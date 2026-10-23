#!/usr/bin/env python
"""
Smoke benchmark for the NTC (OptimalNetTransferCapacity) driver.

It is meant to compare branches of the fork on the same model, not to be a
pytest test (the file name does not start with ``test_``).

Usage
-----
Inspect a model (areas, tie lines, HVDC, sizes)::

    PYTHONPATH=src python src/tests/bench_ntc_smoke.py --grid <file.gridcal> --inspect

Run the snapshot NTC driver::

    PYTHONPATH=src python src/tests/bench_ntc_smoke.py --grid <file.gridcal> \
        --area-from 0 --area-to 1 --time-limit-ms 20000 --json out.json

The script patches ``OpfNTC.formulate`` only to capture the LP size
(number of variables/constraints) of the generated problem. The formulation
itself is untouched.
"""

import argparse
import json
import os
import platform
import sys
import time
import traceback

import numpy as np

from GridCal.Engine import FileOpen
from GridCal.Engine import Devices as dev
from GridCal.Engine import basic_structures as bs
from GridCal.Engine.Simulations.ATC.available_transfer_capacity_driver import AvailableTransferMode
from GridCal.Engine.Simulations.NTC.ntc_options import OptimalNetTransferCapacityOptions
from GridCal.Engine.Simulations.NTC.ntc_driver import OptimalNetTransferCapacityDriver
from GridCal.Engine.Simulations.NTC.ntc_opf import OpfNTC
from GridCal.Engine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
from GridCal.Engine.basic_structures import SolverType


FORMULATION_STATS = list()


def patch_formulation_stats():
    """Record LP size right after each OpfNTC.formulate() call."""
    original = OpfNTC.formulate
    if getattr(original, '_bench_patched', False):
        return

    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = original(self, *args, **kwargs)
        elapsed = time.perf_counter() - t0

        try:
            n_vars = int(self.solver.NumVariables())
            n_cons = int(self.solver.NumConstraints())
        except Exception:
            n_vars = -1
            n_cons = -1

        stats = {
            'formulation_s': elapsed,
            'lp_variables': n_vars,
            'lp_constraints': n_cons,
        }
        self._bench_stats = stats
        FORMULATION_STATS.append(stats)
        return out

    wrapper._bench_patched = True
    OpfNTC.formulate = wrapper


def _safe_len(obj):
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    try:
        return len(obj)
    except Exception:
        return -1


def _count(circuit, name):
    """Count elements of ``circuit`` by ``name`` using the getter when available."""
    getter = getattr(circuit, f'get_{name}', None)
    try:
        if callable(getter):
            return _safe_len(getter())
    except Exception:
        return -1
    return _safe_len(getattr(circuit, name, []))


def model_summary(circuit):
    time_profile = getattr(circuit, 'time_profile', None)
    return {
        'buses': _count(circuit, 'buses'),
        'branches_ac': _count(circuit, 'branches_wo_hvdc'),
        'branches_total': _count(circuit, 'branches'),
        'transformers3w': _safe_len(getattr(circuit, 'transformers3w', [])),
        'hvdc': _count(circuit, 'hvdc'),
        'vsc': _count(circuit, 'vsc'),
        'generators': _count(circuit, 'generators'),
        'loads': _count(circuit, 'loads'),
        'batteries': _count(circuit, 'batteries'),
        'areas': _count(circuit, 'area_number'),
        'time_steps': 0 if time_profile is None else _safe_len(time_profile),
    }


def inspect_grid(circuit, max_pairs=12):
    summary = model_summary(circuit)
    print('Model summary')
    for key, value in summary.items():
        print(f'  {key:16s} {value}')

    print('\nAreas')
    areas = list(circuit.areas)
    areas_info = list()
    for i, area in enumerate(areas):
        try:
            buses = circuit.get_area_buses(area)
        except Exception:
            buses = list()
        areas_info.append({
            'index': i,
            'name': getattr(area, 'name', f'area_{i}'),
            'buses': len(buses),
        })
        print(f'  [{i}] {areas_info[-1]["name"]!r:40s} buses={len(buses)}')

    print('\nCandidate area pairs with tie lines / HVDC')
    n = len(areas)
    pairs = 0
    for i in range(n):
        for j in range(n):
            if i == j or pairs >= max_pairs:
                continue
            a_i = [areas[i]]
            a_j = [areas[j]]
            try:
                branches = circuit.get_inter_areas_branches(a_i, a_j)
            except Exception:
                branches = list()
            try:
                hvdc = circuit.get_inter_areas_hvdc_branches(a_i, a_j)
            except Exception:
                hvdc = list()
            if len(branches) > 0 or len(hvdc) > 0:
                print(f'  {i} -> {j}: inter-area branches={len(branches)}, '
                      f'inter-area HVDC={len(hvdc)}')
                pairs += 1

    return {
        'summary': summary,
        'areas': areas_info,
    }


def add_synthetic_contingencies(circuit, max_contingencies=None):
    """Create one single-branch contingency group per branch.

    ``LinearAnalysis`` builds ``LODF_NX`` from the contingency objects stored in
    the model. Test grids usually have no contingencies, which silently turns
    N-1 off. This helper provides the equivalent of "all branches N-1" so the
    benchmark can exercise the contingency block.
    """
    from GridCal.Engine.Devices.contingency import Contingency
    from GridCal.Engine.Devices.contingency_group import ContingencyGroup

    if len(circuit.contingencies) > 0:
        return 0

    # LODF_NX only covers AC branches; a contingency on an HVDC link makes
    # make_lodf_nx index outside the LODF matrix.
    branches = list(circuit.get_branches_wo_hvdc())
    if max_contingencies is not None:
        branches = branches[:max_contingencies]

    for i, branch in enumerate(branches):
        group = ContingencyGroup(name='N-1 {0} {1}'.format(branch.name, i))
        circuit.contingency_groups.append(group)
        circuit.contingencies.append(
            Contingency(device_idtag=branch.idtag,
                        name='CTGC {0}'.format(i),
                        group=group))

    return len(branches)


def configure_hvdc(circuit, mode, droop, pset):
    """Optionally switch the HVDC links to Pmode3 (``type_0_free``) or Pmode1."""
    from GridCal.Engine.Devices.enumerations import HvdcControlType

    if mode == 'keep':
        return 0

    hvdc = list(circuit.get_hvdc())
    for link in hvdc:
        if pset is not None:
            link.Pset = float(pset)
        if droop is not None:
            link.angle_droop = float(droop)
        if mode == 'free':
            link.control_mode = HvdcControlType.type_0_free
        else:
            link.control_mode = HvdcControlType.type_1_Pset

    return len(hvdc)


def build_options(args, circuit):
    areas = list(circuit.areas)
    if args.area_from >= len(areas) or args.area_to >= len(areas):
        raise IndexError(
            f'Area index out of range: from={args.area_from}, to={args.area_to}, '
            f'available={len(areas)}')

    areas_from = [areas[args.area_from]]
    areas_to = [areas[args.area_to]]

    lst_from = circuit.get_areas_buses(areas_from)
    lst_to = circuit.get_areas_buses(areas_to)

    idx_from = np.array([i for i, bus in lst_from], dtype=int)
    idx_to = np.array([i for i, bus in lst_to], dtype=int)

    if idx_from.size == 0 or idx_to.size == 0:
        raise ValueError(
            f'Empty area: from buses={idx_from.size}, to buses={idx_to.size}')

    solver = getattr(bs.MIPSolvers, args.solver)

    options = OptimalNetTransferCapacityOptions(
        area_from_bus_idx=idx_from,
        area_to_bus_idx=idx_to,
        mip_solver=solver,
        generation_formulation=dev.GenerationNtcFormulation.Proportional,
        monitor_only_sensitive_branches=not args.no_sensitive_filter,
        branch_sensitivity_threshold=args.sensitivity_threshold,
        skip_generation_limits=True,
        consider_contingencies=not args.no_contingencies,
        consider_gen_contingencies=not args.no_gen_contingencies,
        consider_hvdc_contingencies=not args.no_hvdc_contingencies,
        consider_nx_contingencies=not args.no_nx_contingencies,
        dispatch_all_areas=False,
        generation_contingency_threshold=args.gen_contingency_threshold,
        tolerance=1e-2,
        sensitivity_dT=args.sensitivity_dt,
        transfer_method=AvailableTransferMode.InstalledPower,
        time_limit_ms=args.time_limit_ms,
        loading_threshold_to_report=98,
        weight_power_shift=1e5,
        weight_generation_cost=1e2,
    )
    return options, idx_from, idx_to


def _as_float(value, default=float('nan')):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def summarize_results(driver):
    res = driver.results
    if res is None:
        return {'results': None}

    try:
        exchange = _as_float(res.get_exchange_power())
    except Exception:
        exchange = float('nan')

    out = {
        'exchange_power': exchange,
        'structural_ntc': _as_float(getattr(res, 'structural_ntc', None)),
    }

    for attr in ('monitor', 'monitor_by_sensitivity', 'monitor_by_unrealistic_ntc',
                 'monitor_by_zero_exchange'):
        value = getattr(res, attr, None)
        try:
            out[attr] = int(np.sum(np.asarray(value, dtype=bool)))
        except Exception:
            out[attr] = -1

    try:
        loading = np.asarray(res.loading, dtype=float)
        out['max_loading'] = float(np.nanmax(loading)) if loading.size else float('nan')
    except Exception:
        out['max_loading'] = float('nan')

    try:
        out['inter_area_branches'] = int(len(res.inter_area_branches))
    except Exception:
        out['inter_area_branches'] = -1

    try:
        out['inter_area_hvdc'] = int(len(res.inter_area_hvdc))
    except Exception:
        out['inter_area_hvdc'] = -1

    for key, attr in (('contingency_branch_rows', 'contingency_branch_flows_list'),
                      ('contingency_gen_rows', 'contingency_generation_flows_list'),
                      ('contingency_hvdc_rows', 'contingency_hvdc_flows_list')):
        value = getattr(res, attr, None)
        out[key] = -1 if value is None else _safe_len(value)

    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='NTC smoke benchmark')
    parser.add_argument('--grid', required=True, help='path to .gridcal model')
    parser.add_argument('--inspect', action='store_true', help='only inspect the model')
    parser.add_argument('--area-from', type=int, default=0)
    parser.add_argument('--area-to', type=int, default=1)
    parser.add_argument('--solver', default='CBC', choices=['CBC', 'SCIP', 'GLOP'])
    parser.add_argument('--time-limit-ms', type=float, default=20000.0)
    parser.add_argument('--sensitivity-threshold', type=float, default=0.05)
    parser.add_argument('--sensitivity-dt', type=float, default=100.0)
    parser.add_argument('--gen-contingency-threshold', type=float, default=1000.0)
    parser.add_argument('--no-contingencies', action='store_true',
                        help='disable all N-1 contingencies')
    parser.add_argument('--no-nx-contingencies', action='store_true')
    parser.add_argument('--no-gen-contingencies', action='store_true')
    parser.add_argument('--no-hvdc-contingencies', action='store_true')
    parser.add_argument('--no-sensitive-filter', action='store_true',
                        help='monitor every branch, not only the sensitive ones')
    parser.add_argument('--synthetic-contingencies', action='store_true',
                        help='add one N-1 contingency per branch when the model has none')
    parser.add_argument('--hvdc-mode', default='keep', choices=['keep', 'free', 'pset'],
                        help="'free' is Pmode3 (droop), 'pset' is Pmode1")
    parser.add_argument('--hvdc-droop', type=float, default=None,
                        help='HVDC droop in MW/degree for Pmode3')
    parser.add_argument('--hvdc-pset', type=float, default=None,
                        help='HVDC Pset in MW')
    parser.add_argument('--max-contingencies', type=int, default=None,
                        help='limit the number of synthetic contingencies')
    parser.add_argument('--state', type=int, default=None,
                        help='optional time step applied with circuit.set_state(t)')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--lp', default=None, help='optional path to dump the LP')
    parser.add_argument('--json', default=None, help='optional path for a JSON report')
    parser.add_argument('--label', default=None, help='tag written in the JSON report')

    args = parser.parse_args(argv)

    t_load = time.perf_counter()
    circuit = FileOpen(args.grid).open()
    load_s = time.perf_counter() - t_load

    if args.inspect:
        info = inspect_grid(circuit)
        info['file'] = os.path.abspath(args.grid)
        info['load_s'] = load_s
        if args.json:
            with open(args.json, 'w') as fh:
                json.dump(info, fh, indent=2)
        return 0

    if args.state is not None:
        circuit.set_state(args.state)

    if args.synthetic_contingencies:
        added = add_synthetic_contingencies(circuit, args.max_contingencies)
        if added > 0:
            print(f'  synthetic contingencies: {added} (one per branch)')
        else:
            print('  synthetic contingencies: model already has contingencies, nothing added')

    if args.hvdc_mode != 'keep':
        n_hvdc = configure_hvdc(circuit, args.hvdc_mode, args.hvdc_droop, args.hvdc_pset)
        print(f'  HVDC mode: {args.hvdc_mode} on {n_hvdc} link(s) '
              f'(droop={args.hvdc_droop}, Pset={args.hvdc_pset})')

    options, idx_from, idx_to = build_options(args, circuit)
    pf_options = PowerFlowOptions(solver_type=SolverType.DC)

    print(f'Model: {os.path.abspath(args.grid)}')
    print(f'  load time          : {load_s:.2f} s')
    print(f'  area from buses    : {idx_from.size}')
    print(f'  area to buses      : {idx_to.size}')
    print(f'  contingencies      : {not args.no_contingencies}')
    print(f'  N-x contingencies  : {not args.no_nx_contingencies}')
    print(f'  gen contingencies  : {not args.no_gen_contingencies}')
    print(f'  HVDC contingencies : {not args.no_hvdc_contingencies}')
    print(f'  solver             : {args.solver}')

    patch_formulation_stats()

    runs = list()
    for run_idx in range(max(args.repeat, 1)):
        FORMULATION_STATS.clear()

        driver = OptimalNetTransferCapacityDriver(
            grid=circuit,
            options=options,
            pf_options=pf_options,
        )

        t0 = time.perf_counter()
        try:
            driver.run()
            error = None
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
        total_s = time.perf_counter() - t0

        if args.lp is not None and run_idx == 0 and hasattr(driver, 'results') \
                and driver.results is not None:
            pass  # LP dumping is done through OpfNTC.save_lp; not kept by the driver

        summary = {
            'run': run_idx,
            'elapsed_s': total_s,
            'error': error,
            'formulation': dict(FORMULATION_STATS[0]) if FORMULATION_STATS else None,
            'driver_elapsed_s': getattr(driver, 'elapsed', None),
        }

        if error is None:
            try:
                summary.update(summarize_results(driver))
            except Exception as exc:
                summary['summary_error'] = f'{type(exc).__name__}: {exc}'

            n_errors = 0
            try:
                n_errors = sum(1 for e in driver.logger.entries
                               if getattr(e, 'severity', None) is not None
                               and str(getattr(e, 'severity', '')).endswith('Error'))
            except Exception:
                pass
            summary['logger_errors'] = n_errors
            summary['logger_entries'] = len(getattr(driver.logger, 'entries', []))

        runs.append(summary)

        print(f'\nRun {run_idx}')
        print(f'  total time         : {total_s:.3f} s')
        if summary.get('formulation'):
            f = summary['formulation']
            print(f'  formulation time   : {f["formulation_s"]:.3f} s')
            print(f'  LP variables       : {f["lp_variables"]}')
            print(f'  LP constraints     : {f["lp_constraints"]}')
        if error is None:
            print(f'  exchange power     : {summary.get("exchange_power")}')
            print(f'  structural NTC     : {summary.get("structural_ntc")}')
            print(f'  monitored branches : {summary.get("monitor")}')
            print(f'  sensitive branches : {summary.get("monitor_by_sensitivity")}')
            print(f'  max loading (p.u.) : {summary.get("max_loading")}')
            print(f'  contingency rows   : {summary.get("contingency_branch_rows")}')
            print(f'  logger entries     : {summary.get("logger_entries")} '
                  f'(errors={summary.get("logger_errors")})')
            if summary.get('summary_error'):
                print(f'  summary warning    : {summary["summary_error"]}')
        else:
            print(f'  ERROR              : {error}')

    report = {
        'file': os.path.abspath(args.grid),
        'label': args.label,
        'python': sys.version,
        'platform': platform.platform(),
        'solver': args.solver,
        'area_from': args.area_from,
        'area_to': args.area_to,
        'time_limit_ms': args.time_limit_ms,
        'contingencies': not args.no_contingencies,
        'nx_contingencies': not args.no_nx_contingencies,
        'gen_contingencies': not args.no_gen_contingencies,
        'hvdc_contingencies': not args.no_hvdc_contingencies,
        'sensitive_filter': not args.no_sensitive_filter,
        'model': model_summary(circuit),
        'runs': runs,
    }

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(report, fh, indent=2, default=float)

    if any(run.get('error') for run in runs):
        return 2

    return 0


if __name__ == '__main__':
    sys.exit(main())
