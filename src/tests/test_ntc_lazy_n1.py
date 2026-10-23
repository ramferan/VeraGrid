# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import os
import unittest

import numpy as np

try:
    from ortools.linear_solver import pywraplp
except ImportError:
    pywraplp = None

from GridCal.Engine import FileOpen
from GridCal.Engine import Devices as dev
from GridCal.Engine import basic_structures as bs
from GridCal.Engine.Simulations.ATC.available_transfer_capacity_driver import AvailableTransferMode
from GridCal.Engine.Simulations.NTC.ntc_driver import OptimalNetTransferCapacityDriver
from GridCal.Engine.Simulations.NTC.ntc_opf import OpfNTC
from GridCal.Engine.Simulations.NTC.ntc_options import OptimalNetTransferCapacityOptions
from GridCal.Engine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
from GridCal.Engine.basic_structures import SolverType

# this file lives in <repo>/src/tests
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRID = os.path.join(_REPO_ROOT, 'Grids_and_profiles', 'grids',
                    'IEEE14 - ntc areas_voltages_hvdc.gridcal')


def add_n1_contingencies(circuit):
    """One single-branch contingency group per AC branch (all-branches N-1)."""
    from GridCal.Engine.Devices.contingency import Contingency
    from GridCal.Engine.Devices.contingency_group import ContingencyGroup

    for i, branch in enumerate(circuit.get_branches_wo_hvdc()):
        group = ContingencyGroup(name='N-1 {0} {1}'.format(branch.name, i))
        circuit.contingency_groups.append(group)
        circuit.contingencies.append(
            Contingency(device_idtag=branch.idtag, name='CTGC {0}'.format(i), group=group))


@unittest.skipUnless(pywraplp is not None, 'ortools is not available')
@unittest.skipUnless(os.path.exists(GRID), 'NTC test grid is not available')
class TestLazyN1(unittest.TestCase):
    """
    The lazy mode must reach the same TTC as the static N-1 block while adding
    only the contingency rows that actually bind.
    """

    def _run(self, use_lazy_n1):
        circuit = FileOpen(GRID).open()
        add_n1_contingencies(circuit)

        areas = list(circuit.areas)
        idx_from = np.array([i for i, bus in circuit.get_areas_buses([areas[0]])], dtype=int)
        idx_to = np.array([i for i, bus in circuit.get_areas_buses([areas[1]])], dtype=int)

        options = OptimalNetTransferCapacityOptions(
            area_from_bus_idx=idx_from,
            area_to_bus_idx=idx_to,
            mip_solver=bs.MIPSolvers.CBC,
            generation_formulation=dev.GenerationNtcFormulation.Proportional,
            monitor_only_sensitive_branches=True,
            branch_sensitivity_threshold=0.05,
            skip_generation_limits=True,
            consider_contingencies=True,
            consider_gen_contingencies=False,
            consider_hvdc_contingencies=False,
            dispatch_all_areas=False,
            tolerance=1e-2,
            sensitivity_dT=100.0,
            transfer_method=AvailableTransferMode.InstalledPower,
            time_limit_ms=60000,
            use_lazy_n1=use_lazy_n1,
        )

        formulation_sizes = list()
        original = OpfNTC.formulate

        def capture(problem, *args, **kwargs):
            out = original(problem, *args, **kwargs)
            formulation_sizes.append(int(problem.solver.NumConstraints()))
            return out

        OpfNTC.formulate = capture
        try:
            driver = OptimalNetTransferCapacityDriver(
                grid=circuit,
                options=options,
                pf_options=PowerFlowOptions(solver_type=SolverType.DC))
            driver.run()
        finally:
            OpfNTC.formulate = original

        return driver, formulation_sizes[0]

    def test_lazy_matches_full_n1(self):
        full_driver, full_constraints = self._run(use_lazy_n1=False)
        lazy_driver, lazy_constraints = self._run(use_lazy_n1=True)

        full_res = full_driver.results
        lazy_res = lazy_driver.results

        self.assertIsNotNone(full_res)
        self.assertIsNotNone(lazy_res)

        full_ttc = float(full_res.get_exchange_power())
        lazy_ttc = float(lazy_res.get_exchange_power())

        self.assertTrue(np.isclose(lazy_ttc, full_ttc, rtol=0.0, atol=1e-6),
                        'lazy TTC {0} != full TTC {1}'.format(lazy_ttc, full_ttc))

        # The initial lazy LP has no contingency rows at all
        self.assertLess(lazy_constraints, full_constraints)

        # The reports must still cover every screened pair
        self.assertEqual(len(lazy_res.contingency_branch_indices_list),
                         len(full_res.contingency_branch_indices_list))

        # Every reported post-contingency flow must respect its rate
        rates = lazy_res.rates
        for (m, _), flow in zip(lazy_res.contingency_branch_indices_list,
                                lazy_res.contingency_branch_flows_list):
            self.assertLessEqual(abs(flow), rates[m] + 1e-3,
                                 'lazy N-1 violation on branch {0}'.format(m))

        # The lazy loop must have converged and said so
        messages = [str(getattr(entry, 'msg', '')) for entry in lazy_driver.logger.entries]
        self.assertTrue(any('Lazy N-1 converged' in text for text in messages))


if __name__ == '__main__':
    unittest.main()
