# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Modular Remedial Actions for NTC Optimization.

This module encapsulates automatic post-contingency remedial actions
as modular, independent and parameterizable methods:

1. HVDC Fast Modulation Modules (Run-Back / Run-Up / Emergency Power Control)
2. SRAP (Sistema de Reducción Automática de Potencia - Generator Tripping)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Union

from VeraGridEngine.basic_structures import Vec, IntVec, BoolVec, Mat
from VeraGridEngine.Utils.MIP.selected_interface import LpModel, LpVar, LpExp, join
from VeraGridEngine.DataStructures.generator_data import GeneratorData
from VeraGridEngine.DataStructures.passive_branch_data import PassiveBranchData
from VeraGridEngine.Simulations.LinearFactors.linear_analysis import LinearMultiContingency


def add_hvdc_modulation_formulation(prob: LpModel,
                                    t_idx: int,
                                    c: int,
                                    Sbase: float,
                                    contingency: LinearMultiContingency,
                                    base_vsc_flows: Vec,
                                    vsc_rates: Vec,
                                    vsc_active: BoolVec | None,
                                    base_hvdc_flows: Vec,
                                    hvdc_rates: Vec,
                                    hvdc_active: BoolVec | None,
                                    row_filter: BoolVec | None,
                                    vsc_delta_vars: Dict[int, LpVar],
                                    hvdc_delta_vars: Dict[int, LpVar],
                                    branch_terms: Dict[int, LpExp]) -> None:
    """Add HVDC fast modulation (EPC / run-back / run-up) remedial action to the LP.

    Allows post-contingency set-point adjustments Δ_d bounded by converter rating.
    """
    from VeraGridEngine.Simulations.NTC.ntc_opf import add_corrective_converter_deltas

    # VSC converter deltas
    add_corrective_converter_deltas(
        prob=prob, t_idx=t_idx, c=c, Sbase=Sbase,
        compensated_df=contingency.compensated_vsc_df,
        base_flows=base_vsc_flows, rates_mw=vsc_rates,
        outaged_indices=contingency.vsc_indices, active_mask=vsc_active,
        name_tag="vsc", row_filter=row_filter,
        delta_vars=vsc_delta_vars, branch_terms=branch_terms
    )

    # HVDC line deltas
    add_corrective_converter_deltas(
        prob=prob, t_idx=t_idx, c=c, Sbase=Sbase,
        compensated_df=contingency.compensated_hvdc_df,
        base_flows=base_hvdc_flows, rates_mw=hvdc_rates,
        outaged_indices=contingency.hvdc_indices, active_mask=hvdc_active,
        name_tag="hvdc", row_filter=row_filter,
        delta_vars=hvdc_delta_vars, branch_terms=branch_terms
    )


class SrapPrecomputed:
    """Precomputed data structures for ultra-fast SRAP constraint formulation."""
    __slots__ = ("candidate_gens", "bus_to_gens", "candidate_buses", "ptdf_srap")

    def __init__(self, candidate_gens: List[int], bus_to_gens: Dict[int, List[int]],
                 candidate_buses: np.ndarray, ptdf_srap: np.ndarray):
        self.candidate_gens = candidate_gens
        self.bus_to_gens = bus_to_gens
        self.candidate_buses = candidate_buses
        self.ptdf_srap = ptdf_srap


def precompute_srap_data(generator_data: GeneratorData,
                         ptdf: Mat,
                         allowed_gen_indices: IntVec | None = None) -> SrapPrecomputed | None:
    """Precompute SRAP candidate structures and PTDF column slice once for all contingencies."""
    if generator_data is None or generator_data.nelm == 0:
        return None

    gen_iterable = range(generator_data.nelm) if allowed_gen_indices is None else allowed_gen_indices
    candidate_gens: List[int] = []

    for g in gen_iterable:
        if bool(generator_data.active[g]):
            if getattr(generator_data, 'srap_enabled', None) is not None:
                if bool(generator_data.srap_enabled[g]) and float(generator_data.p[g]) > 1e-4:
                    candidate_gens.append(int(g))
            elif float(generator_data.p[g]) > 1e-4:
                candidate_gens.append(int(g))

    if len(candidate_gens) == 0:
        return None

    bus_to_gens: Dict[int, List[int]] = {}
    for g in candidate_gens:
        b = int(generator_data.bus_idx[g])
        bus_to_gens.setdefault(b, []).append(g)

    candidate_buses = np.array(list(bus_to_gens.keys()), dtype=int)
    if len(candidate_buses) == 0:
        return None

    ptdf_srap = ptdf[:, candidate_buses]
    if sp.issparse(ptdf_srap):
        ptdf_srap = ptdf_srap.toarray()
    else:
        ptdf_srap = np.asarray(ptdf_srap)

    return SrapPrecomputed(candidate_gens, bus_to_gens, candidate_buses, ptdf_srap)


def add_srap_formulation(prob: LpModel,
                         t_idx: int,
                         c: int,
                         Sbase: float,
                         contingency: LinearMultiContingency,
                         generator_data: GeneratorData,
                         srap_data: SrapPrecomputed,
                         srap_max_power_mw: float,
                         row_filter: BoolVec | None,
                         srap_delta_vars: Dict[int, LpVar],
                         branch_terms: Dict[int, LpExp],
                         target_branch_indices: IntVec | None = None,
                         sensitivity_threshold: float = 0.05) -> None:
    """Add SRAP (Sistema de Reducción Automática de Potencia) remedial action to the LP.

    Physics and operational constraints:
    1. A generator cannot trip more power than it is currently generating:
       0 <= -ΔP_g <= P0_g  ==>  -P0_g <= ΔP_g <= 0
    2. Total tripped power across the system cannot exceed the operational cap:
       sum(-ΔP_g) <= SRAP_max  ==>  sum(ΔP_g) >= -SRAP_max
    3. Branch flow relief on target constrained branches:
       f_m^(c) += sum_{g in SRAP} cPTDF_{m, bus(g)} * ΔP_g
    """
    if srap_data is None:
        return

    # Determine target branches that need relief
    if target_branch_indices is not None and len(target_branch_indices) > 0:
        target_rows = np.array([int(m) for m in target_branch_indices], dtype=int)
    elif row_filter is not None:
        target_rows = np.flatnonzero(row_filter)
    else:
        target_rows = np.arange(srap_data.ptdf_srap.shape[0], dtype=int)

    if len(target_rows) == 0:
        return

    candidate_buses = srap_data.candidate_buses
    bus_to_gens = srap_data.bus_to_gens
    ptdf_srap = srap_data.ptdf_srap

    # Compute compensated sensitivity matrix ONLY for (target_rows, candidate_buses)
    outaged_br = contingency.branch_indices
    ptdf_target = ptdf_srap[target_rows, :]

    if len(outaged_br) > 0 and contingency.mlodf_factors is not None and contingency.mlodf_factors.nnz > 0:
        mlodf_target = contingency.mlodf_factors[target_rows, :]
        ptdf_out = ptdf_srap[outaged_br, :]
        comp_target = ptdf_target + mlodf_target.dot(ptdf_out)
    else:
        comp_target = ptdf_target

    if sp.issparse(comp_target):
        comp_target = comp_target.toarray()

    # Filter to buses with significant sensitivity (|sens| >= sensitivity_threshold)
    max_sens_per_bus = np.max(np.abs(comp_target), axis=0)
    effective_col_mask = max_sens_per_bus >= sensitivity_threshold
    effective_cols = np.flatnonzero(effective_col_mask)

    if len(effective_cols) == 0:
        return

    if len(effective_cols) > 5:
        top_indices = np.argsort(-max_sens_per_bus[effective_cols])[:5]
        effective_cols = effective_cols[top_indices]

    # Declare LP variables for generators at effective buses
    sum_deltas: List[LpVar] = []
    bus_deltas: Dict[int, LpExp] = {}

    for col_idx in effective_cols:
        b_idx = int(candidate_buses[col_idx])
        gens_at_b = bus_to_gens[b_idx]
        b_expr_terms: List[LpVar] = []

        for g in gens_at_b:
            if g in srap_delta_vars:
                v = srap_delta_vars[g]
            else:
                p0_pu = float(generator_data.p[g]) / Sbase
                v = prob.add_var(lb=-p0_pu, ub=0.0, name=join("srap_g_", [t_idx, c, g], "_"))
                srap_delta_vars[g] = v
                sum_deltas.append(v)
            b_expr_terms.append(v)

        if len(b_expr_terms) == 1:
            bus_deltas[b_idx] = b_expr_terms[0]
        else:
            bus_deltas[b_idx] = prob.sum(b_expr_terms)

    # Global system cap: sum(ΔP_g) >= -SRAP_max / Sbase
    # Only add the constraint once for this contingency
    cst_key = -1
    if cst_key not in srap_delta_vars and len(sum_deltas) > 0 and srap_max_power_mw > 0:
        max_trip_pu = srap_max_power_mw / Sbase
        expr_sum = prob.sum(sum_deltas)
        prob.add_cst(cst=expr_sum >= -max_trip_pu, name=join("srap_max_trip_", [t_idx, c], "_"))
        srap_delta_vars[cst_key] = None

    # Add relief terms directly to branch_terms for target branches
    for row_idx, m in enumerate(target_rows):
        terms_m: List[LpExp] = []
        for col_idx in effective_cols:
            sens = float(comp_target[row_idx, col_idx])
            if abs(sens) >= 1e-4:
                b_idx = int(candidate_buses[col_idx])
                term = sens * bus_deltas[b_idx]
                terms_m.append(term)

        if len(terms_m) > 0:
            sum_m = prob.sum(terms_m)
            m_int = int(m)
            if m_int in branch_terms:
                branch_terms[m_int] = branch_terms[m_int] + sum_m
            else:
                branch_terms[m_int] = sum_m
