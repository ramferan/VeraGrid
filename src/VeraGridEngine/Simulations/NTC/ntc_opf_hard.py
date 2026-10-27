# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Hard NTC policy: one exchange variable, structural area balance, no area slacks.

This module is the modern-engine counterpart of the REE 4.8 NTC policy.  It
reuses the strict formulation (hard thermal limits, hard N-1, compact/lazy
admission and ``add_clipped_flow`` for Pmode3) and replaces only the injection
and area-balance block:

* one ``power_shift`` variable for the whole exchange;
* generation deltas ``power_shift * proportion * sense`` in both areas;
* no ``delta_sl_1/2``: the area balance is structural;
* the physical inter-area objective is the one already present in the strict
  formulation (its branch/HVDC/VSC functions add ``-sum(flow * sense)``).

The injection returns a zero objective because the strict formulation adds the
physical exchange objective itself.  The hard module must not add it twice.
"""
from __future__ import annotations

from VeraGridEngine.Simulations.NTC.ntc_opf import get_exchange_proportions
from VeraGridEngine.Simulations.NTC.ntc_opf_strict import (
    get_base_power,
    get_transfer_power_scaling_per_bus,
)


def add_linear_injections_formulation_hard(t,
                                           Sbase,
                                           gen_data_t,
                                           batt_data_t,
                                           load_data_t,
                                           bus_data_t,
                                           branch_data_t,
                                           active_branch_data_t,
                                           hvdc_data_t,
                                           bus_a1_idx,
                                           bus_a2_idx,
                                           transfer_method,
                                           skip_generation_limits,
                                           ntc_vars,
                                           prob,
                                           logger,
                                           structural_area_balance: bool = True,
                                           **kwargs):
    """Inject the hard policy into the strict formulation.

    The signature matches ``ntc_opf_strict.add_linear_injections_formulation``
    so this function can be used as a drop-in replacement.
    """
    base_power = get_base_power(Sbase=Sbase,
                                gen_data_t=gen_data_t,
                                batt_data_t=batt_data_t,
                                load_data_t=load_data_t,
                                branch_data_t=branch_data_t,
                                active_branch_data_t=active_branch_data_t,
                                hvdc_data_t=hvdc_data_t,
                                logger=logger)

    bus_pref_t, bus_pmax_t, bus_pmin_t = get_transfer_power_scaling_per_bus(
        bus_data_t=bus_data_t,
        gen_data_t=gen_data_t,
        load_data_t=load_data_t,
        transfer_method=transfer_method,
        skip_generation_limits=skip_generation_limits,
        inf_value=prob.INFINITY,
        Sbase=Sbase
    )

    proportions = get_exchange_proportions(
        power=bus_pref_t,
        bus_a1_idx=bus_a1_idx,
        bus_a2_idx=bus_a2_idx,
        logger=logger,
        decimals=6
    )
    ntc_vars.bus_vars.proportions[t, :] = proportions

    # One exchange variable for both areas.  The proportion carries the sign
    # (positive in area 1, negative in area 2), so the same variable increases
    # one area and decreases the other exactly by the same amount.
    power_shift = prob.add_var(lb=-prob.INFINITY,
                               ub=prob.INFINITY,
                               name=f'power_shift_{t}')
    ntc_vars.power_shift[t] = power_shift
    ntc_vars.delta_1[t] = power_shift
    ntc_vars.delta_2[t] = power_shift

    # Keep the result/reporting contract of the strict formulation, but with
    # zero area slacks: the balance is structural, not relaxed.
    ntc_vars.delta_sl_1[t] = prob.add_var(lb=0, ub=0, name=f'DeltaSL_up_zero_{t}')
    ntc_vars.delta_sl_2[t] = prob.add_var(lb=0, ub=0, name=f'DeltaSL_down_zero_{t}')

    for k in bus_a1_idx:
        if bus_data_t.active[k] and proportions[k] != 0:
            ntc_vars.bus_vars.delta_p[t, k] = power_shift * proportions[k]

    for k in bus_a2_idx:
        if bus_data_t.active[k] and proportions[k] != 0:
            ntc_vars.bus_vars.delta_p[t, k] = power_shift * proportions[k]

    for k in range(bus_data_t.nbus):
        ntc_vars.bus_vars.Pinj[t, k] += base_power[k] + ntc_vars.bus_vars.delta_p[t, k]
        ntc_vars.bus_vars.Pbalance[t, k] += ntc_vars.bus_vars.Pinj[t, k]

    # The strict branch/HVDC/VSC formulations add the physical exchange to the
    # objective.  Returning zero here avoids double counting it.
    return 0.0, base_power


def run_linear_ntc_opf_hard(*args, **kwargs):
    """Run the strict formulation with only the hard injection patched in."""
    import VeraGridEngine.Simulations.NTC.ntc_opf_strict as strict

    original = strict.add_linear_injections_formulation
    strict.add_linear_injections_formulation = add_linear_injections_formulation_hard
    try:
        return strict.run_linear_ntc_opf_strict(*args, **kwargs)
    finally:
        strict.add_linear_injections_formulation = original
