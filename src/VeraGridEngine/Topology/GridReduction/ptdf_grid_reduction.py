# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import numpy as np
from typing import Tuple, TYPE_CHECKING
from scipy.sparse import coo_matrix
from VeraGridEngine.basic_structures import IntVec, Mat, Logger, Vector
from VeraGridEngine.enumerations import DeviceType
from VeraGridEngine.Compilers.circuit_to_data import compile_numerical_circuit_at
from VeraGridEngine.Devices.Injections.generator import Generator
from VeraGridEngine.Devices.Injections.battery import Battery
from VeraGridEngine.Devices.Injections.static_generator import StaticGenerator
from VeraGridEngine.Devices.Injections.load import Load
from VeraGridEngine.Topology.topology import find_islands, build_branches_C_coo_3, find_different_states

if TYPE_CHECKING:
    from VeraGridEngine.Devices.multi_circuit import MultiCircuit
    from VeraGridEngine.Simulations.PowerFlow.power_flow_results import PowerFlowResults
    from VeraGridEngine.Simulations.LinearFactors.linear_analysis import LinearAnalysisTs


def ptdf_reduction(grid: MultiCircuit,
                   reduction_bus_indices: IntVec,
                   PTDF: Mat,
                   lin_ts: LinearAnalysisTs,
                   tol=1e-8,
                   aggregate_devices: bool = False) -> Tuple[MultiCircuit, Logger]:
    """
    In-place Grid reduction using the PTDF injection mirroring
    No theory available
    :param grid: MultiCircuit
    :param reduction_bus_indices: Bus indices of the buses to delete
    :param PTDF: PTDF matrix
    :param lin_ts: LinearAnalysisTs
    :param tol: Tolerance, any equivalent power value under this is omitted
    :param aggregate_devices: Aggregate boundary devices (optional)
    """
    logger = Logger()

    # find the boundary set: buses from the internal set the join to the external set
    e_buses, b_buses, i_buses, b_branches = grid.get_reduction_sets(reduction_bus_indices=reduction_bus_indices)

    if len(e_buses) == 0:
        logger.add_info(msg="Nothing to reduce")
        return grid, logger

    if len(i_buses) == 0:
        logger.add_info(msg="Nothing to keep (null grid as a result)")
        return grid, logger

    if len(b_buses) == 0:
        logger.add_info(msg="The reducible and non reducible sets are disjoint and cannot be reduced")
        return grid, logger

    # Start moving objects
    e_buses_set = set(e_buses)
    bus_dict = grid.get_bus_index_dict()
    has_ts = grid.has_time_series

    if has_ts and lin_ts is None:
        logger.add_error("You must provide the lin_ts parameter")
        return grid, logger

    nb = len(b_buses)
    boundary_generators_with_srap = np.zeros(nb, dtype=int)
    boundary_generators_count = np.zeros(nb, dtype=int)
    boundary_generators = Vector(nb, value=list())
    boundary_batteries = Vector(nb, value=list())
    boundary_loads = Vector(nb, value=list())
    boundary_stagen = Vector(nb, value=list())

    for elm in grid.get_injection_devices():
        if elm.bus is not None:
            i = bus_dict[elm.bus]  # bus index where it is currently connected

            if i in e_buses_set:
                # this injection is to be reduced

                for b in range(len(b_buses)):
                    bus_idx = b_buses[b]
                    branch_idx = b_branches[b]
                    bus = grid.buses[bus_idx]
                    ptdf_val = PTDF[branch_idx, bus_idx]

                    if abs(ptdf_val) > tol:

                        # create new device at the boundary bus
                        if elm.device_type == DeviceType.GeneratorDevice:
                            new_elm = elm.copy()
                            elm.bus = bus
                            new_elm.comment = "PTDF reduced equivalent generator"
                            new_elm.P = ptdf_val * elm.P
                            if has_ts:
                                new_elm.P_prof = lin_ts.get_time_flow(branch_idx, bus_idx, elm.P_prof.toarray())

                            new_elm.comment = "PTDF reduced equivalent generator"
                            if aggregate_devices:
                                boundary_generators[b].append(new_elm)
                                boundary_generators_count[b] += 1
                                if elm.srap_enabled:
                                    boundary_generators_with_srap[b] += 1
                            else:
                                grid.add_generator(bus=bus, api_obj=new_elm)

                        elif elm.device_type == DeviceType.BatteryDevice:
                            new_elm = elm.copy()
                            elm.bus = bus
                            new_elm.P = ptdf_val * elm.P
                            if has_ts:
                                new_elm.P_prof = lin_ts.get_time_flow(branch_idx, bus_idx, elm.P_prof.toarray())

                            new_elm.comment = "PTDF reduced equivalent battery"

                            if aggregate_devices:
                                boundary_batteries[b].append(new_elm)
                            else:
                                grid.add_battery(bus=bus, api_obj=new_elm)

                        elif elm.device_type == DeviceType.StaticGeneratorDevice:
                            new_elm = elm.copy()
                            elm.bus = bus
                            new_elm.P = ptdf_val * elm.P
                            new_elm.Q = ptdf_val * elm.Q
                            if has_ts:
                                new_elm.P_prof = lin_ts.get_time_flow(branch_idx, bus_idx, elm.P_prof.toarray())
                                new_elm.Q_prof = lin_ts.get_time_flow(branch_idx, bus_idx, elm.Q_prof.toarray())

                            new_elm.comment = "PTDF reduced equivalent static generator"

                            if aggregate_devices:
                                boundary_stagen[b].append(new_elm)
                            else:
                                grid.add_static_generator(bus=bus, api_obj=new_elm)

                        elif elm.device_type == DeviceType.LoadDevice:
                            new_elm = elm.copy()
                            elm.bus = bus
                            new_elm.P = ptdf_val * elm.P
                            new_elm.Q = ptdf_val * elm.Q
                            if has_ts:
                                new_elm.P_prof = lin_ts.get_time_flow(branch_idx, bus_idx, elm.P_prof.toarray())
                                new_elm.Q_prof = lin_ts.get_time_flow(branch_idx, bus_idx, elm.Q_prof.toarray())

                            new_elm.comment = "PTDF reduced equivalent load"

                            if aggregate_devices:
                                boundary_loads[b].append(new_elm)
                            else:
                                grid.add_load(bus=bus, api_obj=new_elm)

                        else:
                            # device I don't care about
                            logger.add_warning(msg="Ignored device",
                                               device=str(elm),
                                               device_class=elm.device_type.value)

    if aggregate_devices:

        for b in range(nb):  # for every boundary bus ...

            bus_idx = b_buses[b]
            bus = grid.buses[bus_idx]

            # Generators -----------------------------------------------------------------------------------------------
            gen_list = boundary_generators[b]
            srap_gen_count = boundary_generators_with_srap[b]
            total_gen_count = boundary_generators_count[b]

            if total_gen_count > 0:

                if srap_gen_count > 0:
                    # we make 2 generator because of things
                    gen_no_srap = Generator(name=f"Gen no Srap")
                    gen_srap = Generator(name=f"Gen with Srap", srap_enabled=True)

                    for gen in gen_list:
                        if gen.srap_enabled:
                            gen_srap += gen
                        else:
                            gen_no_srap += gen

                    grid.add_generator(bus=bus, api_obj=gen_srap)
                    grid.add_generator(bus=bus, api_obj=gen_no_srap)

                else:
                    gen_no_srap = Generator(name=f"Equivalent boundary gen")
                    for gen in gen_list:
                        gen_no_srap += gen
                    grid.add_generator(bus=bus, api_obj=gen_no_srap)
            else:
                pass

            # Loads ----------------------------------------------------------------------------------------------------
            load = Load(name=f"Equivalent boundary load")
            for elm in boundary_loads[b]:
                load += elm
            grid.add_load(bus=bus, api_obj=load)

            # StaticGenerator ------------------------------------------------------------------------------------------
            stagen = StaticGenerator(name=f"Equivalent boundary load")
            for elm in boundary_stagen[b]:
                stagen += elm
            grid.add_static_generator(bus=bus, api_obj=stagen)

            # Batteries ------------------------------------------------------------------------------------------------
            batt = Battery(name=f"Equivalent boundary battery")
            for elm in boundary_batteries[b]:
                batt += elm
            grid.add_battery(bus=bus, api_obj=batt)

    # Delete the external buses
    to_be_deleted = [grid.buses[e] for e in e_buses]
    for bus in to_be_deleted:
        grid.delete_bus(obj=bus, delete_associated=True)

    return grid, logger


def ptdf_reduction_with_islands(grid: MultiCircuit,
                                reduction_bus_indices: IntVec,
                                PTDF: Mat,
                                tol=1e-8) -> Tuple[MultiCircuit, Logger]:
    """
    In-place Grid reduction using the PTDF injection mirroring power by island
    No theory available
    :param grid: MultiCircuit
    :param reduction_bus_indices: Bus indices of the buses to delete
    :param PTDF: PTDF matrix
    :param tol: Tolerance, any equivalent power value under this is omitted
    """
    logger = Logger()

    # find the boundary set: buses from the internal set the join to the external set
    e_buses, b_buses, i_buses, b_branches = grid.get_reduction_sets(reduction_bus_indices=reduction_bus_indices)

    if len(e_buses) == 0:
        logger.add_info(msg="Nothing to reduce")
        return grid, logger

    if len(i_buses) == 0:
        logger.add_info(msg="Nothing to keep (null grid as a result)")
        return grid, logger

    if len(b_buses) == 0:
        logger.add_info(msg="The reducible and non reducible sets are disjoint and cannot be reduced")
        return grid, logger

    # get the islands --------------------------------------------------------------------------------------------------
    nbus = grid.get_bus_number()
    nc = compile_numerical_circuit_at(circuit=grid, t_idx=None)

    # Get the arrays to prepare the topology
    (bus_active,
     branch_active, branch_F, branch_T,
     hvdc_active, hvdc_F, hvdc_T,
     vsc_active, vsc_F, vsc_T) = grid.get_topology_data(t_idx=None)

    branch_active[b_branches] = 0  # deactivate boundary branches

    i, j, data, n_elm = build_branches_C_coo_3(
        bus_active=bus_active,
        F1=branch_F, T1=branch_T, active1=branch_active,
        F2=vsc_F, T2=vsc_T, FN2=np.full(len(vsc_F), -1, dtype=int), active2=vsc_active,
        F3=hvdc_F, T3=hvdc_T, active3=hvdc_active,
    )

    C = coo_matrix((data, (i, j)), shape=(n_elm, nbus), dtype=int)
    adj = (C.T @ C).tocsc()

    idx_islands = find_islands(adj=adj, active=nc.bus_data.active)

    # Start moving objects ---------------------------------------------------------------------------------------------
    e_buses_set = set(e_buses)
    bus_dict = grid.get_bus_index_dict()
    has_ts = grid.has_time_series

    for elm in grid.get_injection_devices():
        if elm.bus is not None:
            i = bus_dict[elm.bus]  # bus index where it is currently connected

            if i in e_buses_set:
                # this generator is to be reduced

                for b in range(len(b_buses)):
                    bus_idx = b_buses[b]
                    branch_idx = b_branches[b]
                    bus = grid.buses[bus_idx]
                    ptdf_val = PTDF[branch_idx, bus_idx]

                    if abs(ptdf_val) > tol:

                        # create new device at the boundary bus
                        if elm.device_type == DeviceType.GeneratorDevice:
                            new_elm = elm.copy()
                            new_elm.comment = "PTDF reduced equivalent generator"
                            new_elm.P = ptdf_val * elm.P
                            if has_ts:
                                new_elm.P_prof = ptdf_val * elm.P_prof.toarray()
                            new_elm.comment = "Equivalent generator"
                            grid.add_generator(bus=bus, api_obj=new_elm)

                        elif elm.device_type == DeviceType.BatteryDevice:
                            new_elm = elm.copy()
                            new_elm.P = ptdf_val * elm.P
                            if has_ts:
                                new_elm.P_prof = ptdf_val * elm.P_prof.toarray()
                            new_elm.comment = "Equivalent battery"
                            grid.add_battery(bus=bus, api_obj=new_elm)

                        elif elm.device_type == DeviceType.StaticGeneratorDevice:
                            new_elm = elm.copy()
                            new_elm.P = ptdf_val * elm.P
                            new_elm.Q = ptdf_val * elm.Q
                            if has_ts:
                                new_elm.P_prof = ptdf_val * elm.P_prof.toarray()
                                new_elm.Q_prof = ptdf_val * elm.Q_prof.toarray()
                            new_elm.comment = "Equivalent static generator"
                            grid.add_static_generator(bus=bus, api_obj=new_elm)

                        elif elm.device_type == DeviceType.LoadDevice:
                            new_elm = elm.copy()
                            new_elm.P = ptdf_val * elm.P
                            new_elm.Q = ptdf_val * elm.Q
                            if has_ts:
                                new_elm.P_prof = ptdf_val * elm.P_prof.toarray()
                                new_elm.Q_prof = ptdf_val * elm.Q_prof.toarray()
                            new_elm.comment = "Equivalent load"
                            grid.add_load(bus=bus, api_obj=new_elm)
                        else:
                            # device I don't care about
                            logger.add_warning(msg="Ignored device",
                                               device=str(elm),
                                               device_class=elm.device_type.value)

    # Delete the external buses
    to_be_deleted = [grid.buses[e] for e in e_buses]
    for bus in to_be_deleted:
        grid.delete_bus(obj=bus, delete_associated=True)

    return grid, logger
