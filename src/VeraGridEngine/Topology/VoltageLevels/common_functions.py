# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations
from typing import Tuple, List
import VeraGridEngine.Devices as dev
from VeraGridEngine import Country, BusGraphicType, SwitchGraphicType
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Topology.VoltageLevels.single_bar import (
    create_single_bar,
    create_single_bar_with_disconnectors,
    create_single_bar_with_splitter,
    create_single_bar_with_splitter_with_disconnectors,
    create_single_bar_with_bypass,
    create_single_bar_with_bypass_with_disconnectors,
    connect_bar_segments)
from VeraGridEngine.Topology.VoltageLevels.double_bar import (
    create_double_bar,
    create_double_bar_with_disconnectors,
    create_double_bar_with_transference_bar,
    create_double_bar_with_transference_bar_with_disconnectors)
from VeraGridEngine.Topology.VoltageLevels.breaker_and_a_half import (
    create_breaker_and_a_half,
    create_breaker_and_a_half_with_disconnectors
)
from VeraGridEngine.Topology.VoltageLevels.ring import (
    create_ring,
    create_ring_with_disconnectors
)
from VeraGridEngine.enumerations import VoltageLevelTypes


def transform_bus_to_connectivity_grid(grid: MultiCircuit, busbar: dev.Bus) -> Tuple[List[dev.Bus], List[dev.Line]]:
    """
    Transform a BusBar into multiple Connectivity buses connected by branches.
    This is to be able to compute the power that passes through a busbar
    for specific busbar power studies

    :param grid: MultiCircuit instance
    :param busbar: the Bus object (BusGraphicType.BusBar) to transform
    :return: list of new Connectivity buses, list of branches between them
    """
    # Collect all connections (busbar side of each device)
    associated_branches, associated_injections = grid.get_bus_devices(bus=busbar)

    # Create a new Connectivity bus for each connection
    new_buses = []
    x_offset = 0
    for idx, elem in enumerate(associated_branches):
        new_bus = dev.Bus(
            name=f"{busbar.name}_conn_{idx}",
            substation=busbar.substation,
            Vnom=busbar.Vnom,
            voltage_level=busbar.voltage_level,
            xpos=busbar.x + x_offset,  # offset a bit to spread them visually
            ypos=busbar.y,
            country=busbar.country,
            graphic_type=BusGraphicType.Connectivity
        )
        grid.add_bus(new_bus)
        new_buses.append(new_bus)

        # Redirect the element to connect to this new bus instead of the busbar
        if elem.bus_from == busbar:
            elem.bus_from = new_bus

        elif elem.bus_to == busbar:
            elem.bus_to = new_bus

        x_offset += 100

    for idx, elem in enumerate(associated_injections):
        new_bus = dev.Bus(
            name=f"{busbar.name}_conn_{idx}",
            substation=busbar.substation,
            Vnom=busbar.Vnom,
            voltage_level=busbar.voltage_level,
            xpos=busbar.x + x_offset,  # offset a bit to spread them visually
            ypos=busbar.y,
            country=busbar.country,
            graphic_type=BusGraphicType.Connectivity
        )
        grid.add_bus(new_bus)
        new_buses.append(new_bus)

        # Redirect the element to connect to this new bus instead of the busbar
        elem.bus = new_bus

        x_offset += 100

    # Electrically tie all new buses with line branches
    new_branches = list()
    for i in range(len(new_buses) - 1):
        ln = dev.Line(
            name=f"{busbar.name}_backbone_{i}",
            bus_from=new_buses[i],
            bus_to=new_buses[i + 1],
        )
        grid.add_line(ln)
        new_branches.append(ln)

    # Remove the original busbar
    grid.delete_bus(busbar)

    return new_buses, new_branches


def transform_bus_into_voltage_level(
        grid: MultiCircuit,
        bus: dev.Bus,
        vl_type=VoltageLevelTypes.SingleBar,
        add_disconnectors: bool = False,
        bar_by_segments: bool = False
) -> List[dev.Bus]:
    """
    Transform a bus into a voltage level
    :param grid: MultiCircuit to add devices to
    :param bus: Bus device to transform
    :param vl_type: VoltageLevelTypes
    :param add_disconnectors: add voltage level disconnectors?
    :param bar_by_segments: Have the bar with connectivities and impedances instead of a single bus-bar?
    :return: List of all voltage level buses
    """

    # get the associations of the bus
    associated_branches, associated_injections = grid.get_bus_devices(bus=bus)

    # compute the number of bays (positions)
    n_bays = len(associated_branches) + len(associated_injections)
    all_buses: List[dev.Bus] = list()

    if vl_type == VoltageLevelTypes.SingleBar:

        if add_disconnectors:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_single_bar_with_disconnectors(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                bar_by_segments=bar_by_segments,
                offset_x=bus.x,
                offset_y=bus.y,
            )
        else:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_single_bar(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                bar_by_segments=bar_by_segments,
                offset_x=bus.x,
                offset_y=bus.y,
            )

    elif vl_type == VoltageLevelTypes.SingleBarWithBypass:

        if add_disconnectors:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_single_bar_with_bypass_with_disconnectors(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                offset_x=bus.x,
                offset_y=bus.y,
            )
        else:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_single_bar_with_bypass(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                offset_x=bus.x,
                offset_y=bus.y,
            )

    elif vl_type == VoltageLevelTypes.SingleBarWithSplitter:

        if add_disconnectors:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_single_bar_with_splitter_with_disconnectors(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                offset_x=bus.x,
                offset_y=bus.y,
            )
        else:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_single_bar_with_splitter(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                offset_x=bus.x,
                offset_y=bus.y,
            )

    elif vl_type == VoltageLevelTypes.DoubleBar:

        if add_disconnectors:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_double_bar_with_disconnectors(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                offset_x=bus.x,
                offset_y=bus.y,
            )
        else:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_double_bar(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                offset_x=bus.x,
                offset_y=bus.y,
            )

    elif vl_type == VoltageLevelTypes.DoubleBarWithBypass:
        # TODO: Implement
        return all_buses

    elif vl_type == VoltageLevelTypes.DoubleBarWithTransference:

        if add_disconnectors:
            (vl, conn_buses, all_buses,
             offset_total_x, offset_total_y) = create_double_bar_with_transference_bar_with_disconnectors(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                offset_x=bus.x,
                offset_y=bus.y,
            )
        else:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_double_bar_with_transference_bar(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                offset_x=bus.x,
                offset_y=bus.y,
            )

    elif vl_type == VoltageLevelTypes.DoubleBarDuplex:
        # TODO: Implement
        return all_buses

    elif vl_type == VoltageLevelTypes.Ring:

        if add_disconnectors:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_ring_with_disconnectors(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                offset_x=bus.x,
                offset_y=bus.y,
            )
        else:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_ring(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                offset_x=bus.x,
                offset_y=bus.y,
            )

    elif vl_type == VoltageLevelTypes.BreakerAndAHalf:

        if add_disconnectors:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_breaker_and_a_half_with_disconnectors(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                offset_x=bus.x,
                offset_y=bus.y,
            )
        else:
            vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_breaker_and_a_half(
                name=bus.name,
                grid=grid,
                n_bays=n_bays,
                v_nom=bus.Vnom,
                substation=bus.substation,
                country=bus.country,
                offset_x=bus.x,
                offset_y=bus.y,
            )

    else:
        print(f"{vl_type} not implemented :/")
        return all_buses

    # re-connect the branches and injections to the new position-buses
    j = 0
    for elem in associated_branches:
        if elem.bus_from == bus:
            elem.bus_from = conn_buses[j]

        elif elem.bus_to == bus:
            elem.bus_to = conn_buses[j]

        j += 1

    for elem in associated_injections:
        elem.bus = conn_buses[j]
        j += 1

    return all_buses


def create_substation(
        grid: MultiCircuit,
        se_name: str,
        se_code: str,
        lat: float,
        lon: float,
        vl_templates: List[dev.VoltageLevelTemplate]
) -> Tuple[dev.Substation, List[dev.VoltageLevel]]:
    """
    Create a complete substation
    :param grid: MultiCircuit instance
    :param se_name: Substation name
    :param se_code: Substation code
    :param lat: Latitude
    :param lon: Longitude
    :param vl_templates: List of VoltageLevelTemplates to convert
    :return: se_object, [vl list]
    """
    # create the SE
    se_object = dev.Substation(name=se_name,
                               code=se_code,
                               latitude=lat,
                               longitude=lon)

    grid.add_substation(obj=se_object)
    # substation_graphics = self.add_api_substation(api_object=se_object, lat=lat, lon=lon)

    voltage_levels = list()

    offset_x = 0
    offset_y = 0
    for vl_template in vl_templates:

        if vl_template.vl_type == VoltageLevelTypes.SingleBar:

            if vl_template.add_disconnectors:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_single_bar_with_disconnectors(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            else:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_single_bar(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )

            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        elif vl_template.vl_type == VoltageLevelTypes.SingleBarWithBypass:

            if vl_template.add_disconnectors:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_single_bar_with_bypass_with_disconnectors(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            else:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_single_bar_with_bypass(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )

            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        elif vl_template.vl_type == VoltageLevelTypes.SingleBarWithSplitter:

            if vl_template.add_disconnectors:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_single_bar_with_splitter_with_disconnectors(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            else:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_single_bar_with_splitter(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )

            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        elif vl_template.vl_type == VoltageLevelTypes.DoubleBar:

            if vl_template.add_disconnectors:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_double_bar_with_disconnectors(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            else:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_double_bar(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )

            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        elif vl_template.vl_type == VoltageLevelTypes.DoubleBarWithBypass:
            # TODO: Implement
            pass

        elif vl_template.vl_type == VoltageLevelTypes.DoubleBarWithTransference:

            if vl_template.add_disconnectors:
                (vl, conn_buses, all_buses,
                 offset_total_x, offset_total_y) = create_double_bar_with_transference_bar_with_disconnectors(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            else:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_double_bar_with_transference_bar(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )

            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        elif vl_template.vl_type == VoltageLevelTypes.DoubleBarDuplex:
            # TODO: Implement
            pass

        elif vl_template.vl_type == VoltageLevelTypes.Ring:

            if vl_template.add_disconnectors:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_ring_with_disconnectors(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            else:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_ring(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )

            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        elif vl_template.vl_type == VoltageLevelTypes.BreakerAndAHalf:

            if vl_template.add_disconnectors:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_breaker_and_a_half_with_disconnectors(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            else:
                vl, conn_buses, all_buses, offset_total_x, offset_total_y = create_breaker_and_a_half(
                    name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                    grid=grid,
                    n_bays=vl_template.n_bays,
                    v_nom=vl_template.voltage,
                    substation=se_object,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )

            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        else:
            print(f"{vl_template.vl_type} not implemented :/")

    return se_object, voltage_levels
