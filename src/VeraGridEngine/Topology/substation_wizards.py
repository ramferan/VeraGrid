# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple, List
import math
import VeraGridEngine.Devices as dev
from VeraGridEngine import Country, BusGraphicType, SwitchGraphicType
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.enumerations import SubstationTypes


def create_single_bar(name, grid: MultiCircuit, n_bays: int, v_nom: float,
                      substation: dev.Substation, country: Country = None,
                      include_disconnectors: bool = True,
                      offset_x=0, offset_y=0) -> Tuple[dev.VoltageLevel, int, int]:
    """

    :param name:
    :param grid:
    :param n_bays:
    :param v_nom:
    :param substation:
    :param country:
    :param include_disconnectors:
    :param offset_x:
    :param offset_y:
    :return:
    """

    vl = dev.VoltageLevel(name=name, substation=substation, Vnom=v_nom)
    grid.add_voltage_level(vl)

    bus_width = 120
    x_dist = bus_width * 2
    y_dist = bus_width * 1.5
    l_x_pos = []
    l_y_pos = []

    if include_disconnectors:
        width = n_bays * bus_width + bus_width * 2 + (x_dist - bus_width) * (n_bays - 1)
        bar = dev.Bus(name=f"{name} bar", substation=substation, Vnom=v_nom, voltage_level=vl, width=width,
                      xpos=offset_x - bus_width, ypos=offset_y + y_dist * 3, country=country,
                      graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar)
        l_x_pos.append(bar.x)
        l_y_pos.append(bar.y)

        for i in range(n_bays):
            bus1 = dev.Bus(name=f"{name}_bay_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y + 0, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)

            bus2 = dev.Bus(name=f"LineBus2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y + y_dist, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)

            bus3 = dev.Bus(name=f"LineBus3_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 2, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)

            dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
            cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus2, bus_to=bus3, graphic_type=SwitchGraphicType.CircuitBreaker)
            dis2 = dev.Switch(name=f"Dis2_{i}", bus_from=bar, bus_to=bus3, graphic_type=SwitchGraphicType.Disconnector)

            grid.add_bus(bus1)
            l_x_pos.append(bus1.x)
            l_y_pos.append(bus1.y)

            grid.add_bus(bus2)
            l_x_pos.append(bus2.x)
            l_y_pos.append(bus2.y)

            grid.add_bus(bus3)
            l_x_pos.append(bus3.x)
            l_y_pos.append(bus3.y)

            grid.add_switch(dis1)
            grid.add_switch(cb1)
            grid.add_switch(dis2)

        # for i in range(n_trafos):
        #     bus1 = dev.Bus(name=f"{name}_trafo_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 5, width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #
        #     bus2 = dev.Bus(name=f"trafo2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 4, width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #
        #     dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bar, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
        #     cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.CircuitBreaker)
        #
        #     grid.add_bus(bus1)
        #     l_x_pos.append(bus1.x)
        #     l_y_pos.append(bus1.y)
        #
        #     grid.add_bus(bus2)
        #     l_x_pos.append(bus2.x)
        #     l_y_pos.append(bus2.y)
        #
        #     grid.add_switch(dis1)
        #     grid.add_switch(cb1)

    else:
        width = n_bays * bus_width + bus_width * 2 + (x_dist - bus_width) * (n_bays - 1)
        bar = dev.Bus(name=f"{name} bar", substation=substation, Vnom=v_nom, voltage_level=vl, width=width,
                      xpos=offset_x - bus_width, ypos=offset_y + y_dist, country=country,
                      graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar)
        l_x_pos.append(bar.x)
        l_y_pos.append(bar.y)

        for i in range(n_bays):
            bus1 = dev.Bus(name=f"{name}_bay_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)
            cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus1, bus_to=bar, graphic_type=SwitchGraphicType.CircuitBreaker)

            grid.add_bus(bus1)
            l_x_pos.append(bus1.x)
            l_y_pos.append(bus1.y)

            grid.add_switch(cb1)

        # for i in range(n_trafos):
        #     bus1 = dev.Bus(name=f"{name}_trafo_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 2, width=bus_width, country=country,
        #                    graphic_type=BusGraphicType.Connectivity)
        #     cb1 = dev.Switch(name=f"CB_{i}", bus_from=bar, bus_to=bus1, graphic_type=SwitchGraphicType.CircuitBreaker)
        #
        #     grid.add_bus(bus1)
        #     l_x_pos.append(bus1.x)
        #     l_y_pos.append(bus1.y)
        #
        #     grid.add_switch(cb1)

    offset_total_x = max(l_x_pos, default=0) + x_dist
    offset_total_y = max(l_y_pos, default=0) + y_dist

    return vl, offset_total_x, offset_total_y


def create_single_bar_with_bypass(name, grid: MultiCircuit, n_bays: int, v_nom: float,
                                  substation: dev.Substation, country: Country = None,
                                  include_disconnectors: bool = True,
                                  offset_x=0, offset_y=0) -> Tuple[dev.VoltageLevel, int, int]:
    """

    :param name:
    :param grid:
    :param n_bays:
    :param v_nom:
    :param substation:
    :param country:
    :param include_disconnectors:
    :param offset_x:
    :param offset_y:
    :return:
    """

    vl = dev.VoltageLevel(name=name, substation=substation, Vnom=v_nom)
    grid.add_voltage_level(vl)

    bus_width = 120
    x_dist = bus_width * 2
    y_dist = bus_width * 1.5
    l_x_pos = []
    l_y_pos = []

    if include_disconnectors:
        width = n_bays * bus_width + bus_width * 2 + (x_dist - bus_width) * (n_bays - 1)
        bar = dev.Bus(f"{name} bar", substation=substation, Vnom=v_nom, voltage_level=vl,
                      width=width,
                      xpos=offset_x - bus_width, ypos=offset_y + y_dist * 3, country=country,
                      graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar)
        l_x_pos.append(bar.x)
        l_y_pos.append(bar.y)

        for i in range(n_bays):
            bus1 = dev.Bus(f"{name}_bay_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y + 0, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus2 = dev.Bus(f"BayBus2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y + y_dist, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus3 = dev.Bus(f"BayBus3_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 2, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)
            dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
            cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus2, bus_to=bus3, graphic_type=SwitchGraphicType.CircuitBreaker)
            dis2 = dev.Switch(name=f"Dis2_{i}", bus_from=bus3, bus_to=bar, graphic_type=SwitchGraphicType.Disconnector)
            bypass_dis = dev.Switch(name=f"Bypass_Dis_{i}", bus_from=bus1, bus_to=bar,
                                    graphic_type=SwitchGraphicType.Disconnector)

            grid.add_bus(bus1)
            l_x_pos.append(bus1.x)
            l_y_pos.append(bus1.y)

            grid.add_bus(bus2)
            l_x_pos.append(bus2.x)
            l_y_pos.append(bus2.y)

            grid.add_bus(bus3)
            l_x_pos.append(bus3.x)
            l_y_pos.append(bus3.y)

            grid.add_switch(dis1)
            grid.add_switch(cb1)
            grid.add_switch(dis2)
            grid.add_switch(bypass_dis)

        # for i in range(n_trafos):
        #     bus1 = dev.Bus(f"{name}_trafo_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 5, width=bus_width, country=country,
        #                    graphic_type=BusGraphicType.Connectivity)
        #     bus2 = dev.Bus(f"trafo2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 4, width=bus_width, country=country,
        #                    graphic_type=BusGraphicType.Connectivity)
        #     dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bar, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
        #     cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus2, bus_to=bus1, graphic_type=SwitchGraphicType.CircuitBreaker)
        #     bypass_dis = dev.Switch(name=f"Bypass_Dis_{i}", bus_from=bus1, bus_to=bar,
        #                             graphic_type=SwitchGraphicType.Disconnector)
        #
        #     grid.add_bus(bus1)
        #     l_x_pos.append(bus1.x)
        #     l_y_pos.append(bus1.y)
        #
        #     grid.add_bus(bus2)
        #     l_x_pos.append(bus2.x)
        #     l_y_pos.append(bus2.y)
        #
        #     grid.add_switch(dis1)
        #     grid.add_switch(cb1)
        #     grid.add_switch(bypass_dis)
    else:
        width = n_bays * bus_width + bus_width * 2 + (x_dist - bus_width) * (n_bays - 1)
        bar = dev.Bus(name=f"{name} bar",
                      substation=substation, Vnom=v_nom, voltage_level=vl,
                      width=width,
                      xpos=offset_x - bus_width,
                      ypos=offset_y + y_dist * 1,
                      country=country,
                      graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar)
        l_x_pos.append(bar.x)
        l_y_pos.append(bar.y)

        for i in range(n_bays):
            bus1 = dev.Bus(name=f"{name}_bay_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)

            cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus1, bus_to=bar, graphic_type=SwitchGraphicType.CircuitBreaker)
            bypass_dis = dev.Switch(name=f"Bypass_Dis_{i}", bus_from=bus1, bus_to=bar,
                                    graphic_type=SwitchGraphicType.Disconnector)

            grid.add_bus(bus1)
            l_x_pos.append(bus1.x)
            l_y_pos.append(bus1.y)

            grid.add_switch(cb1)
            grid.add_switch(bypass_dis)

        # for i in range(n_trafos):
        #     bus1 = dev.Bus(f"{name}_trafo_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 2, width=bus_width, country=country,
        #                    graphic_type=BusGraphicType.Connectivity)
        #     cb1 = dev.Switch(name=f"CB_{i}", bus_from=bar, bus_to=bus1, graphic_type=SwitchGraphicType.CircuitBreaker)
        #     bypass_dis = dev.Switch(name=f"Bypass_Dis_{i}", bus_from=bus1, bus_to=bar,
        #                             graphic_type=SwitchGraphicType.Disconnector)
        #
        #     grid.add_bus(bus1)
        #     l_x_pos.append(bus1.x)
        #     l_y_pos.append(bus1.y)
        #
        #     grid.add_switch(cb1)
        #     grid.add_switch(bypass_dis)

    offset_total_x = max(l_x_pos, default=0) + x_dist
    offset_total_y = max(l_y_pos, default=0) + y_dist

    return vl, offset_total_x, offset_total_y


def create_single_bar_with_splitter(name, grid: MultiCircuit, n_bays: int, v_nom: float,
                                    substation: dev.Substation, country: Country = None,
                                    include_disconnectors: bool = True,
                                    offset_x=0, offset_y=0) -> Tuple[dev.VoltageLevel, int, int]:
    """

    :param name:
    :param grid:
    :param n_bays:
    :param v_nom:
    :param substation:
    :param country:
    :param include_disconnectors:
    :param offset_x:
    :param offset_y:
    :return:
    """

    vl = dev.VoltageLevel(name=name, substation=substation, Vnom=v_nom)
    grid.add_voltage_level(vl)

    bus_width = 120
    x_dist = bus_width * 2
    y_dist = bus_width * 1.5
    bar_2_x_offset = bus_width * 1.2
    bar_2_y_offset = bus_width * 1.2
    l_x_pos = []
    l_y_pos = []

    n_bays_bar_1 = n_bays // 2
    n_bays_bar_2 = n_bays - n_bays_bar_1

    width_bar_1 = n_bays_bar_1 * bus_width + bus_width * 2 + (x_dist - bus_width) * (n_bays_bar_1 - 1)
    width_bar_2 = n_bays_bar_2 * bus_width + bus_width * 2 + (x_dist - bus_width) * (n_bays_bar_2 - 1)

    if include_disconnectors:

        bar1 = dev.Bus(f"{name} bar 1", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=width_bar_1, xpos=offset_x - bus_width, ypos=offset_y + y_dist * 3, country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar1)
        l_x_pos.append(bar1.x)
        l_y_pos.append(bar1.y)

        bar2 = dev.Bus(f"{name} bar 2", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=width_bar_2, xpos=offset_x + width_bar_1 + bar_2_x_offset,
                       ypos=offset_y + y_dist * 3 + bar_2_y_offset,
                       country=country, graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar2)
        l_x_pos.append(bar2.x)
        l_y_pos.append(bar2.y)

        cb_bars = dev.Switch(name=f"CB_bars", bus_from=bar1, bus_to=bar2, graphic_type=SwitchGraphicType.CircuitBreaker)
        grid.add_switch(cb_bars)

        for i in range(n_bays):
            if i < n_bays_bar_1:
                bar = bar1
                x_offset = 0
                y_offset = 0
            else:
                bar = bar2
                x_offset = bar_2_x_offset + 2 * bus_width
                y_offset = bar_2_y_offset

            bus1 = dev.Bus(f"{name}_bay_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist + x_offset, ypos=offset_y + y_offset, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus2 = dev.Bus(f"LineBus2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist + x_offset, ypos=offset_y + y_dist + y_offset, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus3 = dev.Bus(f"LineBus3_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist + x_offset, ypos=offset_y + y_dist * 2 + y_offset,
                           width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)
            dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
            cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus2, bus_to=bus3, graphic_type=SwitchGraphicType.CircuitBreaker)
            dis2 = dev.Switch(name=f"Dis2_{i}", bus_from=bar, bus_to=bus3, graphic_type=SwitchGraphicType.Disconnector)

            grid.add_bus(bus1)
            l_x_pos.append(bus1.x)
            l_y_pos.append(bus1.y)

            grid.add_bus(bus2)
            l_x_pos.append(bus2.x)
            l_y_pos.append(bus2.y)

            grid.add_bus(bus3)
            l_x_pos.append(bus3.x)
            l_y_pos.append(bus3.y)

            grid.add_switch(dis1)
            grid.add_switch(cb1)
            grid.add_switch(dis2)

        # for i in range(n_trafos):
        #     if i < n_trafos_bar_1:
        #         bar = bar1
        #         x_offset = 0
        #         y_offset = 0
        #     else:
        #         bar = bar2
        #         x_offset = bar_2_x_offset + 2 * bus_width
        #         y_offset = bar_2_y_offset
        # 
        #     bus1 = dev.Bus(f"{name}_trafo_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist + x_offset, ypos=offset_y + y_dist * 5 + y_offset,
        #                    width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #     bus2 = dev.Bus(f"trafo2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist + x_offset, ypos=offset_y + y_dist * 4 + y_offset,
        #                    width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #     dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bar, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
        #     cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.CircuitBreaker)
        # 
        #     grid.add_bus(bus1)
        #     l_x_pos.append(bus1.x)
        #     l_y_pos.append(bus1.y)
        # 
        #     grid.add_bus(bus2)
        #     l_x_pos.append(bus2.x)
        #     l_y_pos.append(bus2.y)
        # 
        #     grid.add_switch(dis1)
        #     grid.add_switch(cb1)
    else:

        bar1 = dev.Bus(f"{name} bar 1", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=width_bar_1, xpos=offset_x - bus_width, ypos=offset_y + y_dist * 1, country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar1)
        l_x_pos.append(bar1.x)
        l_y_pos.append(bar1.y)

        bar2 = dev.Bus(f"{name} bar 2", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=width_bar_2, xpos=offset_x + width_bar_1 + bar_2_x_offset,
                       ypos=offset_y + y_dist * 1 + bar_2_y_offset,
                       country=country, graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar2)
        l_x_pos.append(bar2.x)
        l_y_pos.append(bar2.y)

        cb_bars = dev.Switch(name=f"CB_bars", bus_from=bar1, bus_to=bar2, graphic_type=SwitchGraphicType.CircuitBreaker)
        grid.add_switch(cb_bars)

        for i in range(n_bays):
            if i < n_bays_bar_1:
                bar = bar1
                x_offset = 0
                y_offset = 0
            else:
                bar = bar2
                x_offset = bar_2_x_offset + 2 * bus_width
                y_offset = bar_2_y_offset

            bus1 = dev.Bus(f"{name}_bay_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist + x_offset, ypos=offset_y + y_dist * 0 + y_offset,
                           width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)
            cb1 = dev.Switch(name=f"CB_{i}", bus_from=bar, bus_to=bus1, graphic_type=SwitchGraphicType.CircuitBreaker)

            grid.add_bus(bus1)
            l_x_pos.append(bus1.x)
            l_y_pos.append(bus1.y)

            grid.add_switch(cb1)

        # for i in range(n_trafos):
        #     if i < n_trafos_bar_1:
        #         bar = bar1
        #         x_offset = 0
        #         y_offset = 0
        #     else:
        #         bar = bar2
        #         x_offset = bar_2_x_offset + 2 * bus_width
        #         y_offset = bar_2_y_offset
        # 
        #     bus1 = dev.Bus(f"{name}_trafo_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist + x_offset, ypos=offset_y + y_dist * 2 + y_offset,
        #                    width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #     cb1 = dev.Switch(name=f"CB_{i}", bus_from=bar, bus_to=bus1, graphic_type=SwitchGraphicType.CircuitBreaker)
        # 
        #     grid.add_bus(bus1)
        #     l_x_pos.append(bus1.x)
        #     l_y_pos.append(bus1.y)
        # 
        #     grid.add_switch(cb1)

    offset_total_x = max(l_x_pos, default=0) + x_dist
    offset_total_y = max(l_y_pos, default=0) + y_dist

    return vl, offset_total_x, offset_total_y


def create_double_bar(name, grid: MultiCircuit, n_bays: int, v_nom: float,
                      substation: dev.Substation, country: Country = None,
                      include_disconnectors: bool = True,
                      offset_x=0, offset_y=0) -> Tuple[dev.VoltageLevel, int, int]:
    """

    :param name:
    :param grid:
    :param n_bays:
    :param v_nom:
    :param substation:
    :param country:
    :param include_disconnectors:
    :param offset_x:
    :param offset_y:
    :return:
    """

    vl = dev.VoltageLevel(name=name, substation=substation, Vnom=v_nom)
    grid.add_voltage_level(vl)

    bus_width = 120
    x_dist = bus_width * 2
    y_dist = bus_width * 1.5
    l_x_pos = []
    l_y_pos = []

    if include_disconnectors:
        width = (n_bays + 1) * bus_width + bus_width * 2 + (x_dist - bus_width) * n_bays
        bar1 = dev.Bus(name=f"{name} bar1", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=width,
                       xpos=offset_x - bus_width, ypos=offset_y + y_dist * 3, country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar1)
        l_x_pos.append(bar1.x)
        l_y_pos.append(bar1.y)

        bar2 = dev.Bus(name=f"{name} bar2", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=width,
                       xpos=offset_x - bus_width, ypos=offset_y + y_dist * 4, country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar2)
        l_x_pos.append(bar2.x)
        l_y_pos.append(bar2.y)

        for i in range(n_bays):
            bus1 = dev.Bus(f"{name}_bay_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus2 = dev.Bus(f"LineBus2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y + y_dist, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus3 = dev.Bus(f"LineBus3_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 2, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)
            dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
            cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus2, bus_to=bus3, graphic_type=SwitchGraphicType.CircuitBreaker)
            dis2 = dev.Switch(name=f"Dis2_{i}", bus_from=bar1, bus_to=bus3, graphic_type=SwitchGraphicType.Disconnector)
            dis3 = dev.Switch(name=f"Dis3_{i}", bus_from=bar2, bus_to=bus3, graphic_type=SwitchGraphicType.Disconnector)

            grid.add_bus(bus1)
            l_x_pos.append(bus1.x)
            l_y_pos.append(bus1.y)

            grid.add_bus(bus2)
            l_x_pos.append(bus2.x)
            l_y_pos.append(bus2.y)

            grid.add_bus(bus3)
            l_x_pos.append(bus3.x)
            l_y_pos.append(bus3.y)

            grid.add_switch(dis1)
            grid.add_switch(cb1)
            grid.add_switch(dis2)
            grid.add_switch(dis3)

        # for i in range(n_trafos):
        #     bus1 = dev.Bus(f"{name}_trafo_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 6, width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #     bus2 = dev.Bus(f"trafo2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 5, width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #     dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bar1, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
        #     dis2 = dev.Switch(name=f"Dis2_{i}", bus_from=bar2, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
        #     cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.CircuitBreaker)
        # 
        #     grid.add_bus(bus1)
        #     l_x_pos.append(bus1.x)
        #     l_y_pos.append(bus1.y)
        # 
        #     grid.add_bus(bus2)
        #     l_x_pos.append(bus2.x)
        #     l_y_pos.append(bus2.y)
        # 
        #     grid.add_switch(dis1)
        #     grid.add_switch(dis2)
        #     grid.add_switch(cb1)

        # coupling
        bus1 = dev.Bus(f"{name}_coupling_bar1", substation=substation, Vnom=v_nom, voltage_level=vl,
                       xpos=offset_x + n_bays * x_dist,
                       ypos=offset_y + y_dist * 3.6,
                       width=bus_width,
                       country=country,
                       graphic_type=BusGraphicType.Connectivity)
        bus2 = dev.Bus(f"{name}_coupling_bar2", substation=substation, Vnom=v_nom, voltage_level=vl,
                       xpos=offset_x + n_bays * x_dist + x_dist * 0.5,
                       ypos=offset_y + y_dist * 3.6,
                       width=bus_width,
                       country=country,
                       graphic_type=BusGraphicType.Connectivity)
        dis1 = dev.Switch(name="Dis_bar1", bus_from=bar1, bus_to=bus1, graphic_type=SwitchGraphicType.Disconnector)
        dis2 = dev.Switch(name="Dis_bar2", bus_from=bar2, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
        cb1 = dev.Switch(name="CB_coupling", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.CircuitBreaker)

        grid.add_bus(bus1)
        l_x_pos.append(bus1.x)
        l_y_pos.append(bus1.y)

        grid.add_bus(bus2)
        l_x_pos.append(bus2.x)
        l_y_pos.append(bus2.y)

        grid.add_switch(dis1)
        grid.add_switch(dis2)
        grid.add_switch(cb1)

    else:

        bar1 = dev.Bus(f"{name} bar1", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=(n_bays + 1) * bus_width + bus_width * 2 + (x_dist - bus_width) * n_bays,
                       xpos=offset_x - bus_width,
                       ypos=offset_y + y_dist * 2, country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar1)
        l_x_pos.append(bar1.x)
        l_y_pos.append(bar1.y)

        bar2 = dev.Bus(f"{name} bar2", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=(n_bays + 1) * bus_width + bus_width * 2 + (x_dist - bus_width) * n_bays,
                       xpos=offset_x - bus_width,
                       ypos=offset_y + y_dist * 3,
                       country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar2)
        l_x_pos.append(bar2.x)
        l_y_pos.append(bar2.y)

        for i in range(n_bays):
            bus1 = dev.Bus(f"{name}_bay_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 0, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)

            bus2 = dev.Bus(f"LineBus2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist, ypos=offset_y + y_dist, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)

            cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.CircuitBreaker)
            dis1 = dev.Switch(name=f"Dis2_{i}", bus_from=bar1, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
            dis2 = dev.Switch(name=f"Dis3_{i}", bus_from=bar2, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)

            grid.add_bus(bus1)
            l_x_pos.append(bus1.x)
            l_y_pos.append(bus1.y)

            grid.add_bus(bus2)
            l_x_pos.append(bus2.x)
            l_y_pos.append(bus2.y)

            grid.add_switch(cb1)
            grid.add_switch(dis1)  # this disconnectors must be included to respect the SE geometry
            grid.add_switch(dis2)  # this disconnectors must be included to respect the SE geometry

        # for i in range(n_trafos):
        #     bus1 = dev.Bus(f"{name}_trafo_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 5, width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #     bus2 = dev.Bus(f"trafo2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 4, width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #     dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bar1, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
        #     dis2 = dev.Switch(name=f"Dis2_{i}", bus_from=bar2, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
        #     cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.CircuitBreaker)
        # 
        #     grid.add_bus(bus1)
        #     l_x_pos.append(bus1.x)
        #     l_y_pos.append(bus1.y)
        # 
        #     grid.add_bus(bus2)
        #     l_x_pos.append(bus2.x)
        #     l_y_pos.append(bus2.y)
        # 
        #     grid.add_switch(dis1)  # this disconnectors must be included to respect the SE geometry
        #     grid.add_switch(dis2)  # this disconnectors must be included to respect the SE geometry
        #     grid.add_switch(cb1)

        # coupling
        cb1 = dev.Switch(name="CB_coupling", bus_from=bar1, bus_to=bar2, graphic_type=SwitchGraphicType.CircuitBreaker)
        grid.add_switch(cb1)

    offset_total_x = max(l_x_pos, default=0) + x_dist
    offset_total_y = max(l_y_pos, default=0) + y_dist

    return vl, offset_total_x, offset_total_y


def create_double_bar_with_transference_bar(name: str,
                                            grid: MultiCircuit,
                                            n_bays: int, v_nom: float,
                                            substation: dev.Substation, country: Country = None,
                                            include_disconnectors: bool = True,
                                            offset_x=0, offset_y=0) -> Tuple[dev.VoltageLevel, int, int]:
    """

    :param name:
    :param grid:
    :param n_bays:
    :param v_nom:
    :param substation:
    :param country:
    :param include_disconnectors:
    :param offset_x:
    :param offset_y:
    :return:
    """

    vl = dev.VoltageLevel(name=name, substation=substation, Vnom=v_nom)
    grid.add_voltage_level(vl)

    bus_width = 120
    x_dist = bus_width * 2
    y_dist = bus_width * 1.5
    l_x_pos = []
    l_y_pos = []

    if include_disconnectors:

        bar1 = dev.Bus(name=f"{name} bar1", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=(n_bays + 1) * bus_width + bus_width * 3 + (x_dist - bus_width) * n_bays,
                       xpos=offset_x - bus_width,
                       ypos=offset_y + y_dist * 3,
                       country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar1)
        l_x_pos.append(bar1.x)
        l_y_pos.append(bar1.y)

        bar2 = dev.Bus(name=f"{name} bar2", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=(n_bays + 1) * bus_width + bus_width * 3 + (x_dist - bus_width) * n_bays,
                       xpos=offset_x - bus_width,
                       ypos=offset_y + y_dist * 4, country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar2)
        l_x_pos.append(bar2.x)
        l_y_pos.append(bar2.y)

        transfer_bar = dev.Bus(name=f"{name} transfer bar", substation=substation, Vnom=v_nom, voltage_level=vl,
                               width=(n_bays + 1) * bus_width + bus_width * 3 + (x_dist - bus_width) * n_bays,
                               xpos=offset_x - bus_width, ypos=offset_y + y_dist * 5, country=country,
                               graphic_type=BusGraphicType.BusBar)
        grid.add_bus(transfer_bar)
        l_x_pos.append(transfer_bar.x)
        l_y_pos.append(transfer_bar.y)

        for i in range(n_bays):
            bus1 = dev.Bus(name=f"{name}_bay_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist + x_dist * 0.2,
                           ypos=offset_y, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)

            bus2 = dev.Bus(name=f"BayBus2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist - x_dist * 0.25,
                           ypos=offset_y + y_dist, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)

            bus3 = dev.Bus(name=f"BayBus3_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist - x_dist * 0.25,
                           ypos=offset_y + y_dist * 2, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)

            dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
            cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus2, bus_to=bus3, graphic_type=SwitchGraphicType.CircuitBreaker)
            dis2 = dev.Switch(name=f"Dis2_{i}", bus_from=bar1, bus_to=bus3, graphic_type=SwitchGraphicType.Disconnector)
            dis3 = dev.Switch(name=f"Dis3_{i}", bus_from=bar2, bus_to=bus3, graphic_type=SwitchGraphicType.Disconnector)
            dis4 = dev.Switch(name=f"Dis4_{i}", bus_from=bus1, bus_to=transfer_bar,
                              graphic_type=SwitchGraphicType.Disconnector)

            grid.add_bus(bus1)
            l_x_pos.append(bus1.x)
            l_y_pos.append(bus1.y)

            grid.add_bus(bus2)
            l_x_pos.append(bus2.x)
            l_y_pos.append(bus2.y)

            grid.add_bus(bus3)
            l_x_pos.append(bus3.x)
            l_y_pos.append(bus3.y)

            grid.add_switch(dis1)
            grid.add_switch(cb1)
            grid.add_switch(dis2)
            grid.add_switch(dis3)
            grid.add_switch(dis4)

        # for i in range(n_trafos):
        #     bus1 = dev.Bus(f"{name}_trafo_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 8, width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #     bus2 = dev.Bus(f"trafo2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 7, width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #     bus3 = dev.Bus(f"trafo3_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 6, width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #     dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.Disconnector)
        #     cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus2, bus_to=bus3, graphic_type=SwitchGraphicType.CircuitBreaker)
        #     dis2 = dev.Switch(name=f"Dis2_{i}", bus_from=bus3, bus_to=bar1, graphic_type=SwitchGraphicType.Disconnector)
        #     dis3 = dev.Switch(name=f"Dis3_{i}", bus_from=bus3, bus_to=bar2, graphic_type=SwitchGraphicType.Disconnector)
        #     dis4 = dev.Switch(name=f"Dis4_{i}", bus_from=bus1, bus_to=transfer_bar,
        #                       graphic_type=SwitchGraphicType.Disconnector)
        # 
        #     grid.add_bus(bus1)
        #     l_x_pos.append(bus1.x)
        #     l_y_pos.append(bus1.y)
        # 
        #     grid.add_bus(bus2)
        #     l_x_pos.append(bus2.x)
        #     l_y_pos.append(bus2.y)
        # 
        #     grid.add_bus(bus3)
        #     l_x_pos.append(bus3.x)
        #     l_y_pos.append(bus3.y)
        # 
        #     grid.add_switch(dis1)
        #     grid.add_switch(dis2)
        #     grid.add_switch(cb1)
        #     grid.add_switch(dis3)
        #     grid.add_switch(dis4)

        # coupling
        bus1 = dev.Bus(name=f"{name}_coupling_bar1", substation=substation, Vnom=v_nom, voltage_level=vl,
                       xpos=offset_x + n_bays * x_dist + x_dist * 0.25,
                       ypos=offset_y + y_dist * 3.6,
                       width=bus_width,
                       country=country,
                       graphic_type=BusGraphicType.Connectivity)

        bus2 = dev.Bus(name=f"{name}_coupling_bar2", substation=substation, Vnom=v_nom, voltage_level=vl,
                       xpos=offset_x + n_bays * x_dist + x_dist * 0.25,
                       ypos=offset_y + y_dist * 4.6,
                       width=bus_width,
                       country=country,
                       graphic_type=BusGraphicType.Connectivity)

        dis1 = dev.Switch(name="Dis_bar1", bus_from=bus1, bus_to=bar1, graphic_type=SwitchGraphicType.Disconnector)
        dis2 = dev.Switch(name="Dis_bar2", bus_from=bus1, bus_to=bar2, graphic_type=SwitchGraphicType.Disconnector)
        cb1 = dev.Switch(name="CB_coupling", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.CircuitBreaker)
        dis3 = dev.Switch(name="Dis_coupling", bus_from=bus2, bus_to=transfer_bar,
                          graphic_type=SwitchGraphicType.Disconnector)

        grid.add_bus(bus1)
        l_x_pos.append(bus1.x)
        l_y_pos.append(bus1.y)

        grid.add_bus(bus2)
        l_x_pos.append(bus2.x)
        l_y_pos.append(bus2.y)

        grid.add_switch(dis1)
        grid.add_switch(dis2)
        grid.add_switch(cb1)
        grid.add_switch(dis3)

    else:

        bar1 = dev.Bus(name=f"{name} bar1", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=(n_bays + 1) * bus_width + bus_width * 3 + (x_dist - bus_width) * n_bays,
                       xpos=offset_x - bus_width, ypos=offset_y + y_dist * 2, country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar1)
        l_x_pos.append(bar1.x)
        l_y_pos.append(bar1.y)

        bar2 = dev.Bus(name=f"{name} bar2", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=(n_bays + 1) * bus_width + bus_width * 3 + (x_dist - bus_width) * n_bays,
                       xpos=offset_x - bus_width, ypos=offset_y + y_dist * 3, country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar2)
        l_x_pos.append(bar2.x)
        l_y_pos.append(bar2.y)

        transfer_bar = dev.Bus(name=f"{name} transfer bar", substation=substation, Vnom=v_nom, voltage_level=vl,
                               width=(n_bays + 1) * bus_width + bus_width * 3 + (x_dist - bus_width) * n_bays,
                               xpos=offset_x - bus_width, ypos=offset_y + y_dist * 4, country=country,
                               graphic_type=BusGraphicType.BusBar)
        grid.add_bus(transfer_bar)
        l_x_pos.append(transfer_bar.x)
        l_y_pos.append(transfer_bar.y)

        for i in range(n_bays):
            bus1 = dev.Bus(name=f"{name}_bay_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist + x_dist * 0.1, ypos=offset_y, width=bus_width, country=country,
                           graphic_type=BusGraphicType.Connectivity)

            bus2 = dev.Bus(f"BayBus2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist - x_dist * 0.1, ypos=offset_y + y_dist, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)

            cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.CircuitBreaker)
            dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bus2, bus_to=bar1, graphic_type=SwitchGraphicType.Disconnector)
            dis2 = dev.Switch(name=f"Dis2_{i}", bus_from=bus2, bus_to=bar2, graphic_type=SwitchGraphicType.Disconnector)
            dis3 = dev.Switch(name=f"Dis3_{i}", bus_from=bus1, bus_to=transfer_bar,
                              graphic_type=SwitchGraphicType.Disconnector)

            grid.add_bus(bus1)
            l_x_pos.append(bus1.x)
            l_y_pos.append(bus1.y)

            grid.add_bus(bus2)
            l_x_pos.append(bus2.x)
            l_y_pos.append(bus2.y)

            grid.add_switch(cb1)
            grid.add_switch(dis1)  # this disconnector must be included to respect the SE geometry
            grid.add_switch(dis2)  # this disconnector must be included to respect the SE geometry
            grid.add_switch(dis3)  # this disconnector must be included to respect the SE geometry

        # for i in range(n_trafos):
        #     bus1 = dev.Bus(f"{name}_trafo_conn_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 6, width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #     bus2 = dev.Bus(f"trafo2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
        #                    xpos=offset_x + i * x_dist, ypos=offset_y + y_dist * 5, width=bus_width,
        #                    country=country, graphic_type=BusGraphicType.Connectivity)
        #     dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bus2, bus_to=bar1, graphic_type=SwitchGraphicType.Disconnector)
        #     dis2 = dev.Switch(name=f"Dis2_{i}", bus_from=bus2, bus_to=bar2, graphic_type=SwitchGraphicType.Disconnector)
        #     cb1 = dev.Switch(name=f"CB_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.CircuitBreaker)
        #     dis3 = dev.Switch(name=f"Dis3_{i}", bus_from=bus1, bus_to=transfer_bar,
        #                       graphic_type=SwitchGraphicType.Disconnector)
        # 
        #     grid.add_bus(bus1)
        #     l_x_pos.append(bus1.x)
        #     l_y_pos.append(bus1.y)
        # 
        #     grid.add_bus(bus2)
        #     l_x_pos.append(bus2.x)
        #     l_y_pos.append(bus2.y)
        # 
        #     grid.add_switch(dis1)
        #     grid.add_switch(dis2)  # this disconnector must be included to respect the SE geometry
        #     grid.add_switch(cb1)  # this disconnector must be included to respect the SE geometry
        #     grid.add_switch(dis3)  # this disconnector must be included to respect the SE geometry

        # coupling
        bus1 = dev.Bus(f"{name}_coupling_bar1", substation=substation, Vnom=v_nom, voltage_level=vl,
                       xpos=offset_x + n_bays * x_dist + x_dist * 0.25,
                       ypos=offset_y + y_dist * 3.6,
                       width=bus_width,
                       country=country,
                       graphic_type=BusGraphicType.Connectivity)
        dis1 = dev.Switch(name="Dis_bar1", bus_from=bus1, bus_to=bar1, graphic_type=SwitchGraphicType.Disconnector)
        dis2 = dev.Switch(name="Dis_bar2", bus_from=bus1, bus_to=bar2, graphic_type=SwitchGraphicType.Disconnector)
        cb1 = dev.Switch(name="CB_coupling", bus_from=bus1, bus_to=transfer_bar,
                         graphic_type=SwitchGraphicType.CircuitBreaker)

        grid.add_bus(bus1)
        l_x_pos.append(bus1.x)
        l_y_pos.append(bus1.y)

        grid.add_switch(dis1)  # this disconnector must be included to respect the SE geometry
        grid.add_switch(dis2)  # this disconnector must be included to respect the SE geometry
        grid.add_switch(cb1)

    offset_total_x = max(l_x_pos, default=0) + x_dist
    offset_total_y = max(l_y_pos, default=0) + y_dist

    return vl, offset_total_x, offset_total_y


def create_breaker_and_a_half(name, grid: MultiCircuit, n_bays: int, v_nom: float,
                              substation: dev.Substation, country: Country = None,
                              include_disconnectors: bool = True,
                              offset_x=0, offset_y=0) -> Tuple[dev.VoltageLevel, int, int]:
    """

    :param name:
    :param grid:
    :param n_bays:
    :param v_nom:
    :param substation:
    :param country:
    :param include_disconnectors:
    :param offset_x:
    :param offset_y:
    :return:
    """

    vl = dev.VoltageLevel(name=name, substation=substation, Vnom=v_nom)
    grid.add_voltage_level(vl)

    bus_width = 120
    x_dist = bus_width * 2
    y_dist = bus_width * 1.5
    l_x_pos = []
    l_y_pos = []

    if include_disconnectors:

        bar1 = dev.Bus(f"{name} bar1", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=(n_bays + n_bays % 2) * x_dist, xpos=offset_x - x_dist, ypos=offset_y, country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar1)
        l_x_pos.append(bar1.x)
        l_y_pos.append(bar1.y)

        bar2 = dev.Bus(f"{name} bar2", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=(n_bays + n_bays % 2) * x_dist, xpos=offset_x - x_dist, ypos=offset_y + y_dist * 9,
                       country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar2)
        l_x_pos.append(bar2.x)
        l_y_pos.append(bar2.y)

        for i in range(0, n_bays, 2):
            bus1 = dev.Bus(f"LineBus1_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist - bus_width / 2, ypos=offset_y + y_dist, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus2 = dev.Bus(f"LineBus2_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist - bus_width / 2, ypos=offset_y + y_dist * 2, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus3 = dev.Bus(f"LineBus3_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist - bus_width / 2, ypos=offset_y + y_dist * 3, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus_line_connection_1 = dev.Bus(f"{name}_bay_conn_{i}", substation=substation, Vnom=v_nom,
                                            voltage_level=vl,
                                            xpos=offset_x + (i - 1) * x_dist - bus_width / 2,
                                            ypos=offset_y + y_dist * 2.7, width=0,
                                            country=country,
                                            graphic_type=BusGraphicType.Connectivity)
            bus4 = dev.Bus(f"LineBus4_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist - bus_width / 2, ypos=offset_y + y_dist * 4, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus5 = dev.Bus(f"LineBus4_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist - bus_width / 2, ypos=offset_y + y_dist * 5, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus6 = dev.Bus(f"LineBus6_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist - bus_width / 2, ypos=offset_y + y_dist * 6, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus_line_connection_2 = dev.Bus(f"{name}_bay_conn_{i + 1}", substation=substation, Vnom=v_nom,
                                            voltage_level=vl,
                                            xpos=offset_x + (i - 1) * x_dist - bus_width / 2,
                                            ypos=offset_y + y_dist * 5.7, width=0,
                                            country=country,
                                            graphic_type=BusGraphicType.Connectivity)
            bus7 = dev.Bus(f"LineBus7_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist - bus_width / 2, ypos=offset_y + y_dist * 7, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus8 = dev.Bus(f"LineBus8_{i}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=offset_x + i * x_dist - bus_width / 2, ypos=offset_y + y_dist * 8, width=bus_width,
                           country=country,
                           graphic_type=BusGraphicType.Connectivity)
            dis1 = dev.Switch(name=f"Dis1_{i}", bus_from=bar1, bus_to=bus1, graphic_type=SwitchGraphicType.Disconnector)
            cb1 = dev.Switch(name=f"SW1_{i}", bus_from=bus1, bus_to=bus2, graphic_type=SwitchGraphicType.CircuitBreaker)
            dis2 = dev.Switch(name=f"Dis2_{i}", bus_from=bus2, bus_to=bus3, graphic_type=SwitchGraphicType.Disconnector)
            dis3 = dev.Switch(name=f"Dis3_{i}", bus_from=bus3, bus_to=bus_line_connection_1,
                              graphic_type=SwitchGraphicType.CircuitBreaker)
            dis4 = dev.Switch(name=f"Dis4_{i}", bus_from=bus3, bus_to=bus4, graphic_type=SwitchGraphicType.Disconnector)
            cb2 = dev.Switch(name=f"SW2_{i}", bus_from=bus4, bus_to=bus5, graphic_type=SwitchGraphicType.Disconnector)
            dis5 = dev.Switch(name=f"Dis5_{i}", bus_from=bus5, bus_to=bus6, graphic_type=SwitchGraphicType.Disconnector)
            dis6 = dev.Switch(name=f"Dis6_{i}", bus_from=bus6, bus_to=bus_line_connection_2,
                              graphic_type=SwitchGraphicType.CircuitBreaker)
            dis7 = dev.Switch(name=f"Dis6_{i}", bus_from=bus6, bus_to=bus7, graphic_type=SwitchGraphicType.Disconnector)
            cb3 = dev.Switch(name=f"SW3_{i}", bus_from=bus7, bus_to=bus8, graphic_type=SwitchGraphicType.Disconnector)
            dis8 = dev.Switch(name=f"Dis6_{i}", bus_from=bus8, bus_to=bar2, graphic_type=SwitchGraphicType.Disconnector)

            grid.add_bus(bus1)
            l_x_pos.append(bus1.x)
            l_y_pos.append(bus1.y)

            grid.add_bus(bus2)
            l_x_pos.append(bus2.x)
            l_y_pos.append(bus2.y)

            grid.add_bus(bus3)
            l_x_pos.append(bus3.x)
            l_y_pos.append(bus3.y)

            grid.add_bus(bus4)
            l_x_pos.append(bus4.x)
            l_y_pos.append(bus4.y)

            grid.add_bus(bus5)
            l_x_pos.append(bus5.x)
            l_y_pos.append(bus5.y)

            grid.add_bus(bus6)
            l_x_pos.append(bus6.x)
            l_y_pos.append(bus6.y)

            grid.add_bus(bus7)
            l_x_pos.append(bus7.x)
            l_y_pos.append(bus7.y)

            grid.add_bus(bus8)
            l_x_pos.append(bus8.x)
            l_y_pos.append(bus8.y)

            grid.add_bus(bus_line_connection_1)
            l_x_pos.append(bus_line_connection_1.x)
            l_y_pos.append(bus_line_connection_1.y)

            grid.add_bus(bus_line_connection_2)
            l_x_pos.append(bus_line_connection_2.x)
            l_y_pos.append(bus_line_connection_2.y)

            grid.add_switch(dis1)
            grid.add_switch(cb1)
            grid.add_switch(dis2)
            grid.add_switch(dis3)
            grid.add_switch(dis4)
            grid.add_switch(cb2)
            grid.add_switch(dis5)
            grid.add_switch(dis6)
            grid.add_switch(dis7)
            grid.add_switch(cb3)
            grid.add_switch(dis8)

    else:

        bar1 = dev.Bus(f"{name} bar1", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=(n_bays + n_bays % 2) * x_dist, xpos=offset_x - x_dist, ypos=offset_y, country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar1)
        l_x_pos.append(bar1.x)
        l_y_pos.append(bar1.y)

        bar2 = dev.Bus(f"{name} bar2", substation=substation, Vnom=v_nom, voltage_level=vl,
                       width=(n_bays + n_bays % 2) * x_dist, xpos=offset_x - x_dist, ypos=offset_y + y_dist * 3,
                       country=country,
                       graphic_type=BusGraphicType.BusBar)
        grid.add_bus(bar2)
        l_x_pos.append(bar2.x)
        l_y_pos.append(bar2.y)

        for i in range(0, n_bays, 2):
            bus_line_connection_1 = dev.Bus(f"{name}_bay_conn_{i}", substation=substation, Vnom=v_nom,
                                            voltage_level=vl,
                                            xpos=offset_x + i * x_dist - bus_width / 2, ypos=offset_y + y_dist, width=0,
                                            country=country,
                                            graphic_type=BusGraphicType.Connectivity)
            bus_line_connection_2 = dev.Bus(f"{name}_bay_conn_{i + 1}", substation=substation, Vnom=v_nom,
                                            voltage_level=vl,
                                            xpos=offset_x + i * x_dist - bus_width / 2, ypos=offset_y + y_dist * 2,
                                            width=0,
                                            country=country,
                                            graphic_type=BusGraphicType.Connectivity)
            cb1 = dev.Switch(name=f"SW1_{i}", bus_from=bar1, bus_to=bus_line_connection_1,
                             graphic_type=SwitchGraphicType.CircuitBreaker)
            cb2 = dev.Switch(name=f"SW2_{i}", bus_from=bus_line_connection_1, bus_to=bus_line_connection_2,
                             graphic_type=SwitchGraphicType.CircuitBreaker)
            cb3 = dev.Switch(name=f"SW3_{i}", bus_from=bus_line_connection_2, bus_to=bar2,
                             graphic_type=SwitchGraphicType.CircuitBreaker)

            grid.add_bus(bus_line_connection_1)
            l_x_pos.append(bus_line_connection_1.x)
            l_y_pos.append(bus_line_connection_1.y)

            grid.add_bus(bus_line_connection_2)
            l_x_pos.append(bus_line_connection_2.x)
            l_y_pos.append(bus_line_connection_2.y)

            grid.add_switch(cb1)
            grid.add_switch(cb2)
            grid.add_switch(cb3)

    offset_total_x = max(l_x_pos, default=0) + x_dist
    offset_total_y = max(l_y_pos, default=0) + y_dist

    return vl, offset_total_x, offset_total_y


def create_ring(name, grid: MultiCircuit, n_bays: int, v_nom: float,
                substation: dev.Substation, country: Country = None,
                include_disconnectors: bool = True,
                offset_x=0, offset_y=0) -> Tuple[dev.VoltageLevel, int, int]:
    """

    :param name:
    :param grid:
    :param n_bays:
    :param v_nom:
    :param substation:
    :param country:
    :param include_disconnectors:
    :param offset_x:
    :param offset_y:
    :return:
    """

    vl = dev.VoltageLevel(name=name, substation=substation, Vnom=v_nom)
    grid.add_voltage_level(vl)

    bus_width = 80
    bus_height = 80
    x_dist = bus_width * 6
    y_dist = bus_width * 6
    l_x_pos = []
    l_y_pos = []

    n_positions = max(n_bays, 3)

    radius = x_dist / (2 * math.sin(math.pi / n_positions))
    cx = offset_x + radius
    cy = offset_y + radius

    if include_disconnectors:
        for n in range(n_positions):

            angle1 = 2 * math.pi * n / n_positions
            x1 = cx + radius * math.cos(angle1 + math.radians(25))
            y1 = cy + radius * math.sin(angle1 + math.radians(25))

            angle2 = 2 * math.pi * (n + 1) / n_positions
            x2 = cx + radius * math.cos(angle2 + math.radians(25))
            y2 = cy + radius * math.sin(angle2 + math.radians(25))

            x13, y13 = (x1 + (x2 - x1) / 3, y1 + (y2 - y1) / 3)
            x23, y23 = (x1 + 2 * (x2 - x1) / 3, y1 + 2 * (y2 - y1) / 3)

            bus1 = dev.Bus(f"{name}_position_{n}", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=x1, ypos=y1, width=bus_width, height=bus_height, country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus2 = dev.Bus(f"{name}_cb_{n}.1", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=x13, ypos=y13, width=bus_width, height=bus_height, country=country,
                           graphic_type=BusGraphicType.Connectivity)
            bus3 = dev.Bus(f"{name}_cb_{n}.2", substation=substation, Vnom=v_nom, voltage_level=vl,
                           xpos=x23, ypos=y23, width=bus_width, height=bus_height, country=country,
                           graphic_type=BusGraphicType.Connectivity)

            cb = dev.Switch(name=f"CB_{n}", bus_from=bus2, bus_to=bus3,
                            graphic_type=SwitchGraphicType.CircuitBreaker)
            dis1 = dev.Switch(name=f"dis_{n}.1", bus_from=bus1, bus_to=bus2,
                              graphic_type=SwitchGraphicType.Disconnector)

            if n == 0:
                first_bus = bus1
            elif n + 1 == n_positions:
                dis2 = dev.Switch(name=f"CB_{n}.2", bus_from=bus3, bus_to=first_bus,
                                  graphic_type=SwitchGraphicType.Disconnector)
                grid.add_switch(dis2)

                dis2 = dev.Switch(name=f"dis_{n - 1}.2", bus_from=prev_bus, bus_to=bus1,
                                  graphic_type=SwitchGraphicType.Disconnector)
                grid.add_switch(dis2)

            else:
                dis2 = dev.Switch(name=f"dis_{n}12", bus_from=prev_bus, bus_to=bus1,
                                  graphic_type=SwitchGraphicType.Disconnector)
                grid.add_switch(dis2)

            grid.add_bus(bus1)
            l_x_pos.append(bus1.x)
            l_y_pos.append(bus1.y)

            grid.add_bus(bus2)
            l_x_pos.append(bus2.x)
            l_y_pos.append(bus2.y)

            grid.add_bus(bus3)
            l_x_pos.append(bus3.x)
            l_y_pos.append(bus3.y)

            grid.add_switch(cb)
            grid.add_switch(dis1)

            prev_bus = bus3

    else:

        for n in range(n_positions):

            angle = 2 * math.pi * n / n_positions
            x = cx + radius * math.cos(angle + math.radians(25))
            y = cy + radius * math.sin(angle + math.radians(25))

            bus = dev.Bus(f"{name}_position_{n}", substation=substation, Vnom=v_nom, voltage_level=vl,
                          xpos=x, ypos=y, width=bus_width, height=bus_height, country=country,
                          graphic_type=BusGraphicType.Connectivity)
            if n == 0:
                first_bus = bus
            elif n + 1 == n_positions:
                cb = dev.Switch(name=f"CB_{n}", bus_from=bus, bus_to=first_bus,
                                graphic_type=SwitchGraphicType.CircuitBreaker)
                grid.add_switch(cb)

                cb = dev.Switch(name=f"CB_{n - 1}", bus_from=prev_bus, bus_to=bus,
                                graphic_type=SwitchGraphicType.CircuitBreaker)
                grid.add_switch(cb)
            else:
                cb = dev.Switch(name=f"CB_{n}", bus_from=prev_bus, bus_to=bus,
                                graphic_type=SwitchGraphicType.CircuitBreaker)
                grid.add_switch(cb)

            grid.add_bus(bus)
            l_x_pos.append(bus.x)
            l_y_pos.append(bus.y)

            prev_bus = bus

    offset_total_x = max(l_x_pos, default=0) + x_dist
    offset_total_y = max(l_y_pos, default=0) + y_dist

    return vl, offset_total_x, offset_total_y


def transform_busbar_to_connectivity_grid(grid: MultiCircuit, busbar: dev.Bus):
    """
    Transform a BusBar into multiple Connectivity buses connected by branches.
    This is to be able to compute the power that passes through a busbar
    for specific busbar power studies

    :param grid: MultiCircuit instance
    :param busbar: the Bus object (BusGraphicType.BusBar) to transform
    :return: list of new Connectivity buses
    """
    # Collect all connections (busbar side of each device)
    associated_branches = []
    for elm in grid.get_branches_iter(add_vsc=True, add_hvdc=True, add_switch=True):
        if elm.bus_from == busbar or elm.bus_to == busbar:
            associated_branches.append(elm)

    associated_injections = []
    for elm in grid.get_injection_devices_iter():
        if elm.bus == busbar:
            associated_injections.append(elm)

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
            width=busbar.w,
            country=busbar.country,
            graphic_type=BusGraphicType.Connectivity
        )
        grid.add_bus(new_bus)
        new_buses.append(new_bus)

        # Redirect the element to connect to this new bus instead of the busbar
        if elem.bus_from == busbar:
            elem.bus_from = new_bus

        if elem.bus_to == busbar:
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
            width=busbar.w,
            country=busbar.country,
            graphic_type=BusGraphicType.Connectivity
        )
        grid.add_bus(new_bus)
        new_buses.append(new_bus)

        # Redirect the element to connect to this new bus instead of the busbar
        elem.bus = new_bus

        x_offset += 100

    # Electrically tie all new buses with line branches
    for i in range(len(new_buses) - 1):
        ln = dev.Line(
            name=f"{busbar.name}_backbone_{i}",
            bus_from=new_buses[i],
            bus_to=new_buses[i + 1],
        )
        grid.add_line(ln)

    # Remove the original busbar
    grid.delete_bus(busbar)

    return new_buses


def create_substation(grid: MultiCircuit,
                      se_name: str,
                      se_code: str,
                      lat: float,
                      lon: float,
                      vl_templates: List[dev.VoltageLevelTemplate]) -> Tuple[dev.Substation, List[dev.VoltageLevel]]:
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

        if vl_template.vl_type == SubstationTypes.SingleBar:
            vl, offset_total_x, offset_total_y = create_single_bar(
                name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                grid=grid,
                n_bays=vl_template.n_bays,
                v_nom=vl_template.voltage,
                substation=se_object,
                # country: Country = None,
                include_disconnectors=vl_template.add_disconnectors,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        elif vl_template.vl_type == SubstationTypes.SingleBarWithBypass:
            vl, offset_total_x, offset_total_y = create_single_bar_with_bypass(
                name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                grid=grid,
                n_bays=vl_template.n_bays,
                v_nom=vl_template.voltage,
                substation=se_object,
                # country: Country = None,
                include_disconnectors=vl_template.add_disconnectors,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        elif vl_template.vl_type == SubstationTypes.SingleBarWithSplitter:
            vl, offset_total_x, offset_total_y = create_single_bar_with_splitter(
                name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                grid=grid,
                n_bays=vl_template.n_bays,
                v_nom=vl_template.voltage,
                substation=se_object,
                # country: Country = None,
                include_disconnectors=vl_template.add_disconnectors,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        elif vl_template.vl_type == SubstationTypes.DoubleBar:
            vl, offset_total_x, offset_total_y = create_double_bar(
                name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                grid=grid,
                n_bays=vl_template.n_bays,
                v_nom=vl_template.voltage,
                substation=se_object,
                # country: Country = None,
                include_disconnectors=vl_template.add_disconnectors,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        elif vl_template.vl_type == SubstationTypes.DoubleBarWithBypass:
            # TODO: Implement
            pass

        elif vl_template.vl_type == SubstationTypes.DoubleBarWithTransference:
            vl, offset_total_x, offset_total_y = create_double_bar_with_transference_bar(
                name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                grid=grid,
                n_bays=vl_template.n_bays,
                v_nom=vl_template.voltage,
                substation=se_object,
                # country: Country = None,
                include_disconnectors=vl_template.add_disconnectors,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        elif vl_template.vl_type == SubstationTypes.DoubleBarDuplex:
            # TODO: Implement
            pass

        elif vl_template.vl_type == SubstationTypes.Ring:
            vl, offset_total_x, offset_total_y = create_ring(
                name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                grid=grid,
                n_bays=vl_template.n_bays,
                v_nom=vl_template.voltage,
                substation=se_object,
                # country: Country = None,
                include_disconnectors=vl_template.add_disconnectors,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        elif vl_template.vl_type == SubstationTypes.BreakerAndAHalf:
            vl, offset_total_x, offset_total_y = create_breaker_and_a_half(
                name=f"{se_object.name}-@{vl_template.name} @{vl_template.voltage} kV VL",
                grid=grid,
                n_bays=vl_template.n_bays,
                v_nom=vl_template.voltage,
                substation=se_object,
                # country: Country = None,
                include_disconnectors=vl_template.add_disconnectors,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            offset_x = offset_total_x
            offset_y = offset_total_y
            voltage_levels.append(vl)

        else:
            print(f"{vl_template.vl_type} not implemented :/")

    return se_object, voltage_levels
