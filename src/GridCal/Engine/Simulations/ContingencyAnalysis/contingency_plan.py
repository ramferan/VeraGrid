# GridCal
# Copyright (C) 2022 Santiago Peñate Vera
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
import uuid
from typing import List, Tuple
from GridCal.Engine.Core.multi_circuit import MultiCircuit, DeviceType
from GridCal.Engine.Devices.contingency import Contingency, ContingencyGroup


def add_n1_contingencies(branches, vmin, vmax, filter_branches_by_voltage, branch_types):
    """

    :param plan:
    :param branches:
    :param vmin:
    :param vmax:
    :param filter_branches_by_voltage:
    :param branch_types:
    :return:
    """
    contingencies = list()
    groups = list()

    for i, b in enumerate(branches):

        vi = b.get_max_bus_nominal_voltage()

        filter_ok_i = (vmin <= vi <= vmax) if filter_branches_by_voltage else True

        if filter_ok_i and b.device_type in branch_types:

            group = ContingencyGroup(
                name=b.name,
                category='single',
            )
            contingency = Contingency(
                device_idtag=b.idtag,
                name=b.name,
                code=b.code,
                prop='active',
                value=0,
                group=group
            )

            contingencies.append(contingency)
            groups.append(group)

    return contingencies, groups


def add_n2_contingencies(branches, vmin, vmax, filter_branches_by_voltage, branch_types):
    """

    :param plan:
    :param branches:
    :param vmin:
    :param vmax:
    :param filter_branches_by_voltage:
    :param branch_types:
    :return:
    """
    contingencies = list()
    groups = list()

    for i, branch_i in enumerate(branches):

        vi = branch_i.get_max_bus_nominal_voltage()

        filter_ok_i = (vmin <= vi <= vmax) if filter_branches_by_voltage else True

        if filter_ok_i and branch_i.device_type in branch_types:

            for j, branch_j in enumerate(branches):

                if j != i:

                    vj = branch_j.get_max_bus_nominal_voltage()

                    filter_ok_j = (vmin <= vj <= vmax) if filter_branches_by_voltage else True

                    if filter_ok_j and branch_j.device_type in branch_types:

                        group = ContingencyGroup(
                            name=branch_i.name + " " + branch_j.name,
                            category='double',
                        )

                        contingency1 = Contingency(
                            device_idtag=branch_i.idtag,
                            name=branch_i.name,
                            code=branch_i.code,
                            prop='active',
                            value=0,
                            group=group
                        )

                        contingency2 = Contingency(
                            device_idtag=branch_j.idtag,
                            name=branch_j.name,
                            code=branch_j.code,
                            prop='active',
                            value=0,
                            group=group
                        )

                        contingencies.append(contingency1)
                        contingencies.append(contingency2)
                        groups.append(group)

    return contingencies, groups

def add_injections_contingencies(injections, pmin, pmax, filter_injections_by_power, injection_types, contingency_perc):
    """

    :param injections:
    :param pmin:
    :param pmax:
    :param filter_injections_by_power:
    :param injection_types:
    :param contingency_perc:
    :return:
    """
    contingencies = list()
    groups = list()

    for i, d in enumerate(injections):

        pi = d.Pmax

        filter_ok_i = (pmin <= pi <= pmax) if filter_injections_by_power else True

        if filter_ok_i and d.device_type in injection_types:

            group = ContingencyGroup(
                name=d.name,
                category='injection',
            )

            p = d.Pmax * (1 - contingency_perc/100)
            contingency = Contingency(
                device_idtag=d.idtag,
                name=d.name,
                code=d.code,
                prop='p',
                value=p, # todo: ojo que esto podría aumentar la generación en algunos casos. Ver como definir esto para que funcione mejor en la propia contingencia
                group=group
            )

            contingencies.append(contingency)
            groups.append(group)

    return contingencies, groups

def generate_automatic_contingency_plan(grid: MultiCircuit, k: int, add_branches_to_contingencies:bool = True,
                                        filter_branches_by_voltage: bool = False, vmin=0, vmax=1000,
                                        branch_types: List[DeviceType] = list(), add_injections_to_contingencies: bool = True,
                                        filter_injections_by_power: bool = False, contingency_perc=100.0, pmin=0, pmax=10000,
                                        injection_types: List[DeviceType] = list()) -> Tuple[List[Contingency], List[ContingencyGroup]]:
    """

    :param grid: MultiCircuit instance
    :param k: index (1 for N-1, 2 for N-2, other values of k will fail)
    :param add_branches_to_contingencies:
    :param filter_branches_by_voltage:
    :param vmin:
    :param vmax:
    :param branch_types: List of allowed branch types
    :param add_injections_to_contingencies:
    :param filter_injections_by_power:
    :param contingency_perc:
    :param pmin:
    :param pmax:
    :param injection_types: List of allowed injection types
    :return:
    """

    assert (k in [1, 2])

    # branches = grid.get_branches_wo_hvdc()
    branches = grid.get_branches()
    injections = grid.get_generators() + grid.get_batteries()

    if k == 1:
        if add_branches_to_contingencies:
            contingencies, groups = add_n1_contingencies(
                branches=branches,
                vmin=vmin,
                vmax=vmax,
                filter_branches_by_voltage=filter_branches_by_voltage,
                branch_types=branch_types,
            )
        else:
            contingencies, groups = list(), list()

        if add_injections_to_contingencies:
            contingencies2, groups2 = add_injections_contingencies(
                injections=injections,
                filter_injections_by_power=filter_injections_by_power,
                pmin=pmin,
                pmax=pmax,
                injection_types=injection_types,
                contingency_perc=contingency_perc,
            )

            contingencies.extend(contingencies2)
            groups.extend(groups2)

    elif k == 2:
        if add_branches_to_contingencies:
            contingencies, groups = add_n1_contingencies(branches, vmin, vmax, filter_branches_by_voltage, branch_types)
            contingencies2, groups2 = add_n2_contingencies(branches, vmin, vmax, filter_branches_by_voltage, branch_types)
        else:
            contingencies, groups = list(), list()
            contingencies2, groups2 = list(), list()

        contingencies += contingencies2
        groups += groups2


    else:
        contingencies = list()
        groups = list()

    return contingencies, groups
