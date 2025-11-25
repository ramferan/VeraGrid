# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import List, Dict

from VeraGridEngine.Devices.types import ALL_RMS_TEMPLATES_TYPE
from VeraGridEngine.enumerations import DeviceType

from VeraGridEngine.Templates.Rms.bus_rms_template import BusRmsTemplate
from VeraGridEngine.Templates.Rms.generator_0_rms_template import Generator_0_RmsTemplate
from VeraGridEngine.Templates.Rms.generator_1_rms_template import Generator_1_RmsTemplate
from VeraGridEngine.Templates.Rms.line_0_rms_template import Line_0_RmsTemplate
from VeraGridEngine.Templates.Rms.line_1_rms_template import Line_1_RmsTemplate
from VeraGridEngine.Templates.Rms.load_rms_template import LoadRmsTemplate

templ_bus = BusRmsTemplate()

templ_gen_0 = Generator_0_RmsTemplate()
templ_gen_1 = Generator_1_RmsTemplate()

templ_line_0 = Line_0_RmsTemplate()
templ_line_1 = Line_1_RmsTemplate()

templ_load = LoadRmsTemplate()

Templates: List[ALL_RMS_TEMPLATES_TYPE] = [templ_bus, templ_gen_0, templ_gen_1, templ_line_0, templ_line_1,
                                           templ_load]


def get_generator_catalogue() -> tuple[List[str], Dict[str, ALL_RMS_TEMPLATES_TYPE]]:
    generator_templ_catalogue = dict()
    generator_templ_list = []
    for templ in Templates:
        if templ.tpe == DeviceType.GeneratorDevice:
            generator_templ_list.append(templ.name)
            generator_templ_catalogue[templ.name] = templ

    return generator_templ_list, generator_templ_catalogue


def get_bus_catalogue() -> tuple[List[str], Dict[str, ALL_RMS_TEMPLATES_TYPE]]:
    bus_templ_catalogue = dict()
    bus_templ_list = []
    for templ in Templates:
        if templ.tpe == DeviceType.BusDevice:
            bus_templ_list.append(templ.name)
            bus_templ_catalogue[templ.name] = templ

    return bus_templ_list, bus_templ_catalogue


def get_line_catalogue() -> tuple[List[str], Dict[str, ALL_RMS_TEMPLATES_TYPE]]:
    line_templ_catalogue = dict()
    line_templ_list = []
    for templ in Templates:
        if templ.tpe == DeviceType.LineDevice:
            line_templ_list.append(templ.name)
            line_templ_catalogue[templ.name] = templ

    return line_templ_list, line_templ_catalogue


def get_load_catalogue() -> tuple[List[str], Dict[str, ALL_RMS_TEMPLATES_TYPE]]:
    load_templ_catalogue = dict()
    load_templ_list = []
    for templ in Templates:
        if templ.tpe == DeviceType.LoadDevice:
            load_templ_list.append(templ.name)
            load_templ_catalogue[templ.name] = templ

    return load_templ_list, load_templ_catalogue
