# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as plt_colors
from typing import List, Tuple, Dict, Union

# from VeraGrid.Gui.rms_plot_variables_dialog import RmsPlotDialog
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.Devices.Parents.branch_parent import BranchParent
from VeraGridEngine.Devices.Parents.injection_parent import InjectionParent

from VeraGridEngine.Devices.Parents.physical_device import PhysicalDevice
from VeraGridEngine.Simulations.results_table import ResultsTable
from VeraGridEngine.Simulations.results_template import ResultsTemplate
from VeraGridEngine.DataStructures.numerical_circuit import NumericalCircuit
from VeraGridEngine.basic_structures import IntVec, Vec, StrVec, CxVec, ConvergenceReport, Logger, DateVec
from VeraGridEngine.enumerations import StudyResultsType, ResultTypes, DeviceType
from VeraGridEngine.Utils.Symbolic.symbolic import Var


class RmsResults(ResultsTemplate):

    def __init__(self,
                 values: np.ndarray,
                 time_array: DateVec,
                 stat_vars: List[Var],
                 algeb_vars: List[Var],
                 uid2idx: Dict[int, int],
                 vars_glob_name2uid: Dict[str, int],
                 devices: List[Union[Bus, BranchParent, InjectionParent]],
                 units: str = "",
                 ):
        ResultsTemplate.__init__(
            self,
            name='RMS simulation',
            available_results=[ResultTypes.RmsSimulationReport, ResultTypes.RmsPlotResults],
            time_array=time_array,
            clustering_results=None,
            study_results_type=StudyResultsType.RmsSimulation
        )

        variables = stat_vars + algeb_vars
        variable_names = [str(var) for var in variables]
        self.devices = devices
        self.uid2idx = uid2idx
        self.vars_glob_name2uid = vars_glob_name2uid
        self.variable_array = np.array(variable_names, dtype=np.str_)

        self.values = values
        self.units = units
        self.register(name='values', tpe=Vec)


    def mdl(self, result_type: ResultTypes) -> ResultsTable:
        """
        Export the results as a ResultsTable for plotting.
        """
        if result_type == ResultTypes.RmsSimulationReport:
            return ResultsTable(
                data=np.array(self.values),
                index=np.array(pd.to_datetime(self.time_array).astype(str), dtype=np.str_),
                columns=self.variable_array,
                title="Rms Simulation Results",
                units=self.units,
                idx_device_type=DeviceType.TimeDevice,
                cols_device_type=DeviceType.NoDevice
            )
        elif result_type == ResultTypes.RmsPlotResults:

            results_table = ResultsTable(
                data=np.array(self.values),
                index=np.array(pd.to_datetime(self.time_array).astype(str), dtype=np.str_),
                columns=self.variable_array,
                title="Rms Simulation Results",
                units=self.units,
                idx_device_type=DeviceType.TimeDevice,
                cols_device_type=DeviceType.NoDevice,
                xlabel=" time (s)",
                ylabel="",
            )

            devices_options = {
            }

            for device in self.devices:
                vars_list = []
                for var in device.rms_model.model.state_vars + device.rms_model.model.algebraic_vars:
                    var_glob_name = next((varname for varname, uid in self.vars_glob_name2uid.items() if uid == var.uid), None)
                    vars_list.append(var_glob_name)
                devices_options[device.name] = vars_list

            # dlg = RmsPlotDialog(devices_options, results_table, self.uid2idx, self.vars_glob_name2uid)
            # dlg.exec()

        else:
            raise Exception(f"Result type not understood: {result_type}")