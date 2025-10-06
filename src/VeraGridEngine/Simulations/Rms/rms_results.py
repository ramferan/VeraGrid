# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as plt_colors
from typing import List, Tuple, Dict

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
                 vars2device: Dict[int, PhysicalDevice],
                 units: str = "",
                 ):
        ResultsTemplate.__init__(
            self,
            name='RMS simulation',
            available_results=[ResultTypes.RmsSimulationReport],
            time_array=time_array,
            clustering_results=None,
            study_results_type=StudyResultsType.RmsSimulation
        )

        # if values.shape[1] != len(stat_vars) + len(algeb_vars):
        #     raise ValueError("Number of columns in values must match number of variable_names")
        variables = stat_vars + algeb_vars
        variable_names = [str(var) + vars2device[var.uid].name for var in variables]

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
        else:
            raise Exception(f"Result type not understood: {result_type}")

    # def add(self, dt: float, y: Vec):
    #     """
    #
    #     :param dt:
    #     :param y:
    #     :return:
    #     """
    #     self.dt.append(dt)
    #     self.y.append(y)
