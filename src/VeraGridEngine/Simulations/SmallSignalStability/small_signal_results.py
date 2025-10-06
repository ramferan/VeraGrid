# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Union
import matplotlib.colors as plt_colors
from typing import List, Tuple, Dict

from VeraGridEngine.Devices.Parents.physical_device import PhysicalDevice
from VeraGridEngine.Simulations.results_table import ResultsTable
from VeraGridEngine.Simulations.results_template import ResultsTemplate
from VeraGridEngine.DataStructures.numerical_circuit import NumericalCircuit
from VeraGridEngine.basic_structures import IntVec, Vec, StrVec, CxVec, ConvergenceReport, Logger, DateVec, Mat
from VeraGridEngine.enumerations import StudyResultsType, ResultTypes, DeviceType
from VeraGridEngine.Utils.Symbolic.symbolic import Var


class SmallSignalStabilityResults(ResultsTemplate):

    def __init__(self,
                 stability: str,
                 Eigenvalues: np.ndarray,
                 PF: np.ndarray,
                 stat_vars: List[Var],
                 vars2device: Dict[int, PhysicalDevice]):
        """

        :param stability:
        :param Eigenvalues:
        :param PF:
        :param stat_vars:
        :param vars2device:
        """
        ResultsTemplate.__init__(
            self,
            name='Small Signal Stability',
            available_results=[
                ResultTypes.Modes,
                ResultTypes.ParticipationFactors,
                ResultTypes.SDomainPlot
            ],
            time_array=None,
            clustering_results=None,
            study_results_type=StudyResultsType.SmallSignalStability
        )

        stat_vars_names = [f"{var}{i // 2 + 1}" for i, var in enumerate(stat_vars)]

        self.stat_vars_array = np.array(stat_vars_names, dtype=np.str_)

        self.stability = stability
        self.eigenvalues = Eigenvalues
        self.participation_factors = PF
        # self.register(name='Stability', tpe=Vec)
        self.register(name='eigenvalues', tpe=Vec)
        self.register(name='participation_factors', tpe=Vec)

    def mdl(self, result_type: ResultTypes) -> ResultsTable:
        """
        Export the results as a ResultsTable for plotting.
        """
        if result_type == ResultTypes.ParticipationFactors:

            return ResultsTable(
                data=self.participation_factors,
                index=np.array(self.stat_vars_array.astype(str), dtype=np.str_),
                columns=np.array([f"Mode {i}" for i in range(len(self.eigenvalues))], dtype=np.str_),  #
                title="Participation factors for each eigenvalue",
                idx_device_type=DeviceType.NoDevice,
                cols_device_type=DeviceType.NoDevice
            )

        elif result_type == ResultTypes.Modes:
            return ResultsTable(
                data=np.array([np.real(self.eigenvalues), np.imag(self.eigenvalues)]).T,
                index=np.array([f"Mode {i}" for i in range(len(self.eigenvalues))], dtype=np.str_),
                columns=np.array(["Real", "Imaginary"]),
                title="Eigenvalues",
                idx_device_type=DeviceType.NoDevice,
                cols_device_type=DeviceType.NoDevice
            )
        elif result_type == ResultTypes.SDomainPlot:
            re = self.eigenvalues.real
            im = self.eigenvalues.imag
            data = np.c_[re, im]

            d = np.abs(np.nan_to_num(re))
            colors = (d / d.max())

            if self.plotting_allowed():
                plt.ion()
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                sc = ax.scatter(re, im, c=colors, cmap='winter', s=120, alpha=0.8)
                fig.suptitle("S-Domain Stability plot")
                ax.set_xlabel(r'Real  $ [s^{-1}]$')
                ax.set_ylabel(r'Imaginary  $ [s^{-1}]$')
                ax.axhline(0, color='black', linewidth=1)  # eje horizontal (y = 0)
                ax.axvline(0, color='black', linewidth=1)
                plt.tight_layout()
                plt.show()
                annot = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                    textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc="w"),
                                    arrowprops=dict(arrowstyle="->"),
                                    fontsize=8)
                annot.set_visible(False)

                def update_annotation(ind):
                    """
                    :param ind:
                    :return:
                    """
                    pos = sc.get_offsets()[ind["ind"][0]]
                    annot.xy = pos
                    text = f"Re={pos[0]:.2f}, Im={pos[1]:.2f}"
                    annot.set_text(text)
                    annot.get_bbox_patch().set_alpha(0.8)

                def hover(event):
                    if event.inaxes == ax:
                        cont, ind = sc.contains(event)
                        if cont:
                            update_annotation(ind)
                            annot.set_visible(True)
                            fig.canvas.draw_idle()
                        else:
                            if annot.get_visible():
                                annot.set_visible(False)
                                fig.canvas.draw_idle()

                fig.canvas.mpl_connect("motion_notify_event", hover)

            return ResultsTable(data=data,
                                index=np.empty(len(self.eigenvalues), dtype=np.str_),
                                idx_device_type=DeviceType.NoDevice,
                                columns=np.array(['Real', 'Imag']),
                                cols_device_type=DeviceType.NoDevice,
                                title="S-Domain Stability plot"
                                )
        else:
            raise Exception(f"Result type not understood: {result_type}")

    # def plot(self, fig, ax):
    #     """
    #     Plot the S-Domain modes plot
    #     :param fig: Matplotlib figure. If None, one will be created
    #     :param ax: Matplotlib Axis. If None, one will be created
    #     """
    #     if ax is None:
    #         fig = plt.figure(figsize=(8, 7))
    #         ax = fig.add_subplot(111)
    #
    #     x = self.eigenvalues.real
    #     y = self.eigenvalues.imag
    #
    #     ax.scatter(x, y, 'k', linewidth=2)
    #
    #     ax.set_title(r'$S-Domain$ plot')
    #     ax.set_xlabel(r'$Imaginary [s^{-1}]$')
    #     ax.set_ylabel(r'$Real [s^{-1}]$')
