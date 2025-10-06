# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from typing import Dict, Union
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Utils.Symbolic.block_solver import BlockSolver
from VeraGridEngine.Simulations.driver_template import DriverTemplate
from VeraGridEngine.Simulations.SmallSignalStability.small_signal_options import SmallSignalStabilityOptions
from VeraGridEngine.Simulations.SmallSignalStability.small_signal_results import SmallSignalStabilityResults
from VeraGridEngine.enumerations import EngineType, SimulationTypes
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowResults
from VeraGridEngine.Simulations.Rms.initialization import initialize_rms


class SmallSignalStabilityDriver(DriverTemplate):
    """
    SmallSignal_Stability_Driver
    """
    name = 'Small Signal Stability Simulation'
    tpe = SimulationTypes.SmallSignal_run

    def __init__(self, grid: MultiCircuit,
                 options: Union[SmallSignalStabilityOptions, None] = None,
                 pf_results: Union[PowerFlowResults, None] = None,
                 engine: EngineType = EngineType.VeraGrid):

        """
        DynamicDriver class constructor
        :param grid: MultiCircuit instance
        :param options: SmallSignalOptions instance
        :param pf_results: PowerFlowResults
        :param engine: EngineType (i.e., EngineType.VeraGrid) (optional)
        """

        DriverTemplate.__init__(self, grid=grid, engine=engine)

        self.grid = grid

        self.pf_results: Union[PowerFlowResults, None] = pf_results

        self.options: SmallSignalStabilityOptions = SmallSignalStabilityOptions() if options is None else options

        self.results: SmallSignalStabilityResults = SmallSignalStabilityResults(stability="",
                                                                                Eigenvalues=np.empty(0),
                                                                                PF=np.empty(0),
                                                                                stat_vars=[],
                                                                                vars2device={})

        self.__cancel__ = False

    def run(self):
        """
        Main function to initialize and run the system simulation.

        This function sets up logging, starts the dynamic simulation, and
        logs the outcome. It handles and logs any exceptions raised during execution.
        :return:
        """
        # Run the dynamic simulation
        self.run_small_signal_stability()

    def run_small_signal_stability(self):
        """
        Performs the numerical integration using the chosen method.
        :return:
        """
        self.tic()

        params_mapping: Dict = dict()

        ss, init_guess = initialize_rms(self.grid, self.pf_results)
        slv = BlockSolver(ss, self.grid.time)

        params0 = slv.build_init_params_vector(params_mapping)
        x0 = slv.build_init_vars_vector_from_uid(init_guess)

        if not self.options.ss_assessment_time == 0:
            # Get integration method
            if self.options.integration_method == "trapezoid":
                integrator = "trapezoid"
            elif self.options.integration_method == "implicit euler":
                integrator = "implicit_euler"
            else:
                raise ValueError(f"integrator not implemented :( {self.options.integration_method}")

            t, y = slv.simulate(
                t0=0,
                t_end=self.options.ss_assessment_time,
                h=self.options.time_step,
                x0=x0,
                params0=params0,
                method=integrator
            )

            i = int(self.options.ss_assessment_time / self.options.time_step)
            stab, Eigenvalues, pfactors = slv.run_small_signal_stability(x=y[i], params=params0, plot=False)

        else:

            stab, Eigenvalues, pfactors = slv.run_small_signal_stability(x=x0, params=params0, plot=False)

        self.results: SmallSignalStabilityResults = SmallSignalStabilityResults(
            stability=stab,
            Eigenvalues=Eigenvalues,
            PF=pfactors,
            stat_vars=slv.state_vars,
            vars2device=slv.vars2device
        )

        self.toc()
