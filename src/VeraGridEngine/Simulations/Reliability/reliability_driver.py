# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.  
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from VeraGridEngine.enumerations import SimulationTypes, ReliabilityMode
from VeraGridEngine.Simulations.PowerFlow.power_flow_worker import PowerFlowOptions, multi_island_pf_nc
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Simulations.driver_template import DriverTemplate
from VeraGridEngine.Simulations.OPF.simple_dispatch_ts import GreedyDispatchInputs
from VeraGridEngine.Simulations.Reliability.reliability import (reliability_simulation, generate_states_matrix,
                                                                find_time_blocks)
from VeraGridEngine.Simulations.Reliability.reliability_results import ReliabilityResults
from VeraGridEngine.Compilers.circuit_to_data import compile_numerical_circuit_at


class ReliabilityStudyDriver(DriverTemplate):
    name = 'Reliability analysis'
    tpe = SimulationTypes.Reliability_run

    def __init__(self,
                 grid: MultiCircuit,
                 pf_options: PowerFlowOptions,
                 reliability_mode: ReliabilityMode = ReliabilityMode.GenerationAdequacy,
                 time_indices=None,
                 n_sim: int = 1000000):
        """
        ContinuationPowerFlowDriver constructor
        :param grid: NumericalCircuit instance
        :param pf_options: power flow options instance
        :param reliability_mode: ReliabilityMode
        :param time_indices:
        :param n_sim: Number of Monte-Carlo simulations
        """
        DriverTemplate.__init__(self, grid=grid)

        # voltage stability options
        self.pf_options = pf_options

        self.reliability_mode = reliability_mode

        self.n_sim = n_sim

        if time_indices is None:
            self.time_indices = self.grid.get_all_time_indices()
        else:
            self.time_indices = time_indices

        self.results = ReliabilityResults(nsim=n_sim)

        self.greedy_dispatch_inputs = GreedyDispatchInputs(grid=self.grid,
                                                           time_indices=self.time_indices,
                                                           logger=self.logger)

        self.__cancel__ = False

    def progress_callback(self, lmbda: float):
        """
        Send progress report
        :param lmbda: lambda value
        :return: None
        """
        self.report_text('Running voltage collapse lambda:' + "{0:.2f}".format(lmbda) + '...')

    def run(self):
        """
        Run reliability
        """
        self.report_text("Running reliability study...")
        self.tic()

        if self.reliability_mode == ReliabilityMode.GenerationAdequacy:
            self.run_adequacy_reliability()

        elif self.reliability_mode == ReliabilityMode.GridMetrics:
            self.run_grid_reliability()

        self.toc()
        self.report_text("Done!")
        self.done_signal.emit()

    def run_adequacy_reliability(self):
        """
        run the voltage collapse simulation
        @return:
        """

        horizon = self.grid.get_time_number()

        n_gen = self.grid.get_generators_number()

        gen_pmax = np.empty((horizon, n_gen), dtype=float)
        gen_mttf = np.zeros(n_gen)
        gen_mttr = np.zeros(n_gen)
        for k, gen in enumerate(self.grid.generators):
            gen_mttf[k] = gen.mttf
            gen_mttr[k] = gen.mttr
            if gen.enabled_dispatch:
                gen_pmax[:, k] = gen.Snom * gen.active_prof.toarray()
            else:
                gen_pmax[:, k] = gen.P_prof.toarray() * gen.active_prof.toarray()

        lole, _, _ = reliability_simulation(
            n_sim=self.n_sim,
            load_profile=self.greedy_dispatch_inputs.load_profile,

            gen_profile=self.greedy_dispatch_inputs.gen_profile,
            gen_p_max=gen_pmax,
            gen_p_min=self.greedy_dispatch_inputs.gen_p_min,
            gen_dispatchable=self.greedy_dispatch_inputs.gen_dispatchable,
            gen_active=self.greedy_dispatch_inputs.gen_active,
            gen_cost=self.greedy_dispatch_inputs.gen_cost,
            gen_mttf=gen_mttf,
            gen_mttr=gen_mttr,

            batt_active=self.greedy_dispatch_inputs.batt_active,
            batt_p_max_charge=self.greedy_dispatch_inputs.batt_p_max_charge,
            batt_p_max_discharge=self.greedy_dispatch_inputs.batt_p_max_discharge,
            batt_energy_max=self.greedy_dispatch_inputs.batt_energy_max,
            batt_eff_charge=self.greedy_dispatch_inputs.batt_eff_charge,
            batt_eff_discharge=self.greedy_dispatch_inputs.batt_eff_discharge,
            batt_cost=self.greedy_dispatch_inputs.batt_cost,
            batt_soc0=self.greedy_dispatch_inputs.batt_soc0,
            batt_soc_min=self.greedy_dispatch_inputs.batt_soc_min,
            dt=self.greedy_dispatch_inputs.dt,
            force_charge_if_low=True,
            tol=1e-6
        )

        self.results.lole_evolution = np.cumsum(lole) / (np.arange(len(lole)) + 1)
        print(f"LOLE: {lole.mean()} MWh/year")

    def run_grid_reliability(self) -> None:
        """
        run the voltage collapse simulation
        @return:
        """

        horizon = self.grid.get_time_number()

        nc = compile_numerical_circuit_at(self.grid)

        nc2 = nc.copy()

        ENS_arr = np.zeros(self.n_sim)

        for sim_idx in range(self.n_sim):

            if self.__cancel__:
                return

            gen_actives, gen_n_failures = generate_states_matrix(mttf=nc.generator_data.mttf,
                                                                 mttr=nc.generator_data.mttr,
                                                                 horizon=horizon,
                                                                 initially_working=True)

            batt_actives, batt_n_failures = generate_states_matrix(mttf=nc.battery_data.mttf,
                                                                   mttr=nc.battery_data.mttr,
                                                                   horizon=horizon,
                                                                   initially_working=True)

            simulated_branch_actives, br_n_failures = generate_states_matrix(mttf=nc.passive_branch_data.mttf,
                                                                             mttr=nc.passive_branch_data.mttr,
                                                                             horizon=horizon,
                                                                             initially_working=True)

            total_n_customers = sum([load.n_customers for load in self.grid.loads])

            if gen_n_failures + br_n_failures:

                all_actives = np.c_[gen_actives, batt_actives, simulated_branch_actives]

                blocks = find_time_blocks(horizon, all_actives)

                for idx_list in blocks:

                    batt_e_nom = nc.battery_data.enom.copy()
                    total_failure_time = 0.0
                    total_affected_customers = 0

                    for t in idx_list:  # time_steps

                        # get the time increment
                        dt = self.greedy_dispatch_inputs.dt[t]
                        total_failure_time += dt

                        # modify active states
                        nc2.passive_branch_data.active = simulated_branch_actives[t, :]
                        nc2.generator_data.active = gen_actives[t, :]
                        nc2.battery_data.active = batt_actives[t, :]

                        E_not_supplied = 0
                        islands = nc2.split_into_islands(ignore_single_node_islands=False)
                        for island in islands:

                            total_affected_customers += sum([self.grid.loads[ii].n_customers
                                                             for ii in nc2.load_data.original_idx])

                            if island.generator_data.active.sum() == 0:
                                if island.battery_data.active.sum() == 0:
                                    E_not_supplied += island.load_data.S.sum() * dt
                                else:
                                    # check the battery life
                                    island_energy_demand = island.load_data.S.sum().real * dt

                                    unsatisfied_demand = island_energy_demand
                                    for i in island.battery_data.original_idx:
                                        if unsatisfied_demand > batt_e_nom[i]:
                                            # we deplete the battery
                                            unsatisfied_demand -= batt_e_nom[i]
                                            batt_e_nom[i] = 0
                                        else:
                                            # there is less demand that battery capacity
                                            batt_e_nom[i] -= unsatisfied_demand
                                            unsatisfied_demand = 0

                                    E_not_supplied += unsatisfied_demand

                        # revert active states
                        nc2.passive_branch_data.active = nc.passive_branch_data.active
                        nc2.generator_data.active = nc.generator_data.active
                        nc2.battery_data.active = nc.battery_data.active

                        # Energy not supplied (MWh)
                        ENS_arr[sim_idx] += E_not_supplied.real

            self.report_progress2(current=sim_idx, total=self.n_sim)

        self.results.lole_evolution = np.cumsum(ENS_arr) / (np.arange(len(ENS_arr)) + 1)

    def cancel(self):
        self.__cancel__ = True
