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


import numpy as np
import time
import os

from GridCal.Engine.Core.multi_circuit import MultiCircuit
from GridCal.Engine.Core.time_series_opf_data import compile_opf_time_circuit, OpfTimeCircuit
from GridCal.Engine.Simulations.NTC.ntc_opf import OpfNTC
from GridCal.Engine.Simulations.NTC.ntc_driver import OptimalNetTransferCapacityOptions, OptimalNetTransferCapacityResults
from GridCal.Engine.Simulations.NTC.ntc_ts_results import OptimalNetTransferCapacityTimeSeriesResults
from GridCal.Engine.Simulations.driver_types import SimulationTypes
from GridCal.Engine.Simulations.driver_template import TimeSeriesDriverTemplate
from GridCal.Engine.Simulations.Clustering.clustering import kmeans_sampling
from GridCal.Engine.Simulations.ATC.available_transfer_capacity_driver import compute_alpha
from GridCal.Engine.Simulations.LinearFactors.linear_analysis import LinearAnalysis
from GridCal.Engine.Simulations.ATC.available_transfer_capacity_driver import AvailableTransferMode
from GridCal.Engine.basic_structures import Logger

try:
    from ortools.linear_solver import pywraplp
except ModuleNotFoundError:
    print('ORTOOLS not found :(')



class OptimalNetTransferCapacityTimeSeriesDriver(TimeSeriesDriverTemplate):

    tpe = SimulationTypes.OptimalNetTransferCapacityTimeSeries_run

    def __init__(self, grid: MultiCircuit, options: OptimalNetTransferCapacityOptions, start_=0, end_=None,
                 use_clustering=False, cluster_number=100):
        """

        :param grid: MultiCircuit Object
        :param options: Optimal net transfer capacity options
        :param start_: time index to start (optional)
        :param end_: time index to end (optional)
        """
        TimeSeriesDriverTemplate.__init__(
            self,
            grid=grid,
            start_=start_,
            end_=end_)

        # Options to use

        self.options = options
        self.unresolved_counter = 0

        self.use_clustering = use_clustering
        self.cluster_number = cluster_number

        self.logger = Logger()

        self.results = OptimalNetTransferCapacityTimeSeriesResults(
            branch_names=[],
            bus_names=[],
            generator_names=[],
            load_names=[],
            rates=[],
            contingency_rates=[],
            time_array=[],
            time_indices=[],
            sampled_probabilities=[],
            trm=self.options.trm,
            ntc_load_rule=self.options.ntc_load_rule,
            loading_threshold_to_report=self.options.loading_threshold_to_report,
            reversed_sort_loading=self.options.reversed_sort_loading,
        )

        self.installed_alpha = None
        self.installed_alpha_n1 = None

    name = tpe.value

    def compute_exchange_sensitivity(self, linear, numerical_circuit: OpfTimeCircuit, t, with_n1=True):

        # compute the branch exchange sensitivity (alpha)
        tm0 = time.time()
        alpha, alpha_n1 = compute_alpha(
            ptdf=linear.PTDF,
            lodf=linear.LODF if with_n1 else None,
            P0=numerical_circuit.Sbus.real[:, t],
            Pinstalled=numerical_circuit.bus_installed_power,
            Pgen=numerical_circuit.generator_data.get_injections_per_bus()[:, t].real,
            Pload=numerical_circuit.load_data.get_injections_per_bus()[:, t].real,
            idx1=self.options.area_from_bus_idx,
            idx2=self.options.area_to_bus_idx,
            dT=self.options.sensitivity_dT,
            mode=self.options.transfer_method.value)

        # self.logger.add_info('Exchange sensibility computed in {0:.2f} scs.'.format(time.time()-tm0))
        return alpha, alpha_n1

    def opf(self):
        """
        Run thread
        """

        self.progress_signal.emit(0)

        if self.progress_text is not None:
            self.progress_text.emit('Compiling circuit...')
        else:
            print('Compiling cicuit...')

        tm0 = time.time()
        nc = compile_opf_time_circuit(self.grid)
        self.logger.add_info(f'Time circuit compiled in {time.time()-tm0:.2f} scs')
        print(f'Time circuit compiled in {time.time()-tm0:.2f} scs')

        # declare the linear analysis
        if self.progress_text is not None:
            self.progress_text.emit('Computing linear analysis...')
        else:
            print('Computing linear analysis...')

        linear = LinearAnalysis(
            grid=self.grid,
            distributed_slack=False,
            correct_values=False,
            with_nx=self.options.consider_nx_contingencies,
        )

        tm0 = time.time()
        linear.run()

        self.logger.add_info(f'Linear analysis computed in {time.time()-tm0:.2f} scs.')
        print(f'Linear analysis computed in {time.time()-tm0:.2f} scs.')

        time_indices = self.get_time_indices()

        if self.use_clustering:

            if self.progress_text is not None:
                self.progress_text.emit('Clustering...')

            else:
                print('Clustering...')

            X = nc.Sbus
            X = X[:, time_indices].real.T

            # cluster and re-assign the time indices
            tm1 = time.time()
            time_indices, sampled_probabilities = kmeans_sampling(
                X=X,
                n_points=self.cluster_number,
            )

            self.logger.add_info(f'Kmeans sampling computed in {time.time()-tm1:.2f} scs. [{len(time_indices)} points]')
            print(f'Kmeans sampling computed in {time.time()-tm1:.2f} scs. [{len(time_indices)} points]')

        else:
            sampled_probabilities = np.full(len(self.indices), 1/len(time_indices))

        nt = len(time_indices)

        # Initialize results object
        self.results = OptimalNetTransferCapacityTimeSeriesResults(
            branch_names=linear.numerical_circuit.branch_names,
            bus_names=linear.numerical_circuit.bus_names,
            generator_names=linear.numerical_circuit.generator_names,
            load_names=linear.numerical_circuit.load_names,
            rates=nc.Rates,
            contingency_rates=nc.ContingencyRates,
            time_array=nc.time_array[time_indices],
            time_indices=time_indices,
            sampled_probabilities=sampled_probabilities,
            trm=self.options.trm,
            loading_threshold_to_report=self.options.loading_threshold_to_report,
            ntc_load_rule=self.options.ntc_load_rule)

        if self.options.transfer_method == AvailableTransferMode.InstalledPower:
            alpha, alpha_n1 = self.compute_exchange_sensitivity(
                linear=linear,
                numerical_circuit=nc,
                t=0,
                with_n1=self.options.n1_consideration
            )
        else:
            alpha = np.ones(nc.nbr),
            alpha_n1 = np.ones((nc.nbr, nc.nbr)),

        for t_idx, t in enumerate(time_indices):

            # Initialize problem object (needed to reset solver variable names)
            problem = OpfNTC(
                numerical_circuit=nc,
                area_from_bus_idx=self.options.area_from_bus_idx,
                area_to_bus_idx=self.options.area_to_bus_idx,
                LODF=linear.LODF,
                LODF_NX=linear.LODF_NX,
                PTDF=linear.PTDF,
                alpha=alpha,
                alpha_n1=alpha_n1,
                solver_type=self.options.mip_solver,
                generation_formulation=self.options.generation_formulation,
                monitor_only_sensitive_branches=self.options.monitor_only_sensitive_branches,
                monitor_only_ntc_load_rule_branches=self.options.monitor_only_ntc_load_rule_branches,
                branch_sensitivity_threshold=self.options.branch_sensitivity_threshold,
                skip_generation_limits=self.options.skip_generation_limits,
                dispatch_all_areas=self.options.dispatch_all_areas,
                tolerance=self.options.tolerance,
                weight_power_shift=self.options.weight_power_shift,
                weight_generation_cost=self.options.weight_generation_cost,
                consider_contingencies=self.options.consider_contingencies,
                consider_hvdc_contingencies=self.options.consider_hvdc_contingencies,
                consider_gen_contingencies=self.options.consider_gen_contingencies,
                generation_contingency_threshold=self.options.generation_contingency_threshold,
                match_gen_load=self.options.match_gen_load,
                ntc_load_rule=self.options.ntc_load_rule,
                transfer_method=self.options.transfer_method,
                logger=self.logger
            )

            # update progress bar
            progress = (t_idx + 1) / len(time_indices) * 100
            self.progress_signal.emit(progress)

            if self.progress_text is not None:
                self.progress_text.emit('Optimal net transfer capacity at ' + str(self.grid.time_profile[t]))

            else:
                print('Optimal net transfer capacity at ' + str(self.grid.time_profile[t]))

            # sensitivities
            if self.options.monitor_only_sensitive_branches or self.options.monitor_only_ntc_load_rule_branches:

                if self.options.transfer_method != AvailableTransferMode.InstalledPower:
                    problem.alpha, problem.alpha_n1 = self.compute_exchange_sensitivity(
                        linear=linear,
                        numerical_circuit=nc,
                        t=t,
                        with_n1=self.options.n1_consideration)

            time_str = str(nc.time_array[time_indices][t_idx])

            # Define the problem
            self.progress_text.emit('Formulating NTC OPF...['+time_str+']')
            problem.formulate_ts(t=t)

            # Solve

            scenario = '[' + time_str + ']'
            self.progress_text.emit('Solving NTC OPF...'+scenario)
            solved = problem.solve_ts(
                t=t,
                time_limit_ms=self.options.time_limit_ms
            )
            # print('Problem solved in {0:.2f} scs.'.format(time.time() - tm0))

            self.logger += problem.logger

            if solved:
                self.results.optimal_idx.append(t)

            else:

                if problem.status == pywraplp.Solver.FEASIBLE:
                    self.results.feasible_idx.append(t)
                    self.logger.add_error(
                        'Feasible solution, not optimal or timeout for {0}'.format(scenario),
                        'NTC OPF')

                if problem.status == pywraplp.Solver.INFEASIBLE:
                    self.results.infeasible_idx.append(t)
                    self.logger.add_error(
                        'Unfeasible solution {0}'.format(scenario),
                        'NTC OPF')

                if problem.status == pywraplp.Solver.UNBOUNDED:
                    self.results.unbounded_idx.append(t)
                    self.logger.add_error(
                        'Proved unbounded {0}'.format(scenario),
                        'NTC OPF')

                if problem.status == pywraplp.Solver.ABNORMAL:
                    self.results.abnormal_idx.append(t)
                    self.logger.add_error(
                        'Abnormal solution, some error occurred {0}'.format(scenario),
                        'NTC OPF')

                if problem.status == pywraplp.Solver.NOT_SOLVED:
                    self.results.not_solved.append(t)
                    self.logger.add_error(
                        'Not solved, maybe timeout occurs {0}'.format(scenario),
                        'NTC OPF')

            # pack the results
            idx_w = np.argmax(np.abs(problem.alpha_n1), axis=1)
            alpha_w = np.take_along_axis(problem.alpha_n1, np.expand_dims(idx_w, axis=1), axis=1)

            result = OptimalNetTransferCapacityResults(
                bus_names=nc.bus_data.names,
                branch_names=nc.branch_data.names,
                branch_data_F=nc.branch_data.F,
                branch_data_T=nc.branch_data.T,
                branch_x=nc.branch_data.X,
                load_names=nc.load_data.names,
                generator_names=nc.generator_data.names,
                battery_names=nc.battery_data.names,
                hvdc_names=nc.hvdc_data.names,
                trm=self.options.trm,
                ntc_load_rule=self.options.ntc_load_rule,
                branch_control_modes=nc.branch_data.control_mode,
                hvdc_control_modes=nc.hvdc_data.control_mode,
                Sbus=problem.get_power_injections(),
                voltage=problem.get_voltage(),
                battery_power=np.zeros((nc.nbatt, 1)),
                controlled_generation_power=problem.get_generator_power(),
                Sf=problem.get_branch_power_from(),
                loading=problem.get_loading(),
                solved=bool(solved),
                bus_types=nc.bus_types,
                hvdc_flow=problem.get_hvdc_flow(),
                hvdc_loading=problem.get_hvdc_loading(),
                phase_shift=problem.get_phase_angles(),
                generation_delta=problem.get_generator_delta(),
                # hvdc_angle_slack=problem.get_hvdc_angle_slacks(),
                inter_area_branches=problem.inter_area_branches,
                inter_area_hvdc=problem.inter_area_hvdc,
                alpha=problem.alpha,
                alpha_n1=problem.alpha_n1,
                alpha_w=alpha_w,
                contingency_branch_flows_list=problem.get_contingency_flows_list(),
                contingency_branch_indices_list=problem.contingency_indices_list,
                contingency_generation_flows_list=problem.get_contingency_gen_flows_list(),
                contingency_generation_indices_list=problem.contingency_gen_indices_list,
                contingency_hvdc_flows_list=problem.get_contingency_hvdc_flows_list(),
                contingency_hvdc_indices_list=problem.contingency_hvdc_indices_list,
                contingency_branch_alpha_list=problem.contingency_branch_alpha_list,
                contingency_generation_alpha_list=problem.contingency_generation_alpha_list,
                contingency_hvdc_alpha_list=problem.contingency_hvdc_alpha_list,
                branch_ntc_load_rule=problem.get_branch_ntc_load_rule(),
                rates=nc.branch_data.rates[:, t],
                contingency_rates=nc.branch_data.contingency_rates[:, t],
                area_from_bus_idx=self.options.area_from_bus_idx,
                area_to_bus_idx=self.options.area_to_bus_idx,
                structural_ntc=problem.structural_ntc,
                sbase=nc.Sbase,
                monitor=problem.monitor,
                monitor_loading=problem.monitor_loading,
                monitor_by_sensitivity=problem.monitor_by_sensitivity,
                monitor_by_unrealistic_ntc=problem.monitor_by_unrealistic_ntc,
                monitor_by_zero_exchange=problem.monitor_by_zero_exchange,
                loading_threshold=self.options.loading_threshold_to_report,
                reversed_sort_loading=self.options.reversed_sort_loading,
            )

            self.progress_text.emit('Creating report...['+time_str+']')

            result.create_all_reports(
                loading_threshold=self.options.loading_threshold_to_report,
                reverse=self.options.reversed_sort_loading,
                save_memory=True,  # todo: check if needed
            )
            self.results.results_dict[t] = result

            if self.progress_signal is not None:
                self.progress_signal.emit((t_idx + 1) / nt * 100)

            if self.__cancel__:
                break

        self.progress_text.emit('Creating final reports...')

        self.results.create_all_reports(
            loading_threshold=self.options.loading_threshold_to_report,
            reverse=self.options.reversed_sort_loading,

        )

        self.progress_text.emit('Done!')

        self.logger.add_info('Ejecutado en {0:.2f} scs. para {1} casos'.format(
            time.time()-tm0, len(self.results.time_array)))

    def run(self):
        """

        :return:
        """
        start = time.time()

        self.opf()
        self.progress_text.emit('Done!')

        end = time.time()
        self.results.elapsed = end - start


if __name__ == '__main__':

    import GridCal.Engine.basic_structures as bs
    import GridCal.Engine.Devices as dev
    from GridCal.Engine import FileOpen

    folder = r'\\mornt4.ree.es\DESRED\DPE-Internacional\Interconexiones\FRANCIA\2023 Tracking changes\v0-v4-comparison\Pmode3-5GW'
    fname = os.path.join(folder, '23-2-30-9_00_pmode3.gridcal')

    circuit = FileOpen(fname).open()

    areas_from_idx = [0]
    areas_to_idx = [1]

    # areas_from_idx = [7]
    # areas_to_idx = [0, 1, 2, 3, 4]

    areas_from = [circuit.areas[i] for i in areas_from_idx]
    areas_to = [circuit.areas[i] for i in areas_to_idx]

    compatible_areas = True
    for a1 in areas_from:
        if a1 in areas_to:
            compatible_areas = False
            print("The area from '{0}' is in the list of areas to. This cannot be.".format(a1.name),
                  'Incompatible areas')

    for a2 in areas_to:
        if a2 in areas_from:
            compatible_areas = False
            print("The area to '{0}' is in the list of areas from. This cannot be.".format(a2.name),
                  'Incompatible areas')

    lst_from = circuit.get_areas_buses(areas_from)
    lst_to = circuit.get_areas_buses(areas_to)
    lst_br = circuit.get_inter_areas_branches(areas_from, areas_to)
    lst_br_hvdc = circuit.get_inter_areas_hvdc_branches(areas_from, areas_to)

    idx_from = np.array([i for i, bus in lst_from])
    idx_to = np.array([i for i, bus in lst_to])
    idx_br = np.array([i for i, bus, sense in lst_br])
    sense_br = np.array([sense for i, bus, sense in lst_br])
    idx_hvdc_br = np.array([i for i, bus, sense in lst_br_hvdc])
    sense_hvdc_br = np.array([sense for i, bus, sense in lst_br_hvdc])

    if len(idx_from) == 0:
        print('The area "from" has no buses!')

    if len(idx_to) == 0:
        print('The area "to" has no buses!')

    if len(idx_br) == 0:
        print('There are no inter-area branches!')


    options = OptimalNetTransferCapacityOptions(
        area_from_bus_idx=idx_from,
        area_to_bus_idx=idx_to,
        mip_solver=bs.MIPSolvers.CBC,
        generation_formulation=dev.GenerationNtcFormulation.Proportional,
        monitor_only_sensitive_branches=True,
        branch_sensitivity_threshold=0.05,
        skip_generation_limits=True,
        consider_contingencies=True,
        consider_gen_contingencies=True,
        consider_hvdc_contingencies=True,
        consider_nx_contingencies=True,
        generation_contingency_threshold=1000,
        dispatch_all_areas=False,
        tolerance=1e-2,
        sensitivity_dT=100.0,
        transfer_method=AvailableTransferMode.InstalledPower,
        # todo: checkear si queremos el ptdf por potencia generada
        perform_previous_checks=False,
        weight_power_shift=1e5,
        weight_generation_cost=1e2,
        time_limit_ms=1e4,
        loading_threshold_to_report=.98
    )

    print('Running optimal net transfer capacity...')

    # set optimal net transfer capacity driver instance
    start = 0
    end = 1  #circuit.get_time_number()-1

    driver = OptimalNetTransferCapacityTimeSeriesDriver(
        grid=circuit,
        options=options,
        start_=start,
        end_=end,
        use_clustering=False,
        cluster_number=1)

    driver.run()

    driver.results.save_report(path_out=folder)
    # driver.results.make_report()
