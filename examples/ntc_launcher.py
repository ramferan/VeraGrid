
import os
import time
import numpy as np
import time
import GridCal.Engine.Devices as dev
import GridCal.Engine.basic_structures as bs
from GridCal.Engine import FileOpen
from GridCal.Engine.Simulations.ATC.available_transfer_capacity_driver import AvailableTransferMode
from GridCal.Engine.Simulations.NTC.ntc_options import OptimalNetTransferCapacityOptions
from GridCal.Engine.Simulations.NTC.ntc_ts_driver import OptimalNetTransferCapacityTimeSeriesDriver


def ntc_launcher_by_path(gridcal_path):
    tm0 = time.time()
    print('Loading grical circuit... ')
    grid = FileOpen(gridcal_path).open()
    print(f'[{time.time() - tm0:.2f} scs.]')

    return ntc_launcher(grid=grid)

def ntc_launcher(
        grid,
        start=0,
        end=None,
        areas_from_idx=[0],
        areas_to_idx=[1],
        consider_contingencies=True,
        consider_gen_contingencies=True,
        consider_hvdc_contingencies=True,
        consider_nx_contingencies=True,
        monitor_only_sensitive_branches=True,
        branch_sensitivity_threshold=0.05,
        skip_generation_limits=True,
        generation_contingency_threshold=1000,
        time_limit_ms=1e5,
        loading_threshold_to_report=98,
        use_clustering=True,
        cluster_number=200):

    if end is None:
        end = grid.get_time_number() - 1

    areas_from = [grid.areas[i] for i in areas_from_idx]
    areas_to = [grid.areas[i] for i in areas_to_idx]

    for a1 in areas_from:
        if a1 in areas_to:
            print("The area from '{0}' is in the list of areas to. This cannot be.".format(a1.name),
                  'Incompatible areas')

    for a2 in areas_to:
        if a2 in areas_from:
            print("The area to '{0}' is in the list of areas from. This cannot be.".format(a2.name),
                  'Incompatible areas')

    lst_from = grid.get_areas_buses(areas_from)
    lst_to = grid.get_areas_buses(areas_to)
    lst_br = grid.get_inter_areas_branches(areas_from, areas_to)

    idx_from = np.array([i for i, bus in lst_from])
    idx_to = np.array([i for i, bus in lst_to])
    idx_br = np.array([i for i, bus, sense in lst_br])

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
        monitor_only_sensitive_branches=monitor_only_sensitive_branches,
        branch_sensitivity_threshold=branch_sensitivity_threshold,
        skip_generation_limits=skip_generation_limits,
        consider_contingencies=consider_contingencies,
        consider_gen_contingencies=consider_gen_contingencies,
        consider_hvdc_contingencies=consider_hvdc_contingencies,
        consider_nx_contingencies=consider_nx_contingencies,
        dispatch_all_areas=False,
        generation_contingency_threshold=generation_contingency_threshold,
        tolerance=1e-2,
        sensitivity_dT=100.0,
        transfer_method=AvailableTransferMode.InstalledPower,
        # todo: checkear si queremos el ptdf por potencia generada
        perform_previous_checks=False,
        weight_power_shift=1e5,
        weight_generation_cost=1e2,
        time_limit_ms=time_limit_ms,
        loading_threshold_to_report=loading_threshold_to_report,
    )

    print('Running optimal net transfer capacity...')
    # set optimal net transfer capacity driver instance
    driver = OptimalNetTransferCapacityTimeSeriesDriver(
        grid=grid,
        options=options,
        start_=start,
        end_=end,
        use_clustering=use_clustering,
        cluster_number=cluster_number)

    driver.run()

    # driver.results.create_all_reports(
    #     loading_threshold=loading_threshold_to_report,
    #     reverse=True,
    # )

    return driver.results.get_exchange_power()
