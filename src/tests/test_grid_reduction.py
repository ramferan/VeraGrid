# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import os
import numpy as np
import VeraGridEngine.api as gce
from VeraGridEngine.Topology.GridReduction.ptdf_grid_reduction import ptdf_reduction, ptdf_reduction_projected
from VeraGridEngine.Topology.GridReduction.ward_equivalents import ward_standard_reduction
from VeraGridEngine.Topology.GridReduction.di_shi_grid_reduction import di_shi_reduction


def test_ward_reduction():
    """
    Test to check the PTDF reduction
    :return:
    """
    fname = os.path.join('data', 'grids', 'case89pegase.m')
    grid = gce.open_file(filename=fname)

    remove_bus_idx = np.array([21, 36, 44, 50, 53])
    expected_boundary_idx = np.sort(np.array([20, 77, 15, 32]))

    external, boundary, internal, boundary_branches, internal_branches = grid.get_reduction_sets(reduction_bus_indices=remove_bus_idx)

    assert np.equal(expected_boundary_idx, boundary).all()

    pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.Linear)

    pf_res = gce.power_flow(grid=grid, options=pf_options)

    # gce.ward_reduction(grid=grid, reduction_bus_indices=remove_bus_idx, pf_res=pf_res)
    nc = gce.compile_numerical_circuit_at(circuit=grid, t_idx=None)
    lin = gce.LinearAnalysis(nc=nc)

    P0 = grid.get_Pbus()
    Flows0 = lin.get_flows(P0)

    # if grid.has_time_series:
    #     lin_ts = gce.LinearAnalysisTs(grid=grid)
    # else:
    #     lin_ts = None

    grid2, logger = ptdf_reduction(grid=grid,
                                   reduction_bus_indices=remove_bus_idx)

    nc2 = gce.compile_numerical_circuit_at(circuit=grid2, t_idx=None)
    lin2 = gce.LinearAnalysis(nc=nc2)

    # proof that the flows are actually the same
    Pbus4 = grid.get_Pbus()
    Flows4 = lin2.PTDF @ Pbus4
    diff = Flows0[internal_branches] - Flows4

    ok = np.allclose(Flows4, Flows0[internal_branches], atol=1e-10)
    assert ok

    
def test_ptdf_projected_14_reduction():
    """
    Test to check the PTDF projected reduction
    :return:
    """
    remove_bus_idx_list = [np.array([8, 11]), 
                           np.array([9, 10]),
                           np.array([8, 11, 9, 10]),
                           np.array([0, 1, 2])]

    pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR)

    # Open a new grid instance every time
    # Check the reduction works with any combination of buses to remove
    for remove_bus_idx in remove_bus_idx_list:
        fname = os.path.join('data', 'grids', 'case14_to_reduce.veragrid')
        grid = gce.open_file(filename=fname)
        red_grid, logger = ptdf_reduction_projected(grid=grid, reduction_bus_indices=remove_bus_idx)
        pf_res_reduced = gce.power_flow(grid=red_grid, options=pf_options)

        assert pf_res_reduced.converged

    return None

    
def test_ptdf_projected_14_complex_reduction():
    """
    Test to check the PTDF projected reduction
    :return:
    """
    remove_bus_idx_list = [np.array([8, 11]), 
                           np.array([9, 10]),
                           np.array([8, 11, 9, 10]),
                           np.array([0, 1, 2]),
                           np.array([21]),
                           np.array([14, 17]),
                           np.array([14, 17, 21]),
                           np.array([21, 14, 17, 0, 3])]

    pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR)

    # Open a new grid instance every time
    # Check the reduction works with any combination of buses to remove
    for remove_bus_idx in remove_bus_idx_list:
        fname = os.path.join('data', 'grids', 'case14_complex_to_reduce.veragrid')
        # fname = os.path.join('src', 'tests', 'data', 'grids', 'case14_complex_to_reduce.veragrid')
        grid = gce.open_file(filename=fname)
        red_grid, logger = ptdf_reduction_projected(grid=grid, reduction_bus_indices=remove_bus_idx)
        pf_res_reduced = gce.power_flow(grid=red_grid, options=pf_options)

        assert pf_res_reduced.converged


def test_ptdf_projected_14_complex_inactive_reduction():
    """
    Test to check the PTDF projected reduction
    :return:
    """
    remove_bus_idx_list = [np.array([8, 11]), 
                           np.array([9, 10]),
                           np.array([8, 11, 9, 10]),
                           np.array([0, 1, 2]),
                           np.array([21]),
                           np.array([14, 17]),
                           np.array([14, 17, 21]),
                           np.array([21, 14, 17, 0, 3]),
                           np.array([22]),
                           np.array([22, 21]),
                           np.array([22, 21, 14, 17]),
                           np.array([22, 21, 14, 4])]

    pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR)

    # Open a new grid instance every time
    # Check the reduction works with any combination of buses to remove
    for remove_bus_idx in remove_bus_idx_list:
        fname = os.path.join('data', 'grids', 'case14_to_reduce_inactive.veragrid')
        # fname = os.path.join('src', 'tests', 'data', 'grids', 'case14_to_reduce_inactive.veragrid')
        grid = gce.open_file(filename=fname)
        red_grid, logger = ptdf_reduction_projected(grid=grid, reduction_bus_indices=remove_bus_idx)
        pf_res_reduced = gce.power_flow(grid=red_grid, options=pf_options)

        assert pf_res_reduced.converged

        
def test_ptdf_projected():
    """
    Test to check the PTDF projected reduction in a very simple grid
    :return:
    """
    fname = os.path.join('data', 'grids', '5bus_linear.veragrid')
    # fname = os.path.join('src', 'tests', 'data', 'grids', '5bus_linear.veragrid')
    grid = gce.open_file(filename=fname)

    # First run basic linear analysis
    flows_dr = gce.LinearAnalysisDriver(grid=grid, options=gce.LinearAnalysisOptions(distribute_slack=True))
    flows_dr.run()
    flows_branches = flows_dr.results.Sf

    # Then reduce the network
    bus_to_remove = np.array([1])
    red_grid, logger = ptdf_reduction_projected(grid=grid, reduction_bus_indices=bus_to_remove)
    flows_dr_red = gce.LinearAnalysisDriver(grid=red_grid, options=gce.LinearAnalysisOptions(distribute_slack=True))
    flows_dr_red.run()
    flows_branches_red = flows_dr_red.results.Sf

    # print(flows_branches[[2, 3, 4, 5, 6]])
    # print(flows_branches_red)

    assert np.allclose(flows_branches[[2, 3, 4, 5, 6]], flows_branches_red, atol=1e-5)

    
def test_reduction_flows():
    """
    Test to check the reduction flows
    :return:
    """
    # fname = os.path.join('data', 'grids', 'case14.m')
    fname = os.path.join('data', 'grids', 'case14.m')
    grid = gce.open_file(filename=fname)
    pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR)

    # Original power flow
    pf_original_res = gce.power_flow(grid=grid, options=pf_options)

    reduction_buses = np.array([10, 13])

    # Reduced grid with ward equivalent
    fname = os.path.join('data', 'grids', 'case14.m')
    grid = gce.open_file(filename=fname)
    red_grid, logger = ward_standard_reduction(grid=grid,
                                               reduction_bus_indices=reduction_buses,
                                               V0=pf_original_res.voltage)
    pf_reduced_res = gce.power_flow(grid=red_grid, options=pf_options)
    Vabs_ward = np.array([1.06, 1.045, 1.01, 1.01756835, 1.01944961, 1.07, 1.06120408, 1.09, 1.0553105, 1.05036322, 1.05532085, 1.05064385])
    assert np.allclose(abs(pf_reduced_res.voltage), Vabs_ward, atol=1e-4)

    # Reduced grid with di-shi equivalent
    fname = os.path.join('data', 'grids', 'case14.m')
    grid = gce.open_file(filename=fname)
    red_grid, logger = di_shi_reduction(grid=grid,
                                        reduction_bus_indices=reduction_buses,
                                        V0=pf_original_res.voltage)
    pf_reduced_res = gce.power_flow(grid=red_grid, options=pf_options)
    Vabs_dishi = np.array([1.06, 1.045, 1.01, 1.01767086, 1.01951387, 1.07, 1.06151954, 1.09, 1.05593173, 1.05098464, 1.05518856, 1.05038172])
    assert np.allclose(abs(pf_reduced_res.voltage), Vabs_dishi, atol=1e-4)

    # Reduced grid with ptdf equivalent
    fname = os.path.join('data', 'grids', 'case14.m')
    grid = gce.open_file(filename=fname)
    red_grid, logger = ptdf_reduction(grid=grid,
                                        reduction_bus_indices=reduction_buses)
    pf_reduced_res = gce.power_flow(grid=red_grid, options=pf_options)
    Vabs_ptdf = np.array([1.06, 1.045, 1.01, 1.01530529, 1.01762342, 1.07, 1.05771147, 1.09, 1.04927468, 1.0424791, 1.05525675, 1.05166266])
    assert np.allclose(abs(pf_reduced_res.voltage), Vabs_ptdf, atol=1e-4)


if __name__ == '__main__':
    # test_ward_reduction()
    # test_ptdf_projected_14_reduction()
    # test_ptdf_projected_14_complex_reduction()
    # test_ptdf_projected_14_complex_inactive_reduction()
    # test_reduction_flows()
    test_ptdf_projected()