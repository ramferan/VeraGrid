
import os

import numpy as np

import VeraGridEngine as vg


def test_linear_time_series_with_time_simplification():
    """
    Base test to check equality between a time series linear power flow and a LinearTs analysis
    """
    fname = os.path.join('data', 'grids', 'IEEE39_1W.gridcal')

    grid = vg.open_file(fname)

    # first run a lieear power flow time series to compare to
    opts = vg.PowerFlowOptions(solver_type=vg.SolverType.Linear)
    pf_ts = vg.PowerFlowTimeSeriesDriver(grid=grid, options=opts)
    pf_ts.run()
    Pf_true = pf_ts.results.Sf.real

    lin_ts = vg.LinearAnalysisTs(grid=grid, distributed_slack=False, correct_values=False)
    P = grid.get_Pbus_prof()
    Pf = lin_ts.get_time_flows(P=P)

    assert np.allclose(Pf, Pf_true)


def test_linear_time_series_with_time_simplification_randomized():
    """
    Check equality between a time series linear power flow and a LinearTs analysis with random line outages
    """
    fname = os.path.join('data', 'grids', 'IEEE39_1W.gridcal')

    grid = vg.open_file(fname)

    nbr = grid.get_branch_number()
    nt = grid.get_time_number()

    branches = grid.get_branches()

    # simulate 10 failures
    for i in range(10):
        t_idx = np.random.randint(low=0, high=nt)
        br_idx = np.random.randint(low=0, high=nbr)
        branches[br_idx].active_prof[t_idx] = 0

    # first run a lieear power flow time series to compare to
    opts = vg.PowerFlowOptions(solver_type=vg.SolverType.Linear)
    pf_ts = vg.PowerFlowTimeSeriesDriver(grid=grid, options=opts)
    pf_ts.run()
    Pf_true = pf_ts.results.Sf.real

    lin_ts = vg.LinearAnalysisTs(grid=grid, distributed_slack=False, correct_values=False)
    P = grid.get_Pbus_prof()
    Pf = lin_ts.get_time_flows(P=P)

    assert np.allclose(Pf, Pf_true)


