# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import unittest

import numpy as np

try:
    from ortools.linear_solver import pywraplp
except ImportError:
    pywraplp = None

from GridCal.Engine.Simulations.NTC.ntc_opf import formulate_hvdc_Pmode3_single_flow


class TestPmode3Clip(unittest.TestCase):
    """
    Verify that the clipped Pmode3 law is exact in the three regions:

        flow = clip(P0 + k * (theta_f - theta_t), -rate, rate)
    """

    def _solver(self):
        if pywraplp is None:
            self.skipTest('ortools is not available')
        solver = pywraplp.Solver.CreateSolver('CBC')
        if solver is None:
            self.skipTest('CBC solver is not available')
        return solver

    def _flow(self, theta_f, theta_t, P0=0.0, rate_mw=100.0, sbase=100.0):
        solver = self._solver()

        angle_f = solver.NumVar(-np.pi, np.pi, 'angle_f')
        angle_t = solver.NumVar(-np.pi, np.pi, 'angle_t')
        solver.Add(angle_f == theta_f)
        solver.Add(angle_t == theta_t)

        # k = angle_droop * 57.295779513 / sbase = 1 pu/rad
        angle_droop = sbase / 57.295779513

        flow = formulate_hvdc_Pmode3_single_flow(
            solver=solver,
            active=True,
            P0=P0,
            rate=rate_mw,
            Sbase=sbase,
            angle_droop=angle_droop,
            angle_max_f=np.pi,
            angle_max_t=np.pi,
            suffix='test',
            angle_f=angle_f,
            angle_t=angle_t,
            inf=1e20,
        )

        status = solver.Solve()
        self.assertEqual(status, pywraplp.Solver.OPTIMAL)
        return flow.solution_value()

    def test_linear_region(self):
        self.assertAlmostEqual(self._flow(0.5, 0.0), 0.5, places=6)

    def test_linear_region_with_offset(self):
        self.assertAlmostEqual(self._flow(0.3, 0.0, P0=0.2), 0.5, places=6)

    def test_upper_saturation(self):
        self.assertAlmostEqual(self._flow(2.0, 0.0), 1.0, places=6)

    def test_lower_saturation(self):
        self.assertAlmostEqual(self._flow(-2.0, 0.0), -1.0, places=6)


if __name__ == '__main__':
    unittest.main()
