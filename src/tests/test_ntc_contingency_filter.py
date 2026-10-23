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

from GridCal.Engine.basic_structures import Logger
from GridCal.Engine.Simulations.NTC.ntc_opf import formulate_contingency


class TestNtcContingencyFilter(unittest.TestCase):
    """
    Regression tests for the N-1 contingency screening in the REE NTC OPF.

    The monitorisation logic uses max(|alpha|, max|alpha_n1|), i.e. a branch
    is monitored when it is sensitive in N **or** in N-1. The contingency
    formulation must use the same criterion. Before the fix it used
    ``(alpha > threshold) and (alpha_n1 > threshold)``, so a branch that was
    only sensitive under contingency was monitored in the base case but its
    N-1 constraint was never added.
    """

    def _solver(self):
        if pywraplp is None:
            self.skipTest('ortools is not available')
        solver = pywraplp.Solver.CreateSolver('CBC')
        if solver is None:
            self.skipTest('CBC solver is not available')
        return solver

    def _flows(self, solver, n=3):
        return np.array([
            solver.NumVar(-100.0, 100.0, 'flow_{0}'.format(i))
            for i in range(n)
        ], dtype=object)

    def _run(self, solver, alpha, alpha_n1, contingency_rates):
        """
        Build a tiny 3-branch case. Branch 0 is the outaged branch, branch 1
        is the candidate monitored branch and branch 2 is a non-sensitive
        branch. The LODF of branch 1 with respect to the outage is 1.0.
        """
        n = 3
        rates = np.asarray(contingency_rates, dtype=float)
        names = np.array(['b0', 'b1', 'b2'], dtype=object)

        # LODF_NX stores (outaged_branch_indices, lodf_nx), where lodf_nx has
        # shape (n_branches, n_outaged_branches). For a single outage this is
        # (n, 1), not (n, n).
        lodfnx = np.zeros((n, 1), dtype=float)
        lodfnx[1, 0] = 1.0

        lodf_nx = [(np.array([0], dtype=int), lodfnx)]

        flow_f = self._flows(solver=solver, n=n)
        monitor = np.array([True, True, True], dtype=bool)

        return formulate_contingency(
            solver=solver,
            ContingencyRates=rates,
            Sbase=100.0,
            branch_names=names,
            LODF_NX=lodf_nx,
            F=np.array([0, 1, 2], dtype=int),
            T=np.array([1, 2, 0], dtype=int),
            branch_sensitivity_threshold=0.05,
            flow_f=flow_f,
            monitor=monitor,
            alpha=np.asarray(alpha, dtype=float),
            alpha_n1=np.asarray(alpha_n1, dtype=float),
            logger=Logger(),
        )

    def test_branch_sensitive_only_in_n1_is_constrained(self):
        solver = self._solver()

        alpha = np.array([0.0, 0.0, 1.0])
        alpha_n1 = np.zeros((3, 3), dtype=float)
        alpha_n1[1, 0] = 1.0

        flow_n1f, con_alpha, con_idx = self._run(
            solver=solver,
            alpha=alpha,
            alpha_n1=alpha_n1,
            contingency_rates=[100.0, 100.0, 100.0],
        )

        self.assertEqual(len(flow_n1f), 1)
        self.assertEqual(con_idx[0][0], 1)
        self.assertAlmostEqual(float(con_alpha[0]), 1.0)

    def test_branch_sensitive_only_in_n_is_constrained(self):
        solver = self._solver()

        alpha = np.array([0.0, 1.0, 0.0])
        alpha_n1 = np.zeros((3, 3), dtype=float)

        flow_n1f, con_alpha, con_idx = self._run(
            solver=solver,
            alpha=alpha,
            alpha_n1=alpha_n1,
            contingency_rates=[100.0, 100.0, 100.0],
        )

        self.assertEqual(len(flow_n1f), 1)
        self.assertEqual(con_idx[0][0], 1)
        self.assertAlmostEqual(float(con_alpha[0]), 0.0)

    def test_zero_contingency_rate_is_skipped(self):
        solver = self._solver()

        alpha = np.array([0.0, 0.0, 1.0])
        alpha_n1 = np.zeros((3, 3), dtype=float)
        alpha_n1[1, 0] = 1.0

        flow_n1f, con_alpha, con_idx = self._run(
            solver=solver,
            alpha=alpha,
            alpha_n1=alpha_n1,
            contingency_rates=[100.0, 0.0, 100.0],
        )

        # The branch has no N-1 rate: the constraint is skipped instead of
        # forcing a zero post-contingency flow.
        self.assertEqual(len(flow_n1f), 0)
        self.assertEqual(len(con_idx), 0)


if __name__ == '__main__':
    unittest.main()
