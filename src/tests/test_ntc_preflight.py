# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import unittest

import numpy as np
from scipy.sparse import csc_matrix

try:
    from ortools.linear_solver import pywraplp
except ImportError:
    pywraplp = None

from GridCal.Engine.basic_structures import Logger
from GridCal.Engine.Simulations.NTC.ntc_opf import (
    NtcFormulationError,
    formulate_proportional_generation,
    validate_inter_area_links,
)


class TestNtcPreflight(unittest.TestCase):
    """
    The proportional formulation divides by the reference power of each area.
    With no participating generation the shares are 0/0, the NaN coefficients
    reach the LP and CBC reports a plain "infeasible". These tests pin the
    early, explicit failure instead.
    """

    def _solver(self):
        if pywraplp is None:
            self.skipTest('ortools is not available')
        solver = pywraplp.Solver.CreateSolver('CBC')
        if solver is None:
            self.skipTest('CBC solver is not available')
        return solver

    @staticmethod
    def _cgen(bus_of_gen, n_bus=3):
        """Cgen[generator, bus] = 1 connectivity matrix (ngen x nbus)."""
        n_gen = len(bus_of_gen)
        rows = np.arange(n_gen)
        cols = np.asarray(bus_of_gen, dtype=int)
        return csc_matrix((np.ones(n_gen), (rows, cols)), shape=(n_gen, n_bus))

    def _formulate(self, **overrides):
        params = dict(
            solver=self._solver(),
            generator_active=np.array([True, True]),
            generator_dispatchable=np.array([True, True]),
            generator_cost=np.ones(2),
            generator_names=['gen_a', 'gen_b'],
            inf=1e20,
            ngen=2,
            Cgen=self._cgen((0, 1)),
            Pgen=np.array([0.2, 0.2]),
            Pmax=np.array([1.0, 1.0]),
            Pmin=np.array([0.0, 0.0]),
            Pref=np.array([1.0, 1.0]),
            a1=[0],
            a2=[1],
            logger=Logger(),
        )
        params.update(overrides)
        return formulate_proportional_generation(**params)

    def test_area_without_generators_raises(self):
        with self.assertRaises(NtcFormulationError) as ctx:
            self._formulate(
                generator_active=np.array([True]),
                generator_dispatchable=np.array([True]),
                generator_cost=np.ones(1),
                generator_names=['gen_a'],
                ngen=1,
                Cgen=self._cgen((0,)),
                Pgen=np.array([0.2]),
                Pmax=np.array([1.0]),
                Pmin=np.array([0.0]),
                Pref=np.array([1.0]),
            )

        self.assertIn('Area 2 has no generator', str(ctx.exception))

    def test_non_dispatchable_area_raises(self):
        with self.assertRaises(NtcFormulationError) as ctx:
            self._formulate(
                generator_dispatchable=np.array([False, False]),
            )

        message = str(ctx.exception)
        self.assertIn('Area 1 has no generator able to participate', message)
        self.assertIn('dispatchable=False', message)

    def test_zero_reference_power_raises(self):
        # This is the IEEE 118 case: generators exist but their reference
        # power is zero, so the shares are 0/0.
        with self.assertRaises(NtcFormulationError) as ctx:
            self._formulate(
                Pref=np.array([0.0, 0.0]),
            )

        self.assertIn('zero reference power', str(ctx.exception))

    def test_valid_participation_formulates(self):
        generation, delta, a1_idx, a2_idx, power_shift, _ = self._formulate()

        self.assertEqual(sorted(a1_idx), [0])
        self.assertEqual(sorted(a2_idx), [1])
        self.assertIsNotNone(power_shift)
        self.assertEqual(len(generation), 2)

    def test_no_inter_area_link_raises(self):
        with self.assertRaises(NtcFormulationError):
            validate_inter_area_links([], [])

    def test_inter_area_link_is_accepted(self):
        # (branch index, sign) tuples as returned by get_inter_areas_branches
        validate_inter_area_links([(0, 1.0)], [])


if __name__ == '__main__':
    unittest.main()
