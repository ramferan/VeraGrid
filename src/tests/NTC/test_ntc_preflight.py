# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import importlib.util
import os
import unittest

import numpy as np

# The preflight helpers only depend on numpy. Load the module by path so this
# test runs without importing the whole engine (and its heavy dependencies).
_MODULE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..',
    'VeraGridEngine', 'Simulations', 'NTC', 'ntc_preflight.py'))


def _load_module():
    spec = importlib.util.spec_from_file_location('ntc_preflight', _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Logger:
    """Minimal logger stub: only what the preflight helpers use."""

    def __init__(self):
        self.warnings = list()

    def add_warning(self, msg, device=''):
        self.warnings.append(msg)


class TestNtcPreflight(unittest.TestCase):

    def setUp(self):
        self.preflight = _load_module()

    def test_participation_uses_only_area_buses(self):
        power = np.array([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(self.preflight.participation_per_area(power, [0, 2]), 4.0)
        self.assertEqual(self.preflight.participation_per_area(power, []), 0.0)
        self.assertEqual(self.preflight.participation_per_area(power, [1, 2, 3]), 9.0)

    def test_zero_participation_warns_by_default(self):
        logger = _Logger()
        power = np.zeros(4)

        total = self.preflight.check_participation(
            power=power, bus_idx=[0, 1], area_label='Area 1 (sending)', logger=logger)

        self.assertEqual(total, 0.0)
        self.assertEqual(len(logger.warnings), 1)
        self.assertIn('Area 1 (sending)', logger.warnings[0])
        self.assertIn('relaxed', logger.warnings[0])

    def test_zero_participation_can_be_fatal(self):
        power = np.zeros(4)

        with self.assertRaises(self.preflight.NtcFormulationError):
            self.preflight.check_participation(
                power=power, bus_idx=[0, 1],
                area_label='Area 2 (receiving)', raise_on_zero=True)

    def test_empty_area_message(self):
        logger = _Logger()

        self.preflight.check_participation(
            power=np.ones(4), bus_idx=[], area_label='Area 2', logger=logger)

        self.assertIn('has no bus', logger.warnings[0])

    def test_valid_participation_is_silent(self):
        logger = _Logger()

        total = self.preflight.check_participation(
            power=np.array([1.0, -2.0, 3.0]), bus_idx=[0, 1], logger=logger)

        self.assertEqual(total, 3.0)
        self.assertEqual(logger.warnings, [])

    def test_no_inter_area_link_raises(self):
        with self.assertRaises(self.preflight.NtcFormulationError):
            self.preflight.validate_inter_area_links([], [])

        with self.assertRaises(self.preflight.NtcFormulationError):
            self.preflight.validate_inter_area_links([], [], [])

    def test_any_inter_area_element_is_accepted(self):
        self.preflight.validate_inter_area_links([(0, 1.0)], [])
        self.preflight.validate_inter_area_links([], [(0, -1.0)])
        self.preflight.validate_inter_area_links([], [], [(0, 1.0)])

    def test_empty_n1_is_reported(self):
        logger = _Logger()

        self.preflight.warn_if_n1_is_empty(logger, lodf_nx=[], contingency_rates=np.ones(3))
        self.assertIn('no contingency groups', logger.warnings[0])

        logger = _Logger()
        self.preflight.warn_if_n1_is_empty(logger, lodf_nx=[1], contingency_rates=np.zeros(3))
        self.assertIn('contingency rates', logger.warnings[0])

        logger = _Logger()
        self.preflight.warn_if_n1_is_empty(logger, lodf_nx=[1], contingency_rates=np.ones(3))
        self.assertEqual(logger.warnings, [])


if __name__ == '__main__':
    unittest.main()
