# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
NTC input validation.

These checks are deliberately independent from the rest of the engine (only
numpy is imported) so they can be unit-tested and reused by both the regular
and the strict NTC formulations.

The failure modes covered here are the ones that otherwise reach the LP as a
bare "infeasible":

- an area without participating generation: the exchange proportions become
  0/0 and only the area slack absorbs the transfer, so the reported NTC is a
  relaxed one;
- selected areas with no AC branch, HVDC link or VSC converter between them:
  the exchange objective is undefined.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


class NtcFormulationError(ValueError):
    """The model data cannot produce a valid NTC formulation."""


def participation_per_area(power: np.ndarray, bus_idx: Iterable[int]) -> float:
    """
    Sum of the absolute reference power of the elements inside the area.

    :param power: per-bus reference power (p.u.)
    :param bus_idx: bus indices of the area
    :return: sum of |power| over the area buses
    """
    power_array = np.asarray(power, dtype=float)
    idx = np.asarray(list(bus_idx), dtype=int)

    if idx.size == 0:
        return 0.0

    mask = np.isin(np.arange(power_array.size), idx, assume_unique=False)

    return float(np.sum(np.abs(power_array[mask])))


def check_participation(power: np.ndarray,
                        bus_idx: Iterable[int],
                        area_label: str = 'area',
                        raise_on_zero: bool = False,
                        logger: Any = None,
                        tolerance: float = 1e-12) -> float:
    """
    Check that an area has elements able to take part in the exchange.

    With ``raise_on_zero=False`` (the historical behaviour) a warning is
    logged and the caller keeps going, which yields a solution relaxed by the
    area slack. With ``raise_on_zero=True`` the formulation is aborted with a
    message that names the area.

    :param power: per-bus reference power (p.u.)
    :param bus_idx: bus indices of the area
    :param area_label: human readable area name for the message
    :param raise_on_zero: raise instead of warning when there is no participation
    :param logger: optional logger (``add_warning`` is used)
    :param tolerance: values below this are considered zero
    :return: sum of |power| over the area
    """
    total = participation_per_area(power, bus_idx)

    if total > tolerance:
        return total

    idx = list(bus_idx)

    if len(idx) == 0:
        message = (f'{area_label} has no bus: the exchange cannot be balanced. '
                   f'Check the area assignment of the buses.')
    else:
        message = (f'{area_label} has no element able to participate in the '
                   f'exchange: every generator is inactive, non-dispatchable '
                   f'or has zero reference power. The NTC would be relaxed by '
                   f'the area slack.')

    if raise_on_zero:
        raise NtcFormulationError(message)

    if logger is not None:
        logger.add_warning(message, device='NTC OPF')

    return total


def validate_inter_area_links(inter_area_branches: Iterable[Any],
                              inter_area_hvdcs: Iterable[Any],
                              inter_area_vsc: Iterable[Any] | None = None) -> None:
    """
    Fail when the selected areas share no element that can carry the exchange.

    :param inter_area_branches: AC branches between the areas
    :param inter_area_hvdcs: HVDC links between the areas
    :param inter_area_vsc: VSC converters between the areas (optional)
    :raises NtcFormulationError: when the three collections are empty
    """
    n_ac = len(list(inter_area_branches)) if inter_area_branches is not None else 0
    n_hvdc = len(list(inter_area_hvdcs)) if inter_area_hvdcs is not None else 0
    n_vsc = len(list(inter_area_vsc)) if inter_area_vsc is not None else 0

    if (n_ac + n_hvdc + n_vsc) == 0:
        raise NtcFormulationError(
            'There is no AC branch, HVDC link or VSC converter between the '
            'selected areas, so the exchange is undefined. Check that the '
            'areas are adjacent and that the tie elements are active.')


def warn_if_n1_is_empty(logger: Any, lodf_nx: Iterable[Any], contingency_rates: np.ndarray) -> None:
    """
    Make the silent "N-1 turned off" cases visible.

    ``LODF_NX`` is built from the contingency groups stored in the model, so a
    model without contingencies runs a plain NTC while the user believes N-1
    is enforced.

    :param logger: logger instance
    :param lodf_nx: contingency screening structure
    :param contingency_rates: per-branch contingency rates
    """
    if logger is None:
        return

    if len(list(lodf_nx)) == 0:
        logger.add_warning(
            'The model has no contingency groups: N-1 constraints are skipped',
            device='NTC OPF')
        return

    rates = np.asarray(contingency_rates, dtype=float)

    if rates.size > 0 and np.all(rates <= 0):
        logger.add_warning(
            'All contingency rates are <= 0: N-1 constraints will be empty',
            device='NTC OPF')
