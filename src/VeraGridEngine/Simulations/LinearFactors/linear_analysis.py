# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from collections import deque
import numpy as np
import numba as nb
import warnings
import scipy.sparse as sp
from typing import Any, List, Dict, Tuple, TYPE_CHECKING

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve as scipy_spsolve

from VeraGridEngine.enumerations import DeviceType
from VeraGridEngine.basic_structures import Logger, Vec, IntVec, CxVec, Mat, ObjVec, CxMat, BoolVec, IntMat, Vector
from VeraGridEngine.DataStructures.numerical_circuit import NumericalCircuit
from VeraGridEngine.Compilers.circuit_to_data import compile_numerical_circuit_at
from VeraGridEngine.Devices.Events.contingency_group import ContingencyGroup
from VeraGridEngine.Devices.Events.contingency import Contingency
from VeraGridEngine.Simulations.Derivatives.ac_jacobian import AC_jacobian
from VeraGridEngine.Simulations.Derivatives.csc_derivatives import dSf_dV_csc
from VeraGridEngine.Utils.Sparse.csc import dense_to_csc
import VeraGridEngine.Utils.Sparse.csc2 as csc
from VeraGridEngine.Utils.MIP.selected_interface import lpDot1D_changes
from VeraGridEngine.enumerations import ContingencyOperationTypes, ConverterControlType, HvdcControlType
from VeraGridEngine.Topology.topology import find_different_states

if TYPE_CHECKING:
    from VeraGridEngine.Devices.multi_circuit import MultiCircuit


@nb.jit()
def make_contingency_flows(base_flow: Vec,
                           lodf_factors: sp.csc_matrix,
                           ptdf_factors: sp.csc_matrix,
                           injections: Vec,
                           threshold):
    """
    Compute the general contingency flows
    :param base_flow: base flow (number of branches)
    :param lodf_factors: LODF factors (number of branches, number of branch contingencies)
    :param ptdf_factors: PTDF factors (number of branches, number of injection contingencies)
    :param injections: Array of contingency injections)
    :param threshold: PTDF and LODF threshold
    :return: contingency flows (number of branches)
    """
    branch_number = lodf_factors.shape[0]
    branch_contingency_number = lodf_factors.shape[1]
    injection_number = ptdf_factors.shape[1]
    flow_n1: Vec = np.zeros(branch_number)

    if branch_number < 100:
        for m in range(branch_number):

            # copy the base flow
            flow_n1[m] = base_flow[m]

            # add the branch contingency influences
            for c in range(branch_contingency_number):
                if abs(lodf_factors[m, c]) > threshold:
                    flow_n1[m] += lodf_factors[m, c] * base_flow[c]

            # add the injection influences
            for c in range(injection_number):
                if abs(ptdf_factors[m, c]) > threshold:
                    flow_n1[m] += ptdf_factors[m, c] * injections[c]
    else:

        # parallel version

        for m in nb.prange(branch_number):

            # copy the base flow
            flow_n1[m] = base_flow[m]

            for c in range(branch_contingency_number):
                if abs(lodf_factors[m, c]) > threshold:
                    flow_n1[m] += lodf_factors[m, c] * base_flow[c]

            for c in range(injection_number):
                if abs(ptdf_factors[m, c]) > threshold:
                    flow_n1[m] += ptdf_factors[m, c] * injections[c]

    return flow_n1


def make_jacobian_ptdf(Ybus: sp.csc_matrix,
                       Yf: sp.csc_matrix,
                       F: IntVec,
                       T: IntVec,
                       V: CxVec,
                       pq: IntVec,
                       pv: IntVec,
                       bus_types: IntVec,
                       distribute_slack: bool = False) -> Mat:
    """
    Compute the AC-PTDF
    :param Ybus: admittance matrix
    :param Yf: Admittance matrix of the buses "from"
    :param F: array if Branches "from" bus indices
    :param T: array if Branches "to" bus indices
    :param V: voltages array
    :param pq: array of pq node indices
    :param pv: array of pv node indices
    :param bus_types: Array of bus types
    :param distribute_slack: distribute slack?
    :return: AC-PTDF matrix (Branches, buses)
    """
    n = len(V)
    pvpq = np.r_[pv, pq]
    npq = len(pq)
    npv = len(pv)

    # compute the Jacobian
    J = AC_jacobian(Ybus, V, pvpq, pq)

    # if distribute_slack:
    #     dP: Mat = np.ones((n, n)) * (-1 / (n - 1))
    #     for i in range(n):
    #         dP[i, i] = 1.0
    # else:
    #     dP = np.eye(n, n)
    dP = make_dP_from_bus_types(bus_types=bus_types, distribute_slack=distribute_slack)

    # compose the compatible array (the Q increments are considered zero
    dQ = np.zeros((npq, n))
    dS = np.r_[dP[pvpq, :], dQ]

    # solve the voltage increments
    dx, ok = csc.spsolve_csc(J, dS)

    # compute branch derivatives
    dSf_dVm, dSf_dVa = dSf_dV_csc(Yf.tocsc(), V, F, T)

    # compose the final AC-PTDF
    dPf_dVa = csc.mat_to_scipy(dSf_dVa.real)[:, pvpq]
    dPf_dVm = csc.mat_to_scipy(dSf_dVm.real)[:, pq]
    PTDF = sp.hstack((dPf_dVa, dPf_dVm)) * dx

    return PTDF


@nb.jit(cache=True)
def make_dP_from_bus_types(bus_types: IntVec, distribute_slack: bool) -> Mat:
    """
    Build the dP / Dij matrix used to modify the PTDF.

    bus_types:
        1 -> PQ
        2 -> PV
        3 -> Slack

    If distribute_slack=True:
        The balancing injection is distributed equally among PV + Slack buses.

        P_eff = dP @ P
        P_eff = P - w * sum(P)

    If distribute_slack=False:
        Return identity, preserving the current single-slack PTDF behavior.
    """

    PQ = 1
    PV = 2
    SLACK = 3

    n = bus_types.shape[0]

    # Start with identity
    dP = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        dP[i, i] = 1.0

    if not distribute_slack:
        return dP

    # Count participating buses: PV + Slack
    n_participating = 0

    for i in range(n):
        if bus_types[i] == PV or bus_types[i] == SLACK:
            n_participating += 1

    if n_participating == 0:
        raise ValueError("No PV or Slack buses found for distributed slack")

    share = 1.0 / n_participating

    # dP = I - w * 1^T
    #
    # Since w_i = share for PV/Slack buses and zero for PQ buses,
    # only the rows of PV/Slack buses are modified.
    for i in range(n):
        if bus_types[i] == PV or bus_types[i] == SLACK:
            for j in range(n):
                dP[i, j] -= share

    return dP


@nb.jit(cache=True)
def make_corrected_injections(P: Vec, bus_types: IntVec, distribute_slack: bool) -> Vec:
    """
    Compute the effective active-power injection vector used for KCL checks.

    If distribute_slack=True:
        mismatch is shared equally among PV + Slack buses.

    If distribute_slack=False:
        mismatch is assigned only to the Slack bus or buses.
    """

    PQ = 1
    PV = 2
    SLACK = 3

    n = P.shape[0]

    P_eff = P.copy()

    mismatch = 0.0

    for i in range(n):
        mismatch += P[i]

    n_participating = 0

    for i in range(n):
        if distribute_slack:
            if bus_types[i] == PV or bus_types[i] == SLACK:
                n_participating += 1
        else:
            if bus_types[i] == SLACK:
                n_participating += 1

    if n_participating == 0:
        raise ValueError("No slack-participating buses found")

    share = mismatch / n_participating

    for i in range(n):
        if distribute_slack:
            if bus_types[i] == PV or bus_types[i] == SLACK:
                P_eff[i] -= share
        else:
            if bus_types[i] == SLACK:
                P_eff[i] -= share

    return P_eff


@nb.jit(cache=True)
def make_corrected_injections_2d(P: Mat, bus_types: IntVec, distribute_slack: bool) -> Mat:
    """
    Compute the effective active-power injection vector used for KCL checks.

    If distribute_slack=True:
        mismatch is shared equally among PV + Slack buses.

    If distribute_slack=False:
        mismatch is assigned only to the Slack bus or buses.
    """

    PQ = 1
    PV = 2
    SLACK = 3

    nt, n = P.shape

    P_eff = P.copy()

    n_participating = 0
    for i in range(n):
        if distribute_slack:
            if bus_types[i] == PV or bus_types[i] == SLACK:
                n_participating += 1
        else:
            if bus_types[i] == SLACK:
                n_participating += 1

    if n_participating == 0:
        raise ValueError("No slack-participating buses found")

    mismatch = np.zeros(nt)

    for t in range(nt):
        for i in range(n):
            mismatch[t] += P[t, i]

        share = mismatch[t] / n_participating

        for i in range(n):
            if distribute_slack:
                if bus_types[i] == PV or bus_types[i] == SLACK:
                    P_eff[t, i] -= share
            else:
                if bus_types[i] == SLACK:
                    P_eff[t, i] -= share

    return P_eff


def make_ptdf(Bpqpv: sp.csc_matrix,
              Bf: sp.csc_matrix,
              no_slack: IntVec,
              bus_types: IntVec,
              distribute_slack: bool = True) -> Mat:
    """
    Build the PTDF matrix
    :param Bpqpv: DC-linear susceptance matrix already sliced
    :param Bf: Bus-branch "from" susceptance matrix
    :param no_slack: array of sorted pq and pv node indices
    :param bus_types: array of bus types
    :param distribute_slack: distribute the slack?
    :return: PTDF matrix. It is a full matrix of dimensions Branches x buses
    """

    n = Bf.shape[1]  # number of buses

    # if distribute_slack:
    #     dP: Mat = np.ones((n, n)) * (-1 / (n - 1))
    #     for i in range(n):
    #         dP[i, i] = 1.0
    # else:
    #     dP: Mat = np.eye(n, n)
    dP = make_dP_from_bus_types(bus_types=bus_types, distribute_slack=distribute_slack)

    # solve for change in voltage angles
    dTheta = np.zeros((n, n))
    # Bref = Bbus[noslack, :][:, noref].tocsc()
    dtheta_ref = scipy_spsolve(Bpqpv, dP[no_slack, :])

    if sp.issparse(dtheta_ref):
        dTheta[no_slack, :] = dtheta_ref.toarray()
    else:
        dTheta[no_slack, :] = dtheta_ref

    # compute the corresponding change in branch Sf
    # Bf is a sparse matrix
    H = Bf @ dTheta

    # PTDF = Bf x (B^-1 x P)

    return H


def make_acdc_ptdf(nc: NumericalCircuit,
                   logger: Logger,
                   bus_types: IntVec,
                   distribute_slack: bool = False,
                   converters_as_setpoint: bool = False) -> Mat:
    """
    Build the ACDC PTDF matrix
    :param nc: NumericalCircuit
    :param logger: Logger
    :param bus_types: Array of bus types for the distributed slack
    :param distribute_slack: distribute the slack?
    :param converters_as_setpoint: treat every converter (VSC and HVDC) as a set-point
        device with the small admittance, ignoring droop control modes
    :return: PTDF matrix. It is a full matrix of dimensions Branches x buses
    """
    n = nc.nbus

    # mount the base matrix
    A = lil_matrix((n, n))
    Af = lil_matrix((nc.nbr, n))

    for k in range(nc.nbr):
        f = nc.passive_branch_data.F[k]
        t = nc.passive_branch_data.T[k]

        if nc.bus_data.is_dc[f] and nc.bus_data.is_dc[t]:
            # this is a dc branch
            ys: float = float(nc.passive_branch_data.active[k]) / (nc.passive_branch_data.R[k] + 1e-20)

        elif not nc.bus_data.is_dc[f] and not nc.bus_data.is_dc[t]:
            # AC branch: the DC susceptance includes the tap module, same as Bf
            m: float = float(nc.active_branch_data.tap_module[k])
            ys = float(nc.passive_branch_data.active[k]) / (nc.passive_branch_data.X[k] * m + 1e-20)

        else:
            # this is an error
            raise AttributeError(f"The branch {k} is nether fully AC not fully DC :(")

        Af[k, f] = ys
        Af[k, t] = -ys

        A[f, f] += ys
        A[f, t] -= ys
        A[t, f] -= ys
        A[t, t] += ys

    # Fake impedances for converters.
    # Set-point devices get a small coupling that only keeps the AC-DC topology visible
    # In the NTC the free mode has to have a higher or equal NTC than the Pmode3
    for k in range(nc.nvsc):
        f = nc.vsc_data.F[k]
        t = nc.vsc_data.T[k]

        if (not converters_as_setpoint and
                nc.vsc_data.control1_int[k] == ConverterControlType.Pdc_angle_droop.idx()):
            # P-MODE 3: The VSC behaves as a droop control
            # P = P0 + k * (theta_f - theta_t)
            # k is in MW/deg, we need it in p.u./rad
            ys = nc.vsc_data.control1_val[k] * 57.295779513 / nc.Sbase
        else:
            # Non-droop VSCs: pure setpoint injections without fictitious AC admittance
            ys = 0.0

        A[f, f] += ys
        A[f, t] -= ys
        A[t, f] -= ys
        A[t, t] += ys

    # fake impedances for hvdc, same reasoning as the VSCs
    for k in range(nc.nhvdc):
        f = nc.hvdc_data.F[k]
        t = nc.hvdc_data.T[k]

        if (not converters_as_setpoint and
                nc.hvdc_data.control_mode_int[k] == HvdcControlType.type_0_free.idx()):
            # Free mode: P = Pset + angle_droop * (theta_f - theta_t)
            # angle_droop is in MW/deg, we need it in p.u./rad
            ys = nc.hvdc_data.angle_droop[k] * 57.295779513 / nc.Sbase
        else:
            # Pset mode: pure setpoint injections without fictitious AC admittance
            ys = 0.0

        A[f, f] += ys
        A[f, t] -= ys
        A[t, f] -= ys
        A[t, t] += ys

    # detect how to slice
    no_slack = list()
    dc_sl = list()
    ac_sl = list()
    for i in range(n):
        if nc.bus_data.is_dc[i]:
            if nc.bus_data.is_vm_controlled[i]:
                dc_sl.append(i)
            else:
                no_slack.append(i)
        else:
            if nc.bus_data.is_vm_controlled[i] and nc.bus_data.is_va_controlled[i]:
                ac_sl.append(i)
            else:
                no_slack.append(i)

    # if distribute_slack:
    #     dP: Mat = np.eye(n) - 1.0 / n
    # else:
    #     dP = np.eye(n, n)
    dP = make_dP_from_bus_types(bus_types=bus_types, distribute_slack=distribute_slack)

    Ared = A[no_slack, :][:, no_slack]
    Pred = dP[no_slack, :]

    dTheta = np.zeros((n, n))

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            dtheta_ref = sp.linalg.spsolve(Ared.tocsc(), Pred)
            dTheta[no_slack, :] = dtheta_ref
        except sp.linalg.MatrixRankWarning as e:
            logger.add_error("ACDC PTDF singular matrix. Does each subgrid have a slack?")

    H = Af @ dTheta

    return H


def make_lodf(Cf: sp.csc_matrix,
              Ct: sp.csc_matrix,
              PTDF: Mat,
              correct_values: bool = False,
              numerical_zero: float = 1e-10) -> Mat:
    """
    Compute the LODF matrix
    :param Cf: Branch "from" -bus connectivity matrix
    :param Ct: Branch "to" -bus connectivity matrix
    :param PTDF: PTDF matrix in numpy array form (Branches, buses)
    :param correct_values: correct values out of the interval
    :param numerical_zero: value considered zero in numerical terms (i.e. 1e-10)
    :return: LODF matrix of dimensions (Branches, Branches)
    """
    nl = PTDF.shape[0]

    # compute the connectivity matrix
    Cft = Cf - Ct
    H = PTDF * Cft.T

    # this loop avoids the divisions by zero
    # in those cases the LODF column should be zero
    LODF: Mat = np.zeros((nl, nl))
    div = 1 - H.diagonal()
    for j in range(H.shape[1]):
        if abs(div[j]) > numerical_zero:
            LODF[:, j] = H[:, j] / div[j]

    # replace the diagonal elements by -1
    # old code
    # LODF = LODF - sp.diags(LODF.diagonal()) - sp.eye(nl, nl), replaced by:
    for i in range(nl):
        LODF[i, i] = - 1.0

    if correct_values:
        LODF[LODF > 1.2] = 0
        LODF[LODF < -1.2] = 0
        # LODF[LODF > 1.] = 1.
        # LODF[LODF < -1.] = 1.

    return LODF


@nb.jit(cache=True)
def make_transfer_limits(ptdf: Mat,
                         flows: Vec,
                         rates: Vec) -> Vec:
    """
    Compute the maximum transfer limits of each branch in normal operation
    :param ptdf: power transfer distribution factors matrix (n-branch, n-bus)
    :param flows: base Sf in MW
    :param rates: array of branch rates
    :return: Max transfer limits vector  (n-branch)
    """
    nbr = ptdf.shape[0]
    nbus = ptdf.shape[1]
    tmc: Vec = np.zeros(nbr)

    for m in range(nbr):
        for i in range(nbus):

            if ptdf[m, i] != 0.0:
                val = (rates[m] - flows[m]) / ptdf[m, i]  # I want it with sign

                # update the transference value
                if abs(val) > abs(tmc[m]):
                    tmc[m] = val

    return tmc


@nb.jit(cache=True)
def create_M_numba(lodf: Mat, branch_contingency_indices) -> Mat:
    """

    :param lodf:
    :param branch_contingency_indices:
    :return:
    """
    M: Mat = np.empty((len(branch_contingency_indices), len(branch_contingency_indices)))
    for i in range(len(branch_contingency_indices)):
        for j in range(len(branch_contingency_indices)):
            if i == j:
                M[i, j] = 1.0
            else:
                M[i, j] = -lodf[branch_contingency_indices[i], branch_contingency_indices[j]]
    return M


def compute_phase_shift_terms(nc: NumericalCircuit) -> Tuple[Vec, Vec]:
    """
    Build the per-unit contribution of the branch phase shifts to the DC branch flows.

    A branch with susceptance ``b = 1 / (X · m)`` and phase shift ``tau`` has
    ``Pf = b · (theta_f - theta_t - tau)``. This is the same convention as
    ``compute_linear_admittances`` and ``power_flow_post_process_linear``.
    The tap ratio is ``t = m · exp(j · tau)``, so both the tap module and the
    tap phase enter the DC flow.

    Separating the constant part turns the DC system into ``B·theta = P - Pshift``, so the flows are
    ``Pf = PTDF @ (P - Pshift) + shift_flow`` with ``shift_flow = -b·tau`` and
    ``Pshift[from] = -b·tau``, ``Pshift[to] = +b·tau``.
    Without these two terms the PTDF flows are simply blind to any phase shifter.

    :param nc: numerical circuit
    :return: direct flow term per branch, equivalent bus injections, in p.u.
    """
    tau_flows: Vec = np.zeros(nc.nbr, dtype=float)
    tau_injections: Vec = np.zeros(nc.nbus, dtype=float)

    for k in range(nc.nbr):
        tau: float = float(nc.active_branch_data.tap_angle[k])
        x: float = float(nc.passive_branch_data.X[k])
        m: float = float(nc.active_branch_data.tap_module[k])
        is_dc: bool = bool(nc.passive_branch_data.dc[k])
        is_active: bool = bool(nc.passive_branch_data.active[k])

        if (not is_active) or is_dc or (tau == 0.0) or (x == 0.0) or (m == 0.0):
            # no contribution: the branch is off, DC, unshifted, or has no usable susceptance
            pass
        else:
            # b = 1/(X*m), add m to the contribution
            contribution: float = -tau / (x * m)
            tau_flows[k] = contribution
            tau_injections[nc.passive_branch_data.F[k]] += contribution
            tau_injections[nc.passive_branch_data.T[k]] -= contribution

    return tau_flows, tau_injections


class LinearAnalysis:
    """
    Linear Analysis
    """

    __slots__ = (
        "logger",
        "islands",
        "PTDF",
        "PTDF_by_island",
        "LODF",
        "HvdcDF",
        "HvdcODF",
        "VscDF",
        "VscODF",
        "bus_types",
        "distributed_slack",
        "Sbase",
        "tau_flows",
        "tau_injections",
        "branch_X",
        "branch_from",
        "branch_to",
        "hvdc_rates_pu",
        "hvdc_droop_pu",
        "hvdc_active_pmode3",
        "hvdc_p3_idx",
        "hvdc_phi",
        "hvdc_Z0",
        "hvdc_K_p3",
        "branch_is_dc",
        "branch_R",
        "branch_active",
        "branch_rates_pu",
        "branch_contingency_rates_pu",
        "bus_is_dc",
        "vsc_from",
        "vsc_to",
        "vsc_active",
        "vsc_raw_rates_pu",
        "vsc_control1_int",
        "vsc_control2_int",
        "vsc_rates_pu",
        "vsc_droop_pu",
        "vsc_active_pmode3",
        "vsc_p3_idx",
        "vsc_droop_df",
        "vsc_meas_df",
        "droop_phi_inj",
        "droop_phi_meas",
        "droop_Z0",
        "droop_K_p3",
    )

    def __init__(self,
                 nc: NumericalCircuit,
                 distributed_slack: bool = False,
                 correct_values: bool = False,
                 converters_as_setpoint: bool = False,
                 logger: Logger = Logger(),
                 use_jacobian_ptdf: bool = False,
                 compute_lodf: bool = True):
        """
        Linear Analysis constructor
        :param nc: numerical circuit instance
        :param distributed_slack: boolean to distribute slack
        :param correct_values: boolean to fix out layer values
        :param converters_as_setpoint: build the AC-DC PTDF with every converter treated
            as a set-point device regardless of its control mode
        :param use_jacobian_ptdf: use the AC Jacobian PTDF for AC islands
        :param compute_lodf: whether to compute the LODF matrix (can be False if only PTDF is needed)
        """

        self.logger: Logger = logger

        self.islands: List[NumericalCircuit] = nc.split_into_islands()
        n_br = nc.nbr
        n_bus = nc.nbus
        n_hvdc = nc.hvdc_data.nelm
        n_vsc = nc.vsc_data.nelm

        self.PTDF: Mat = np.zeros((n_br, n_bus))
        self.PTDF_by_island: List[Mat] = list()

        self.LODF: Mat = np.zeros((n_br, n_br))

        self.HvdcDF: Mat = np.zeros((n_br, n_hvdc))
        self.HvdcODF: Mat = np.zeros((n_br, n_hvdc))

        self.VscDF: Mat = np.zeros((n_br, n_vsc))
        self.VscODF: Mat = np.zeros((n_br, n_vsc))

        self.bus_types = nc.bus_data.bus_types.copy()
        self.distributed_slack = distributed_slack
        self.Sbase = nc.Sbase

        # phase-shift contribution in MW, applied by get_flows / get_flows2d
        tau_flows_pu, tau_injections_pu = compute_phase_shift_terms(nc=nc)
        self.tau_flows = tau_flows_pu * nc.Sbase
        self.tau_injections = tau_injections_pu * nc.Sbase

        # compute the PTDF per islands
        if len(self.islands) > 0:
            for n_island, island in enumerate(self.islands):

                indices = island.get_simulation_indices()

                # no slacks will make it impossible to compute the PTDF analytically
                if len(indices.vd) == 1:
                    if len(indices.no_slack) > 0:

                        if island.bus_data.is_dc.any() or (island.nhvdc > 0 and not converters_as_setpoint):
                            ptdf_island = make_acdc_ptdf(
                                nc=island,
                                logger=self.logger,
                                bus_types=island.bus_data.bus_types,
                                distribute_slack=distributed_slack,
                                converters_as_setpoint=converters_as_setpoint
                            )

                        else:
                            if use_jacobian_ptdf:
                                # Use the nonlinear operating-point derivatives only when explicitly requested.
                                adm = island.get_admittance_matrices()
                                ptdf_island = make_jacobian_ptdf(
                                    Ybus=adm.Ybus,
                                    Yf=adm.Yf,
                                    F=island.passive_branch_data.F,
                                    T=island.passive_branch_data.T,
                                    V=island.bus_data.Vbus,
                                    pq=indices.pq,
                                    pv=indices.pv,
                                    bus_types=island.bus_data.bus_types,
                                    distribute_slack=distributed_slack
                                )
                            else:
                                adml = island.get_linear_admittance_matrices(indices=indices)
                                Bpqpv = adml.get_Bred(pqpv=indices.no_slack)

                                # Keep the established DC PTDF implementation as the default path.
                                ptdf_island = make_ptdf(
                                    Bpqpv=Bpqpv,
                                    Bf=adml.Bf,
                                    no_slack=indices.no_slack,
                                    bus_types=island.bus_data.bus_types,
                                    distribute_slack=distributed_slack
                                )

                        # store maybe for later
                        self.PTDF_by_island.append(ptdf_island)

                        # assign the PTDF to the main PTDF matrix
                        self.PTDF[np.ix_(island.passive_branch_data.original_idx,
                                         island.bus_data.original_idx)] = ptdf_island

                        if compute_lodf:
                            # compute the island LODF
                            lodf_island = make_lodf(
                                Cf=island.passive_branch_data.Cf.tocsc(),
                                Ct=island.passive_branch_data.Ct.tocsc(),
                                PTDF=ptdf_island,
                                correct_values=correct_values
                            )

                            # assign the LODF to the main LODF matrix
                            self.LODF[np.ix_(island.passive_branch_data.original_idx,
                                             island.passive_branch_data.original_idx)] = lodf_island
                    else:
                        self.logger.add_error('No PQ or PV nodes', 'Island {}'.format(n_island))

                elif len(indices.vd) == 0:
                    self.logger.add_warning('No slack bus', 'Island {}'.format(n_island))

                else:
                    self.logger.add_error('More than one slack bus', 'Island {}'.format(n_island))
        else:
            # there are no islands
            pass

        # compute the HVDC PTDF (HVDC lines, Buses)
        # A_hvdc = lil_matrix((n_bus, n_hvdc))
        for k in range(n_hvdc):
            f = nc.hvdc_data.F[k]
            t = nc.hvdc_data.T[k]
            # A_hvdc[f, k] = -1  # subtracts power at the "from" side
            # A_hvdc[t, k] = 1  # injects power at the "to" side
            self.HvdcDF[:, k] = self.PTDF[:, t] - self.PTDF[:, f]
            self.HvdcODF[:, k] = self.PTDF[:, f] - self.PTDF[:, t]

        # self.HvdcDF = self.PTDF @ A_hvdc

        # compute the VSC PTDF (HVDC lines, Buses)
        # A_vsc = lil_matrix((n_bus, n_vsc))
        for k in range(n_vsc):
            f = nc.vsc_data.F[k]
            t = nc.vsc_data.T[k]
            # A_vsc[f, k] = -1  # subtracts power at the "from" side
            # A_vsc[t, k] = 1  # injects power at the "to" side
            self.VscDF[:, k] = self.PTDF[:, t] - self.PTDF[:, f]
            self.VscODF[:, k] = self.PTDF[:, f] - self.PTDF[:, t]

        # self.VscDF = self.PTDF @ A_vsc

        # Cache branch & converter properties for AC-contingency droop and DC-cable pole outages
        self.branch_X = nc.passive_branch_data.X.copy()
        self.branch_from = nc.passive_branch_data.F.copy()
        self.branch_to = nc.passive_branch_data.T.copy()
        is_dc_bus = nc.bus_data.is_dc.astype(bool)
        self.bus_is_dc = is_dc_bus.copy()
        self.branch_is_dc = nc.passive_branch_data.dc.astype(bool).copy()
        self.branch_R = nc.passive_branch_data.R.copy()
        self.branch_active = nc.passive_branch_data.active.astype(bool).copy()
        self.branch_rates_pu = nc.passive_branch_data.rates / nc.Sbase
        self.branch_contingency_rates_pu = nc.passive_branch_data.contingency_rates / nc.Sbase
        self.vsc_from = nc.vsc_data.F.copy()
        self.vsc_to = nc.vsc_data.T.copy()
        self.vsc_active = nc.vsc_data.active.astype(bool).copy()
        self.vsc_raw_rates_pu = nc.vsc_data.rates / nc.Sbase
        self.vsc_control1_int = nc.vsc_data.control1_int.copy()
        self.vsc_control2_int = nc.vsc_data.control2_int.copy()

        # Cache HVDC Pmode3 droop and ratings
        self.hvdc_rates_pu = nc.hvdc_data.rates / nc.Sbase
        self.hvdc_droop_pu = np.zeros(n_hvdc, dtype=float)
        self.hvdc_active_pmode3 = np.zeros(n_hvdc, dtype=bool)
        for h in range(n_hvdc):
            if nc.hvdc_data.active[h] and nc.hvdc_data.control_mode_int[h] == HvdcControlType.type_0_free.idx():
                self.hvdc_active_pmode3[h] = True
                self.hvdc_droop_pu[h] = nc.hvdc_data.get_angle_droop_in_pu_rad_at(h, nc.Sbase)

        # Cache VSC Pmode3 droop, effective DC-corridor ratings, and AC-side distribution factors
        dc_rate_per_bus = np.zeros(n_bus, dtype=float)
        dc_parent = np.arange(n_bus, dtype=int)

        def _dc_find(x: int) -> int:
            while dc_parent[x] != x:
                dc_parent[x] = dc_parent[dc_parent[x]]
                x = dc_parent[x]
            return x

        for k_br in range(n_br):
            if nc.passive_branch_data.active[k_br]:
                f_b = int(nc.passive_branch_data.F[k_br])
                t_b = int(nc.passive_branch_data.T[k_br])
                if is_dc_bus[f_b] and is_dc_bus[t_b]:
                    dc_rate_per_bus[f_b] += nc.passive_branch_data.rates[k_br]
                    dc_rate_per_bus[t_b] += nc.passive_branch_data.rates[k_br]
                    ra, rb = _dc_find(f_b), _dc_find(t_b)
                    if ra != rb:
                        dc_parent[rb] = ra

        dc_comp_slacks: Dict[int, List[int]] = dict()
        dc_comp_vscs: Dict[int, List[int]] = dict()
        for v in range(n_vsc):
            if not nc.vsc_data.active[v]:
                continue
            f_v = int(nc.vsc_data.F[v])
            t_v = int(nc.vsc_data.T[v])
            dc_v = f_v if is_dc_bus[f_v] else (t_v if is_dc_bus[t_v] else -1)
            ac_v = t_v if is_dc_bus[f_v] else (f_v if is_dc_bus[t_v] else -1)
            if dc_v >= 0:
                dc_comp_vscs.setdefault(_dc_find(dc_v), []).append(v)
            c1 = nc.vsc_data.control1_int[v]
            c2 = nc.vsc_data.control2_int[v]
            is_vm_dc = (
                c1 == ConverterControlType.Vm_dc.idx() or c2 == ConverterControlType.Vm_dc.idx()
            ) and c1 != ConverterControlType.Pdc_angle_droop.idx()
            if is_vm_dc and dc_v >= 0 and ac_v >= 0:
                dc_comp_slacks.setdefault(_dc_find(dc_v), []).append(ac_v)

        self.vsc_rates_pu = nc.vsc_data.rates.copy()
        for v in range(n_vsc):
            f_v = int(nc.vsc_data.F[v])
            if is_dc_bus[f_v] and dc_rate_per_bus[f_v] > 0.0:
                self.vsc_rates_pu[v] = min(self.vsc_rates_pu[v], dc_rate_per_bus[f_v])
        for v_list in dc_comp_vscs.values():
            if len(v_list) == 2:
                pair_cap = min(float(self.vsc_rates_pu[v_list[0]]), float(self.vsc_rates_pu[v_list[1]]))
                self.vsc_rates_pu[v_list[0]] = pair_cap
                self.vsc_rates_pu[v_list[1]] = pair_cap
        self.vsc_rates_pu = self.vsc_rates_pu / nc.Sbase

        self.vsc_droop_pu = np.zeros(n_vsc, dtype=float)
        self.vsc_active_pmode3 = np.zeros(n_vsc, dtype=bool)
        self.vsc_droop_df = np.zeros((n_br, n_vsc), dtype=float)
        self.vsc_meas_df = np.zeros((n_br, n_vsc), dtype=float)
        vsc_return_buses: Dict[int, List[int]] = dict()
        vsc_ctrl_buses: Dict[int, int] = dict()

        for v in range(n_vsc):
            if (nc.vsc_data.active[v]
                    and nc.vsc_data.control1_int[v] == ConverterControlType.Pdc_angle_droop.idx()
                    and nc.vsc_data.control2_int[v] == ConverterControlType.Pac.idx()):
                ctrl = int(nc.vsc_data.control1_bus_idx[v])
                f_v = int(nc.vsc_data.F[v])
                t_v = int(nc.vsc_data.T[v])
                if 0 <= ctrl < n_bus and not is_dc_bus[ctrl] and not is_dc_bus[t_v]:
                    slacks = dc_comp_slacks.get(_dc_find(f_v), [])
                    ret = slacks if len(slacks) > 0 else [ctrl]
                    self.vsc_active_pmode3[v] = True
                    self.vsc_droop_pu[v] = float(nc.vsc_data.control1_val[v]) * 57.295779513 / nc.Sbase
                    vsc_return_buses[v] = ret
                    vsc_ctrl_buses[v] = ctrl
                    ac_ret_ptdf = np.mean(self.PTDF[:, ret], axis=1)
                    self.vsc_droop_df[:, v] = self.PTDF[:, t_v] - ac_ret_ptdf - self.PTDF[:, f_v]
                    self.vsc_meas_df[:, v] = self.PTDF[:, t_v] - self.PTDF[:, ctrl]

        # Precompute closed-loop Thevenin impedance seen by active Pmode3 HVDC links and VSC converters
        self.hvdc_p3_idx = np.where(self.hvdc_active_pmode3)[0]
        self.vsc_p3_idx = np.where(self.vsc_active_pmode3)[0]
        self.hvdc_phi = None
        self.hvdc_Z0 = None
        self.hvdc_K_p3 = None
        self.droop_phi_inj = None
        self.droop_phi_meas = None
        self.droop_Z0 = None
        self.droop_K_p3 = None

        n_hp3 = len(self.hvdc_p3_idx)
        n_vp3 = len(self.vsc_p3_idx)
        n_p3 = n_hp3 + n_vp3
        if n_p3 > 0 and np.any(nc.bus_data.bus_types == 3):
            import scipy.sparse.linalg as spla
            from scipy.sparse.csgraph import connected_components

            br_act = nc.passive_branch_data.active
            br_x = nc.passive_branch_data.X
            f_br = nc.passive_branch_data.F
            t_br = nc.passive_branch_data.T
            ac_br_valid = (br_act != 0) & (~is_dc_bus[f_br]) & (~is_dc_bus[t_br]) & (np.abs(br_x) > 1e-12)
            b_val = np.where(ac_br_valid, 1.0 / np.where(ac_br_valid, br_x, 1.0), 0.0)
            Bbus = sp.csc_matrix(
                (np.concatenate([b_val, b_val, -b_val, -b_val]),
                 (np.concatenate([f_br, t_br, f_br, t_br]),
                  np.concatenate([f_br, t_br, t_br, f_br]))),
                shape=(n_bus, n_bus),
            )
            diag_abs = np.abs(Bbus.diagonal())
            ac_active_bus = (~is_dc_bus) & (diag_abs > 1e-12)
            bus_mask = np.zeros(n_bus, dtype=bool)
            if np.any(ac_active_bus):
                _, comp_labels = connected_components(Bbus, directed=False)
                comp_groups: Dict[int, List[int]] = dict()
                for b_i in np.flatnonzero(ac_active_bus):
                    comp_groups.setdefault(int(comp_labels[b_i]), []).append(int(b_i))
                bus_mask[:] = ac_active_bus
                for members in comp_groups.values():
                    vd_members = [b_i for b_i in members if nc.bus_data.bus_types[b_i] == 3]
                    ref_bus = vd_members[0] if vd_members else members[0]
                    bus_mask[ref_bus] = False

            if np.any(bus_mask):
                Bred = Bbus[np.ix_(bus_mask, bus_mask)].tocsc()
                solve_B = spla.factorized(Bred)

                phi_inj = np.zeros((n_bus, n_p3), dtype=float)
                phi_meas = np.zeros((n_bus, n_p3), dtype=float)
                ctrl_rows = np.zeros(n_p3, dtype=int)
                to_rows = np.zeros(n_p3, dtype=int)
                k_all = np.zeros(n_p3, dtype=float)

                for j, h in enumerate(self.hvdc_p3_idx):
                    hf = int(nc.hvdc_data.F[h])
                    ht = int(nc.hvdc_data.T[h])
                    rhs = np.zeros(n_bus, dtype=float)
                    rhs[hf] = 1.0
                    rhs[ht] = -1.0
                    sol = np.zeros(n_bus, dtype=float)
                    sol[bus_mask] = solve_B(rhs[bus_mask])
                    phi_inj[:, j] = sol
                    phi_meas[:, j] = sol
                    ctrl_rows[j] = hf
                    to_rows[j] = ht
                    k_all[j] = self.hvdc_droop_pu[h]

                for j_v, v in enumerate(self.vsc_p3_idx):
                    col = n_hp3 + j_v
                    ctrl = vsc_ctrl_buses[int(v)]
                    vt = int(nc.vsc_data.T[v])
                    ret = vsc_return_buses[int(v)]

                    rhs_inj = np.zeros(n_bus, dtype=float)
                    w_ret = 1.0 / len(ret)
                    for rb in ret:
                        rhs_inj[rb] += w_ret
                    rhs_inj[vt] -= 1.0
                    sol_inj = np.zeros(n_bus, dtype=float)
                    sol_inj[bus_mask] = solve_B(rhs_inj[bus_mask])
                    phi_inj[:, col] = sol_inj

                    if len(ret) == 1 and ret[0] == ctrl:
                        phi_meas[:, col] = sol_inj
                    else:
                        rhs_m = np.zeros(n_bus, dtype=float)
                        rhs_m[ctrl] = 1.0
                        rhs_m[vt] = -1.0
                        sol_m = np.zeros(n_bus, dtype=float)
                        sol_m[bus_mask] = solve_B(rhs_m[bus_mask])
                        phi_meas[:, col] = sol_m

                    ctrl_rows[col] = ctrl
                    to_rows[col] = vt
                    k_all[col] = self.vsc_droop_pu[v]

                self.droop_phi_inj = phi_inj
                self.droop_phi_meas = phi_meas
                self.droop_Z0 = phi_inj[ctrl_rows, :] - phi_inj[to_rows, :]
                self.droop_K_p3 = np.diag(k_all)

                if n_hp3 > 0:
                    self.hvdc_phi = phi_inj[:, :n_hp3]
                    self.hvdc_Z0 = self.droop_Z0[:n_hp3, :n_hp3]
                    self.hvdc_K_p3 = np.diag(self.hvdc_droop_pu[self.hvdc_p3_idx])

    def get_transfer_limits(self, flows: np.ndarray, rates: Vec):
        """
        Compute the maximum transfer limits of each branch in normal operation
        :param flows: base Sf in MW
        :param rates: rates in MW
        :return: Max transfer limits vector (n-branch)
        """
        return make_transfer_limits(
            ptdf=self.PTDF,
            flows=flows,
            rates=rates
        )

    def get_flows(self, Sbus: CxVec | Vec, P_hvdc: Vec | None = None,
                  P_vsc: Vec | None = None) -> CxVec | Vec:
        """
        Compute the time series branch Sf using the PTDF

        Everything here is expected to be in MW 

        :param Sbus: Power Injections time series array (nbus) [MW]
        :param P_hvdc: Power from HvdcLines (nhvdc) [MW] (optional)
        :param P_vsc: Power from Vsc (nvsc) [MW] (optional)
        :return: branch active power Sf (nbranch) [MW]
        """
        if Sbus.ndim == 1:
            # the phase shift acts as a pair of equivalent injections plus a direct term
            Pf = self.PTDF @ (Sbus.real - self.tau_injections) + self.tau_flows

            if P_hvdc is not None:
                Pf += self.HvdcDF @ P_hvdc

            if P_vsc is not None:
                Pf += self.VscDF @ P_vsc

            return Pf
        else:
            raise Exception(f'Sbus has unsupported dimensions: {Sbus.shape}')

    def get_flows2d(self, Sbus: CxMat | Mat, P_hvdc: Mat | None = None,
                    P_vsc: Mat | None = None) -> CxMat | Mat:
        """
        Compute the time series branch Sf using the PTDF, everything in MW, see get_flows
        :param Sbus: Power Injections time series array (time, nbus) for 2D [MW]
        :param P_hvdc: Power from HvdcLines (nhvdc) [MW] (optional)
        :param P_vsc: Power from Vsc (nvsc) [MW] (optional)
        :return: branch active power Sf (time, nbranch) [MW]
        """
        if Sbus.ndim == 2:
            # same phase-shift correction as get_flows, broadcast over the time axis
            Pf = np.dot(self.PTDF, (Sbus.real - self.tau_injections).T).T + self.tau_flows

            if P_hvdc is not None:
                Pf += np.dot(self.HvdcDF, P_hvdc.T).T

            if P_vsc is not None:
                Pf += np.dot(self.VscDF, P_vsc.T).T

            return Pf
        else:
            raise Exception(f'Sbus has unsupported dimensions: {Sbus.shape}')

    def get_injections(self, flows: Vec) -> Vec:
        """
        Get injections that satisfy the flows
        :param flows: Array of flows (nbranch)
        :return: Array of injections (nbus)
        """
        if flows.ndim == 1:

            Pbus, _, _, _ = np.linalg.lstsq(self.PTDF, flows)

            return Pbus
        else:
            raise Exception(f'flows has unsupported dimensions: {flows.shape}')

    def get_reverse_injections_2d(self, flows: Mat) -> Mat:
        """
        Get injections that satisfy the flows
        :param flows: Matrix of flows (time, nbranch)
        :return: Matrix of injections (time, nbus)
        """
        if flows.ndim == 2:

            Pbus_t, _, _, _ = np.linalg.lstsq(self.PTDF, flows.T)

            return Pbus_t.T
        else:
            raise Exception(f'flows has unsupported dimensions: {flows.shape}')

    def get_corrected_injections(self, P: Vec) -> Vec:
        """
        This function corrects the original injections to match the PTDF flows
        :param P: Nodal injections
        :return: Corrected nodal injections
        """
        # # Now we need to recompose the corrected injections
        # P_eff = np.zeros_like(P, dtype=float)
        #
        # # Correction of the injections vector --------------------------------------------------------------------------
        # for island in self.islands:
        #
        #     island_nbus = len(island.bus_data.original_idx)
        #
        #     if island_nbus != 0:
        #         indices = island.get_simulation_indices()
        #         if self.distributed_slack:
        #             # We need to apply the distributed slack factors to the injections
        #
        #             if island_nbus <= 1:
        #                 P_eff[island.bus_data.original_idx] = 0.0
        #             else:
        #                 # VeraGrid current dP convention:
        #                 # P_eff_i = P_i - sum(P_j for j != i) / (m - 1)
        #                 Pi = P[island.bus_data.original_idx]
        #                 P_eff[island.bus_data.original_idx] = Pi - (Pi.sum() - Pi) / (island_nbus - 1)
        #         else:
        #             # We need to compute the slack power injection
        #             if len(indices.vd) == 1:
        #
        #                 slack_local = indices.vd[0]
        #                 slack_global = island.bus_data.original_idx[slack_local]
        #                 Pi = P[island.bus_data.original_idx]
        #
        #                 # Non-slack buses keep their specified injections.
        #                 P_eff[island.bus_data.original_idx] = Pi
        #
        #                 # The slack bus absorbs the island mismatch.
        #                 # Equivalent to: P_eff[slack] = -sum(P_non_slack)
        #                 P_eff[slack_global] -= Pi.sum()
        #             else:
        #                 print(f"Expected exactly one slack bus per island, got {len(indices.vd)}")
        #
        # return P_eff
        return make_corrected_injections(
            P=P,
            bus_types=self.bus_types,
            distribute_slack=self.distributed_slack
        )

    def get_corrected_injections2d(self, P: Mat) -> Vec:
        """
        This function corrects the original injections to match the PTDF flows
        :param P: Nodal injections
        :return: Corrected nodal injections
        """
        return make_corrected_injections_2d(
            P=P,
            bus_types=self.bus_types,
            distribute_slack=self.distributed_slack
        )


class LinearMultiContingency:
    """
    LinearMultiContingency
    """

    __slots__ = (
        "branch_indices",
        "hvdc_indices",
        "vsc_indices",
        "bus_indices",
        "mlodf_factors",
        "compensated_ptdf_factors",
        "injections_factor",
        "hvdc_odf",
        "vsc_odf",
        "compensated_vsc_df",
        "compensated_hvdc_df",
        "hvdc_factors",
        "vsc_factors",
        "hvdc_surviving_rates",
        "vsc_surviving_rates",
        "hvdc_droop_factors",
        "effective_droop_lodf",
        "hvdc_rates",
        "hvdc_df",
        "vsc_droop_factors",
        "vsc_rates",
        "vsc_df",
        "enabled",
    )

    def __init__(self,
                 branch_indices: IntVec,
                 hvdc_indices: IntVec,
                 vsc_indices: IntVec,
                 bus_indices: IntVec,
                 mlodf_factors: sp.csc_matrix,
                 compensated_ptdf_factors: sp.csc_matrix,
                 hvdc_odf: sp.csc_matrix,
                 vsc_odf: sp.csc_matrix,
                 injections_factor: Vec,
                 compensated_vsc_df: sp.csc_matrix | None = None,
                 compensated_hvdc_df: sp.csc_matrix | None = None,
                 hvdc_factors: Vec | None = None,
                 vsc_factors: Vec | None = None,
                 hvdc_surviving_rates: Vec | None = None,
                 vsc_surviving_rates: Vec | None = None,
                 hvdc_droop_factors: Mat | None = None,
                 effective_droop_lodf: sp.csc_matrix | None = None,
                 hvdc_rates: Vec | None = None,
                 hvdc_df: Mat | None = None,
                 vsc_droop_factors: Mat | None = None,
                 vsc_rates: Vec | None = None,
                 vsc_df: Mat | None = None):
        """
        Linear multi contingency object
        :param branch_indices: contingency branch indices.
        :param hvdc_indices: HvdcLine indices that belong into the contingency
        :param vsc_indices: VSC indices that belong into the contingency
        :param bus_indices: contingency bus indices.

        :param mlodf_factors: MLODF factors applicable (all_branches, contingency branches).
                             Should be: MLODF[k, βδ]

        :param compensated_ptdf_factors: compensated PTDF factors applicable (all_branches, contingency buses)
                                         should be: MLODF[k, βδ] x PTDF[βδ, i] + PTDF[k, i]

        :param injections_factor: Injection contingency factors,
                                  i.e percentage to decrease an injection (len(bus indices))

        :param compensated_vsc_df: post-contingency VSC distribution factors (all_branches, n_vsc), used for
                                   corrective N-1, i.e. the change of branch m flow per unit change of VSC v
                                   set-point given the outaged elements of this contingency.
                                   Should be: VscDF[k, v] + MLODF[k, βδ] x VscDF[βδ, v]

        :param compensated_hvdc_df: post-contingency HVDC distribution factors (all_branches, n_hvdc), used for
                                    corrective N-1 (analogous to compensated_vsc_df).
        """

        assert len(bus_indices) == len(injections_factor)

        self.branch_indices: IntVec = branch_indices
        self.hvdc_indices: IntVec = hvdc_indices
        self.vsc_indices: IntVec = vsc_indices
        self.bus_indices: IntVec = bus_indices

        # MLODF[k, βδ]
        self.mlodf_factors: sp.csc_matrix = mlodf_factors

        # MLODF[k, βδ] x PTDF[βδ, i] + PTDF[k, i]
        # if not isinstance(compensated_ptdf_factors, sp.csc_matrix):
        #     print()
        self.compensated_ptdf_factors: sp.csc_matrix = compensated_ptdf_factors

        # percentage to decrease an injection, used to compute ΔP
        self.injections_factor: Vec = injections_factor

        self.hvdc_odf = hvdc_odf
        self.vsc_odf = vsc_odf

        # partial power outage / capacity factors (1.0 = 100% loss)
        self.hvdc_factors: Vec = hvdc_factors if hvdc_factors is not None else np.ones(len(hvdc_indices), dtype=float)
        self.vsc_factors: Vec = vsc_factors if vsc_factors is not None else np.ones(len(vsc_indices), dtype=float)

        # surviving ratings in p.u. after contingency
        self.hvdc_surviving_rates: Vec = hvdc_surviving_rates if hvdc_surviving_rates is not None else np.zeros(len(hvdc_indices), dtype=float)
        self.vsc_surviving_rates: Vec = vsc_surviving_rates if vsc_surviving_rates is not None else np.zeros(len(vsc_indices), dtype=float)

        # post-contingency converter distribution factors for corrective N-1 (may be None if not requested)
        self.compensated_vsc_df: sp.csc_matrix | None = compensated_vsc_df
        self.compensated_hvdc_df: sp.csc_matrix | None = compensated_hvdc_df

        # passive Pmode3 HVDC & VSC droop coupling for AC branch outages
        self.hvdc_droop_factors: Mat | None = hvdc_droop_factors
        self.effective_droop_lodf: sp.csc_matrix | None = effective_droop_lodf
        self.hvdc_rates: Vec | None = hvdc_rates
        self.hvdc_df: Mat | None = hvdc_df
        self.vsc_droop_factors: Mat | None = vsc_droop_factors
        self.vsc_rates: Vec | None = vsc_rates
        self.vsc_df: Mat | None = vsc_df

        # store flag for contingency emptyness
        # self.enabled: bool = (self.mlodf_factors.nnz > 0
        #                       or self.compensated_ptdf_factors.nnz > 0
        #                       or injections_factor.sum() > 1)
        self.enabled: bool = True

    def has_injection_contingencies(self) -> bool:
        """
        Check if this multi-contingency has bus injection modifications
        :return: true / false
        """
        return len(self.bus_indices) > 0

    def get_contingency_flows(self,
                              base_branches_flow: Vec,
                              injections: Vec,
                              hvdc_flow: Vec | None = None,
                              vsc_flow: Vec | None = None) -> Vec:
        """
        Get contingency flows
        :param base_branches_flow: Base branch flows (nbranch)
        :param injections: Bus injections increments (nbus)
        :param hvdc_flow: Base HvdcLine flows (n_hvdc)
        :param vsc_flow: Base Vsc flows (n_vsc)
        :return: New flows (nbranch)
        """

        flow = base_branches_flow.copy()

        if len(self.branch_indices):
            # MLODF[k, βδ] x Pf0[βδ]
            flow += self.mlodf_factors @ base_branches_flow[self.branch_indices]

            # Physical passive droop response of Pmode3 HVDC links and VSCs to AC branch outages
            # projected onto the post-contingency network: cDF = DF + MLODF * DF[br_con, :].
            if self.hvdc_droop_factors is not None and hvdc_flow is not None and self.hvdc_rates is not None and self.hvdc_df is not None:
                dh_droop = self.hvdc_droop_factors @ base_branches_flow[self.branch_indices]
                h_post = np.clip(hvdc_flow + dh_droop, -self.hvdc_rates, self.hvdc_rates)
                delta_h = h_post - hvdc_flow
                flow += self.hvdc_df @ delta_h
                flow += self.mlodf_factors @ (self.hvdc_df[self.branch_indices, :] @ delta_h)

            if self.vsc_droop_factors is not None and vsc_flow is not None and self.vsc_rates is not None and self.vsc_df is not None:
                dv_droop = self.vsc_droop_factors @ base_branches_flow[self.branch_indices]
                v_post = np.clip(vsc_flow + dv_droop, -self.vsc_rates, self.vsc_rates)
                delta_v = v_post - vsc_flow
                flow += self.vsc_df @ delta_v
                flow += self.mlodf_factors @ (self.vsc_df[self.branch_indices, :] @ delta_v)

        if len(self.hvdc_indices) and hvdc_flow is not None:
            h_flows = hvdc_flow[self.hvdc_indices]
            h_loss = np.maximum(0.0, np.abs(h_flows) - self.hvdc_surviving_rates) * np.sign(h_flows)
            flow += self.hvdc_odf @ h_loss

        if len(self.vsc_indices) and vsc_flow is not None:
            v_flows = vsc_flow[self.vsc_indices]
            v_loss = np.maximum(0.0, np.abs(v_flows) - self.vsc_surviving_rates) * np.sign(v_flows)
            flow += self.vsc_odf @ v_loss

        if len(self.bus_indices):
            injection_delta = injections[self.bus_indices]

            # (MLODF[k, βδ] x PTDF[βδ, i] + PTDF[k, i]) x ΔP[i]
            flow += self.compensated_ptdf_factors @ injection_delta

        return flow

    @staticmethod
    def pole_spill(flow_value: Any, r_surv: float) -> Any:
        """
        Spilled power of the surviving pole, respecting the sign of the flow.

        The surviving pole keeps its rating ``±R_surv`` **in the original flow direction**,
        so only the excess spills into the AC grid:

        .. math:: s = \\operatorname{sign}(f)\\,\\max(0, |f| - R_{surv})

        The plain algebraic ``f - R_surv`` is only right while ``f > 0``: with
        ``f = -1622 MW`` and ``R_surv = 1000 MW`` it declared ``-2622 MW`` of spill when the
        correct figure is ``-622 MW``. That over-counting inflates every AC branch fed by the
        pole and is fixed here the same way the N-1 screening already does it.

        :param flow_value: pre-contingency flow of the link (pu). A number, or an LP variable.
        :param r_surv: surviving rating of the pole (pu)
        :return: spill in the same units as ``flow_value``
        """
        if r_surv <= 0.0:
            # the pole is gone: everything spills
            return flow_value
        try:
            f = float(flow_value)
        except (TypeError, ValueError):
            # LP variable (`LpVar` has no __float__): the exact spill cannot be evaluated
            # without the base operating point, so the linearised expression is kept. The
            # caller that knows the base flow must pass the number instead, which then takes
            # the branch below. Do NOT return the bare variable here: dropping the -r_surv
            # offset silently tells the LP that the surviving pole can carry nothing, and
            # the resulting TTC changes by gigawatts (measured: H13 5.298 -> 2.850 MW).
            return flow_value - r_surv
        if abs(f) <= r_surv:
            return 0.0
        return f - (r_surv if f > 0.0 else -r_surv)

    def compute_hvdc_droop_addon(self, base_branch_flow: Vec, hvdc_flow: Vec | None, vsc_flow: Vec | None = None) -> Vec | None:
        """
        Evaluate the physical saturated droop flow change on post-contingency AC branches for Pmode3 HVDC links and VSCs.
        :param base_branch_flow: solved base AC branch flows (pu)
        :param hvdc_flow: solved base HVDC link flows (pu)
        :param vsc_flow: solved base VSC converter flows (pu)
        :return: flow increment on AC branches from saturated droop response (pu)
        """
        if len(self.branch_indices) == 0:
            return None
        total_addon: Vec | None = None
        if (self.hvdc_droop_factors is not None and self.hvdc_rates is not None
                and self.hvdc_df is not None and hvdc_flow is not None):
            dh_droop = self.hvdc_droop_factors @ base_branch_flow[self.branch_indices]
            h_post = np.clip(hvdc_flow + dh_droop, -self.hvdc_rates, self.hvdc_rates)
            delta_h = h_post - hvdc_flow
            total_addon = (self.hvdc_df @ delta_h) + self.mlodf_factors @ (self.hvdc_df[self.branch_indices, :] @ delta_h)
        if (self.vsc_droop_factors is not None and self.vsc_rates is not None
                and self.vsc_df is not None and vsc_flow is not None):
            dv_droop = self.vsc_droop_factors @ base_branch_flow[self.branch_indices]
            v_post = np.clip(vsc_flow + dv_droop, -self.vsc_rates, self.vsc_rates)
            delta_v = v_post - vsc_flow
            vsc_addon = (self.vsc_df @ delta_v) + self.mlodf_factors @ (self.vsc_df[self.branch_indices, :] @ delta_v)
            total_addon = vsc_addon if total_addon is None else (total_addon + vsc_addon)
        return total_addon

    def compute_hvdc_droop_lp_addon(self,
                                    base_branch_flow_lp: ObjVec,
                                    base_branch_flow_num: Vec,
                                    hvdc_flow_lp: ObjVec | None,
                                    hvdc_flow_num: Vec | None,
                                    row_filter: BoolVec | None = None,
                                    vsc_flow_lp: ObjVec | None = None,
                                    vsc_flow_num: Vec | None = None) -> ObjVec | None:
        """
        Compute linear LP expressions for passive Pmode3 HVDC and VSC droop response to AC branch outages.
        """
        if len(self.branch_indices) == 0:
            return None

        def _collect_channel(droop_factors: Mat | None,
                             rates: Vec | None,
                             df_mat: Mat | None,
                             flow_lp: ObjVec | None,
                             flow_num: Vec | None):
            if droop_factors is None or rates is None or df_mat is None or flow_lp is None or flow_num is None:
                return None, []
            d_droop = droop_factors @ base_branch_flow_num[self.branch_indices]
            n_dev = droop_factors.shape[0]
            delta_lp = np.zeros(n_dev, dtype=object)
            active_dev = []
            for d_idx in range(n_dev):
                if not np.any(np.abs(droop_factors[d_idx, :]) > 1e-12):
                    continue
                f_unclipped = float(flow_num[d_idx] + d_droop[d_idx])
                r_d = float(rates[d_idx])
                if f_unclipped > r_d:
                    delta_lp[d_idx] = r_d - flow_lp[d_idx]
                elif f_unclipped < -r_d:
                    delta_lp[d_idx] = -r_d - flow_lp[d_idx]
                else:
                    expr_d = 0.0
                    for k_col, c_br in enumerate(self.branch_indices):
                        coef = float(droop_factors[d_idx, k_col])
                        if abs(coef) > 1e-12:
                            expr_d += coef * base_branch_flow_lp[c_br]
                    delta_lp[d_idx] = expr_d
                active_dev.append(d_idx)
            if not active_dev:
                return None, []
            return delta_lp, active_dev

        delta_h_lp, active_h = _collect_channel(
            self.hvdc_droop_factors, self.hvdc_rates, self.hvdc_df, hvdc_flow_lp, hvdc_flow_num
        )
        delta_v_lp, active_v = _collect_channel(
            self.vsc_droop_factors, self.vsc_rates, self.vsc_df, vsc_flow_lp, vsc_flow_num
        )

        if not active_h and not active_v:
            return None

        n_br = self.hvdc_df.shape[0] if active_h else self.vsc_df.shape[0]
        addon = np.zeros(n_br, dtype=object)
        rows = np.flatnonzero(row_filter) if row_filter is not None else np.arange(n_br)
        if len(rows) == 0:
            return addon
        mlodf_rows = self.mlodf_factors.tocsr()[rows, :]
        c_hvdc_rows = (
            self.hvdc_df[np.ix_(rows, active_h)]
            + mlodf_rows @ self.hvdc_df[np.ix_(self.branch_indices, active_h)]
        ) if active_h else None
        c_vsc_rows = (
            self.vsc_df[np.ix_(rows, active_v)]
            + mlodf_rows @ self.vsc_df[np.ix_(self.branch_indices, active_v)]
        ) if active_v else None
        for pos_r, m in enumerate(rows):
            expr_m = 0.0
            for pos_h, h in enumerate(active_h):
                w = float(c_hvdc_rows[pos_r, pos_h])
                if abs(w) > 1e-12:
                    expr_m += w * delta_h_lp[h]
            for pos_v, v in enumerate(active_v):
                w = float(c_vsc_rows[pos_r, pos_v])
                if abs(w) > 1e-12:
                    expr_m += w * delta_v_lp[v]
            addon[m] = expr_m
        return addon

    def get_lp_contingency_flows(self,
                                 base_flow: ObjVec,
                                 injections: ObjVec,
                                 hvdc_flow: ObjVec | None = None,
                                 vsc_flow: ObjVec | None = None,
                                 row_filter: BoolVec | None = None,
                                 droop_addon: Vec | None = None,
                                 pole_spill: ObjVec | None = None,
                                 vsc_pole_spill: ObjVec | None = None) -> Tuple[ObjVec, BoolVec, IntVec]:
        """
        Get contingency flows using the LP interface equations
        :param base_flow: Base branch flows (nbranch)
        :param injections: Bus injections (nbus)
        :param hvdc_flow: Base HvdcLine flows (n_hvdc)
        :param vsc_flow: Base Vsc flows (n_vsc)
        :param row_filter: per-branch flag, True for the branches the caller is going to constrain.
        :param droop_addon: numeric AC branch flow offset from saturated Pmode3 HVDC droop response
        :param pole_spill: spill of the surviving pole, already built as an LP expression and aligned with
                           ``self.hvdc_indices``. When given it is used instead of evaluating
                           ``pole_spill()`` on the flow, which is impossible with solver variables.
                           The expression must be ``sign(f) * max(0, |f| - R_surv)``: the source can
                           build it exactly with ``add_clipped_flow`` on the surviving pole.
        :param vsc_pole_spill: same as ``pole_spill`` for the VSC converters
        :return: New flows (nbranch), the mask of branches that changed, and their indices
        """
        mask: BoolVec = np.zeros(len(base_flow), dtype=bool)
        flow: ObjVec = base_flow.copy()

        if len(self.branch_indices) > 0:
            inc, block_idx = lpDot1D_changes(self.mlodf_factors, base_flow[self.branch_indices], row_filter)
            if len(block_idx) > 0:
                mask[block_idx] = True
                flow[block_idx] += inc[block_idx]

            # Physical saturated droop addon from Pmode3 HVDC links responding to AC contingency
            if droop_addon is not None and len(droop_addon) == len(flow):
                from VeraGridEngine.Utils.MIP.selected_interface import LpExp
                indices = np.flatnonzero(row_filter) if row_filter is not None else range(len(flow))
                for idx_a in indices:
                    val = droop_addon[idx_a]
                    if isinstance(val, LpExp):
                        mask[idx_a] = True
                        flow[idx_a] += val
                    elif isinstance(val, (int, float, np.floating)) and abs(float(val)) > 1e-6:
                        mask[idx_a] = True
                        flow[idx_a] += float(val)

        if len(self.hvdc_indices) > 0 and hvdc_flow is not None:
            if pole_spill is not None and len(pole_spill) == len(self.hvdc_indices):
                hvdc_loss_list = list(pole_spill)
            else:
                hvdc_loss_list = []
                for k, h_idx in enumerate(self.hvdc_indices):
                    r_surv = float(self.hvdc_surviving_rates[k])
                    hvdc_loss_list.append(self.pole_spill(hvdc_flow[h_idx], r_surv))
            arr_hvdc = np.empty(len(hvdc_loss_list), dtype=object)
            arr_hvdc[:] = hvdc_loss_list
            inc, block_idx = lpDot1D_changes(self.hvdc_odf, arr_hvdc, row_filter)
            if len(block_idx) > 0:
                mask[block_idx] = True
                flow[block_idx] += inc[block_idx]

        if len(self.vsc_indices) > 0 and vsc_flow is not None:
            if vsc_pole_spill is not None and len(vsc_pole_spill) == len(self.vsc_indices):
                vsc_loss_list = list(vsc_pole_spill)
            else:
                vsc_loss_list = []
                for k, v_idx in enumerate(self.vsc_indices):
                    r_surv = float(self.vsc_surviving_rates[k])
                    vsc_loss_list.append(self.pole_spill(vsc_flow[v_idx], r_surv))
            arr_vsc = np.empty(len(vsc_loss_list), dtype=object)
            arr_vsc[:] = vsc_loss_list
            inc, block_idx = lpDot1D_changes(self.vsc_odf, arr_vsc, row_filter)
            if len(block_idx) > 0:
                mask[block_idx] = True
                flow[block_idx] += inc[block_idx]

        if len(self.bus_indices) > 0:
            injection_delta: ObjVec = self.injections_factor * injections[self.bus_indices]
            inc, block_idx = lpDot1D_changes(self.compensated_ptdf_factors, injection_delta, row_filter)
            if len(block_idx) > 0:
                mask[block_idx] = True
                flow[block_idx] += inc[block_idx]

        # Derive the changed indices from the accumulated mask
        changed_idx: IntVec = np.flatnonzero(mask)

        return flow, mask, changed_idx

    def get_alpha_n1(self, dP: Vec, dT: float):
        """
        Compute the N-1 sensitivities to the inter-area exchange
        :param dP: Inter-area power exchanges computed with (compute_dP)
        :param dT: Exchange amount (MW) usually a unitary increment is sufficient (use the value used to compute dP)
        :return: N-1 branch exchange sensitivities
        """

        # (MLODF[k, βδ] x PTDF[βδ, i] + PTDF[k, i]) x ΔP[i]
        dflow = self.compensated_ptdf_factors @ dP[self.bus_indices]

        # dflow_n1 = dflow[m] + lodf[m, c] * dflow[c]

        # MLODF[k, βδ] x Pf0[βδ]
        dflow_n1 = self.mlodf_factors @ dflow[self.branch_indices]

        alpha_n1 = dflow_n1 / dT

        return alpha_n1


class ContingencyIndices:
    """
    Contingency indices
    """

    __slots__ = (
        "branch_contingency_indices",
        "hvdc_contingency_indices",
        "vsc_contingency_indices",
        "bus_contingency_indices",
        "injections_factors",
        "hvdc_factors",
        "vsc_factors",
    )

    def __init__(self,
                 contingency_group: ContingencyGroup,
                 contingency_group_dict: Dict[str, List[Contingency]],
                 branches_dict: Dict[str, int],
                 hvdc_dict: Dict[str, int],
                 vsc_dict: Dict[str, int],
                 injections_bus_index_dict: Dict[str, int]):
        """
        Contingency indices
        :param contingency_group: ContingencyGroup
        :param contingency_group_dict: dictionary to get the list of contingencies matching a contingency group
        :param branches_dict: dictionary to get the branch index by the branch idtag
        :param hvdc_dict: dictionary to get the HvdcLine index by the branch idtag
        :param vsc_dict: dictionary to get the Vsc index by the branch idtag
        :param injections_bus_index_dict: dictionary to get the injection device bus index by the generator idtag
        """

        # get the group's contingencies
        contingencies: List[Contingency] = contingency_group_dict[contingency_group.idtag]

        branch_contingency_indices_list = list()
        hvdc_contingency_indices_list = list()
        vsc_contingency_indices_list = list()
        bus_contingency_indices_list = list()
        injections_factors_list = list()
        hvdc_factors_list = list()
        vsc_factors_list = list()

        # apply the contingencies
        for cnt in contingencies:

            if cnt.prop == ContingencyOperationTypes.Active:

                if cnt.tpe == DeviceType.HVDCLineDevice:
                    hvdc_idx = hvdc_dict.get(cnt.device_idtag, None)
                    if hvdc_idx is not None:
                        hvdc_contingency_indices_list.append(hvdc_idx)
                        hvdc_factors_list.append(1.0)

                elif cnt.tpe == DeviceType.VscDevice:
                    vsc_idx = vsc_dict.get(cnt.device_idtag, None)
                    if vsc_idx is not None:
                        vsc_contingency_indices_list.append(vsc_idx)
                        vsc_factors_list.append(1.0)

                elif cnt.tpe in [DeviceType.GeneratorDevice,
                                 DeviceType.BatteryDevice,
                                 DeviceType.LoadDevice,
                                 DeviceType.StaticGeneratorDevice]:

                    # NOTE: We treat injection active contingencies as full power outages (100% power reduction)

                    bus_idx = injections_bus_index_dict.get(cnt.device_idtag, None)
                    if bus_idx is not None:
                        bus_contingency_indices_list.append(bus_idx)
                        injections_factors_list.append(1.0)  # 100% power contingency
                    else:
                        print(f"contingency generator {cnt.device_idtag} not found")

                else:
                    # search for the contingency in the Branches
                    br_idx = branches_dict.get(cnt.device_idtag, None)
                    if br_idx is not None:
                        branch_contingency_indices_list.append(br_idx)
                    else:
                        print(f"contingency device {cnt.device_type.value}:{cnt.device_idtag} not found")

            elif cnt.prop == ContingencyOperationTypes.PowerPercentage:
                if cnt.tpe == DeviceType.HVDCLineDevice:
                    hvdc_idx = hvdc_dict.get(cnt.device_idtag, None)
                    if hvdc_idx is not None:
                        hvdc_contingency_indices_list.append(hvdc_idx)
                        hvdc_factors_list.append(cnt.value / 100.0)
                elif cnt.tpe == DeviceType.VscDevice:
                    vsc_idx = vsc_dict.get(cnt.device_idtag, None)
                    if vsc_idx is not None:
                        vsc_contingency_indices_list.append(vsc_idx)
                        vsc_factors_list.append(cnt.value / 100.0)
                elif cnt.tpe in [DeviceType.LineDevice, DeviceType.Transformer2WDevice, DeviceType.WindingDevice]:
                    print(f"Warning: PowerPercentage is not supported for AC branch {cnt.device_idtag}. Ignored.")
                else:
                    bus_idx = injections_bus_index_dict.get(cnt.device_idtag, None)
                    if bus_idx is not None:
                        bus_contingency_indices_list.append(bus_idx)
                        injections_factors_list.append(cnt.value / 100.0)
                    else:
                        print(f"contingency generator {cnt.device_idtag} not found")
            else:
                print(f'Unknown branch contingency property {cnt.prop} at {cnt.name} {cnt.idtag}')

        self.branch_contingency_indices: IntVec = np.array(branch_contingency_indices_list, dtype=int)
        self.hvdc_contingency_indices: IntVec = np.array(hvdc_contingency_indices_list, dtype=int)
        self.vsc_contingency_indices: IntVec = np.array(vsc_contingency_indices_list, dtype=int)
        self.bus_contingency_indices: IntVec = np.array(bus_contingency_indices_list, dtype=int)
        self.injections_factors: Vec = np.array(injections_factors_list, dtype=float)
        self.hvdc_factors: Vec = np.array(hvdc_factors_list, dtype=float)
        self.vsc_factors: Vec = np.array(vsc_factors_list, dtype=float)


def _solve_dc_corridor_rating_pu(dc_ref: int,
                                 dc_other: int,
                                 dc_branches: List[int],
                                 lin: LinearAnalysis,
                                 use_contingency_rates: bool) -> float:
    """
    Compute the maximum power transfer (p.u.) between two DC terminals over surviving DC branches.
    Returns 0.0 if disconnected, or np.inf if the two terminals coincide on the same DC bus.
    """
    if dc_ref == dc_other:
        return float(np.inf)

    adj: Dict[int, List[int]] = dict()
    for b in dc_branches:
        fb = int(lin.branch_from[b])
        tb = int(lin.branch_to[b])
        adj.setdefault(fb, []).append(tb)
        adj.setdefault(tb, []).append(fb)

    reachable = {dc_other}
    queue = deque([dc_other])
    while queue:
        cur = queue.popleft()
        for nb_bus in adj.get(cur, ()):
            if nb_bus not in reachable:
                reachable.add(nb_bus)
                queue.append(nb_bus)

    if dc_ref not in reachable:
        return 0.0

    loc = {bus_i: idx for idx, bus_i in enumerate(sorted(reachable))}
    n_loc = len(loc)
    G = np.zeros((n_loc, n_loc), dtype=float)
    branch_terms: List[Tuple[int, int, int, float]] = list()
    for b in dc_branches:
        fb = int(lin.branch_from[b])
        tb = int(lin.branch_to[b])
        if fb not in loc or tb not in loc:
            continue
        i, j = loc[fb], loc[tb]
        g = 1.0 / max(abs(float(lin.branch_R[b])), 1e-6)
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g
        branch_terms.append((b, i, j, g))

    if not branch_terms:
        return 0.0

    ref_i = loc[dc_other]
    keep = [idx for idx in range(n_loc) if idx != ref_i]
    inj = np.zeros(n_loc, dtype=float)
    inj[loc[dc_ref]] = -1.0
    vdc = np.zeros(n_loc, dtype=float)
    if keep:
        G_red = G[np.ix_(keep, keep)]
        try:
            vdc[keep] = np.linalg.solve(G_red, inj[keep])
        except np.linalg.LinAlgError:
            vdc[keep] = np.linalg.lstsq(G_red, inj[keep], rcond=None)[0]

    rates_arr = lin.branch_contingency_rates_pu if use_contingency_rates else lin.branch_rates_pu
    cap = float(np.inf)
    for b, i, j, g in branch_terms:
        flow_b = (vdc[i] - vdc[j]) * g
        if abs(flow_b) > 1e-9:
            cap = min(cap, abs(float(rates_arr[b])) / abs(flow_b))
    return cap


class LinearMultiContingencies:
    """
    LinearMultiContingencies
    """

    __slots__ = (
        "grid",
        "contingency_groups_used",
        "__contingency_group_dict",
        "__branches_dict",
        "__hvdc_dict",
        "__vsc_dict",
        "__injections_bus_index_dict",
        "contingency_indices_list",
        "multi_contingencies",
    )

    def __init__(self, grid: MultiCircuit, contingency_groups_used: List[ContingencyGroup]):
        """
        Constructor
        :param grid: MultiCircuit
        """
        self.grid: MultiCircuit = grid

        self.contingency_groups_used = contingency_groups_used

        # auxiliary structures
        self.__contingency_group_dict = grid.get_contingency_group_dict()
        bus_index_dict = grid.get_bus_index_dict()
        self.__branches_dict = grid.get_branches_index_dict2(add_vsc=False, add_hvdc=False, add_switch=True)
        self.__hvdc_dict = grid.get_hvdc_index_dict()
        self.__vsc_dict = grid.get_vsc_index_dict()
        self.__injections_bus_index_dict = grid.get_injections_bus_index_dict(bus_index_dict=bus_index_dict)

        self.contingency_indices_list = Vector(len(self.contingency_groups_used))

        # for each contingency group
        for ic, contingency_group in enumerate(self.contingency_groups_used):
            self.contingency_indices_list[ic] = ContingencyIndices(
                contingency_group=contingency_group,
                contingency_group_dict=self.__contingency_group_dict,
                branches_dict=self.__branches_dict,
                hvdc_dict=self.__hvdc_dict,
                vsc_dict=self.__vsc_dict,
                injections_bus_index_dict=self.__injections_bus_index_dict
            )

        # list of LinearMultiContingency objects that are used later to compute the contingency flows
        self.multi_contingencies: List[LinearMultiContingency] = list()

    @property
    def contingency_group_dict(self) -> Dict[str, List[Contingency]]:
        """
        get the contingency grooups dictionary
        :return: Dict[str, List[Contingency]]
        """
        return self.__contingency_group_dict

    def get_contingency_group_names(self) -> List[str]:
        """
        Returns a list of the names of the used contingency groups
        :return: List[str]
        """
        return [elm.name for elm in self.contingency_groups_used]

    def compute(self,
                lin: LinearAnalysis,
                ptdf_threshold: float = 0.0001,
                lodf_threshold: float = 0.0001,
                with_corrective_converter_df: bool = False) -> None:
        """
        Make the LODF with any contingency combination using the declared contingency objects
        :param lin: LinearAnalysis instance
        :param ptdf_threshold: threshold to discard values
        :param lodf_threshold: Threshold for LODF conversion to sparse
        :param with_corrective_converter_df: if True, also compute the post-contingency converter distribution
                                             factors (compensated_vsc_df / compensated_hvdc_df) used to allow
                                             corrective (N-1) re-dispatch of VSC/HVDC set-points.
        :return: None
        """

        # lodf: Mat = lin.LODF
        # ptdf: Mat = lin.PTDF
        self.multi_contingencies.clear()

        # Precompute active DC subgrids so DcLine outages can be translated into converter run-back + AC spillover
        has_dc_subgrid_info = (
            getattr(lin, "branch_is_dc", None) is not None
            and getattr(lin, "branch_R", None) is not None
            and getattr(lin, "bus_is_dc", None) is not None
            and getattr(lin, "vsc_active", None) is not None
            and np.any(lin.branch_is_dc)
        )
        dc_br_to_comp: Dict[int, int] = dict()
        dc_comp_records: Dict[int, Tuple[List[int], List[int]]] = dict()
        if has_dc_subgrid_info:
            n_bus_lin = len(lin.bus_is_dc)
            dc_par = np.arange(n_bus_lin, dtype=int)

            def _find_dc_root(x: int) -> int:
                while dc_par[x] != x:
                    dc_par[x] = dc_par[dc_par[x]]
                    x = dc_par[x]
                return x

            act_dc_branches = np.flatnonzero(lin.branch_is_dc & lin.branch_active)
            for b_dc in act_dc_branches:
                fb = int(lin.branch_from[b_dc])
                tb = int(lin.branch_to[b_dc])
                ra, rb = _find_dc_root(fb), _find_dc_root(tb)
                if ra != rb:
                    dc_par[rb] = ra

            comp_dc_brs: Dict[int, List[int]] = dict()
            comp_vscs: Dict[int, List[int]] = dict()
            for b_dc in act_dc_branches:
                b_i = int(b_dc)
                r = _find_dc_root(int(lin.branch_from[b_i]))
                comp_dc_brs.setdefault(r, []).append(b_i)
                dc_br_to_comp[b_i] = r

            for v_i in np.flatnonzero(lin.vsc_active):
                v_int = int(v_i)
                fv = int(lin.vsc_from[v_int])
                tv = int(lin.vsc_to[v_int])
                dc_b = fv if lin.bus_is_dc[fv] else (tv if lin.bus_is_dc[tv] else -1)
                ac_b = tv if lin.bus_is_dc[fv] else (fv if lin.bus_is_dc[tv] else -1)
                if dc_b >= 0 and ac_b >= 0:
                    comp_vscs.setdefault(_find_dc_root(dc_b), []).append(v_int)

            for r, dc_list in comp_dc_brs.items():
                dc_comp_records[r] = (comp_vscs.get(r, []), dc_list)

        def _vsc_dc_ac_sgn(v_i: int) -> Tuple[int, int, float]:
            fv = int(lin.vsc_from[v_i])
            tv = int(lin.vsc_to[v_i])
            if lin.bus_is_dc[fv] and not lin.bus_is_dc[tv]:
                return fv, tv, 1.0
            if lin.bus_is_dc[tv] and not lin.bus_is_dc[fv]:
                return tv, fv, -1.0
            return -1, -1, 1.0

        p3_c1 = ConverterControlType.Pdc_angle_droop.idx()
        vm_dc_c = ConverterControlType.Vm_dc.idx()

        # for each contingency group
        for ic, contingency_group in enumerate(self.contingency_groups_used):

            contingency_indices: ContingencyIndices = self.contingency_indices_list[ic]
            raw_br_idx: IntVec = contingency_indices.branch_contingency_indices

            # Translate outages of DC cables in 2-terminal DC subgrids into surviving corridor run-back + AC spillover
            handled_dc_br: set[int] = set()
            extra_vsc_spills: List[Tuple[int, float, Vec, float]] = list()
            if has_dc_subgrid_info and len(raw_br_idx) > 0:
                dc_by_comp: Dict[int, set[int]] = dict()
                for b_raw in raw_br_idx:
                    b_int = int(b_raw)
                    if lin.branch_is_dc[b_int] and b_int in dc_br_to_comp:
                        dc_by_comp.setdefault(dc_br_to_comp[b_int], set()).add(b_int)

                for comp_r, out_dc_set in dc_by_comp.items():
                    vscs_r, dc_list_r = dc_comp_records.get(comp_r, ([], []))
                    if len(vscs_r) != 2:
                        continue
                    v_ref, v_other = sorted(
                        vscs_r,
                        key=lambda vi: (
                            0 if int(lin.vsc_control1_int[vi]) == p3_c1 else 1,
                            0 if (int(lin.vsc_control1_int[vi]) != vm_dc_c and int(lin.vsc_control2_int[vi]) != vm_dc_c) else 1,
                            vi,
                        ),
                    )
                    dc_ref, ac_ref, sgn_ref = _vsc_dc_ac_sgn(v_ref)
                    dc_other, ac_other, _ = _vsc_dc_ac_sgn(v_other)
                    if dc_ref < 0 or dc_other < 0:
                        continue

                    handled_dc_br.update(out_dc_set)
                    r_vsc_cap = min(float(lin.vsc_raw_rates_pu[v_ref]), float(lin.vsc_raw_rates_pu[v_other]))
                    r_pre_pu = min(
                        r_vsc_cap,
                        _solve_dc_corridor_rating_pu(dc_ref, dc_other, dc_list_r, lin, use_contingency_rates=False),
                    )
                    surv_dc_brs = [b for b in dc_list_r if b not in out_dc_set]
                    r_surv_pu = min(
                        r_vsc_cap,
                        _solve_dc_corridor_rating_pu(dc_ref, dc_other, surv_dc_brs, lin, use_contingency_rates=True),
                    )
                    if r_surv_pu < r_pre_pu - 1e-9:
                        odf_base = sgn_ref * (lin.PTDF[:, ac_other] - lin.PTDF[:, ac_ref])
                        odf_base[lin.branch_is_dc] = 0.0
                        frac_eff = 1.0 - (max(r_surv_pu, 0.0) / r_pre_pu) if r_pre_pu > 1e-9 else 1.0
                        extra_vsc_spills.append((v_ref, float(max(r_surv_pu, 0.0)), odf_base, float(min(1.0, frac_eff))))

            eff_br_idx: IntVec = (
                np.array([int(b) for b in raw_br_idx if int(b) not in handled_dc_br], dtype=int)
                if handled_dc_br
                else raw_br_idx
            )

            if len(eff_br_idx) > 1:

                # Flow =
                #   Pf0[k]
                # + MLODF[k, bd] * Pf0[bd]
                # + MLODF[k, bd] * PTDF[bd, i] * dP[i]
                # + PTDF[k, i] * dPi

                # Compute M matrix [n, n] (lodf relating the outaged lines to each other)
                M = create_M_numba(lodf=lin.LODF,
                                   branch_contingency_indices=eff_br_idx)
                L = lin.LODF[:, eff_br_idx]

                try:
                    # Compute LODF for the multiple failure MLODF[k, βδ]
                    mlodf_factors = dense_to_csc(mat=L @ np.linalg.inv(M),
                                                 threshold=lodf_threshold)

                except np.linalg.LinAlgError:
                    # Done to capture antenna when computing multiples contingencies
                    mlodf_factors = dense_to_csc(mat=L @ np.linalg.pinv(M),
                                                 threshold=lodf_threshold)

                if len(contingency_indices.bus_contingency_indices) > 0:
                    # this is PTDF[k, i]
                    ptdf_k_i = dense_to_csc(mat=lin.PTDF[:, contingency_indices.bus_contingency_indices],
                                            threshold=ptdf_threshold)
                    # PTDF[βδ, i]
                    ptdf_bd_i = dense_to_csc(
                        mat=lin.PTDF[np.ix_(eff_br_idx,
                                            contingency_indices.bus_contingency_indices)],
                        threshold=ptdf_threshold
                    )

                else:
                    ptdf_k_i = sp.csc_matrix((lin.PTDF.shape[0], lin.PTDF.shape[1]))

                    # PTDF[βδ, i]
                    ptdf_bd_i = dense_to_csc(mat=lin.PTDF[eff_br_idx, :],
                                             threshold=ptdf_threshold)

                # must compute: MLODF[k, βδ] x PTDF[βδ, i] + PTDF[k, i]
                compensated_ptdf_factors = mlodf_factors @ ptdf_bd_i + ptdf_k_i

            elif len(eff_br_idx) == 1:

                # Pf0[k]
                # + LODF[k, c] * Pf0[c]
                # + LODF[k, c] * PTDF[c, i] * dPi
                # + PTDF[k, i] * dPi

                # append values
                mlodf_factors = dense_to_csc(mat=lin.LODF[:, eff_br_idx],
                                             threshold=lodf_threshold)

                if len(contingency_indices.bus_contingency_indices) > 0:
                    # single branch and single bus contingency

                    # this is PTDF[k, i]
                    ptdf_k_i = dense_to_csc(mat=lin.PTDF[:, contingency_indices.bus_contingency_indices],
                                            threshold=ptdf_threshold)
                    # PTDF[βδ, i]
                    ptdf_bd_i = dense_to_csc(
                        mat=lin.PTDF[np.ix_(eff_br_idx,
                                            contingency_indices.bus_contingency_indices)],
                        threshold=ptdf_threshold
                    )

                    # must compute: MLODF[k, βδ] x PTDF[βδ, i] + PTDF[k, i]
                    compensated_ptdf_factors = mlodf_factors @ ptdf_bd_i + ptdf_k_i
                else:
                    # single branch contingency, no bus contingency
                    compensated_ptdf_factors = sp.csc_matrix(([], [], [0]), shape=(lin.LODF.shape[0], 0))

            else:
                mlodf_factors = sp.csc_matrix(([], [], [0]), shape=(lin.LODF.shape[0], 0))
                if len(contingency_indices.bus_contingency_indices) > 0:
                    # only bus contingencies
                    compensated_ptdf_factors = dense_to_csc(
                        mat=lin.PTDF[:, contingency_indices.bus_contingency_indices],
                        threshold=ptdf_threshold
                    )
                else:
                    # no bus or branch contingencies
                    compensated_ptdf_factors = sp.csc_matrix(([], [], [0]), shape=(lin.LODF.shape[0], 0))

            # compute the hvdc and vsc contingency distribution factor matrices
            hvdc_odf: sp.csc_matrix = dense_to_csc(
                mat=lin.HvdcODF[:, contingency_indices.hvdc_contingency_indices],
                threshold=lodf_threshold
            )
            vsc_odf: sp.csc_matrix = dense_to_csc(
                mat=lin.VscODF[:, contingency_indices.vsc_contingency_indices],
                threshold=lodf_threshold
            )

            # Post-contingency converter distribution factors for corrective N-1
            # The branch changes by cDF[k, d] = DF[k, d] + MLODF[k, βδ] x DF[βδ, d]
            # where DF is VscDF / HvdcDF (= PTDF[:, to] - PTDF[:, from]).
            compensated_vsc_df: sp.csc_matrix | None = None
            compensated_hvdc_df: sp.csc_matrix | None = None
            if with_corrective_converter_df:
                # indices of the branches tripped by this contingency (empty for pure injection contingencies)
                br_idx: IntVec = eff_br_idx

                # VSC converters: compensate the base factors with the outaged branches when there are any
                if lin.VscDF.shape[1] > 0:
                    if len(br_idx) > 0:
                        vsc_df: Mat = np.asarray(lin.VscDF + (mlodf_factors @ lin.VscDF[br_idx, :]))
                    else:
                        # no branch is tripped (e.g. injection-only contingency): the base factors already apply
                        vsc_df = np.asarray(lin.VscDF)
                    compensated_vsc_df = dense_to_csc(mat=vsc_df, threshold=lodf_threshold)
                else:
                    # the grid has no VSC converters: leave the factor matrix as None
                    pass

                # HVDC links: identical compensation to the VSC converters
                if lin.HvdcDF.shape[1] > 0:
                    if len(br_idx) > 0:
                        hvdc_df: Mat = np.asarray(lin.HvdcDF + (mlodf_factors @ lin.HvdcDF[br_idx, :]))
                    else:
                        # no branch is tripped: the base factors already apply
                        hvdc_df = np.asarray(lin.HvdcDF)
                    compensated_hvdc_df = dense_to_csc(mat=hvdc_df, threshold=lodf_threshold)
                else:
                    # the grid has no HVDC links: leave the factor matrix as None
                    pass
            else:
                # corrective N-1 was not requested: keep the converter factors as None (preventive behaviour)
                pass

            # compute surviving ratings after contingency
            hvdc_surviving_rates = []
            for h_idx, frac in zip(contingency_indices.hvdc_contingency_indices, contingency_indices.hvdc_factors):
                rate_mw = float(self.grid.hvdc_lines[h_idx].rate)
                rate_pu = rate_mw / lin.Sbase
                hvdc_surviving_rates.append(rate_pu * (1.0 - frac))

            eff_vsc_indices = list(int(x) for x in contingency_indices.vsc_contingency_indices)
            eff_vsc_factors = list(float(x) for x in contingency_indices.vsc_factors)
            vsc_surviving_rates = []
            for v_idx, frac in zip(eff_vsc_indices, eff_vsc_factors):
                rate_mw = float(self.grid.vsc_devices[v_idx].rate)
                rate_pu = rate_mw / lin.Sbase
                vsc_surviving_rates.append(rate_pu * (1.0 - frac))

            if extra_vsc_spills:
                extra_odf_cols = []
                for v_ref, r_surv_pu, odf_base, frac_eff in extra_vsc_spills:
                    if v_ref in eff_vsc_indices:
                        pos_v = eff_vsc_indices.index(v_ref)
                        if r_surv_pu < vsc_surviving_rates[pos_v]:
                            vsc_surviving_rates[pos_v] = r_surv_pu
                            eff_vsc_factors[pos_v] = max(eff_vsc_factors[pos_v], frac_eff)
                        continue
                    odf_col = (
                        odf_base + (mlodf_factors @ odf_base[eff_br_idx])
                        if len(eff_br_idx) > 0
                        else odf_base
                    )
                    if getattr(lin, "branch_is_dc", None) is not None:
                        odf_col[lin.branch_is_dc] = 0.0
                    eff_vsc_indices.append(v_ref)
                    eff_vsc_factors.append(frac_eff)
                    vsc_surviving_rates.append(r_surv_pu)
                    extra_odf_cols.append(odf_col[:, None])
                if extra_odf_cols:
                    extra_odf_csc = dense_to_csc(mat=np.hstack(extra_odf_cols), threshold=lodf_threshold)
                    vsc_odf = sp.hstack([vsc_odf, extra_odf_csc], format="csc")

            # Calculate passive Pmode3 HVDC & VSC droop coupling for AC branch outages
            hvdc_droop_factors: Mat | None = None
            vsc_droop_factors: Mat | None = None
            effective_droop_lodf: sp.csc_matrix | None = None
            br_con_idx: IntVec = eff_br_idx
            has_h_p3 = hasattr(lin, "hvdc_active_pmode3") and np.any(lin.hvdc_active_pmode3)
            has_v_p3 = hasattr(lin, "vsc_active_pmode3") and np.any(lin.vsc_active_pmode3)
            if len(br_con_idx) > 0 and (has_h_p3 or has_v_p3):
                n_h: int = len(lin.hvdc_active_pmode3)
                n_v: int = len(lin.vsc_active_pmode3) if hasattr(lin, "vsc_active_pmode3") else 0
                n_c: int = len(br_con_idx)
                droop_mat_h: Mat = np.zeros((n_h, n_c), dtype=float)
                droop_mat_v: Mat = np.zeros((n_v, n_c), dtype=float)
                is_dc_br = getattr(lin, "branch_is_dc", None)
                ac_pos = np.flatnonzero(~is_dc_br[br_con_idx]) if is_dc_br is not None else np.arange(n_c)

                hp3_idx = getattr(lin, "hvdc_p3_idx", np.where(lin.hvdc_active_pmode3)[0])
                vp3_idx = getattr(lin, "vsc_p3_idx", np.where(lin.vsc_active_pmode3)[0]) if has_v_p3 else np.zeros(0, dtype=int)
                n_hp3 = len(hp3_idx)
                n_vp3 = len(vp3_idx)

                if len(ac_pos) == 1:
                    col_pos = int(ac_pos[0])
                    c_single: int = int(br_con_idx[col_pos])
                    x_c: float = float(lin.branch_X[c_single])
                    f_node: int = int(lin.branch_from[c_single])
                    t_node: int = int(lin.branch_to[c_single])
                    denom: float = 1.0 - float(lin.PTDF[c_single, f_node] - lin.PTDF[c_single, t_node])
                    if abs(denom) < 1e-10:
                        denom = 1.0

                    g_open_h = np.array([
                        - lin.hvdc_droop_pu[h_idx] * x_c * lin.HvdcDF[c_single, h_idx] / denom
                        for h_idx in hp3_idx
                    ], dtype=float)
                    g_open_v = np.array([
                        - lin.vsc_droop_pu[v_idx] * x_c * lin.vsc_meas_df[c_single, v_idx] / denom
                        for v_idx in vp3_idx
                    ], dtype=float)
                    g_open_all = np.concatenate([g_open_h, g_open_v])

                    if getattr(lin, "droop_Z0", None) is not None and getattr(lin, "droop_phi_inj", None) is not None:
                        phi_inj_c = lin.droop_phi_inj[f_node, :] - lin.droop_phi_inj[t_node, :]
                        phi_meas_src = getattr(lin, "droop_phi_meas", lin.droop_phi_inj)
                        phi_meas_c = phi_meas_src[f_node, :] - phi_meas_src[t_node, :]
                        if abs(x_c) > 1e-12:
                            Zc = lin.droop_Z0 + np.outer(phi_meas_c, phi_inj_c) / (x_c * denom)
                        else:
                            Zc = lin.droop_Z0.copy()
                        g_closed_all = np.linalg.solve(np.eye(n_hp3 + n_vp3) + lin.droop_K_p3 @ Zc, g_open_all)
                        if n_hp3 > 0:
                            droop_mat_h[hp3_idx, col_pos] = g_closed_all[:n_hp3]
                        if n_vp3 > 0:
                            droop_mat_v[vp3_idx, col_pos] = g_closed_all[n_hp3:]
                    else:
                        if n_hp3 > 0:
                            droop_mat_h[hp3_idx, col_pos] = g_open_h
                        if n_vp3 > 0:
                            droop_mat_v[vp3_idx, col_pos] = g_open_v

                elif len(ac_pos) > 1:
                    ac_br_idx = br_con_idx[ac_pos]
                    X_diag: Vec = lin.branch_X[ac_br_idx]
                    f_nodes: IntVec = lin.branch_from[ac_br_idx]
                    t_nodes: IntVec = lin.branch_to[ac_br_idx]
                    H_block: Mat = lin.PTDF[np.ix_(ac_br_idx, t_nodes)] - lin.PTDF[np.ix_(ac_br_idx, f_nodes)]
                    M_block: Mat = np.eye(len(ac_pos)) + H_block
                    try:
                        M_inv: Mat = np.linalg.inv(M_block)
                    except np.linalg.LinAlgError:
                        M_inv = np.linalg.pinv(M_block)
                    for h_idx in hp3_idx:
                        droop_mat_h[h_idx, ac_pos] = - lin.hvdc_droop_pu[h_idx] * (lin.HvdcDF[ac_br_idx, h_idx] * X_diag) @ M_inv
                    for v_idx in vp3_idx:
                        droop_mat_v[v_idx, ac_pos] = - lin.vsc_droop_pu[v_idx] * (lin.vsc_meas_df[ac_br_idx, v_idx] * X_diag) @ M_inv

                eff_mat = np.zeros((lin.PTDF.shape[0], n_c), dtype=float)
                if has_h_p3:
                    hvdc_droop_factors = droop_mat_h
                    eff_mat = eff_mat + (lin.HvdcDF @ droop_mat_h) + (mlodf_factors @ (lin.HvdcDF[br_con_idx, :] @ droop_mat_h))
                if has_v_p3:
                    vsc_droop_factors = droop_mat_v
                    eff_mat = eff_mat + (lin.vsc_droop_df @ droop_mat_v) + (mlodf_factors @ (lin.vsc_droop_df[br_con_idx, :] @ droop_mat_v))
                effective_droop_lodf = dense_to_csc(eff_mat, threshold=lodf_threshold)

            # append values
            self.multi_contingencies.append(
                LinearMultiContingency(
                    branch_indices=eff_br_idx,
                    hvdc_indices=contingency_indices.hvdc_contingency_indices,
                    vsc_indices=np.array(eff_vsc_indices, dtype=int),
                    bus_indices=contingency_indices.bus_contingency_indices,
                    mlodf_factors=mlodf_factors,
                    compensated_ptdf_factors=compensated_ptdf_factors,
                    injections_factor=contingency_indices.injections_factors,
                    hvdc_odf=hvdc_odf,
                    vsc_odf=vsc_odf,
                    compensated_vsc_df=compensated_vsc_df,
                    compensated_hvdc_df=compensated_hvdc_df,
                    hvdc_factors=contingency_indices.hvdc_factors,
                    vsc_factors=np.array(eff_vsc_factors, dtype=float),
                    hvdc_surviving_rates=np.array(hvdc_surviving_rates, dtype=float),
                    vsc_surviving_rates=np.array(vsc_surviving_rates, dtype=float),
                    hvdc_droop_factors=hvdc_droop_factors,
                    effective_droop_lodf=effective_droop_lodf,
                    hvdc_rates=lin.hvdc_rates_pu if hasattr(lin, "hvdc_rates_pu") else None,
                    hvdc_df=lin.HvdcDF,
                    vsc_droop_factors=vsc_droop_factors,
                    vsc_rates=getattr(lin, "vsc_rates_pu", None),
                    vsc_df=getattr(lin, "vsc_droop_df", None),
                )
            )

    def get_single_con_branch_idx(self) -> Tuple[IntVec, IntVec]:
        """
        Get the branch index array and the contingency group it belongs array
        :return: array of single contingency branch indices,
                 array of the matching contingency groups
        """
        con_idx = list()
        cg_idx = list()
        for i, ci in enumerate(self.contingency_indices_list):
            if len(ci.branch_contingency_indices) == 1:
                con_idx.append(ci.branch_contingency_indices[0])
                cg_idx.append(i)

        return np.array(con_idx), np.array(cg_idx)


class LinearAnalysisTs:
    """
    Class to compute the different linear states of a grid
    """

    __slots__ = (
        "time_indices",
        "_inverse_time_index",
        "groups",
        "mapping",
        "_linear_analysis",
        "_linear_multi_contingencies",
        "nbr",
        "nbus",
        "nt",
    )

    def __init__(self, grid: MultiCircuit,
                 distributed_slack: bool = False,
                 correct_values: bool = False,
                 time_indices: IntVec | None = None,
                 contingency_groups_used: List[ContingencyGroup] | None = None,
                 ptdf_threshold: float = 1e-4,
                 lodf_threshold: float = 1e-4,
                 compute_multi_contingencies: bool = True,
                 use_jacobian_ptdf: bool = False):
        """
        Constructor
        :param grid: MultiCircuit instance
        :param distributed_slack: boolean to distribute slack
        :param correct_values: boolean to fix out layer values
        :param time_indices: Array of time indices
        :param contingency_groups_used: list of contingency groups to use (all the grid's if None)
        :param ptdf_threshold: threshold for PTDF's to be converted to sparse
        :param lodf_threshold: threshold for LODF's to be converted to sparse
        :param compute_multi_contingencies: also pre-compute the LinearMultiContingencies
        :param use_jacobian_ptdf: use the AC Jacobian PTDF implementation?
        """

        if not grid.has_time_series:
            raise Exception("The grid does not have any time series :(")

        self.time_indices = grid.get_all_time_indices() if time_indices is None else time_indices

        self._inverse_time_index = {t: i for i, t in enumerate(self.time_indices)}

        # get the states matrix
        mat: IntMat = grid.get_branch_active_time_array()[self.time_indices, :]

        # analyze how many PTDF's we need to get
        self.groups, self.mapping = find_different_states(mat)

        self._linear_analysis: Dict[int, LinearAnalysis] = dict()
        self._linear_multi_contingencies: Dict[int, LinearMultiContingencies] = dict()

        # compile the linear analysis for the different time steps
        for t_idx, list_of_represented_time_steps in self.groups.items():
            # Compile the snapshot
            nc = compile_numerical_circuit_at(circuit=grid, t_idx=t_idx)

            # Linear analysis (PTDF & LODF)
            lin = LinearAnalysis(nc=nc,
                                 distributed_slack=distributed_slack,
                                 correct_values=correct_values,
                                 use_jacobian_ptdf=use_jacobian_ptdf)

            self._linear_analysis[t_idx] = lin

            if compute_multi_contingencies:
                linear_multiple_contingencies = LinearMultiContingencies(
                    grid=grid,
                    contingency_groups_used=contingency_groups_used
                    if contingency_groups_used is not None
                    else grid.contingency_groups
                )
                linear_multiple_contingencies.compute(
                    lin=lin,
                    ptdf_threshold=ptdf_threshold,
                    lodf_threshold=lodf_threshold
                )

                self._linear_multi_contingencies[t_idx] = linear_multiple_contingencies

        self.nbr = grid.get_branch_number()
        self.nbus = grid.get_bus_number()
        self.nt = len(self.time_indices)

    def get_linear_analysis_at(self, t_idx: int) -> LinearAnalysis:
        """

        :param t_idx: time index in the general schema
        :return:
        """
        # get the base index (i.e. when clustering)
        it = self._inverse_time_index[t_idx]

        # get the mapped reduced index
        t_i = self.mapping[it]

        # get the linear analysis
        lin: LinearAnalysis = self._linear_analysis[t_i]

        return lin

    def get_multiple_contingencies_at(self, t_idx: int) -> LinearMultiContingencies:
        """

        :param t_idx:
        :return:
        """
        # get the base index (i.e. when clustering)
        it = self._inverse_time_index[t_idx]

        # get the mapped reduced index
        t_i = self.mapping[it]

        # get the linear analysis
        lin: LinearMultiContingencies = self._linear_multi_contingencies[t_i]

        return lin

    def get_flows_at(self, t_idx: int, P: CxVec | Vec) -> CxVec | Vec:
        """
        Get the flows at a time step
        :param t_idx: Time index
        :param P: Injections vector
        :return: branch flows vector
        """
        # get the linear analysis
        lin = self.get_linear_analysis_at(t_idx=t_idx)

        return lin.get_flows(Sbus=P)

    def get_branch_flow_ts(self, branch_idx: int, bus_idx: int, P: CxVec | Vec) -> CxVec | Vec:
        """
        Get the flow time series of a single branch given the injection time series of a single bus
        :param branch_idx: Branch index
        :param bus_idx: Bus index
        :param P: Bus injection time series
        :return: Branch flow time series
        """
        # must have the same size
        assert len(P) == self.nt

        flow_ts: Vec = np.zeros_like(P)

        for t_idx, list_of_represented_time_steps in self.groups.items():
            # get the linear analysis
            lin: LinearAnalysis = self._linear_analysis[t_idx]

            flow_ts[list_of_represented_time_steps] = lin.PTDF[branch_idx, bus_idx] * P[list_of_represented_time_steps]

        return flow_ts

    def get_flows_ts(self, P: CxMat | Mat, progress_func=None, progress_text=None) -> CxMat | Mat:
        """
        Get the flow time series of all branches given the injection time series all buses
        :param P: Bus injection time series
        :param progress_func: Progres function
        :param progress_text: Progress text function
        :return: Branches flow time series
        """
        # must have the same size
        assert P.shape[0] == self.nt
        assert P.shape[1] == self.nbus

        flow_ts = np.zeros((self.nt, self.nbr))
        ii = 0
        nn = len(self.groups)
        for t_idx, list_of_represented_time_steps in self.groups.items():

            if progress_text is not None:
                progress_text(f"Computing flows for time group {t_idx}")

            # get the linear analysis
            lin: LinearAnalysis = self._linear_analysis[t_idx]

            flow_ts[list_of_represented_time_steps, :] = lin.get_flows2d(Sbus=P[list_of_represented_time_steps, :])

            if progress_func is not None:
                ii += 1
                progress_func(ii / nn * 100.0)

        return flow_ts

    def get_reverse_injections_ts(self, flows_ts: CxMat | Mat) -> CxMat | Mat:
        """
        Get the flow time series of all branches given the injection time series all buses
        :param flows_ts: Branches flow time series
        :return: Bus injection time series
        """
        # must have the same size
        assert flows_ts.shape[0] == self.nt
        assert flows_ts.shape[1] == self.nbr

        Pbus = np.zeros((self.nt, self.nbus))

        for t_idx, list_of_represented_time_steps in self.groups.items():
            # get the linear analysis
            lin: LinearAnalysis = self._linear_analysis[t_idx]

            Pbus[list_of_represented_time_steps, :] = lin.get_reverse_injections_2d(
                flows=flows_ts[list_of_represented_time_steps, :]
            )

        return Pbus

    def get_corrected_injections_ts(self, P_ts: CxMat | Mat) -> CxMat | Mat:
        """
        Get the corrected injections time series
        :param P_ts: Time series injections
        :return: Bus injection time series
        """
        # must have the same size
        assert P_ts.shape[0] == self.nt
        assert P_ts.shape[1] == self.nbus

        Pbus = np.zeros((self.nt, self.nbus))

        for t_idx, list_of_represented_time_steps in self.groups.items():
            # get the linear analysis
            lin: LinearAnalysis = self._linear_analysis[t_idx]

            Pbus[list_of_represented_time_steps, :] = lin.get_corrected_injections2d(
                P=P_ts[list_of_represented_time_steps, :]
            )

        return Pbus

    def get_hvdc_flows_ts(self, Pdc_hvdc_ts: Mat) -> Mat:
        """
        Get the AC branch flow time series due to the HVDC transfers
        :param Pdc_hvdc_ts: (nt, nhvdc) HVDC from->to transfer time series (see get_hvdc_Pdc_ts), in MW
        :return: (nt, nbr) branch flows due to the HVDC lines, in MW
        """
        assert Pdc_hvdc_ts.shape[0] == self.nt

        flows = np.zeros((self.nt, self.nbr))

        for t_idx, time_steps in self.groups.items():
            lin: LinearAnalysis = self._linear_analysis[t_idx]
            flows[time_steps, :] = (lin.HvdcDF @ Pdc_hvdc_ts[time_steps, :].T).T

        return flows


def get_hvdc_Pdc_ts(grid: MultiCircuit) -> Mat:
    """
    Get the HVDC scheduled from->to transfer time series in MW
    (this equals -Pf of HvdcData.get_power evaluated at theta=0)
    :param grid: MultiCircuit
    :return: (nt, nhvdc) transfer of each HvdcLine, positive from->to, in MW
    """
    nt = grid.get_time_number()
    nhvdc = len(grid.hvdc_lines)
    Pdc = np.zeros((nt, nhvdc), dtype=float)

    for i, elm in enumerate(grid.hvdc_lines):
        active = elm.active_prof.toarray().astype(bool)

        if elm.control_mode == HvdcControlType.type_1_Pset:
            Pcalc = elm.Pset_prof.toarray() * active
        elif elm.control_mode == HvdcControlType.type_0_free:
            # at theta=0 the angle droop term is zero; clip to the rating
            rates = elm.rate_prof.toarray()
            Pcalc = np.clip(elm.Pset_prof.toarray(), -rates, rates) * active
        else:
            Pcalc = np.zeros(nt)

        # -Pf as in HvdcData.get_power: Pcalc (+ losses on the sending side when Pcalc < 0)
        Vnt = elm.bus_to.Vnom
        Vset_t = elm.Vset_t_prof.toarray()

        pos = Pcalc > 0.0
        neg = Pcalc < 0.0

        Pdc[pos, i] = Pcalc[pos]

        I_neg = Pcalc[neg] / (Vnt * Vset_t[neg])  # kA
        losses_neg = elm.r * I_neg * I_neg  # MW
        Pdc[neg, i] = Pcalc[neg] + losses_neg

    return Pdc
