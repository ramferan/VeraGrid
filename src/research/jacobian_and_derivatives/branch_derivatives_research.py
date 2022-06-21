import numpy as np
import numba as nb
import scipy.sparse as sp
from scipy.sparse import lil_matrix, diags
import GridCal.Engine as gc

# ----------------------------------------------------------------------------------------------------------------------
#  base (less efficient) derivatives
# ----------------------------------------------------------------------------------------------------------------------


def dSbus_dV(Ybus, V):
    """
    Derivatives of the power injections w.r.t the voltage
    :param Ybus: Admittance matrix
    :param V: complex voltage arrays
    :return: dSbus_dVa, dSbus_dVm
    """
    diagV = diags(V)
    diagE = diags(V / np.abs(V))
    Ibus = Ybus * V
    diagIbus = diags(Ibus)

    dSbus_dVa = 1j * diagV * np.conj(diagIbus - Ybus * diagV)  # dSbus / dVa
    dSbus_dVm = diagV * np.conj(Ybus * diagE) + np.conj(diagIbus) * diagE  # dSbus / dVm

    return dSbus_dVa, dSbus_dVm


def dSbr_dV(Yf, Yt, V, F, T, Cf, Ct):
    """
    Derivatives of the branch power w.r.t the branch voltage modules and angles
    :param Yf: Admittances matrix of the branches with the "from" buses
    :param Yt: Admittances matrix of the branches with the "to" buses
    :param V: Array of voltages
    :param F: Array of branch "from" bus indices
    :param T: Array of branch "to" bus indices
    :param Cf: Connectivity matrix of the branches with the "from" buses
    :param Ct: Connectivity matrix of the branches with the "to" buses
    :return: dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm
    """
    Yfc = np.conj(Yf)
    Ytc = np.conj(Yt)
    Vc = np.conj(V)
    Ifc = Yfc * Vc  # conjugate  of "from"  current
    Itc = Ytc * Vc  # conjugate of "to" current

    diagIfc = diags(Ifc)
    diagItc = diags(Itc)
    Vf = V[F]
    Vt = V[T]
    diagVf = diags(Vf)
    diagVt = diags(Vt)
    diagVc = diags(Vc)

    Vnorm = V / np.abs(V)
    diagVnorm = diags(Vnorm)
    diagV = diags(V)

    CVf = Cf * diagV
    CVt = Ct * diagV
    CVnf = Cf * diagVnorm
    CVnt = Ct * diagVnorm

    dSf_dVa = 1j * (diagIfc * CVf - diagVf * Yfc * diagVc)
    dSf_dVm = diagVf * np.conj(Yf * diagVnorm) + diagIfc * CVnf
    dSt_dVa = 1j * (diagItc * CVt - diagVt * Ytc * diagVc)
    dSt_dVm = diagVt * np.conj(Yt * diagVnorm) + diagItc * CVnt

    return dSf_dVa.tocsc(), dSf_dVm.tocsc(), dSt_dVa.tocsc(), dSt_dVm.tocsc()


def calc_dPbr_dVa(dP_dVa, F, T, n, m):

    val = np.zeros((m, n))
    k = 0
    for i, j in zip(F, T):
        val[k, i] = -dP_dVa[i, j]
        val[k, j] = dP_dVa[i, j]
        k += 1
    return val


def calc_Sbr(Yf, Yt, Cf, Ct, V):
    # Branches current, loading, etc
    Vf = Cf * V
    Vt = Ct * V
    If = Yf * V
    It = Yt * V
    Sf = Vf * np.conj(If)
    St = Vt * np.conj(It)
    return Sf, St



# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def calc_dPbr_dVm(dP_dVm, Vm, Pbr, F, T, n, m):
    val = np.zeros((m, n))
    k = 0
    for i, j in zip(F, T):
        val[k, i] = (2 * Pbr[k] - dP_dVm[i, j]) / Vm[i]
        val[k, j] = dP_dVm[i, j] / Vm[j]
        k += 1
    return val


def branch_derivatives(G, B, Vm, Va, b_shunt_br, tap_m, tap_angle, F, T, m, n):
    """
    Branch derivatives according to gomez exposito book page (181)
    :param G:
    :param B:
    :param Vm:
    :param Va:
    :param b_shunt_br:
    :param F:
    :param T:
    :param m:
    :param n:
    :return:
    """

    # compute the angle difference sine and cosine
    Va_ij = Va[F] - Va[T] + tap_angle
    sinVa = np.sin(Va_ij)
    cosVa = np.cos(Va_ij)

    dPij_dVm = np.zeros((m, n))
    dQij_dVm = np.zeros((m, n))
    dPij_dVa = np.zeros((m, n))
    dQij_dVa = np.zeros((m, n))

    k = 0
    for i, j in zip(F, T):
        # dPij_dVm (ok)
        dPij_dVm[k, i] = Vm[j] * (G[i, j] * cosVa[k] + B[i, j] * sinVa[k]) - 2 * G[i, j] * Vm[i]
        dPij_dVm[k, j] = Vm[i] * (G[i, j] * cosVa[k] + B[i, j] * sinVa[k])

        # dQij_dVm (still failing)
        dQij_dVm[k, i] = Vm[j] * (G[i, j] * sinVa[k] - B[i, j] * cosVa[k]) + 2 * Vm[i] * (B[i, j] - b_shunt_br[k] * tap_m[k] * tap_m[k])
        dQij_dVm[k, j] = Vm[i] * (G[i, j] * sinVa[k] - B[i, j] * cosVa[k])

        # dPij_dVa (ok)
        dPij_dVa[k, i] = Vm[i] * Vm[j] * (- G[i, j] * sinVa[k] + B[i, j] * cosVa[k])
        dPij_dVa[k, j] = Vm[i] * Vm[j] * (+ G[i, j] * sinVa[k] - B[i, j] * cosVa[k])

        # dQij_dVa (ok)
        dQij_dVa[k, i] = + Vm[i] * Vm[j] * (G[i, j] * cosVa[k] + B[i, j] * sinVa[k])
        dQij_dVa[k, j] = - Vm[i] * Vm[j] * (G[i, j] * cosVa[k] + B[i, j] * sinVa[k])

        k += 1

    return dPij_dVa, dQij_dVa, dPij_dVm, dQij_dVm


def test(nc):

    V = nc.Vbus
    dSbus_dVa_, dSbus_dVm_ = dSbus_dV(Ybus=nc.Ybus, V=V)
    dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm = dSbr_dV(Yf=nc.Yf, Yt=nc.Yt, V=V, F=nc.F, T=nc.T, Cf=nc.Cf, Ct=nc.Ct)
    Sf, St = calc_Sbr(Yf=nc.Yf, Yt=nc.Yt, Cf=nc.Cf, Ct=nc.Ct, V=V)

    dP_dVa, dQ_dVa = dSbus_dVa_.real, dSbus_dVa_.imag
    dP_dVm, dQ_dVm = dSbus_dVm_.real, dSbus_dVm_.imag

    dPf_dVa, dQf_dVa = dSf_dVa.real, dSf_dVa.imag
    dPf_dVm, dQf_dVm = dSf_dVm.real, dSf_dVm.imag

    # print('dPf_dVa (original)\n', dPf_dVa.toarray())
    dPf_dVa2 = calc_dPbr_dVa(dP_dVa=dP_dVa, F=nc.F, T=nc.T, n=nc.nbus, m=nc.nbr)
    # print('dPf_dVa (new)\n', dPf_dVa2)
    print('dPf_dVa ok:', np.allclose(dPf_dVa.toarray(), dPf_dVa2))

    print('dPf_dVm2 (original)\n', dPf_dVm.toarray())
    dPf_dVm2 = calc_dPbr_dVm(dP_dVm=dP_dVm, Vm=np.abs(V), Pbr=Sf.real, F=nc.F, T=nc.T, n=nc.nbus, m=nc.nbr)
    print('dPf_dVm2 (new)\n', dPf_dVm2)
    print('dPf_dVm ok:', np.allclose(dPf_dVm.toarray(), dPf_dVm2))


def test2(nc):

    V = nc.Vbus
    dSbus_dVa_, dSbus_dVm_ = dSbus_dV(Ybus=nc.Ybus, V=V)
    dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm = dSbr_dV(Yf=nc.Yf, Yt=nc.Yt, V=V, F=nc.F, T=nc.T, Cf=nc.Cf, Ct=nc.Ct)
    Sf, St = calc_Sbr(Yf=nc.Yf, Yt=nc.Yt, Cf=nc.Cf, Ct=nc.Ct, V=V)

    dP_dVa, dQ_dVa = dSbus_dVa_.real, dSbus_dVa_.imag
    dP_dVm, dQ_dVm = dSbus_dVm_.real, dSbus_dVm_.imag

    dPf_dVa, dQf_dVa = dSf_dVa.real, dSf_dVa.imag
    dPf_dVm, dQf_dVm = dSf_dVm.real, dSf_dVm.imag

    dPf_dVa2, dQf_dVa2, dPf_dVm2, dQf_dVm2 = branch_derivatives(G=nc.Ybus.real, B=nc.Ybus.imag,
                                                                Vm=np.abs(V), Va=np.angle(V),
                                                                b_shunt_br=nc.branch_data.B / 2.0,
                                                                tap_m=nc.branch_data.m[:, 0],
                                                                tap_angle=nc.branch_data.theta[:, 0],
                                                                F=nc.F, T=nc.T, m=nc.nbr, n=nc.nbus)

    # print('dPf_dVa (original)\n', dPf_dVa.toarray())
    # print('dPf_dVa (new)\n', dPf_dVa2)


    # print('dPf_dVm2 (original)\n', dPf_dVm.toarray())
    # print('dPf_dVm2 (new)\n', dPf_dVm2)

    print('dQf_dVm2 (original)\n', dQf_dVm.toarray())
    print('dQf_dVm2 (new)\n', dQf_dVm2)
    print('dQf_dVm2 (diff)\n', dQf_dVm.toarray() - dQf_dVm2)

    print('dPf_dVm ok:', np.allclose(dPf_dVm.toarray(), dPf_dVm2))
    print('dQf_dVm ok:', np.allclose(dQf_dVm.toarray(), dQf_dVm2))
    print('dPf_dVa ok:', np.allclose(dPf_dVa.toarray(), dPf_dVa2))
    print('dQf_dVa ok:', np.allclose(dQf_dVa.toarray(), dQf_dVa2))


if __name__ == '__main__':
    # fname = '/home/santi/Documentos/Git/GitHub/GridCal/Grids_and_profiles/grids/Lynn 5 Bus (pq).gridcal'
    fname = '/home/santi/Documentos/Git/GitHub/GridCal/Grids_and_profiles/grids/IEEE14 - ntc areas.gridcal'
    # fname = r'C:\Users\SPV86\Documents\Git\GitHub\GridCal\Grids_and_profiles\grids\IEEE14 - ntc areas.gridcal'
    # fname = '/home/santi/Documentos/Git/GitLab/newton-solver/demo/data/IEEE14.json'
    grid = gc.FileOpen(fname).open()
    nc = gc.compile_snapshot_circuit(grid)

    test2(nc)
