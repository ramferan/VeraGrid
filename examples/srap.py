import os
import numpy as np
import pandas as pd
import time

from GridCal.Engine import FileOpen
from GridCal.Engine.Core.time_series_opf_data import compile_opf_time_circuit
from GridCal.Engine.Simulations.LinearFactors.linear_analysis import LinearAnalysis
from GridCal.Engine.Simulations.ContingencyAnalysis.contingency_analysis_driver import ContingencyAnalysisOptions, ContingencyAnalysisDriver
from GridCal.Engine.Simulations.Clustering import kmeans_sampling, kmeans_approximate_sampling
from GridCal.Engine.basic_structures import BranchImpedanceMode
from examples.ntc_launcher import ntc_launcher


def get_PTDF_LODF_NX(ptdf, lodf, idx_fail):
    num_branches = lodf.shape[0]
    num_fails = len(idx_fail)

    # Init LODF_NX
    lodf_nx = np.zeros((num_branches, num_branches))

    # Compute L vector
    L = lodf[:, idx_fail]  # Take the columns of the LODF associated with the contingencies

    # Compute M matrix [n, n] (lodf relating the outaged lines to each other)
    M = np.ones((num_fails, num_fails))
    for i in range(num_fails):
        for j in range(num_fails):
            if not (i == j):
                M[i, j] = -lodf[idx_fail[i], idx_fail[j]]

    # Compute LODF_NX
    lodf_nx[:, idx_fail] = np.matmul(L, np.linalg.inv(M))

    # Compute PTDF_LODF_NX
    PTDF_LODF_NX = np.matmul(lodf_nx + np.eye(num_branches), ptdf)

    return PTDF_LODF_NX


def compute_srap(p_available, ov, ptdf, lodf, idx_fail, partial=False):
    # Este codigo me permite, partiendo de una ejecución de flujo de cargas, para una hora y unas contingecias determinadas, establecer si el srap seria capaz de eliminar las sobrecargas

    # Entradas
    pmax = 15  # Maxima potencia a recortar con SRAP
    partial = False
    ov = np.array(
        [0, -6, 7, 0])  # Vector que me indica las desviaciones respecto de los rates para cada una de las lineas
    p_available = np.array([9, 5, 3, 2])  # Este vector indica la potencia disponible de cada grupo para
    # PTDF_LODF_NX #matriz que me dice sensibilidades ante el fallo, para calcular esta matriz necsitamos:
    # - PTDF original
    # - LODF original
    # - lineas falladas en el caso de estudio
    PTDF_LODF_NX = get_PTDF_LODF_NX(ptdf, lodf, idx_fail)

    # Considero unicamente aquellas lineas con sobrecargas, ya sean positivas o negativas
    ov_exists = np.where(ov != 0)[0]

    # Creo un vector para determinar si el srap ha sido capaz de eliminar tal sobrecarga
    ov_solved = [False] * len(ov_exists)

    # Asumimos que analizamos cada sobrecarga por separado
    for ov_exist in ov_exists:
        sens = PTDF_LODF_NX[ov_exists,
               :]  # Vector de sensibilidades nudo rama (Fila de la PTDF o PTDF_LODF_NX en el caso de fallo multiple). Hacer función para obtenerlo
        # sens = np.array([1, 5, 3, 4])  # Eliminar cuando se tenga la de arriba activa

        # Busco si la sobrecarga es positiva o negativa, el orden de buses que afectan más
        if ov[ov_exist] > 0:
            i_sens = np.argsort(-sens[0, :], axis=0)  # Si la sobrecarga es positiva, ordeno de mayor a menor
        else:
            i_sens = np.argsort(sens[0, :], axis=0)  # Si la sobrecarga es negativa, ordeno de menor a mayor

        # Calculo del indice del ultimo generador antes de llegar a la maxima potencia
        imax = np.max(np.where(np.cumsum(p_available[i_sens]) <= pmax))

        # Calculo del producto de la potencia disponible con su sensibilidad hasta el imax, ambas ordenadas
        max_correct = np.sum(p_available[i_sens][0:imax] * sens[i_sens][0:imax])

        if partial:
            # calculo de la potencia disparada
            p_triggered = np.sum(p_available[i_sens][0:imax])

            # additional correct
            add_correct = (pmax - p_triggered) * sens[i_sens][imax + 1]
            max_correct += add_correct

        # Calculo si la corrección es suficiente, en ese caso marco como True
        c1 = (ov[ov_exist] > 0) and (max_correct >= ov[ov_exist])  # positive ov
        c2 = (ov[ov_exist] < 0) and (max_correct <= ov[ov_exist])  # negative ov

        if c1 or c2:
            ov_solved[ov_exist] = True

    return ov_solved


def run_srap(gridcal_path):

    tm0 = time.time()
    print('Loading grical circuit... ', sep=' ')
    grid = FileOpen(gridcal_path).open()
    print(f'[{time.time() - tm0:.2f} scs.]')

    tm_ = time.time()

    options = ContingencyAnalysisOptions(
        distributed_slack=True,
        correct_values=True,
        use_provided_flows=False,
        Pf=None,
        pf_results=None,
        nonlinear=False)

    driver = ContingencyAnalysisDriver(
        grid=grid,
        options=options)

    driver.run()

    print(f'Contingency analysis computed in {time.time() - tm_:.2f} scs.')

    #take rates and monitoring logic, convert them to matrix
    rates = [branch.rate for branch in grid.get_branches_wo_hvdc()]
    monitor = [branch.monitor_loading for branch in grid.get_branches_wo_hvdc()]

    rates_matrix = np.tile(rates, (len(rates), 1)).T
    monitor_matrix = np.tile(monitor, (len(monitor), 1)).T

    #set a condition to review overloads
    cond = np.multiply(((np.abs(driver.results.loading.real) - 1) > 0), monitor_matrix)

    #create an overload matrix for each contingency
    ov = np.zeros((len(rates), len(rates)))
    ov[cond] = np.multiply(driver.results.loading.real[cond], rates_matrix[cond])





    #OJO, y el Sbase?


    ov_solved = compute_srap(p_available, ov, ptdf, lodf, idx_fail, partial=False)



if __name__ == '__main__':

    path = os.path.join(
        r'C:\Users\posmarfe\OneDrive - REDEIA\Escritorio\2023 MoU Pmode1-3\srap',
        '1_hour_MOU_2022_5GW_v6h-B_pmode1_withcont_1link.gridcal'
    )

    run_srap(gridcal_path = path)











