import os
import numpy as np
import pandas as pd
import time
from numba import jit
import cProfile
import math
import pstats
from scipy import sparse
import numexpr as ne

from GridCal.Engine import FileOpen
from GridCal.Engine.Core.time_series_opf_data import compile_opf_time_circuit
from GridCal.Engine.Simulations.LinearFactors.linear_analysis import LinearAnalysis
from GridCal.Engine.Simulations.ContingencyAnalysis.contingency_analysis_driver import ContingencyAnalysisOptions, ContingencyAnalysisDriver
from GridCal.Engine.Simulations.Clustering import kmeans_sampling, kmeans_approximate_sampling
from GridCal.Engine.basic_structures import BranchImpedanceMode
from examples.ntc_launcher import ntc_launcher


#@jit(nopython=True)
def multiply_fast(a,b):
    return np.dot(a,b)


def block1_PTDF(ptdf, lodf, failed_lines,ov_exists):
    num_branches = lodf.shape[0]
    num_failed_lines = len(failed_lines)

    # Init LODF_NX
    lodf_nx = np.zeros((num_branches, num_branches))

    # Compute L vector
    L = lodf[:, list(failed_lines)] #wo numba

    return num_failed_lines,L,lodf, lodf_nx, failed_lines,num_branches

def block2_PTDF (num_failed_lines,L,lodf, lodf_nx, failed_lines):

    M = np.ones((num_failed_lines, num_failed_lines))

    for i in range(num_failed_lines):
        for j in range(num_failed_lines):
            if not (i == j):
                M[i, j] = -lodf[failed_lines[i], failed_lines[j]]

    L = sparse.coo_matrix(L)

     # Compute LODF_NX
    #lodf_nx[:, list(failed_lines)] = np.dot(L, np.linalg.inv(M))
    lodf_nx[:, list(failed_lines)] = L.dot(np.linalg.inv(M))

    return lodf_nx


def block3_PTDF(ov_exists,num_branches,lodf_nx):
    # Este modulo simplemente suma un vector unitario a lodf_nx de una forma mas rapida
    num_over = len(ov_exists)
    eye_red = np.zeros((num_over, num_branches))
    eye_red[np.arange(num_over), ov_exists] = 1
    # for n_over in range(num_over): #Numba solo soporta que uno de los argumentos de indexacion sea vector, por eso hay que hacer el bucle for. Si se quiere hacer sin bucle for sL puede comentar este y usar la linea de abajo
    #    eye_red[n_over, ov_exists[n_over]] = 1

    lodf_nx = lodf_nx[ov_exists, :] + eye_red

    return lodf_nx


def block4_PTDF(lodf_nx,ptdf):

    lodf_nx1 = sparse.coo_matrix(lodf_nx)


    # PTDF_LODF_NX = np.dot(lodf_nx, ptdf)
    PTDF_LODF_NX = lodf_nx1.dot(ptdf)
    PTDF_LODF_NX.toarray()

    return PTDF_LODF_NX


def block5_gen_function(ov, cont):
    ov_c = ov[:, cont]

    if cont == 0:
        failed_lines = np.array([])
    else:
        failed_lines = np.array( [cont])  # en este caso si se estuvisen realizando fallos multiples esto no sería solo el indice de la columna, sino que sería algo más

    # Considero unicamente aquellas lineas con sobrecargas, ya sean positivas o negativas
    ov_exists = np.where(ov_c != 0)[0]

    return ov_exists, failed_lines, ov_c


def block6_gen_function(ptdf, lodf, failed_lines, ov_exists):
    PTDF_LODF_NX = get_PTDF_LODF_NX(ptdf, lodf, failed_lines, ov_exists)
    return PTDF_LODF_NX

def block7_gen_function(PTDF_LODF_NX,i_ov,ov_c,ov_exist):
    sens = PTDF_LODF_NX[i_ov,:]  # Vector de sensibilidades nudo rama (Fila de la PTDF o PTDF_LODF_NX en el caso de fallo multiple). Hacer función para obtenerlo
    # sens = np.array([1, 5, 3, 4])  # Eliminar cuando se tenga la de arriba activa

    # Busco si la sobrecarga es positiva o negativa, el orden de buses que afectan más
    if ov_c[ov_exist] > 0:
        i_sens = np.argsort(-sens, axis=0)  # Si la sobrecarga es positiva, ordeno de mayor a menor
    else:
        i_sens = np.argsort(sens, axis=0)  # Si la sobrecarga es negativa, ordeno de menor a mayor

    return i_sens, sens


def block8_gen_function(p_available,i_sens,pmax,sens):
    # Calculo del indice del ultimo generador antes de llegar a la maxima potencia
    imax = np.max(np.where(np.cumsum(p_available[i_sens]) <= pmax))

    # Calculo del producto de la potencia disponible con su sensibilidad hasta el imax, ambas ordenadas
    max_correct = np.sum(p_available[i_sens][0:imax] * sens[i_sens][0:imax])


    return imax, max_correct

def block9_gen_function(partial,p_available,i_sens,sens,imax,pmax,max_correct):
    if partial:
        # calculo de la potencia disparada
        p_triggered = np.sum(p_available[i_sens][0:imax])

        # additional correct
        add_correct = (pmax - p_triggered) * sens[i_sens][imax + 1]
        max_correct += add_correct
    return max_correct


def block10_gen_function(ov_c,ov_exist,max_correct,ov_solved,cont):
    # Calculo si la corrección es suficiente, en ese caso marco como True
    c1 = (ov_c[ov_exist] > 0) and (max_correct >= ov_c[ov_exist])  # positive ov
    c2 = (ov_c[ov_exist] < 0) and (max_correct <= ov_c[ov_exist])  # negative ov

    if c1 or c2:
        ov_solved[ov_exist, cont] = True

    return ov_solved

def block11_gen_function():
    return


def block12_gen_function():
    return

def block13_gen_function():
    return


def block14_gen_function():
    return



#@jit(nopython=True)
def get_PTDF_LODF_NX(ptdf, lodf, failed_lines,ov_exists):
    num_failed_lines, L, lodf, lodf_nx, failed_lines, num_branches = block1_PTDF(ptdf, lodf, failed_lines, ov_exists)

    lodf_nx = block2_PTDF(num_failed_lines, L, lodf, lodf_nx, failed_lines)

    lodf_nx = block3_PTDF(ov_exists, num_branches, lodf_nx)

    PTDF_LODF_NX = block4_PTDF(lodf_nx, ptdf)

    return PTDF_LODF_NX

#@jit(nopython=True)
def compute_srap(p_available, ov, pmax, ptdf, lodf,  partial=False):
    # Este codigo me permite, partiendo de una ejecución de flujo de cargas, para una hora y unas contingecias determinadas, establecer si el srap seria capaz de eliminar las sobrecargas

    # Entradas
    #pmax = 15  # Maxima potencia a recortar con SRAP
    #partial = False
    #ov = np.array([0, -6, 7, 0])  # Vector que me indica las desviaciones respecto de los rates para cada una de las lineas
    #p_available = np.array([9, 5, 3, 2])  # Este vector indica la potencia disponible de cada grupo para

    #Cambiar porcentaje
    ptdf = sparse.coo_matrix(np.where(np.abs(ptdf)<0.999,0,ptdf)) #este 99 es tan solo para hacer mas hueca la matriz, representando que solo consideraria el 1% de los generadores para esto, un 0.999 equivale a una sensibilidad del 20% min
    lodf = np.where(np.abs(lodf) < 0.01, 0, lodf)

    #Aqui vendría un bucle for para recorrer todas las posibles contingencias
    num_cont = ov.shape[1] #numero de gruposde contingencias analizados

    # Creo un vector para determinar si el srap ha sido capaz de eliminar tal sobrecarga
    ov_solved = np.full((ov.shape[0],ov.shape[1]),False)

    for cont in range(num_cont):

        ov_exists, failed_lines , ov_c = block5_gen_function(ov, cont)

        if len(ov_exists): #si tenemos alguna sobrecarga

            # PTDF_LODF_NX #matriz que me dice sensibilidades ante el fallo, para calcular esta matriz necsitamos:
            # - PTDF original
            # - LODF original
            # - lineas falladas en el caso de estudio

            PTDF_LODF_NX = block6_gen_function(ptdf, lodf, failed_lines, ov_exists)


            # Asumimos que analizamos cada sobrecarga por separado
            for i_ov,ov_exist in enumerate(ov_exists):

                i_sens, sens = block7_gen_function (PTDF_LODF_NX, i_ov,  ov_c, ov_exist)

                imax, max_correct = block8_gen_function(p_available, i_sens, pmax, sens)

                max_correct = block9_gen_function(partial, p_available, i_sens, sens, imax, pmax, max_correct)

                ov_solved = block10_gen_function(ov_c, ov_exist, max_correct, ov_solved, cont)


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
    Sbase = grid.Sbase

    rates = np.array([branch.rate for branch in grid.get_branches_wo_hvdc()])/Sbase
    monitor = np.array([branch.monitor_loading for branch in grid.get_branches_wo_hvdc()])
    srap_rate = rates*1.4

    rates_matrix = np.tile(rates, (len(rates), 1)).T
    monitor_matrix = np.tile(monitor, (len(monitor), 1)).T
    srap_rate_matrix = np.tile(srap_rate, (len(srap_rate), 1)).T

    #set a condition to review overloads
    cond_overload = np.multiply(((np.abs(driver.results.loading.real) - 1) > 0), monitor_matrix)
    cond_srap = (np.abs(driver.results.Sf.real)/Sbase <= srap_rate_matrix)
    cond = np.multiply(cond_overload, cond_srap)

    #create an overload matrix for each contingency
    ov = np.zeros((len(rates), len(rates)))
    ov[cond] = driver.results.Sf.real[cond]/Sbase

    1+1

    num_buses = 5482
    num_branches = 7600

    p_available = np.ones(num_buses)*100/Sbase
    pmax = 1300 / Sbase
    lodf = driver.results.otdf
    ptdf = np.random.rand(num_branches,num_buses)


    #p_available = /Sbase
    #pmax = /Sbase
    #ptdf =
    #lodf =

    tm_srap = time.time()




    ov_solved = compute_srap(p_available, ov, pmax, ptdf, lodf, partial=False)
    print(f'SRAP computed in {time.time() - tm_srap:.2f} scs.')

    solved_by_srap = len(np.where(ov_solved)[0])/len(np.where(cond)[0])*100
    print(f'SRAP solved {solved_by_srap:.2f} % of the cases')



if __name__ == '__main__':

    path = os.path.join(
        r'C:\Users\posmarfe\OneDrive - REDEIA\Escritorio\2023 MoU Pmode1-3\srap',
        '1_hour_MOU_2022_5GW_v6h-B_pmode1_withcont_1link.gridcal'
    )

    pr = cProfile.Profile()
    cProfile.run('run_srap(gridcal_path = path)')
    ps = pstats.Stats(pr)
    #ps.strip_dirs().sort_stats('cumtime').print_stats(1)
    ps.dump_stats(r'C:\Users\posmarfe\OneDrive - REDEIA\Escritorio')


    #run_srap(gridcal_path = path)


