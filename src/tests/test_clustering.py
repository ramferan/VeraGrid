
import os
import time

from GridCal.Engine.Core.time_series_opf_data import compile_opf_time_circuit
from GridCal.Engine.Simulations.Clustering import kmeans_sampling, kmeans_approximate_sampling

from GridCal.Engine import FileOpen
from GridCal.Engine.basic_structures import BranchImpedanceMode

def test_clustering():
    folder = r'\\mornt4.ree.es\DESRED\DPE-Internacional\Interconexiones\FRANCIA\2023 Tracking changes\v0-v4-comparison\Pmode3\k_mod\5GW'
    fname = os.path.join(folder, 'MOU_2022_5GW_v6h-B_pmode3_withcont_1link.gridcal')


    tm0 = time.time()
    main_circuit = FileOpen(fname).open()
    print(f'circuit opened in {time.time() - tm0:.2f} scs.')

    tm0 = time.time()
    nc = compile_opf_time_circuit(
        circuit=main_circuit,
        apply_temperature=False,
        branch_tolerance_mode=BranchImpedanceMode.Specified
    )
    print(f'numerical circuit computed in {time.time() - tm0:.2f} scs.')

    X = nc.Sbus.real.T

    tm1 = time.time()
    time_indices, sampled_probabilities = kmeans_sampling(
        X=X,
        n_points=200)
    print(f'kmeans computed in {time.time() - tm1:.2f} scs.')

    tm2 = time.time()
    time_indices_, sampled_probabilities_ = kmeans_approximate_sampling(
        X=X,
        n_points=200)
    print(f'kmeans_sampling computed in {time.time() - tm2:.2f} scs.')

    return time_indices == time_indices_ and sampled_probabilities == sampled_probabilities_


