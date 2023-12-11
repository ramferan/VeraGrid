
import os
import numpy as np
import pandas as pd
import time

from GridCal.Engine import FileOpen
from GridCal.Engine.Core.time_series_opf_data import compile_opf_time_circuit
from GridCal.Engine.Simulations.Clustering import kmeans_sampling, kmeans_approximate_sampling
from GridCal.Engine.basic_structures import BranchImpedanceMode
from examples.ntc_launcher import ntc_launcher


def expand_kmeans():



    tm0 = time.time()
    folder = r'\\mornt4.ree.es\DESRED\DPE-Internacional\Interconexiones\FRANCIA\2023 MoU Pmode1-3'
    xlsx_path = os.path.join(folder, 'kmeans_expansion.xlsx')

    scenarios = pd.read_excel(xlsx_path, sheet_name='Hoja1')
    scenarios.columns = scenarios.columns.str.lower()

    for i, row in scenarios.iterrows():
        results_path = os.path.join(os.path.dirname(row['path']), row['results'])

        tm0 = time.time()
        print('Loading grical circuit... ', sep=' ')
        grid = FileOpen(row['path']).open()
        print(f'[{time.time() - tm0:.2f} scs.]')

        time_indices, sampled_probabilities, samples = get_kmeans_with_samples(grid=grid)

        samples = pd.DataFrame(samples, columns=['cluster_index']).reset_index(names=['time_index'])

        df = pd.read_csv(results_path)[['Time index', 'Time', 'TTC']].drop_duplicates()
        df = df.rename(columns={'Time index': 'time_index'})

        df = df.merge(samples, on='time_index')
        df = df.drop(columns=['time_index'])

        results = samples.merge(df, on='cluster_index').sort_values(by='time_index')

        results = results.rename(
            columns={
                'cluster_index': 'Cluster Index',
                'Time': 'Cluster Time'
            }
        )

        trm = np.full(results.shape[0], row['trm'])
        trm[np.where(results['TTC'] < 0)] = -row['trm']

        results['NTC'] = results['TTC'] - trm

        results.to_csv(os.path.join(os.path.dirname(row['path']), row['sense']+'_expanded_kmeans.csv'),index=False)
        1+1


        """""

        lost_cluster = [i for i in samples[['cluster_index']].drop_duplicates().values if
                        i not in results['Cluster Index'].drop_duplicates().values][0]

        wrong_time_index = samples[samples['cluster_index'] == lost_cluster[0]]['time_index']
        right_time_index = np.unique([i for i in results['time_index'].values if i not in wrong_time_index.values])

        # results = results['']
        lost_time = time_indices[lost_cluster][0]

        print('Computing ntc...')
        ntc = ntc_launcher(
            grid=grid,
            start=lost_time,
            end=lost_time,
            areas_from_idx=[0],
            areas_to_idx=[1],
            consider_contingencies=True,
            consider_gen_contingencies=False,
            consider_hvdc_contingencies=True,
            consider_nx_contingencies=True,
            monitor_only_sensitive_branches=True,
            branch_sensitivity_threshold=0.05,
            skip_generation_limits=True,
            generation_contingency_threshold=1000,
            time_limit_ms=1e5,
            loading_threshold_to_report=98,
            use_clustering=False,
            cluster_number=200)
        k=1
        
        """""


def get_kmeans_with_samples(grid):

    tm0 = time.time()
    print('Computing numerical circuit', end=' ')
    nc = compile_opf_time_circuit(
        circuit=grid,
        apply_temperature=False,
        branch_tolerance_mode=BranchImpedanceMode.Specified
    )
    print(f'[{time.time() - tm0:.2f} scs.]')

    X = nc.Sbus.real.T

    tm1 = time.time()
    print('kmeans sampling... ', end=' ')
    time_indices, sampled_probabilities, samples = kmeans_sampling(
            X=X,
            n_points=200,
            with_samples=True
    )

    print(f'[{time.time() - tm1:.2f} scs.]')

    return time_indices, sampled_probabilities, samples

if __name__ == '__main__':
    expand_kmeans()
