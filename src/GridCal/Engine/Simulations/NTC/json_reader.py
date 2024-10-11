import json
from GridCal.Engine.Simulations.LinearFactors.linear_analysis import make_lodf


def read_contingency_json():
    # Opening JSON file
    with open(r'C:\Users\POSMARFE\OneDrive - REDEIA\Documentos\AprendiendoPython\OptPython\NX_v2.json') as json_file:
        contingency_dict = json.load(json_file)

    return contingency_dict


if __name__ == '__main__':
    import time
    import os
    import numpy as np
    from GridCal.Engine.Core.snapshot_opf_data import compile_snapshot_opf_circuit
    from GridCal.Engine.IO.file_handler import FileOpen
    from GridCal.Engine.basic_structures import BranchImpedanceMode
    from GridCal.Engine.Simulations.LinearFactors.linear_analysis import LinearAnalysis


    folder = r'\\mornt4\DESRED\DPE-Internacional\Interconexiones\FRANCIA\2022 MoU\5GW 8.0\Con N-x\merged\GridCal'
    fname = os.path.join(folder, 'MOU_2022_5GW_v6h-B_pmode1.gridcal')

    tm0 = time.time()
    print(f'circuit opened in {time.time() - tm0:.2f} scs.')


    main_circuit = FileOpen(fname).open()
    numerical_circuit_ = compile_snapshot_opf_circuit(
        circuit=main_circuit,
        apply_temperature=False,
        branch_tolerance_mode=BranchImpedanceMode.Specified
    )

    linear = LinearAnalysis(
        grid=main_circuit,
        distributed_slack=False,
        correct_values=False
    )
    linear.run()
    lodf=linear.LODF

    contingencies = read_contingency_json()

    for item1 in contingencies['contingencies']:
        a=np.array([])
        for each_cont in range(len(contingencies['contingencies'][item1]['elements'])):
            new_element = contingencies['contingencies'][item1]['elements'][each_cont]['element']
            a = np.append(a, new_element)

        #a = np.array(contingencies['contingencies'][item1]['elements'])
        b = numerical_circuit_.branch_data.codes
        contingency_branches_index = np.array(np.where(np.isin(b,a))) #Find the indexes of the branches associated with the contingencies
        num_contingency_branches = contingency_branches_index.size  # Number of branches with contingency
        if(num_contingency_branches<a.size):
            contingencies['contingencies'][item1]['contingency_exists']='false'
            contingencies['contingencies'][item1]['sensitivity_array'] = []
        else:
            contingencies['contingencies'][item1]['contingency_exists'] = 'true'
            np.sort(contingency_branches_index)  # Order the index of branches with contingency
            L = lodf[:, contingency_branches_index[0]]  # Take the columns of the LODF associated with the contingencies
            M = np.zeros((num_contingency_branches,num_contingency_branches))  # Create a zeros matrix to populate in the following loop

            for j in range(num_contingency_branches):
                for k in range(num_contingency_branches):
                    if (j == k):
                        M[j, k] = 1
                    else:
                        M[j, k] = -lodf[contingency_branches_index[0][j], contingency_branches_index[0][k]]  # Take the elements of the LODF associated with the contingencies

            sensitivity_array = np.matmul(L,np.linalg.inv(M))
            contingencies['contingencies'][item1]['sensitivity_array'] = sensitivity_array  # Add to the dictionary
    1+1




























