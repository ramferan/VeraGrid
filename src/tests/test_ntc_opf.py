import os
import time
import numpy as np
from unittest import TestCase, TestLoader
from GridCal.Engine import basic_structures as bs
from GridCal.Engine import Devices as dev
from GridCal.Engine.Simulations.ATC.available_transfer_capacity_driver import AvailableTransferMode
from GridCal.Engine.Simulations.NTC.ntc_options import OptimalNetTransferCapacityOptions
from GridCal.Engine.Simulations.NTC.ntc_driver import OptimalNetTransferCapacityDriver
from GridCal.Engine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
from GridCal.Engine.basic_structures import SolverType
from GridCal.Engine import FileOpen


class TestCases(TestCase):

    # TestLoader.sortTestMethodsUsing = None

    def test_pmode1(self):
        tm0 = time.time()

        folder = r'\\mornt4.ree.es\DESRED\DPE-Internacional\Interconexiones\FRANCIA\2023 MoU Pmode3\Pmode3_conting\5GW\h_pmode1_esc_inst'
        fname = os.path.join(folder, r'unahorita_23_2_pmode1.gridcal')

        circuit = FileOpen(fname).open()

        areas_from_idx = [0]
        areas_to_idx = [1]

        areas_from = [circuit.areas[i] for i in areas_from_idx]
        areas_to = [circuit.areas[i] for i in areas_to_idx]

        lst_from = circuit.get_areas_buses(areas_from)
        lst_to = circuit.get_areas_buses(areas_to)

        idx_from = np.array([i for i, bus in lst_from])
        idx_to = np.array([i for i, bus in lst_to])

        options = OptimalNetTransferCapacityOptions(
            area_from_bus_idx=idx_from,
            area_to_bus_idx=idx_to,
            mip_solver=bs.MIPSolvers.CBC,
            generation_formulation=dev.GenerationNtcFormulation.Proportional,
            monitor_only_sensitive_branches=True,
            branch_sensitivity_threshold=0.05,
            skip_generation_limits=True,
            consider_contingencies=True,
            consider_gen_contingencies=True,
            consider_hvdc_contingencies=True,
            consider_nx_contingencies=True,
            dispatch_all_areas=False,
            generation_contingency_threshold=1000,
            tolerance=1e-2,
            sensitivity_dT=100.0,
            transfer_method=AvailableTransferMode.InstalledPower,
            time_limit_ms=1e4,
            loading_threshold_to_report=98,
        )

        pf_options = PowerFlowOptions(
            solver_type=SolverType.DC
        )
        print('Running optimal net transfer capacity...')

        # set optimal net transfer capacity driver instance
        driver = OptimalNetTransferCapacityDriver(
            grid=circuit,
            options=options,
            pf_options=pf_options,
        )

        driver.run()

        ttc = np.floor(driver.results.get_exchange_power())
        ettc = 5502.0
        result = np.isclose(ttc, ettc, atol=1)

        print(f'The computed TTC is {ttc}, the expected value is {ettc}')
        print(f'Test result is {result}. Computed in {time.time()-tm0:.2f} scs.')

        self.assertTrue(result)


if __name__ == '__main':
    TestCases.run()