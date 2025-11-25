# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import List
import numpy as np

from VeraGridEngine.enumerations import DeviceType
from VeraGridEngine.Devices.Dynamic.rms_template import RmsModelTemplate
from VeraGridEngine.Utils.Symbolic.block import Block, Var, Const, Expr, VarPowerFlowRefferenceType
from VeraGridEngine.Utils.Symbolic.symbolic import cos, sin, real, imag, conj, angle, exp, log, abs, UndefinedConst


class Line_0_RmsTemplate(RmsModelTemplate):

    def __init__(self, name: str = "rms_line_0_template"):
        super().__init__(name=name)

        self.tpe: DeviceType = DeviceType.LineDevice

        self.Vmf: Var = Var('')
        self.Vaf: Var = Var('')
        self.Vmt: Var = Var('')
        self.Vat: Var = Var('')

        self.Qf = Var("Qf")
        self.Qt = Var("Qt")
        self.Pf = Var("Pf")
        self.Pt = Var("Pt")

        self.R = Const(1)
        self.X = Const(1)
        self.B = Const(1)

        self.g =  Const((1.0 / complex(self.R.value, self.X.value)).real)
        self.b = Const((1.0 / complex(self.R.value, self.X.value)).imag)
        self.bsh = self.B


    def get_block(self):
        block = Block(
            algebraic_vars=[self.Pf, self.Pt, self.Qf, self.Qt],
            algebraic_eqs=[
                self.Pf - ((self.Vmf ** 2 * self.g) - self.g * self.Vmf * self.Vmt * cos(self.Vaf - self.Vat) + self.b * self.Vmf * self.Vmt * cos(self.Vaf - self.Vat + np.pi / 2)),
                self.Qf - (self.Vmf ** 2 * (-self.bsh / 2 - self.b) - self.g * self.Vmf * self.Vmt * sin(self.Vaf - self.Vat) + self.b * self.Vmf * self.Vmt * sin(
                    self.Vaf - self.Vat + np.pi / 2)),
                self.Pt - ((self.Vmt ** 2 * self.g) - self.g * self.Vmt * self.Vmf * cos(self.Vat - self.Vaf) + self.b * self.Vmt * self.Vmf * cos(self.Vat - self.Vaf + np.pi / 2)),
                self.Qt - (self.Vmt ** 2 * (-self.bsh / 2 - self.b) - self.g * self.Vmt * self.Vmf * sin(self.Vat - self.Vaf) + self.b * self.Vmt * self.Vmf * sin(
                    self.Vat - self.Vaf + np.pi / 2)),
            ])

        block.external_mapping={
                VarPowerFlowRefferenceType.Pf: self.Pf,
                VarPowerFlowRefferenceType.Pt: self.Pt,
                VarPowerFlowRefferenceType.Qf: self.Qf,
                VarPowerFlowRefferenceType.Qt: self.Qt,
            }

        return block
