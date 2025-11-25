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


class Generator_0_RmsTemplate(RmsModelTemplate):

    def __init__(self, name: str = "rms_generator_0_template"):
        super().__init__(name=name)


        self.tpe: DeviceType = DeviceType.GeneratorDevice


        self.Vm: Var = Var('Vm_placeholder')
        self.Va: Var = Var('Va_placeholder')

        self.P_g: Var = Var('P_g')
        self.Q_g: Var = Var('Q_g')

        self.delta = Var("delta")
        self.omega = Var("omega")
        self.psid = Var("psid")
        self.psiq = Var("psiq")
        self.i_d = Var("i_d")
        self.i_q = Var("i_q")
        self.v_d = Var("v_d")
        self.v_q = Var("v_q")
        self.te = Var("te")
        self.et = Var("et")
        self.tm = Var("tm")

        self.R1 = Var("R1")
        self.X1 = Var("X1")
        self.freq = Var("frequ")
        self.M = Var("M")
        self.D = Var("D")
        self.omega_ref = Var("omega_ref")
        self.Kp = Var("Kp")
        self.Ki = Var("Ki")

        self.event_dict = {self.R1: Const(0.0),
                      self.X1: Const(0.3),
                      self.freq: Const(60.0),
                      self.M: Const(4.0),
                      self.D: Const(1.0),
                      self.omega_ref: Const(1.0),
                      self.Kp: Const(0.0),
                      self.Ki: Const(0.0)}


        self.vf = UndefinedConst() # this will disappear when the generator and the exciter model are decoupled
        self.tm0 = UndefinedConst()


    def get_block(self):
        """

        :return:
        """
        block = Block(
            state_vars=[self.delta, self.omega],
            state_eqs=[
                (2 * np.pi * self.freq) * (self.omega - self.omega_ref),
                (self.tm - self.te - self.D * (self.omega - self.omega_ref)) / self.M,
            ],
            algebraic_vars=[self.P_g, self.Q_g, self.v_d, self.v_q, self.i_d, self.i_q, self.psid, self.psiq, self.te, self.tm, self.et],
            algebraic_eqs=[
                self.psid - (self.R1 * self.i_q +self. v_q),
                self.psiq + (self.R1 * self.i_d + self.v_d),
                0 - (self.psid + self.X1 * self.i_d - self.vf),
                0 - (self.psiq + self.X1 * self.i_q),
                self.v_d - (self.Vm * sin(self.delta - self.Va)),
                self.v_q - (self.Vm * cos(self.delta - self.Va)),
                self.te - (self.psid * self.i_q - self.psiq * self.i_d),
                self.P_g - (self.v_d * self.i_d + self.v_q * self.i_q),
                self.Q_g - (self.v_q * self.i_d - self.v_d * self.i_q),
                self.tm - (self.tm0 + self.Kp * (self.omega - self.omega_ref) + self.Ki * self.et),
                2 * np.pi * self.freq * self.et - self.delta,  #
            ],

            init_eqs={
                self.delta: imag(
                    log((self.Vm * exp(1j * self.Va) + (self.R1 + 1j * self.X1) * (
                        conj((self.P_g + 1j * self.Q_g) / (self.Vm * exp(1j * self.Va))))) / (
                            abs(self.Vm * exp(1j * self.Va) + (self.R1 + 1j * self.X1) * (
                                conj((self.P_g + 1j * self.Q_g) / (self.Vm * exp(1j * self.Va)))))))),
                self.omega: self.omega_ref,
                self.v_d: real((self.Vm * exp(1j * self.Va)) * exp(-1j * (self.delta - np.pi / 2))),
                self.v_q: imag((self.Vm * exp(1j * self.Va)) * exp(-1j * (self.delta - np.pi / 2))),
                self.i_d: real(
                    conj((self.P_g + 1j * self.Q_g) / (self.Vm * exp(1j * self.Va))) * exp(-1j * (self.delta - np.pi / 2))),
                self.i_q: imag(
                    conj((self.P_g + 1j * self.Q_g) / (self.Vm * exp(1j * self.Va))) * exp(-1j * (self.delta - np.pi / 2))),
                self.psid: self.R1 * self.i_q + self.v_q,
                self.psiq: -self.R1 * self.i_d - self.v_d,
                self.te: self.psid * self.i_q - self.psiq * self.i_d,
                self.tm: self.te,
                self.et: Const(0),
            })

        block.fix_vars = [self.tm0, self.vf]
        block.fix_vars_eqs = {self.tm0.uid: self.tm,
                        self.vf.uid: self.psid + self.X1 * self.i_d}

        block.external_mapping = {
            VarPowerFlowRefferenceType.P: self.P_g,
            VarPowerFlowRefferenceType.Q: self.Q_g
        }

        block.event_dict = self.event_dict

        return block
