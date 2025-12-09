# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import numpy as np
import math
import uuid
import warnings
import scipy.sparse as sp
from typing import Optional

from VeraGridEngine.Devices.Dynamic.events import RmsEvents
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.block_solver import BlockSolver, _compile_parameters_equations, _compile_equations
from VeraGridEngine.Utils.Symbolic.symbolic import Var, Const, Expr, Func, cos, sin, _emit
from VeraGridEngine.Utils.MultiLinear.differential_var import DiffVar, LagVar
from VeraGridEngine.Utils.MultiLinear.diff_blocksolver import DiffBlock

def block2diffblock(block: Block):
    diff_block = DiffBlock.from_block(block)
    diff_block.state_eqs = []
    diff_block.state_vars = []

    for i, eq in enumerate(block.state_eqs):
        state_var = block.state_vars[i]
        dt_var = DiffVar.get_or_create(name = 'dt_' + state_var.name, base_var=state_var)
        eq = eq - dt_var
        diff_block.algebraic_eqs.append(eq)
        diff_block.algebraic_vars.append(state_var)
        diff_block.diff_vars.append(dt_var)

    for i, child_block in enumerate(diff_block.children):
        diff_block.children[i] = block2diffblock(child_block)
    
    return diff_block

        