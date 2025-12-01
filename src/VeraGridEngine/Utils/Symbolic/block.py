# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Sequence, List, Dict, Any, Optional

from VeraGridEngine.Devices.Dynamic.events import RmsEvent
from VeraGridEngine.Devices.Parents.physical_device import PhysicalDevice
from VeraGridEngine.Utils.Symbolic.symbolic import Var, LagVar, Const, Expr, make_symbolic, UndefinedConst, cos, sin, \
    real, sqrt, atan, \
    imag, conj, angle, exp, log, abs, piecewise, DiffVar, hard_sat, f_exc
from VeraGridEngine.enumerations import VarPowerFlowRefferenceType


def _new_uid() -> int:
    """
    Generate a fresh UUID‑v4 string.
    :return: UUIDv4 in integer format
    """
    return uuid.uuid4().int


def _serialize_expr_list(exprs: List[Expr]) -> List[Dict[str, Any]]:
    """

    :param exprs:
    :return:
    """
    return [expr.to_dict() for expr in exprs]


def _serialize_var_list(vars_: List[Var | Const]) -> List[Dict[str, Any]]:
    """
    Serialize list of variables or constants
    :param vars_: list of Var or Const
    :return: List of dictionaries with the serialized data
    """
    return [v.to_dict() for v in vars_]


def _serialize_undefinedconst_list(undefconsts_: List[UndefinedConst]) -> List[Dict[str, Any]]:
    """
    Serialize list of variables or constants
    :param vars_: list of Var or Const
    :return: List of dictionaries with the serialized data
    """
    return [undefconst.to_dict() for undefconst in undefconsts_]


def _deserialize_expr_list(expr_dicts: List[Dict[str, Any]]) -> List[Expr]:
    """

    :param expr_dicts:
    :return:
    """
    return [Expr.from_dict(d) for d in expr_dicts]


def _deserialize_var_list(var_dicts: List[Dict[str, Any]]) -> List[Var | Const | UndefinedConst]:
    """
    De-serialize previously serialized data into List of Vars or Const
    :param var_dicts: List of serialized data
    :return: List of Vars or Const
    """
    result = list()
    for d in var_dicts:
        if d["type"] == "Var":
            result.append(Var(name=d["name"], uid=d["uid"]))
        elif d["type"] == "Const":
            result.append(Const(value=d["value"], uid=d["uid"]))
        elif d['type'] == "UndefinedConst":
            result.append(UndefinedConst())
        else:
            raise ValueError(f"Unknown variable type {d['type']}")
    return result


def block2diffblock(block: Block):
    diff_block = DiffBlock.from_block(block)

    for i, eq in enumerate(block.state_eqs):
        state_var = block.state_vars[i]
        dt_var = DiffVar.get_or_create(name='dt_' + state_var.name, base_var=state_var)
        eq = eq - dt_var
        diff_block.algebraic_eqs.append(eq)
        diff_block.algebraic_vars.append(state_var)
        if dt_var not in diff_block.diff_vars:
            diff_block.diff_vars.append(dt_var)

    diff_block.state_eqs = []
    diff_block.state_vars = []

    for i, child_block in enumerate(diff_block.children):
        diff_block.children[i] = block2diffblock(child_block)

    return diff_block


def tf_to_diffblock(num: np.ndarray, den: np.ndarray, x: Var | Expr, y: Var = None, name: Optional[str] = ''):
    """
    num: list of numerator coefficients [b0, b1, ..., bm]
    den: list of denominator coefficients [a0, a1, ..., an]
    x:   sympy symbol for input
    y:   sympy function of t for output
    """
    if len(num) > len(den):
        raise ValueError("Transfer function is improper: numerator degree > denominator degree.")
    if num[-1] == 0 or den[-1] == 0:
        raise ValueError("Leading coefficient of cannot be zero.")

    if y is None:
        y = Var('y_' + name)

    aux_eqs = []
    aux_vars = []
    # check if is an expression
    if not isinstance(x, Var):
        u = Var('u_' + name)
        aux_eqs.append((u - x).simplify())
        aux_vars.append(u)
        x = u

    diff_vars_x = [x]
    diff_vars_y = [y]
    diff_vars = []
    base_var = x
    for i in range(1, len(num)):
        in_registry = base_var.uid in DiffVar._registry
        new_diff = DiffVar.get_or_create(name=f'dt_{i}_' + x.name, base_var=base_var)
        diff_vars_x.append(new_diff)
        if not in_registry:
            diff_vars.append(new_diff)
        base_var = new_diff
    base_var = y
    for i in range(1, len(den)):
        in_registry = base_var.uid in DiffVar._registry
        new_diff = DiffVar.get_or_create(name=f'dt_{i}_' + y.name, base_var=base_var)
        diff_vars_y.append(new_diff)
        if not in_registry:
            diff_vars.append(new_diff)
        base_var = new_diff

    # Create the diff equation
    rhs = np.array(diff_vars_x) @ np.array(num)
    lhs = np.array(diff_vars_y) @ np.array(den)

    block = DiffBlock()
    block.algebraic_vars = [y] + aux_vars
    block.algebraic_eqs = [lhs - rhs] + aux_eqs
    block.diff_vars = diff_vars

    return block, y


def tf_to_diffblock_with_states(num: np.ndarray, den: np.ndarray, x, y: Var = None, name: Optional[str] = ''):
    """
    num: numerator coefficients [b0,...,bm]
    den: denominator coefficients [a0,...,an], with an != 0
    u:   input Var
    y:   output Var
    """
    if len(num) > len(den):
        raise ValueError("Transfer function is improper: numerator degree > denominator degree.")
    if num[-1] == 0 or den[-1] == 0:
        raise ValueError("Leading coefficient of cannot be zero.")

    if y is None:
        y = Var('y_' + name)

    aux_eqs = []
    aux_vars = []
    # check if x is a expression
    if not isinstance(x, Var):
        u = Var('u_' + name)
        aux_eqs.append(u - x)
        aux_vars.append(u)
        x = u

    # Normalize
    den = [c / den[-1] for c in den]
    num = [c / den[-1] for c in num]

    # States
    x_states = [x]
    y_states = [y]
    y_states.extend([Var(f"y{i}") for i in range(1, len(den))])  # y1,...,yn
    x_states.extend([Var(f"x{i}") for i in range(1, len(num))])  # x1,...,xm

    # Differential equations (canonical form)
    diff_eqs = []
    diff_vars = []
    for i in range(len(y_states) - 1):
        in_registry = y_states[i].uid in DiffVar._registry
        dy = DiffVar.get_or_create(f"d_{y_states[i].name}", base_var=y_states[i])
        diff_eqs.append(y_states[i + 1] - dy)
        if not in_registry:
            diff_vars.append(dy)

    for i in range(len(num) - 1):
        in_registry = x_states[i].uid in DiffVar._registry
        dx = DiffVar.get_or_create(f"d^{i}_{x.name}", base_var=x_states[i])
        diff_eqs.append(x_states[i + 1] - dx)
        if not in_registry:
            diff_vars.append(dx)

            # Last equation: linear combination
    last_eq = (
            sum(den[i] * y_states[i] for i in range(len(y_states)))
            - sum(num[i] * x_states[i] for i in range(len(x_states)))
    )
    diff_eqs.append(last_eq)

    block = DiffBlock(
        algebraic_vars=y_states + x_states[1:] + aux_vars,
        algebraic_eqs=diff_eqs + aux_eqs,
    )
    block.diff_vars = diff_vars
    block.name = 'TF'
    return block, y_states[0]


def tf_to_block_with_states(num: np.ndarray, den: np.ndarray, x):
    """
    num: numerator coefficients [b0,...,bm]
    den: denominator coefficients [a0,...,an], with an != 0
    u:   input Expr or Var
    y:   output Var
    """
    if len(num) >= len(den):
        raise ValueError("Transfer function is improper: numerator degree > denominator degree.")
    if num[-1] == 0 or den[-1] == 0:
        raise ValueError("Leading coefficient of cannot be zero.")

    aux_eqs = []
    aux_vars = []
    # check if xis a expression
    if not isinstance(x, Var):
        u = Var('u')
        aux_eqs.append(u - x)
        aux_vars.append(u)
        x = u

    # Normalize
    den = [c / den[-1] for c in den]
    num = [c / den[-1] for c in num]

    order = len(den)  # system order
    x_states = [x]  # x0
    x_states.extend([Var(f"x{i}") for i in range(1, order + 1)])  # x1,...,xn

    # Differential equations (canonical form)
    state_eqs = []
    state_vars = []
    for i in range(1, order):
        state_eqs.append(x_states[i])
        state_vars.append(x_states[i + 1])

    y_states = [Var(f"y{i}") for i in range(len(num) + 1)]  # x1,...,xn
    for i in range(len(num)):
        state_eqs.append(y_states[i])
        state_vars.append(y_states[i + 1])

    # Last equation: linear combination
    last_eq = (
            sum(den[i] * y_states[i] for i in range(order))
            - sum(num[i] * x_states[i] for i in range(len(num)))
    )

    block = Block(
        state_eqs=state_eqs,
        state_vars=state_vars,
        algebraic_eqs=[last_eq] + aux_eqs,
        algebraic_vars=x_states[0] + y_states[0] + aux_vars,
    )
    return block, y_states[0]


class Block:
    """
    Class representing a Block
    """

    def __init__(self,
                 state_vars: List[Var] | None = None,
                 state_eqs: List[Expr] | None = None,
                 algebraic_vars: List[Var] | None = None,
                 algebraic_eqs: List[Expr] | None = None,
                 parameters: Dict[str, Const] | None = None,
                 init_eqs: Dict[Var, Expr] | None = None,
                 children: List["Block"] | None = None,
                 in_vars: Dict[str, Var] | None = None,
                 out_vars: Dict[str, Var] | None = None):

        # ,
        # external_mapping: Dict[VarPowerFlowRefferenceType, Var] | None = None
        """
        This represents a group of equations or a group of blocks

        :param algebraic_vars: List of non differential variables (AKA algebraic)
        :param algebraic_eqs: List of equations that provide values for the algebraic variables
        :param state_vars: List of differential variables (AKA state variables)
        :param state_eqs: List of equations that provide values for the state variables
        :param children: List of other blocks to be flattened later into this block
        :param in_vars: List of variables from other blocks that we use here
        :param out_vars: List of variables that already exist in algebraic_vars or state_vars that we want to expose
        :param init_eqs: List of equations that help initializing the block variables (algebraic and state)
        """

        self.name: str = ""

        self.uid: int = _new_uid()
        self.vars2device: Dict[int, PhysicalDevice] = dict()
        self.vars_glob_name2uid: Dict[str, int] = dict()

        # TODO: Change to dictionary of Var : Expr
        self.state_vars: List[Var] = list() if state_vars is None else state_vars
        self.state_eqs: List[Expr] = list() if state_eqs is None else state_eqs

        # TODO: Change to dictionary of Var : Expr
        self.algebraic_vars: List[Var] = list() if algebraic_vars is None else algebraic_vars
        self.algebraic_eqs: List[Expr] = list() if algebraic_eqs is None else algebraic_eqs

        # initialization
        self.init_eqs: Dict[Var, Expr] = dict() if init_eqs is None else init_eqs

        # vars to make this recursive
        self.children: List["Block"] = list() if children is None else children

        self.in_vars: Dict[str, Var] = dict() if in_vars is None else in_vars
        self.out_vars: Dict[str, Var] = dict() if out_vars is None else out_vars

        self.parameters: Dict[str, Const] = dict() if parameters is None else parameters

        self.external_mapping: Dict[VarPowerFlowRefferenceType, Var] = dict()

        # fix vars will disappear when the exciter and governor models are decoupled from generator
        self.fix_vars: List[UndefinedConst] = list()
        self.fix_vars_eqs: Dict[Any, Expr] = dict()

        # ---------------------------------------------------------------------------------------------------------------
        self.var_mapping = {v.name: v for v in self.algebraic_vars}

        # Dictionary of Variables and their Expressions that appear due to an event
        # this is the dictionary of "parameters" that may change and their equations
        self.event_dict: Dict[Var, Expr] = dict()

    def empty(self) -> bool:
        """

        :return:
        """
        return (len(self.state_vars) + len(self.algebraic_vars)) == 0

    def E(self, d: VarPowerFlowRefferenceType) -> Var:
        """

        :param d:
        :return:
        """
        return self.external_mapping[d]

    def V(self, d: str) -> Var:
        """

        :param d:
        :return:
        """
        return self.var_mapping[d]

    def add(self, val: "Block"):
        """
        Add another block
        :param val: Block
        """
        self.children.append(val)

    def remove(self, val: Block):
        """
        Remove a block from block children
        :param val: Block
        """
        self.children.remove(val)

    def apply_events(self, events: List[RmsEvent]):
        """
        Apply events to this block
        :param events:
        :return:
        """
        # TODO Implement filling self.event_dicts with the RmsEvents info
        pass

    def get_all_blocks(self) -> List[Block]:
        """
        Depth-first collection of all *primitive* Blocks.
        """

        flat: List[Block] = [self]
        for el in self.children:
            flat.extend(el.get_all_blocks())

        return flat

    def get_vars(self) -> List[Var]:
        """
        Get a list of algebraic + state vars
        :return: List[Var]
        """
        return self.algebraic_vars + self.state_vars

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "vars2device": {var_uid: dev.uid for var_uid, dev in self.vars2device.items()},
            "vars_glob_name2uid": self.vars_glob_name2uid,
            "state_vars": _serialize_var_list(self.state_vars),
            "state_eqs": _serialize_expr_list(self.state_eqs),
            "algebraic_vars": _serialize_var_list(self.algebraic_vars),
            "algebraic_eqs": _serialize_expr_list(self.algebraic_eqs),
            "init_eqs": [
                {"var": var.to_dict(), "expr": expr.to_dict()}
                for var, expr in self.init_eqs.items()
            ],
            "fix_vars": _serialize_undefinedconst_list(self.fix_vars),
            "fix_vars_eqs": [
                {"uid": undef.uid, "expr": expr.to_dict()}
                for undef, expr in self.fix_vars_eqs.items()
            ],
            "external_mapping": {str(dynvartype): var.to_dict() for dynvartype, var in self.external_mapping.items()},
            "event_dict": [
                {"var": var.to_dict(), "expr": expr.to_dict()}
                for var, expr in self.event_dict.items()
            ],
            "name": self.name,
            "children": [child.to_dict() for child in self.children],
            "in_vars": [{"number": number, "var": var.to_dict()} for number, var in self.in_vars.items()],
            "out_vars": [{"number": number, "var": var.to_dict()} for number, var in self.out_vars.items()]
        }

    @staticmethod
    def parse(data: Dict[str, Any]) -> "Block":
        state_vars = _deserialize_var_list(data.get("state_vars", []))
        state_eqs = _deserialize_expr_list(data.get("state_eqs", []))
        algebraic_vars = _deserialize_var_list(data.get("algebraic_vars", []))
        algebraic_eqs = _deserialize_expr_list(data.get("algebraic_eqs", []))
        children = [Block.parse(child) for child in data.get("children", [])]

        block = Block(
            state_vars=state_vars,
            state_eqs=state_eqs,
            algebraic_vars=algebraic_vars,
            algebraic_eqs=algebraic_eqs,
            init_eqs={},
            children=children,
        )

        block.uid = int(data.get("uid", block.uid))
        block.name = data.get("name", "")
        block.vars_glob_name2uid = data.get("vars_glob_name2uid", {})
        block.vars2device = {int(k): v for k, v in data.get("vars2device", {}).items()}

        block.init_eqs = {}
        for pair in data.get("init_eqs", []):
            var_dict = pair["var"]
            expr_dict = pair["expr"]
            var = Var(name=var_dict["name"], uid=var_dict.get("uid"))
            expr = Expr.from_dict(expr_dict)
            block.init_eqs[var] = expr

        fv_list = _deserialize_var_list(data.get("fix_vars", []))
        block.fix_vars = [v for v in fv_list if isinstance(v, UndefinedConst)]

        block.fix_vars_eqs = {}
        undef_by_uid = {getattr(u, "uid", None): u for u in block.fix_vars}
        for item in data.get("fix_vars_eqs", []):
            uid = item.get("uid")
            expr = Expr.from_dict(item["expr"])
            undef = undef_by_uid.get(uid)
            if undef is not None:
                block.fix_vars_eqs[undef] = expr
            else:
                new_undef = UndefinedConst()
                block.fix_vars_eqs[new_undef] = expr
                block.fix_vars.append(new_undef)

        block.external_mapping = {}
        for key_str, var_dict in data.get("external_mapping", {}).items():
            try:
                key = VarPowerFlowRefferenceType[
                    key_str] if key_str in VarPowerFlowRefferenceType.__members__ else VarPowerFlowRefferenceType(
                    key_str)
            except Exception:
                key = key_str
            var = Var(name=var_dict["name"], uid=var_dict.get("uid"))
            block.external_mapping[key] = var

        block.event_dict = {}
        for pair in data.get("event_dict", []):
            var_dict = pair["var"]
            expr_dict = pair["expr"]
            var = Var(name=var_dict["name"], uid=var_dict.get("uid"))
            expr = Expr.from_dict(expr_dict)
            block.event_dict[var] = expr

        block.in_vars = {}
        for pair in data.get("in_vars", []):
            number = pair["number"]
            var_dict = pair["var"]
            var = Var(name=var_dict["name"], uid=var_dict.get("uid"))
            block.in_vars[number] = var

        block.out_vars = {}
        for pair in data.get("out_vars", []):
            number = pair["number"]
            var_dict = pair["var"]
            var = Var(name=var_dict["name"], uid=var_dict.get("uid"))
            block.out_vars[number] = var

        block.var_mapping = {v.name: v for v in block.algebraic_vars}
        return block

    def copy(self) -> "Block":
        """
        Make a deep copy of this
        :return: deep copy Block
        """
        return Block.parse(data=self.to_dict())

    def __eq__(self, other):
        if isinstance(other, Block):
            return self.to_dict() == other.to_dict()
        else:
            return False


class DiffBlock(Block):
    diff_vars: List[DiffVar]
    lag_vars: List[LagVar]
    reformulated_vars: List[DiffVar]
    differential_eqs: List[Expr]
    pseudo_transient: bool = False

    @classmethod
    def from_block(cls, block: Block, **kwargs):
        if isinstance(block, DiffBlock):
            return block  # already a DiffBlock

        # Create a new instance of DiffBlock, copying fields from the original Block
        obj = cls.__new__(cls)  # create instance without __init__
        obj.__dict__ = block.__dict__.copy()

        # Ensure DiffBlock-specific fields are always initialised
        obj.diff_vars = []
        obj.lag_vars = []
        obj.reformulated_vars = []
        obj.differential_eqs = []

        obj.__dict__.update(kwargs)  # add DiffBlock-specific fields
        return obj


# ----------------------------------------------------------------------------------------------------------------------
# Pre defined blocks
# ----------------------------------------------------------------------------------------------------------------------

def constant(item_name: str = "") -> Block:
    name: str = "const_"
    y = Var(name + item_name)
    param = Var("param_" + item_name)

    blk = Block(algebraic_vars=[y], algebraic_eqs=[y - param])
    blk.event_dict[param] = Const(0.0)
    blk.in_vars = dict()
    blk.out_vars[str(0)] = y
    blk.name = "const"
    return blk


def gain(item_name: str = "") -> Block:
    inputs = [Var("inp_num_" + item_name)]
    name: str = "gain"
    y = Var(name + item_name)
    gain_param = Var("gain_param_" + item_name)
    in_vars_dict: Dict[str, Var] = dict()
    for i, inpt in enumerate(inputs):
        in_vars_dict[str(i)] = inpt
    expr: Expr = gain_param * inputs[0]
    blk = Block(algebraic_vars=[y], algebraic_eqs=[y - expr])
    blk.event_dict[gain_param] = Const(0.0)
    blk.in_vars = in_vars_dict
    blk.out_vars[str(0)] = y
    blk.name = "gain"
    return blk


def variable(name: str = "variable_", vartype: str = "vartype_") -> Tuple[Var, Block]:
    y = Var(name)
    if vartype == 'state':
        blk = Block(state_vars=[y])
    else:
        blk = Block(algebraic_vars=[y])
    return y, blk


def equation(expr: str = 'expression', etype: str = "etype") -> Block:
    if etype == 'state':
        blk = Block(state_eqs=[make_symbolic(expr)])
    else:
        blk = Block(algebraic_eqs=[make_symbolic(expr)])
    return blk


def adder(item_name: str = "") -> Block:
    inputs = [Var("sum_in_1_" + item_name),
              Var("sum_in_2_" + item_name)]  # will not be specified if inputs can be more than 2
    y = Var("sum_out_" + item_name)
    in_vars_dict: Dict[str, Var] = dict()
    expr: Expr = inputs[0]
    for i, inpt in enumerate(inputs):
        in_vars_dict[str(i)] = inpt
        if i > 0:
            expr += inpt
    blk = Block(algebraic_vars=[y], algebraic_eqs=[y - expr])
    blk.in_vars = in_vars_dict
    blk.out_vars[str(0)] = y
    blk.name = "sum"
    return blk


def substract(item_name: str = "") -> Block:
    inputs = [Var("minuend_" + item_name), Var("subtrahend_" + item_name)]
    y = Var("difference_" + item_name)
    in_vars_dict: Dict[str, Var] = dict()
    expr: Expr = inputs[0] - inputs[1]
    for i, inpt in enumerate(inputs):
        in_vars_dict[str(i)] = inpt
    blk = Block(algebraic_vars=[y], algebraic_eqs=[y - expr])
    blk.in_vars = in_vars_dict
    blk.out_vars[str(0)] = y
    blk.name = "substraction"
    return blk


def product(item_name: str = "") -> Block:
    inputs = [Var("factor1_" + item_name),
              Var("factor2_" + item_name)]  # will not be specified if inputs can be more than 2
    y = Var("product_out_" + item_name)
    in_vars_dict: Dict[str, Var] = dict()
    expr: Expr = inputs[0] * inputs[1]
    for i, inpt in enumerate(inputs):
        in_vars_dict[str(i)] = inpt
    blk = Block(algebraic_vars=[y], algebraic_eqs=[y - expr])
    blk.in_vars = in_vars_dict
    blk.out_vars[str(0)] = y
    blk.name = "product"
    return blk


def divide(item_name: str = "") -> Block:
    inputs = [Var("divident_" + item_name),
              Var("divisor_" + item_name)]  # will not be specified if inputs can be more than 2
    y = Var("quotient_" + item_name)
    in_vars_dict: Dict[str, Var] = dict()
    expr: Expr = inputs[0] / inputs[1]
    for i, inpt in enumerate(inputs):
        in_vars_dict[str(i)] = inpt
    blk = Block(algebraic_vars=[y], algebraic_eqs=[y - expr])
    blk.in_vars = in_vars_dict
    blk.out_vars[str(0)] = y
    blk.name = "divide"
    return blk


def absolut(item_name: str = "") -> Block:
    inputs = [Var("inp_num_" + item_name)]  # will not be specified if inputs can be more than 2
    y = Var("absolut_" + item_name)
    in_vars_dict: Dict[str, Var] = dict()
    expr: Expr = abs(inputs[0])
    for i, inpt in enumerate(inputs):
        in_vars_dict[str(i)] = inpt
    blk = Block(algebraic_vars=[y], algebraic_eqs=[y - expr])
    blk.in_vars = in_vars_dict
    blk.out_vars[str(0)] = y
    blk.name = "abs"
    return blk


def integrator(u: Var | Const, name: str = "x") -> Tuple[Var, Block]:
    x = Var(name)
    blk = Block(state_vars=[x], state_eqs=[u])
    return x, blk


def pi_controller(err: Var, kp: float, ki: float, name: str = "pi") -> Block:
    up, blk_kp = gain(kp, err, f"{name}_up")
    ie, blk_int = integrator(err, f"{name}_int")
    ui, blk_ki = gain(ki, ie, f"{name}_ui")
    u, blk_sum = adder([up, ui], f"{name}_u")
    return Block(name="",
                 children=[blk_kp, blk_int, blk_ki, blk_sum],
                 in_vars=[err],
                 out_vars=[u])


class GenqecBuild:
    def __init__(self, name: str = ""):
        self.name: str = name

    def genqec(self) -> Block:
        """
         generator with quadratic saturation
        """
        # Inputs
        # Vm: Bus voltage module
        # Va: Bus voltage angle
        # Tm: mechanical torque (from governor)
        # Vf: excitation voltage (from exciter)

        inputs: List[Var] = [Var("Vm_" + self.name), Var("Va" + self.name), Var("Tm_" + self.name),
                             Var("Vf_" + self.name)]

        in_vars_dict: Dict[str, Var] = dict()
        for i, inpt in enumerate(inputs):
            in_vars_dict[str(i)] = inpt

        # ______________________________________________________________________________________
        #                                    variables
        # ______________________________________________________________________________________

        # Saturation factors
        Sat_d = Var('Sat_d' + self.name)
        Sat_q = Var('Sat_q' + self.name)

        # States
        delta = Var("delta" + self.name)  # rotor angle
        omega = Var("omega" + self.name)  # rotor electrical speed
        Eq1 = Var("Eq1" + self.name)  # internal emf behind Xd'
        Ed1 = Var("Ed1" + self.name)
        Eq2 = Var("Eq2" + self.name)
        Ed2 = Var("Ed2" + self.name)
        Eq_prime = Var("Eq_prime" + self.name)  # transient voltage q-axis
        Ed_prime = Var("Ed_prime" + self.name)  # transient voltage d-axis
        Eq_2prime = Var("Eq_2prime" + self.name)  # subtransient voltage q-axis
        Ed_2prime = Var("Ed_2prime" + self.name)  # subtransient voltage d-axis

        # Algebraic variables
        Id = Var("Id" + self.name)
        Iq = Var("Iq" + self.name)
        Vd = Var("Vd" + self.name)
        Vq = Var("Vq" + self.name)
        Psid = Var("Psid" + self.name)
        Psiq = Var("Psiq" + self.name)
        Te = Var("Te" + self.name)

        IRPu = Var("IRPu")

        # Saturated resistances
        Xd_sat = Var('Xd_sat' + self.name)
        Xq_sat = Var('Xq_sat' + self.name)
        Xd_prime_sat = Var('Xd_prime_sat' + self.name)
        Xq_prime_sat = Var('Xq_prime_sat' + self.name)
        Xd_2prime_sat = Var('Xd_2prime_sat' + self.name)
        Xq_2prime_sat = Var('Xq_2prime_sat' + self.name)
        Ed2_coef = Var('Ed2_coef' + self.name)
        Eq2_coef = Var('Eq2_coef' + self.name)

        # Vg = Var('Vg')
        # dg = Var('dg')
        Pg = Var('Pg' + self.name)
        Qg = Var('Qg' + self.name)

        Sa = Var('Sa')
        V_qag = Var('V_qag')
        V_dag = Var('V_dag')
        Psi_ag = Var('Psi_ag')

        # ______________________________________________________________________________________
        #                                    parameters
        # ______________________________________________________________________________________
        fn = Var('fn')  # system frequency [Hz]
        ws = Var('ws')  # synchronous speed [rad/s]
        M = Var('M')  # inertia constant
        D = Var('D')  # damping (optional)
        Rs = Var('Rs')  # stator resistance
        Ra = Var('Ra')  # armature resistance (if distinct)

        # Reactances
        Xd = Var('Xd')
        Xq = Var('Xq')
        Xd_prime = Var('Xd_prime')
        Xq_prime = Var('Xq_prime')
        Xd_2prime = Var('Xd_2prime')
        Xq_2prime = Var('Xq_2prime')
        Xl = Var('Xl')

        # Time constants
        Td0_prime = Var('Td0_prime')
        Tq0_prime = Var('Tq0_prime')
        Td0_2prime = Var('Td0_2prime')
        Tq0_2prime = Var('Tq0_2prime')

        A = Var('A')
        B = Var("B")

        event_dict = {
            fn: Const(50.0),
            ws: Const(1.0),
            M: Const(3.5),
            D: Const(10.0),
            Rs: Const(0.003),
            Ra: Const(0.003),

            # Reactances
            Xd: Const(1.8),
            Xq: Const(1.7),
            Xd_prime: Const(0.3),
            Xq_prime: Const(0.55),
            Xd_2prime: Const(0.25),
            Xq_2prime: Const(0.25),
            Xl: Const(0.15),

            # Time constants
            Td0_prime: Const(8.0),
            Tq0_prime: Const(0.4),
            Td0_2prime: Const(0.03),
            Tq0_2prime: Const(0.05),

            A: Const(5.0),
            B: Const(1.0)
        }

        generator_block = Block(
            state_eqs=[
                (omega - Const(1)) * ws,  # dδ/dt
                (inputs[2] - Te - D * (omega - Const(1))) * (1 / M),  # dω/dt
                inputs[3] / Td0_prime - Sat_q * Eq1 / Td0_prime,  # dEq'/dt
                -Sat_q * Ed1 / Tq0_prime,  # dEd'/dt
                Eq2_coef * (-Eq2),  # dEq''/dt
                Ed2_coef * (-Ed2),  # dEd''/dt
            ],
            state_vars=[delta, omega, Eq_prime, Ed_prime, Eq_2prime, Ed_2prime],
            algebraic_eqs=[
                Vd - (-inputs[2] * sin(inputs[3] - delta)),  # from input block
                Vq - (inputs[2] * cos(inputs[3] - delta)),  # from input block
                Pg - (Vd * Id + Vq * Iq),  # from input block
                Qg - (Vq * Id - Vd * Iq),  # from input block
                Vd - (omega * Ed_2prime + Iq * Xq_2prime_sat - Id * Ra),
                Vq - (omega * Eq_2prime - Id * Xd_2prime_sat - Iq * Ra),
                Psid - (Eq_prime - Id * Xd_2prime_sat),
                Psiq - (-Ed_prime - Iq * Xq_2prime_sat),
                Te - (Psid * Iq - Psiq * Id),
                (Xd_sat - Xd_prime_sat) * Eq1 - (
                        (Xd_sat - Xd_prime_sat) * Eq_2prime + (Eq_prime - Eq_2prime) * (Xd_sat - Xd_2prime_sat)),
                # Eq1 definition
                (Xq_sat - Xq_prime_sat) * Ed1 - (
                        (Xq_sat - Xq_prime_sat) * Ed_2prime + (Ed_prime - Ed_2prime) * (Xq_sat - Xq_2prime_sat)),
                # Ed1 definition
                (Xd_sat - Xd_2prime_sat) * Eq2 - (-1) * (
                        (Eq_prime - Eq_2prime) * (Xd_sat - Xd_prime_sat) + Id * (Xd_sat - Xd_2prime_sat) ** 2),
                # Ed2 definition
                (Xq_sat - Xq_2prime_sat) * Ed2 - (-1) * (
                        (Ed_prime - Ed_2prime) * (Xq_sat - Xq_prime_sat) + Iq * (Xq_sat - Xq_2prime_sat) ** 2),
                # Eq2 definition
                Eq2_coef * (Tq0_2prime * (Xq_sat - Xq_2prime_sat)) - (Xq_prime_sat - Xq_2prime_sat) * Sat_q,
                Ed2_coef * (Td0_2prime * (Xd_sat - Xd_2prime_sat)) - (Xd_prime_sat - Xd_2prime_sat) * Sat_d,
                # saturated resistance
                Sat_d * Xd_2prime_sat - (Xd_2prime - Xl + Xl * Sat_d),
                Sat_q * Xq_2prime_sat - (Xq_2prime - Xl + Xl * Sat_q),
                Sat_d * Xd_prime_sat - (Xd_prime - Xl + Xl * Sat_d),
                Sat_q * Xq_prime_sat - (Xq_prime - Xl + Xl * Sat_q),
                Sat_d * Xd_sat - (Xd - Xl + Xl * Sat_d),
                Sat_q * Xq_sat - (Xq - Xl + Xl * Sat_q),
                # flux
                V_dag - (Vd - Ra * Id + Xq_2prime_sat * Iq + Iq * Ra + Id * Xl),
                V_qag - (Vq - Ra * Iq + Xd_2prime_sat * Id + Id * Ra - Iq * Xl),
                (omega) * Psi_ag - (ws) * sqrt(V_qag * V_qag + V_dag * V_dag),
                Sat_d - (Const(1) + Sa),
                Sat_q - (Const(1) + Sa),
                # saturations (quadratic)
                Sa - A * ((Psi_ag - B) + sqrt((Psi_ag - B) ** 2 + Const(1e-4))),
                IRPu - Eq1 * (1 + Sa)
            ],
            algebraic_vars=[Vd, Vq, Psid, Psiq, Ed2_coef, Eq2_coef, Te, Ed1, Eq1, Ed2, Eq2, Id, Iq,
                            # saturated resistance
                            Xq_sat, Xd_sat, Xq_prime_sat, Xd_prime_sat, Xq_2prime_sat, Xd_2prime_sat,
                            # flux
                            Sa, V_dag, V_qag, Sat_d, Sat_q, Psi_ag,
                            IRPu],
            init_eqs={
                omega: Const(0),
                V_dag: inputs[0] * sin(inputs[1]) + imag(
                    conj(Pg + 1j * Qg) / conj(inputs[0] * exp(1j * inputs[1]))) * Ra + real(
                    conj(Pg + 1j * Qg) / conj(inputs[0] * exp(1j * inputs[1]))) * Xl,
                V_qag: inputs[0] * cos(inputs[1]) + real(
                    conj(Pg + 1j * Qg) / conj(inputs[0] * exp(1j * inputs[1]))) * Ra + imag(
                    conj(Pg + 1j * Qg) / conj(inputs[0] * exp(1j * inputs[1]))) * Xl,
                Psi_ag: sqrt(V_dag ** 2 + V_qag ** 2),
                Sa: A * (Psi_ag - B) ** 2,
                delta: atan(
                    (inputs[0] * sin(inputs[1]) + imag(
                        conj(Pg + 1j * Qg) / conj(inputs[0] * exp(1j * inputs[1]))) * Ra + real(
                        conj(Pg + 1j * Qg) / conj(inputs[0] * exp(1j * inputs[1]))) *
                     ((Xq - Xl) / (Const(1) + Sa) + Xl)) /
                    (inputs[0] * cos(inputs[1]) + real(
                        conj(Pg + 1j * Qg) / conj(inputs[0] * exp(1j * inputs[1]))) * Ra + imag(
                        conj(Pg + 1j * Qg) / conj(inputs[0] * exp(1j * inputs[1]))) *
                     ((Xq - Xl) / (Const(1) + Sa) + Xl))),
                Vd: (inputs[0] * sin(delta - inputs[1])),
                Vq: (inputs[0] * cos(delta - inputs[1])),
                Id: real(conj(Pg + 1j * Qg) / conj(inputs[0] * exp(1j * inputs[1]))) * sin(delta) - real(
                    conj(Pg + 1j * Qg) / conj(inputs[0] * exp(1j * inputs[1]))) * cos(delta),
                Iq: real(conj(Pg + 1j * Qg) / conj(inputs[0] * exp(1j * inputs[1]))) * cos(delta) + real(
                    conj(Pg + 1j * Qg) / conj(inputs[0] * exp(1j * inputs[1]))) * sin(delta),
                Ed_prime: (Xq - Xq_prime) * Iq / (Const(1) + Sa),
                Eq_prime: Ed_prime + (Xq_prime - Xl) * (Iq / (Const(1) + Sa)) + (Xq_prime - Xl) * (
                        Iq / (Const(1) + Sa)),
                Eq_2prime: Vd,
                Ed_2prime: Vq,
            })

        generator_block.external_mapping = {
            VarPowerFlowRefferenceType.P: Pg,
            VarPowerFlowRefferenceType.Q: Qg,
        }
        generator_block.name = "genqec"
        generator_block.event_dict = event_dict
        generator_block.in_vars = in_vars_dict
        generator_block.out_vars[str(0)] = Pg
        generator_block.out_vars[str(1)] = Qg
        generator_block.out_vars[str(2)] = omega
        generator_block.out_vars[str(3)] = IRPu

        return generator_block


class GovernorBuild:
    def __init__(self, name: str = ""):
        self.name: str = name

        self.parameters = {
            # Time constants
            "T1": Const(1.0),  # governor time constant (s)
            "T2": Const(1.0),  # reheater time constant (s)
            "T3": Const(10.0),  # crossover time constant (s)
            "T4": Const(0.2),  # lead/lag constant (s)
            "T5": Const(0.5),  # lead/lag constant (s)
            "T6": Const(0.1),  # lead/lag constant (s)
            "T7": Const(0.05),  # lead/lag constant (s)

            # Steam fractions (distribution factors)
            "K1": Const(0.5),
            "K2": Const(0.5),
            "K3": Const(0.0),
            "K4": Const(0.0),
            "K5": Const(0.0),
            "K6": Const(0.0),
            "K7": Const(0.0),
            "K8": Const(0.0),
        }

    def governor(self) -> Block:
        """
            governor model with no load control and with no free reference
        """

        # Inputs
        # omega: omega (from generator)

        inputs = [Var("omega_" + self.name)]

        in_vars_dict: Dict[str, Var] = dict()
        for i, inpt in enumerate(inputs):
            in_vars_dict[str(i)] = inpt

        # ______________________________________________________________________________________
        #                                    variables
        # ______________________________________________________________________________________

        Tm = Var("Tm")  # Mechanical power input (pu
        et = Var("et")
        algebraic_eqs = []
        algebraic_vars = []

        # ______________________________________________________________________________________
        #                                    parameters
        # ______________________________________________________________________________________

        # Time constants
        T1 = Const(1.0)  # governor time constant (s)
        T2 = Const(1.0)  # reheater time constant (s)
        T3 = Const(10.0)  # crossover time constant (s)
        T4 = Const(0.2)  # lead/lag constant (s)
        T5 = Const(0.5)  # lead/lag constant (s)
        T6 = Const(0.1)  # lead/lag constant (s)
        T7 = Const(0.05)  # lead/lag constant (s)

        # Steam fractions (distribution factors)
        K1 = Const(0.5)
        K2 = Const(0.5)
        K3 = Const(0.0)
        K4 = Const(0.0)
        K5 = Const(0.0)
        K6 = Const(0.0)
        K7 = Const(0.0)
        K8 = Const(0.0)

        # Gains and limits
        K = Var("K")  # governor gain (inverse droop)
        Pmax = Var("Pmax")  # max mechanical power (pu)
        Pmin = Var("Pmin")  # min mechanical power (pu)
        Uc = Var("Uc")  # max valve closing rate (pu/s)
        Uo = Var("Uo")  # max valve opening rate (pu/s)
        T_aux = Var("T_aux")

        # Control
        Kp = Var("Kp")
        Ki = Var("Ki")
        omega_ref = Var('omega_ref')
        p0 = Var('p0')
        P0 = Var('P0')

        # reference
        Pm_ref = Var('Pm_ref')

        events_dict = {
            # control parameters
            Kp: Const(-0.01),
            Ki: Const(-0.01),
            p0: Const(1.0),
            P0: Const(0.01),
            omega_ref: Const(1),

            # Governor parameters
            K: Const(10.0),  # governor gain (inverse droop)
            Pmax: Const(2.0),  # max mechanical power (pu)
            Pmin: Const(0.0),  # min mechanical power (pu)
            Uc: Const(-0.1),  # max valve closing rate (pu/s)
            Uo: Const(0.1),  # max valve opening rate (pu/s)
            T_aux: Const(0.0),

            # reference
            Pm_ref: Const(0.0963287180659294)
        }
        controller_block = Block(
            state_eqs=[
                P0 * (inputs[0] - omega_ref)
            ],
            state_vars=[et],
            algebraic_eqs=[
                T_aux - (Kp * (inputs[0] - omega_ref) + Ki * et),
            ],
            algebraic_vars=[T_aux],
        )

        u1 = inputs[0] - omega_ref
        lead_lag_block, y1 = tf_to_diffblock(
            num=np.array([1, self.parameters["T2"]]),
            den=np.array([1, self.parameters["T1"]]),
            x=u1,
            name='gov0',
        )

        # ==============================
        # First Feed back Loop
        y2_3 = Var('y2_3_gov')
        algebraic_vars.append(y2_3)
        x2 = Pm_ref - K * y1 - y2_3

        y2 = x2 * (1 / self.parameters["T3"])
        y2_1 = hard_sat(y2, Uc, Uo)
        tf1, y2_2 = tf_to_diffblock(
            num=np.array([1]),
            den=np.array([0, 1]),
            x=y2_1,
            name='gov1',
        )
        algebraic_eqs.append(y2_3 - hard_sat(y2_2, Pmin, Pmax))

        # ==============================
        # We compute different outputs for every tf
        tf2, y3_1 = tf_to_diffblock(
            num=np.array([1]),
            den=np.array([1, self.parameters["T4"]]),
            x=y2_3,
            name='gov2',
        )
        tf3, y3_2 = tf_to_diffblock(
            num=np.array([1]),
            den=np.array([1, self.parameters["T5"]]),
            x=y3_1,
            name='gov3',
        )
        tf4, y3_3 = tf_to_diffblock(
            num=np.array([1]),
            den=np.array([1, self.parameters["T6"]]),
            x=y3_2,
            name='gov4',
        )
        tf5, y3_4 = tf_to_diffblock(
            num=np.array([1]),
            den=np.array([1, self.parameters["T7"]]),
            x=y3_3,
            name='gov5',
        )

        u = self.parameters["K1"] * y3_1 + self.parameters["K2"] * y3_2 + self.parameters["K3"] * y3_3 + \
            self.parameters["K4"] * y3_4 + T_aux
        aux_block = Block(
            algebraic_eqs=[u - Tm] + algebraic_eqs,
            algebraic_vars=[Tm] + algebraic_vars,
        )

        governor_block = Block(
            children=[lead_lag_block, tf1, tf2, tf3, tf4, tf5, aux_block, controller_block]
        )

        governor_block.name = "governor"

        governor_block.event_dict = events_dict

        governor_block.parameters = self.parameters

        governor_block.out_vars[str(0)] = Tm

        governor_block.in_vars = in_vars_dict

        return governor_block


class StabilizerBuild:
    def __init__(self, name: str = ""):
        self.name: str = name

        self.parameters = {
            # Stabilizer parameters
            "A1": Const(1.0),  # notch filter coefficient 1
            "A2": Const(1.0),  # notch filter coefficient 2
            "t1": Const(0.1),  # lead time constant
            "t2": Const(0.02),  # lag time constant
            "t3": Const(0.02),  # lag time constant
            "t4": Const(0.1),  # second lag time constant
            "t5": Const(10.0),  # washout time constant
            "t6": Const(0.02),  # transducer time constant
        }

    def stabilizer(self):
        """
           stabilizer model
        """

        # input variables
        # omega: omega from generator

        inputs = [Var("omega_" + self.name)]

        in_vars_dict: Dict[str, Var] = dict()
        for i, inpt in enumerate(inputs):
            in_vars_dict[str(i)] = inpt

        # PSS parameters with typical values

        Ks = Var("Ks")  # stabilizer gain
        VPssMaxPu = Var("VPssMaxPu")  # max stabilizer output
        VPssMinPu = Var("VPssMinPu")  # min stabilizer output
        SNom = Var("SNom")  # nominal apparent power

        events_dict = {
            # Stabilizer parameters
            Ks: Const(20.0),  # stabilizer gain
            VPssMaxPu: Const(1.0),  # max stabilizer output
            VPssMinPu: Const(-1.0),  # min stabilizer output
            SNom: Const(1.0),  # nominal apparent power
        }

        # variables
        Vpss = Var('V_pss')

        vars_block = Block(
            algebraic_vars=[],
        )

        tf, y = tf_to_diffblock_with_states(
            num=np.array([1.0]),
            den=np.array([1, self.parameters["t6"]]),
            x=inputs[0],
            name='stabilizer1',
        )

        tf2, y2 = tf_to_diffblock_with_states(
            num=np.array([0, self.parameters["t5"]]),
            den=np.array([1, self.parameters["t5"]]),
            x=Ks * y,
            name='stabilizer2',
        )
        tf3, y3 = tf_to_diffblock_with_states(
            num=np.array([1]),
            den=np.array([1, self.parameters["A1"], self.parameters["A2"]]),
            x=y2,
            name='stabilizer3',
        )
        tf4, y4 = tf_to_diffblock_with_states(
            num=np.array([1, self.parameters["t1"]]),
            den=np.array([1, self.parameters["t2"]]),
            x=y3,
            name='stabilizer4',
        )
        tf5, y5 = tf_to_diffblock_with_states(
            num=np.array([1, self.parameters["t3"]]),
            den=np.array([1, self.parameters["t4"]]),
            x=y4,
            name='stabilizer5',
        )

        algebraic_eqs = list()
        algebraic_eqs.append(hard_sat(y5, VPssMinPu, VPssMaxPu) - Vpss)
        block_1 = Block()

        stabilizer_block = Block(
            children=[tf, tf2, tf3, tf4, tf5],
            algebraic_eqs=algebraic_eqs,
            algebraic_vars=[Vpss],
        )

        stabilizer_block.in_vars = in_vars_dict
        stabilizer_block.add(vars_block)
        stabilizer_block.add(block_1)
        stabilizer_block.out_vars[str(0)] = Vpss
        stabilizer_block.event_dict = events_dict
        stabilizer_block.parameters = self.parameters
        stabilizer_block.name = "stabilizer"

        return stabilizer_block


class ExciterBuild:
    def __init__(self, name: str = ""):
        self.name: str = name

        self.parameters = {
            # Exciter (AVR) parameters
            "Ka": Const(200.0),  # AVR gain
            "Kf": Const(0.03),  # exciter rate feedback gain

            # Time constants
            "tA": Const(0.02),  # AVR time constant (s)
            "tB": Const(10.0),  # lead-lag: lag time constant (s)
            "tC": Const(1.0),  # lead-lag: lead time constant (s)
            "tE": Const(0.5),  # exciter field time constant (s)
            "tF": Const(1.0),  # rate feedback time constant (s)
            "tR": Const(0.02),  # stator voltage filter time constant (s)

            # Exciter submodel parameters
            "Kc": Const(0.2),  # rectifier loading factor
            "Kd": Const(0.1),  # demagnetizing factor
            "Ke": Const(1.0),  # field resistance constant

        }

    def exciter(self):
        """
        exciter model
        """

        # input variables
        # IRPu: rotor current (pu) ???
        # Va: measured stator voltage (from generator) (pu)
        # Vpss: output from power system stabilizer (pu)

        inputs = [Var("IRPu_" + self.name), Var("Va_" + self.name), Var("Vpss_" + self.name)]

        in_vars_dict: Dict[str, Var] = dict()
        for i, inpt in enumerate(inputs):
            in_vars_dict[str(i)] = inpt

        algebraic_vars = []

        # ______________________________________________________________________________________
        #                                    variables
        # ______________________________________________________________________________________

        Vf = Var("Vf")
        Efe = Var('Efe')

        # Exciter internal variables
        VeMaxPu = Var('VeMaxPu')
        u_aux = Var('u_aux')

        # ______________________________________________________________________________________
        #                                    parameters
        # ______________________________________________________________________________________

        # ---- Exciter (AVR) parameters ----
        UsRefPu = Var("UsRefPu")  # reference voltage (pu)
        AEz = Var("AEz")  # saturation gain
        BEz = Var("BEz")  # saturation exponential coefficient
        EfeMaxPu = Var("EfeMaxPu")  # max exciter field voltage (pu)
        EfeMinPu = Var("EfeMinPu")  # min exciter field voltage (pu)

        # ---- Exciter (AVR) time constants and limits ----

        TolLi = Var("TolLi")  # limiter crossing tolerance (fraction)

        VaMaxPu = Var("VaMaxPu")  # AVR output max (pu)
        VaMinPu = Var("VaMinPu")  # AVR output min (pu)
        VeMinPu = Var("VeMinPu")  # min exciter output voltage (pu)
        VfeMaxPu = Var("VfeMaxPu")  # max exciter field current signal (pu)

        # exciter submodel parameters
        AEx = Var("AEx")  # Gain of saturation function
        BEx = Var("BEx")  # Exponential coefficient of saturation function
        ToLLi = Var("ToLLi")  # Tolerance on limit crossing
        VeMinPu_submodel = Var("VeMinPu_submodel")  # Minimum exciter output voltage (pu)
        VfeMaxPu_submodel = Var("VfeMaxPu_submodel")  # Maximum exciter field current signal (pu)
        F_rectifier = Var("F_rectifier")  # Rectifier factor

        events_dict = {
            # Exciter (AVR) parameters
            UsRefPu: Const(1.0),  # reference voltage (pu)
            AEz: Const(0.02),  # saturation gain
            BEz: Const(1.5),  # saturation exponential coefficient
            EfeMaxPu: Const(15.0),  # max exciter field voltage (pu)
            EfeMinPu: Const(-5.0),  # min exciter field voltage (pu)

            # Time constants
            TolLi: Const(0.05),  # limiter crossing tolerance (fraction)

            # Limits
            VaMaxPu: Const(20.0),  # AVR output max (pu)
            VaMinPu: Const(-10.0),  # AVR output min (pu)
            VeMinPu: Const(-1.0),  # min exciter output voltage (pu)
            VfeMaxPu: Const(5.0),  # max exciter field current signal (pu)

            # Exciter submodel parameters
            AEx: Const(0.02),  # saturation gain
            BEx: Const(-0.01),  # exponential coeff of saturation function
            ToLLi: Const(0.05),  # tolerance on limit crossing
            VeMinPu_submodel: Const(-0.1),  # minimum exciter output voltage
            VfeMaxPu_submodel: Const(5.0),  # max exciter field current signal
            F_rectifier: Const(1.0),  # rectifier factor (1=DC, 0.5=AC)
        }

        # ---Internal Blocks---
        tf1, y1 = tf_to_diffblock(
            num=np.array([1]),
            den=np.array([1, self.parameters["tR"]]),
            x=inputs[1],
            name='exciter1',
        )  # filtered stator voltage

        # error1 = UPssPu - y + UsRefPu
        error1 = (- y1 + UsRefPu) + inputs[2]
        tf2, y2 = tf_to_diffblock(
            num=np.array([0, self.parameters["Kf"]]),
            den=np.array([1, self.parameters["tF"]]),
            x=Vf,
            name='exciter2',
        )
        error2 = error1 - y2

        tf3, y3 = tf_to_diffblock(
            num=np.array([1, self.parameters["tC"]]),
            den=np.array([1, self.parameters["tB"]]),
            x=error2,
            name='exciter3',
        )
        tf4, y4 = tf_to_diffblock(
            num=np.array([self.parameters["Ka"]]),
            den=np.array([1, self.parameters["tA"]]),
            x=y3,
            name='exciter4',
        )

        y5 = hard_sat(y4, VaMinPu, VaMaxPu)

        min_const = Const(max(events_dict[VaMinPu], events_dict[EfeMinPu]))
        max_const = Const(min(events_dict[VaMaxPu], events_dict[EfeMaxPu]))
        y6 = hard_sat(y4, min_const, max_const)

        # exciter submodel

        algebraic_eqs_submodel = []
        algebraic_vars_submodel = []

        x1 = VfeMaxPu - inputs[0] * self.parameters["Kd"]
        error1 = Efe - (inputs[0] * self.parameters["Kd"] + u_aux)

        tf1, Ve_presat = tf_to_diffblock(
            num=np.array([1]),
            den=np.array([0, self.parameters["tE"]]),
            x=error1,
            name='subexciter1',
        )

        Ve = hard_sat(Ve_presat, VeMinPu, Const(1000))
        aux_expr = self.parameters["Ke"] * Ve + AEx * Ve * exp(BEx * Ve)
        algebraic_eqs_submodel.append(u_aux - aux_expr)
        algebraic_eqs_submodel.append(VeMaxPu * u_aux - x1)

        f_input = Var('f_input')
        f_output = f_exc(f_input)
        algebraic_vars_submodel.append(f_input)
        algebraic_eqs_submodel.append(f_input * Ve - inputs[0] * self.parameters["Kc"])

        algebraic_eqs_submodel.append(Vf - f_output * Ve)

        aux_model = DiffBlock(
            algebraic_eqs=algebraic_eqs_submodel,
            algebraic_vars=[u_aux, VeMaxPu, Vf] + algebraic_vars_submodel
        )

        exciter_submodel = DiffBlock(children=[tf1, aux_model])

        linking_block = Block(
            algebraic_eqs=[y6 - Efe],
            algebraic_vars=[Efe] + algebraic_vars,

        )

        exciter_model = Block(children=[tf1, tf2, tf3, tf4, exciter_submodel, linking_block])
        exciter_model.out_vars[str(0)] = Vf
        exciter_model.in_vars = in_vars_dict

        return exciter_model


def exciter(item_name: str = "") -> Block:
    vf = Var("vf" + item_name)
    blk = Block()
    blk.in_vars = dict()
    blk.out_vars[str(0)] = vf
    blk.name = "const"
    return blk


def generator(name: str = "") -> Block:
    inputs = [Var("Vm_" + name), Var("Va_" + name)]

    in_vars_dict: Dict[str, Var] = dict()
    for i, inpt in enumerate(inputs):
        in_vars_dict[str(i)] = inpt

    delta = Var("delta")
    omega = Var("omega")
    psid = Var("psid")
    psiq = Var("psiq")
    i_d = Var("i_d")
    i_q = Var("i_q")
    v_d = Var("v_d")
    v_q = Var("v_q")
    te = Var("te")
    et = Var("et")
    tm = Var("tm")
    P_g = Var("P_g")
    Q_g = Var("Q_g")

    R1 = Var("R1")
    X1 = Var("X1")
    freq = Var("frequ")
    M = Var("M")
    D = Var("D")
    omega_ref = Var("omega_ref")
    Kp = Var("Kp")
    Ki = Var("Ki")

    # vf = UndefinedConst() # this will disappear when the generator and the exciter model are decoupled
    # vf.name ="vf" # this will disappear when the generator and the exciter model are decoupled
    # tm0 = UndefinedConst() # this will disappear when the generator and the exciter model are decoupled
    # tm0.name = "tm0" # this will disappear when the generator and the exciter model are decoupled

    vf = Var("Vf")
    tm0 = Var("tm0")
    event_dict = {R1: Const(0.0),
                  X1: Const(0.3),
                  freq: Const(60.0),
                  M: Const(4.0),
                  D: Const(1.0),
                  omega_ref: Const(1.0),
                  Kp: Const(0.0),
                  Ki: Const(0.0)}

    gen_block = Block(
        state_eqs=[
            (2 * np.pi * freq) * (omega - omega_ref),
            (tm - te - D * (omega - omega_ref)) / M,
        ],
        state_vars=[delta, omega],
        algebraic_eqs=[
            psid - (R1 * i_q + v_q),
            psiq + (R1 * i_d + v_d),
            0 - (psid + X1 * i_d - inputs[1]),
            0 - (psiq + X1 * i_q),
            v_d - (inputs[0] * sin(delta - inputs[1])),
            v_q - (inputs[0] * cos(delta - inputs[1])),
            te - (psid * i_q - psiq * i_d),
            P_g - (v_d * i_d + v_q * i_q),
            Q_g - (v_q * i_d - v_d * i_q),
            tm - (tm0 + Kp * (omega - omega_ref) + Ki * et),
            2 * np.pi * freq * et - delta,  #
        ],
        algebraic_vars=[P_g, Q_g, v_d, v_q, i_d, i_q, psid, psiq, te, tm, et],
        init_eqs={
            delta: imag(
                log((inputs[0] * exp(1j * inputs[1]) + (R1 + 1j * X1) * (
                    conj((P_g + 1j * Q_g) / (inputs[0] * exp(1j * inputs[1]))))) / (
                        abs(inputs[0] * exp(1j * inputs[1]) + (R1 + 1j * X1) * (
                            conj((P_g + 1j * Q_g) / (inputs[0] * exp(1j * inputs[1])))))))),
            omega: omega_ref,
            v_d: real((inputs[0] * exp(1j * inputs[1])) * exp(-1j * (delta - np.pi / 2))),
            v_q: imag((inputs[0] * exp(1j * inputs[1])) * exp(-1j * (delta - np.pi / 2))),
            i_d: real(conj((P_g + 1j * Q_g) / (inputs[0] * exp(1j * inputs[1]))) * exp(-1j * (delta - np.pi / 2))),
            i_q: imag(conj((P_g + 1j * Q_g) / (inputs[0] * exp(1j * inputs[1]))) * exp(-1j * (delta - np.pi / 2))),
            psid: R1 * i_q + v_q,
            psiq: -R1 * i_d - v_d,
            te: psid * i_q - psiq * i_d,
            tm: te,
            et: Const(0),
        },
    )
    # gen_block.fix_vars = [tm0, vf]
    # gen_block.fix_vars_eqs = {tm0.uid: tm,
    #                       vf.uid: psid + X1 * i_d}

    gen_block.external_mapping = {
        VarPowerFlowRefferenceType.P: P_g,
        VarPowerFlowRefferenceType.Q: Q_g
    }

    gen_block.event_dict = event_dict
    gen_block.in_vars = in_vars_dict
    gen_block.out_vars[str(0)] = P_g
    gen_block.out_vars[str(1)] = Q_g
    return gen_block


def load(name: str = "load") -> Block:
    Ql = Var("Ql")
    Pl = Var("Pl")

    load_block = Block(
        algebraic_eqs=[
            # Pl - load_object.Pl0,
            # Ql - load_object.Ql0
        ],
        algebraic_vars=[Pl, Ql],
        init_eqs={})

    load_block.external_mapping = {
        VarPowerFlowRefferenceType.P: Pl,
        VarPowerFlowRefferenceType.Q: Ql
    }
    return load_block


def line(item_name: str, api_object: Any) -> Block:
    line_object = api_object
    Qf = Var("Qf_" + item_name)
    Qt = Var("Qt_" + item_name)
    Pf = Var("Pf_" + item_name)
    Pt = Var("Pt_" + item_name)

    g = Const(1.0 / complex(line_object.R, line_object.X).real)
    b = Const(1.0 / complex(line_object.R, line_object.X).imag)
    bsh = Const(line_object.B)

    Vmf = line_object.bus_from.rms_model.model.E(VarPowerFlowRefferenceType.Vm)
    Vaf = line_object.bus_from.rms_model.model.E(VarPowerFlowRefferenceType.Va)
    Vmt = line_object.bus_to.rms_model.model.E(VarPowerFlowRefferenceType.Vm)
    Vat = line_object.bus_to.rms_model.model.E(VarPowerFlowRefferenceType.Va)

    line_block = Block(
        algebraic_eqs=[
            Pf - ((Vmf ** 2 * g) - g * Vmf * Vmt * cos(Vaf - Vat) + b * Vmf * Vmt * cos(Vaf - Vat + np.pi / 2)),
            Qf - (Vmf ** 2 * (-bsh / 2 - b) - g * Vmf * Vmt * sin(Vaf - Vat) + b * Vmf * Vmt * sin(
                Vaf - Vat + np.pi / 2)),
            Pt - ((Vmt ** 2 * g) - g * Vmt * Vmf * cos(Vat - Vaf) + b * Vmt * Vmf * cos(Vat - Vaf + np.pi / 2)),
            Qt - (Vmt ** 2 * (-bsh / 2 - b) - g * Vmt * Vmf * sin(Vat - Vaf) + b * Vmt * Vmf * sin(
                Vat - Vaf + np.pi / 2)),
        ],
        algebraic_vars=[Pf, Pt, Qf, Qt],
        init_eqs={},
        parameters=[])
    line_block.external_mapping = {
        VarPowerFlowRefferenceType.Pf: Pf,
        VarPowerFlowRefferenceType.Pt: Pt,
        VarPowerFlowRefferenceType.Qf: Qf,
        VarPowerFlowRefferenceType.Qt: Qt,
    }
    return line_block


def generic(state_inputs: int, state_outputs: Sequence[str], algebraic_inputs: int,
            algebraic_outputs: Sequence[str]) -> Block:
    blk = Block()
    blk.name = "generic"
    input_vars = [Var(f"Vport{i}") for i in range(state_inputs + algebraic_inputs)]

    for i, v in enumerate(state_outputs):
        var = Var(v)
        blk.state_vars.append(var)
        blk.out_vars[str(i)] = var

    for i, v in enumerate(algebraic_outputs):
        var = Var(v)
        blk.algebraic_vars.append(var)
        blk.out_vars[str(i)] = var

    in_vars_dict: Dict[str, Var] = dict()
    for i, inpt in enumerate(input_vars):
        in_vars_dict[str(i)] = inpt
    blk.in_vars = in_vars_dict

    return blk
