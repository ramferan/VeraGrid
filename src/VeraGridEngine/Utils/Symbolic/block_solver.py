# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

import pdb
import sys
import os
import time

import cProfile as profile

pr = profile.Profile()
pr.disable()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..", "src")))

from typing import Tuple, Any
import pandas as pd
import numpy as np
import numba as nb
import math
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import gmres, spilu, LinearOperator, spsolve, eigs
from typing import Dict, List, Literal, Any, Callable, Sequence
from matplotlib import pyplot as plt

from VeraGridEngine.Devices.Dynamic.events import RmsEvents
from VeraGridEngine.Utils.Symbolic.symbolic import Var, Expr, Const, _emit, _emit_params_eq, _heaviside
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Sparse.csc import pack_4_by_4_scipy
from VeraGridEngine.basic_structures import Vec


def _fully_substitute(expr: Expr, mapping: Dict[Var, Expr], max_iter: int = 10) -> Expr:
    cur = expr
    for _ in range(max_iter):
        nxt = cur.subs(mapping).simplify()
        if str(nxt) == str(cur):  # no further change
            break
        cur = nxt
    return cur


def _compile_equations(eqs: Sequence[Expr],
                       uid2sym_vars: Dict[int, str],
                       uid2sym_params: Dict[int, str],
                       add_doc_string: bool = True) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Compile the array of expressions to a function that returns an array of values for those expressions
    :param eqs: Iterable of expressions (Expr)
    :param uid2sym_vars: dictionary relating the uid of a var with its array name (i.e. var[0])
    :param uid2sym_params:
    :param add_doc_string: add the docstring?
    :return: Function pointer that returns an array
    """
    # Build source
    src = f"def _f(vars, params):\n"
    src += f"    out = np.zeros({len(eqs)})\n"
    src += "\n".join([f"    out[{i}] = {_emit(e, uid2sym_vars, uid2sym_params)}" for i, e in enumerate(eqs)]) + "\n"
    src += f"    return out"
    ns: Dict[str, Any] = {"math": math, "np": np}
    exec(src, ns)
    fn = nb.njit(ns["_f"], fastmath=True)

    if add_doc_string:
        fn.__doc__ = "def _f(vars)"
    return fn


def _compile_parameters_equations(eqs: Sequence[Expr],
                                  uid2sym_t: Dict[int, str],
                                  add_doc_string: bool = True) -> Callable[[float], np.ndarray]:
    """
    Compile the array of expressions to a function that returns an array of values for those expressions
    :param eqs: Iterable of expressions (Expr)
    :param uid2sym_vars: dictionary relating the uid of a var with its array name (i.e. var[0])
    :param uid2sym_params:
    :param add_doc_string: add the docstring?
    :return: Function pointer that returns an array
    """
    # Build source
    src = f"def _f(glob_time):\n"
    src += f"    out = np.zeros({len(eqs)})\n"
    src += "\n".join([f"    out[{i}] = {_emit_params_eq(e, uid2sym_t)}" for i, e in enumerate(eqs)]) + "\n"
    src += f"    return out"
    ns: Dict[str, Any] = {"math": math, "np": np, "_heaviside": _heaviside}
    exec(src, ns)
    fn = nb.njit(ns["_f"], fastmath=True)
    if add_doc_string:
        fn.__doc__ = "def _f(vars)"
    return fn


# full jacobian
def _get_full_jacobian(eqs: List[Expr],
                       variables: List[Var],
                       uid2sym_vars: Dict[int, str],
                       uid2sym_params: Dict[int, str], ):
    """
    JIT‑compile a sparse Jacobian evaluator for *equations* w.r.t *variables*.
    :param eqs: Array of equations
    :param variables: Array of variables to differentiate against
    :param uid2sym_vars: dictionary relating the uid of a var with its array name (i.e. var[0])
    :param uid2sym_params:
    :return:
            jac_fn : callable(values: np.ndarray, params: np.ndarray) -> scipy.sparse.csc_matrix
                Fast evaluator in which *values* is a 1‑D NumPy vector of length
                ``len(variables)``and *params* is a 1‑D NumPy vector of length
                ``len(parameters)``
            sparsity_pattern : tuple(np.ndarray, np.ndarray)
                Row/col indices of structurally non‑zero entries.
    """

    # Ensure deterministic variable order
    check_set = set()
    for v in variables:
        if v in check_set:
            raise ValueError(f"Repeated var {v.name} in the variables' list :(")
        else:
            check_set.add(v)

    triplets: List[Tuple[int, int, Callable]] = []  # (col, row, fn)

    for row, eq in enumerate(eqs):
        for col, var in enumerate(variables):
            d_expression = eq.diff(var).simplify()
            if isinstance(d_expression, Const) and d_expression.value == 0:
                continue  # structural zero
            triplets.append((col, row, d_expression))

    triplets.sort(key=lambda t: (t[0], t[1]))
    cols_sorted, rows_sorted, equations_sorted = zip(*triplets) if triplets else ([], [], [])
    functions_ptr = _compile_equations(eqs=equations_sorted, uid2sym_vars=uid2sym_vars,
                                       uid2sym_params=uid2sym_params)

    nnz = len(cols_sorted)
    indices = np.fromiter(rows_sorted, dtype=np.int32, count=nnz)

    indptr = np.zeros(len(variables) + 1, dtype=np.int32)
    for c in cols_sorted:
        indptr[c + 1] += 1
    np.cumsum(indptr, out=indptr)
    # template with zeros but correct structure
    template_csc = sp.csc_matrix((np.zeros(nnz, dtype=np.float64),
                                  indices.copy(),
                                  indptr.copy()),
                                 shape=(len(eqs), len(variables)))

    def full_jac_fn(values: np.ndarray, params: np.ndarray) -> tuple[csc_matrix, float, float]:  # noqa: D401 – simple
        assert len(values) >= len(variables)

        start_jac = time.time()
        jac_values = functions_ptr(values, params)
        end_jac = time.time()
        jac_eval_time = end_jac - start_jac

        # data = np.array(jac_values, dtype=np.float64)

        start_csc_matrix = time.time()
        csc_matrix = template_csc.copy()
        csc_matrix.data[:] = jac_values
        # csc_matrix = sp.csc_matrix((data, indices, indptr), shape=(len(eqs), len(variables)))
        end_csc_matrix = time.time()
        csc_matrix_time = end_csc_matrix - start_csc_matrix

        return csc_matrix, jac_eval_time, csc_matrix_time

    return full_jac_fn


def _get_jacobian(eqs: List[Expr],
                  variables: List[Var],
                  uid2sym_vars: Dict[int, str],
                  uid2sym_params: Dict[int, str], ):
    """
    JIT‑compile a sparse Jacobian evaluator for *equations* w.r.t *variables*.
    :param eqs: Array of equations
    :param variables: Array of variables to differentiate against
    :param uid2sym_vars: dictionary relating the uid of a var with its array name (i.e. var[0])
    :param uid2sym_params:
    :return:
            jac_fn : callable(values: np.ndarray, params: np.ndarray) -> scipy.sparse.csc_matrix
                Fast evaluator in which *values* is a 1‑D NumPy vector of length
                ``len(variables)``and *params* is a 1‑D NumPy vector of length
                ``len(parameters)``
            sparsity_pattern : tuple(np.ndarray, np.ndarray)
                Row/col indices of structurally non‑zero entries.
    """

    # Ensure deterministic variable order
    check_set = set()
    for v in variables:
        if v in check_set:
            raise ValueError(f"Repeated var {v.name} in the variables' list :(")
        else:
            check_set.add(v)

    triplets: List[Tuple[int, int, Callable]] = []  # (col, row, fn)

    for row, eq in enumerate(eqs):
        for col, var in enumerate(variables):
            d_expression = eq.diff(var).simplify()
            if isinstance(d_expression, Const) and d_expression.value == 0:
                continue  # structural zero
            triplets.append((col, row, d_expression))

    triplets.sort(key=lambda t: (t[0], t[1]))
    cols_sorted, rows_sorted, equations_sorted = zip(*triplets) if triplets else ([], [], [])
    functions_ptr = _compile_equations(eqs=equations_sorted, uid2sym_vars=uid2sym_vars,
                                       uid2sym_params=uid2sym_params)

    nnz = len(cols_sorted)
    indices = np.fromiter(rows_sorted, dtype=np.int32, count=nnz)

    indptr = np.zeros(len(variables) + 1, dtype=np.int32)
    for c in cols_sorted:
        indptr[c + 1] += 1
    np.cumsum(indptr, out=indptr)
    # template with zeros but correct structure
    template_csc = sp.csc_matrix((np.zeros(nnz, dtype=np.float64),
                                  indices.copy(),
                                  indptr.copy()),
                                 shape=(len(eqs), len(variables)))

    def jac_fn(values: np.ndarray, params: np.ndarray) -> tuple[csc_matrix, float, float]:  # noqa: D401 – simple
        assert len(values) >= len(variables)
        pr.enable()

        start_jac = time.time()
        jac_values = functions_ptr(values, params)
        end_jac = time.time()
        jac_eval_time = end_jac - start_jac

        # data = np.array(jac_values, dtype=np.float64)

        start_csc_matrix = time.time()
        csc_matrix = template_csc.copy()
        csc_matrix.data[:] = jac_values
        # csc_matrix = sp.csc_matrix((data, indices, indptr), shape=(len(eqs), len(variables)))
        end_csc_matrix = time.time()
        csc_matrix_time = end_csc_matrix - start_csc_matrix

        pr.disable()
        pr.dump_stats('profile.pstat')

        return csc_matrix, jac_eval_time, csc_matrix_time

    return jac_fn


class BlockSolver:
    """
    A network of Blocks that behaves roughly like a Simulink diagram.
    """

    def __init__(self, block_system: Block, glob_time: Var):
        """
        Constructor        
        :param block_system: BlockSystem
        """
        self.block_system: Block = block_system
        # TODO: uids, system vars,.. have been already processed in block_system and can be retrived from there.
        # Flatten the block lists, preserving declaration order
        self._algebraic_vars: List[Var] = list()
        self._algebraic_eqs: List[Expr] = list()
        self._state_vars: List[Var] = list()
        self._state_eqs: List[Expr] = list()
        self._parameters: List[Const] = list()
        self._parameters_eqs: List[Expr] = list()
        self.glob_time: Var = glob_time
        self.vars2device = block_system.vars2device

        for b in self.block_system.get_all_blocks():
            self._algebraic_vars.extend(b.algebraic_vars)
            self._algebraic_eqs.extend(b.algebraic_eqs)
            self._state_vars.extend(b.state_vars)
            self._state_eqs.extend(b.state_eqs)
            self._parameters.extend(b.parameters)
            self._parameters_eqs.extend(b.parameters_eqs)

        self._n_state = len(self._state_vars)
        self._n_alg = len(self._algebraic_vars)
        self._n_vars = self._n_state + self._n_alg
        self._n_params = len(self._parameters)

        # generate the in-code names for each variable
        # inside the compiled functions the variables are
        # going to be represented by an array called vars[]

        uid2sym_vars: Dict[int, str] = dict()
        uid2sym_params: Dict[int, str] = dict()
        uid2sym_t: Dict[int, str] = dict()
        self.uid2var: Dict[int, Var] = dict()
        self.uid2idx_vars: Dict[int, int] = dict()
        self.uid2idx_params: Dict[int, int] = dict()
        self.uid2idx_t: Dict[int, int] = dict()
        i = 0
        for v in self._state_vars:
            uid2sym_vars[v.uid] = f"vars[{i}]"
            self.uid2var[v.uid] = v
            self.uid2idx_vars[v.uid] = i
            i += 1

        for v in self._algebraic_vars:
            uid2sym_vars[v.uid] = f"vars[{i}]"
            self.uid2var[v.uid] = v
            self.uid2idx_vars[v.uid] = i

            i += 1

        j = 0
        for j, ep in enumerate(self._parameters):
            uid2sym_params[ep.uid] = f"params[{j}]"
            self.uid2idx_params[ep.uid] = j
            j += 1

        k = 0
        uid2sym_t[self.glob_time.uid] = f"glob_time"
        self.uid2idx_t[self.glob_time.uid] = k

        # Compile RHS and Jacobian
        """
                   state Var   algeb var  
        state eq |J11        | J12       |    | ∆ state var|    | ∆ state eq |
                 |           |           |    |            |    |            |
                 ------------------------- x  |------------|  = |------------|
        algeb eq |J21        | J22       |    | ∆ algeb var|    | ∆ algeb eq |
                 |           |           |    |            |    |            |
        """
        start_compiling = time.time()
        print("Compiling...", end="")

        # all_eqs = self._state_eqs + self._algebraic_eqs
        # all_vars = self._state_vars + self._algebraic_vars

        # self._full_jacobian_fn = _get_full_jacobian(eqs=all_eqs, variables=all_vars, uid2sym_vars=uid2sym_vars, uid2sym_params=uid2sym_params)

        self._rhs_state_fn = _compile_equations(eqs=self._state_eqs, uid2sym_vars=uid2sym_vars,
                                                uid2sym_params=uid2sym_params)

        self._rhs_algeb_fn = _compile_equations(eqs=self._algebraic_eqs, uid2sym_vars=uid2sym_vars,
                                                uid2sym_params=uid2sym_params)

        self._params_fn = _compile_parameters_equations(eqs=self._parameters_eqs, uid2sym_t=uid2sym_t)

        self._j11_fn = _get_jacobian(eqs=self._state_eqs, variables=self._state_vars, uid2sym_vars=uid2sym_vars,
                                     uid2sym_params=uid2sym_params)
        self._j12_fn = _get_jacobian(eqs=self._state_eqs, variables=self._algebraic_vars, uid2sym_vars=uid2sym_vars,
                                     uid2sym_params=uid2sym_params)
        self._j21_fn = _get_jacobian(eqs=self._algebraic_eqs, variables=self._state_vars, uid2sym_vars=uid2sym_vars,
                                     uid2sym_params=uid2sym_params)
        self._j22_fn = _get_jacobian(eqs=self._algebraic_eqs, variables=self._algebraic_vars, uid2sym_vars=uid2sym_vars,
                                     uid2sym_params=uid2sym_params)

        print("done!")
        end_compiling = time.time()
        compilation_time = end_compiling - start_compiling
        print(f"all compilation time = {compilation_time:.6f} [s]")

    @property
    def state_vars(self) -> List[Var]:
        """
        Get the state vars
        :return: List[Var]
        """
        return self._state_vars

    def get_var_idx(self, v: Var) -> int:
        """

        :param v:
        :return:
        """
        return self.uid2idx_vars[v.uid]

    def get_vars_idx(self, variables: Sequence[Var]) -> np.ndarray:
        """

        :param variables:
        :return:
        """
        return np.array([self.uid2idx_vars[v.uid] for v in variables])

    def sort_vars(self, mapping: dict[Var, float]) -> np.ndarray:
        """
        Helper function to build the initial vector
        :param mapping: var->initial value mapping
        :return: array matching with the mapping, matching the solver ordering
        """
        x = np.zeros(len(self._state_vars) + len(self._algebraic_vars), dtype=object)

        for key, val in mapping.items():
            i = self.uid2idx_vars[key.uid]
            x[i] = key

        return x

    def sort_vars_from_uid(self, mapping: dict[tuple[int, str], float]) -> np.ndarray:
        """
        Helper function to build the initial vector
        :param mapping: var->initial value mapping
        :return: array matching with the mapping, matching the solver ordering
        """
        x = np.zeros(len(self._state_vars) + len(self._algebraic_vars), dtype=object)

        for key, val in mapping.items():
            uid, name = key
            i = self.uid2idx_vars[uid]
            x[i] = uid

        return x

    def build_init_vars_vector(self, mapping: dict[Var, float]) -> np.ndarray:
        """
        Helper function to build the initial vector
        :param mapping: var->initial value mapping
        :return: array matching with the mapping, matching the solver ordering
        """
        x = np.zeros(len(self._state_vars) + len(self._algebraic_vars))

        for key, val in mapping.items():
            if key.uid in self.uid2idx_vars.keys():
                i = self.uid2idx_vars[key.uid]
                x[i] = val
            else:
                raise ValueError(f"Missing variable {key} definition")

        return x

    def build_init_vars_vector_from_uid(self, mapping: dict[tuple[int, str], float]) -> np.ndarray:
        """
        Helper function to build the initial vector
        :param mapping: var->initial value mapping
        :return: array matching with the mapping, matching the solver ordering
        """
        x = np.zeros(len(self._state_vars) + len(self._algebraic_vars))

        for key, val in mapping.items():
            uid, name = key
            if uid in self.uid2idx_vars.keys():
                i = self.uid2idx_vars[uid]
                x[i] = val
            else:
                raise ValueError(f"Missing uid {key} definition")

        return x

    def build_init_params_vector(self, mapping: dict[Var, float]) -> np.ndarray:
        """
        Helper function to build the initial vector
        :param mapping: var->initial value mapping
        :return: array matching with the mapping, matching the solver ordering
        """
        x = np.zeros(self._n_params)

        for key, val in mapping.items():
            i = self.uid2idx_params[key.uid]

            x[i] = val

        return x

    def rhs_fixed(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Return 𝑑x/dt given the current *state* vector.
        :param x: get the right-hand-side give a state vector
        :return [f_state_update, f_algeb]
        """
        f_algeb = np.array(self._rhs_algeb_fn(x, params))

        if self._n_state > 0:
            f_state = np.array(self._rhs_state_fn(x, params))
            return np.r_[f_state, f_algeb]
        else:
            return f_algeb

    def rhs_implicit(self, x: np.ndarray, xn: np.ndarray, params: np.ndarray, sim_step, h: float) -> np.ndarray:
        """
        Return 𝑑x/dt given the current *state* vector.
        :param x: get the right-hand-side give a state vector
        :param xn:
        :param params: params array
        :param sim_step: simulation step
        :param h: simulation step
        :return [f_state_update, f_algeb]
        """
        f_algeb = np.array(self._rhs_algeb_fn(x, params))
        sim_step = sim_step
        if self._n_state > 0:
            f_state = np.array(self._rhs_state_fn(x, params))
            f_state_update = x[:self._n_state] - xn[:self._n_state] - h * f_state
            return np.r_[f_state_update, f_algeb]

        else:
            return f_algeb

    # def jacobian_implicit(
    #         self,
    #         x: np.ndarray,
    #         params: np.ndarray,
    #         h: float,
    #         n_processes: int = 4
    # ) -> Tuple[sp.csc_matrix, float, float]:
    #     """
    #     Compute the implicit Jacobian:
    #         | I - h*J11   -h*J12 |
    #         | J21         J22    |
    #     using pathos multiprocessing to parallelize the four Jacobians.
    #     """
    #
    #     tasks = [
    #         (self._j11_fn, (x, params)),
    #         (self._j12_fn, (x, params)),
    #         (self._j21_fn, (x, params)),
    #         (self._j22_fn, (x, params))
    #     ]
    #
    #     # Use pathos ProcessingPool
    #     with Pool(nodes=n_processes) as pool:
    #         # pathos can map closures/lambdas directly
    #         results = pool.map(lambda t: t[0](*t[1]), tasks)
    #
    #     # Unpack results
    #     (j11_val, jac_time11, csc_time11), \
    #         (j12_val, jac_time12, csc_time12), \
    #         (j21_val, jac_time21, csc_time21), \
    #         (j22_val, jac_time22, csc_time22) = results
    #
    #     # Total timing
    #     jac_time = jac_time11 + jac_time12 + jac_time21 + jac_time22
    #     csc_time = csc_time11 + csc_time12 + csc_time21 + csc_time22
    #
    #     # Build blocks
    #     I = sp.eye(self._n_state, self._n_state)
    #     j11: sp.csc_matrix = (I - h * j11_val).tocsc()
    #     j12: sp.csc_matrix = - h * j12_val
    #     j21: sp.csc_matrix = j21_val
    #     j22: sp.csc_matrix = j22_val
    #
    #     J = pack_4_by_4_scipy(j11, j12, j21, j22)
    #
    #     return J, jac_time, csc_time

    # def full_jacobian_implicit(self, x: np.ndarray, params: np.ndarray, h: float) -> tuple[sp.csc_matrix, float, float]:
    #     """
    #     :param x: vector or variables' values
    #     :param params: params array
    #     :param h: step
    #     :return:
    #     """
    #
    #     """
    #               state Var    algeb var
    #     state eq |I - h * J11 | - h* J12  |    | ∆ state var|    | ∆ state eq |
    #              |            |           |    |            |    |            |
    #              -------------------------- x  |------------|  = |------------|
    #     algeb eq |J21         | J22       |    | ∆ algeb var|    | ∆ algeb eq |
    #              |            |           |    |            |    |            |
    #     """
    #     ######################## to del
    #
    #     J, full_jac_time, csc_time = self._full_jacobian_fn(x, params)
    #
    #     j11_val, jac_time11, csc_time11 = self._j11_fn(x, params)
    #     j12_val, jac_time12, csc_time12 = self._j12_fn(x, params)
    #     j21_val, jac_time21, csc_time21 = self._j21_fn(x, params)
    #     j22_val, jac_time22, csc_time22 = self._j22_fn(x, params)
    #
    #     jac_time = jac_time11 + jac_time12 + jac_time21 + jac_time22
    #
    #     return J, jac_time, full_jac_time, csc_time

    def jacobian_implicit(self, x: np.ndarray, params: np.ndarray, h: float) -> tuple[
        sp.csc_matrix, float, float, float]:
        """
        :param x: vector or variables' values
        :param params: params array
        :param h: step
        :return:
        """

        """
                  state Var    algeb var
        state eq |I - h * J11 | - h* J12  |    | ∆ state var|    | ∆ state eq |
                 |            |           |    |            |    |            |
                 -------------------------- x  |------------|  = |------------|
        algeb eq |J21         | J22       |    | ∆ algeb var|    | ∆ algeb eq |
                 |            |           |    |            |    |            |
        """
        ########################to del

        j11_val, jac_time11, csc_time11 = self._j11_fn(x, params)
        j12_val, jac_time12, csc_time12 = self._j12_fn(x, params)
        j21_val, jac_time21, csc_time21 = self._j21_fn(x, params)
        j22_val, jac_time22, csc_time22 = self._j22_fn(x, params)

        jac_time = jac_time11 + jac_time12 + jac_time21 + jac_time22
        csc_time = csc_time11 + csc_time12 + csc_time21 + csc_time22

        I = sp.eye(m=self._n_state, n=self._n_state)
        j11: sp.csc_matrix = (I - h * j11_val).tocsc()
        j12: sp.csc_matrix = - h * j12_val
        j21: sp.csc_matrix = j21_val
        j22: sp.csc_matrix = j22_val

        J = pack_4_by_4_scipy(j11, j12, j21, j22)

        # j11: sp.csc_matrix = (I - h * self._j11_fn(x, params)).tocsc()
        # j12: sp.csc_matrix = - h * self._j12_fn(x, params)
        # j21: sp.csc_matrix = self._j21_fn(x, params)
        # j22: sp.csc_matrix = self._j22_fn(x, params)

        return J, jac_time, csc_time

    def residual_init(self, z: np.ndarray, params: np.ndarray):
        # concatenate state & algebraic residuals
        f_s = np.array(self._rhs_state_fn(z, params))  # f_state(x)        == 0 at t=0
        f_a = np.array(self._rhs_algeb_fn(z, params))  # g_algeb(x, y)     == 0
        return np.r_[f_s, f_a]

    def jacobian_init(self, z: np.ndarray, params: np.ndarray):
        J11 = self._j11_fn(z, params)  # ∂f_state/∂x
        J12 = self._j12_fn(z, params)  # ∂f_state/∂y
        J21 = self._j21_fn(z, params)  # ∂g/∂x
        J22 = self._j22_fn(z, params)  # ∂g/∂y
        return pack_4_by_4_scipy(J11, J12, J21, J22)  # → sparse 2×2 block Jacobian

    def initialize_with_newton(self, x0: np.ndarray, params0: np.ndarray,
                               tol=1e-6, max_it=20, use_preconditioner: bool = True):
        """
        DAE system initialization using a Newton-Krylov method
        :param x0: base solution (rough)
        :param params0: base parameters array
        :param tol: Tolerance
        :param max_it: maximum newton iterations
        :param use_preconditioner: Use a GMRES preconditioner?
        :return: new, hopefully better initial point
        """
        x_vec = x0.copy()  # initial guess vector

        for k in range(max_it):
            F = self.residual_init(x_vec,
                                   params=params0)  # F is a numpy array containing the f and g functions residuals
            if np.linalg.norm(F,
                              np.inf) < tol:  # use infinit norm: The maximum absolute value of the components of the vector
                print(f"Converged in {k} iterations :", x_vec)
                break

            J = self.jacobian_init(x_vec, params=params0)

            # --------- iterative GMRES  ------------------------------------------
            # Build an ILU(0) factorisation as a right-preconditioner

            if use_preconditioner:
                eps = 1e-8 * sp.linalg.norm(J, np.inf)
                Jreg = J + eps * sp.eye(J.shape[0], format='csc')
                ilu = spilu(Jreg, drop_tol=1e-6, fill_factor=10)
                M = LinearOperator(J.shape, ilu.solve)  # M ≈ J⁻¹
                dx, info = gmres(A=J, b=-F, M=M, atol=1e-9, restart=200)
            else:
                dx, info = gmres(A=J, b=-F, atol=1e-9, restart=500)

            if info != 0:
                raise RuntimeError(f"GMRES failed (info = {info}) at Newton iter {k}")

            x_vec += dx

        else:
            raise RuntimeError("Initialisation did not converge")

        return x_vec

    def resid_gamma(self, z, z_prev, dt, n_s, params0):
        """Pseudo-transient residual  R(z) = 0  (index-1 semi-explicit DAE)."""
        f_s = np.array(self._rhs_state_fn(z, params0))  # f(x,y)
        f_a = np.array(self._rhs_algeb_fn(z, params0))  # g(x,y)
        return np.r_[(z[:n_s] - z_prev[:n_s]) / dt - f_s,  # BE for states
        f_a]

    def jacob_gamma(self, z, dt, I_s, params0):
        """Block Jacobian of the Ψtc residual."""
        J11 = self._j11_fn(z, params0)  # ∂f_state/∂x
        J12 = self._j12_fn(z, params0)  # ∂f_state/∂y
        J21 = self._j21_fn(z, params0)  # ∂g/∂x
        J22 = self._j22_fn(z, params0)  # ∂g/∂y

        # Assemble:  [ I/dt -J11   -J12 ]
        #            [   J21        J22 ]
        top_left = I_s / dt - J11
        top_right = -J12
        bottom = sp.hstack([J21, J22], format='csc')
        return sp.vstack([sp.hstack([top_left, top_right], format='csc'),
                          bottom], format='csc')

    def initialize_with_pseudo_transient_gamma(self, x0: np.ndarray, params0: np.ndarray,
                                               dt0=1.0, beta=0.5,
                                               tol=1e-6, max_it=20):
        """
        DAE system initialization using a Pseudo-transient (Ψtc) continuation method
        :param x0: base solution (rough)
        :param params0: base parameters array
        :param dt0: starting pseudo-time step (s)
        :param beta: step-length reduction factor (< 1)
        :param tol: Tolerance
        :param max_it: maximum newton iterations
        :return: new, hopefully better initial point
        """
        x_vec = x0.copy()

        z_vec = x0.copy()  # internal ordering
        n_s = self._n_state
        I_s = sp.eye(n_s, format='csc')  # identity for ∂x/∂x

        dt = dt0
        step = 0
        while True:
            # Newton on    R(z) = 0   with current Δτ
            z_prev = z_vec.copy()
            for it in range(max_it):
                R = self.resid_gamma(z_vec, z_prev, dt, n_s, params0)

                if np.linalg.norm(R, np.inf) < tol:
                    break

                J = self.jacob_gamma(z_vec, dt, I_s, params0)

                # ILU-preconditioned GMRES  (robust & memory-friendly)
                try:
                    M = LinearOperator(J.shape, spilu(J, drop_tol=1e-5,
                                                      fill_factor=10).solve)
                except RuntimeError:
                    M = None  # fallback – still works thanks to I/dt term

                dz, info = gmres(A=J, b=-R, M=M, atol=1e-9, restart=200)
                if info != 0:
                    raise RuntimeError(f"GMRES failed at Ψtc step {step}, "
                                       f"Newton iter {it} (info={info})")
                z_vec += dz

            # Check global convergence
            res_norm = np.linalg.norm(self._rhs_state_fn(z_vec, params0), np.inf)
            res_norm = max(res_norm,
                           np.linalg.norm(self._rhs_algeb_fn(z_vec, params0), np.inf))

            if res_norm < tol:
                print(f"Ψtc converged in {step + 1} steps, residual {res_norm:.2e}")
                break

            # Reduce pseudo-time-step and continue
            dt *= beta
            step += 1
            if dt < 1e-9:
                raise RuntimeError("Ψtc failed: Δτ became too small "
                                   "(model may be ill-posed)")

        return z_vec

    def _apply_lambda(self,
                      params: np.ndarray,
                      ramps_descr: list[tuple[int, float, float, Const]],
                      lam: float) -> None:
        """
        Update the parameter vector *in-place* for a given λ ∈ [0,1].

        Some Const implementations expose a writable .value, others make it
        read-only.  We therefore update the associated Const object only if
        it provides a usable mutator; otherwise we rely on the explicit
        'params' array that is already passed to every RHS / Jacobian call.
        """
        for idx, start, end, const_ref in ramps_descr:
            val = start + lam * (end - start)
            params[idx] = val

    def _psi_tc(self,
                z_init: np.ndarray,
                params: np.ndarray,
                dt0: float,
                beta: float,
                psi_tol: float,
                newton_tol: float,
                newton_max: int) -> np.ndarray:
        """
        Pseudo-transient continuation inner solver (index-1 semi-explicit DAE).
        Returns a *consistent* z vector for the **current** parameter set.
        """
        n_s = self._n_state
        I_s = sp.eye(n_s, format="csc") if n_s else None
        z = z_init.copy()
        dt = dt0

        while True:
            z_prev = z.copy()
            # ------------ Newton on Ψtc residual -----------------------------
            for _ in range(newton_max):
                f_a = np.array(self._rhs_algeb_fn(z, params))
                if n_s:
                    f_s = np.array(self._rhs_state_fn(z, params))
                    R = np.r_[(z[:n_s] - z_prev[:n_s]) / dt - f_s, f_a]
                else:
                    R = f_a

                if np.linalg.norm(R, np.inf) < newton_tol:
                    break

                J22 = self._j22_fn(z, params)
                J12 = self._j12_fn(z, params)
                J21 = self._j21_fn(z, params)

                if n_s:
                    J11 = self._j11_fn(z, params)
                    JL = I_s / dt - J11
                    JR = -J12
                    top = sp.hstack([JL, JR], format="csc")
                    bot = sp.hstack([J21, J22], format="csc")
                    J = sp.vstack([top, bot], format="csc")
                else:
                    J = J22

                try:
                    M = LinearOperator(J.shape,
                                       spilu(J, drop_tol=1e-5, fill_factor=10).solve)
                except RuntimeError:
                    M = None

                dz, info = gmres(J, -R, M=M, atol=1e-9, restart=200)
                if info != 0:
                    raise RuntimeError("GMRES failed during Ψtc")
                z += dz

            # ------------ stop when original residual is small ---------------
            res = np.linalg.norm(self._rhs_algeb_fn(z, params), np.inf)
            if n_s:
                res = max(res, np.linalg.norm(self._rhs_state_fn(z, params), np.inf))
            if res < psi_tol:
                return z

            dt *= beta
            if dt < 1e-9:
                raise RuntimeError("Ψtc stalled ⇒ no equilibrium?")

    def initialise_homotopy(self,
                            z0,
                            params,
                            ramps: list[tuple[Const | Var, float]] | None = None,
                            lam_steps: int = 21,
                            psi_dt0: float = 5.0,
                            psi_beta: float = 0.4,
                            psi_tol: float = 1e-10,
                            newton_tol: float = 1e-12,
                            newton_max: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Robust, guess-free initialiser:
          • cold start  →  Ψtc  →  homotopy λ∈[0,1]
        Returns (x0, param_vector).
        """

        # 1 ── cold start
        z = z0.copy()

        # 2 ── ramp description
        if ramps is None:  # ramp all Consts 0 → nominal
            ramps = [(p, 0.0) for p in self._parameters]

        # params = np.zeros(self._n_params)
        # for p in self._parameters:
        #     params[self.uid2idx_params[p.uid]] = p.value  # :contentReference[oaicite:2]{index=2}

        ramps_descr = []
        # for const_obj, start_val in ramps:
        #     idx = self.uid2idx_params[const_obj.uid]
        #     ramps_descr.append((idx, start_val, const_obj.value, const_obj))

        # 3 ── homotopy outer loop
        for lam in np.linspace(0.0, 1.0, lam_steps):
            self._apply_lambda(params, ramps_descr, lam)
            z = self._psi_tc(z, params, psi_dt0, psi_beta,
                             psi_tol, newton_tol, newton_max)

        print("Homotopy initialisation succeeded ✔")
        return z, params

    def initialise_homotopy_adaptive_lambda(self,
                                            z0,
                                            params,
                                            ramps: list[tuple[Const | Var, float]] | None = None,
                                            psi_dt0: float = 5.0,
                                            psi_beta: float = 0.4,
                                            psi_tol: float = 1e-10,
                                            newton_tol: float = 1e-12,
                                            newton_max: int = 10,
                                            delta_lam_init: float = 0.05,
                                            min_step: float = 1e-4,
                                            max_step: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """
        Homotopy with adaptive lambda stepping.
        :param z0:
        :param params:
        :param ramps:
        :param psi_dt0:
        :param psi_beta:
        :param psi_tol:
        :param newton_tol:
        :param newton_max:
        :param delta_lam_init:
        :param min_step:
        :param max_step:
        :return:
        """
        z = z0.copy()

        if ramps is None:
            ramps = [(p, 0.0) for p in self._parameters]

        ramps_descr = []
        for const_obj, start_val in ramps:
            idx = self.uid2idx_params[const_obj.uid]
            ramps_descr.append((idx, start_val, const_obj.value, const_obj))

        lam = 0.0
        lam_eps = 1e-6
        delta_lam = delta_lam_init

        while lam < 1.0 - lam_eps:
            try:
                self._apply_lambda(params, ramps_descr, lam)
                z = self._psi_tc(z, params, psi_dt0, psi_beta,
                                 psi_tol, newton_tol, newton_max)

                lam_new = min(lam + delta_lam, 1.0)
                self._apply_lambda(params, ramps_descr, lam_new)
                z_new = self._psi_tc(z, params, psi_dt0, psi_beta,
                                     psi_tol, newton_tol, newton_max)

                # Step succeeded, accept and increase step size
                lam = lam_new
                z = z_new
                delta_lam = min(delta_lam * 1.5, max_step)
                print(f"Adaptive λ stepping: λ={lam:.4f}, step={delta_lam:.4f}")

            except RuntimeError as e:
                # Step failed, reduce step size and retry
                delta_lam *= 0.5
                print(f"Step failed at λ={lam:.4f}, reducing step to {delta_lam:.4f}")
                if delta_lam < min_step:
                    raise RuntimeError(f"Adaptive homotopy failed at λ={lam:.4f}") from e

        # Ensure final λ=1
        self._apply_lambda(params, ramps_descr, 1.0)
        z = self._psi_tc(z, params, psi_dt0, psi_beta,
                         psi_tol, newton_tol, newton_max)

        print("Homotopy initialisation (adaptive lambda) succeeded ✔")
        return z, params

    def get_dummy_x0(self):
        return np.zeros(self._n_state)

    def equations(self) -> Tuple[List[Expr], List[Expr]]:
        """
        Return (algebraic_eqs, state_eqs) as *originally declared* (no substitution).
        """
        return self._algebraic_eqs, self._state_eqs

    def build_params_matrix(self, n_steps: int, params0: np.ndarray, events_list: RmsEvents) -> csr_matrix:
        """

        :param n_steps:
        :param params0:
        :param events_list:
        :return:
        """
        # TODO: reconsider this algorithm entirely,
        #  this all looks like it can be built from the rows, cols, values info

        diff_params_matrix = np.zeros((n_steps, len(params0)))
        params_matrix_current = params0.copy()

        # get events info
        rows, cols, values = events_list.build_triplets_list()

        # build diff sparse matrix
        for time_step in range(n_steps):
            if time_step in rows:  # TODO: very expensive
                for position in np.where(rows == time_step):
                    prop_idx = self.uid2idx_params[cols[position][0].uid]
                    value = values[position]
                    diff_val = value - params_matrix_current[prop_idx]
                    diff_params_matrix[time_step, prop_idx] += diff_val
                    params_matrix_current[prop_idx] = value

        # make params matrix sparse
        diff_params_matrix_spa = csr_matrix(diff_params_matrix)
        return diff_params_matrix_spa

    def simulate(
            self,
            t0: float,
            t_end: float,
            h: float,
            x0: np.ndarray,
            params0: np.ndarray,
            method: str,
            newton_tol: float = 1e-8,
            newton_max_iter: int = 1000,

    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param events_list:
        :param params0:
        :param t0: start time
        :param t_end: end time
        :param h: step
        :param x0: initial values
        :param method: method
        :param newton_tol:
        :param newton_max_iter:
        :return: 1D time array, 2D array of simulated variables
        """
        if method == "euler":
            return self._simulate_fixed(t0, t_end, h, x0, params0, stepper="euler")
        if method == "rk4":
            return self._simulate_fixed(t0, t_end, h, x0, params0, stepper="rk4")
        if method == "implicit_euler":
            return self._simulate_implicit_euler(
                t0=t0, t_end=t_end, h=h, x0=x0, params0=params0,
                tol=newton_tol, max_iter=newton_max_iter,
            )
        raise ValueError(f"Unknown method '{method}'")

    def _simulate_fixed(self, t0, t_end, h, x0, params, stepper="euler"):
        """
        Fixed‑step helpers (Euler, RK‑4)
        :param t0:
        :param t_end:
        :param h:
        :param x0:
        :param stepper:
        :return:
        """
        steps = int(np.ceil((t_end - t0) / h))
        t = np.empty(steps + 1)
        y = np.empty((steps + 1, self._n_vars))
        t[0] = t0
        y[0, :] = x0.copy()

        for i in range(steps):
            tn = t[i]
            xn = y[i]
            if stepper == "euler":
                k1 = self.rhs_fixed(xn, params)
                y[i + 1] = xn + h * k1
            elif stepper == "rk4":
                k1 = self.rhs_fixed(xn, params)
                k2 = self.rhs_fixed(xn + 0.5 * h * k1, params)
                k3 = self.rhs_fixed(xn + 0.5 * h * k2, params)
                k4 = self.rhs_fixed(xn + h * k3, params)
                y[i + 1] = xn + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise RuntimeError("unknown stepper")
            t[i + 1] = tn + h
        return t, y

    def _simulate_implicit_euler(self, t0: float, t_end: float, h: float,
                                 x0: np.ndarray,
                                 params0: np.ndarray,
                                 tol=1e-6,
                                 max_iter=1000):
        """
        :param t0:
        :param t_end:
        :param h:
        :param x0:
        :params_matrix:
        :param tol:
        :param max_iter:
        :return:
        """
        steps = int(np.ceil((t_end - t0) / h))
        t = np.empty(steps + 1)
        y = np.empty((steps + 1, self._n_vars))
        t[0] = t0
        y[0] = x0.copy()
        jacobian_time = 0
        functions_time = 0
        params_time = 0
        residual_time = 0
        solv_time = 0
        total_jac_time = 0
        total_csc_time = 0
        for step_idx in range(steps):
            xn = y[step_idx]
            x_new = xn.copy()  # initial guess
            converged = False
            n_iter = 0
            current_time = t[step_idx]

            start_params_calculation = time.time()
            params_current = self._params_fn(float(current_time))
            end_params_calculation = time.time()
            params_calculation_time = end_params_calculation - start_params_calculation
            params_time += params_calculation_time

            while not converged and n_iter < max_iter:

                start_functions_calc = time.time()
                rhs = self.rhs_implicit(x_new, xn, params_current, step_idx, h)
                end_functions_calc = time.time()
                calc_functions_time = end_functions_calc - start_functions_calc
                functions_time += calc_functions_time

                start_residual_calc = time.time()
                residual = np.linalg.norm(rhs, np.inf)
                end_residual_calc = time.time()
                calc_residual_time = end_residual_calc - start_residual_calc
                residual_time += calc_residual_time

                converged = residual < tol

                if step_idx == 0:
                    if converged:
                        print("System well initailzed.")
                    else:
                        print(f"System bad initilaized. DAE resiudal is {residual}.")

                if converged:
                    break

                start_jac_calc = time.time()
                Jf, jac_eval_time, csc_matrix_time = self.jacobian_implicit(x_new, params_current, h)  # sparse matrix
                end_jac_calc = time.time()
                calc_jac_time = end_jac_calc - start_jac_calc
                jacobian_time += calc_jac_time
                total_jac_time += jac_eval_time
                total_csc_time += csc_matrix_time

                start_solv = time.time()
                delta = sp.linalg.spsolve(Jf, -rhs)
                end_solv = time.time()
                solv_time = end_solv - start_solv
                solv_time += calc_jac_time

                x_new += delta
                n_iter += 1

            if converged:

                y[step_idx + 1] = x_new
                t[step_idx + 1] = t[step_idx] + h



            else:
                print(f"Failed to converge at step {step_idx}")
                break
        print((f"jacobian_total_time = {jacobian_time:.6f} [s]"))

        print((f"functions_total_time = {functions_time:.6f} [s]"))
        print((f"params_total_time = {params_time:.6f} [s]"))
        print((f"residual_total_time = {residual_time:.6f} [s]"))
        print((f"solv_time = {solv_time:.6f} [s]"))
        print((f"total_jac_time = {total_jac_time:.6f} [s]"))
        print((f"total_csc_time = {total_csc_time:.6f} [s]"))

        return t, y

    def save_simulation_to_csv(self, filename, t, y, csv_saving=False):
        """
        Save the simulation results to a CSV file.

        Parameters:
        ----------
        filename : str
            The path and name of the CSV file to save.
        t : np.ndarray
            Time vector.
        y : np.ndarray
            Simulation results array (rows: time steps, columns: variable values).

        Returns:
        -------
        None
        """
        # Combine state and algebraic variables
        all_vars = self._state_vars + self._algebraic_vars
        var_names = [str(var) + '_VeraGrid' for var in all_vars]

        # Create DataFrame with time and variable data
        df_simulation_results = pd.DataFrame(data=y, columns=var_names)
        df_simulation_results.insert(0, 'Time [s]', t)

        if csv_saving:
            df_simulation_results.to_csv(filename, index=False)
            print(f"Simulation results saved to: {filename}")
        return df_simulation_results

    def run_small_signal_stability(self, x: Vec, params: Vec, tol=1e-6, plot=True):
        """
        Small Signal Stability analysis
        :param x: variables (1D numpy array)
        :param params: parameters (1D numpy array)
        :param tol:  numerical tolerance for eigenvalues = 0
        :param plot: True(default) if S-domain eigenvalues plot wanted. Else: False
        :return:
            stability: str
                "Unstable", "Marginally stable" or "Asymptotically stable"
            eigenvalues:  1D row numpy array
            participation factors: 2D array csc matrix.
                Participation factors of mode i stored in PF[:,i]
        """

        """
        Small Signal Stability analysis:
        1. Calculate the state matrix (A) from the state space model. From the DAE model:
            Tx'=f(x,y)
            0=g(x,y)
            the A matrix is computed as:
            A = T^-1(f_x - f_y * g_y^{-1} * g_x)   #T is implicit in the jacobian!

        2. Find eigenvalues and right(V) and left(W) eigenvectors

        3. Perform stability assessment

        4. Calculate normalized participation factors PF = W · V
        """

        fx, _, _ = self._j11_fn(x, params)  # ∂f/∂x
        fy, _, _ = self._j12_fn(x, params)  # ∂f/∂y
        gx, _, _ = self._j21_fn(x, params)  # ∂g/∂x
        gy, _, _ = self._j22_fn(x, params)  # ∂g/∂y

        gyx = spsolve(gy, gx)
        A = (fx - fy @ gyx)  # sparse state matrix csc matrix
        An = A.toarray()  # TODO: always use sparse algebra

        num_states = A.shape[0]

        eigen_values, W, V = scipy.linalg.eig(An, left=True, right=True)
        # eigen_values, eigen_vectors = eigs(A)  # TODO: always use sparse algebra
        V = sp.csc_matrix(V)  # right
        W = sp.csc_matrix(W)  # left

        # find participation factors
        participation_factors = sp.lil_matrix(A.shape)
        for row in range(W.shape[0]):
            for column in range(W.shape[0]):
                participation_factors[row, column] = abs(W[row, column]) * abs(V[row, column])

        # normalize participation factors
        PF_abs = sp.csc_matrix(np.ones(num_states)) @ participation_factors
        for i in range(len(eigen_values)):
            participation_factors[:, i] /= PF_abs[0, i]

            # Stability: select positive and zero eigenvalues

        unstable_eigs = eigen_values[np.real(eigen_values) > tol]
        zero_eigs = eigen_values[abs(np.real(eigen_values)) <= tol]
        stable_eigs = eigen_values[np.real(eigen_values) < -tol]

        if unstable_eigs.size == 0:
            if zero_eigs.size == 0:
                stability = "Asymptotically stable"  # TODO: always use Enums for known states
            else:
                stability = "Marginally stable"
        else:
            stability = "Unstable"

        if plot:
            x = eigen_values.real
            y = eigen_values.imag

            plt.scatter(x, y, marker='x', color='blue')
            plt.xlabel("Re [s -1]")
            plt.ylabel("Im [s -1]")
            plt.title("Stability plot")
            plt.axhline(0, color='black', linewidth=1)  # eje horizontal (y = 0)
            plt.axvline(0, color='black', linewidth=1)
            plt.tight_layout()
            plt.show()

        return stability, eigen_values, participation_factors
