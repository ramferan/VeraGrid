# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple, Any, Sequence, Callable, Dict
import math
import numba as nb
import numpy as np
import copy
from scipy.optimize import newton_krylov

from VeraGridEngine.Devices.Parents.physical_device import PhysicalDevice
from VeraGridEngine.Utils.Symbolic import BlockSolver
from VeraGridEngine.Utils.Symbolic.symbolic import _emit, _emit_one
from VeraGridEngine.Utils.Symbolic.block import Block, Expr, Var
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Simulations.PowerFlow.power_flow_results import PowerFlowResults
from VeraGridEngine.enumerations import VarPowerFlowRefferenceType
from VeraGridEngine.basic_structures import Logger, ObjVec, BoolVec
from VeraGridEngine.Utils.Symbolic.symbolic import Var, Expr, Const, _emit, _emit_params_eq, _heaviside, piecewise


def _compile_equation(eqs: Sequence[Expr],
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
    # TODO: Why is there a second compile equation thing here?
    # Build source
    src = f"def _f(vars, params):\n"
    src += f"    out = np.zeros({len(eqs)})\n"
    src += "\n".join([f"    out[{i}] = {_emit_one(e, uid2sym_vars, uid2sym_params)}" for i, e in enumerate(eqs)]) + "\n"
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
    :param uid2sym_t: dictionary relating the uid of a var with its array name (i.e. var[0])
    :param add_doc_string: add the docstring?
    :return: Function pointer that returns an array
    """

    # Build source
    src = f"def _f(glob_time):\n"
    src += f"    out = np.zeros({len(eqs)})\n"
    src += "\n".join(
        [f"    out[{i}] = {_emit_params_eq(e, uid2sym_t)}" for i, e in enumerate(eqs)]) + "\n"
    src += f"    return out"
    ns: Dict[str, Any] = {
                "math": math,
                "np": np,
                "nb": nb,
                "_heaviside": _heaviside,
            }
    exec(src, ns)
    fn = nb.njit(ns["_f"], fastmath=True)

    if add_doc_string:
        fn.__doc__ = "def _f(vars)"
    return fn



def compose_system_block(time: Var, grid: MultiCircuit,
                          power_flow_results: PowerFlowResults, vars2device:Dict[int, PhysicalDevice], vars_glob_name2uid: Dict[str, int]) -> Tuple[Var, Block, Dict[Tuple[int, str], float]]:

    """
    Compose all RMS models
    :param time: time of the simulation
    :param grid:
    :param power_flow_results:
    :param vars2device: dictionary relating uid of vars with the device they belong to
    :param vars_glob_name2uid: dictionary relating global name of the variable and uid (used when showing results)
    :return: System block and initial guess dictionary
    """
    # already computed grid power flow
    res = power_flow_results

    Sf = res.Sf / grid.Sbase
    St = res.St / grid.Sbase

    # create the system block
    sys_block = Block(children=[], in_vars=[])

    sys_block.vars2device = vars2device
    sys_block.vars_glob_name2uid = vars_glob_name2uid

    # initialize containers
    init_guess: Dict[Tuple[int, str], float] = {}
    sys_vars: list[Tuple[int, str]] = []
    seen_vars: set[Tuple[int, str]] = set()

    uid2sym_vars: Dict[int, str] = {}
    uid2idx_vars: Dict[int, int] = {}
    uid2sym_t: Dict[int, str] = dict()
    uid2idx_t: Dict[int, int] = dict()

    # fill uid2sym_t and uid2idx_t dicts
    array_index_t = 0
    uid2sym_t[time.uid] = f"glob_time"
    uid2idx_t[time.uid] = array_index_t



    array_index = 0

    # buses
    for i, elm in enumerate(grid.buses):

        # get model and model vars
        mdl = elm.rms_model.model
        mdl_vars = mdl.state_vars + mdl.algebraic_vars

        # fill system variables list
        for var in mdl_vars:
            key = (var.uid, var.name)
            if key not in seen_vars:
                sys_vars.append(key)
                seen_vars.add(key)

        # fill uid2sym and uid2idx dicts
        for v in mdl_vars:
            uid2sym_vars[v.uid] = f"vars[{array_index}]"
            uid2idx_vars[v.uid] = array_index
            array_index += 1

        # fill init_guess
        # TODO: initialization from power injections of PFlows results needs
        #  to be addressed when multiple devices are connected to the same bus.
        # (Shunt Load, for the benchmark case, two generators,...)
        init_guess[(mdl.external_mapping[VarPowerFlowRefferenceType.Vm].uid,
                    mdl.external_mapping[VarPowerFlowRefferenceType.Vm].name)] = float(np.abs(res.voltage[i]))
        init_guess[(mdl.external_mapping[VarPowerFlowRefferenceType.Va].uid,
                    mdl.external_mapping[VarPowerFlowRefferenceType.Va].name)] = float(np.angle(res.voltage[i]))
        init_guess[(mdl.external_mapping[VarPowerFlowRefferenceType.P].uid,
                    mdl.external_mapping[VarPowerFlowRefferenceType.P].name)] = float(np.real(res.Sbus[i] / grid.Sbase))
        init_guess[(mdl.external_mapping[VarPowerFlowRefferenceType.Q].uid,
                    mdl.external_mapping[VarPowerFlowRefferenceType.Q].name)] = float(np.imag(res.Sbus[i] / grid.Sbase))

        sys_block.add(mdl)

    # branches
    for i, elm in enumerate(grid.get_branches_iter(add_vsc=True, add_hvdc=True, add_switch=True)):
        mdl = elm.rms_model.model
        mdl_vars = mdl.state_vars + mdl.algebraic_vars

        # fill system variables list
        for var in mdl_vars:
            key = (var.uid, var.name)
            if key not in seen_vars:
                sys_vars.append(key)
                seen_vars.add(key)

        # fill uid2sym and uid2idx dicts
        for v in mdl_vars:
            uid2sym_vars[v.uid] = f"vars[{array_index}]"
            uid2idx_vars[v.uid] = array_index

            array_index += 1

        # fill init_guess
        init_guess[(mdl.external_mapping[VarPowerFlowRefferenceType.Pf].uid,
                    mdl.external_mapping[VarPowerFlowRefferenceType.Pf].name)] = Sf[i].real
        init_guess[(mdl.external_mapping[VarPowerFlowRefferenceType.Qf].uid,
                    mdl.external_mapping[VarPowerFlowRefferenceType.Qf].name)] = Sf[i].imag
        init_guess[(mdl.external_mapping[VarPowerFlowRefferenceType.Pt].uid,
                    mdl.external_mapping[VarPowerFlowRefferenceType.Pt].name)] = St[i].real
        init_guess[(mdl.external_mapping[VarPowerFlowRefferenceType.Qt].uid,
                    mdl.external_mapping[VarPowerFlowRefferenceType.Qt].name)] = St[i].imag

        sys_block.add(mdl)

    # injections

    # initialize uid2sym_params dict

    uid2sym_params: Dict[int, str] = {}
    uid2idx_params: Dict[int, int] = {}

    # already known variables:

    for elm in grid.get_injection_devices_iter():
        bus_rms_mdl = elm.bus.rms_model.model
        mdl = elm.rms_model.model


        init_guess[(mdl.external_mapping[VarPowerFlowRefferenceType.P].uid, mdl.external_mapping[VarPowerFlowRefferenceType.P].name)] = init_guess[(bus_rms_mdl.external_mapping[VarPowerFlowRefferenceType.P].uid, bus_rms_mdl.external_mapping[VarPowerFlowRefferenceType.P].name)]
        init_guess[(mdl.external_mapping[VarPowerFlowRefferenceType.Q].uid, mdl.external_mapping[VarPowerFlowRefferenceType.Q].name)] = init_guess[(bus_rms_mdl.external_mapping[VarPowerFlowRefferenceType.Q].uid, bus_rms_mdl.external_mapping[VarPowerFlowRefferenceType.Q].name)]

        mdl_vars = mdl.state_vars + mdl.algebraic_vars

        for var in mdl_vars:
            key = (var.uid, var.name)
            if key not in seen_vars:
                sys_vars.append(key)
                seen_vars.add(key)

        for v in mdl_vars:
            if v.uid not in uid2sym_vars:
                uid2sym_vars[v.uid] = f"vars[{array_index}]"
                uid2idx_vars[v.uid] = array_index
                array_index += 1

        params_array_index = 0

        for param in mdl.event_dict.keys():
            if param.uid not in uid2sym_params:
                uid2sym_params[param.uid] = f"params[{params_array_index}]"
                uid2idx_params[param.uid] =params_array_index
                params_array_index += 1


        # initialize array for model variables
        x = np.zeros(len(sys_vars))

        # assign initial guesses for known variables
        for uid, name in sys_vars:
            key = (uid, name)
            if key in init_guess:
                x[uid2idx_vars[uid]] = init_guess[key]

        # initialize array for model params
        params_array = np.zeros(len(mdl.event_dict.keys()))

        # compute and assign parameters value
        for param in mdl.event_dict.keys():
            eq = mdl.event_dict[param]
            eq_fn = _compile_parameters_equations([eq], uid2sym_t)
            param_val = float(eq_fn(0.0))
            params_array[uid2idx_params[param.uid]] = param_val

        # compute and assign missing init_vars

        init_guess = init_guess.copy()


        for var in mdl.init_eqs.keys():
            key = (var.uid, var.name)
            if key in init_guess:
                x[uid2idx_vars[var.uid]] = init_guess[key]
            else:
                eq = mdl.init_eqs[var]
                eq_fn = _compile_equation([eq], uid2sym_vars, uid2sym_params)
                init_val = float(eq_fn(x, params_array))
                init_guess[key] = init_val
                x[uid2idx_vars[var.uid]] = init_val
        # TODO: change model to generator, exciter and governor separated and remove this
        for var in mdl.fix_vars:
            eq = mdl.fix_vars_eqs[var.uid]
            eq_fn = _compile_equation([eq], uid2sym_vars, uid2sym_params)
            init_val = float(eq_fn(x, params_array))
            var.value = init_val
        sys_block.add(mdl)

    # del buses P, Q
    for i, elm in enumerate(grid.buses):
            mdl = elm.rms_model.model
            del init_guess[(mdl.external_mapping[VarPowerFlowRefferenceType.P].uid,
                        mdl.external_mapping[VarPowerFlowRefferenceType.P].name)]
            del init_guess[(mdl.external_mapping[VarPowerFlowRefferenceType.Q].uid,
                        mdl.external_mapping[VarPowerFlowRefferenceType.Q].name)]

    return time, sys_block, init_guess


def setP(P: ObjVec, P_used: BoolVec, k: int, val: object):
    if not P_used[k]:
        P[k] = val
        P_used[k] = 1
    else:
        P[k] += val


def setQ(Q: ObjVec, Q_used: BoolVec, k: int, val: object):
    if not Q_used[k]:
        Q[k] = val
        Q_used[k] = 1
    else:
        Q[k] += val


def initialize_rms(grid: MultiCircuit, power_flow_results, logger: Logger = Logger()):
    """
    Initialize all RMS models
    """

    # create time variable

    time = Var("time")

    # instantiate vars2device  and vars_glob_name2uid dicts

    vars2device:Dict[int, PhysicalDevice] = dict()
    vars_glob_name2uid: Dict[str, int] = dict()
    # find events
    rms_events = grid.rms_events

    # already computed grid power flow

    bus_dict = dict()

    # balance equation arrays
    n = len(grid.buses)
    P: ObjVec = np.zeros(n, dtype=object)
    Q: ObjVec = np.zeros(n, dtype=object)
    P_used = np.zeros(n, dtype=int)
    Q_used = np.zeros(n, dtype=int)



    # initialize buses
    for i, elm in enumerate(grid.buses):
        elm.initialize_rms()
        bus_dict[elm] = i
        for state_var in elm.rms_model.model.state_vars:
            vars2device[state_var.uid] = elm
            vars_glob_name2uid[state_var.name + elm.name] = state_var.uid
        for algeb_var in elm.rms_model.model.algebraic_vars:
            vars2device[algeb_var.uid] = elm
            vars_glob_name2uid[algeb_var.name + elm.name] = algeb_var.uid

    # initialize branches
    for elm in grid.get_branches_iter(add_vsc=True, add_hvdc=True, add_switch=True):
        elm.initialize_rms()
        mdl = elm.rms_model.model
        f = bus_dict[elm.bus_from]
        t = bus_dict[elm.bus_to]
        # add variable to conservation equations of the bus to which the element is connected
        setP(P, P_used, f, -mdl.E(VarPowerFlowRefferenceType.Pf))
        setP(P, P_used, t, -mdl.E(VarPowerFlowRefferenceType.Pt))
        setQ(Q, Q_used, f, -mdl.E(VarPowerFlowRefferenceType.Qf))
        setQ(Q, Q_used, t, -mdl.E(VarPowerFlowRefferenceType.Qt))

        for state_var in elm.rms_model.model.state_vars:
            vars2device[state_var.uid] = elm
            vars_glob_name2uid[state_var.name + elm.name] = state_var.uid
        for algeb_var in elm.rms_model.model.algebraic_vars:
            vars2device[algeb_var.uid] = elm
            vars_glob_name2uid[algeb_var.name + elm.name] = algeb_var.uid

    # initialize injections
    for elm in grid.get_injection_devices_iter():
        elm.initialize_rms()
        # create dictionary for collecting events of the same parameter

        if elm.rms_model.model.event_dict is not None:
            collect_events = {
                key: {"times": [], "values": []}
                for key in elm.rms_model.model.event_dict.keys()
            }
            # find out if there are events affecting the device parameters
            rms_evts = [rms_evt for rms_evt in rms_events if rms_evt.device_idtag == elm.idtag]
            if len(rms_evts) !=0:
                for rms_evt in rms_evts:
                    collect_events[rms_evt.parameter]["times"].append(rms_evt.time)
                    collect_events[rms_evt.parameter]["values"].append(rms_evt.value)
                    # TODO: implement the function in block: apply_event
                for param, events_info in collect_events.items():
                    default_value = copy.deepcopy(elm.rms_model.model.event_dict[param])
                    elm.rms_model.model.event_dict[param] = piecewise(time, np.array(events_info["times"]),
                                                                  np.array(events_info["values"]), default_value)


        # after applying the events the model has to be "build"
        mdl = elm.rms_model.model
        # add variable to conservation equations of the bus to which the element is connected
        k = bus_dict[elm.bus]
        setP(P, P_used, k, mdl.E(VarPowerFlowRefferenceType.P))
        setQ(Q, Q_used, k, mdl.E(VarPowerFlowRefferenceType.Q))

        for state_var in elm.rms_model.model.state_vars:
            vars2device[state_var.uid] = elm
            vars_glob_name2uid[state_var.name + elm.name] = state_var.uid
        for algeb_var in elm.rms_model.model.algebraic_vars:
            vars2device[algeb_var.uid] = elm
            vars_glob_name2uid[algeb_var.name + elm.name] = algeb_var.uid

    # add the nodal balance equations
    for i, elm in enumerate(grid.buses):
        mdl = elm.rms_model.model
        if len(mdl.algebraic_eqs) == 0:
            if P_used[i] == 0 and Q_used[i] == 0:
                logger.add_error("Isolated bus", value=i)
            else:
                mdl.algebraic_eqs.append(P[i])
                mdl.algebraic_eqs.append(Q[i])

    return compose_system_block(time, grid, power_flow_results, vars2device, vars_glob_name2uid)
