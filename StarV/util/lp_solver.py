#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
##                                                                     ##
##   Primary contacts: Hoang Dung Tran <dungtran@ufl.edu> (UF)         ##
##                     Sung Woo Choi <sungwoo.choi@ufl.edu> (UF)       ##
##                     Yuntao Li <yli17@ufl.edu> (UF)                  ##
##                     Qing Liu <qliu1@ufl.edu> (UF)                   ##
##                                                                     ##
##   See CONTRIBUTORS for full author contacts and affiliations.       ##
##   This program is licensed under the BSD 3‑Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
LP solver interface for StarV reachability methods
Author: Sung Woo Choi
Created: 03/12/2026

"""
import io
import os
import sys
import warnings
import contextlib
import numpy as np
import scipy.sparse as sp

GUROBI_OPT_TOL = 1e-6

_CUPDLP_CALLBACK = None


def _to_1d(x, dtype=np.float64):
    return np.asarray(x, dtype=dtype).reshape(-1)

# context manager to suppress native stdout/stderr output from external solvers (e.g., cuPDLPx) 
# that do not respect Python-level redirection
def _suppress_native_stdio():
    @contextlib.contextmanager
    def _ctx():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        out_fd = os.dup(1)
        err_fd = os.dup(2)
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
        finally:
            os.dup2(out_fd, 1)
            os.dup2(err_fd, 2)
            os.close(out_fd)
            os.close(err_fd)
            os.close(devnull_fd)

    return _ctx()

def set_cupdlp_callback(callback):
    """Register a global cuPDLP callback for lp_solver='cupdlp' paths."""
    global _CUPDLP_CALLBACK
    if callback is not None and not callable(callback):
        raise TypeError("error: cuPDLP callback must be callable or None")
    _CUPDLP_CALLBACK = callback

def resolve_cupdlp_callback(solver_opts=None):
    solver_opts = {} if solver_opts is None else solver_opts
    return solver_opts.get("cupdlp_callback", _CUPDLP_CALLBACK)

def resolve_cupdlp_fallback_solver(solver_opts=None):
    solver_opts = {} if solver_opts is None else solver_opts
    fallback_solver = solver_opts.get("cupdlp_fallback_solver", "gurobi")
    if fallback_solver == "cupdlp-gurobi":
        fallback_solver = "gurobi"
    return fallback_solver

def is_status_optimal(status):
    if status is None:
        return True
    if isinstance(status, str):
        return status.lower() in ("optimal", "opt", "success", "solved", "0")
    if isinstance(status, (bool, np.bool_)):
        return bool(status)
    try:
        return int(status) == 0
    except Exception:
        return False

def parse_cupdlp_callback_output(out, num_vars, f):
    status, obj, x, y = None, None, None, None

    if isinstance(out, dict):
        status = out.get("status", out.get("exitflag", out.get("code", None)))
        obj = out.get("obj", out.get("objective", out.get("fun", None)))
        x = out.get("x", out.get("primal", out.get("primal_solution", None)))
        y = out.get("y", out.get("dual", out.get("dual_solution", None)))
    elif isinstance(out, (tuple, list)):
        if len(out) == 3:
            status, obj, x = out
        elif len(out) == 2:
            if np.isscalar(out[0]):
                obj, x = out
            else:
                x, obj = out
            status = 0
        elif len(out) == 1:
            x = out[0]
            status = 0
        else:
            raise Exception("error: invalid callback return format for cuPDLP callback")
    else:
        x = out
        status = 0

    if not is_status_optimal(status):
        raise Exception("error: cuPDLP callback did not return an optimal status: {}".format(status))

    if x is not None:
        x = np.asarray(x).reshape(-1)
        if x.shape[0] != num_vars:
            raise Exception(
                "error: invalid primal vector size from cuPDLP callback, expected {}, got {}".format(
                    num_vars, x.shape[0]
                )
            )

    if y is not None:
        y = np.asarray(y).reshape(-1)

    if obj is None:
        if x is None:
            raise Exception("error: cuPDLP callback must return either objective value or primal solution")
        obj = float(_to_1d(f) @ x)
    else:
        obj = float(obj)

    return obj, x, y


def cupdlpx_solver(f, A_ub, b_ub, lb, ub, sense="min", cupdlp_params=None, show=False):
    """Solve LP with cupdlpx and return callback-style output dict."""
    try:
        from cupdlpx import Model as CuPDLPxModel
        from cupdlpx import PDLP as CuPDLPxPDLP
    except Exception as e:
        raise Exception(
            "error: cuPDLP backend not available. Install Python package 'cupdlpx' "
            "(pip install cupdlpx) or pass solver_opts={'cupdlp_callback': fn}."
        ) from e

    cupdlp_params = {} if cupdlp_params is None else dict(cupdlp_params)

    f_vec = _to_1d(f)
    b_vec = _to_1d(b_ub)
    lb_vec = _to_1d(lb)
    ub_vec = _to_1d(ub)
    if f_vec.shape[0] != lb_vec.shape[0] or f_vec.shape[0] != ub_vec.shape[0]:
        raise Exception("error: inconsistent dimensions among f, lb, and ub")

    if sp.issparse(A_ub):
        # cupdlpx expects scipy.sparse matrix classes, not sparse arrays.
        A_mat = sp.csr_matrix(A_ub, dtype=np.float64)
    else:
        A_mat = np.asarray(A_ub, dtype=np.float64)
        if A_mat.ndim == 1:
            A_mat = A_mat.reshape(1, -1)
    if A_mat.shape[0] != b_vec.shape[0]:
        raise Exception("error: inconsistent dimensions between A_ub and b_ub")

    con_lb = np.full(b_vec.shape[0], -np.inf, dtype=np.float64)
    model = CuPDLPxModel(
        objective_vector=f_vec,
        constraint_matrix=A_mat,
        constraint_lower_bound=con_lb,
        constraint_upper_bound=b_vec,
        variable_lower_bound=lb_vec,
        variable_upper_bound=ub_vec,
    )

    if "OutputFlag" not in cupdlp_params and "LogToConsole" not in cupdlp_params:
        try:
            model.setParams(OutputFlag=False)
        except Exception:
            pass
        for key, val in (("LogToConsole", False), ("Verbose", False), ("PrintLevel", 0)):
            try:
                model.setParam(key, val)
            except Exception:
                pass

    if cupdlp_params:
        try:
            model.setParams(**cupdlp_params)
        except Exception:
            for key, val in cupdlp_params.items():
                model.setParam(key, val)

    local_sense = str(sense).lower()
    if local_sense == "max":
        model.ModelSense = CuPDLPxPDLP.MAXIMIZE
    elif hasattr(CuPDLPxPDLP, "MINIMIZE"):
        model.ModelSense = CuPDLPxPDLP.MINIMIZE

    if show:
        model.optimize()
    else:
        with _suppress_native_stdio():
            model.optimize()

    x = getattr(model, "X", None)
    if x is None:
        return {"status": getattr(model, "Status", 1), "obj": None, "x": None}

    x = np.asarray(x).reshape(-1)
    obj = float(getattr(model, "ObjVal", float(f_vec @ x)))
    y = getattr(model, "Pi", None)
    return {"status": 0, "obj": obj, "x": x, "y": y}


def call_cupdlp_callback(callback, f, A_ub, b_ub, lb, ub, sense="min"):
    if callback is None:
        raise Exception(
            "error: lp_solver='cupdlp' requires a callback. "
            "Register one via StarV.util.lp_solver.set_cupdlp_callback(...) "
            "or pass solver_opts={'cupdlp_callback': fn}"
        )
    try:
        out = callback(f=f, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, sense=sense)
    except TypeError as e:
        msg = str(e)
        if (
            "unexpected keyword argument" in msg
            or "positional argument" in msg
            or "required positional argument" in msg
        ):
            out = callback(f, A_ub, b_ub, lb, ub, sense)
        else:
            raise
    return parse_cupdlp_callback_output(out=out, num_vars=_to_1d(lb).shape[0], f=f)


def _build_cupdlpx_callback(solver_opts=None, show=False):
    solver_opts = {} if solver_opts is None else solver_opts
    cupdlp_params = solver_opts.get("cupdlp_params", None)

    def _callback(f, A_ub, b_ub, lb, ub, sense="min"):
        return cupdlpx_solver(f=f, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, sense=sense,
            cupdlp_params=cupdlp_params, show=show)
    return _callback


def build_gurobi_lp_model(A_ub, b_ub, lb, ub, binary_idx=None):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise Exception("error: gurobipy is required for lp_solver='gurobi'") from e

    lb_vec = _to_1d(lb)
    ub_vec = _to_1d(ub)
    num_pred = lb_vec.shape[0]
    if binary_idx is None:
        binary_idx = np.empty(0, dtype=np.int32)

    model = gp.Model()
    model.Params.LogToConsole = 0
    model.Params.OptimalityTol = GUROBI_OPT_TOL

    if binary_idx.size > 0:
        vtype = [GRB.CONTINUOUS] * num_pred
        for idx in binary_idx.tolist():
            vtype[idx] = GRB.BINARY
    else:
        vtype = GRB.CONTINUOUS

    x = model.addMVar(shape=num_pred, lb=lb_vec, ub=ub_vec, vtype=vtype)
    constr = model.addConstr(A_ub @ x <= _to_1d(b_ub))
    return model, x, constr


def apply_gurobi_warm_start(x, constr, num_pred, primal_start=None, dual_start=None):
    try:
        from gurobipy import GRB
    except Exception:
        return

    if primal_start is not None:
        p = np.asarray(primal_start).reshape(-1)
        if p.shape[0] == num_pred:
            try:
                x.setAttr(GRB.Attr.PStart, p)
            except Exception:
                try:
                    x.setAttr(GRB.Attr.Start, p)
                except Exception:
                    pass

    if dual_start is not None:
        y = np.asarray(dual_start).reshape(-1)
        try:
            n_constr = constr.shape[0]
        except Exception:
            n_constr = None
        if n_constr is None or y.shape[0] == n_constr:
            try:
                constr.setAttr(GRB.Attr.DStart, y)
            except Exception:
                pass


def gurobi_solver(f, A_ub, b_ub, lb, ub, binary_idx=None, sense="min",
    center=0.0, model_pack=None, primal_start=None, dual_start=None,
):
    try:
        from gurobipy import GRB
    except Exception as e:
        raise Exception("error: gurobipy is required for lp_solver='gurobi'") from e

    if binary_idx is None:
        binary_idx = np.empty(0, dtype=np.int32)

    if model_pack is None:
        model, x, constr = build_gurobi_lp_model(A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, binary_idx=binary_idx)
    else:
        model, x, constr = model_pack

    f_vec = _to_1d(f)
    apply_gurobi_warm_start(
        x, constr, num_pred=f_vec.shape[0], primal_start=primal_start, dual_start=dual_start
    )

    model.setObjective(f_vec @ x, GRB.MINIMIZE if sense == "min" else GRB.MAXIMIZE)
    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise Exception("error: cannot find an optimal solution, exitflag = {}".format(model.status))
    return float(model.objVal) + float(center)


def scipy_milp_solver(f, A_ub, b_ub, lb, ub, binary_idx=None, sense="min", center=0.0):
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception as e:
        raise Exception(
            "error: lp_solver='scipy-milp' requires scipy.optimize.milp "
            "(available in newer SciPy versions)."
        ) from e

    if binary_idx is None:
        binary_idx = np.empty(0, dtype=np.int32)

    A = sp.csr_matrix(A_ub) if sp.issparse(A_ub) else np.asarray(A_ub, dtype=np.float64)
    b = _to_1d(b_ub)
    lb_vec = _to_1d(lb)
    ub_vec = _to_1d(ub)

    constraints = LinearConstraint(A, -np.inf, b)
    bounds = Bounds(lb_vec, ub_vec)
    integrality = np.zeros(lb_vec.shape[0], dtype=np.int32)
    if binary_idx.size > 0:
        integrality[binary_idx] = 1

    c = _to_1d(f) if sense == "min" else -_to_1d(f)
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if not getattr(res, "success", False):
        raise Exception(
            "error: cannot find an optimal solution, exitflag = {}, message = {}".format(
                res.status, getattr(res, "message", "unknown")
            )
        )
    obj = float(res.fun)
    return (obj if sense == "min" else -obj) + float(center)


def glpk_solver(f, A_ub, b_ub, lb, ub, binary_idx=None, sense="min", center=0.0):
    try:
        import glpk
    except ImportError as e:
        raise ImportError("error: glpk is not installed") from e

    if binary_idx is None:
        binary_idx = np.empty(0, dtype=np.int32)

    f_vec = _to_1d(f)
    b_vec = _to_1d(b_ub)
    lb_vec = _to_1d(lb)
    ub_vec = _to_1d(ub)
    A = sp.csr_matrix(A_ub) if sp.issparse(A_ub) else np.asarray(A_ub, dtype=np.float64)
    if not sp.issparse(A):
        A = sp.csr_matrix(A)

    glpk.env.term_on = False
    lp = glpk.LPX()
    lp.obj.maximize = (sense == "max")

    lp.rows.add(A.shape[0])
    for r in lp.rows:
        r.name = chr(ord("p") + r.index)
        lp.rows[r.index].bounds = None, b_vec[r.index]

    lp.cols.add(lb_vec.shape[0])
    binary_idx_set = set(binary_idx.tolist()) if binary_idx.size > 0 else set()
    for c in lp.cols:
        c.name = "x%d" % c.index
        c.bounds = lb_vec[c.index], ub_vec[c.index]
        if c.index in binary_idx_set:
            c.kind = int

    lp.obj[:] = f_vec.tolist()
    B = A.toarray().reshape(A.shape[0] * A.shape[1],)
    lp.matrix = B.tolist()
    lp.simplex()

    if binary_idx.size > 0:
        if hasattr(lp, "integer"):
            lp.integer()
        elif hasattr(lp, "intopt"):
            lp.intopt()
        else:
            raise Exception("error: current glpk binding does not expose a MIP optimizer method")

    if lp.status != "opt":
        raise Exception("error: cannot find an optimal solution, lp.status = {}".format(lp.status))
    return float(lp.obj.value) + float(center)


def linprog_solver(f, A_ub, b_ub, lb, ub, sense="min", center=0.0):
    try:
        from scipy.optimize import linprog
    except ImportError as e:
        raise ImportError("error: scipy is not installed") from e

    c = _to_1d(f) if sense == "min" else -_to_1d(f)
    bounds = np.column_stack((_to_1d(lb), _to_1d(ub)))
    res = linprog(c, A_ub=A_ub, b_ub=_to_1d(b_ub), bounds=bounds, method="highs")
    if res.status != 0:
        raise Exception("error: cannot find an optimal solution, exitflag = {}".format(res.status))
    obj = float(res.fun)
    return (obj if sense == "min" else -obj) + float(center)


def solve_lp(f, A_ub, b_ub, lb, ub, lp_solver="gurobi", sense="min",
    center=0.0, binary_idx=None, solver_opts=None, model_pack=None, show=False,
):
    """
    Solve an LP or MILP with specified solver and return the optimal objective value.
    Args:
        - f: objective vector
        - A_ub, b_ub: inequality constraint matrix and vector (A_ub @ x <= b_ub)
        - lb, ub: predicate variable lower and upper bounds
        - lp_solver: which LP solver to use (default: "gurobi")
        - sense: "min" or "max" for the optimization direction
        - center: center vector of the reachable set for objective shifting (default: 0.0)
        - binary_idx: optional list or array of variable indices that should be treated as binary (0-1) variables for MILP problems
        - solver_opts: additional options to pass to the LP solver (e.g., callback for "cupdlp", or "cupdlp_params" for cuPDLPx)
        - model_pack: optional pre-built model components for warm-starting (currently only supported for "gurobi")
        - show: whether to print solver output/logging (default: False)
    Returns:
        The optimal objective value for the specified problem and solver.
    """
    
    solver_opts = {} if solver_opts is None else solver_opts
    binary_idx = np.empty(0, dtype=np.int32) if binary_idx is None else np.asarray(binary_idx, dtype=np.int32).reshape(-1)
    has_binary = binary_idx.size > 0
    
    if has_binary and lp_solver == "cupdlp":
        warnings.warn(
            (
                "lp_solver='cupdlp' cannot solve MILP directly. "
                "Using lp_solver='cupdlp-gurobi' instead."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        lp_solver = "cupdlp-gurobi"

    if has_binary and lp_solver == "linprog":
        raise Exception(
            "error: set contains binary predicates (MILP), but lp_solver='{}' solves LP relaxation only. "
            "Use lp_solver='gurobi' or 'scipy-milp' (or 'cupdlp-gurobi' for warm-start).".format(lp_solver)
        )

    if show:
        print(f"Solving LP (sense={sense}) with lp_solver='{lp_solver}'...")
            
    if lp_solver == "gurobi":
        return gurobi_solver(f=f, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, binary_idx=binary_idx, sense=sense,
            center=center, model_pack=model_pack,
        )

    if lp_solver == "cupdlp":
        # first try to resolve a callback from solver_opts or global registration, then try to build a default cupdlpx callback, and finally fallback to another solver if allowed
        callback = resolve_cupdlp_callback(solver_opts=solver_opts)
        if callback is None:
            try:
                callback = _build_cupdlpx_callback(solver_opts=solver_opts, show=show)
            except Exception:
                callback = None

        if callback is None:
            # if fallback is not allowed, raise an exception instead of warning and falling back
            # change to False if you want to enforce the requirement of a callback for 'cupdlp' solver
            # change to True if you want to allow silent fallback to another solver when callback is not provided for 'cupdlp' solver
            if not solver_opts.get("allow_cupdlp_fallback", False):
                raise Exception(
                    "error: lp_solver='cupdlp' requires a callback. "
                    "Register one via set_cupdlp_callback(...) or pass solver_opts={'cupdlp_callback': fn}"
                )

            fallback_solver = resolve_cupdlp_fallback_solver(solver_opts=solver_opts)
            if fallback_solver == "cupdlp":
                raise Exception(
                    "error: invalid cupdlp fallback solver '{}'; choose one of "
                    "['gurobi', 'linprog', 'glpk', 'scipy-milp']".format(fallback_solver)
                )
            warnings.warn(
                (
                    "lp_solver='cupdlp' could not be executed (missing callback/backend). "
                    "Falling back to lp_solver='{}'.".format(fallback_solver)
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            return solve_lp(f=f, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, lp_solver=fallback_solver, sense=sense,
                center=center, binary_idx=binary_idx, solver_opts=solver_opts, model_pack=model_pack, show=show,
            )

        obj, _, _ = call_cupdlp_callback(callback, f=f, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, sense=sense)
        return obj + float(center)

    if lp_solver == "cupdlp-gurobi":
        callback = resolve_cupdlp_callback(solver_opts=solver_opts)
        if callback is None:
            try:
                callback = _build_cupdlpx_callback(solver_opts=solver_opts, show=show)
            except Exception:
                callback = None

        if callback is None:
            x_ws, y_ws = None, None
        else:
            try:
                _, x_ws, y_ws = call_cupdlp_callback(
                    callback, f=f, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, sense=sense
                )
            except Exception:
                if not solver_opts.get("allow_cupdlp_failure", True):
                    raise
                x_ws, y_ws = None, None

        return gurobi_solver(
            f=f, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, binary_idx=binary_idx, sense=sense, center=center,
            model_pack=model_pack, primal_start=x_ws, dual_start=y_ws,
        )

    if lp_solver == "scipy-milp":
        return scipy_milp_solver(f=f, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, binary_idx=binary_idx, sense=sense, center=center)

    if lp_solver == "linprog":
        return linprog_solver(f=f, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, sense=sense, center=center)

    if lp_solver == "glpk":
        return glpk_solver(f=f, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub, binary_idx=binary_idx, sense=sense, center=center)

    raise Exception(
        "error: unknown lp solver, should be one of "
        "['gurobi', 'cupdlp', 'cupdlp-gurobi', 'scipy-milp', 'linprog', 'glpk']"
    )


def solve_index_lp(reach_set, index, sense="min", lp_solver="gurobi", solver_opts=None, model_pack=None, show=False):
    """Solve objective for one flattened state index from a Star-like set.
    Arg:
        reach_set: a reachable set object (e.g., Star or ProbStar)
        index: the flattened state index to solve for
        sense: "min" or "max" for the optimization direction
        lp_solver: which LP solver to use (default: "gurobi")
        solver_opts: additional options to pass to the LP solver (e.g., callback for "cupdlp")
        model_pack: optional pre-built model components for warm-starting (currently only supported for "gurobi")
        show: whether to print solver output/logging
    Returns:
        The optimal objective value for the specified index and optimization sense.
    """
    if hasattr(reach_set, "get_objective_and_center"):
        f, center = reach_set.get_objective_and_center(index)
        if f is None:
            return center
    else:
        assert index >= 0 and index < reach_set.V.shape[0], "error: invalid index"
        if isinstance(reach_set.V, np.ndarray):
            f = reach_set.V[index, 1:]
            center = reach_set.V[index, 0]
            if (f == 0).all():
                return center
        else:
            row = reach_set.V[[index]] if hasattr(reach_set.V, "__getitem__") else reach_set.V._getrow(index)
            center = reach_set.c[index]
            if row.nnz == 0:
                return center
            f = np.asarray(row.toarray()).reshape(-1)

    if hasattr(reach_set, "get_lp_ub"):
        A_ub, b_ub = reach_set.get_lp_ub()
    else:
        if len(reach_set.d) == 0:
            A_ub = sp.csr_array((1, reach_set.num_pred))
            b_ub = np.zeros(1)
        else:
            A_ub, b_ub = reach_set.C, reach_set.d

    if hasattr(reach_set, "get_binary_predicate_indices"):
        binary_idx = reach_set.get_binary_predicate_indices()
    else:
        binary_idx = np.empty(0, dtype=np.int32)

    return solve_lp(f=f, A_ub=A_ub, b_ub=b_ub, lb=reach_set.pred_lb, ub=reach_set.pred_ub,
        lp_solver=lp_solver, sense=sense, center=center, binary_idx=binary_idx,
        solver_opts=solver_opts, model_pack=model_pack, show=show,
    )
