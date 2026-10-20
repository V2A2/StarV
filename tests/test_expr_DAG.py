"""
Modular example for:
1) expression DAG lifting
2) 6D observable construction
3) state propagation from lifted Star
4) symbolic next-step lifted observable derivation skeleton

Author: Zhuoyang Zhou
Date: 03/02/2026 Updated: 03/07/2026
"""

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
pkg_root = os.path.abspath(os.path.join(current_dir, ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from StarV.lifting.exprnode import ExprNode
from StarV.lifting.exprgraph import ExprGraph
from StarV.lifting.executor import LiftExecutor
from StarV.set.star import Star


# ============================================================
# 1) Build 6D observable graph
#    z = [x, theta, x2, x3, sin_t, mul]
#      = [x, theta, x^2, x^3, sin(theta), x^3 sin(theta)]
# ============================================================

def make_graph():
    n1 = ExprNode(id="x", op="var", params={"name": "x"})
    n2 = ExprNode(id="theta", op="var", params={"name": "theta"})
    n3 = ExprNode(id="x2", op="powEven", inputs=["x"], params={"n": 2})
    n4 = ExprNode(id="x3", op="powOdd", inputs=["x"], params={"n": 3})
    n5 = ExprNode(id="sin_t", op="sin", inputs=["theta"])
    n6 = ExprNode(id="mul", op="mul", inputs=["x3", "sin_t"])

    graph = ExprGraph([n1, n2, n3, n4, n5, n6])
    order = graph.topoSort()
    return graph, order


# ============================================================
# 2) Initial state Star
# ============================================================

def make_star(lb, ub):
    return Star(
        np.array(lb, dtype=np.float64),
        np.array(ub, dtype=np.float64)
    )


# ============================================================
# 3) Lift initial state Star to 6D observable Star
# ============================================================

def lift_star(S, order, lp_solver="gurobi", RF=0.0, verbose=True):
    exe = LiftExecutor(lpSolver=lp_solver, RF=RF, verbose=verbose)
    S_hat, idx_map = exe.run(
        initialStar=S,
        orderedNodes=order,
        varIndexMap={"x": 0, "theta": 1},
    )
    return S_hat, idx_map, exe


# ============================================================
# 4) Build state transition matrix from 6D observable Star
#
#    z = [x, theta, x2, x3, sin_t, mul]
#
#    x_{k+1}     = x2 + mul
#    theta_{k+1} = theta + x2 + mul
# ============================================================

def make_k(idx_map, z_dim):
    K = np.zeros((2, z_dim), dtype=np.float64)
    b = np.zeros((2,), dtype=np.float64)

    i_theta = idx_map["theta"]
    i_x2 = idx_map["x2"]
    i_mul = idx_map["mul"]

    # x_{k+1} = x2 + mul
    K[0, i_x2] = 1.0
    K[0, i_mul] = 1.0

    # theta_{k+1} = theta + x2 + mul
    K[1, i_theta] = 1.0
    K[1, i_x2] = 1.0
    K[1, i_mul] = 1.0

    return K, b


# ============================================================
# 5) Propagate next-step state from lifted 6D Star
# ============================================================

def prop_state(S_hat, K, b):
    return S_hat.affineMap(K, b)


# ============================================================
# 6) Re-lift next-step state Star
#
# Note:
# This is NOT yet the true 6D->6D Koopman propagation.
# This is only:
#    low-dim state -> lift again
# ============================================================

def relift(S, order, exe):
    S_hat, idx_map = exe.run(
        initialStar=S,
        orderedNodes=order,
        varIndexMap={"x": 0, "theta": 1},
    )
    return S_hat, idx_map


# ============================================================
# 7) Ground truth comparison by grid sampling
# ============================================================

def gt_box(lb, ub, Nx=401, Nt=401):
    xs = np.linspace(lb[0], ub[0], Nx)
    ths = np.linspace(lb[1], ub[1], Nt)

    X, T = np.meshgrid(xs, ths, indexing="ij")
    Xn = X**2 + (X**3) * np.sin(T)
    Tn = T + X**2 + (X**3) * np.sin(T)

    x_min = float(np.min(Xn))
    x_max = float(np.max(Xn))
    t_min = float(np.min(Tn))
    t_max = float(np.max(Tn))

    return (x_min, x_max), (t_min, t_max)


# ============================================================
# 8) Symbolic next-step lifted observable expressions
#
# Current 6D observable:
#   z1 = x
#   z2 = theta
#   z3 = x^2
#   z4 = x^3
#   z5 = sin(theta)
#   z6 = x^3 sin(theta)
#
# Then:
#   x_next     = z3 + z6
#   theta_next = z2 + z3 + z6
#
# Next-step 6D observables would be:
#   z1_next = x_next
#   z2_next = theta_next
#   z3_next = x_next^2
#   z4_next = x_next^3
#   z5_next = sin(theta_next)
#   z6_next = x_next^3 * sin(theta_next)
# ============================================================

def next_expr():
    expr = {}
    expr["z1_next"] = "z3 + z6"
    expr["z2_next"] = "z2 + z3 + z6"
    expr["z3_next"] = "(z3 + z6)^2"
    expr["z4_next"] = "(z3 + z6)^3"
    expr["z5_next"] = "sin(z2 + z3 + z6)"
    expr["z6_next"] = "(z3 + z6)^3 * sin(z2 + z3 + z6)"
    return expr


# ============================================================
# 9) Placeholder for finite truncation:
#    z_{k+1} = K z_k + e_k,  e_k in R
# ============================================================

def trunc6():
    raise NotImplementedError(
        "TODO: derive finite 6D-to-6D Koopman-like update "
        "z_{k+1} = K z_k + e_k, with residual Star R. "
        "This is the current stopping point."
    )


# ============================================================
# 10) Utility printing
# ============================================================

def show_star(name, S, lp_solver="gurobi"):
    print(f"\n========== {name} ==========")
    print("dim   =", S.dim)
    print("nVars =", S.nVars)
    lb, ub = S.getRanges(lp_solver=lp_solver)
    print("ranges lb =", lb)
    print("ranges ub =", ub)
    return lb, ub


def show_expr():
    expr = next_expr()
    print("\n========== Symbolic next-step 6D observables ==========")
    for k, v in expr.items():
        print(f"{k} = {v}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # --------------------------------
    # A. Build graph
    # --------------------------------
    graph, order = make_graph()
    print("Topo order:", [node.id for node in order])

    # --------------------------------
    # B. Initial state
    # --------------------------------
    lb = [-0.1, -0.07]
    ub = [0.1, 0.14]
    S0 = make_star(lb, ub)
    l0, u0 = show_star("Initial State Star S_k (x, theta)", S0)

    # --------------------------------
    # C. Lift to 6D observable Star
    # --------------------------------
    S0_hat, idx0, exe = lift_star(
        S0, order, lp_solver="gurobi", RF=0.0, verbose=True
    )

    lh0, uh0 = show_star("Lifted 6D Star S_hat_k", S0_hat)
    print("idx_map =", idx0)
    print("Lifted Star object:")
    print(S0_hat)

    # --------------------------------
    # D. State propagation from lifted 6D Star
    # --------------------------------
    K, b = make_k(idx0, S0_hat.dim)

    print("\n========== State transition from 6D observable ==========")
    print("K shape =", K.shape)
    print(K)

    S1 = prop_state(S0_hat, K, b)
    l1, u1 = show_star("Propagated State Star S_k1", S1)
    print("S_k1 Star object:")
    print(S1)

    # --------------------------------
    # E. Re-lift next-step state
    # --------------------------------
    S1_hat, idx1 = relift(S1, order, exe)
    lh1, uh1 = show_star("Re-lifted 6D Star S_hat_k1", S1_hat)
    print("idx_map_k1 =", idx1)

    # --------------------------------
    # F. Ground truth comparison
    # --------------------------------
    (gt_x_min, gt_x_max), (gt_t_min, gt_t_max) = gt_box(lb, ub)

    print("\n========== Ground Truth (grid sampling) ==========")
    print(f"GT x_(k+1)     in [{gt_x_min:.6f}, {gt_x_max:.6f}]")
    print(f"GT theta_(k+1) in [{gt_t_min:.6f}, {gt_t_max:.6f}]")

    print("\n========== Comparison (Star vs GT) ==========")
    print(f"Star x_(k+1)     in [{l1[0]:.6f}, {u1[0]:.6f}]")
    print(f"Star theta_(k+1) in [{l1[1]:.6f}, {u1[1]:.6f}]")

    print("\nOver-approx gaps:")
    print("x lower gap     =", l1[0] - gt_x_min)
    print("x upper gap     =", gt_x_max - u1[0])
    print("theta lower gap =", l1[1] - gt_t_min)
    print("theta upper gap =", gt_t_max - u1[1])

    # --------------------------------
    # G. Show current stopping point
    # --------------------------------
    show_expr()

    print("\n========== Current stopping point ==========")
    print("Need to derive a finite 6D-to-6D model:")
    print("    z_{k+1} = K z_k + e_k,  e_k in R")
    print("for the following coupled terms:")
    print("    (z3 + z6)^2")
    print("    (z3 + z6)^3")
    print("    sin(z2 + z3 + z6)")
    print("    (z3 + z6)^3 * sin(z2 + z3 + z6)")