"""
Example test for expression DAG + lifting + Koopman affine propagation + ground truth compare.
Author: Zhuoyang Zhou
Date: 03/02/2026
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
# 0) Define nonlinear system (for ground truth comparison)
#    x_{k+1} = x^2 + x^3*sin(theta)
#    theta_{k+1} = theta
# ============================================================

# ============================================================
# 1) Build expression DAG:
#    x2 = x^2
#    x3 = x^3
#    sin_t = sin(theta)
#    mul = x3 * sin_t
#    final = x2 + mul   (this equals x_{k+1})
# ============================================================

n1 = ExprNode(id="x", op="var", params={"name": "x"})
n2 = ExprNode(id="theta", op="var", params={"name": "theta"})
n3 = ExprNode(id="x2", op="powEven", inputs=["x"], params={"n": 2})
n4 = ExprNode(id="x3", op="powOdd", inputs=["x"], params={"n": 3})
n5 = ExprNode(id="sin_t", op="sin", inputs=["theta"])
n6 = ExprNode(id="mul", op="mul", inputs=["x3", "sin_t"])
n7 = ExprNode(id="final", op="add", inputs=["x2", "mul"])

graph = ExprGraph([n1, n2, n3, n4, n5, n6, n7])
exec_order = graph.topoSort()
print("Topo order:", [node.id for node in exec_order])


# ============================================================
# 2) Initialize state set S_k = Star(x_k, theta_k)
# ============================================================

lb = np.array([-0.1, -0.07], dtype=np.float64)   # [x, theta]
ub = np.array([ 0.1,  0.14], dtype=np.float64)
S_k = Star(lb, ub)

print("\n========== Initial State Star S_k (x, theta) ==========")
print("dim   =", S_k.dim)
print("nVars =", S_k.nVars)
l0, u0 = S_k.getRanges(lp_solver="gurobi")
print("ranges lb =", l0)
print("ranges ub =", u0)


# ============================================================
# 3) Lift to observable space: construct lifted Star Ŝ_k = Star(z_k)
# ============================================================

executor = LiftExecutor(lpSolver="gurobi", RF=0.0, verbose=True)

S_hat_k, idToIndex = executor.run(
    initialStar=S_k,
    orderedNodes=exec_order,
    varIndexMap={"x": 0, "theta": 1},
)

print("\n========== Lifted Star Ŝ_k ==========")
print("idToIndex =", idToIndex)
print("dim   =", S_hat_k.dim)
print("nVars =", S_hat_k.nVars)

l_hat, u_hat = S_hat_k.getRanges(lp_solver="gurobi")
print("lifted ranges lb =", l_hat)
print("lifted ranges ub =", u_hat)

print("\nLifted Star object:")
print(S_hat_k)


# ============================================================
# 4) Define Koopman-based lifting model (manual)
#    z_{k+1} ≈ K z_k + e_k
#
# Here we only propagate the STATE (x_{k+1}, theta_{k+1}) by selecting
# components from z_k:
#   x_{k+1} = final = z[ idToIndex['final'] ]
#   theta_{k+1} = theta = z[ idToIndex['theta'] ]
#
# So define K_state as a 2 x dim(z) matrix:
#   [x_{k+1}]     [0 ... 1(at 'final') ... 0] [z_k]
#   [theta_{k+1}] = [0 ... 1(at 'theta') ... 0] [z_k]
# ============================================================

zDim = S_hat_k.dim
K_state = np.zeros((2, zDim), dtype=np.float64)

idx_final = idToIndex["final"]
idx_theta = idToIndex["theta"]

K_state[0, idx_final] = 1.0   # x_{k+1} = final
K_state[1, idx_theta] = 1.0   # theta_{k+1} = theta

b_state = np.zeros((2,), dtype=np.float64)      # no offset
# residual e_k ignored in this test (set to 0)


print("\n========== Koopman-based state lifting model ==========")
print("K_state shape =", K_state.shape)
print("K_state row0 selects 'final' index =", idx_final)
print("K_state row1 selects 'theta' index =", idx_theta)


# ============================================================
# 5) Lifted affine propagation:
#    S_{k+1} = K_state * Ŝ_k + b_state   (residual ignored)
# ============================================================

S_k1 = S_hat_k.affineMap(K_state, b_state)

print("\n========== Propagated State Star S_{k+1} ==========")
print("dim   =", S_k1.dim)
print("nVars =", S_k1.nVars)

l1, u1 = S_k1.getRanges(lp_solver="gurobi")
print("S_{k+1} ranges lb =", l1)
print("S_{k+1} ranges ub =", u1)

print("\nS_{k+1} Star object:")
print(S_k1)


# ============================================================
# 6) Update lifted reachable set for timestep k+1:
#    Ŝ_{k+1} = Lift(S_{k+1})
# ============================================================

S_hat_k1, idToIndex_k1 = executor.run(
    initialStar=S_k1,                 # now state is (x_{k+1}, theta_{k+1})
    orderedNodes=exec_order,
    varIndexMap={"x": 0, "theta": 1},
)

print("\n========== Updated Lifted Star Ŝ_{k+1} ==========")
print("idToIndex_k1 =", idToIndex_k1)
print("dim   =", S_hat_k1.dim)
print("nVars =", S_hat_k1.nVars)

l_hat1, u_hat1 = S_hat_k1.getRanges(lp_solver="gurobi")
print("Ŝ_{k+1} lifted ranges lb =", l_hat1)
print("Ŝ_{k+1} lifted ranges ub =", u_hat1)


# ============================================================
# 7) Compare next-step state bounds vs ground truth (sampling)
#    Ground truth (approx) by grid sampling over initial box:
#       x in [-1,1], theta in [-0.7,0.7]
#       x_{k+1} = x^2 + x^3*sin(theta)
#       theta_{k+1} = theta
# ============================================================

# sampling resolution (increase if you want tighter GT approximation)
Nx = 401
Nt = 401

xs = np.linspace(lb[0], ub[0], Nx)
ths = np.linspace(lb[1], ub[1], Nt)

X, T = np.meshgrid(xs, ths, indexing="ij")
X_next = X**2 + (X**3) * np.sin(T)
T_next = T

gt_x_min = float(np.min(X_next))
gt_x_max = float(np.max(X_next))
gt_t_min = float(np.min(T_next))
gt_t_max = float(np.max(T_next))

print("\n========== Ground Truth (grid sampling) ==========")
print(f"GT x_(k+1) in [{gt_x_min:.6f}, {gt_x_max:.6f}]")
print(f"GT theta_(k+1) in [{gt_t_min:.6f}, {gt_t_max:.6f}]")

print("\n========== Comparison (Star vs GT) ==========")
print(f"Star x_(k+1) in [{l1[0]:.6f}, {u1[0]:.6f}]")
print(f"Star theta_(k+1) in [{l1[1]:.6f}, {u1[1]:.6f}]")

print("\nOver-approx gaps:")
print("x lower gap  =", l1[0] - gt_x_min)
print("x upper gap  =", gt_x_max - u1[0])
print("theta lower gap =", l1[1] - gt_t_min)
print("theta upper gap =", gt_t_max - u1[1])