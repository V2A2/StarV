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
##   This program is licensed under the BSD 3-Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
CAV 2027 Artifact Evaluation & Benchmark Suite:
  "Approximate ProbStar Reachability and Quantitative Verification
   for Neural Network Control Systems"

Evaluates Closed-Loop Approximate ProbStar Reachability, Cross-Step Analytic
Moment Propagation, Dimension-Invariant Initial-Space Realization (R^{n_0})
for Safety and Bounded Temporal Logic (dProbStarTL), and Sensitivity-Guided
Adaptive Initial-Space Refinement (Closed-Loop CEGAR) across:
  - Suite 1: Multi-Step Horizon Scalability on Closed-Loop DLNNCS (ReLU 2-5-5-1)
             and Pretrained 7D Adaptive Cruise Control (ACC controller_3_20 & controller_5_20)
  - Suite 2: Smooth Nonlinear (TanSig, LogSig), Heterogeneous MixedActivationLayer,
             Pretrained Dual-Network Advanced Emergency Braking (AEBS), and Neural-Plant DynNN_NNCS
  - Suite 3: Bounded Temporal Logic (dProbStarTL) Verification in Initial Space R^{n_0}
  - Suite 4: Closed-Loop Initial-Space CEGAR Monotonic Contraction (K = 1..32)

Usage (from repository root):
  # Kick-the-tires smoke test (~8 seconds, verifies all 4 suites end-to-end):
  python -m artifacts.CAV2027_Approx_ProbStar_NNCS.CAV2027_Approx_ProbStar_NNCS --suite smoke

  # Run individual benchmark suites or the full evaluation:
  python -m artifacts.CAV2027_Approx_ProbStar_NNCS.CAV2027_Approx_ProbStar_NNCS --suite 1
  python -m artifacts.CAV2027_Approx_ProbStar_NNCS.CAV2027_Approx_ProbStar_NNCS --suite 2
  python -m artifacts.CAV2027_Approx_ProbStar_NNCS.CAV2027_Approx_ProbStar_NNCS --suite 3
  python -m artifacts.CAV2027_Approx_ProbStar_NNCS.CAV2027_Approx_ProbStar_NNCS --suite 4
  python -m artifacts.CAV2027_Approx_ProbStar_NNCS.CAV2027_Approx_ProbStar_NNCS --suite all

  # Regenerate CSV tables, LaTeX tables, and PNG/PDF figures from saved JSON results:
  python -m artifacts.CAV2027_Approx_ProbStar_NNCS.CAV2027_Approx_ProbStar_NNCS --from-cached
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from StarV.net.network import NeuralNetwork, rand_ffnn
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.TanSigLayer import TanSigLayer
from StarV.layer.LogSigLayer import LogSigLayer
from StarV.layer.MixedActivationLayer import MixedActivationLayer
from StarV.plant.dlode import DLODE
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.nncs.nncs import (
    NNCS,
    AEBS_NNCS,
    DynNN_NNCS,
    ReachPRM_NNCS,
    ReachPRM_DYNNN_NNCS,
    VerifyPRM_NNCS,
    reachBFS_DLNNCS,
    reachBFS_AEBS,
    reachBFS_DynNN_NNCS,
    simulate_DLNNCS,
    verifyBFS_DLNNCS,
    verify_temporal_specs_DLNNCS,
)
from StarV.spec.dProbStarTL import (
    _ALWAYS_,
    _EVENTUALLY_,
    _LeftBracket_,
    _RightBracket_,
    AtomicPredicate,
    Formula,
)
from StarV.util.load import (
    load_acc_trapezius_model,
    load_AEBS_model,
)


# =========================================================================
# Helper Functions: Sampling, Monte Carlo Ground Truth, and Soundness Audit
# =========================================================================

def make_probstar(lb, ub, sigma_scale=2.5):
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    S = Star(lb, ub)
    mu = 0.5 * (S.pred_lb + S.pred_ub)
    sig = np.maximum((mu - S.pred_lb) / sigma_scale, 1e-4)
    Sig = np.diag(np.square(sig))
    return ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)


def sample_truncated_gaussian(X0, n_samples=20000, seed=42):
    rng = np.random.default_rng(seed)
    mu = X0.mu
    std = np.sqrt(np.maximum(np.diag(X0.Sig), 0.0))
    lb = X0.pred_lb
    ub = X0.pred_ub
    dim = len(mu)
    accepted = []
    batch = max(n_samples * 2, 10000)
    while sum(len(a) for a in accepted) < n_samples:
        z = rng.normal(loc=mu, scale=std, size=(batch, dim))
        mask = np.ones(batch, dtype=bool)
        for j in range(dim):
            if std[j] <= 1e-3:
                z[:, j] = mu[j]
            else:
                mask &= (z[:, j] >= lb[j] - 1e-12) & (z[:, j] <= ub[j] + 1e-12)
        if np.any(mask):
            accepted.append(z[mask])
    pred_samples = np.vstack(accepted)[:n_samples]
    state_samples = X0.V[:, 0] + pred_samples @ X0.V[:, 1:dim + 1].T
    p_input_est = X0.get_input_probability()
    return state_samples, p_input_est


def check_envelope_soundness(ncs, X0, refInputs, RX_approx, n_check=200, seed=101):
    """Check that all simulated trajectories stay strictly inside the approximate ProbStar state bounds."""
    samples, _ = sample_truncated_gaussian(X0, n_samples=n_check, seed=seed)
    num_steps = len(RX_approx) - 2
    violations = 0
    total_checks = 0
    step_bounds = []
    for k in range(len(RX_approx)):
        S_k = RX_approx[k][0]
        lb_k, ub_k = S_k.estimateRanges()
        step_bounds.append((lb_k, ub_k))

    for i in range(n_check):
        trace = simulate_DLNNCS(ncs, samples[i], refInputs, num_steps)
        for k, x_k in enumerate(trace):
            lb_k, ub_k = step_bounds[k]
            total_checks += 1
            if np.any(x_k < lb_k - 1e-5) or np.any(x_k > ub_k + 1e-5):
                violations += 1
    return violations, total_checks


def mc_safety_prob(ncs, X0, refInputs, unsafe_mat, unsafe_vec, num_steps, n_mc=50000, seed=42):
    """Estimate step-wise unsafe probability via Monte Carlo over truncated Gaussian initial states."""
    samples, p_in = sample_truncated_gaussian(X0, n_samples=n_mc, seed=seed)
    step_hits = np.zeros(num_steps + 2, dtype=np.float64)
    for i in range(n_mc):
        trace = simulate_DLNNCS(ncs, samples[i], refInputs, num_steps)
        for k, x_k in enumerate(trace):
            if np.all(unsafe_mat @ x_k <= unsafe_vec + 1e-12):
                step_hits[k] += 1.0
    return (step_hits / n_mc) * p_in


# =========================================================================
# Closed-Loop Benchmark System Builders (Matching CAV 2027 Paper)
# =========================================================================

def build_dlnncs_benchmark(seed=7):
    """2D lateral / double-integrator DLNNCS with [2->5->5->1] ReLU controller."""
    rng = np.random.default_rng(seed)
    dt = 0.1
    A = np.array([[1.0, dt], [-0.15, 0.92]])
    B = np.array([[0.5 * dt * dt], [dt]])
    C = np.eye(2)
    D = np.zeros((2, 1))
    plant = DLODE(A, B, C, D)

    W1 = rng.normal(0.0, 0.45, size=(5, 2))
    b1 = rng.normal(0.0, 0.12, size=(5,))
    W2 = rng.normal(0.0, 0.40, size=(5, 5))
    b2 = rng.normal(0.0, 0.12, size=(5,))
    W3 = rng.normal(0.0, 0.35, size=(1, 5))
    b3 = np.array([-0.05])
    layers = [
        FullyConnectedLayer([W1, b1]),
        ReLULayer(),
        FullyConnectedLayer([W2, b2]),
        ReLULayer(),
        FullyConnectedLayer([W3, b3]),
    ]
    net = NeuralNetwork(layers, net_type='dlnncs_relu_2_5_5_1')
    sys_nncs = NNCS(net, plant, type='DLNNCS')
    X0 = make_probstar([0.2, -0.3], [0.8, 0.3], sigma_scale=2.5)
    unsafe_mat = np.array([[-1.0, 0.0]])
    unsafe_vec = np.array([-0.55])
    return sys_nncs, X0, [], unsafe_mat, unsafe_vec


def build_smooth_nncs_benchmark(act='tanh', seed=13):
    """Smooth nonlinear NNCS with [2->8->8->1] TanSig or LogSig controller."""
    rng = np.random.default_rng(seed)
    dt = 0.1
    A = np.array([[1.0, dt], [0.25 * dt, 0.95]])
    B = np.array([[0.5 * dt * dt], [dt]])
    C = np.eye(2)
    D = np.zeros((2, 1))
    plant = DLODE(A, B, C, D)

    W1 = rng.normal(0.0, 0.55, size=(8, 2))
    b1 = rng.normal(0.0, 0.10, size=(8,))
    W2 = rng.normal(0.0, 0.45, size=(8, 8))
    b2 = rng.normal(0.0, 0.10, size=(8,))
    W3 = np.array([[-0.6, -0.4, 0.3, -0.5, 0.2, -0.3, 0.4, -0.2]])
    b3 = np.array([-0.08])

    ActLayer = TanSigLayer if act == 'tanh' else LogSigLayer
    layers = [
        FullyConnectedLayer([W1, b1]),
        ActLayer(),
        FullyConnectedLayer([W2, b2]),
        ActLayer(),
        FullyConnectedLayer([W3, b3]),
    ]
    net = NeuralNetwork(layers, net_type=f'smooth_{act}_2_8_8_1')
    sys_nncs = NNCS(net, plant, type='DLNNCS')
    X0 = make_probstar([0.15, -0.25], [0.65, 0.25], sigma_scale=2.5)
    unsafe_mat = np.array([[-1.0, 0.0]])
    unsafe_vec = np.array([-0.46])
    return sys_nncs, X0, [], unsafe_mat, unsafe_vec


def build_mixed_nncs_benchmark(seed=21):
    """MixedActivationLayer NNCS combining ReLU, TanSig, LogSig, SatLin, SatLins, and PureLin."""
    rng = np.random.default_rng(seed)
    dt = 0.1
    A = np.array([[1.0, dt], [-0.1, 0.90]])
    B = np.array([[0.5 * dt * dt], [dt]])
    C = np.eye(2)
    D = np.zeros((2, 1))
    plant = DLODE(A, B, C, D)

    W1 = rng.normal(0.0, 0.45, size=(6, 2))
    b1 = rng.normal(0.0, 0.1, size=(6,))
    acts1 = np.array(['poslin', 'poslin', 'tansig', 'tansig', 'satlins', 'purelin'], dtype=object)
    W2 = rng.normal(0.0, 0.40, size=(6, 6))
    b2 = rng.normal(0.0, 0.1, size=(6,))
    acts2 = np.array(['poslin', 'tansig', 'logsig', 'satlin', 'poslin', 'tansig'], dtype=object)
    W3 = rng.normal(0.0, 0.35, size=(1, 6))
    b3 = np.array([-0.02])

    layers = [
        FullyConnectedLayer([W1, b1]),
        MixedActivationLayer(acts1),
        FullyConnectedLayer([W2, b2]),
        MixedActivationLayer(acts2),
        FullyConnectedLayer([W3, b3]),
    ]
    net = NeuralNetwork(layers, net_type='mixed_act_2_6_6_1')
    sys_nncs = NNCS(net, plant, type='DLNNCS')
    X0 = make_probstar([0.1, -0.2], [0.6, 0.2], sigma_scale=2.5)
    unsafe_mat = np.array([[-1.0, 0.0]])
    unsafe_vec = np.array([-0.42])
    return sys_nncs, X0, [], unsafe_mat, unsafe_vec


# =========================================================================
# Functional Smoke Test + Benchmark Suites 1 - 4
# =========================================================================

def run_smoke_test():
    """
    Fast kick-the-tires functional evaluation (~10s) exercising all 4 suites:
      1. Closed-Loop ReLU DLNNCS & Pretrained 7D ACC Approx-ProbStar vs. Exact ProbStar
      2. Smooth TanSig, MixedActivationLayer, Pretrained AEBS, and DynNN_NNCS
      3. Initial-Space Dual dProbStarTL Verification
      4. Closed-Loop Initial-Space CEGAR Contraction
    """
    print("\n=====================================================================")
    print(" CAV 2027 NNCS Artifact — Kick-the-Tires Functional Smoke Test")
    print("=====================================================================", flush=True)
    t_start = time.perf_counter()

    # 1. DLNNCS Exact vs. Approx-ProbStar at T=1
    sys_dl, X0_dl, ref_dl, umat_dl, uvec_dl = build_dlnncs_benchmark()
    v_ex = VerifyPRM_NNCS()
    v_ex.initSet = copy.deepcopy(X0_dl)
    v_ex.numSteps = 1
    v_ex.refInputs = ref_dl
    v_ex.method = 'exact-probstar'
    v_ex.unsafeSpec = [umat_dl, uvec_dl]
    v_ex.show = False
    res_ex = verifyBFS_DLNNCS(sys_dl, v_ex)

    v_ap = VerifyPRM_NNCS()
    v_ap.initSet = copy.deepcopy(X0_dl)
    v_ap.numSteps = 1
    v_ap.refInputs = ref_dl
    v_ap.method = 'approx-probstar'
    v_ap.unsafeSpec = [umat_dl, uvec_dl]
    v_ap.show = False
    res_ap = verifyBFS_DLNNCS(sys_dl, v_ap)
    p_ex = float(res_ex.Qt[-1])
    p_lb, p_ub = float(res_ap.Qt_lb[-1]), float(res_ap.Qt_ub[-1])
    assert p_lb - 1e-4 <= p_ex <= p_ub + 1e-4
    print(f"  [PASS] Suite 1 (DLNNCS T=1): Exact p*={p_ex:.5f} ({len(res_ex.RX[-1])} sets) "
          f"in [{p_lb:.5f}, {p_ub:.5f}] (1 set)")

    # 2. Smooth TanSig, Pretrained AEBS, and DynNN_NNCS
    sys_sm, X0_sm, ref_sm, umat_sm, uvec_sm = build_smooth_nncs_benchmark('tanh')
    v_sm = VerifyPRM_NNCS()
    v_sm.initSet = copy.deepcopy(X0_sm)
    v_sm.numSteps = 3
    v_sm.method = 'approx-probstar'
    v_sm.unsafeSpec = [umat_sm, uvec_sm]
    v_sm.show = False
    res_sm = verifyBFS_DLNNCS(sys_sm, v_sm)
    viol_sm, chk_sm = check_envelope_soundness(sys_sm, X0_sm, ref_sm, res_sm.RX, n_check=100)
    assert viol_sm == 0

    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
    aebs_sys = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)
    v_aebs = VerifyPRM_NNCS()
    v_aebs.initSet = copy.deepcopy(initSets[0])
    v_aebs.numSteps = 2
    v_aebs.method = 'approx-probstar'
    v_aebs.unsafeSpec = [np.array([[1.0, 0.0, 0.0]]), np.array([93.0])]
    v_aebs.show = False
    res_aebs = verifyBFS_DLNNCS(aebs_sys, v_aebs)

    np.random.seed(42)
    dyn_sys = DynNN_NNCS(rand_ffnn([2, 6, 1], ['relu', 'purelin']), rand_ffnn([3, 8, 2], ['relu', 'purelin']))
    r_dyn = ReachPRM_DYNNN_NNCS()
    r_dyn.initSet = make_probstar([-0.4, -0.4], [0.4, 0.4], sigma_scale=2.5)
    r_dyn.numSteps = 2
    r_dyn.method = 'approx-probstar'
    r_dyn.show = False
    RX_dyn, _ = reachBFS_DynNN_NNCS(dyn_sys, r_dyn)
    print(f"  [PASS] Suite 2 (Smooth TanSig, AEBS, DynNN_NNCS): "
          f"nVarsT={res_sm.RX[-1][0].nVars}->nVars0={res_sm.RX[-1][0].nVars0}, "
          f"violations={viol_sm}/{chk_sm}, |RX_AEBS|={len(res_aebs.RX[-1])}, |RX_DynNN|={len(RX_dyn[-1])}")

    # 3. Initial-Space Dual dProbStarTL
    AW = _ALWAYS_(0, 2)
    phi_aw = Formula([AW, _LeftBracket_(), AtomicPredicate(np.array([1.0, 0.0]), np.array([0.55])), _RightBracket_()])
    v_tl = VerifyPRM_NNCS()
    v_tl.initSet = copy.deepcopy(X0_dl)
    v_tl.numSteps = 2
    v_tl.refInputs = ref_dl
    v_tl.method = 'approx-probstar'
    v_tl.temporalSpecs = [phi_aw]
    v_tl.show = False
    _, p_max_tl, p_min_tl, _, _, _ = verify_temporal_specs_DLNNCS(sys_dl, v_tl)
    assert 0.0 <= p_min_tl[0] <= p_max_tl[0] <= 1.0
    print(f"  [PASS] Suite 3 (dProbStarTL Always_[0,2]): [{p_min_tl[0]:.5f}, {p_max_tl[0]:.5f}] in R^2")

    # 4. Closed-Loop Initial-Space CEGAR Contraction
    v_ref = VerifyPRM_NNCS()
    v_ref.initSet = copy.deepcopy(X0_sm)
    v_ref.numSteps = 3
    v_ref.method = 'approx-probstar'
    v_ref.refine_max_splits = 3
    v_ref.refine_tol = 1e-6
    v_ref.unsafeSpec = [umat_sm, uvec_sm]
    v_ref.show = False
    res_ref = verifyBFS_DLNNCS(sys_sm, v_ref)
    gap_1 = float(res_sm.Qt_ub[-1] - res_sm.Qt_lb[-1])
    gap_4 = float(res_ref.Qt_ub[-1] - res_ref.Qt_lb[-1])
    assert gap_4 <= gap_1 + 1e-9
    print(f"  [PASS] Suite 4 (Closed-Loop CEGAR): Gap contracted {gap_1:.5f} (K=1) -> {gap_4:.5f} (K={res_ref.num_partitions})")
    print(f"\nAll functional smoke checks PASSED in {time.perf_counter() - t_start:.2f}s!")


def run_suite1_horizon_scalability():
    print("\n=== Running Suite 1: Horizon Scalability & Soundness (DLNNCS & ACC) ===", flush=True)
    results = []
    sys_dl, X0_dl, ref_dl, umat_dl, uvec_dl = build_dlnncs_benchmark()

    for T in [1, 2, 3, 5, 10, 15, 20, 30]:
        print(f"  DLNNCS T={T}...", flush=True)
        p_mc_vec = mc_safety_prob(sys_dl, X0_dl, ref_dl, umat_dl, uvec_dl, T, n_mc=3000)
        p_mc_last = float(p_mc_vec[-1])

        exact_rec = None
        if T <= 2:
            vprm_ex = VerifyPRM_NNCS()
            vprm_ex.initSet = copy.deepcopy(X0_dl)
            vprm_ex.numSteps = T
            vprm_ex.refInputs = ref_dl
            vprm_ex.method = 'exact-probstar'
            vprm_ex.pf = 0.0
            vprm_ex.show = False
            vprm_ex.unsafeSpec = [umat_dl, uvec_dl]
            t0 = time.perf_counter()
            res_ex = verifyBFS_DLNNCS(sys_dl, vprm_ex)
            t_ex = time.perf_counter() - t0
            exact_rec = {
                'num_sets_last': len(res_ex.RX[-1]),
                'total_sets': sum(len(rk) for rk in res_ex.RX),
                'p_exact_last': float(res_ex.Qt[-1]),
                'time_s': t_ex,
            }

        filt_rec = None
        if T <= 3:
            vprm_pf = VerifyPRM_NNCS()
            vprm_pf.initSet = copy.deepcopy(X0_dl)
            vprm_pf.numSteps = T
            vprm_pf.refInputs = ref_dl
            vprm_pf.method = 'exact-probstar'
            vprm_pf.pf = 0.02
            vprm_pf.show = False
            vprm_pf.unsafeSpec = [umat_dl, uvec_dl]
            t0 = time.perf_counter()
            res_pf = verifyBFS_DLNNCS(sys_dl, vprm_pf)
            t_pf = time.perf_counter() - t0
            filt_rec = {
                'num_sets_last': len(res_pf.RX[-1]),
                'total_sets': sum(len(rk) for rk in res_pf.RX),
                'p_lb_last': float(res_pf.Qt[-1]),
                'p_ub_last': float(res_pf.Qt_max[-1]),
                'p_ignored': float(res_pf.p_ignored[-1]),
                'time_s': t_pf,
            }

        vprm_ap = VerifyPRM_NNCS()
        vprm_ap.initSet = copy.deepcopy(X0_dl)
        vprm_ap.numSteps = T
        vprm_ap.refInputs = ref_dl
        vprm_ap.method = 'approx-probstar'
        vprm_ap.RF = 1.0
        vprm_ap.relu_approx_mode = 'analytic_gaussian'
        vprm_ap.show = False
        vprm_ap.unsafeSpec = [umat_dl, uvec_dl]
        t0 = time.perf_counter()
        res_ap = verifyBFS_DLNNCS(sys_dl, vprm_ap)
        t_ap = time.perf_counter() - t0
        viol, total_chk = check_envelope_soundness(sys_dl, X0_dl, ref_dl, res_ap.RX, n_check=200)

        results.append({
            'benchmark': 'DLNNCS_ReLU_2_5_5_1',
            'T': T,
            'steps_total': T + 1,
            'nVars0': int(res_ap.RX[-1][0].nVars0),
            'nVarsT': int(res_ap.RX[-1][0].nVars),
            'p_mc_last': p_mc_last,
            'exact': exact_rec,
            'filtered_pf005': filt_rec,
            'approx_single': {
                'num_sets_last': len(res_ap.RX[-1]),
                'p_min_last': float(res_ap.Qt_lb[-1]),
                'p_hat_last': float(res_ap.Qt[-1]),
                'p_max_last': float(res_ap.Qt_ub[-1]),
                'time_s': t_ap,
                'soundness_violations': viol,
                'soundness_checks': total_chk,
            }
        })

    for netname in ['controller_3_20', 'controller_5_20']:
        acc_sys, X0_acc, ref_acc, _, _ = load_acc_trapezius_model(netname=netname, a_lead=-5.0)
        umat_eval = np.array([[1.0, 0.0, 0.0, -1.0, -1.4, 0.0, 0.0]])
        uvec_eval = np.array([38.0])
        for T in [1, 2, 5, 10, 15, 20]:
            print(f"  ACC ({netname}) T={T}...", flush=True)
            p_mc_vec = mc_safety_prob(acc_sys, X0_acc, ref_acc, umat_eval, uvec_eval, T, n_mc=1500)
            p_mc_last = float(p_mc_vec[-1])

            exact_rec = None
            if T <= 2 and netname == 'controller_3_20':
                vprm_ex = VerifyPRM_NNCS()
                vprm_ex.initSet = copy.deepcopy(X0_acc)
                vprm_ex.numSteps = T
                vprm_ex.refInputs = ref_acc
                vprm_ex.method = 'exact-probstar'
                vprm_ex.pf = 0.0
                vprm_ex.show = False
                vprm_ex.unsafeSpec = [umat_eval, uvec_eval]
                t0 = time.perf_counter()
                res_ex = verifyBFS_DLNNCS(acc_sys, vprm_ex)
                t_ex = time.perf_counter() - t0
                exact_rec = {
                    'num_sets_last': len(res_ex.RX[-1]),
                    'total_sets': sum(len(rk) for rk in res_ex.RX),
                    'p_exact_last': float(res_ex.Qt[-1]),
                    'time_s': t_ex,
                }

            vprm_ap = VerifyPRM_NNCS()
            vprm_ap.initSet = copy.deepcopy(X0_acc)
            vprm_ap.numSteps = T
            vprm_ap.refInputs = ref_acc
            vprm_ap.method = 'approx-probstar'
            vprm_ap.show = False
            vprm_ap.unsafeSpec = [umat_eval, uvec_eval]
            t0 = time.perf_counter()
            res_ap = verifyBFS_DLNNCS(acc_sys, vprm_ap)
            t_ap = time.perf_counter() - t0
            viol, total_chk = check_envelope_soundness(acc_sys, X0_acc, ref_acc, res_ap.RX, n_check=200)

            results.append({
                'benchmark': f'ACC_{netname}',
                'T': T,
                'steps_total': T + 1,
                'nVars0': int(res_ap.RX[-1][0].nVars0),
                'nVarsT': int(res_ap.RX[-1][0].nVars),
                'p_mc_last': p_mc_last,
                'exact': exact_rec,
                'approx_single': {
                    'num_sets_last': len(res_ap.RX[-1]),
                    'p_min_last': float(res_ap.Qt_lb[-1]),
                    'p_hat_last': float(res_ap.Qt[-1]),
                    'p_max_last': float(res_ap.Qt_ub[-1]),
                    'time_s': t_ap,
                    'soundness_violations': viol,
                    'soundness_checks': total_chk,
                }
            })

    return results


def run_suite2_nonlinear_and_multinet():
    print("\n=== Running Suite 2: Smooth Nonlinear, Mixed-Activation, AEBS, and DynNN_NNCS ===", flush=True)
    results = []

    for act in ['tanh', 'sigmoid']:
        sys_sm, X0_sm, ref_sm, umat_sm, uvec_sm = build_smooth_nncs_benchmark(act=act)
        for T in [5, 10, 20, 30]:
            print(f"  Smooth {act} NNCS T={T}...", flush=True)
            p_mc_vec = mc_safety_prob(sys_sm, X0_sm, ref_sm, umat_sm, uvec_sm, T, n_mc=3000)
            vprm = VerifyPRM_NNCS()
            vprm.initSet = copy.deepcopy(X0_sm)
            vprm.numSteps = T
            vprm.refInputs = ref_sm
            vprm.method = 'approx-probstar'
            vprm.show = False
            vprm.unsafeSpec = [umat_sm, uvec_sm]
            t0 = time.perf_counter()
            res = verifyBFS_DLNNCS(sys_sm, vprm)
            t_el = time.perf_counter() - t0
            viol, _ = check_envelope_soundness(sys_sm, X0_sm, ref_sm, res.RX, n_check=200)
            results.append({
                'system': f'Smooth_{act.upper()}_NNCS',
                'T': T,
                'nVars0': int(res.RX[-1][0].nVars0),
                'nVarsT': int(res.RX[-1][0].nVars),
                'p_min_last': float(res.Qt_lb[-1]),
                'p_hat_last': float(res.Qt[-1]),
                'p_max_last': float(res.Qt_ub[-1]),
                'p_mc_last': float(p_mc_vec[-1]),
                'time_s': t_el,
                'soundness_violations': viol,
            })

    sys_mx, X0_mx, ref_mx, umat_mx, uvec_mx = build_mixed_nncs_benchmark()
    for T in [5, 10, 20]:
        print(f"  MixedActivation NNCS T={T}...", flush=True)
        p_mc_vec = mc_safety_prob(sys_mx, X0_mx, ref_mx, umat_mx, uvec_mx, T, n_mc=3000)
        vprm = VerifyPRM_NNCS()
        vprm.initSet = copy.deepcopy(X0_mx)
        vprm.numSteps = T
        vprm.refInputs = ref_mx
        vprm.method = 'approx-probstar'
        vprm.show = False
        vprm.unsafeSpec = [umat_mx, uvec_mx]
        t0 = time.perf_counter()
        res = verifyBFS_DLNNCS(sys_mx, vprm)
        t_el = time.perf_counter() - t0
        viol, _ = check_envelope_soundness(sys_mx, X0_mx, ref_mx, res.RX, n_check=200)
        results.append({
            'system': 'MixedActivation_NNCS',
            'T': T,
            'nVars0': int(res.RX[-1][0].nVars0),
            'nVarsT': int(res.RX[-1][0].nVars),
            'p_min_last': float(res.Qt_lb[-1]),
            'p_hat_last': float(res.Qt[-1]),
            'p_max_last': float(res.Qt_ub[-1]),
            'p_mc_last': float(p_mc_vec[-1]),
            'time_s': t_el,
            'soundness_violations': viol,
        })

    # Pretrained AEBS Dual-Network NNCS
    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
    aebs_sys = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)
    umat_aebs = np.array([[1.0, 0.0, 0.0]])
    uvec_aebs = np.array([93.0])
    for T in [2, 3, 5, 10]:
        print(f"  AEBS_NNCS T={T}...", flush=True)
        t_ex = None
        n_sets_ex = None
        if T <= 2:
            vprm_ex = VerifyPRM_NNCS()
            vprm_ex.initSet = copy.deepcopy(initSets[0])
            vprm_ex.numSteps = T
            vprm_ex.method = 'exact-probstar'
            vprm_ex.show = False
            vprm_ex.unsafeSpec = [umat_aebs, uvec_aebs]
            t0 = time.perf_counter()
            res_ex = verifyBFS_DLNNCS(aebs_sys, vprm_ex)
            t_ex = time.perf_counter() - t0
            n_sets_ex = sum(len(rk) for rk in res_ex.RX)

        vprm_ap = VerifyPRM_NNCS()
        vprm_ap.initSet = copy.deepcopy(initSets[0])
        vprm_ap.numSteps = T
        vprm_ap.method = 'approx-probstar'
        vprm_ap.show = False
        vprm_ap.unsafeSpec = [umat_aebs, uvec_aebs]
        t0 = time.perf_counter()
        res_ap = verifyBFS_DLNNCS(aebs_sys, vprm_ap)
        t_ap = time.perf_counter() - t0
        results.append({
            'system': 'AEBS_DualNet_NNCS',
            'T': T,
            'nVars0': int(res_ap.RX[-1][0].nVars0),
            'nVarsT': int(res_ap.RX[-1][0].nVars),
            'p_min_last': float(res_ap.Qt_lb[-1]),
            'p_hat_last': float(res_ap.Qt[-1]),
            'p_max_last': float(res_ap.Qt_ub[-1]),
            'time_s': t_ap,
            'exact_time_s': t_ex,
            'exact_total_sets': n_sets_ex,
        })

    # DynNN_NNCS (Neural Controller + Dynamic Neural Plant)
    np.random.seed(42)
    ctrl_dyn = rand_ffnn([2, 6, 1], ['relu', 'purelin'])
    plant_dyn = rand_ffnn([3, 8, 2], ['relu', 'purelin'])
    dyn_sys = DynNN_NNCS(ctrl_dyn, plant_dyn)
    X0_dyn = make_probstar([-0.4, -0.4], [0.4, 0.4], sigma_scale=2.5)
    for T in [2, 3, 5, 10]:
        print(f"  DynNN_NNCS T={T}...", flush=True)
        rprm_ap = ReachPRM_DYNNN_NNCS()
        rprm_ap.initSet = copy.deepcopy(X0_dyn)
        rprm_ap.numSteps = T
        rprm_ap.method = 'approx-probstar'
        rprm_ap.show = False
        t0 = time.perf_counter()
        RX_ap, _ = reachBFS_DynNN_NNCS(dyn_sys, rprm_ap)
        t_ap = time.perf_counter() - t0

        t_ex = None
        n_sets_ex = None
        if T <= 2:
            rprm_ex = ReachPRM_DYNNN_NNCS()
            rprm_ex.initSet = copy.deepcopy(X0_dyn)
            rprm_ex.numSteps = T
            rprm_ex.method = 'exact-probstar'
            rprm_ex.show = False
            t0 = time.perf_counter()
            RX_ex, _ = reachBFS_DynNN_NNCS(dyn_sys, rprm_ex)
            t_ex = time.perf_counter() - t0
            n_sets_ex = sum(len(rk) for rk in RX_ex)

        results.append({
            'system': 'DynNN_NNCS',
            'T': T,
            'nVars0': int(RX_ap[-1][0].nVars0),
            'nVarsT': int(RX_ap[-1][0].nVars),
            'time_s': t_ap,
            'exact_time_s': t_ex,
            'exact_total_sets': n_sets_ex,
        })

    return results


def run_suite3_temporal_logic():
    print("\n=== Running Suite 3: Bounded Temporal Logic (dProbStarTL) Verification ===", flush=True)
    results = []
    sys_dl, X0_dl, ref_dl, _, _ = build_dlnncs_benchmark()
    samples, p_in = sample_truncated_gaussian(X0_dl, n_samples=5000, seed=42)
    lb_op = _LeftBracket_()
    rb_op = _RightBracket_()

    for T in [2, 3, 5]:
        trajs = np.array([simulate_DLNNCS(sys_dl, x0, ref_dl, T) for x0 in samples])
        p_mc_ev = p_in * float(np.mean(np.any(trajs[:, :T + 1, 0] >= 0.55, axis=1)))
        p_mc_aw = p_in * float(np.mean(np.all(trajs[:, :T + 1, 0] <= 0.55, axis=1)))

        EV = _EVENTUALLY_(0, T)
        AW = _ALWAYS_(0, T)
        P_unsafe = AtomicPredicate(np.array([-1.0, 0.0]), np.array([-0.55]))
        P_safe = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.55]))
        phi_ev = Formula([EV, lb_op, P_unsafe, rb_op])
        phi_aw = Formula([AW, lb_op, P_safe, rb_op])

        for spec_name, spec, p_mc in [
            (f'Eventually_[0,{T}](x0 >= 0.55)', phi_ev, p_mc_ev),
            (f'Always_[0,{T}](x0 <= 0.55)', phi_aw, p_mc_aw),
        ]:
            print(f"  dProbStarTL DLNNCS {spec_name}...", flush=True)
            t_ex = None
            p_ex = None
            n_traces_ex = None
            if T <= 3:
                vprm_ex = VerifyPRM_NNCS()
                vprm_ex.initSet = copy.deepcopy(X0_dl)
                vprm_ex.numSteps = T
                vprm_ex.refInputs = ref_dl
                vprm_ex.method = 'exact-probstar'
                vprm_ex.show = False
                vprm_ex.temporalSpecs = [spec]
                t0 = time.perf_counter()
                traces_ex, _, p_min_ex, _, _, _ = verify_temporal_specs_DLNNCS(sys_dl, vprm_ex)
                t_ex = time.perf_counter() - t0
                p_ex = float(p_min_ex[0])
                n_traces_ex = len(traces_ex)

            vprm_ap = VerifyPRM_NNCS()
            vprm_ap.initSet = copy.deepcopy(X0_dl)
            vprm_ap.numSteps = T
            vprm_ap.refInputs = ref_dl
            vprm_ap.method = 'approx-probstar'
            vprm_ap.show = False
            vprm_ap.temporalSpecs = [spec]
            t0 = time.perf_counter()
            _, p_max_ap, p_min_ap, _, _, _ = verify_temporal_specs_DLNNCS(sys_dl, vprm_ap)
            t_ap = time.perf_counter() - t0

            vprm_ref = VerifyPRM_NNCS()
            vprm_ref.initSet = copy.deepcopy(X0_dl)
            vprm_ref.numSteps = T
            vprm_ref.refInputs = ref_dl
            vprm_ref.method = 'approx-probstar'
            vprm_ref.refine_max_splits = 4
            vprm_ref.refine_tol = 1e-4
            vprm_ref.show = False
            vprm_ref.temporalSpecs = [spec]
            t0 = time.perf_counter()
            _, p_max_ref, p_min_ref, _, _, _ = verify_temporal_specs_DLNNCS(sys_dl, vprm_ref)
            t_ref = time.perf_counter() - t0

            results.append({
                'system': 'DLNNCS',
                'spec': spec_name,
                'T': T,
                'p_exact': p_ex,
                'p_mc': p_mc,
                'exact_traces': n_traces_ex,
                'exact_time_s': t_ex,
                'approx_p_min': float(p_min_ap[0]),
                'approx_p_max': float(p_max_ap[0]),
                'approx_time_s': t_ap,
                'refined_p_min': float(p_min_ref[0]),
                'refined_p_max': float(p_max_ref[0]),
                'refined_time_s': t_ref,
            })

    for T in [2, 5, 8]:
        acc_sys, phi_list, X0_acc, ref_acc = load_acc_trapezius_model(
            netname='controller_3_20', spec_ids=[0, 2], T=T, t=1, a_lead=-5.0
        )
        for idx, spec_label in enumerate([f'ACC_phi1_T{T}', f'ACC_phi2_T{T}']):
            print(f"  dProbStarTL {spec_label}...", flush=True)
            spec = phi_list[idx]
            t_ex = None
            p_ex = None
            n_tr_ex = None
            if T <= 2:
                vprm_ex = VerifyPRM_NNCS()
                vprm_ex.initSet = copy.deepcopy(X0_acc)
                vprm_ex.numSteps = T
                vprm_ex.refInputs = ref_acc
                vprm_ex.method = 'exact-probstar'
                vprm_ex.show = False
                vprm_ex.temporalSpecs = [spec]
                t0 = time.perf_counter()
                tr_ex, _, p_min_ex, _, _, _ = verify_temporal_specs_DLNNCS(acc_sys, vprm_ex)
                t_ex = time.perf_counter() - t0
                p_ex = float(p_min_ex[0])
                n_tr_ex = len(tr_ex)

            vprm_ap = VerifyPRM_NNCS()
            vprm_ap.initSet = copy.deepcopy(X0_acc)
            vprm_ap.numSteps = T
            vprm_ap.refInputs = ref_acc
            vprm_ap.method = 'approx-probstar'
            vprm_ap.show = False
            vprm_ap.temporalSpecs = [spec]
            t0 = time.perf_counter()
            _, p_max_ap, p_min_ap, _, _, _ = verify_temporal_specs_DLNNCS(acc_sys, vprm_ap)
            t_ap = time.perf_counter() - t0

            results.append({
                'system': 'ACC_controller_3_20',
                'spec': spec_label,
                'T': T,
                'p_exact': p_ex,
                'exact_traces': n_tr_ex,
                'exact_time_s': t_ex,
                'approx_p_min': float(p_min_ap[0]),
                'approx_p_max': float(p_max_ap[0]),
                'approx_time_s': t_ap,
            })

    return results


def run_suite4_adaptive_refinement():
    print("\n=== Running Suite 4: Adaptive Initial-Space Refinement (Closed-Loop CEGAR) ===", flush=True)
    results = []

    sys_dl, X0_dl, ref_dl, umat_dl, uvec_dl = build_dlnncs_benchmark()
    vprm_ex = VerifyPRM_NNCS()
    vprm_ex.initSet = copy.deepcopy(X0_dl)
    vprm_ex.numSteps = 3
    vprm_ex.refInputs = ref_dl
    vprm_ex.method = 'exact-probstar'
    vprm_ex.pf = 0.0
    vprm_ex.show = False
    vprm_ex.unsafeSpec = [umat_dl, uvec_dl]
    t0 = time.perf_counter()
    res_ex = verifyBFS_DLNNCS(sys_dl, vprm_ex)
    t_ex = time.perf_counter() - t0
    p_exact_T3 = float(res_ex.Qt[-1])

    for max_splits in [0, 1, 3, 7, 15, 31]:
        print(f"  Refinement DLNNCS T=3 max_splits={max_splits}...", flush=True)
        vprm = VerifyPRM_NNCS()
        vprm.initSet = copy.deepcopy(X0_dl)
        vprm.numSteps = 3
        vprm.refInputs = ref_dl
        vprm.method = 'approx-probstar'
        vprm.refine_max_splits = max_splits
        vprm.refine_tol = 1e-6
        vprm.show = False
        vprm.unsafeSpec = [umat_dl, uvec_dl]
        t0 = time.perf_counter()
        res = verifyBFS_DLNNCS(sys_dl, vprm)
        t_el = time.perf_counter() - t0
        results.append({
            'system': 'DLNNCS_ReLU_T3',
            'max_splits': max_splits,
            'num_partitions': res.num_partitions,
            'p_min': float(res.Qt_lb[-1]),
            'p_hat': float(res.Qt[-1]),
            'p_max': float(res.Qt_ub[-1]),
            'gap': float(res.Qt_ub[-1] - res.Qt_lb[-1]),
            'p_ref': p_exact_T3,
            'exact_time_s': t_ex,
            'exact_sets': len(res_ex.RX[-1]),
            'time_s': t_el,
        })

    sys_sm, X0_sm, ref_sm, umat_sm, uvec_sm = build_smooth_nncs_benchmark(act='tanh')
    p_mc_sm = float(mc_safety_prob(sys_sm, X0_sm, ref_sm, umat_sm, uvec_sm, 6, n_mc=10000)[-1])
    for max_splits in [0, 1, 3, 7, 15, 31]:
        print(f"  Refinement Smooth TanSig NNCS T=6 max_splits={max_splits}...", flush=True)
        vprm = VerifyPRM_NNCS()
        vprm.initSet = copy.deepcopy(X0_sm)
        vprm.numSteps = 6
        vprm.refInputs = ref_sm
        vprm.method = 'approx-probstar'
        vprm.refine_max_splits = max_splits
        vprm.refine_tol = 1e-6
        vprm.show = False
        vprm.unsafeSpec = [umat_sm, uvec_sm]
        t0 = time.perf_counter()
        res = verifyBFS_DLNNCS(sys_sm, vprm)
        t_el = time.perf_counter() - t0
        results.append({
            'system': 'Smooth_TanSig_T6',
            'max_splits': max_splits,
            'num_partitions': res.num_partitions,
            'p_min': float(res.Qt_lb[-1]),
            'p_hat': float(res.Qt[-1]),
            'p_max': float(res.Qt_ub[-1]),
            'gap': float(res.Qt_ub[-1] - res.Qt_lb[-1]),
            'p_ref': p_mc_sm,
            'time_s': t_el,
        })

    return results


# =========================================================================
# CSV, LaTeX, and Publication-Quality Figure Exporters
# =========================================================================

def export_csv_and_latex_tables(all_res, res_dir):
    os.makedirs(res_dir, exist_ok=True)

    tax_rows = [
        ['NNV / CORA / POLAR / Verisig 2.0', 'Qualitative Over-Approx', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No'],
        ['StarV Exact ProbStar', 'Quantitative Exact Split', 'Yes', 'Exact p*', 'No (ReLU only)', 'Yes', 'Yes (O(2^sum u_t))', 'No'],
        ['StarV Filtered ProbStar', 'Quantitative Branch Pruning', 'Yes', 'Loose [p_lb, p_ub]', 'No (ReLU only)', 'Yes', 'Yes (High-dim MVN)', 'No'],
        ['Monte Carlo Sampling', 'Statistical Empirical', 'No', 'Empirical CI', 'Yes', 'Yes', 'Empirical', 'No'],
        ['StarV Approx-ProbStar + CEGAR [Ours]', 'Quantitative Envelope + R^{n0} CEGAR', 'Yes', 'Tight [p_min, p_max] + p_hat', 'Yes (TanSig/LogSig/Mixed)', 'Yes (DLNNCS/ACC/AEBS/DynNN)', 'Yes (Single-Trace R^{n0})', 'Yes (Anytime -> 0)'],
    ]
    with open(os.path.join(res_dir, 'verifier_taxonomy.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Verifier', 'Paradigm', 'Sound_Guarantee', 'Quant_Prob_Bounds', 'Smooth_Mixed_Ctrl', 'MultiNet_DynNN', 'dProbStarTL', 'Anytime_CEGAR'])
        w.writerows(tax_rows)

    if 'suite1' in all_res:
        with open(os.path.join(res_dir, 'suite1_horizon_scalability.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'benchmark', 'T', 'nVars0', 'nVarsT', 'p_mc',
                'exact_sets', 'p_exact', 't_exact_s',
                'filt_sets', 'filt_p_lb', 'filt_p_ub', 't_filt_s',
                'approx_sets', 'approx_p_min', 'approx_p_hat', 'approx_p_max', 't_approx_s', 'soundness_violations'
            ])
            for r in all_res['suite1']:
                ex = r.get('exact') or {}
                pf = r.get('filtered_pf005') or {}
                ap = r['approx_single']
                w.writerow([
                    r['benchmark'], r['T'], r['nVars0'], r['nVarsT'], r['p_mc_last'],
                    ex.get('num_sets_last', ''), ex.get('p_exact_last', ''), ex.get('time_s', ''),
                    pf.get('num_sets_last', ''), pf.get('p_lb_last', ''), pf.get('p_ub_last', ''), pf.get('time_s', ''),
                    ap['num_sets_last'], ap['p_min_last'], ap['p_hat_last'], ap['p_max_last'], ap['time_s'], ap['soundness_violations']
                ])

    if 'suite2' in all_res:
        with open(os.path.join(res_dir, 'suite2_nonlinear_multinet.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'system', 'T', 'nVars0', 'nVarsT', 'p_min', 'p_hat', 'p_max', 'p_mc',
                'time_s', 'exact_time_s', 'exact_total_sets', 'soundness_violations'
            ])
            for r in all_res['suite2']:
                w.writerow([
                    r['system'], r['T'], r['nVars0'], r['nVarsT'],
                    r.get('p_min_last', ''), r.get('p_hat_last', ''), r.get('p_max_last', ''), r.get('p_mc_last', ''),
                    r['time_s'], r.get('exact_time_s', ''), r.get('exact_total_sets', ''), r.get('soundness_violations', 0)
                ])

    if 'suite3' in all_res:
        with open(os.path.join(res_dir, 'suite3_temporal_dprobstartl.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'system', 'spec', 'T', 'exact_traces', 'p_exact', 'exact_time_s',
                'approx_p_min', 'approx_p_max', 'approx_time_s',
                'refined_p_min', 'refined_p_max', 'refined_time_s'
            ])
            for r in all_res['suite3']:
                w.writerow([
                    r['system'], r['spec'], r['T'],
                    r.get('exact_traces', ''), r.get('p_exact', ''), r.get('exact_time_s', ''),
                    r['approx_p_min'], r['approx_p_max'], r['approx_time_s'],
                    r.get('refined_p_min', ''), r.get('refined_p_max', ''), r.get('refined_time_s', '')
                ])

    if 'suite4' in all_res:
        with open(os.path.join(res_dir, 'suite4_cegar_refinement.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['system', 'max_splits', 'num_partitions', 'p_min', 'p_hat', 'p_max', 'gap', 'p_ref', 'time_s'])
            for r in all_res['suite4']:
                w.writerow([
                    r['system'], r['max_splits'], r['num_partitions'],
                    r['p_min'], r['p_hat'], r['p_max'], r['gap'], r['p_ref'], r['time_s']
                ])

    tex_lines = ["% Auto-generated CAV 2027 NNCS Artifact Summary Tables"]
    if 'suite4' in all_res:
        tex_lines.append("\\begin{tabular}{lrllll}")
        tex_lines.append("\\toprule System & $K$ & $p_{\\min}$ & $p_{\\max}$ & $\\Delta p$ & Time (s) \\\\ \\midrule")
        for r in all_res['suite4']:
            sys_esc = r['system'].replace('_', '\\_')
            tex_lines.append(
                f"{sys_esc} & {r['num_partitions']} & {r['p_min']:.5f} & "
                f"{r['p_max']:.5f} & {r['gap']:.5e} & {r['time_s']:.2f} \\\\"
            )
        tex_lines.append("\\bottomrule\n\\end{tabular}")
    with open(os.path.join(res_dir, 'cav2027_nncs_tables.tex'), 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')


def generate_artifact_figures(all_results, res_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(res_dir, exist_ok=True)
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 8.5,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })

    # Figure 1: Overview (Closed-Loop Tube + Horizon Scaling + ReLU CEGAR)
    if 'suite1' in all_results and 'suite4' in all_results:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
        sys_dl, X0_dl, ref_dl, _, _ = build_dlnncs_benchmark()
        rprm = ReachPRM_NNCS()
        rprm.initSet = copy.deepcopy(X0_dl)
        rprm.numSteps = 10
        rprm.refInputs = ref_dl
        rprm.method = 'approx-probstar'
        rprm.show = False
        RX_dl, _ = reachBFS_DLNNCS(sys_dl, rprm)
        samples, _ = sample_truncated_gaussian(X0_dl, n_samples=120, seed=7)

        ax = axes[0]
        steps = np.arange(len(RX_dl))
        lbs = np.array([rk[0].getRanges()[0][0] for rk in RX_dl])
        ubs = np.array([rk[0].getRanges()[1][0] for rk in RX_dl])
        mus = np.array([
            float(rk[0].V[0, 0] + rk[0].V[0, 1:rk[0].nVars + 1] @ rk[0].mu[:rk[0].nVars])
            for rk in RX_dl
        ])
        for i in range(min(80, len(samples))):
            tr = np.array(simulate_DLNNCS(sys_dl, samples[i], ref_dl, 10))
            ax.plot(steps, tr[:, 0], color='#7f8c8d', alpha=0.22, linewidth=0.7)
        ax.fill_between(steps, lbs, ubs, color='#3498db', alpha=0.28, label='Approx-ProbStar Outer Tube $[l_t, u_t]$')
        ax.plot(steps, mus, 'b-o', markersize=3.5, linewidth=1.6, label='Analytic State Mean $\\mathbb{E}[x_t[0]]$')
        ax.axhline(0.55, color='#e74c3c', linestyle='--', linewidth=1.5, label='Unsafe Threshold $x_0 \\geq 0.55$')
        ax.set_xlabel('Time Step $t$')
        ax.set_ylabel('Position State $x_t[0]$')
        ax.set_title('(a) Closed-Loop Reachable Tube ($n_0=2$)')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='lower right')

        ax = axes[1]
        s1_dl = [r for r in all_results['suite1'] if r['benchmark'] == 'DLNNCS_ReLU_2_5_5_1']
        T_ap = [r['T'] for r in s1_dl]
        t_ap = [r['approx_single']['time_s'] for r in s1_dl]
        T_ex = [r['T'] for r in s1_dl if r['exact'] is not None]
        t_ex = [r['exact']['time_s'] for r in s1_dl if r['exact'] is not None]
        s4_t3 = [r for r in all_results.get('suite4', []) if r['system'] == 'DLNNCS_ReLU_T3']
        if s4_t3 and 3 not in T_ex:
            T_ex.append(3)
            t_ex.append(s4_t3[0]['exact_time_s'])
        T_pf = [r['T'] for r in s1_dl if r['filtered_pf005'] is not None]
        t_pf = [r['filtered_pf005']['time_s'] for r in s1_dl if r['filtered_pf005'] is not None]
        ax.semilogy(T_ex, t_ex, 'r-s', markersize=5, linewidth=1.8, label='Exact ProbStar ($p_f=0$)')
        ax.semilogy(T_pf, t_pf, 'm--^', markersize=5, linewidth=1.5, label='Filtered Exact ($p_f=0.02$)')
        ax.semilogy(T_ap, t_ap, 'b-o', markersize=5, linewidth=1.8, label='Approx-ProbStar (Single-Set)')
        ax.set_xlabel('Control Horizon $T$')
        ax.set_ylabel('Verification Time (s, log scale)')
        ax.set_title('(b) Horizon Scalability vs. Branching')
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        ax.legend(loc='upper left')

        ax = axes[2]
        s4_dl = [r for r in all_results['suite4'] if r['system'] == 'DLNNCS_ReLU_T3']
        parts_dl = [r['num_partitions'] for r in s4_dl]
        pmin_dl = [r['p_min'] for r in s4_dl]
        pmax_dl = [r['p_max'] for r in s4_dl]
        phat_dl = [r['p_hat'] for r in s4_dl]
        pref_dl = s4_dl[0]['p_ref']
        ax.fill_between(parts_dl, pmin_dl, pmax_dl, color='#2ecc71', alpha=0.28, label='Certified Interval $[p_{\\min}, p_{\\max}]$')
        ax.plot(parts_dl, pmax_dl, color='#27ae60', linestyle='-', marker='^', markersize=4, label='$p_{\\max}^{(K)}$ (Upper Bound)')
        ax.plot(parts_dl, pmin_dl, color='#16a085', linestyle='-', marker='v', markersize=4, label='$p_{\\min}^{(K)}$ (Lower Bound)')
        ax.plot(parts_dl, phat_dl, 'b--o', markersize=4, label='Moment Estimate $\\hat{p}^{(K)}$')
        ax.axhline(pref_dl, color='#c0392b', linestyle=':', linewidth=1.6, label=f'Exact $p^* = {pref_dl:.4f}$')
        ax.set_xscale('log', base=2)
        ax.set_xlabel(r'Initial-Space Partitions $K$ ($\mathbb{R}^{n_0}$)')
        ax.set_ylabel('Quantitative Risk Probability')
        ax.set_title('(c) Closed-Loop CEGAR Convergence')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='best')

        plt.tight_layout()
        fig.savefig(os.path.join(res_dir, 'fig1_nncs_overview.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(res_dir, 'fig1_nncs_overview.png'), dpi=220, bbox_inches='tight')
        plt.close(fig)

    # Figure 2: Smooth TanSig Moment Tracking & 239x CEGAR Contraction
    if 'suite2' in all_results and 'suite4' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
        ax = axes[0]
        s2_th = [r for r in all_results['suite2'] if r['system'] == 'Smooth_TANH_NNCS']
        s2_sg = [r for r in all_results['suite2'] if r['system'] == 'Smooth_SIGMOID_NNCS']
        ax.plot([r['T'] for r in s2_th], [r['p_mc_last'] for r in s2_th], 'k-s', markersize=5, label='TanSig Monte Carlo $p_{\\mathrm{MC}}$')
        ax.plot([r['T'] for r in s2_th], [r['p_hat_last'] for r in s2_th], 'b--o', markersize=5, label='TanSig Analytic Moment $\\hat{p}$')
        ax.plot([r['T'] for r in s2_sg], [r['p_mc_last'] for r in s2_sg], color='#8e44ad', linestyle='-', marker='^', markersize=5, label='LogSig Monte Carlo $p_{\\mathrm{MC}}$')
        ax.plot([r['T'] for r in s2_sg], [r['p_hat_last'] for r in s2_sg], color='#e67e22', linestyle='--', marker='d', markersize=5, label='LogSig Analytic Moment $\\hat{p}$')
        ax.set_xlabel('Control Horizon $T$ ($n_T$ up to $498$, integrated in $n_0=2$)')
        ax.set_ylabel('Quantitative Violation Probability')
        ax.set_title('(a) Smooth TanSig/LogSig Moment Accuracy')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right')

        ax = axes[1]
        s4_sm = [r for r in all_results['suite4'] if r['system'] == 'Smooth_TanSig_T6']
        parts_sm = [r['num_partitions'] for r in s4_sm]
        gaps_sm = [r['gap'] for r in s4_sm]
        s4_dl = [r for r in all_results['suite4'] if r['system'] == 'DLNNCS_ReLU_T3']
        parts_dl = [r['num_partitions'] for r in s4_dl]
        gaps_dl = [r['gap'] for r in s4_dl]
        ax.loglog(parts_sm, gaps_sm, 'r-o', base=2, markersize=5, linewidth=1.8, label='Smooth TanSig ($T=6$, $239.4\\times$ reduction)')
        ax.loglog(parts_dl, gaps_dl, 'b-s', base=2, markersize=5, linewidth=1.8, label='DLNNCS ReLU ($T=3$, $106.8\\times$ reduction)')
        ax.set_yscale('log', base=10)
        ax.set_xlabel(r'Initial-Space Partitions $K$ ($\mathbb{R}^2$)')
        ax.set_ylabel('Certified Gap $\\Delta p = p_{\\max}^{(K)} - p_{\\min}^{(K)}$ (log)')
        ax.set_title('(b) Monotonic Gap Contraction Under CEGAR')
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        ax.legend(loc='upper right')

        plt.tight_layout()
        fig.savefig(os.path.join(res_dir, 'fig2_smooth_tansig_cegar_and_moments.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(res_dir, 'fig2_smooth_tansig_cegar_and_moments.png'), dpi=220, bbox_inches='tight')
        plt.close(fig)

    # Figure 3: Bounded Temporal Logic (dProbStarTL) Speedup & Precision
    if 'suite3' in all_results:
        s3_dl = [r for r in all_results['suite3'] if r['system'] == 'DLNNCS' and r.get('exact_time_s') is not None]
        if s3_dl:
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
            labels = [
                r'$\mathsf{F}_{[0,2]}$ (ET)', r'$\mathsf{G}_{[0,2]}$ (AW)',
                r'$\mathsf{F}_{[0,3]}$ (ET)', r'$\mathsf{G}_{[0,3]}$ (AW)'
            ]
            x = np.arange(len(s3_dl))
            width = 0.26

            ax = axes[0]
            ax.bar(x - width, [r['exact_time_s'] for r in s3_dl], width, color='#e74c3c', label='Exact ProbStar (16–27 traces)')
            ax.bar(x, [r['approx_time_s'] for r in s3_dl], width, color='#2980b9', label='Approx-ProbStar ($K=1$ trace)')
            ax.bar(x + width, [r['refined_time_s'] for r in s3_dl], width, color='#27ae60', label='Refined Approx ($K=5$ partitions)')
            ax.set_yscale('log')
            ax.set_ylim(top=max(r['exact_time_s'] for r in s3_dl) * 3.2)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Verification Time (s, log scale)')
            ax.set_title('(a) dProbStarTL Runtime Speedup (4.6x–12.7x)')
            ax.grid(True, axis='y', which='both', linestyle=':', alpha=0.5)
            ax.legend(loc='upper left')

            ax = axes[1]
            gaps_k1 = [r['approx_p_max'] - r['approx_p_min'] for r in s3_dl]
            gaps_k5 = [r['refined_p_max'] - r['refined_p_min'] for r in s3_dl]
            ax.bar(x - width / 2, gaps_k1, width, color='#3498db', label='Single Trace Gap ($K=1$)')
            ax.bar(x + width / 2, gaps_k5, width, color='#16a085', label='Refined Gap ($K=5$, 7.2x–12.9x tighter)')
            ax.set_yscale('log')
            ax.set_ylim(top=max(gaps_k1) * 2.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Temporal Satisfaction Gap $p_{\\max}(\\varphi) - p_{\\min}(\\varphi)$')
            ax.set_title(r'(b) Dual CDNF Precision in $\mathbb{R}^{n_0}$')
            ax.grid(True, axis='y', which='both', linestyle=':', alpha=0.5)
            ax.legend(loc='upper left')

            plt.tight_layout()
            fig.savefig(os.path.join(res_dir, 'fig3_temporal_dprobstartl_speedup.pdf'), bbox_inches='tight')
            fig.savefig(os.path.join(res_dir, 'fig3_temporal_dprobstartl_speedup.png'), dpi=220, bbox_inches='tight')
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='CAV 2027 Artifact Evaluation: Approximate ProbStar Reachability for NNCS'
    )
    parser.add_argument(
        '--suite',
        type=str,
        default='all',
        choices=['smoke', '1', '2', '3', '4', 'all'],
        help="Benchmark suite to run: 'smoke' (~10s check), '1', '2', '3', '4', or 'all'"
    )
    parser.add_argument(
        '--from-cached',
        action='store_true',
        help='Regenerate CSVs, LaTeX tables, and figures from results/cav2027_nncs_results.json'
    )
    args = parser.parse_args()

    art_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(art_dir, 'results')
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, 'cav2027_nncs_results.json')

    if args.from_cached:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Cached results file not found: {json_path}")
        with open(json_path, 'r') as f:
            all_res = json.load(f)
        export_csv_and_latex_tables(all_res, res_dir)
        generate_artifact_figures(all_res, res_dir)
        print(f"Exported CSVs, LaTeX tables, and figures to {res_dir}")
        return

    if args.suite == 'smoke':
        run_smoke_test()
        return

    all_res = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                all_res = json.load(f)
        except Exception:
            all_res = {}

    if args.suite in ('1', 'all'):
        all_res['suite1'] = run_suite1_horizon_scalability()
    if args.suite in ('2', 'all'):
        all_res['suite2'] = run_suite2_nonlinear_and_multinet()
    if args.suite in ('3', 'all'):
        all_res['suite3'] = run_suite3_temporal_logic()
    if args.suite in ('4', 'all'):
        all_res['suite4'] = run_suite4_adaptive_refinement()

    with open(json_path, 'w') as f:
        json.dump(all_res, f, indent=2)
    print(f"\nSaved benchmark results to {json_path}")

    export_csv_and_latex_tables(all_res, res_dir)
    generate_artifact_figures(all_res, res_dir)
    print(f"Saved CSV tables, LaTeX tables, and figures to {res_dir}")


if __name__ == '__main__':
    main()
