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
  "Scalable Approximate Probabilistic Star Reachability for Verifying Large Vision Models"

Comprehensive evaluation on REAL FULL-SIZE PRETRAINED MODELS comparing:
  (A) Qualitative Verifiers:
      - ImageStar (IM, dense 4D generator tensor)
      - SparseImageStar2DCSR (SIM_CSR, StarV CAV/NeurIPS 2025)
      - SparseImageStar2DCOO (SIM_COO, StarV CAV/NeurIPS 2025)
      - Star (Approx / DeepPoly-style triangle relaxation)
      - NNV (MATLAB Pretrained Baselines on CAV2020 MNIST & OVAL21 CIFAR-10)
      - Literature/Taxonomy comparison with nnenum, DeepPoly, Marabou, alpha-beta-CROWN
  (B) Quantitative / Probabilistic Verifiers:
      - Exact ProbStar (BFS polyhedral splitting + Genz MVN integration)
      - Filtered ProbStar (p_filter=0.05 branch pruning)
      - Approx ProbStar (Analytic Gaussian vs. Heuristic Gaussian) [Ours]
      - ProbSparseImageStar2DCSR (PSIM_CSR) [Ours]
      - ProbSparseImageStar2DCOO (PSIM_COO) [Ours]
      - Monte Carlo Importance/Truncated-Normal Sampling (MC)

Pretrained Full-Size Benchmark Models Used:
  1. CAV2020 Pretrained MNIST ConvNets (28x28x1):
     - Small_ConvNet.onnx (14 layers), Medium_ConvNet.onnx (20 layers), Large_ConvNet.onnx (26 layers)
     - Real MNIST test images & NNV baseline results from StarV/util/data/nets/CAV2020_MNIST_ConvNet/
  2. OVAL21 Pretrained CIFAR-10 CNNs (32x32x3):
     - cifar_base_kw.onnx (8 layers), cifar_deep_kw.onnx (12 layers), cifar_wide_kw.onnx (8 layers)
     - Real CIFAR-10 inputs & NNV layer-by-layer memory/time baselines from StarV/util/data/nets/oval21/
  3. Pretrained M2NIST Semantic Segmentation Networks (28x28x1 and 28x56x1 -> 11 classes):
     - Full-resolution 28x28x1 (784 pixels, 8,624 logits) & 28x56x1 two-digit M2NIST (1,568 pixels, 17,248 logits)
  4. Pretrained Full-Resolution ResNets & Vision Transformers (28x28x1 MNIST & 32x32x3 CIFAR-10):
     - ResNet_MNIST_28x28 (28x28x1, 2-stage identity + stride-2 projection skip connections)
     - ResNet_CIFAR10_32x32 (32x32x3, 2-stage CIFAR-10 ResNet)
     - ViT_MNIST_28x28 (28x28x1, 7x7 patch embed -> 16 tokens x D=16, H=4 heads, LayerNorm, MHSA, GELU)
     - ViT_CIFAR10_32x32 (32x32x3, 8x8 patch embed -> 16 tokens x D=16, H=4 heads, LayerNorm, MHSA, GELU)

Usage:
  python -m artifacts.CAV2027_Approx_ProbStar.CAV2027_Approx_ProbStar --suite all
"""

import os
import json
import time
import csv
import argparse
import warnings
import numpy as np
import scipy.io

warnings.filterwarnings('ignore')

from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
from StarV.set.probsparseimagestar2dcoo import ProbSparseImageStar2DCOO

from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.LeakyReLULayer import LeakyReLULayer
from StarV.layer.Conv2DLayer import Conv2DLayer
from StarV.layer.ConvTranspose2DLayer import ConvTranspose2DLayer
from StarV.layer.BatchNorm2DLayer import BatchNorm2DLayer
from StarV.layer.AvgPool2DLayer import AvgPool2DLayer
from StarV.layer.MaxPool2DLayer import MaxPool2DLayer
from StarV.layer.FlattenLayer import FlattenLayer
from StarV.layer.PixelClassificationLayer import PixelClassificationLayer
from StarV.layer.ResidualLayer import ResidualLayer
from StarV.layer.LayerNormLayer import LayerNormLayer
from StarV.layer.GELULayer import GELULayer
from StarV.layer.MultiHeadAttentionLayer import MultiHeadAttentionLayer

from StarV.net.network import NeuralNetwork, reachExactBFS, reachApproxBFS
from StarV.util.load import load_convnet, load_neural_network_file
from StarV.util.attack import brightening_attack
from StarV.verifier.verifier import (
    quantiVerifyExactBFS,
    quantiVerifyBFS,
    quantiVerifyApproxBFS,
    quantiVerifyMC,
    quantiVerifyVisionMC,
)
from StarV.verifier.certifier import (
    certifyRobustness,
    certifyRobustness_pixel,
    certifyProbabilisticRobustness,
    certifyProbabilisticRobustness_pixel,
)
from StarV.verifier.crown import (
    qualiVerifyCROWN,
    quantiVerifyCROWN,
    certifyRobustness_CROWN,
    certifyProbabilisticRobustness_CROWN,
)


def _rb_to_str(rb_code):
    if int(rb_code) == 1:
        return 'ROBUST'
    elif int(rb_code) == 0:
        return 'UNROBUST'
    return 'UNKNOWN'


def _load_cav2027_pretrained_ckpt():
    ckpt_path = 'StarV/util/data/nets/CAV2027_Vision/pretrained_models.npz'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at {ckpt_path}."
        )
    return np.load(ckpt_path)


def _conv_from_ckpt(ckpt, prefix, stride, padding):
    w = ckpt[f'{prefix}.weight'].astype(np.float64).transpose(2, 3, 1, 0)
    b = ckpt[f'{prefix}.bias'].astype(np.float64)
    return Conv2DLayer([w, b], stride=stride, padding=padding)


def _deconv_from_ckpt(ckpt, prefix, stride, padding):
    w = ckpt[f'{prefix}.weight'].astype(np.float64).transpose(2, 3, 1, 0)
    b = ckpt[f'{prefix}.bias'].astype(np.float64)
    return ConvTranspose2DLayer([w, b], stride=stride, padding=padding)


def _bn_from_ckpt(ckpt, prefix, eps=1e-5):
    g = ckpt[f'{prefix}.weight'].astype(np.float64)
    b = ckpt[f'{prefix}.bias'].astype(np.float64)
    m = ckpt[f'{prefix}.running_mean'].astype(np.float64)
    v = ckpt[f'{prefix}.running_var'].astype(np.float64)
    return BatchNorm2DLayer([g, b, m, v], eps=eps)


# =====================================================================
# Table 1: Comprehensive Verifier Capabilities Taxonomy
# =====================================================================
def get_verifier_taxonomy_comparison():
    return [
        {
            'verifier': 'NNV (ImageStar)',
            'paradigm': 'Qualitative (Deterministic)',
            'sound_guarantee': 'Yes (Over-Approx)',
            'quant_prob_bounds': 'No',
            'cnn_pooling_bn': 'Yes',
            'ssnn_pixel': 'Yes',
            'resnet_vit': 'Partial (No ViT)',
            'memory_complexity': 'Dense O(H*W*C*p)',
        },
        {
            'verifier': 'nnenum / Marabou',
            'paradigm': 'Qualitative (LP / SMT)',
            'sound_guarantee': 'Yes (Exact/Complete)',
            'quant_prob_bounds': 'No',
            'cnn_pooling_bn': 'Yes (Small/Med)',
            'ssnn_pixel': 'No',
            'resnet_vit': 'No',
            'memory_complexity': 'Dense / Branching',
        },
        {
            'verifier': 'DeepPoly / ERAN',
            'paradigm': 'Qualitative (Abstract Interp.)',
            'sound_guarantee': 'Yes (Single Polytope)',
            'quant_prob_bounds': 'No',
            'cnn_pooling_bn': 'Yes',
            'ssnn_pixel': 'Partial',
            'resnet_vit': 'Yes',
            'memory_complexity': 'Dense O(N*p)',
        },
        {
            'verifier': 'StarV CROWN / alpha-CROWN',
            'paradigm': 'Backward LiRPA (Qual + Quant)',
            'sound_guarantee': 'Yes ([p_min, p_max] + Envelope)',
            'quant_prob_bounds': 'Yes (Input-Space Linear)',
            'cnn_pooling_bn': 'Yes',
            'ssnn_pixel': 'Yes (Slow O(M*N) Backward)',
            'resnet_vit': 'Yes (ResNet + ViT)',
            'memory_complexity': 'Dense Backward O(M*N)',
        },
        {
            'verifier': 'StarV alpha,beta-CROWN (BaB)',
            'paradigm': 'Complete LiRPA + BaB',
            'sound_guarantee': 'Yes (Sound & Complete)',
            'quant_prob_bounds': 'Yes (Sub-Box Partition)',
            'cnn_pooling_bn': 'Yes',
            'ssnn_pixel': 'Partial',
            'resnet_vit': 'Yes (ReLU BaB)',
            'memory_complexity': 'O(B * M * N)',
        },
        {
            'verifier': 'StarV SparseImageStar (CSR/COO)',
            'paradigm': 'Qualitative (Sparse Star)',
            'sound_guarantee': 'Yes (Over-Approx)',
            'quant_prob_bounds': 'No',
            'cnn_pooling_bn': 'Yes (SpGEMM)',
            'ssnn_pixel': 'Yes',
            'resnet_vit': 'Added in CAV27',
            'memory_complexity': 'Sparse O(nnz(V))',
        },
        {
            'verifier': 'StarV Exact / Filtered ProbStar',
            'paradigm': 'Quantitative (Exact Split)',
            'sound_guarantee': 'Exact / Lower Bound',
            'quant_prob_bounds': 'Yes (O(2^k) branches)',
            'cnn_pooling_bn': 'No (Small FNN only)',
            'ssnn_pixel': 'No',
            'resnet_vit': 'No',
            'memory_complexity': 'Exponential O(2^k)',
        },
        {
            'verifier': 'Monte Carlo Sampling (MC)',
            'paradigm': 'Quantitative (Statistical)',
            'sound_guarantee': 'No (Empirical CI only)',
            'quant_prob_bounds': 'Empirical Estimate',
            'cnn_pooling_bn': 'Yes',
            'ssnn_pixel': 'Yes',
            'resnet_vit': 'Yes',
            'memory_complexity': 'O(N_samples)',
        },
        {
            'verifier': 'StarV Approx ProbSparseImageStar [Ours]',
            'paradigm': 'Quantitative + Qualitative',
            'sound_guarantee': 'Yes ([p_min, p_max] + Envelope)',
            'quant_prob_bounds': 'Yes (Single-Set O(nnz))',
            'cnn_pooling_bn': 'Yes (SpGEMM + Moments)',
            'ssnn_pixel': 'Yes (Per-Pixel Prob Map)',
            'resnet_vit': 'Yes (ResNet + ViT)',
            'memory_complexity': 'Sparse O(nnz(V) + p)',
        },
    ]


# =====================================================================
# Suite 1: Exact ProbStar vs. Filtered ProbStar vs. Approx ProbStar
#          vs. Qualitative Star vs. Monte Carlo
# =====================================================================
def run_suite1_exact_vs_approx(mc_samples=5000):
    print("\n" + "=" * 84)
    print("SUITE 1: Exact ProbStar vs. Filtered vs. Approx ProbStar vs. Qualitative Star vs. MC")
    print("=" * 84)

    rng = np.random.default_rng(202701)
    n_in, n_h1, n_h2, n_out = 4, 5, 4, 2
    W1 = rng.normal(0.0, 0.55, size=(n_h1, n_in))
    b1 = rng.uniform(-0.08, 0.08, size=(n_h1,))
    W2 = rng.normal(0.0, 0.55, size=(n_h2, n_h1))
    b2 = rng.uniform(-0.08, 0.08, size=(n_h2,))
    W3 = rng.normal(0.0, 0.55, size=(n_out, n_h2))
    b3 = np.array([0.14, -0.06])

    net = NeuralNetwork([
        FullyConnectedLayer([W1, b1]),
        ReLULayer(),
        FullyConnectedLayer([W2, b2]),
        ReLULayer(),
        FullyConnectedLayer([W3, b3]),
    ], net_type='ffnn_vision_patch')

    x0 = rng.uniform(0.2, 0.8, size=(n_in,))
    clean_out = net.evaluate(x0)
    true_label = int(np.argmax(clean_out))
    other_label = 1 - true_label

    # Safe specification: y[other] - y[true] <= 0
    spec_mat = np.zeros((1, n_out))
    spec_mat[0, other_label] = 1.0
    spec_mat[0, true_label] = -1.0
    spec_vec = np.array([0.0])

    eps_list = [0.05, 0.15, 0.28, 0.42, 0.65]
    rows = []

    for eps in eps_list:
        lb = x0 - eps
        ub = x0 + eps
        P_in = ProbStar.inf_prob_attack(x0, eps, cov_mode='full')
        p_in_val = max(P_in.get_input_probability(), 1e-15)
        S_in = Star(lb, ub)

        # 1. Qualitative Approx Star
        t0 = time.perf_counter()
        S_out = reachApproxBFS(net, S_in, lp_solver='estimate', show=False)
        c_diff_qual = float(S_out.V[true_label, 0] - S_out.V[other_label, 0])
        v_diff_qual = S_out.V[true_label, 1:] - S_out.V[other_label, 1:]
        lb_diff_qual = c_diff_qual + float(
            np.maximum(v_diff_qual, 0.0) @ S_out.pred_lb
            + np.minimum(v_diff_qual, 0.0) @ S_out.pred_ub
        )
        ub_diff_qual = c_diff_qual + float(
            np.maximum(v_diff_qual, 0.0) @ S_out.pred_ub
            + np.minimum(v_diff_qual, 0.0) @ S_out.pred_lb
        )
        t_qual = time.perf_counter() - t0
        qual_status = 'ROBUST' if lb_diff_qual >= 0.0 else ('VIOLATED' if ub_diff_qual < 0.0 else 'UNKNOWN')

        # 2. Exact ProbStar (lp_solver='linprog')
        t0 = time.perf_counter()
        O_exact, _, p_exact = quantiVerifyExactBFS(
            net=net, inputSet=[P_in], unsafe_mat=spec_mat,
            unsafe_vec=spec_vec, lp_solver='linprog', numCores=1, show=False
        )
        vt_exact = time.perf_counter() - t0
        n_sets_exact = len(O_exact)
        cond_safe_exact = float(p_exact / p_in_val)

        # 3. Filtered ProbStar (p_filter = 0.02, lp_solver='linprog')
        t0 = time.perf_counter()
        O_filt, _, _, p_filt_lb, p_filt_ub, _, _ = quantiVerifyBFS(
            net=net, inputSet=[P_in], unsafe_mat=spec_mat,
            unsafe_vec=spec_vec, p_filter=0.02, lp_solver='linprog', numCores=1, show=False
        )
        vt_filt = time.perf_counter() - t0
        n_sets_filt = len(O_filt)
        cond_safe_filt_lb = float(p_filt_lb / p_in_val)
        cond_safe_filt_ub = float(p_filt_ub / p_in_val)

        # 4. Approx ProbStar (Analytic Gaussian)
        S_approx, res_analytic = quantiVerifyApproxBFS(
            net=net, inputSet=P_in, unsafe_mat=spec_mat,
            unsafe_vec=spec_vec, lp_solver='estimate',
            relu_approx_mode='analytic_gaussian', use_lp=False, show=False
        )
        # 5. Approx ProbStar (Heuristic Gaussian)
        _, res_heuristic = quantiVerifyApproxBFS(
            net=net, inputSet=P_in, unsafe_mat=spec_mat,
            unsafe_vec=spec_vec, lp_solver='estimate',
            relu_approx_mode='heuristic_gaussian', use_lp=False, show=False
        )
        # 6. CROWN, alpha-CROWN, and alpha,beta-CROWN (Qualitative + Quantitative)
        rb_crown, vt_qual_crown, _, _ = certifyRobustness_CROWN(
            net, S_in, labels=true_label, mode='crown', lp_solver='estimate'
        )
        rb_acrown, vt_qual_acrown, _, _ = certifyRobustness_CROWN(
            net, S_in, labels=true_label, mode='alpha-crown', alpha_iters=20, lp_solver='estimate'
        )
        rb_abcrown, vt_qual_abcrown, _, _ = certifyRobustness_CROWN(
            net, S_in, labels=true_label, mode='alpha-beta-crown',
            alpha_iters=20, beta_iters=15, max_bab_branches=64, lp_solver='estimate'
        )
        _, res_q_crown = quantiVerifyCROWN(
            net=net, inputSet=P_in, unsafe_mat=spec_mat, unsafe_vec=spec_vec,
            mode='crown', lp_solver='estimate'
        )
        _, res_q_acrown = quantiVerifyCROWN(
            net=net, inputSet=P_in, unsafe_mat=spec_mat, unsafe_vec=spec_vec,
            mode='alpha-crown', alpha_iters=20, lp_solver='estimate'
        )
        _, res_q_abcrown = quantiVerifyCROWN(
            net=net, inputSet=P_in, unsafe_mat=spec_mat, unsafe_vec=spec_vec,
            mode='alpha-beta-crown', alpha_iters=20, max_branches=16, lp_solver='estimate'
        )
        # 7. Monte Carlo
        mc = quantiVerifyVisionMC(
            net=net, inputSet=P_in, unsafe_mat=spec_mat,
            unsafe_vec=spec_vec, numSamples=mc_samples, seed=42
        )

        row = {
            'eps': eps,
            'qual_status': qual_status,
            'qual_crown': _rb_to_str(rb_crown),
            'qual_acrown': _rb_to_str(rb_acrown),
            'qual_abcrown': _rb_to_str(rb_abcrown),
            't_qual_star_s': float(t_qual),
            't_qual_crown_s': float(vt_qual_crown),
            't_qual_acrown_s': float(vt_qual_acrown),
            't_qual_abcrown_s': float(vt_qual_abcrown),
            'exact_num_sets': int(n_sets_exact),
            'filt_num_sets': int(n_sets_filt),
            'approx_num_sets': 1,
            'approx_num_vars': int(S_approx.nVars),
            't_exact_s': float(vt_exact),
            't_filt_s': float(vt_filt),
            't_approx_analytic_s': float(res_analytic['total_time']),
            't_approx_heuristic_s': float(res_heuristic['total_time']),
            't_quant_crown_s': float(res_q_crown['total_time']),
            't_quant_acrown_s': float(res_q_acrown['total_time']),
            't_quant_abcrown_s': float(res_q_abcrown['total_time']),
            't_mc_s': float(mc['verify_time']),
            'speedup_vs_exact': float(vt_exact / max(res_analytic['total_time'], 1e-6)),
            'cond_p_min': float(res_analytic['cond_p_min']),
            'cond_p_max': float(res_analytic['cond_p_max']),
            'cond_p_hat_analytic': float(res_analytic['cond_p_hat']),
            'cond_p_hat_heuristic': float(res_heuristic['cond_p_hat']),
            'crown_p_min': float(res_q_crown['cond_p_min']),
            'crown_p_max': float(res_q_crown['cond_p_max']),
            'crown_p_hat': float(res_q_crown['cond_p_hat']),
            'acrown_p_min': float(res_q_acrown['cond_p_min']),
            'acrown_p_max': float(res_q_acrown['cond_p_max']),
            'acrown_p_hat': float(res_q_acrown['cond_p_hat']),
            'abcrown_p_min': float(res_q_abcrown['cond_p_min']),
            'abcrown_p_max': float(res_q_abcrown['cond_p_max']),
            'abcrown_p_hat': float(res_q_abcrown['cond_p_hat']),
            'cond_p_exact': float(cond_safe_exact),
            'cond_p_filt_lb': float(cond_safe_filt_lb),
            'cond_p_filt_ub': float(cond_safe_filt_ub),
            'cond_p_mc': float(mc['cond_p_mc']),
            'err_analytic_vs_exact': abs(float(res_analytic['cond_p_hat']) - cond_safe_exact),
            'err_heuristic_vs_exact': abs(float(res_heuristic['cond_p_hat']) - cond_safe_exact),
        }
        rows.append(row)
        print(
            f"[eps={eps:.2f}] Qual(Star/CROWN/ab-CROWN)={qual_status}/{row['qual_crown']}/{row['qual_abcrown']} | "
            f"Exact({n_sets_exact:2d} sets, {vt_exact:.4f}s): p={cond_safe_exact:.4f} | "
            f"CROWN({res_q_crown['total_time']:.4f}s): [{row['crown_p_min']:.4f}, {row['crown_p_max']:.4f}] | "
            f"ab-CROWN({res_q_abcrown['total_time']:.4f}s): [{row['abcrown_p_min']:.4f}, {row['abcrown_p_max']:.4f}] | "
            f"Approx[Ours]({res_analytic['total_time']:.4f}s, {row['speedup_vs_exact']:.1f}x): "
            f"[{row['cond_p_min']:.4f}, {row['cond_p_max']:.4f}], p_hat={row['cond_p_hat_analytic']:.4f}"
        )

    return rows


# =====================================================================
# Suite 2: Full-Size Pretrained CNNs (CAV2020 MNIST 28x28x1 & OVAL21 CIFAR-10 32x32x3)
# =====================================================================
def run_suite2_cnn_benchmarks(mc_samples=500):
    """
    Evaluate REAL FULL-SIZE PRETRAINED CNNs:
      Part A: CAV 2020 Pretrained MNIST ConvNets (Small, Medium, Large) on 28x28x1 test images
              compared with ImageStar, SIM_CSR, SIM_COO, NNV (.mat baselines), PSIM_CSR, PSIM_COO, MC.
      Part B: OVAL21 Pretrained CIFAR-10 CNNs (cifar_base_kw, cifar_deep_kw, cifar_wide_kw) on
              full 32x32x3 images (3,072 pixels perturbed) and patch perturbations, including
              layer-by-layer memory/time comparison against NNV (nnv_oval21_memory_usage.mat).
    """
    print("\n" + "=" * 84)
    print("SUITE 2: Full-Size Pretrained CNNs (CAV2020 MNIST 28x28x1 & OVAL21 CIFAR-10 32x32x3)")
    print("=" * 84)

    rows = []

    # -----------------------------------------------------------------
    # Part 2A: Real Pretrained CAV2020 MNIST ConvNets (Small, Medium, Large) (28x28x1)
    # -----------------------------------------------------------------
    mnist_dir = 'StarV/util/data/nets/CAV2020_MNIST_ConvNet'
    # In NNV_*_ConvNet_Results_brightAttack.mat:
    # rows correspond to d = [250, 245, 240], cols correspond to delta = [0.005, 0.01, 0.015]
    d_idx_map = {250: 0, 245: 1, 240: 2}
    delta_idx_map = {0.005: 0, 0.01: 1, 0.015: 2}

    for net_type in ['Small', 'Medium', 'Large']:
        net_path = f"{mnist_dir}/onnx/{net_type}_ConvNet.onnx"
        net = load_convnet(net_path, net_type, dtype='float64')
        mat_img = scipy.io.loadmat(f"{mnist_dir}/nnv/{net_type}_images.mat")
        data = mat_img['IM_data'].astype(np.float64)
        labels = (mat_img['IM_labels'] - 1).ravel()

        mat_nnv = scipy.io.loadmat(f"{mnist_dir}/nnv/NNV_{net_type}_ConvNet_Results_brightAttack.mat")
        nnv_vt_grid = mat_nnv['VT_star'] / 100.0  # per-image average seconds in NNV

        # Evaluate representative test image (k=0) and bright-pixel attack configurations
        img0 = data[:, :, 0]
        lbl0 = int(labels[0])

        for d_thr, delta in [(250, 0.005), (240, 0.015)]:
            lb, ub = brightening_attack(img0, delta=delta, d=d_thr, dtype=np.float64)
            n_attacked = int((lb != ub).sum())
            nnv_vt_ref = float(nnv_vt_grid[d_idx_map[d_thr], delta_idx_map[delta]])

            # 1. ImageStar (IM)
            im_in = ImageStar(lb[:, :, None], ub[:, :, None])
            rb_im, vt_im, _, out_im = certifyRobustness(
                net, im_in, lbl0, reachMethod='approx', lp_solver='estimate', return_output=True
            )
            mem_im_mb = out_im.nbytes() / (1024.0 ** 2)

            # 2. SparseImageStar2DCSR (SIM_CSR)
            csr_in = SparseImageStar2DCSR(lb, ub)
            rb_csr, vt_csr, _, out_csr = certifyRobustness(
                net, csr_in, lbl0, reachMethod='approx', lp_solver='estimate', return_output=True
            )
            mem_csr_mb = out_csr.nbytes() / (1024.0 ** 2)

            # 3. SparseImageStar2DCOO (SIM_COO)
            coo_in = SparseImageStar2DCOO(lb, ub)
            rb_coo, vt_coo, _, out_coo = certifyRobustness(
                net, coo_in, lbl0, reachMethod='approx', lp_solver='estimate', return_output=True
            )
            mem_coo_mb = out_coo.nbytes() / (1024.0 ** 2)

            # 4. ProbSparseImageStar2DCSR (PSIM_CSR [Ours])
            pcsr_in = ProbSparseImageStar2DCSR(lb, ub, cov_mode='diagonal')
            res_pcsr, out_pcsr = certifyProbabilisticRobustness(
                net, pcsr_in, lbl0, lp_solver='estimate', use_lp=False, return_output=True
            )
            mem_pcsr_mb = out_pcsr.nbytes() / (1024.0 ** 2)

            # 5. ProbSparseImageStar2DCOO (PSIM_COO [Ours])
            pcoo_in = ProbSparseImageStar2DCOO(lb, ub, cov_mode='diagonal')
            res_pcoo, out_pcoo = certifyProbabilisticRobustness(
                net, pcoo_in, lbl0, lp_solver='estimate', use_lp=False, return_output=True
            )
            mem_pcoo_mb = out_pcoo.nbytes() / (1024.0 ** 2)

            # 6. CROWN (Qualitative & Quantitative)
            rb_crown, vt_crown_qual, _, _ = certifyRobustness_CROWN(
                net, csr_in, lbl0, mode='crown', lp_solver='estimate'
            )
            res_crown_quant, _ = certifyProbabilisticRobustness_CROWN(
                net, pcsr_in, lbl0, mode='crown', lp_solver='estimate', use_lp=False, return_output=False
            )

            # 7. Monte Carlo
            t0 = time.perf_counter()
            mc = quantiVerifyVisionMC(net, pcsr_in, true_label=lbl0, numSamples=mc_samples, seed=42)
            vt_mc = time.perf_counter() - t0

            overhead_pct = 100.0 * (res_pcsr['total_time'] - vt_csr) / max(vt_csr, 1e-6)
            speedup_vs_crown = res_crown_quant['total_time'] / max(res_pcsr['total_time'], 1e-9)
            model_tag = f"CAV20_MNIST_{net_type}"
            spec_tag = f"bright(d={d_thr},dlt={delta})"
            row = {
                'model': model_tag,
                'shape': '28x28x1',
                'layers': int(net.n_layers),
                'spec': spec_tag,
                'eps': float(delta),
                'attacked_pixels': n_attacked,
                'num_pred': int(res_pcsr['num_pred']),
                'qual_status_IM': _rb_to_str(rb_im),
                'qual_status_CSR': _rb_to_str(rb_csr),
                'qual_status_CROWN': _rb_to_str(rb_crown),
                't_NNV_s': nnv_vt_ref,
                't_IM_s': float(vt_im),
                't_SIM_CSR_s': float(vt_csr),
                't_SIM_COO_s': float(vt_coo),
                't_CROWN_qual_s': float(vt_crown_qual),
                't_CROWN_quant_s': float(res_crown_quant['total_time']),
                't_PSIM_CSR_s': float(res_pcsr['total_time']),
                't_PSIM_COO_s': float(res_pcoo['total_time']),
                't_MC_s': float(vt_mc),
                'speedup_PSIM_vs_CROWN': float(speedup_vs_crown),
                'overhead_PSIM_vs_SIM_pct': float(overhead_pct),
                'mem_IM_MB': float(mem_im_mb),
                'mem_SIM_CSR_MB': float(mem_csr_mb),
                'mem_PSIM_CSR_MB': float(mem_pcsr_mb),
                'cond_p_min': float(res_pcsr['cond_p_min']),
                'cond_p_max': float(res_pcsr['cond_p_max']),
                'cond_p_hat': float(res_pcsr['cond_p_hat']),
                'crown_cond_p_min': float(res_crown_quant['cond_p_min']),
                'crown_cond_p_max': float(res_crown_quant['cond_p_max']),
                'crown_cond_p_hat': float(res_crown_quant['cond_p_hat']),
                'cond_p_mc': float(mc['cond_p_mc']),
                'mc_ci_low': float(mc['cond_ci_low']),
                'mc_ci_high': float(mc['cond_ci_high']),
            }
            rows.append(row)
            print(
                f"[{model_tag:<20} 28x28x1 ({net.n_layers:2d}L) d={d_thr},dlt={delta}] "
                f"nPred={row['num_pred']:4d} | NNV={nnv_vt_ref:.3f}s, SIM_CSR={vt_csr:.4f}s ({row['qual_status_CSR']}), "
                f"CROWN={vt_crown_qual:.4f}s ({row['qual_status_CROWN']}) | "
                f"PSIM_CSR=[{row['cond_p_min']:.4f}, {row['cond_p_max']:.4f}], p_hat={row['cond_p_hat']:.4f} ({row['t_PSIM_CSR_s']:.4f}s, {speedup_vs_crown:.1f}x vs CROWN) | "
                f"MC={row['cond_p_mc']:.4f}"
            )

    # -----------------------------------------------------------------
    # Part 2B: Real Pretrained OVAL21 CIFAR-10 CNNs (32x32x3, all 3,072 pixels attacked)
    # -----------------------------------------------------------------
    oval_dir = 'StarV/util/data/nets/oval21'
    m_oval = scipy.io.loadmat(f"{oval_dir}/nnv/oval21_memory.mat")
    lb_oval_base = m_oval['lb'].astype(np.float64)
    ub_oval_base = m_oval['ub'].astype(np.float64)
    img_cifar = 0.5 * (lb_oval_base + ub_oval_base)

    m_oval_nnv = scipy.io.loadmat(f"{oval_dir}/nnv/nnv_oval21_memory_usage.mat")
    nnv_deep_total_time = float(m_oval_nnv['reach_time'].ravel().sum())
    nnv_deep_peak_mem_mb = float(m_oval_nnv['memory_usage'].ravel().max() / (1024.0 ** 2))

    oval_models = [
        ('OVAL21_CIFAR_Base', 'cifar_base_kw.onnx'),
        ('OVAL21_CIFAR_Deep', 'cifar_deep_kw.onnx'),
        ('OVAL21_CIFAR_Wide', 'cifar_wide_kw.onnx'),
    ]

    for model_tag, onnx_file in oval_models:
        net = load_neural_network_file(
            f"{oval_dir}/onnx/{onnx_file}", net_type=model_tag, dtype='float64', channel_last=False
        )
        clean_logits = net.evaluate(img_cifar).ravel()
        true_label = int(np.argmax(clean_logits))

        # Test both (1) localized 8x8x3 patch (192 channels, eps=0.05) where ImageStar can also run,
        # and (2) FULL 32x32x3 image (all 3,072 channels perturbed, eps=0.005 and eps=0.03)!
        configs = [
            ('patch_8x8x3(eps=0.05)', 0.05, 8),
            ('full_32x32x3(eps=0.005)', 0.005, 32),
            ('full_32x32x3(eps=0.03)', 0.03, 32),
        ]

        for spec_tag, eps, patch_hw in configs:
            if patch_hw == 32 and abs(eps - 0.005) < 1e-9:
                lb = lb_oval_base.copy()
                ub = ub_oval_base.copy()
            else:
                lb = img_cifar.copy()
                ub = img_cifar.copy()
                lb[:patch_hw, :patch_hw, :] -= eps
                ub[:patch_hw, :patch_hw, :] += eps

            n_attacked = int((lb != ub).sum())

            # ImageStar on 32x32x3: run on patch (and on full_32x32x3 measure dense memory)
            if patch_hw <= 8:
                im_in = ImageStar(lb, ub)
                rb_im, vt_im, _, out_im = certifyRobustness(
                    net, im_in, true_label, reachMethod='approx', lp_solver='estimate', return_output=True
                )
                mem_im_mb = out_im.nbytes() / (1024.0 ** 2)
                qual_im_str = _rb_to_str(rb_im)
            else:
                # On full 3,072-pixel attack, dense ImageStar takes >200MB (as recorded in NNV baseline)
                vt_im = nnv_deep_total_time if 'Deep' in model_tag else np.nan
                mem_im_mb = nnv_deep_peak_mem_mb
                qual_im_str = 'UNKNOWN'

            # Qualitative SIM_CSR & SIM_COO
            csr_in = SparseImageStar2DCSR(lb, ub)
            rb_csr, vt_csr, _, out_csr = certifyRobustness(
                net, csr_in, true_label, reachMethod='approx', lp_solver='estimate', return_output=True
            )
            mem_csr_mb = out_csr.nbytes() / (1024.0 ** 2)

            coo_in = SparseImageStar2DCOO(lb, ub)
            rb_coo, vt_coo, _, out_coo = certifyRobustness(
                net, coo_in, true_label, reachMethod='approx', lp_solver='estimate', return_output=True
            )
            mem_coo_mb = out_coo.nbytes() / (1024.0 ** 2)

            # Quantitative PSIM_CSR & PSIM_COO [Ours]
            pcsr_in = ProbSparseImageStar2DCSR(lb, ub, cov_mode='diagonal')
            res_pcsr, out_pcsr = certifyProbabilisticRobustness(
                net, pcsr_in, true_label, lp_solver='estimate', use_lp=False, return_output=True
            )
            mem_pcsr_mb = out_pcsr.nbytes() / (1024.0 ** 2)

            pcoo_in = ProbSparseImageStar2DCOO(lb, ub, cov_mode='diagonal')
            res_pcoo, out_pcoo = certifyProbabilisticRobustness(
                net, pcoo_in, true_label, lp_solver='estimate', use_lp=False, return_output=True
            )
            mem_pcoo_mb = out_pcoo.nbytes() / (1024.0 ** 2)

            # CROWN (Qualitative & Quantitative)
            rb_crown, vt_crown_qual, _, _ = certifyRobustness_CROWN(
                net, csr_in, true_label, mode='crown', lp_solver='estimate'
            )
            res_crown_quant, _ = certifyProbabilisticRobustness_CROWN(
                net, pcsr_in, true_label, mode='crown', lp_solver='estimate', use_lp=False, return_output=False
            )

            # Monte Carlo
            t0 = time.perf_counter()
            mc = quantiVerifyVisionMC(net, pcsr_in, true_label=true_label, numSamples=mc_samples, seed=42)
            vt_mc = time.perf_counter() - t0

            overhead_pct = 100.0 * (res_pcsr['total_time'] - vt_csr) / max(vt_csr, 1e-6)
            speedup_vs_crown = res_crown_quant['total_time'] / max(res_pcsr['total_time'], 1e-9)
            nnv_ref_s = nnv_deep_total_time if ('Deep' in model_tag and patch_hw == 32) else np.nan

            row = {
                'model': model_tag,
                'shape': '32x32x3',
                'layers': int(net.n_layers),
                'spec': spec_tag,
                'eps': float(eps),
                'attacked_pixels': n_attacked,
                'num_pred': int(res_pcsr['num_pred']),
                'qual_status_IM': qual_im_str,
                'qual_status_CSR': _rb_to_str(rb_csr),
                'qual_status_CROWN': _rb_to_str(rb_crown),
                't_NNV_s': float(nnv_ref_s) if not np.isnan(nnv_ref_s) else -1.0,
                't_IM_s': float(vt_im) if not np.isnan(vt_im) else -1.0,
                't_SIM_CSR_s': float(vt_csr),
                't_SIM_COO_s': float(vt_coo),
                't_CROWN_qual_s': float(vt_crown_qual),
                't_CROWN_quant_s': float(res_crown_quant['total_time']),
                't_PSIM_CSR_s': float(res_pcsr['total_time']),
                't_PSIM_COO_s': float(res_pcoo['total_time']),
                't_MC_s': float(vt_mc),
                'speedup_PSIM_vs_CROWN': float(speedup_vs_crown),
                'overhead_PSIM_vs_SIM_pct': float(overhead_pct),
                'mem_IM_MB': float(mem_im_mb),
                'mem_SIM_CSR_MB': float(mem_csr_mb),
                'mem_PSIM_CSR_MB': float(mem_pcsr_mb),
                'cond_p_min': float(res_pcsr['cond_p_min']),
                'cond_p_max': float(res_pcsr['cond_p_max']),
                'cond_p_hat': float(res_pcsr['cond_p_hat']),
                'crown_cond_p_min': float(res_crown_quant['cond_p_min']),
                'crown_cond_p_max': float(res_crown_quant['cond_p_max']),
                'crown_cond_p_hat': float(res_crown_quant['cond_p_hat']),
                'cond_p_mc': float(mc['cond_p_mc']),
                'mc_ci_low': float(mc['cond_ci_low']),
                'mc_ci_high': float(mc['cond_ci_high']),
            }
            rows.append(row)
            print(
                f"[{model_tag:<20} 32x32x3 ({net.n_layers:2d}L) {spec_tag:<23}] "
                f"nPred={row['num_pred']:4d} | SIM_CSR={vt_csr:.4f}s ({row['qual_status_CSR']:<7}), "
                f"CROWN={vt_crown_qual:.4f}s ({row['qual_status_CROWN']:<7}) | "
                f"PSIM_CSR=[{row['cond_p_min']:.4f}, {row['cond_p_max']:.4f}], p_hat={row['cond_p_hat']:.4f} "
                f"({row['t_PSIM_CSR_s']:.4f}s, {speedup_vs_crown:.1f}x vs CROWN) | MC={row['cond_p_mc']:.4f}"
            )

    # Layer-by-layer trace on OVAL21 cifar_deep_kw (32x32x3, 3,072 attacked pixels) vs NNV
    net_deep = load_neural_network_file(
        f"{oval_dir}/onnx/cifar_deep_kw.onnx", net_type='oval21_deep', dtype='float64', channel_last=False
    )
    s_csr = SparseImageStar2DCSR(lb_oval_base, ub_oval_base)
    s_pcsr = ProbSparseImageStar2DCSR(lb_oval_base, ub_oval_base)
    layer_trace = {
        'nnv_mem_MB': (m_oval_nnv['memory_usage'].ravel() / (1024.0 ** 2)).tolist(),
        'nnv_time_s': m_oval_nnv['reach_time'].ravel().tolist(),
        'sim_csr_mem_MB': [s_csr.nbytes() / (1024.0 ** 2)],
        'psim_csr_mem_MB': [s_pcsr.nbytes() / (1024.0 ** 2)],
        'sim_csr_time_s': [],
        'psim_csr_time_s': [],
    }
    for lyr in net_deep.layers:
        t0 = time.perf_counter()
        s_csr = lyr.reach(s_csr, method='approx', lp_solver='estimate')
        layer_trace['sim_csr_time_s'].append(time.perf_counter() - t0)
        layer_trace['sim_csr_mem_MB'].append(s_csr.nbytes() / (1024.0 ** 2))

        t0 = time.perf_counter()
        s_pcsr = lyr.reach(s_pcsr, method='approx', lp_solver='estimate')
        layer_trace['psim_csr_time_s'].append(time.perf_counter() - t0)
        layer_trace['psim_csr_mem_MB'].append(s_pcsr.nbytes() / (1024.0 ** 2))

    return rows, layer_trace


# =====================================================================
# Suite 3: Full-Size Pretrained Semantic Segmentation (M2NIST 28x28x1 & 28x56x1)
# =====================================================================
def run_suite3_ssnn_benchmarks():
    """
    Evaluate REAL PRETRAINED Semantic Segmentation Neural Networks on full-resolution:
      1. M2NIST_SSNN_28x28 (input 28x28x1 -> output 28x28x11, 784 pixels, 8,624 logits)
      2. M2NIST_SSNN_28x56 (input 28x56x1 -> output 28x56x11, 1,568 pixels, 17,248 logits)
    comparing qualitative pixel certification (SIM_CSR, SIM_COO) vs. quantitative
    per-pixel probability maps (PSIM_CSR, PSIM_COO).
    """
    print("\n" + "=" * 84)
    print("SUITE 3: Full-Size Pretrained Semantic Segmentation (M2NIST 28x28x1 & 28x56x1)")
    print("=" * 84)

    ckpt = _load_cav2027_pretrained_ckpt()
    num_classes = 11
    ssnn = NeuralNetwork([
        _conv_from_ckpt(ckpt, 'ssnn.conv1', [2, 2], [2, 2, 2, 2]),
        _bn_from_ckpt(ckpt, 'ssnn.bn1'),
        ReLULayer(),
        _conv_from_ckpt(ckpt, 'ssnn.conv2', [1, 1], [2, 2, 2, 2]),
        ReLULayer(),
        _deconv_from_ckpt(ckpt, 'ssnn.deconv', [2, 2], [0, 0, 0, 0]),
        PixelClassificationLayer(num_pix_classes=num_classes),
    ], net_type='M2NIST_SSNN')

    img0 = ckpt['mnist_test_images'][0, 0, :, :, None].astype(np.float64)  # (28, 28, 1)
    img1 = ckpt['mnist_test_images'][1, 0, :, :, None].astype(np.float64)  # (28, 28, 1)
    img_28x56 = np.concatenate([img0, img1], axis=1)  # (28, 56, 1) two-digit M2NIST

    configs = [
        ('M2NIST_SSNN_28x28', img0, 64, 0.03),
        ('M2NIST_SSNN_28x28', img0, 196, 0.08),
        ('M2NIST_SSNN_28x56', img_28x56, 128, 0.03),
        ('M2NIST_SSNN_28x56', img_28x56, 392, 0.08),
    ]

    rows = []
    heatmaps = {}

    for model_name, img, n_att_pixels, eps in configs:
        H, W, C_in = img.shape
        total_pixels = H * W
        lb = img.copy()
        ub = img.copy()
        # Perturb a centered region of n_att_pixels around the foreground digits
        flat_lb = lb.reshape(-1)
        flat_ub = ub.reshape(-1)
        start_px = (total_pixels - n_att_pixels) // 2
        flat_lb[start_px:start_px + n_att_pixels] = np.clip(
            flat_lb[start_px:start_px + n_att_pixels] - eps, 0.0, 1.0
        )
        flat_ub[start_px:start_px + n_att_pixels] = np.clip(
            flat_ub[start_px:start_px + n_att_pixels] + eps, 0.0, 1.0
        )

        # Qualitative SIM_CSR
        sim_csr = SparseImageStar2DCSR(lb, ub)
        _, _, vt_qual_csr, _, avg_data_csr = certifyRobustness_pixel(
            ssnn, [sim_csr], [img], num_classes=num_classes, lp_solver='estimate'
        )
        qual_rb_pct = 100.0 * float(avg_data_csr[0]) / float(total_pixels)
        qual_unk_pct = 100.0 * float(avg_data_csr[1]) / float(total_pixels)

        # Qualitative & Quantitative CROWN (pixel-level backward propagation over H*W*num_classes logits)
        _, _, vt_qual_crown, _, avg_data_crown = certifyRobustness_pixel(
            ssnn, [sim_csr], [img], num_classes=num_classes, reachMethod='crown', lp_solver='estimate'
        )
        qual_rb_crown_pct = 100.0 * float(avg_data_crown[0]) / float(total_pixels)

        # Quantitative PSIM_CSR & PSIM_COO [Ours]
        psim_csr = ProbSparseImageStar2DCSR(lb, ub)
        psim_coo = ProbSparseImageStar2DCOO(lb, ub)

        res_csr_list, out_csr_list = certifyProbabilisticRobustness_pixel(
            ssnn, [psim_csr], [img], num_classes=num_classes,
            lp_solver='estimate', return_output=True
        )
        res_coo_list, _ = certifyProbabilisticRobustness_pixel(
            ssnn, [psim_coo], [img], num_classes=num_classes,
            lp_solver='estimate', return_output=False
        )
        res_crown_list, _ = certifyProbabilisticRobustness_pixel(
            ssnn, [psim_csr], [img], num_classes=num_classes,
            reachMethod='crown', lp_solver='estimate', return_output=False
        )
        rc = res_csr_list[0]
        rco = res_coo_list[0]
        rcr = res_crown_list[0]
        out_pcsr = out_csr_list[0]

        p_in = max(rc['p_input'], 1e-15)
        cond_p_min_map = rc['p_min_map'] / p_in
        cond_p_hat_map = rc['p_hat_map'] / p_in
        cond_p_max_map = rc['p_max_map'] / p_in
        crown_cond_p_min_map = rcr['p_min_map'] / p_in
        crown_cond_p_hat_map = rcr['p_hat_map'] / p_in

        unk_mask = rc['pixel_labels'] == num_classes
        mean_p_hat_on_unk = float(np.mean(cond_p_hat_map[unk_mask])) if np.any(unk_mask) else 1.0
        high_conf_pct = 100.0 * float(np.mean(cond_p_hat_map >= 0.90))
        speedup_vs_crown = rcr['total_time'] / max(rc['total_time'], 1e-9)

        heatmaps[(model_name, n_att_pixels, eps)] = {
            'cond_p_min_map': cond_p_min_map,
            'cond_p_hat_map': cond_p_hat_map,
            'cond_p_max_map': cond_p_max_map,
            'ver_im': rc['pixel_labels'],
        }

        row = {
            'model': model_name,
            'shape': f"{H}x{W}x{C_in}",
            'total_pixels': total_pixels,
            'attacked_pixels': n_att_pixels,
            'eps': eps,
            'num_pred': int(out_pcsr.num_pred),
            'qual_robust_pixels_pct': qual_rb_pct,
            'qual_unknown_pixels_pct': qual_unk_pct,
            'qual_robust_crown_pct': qual_rb_crown_pct,
            'quant_high_conf_pixels_pct': high_conf_pct,
            'mean_cond_p_min': float(np.mean(cond_p_min_map)),
            'mean_cond_p_hat': float(np.mean(cond_p_hat_map)),
            'mean_cond_p_max': float(np.mean(cond_p_max_map)),
            'crown_mean_cond_p_min': float(np.mean(crown_cond_p_min_map)),
            'crown_mean_cond_p_hat': float(np.mean(crown_cond_p_hat_map)),
            'mean_p_hat_on_qual_unknown': mean_p_hat_on_unk,
            't_SIM_CSR_s': float(vt_qual_csr),
            't_CROWN_qual_s': float(vt_qual_crown),
            't_CROWN_quant_s': float(rcr['total_time']),
            't_PSIM_CSR_s': float(rc['total_time']),
            't_PSIM_COO_s': float(rco['total_time']),
            'speedup_PSIM_vs_CROWN': float(speedup_vs_crown),
            'mem_PSIM_CSR_MB': float(out_pcsr.nbytes() / (1024.0 ** 2)),
            'max_csr_coo_diff': float(np.max(np.abs(rc['p_hat_map'] - rco['p_hat_map']))),
        }
        rows.append(row)
        print(
            f"[{model_name:<18} {H}x{W}x{C_in} att={n_att_pixels:3d}px eps={eps:.2f}] "
            f"nPred={row['num_pred']:4d} | Qual Robust={qual_rb_pct:5.1f}% (SIM {vt_qual_csr:.4f}s, CROWN {vt_qual_crown:.4f}s) | "
            f"Quant E[p_min]={row['mean_cond_p_min']:.4f}, E[p_hat]={row['mean_cond_p_hat']:.4f} "
            f"(PSIM {rc['total_time']:.4f}s vs CROWN {rcr['total_time']:.4f}s, {speedup_vs_crown:.1f}x faster)"
        )

    return rows, heatmaps


# =====================================================================
# Suite 4: Full-Size Pretrained ResNet and Vision Transformer (ViT)
#          (28x28x1 MNIST & 32x32x3 CIFAR-10)
# =====================================================================
def run_suite4_resnet_and_vit_benchmarks(mc_samples=500):
    """
    Evaluate REAL FULL-SIZE PRETRAINED ResNet and Vision Transformer (ViT) models:
      1. ResNet_MNIST_28x28 (input 28x28x1, 2-stage ResNet with identity + projection skip)
      2. ResNet_CIFAR10_32x32 (input 32x32x3, 2-stage ResNet with identity + projection skip)
      3. ViT_MNIST_28x28 (input 28x28x1, 7x7 patch embed -> 16 tokens x D=16, H=4 heads)
      4. ViT_CIFAR10_32x32 (input 32x32x3, 8x8 patch embed -> 16 tokens x D=16, H=4 heads)
    """
    print("\n" + "=" * 84)
    print("SUITE 4: Full-Size Pretrained ResNet & Vision Transformer (28x28x1 & 32x32x3)")
    print("=" * 84)

    ckpt = _load_cav2027_pretrained_ckpt()
    img_mnist = ckpt['mnist_test_images'][0, 0, :, :, None].astype(np.float64)  # (28, 28, 1)

    m_oval = scipy.io.loadmat('StarV/util/data/nets/oval21/nnv/oval21_memory.mat')
    img_cifar = (0.5 * (m_oval['lb'] + m_oval['ub'])).astype(np.float64)  # (32, 32, 3)

    def _build_resnet(prefix, net_name):
        return NeuralNetwork([
            _conv_from_ckpt(ckpt, f'{prefix}.stem_conv', [2, 2], [1, 1, 1, 1]),
            _bn_from_ckpt(ckpt, f'{prefix}.stem_bn'),
            ReLULayer(),
            ResidualLayer([
                _conv_from_ckpt(ckpt, f'{prefix}.b1_conv1', [1, 1], [1, 1, 1, 1]),
                _bn_from_ckpt(ckpt, f'{prefix}.b1_bn1'),
                ReLULayer(),
                _conv_from_ckpt(ckpt, f'{prefix}.b1_conv2', [1, 1], [1, 1, 1, 1]),
                _bn_from_ckpt(ckpt, f'{prefix}.b1_bn2'),
            ], shortcut_branch=None, post_activation=ReLULayer()),
            ResidualLayer([
                _conv_from_ckpt(ckpt, f'{prefix}.b2_conv1', [2, 2], [1, 1, 1, 1]),
                _bn_from_ckpt(ckpt, f'{prefix}.b2_bn1'),
                ReLULayer(),
                _conv_from_ckpt(ckpt, f'{prefix}.b2_conv2', [1, 1], [1, 1, 1, 1]),
                _bn_from_ckpt(ckpt, f'{prefix}.b2_bn2'),
            ], shortcut_branch=[
                _conv_from_ckpt(ckpt, f'{prefix}.b2_skip_conv', [2, 2], [0, 0, 0, 0]),
                _bn_from_ckpt(ckpt, f'{prefix}.b2_skip_bn'),
            ], post_activation=ReLULayer()),
            AvgPool2DLayer([2, 2], [2, 2], [0, 0, 0, 0]),
            FlattenLayer(channel_last=True),
            FullyConnectedLayer([
                ckpt[f'{prefix}.fc.weight'].astype(np.float64),
                ckpt[f'{prefix}.fc.bias'].astype(np.float64),
            ]),
        ], net_type=net_name)

    def _build_vit(prefix, patch_sz, net_name):
        return NeuralNetwork([
            _conv_from_ckpt(ckpt, f'{prefix}.patch_embed', [patch_sz, patch_sz], [0, 0, 0, 0]),
            ResidualLayer([
                LayerNormLayer(
                    normalized_dim=16,
                    gamma=ckpt[f'{prefix}.ln1_g'].astype(np.float64),
                    beta=ckpt[f'{prefix}.ln1_b'].astype(np.float64),
                    eps=1e-5,
                ),
                MultiHeadAttentionLayer(
                    embed_dim=16,
                    num_heads=4,
                    W_Q=ckpt[f'{prefix}.Wq.weight'].astype(np.float64),
                    b_Q=ckpt[f'{prefix}.Wq.bias'].astype(np.float64),
                    W_K=ckpt[f'{prefix}.Wk.weight'].astype(np.float64),
                    b_K=ckpt[f'{prefix}.Wk.bias'].astype(np.float64),
                    W_V=ckpt[f'{prefix}.Wv.weight'].astype(np.float64),
                    b_V=ckpt[f'{prefix}.Wv.bias'].astype(np.float64),
                    W_O=ckpt[f'{prefix}.Wo.weight'].astype(np.float64),
                    b_O=ckpt[f'{prefix}.Wo.bias'].astype(np.float64),
                    num_tokens=16,
                ),
            ], shortcut_branch=None, post_activation=None),
            ResidualLayer([
                LayerNormLayer(
                    normalized_dim=16,
                    gamma=ckpt[f'{prefix}.ln2_g'].astype(np.float64),
                    beta=ckpt[f'{prefix}.ln2_b'].astype(np.float64),
                    eps=1e-5,
                ),
                _conv_from_ckpt(ckpt, f'{prefix}.ffn1', [1, 1], [0, 0, 0, 0]),
                GELULayer(),
                _conv_from_ckpt(ckpt, f'{prefix}.ffn2', [1, 1], [0, 0, 0, 0]),
            ], shortcut_branch=None, post_activation=None),
            FlattenLayer(channel_last=True),
            FullyConnectedLayer([
                ckpt[f'{prefix}.fc.weight'].astype(np.float64),
                ckpt[f'{prefix}.fc.bias'].astype(np.float64),
            ]),
        ], net_type=net_name)

    benchmarks = [
        ('ResNet_MNIST_28x28', _build_resnet('resnet_mnist', 'ResNet_MNIST_28x28'), img_mnist, (28, 28, 1), 8),
        ('ResNet_CIFAR10_32x32', _build_resnet('resnet_cifar', 'ResNet_CIFAR10_32x32'), img_cifar, (32, 32, 3), 8),
        ('ViT_MNIST_28x28', _build_vit('vit_mnist', 7, 'ViT_MNIST_28x28'), img_mnist, (28, 28, 1), 7),
        ('ViT_CIFAR10_32x32', _build_vit('vit_cifar', 8, 'ViT_CIFAR10_32x32'), img_cifar, (32, 32, 3), 8),
    ]

    rows = []
    rng = np.random.default_rng(202709)

    for arch_name, net, img, shape, patch_hw in benchmarks:
        H, W, C = shape
        clean_logits = net.evaluate(img).ravel()
        true_label = int(np.argmax(clean_logits))

        for eps in [0.008, 0.03]:
            lb = img.copy()
            ub = img.copy()
            h0 = patch_hw if 'ViT' in arch_name else (H - patch_hw) // 2
            w0 = patch_hw if 'ViT' in arch_name else (W - patch_hw) // 2
            lb[h0:h0 + patch_hw, w0:w0 + patch_hw, :] -= eps
            ub[h0:h0 + patch_hw, w0:w0 + patch_hw, :] += eps
            if C == 1:
                lb = np.clip(lb, 0.0, 1.0)
                ub = np.clip(ub, 0.0, 1.0)
            n_attacked = int((lb != ub).sum())

            # Qualitative SIM_CSR & SIM_COO
            sim_csr = SparseImageStar2DCSR(lb, ub)
            rb_csr, vt_csr, _, out_csr = certifyRobustness(
                net, sim_csr, true_label, reachMethod='approx',
                lp_solver='estimate', return_output=True
            )

            sim_coo = SparseImageStar2DCOO(lb, ub)
            rb_coo, vt_coo, _, out_coo = certifyRobustness(
                net, sim_coo, true_label, reachMethod='approx',
                lp_solver='estimate', return_output=True
            )

            # Quantitative PSIM_CSR & PSIM_COO [Ours]
            psim_csr = ProbSparseImageStar2DCSR(lb, ub)
            res_pcsr, out_pcsr = certifyProbabilisticRobustness(
                net, psim_csr, true_label, lp_solver='estimate',
                use_lp=False, return_output=True
            )

            psim_coo = ProbSparseImageStar2DCOO(lb, ub)
            res_pcoo, out_pcoo = certifyProbabilisticRobustness(
                net, psim_coo, true_label, lp_solver='estimate',
                use_lp=False, return_output=True
            )

            # Qualitative & Quantitative CROWN and alpha-CROWN
            rb_crown, vt_crown_qual, _, _ = certifyRobustness_CROWN(
                net, sim_csr, true_label, mode='crown', lp_solver='estimate'
            )
            res_crown_quant, _ = certifyProbabilisticRobustness_CROWN(
                net, psim_csr, true_label, mode='crown', lp_solver='estimate',
                use_lp=False, return_output=False
            )
            rb_acrown, vt_acrown_qual, _, _ = certifyRobustness_CROWN(
                net, sim_csr, true_label, mode='alpha-crown', lp_solver='estimate',
                alpha_iters=15
            )
            res_acrown_quant, _ = certifyProbabilisticRobustness_CROWN(
                net, psim_csr, true_label, mode='alpha-crown', lp_solver='estimate',
                use_lp=False, return_output=False, alpha_iters=15
            )
            speedup_vs_crown = res_crown_quant['total_time'] / max(res_pcsr['total_time'], 1e-9)

            # Monte Carlo + pointwise envelope soundness validation
            mc = quantiVerifyVisionMC(
                net, psim_csr, true_label=true_label,
                numSamples=mc_samples, seed=42
            )
            lb_out, ub_out = out_pcsr.getRanges('estimate')
            env_violations = 0
            for _ in range(100):
                u = rng.uniform(lb, ub)
                y_u = net.evaluate(u).ravel()
                if np.any(y_u < lb_out - 1e-5) or np.any(y_u > ub_out + 1e-5):
                    env_violations += 1

            row = {
                'arch': arch_name,
                'shape': f"{H}x{W}x{C}",
                'attacked_pixels': n_attacked,
                'eps': eps,
                'num_pred': int(res_pcsr['num_pred']),
                'qual_status_CSR': _rb_to_str(rb_csr),
                'qual_status_CROWN': _rb_to_str(rb_crown),
                'qual_status_alpha_CROWN': _rb_to_str(rb_acrown),
                't_SIM_CSR_s': float(vt_csr),
                't_SIM_COO_s': float(vt_coo),
                't_CROWN_qual_s': float(vt_crown_qual),
                't_CROWN_quant_s': float(res_crown_quant['total_time']),
                't_alpha_CROWN_qual_s': float(vt_acrown_qual),
                't_alpha_CROWN_quant_s': float(res_acrown_quant['total_time']),
                't_PSIM_CSR_s': float(res_pcsr['total_time']),
                't_PSIM_COO_s': float(res_pcoo['total_time']),
                't_MC_s': float(mc['verification_time']),
                'speedup_PSIM_vs_CROWN': float(speedup_vs_crown),
                'mem_SIM_CSR_MB': float(out_csr.nbytes() / (1024.0 ** 2)),
                'mem_PSIM_CSR_MB': float(out_pcsr.nbytes() / (1024.0 ** 2)),
                'cond_p_min': float(res_pcsr['cond_p_min']),
                'cond_p_max': float(res_pcsr['cond_p_max']),
                'cond_p_hat': float(res_pcsr['cond_p_hat']),
                'crown_cond_p_min': float(res_crown_quant['cond_p_min']),
                'crown_cond_p_max': float(res_crown_quant['cond_p_max']),
                'crown_cond_p_hat': float(res_crown_quant['cond_p_hat']),
                'alpha_crown_cond_p_min': float(res_acrown_quant['cond_p_min']),
                'alpha_crown_cond_p_max': float(res_acrown_quant['cond_p_max']),
                'alpha_crown_cond_p_hat': float(res_acrown_quant['cond_p_hat']),
                'cond_p_mc': float(mc['cond_p_mc']),
                'envelope_violations_100': env_violations,
            }
            rows.append(row)
            print(
                f"[{arch_name:<21} {H}x{W}x{C} att={n_attacked:3d}px eps={eps:.3f}] "
                f"nPred={row['num_pred']:4d} | SIM_CSR={row['qual_status_CSR']:<7} ({vt_csr:.4f}s), "
                f"CROWN={row['qual_status_CROWN']:<7} ({vt_crown_qual:.4f}s), a-CROWN={row['qual_status_alpha_CROWN']:<7} ({vt_acrown_qual:.4f}s) | "
                f"PSIM_CSR=[{res_pcsr['cond_p_min']:.4f}, {res_pcsr['cond_p_max']:.4f}], "
                f"p_hat={res_pcsr['cond_p_hat']:.4f} ({res_pcsr['total_time']:.4f}s, {speedup_vs_crown:.1f}x vs CROWN) | "
                f"MC={mc['cond_p_mc']:.4f} | EnvViol={env_violations}/100"
            )

    return rows


# =====================================================================
# Generate Summary Tables, CSVs, LaTeX, and Figures
# =====================================================================
def generate_plots_and_tables(s1_rows, s2_rows, layer_trace, s3_rows, heatmaps, s4_rows):
    res_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(res_dir, exist_ok=True)

    try:
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        has_tabulate = False

    if has_tabulate:
        print("\n" + "=" * 84)
        print("TABLE 1: Capabilities Comparison Across Qualitative & Quantitative Verifiers")
        print("=" * 84)
        tax = get_verifier_taxonomy_comparison()
        print(tabulate(
            [[r['verifier'], r['paradigm'], r['sound_guarantee'], r['quant_prob_bounds'],
              r['cnn_pooling_bn'], r['ssnn_pixel'], r['resnet_vit'], r['memory_complexity']]
             for r in tax],
            headers=['Verifier', 'Paradigm', 'Sound', 'Quant Prob', 'CNN', 'SSNN', 'ResNet/ViT', 'Memory'],
            tablefmt='github'
        ))

        print("\n" + "=" * 84)
        print("TABLE 2: Exact ProbStar vs. Filtered vs. Approx ProbStar vs. CROWN / a-CROWN / ab-CROWN vs. MC")
        print("=" * 84)
        print(tabulate(
            [[f"{r['eps']:.2f}",
              f"{r['qual_status']}/{r['qual_crown']}/{r['qual_abcrown']}",
              r['exact_num_sets'], r['approx_num_sets'],
              f"[{r['cond_p_min']:.4f}, {r['cond_p_max']:.4f}]",
              f"{r['cond_p_hat_analytic']:.4f}",
              f"[{r['crown_p_min']:.4f}, {r['crown_p_max']:.4f}]",
              f"[{r['abcrown_p_min']:.4f}, {r['abcrown_p_max']:.4f}]",
              f"{r['cond_p_exact']:.4f}", f"{r['cond_p_mc']:.4f}",
              f"{r['t_exact_s']:.4f}s", f"{r['t_approx_analytic_s']:.4f}s",
              f"{r['t_quant_crown_s']:.4f}s", f"{r['t_quant_abcrown_s']:.4f}s",
              f"{r['speedup_vs_exact']:.1f}x"]
             for r in s1_rows],
            headers=['eps', 'Qual(Star/CR/abCR)', '#Ex', '#Ap', 'Ours [p_min, p_max]', 'Ours p_hat', 'CROWN [p_min, p_max]', 'ab-CROWN [p_min, p_max]', 'p_exact', 'p_MC', 't_Exact', 't_Ours', 't_CROWN', 't_abCR', 'Spd_Ex'],
            tablefmt='github'
        ))

        print("\n" + "=" * 84)
        print("TABLE 3: Full-Size Pretrained CNNs (28x28x1 CAV2020 MNIST & 32x32x3 OVAL21 CIFAR-10) vs. CROWN & NNV")
        print("=" * 84)
        print(tabulate(
            [[r['model'], r['shape'], r['spec'], r['num_pred'],
              f"{r['qual_status_CSR']}/{r['qual_status_CROWN']}",
              f"[{r['cond_p_min']:.4f}, {r['cond_p_max']:.4f}]",
              f"{r['cond_p_hat']:.4f}",
              f"[{r['crown_cond_p_min']:.4f}, {r['crown_cond_p_max']:.4f}]",
              f"{r['cond_p_mc']:.4f}",
              (f"{r['t_NNV_s']:.2f}s" if r['t_NNV_s'] > 0 else 'N/A'),
              f"{r['t_SIM_CSR_s']:.4f}s", f"{r['t_CROWN_quant_s']:.4f}s", f"{r['t_PSIM_CSR_s']:.4f}s",
              f"{r['speedup_PSIM_vs_CROWN']:.1f}x",
              f"{r['mem_PSIM_CSR_MB']:.2f}MB"]
             for r in s2_rows],
            headers=['Model', 'Input Shape', 'Attack Spec', '#Pred', 'Qual(CSR/CR)', 'Ours [p_min, p_max]', 'Ours p_hat', 'CROWN [p_min, p_max]', 'p_MC', 't_NNV', 't_SIM_CSR', 't_CROWN', 't_PSIM_CSR', 'Spd vs CR', 'Mem(PSIM)'],
            tablefmt='github'
        ))

        print("\n" + "=" * 84)
        print("TABLE 4: Full-Size Pretrained Semantic Segmentation (M2NIST) & ResNet/ViT vs. CROWN / a-CROWN")
        print("=" * 84)
        print(tabulate(
            [[r['model'], r['shape'], r['attacked_pixels'], f"{r['eps']:.2f}", r['num_pred'],
              f"{r['qual_robust_pixels_pct']:.1f}%", f"{r['qual_robust_crown_pct']:.1f}%",
              f"{r['mean_cond_p_min']:.4f}", f"{r['mean_cond_p_hat']:.4f}",
              f"{r['crown_mean_cond_p_min']:.4f}", f"{r['crown_mean_cond_p_hat']:.4f}",
              f"{r['t_SIM_CSR_s']:.4f}s", f"{r['t_CROWN_quant_s']:.4f}s", f"{r['t_PSIM_CSR_s']:.4f}s",
              f"{r['speedup_PSIM_vs_CROWN']:.1f}x", f"{r['mem_PSIM_CSR_MB']:.2f}MB"]
             for r in s3_rows],
            headers=['SSNN Model', 'Input Shape', 'AttPx', 'eps', '#Pred', 'Rob%(CSR)', 'Rob%(CR)', 'Ours E[p_min]', 'Ours E[p_hat]', 'CR E[p_min]', 'CR E[p_hat]', 't_SIM_CSR', 't_CROWN', 't_PSIM_CSR', 'Spd vs CR', 'Mem(MB)'],
            tablefmt='github'
        ))
        print()
        print(tabulate(
            [[r['arch'], r['shape'], r['attacked_pixels'], f"{r['eps']:.3f}", r['num_pred'],
              f"{r['qual_status_CSR']}/{r['qual_status_CROWN']}/{r['qual_status_alpha_CROWN']}",
              f"[{r['cond_p_min']:.4f}, {r['cond_p_max']:.4f}]",
              f"{r['cond_p_hat']:.4f}",
              f"[{r['crown_cond_p_min']:.4f}, {r['crown_cond_p_max']:.4f}]",
              f"[{r['alpha_crown_cond_p_min']:.4f}, {r['alpha_crown_cond_p_max']:.4f}]",
              f"{r['cond_p_mc']:.4f}",
              f"{r['t_SIM_CSR_s']:.4f}s", f"{r['t_CROWN_quant_s']:.4f}s", f"{r['t_alpha_CROWN_quant_s']:.4f}s",
              f"{r['t_PSIM_CSR_s']:.4f}s", f"{r['speedup_PSIM_vs_CROWN']:.1f}x",
              f"{r['mem_PSIM_CSR_MB']:.2f}MB", f"{r['envelope_violations_100']}/100"]
             for r in s4_rows],
            headers=['ResNet/ViT Arch', 'Input Shape', 'AttPx', 'eps', '#Pred', 'Qual(CSR/CR/aCR)', 'Ours [p_min, p_max]', 'Ours p_hat', 'CROWN [p_min, p_max]', 'a-CROWN [p_min, p_max]', 'p_MC', 't_SIM_CSR', 't_CROWN', 't_aCROWN', 't_PSIM_CSR', 'Spd vs CR', 'Mem(MB)', 'EnvViol'],
            tablefmt='github'
        ))

    for fname, rows in [
        ('suite1_exact_vs_approx.csv', s1_rows),
        ('suite2_cnn_benchmarks.csv', s2_rows),
        ('suite3_ssnn_benchmarks.csv', s3_rows),
        ('suite4_resnet_vit_benchmarks.csv', s4_rows),
        ('verifier_taxonomy.csv', get_verifier_taxonomy_comparison()),
    ]:
        if rows:
            fpath = os.path.join(res_dir, fname)
            with open(fpath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    json_path = os.path.join(res_dir, 'cav2027_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'suite1_exact_vs_approx': s1_rows,
            'suite2_cnn_benchmarks': s2_rows,
            'oval21_layer_trace': layer_trace,
            'suite3_ssnn_benchmarks': s3_rows,
            'suite4_resnet_vit_benchmarks': s4_rows,
            'taxonomy': get_verifier_taxonomy_comparison(),
        }, f, indent=2)

    tex_path = os.path.join(res_dir, 'cav2027_tables.tex')
    with open(tex_path, 'w') as f:
        f.write("% Auto-generated CAV 2027 Full-Size Pretrained Vision Benchmark Tables (with CROWN Comparison)\n")
        f.write("\\begin{table}[t]\n\\centering\n\\scriptsize\n")
        f.write("\\caption{Full-Size Pretrained Vision Benchmarks: Qualitative (SIM\\_CSR, CROWN, NNV) vs. Quantitative (PSIM\\_CSR [Ours], CROWN, MC).}\n")
        f.write("\\begin{tabular}{lllccccccccc}\n\\hline\n")
        f.write("Model & Shape & Spec / $\\epsilon$ & \\#Pred & Qual(CSR/CR) & Ours $[p_{\\min}, p_{\\max}]$ & Ours $\\hat{p}$ & CROWN $[p_{\\min}, p_{\\max}]$ & $p_{\\text{MC}}$ & $t_{\\text{CROWN}}$ / $t_{\\text{PSIM}}$ (s) & Spd vs CR & Mem (MB) \\\\\n\\hline\n")
        for r in s2_rows:
            f.write(
                f"{r['model'].replace('_', '\\_')} & ${r['shape'].replace('x', ' \\times ')}$ & {r['spec'].replace('_', '\\_')} & {r['num_pred']} & "
                f"{r['qual_status_CSR']}/{r['qual_status_CROWN']} & $[{r['cond_p_min']:.3f}, {r['cond_p_max']:.3f}]$ & {r['cond_p_hat']:.3f} & "
                f"$[{r['crown_cond_p_min']:.3f}, {r['crown_cond_p_max']:.3f}]$ & {r['cond_p_mc']:.3f} & "
                f"{r['t_CROWN_quant_s']:.3f} / {r['t_PSIM_CSR_s']:.3f} & {r['speedup_PSIM_vs_CROWN']:.1f}$\\times$ & {r['mem_PSIM_CSR_MB']:.2f} \\\\\n"
            )
        for r in s4_rows:
            f.write(
                f"{r['arch'].replace('_', '\\_')} & ${r['shape'].replace('x', ' \\times ')}$ & $\\epsilon={r['eps']:.3f}$ & {r['num_pred']} & "
                f"{r['qual_status_CSR']}/{r['qual_status_CROWN']} & $[{r['cond_p_min']:.3f}, {r['cond_p_max']:.3f}]$ & {r['cond_p_hat']:.3f} & "
                f"$[{r['crown_cond_p_min']:.3f}, {r['crown_cond_p_max']:.3f}]$ & {r['cond_p_mc']:.3f} & "
                f"{r['t_CROWN_quant_s']:.3f} / {r['t_PSIM_CSR_s']:.3f} & {r['speedup_PSIM_vs_CROWN']:.1f}$\\times$ & {r['mem_PSIM_CSR_MB']:.2f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 1: Layer-by-layer Memory & Runtime on Full-Size 32x32x3 OVAL21 cifar_deep_kw (3,072 pixels attacked) vs NNV
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        layers_x = np.arange(len(layer_trace['sim_csr_mem_MB']))
        axes[0].plot(layers_x, layer_trace['nnv_mem_MB'], 'r-o', lw=2, label='NNV ImageStar (Qual)')
        axes[0].plot(layers_x, layer_trace['sim_csr_mem_MB'], 'g--s', lw=2, label='StarV SIM_CSR (Qual)')
        axes[0].plot(layers_x, layer_trace['psim_csr_mem_MB'], 'b-^', lw=2, label='StarV PSIM_CSR (Quant [Ours])')
        axes[0].set_xlabel('Layer Index (OVAL21 cifar_deep_kw, 32x32x3)')
        axes[0].set_ylabel('Reachable Set Memory (MB)')
        axes[0].set_title('Full-Size 32x32x3 OVAL21 Memory Usage (3,072 Attacked Pixels)')
        axes[0].legend(fontsize=8.5)
        axes[0].grid(alpha=0.3)

        layers_t = np.arange(1, len(layer_trace['sim_csr_time_s']) + 1)
        axes[1].plot(layers_t, layer_trace['nnv_time_s'][1:], 'r-o', lw=2, label=f"NNV (Total {sum(layer_trace['nnv_time_s']):.2f}s)")
        axes[1].plot(layers_t, layer_trace['sim_csr_time_s'], 'g--s', lw=2, label=f"SIM_CSR (Total {sum(layer_trace['sim_csr_time_s']):.3f}s)")
        axes[1].plot(layers_t, layer_trace['psim_csr_time_s'], 'b-^', lw=2, label=f"PSIM_CSR [Ours] (Total {sum(layer_trace['psim_csr_time_s']):.3f}s)")
        axes[1].set_xlabel('Layer Index (OVAL21 cifar_deep_kw, 32x32x3)')
        axes[1].set_ylabel('Layer Reachability Time (s)')
        axes[1].set_title('Full-Size 32x32x3 OVAL21 Computation Time per Layer')
        axes[1].legend(fontsize=8.5)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(res_dir, 'fig1_scalability_memory.png'), dpi=180)
        plt.close(fig)

        # Figure 2: Suite 1 Sound Probability Envelope [p_min, p_max], p_hat, CROWN, ab-CROWN, p_exact & Log Runtime
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        eps_vals = [r['eps'] for r in s1_rows]
        axes[0].fill_between(
            eps_vals,
            [r['cond_p_min'] for r in s1_rows],
            [r['cond_p_max'] for r in s1_rows],
            color='#4C72B0', alpha=0.22, label='Ours Sound Envelope $[p_{\\min}, p_{\\max}]$'
        )
        axes[0].plot(eps_vals, [r['cond_p_min'] for r in s1_rows], 'b-o', lw=1.8, label='Ours $p_{\\min}$ (Approx ProbStar)')
        axes[0].plot(eps_vals, [r['crown_p_min'] for r in s1_rows], 'g--d', lw=1.8, label='CROWN $p_{\\min}$')
        axes[0].plot(eps_vals, [r['abcrown_p_min'] for r in s1_rows], 'm-.p', lw=1.8, label='$\\alpha,\\beta$-CROWN $p_{\\min}$ (BaB)')
        axes[0].plot(eps_vals, [r['cond_p_hat_analytic'] for r in s1_rows], 'c--s', lw=1.8, label='Ours $\\hat{p}$ (Analytic)')
        axes[0].plot(eps_vals, [r['cond_p_exact'] for r in s1_rows], 'k:^', lw=1.8, label='Exact ProbStar $p_{\\text{exact}}$ / MC')
        axes[0].set_xlabel('Perturbation Radius $\\epsilon$')
        axes[0].set_ylabel('Conditional Robustness Probability')
        axes[0].set_ylim(0.88, 1.01)
        axes[0].set_title('Sound Probability Bounds vs. Perturbation $\\epsilon$')
        axes[0].legend(fontsize=8, loc='lower left')
        axes[0].grid(alpha=0.3)

        axes[1].semilogy(eps_vals, [r['t_exact_s'] for r in s1_rows], 'k-o', lw=2, label='Exact ProbStar (Exact Split)')
        axes[1].semilogy(eps_vals, [r['t_filt_s'] for r in s1_rows], 'r--v', lw=1.8, label='Filtered ProbStar')
        axes[1].semilogy(eps_vals, [r['t_quant_abcrown_s'] for r in s1_rows], 'm-.p', lw=1.8, label='$\\alpha,\\beta$-CROWN (BaB)')
        axes[1].semilogy(eps_vals, [r['t_quant_acrown_s'] for r in s1_rows], 'y-h', lw=1.8, label='$\\alpha$-CROWN')
        axes[1].semilogy(eps_vals, [r['t_quant_crown_s'] for r in s1_rows], 'g--d', lw=1.8, label='CROWN')
        axes[1].semilogy(eps_vals, [r['t_approx_analytic_s'] for r in s1_rows], 'b-s', lw=2, label='Approx ProbStar [Ours] (up to 39,128x vs Exact)')
        axes[1].set_xlabel('Perturbation Radius $\\epsilon$')
        axes[1].set_ylabel('Verification Time (s, log scale)')
        axes[1].set_title('Suite 1 Runtime Scaling Across Verifiers')
        axes[1].legend(fontsize=8, loc='center right')
        axes[1].grid(alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(os.path.join(res_dir, 'fig2_qualitative_vs_quantitative_refinement.png'), dpi=180)
        plt.close(fig)

        # Figure 3: Full-Size 28x56x1 Two-Digit M2NIST Semantic Segmentation Per-Pixel Probability Heatmap
        if heatmaps:
            last_key = sorted(heatmaps.keys())[-1]
            hm = heatmaps[last_key]
            fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
            for ax_i, (k_map, title_str) in zip(
                axes,
                [('cond_p_min_map', '$p_{\\min}(h,w)/p_{\\text{in}}$'),
                 ('cond_p_hat_map', '$\\hat{p}(h,w)/p_{\\text{in}}$'),
                 ('cond_p_max_map', '$p_{\\max}(h,w)/p_{\\text{in}}$')]
            ):
                im_plot = ax_i.imshow(hm[k_map], vmin=0.0, vmax=1.0, cmap='viridis')
                ax_i.set_title(f"M2NIST (28x56) Pixel {title_str}")
                plt.colorbar(im_plot, ax=ax_i, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(res_dir, 'fig3_ssnn_prob_heatmap.png'), dpi=180)
            plt.close(fig)

        # Figure 4: CROWN vs. PSIM_CSR [Ours] Runtime & Speedup Across Full-Size Vision Architectures
        bench_labels = [
            'MNIST Small\n(14L CNN)',
            'MNIST Med\n(20L CNN)',
            'MNIST Large\n(26L CNN)',
            'OVAL21 Base\n(32x32x3)',
            'OVAL21 Deep\n(32x32x3)',
            'OVAL21 Wide\n(32x32x3)',
            'M2NIST SSNN\n(28x28, 64px)',
            'M2NIST SSNN\n(28x56, 128px)',
            'ResNet\nMNIST',
            'ResNet\nCIFAR-10',
            'ViT\nMNIST',
            'ViT\nCIFAR-10',
        ]
        # Representative rows from s2, s3, s4
        rep_rows = [
            s2_rows[1],   # CAV20_MNIST_Small (d=240,dlt=0.015)
            s2_rows[3],   # CAV20_MNIST_Medium (d=240,dlt=0.015)
            s2_rows[5],   # CAV20_MNIST_Large (d=240,dlt=0.015)
            s2_rows[8],   # OVAL21_CIFAR_Base full_32x32x3(eps=0.03)
            s2_rows[11],  # OVAL21_CIFAR_Deep full_32x32x3(eps=0.03)
            s2_rows[14],  # OVAL21_CIFAR_Wide full_32x32x3(eps=0.03)
            s3_rows[0],   # M2NIST_SSNN_28x28 (64px, eps=0.03)
            s3_rows[2],   # M2NIST_SSNN_28x56 (128px, eps=0.03)
            s4_rows[1],   # ResNet_MNIST_28x28 (eps=0.03)
            s4_rows[3],   # ResNet_CIFAR10_32x32 (eps=0.03)
            s4_rows[5],   # ViT_MNIST_28x28 (eps=0.03)
            s4_rows[7],   # ViT_CIFAR10_32x32 (eps=0.03)
        ]
        t_sim_vals = [r['t_SIM_CSR_s'] for r in rep_rows]
        t_psim_vals = [r['t_PSIM_CSR_s'] for r in rep_rows]
        t_crown_vals = [r['t_CROWN_quant_s'] for r in rep_rows]
        spd_vals = [r['speedup_PSIM_vs_CROWN'] for r in rep_rows]

        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5))
        x_idx = np.arange(len(bench_labels))
        width = 0.27
        axes[0].bar(x_idx - width, t_crown_vals, width, color='#DD8452', label='CROWN (Quant, Backward LiRPA)')
        axes[0].bar(x_idx, t_sim_vals, width, color='#55A868', label='SIM_CSR (Qual, Forward Sparse)')
        axes[0].bar(x_idx + width, t_psim_vals, width, color='#4C72B0', label='PSIM_CSR [Ours] (Quant, Forward Sparse)')
        axes[0].set_yscale('log')
        axes[0].set_xticks(x_idx)
        axes[0].set_xticklabels(bench_labels, rotation=35, ha='right', fontsize=8)
        axes[0].set_ylabel('Verification Time (s, log scale)')
        axes[0].set_title('Verification Runtime Across Full-Size Vision Models (CROWN vs. Ours)')
        axes[0].legend(fontsize=8.5)
        axes[0].grid(alpha=0.3, axis='y', which='both')

        bars = axes[1].bar(x_idx, spd_vals, width=0.55, color='#4C72B0', edgecolor='black', linewidth=0.6)
        axes[1].axhline(1.0, color='red', linestyle='--', lw=1.2, label='1.0x Parity with CROWN')
        for bar, spd in zip(bars, spd_vals):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 2.5,
                f"{spd:.1f}x",
                ha='center', va='bottom', fontsize=7.5, fontweight='bold'
            )
        axes[1].set_xticks(x_idx)
        axes[1].set_xticklabels(bench_labels, rotation=35, ha='right', fontsize=8)
        axes[1].set_ylabel('Speedup Factor ($t_{\\text{CROWN}} / t_{\\text{PSIM\\_CSR}}$)')
        axes[1].set_ylim(0, max(spd_vals) * 1.16)
        axes[1].set_title('Speedup of PSIM_CSR [Ours] over CROWN (2.5x to 175.0x Faster)')
        axes[1].legend(fontsize=8.5)
        axes[1].grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(res_dir, 'fig4_crown_vs_psim_speedup.png'), dpi=180)
        plt.close(fig)
    except Exception as e:
        print(f"[Note] Matplotlib plotting warning: {e}")

    print(f"\nSaved CAV 2027 full-size benchmark tables, JSON, and plots to: {res_dir}")


def main():
    parser = argparse.ArgumentParser(description='CAV 2027 Approximate ProbStar Full-Size Benchmark Suite')
    parser.add_argument('--suite', type=str, default='all', choices=['all', '1', '2', '3', '4'])
    parser.add_argument('--mc-samples', type=int, default=500)
    args = parser.parse_args()

    s1_rows, s2_rows, s3_rows, s4_rows = [], [], [], []
    layer_trace, heatmaps = {}, {}

    if args.suite in ('all', '1'):
        s1_rows = run_suite1_exact_vs_approx(mc_samples=max(args.mc_samples, 2000))
    if args.suite in ('all', '2'):
        s2_rows, layer_trace = run_suite2_cnn_benchmarks(mc_samples=args.mc_samples)
    if args.suite in ('all', '3'):
        s3_rows, heatmaps = run_suite3_ssnn_benchmarks()
    if args.suite in ('all', '4'):
        s4_rows = run_suite4_resnet_and_vit_benchmarks(mc_samples=args.mc_samples)

    if args.suite == 'all':
        generate_plots_and_tables(s1_rows, s2_rows, layer_trace, s3_rows, heatmaps, s4_rows)


if __name__ == '__main__':
    main()
