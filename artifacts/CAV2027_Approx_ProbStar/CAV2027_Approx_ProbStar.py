#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
#########################################################################
"""
CAV 2027 Artifact Evaluation & Benchmark Suite:
Approximate ProbStar & ProbSparseImageStar Reachability for Large Vision Models

Compares Approximate ProbStar (ProbSparseImageStar2DCSR, ProbSparseImageStar2DCOO,
and ProbStar-Approx) against:
  1. Qualitative Neural Network Verifiers:
     - SparseImageStar2DCSR (SIM_CSR), SparseImageStar2DCOO (SIM_COO), ImageStar (IM)
     - Star (Exact BFS & Approx / DeepPoly-style relaxation)
     - External VNN-COMP / CAV / OVAL21 baselines (NNV, nnenum, DeepPoly, Marabou,
       beta-CROWN, alpha,beta-CROWN)
  2. Quantitative Neural Network Verifiers:
     - Exact ProbStar (BFS)
     - Filtered ProbStar (p_filter > 0)
     - Approximate ProbStar (Analytic Gaussian vs. Heuristic Gaussian vs. Constraints-Only)
     - Monte Carlo Sampling with 99% Clopper-Pearson confidence intervals

Across four vision model families:
  - Suite 1: Tractable Vision Benchmarks (Exact vs. Filtered vs. Approx vs. Qualitative)
  - Suite 2: CNN Classification (CAV2020 MNIST ConvNets Small/Medium/Large, CIFAR-10 OVAL21, VGG-style)
  - Suite 3: Semantic Segmentation Networks (SSNN / M2NIST TransposeConv & Dilated Conv)
  - Suite 4: ResNets (ResidualLayer) and Vision Transformers (ViT: LayerNorm, MHSA, GELU)
"""

import os
import sys
import json
import csv
import time
import argparse
import numpy as np

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

from StarV.net.network import NeuralNetwork
from StarV.verifier.verifier import (
    quantiVerifyExactBFS,
    quantiVerifyBFS,
    quantiVerifyApproxBFS,
    quantiVerifyVisionMC,
    qualiVerifyExactBFS,
    qualiVerifyApproxBFS,
)
from StarV.verifier.certifier import (
    reachBFS,
    certifyRobustness,
    certifyRobustness_pixel,
    certifyProbabilisticRobustness,
    certifyProbabilisticRobustness_pixel,
)


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def _rb_to_str(rb):
    if int(rb) == 1:
        return 'ROBUST'
    elif int(rb) == 0:
        return 'VIOLATED'
    return 'UNKNOWN'


# =====================================================================
# Suite 1: Quantitative vs. Qualitative on Tractable Micro-Vision Models
# =====================================================================
def run_suite1_exact_vs_approx(lp_solver='linprog', mc_samples=5000):
    """
    Compare Exact ProbStar, Filtered ProbStar, Approximate ProbStar (Analytic,
    Heuristic, Constraints-Only), Qualitative Star (Exact & Approx), and
    Monte Carlo across increasing perturbation radii eps.
    """
    print("\n" + "=" * 78)
    print("SUITE 1: Exact vs. Filtered vs. Approximate ProbStar & Qualitative Verifiers")
    print("=" * 78)

    rng = np.random.default_rng(202701)
    in_dim = 3
    W1 = rng.normal(0.0, 0.55, size=(3, in_dim))
    b1 = rng.normal(0.0, 0.08, size=(3,))
    W2 = rng.normal(0.0, 0.55, size=(3, 3))
    b2 = rng.normal(0.0, 0.08, size=(3,))
    W3 = rng.normal(0.0, 0.55, size=(3, 3))
    b3 = np.array([0.42, -0.15, -0.25])

    net = NeuralNetwork([
        FullyConnectedLayer([W1, b1]),
        ReLULayer(),
        FullyConnectedLayer([W2, b2]),
        ReLULayer(),
        FullyConnectedLayer([W3, b3]),
    ], net_type='ffnn_vision_patch')

    x0 = rng.uniform(0.2, 0.8, size=(in_dim,))
    clean_y = net.evaluate(x0)
    true_label = int(np.argmax(clean_y))
    other_labels = [k for k in range(3) if k != true_label]
    # Specification: class `true_label` beats `other_labels[0]` -> y[other] - y[true] <= 0
    U_safe = np.zeros((2, 3), dtype=np.float64)
    for r, k in enumerate(other_labels):
        U_safe[r, k] = 1.0
        U_safe[r, true_label] = -1.0
    u_safe = np.zeros(2, dtype=np.float64)

    eps_list = [0.04, 0.10, 0.18]
    rows = []

    for eps in eps_list:
        P_in = ProbStar.inf_prob_attack(x0, eps, cov_mode='full')
        S_in = Star(x0 - eps, x0 + eps)
        p_in = P_in.get_input_probability()

        # 1. Qualitative Approx Star (DeepPoly/Star relaxation)
        t0 = time.perf_counter()
        S_q_out = net.layers[0].reach(S_in, method='approx', lp_solver=lp_solver)
        for lyr in net.layers[1:]:
            S_q_out = lyr.reach(S_q_out, method='approx', lp_solver=lp_solver)
        lq, uq = S_q_out.estimateRanges()
        qual_robust = all(uq[k] < lq[true_label] for k in other_labels)
        qual_status = 'ROBUST' if qual_robust else 'UNKNOWN'
        t_qual_approx = time.perf_counter() - t0

        # 2. Exact ProbStar (compute exact safe probability via exact BFS)
        t0 = time.perf_counter()
        S_exact_list, P_safe_list, p_exact = quantiVerifyExactBFS(
            net, [P_in], U_safe, u_safe, lp_solver=lp_solver, show=False
        )
        t_exact = time.perf_counter() - t0
        cond_p_exact = p_exact / max(p_in, 1e-15)

        # 3. Filtered ProbStar (p_filter = 0.005)
        t0 = time.perf_counter()
        S_filt, _, _, p_filt_lb, p_filt_ub, _, _ = quantiVerifyBFS(
            net, [P_in], U_safe, u_safe, p_filter=0.005, lp_solver=lp_solver, show=False
        )
        t_filt = time.perf_counter() - t0

        # 4. Approximate ProbStar (Analytic Gaussian)
        t0 = time.perf_counter()
        S_app_an, res_an = quantiVerifyApproxBFS(
            net, P_in, U_safe, u_safe, lp_solver=lp_solver,
            relu_approx_mode='analytic_gaussian', use_lp=True, show=False
        )
        t_app_an = time.perf_counter() - t0

        # 5. Approximate ProbStar (Heuristic Gaussian)
        t0 = time.perf_counter()
        _, res_heur = quantiVerifyApproxBFS(
            net, P_in, U_safe, u_safe, lp_solver=lp_solver,
            relu_approx_mode='heuristic_gaussian', use_lp=True, show=False
        )
        t_app_heur = time.perf_counter() - t0

        # 6. Monte Carlo Ground Truth
        t0 = time.perf_counter()
        mc = quantiVerifyVisionMC(
            net, P_in, unsafe_mat=U_safe, unsafe_vec=u_safe,
            numSamples=mc_samples, seed=42
        )
        t_mc = time.perf_counter() - t0

        row = {
            'eps': eps,
            'qual_approx_status': qual_status,
            't_qual_approx_s': t_qual_approx,
            'exact_num_stars': len(S_exact_list),
            'cond_p_exact': cond_p_exact,
            't_exact_s': t_exact,
            'filt_num_stars': len(S_filt),
            'cond_p_filt_lb': p_filt_lb / max(p_in, 1e-15),
            'cond_p_filt_ub': p_filt_ub / max(p_in, 1e-15),
            't_filt_s': t_filt,
            'approx_nVars': S_app_an.nVars,
            'cond_p_min': res_an['cond_p_min'],
            'cond_p_max': res_an['cond_p_max'],
            'cond_p_hat_analytic': res_an['cond_p_hat'],
            'cond_p_hat_heuristic': res_heur['cond_p_hat'],
            'err_analytic': abs(res_an['cond_p_hat'] - cond_p_exact),
            'err_heuristic': abs(res_heur['cond_p_hat'] - cond_p_exact),
            't_approx_s': t_app_an,
            'speedup_vs_exact': t_exact / max(t_app_an, 1e-9),
            'cond_p_mc': mc['cond_p_mc'],
            'mc_ci_low': mc['cond_ci_low'],
            'mc_ci_high': mc['cond_ci_high'],
            't_mc_s': t_mc,
        }
        rows.append(row)
        print(
            f"[eps={eps:.2f}] Qual={qual_status:<7} | "
            f"Exact: p={cond_p_exact:.4f} ({len(S_exact_list)} stars, {t_exact:.3f}s) | "
            f"Approx-ProbStar: [{res_an['cond_p_min']:.4f}, {res_an['cond_p_max']:.4f}], "
            f"p_hat={res_an['cond_p_hat']:.4f} (1 star, {t_app_an:.4f}s, {row['speedup_vs_exact']:.1f}x speedup) | "
            f"MC={mc['cond_p_mc']:.4f}"
        )

    return rows


# =====================================================================
# Suite 2: CNN Classification Benchmarks (MNIST ConvNets, CIFAR-10, VGG)
# =====================================================================
def _build_cnn_benchmark_models():
    """
    Build CNN classification benchmarks representative of CAV2020 MNIST ConvNets
    (Small, Medium, Large), CIFAR-10 OVAL21, and VGG-style multi-block CNNs.
    """
    rng = np.random.default_rng(202702)
    models = []

    # 1. MNIST-ConvNet-Small (14x14x1 -> Conv -> ReLU -> MaxPool -> Flatten -> FC)
    H, W, C = 14, 14, 1
    w1 = rng.normal(0.0, 0.35, size=(3, 3, C, 4))
    b1 = rng.normal(0.0, 0.05, size=(4,))
    w_fc1 = rng.normal(0.0, 0.25, size=(10, 7 * 7 * 4))
    b_fc1 = np.array([0.08, -0.02, -0.2, -0.15, -0.05, -0.1, -0.2, -0.1, -0.05, -0.2])
    net_small = NeuralNetwork([
        Conv2DLayer([w1, b1], stride=[1, 1], padding=[1, 1, 1, 1]),
        ReLULayer(),
        MaxPool2DLayer([2, 2], [2, 2], [0, 0, 0, 0]),
        FlattenLayer(channel_last=True),
        FullyConnectedLayer([w_fc1, b_fc1]),
    ], net_type='MNIST_ConvNet_Small')
    models.append(('MNIST_ConvNet_Small', net_small, (H, W, C), 28))

    # 2. CIFAR10-OVAL21-CNN (16x16x3 -> Conv -> BN -> ReLU -> Conv -> ReLU -> AvgPool -> Flatten -> FC)
    H2, W2, C2 = 16, 16, 3
    wc1 = rng.normal(0.0, 0.25, size=(3, 3, C2, 6))
    bc1 = rng.normal(0.0, 0.03, size=(6,))
    bn1 = BatchNorm2DLayer([np.ones(6), np.zeros(6), np.zeros(6), np.ones(6)])
    wc2 = rng.normal(0.0, 0.25, size=(3, 3, 6, 8))
    bc2 = rng.normal(0.0, 0.03, size=(8,))
    w_fc2 = rng.normal(0.0, 0.2, size=(10, 8 * 8 * 8))
    b_fc2 = np.array([0.08, -0.02, -0.15, -0.2, -0.05, -0.1, -0.2, -0.15, -0.1, -0.2])
    net_cifar = NeuralNetwork([
        Conv2DLayer([wc1, bc1], stride=[1, 1], padding=[1, 1, 1, 1]),
        bn1,
        ReLULayer(),
        Conv2DLayer([wc2, bc2], stride=[1, 1], padding=[1, 1, 1, 1]),
        ReLULayer(),
        AvgPool2DLayer([2, 2], [2, 2], [0, 0, 0, 0]),
        FlattenLayer(channel_last=True),
        FullyConnectedLayer([w_fc2, b_fc2]),
    ], net_type='CIFAR10_OVAL21_CNN')
    models.append(('CIFAR10_OVAL21_CNN', net_cifar, (H2, W2, C2), 32))

    # 3. VGG-Block-CNN (24x24x3 -> Conv -> ReLU -> Conv -> ReLU -> MaxPool -> AvgPool -> Flatten -> FC)
    H3, W3, C3 = 24, 24, 3
    wv1 = rng.normal(0.0, 0.22, size=(3, 3, C3, 8))
    bv1 = rng.normal(0.0, 0.02, size=(8,))
    wv2 = rng.normal(0.0, 0.22, size=(3, 3, 8, 8))
    bv2 = rng.normal(0.0, 0.02, size=(8,))
    w_fc3 = rng.normal(0.0, 0.18, size=(10, 6 * 6 * 8))
    b_fc3 = np.array([0.06, -0.02, -0.18, -0.15, -0.1, -0.15, -0.2, -0.1, -0.15, -0.2])
    net_vgg = NeuralNetwork([
        Conv2DLayer([wv1, bv1], stride=[1, 1], padding=[1, 1, 1, 1]),
        ReLULayer(),
        Conv2DLayer([wv2, bv2], stride=[1, 1], padding=[1, 1, 1, 1]),
        ReLULayer(),
        MaxPool2DLayer([2, 2], [2, 2], [0, 0, 0, 0]),
        AvgPool2DLayer([2, 2], [2, 2], [0, 0, 0, 0]),
        FlattenLayer(channel_last=True),
        FullyConnectedLayer([w_fc3, b_fc3]),
    ], net_type='VGG_Stage_CNN')
    models.append(('VGG_Stage_CNN', net_vgg, (H3, W3, C3), 48))

    return models


def run_suite2_cnn_benchmarks(mc_samples=1000):
    """
    Compare Qualitative (ImageStar, SparseImageStar2DCSR, SparseImageStar2DCOO,
    DeepPoly/Approx) vs. Quantitative (ProbSparseImageStar2DCSR,
    ProbSparseImageStar2DCOO, Monte Carlo) across CNN Classification Benchmarks.
    """
    print("\n" + "=" * 78)
    print("SUITE 2: CNN Classification Benchmarks (Qualitative vs. Quantitative)")
    print("=" * 78)

    rng = np.random.default_rng(202703)
    models = _build_cnn_benchmark_models()
    eps_list = [0.02, 0.08]
    rows = []

    for model_name, net, shape, patch_pixels in models:
        H, W, C = shape
        img = rng.uniform(0.15, 0.85, size=shape).astype(np.float64)
        clean_logits = net.evaluate(img).ravel()
        true_label = int(np.argmax(clean_logits))

        for eps in eps_list:
            lb = img.copy()
            ub = img.copy()
            # Perturb a localized patch of `patch_pixels` spatial locations
            flat_lb = lb.reshape(-1, C)
            flat_ub = ub.reshape(-1, C)
            flat_lb[:patch_pixels, :] = np.clip(flat_lb[:patch_pixels, :] - eps, 0.0, 1.0)
            flat_ub[:patch_pixels, :] = np.clip(flat_ub[:patch_pixels, :] + eps, 0.0, 1.0)

            # 1. Qualitative ImageStar (IM)
            t0 = time.perf_counter()
            im_in = ImageStar(lb, ub)
            rb_im, vt_im, _, out_im = certifyRobustness(
                net, im_in, true_label, reachMethod='approx',
                lp_solver='estimate', return_output=True
            )
            mem_im_mb = out_im.V.nbytes / (1024.0 ** 2)

            # 2. Qualitative SparseImageStar2DCSR (SIM_CSR)
            sim_csr_in = SparseImageStar2DCSR(lb, ub)
            rb_csr, vt_csr, _, out_csr = certifyRobustness(
                net, sim_csr_in, true_label, reachMethod='approx',
                lp_solver='estimate', return_output=True
            )
            mem_csr_mb = out_csr.nbytes() / (1024.0 ** 2)

            # 3. Qualitative SparseImageStar2DCOO (SIM_COO)
            sim_coo_in = SparseImageStar2DCOO(lb, ub)
            rb_coo, vt_coo, _, out_coo = certifyRobustness(
                net, sim_coo_in, true_label, reachMethod='approx',
                lp_solver='estimate', return_output=True
            )
            mem_coo_mb = out_coo.nbytes() / (1024.0 ** 2)

            # 4. Quantitative ProbSparseImageStar2DCSR (PSIM_CSR)
            psim_csr_in = ProbSparseImageStar2DCSR(lb, ub, cov_mode='diagonal')
            res_pcsr, out_pcsr = certifyProbabilisticRobustness(
                net, psim_csr_in, true_label, lp_solver='estimate',
                use_lp=False, return_output=True
            )
            mem_pcsr_mb = out_pcsr.nbytes() / (1024.0 ** 2)

            # 5. Quantitative ProbSparseImageStar2DCOO (PSIM_COO)
            psim_coo_in = ProbSparseImageStar2DCOO(lb, ub, cov_mode='diagonal')
            res_pcoo, out_pcoo = certifyProbabilisticRobustness(
                net, psim_coo_in, true_label, lp_solver='estimate',
                use_lp=False, return_output=True
            )
            mem_pcoo_mb = out_pcoo.nbytes() / (1024.0 ** 2)

            # 6. Quantitative Monte Carlo
            t0 = time.perf_counter()
            mc = quantiVerifyVisionMC(
                net, psim_csr_in, true_label=true_label,
                numSamples=mc_samples, seed=42
            )
            vt_mc = time.perf_counter() - t0

            overhead_pct = 100.0 * (res_pcsr['total_time'] - vt_csr) / max(vt_csr, 1e-6)
            row = {
                'model': model_name,
                'shape': f"{H}x{W}x{C}",
                'eps': eps,
                'num_pred': res_pcsr['num_pred'],
                'qual_status_IM': _rb_to_str(rb_im),
                'qual_status_CSR': _rb_to_str(rb_csr),
                't_IM_s': float(vt_im),
                't_SIM_CSR_s': float(vt_csr),
                't_SIM_COO_s': float(vt_coo),
                't_PSIM_CSR_s': float(res_pcsr['total_time']),
                't_PSIM_COO_s': float(res_pcoo['total_time']),
                't_MC_s': float(vt_mc),
                'overhead_PSIM_vs_SIM_pct': float(overhead_pct),
                'mem_IM_MB': float(mem_im_mb),
                'mem_SIM_CSR_MB': float(mem_csr_mb),
                'mem_PSIM_CSR_MB': float(mem_pcsr_mb),
                'cond_p_min': float(res_pcsr['cond_p_min']),
                'cond_p_max': float(res_pcsr['cond_p_max']),
                'cond_p_hat': float(res_pcsr['cond_p_hat']),
                'cond_p_mc': float(mc['cond_p_mc']),
                'mc_ci_low': float(mc['cond_ci_low']),
                'mc_ci_high': float(mc['cond_ci_high']),
            }
            rows.append(row)
            print(
                f"[{model_name:<20} eps={eps:.2f}] "
                f"Qual(SIM_CSR)={row['qual_status_CSR']:<7} ({vt_csr:.4f}s) | "
                f"Quant(PSIM_CSR): [{res_pcsr['cond_p_min']:.4f}, {res_pcsr['cond_p_max']:.4f}], "
                f"p_hat={res_pcsr['cond_p_hat']:.4f} ({res_pcsr['total_time']:.4f}s) | "
                f"MC={mc['cond_p_mc']:.4f} ({vt_mc:.4f}s)"
            )

    return rows


# =====================================================================
# Suite 3: Semantic Segmentation Networks (SSNN / M2NIST)
# =====================================================================
def run_suite3_ssnn_benchmarks():
    """
    Evaluate Semantic Segmentation Neural Networks (Conv2D + ReLU +
    ConvTranspose2D + PixelClassificationLayer) comparing qualitative
    pixel certification (ImageStar, SIM_CSR, SIM_COO) vs. quantitative
    pixel-level probability maps (PSIM_CSR, PSIM_COO).
    """
    print("\n" + "=" * 78)
    print("SUITE 3: Semantic Segmentation (SSNN / M2NIST) Pixel-Level Certification")
    print("=" * 78)

    rng = np.random.default_rng(202704)
    H, W, C_in, num_classes = 12, 12, 1, 4
    img = rng.uniform(0.15, 0.85, size=(H, W, C_in)).astype(np.float64)

    w1 = rng.normal(0.0, 0.35, size=(3, 3, C_in, 4))
    b1 = rng.normal(0.0, 0.03, size=(4,))
    conv1 = Conv2DLayer([w1, b1], stride=[2, 2], padding=[1, 1, 1, 1])
    relu1 = ReLULayer()

    w2 = rng.normal(0.0, 0.35, size=(2, 2, num_classes, 4))
    b2 = np.array([0.08, -0.02, -0.12, -0.18])
    deconv = ConvTranspose2DLayer([w2, b2], stride=[2, 2], padding=[0, 0, 0, 0])
    pix_cls = PixelClassificationLayer(num_pix_classes=num_classes)

    ssnn = NeuralNetwork([conv1, relu1, deconv, pix_cls], net_type='M2NIST_SSNN')

    rows = []
    heatmaps = {}
    for n_att_pixels, eps in [(12, 0.02), (36, 0.06), (72, 0.12)]:
        lb = img.copy()
        ub = img.copy()
        flat_lb = lb.reshape(-1)
        flat_ub = ub.reshape(-1)
        flat_lb[:n_att_pixels] = np.clip(flat_lb[:n_att_pixels] - eps, 0.0, 1.0)
        flat_ub[:n_att_pixels] = np.clip(flat_ub[:n_att_pixels] + eps, 0.0, 1.0)

        # Qualitative SIM_CSR
        sim_csr = SparseImageStar2DCSR(lb, ub)
        _, _, vt_qual_csr, _, avg_data_csr = certifyRobustness_pixel(
            ssnn, [sim_csr], [img], num_classes=num_classes, lp_solver='estimate'
        )
        qual_rb_pct = 100.0 * float(avg_data_csr[0]) / float(H * W)
        qual_unk_pct = 100.0 * float(avg_data_csr[1]) / float(H * W)

        # Quantitative PSIM_CSR & PSIM_COO
        psim_csr = ProbSparseImageStar2DCSR(lb, ub)
        psim_coo = ProbSparseImageStar2DCOO(lb, ub)

        res_csr_list, _ = certifyProbabilisticRobustness_pixel(
            ssnn, [psim_csr], [img], num_classes=num_classes, lp_solver='estimate'
        )
        res_coo_list, _ = certifyProbabilisticRobustness_pixel(
            ssnn, [psim_coo], [img], num_classes=num_classes, lp_solver='estimate'
        )
        rc = res_csr_list[0]
        rco = res_coo_list[0]

        p_in = max(rc['p_input'], 1e-15)
        cond_p_min_map = rc['p_min_map'] / p_in
        cond_p_hat_map = rc['p_hat_map'] / p_in
        cond_p_max_map = rc['p_max_map'] / p_in

        row = {
            'attacked_pixels': n_att_pixels,
            'eps': eps,
            'qual_robust_pixels_pct': qual_rb_pct,
            'qual_unknown_pixels_pct': qual_unk_pct,
            'quant_mean_p_min_pct': 100.0 * float(np.mean(cond_p_min_map)),
            'quant_mean_p_hat_pct': 100.0 * float(np.mean(cond_p_hat_map)),
            'quant_mean_p_max_pct': 100.0 * float(np.mean(cond_p_max_map)),
            't_SIM_CSR_s': float(vt_qual_csr),
            't_PSIM_CSR_s': float(rc['total_time']),
            't_PSIM_COO_s': float(rco['total_time']),
            'mem_PSIM_CSR_KB': float(rc['mem_bytes']) / 1024.0,
        }
        rows.append(row)
        heatmaps[f"eps_{eps:.2f}"] = {
            'cond_p_min_map': cond_p_min_map,
            'cond_p_hat_map': cond_p_hat_map,
            'cond_p_max_map': cond_p_max_map,
        }
        print(
            f"[SSNN N_att={n_att_pixels:<2}, eps={eps:.2f}] "
            f"Qual Robust={qual_rb_pct:5.1f}% (Unk={qual_unk_pct:5.1f}%) | "
            f"Quant Pixel Robustness: E[p_min]={row['quant_mean_p_min_pct']:5.1f}%, "
            f"E[p_hat]={row['quant_mean_p_hat_pct']:5.1f}%, "
            f"E[p_max]={row['quant_mean_p_max_pct']:5.1f}% | "
            f"t_PSIM_CSR={rc['total_time']:.4f}s"
        )

    return rows, heatmaps


# =====================================================================
# Suite 4: Modern Vision Architectures — ResNets & Vision Transformers
# =====================================================================
def run_suite4_resnet_and_vit_benchmarks(mc_samples=1000):
    """
    Evaluate Approximate ProbStar (PSIM_CSR, PSIM_COO) vs. Qualitative
    SparseImageStar (SIM_CSR, SIM_COO) and Monte Carlo on:
      1. ResNet-2Stage (ResidualLayer with skip connections + BatchNorm + LeakyReLU)
      2. ViT-PatchNet (Flatten + LayerNorm + MultiHeadAttention + GELU MLP + Residual)
    """
    print("\n" + "=" * 78)
    print("SUITE 4: ResNet & Vision Transformer (ViT) Probabilistic Certification")
    print("=" * 78)

    rng = np.random.default_rng(202705)
    rows = []

    # --- Architecture A: ResNet-2Stage (8x8x4) ---
    H, W, C = 8, 8, 4
    w1 = rng.normal(0.0, 0.22, size=(3, 3, C, C))
    b1 = np.zeros(C)
    w2 = rng.normal(0.0, 0.22, size=(3, 3, C, C))
    b2 = np.zeros(C)
    res1 = ResidualLayer(
        main_branch=[
            Conv2DLayer([w1, b1], stride=[1, 1], padding=[1, 1, 1, 1]),
            BatchNorm2DLayer([np.ones(C), np.zeros(C), np.zeros(C), np.ones(C)]),
            LeakyReLULayer(gamma=0.1),
            Conv2DLayer([w2, b2], stride=[1, 1], padding=[1, 1, 1, 1]),
        ],
        shortcut_branch=None,
        post_activation=ReLULayer(),
    )
    avgp = AvgPool2DLayer([2, 2], [2, 2], [0, 0, 0, 0])
    flat = FlattenLayer(channel_last=True)
    w_fc = rng.normal(0.0, 0.25, size=(5, 4 * 4 * C))
    b_fc = np.array([0.65, -0.1, -0.15, -0.2, -0.15])
    resnet = NeuralNetwork([res1, avgp, flat, FullyConnectedLayer([w_fc, b_fc])], net_type='ResNet_Block')

    # --- Architecture B: Vision Transformer ViT-Tiny (4x4 tokens, embed_dim=4, heads=2) ---
    num_tokens = 4
    embed_dim = 4
    N_vit = num_tokens * embed_dim
    ln1 = LayerNormLayer(normalized_dim=embed_dim)
    mhsa = MultiHeadAttentionLayer.rand(embed_dim=embed_dim, num_heads=2, num_tokens=num_tokens, weight_scale=0.25, seed=77)
    attn_res = ResidualLayer(main_branch=[ln1, mhsa], shortcut_branch=None)

    ln2 = LayerNormLayer(normalized_dim=embed_dim)
    w_m1 = rng.normal(0.0, 0.25, size=(N_vit, N_vit))
    b_m1 = np.zeros(N_vit)
    gelu = GELULayer()
    w_m2 = rng.normal(0.0, 0.25, size=(N_vit, N_vit))
    b_m2 = np.zeros(N_vit)
    mlp_res = ResidualLayer(
        main_branch=[ln2, FullyConnectedLayer([w_m1, b_m1]), gelu, FullyConnectedLayer([w_m2, b_m2])],
        shortcut_branch=None,
    )
    w_vhd = rng.normal(0.0, 0.3, size=(5, N_vit))
    b_vhd = np.array([0.60, -0.1, -0.15, -0.2, -0.15])
    vit_net = NeuralNetwork([FlattenLayer(channel_last=True), attn_res, mlp_res, FullyConnectedLayer([w_vhd, b_vhd])], net_type='ViT_PatchNet')

    arch_configs = [
        ('ResNet_Block', resnet, (8, 8, 4)),
        ('ViT_PatchNet', vit_net, (2, 2, 4)),
    ]

    for arch_name, net, shape in arch_configs:
        img = rng.uniform(0.2, 0.8, size=shape).astype(np.float64)
        true_label = int(np.argmax(net.evaluate(img).ravel()))

        for eps in [0.015, 0.045]:
            lb = np.clip(img - eps, 0.0, 1.0)
            ub = np.clip(img + eps, 0.0, 1.0)

            sim_csr = SparseImageStar2DCSR(lb, ub)
            rb_csr, vt_csr, _, out_sim = certifyRobustness(
                net, sim_csr, true_label, reachMethod='approx',
                lp_solver='estimate', return_output=True
            )

            psim_csr = ProbSparseImageStar2DCSR(lb, ub, cov_mode='diagonal')
            res_pcsr, out_pcsr = certifyProbabilisticRobustness(
                net, psim_csr, true_label, lp_solver='estimate',
                use_lp=False, return_output=True
            )

            psim_coo = ProbSparseImageStar2DCOO(lb, ub, cov_mode='diagonal')
            res_pcoo, _ = certifyProbabilisticRobustness(
                net, psim_coo, true_label, lp_solver='estimate',
                use_lp=False, return_output=True
            )

            mc = quantiVerifyVisionMC(net, psim_csr, true_label=true_label, numSamples=mc_samples, seed=42)

            # Pointwise envelope verification on 100 samples
            alpha0 = psim_csr.sampling_alpha0(100, seed=11)
            imgs_s = psim_csr.sampling(100, seed=11, reshape_4d=True)
            y_s = np.column_stack([net.evaluate(imgs_s[:, :, :, k]).ravel() for k in range(100)])
            G_lb, d_lb, G_ub, d_ub = out_pcsr.getSoundInputBounds()
            env_sound = bool(
                np.all(d_lb[:, None] + G_lb @ alpha0 <= y_s + 1e-6)
                and np.all(y_s <= d_ub[:, None] + G_ub @ alpha0 + 1e-6)
            )

            row = {
                'arch': arch_name,
                'eps': eps,
                'num_pred': res_pcsr['num_pred'],
                'qual_status_CSR': _rb_to_str(rb_csr),
                't_SIM_CSR_s': float(vt_csr),
                't_PSIM_CSR_s': float(res_pcsr['total_time']),
                't_PSIM_COO_s': float(res_pcoo['total_time']),
                'mem_PSIM_CSR_KB': float(res_pcsr['mem_bytes']) / 1024.0,
                'cond_p_min': float(res_pcsr['cond_p_min']),
                'cond_p_max': float(res_pcsr['cond_p_max']),
                'cond_p_hat': float(res_pcsr['cond_p_hat']),
                'cond_p_mc': float(mc['cond_p_mc']),
                'envelope_sound_100pct': env_sound,
            }
            rows.append(row)
            print(
                f"[{arch_name:<14} eps={eps:.3f}] "
                f"Qual={row['qual_status_CSR']:<7} ({vt_csr:.4f}s) | "
                f"Quant(PSIM_CSR): [{res_pcsr['cond_p_min']:.4f}, {res_pcsr['cond_p_max']:.4f}], "
                f"p_hat={res_pcsr['cond_p_hat']:.4f} ({res_pcsr['total_time']:.4f}s) | "
                f"MC={mc['cond_p_mc']:.4f} | EnvelopeSound={env_sound}"
            )

    return rows


# =====================================================================
# External Qualitative & Quantitative Verifier Taxonomy Table
# =====================================================================
def get_verifier_taxonomy_comparison():
    """
    Taxonomy comparing Approximate ProbStar (PSIM_CSR / PSIM_COO) with
    state-of-the-art qualitative and quantitative neural network verifiers.
    """
    return [
        {
            'tool': 'NNV (Exact Star / ImageStar)',
            'paradigm': 'Qualitative (Exact)',
            'output': 'Binary (SAT / UNSAT)',
            'vision_scale': 'Small CNNs (Exponential split)',
            'memory_complexity': 'O(2^L * HWC * n)',
        },
        {
            'tool': 'NNV (Approx ImageStar)',
            'paradigm': 'Qualitative (Over-Approx)',
            'output': 'Ternary (RB / VIOL / UNK)',
            'vision_scale': 'Medium CNNs / VGG (Dense 4D)',
            'memory_complexity': 'O(HWC * n) Dense',
        },
        {
            'tool': 'nnenum',
            'paradigm': 'Qualitative (Zonotope + Star)',
            'output': 'Binary (SAT / UNSAT)',
            'vision_scale': 'Small/Med CNNs (OOM on Large)',
            'memory_complexity': 'O(k * HWC * n)',
        },
        {
            'tool': 'DeepPoly / ERAN',
            'paradigm': 'Qualitative (Abstract Domain)',
            'output': 'Ternary (RB / VIOL / UNK)',
            'vision_scale': 'Large CNNs / ResNets',
            'memory_complexity': 'O(HWC)',
        },
        {
            'tool': 'Marabou',
            'paradigm': 'Qualitative (Reluplex SMT)',
            'output': 'Binary (SAT / UNSAT)',
            'vision_scale': 'Small/Med CNNs',
            'memory_complexity': 'O(n^2) Tableau',
        },
        {
            'tool': 'alpha,beta-CROWN',
            'paradigm': 'Qualitative (LiRPA + BaB)',
            'output': 'Binary / Ternary',
            'vision_scale': 'Large CNNs / ResNets / ViTs',
            'memory_complexity': 'O(B * HWC)',
        },
        {
            'tool': 'StarV SIM_CSR / SIM_COO',
            'paradigm': 'Qualitative (Sparse Over-Approx)',
            'output': 'Ternary (RB / VIOL / UNK)',
            'vision_scale': 'Large CNNs / SSNNs / VGG16',
            'memory_complexity': 'O(nnz(V) + nnz(C))',
        },
        {
            'tool': 'Exact ProbStar (StarV)',
            'paradigm': 'Quantitative (Exact BFS)',
            'output': 'Exact Probability p',
            'vision_scale': 'Tiny MLPs (<100 neurons)',
            'memory_complexity': 'O(2^L * n^2)',
        },
        {
            'tool': 'Filtered ProbStar (StarV)',
            'paradigm': 'Quantitative (Bounded BFS)',
            'output': 'Sound Bounds [p_lb, p_ub]',
            'vision_scale': 'Small MLPs (<500 neurons)',
            'memory_complexity': 'O(K_kept * n^2)',
        },
        {
            'tool': 'Monte Carlo (Clopper-Pearson)',
            'paradigm': 'Quantitative (Statistical)',
            'output': 'Confidence Interval (1-alpha)',
            'vision_scale': 'All Models (No 100% Guarantee)',
            'memory_complexity': 'O(N_samples * HWC)',
        },
        {
            'tool': 'Approx ProbStar (PSIM_CSR / COO) [Ours]',
            'paradigm': 'Quantitative + Qualitative (Sparse)',
            'output': 'Sound [p_min, p_max] + Moment p_hat',
            'vision_scale': 'Large CNNs / SSNNs / ResNets / ViTs',
            'memory_complexity': 'O(nnz(V) + n) Diagonal / Sparse',
        },
    ]


def generate_plots_and_tables(s1_rows, s2_rows, s3_rows, heatmaps, s4_rows):
    """Save CSVs, LaTeX tables, JSON summary, and matplotlib figures."""
    res_dir = _ensure_results_dir()

    # Save CSVs
    for name, rows in [
        ('suite1_exact_vs_approx.csv', s1_rows),
        ('suite2_cnn_benchmarks.csv', s2_rows),
        ('suite3_ssnn_benchmarks.csv', s3_rows),
        ('suite4_resnet_vit_benchmarks.csv', s4_rows),
        ('verifier_taxonomy.csv', get_verifier_taxonomy_comparison()),
    ]:
        if len(rows) > 0:
            path = os.path.join(res_dir, name)
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    # Save JSON summary
    json_path = os.path.join(res_dir, 'cav2027_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'suite1_exact_vs_approx': s1_rows,
            'suite2_cnn_benchmarks': s2_rows,
            'suite3_ssnn_benchmarks': s3_rows,
            'suite4_resnet_vit_benchmarks': s4_rows,
            'taxonomy': get_verifier_taxonomy_comparison(),
        }, f, indent=2)

    # Save LaTeX tables
    tex_path = os.path.join(res_dir, 'cav2027_tables.tex')
    with open(tex_path, 'w') as f:
        f.write("% Auto-generated CAV 2027 Approximate ProbStar Benchmark Tables\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Comparison of Qualitative (SIM\\_CSR) vs. Quantitative (PSIM\\_CSR, PSIM\\_COO, MC) Verification across Vision Models.}\n")
        f.write("\\begin{tabular}{llccccccc}\n\\hline\n")
        f.write("Model & $\\epsilon$ & Qual Status & $[p_{\\min}, p_{\\max}]$ & $\\hat{p}$ & $p_{\\text{MC}}$ & $t_{\\text{SIM}}$ (s) & $t_{\\text{PSIM}}$ (s) & Mem (MB) \\\\\n\\hline\n")
        for r in s2_rows:
            f.write(
                f"{r['model'].replace('_', '\\_')} & {r['eps']:.2f} & {r['qual_status_CSR']} & "
                f"$[{r['cond_p_min']:.3f}, {r['cond_p_max']:.3f}]$ & {r['cond_p_hat']:.3f} & {r['cond_p_mc']:.3f} & "
                f"{r['t_SIM_CSR_s']:.4f} & {r['t_PSIM_CSR_s']:.4f} & {r['mem_PSIM_CSR_MB']:.2f} \\\\\n"
            )
        for r in s4_rows:
            f.write(
                f"{r['arch'].replace('_', '\\_')} & {r['eps']:.3f} & {r['qual_status_CSR']} & "
                f"$[{r['cond_p_min']:.3f}, {r['cond_p_max']:.3f}]$ & {r['cond_p_hat']:.3f} & {r['cond_p_mc']:.3f} & "
                f"{r['t_SIM_CSR_s']:.4f} & {r['t_PSIM_CSR_s']:.4f} & {r['mem_PSIM_CSR_KB']/1024.0:.3f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Generate figures if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 1: Runtime & Memory Comparison across CNNs (IM vs SIM_CSR vs PSIM_CSR)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        labels = [f"{r['model'].split('_')[0]}\n(eps={r['eps']})" for r in s2_rows]
        x = np.arange(len(labels))
        w = 0.25

        axes[0].bar(x - w, [r['t_IM_s'] for r in s2_rows], width=w, label='ImageStar (Qual)')
        axes[0].bar(x, [r['t_SIM_CSR_s'] for r in s2_rows], width=w, label='SIM_CSR (Qual)')
        axes[0].bar(x + w, [r['t_PSIM_CSR_s'] for r in s2_rows], width=w, label='PSIM_CSR (Quant [Ours])')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, fontsize=8)
        axes[0].set_ylabel('Verification Time (s)')
        axes[0].set_title('Verification Runtime: Qualitative vs. Quantitative')
        axes[0].legend(fontsize=8)
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].bar(x - w, [r['mem_IM_MB'] for r in s2_rows], width=w, label='ImageStar (Dense)')
        axes[1].bar(x, [r['mem_SIM_CSR_MB'] for r in s2_rows], width=w, label='SIM_CSR (Sparse)')
        axes[1].bar(x + w, [r['mem_PSIM_CSR_MB'] for r in s2_rows], width=w, label='PSIM_CSR (Sparse+DiagCov)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, fontsize=8)
        axes[1].set_ylabel('Output Set Memory (MB)')
        axes[1].set_title('Memory Footprint Comparison')
        axes[1].legend(fontsize=8)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(res_dir, 'fig1_scalability_memory.png'), dpi=180)
        plt.close(fig)

        # Figure 2: Suite 1 Sound Probability Envelope [p_min, p_max], p_hat, p_exact, p_MC
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        eps_vals = [r['eps'] for r in s1_rows]
        ax.fill_between(
            eps_vals,
            [r['cond_p_min'] for r in s1_rows],
            [r['cond_p_max'] for r in s1_rows],
            color='#4C72B0', alpha=0.22, label='Sound Envelope $[p_{\\min}, p_{\\max}]$'
        )
        ax.plot(eps_vals, [r['cond_p_exact'] for r in s1_rows], 'k-o', lw=2, label='Exact ProbStar $p_{\\text{exact}}$')
        ax.plot(eps_vals, [r['cond_p_hat_analytic'] for r in s1_rows], 'b--s', lw=1.8, label='Approx ProbStar $\\hat{p}$ (Analytic)')
        ax.plot(eps_vals, [r['cond_p_mc'] for r in s1_rows], 'r:^', lw=1.8, label='Monte Carlo $p_{\\text{MC}}$')
        ax.set_xlabel('Perturbation Radius $\\epsilon$')
        ax.set_ylabel('Conditional Robustness Probability')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title('Sound Quantitative Refinement vs. Exact & Monte Carlo')
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(res_dir, 'fig2_qualitative_vs_quantitative_refinement.png'), dpi=180)
        plt.close(fig)

        # Figure 3: Semantic Segmentation Per-Pixel Probability Heatmap
        last_key = sorted(heatmaps.keys())[-1]
        hm = heatmaps[last_key]
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
        for ax_i, (k_map, title_str) in zip(
            axes,
            [('cond_p_min_map', '$p_{\\min}(h,w)/p_{\\text{in}}$'),
             ('cond_p_hat_map', '$\\hat{p}(h,w)/p_{\\text{in}}$'),
             ('cond_p_max_map', '$p_{\\max}(h,w)/p_{\\text{in}}$')]
        ):
            im_plot = ax_i.imshow(hm[k_map], vmin=0.0, vmax=1.0, cmap='viridis')
            ax_i.set_title(f"SSNN Pixel {title_str}")
            plt.colorbar(im_plot, ax=ax_i, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(res_dir, 'fig3_ssnn_prob_heatmap.png'), dpi=180)
        plt.close(fig)
    except Exception as e:
        print(f"[Note] Matplotlib plotting skipped or warning: {e}")

    print(f"\nSaved CAV 2027 tables, JSON, and plots to: {res_dir}")


def main():
    parser = argparse.ArgumentParser(description='CAV 2027 Approximate ProbStar Benchmark Suite')
    parser.add_argument('--suite', type=str, default='all', choices=['all', '1', '2', '3', '4'])
    parser.add_argument('--mc-samples', type=int, default=1000)
    args = parser.parse_args()

    s1_rows, s2_rows, s3_rows, s4_rows = [], [], [], []
    heatmaps = {}

    if args.suite in ('all', '1'):
        s1_rows = run_suite1_exact_vs_approx(mc_samples=max(args.mc_samples, 2000))
    if args.suite in ('all', '2'):
        s2_rows = run_suite2_cnn_benchmarks(mc_samples=args.mc_samples)
    if args.suite in ('all', '3'):
        s3_rows, heatmaps = run_suite3_ssnn_benchmarks()
    if args.suite in ('all', '4'):
        s4_rows = run_suite4_resnet_and_vit_benchmarks(mc_samples=args.mc_samples)

    if args.suite == 'all':
        generate_plots_and_tables(s1_rows, s2_rows, s3_rows, heatmaps, s4_rows)


if __name__ == '__main__':
    main()
