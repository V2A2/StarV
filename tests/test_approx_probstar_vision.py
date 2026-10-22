#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
#########################################################################
"""
Comprehensive Unit & Comparative Verification Tests for Approximate ProbStar
and ProbSparseImageStar (CSR & COO) across Vision Architectures (CAV 2027).
"""

import unittest
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
from StarV.layer.ReshapeLayer import ReshapeLayer
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
    qualiVerifyApproxBFS,
)
from StarV.verifier.certifier import (
    reachBFS,
    certifyRobustness,
    certifyProbabilisticRobustness,
    certifyProbabilisticRobustness_pixel,
)
from StarV.verifier.crown import (
    certifyRobustness_CROWN,
    certifyProbabilisticRobustness_CROWN,
)


def _assert_prob_invariants(res, tol=1e-9):
    """Assert mathematical probability invariants on verification result dict."""
    p_in = res['p_input']
    assert 0.0 - tol <= p_in <= 1.0 + tol
    assert 0.0 - tol <= res['p_min'] <= res['p_max'] + tol
    assert res['p_max'] <= p_in + tol
    assert res['p_min'] - tol <= res['p_hat'] <= res['p_max'] + tol
    assert 0.0 - tol <= res['cond_p_min'] <= res['cond_p_max'] + tol
    assert res['cond_p_max'] <= 1.0 + tol
    assert res['cond_p_min'] - tol <= res['cond_p_hat'] <= res['cond_p_max'] + tol


def test_probstar_approx_vs_exact_and_mc_soundness():
    """
    Compare Approximate ProbStar against Exact ProbStar, Filtered ProbStar,
    Qualitative Star, and Monte Carlo on a 2-layer ReLU network.
    """
    rng = np.random.default_rng(2027)
    W1 = rng.normal(0.0, 0.6, size=(3, 3))
    b1 = rng.normal(0.0, 0.1, size=(3,))
    W2 = rng.normal(0.0, 0.6, size=(2, 3))
    b2 = np.array([0.35, -0.10])

    net = NeuralNetwork([
        FullyConnectedLayer([W1, b1]),
        ReLULayer(),
        FullyConnectedLayer([W2, b2]),
        ReLULayer(),
    ])

    x0 = np.array([0.2, -0.1, 0.3])
    eps = 0.25
    P_in = ProbStar.inf_prob_attack(x0, eps, cov_mode='full')
    S_in = Star(x0 - eps, x0 + eps)

    # Specification: y[1] - y[0] <= 0 (class 0 beats class 1)
    U = np.array([[1.0, -1.0]])  # y[0] <= y[1]  (unsafe spec)
    u_spec = np.array([0.0])

    # 1. Exact ProbStar
    _, _, p_exact = quantiVerifyExactBFS(net, [P_in], U, u_spec, lp_solver='linprog', show=False)

    # 2. Filtered ProbStar
    _, _, _, p_filt_lb, p_filt_ub, _, _ = quantiVerifyBFS(
        net, [P_in], U, u_spec, p_filter=1e-3, lp_solver='linprog', show=False
    )

    # 3. Approximate ProbStar (analytic_gaussian vs heuristic_gaussian vs constraints_only)
    S_approx, res_analytic = quantiVerifyApproxBFS(
        net, P_in, U, u_spec, lp_solver='linprog',
        relu_approx_mode='analytic_gaussian', use_lp=True, show=False
    )
    _, res_heuristic = quantiVerifyApproxBFS(
        net, P_in, U, u_spec, lp_solver='linprog',
        relu_approx_mode='heuristic_gaussian', use_lp=True, show=False
    )
    _, res_conly = quantiVerifyApproxBFS(
        net, P_in, U, u_spec, lp_solver='linprog',
        relu_approx_mode='constraints_only', use_lp=True, show=False
    )

    for res in (res_analytic, res_heuristic, res_conly):
        _assert_prob_invariants(res)
        # Soundness theorem: p_min <= p_exact <= p_max
        assert res['p_min'] - 1e-4 <= p_exact <= res['p_max'] + 1e-4
        assert p_filt_lb - 1e-4 <= p_exact <= p_filt_ub + 1e-4

    # 4. Verify envelope soundness pointwise on 500 random samples
    alpha0_samples = P_in.sampling_alpha0(500, seed=7)
    y_samples = net.evaluate(alpha0_samples)
    G_lb, d_lb, G_ub, d_ub = S_approx.getSoundInputBounds()
    env_lb = d_lb[:, None] + G_lb @ alpha0_samples
    env_ub = d_ub[:, None] + G_ub @ alpha0_samples
    assert np.all(env_lb <= y_samples + 1e-7), "Lower linear envelope violated!"
    assert np.all(y_samples <= env_ub + 1e-7), "Upper linear envelope violated!"

    # 5. Qualitative Star vs Approximate ProbStar geometric equivalence
    S_qual, _, _ = qualiVerifyApproxBFS(net, S_in, U, u_spec, lp_solver='linprog', show=False)
    l_app, u_app = S_approx.estimateRanges()
    l_qual, u_qual = S_qual.estimateRanges()
    np.testing.assert_allclose(l_app, l_qual, atol=1e-10)
    np.testing.assert_allclose(u_app, u_qual, atol=1e-10)


def test_cnn_qualitative_vs_quantitative_csr_coo_imagestar():
    """
    Verify a complete CNN pipeline (Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D ->
    AvgPool2D -> Flatten -> FC) across ImageStar, SparseImageStar2DCSR,
    SparseImageStar2DCOO, ProbSparseImageStar2DCSR, and ProbSparseImageStar2DCOO.
    """
    rng = np.random.default_rng(42)
    H, W, C_in = 8, 8, 2
    img = rng.uniform(0.1, 0.9, size=(H, W, C_in)).astype(np.float64)

    # Sparse localized L_inf attack on a 3x3 patch
    lb = img.copy()
    ub = img.copy()
    lb[2:5, 2:5, :] = np.clip(img[2:5, 2:5, :] - 0.08, 0.0, 1.0)
    ub[2:5, 2:5, :] = np.clip(img[2:5, 2:5, :] + 0.08, 0.0, 1.0)

    # Build CNN layers
    w_conv = rng.normal(0.0, 0.3, size=(3, 3, C_in, 3)).astype(np.float64)
    b_conv = rng.normal(0.0, 0.05, size=(3,)).astype(np.float64)
    conv = Conv2DLayer([w_conv, b_conv], stride=[1, 1], padding=[1, 1, 1, 1])

    mean_bn = np.zeros(3, dtype=np.float64)
    var_bn = np.ones(3, dtype=np.float64) * 0.5
    gamma_bn = np.array([1.1, 0.9, 1.0], dtype=np.float64)
    beta_bn = np.array([0.02, -0.01, 0.0], dtype=np.float64)
    bn = BatchNorm2DLayer([gamma_bn, beta_bn, mean_bn, var_bn])

    relu1 = ReLULayer()
    maxpool = MaxPool2DLayer([2, 2], [2, 2], [0, 0, 0, 0])
    avgpool = AvgPool2DLayer([2, 2], [2, 2], [0, 0, 0, 0])
    flatten = FlattenLayer(channel_last=True)
    # Output after 8x8 -> MaxPool(2,2) -> 4x4 -> AvgPool(2,2) -> 2x2x3 = 12
    w_fc = rng.normal(0.0, 0.4, size=(3, 12)).astype(np.float64)
    b_fc = np.array([0.5, -0.2, -0.3], dtype=np.float64)
    fc = FullyConnectedLayer([w_fc, b_fc])

    net = NeuralNetwork([conv, bn, relu1, maxpool, avgpool, flatten, fc], net_type='cnn')
    clean_logits = net.evaluate(img).ravel()
    true_label = int(np.argmax(clean_logits))

    im_star = ImageStar(lb, ub)
    sim_csr = SparseImageStar2DCSR(lb, ub)
    sim_coo = SparseImageStar2DCOO(lb, ub)
    psim_csr = ProbSparseImageStar2DCSR(lb, ub, cov_mode='diagonal')
    psim_coo = ProbSparseImageStar2DCOO(lb, ub, cov_mode='diagonal')

    out_im, _ = reachBFS(net, im_star, reachMethod='approx', lp_solver='estimate')
    out_csr, _ = reachBFS(net, sim_csr, reachMethod='approx', lp_solver='estimate')
    out_coo, _ = reachBFS(net, sim_coo, reachMethod='approx', lp_solver='estimate')
    out_pcsr, _ = reachBFS(net, psim_csr, reachMethod='approx', lp_solver='estimate')
    out_pcoo, _ = reachBFS(net, psim_coo, reachMethod='approx', lp_solver='estimate')

    assert isinstance(out_pcsr, ProbSparseImageStar2DCSR)
    assert isinstance(out_pcoo, ProbSparseImageStar2DCOO)

    # Qualitative ranges must match across all 5 representations
    l_im, u_im = out_im.estimateRanges()
    l_csr, u_csr = out_csr.estimateRanges()
    l_coo, u_coo = out_coo.estimateRanges()
    l_pcsr, u_pcsr = out_pcsr.estimateRanges()
    l_pcoo, u_pcoo = out_pcoo.estimateRanges()

    np.testing.assert_allclose(l_csr, l_im, atol=1e-6)
    np.testing.assert_allclose(u_csr, u_im, atol=1e-6)
    np.testing.assert_allclose(l_pcsr, l_csr, atol=1e-10)
    np.testing.assert_allclose(u_pcsr, u_csr, atol=1e-10)
    np.testing.assert_allclose(l_pcoo, l_coo, atol=1e-10)
    np.testing.assert_allclose(u_pcoo, u_coo, atol=1e-10)

    # Quantitative probability certification CSR vs COO must match identically
    res_csr, _ = certifyProbabilisticRobustness(
        net, psim_csr, true_label, lp_solver='estimate', use_lp=False
    )
    res_coo, _ = certifyProbabilisticRobustness(
        net, psim_coo, true_label, lp_solver='estimate', use_lp=False
    )
    _assert_prob_invariants(res_csr)
    _assert_prob_invariants(res_coo)
    np.testing.assert_allclose(res_csr['p_min'], res_coo['p_min'], atol=1e-10)
    np.testing.assert_allclose(res_csr['p_max'], res_coo['p_max'], atol=1e-10)
    np.testing.assert_allclose(res_csr['p_hat'], res_coo['p_hat'], atol=1e-10)

    # Pointwise soundness check against Monte Carlo samples
    alpha0 = psim_csr.sampling_alpha0(200, seed=99)
    imgs_mc = psim_csr.sampling(200, seed=99, reshape_4d=True)
    y_mc = np.column_stack([net.evaluate(imgs_mc[:, :, :, i]).ravel() for i in range(200)])

    G_lb, d_lb, G_ub, d_ub = out_pcsr.getSoundInputBounds()
    env_lb = d_lb[:, None] + G_lb @ alpha0
    env_ub = d_ub[:, None] + G_ub @ alpha0
    assert np.all(env_lb <= y_mc + 1e-6)
    assert np.all(y_mc <= env_ub + 1e-6)

    mc_res = quantiVerifyVisionMC(net, psim_csr, true_label=true_label, numSamples=300, seed=99)
    assert res_csr['cond_p_min'] - 0.05 <= mc_res['cond_p_mc'] <= res_csr['cond_p_max'] + 0.05


def test_semantic_segmentation_ssnn_probabilistic_pixels():
    """
    Test Semantic Segmentation Network (Conv2D -> ReLU -> ConvTranspose2D ->
    PixelClassificationLayer) with ProbSparseImageStar2DCSR and ProbSparseImageStar2DCOO.
    """
    rng = np.random.default_rng(101)
    H, W, C_in, num_classes = 6, 6, 1, 3
    img = rng.uniform(0.2, 0.8, size=(H, W, C_in)).astype(np.float64)
    lb = img.copy()
    ub = img.copy()
    lb[1:3, 1:3, 0] -= 0.05
    ub[1:3, 1:3, 0] += 0.05

    w1 = rng.normal(0.0, 0.35, size=(3, 3, C_in, 2)).astype(np.float64)
    b1 = rng.normal(0.0, 0.05, size=(2,)).astype(np.float64)
    conv1 = Conv2DLayer([w1, b1], stride=[2, 2], padding=[1, 1, 1, 1])

    relu = ReLULayer()

    # ConvTranspose2D weight shape in StarV default module: (H_k, W_k, C_out, C_in)
    w2 = rng.normal(0.0, 0.35, size=(2, 2, num_classes, 2)).astype(np.float64)
    b2 = np.array([0.4, -0.1, -0.2], dtype=np.float64)
    deconv = ConvTranspose2DLayer([w2, b2], stride=[2, 2], padding=[0, 0, 0, 0])

    pix_layer = PixelClassificationLayer(num_pix_classes=num_classes)
    ssnn = NeuralNetwork([conv1, relu, deconv, pix_layer], net_type='ssnn')

    psim_csr = ProbSparseImageStar2DCSR(lb, ub)
    psim_coo = ProbSparseImageStar2DCOO(lb, ub)

    res_csr_list, _ = certifyProbabilisticRobustness_pixel(
        ssnn, [psim_csr], [img], num_classes=num_classes, lp_solver='estimate'
    )
    res_coo_list, _ = certifyProbabilisticRobustness_pixel(
        ssnn, [psim_coo], [img], num_classes=num_classes, lp_solver='estimate'
    )

    res_csr = res_csr_list[0]
    res_coo = res_coo_list[0]

    assert res_csr['p_min_map'].shape == (H, W)
    assert res_csr['p_max_map'].shape == (H, W)
    assert res_csr['p_hat_map'].shape == (H, W)
    assert np.all(res_csr['p_min_map'] <= res_csr['p_hat_map'] + 1e-9)
    assert np.all(res_csr['p_hat_map'] <= res_csr['p_max_map'] + 1e-9)
    assert np.all(res_csr['p_max_map'] <= res_csr['p_input'] + 1e-9)
    np.testing.assert_allclose(res_csr['p_hat_map'], res_coo['p_hat_map'], atol=1e-9)


def test_resnet_residual_layer_and_leakyrelu():
    """
    Test ResidualLayer (both identity skip connection and projection shortcut)
    and LeakyReLULayer on Star, ProbStar, SparseImageStar2DCSR, and ProbSparseImageStar2DCSR.
    """
    rng = np.random.default_rng(77)
    H, W, C = 6, 6, 2
    img = rng.uniform(0.1, 0.9, size=(H, W, C)).astype(np.float64)
    lb = np.clip(img - 0.04, 0.0, 1.0)
    ub = np.clip(img + 0.04, 0.0, 1.0)

    w1 = rng.normal(0.0, 0.25, size=(3, 3, C, C)).astype(np.float64)
    b1 = np.zeros(C, dtype=np.float64)
    w2 = rng.normal(0.0, 0.25, size=(3, 3, C, C)).astype(np.float64)
    b2 = np.zeros(C, dtype=np.float64)

    res_block = ResidualLayer(
        main_branch=[
            Conv2DLayer([w1, b1], stride=[1, 1], padding=[1, 1, 1, 1]),
            LeakyReLULayer(gamma=0.1),
            Conv2DLayer([w2, b2], stride=[1, 1], padding=[1, 1, 1, 1]),
        ],
        shortcut_branch=None,
        post_activation=ReLULayer(),
    )
    flatten = FlattenLayer(channel_last=True)
    w_fc = rng.normal(0.0, 0.3, size=(3, H * W * C)).astype(np.float64)
    b_fc = np.array([0.3, -0.1, -0.2], dtype=np.float64)
    fc = FullyConnectedLayer([w_fc, b_fc])

    resnet = NeuralNetwork([res_block, flatten, fc], net_type='resnet')
    true_label = int(np.argmax(resnet.evaluate(img).ravel()))

    sim_csr = SparseImageStar2DCSR(lb, ub)
    psim_csr = ProbSparseImageStar2DCSR(lb, ub)

    out_sim, _ = reachBFS(resnet, sim_csr, reachMethod='approx', lp_solver='estimate')
    res_prob, out_psim = certifyProbabilisticRobustness(
        resnet, psim_csr, true_label, lp_solver='estimate', return_output=True
    )
    _assert_prob_invariants(res_prob)

    l_sim, u_sim = out_sim.estimateRanges()
    l_psim, u_psim = out_psim.estimateRanges()
    np.testing.assert_allclose(l_psim, l_sim, atol=1e-9)
    np.testing.assert_allclose(u_psim, u_sim, atol=1e-9)

    # Verify pointwise envelope soundness on 200 samples
    alpha0 = psim_csr.sampling_alpha0(200, seed=123)
    imgs_mc = psim_csr.sampling(200, seed=123, reshape_4d=True)
    y_mc = np.column_stack([resnet.evaluate(imgs_mc[:, :, :, i]).ravel() for i in range(200)])
    G_lb, d_lb, G_ub, d_ub = out_psim.getSoundInputBounds()
    assert np.all(d_lb[:, None] + G_lb @ alpha0 <= y_mc + 1e-6)
    assert np.all(y_mc <= d_ub[:, None] + G_ub @ alpha0 + 1e-6)


def test_vision_transformer_layernorm_gelu_mhsa_soundness():
    """
    Test Vision Transformer (ViT) layers (LayerNormLayer, MultiHeadAttentionLayer,
    GELULayer, ResidualLayer) on Star, ProbStar, SparseImageStar2DCSR, and
    ProbSparseImageStar2DCSR, verifying 100% pointwise envelope soundness.
    """
    rng = np.random.default_rng(314)
    num_tokens = 4
    embed_dim = 4
    num_heads = 2
    N = num_tokens * embed_dim

    ln1 = LayerNormLayer(normalized_dim=embed_dim)
    mhsa = MultiHeadAttentionLayer.rand(
        embed_dim=embed_dim, num_heads=num_heads, num_tokens=num_tokens,
        weight_scale=0.3, seed=314
    )
    attn_block = ResidualLayer(main_branch=[ln1, mhsa], shortcut_branch=None)

    ln2 = LayerNormLayer(normalized_dim=embed_dim)
    w_mlp1 = rng.normal(0.0, 0.3, size=(N, N)).astype(np.float64)
    b_mlp1 = rng.normal(0.0, 0.02, size=(N,)).astype(np.float64)
    gelu = GELULayer()
    w_mlp2 = rng.normal(0.0, 0.3, size=(N, N)).astype(np.float64)
    b_mlp2 = rng.normal(0.0, 0.02, size=(N,)).astype(np.float64)
    mlp_block = ResidualLayer(
        main_branch=[
            ln2,
            FullyConnectedLayer([w_mlp1, b_mlp1]),
            gelu,
            FullyConnectedLayer([w_mlp2, b_mlp2]),
        ],
        shortcut_branch=None,
    )

    w_head = rng.normal(0.0, 0.4, size=(3, N)).astype(np.float64)
    b_head = np.array([0.4, -0.1, -0.2], dtype=np.float64)
    head = FullyConnectedLayer([w_head, b_head])

    vit_net = NeuralNetwork([attn_block, mlp_block, head], net_type='vit')

    # Test with ProbSparseImageStar2DCSR (patch-token image shape 2 x 2 x 4)
    img = rng.uniform(-0.5, 0.5, size=(2, 2, embed_dim)).astype(np.float64)
    eps = 0.03
    lb = img - eps
    ub = img + eps

    psim_csr = ProbSparseImageStar2DCSR(lb, ub)
    sim_csr = SparseImageStar2DCSR(lb, ub)

    flatten_in = FlattenLayer(channel_last=True)
    vit_img_net = NeuralNetwork([flatten_in, attn_block, mlp_block, head], net_type='vit')
    true_label = int(np.argmax(vit_img_net.evaluate(img).ravel()))

    out_sim, _ = reachBFS(vit_img_net, sim_csr, reachMethod='approx', lp_solver='estimate')
    res_prob, out_psim = certifyProbabilisticRobustness(
        vit_img_net, psim_csr, true_label, lp_solver='estimate', return_output=True
    )
    _assert_prob_invariants(res_prob)

    l_sim, u_sim = out_sim.estimateRanges()
    l_psim, u_psim = out_psim.estimateRanges()
    np.testing.assert_allclose(l_psim, l_sim, atol=1e-9)
    np.testing.assert_allclose(u_psim, u_sim, atol=1e-9)

    # Pointwise envelope soundness on 300 Monte Carlo samples
    alpha0 = psim_csr.sampling_alpha0(300, seed=55)
    imgs_mc = psim_csr.sampling(300, seed=55, reshape_4d=True)
    y_mc = np.column_stack([vit_img_net.evaluate(imgs_mc[:, :, :, i]).ravel() for i in range(300)])

    G_lb, d_lb, G_ub, d_ub = out_psim.getSoundInputBounds()
    env_lb = d_lb[:, None] + G_lb @ alpha0
    env_ub = d_ub[:, None] + G_ub @ alpha0
    assert np.all(env_lb <= y_mc + 1e-6), "ViT lower envelope violated!"
    assert np.all(y_mc <= env_ub + 1e-6), "ViT upper envelope violated!"


def test_crown_vs_approx_probstar_2d_and_norm_layer():
    """
    Test CROWN qualitative & quantitative certification alongside PSIM_CSR
    on a CNN with a leading FullyConnectedLayer(W=None, b=...) normalization
    layer and 2D (H, W) SparseImageStar2DCSR / ProbSparseImageStar2DCSR inputs.
    """
    rng = np.random.default_rng(2027)
    H, W = 6, 6
    img_2d = rng.uniform(0.1, 0.9, size=(H, W)).astype(np.float64)
    lb_2d = img_2d.copy()
    ub_2d = img_2d.copy()
    lb_2d[1:3, 1:3] = np.clip(lb_2d[1:3, 1:3] - 0.04, 0.0, 1.0)
    ub_2d[1:3, 1:3] = np.clip(ub_2d[1:3, 1:3] + 0.04, 0.0, 1.0)

    norm_layer = FullyConnectedLayer([None, np.array([-0.5], dtype=np.float64)])
    w_conv = rng.normal(0.0, 0.35, size=(3, 3, 1, 2)).astype(np.float64)
    b_conv = np.zeros(2, dtype=np.float64)
    conv = Conv2DLayer([w_conv, b_conv], stride=[2, 2], padding=[1, 1, 1, 1])
    relu = ReLULayer()
    mp = MaxPool2DLayer([2, 2], [1, 1], [0, 0, 0, 0])
    flat = FlattenLayer(channel_last=True)
    w_fc = rng.normal(0.0, 0.4, size=(3, 2 * 2 * 2)).astype(np.float64)
    b_fc = np.array([0.4, -0.1, -0.2], dtype=np.float64)
    fc = FullyConnectedLayer([w_fc, b_fc])

    net = NeuralNetwork([norm_layer, conv, relu, mp, flat, fc], net_type='cnn_norm')
    true_label = int(np.argmax(net.evaluate(img_2d[:, :, None]).ravel()))

    sim_csr = SparseImageStar2DCSR(lb_2d, ub_2d)
    psim_csr = ProbSparseImageStar2DCSR(lb_2d, ub_2d)

    rb_sim, _, _, _ = certifyRobustness(
        net, sim_csr, true_label, reachMethod='approx', lp_solver='estimate', return_output=False
    )
    rb_crown, _, _, _ = certifyRobustness_CROWN(net, sim_csr, true_label, mode='crown', lp_solver='estimate')
    assert rb_sim in (0, 1, 2)
    assert rb_crown in (0, 1, 2)

    res_psim, _ = certifyProbabilisticRobustness(
        net, psim_csr, true_label, lp_solver='estimate', return_output=False
    )
    res_crown, _ = certifyProbabilisticRobustness_CROWN(
        net, psim_csr, true_label, mode='crown', lp_solver='estimate', return_output=False
    )
    _assert_prob_invariants(res_psim)
    _assert_prob_invariants(res_crown)


class Test(unittest.TestCase):
    def test_probstar_approx_vs_exact_and_mc(self):
        test_probstar_approx_vs_exact_and_mc_soundness()

    def test_cnn_qualitative_vs_quantitative(self):
        test_cnn_qualitative_vs_quantitative_csr_coo_imagestar()

    def test_ssnn_probabilistic_pixels(self):
        test_semantic_segmentation_ssnn_probabilistic_pixels()

    def test_resnet_residual_and_leakyrelu(self):
        test_resnet_residual_layer_and_leakyrelu()

    def test_vit_layernorm_gelu_mhsa(self):
        test_vision_transformer_layernorm_gelu_mhsa_soundness()

    def test_crown_vs_approx_probstar(self):
        test_crown_vs_approx_probstar_2d_and_norm_layer()


if __name__ == '__main__':
    unittest.main(verbosity=2)

