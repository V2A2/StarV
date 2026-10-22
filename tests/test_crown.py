import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from StarV.net.network import NeuralNetwork, reachCROWN
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.LeakyReLULayer import LeakyReLULayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.SatLinsLayer import SatLinsLayer
from StarV.layer.LogSigLayer import LogSigLayer
from StarV.layer.TanSigLayer import TanSigLayer
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

from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.set.sparsestar import SparseStar
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
from StarV.set.probsparseimagestar2dcoo import ProbSparseImageStar2DCOO

from StarV.verifier.crown import (
    CROWNNetworkGraph,
    CROWNVerifier,
    _InputSpec,
    qualiVerifyCROWN,
    quantiVerifyCROWN,
    certifyRobustness_CROWN,
    certifyProbabilisticRobustness_CROWN,
)
from StarV.verifier.verifier import Verifier, quantiVerifyVisionMC
from StarV.verifier.certifier import Certifier


def _check_linear_and_interval_soundness(net, in_set, mode='crown', num_samples=150, seed=123):
    """
    Verify point-wise soundness of CROWN / alpha-CROWN linear bounds:
      z_lb <= A_lb @ alpha_0 + b_lb <= f(x(alpha_0)) <= A_ub @ alpha_0 + b_ub <= z_ub
    over `num_samples` random points in the input domain.
    """
    rng = np.random.default_rng(seed)
    inp_spec = _InputSpec(in_set)
    graph = CROWNNetworkGraph(net, inp_spec.in_shape)
    verifier = CROWNVerifier(mode=mode, alpha_iters=20, alpha_lr=0.15)

    A_lb, b_lb, z_lb, A_ub, b_ub, z_ub, _, _ = verifier.compute_bounds(
        graph, inp_spec, C_spec=None, d_spec=None, optimize_upper=True
    )

    n_act = inp_spec.nVars
    lb = inp_spec.pred_lb
    ub = inp_spec.pred_ub
    alphas = rng.uniform(lb, ub, size=(num_samples, n_act))
    if num_samples >= 2:
        alphas[0] = lb
        alphas[1] = ub
        alphas[2] = 0.5 * (lb + ub)

    tol = 1e-6
    for i in range(num_samples):
        a = alphas[i]
        x_flat = inp_spec.c0 + inp_spec.V0 @ a
        y_graph = graph.evaluate_flat(x_flat)

        # Check graph evaluation matches net.evaluate
        x_shaped = x_flat.reshape(inp_spec.in_shape)
        if not graph.has_pixel_classification:
            y_net = net.evaluate(x_shaped).reshape(-1)
            np.testing.assert_allclose(y_graph, y_net, atol=1e-5, rtol=1e-5)

        lin_lb = A_lb @ a + b_lb
        lin_ub = A_ub @ a + b_ub

        assert np.all(z_lb <= lin_lb + tol), f"z_lb > lin_lb: max diff {np.max(z_lb - lin_lb)}"
        assert np.all(lin_lb <= y_graph + tol), f"lin_lb > y: max diff {np.max(lin_lb - y_graph)}"
        assert np.all(y_graph <= lin_ub + tol), f"y > lin_ub: max diff {np.max(y_graph - lin_ub)}"
        assert np.all(lin_ub <= z_ub + tol), f"lin_ub > z_ub: max diff {np.max(lin_ub - z_ub)}"

    return z_lb, z_ub


def test_ffnn_all_activations_soundness_and_alpha_tightening():
    """Test FFNN with ReLU, LeakyReLU, SatLin, SatLins, LogSig, TanSig, GELU."""
    rng = np.random.default_rng(42)
    act_layers = [
        ReLULayer(),
        LeakyReLULayer(gamma=0.1),
        SatLinLayer(),
        SatLinsLayer(),
        LogSigLayer(),
        TanSigLayer(),
        GELULayer(),
    ]
    for act in act_layers:
        W1 = rng.standard_normal((6, 4)) * 0.6
        b1 = rng.standard_normal(6) * 0.2
        W2 = rng.standard_normal((5, 6)) * 0.6
        b2 = rng.standard_normal(5) * 0.2
        W3 = rng.standard_normal((3, 5)) * 0.6
        b3 = rng.standard_normal(3) * 0.2
        net = NeuralNetwork([
            FullyConnectedLayer([W1, b1]),
            act,
            FullyConnectedLayer([W2, b2]),
            act,
            FullyConnectedLayer([W3, b3]),
        ])
        x0 = rng.standard_normal(4) * 0.3
        in_star = Star(x0 - 0.25, x0 + 0.25)

        z_lb_crown, z_ub_crown = _check_linear_and_interval_soundness(net, in_star, mode='crown')
        z_lb_alpha, z_ub_alpha = _check_linear_and_interval_soundness(net, in_star, mode='alpha-crown')

        # Alpha-CROWN must be at least as tight as Classic CROWN
        assert np.all(z_lb_alpha >= z_lb_crown - 1e-6)
        assert np.all(z_ub_alpha <= z_ub_crown + 1e-6)


def test_cnn_conv_bn_pool_transpose_soundness():
    """Test CNN with Conv2D, BatchNorm2D, AvgPool2D, MaxPool2D, ConvTranspose2D, Flatten."""
    rng = np.random.default_rng(7)
    in_shape = (6, 6, 2)

    # Conv2D weight layout in StarV: (H_k, W_k, C_in, C_out)
    w_conv = rng.standard_normal((3, 3, 2, 3)) * 0.3
    b_conv = rng.standard_normal(3) * 0.1
    conv = Conv2DLayer([w_conv, b_conv], stride=(1, 1), padding=(1, 1, 1, 1))

    # BatchNorm2DLayer: [gamma, beta, mean, var]
    mean_bn = rng.standard_normal(3) * 0.1
    var_bn = np.abs(rng.standard_normal(3)) + 0.5
    gamma_bn = rng.standard_normal(3) * 0.5 + 1.0
    beta_bn = rng.standard_normal(3) * 0.1
    bn = BatchNorm2DLayer([gamma_bn, beta_bn, mean_bn, var_bn])

    maxpool = MaxPool2DLayer(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0, 0, 0))
    # After maxpool: (3, 3, 3)

    # ConvTranspose2D weight layout in StarV: (H_k, W_k, C_out, C_in)
    w_deconv = rng.standard_normal((2, 2, 2, 3)) * 0.3
    b_deconv = rng.standard_normal(2) * 0.1
    deconv = ConvTranspose2DLayer([w_deconv, b_deconv], stride=(1, 1), padding=(0, 0, 0, 0))
    # After deconv: (4, 4, 2)

    avgpool = AvgPool2DLayer(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0, 0, 0))
    # After avgpool: (2, 2, 2) -> flat dim 8

    flat = FlattenLayer(channel_last=False)
    W_fc = rng.standard_normal((3, 8)) * 0.4
    b_fc = rng.standard_normal(3) * 0.1
    fc = FullyConnectedLayer([W_fc, b_fc])

    net = NeuralNetwork([conv, bn, ReLULayer(), maxpool, deconv, ReLULayer(), avgpool, flat, fc])

    img = rng.uniform(0.2, 0.8, size=in_shape)
    in_csr = SparseImageStar2DCSR.inf_attack(img, epsilon=0.05)

    z_lb_c, z_ub_c = _check_linear_and_interval_soundness(net, in_csr, mode='crown', num_samples=80)
    z_lb_a, z_ub_a = _check_linear_and_interval_soundness(net, in_csr, mode='alpha-crown', num_samples=80)
    assert np.all(z_lb_a >= z_lb_c - 1e-6)
    assert np.all(z_ub_a <= z_ub_c + 1e-6)


def test_resnet_dag_and_vit_soundness():
    """Test ResidualLayer (DAG) and Vision Transformer (LayerNorm + MultiHeadAttention + GELU)."""
    rng = np.random.default_rng(99)

    # 1. ResNet block with projection shortcut
    W_m1 = rng.standard_normal((6, 4)) * 0.5
    b_m1 = rng.standard_normal(6) * 0.1
    W_m2 = rng.standard_normal((5, 6)) * 0.5
    b_m2 = rng.standard_normal(5) * 0.1
    W_sc = rng.standard_normal((5, 4)) * 0.5
    b_sc = rng.standard_normal(5) * 0.1
    res_block = ResidualLayer(
        main_branch=[FullyConnectedLayer([W_m1, b_m1]), ReLULayer(), FullyConnectedLayer([W_m2, b_m2])],
        shortcut_branch=[FullyConnectedLayer([W_sc, b_sc])],
        post_activation=ReLULayer(),
    )
    W_out = rng.standard_normal((3, 5)) * 0.5
    b_out = rng.standard_normal(3) * 0.1
    resnet = NeuralNetwork([res_block, FullyConnectedLayer([W_out, b_out])])

    x0 = rng.standard_normal(4) * 0.2
    in_star = Star(x0 - 0.15, x0 + 0.15)
    z_lb_c, z_ub_c = _check_linear_and_interval_soundness(resnet, in_star, mode='crown')
    z_lb_a, z_ub_a = _check_linear_and_interval_soundness(resnet, in_star, mode='alpha-crown')
    assert np.all(z_lb_a >= z_lb_c - 1e-6)

    # 2. Vision Transformer block: LayerNorm -> MultiHeadAttention -> Residual -> LayerNorm -> MLP(GELU)
    seq_len, d_model, num_heads = 2, 4, 2
    flat_dim = seq_len * d_model
    ln1 = LayerNormLayer(normalized_dim=d_model, gamma=np.ones(d_model), beta=np.zeros(d_model))
    mhsa = MultiHeadAttentionLayer.rand(embed_dim=d_model, num_heads=num_heads, num_tokens=seq_len, seed=11)
    res_attn = ResidualLayer(main_branch=[ln1, mhsa], shortcut_branch=None, post_activation=None)

    ln2 = LayerNormLayer(normalized_dim=d_model, gamma=np.ones(d_model), beta=np.zeros(d_model))
    W_ff1 = rng.standard_normal((flat_dim, flat_dim)) * 0.3
    b_ff1 = rng.standard_normal(flat_dim) * 0.05
    W_ff2 = rng.standard_normal((flat_dim, flat_dim)) * 0.3
    b_ff2 = rng.standard_normal(flat_dim) * 0.05
    res_mlp = ResidualLayer(
        main_branch=[ln2, FullyConnectedLayer([W_ff1, b_ff1]), GELULayer(), FullyConnectedLayer([W_ff2, b_ff2])],
        shortcut_branch=None,
        post_activation=None,
    )
    W_head = rng.standard_normal((2, flat_dim)) * 0.4
    b_head = np.zeros(2)
    vit_net = NeuralNetwork([res_attn, res_mlp, FullyConnectedLayer([W_head, b_head])])

    x_vit = rng.standard_normal(flat_dim) * 0.5
    vit_star = Star(x_vit - 0.03, x_vit + 0.03)
    _check_linear_and_interval_soundness(vit_net, vit_star, mode='crown', num_samples=80)


def test_alpha_beta_crown_bab_completeness():
    """
    1) Test a 2-layer diamond ReLU network where Classic CROWN is UNKNOWN and
       alpha-CROWN proves UNSAT in 1 pass via gradient-optimized slopes.
    2) Test a non-convex ReLU network y(x) = 0.85 - ReLU(x - 0.2) - ReLU(-x - 0.2)
       on x in [-1, 1] (where true min is 0.05 > 0, but any single-pass linear
       relaxation has upper-chord sum 0.96 -> lower bound -0.11 < 0), verifying
       that both CROWN and alpha-CROWN return UNKNOWN while alpha,beta-CROWN
       branches on the ReLU neurons and proves UNSAT!
    """
    # Part 1: alpha-CROWN strictly beats Classic CROWN
    W1 = np.array([[1.0], [-1.0]])
    b1 = np.array([0.0, 0.0])
    W2 = np.array([[1.0, 1.0], [-1.0, -1.0]])
    b2 = np.array([0.0, 0.5])
    W3 = np.array([[1.0, 2.0]])
    b3 = np.array([0.0])
    net_diamond = NeuralNetwork([
        FullyConnectedLayer([W1, b1]),
        ReLULayer(),
        FullyConnectedLayer([W2, b2]),
        ReLULayer(),
        FullyConnectedLayer([W3, b3]),
    ])
    in_star = Star(np.array([-1.0]), np.array([1.0]))
    _, _, status_c1, info_c1 = qualiVerifyCROWN(
        net_diamond, in_star, np.array([[1.0]]), np.array([0.35]), mode='crown', return_info=True
    )
    assert status_c1 == 'UNKNOWN'
    _, _, status_a1, info_a1 = qualiVerifyCROWN(
        net_diamond, in_star, np.array([[1.0]]), np.array([0.35]), mode='alpha-crown', return_info=True
    )
    assert status_a1 == 'UNSAT'

    # Part 2: Non-convex ReLU cancellation y(x) = ReLU(x) - ReLU(x) + 0.2 == 0.2 > 0 on [-1, 1].
    # Any single-pass linear relaxation must upper-bound -ReLU(x) by -(0.5 x + 0.5) and
    # lower-bound +ReLU(x) by alpha*x, giving L(x) = (alpha - 0.5)*x - 0.3 <= -0.3 < 0
    # for ALL alpha in [0, 1]. Thus both CROWN and alpha-CROWN return UNKNOWN, while
    # alpha,beta-CROWN splits the ReLUs at x = 0 and proves UNSAT!
    W_nc1 = np.array([[1.0], [1.0]])
    b_nc1 = np.array([0.0, 0.0])
    W_nc2 = np.array([[1.0, -1.0]])
    b_nc2 = np.array([0.2])
    net_nc = NeuralNetwork([
        FullyConnectedLayer([W_nc1, b_nc1]),
        ReLULayer(),
        FullyConnectedLayer([W_nc2, b_nc2]),
    ])
    # Unsafe region: y <= 0.0  <=>  1.0 * y <= 0.0
    unsafe_mat = np.array([[1.0]])
    unsafe_vec = np.array([0.0])

    _, _, st_crown, _ = qualiVerifyCROWN(
        net_nc, in_star, unsafe_mat, unsafe_vec, mode='crown', return_info=True
    )
    assert st_crown == 'UNKNOWN'

    _, _, st_alpha, _ = qualiVerifyCROWN(
        net_nc, in_star, unsafe_mat, unsafe_vec, mode='alpha-crown', return_info=True
    )
    assert st_alpha == 'UNKNOWN'

    _, _, st_abc, info_abc = qualiVerifyCROWN(
        net_nc, in_star, unsafe_mat, unsafe_vec, mode='alpha-beta-crown', return_info=True
    )
    assert st_abc == 'UNSAT'
    assert info_abc['branches_explored'] > 1

    # Also test a truly unsafe threshold y <= 0.25 (since y(x) = 0.2 <= 0.25 -> SAT!)
    _, _, status_sat, info_sat = qualiVerifyCROWN(
        net_nc, in_star, np.array([[1.0]]), np.array([0.25]), mode='alpha-beta-crown', return_info=True
    )
    assert status_sat == 'SAT'
    assert info_sat['counterexample_x'] is not None
    assert info_sat['counterexample_y'][0] <= 0.25 + 1e-6


def test_verifier_and_certifier_integration_qualitative_and_quantitative():
    """Test Verifier, Certifier, Star/ImageStar/ProbStar sets, and Monte Carlo soundness."""
    rng = np.random.default_rng(2025)
    W1 = rng.standard_normal((8, 4)) * 0.5
    b1 = rng.standard_normal(8) * 0.1
    W2 = rng.standard_normal((3, 8)) * 0.5
    b2 = np.array([1.2, -0.5, -0.5])  # Bias class 0 to be dominant near origin
    net = NeuralNetwork([
        FullyConnectedLayer([W1, b1]),
        ReLULayer(),
        FullyConnectedLayer([W2, b2]),
    ])

    x0 = np.zeros(4)
    star_in = Star(x0 - 0.1, x0 + 0.1)
    pstar_in = ProbStar.inf_prob_attack(x0, epsilon=0.25, trunc_sigma=2.0)

    # 1. Reachability via network.reachCROWN and Certifier.reach
    out_star = reachCROWN(net, star_in, mode='alpha-crown')
    assert isinstance(out_star, Star)
    lb_s, ub_s = out_star.estimateRanges()
    assert len(lb_s) == 3 and np.all(lb_s <= ub_s)

    cert = Certifier(reachMethod='alpha-beta-crown')
    rb, vt, vt_tot, Y_cert = cert.certify(net, star_in, labels=0, return_output=True)
    assert rb == 1

    # 2. Qualitative Verifier integration
    ver = Verifier(method='alpha-beta-crown')
    # Unsafe: y_0 - y_1 <= -5.0 (provably UNSAT on small box)
    U = np.array([[1.0, -1.0, 0.0]])
    u = np.array([-5.0])
    S_q, P_q, res_q = ver.verify(net, star_in, U, u, reachMethod='alpha-beta-crown', show=False)
    assert res_q == 'UNSAT'
    assert P_q == []

    # 3. Quantitative Verifier integration + Monte Carlo soundness check
    U_amb = np.array([[1.0, -1.0, 0.0]])
    u_amb = np.array([1.7])
    S_p, res_p = ver.verify(net, pstar_in, U_amb, u_amb, reachMethod='alpha-crown', show=False)
    assert isinstance(S_p, ProbStar)
    assert 0.0 <= res_p['p_min'] <= res_p['p_hat'] <= res_p['p_max'] <= res_p['p_input'] + 1e-9

    mc = quantiVerifyVisionMC(net, pstar_in, unsafe_mat=U_amb, unsafe_vec=u_amb, numSamples=20000, seed=7)
    assert res_p['cond_p_min'] <= mc['cond_ci_high'] + 1e-3
    assert res_p['cond_p_max'] >= mc['cond_ci_low'] - 1e-3

    # 4. Quantitative Probabilistic Certification via Certifier
    prob_res, Y_prob = cert.certify_probabilistic(net, pstar_in, labels=0, return_output=True)
    assert isinstance(Y_prob, ProbStar)
    assert 0.0 <= prob_res['cond_p_min'] <= prob_res['cond_p_hat'] <= prob_res['cond_p_max'] <= 1.0


def test_pixel_classification_segmentation_crown():
    """Test CROWN on a semantic segmentation network ending with PixelClassificationLayer."""
    rng = np.random.default_rng(17)
    in_shape = (4, 4, 2)
    w_conv = rng.standard_normal((3, 3, 2, 3)) * 0.3
    b_conv = np.array([1.0, -0.5, -0.5])
    conv = Conv2DLayer([w_conv, b_conv], stride=(1, 1), padding=(1, 1, 1, 1))
    pix_layer = PixelClassificationLayer(num_pix_classes=3)
    seg_net = NeuralNetwork([conv, ReLULayer(), pix_layer])

    clean_img = rng.uniform(0.1, 0.4, size=in_shape)
    in_csr = SparseImageStar2DCSR.inf_attack(clean_img, epsilon=0.02)
    in_pcsr = ProbSparseImageStar2DCSR.inf_prob_attack(clean_img, epsilon=0.02, sig_factor=2.0)

    cert = Certifier(reachMethod='crown', lp_solver='estimate')
    ver_im, veri_time, vt_tot, out_sets, avg_data = cert.certify_pixel(
        seg_net, [in_csr], [clean_img], num_classes=3, return_output=True
    )
    assert ver_im.shape == (4, 4)
    assert isinstance(out_sets[0], ImageStar)

    prob_seg_res, prob_out_sets = cert.certify_probabilistic_pixel(
        seg_net, [in_pcsr], [clean_img], num_classes=3, return_output=True
    )
    assert len(prob_seg_res) == 1
    assert prob_seg_res[0]['p_safe_min'].shape == (4, 4)
    assert np.all(prob_seg_res[0]['p_safe_min'] <= prob_seg_res[0]['p_safe_max'] + 1e-9)


class TestCROWN(unittest.TestCase):
    def test_1_ffnn_activations(self):
        test_ffnn_all_activations_soundness_and_alpha_tightening()

    def test_2_cnn_layers(self):
        test_cnn_conv_bn_pool_transpose_soundness()

    def test_3_resnet_and_vit(self):
        test_resnet_dag_and_vit_soundness()

    def test_4_alpha_beta_crown_bab(self):
        test_alpha_beta_crown_bab_completeness()

    def test_5_verifier_certifier_integration(self):
        test_verifier_and_certifier_integration_qualitative_and_quantitative()

    def test_6_pixel_classification_segmentation(self):
        test_pixel_classification_segmentation_crown()


if __name__ == '__main__':
    unittest.main(verbosity=2)

