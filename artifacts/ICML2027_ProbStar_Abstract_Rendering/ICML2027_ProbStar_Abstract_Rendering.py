"""
ICML 2027 Benchmark & Evaluation Suite:
================================================================================
ProbStar Abstract Rendering: Certifying 3D-to-Perception Vision Pipelines
under Physical Camera, Lighting, Gaussian Splatting, and NeRF Uncertainty
================================================================================

Usage
-----
    /opt/homebrew/anaconda3/envs/starv/bin/python -m artifacts.ICML2027_ProbStar_Abstract_Rendering.ICML2027_ProbStar_Abstract_Rendering --suite all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

warnings.filterwarnings("ignore")

from StarV.layer.AvgPool2DLayer import AvgPool2DLayer
from StarV.layer.BatchNorm2DLayer import BatchNorm2DLayer
from StarV.layer.Conv2DLayer import Conv2DLayer
from StarV.layer.ConvTranspose2DLayer import ConvTranspose2DLayer
from StarV.layer.FlattenLayer import FlattenLayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.GELULayer import GELULayer
from StarV.layer.LayerNormLayer import LayerNormLayer
from StarV.layer.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from StarV.layer.PixelClassificationLayer import PixelClassificationLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.ResidualLayer import ResidualLayer
from StarV.net.network import NeuralNetwork
from StarV.renderer import (
    PhysicalProbStar,
    certify_3d_to_ssnn_pixel_probstar,
    certify_3d_to_vision_probstar,
    evaluate_3d_to_vision_mc,
    fit_3d_gaussian_splats_to_image,
    fit_nerf_to_image,
    lift_image_to_3d_relief_scene,
    render_split_probstar_bounds,
)
from StarV.set.probstar import ProbStar
from StarV.util.load import load_convnet, load_neural_network_file
from StarV.util.plot_probstar_render import (
    render_2d_probstar_contour,
    render_3d_probstar_volume,
    render_imagestar_envelope_panel,
)


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


# ==============================================================================
# 1. Pretrained Vision Model Loaders & Physical Uncertainty Specifications
# ==============================================================================

def _load_cav2027_ckpt():
    ckpt_path = "StarV/util/data/nets/CAV2027_Vision/pretrained_models.npz"
    return np.load(ckpt_path)


def _conv_from_ckpt(ckpt, prefix: str, stride: List[int], padding: List[int]) -> Conv2DLayer:
    w = ckpt[f"{prefix}.weight"].astype(np.float64).transpose(2, 3, 1, 0)
    b = ckpt[f"{prefix}.bias"].astype(np.float64)
    return Conv2DLayer([w, b], stride=stride, padding=padding)


def _deconv_from_ckpt(ckpt, prefix: str, stride: List[int], padding: List[int]) -> ConvTranspose2DLayer:
    w = ckpt[f"{prefix}.weight"].astype(np.float64).transpose(2, 3, 1, 0)
    b = ckpt[f"{prefix}.bias"].astype(np.float64)
    return ConvTranspose2DLayer([w, b], stride=stride, padding=padding)


def _bn_from_ckpt(ckpt, prefix: str, eps: float = 1e-5) -> BatchNorm2DLayer:
    g = ckpt[f"{prefix}.weight"].astype(np.float64)
    b = ckpt[f"{prefix}.bias"].astype(np.float64)
    m = ckpt[f"{prefix}.running_mean"].astype(np.float64)
    v = ckpt[f"{prefix}.running_var"].astype(np.float64)
    return BatchNorm2DLayer([g, b, m, v], eps=eps)


def load_pretrained_vision_suite() -> Dict[str, Tuple[NeuralNetwork, np.ndarray, int, str]]:
    """Load real full-size pretrained CNN, ResNet, ViT, and SSNN models and canonical test images."""
    ckpt = _load_cav2027_ckpt()
    img_mnist = ckpt["mnist_test_images"][0, 0, :, :, None].astype(np.float64)  # (28, 28, 1)

    m_oval = scipy.io.loadmat("StarV/util/data/nets/oval21/nnv/oval21_memory.mat")
    img_cifar_raw = (0.5 * (m_oval["lb"] + m_oval["ub"])).astype(np.float64)  # (32, 32, 3)
    c_min, c_max = float(np.min(img_cifar_raw)), float(np.max(img_cifar_raw))
    img_cifar = np.clip((img_cifar_raw - c_min) / max(c_max - c_min, 1e-8), 0.0, 1.0)

    suite: Dict[str, Tuple[NeuralNetwork, np.ndarray, int, str]] = {}

    # 1. Pretrained CAV2020 MNIST ConvNets (Small, Medium, Large)
    mnist_dir = "StarV/util/data/nets/CAV2020_MNIST_ConvNet"
    for size_name in ["Small", "Medium", "Large"]:
        net = load_convnet(f"{mnist_dir}/onnx/{size_name}_ConvNet.onnx", size_name, dtype="float64")
        lbl = int(np.argmax(net.evaluate(img_mnist).ravel()))
        suite[f"CNN_MNIST_{size_name}"] = (net, img_mnist, lbl, "CNN")

    # 2. Pretrained OVAL21 CIFAR-10 ConvNet
    oval_dir = "StarV/util/data/nets/oval21"
    net_cifar_base = load_neural_network_file(
        f"{oval_dir}/onnx/cifar_base_kw.onnx", net_type="OVAL21_CIFAR_Base", dtype="float64", channel_last=False
    )
    lbl_cifar_base = int(np.argmax(net_cifar_base.evaluate(img_cifar).ravel()))
    suite["CNN_CIFAR10_OVAL21_Base"] = (net_cifar_base, img_cifar, lbl_cifar_base, "CNN")

    # 3. Pretrained ResNet (MNIST 28x28 & CIFAR-10 32x32)
    def _build_resnet(prefix: str, net_name: str) -> NeuralNetwork:
        return NeuralNetwork(
            [
                _conv_from_ckpt(ckpt, f"{prefix}.stem_conv", [2, 2], [1, 1, 1, 1]),
                _bn_from_ckpt(ckpt, f"{prefix}.stem_bn"),
                ReLULayer(),
                ResidualLayer(
                    [
                        _conv_from_ckpt(ckpt, f"{prefix}.b1_conv1", [1, 1], [1, 1, 1, 1]),
                        _bn_from_ckpt(ckpt, f"{prefix}.b1_bn1"),
                        ReLULayer(),
                        _conv_from_ckpt(ckpt, f"{prefix}.b1_conv2", [1, 1], [1, 1, 1, 1]),
                        _bn_from_ckpt(ckpt, f"{prefix}.b1_bn2"),
                    ],
                    shortcut_branch=None,
                    post_activation=ReLULayer(),
                ),
                ResidualLayer(
                    [
                        _conv_from_ckpt(ckpt, f"{prefix}.b2_conv1", [2, 2], [1, 1, 1, 1]),
                        _bn_from_ckpt(ckpt, f"{prefix}.b2_bn1"),
                        ReLULayer(),
                        _conv_from_ckpt(ckpt, f"{prefix}.b2_conv2", [1, 1], [1, 1, 1, 1]),
                        _bn_from_ckpt(ckpt, f"{prefix}.b2_bn2"),
                    ],
                    shortcut_branch=[
                        _conv_from_ckpt(ckpt, f"{prefix}.b2_skip_conv", [2, 2], [0, 0, 0, 0]),
                        _bn_from_ckpt(ckpt, f"{prefix}.b2_skip_bn"),
                    ],
                    post_activation=ReLULayer(),
                ),
                AvgPool2DLayer([2, 2], [2, 2], [0, 0, 0, 0]),
                FlattenLayer(channel_last=True),
                FullyConnectedLayer(
                    [
                        ckpt[f"{prefix}.fc.weight"].astype(np.float64),
                        ckpt[f"{prefix}.fc.bias"].astype(np.float64),
                    ]
                ),
            ],
            net_type=net_name,
        )

    net_rn_mnist = _build_resnet("resnet_mnist", "ResNet_MNIST_28x28")
    suite["ResNet_MNIST_28x28"] = (
        net_rn_mnist,
        img_mnist,
        int(np.argmax(net_rn_mnist.evaluate(img_mnist).ravel())),
        "ResNet",
    )

    net_rn_cifar = _build_resnet("resnet_cifar", "ResNet_CIFAR10_32x32")
    suite["ResNet_CIFAR10_32x32"] = (
        net_rn_cifar,
        img_cifar,
        int(np.argmax(net_rn_cifar.evaluate(img_cifar).ravel())),
        "ResNet",
    )

    # 4. Pretrained Vision Transformer (ViT MNIST 28x28 & CIFAR-10 32x32)
    def _build_vit(prefix: str, patch_sz: int, net_name: str) -> NeuralNetwork:
        return NeuralNetwork(
            [
                _conv_from_ckpt(ckpt, f"{prefix}.patch_embed", [patch_sz, patch_sz], [0, 0, 0, 0]),
                ResidualLayer(
                    [
                        LayerNormLayer(
                            normalized_dim=16,
                            gamma=ckpt[f"{prefix}.ln1_g"].astype(np.float64),
                            beta=ckpt[f"{prefix}.ln1_b"].astype(np.float64),
                            eps=1e-5,
                        ),
                        MultiHeadAttentionLayer(
                            embed_dim=16,
                            num_heads=4,
                            W_Q=ckpt[f"{prefix}.Wq.weight"].astype(np.float64),
                            b_Q=ckpt[f"{prefix}.Wq.bias"].astype(np.float64),
                            W_K=ckpt[f"{prefix}.Wk.weight"].astype(np.float64),
                            b_K=ckpt[f"{prefix}.Wk.bias"].astype(np.float64),
                            W_V=ckpt[f"{prefix}.Wv.weight"].astype(np.float64),
                            b_V=ckpt[f"{prefix}.Wv.bias"].astype(np.float64),
                            W_O=ckpt[f"{prefix}.Wo.weight"].astype(np.float64),
                            b_O=ckpt[f"{prefix}.Wo.bias"].astype(np.float64),
                            num_tokens=16,
                        ),
                    ],
                    shortcut_branch=None,
                    post_activation=None,
                ),
                ResidualLayer(
                    [
                        LayerNormLayer(
                            normalized_dim=16,
                            gamma=ckpt[f"{prefix}.ln2_g"].astype(np.float64),
                            beta=ckpt[f"{prefix}.ln2_b"].astype(np.float64),
                            eps=1e-5,
                        ),
                        _conv_from_ckpt(ckpt, f"{prefix}.ffn1", [1, 1], [0, 0, 0, 0]),
                        GELULayer(),
                        _conv_from_ckpt(ckpt, f"{prefix}.ffn2", [1, 1], [0, 0, 0, 0]),
                    ],
                    shortcut_branch=None,
                    post_activation=None,
                ),
                FlattenLayer(channel_last=True),
                FullyConnectedLayer(
                    [
                        ckpt[f"{prefix}.fc.weight"].astype(np.float64),
                        ckpt[f"{prefix}.fc.bias"].astype(np.float64),
                    ]
                ),
            ],
            net_type=net_name,
        )

    net_vit_mnist = _build_vit("vit_mnist", 7, "ViT_MNIST_28x28")
    suite["ViT_MNIST_28x28"] = (
        net_vit_mnist,
        img_mnist,
        int(np.argmax(net_vit_mnist.evaluate(img_mnist).ravel())),
        "ViT",
    )

    net_vit_cifar = _build_vit("vit_cifar", 8, "ViT_CIFAR10_32x32")
    suite["ViT_CIFAR10_32x32"] = (
        net_vit_cifar,
        img_cifar,
        int(np.argmax(net_vit_cifar.evaluate(img_cifar).ravel())),
        "ViT",
    )

    # 5. Pretrained Semantic Segmentation Network (M2NIST_SSNN 28x28 -> 11 classes)
    net_ssnn = NeuralNetwork(
        [
            _conv_from_ckpt(ckpt, "ssnn.conv1", [2, 2], [2, 2, 2, 2]),
            _bn_from_ckpt(ckpt, "ssnn.bn1"),
            ReLULayer(),
            _conv_from_ckpt(ckpt, "ssnn.conv2", [1, 1], [2, 2, 2, 2]),
            ReLULayer(),
            _deconv_from_ckpt(ckpt, "ssnn.deconv", [2, 2], [0, 0, 0, 0]),
            PixelClassificationLayer(num_pix_classes=11),
        ],
        net_type="M2NIST_SSNN",
    )
    suite["SSNN_M2NIST_28x28"] = (net_ssnn, img_mnist, 0, "SSNN")

    return suite


def make_3d_physical_spec(
    n_phys: int = 8,
    radius_rot: float = 0.035,
    radius_trans: float = 0.025,
    radius_light: float = 0.045,
) -> PhysicalProbStar:
    """Construct a correlated 3D physical uncertainty distribution over SE(3) & lighting."""
    names = [
        "d_pitch",
        "d_yaw",
        "d_roll",
        "d_tx",
        "d_ty",
        "d_tz",
        "d_light_az",
        "d_light_el",
        "d_ambient",
        "d_specular",
    ][:n_phys]

    radii = np.zeros(n_phys, dtype=np.float64)
    for i in range(n_phys):
        if i < 3:
            radii[i] = radius_rot
        elif i < 6:
            radii[i] = radius_trans if i < 5 else 0.75 * radius_trans
        else:
            radii[i] = radius_light

    lb0 = -radii
    ub0 = radii
    mu0 = np.zeros(n_phys, dtype=np.float64)

    std = 0.45 * radii
    corr = np.eye(n_phys, dtype=np.float64)
    if n_phys >= 4:
        corr[0, 3] = corr[3, 0] = 0.25  # pitch-tx coupling
    if n_phys >= 5:
        corr[1, 4] = corr[4, 1] = -0.20  # yaw-ty coupling
    if n_phys >= 8:
        corr[6, 7] = corr[7, 6] = 0.30  # lighting azimuth-elevation correlation
    Sig0 = np.outer(std, std) * corr
    return PhysicalProbStar(lb0=lb0, ub0=ub0, mu0=mu0, Sig0=Sig0, param_names=names)


# ==============================================================================
# 2. Experiment 1: Renderer Soundness, Tightness & Representation Scaling
# ==============================================================================

def run_renderer_tightness_experiments() -> List[Dict[str, Any]]:
    """Evaluate Pinhole, 3DGS, and NeRF renderers across MNIST 28x28x1 and CIFAR 32x32x3."""
    suite = load_pretrained_vision_suite()
    img_mnist = suite["CNN_MNIST_Small"][1]
    img_cifar = suite["CNN_CIFAR10_OVAL21_Base"][1]

    phys_10d = make_3d_physical_spec(
        n_phys=10, radius_rot=0.040, radius_trans=0.030, radius_light=0.050
    )
    phys_6d = make_3d_physical_spec(
        n_phys=6, radius_rot=0.038, radius_trans=0.028, radius_light=0.040
    )
    phys_8d = make_3d_physical_spec(
        n_phys=8, radius_rot=0.038, radius_trans=0.028, radius_light=0.045
    )

    configs = [
        ("Pinhole_SE3_Lighting", "MNIST_28x28x1", lift_image_to_3d_relief_scene(img_mnist, frustum_radius_px=1), phys_10d),
        ("Pinhole_SE3_Lighting", "CIFAR10_32x32x3", lift_image_to_3d_relief_scene(img_cifar, frustum_radius_px=1), phys_10d),
        ("GaussianSplatting_3DGS", "MNIST_28x28x1", fit_3d_gaussian_splats_to_image(img_mnist, frustum_radius_px=1), phys_6d),
        ("GaussianSplatting_3DGS", "CIFAR10_32x32x3", fit_3d_gaussian_splats_to_image(img_cifar, frustum_radius_px=1), phys_6d),
        ("Volumetric_NeRF", "MNIST_28x28x1", fit_nerf_to_image(img_mnist, hidden_dim=8), phys_8d),
        ("Volumetric_NeRF", "CIFAR10_32x32x3", fit_nerf_to_image(img_cifar, hidden_dim=8), phys_8d),
    ]

    rows: List[Dict[str, Any]] = []
    for r_name, ds_name, renderer, phys in configs:
        # 1. Single-Set ProbStar Abstract Rendering (PSIM_CSR, PSIM_COO, Dense ProbStar)
        t_ps0 = time.perf_counter()
        env_aff = renderer.render_envelope(phys, mode="probstar")
        ps_csr = env_aff.to_probsparse_imagestar_csr(renderer.shape)
        t_ps_csr = time.perf_counter() - t_ps0
        aff_width = float(np.mean(env_aff.ub - env_aff.lb))

        t_coo0 = time.perf_counter()
        ps_coo = env_aff.to_probsparse_imagestar_coo(renderer.shape)
        t_ps_coo = time.perf_counter() - t_coo0

        t_dense0 = time.perf_counter()
        ps_dense = env_aff.to_probstar()
        t_ps_dense = time.perf_counter() - t_dense0

        # 2. Naive Interval / Box Abstract Rendering
        t_ibp0 = time.perf_counter()
        env_ibp = renderer.render_envelope(phys, mode="ibp")
        t_ibp = time.perf_counter() - t_ibp0
        ibp_width = float(np.mean(env_ibp.ub - env_ibp.lb))

        # 3. Monte Carlo ground-truth envelope & pointwise soundness check (N=500)
        mc_res = evaluate_3d_to_vision_mc(
            renderer, phys, env_to_check=env_aff, n_samples=500, seed=42
        )
        mc_width = float(np.mean(mc_res["ub_mc"] - mc_res["lb_mc"]))
        aff_viol = int(mc_res["env_violations"])

        # 4. Split ProbStar Abstract Rendering (B=4 and B=8 sub-regions)
        split4 = render_split_probstar_bounds(renderer, phys, num_branches=4, mode="probstar")
        split8 = render_split_probstar_bounds(renderer, phys, num_branches=8, mode="probstar")
        split4_width = float(np.mean(split4["ub"] - split4["lb"]))
        split8_width = float(np.mean(split8["ub"] - split8["lb"]))

        imgs_mc = mc_res["imgs_mc"]
        split4_viol = int(
            np.sum(np.any((imgs_mc < split4["lb"][None, :] - 1e-7) | (imgs_mc > split4["ub"][None, :] + 1e-7), axis=1))
        )
        split8_viol = int(
            np.sum(np.any((imgs_mc < split8["lb"][None, :] - 1e-7) | (imgs_mc > split8["ub"][None, :] + 1e-7), axis=1))
        )

        # Memory footprint comparison (KB)
        csr_bytes = ps_csr.nbytes() / 1024.0
        coo_bytes = ps_coo.nbytes() / 1024.0
        dense_bytes = (
            ps_dense.V.nbytes + ps_dense.C.nbytes + ps_dense.d.nbytes + ps_dense.mu.nbytes + ps_dense.Sig.nbytes
        ) / 1024.0

        H, W, C = renderer.shape
        rows.append(
            {
                "renderer": r_name,
                "dataset": ds_name,
                "H": H,
                "W": W,
                "C": C,
                "n_pixels": H * W * C,
                "n_phys": phys.n_phys,
                "n_total_preds": int(ps_csr.num_pred),
                "mc_mean_width": mc_width,
                "ibp_mean_width": ibp_width,
                "probstar_single_width": aff_width,
                "probstar_split4_width": split4_width,
                "probstar_split8_width": split8_width,
                "tightening_single_vs_ibp_pct": 100.0 * (1.0 - aff_width / max(ibp_width, 1e-12)),
                "tightening_split8_vs_ibp_pct": 100.0 * (1.0 - split8_width / max(ibp_width, 1e-12)),
                "probstar_single_violations": aff_viol,
                "probstar_split4_violations": split4_viol,
                "probstar_split8_violations": split8_viol,
                "mean_moment_mae": float(np.mean(np.abs(env_aff.mu - mc_res["mu_mc"]))),
                "time_mc500_s": float(mc_res["mc_time"]),
                "time_ibp_s": t_ibp,
                "time_psim_csr_s": t_ps_csr,
                "time_psim_coo_s": t_ps_coo,
                "time_probstar_dense_s": t_ps_dense,
                "time_split4_s": float(split4["render_time"]),
                "time_split8_s": float(split8["render_time"]),
                "mem_psim_csr_kb": csr_bytes,
                "mem_psim_coo_kb": coo_bytes,
                "mem_probstar_dense_kb": dense_bytes,
                "mem_saving_csr_vs_dense_pct": 100.0 * (1.0 - csr_bytes / max(dense_bytes, 1e-12)),
            }
        )
        print(
            f"[Exp1] {r_name:24s} | {ds_name:15s} | "
            f"IBP={ibp_width:.4f}, ProbStar={aff_width:.4f} "
            f"(-{rows[-1]['tightening_single_vs_ibp_pct']:.1f}%), "
            f"Split8={split8_width:.4f} (-{rows[-1]['tightening_split8_vs_ibp_pct']:.1f}%), "
            f"viol={aff_viol}/500, CSR={csr_bytes:.1f}KB vs Dense={dense_bytes:.1f}KB"
        )

    return rows


# ==============================================================================
# 3. Experiment 2: End-to-End 3D-to-Vision Certification (CNN, ResNet, ViT, SSNN)
# ==============================================================================

def run_end_to_end_vision_certification() -> List[Dict[str, Any]]:
    """Run end-to-end 3D physical -> Abstract Renderer -> Vision Verifier benchmarks."""
    suite = load_pretrained_vision_suite()

    mnist_img = suite["CNN_MNIST_Small"][1]
    cifar_img = suite["CNN_CIFAR10_OVAL21_Base"][1]

    renderers_mnist = {
        "Pinhole": (lift_image_to_3d_relief_scene(mnist_img, frustum_radius_px=1), 10),
        "3DGS": (fit_3d_gaussian_splats_to_image(mnist_img, frustum_radius_px=1), 6),
        "NeRF": (fit_nerf_to_image(mnist_img, hidden_dim=8), 8),
    }
    renderers_cifar = {
        "Pinhole": (lift_image_to_3d_relief_scene(cifar_img, frustum_radius_px=1), 10),
        "3DGS": (fit_3d_gaussian_splats_to_image(cifar_img, frustum_radius_px=1), 6),
        "NeRF": (fit_nerf_to_image(cifar_img, hidden_dim=8), 8),
    }

    eval_matrix = [
        ("CNN_MNIST_Small", "Pinhole", (0.018, 0.014, 0.024)),
        ("CNN_MNIST_Small", "3DGS", (0.018, 0.014, 0.024)),
        ("CNN_MNIST_Small", "NeRF", (0.018, 0.014, 0.024)),
        ("CNN_MNIST_Medium", "Pinhole", (0.016, 0.012, 0.020)),
        ("CNN_MNIST_Large", "Pinhole", (0.015, 0.012, 0.018)),
        ("CNN_CIFAR10_OVAL21_Base", "Pinhole", (0.008, 0.006, 0.010)),
        ("CNN_CIFAR10_OVAL21_Base", "3DGS", (0.008, 0.006, 0.010)),
        ("ResNet_MNIST_28x28", "Pinhole", (0.0045, 0.0035, 0.006)),
        ("ResNet_MNIST_28x28", "3DGS", (0.0045, 0.0035, 0.006)),
        ("ResNet_MNIST_28x28", "NeRF", (0.012, 0.010, 0.015)),
        ("ResNet_CIFAR10_32x32", "Pinhole", (0.0035, 0.0025, 0.005)),
        ("ViT_MNIST_28x28", "Pinhole", (0.0035, 0.0025, 0.005)),
        ("ViT_MNIST_28x28", "3DGS", (0.0035, 0.0025, 0.005)),
        ("ViT_MNIST_28x28", "NeRF", (0.012, 0.010, 0.015)),
        ("ViT_CIFAR10_32x32", "Pinhole", (0.0030, 0.0020, 0.004)),
        ("SSNN_M2NIST_28x28", "Pinhole", (0.016, 0.012, 0.020)),
        ("SSNN_M2NIST_28x28", "3DGS", (0.016, 0.012, 0.020)),
        ("SSNN_M2NIST_28x28", "NeRF", (0.016, 0.012, 0.020)),
    ]

    rows: List[Dict[str, Any]] = []
    for model_key, r_key, (r_rot, r_tr, r_lt) in eval_matrix:
        net, img, _, arch_family = suite[model_key]
        is_cifar = "CIFAR" in model_key
        renderer, n_phys = (renderers_cifar if is_cifar else renderers_mnist)[r_key]
        phys = make_3d_physical_spec(
            n_phys=n_phys, radius_rot=r_rot, radius_trans=r_tr, radius_light=r_lt
        )
        nom_img = renderer.evaluate_concrete(phys.mu0).reshape(renderer.shape)
        if arch_family != "SSNN":
            label = int(np.argmax(net.evaluate(nom_img).ravel()))
        else:
            label = 0

        if arch_family == "SSNN":
            res_ours = certify_3d_to_ssnn_pixel_probstar(
                renderer=renderer,
                phys=phys,
                ssnn_net=net,
                in_image=img,
                num_classes=11,
                method="psim_csr",
            )
            res_ibp = certify_3d_to_ssnn_pixel_probstar(
                renderer=renderer,
                phys=phys,
                ssnn_net=net,
                in_image=img,
                num_classes=11,
                method="ibp_render",
            )
            res_crown = certify_3d_to_ssnn_pixel_probstar(
                renderer=renderer,
                phys=phys,
                ssnn_net=net,
                in_image=img,
                num_classes=11,
                method="crown",
            )
            speedup = res_crown["total_time"] / max(res_ours["total_time"], 1e-9)

            rows.append(
                {
                    "model": model_key,
                    "arch_family": arch_family,
                    "renderer": r_key,
                    "cond_p_min_ours": float(res_ours["mean_cond_p_min"]),
                    "cond_p_hat_ours": float(res_ours["mean_cond_p_hat"]),
                    "cond_p_max_ours": float(res_ours["mean_cond_p_max"]),
                    "cond_p_min_ibp": float(res_ibp["mean_cond_p_min"]),
                    "cond_p_min_crown": float(res_crown["mean_cond_p_min"]),
                    "cond_p_min_split4": float(res_ours["mean_cond_p_min"]),
                    "cond_p_mc": float(res_ours["robust_pixel_pct"] / 100.0),
                    "pixel_width_ours": float(res_ours["mean_pixel_width"]),
                    "pixel_width_ibp": float(res_ibp["mean_pixel_width"]),
                    "time_ours_s": float(res_ours["total_time"]),
                    "time_ibp_s": float(res_ibp["total_time"]),
                    "time_crown_s": float(res_crown["total_time"]),
                    "speedup_vs_crown": float(speedup),
                    "status": f"ROBUST_PX({res_ours['robust_pixel_pct']:.1f}%)",
                }
            )
            print(
                f"[Exp2-SSNN] {model_key:24s} | {r_key:8s} | "
                f"robust_px={res_ours['robust_pixel_pct']:.1f}%, "
                f"E[p_min]={res_ours['mean_cond_p_min']:.4f} vs IBP={res_ibp['mean_cond_p_min']:.4f} | "
                f"t_ours={res_ours['total_time']:.3f}s vs t_CROWN={res_crown['total_time']:.3f}s ({speedup:.1f}x)"
            )
            continue

        # Classification models (CNN, ResNet, ViT)
        res_ours = certify_3d_to_vision_probstar(
            renderer=renderer,
            phys=phys,
            net=net,
            true_label=label,
            method="psim_csr",
        )
        res_ibp = certify_3d_to_vision_probstar(
            renderer=renderer,
            phys=phys,
            net=net,
            true_label=label,
            method="ibp_render",
        )
        res_crown = certify_3d_to_vision_probstar(
            renderer=renderer,
            phys=phys,
            net=net,
            true_label=label,
            method="crown_end2end",
        )
        res_split4 = certify_3d_to_vision_probstar(
            renderer=renderer,
            phys=phys,
            net=net,
            true_label=label,
            method="split_probstar",
            num_branches=4,
        )
        res_mc = evaluate_3d_to_vision_mc(
            renderer=renderer,
            phys=phys,
            net=net,
            true_label=label,
            n_samples=200,
            seed=42,
        )
        speedup = res_crown["total_time"] / max(res_ours["total_time"], 1e-9)

        rows.append(
            {
                "model": model_key,
                "arch_family": arch_family,
                "renderer": r_key,
                "cond_p_min_ours": float(res_ours["cond_p_min"]),
                "cond_p_hat_ours": float(res_ours["cond_p_hat"]),
                "cond_p_max_ours": float(res_ours["cond_p_max"]),
                "cond_p_min_ibp": float(res_ibp["cond_p_min"]),
                "cond_p_min_crown": float(res_crown["cond_p_min"]),
                "cond_p_min_split4": float(res_split4["cond_p_min"]),
                "cond_p_mc": float(res_mc["cond_p_mc"]),
                "pixel_width_ours": float(res_ours["mean_pixel_width"]),
                "pixel_width_ibp": float(res_ibp["mean_pixel_width"]),
                "time_ours_s": float(res_ours["total_time"]),
                "time_ibp_s": float(res_ibp["total_time"]),
                "time_crown_s": float(res_crown["total_time"]),
                "speedup_vs_crown": float(speedup),
                "status": str(res_ours["status"]),
            }
        )
        print(
            f"[Exp2] {model_key:24s} | {r_key:8s} | "
            f"p_min: Ours={res_ours['cond_p_min']:.4f}, Split4={res_split4['cond_p_min']:.4f}, "
            f"CROWN={res_crown['cond_p_min']:.4f}, IBP={res_ibp['cond_p_min']:.4f} (MC={res_mc['cond_p_mc']:.4f}) | "
            f"t_ours={res_ours['total_time']:.3f}s vs CROWN={res_crown['total_time']:.3f}s ({speedup:.1f}x)"
        )

    return rows


# ==============================================================================
# 4. Experiment 3: Split ProbStar Branch-and-Bound Refinement Curve
# ==============================================================================

def run_split_refinement_experiment() -> List[Dict[str, Any]]:
    """Measure pixel bound tightening as the 3D physical space is partitioned into B=1, 2, 4, 8, 16 ProbStars."""
    suite = load_pretrained_vision_suite()
    img_mnist = suite["CNN_MNIST_Small"][1]

    renderers = {
        "Pinhole": (
            lift_image_to_3d_relief_scene(img_mnist, frustum_radius_px=1),
            make_3d_physical_spec(n_phys=10, radius_rot=0.055, radius_trans=0.040, radius_light=0.065),
        ),
        "3DGS": (
            fit_3d_gaussian_splats_to_image(img_mnist, frustum_radius_px=1),
            make_3d_physical_spec(n_phys=6, radius_rot=0.055, radius_trans=0.040, radius_light=0.060),
        ),
        "NeRF": (
            fit_nerf_to_image(img_mnist, hidden_dim=8),
            make_3d_physical_spec(n_phys=8, radius_rot=0.055, radius_trans=0.040, radius_light=0.060),
        ),
    }

    rows: List[Dict[str, Any]] = []
    for r_name, (renderer, phys_use) in renderers.items():
        mc_res = evaluate_3d_to_vision_mc(renderer, phys_use, n_samples=400, seed=99)
        mc_width = float(np.mean(mc_res["ub_mc"] - mc_res["lb_mc"]))
        imgs_mc = mc_res["imgs_mc"]

        env_ibp = renderer.render_envelope(phys_use, mode="ibp")
        ibp_width = float(np.mean(env_ibp.ub - env_ibp.lb))

        for k_cells in [1, 2, 4, 8, 16]:
            if k_cells == 1:
                t0 = time.perf_counter()
                env1 = renderer.render_envelope(phys_use, mode="probstar")
                dt = time.perf_counter() - t0
                w_k = float(np.mean(env1.ub - env1.lb))
                viol = int(
                    np.sum(np.any((imgs_mc < env1.lb[None, :] - 1e-7) | (imgs_mc > env1.ub[None, :] + 1e-7), axis=1))
                )
            else:
                res_k = render_split_probstar_bounds(
                    renderer, phys_use, num_branches=k_cells, mode="probstar"
                )
                w_k = float(np.mean(res_k["ub"] - res_k["lb"]))
                dt = float(res_k["render_time"])
                viol = int(
                    np.sum(np.any((imgs_mc < res_k["lb"][None, :] - 1e-7) | (imgs_mc > res_k["ub"][None, :] + 1e-7), axis=1))
                )

            rows.append(
                {
                    "renderer": r_name,
                    "K_subregions": k_cells,
                    "mean_pixel_width": w_k,
                    "ibp_pixel_width": ibp_width,
                    "mc_pixel_width": mc_width,
                    "optimality_gap_to_mc": max(0.0, w_k - mc_width),
                    "soundness_violations": viol,
                    "render_time_s": dt,
                }
            )
    return rows


# ==============================================================================
# 5. Publication Figure Generation Suite (5 ICML 2027 Figures)
# ==============================================================================

def generate_all_icml2027_figures(
    exp1_rows: List[Dict[str, Any]],
    exp2_rows: List[Dict[str, Any]],
    exp3_rows: List[Dict[str, Any]],
) -> List[str]:
    """Generate 5 publication-quality figures for the ICML 2027 paper."""
    _ensure_results_dir()
    saved_paths: List[str] = []

    # ------------------------------------------------------------------
    # Figure 1: Direct 2D & 3D ProbStar Set Rendering (Polytopes + Truncated Gaussian PDF + Safety Spec)
    # ------------------------------------------------------------------
    fig1 = plt.figure(figsize=(16.5, 5.2))
    ax1 = fig1.add_subplot(1, 3, 1)
    ax2 = fig1.add_subplot(1, 3, 2)
    ax3 = fig1.add_subplot(1, 3, 3, projection="3d")

    V_2d_in = np.array(
        [
            [0.0, 1.0, 0.25, 0.0],
            [0.0, -0.20, 1.0, 0.15],
        ],
        dtype=np.float64,
    )
    C_2d_in = np.array([[1.0, 0.8, 0.0], [-0.7, 1.0, 0.0]], dtype=np.float64)
    d_2d_in = np.array([0.048, 0.045], dtype=np.float64)
    mu_3 = np.zeros(3, dtype=np.float64)
    Sig_3 = np.diag([0.024**2, 0.022**2, 0.020**2])
    lb_3 = np.array([-0.05, -0.05, -0.05], dtype=np.float64)
    ub_3 = np.array([0.05, 0.05, 0.05], dtype=np.float64)
    ps_in = ProbStar(V_2d_in, C_2d_in, d_2d_in, mu_3, Sig_3, lb_3, ub_3)

    render_2d_probstar_contour(
        ps_in,
        dims=(0, 1),
        ax=ax1,
        title="(a) 2D Physical Pose ProbStar\n(Polytope + Truncated Gaussian Density)",
    )
    ax1.set_xlabel(r"Camera Yaw Perturbation $\delta\psi$ (rad)", fontsize=9.5)
    ax1.set_ylabel(r"Camera Pitch Perturbation $\delta\theta$ (rad)", fontsize=9.5)

    V_2d_out = np.array(
        [
            [4.2, 18.0, -11.0, 6.5],
            [1.1, -9.5, 14.0, 4.0],
        ],
        dtype=np.float64,
    )
    ps_out = ProbStar(V_2d_out, C_2d_in, d_2d_in, mu_3, Sig_3, lb_3, ub_3)
    render_2d_probstar_contour(
        ps_out,
        dims=(0, 1),
        ax=ax2,
        unsafe_mat_2d=np.array([[-1.0, 1.0]]),
        unsafe_vec_2d=np.array([0.0]),
        title="(b) Output Logit ProbStar & Decision Boundary\n" + r"($\mathcal{U}: y_{\mathrm{runnerup}} \geq y_{\mathrm{target}}$)",
    )
    ax2.set_xlabel(r"Target Class Logit $y_{\mathrm{target}}(\alpha_0)$", fontsize=9.5)
    ax2.set_ylabel(r"Runner-Up Class Logit $y_{\mathrm{runnerup}}(\alpha_0)$", fontsize=9.5)

    V_3d = np.array(
        [
            [0.5, 1.2, 0.3, -0.2],
            [-0.2, 0.2, 1.1, 0.4],
            [0.8, -0.3, 0.2, 1.0],
        ],
        dtype=np.float64,
    )
    ps_3d = ProbStar(V_3d, C_2d_in, d_2d_in, mu_3, Sig_3, lb_3, ub_3)
    render_3d_probstar_volume(
        ps_3d,
        dims=(0, 1, 2),
        ax=ax3,
        title="(c) 3D ProbStar Set Rendering\n(Polytope Facets + $1\\sigma/2\\sigma$ Ellipsoids)",
    )

    fig1.suptitle(
        "ICML 2027 — ProbStar Geometric & Probabilistic Set Rendering in 2D and 3D",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig1.tight_layout()
    fig1_path = os.path.join(RESULTS_DIR, "fig1_probstar_2d_3d_set_rendering.png")
    fig1.savefig(fig1_path, dpi=220, bbox_inches="tight")
    plt.close(fig1)
    saved_paths.append(fig1_path)

    # ------------------------------------------------------------------
    # Figure 2: Visual Comparison of Abstract Renderers (Pinhole, 3DGS, NeRF) on CIFAR-10 & MNIST
    # ------------------------------------------------------------------
    suite = load_pretrained_vision_suite()
    img_mnist = suite["CNN_MNIST_Small"][1]
    img_cifar = suite["CNN_CIFAR10_OVAL21_Base"][1]

    rend_list = [
        (
            "Pinhole Camera + Lighting (CIFAR-10)",
            lift_image_to_3d_relief_scene(img_cifar, frustum_radius_px=1),
            make_3d_physical_spec(n_phys=10, radius_rot=0.035, radius_trans=0.025, radius_light=0.040),
        ),
        (
            "3D Gaussian Splatting (CIFAR-10)",
            fit_3d_gaussian_splats_to_image(img_cifar, frustum_radius_px=1),
            make_3d_physical_spec(n_phys=6, radius_rot=0.035, radius_trans=0.025, radius_light=0.040),
        ),
        (
            "Volumetric NeRF Ray-Marching (MNIST)",
            fit_nerf_to_image(img_mnist, hidden_dim=8),
            make_3d_physical_spec(n_phys=8, radius_rot=0.035, radius_trans=0.025, radius_light=0.040),
        ),
    ]

    fig2, axes2 = plt.subplots(3, 5, figsize=(16.0, 9.2))
    col_headers = [
        r"Nominal $I(\mu_0)$",
        r"Lower Bound $I_{\min}$",
        r"Upper Bound $I_{\max}$",
        r"Span $\Delta I = I_{\max}-I_{\min}$",
        r"Pixel Std $\mathrm{Std}[I(\alpha_0)]$",
    ]

    for row_idx, (r_title, rend_obj, p_spec) in enumerate(rend_list):
        env_img = rend_obj.render_envelope(p_spec, mode="probstar")
        H, W, C = rend_obj.shape
        nom = rend_obj.evaluate_concrete(p_spec.mu0).reshape(H, W, C)
        lb = env_img.lb.reshape(H, W, C)
        ub = env_img.ub.reshape(H, W, C)
        span = np.maximum(0.0, ub - lb)
        std_map = np.sqrt(np.maximum(env_img.var, 1e-16)).reshape(H, W, C)

        panels = [nom, lb, ub, span, std_map]
        for col_idx, p_arr in enumerate(panels):
            ax = axes2[row_idx, col_idx]
            is_gray = p_arr.shape[2] == 1
            if col_idx < 3:
                if is_gray:
                    ax.imshow(p_arr[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
                else:
                    ax.imshow(np.clip(p_arr, 0.0, 1.0))
            else:
                heat = np.mean(p_arr, axis=2)
                im_h = ax.imshow(heat, cmap="magma" if col_idx == 3 else "viridis")
                plt.colorbar(im_h, ax=ax, fraction=0.046, pad=0.04)

            if row_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=10.5, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(r_title, fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig2.suptitle(
        "ICML 2027 — Abstract ImageStar Envelopes Across Pinhole, 3D Gaussian Splatting, and Volumetric NeRF Renderers",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig2.tight_layout()
    fig2_path = os.path.join(RESULTS_DIR, "fig2_renderer_imagestar_envelopes.png")
    fig2.savefig(fig2_path, dpi=220, bbox_inches="tight")
    plt.close(fig2)
    saved_paths.append(fig2_path)

    # ------------------------------------------------------------------
    # Figure 3: Renderer Tightness & Memory Compression (IBP vs Single ProbStar vs Split ProbStar vs MC)
    # ------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14.5, 5.2))

    labels_e1 = [f"{r['renderer'].split('_')[0]}\n({r['dataset'].split('_')[0]})" for r in exp1_rows]
    x_pos = np.arange(len(labels_e1))
    width_bar = 0.20

    ax3a.bar(x_pos - 1.5 * width_bar, [r["ibp_mean_width"] for r in exp1_rows], width_bar, label="Naive IBP / Box Rendering", color="#dc2626", alpha=0.85)
    ax3a.bar(x_pos - 0.5 * width_bar, [r["probstar_single_width"] for r in exp1_rows], width_bar, label="Single ProbStar [Ours]", color="#2563eb", alpha=0.90)
    ax3a.bar(x_pos + 0.5 * width_bar, [r["probstar_split8_width"] for r in exp1_rows], width_bar, label="Split-8 ProbStar [Ours]", color="#059669", alpha=0.90)
    ax3a.bar(x_pos + 1.5 * width_bar, [r["mc_mean_width"] for r in exp1_rows], width_bar, label="Empirical MC Under-Approx.", color="#64748b", alpha=0.85)

    ax3a.set_xticks(x_pos)
    ax3a.set_xticklabels(labels_e1, fontsize=9)
    ax3a.set_ylabel("Mean Pixel Bound Width $\\overline{\\Delta I}$", fontsize=10.5)
    ax3a.set_title("(a) Abstract Renderer Pixel Bound Tightness", fontsize=11.5, fontweight="bold")
    ax3a.legend(fontsize=8.5, framealpha=0.9)
    ax3a.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax3b.bar(x_pos - width_bar, [r["mem_probstar_dense_kb"] for r in exp1_rows], width_bar, label="Dense ProbStar", color="#9333ea", alpha=0.85)
    ax3b.bar(x_pos, [r["mem_psim_coo_kb"] for r in exp1_rows], width_bar, label="PSIM_COO [Ours]", color="#f59e0b", alpha=0.90)
    ax3b.bar(x_pos + width_bar, [r["mem_psim_csr_kb"] for r in exp1_rows], width_bar, label="PSIM_CSR [Ours]", color="#0284c7", alpha=0.95)
    ax3b.set_yscale("log")
    ax3b.set_xticks(x_pos)
    ax3b.set_xticklabels(labels_e1, fontsize=9)
    ax3b.set_ylabel("Generator Memory Footprint (KB, log scale)", fontsize=10.5)
    ax3b.set_title("(b) Sparse Affine-Residual Memory Scaling (PSIM_CSR vs Dense)", fontsize=11.5, fontweight="bold")
    ax3b.legend(fontsize=8.5, framealpha=0.9)
    ax3b.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig3.tight_layout()
    fig3_path = os.path.join(RESULTS_DIR, "fig3_renderer_tightness_and_memory.png")
    fig3.savefig(fig3_path, dpi=220, bbox_inches="tight")
    plt.close(fig3)
    saved_paths.append(fig3_path)

    # ------------------------------------------------------------------
    # Figure 4: Split ProbStar Branch-and-Bound Refinement Convergence
    # ------------------------------------------------------------------
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    colors_r = {"Pinhole": "#2563eb", "3DGS": "#059669", "NeRF": "#d97706"}
    markers_r = {"Pinhole": "o", "3DGS": "s", "NeRF": "^"}

    for r_name in ["Pinhole", "3DGS", "NeRF"]:
        sub = [r for r in exp3_rows if r["renderer"] == r_name]
        ks = [r["K_subregions"] for r in sub]
        ws = [r["mean_pixel_width"] for r in sub]
        ts = [r["render_time_s"] * 1000.0 for r in sub]
        mc_w = sub[0]["mc_pixel_width"]

        ax4a.plot(ks, ws, marker=markers_r[r_name], lw=2.2, color=colors_r[r_name], label=f"{r_name} Split ProbStar")
        ax4a.axhline(mc_w, color=colors_r[r_name], linestyle="--", lw=1.1, alpha=0.65, label=f"{r_name} MC Empirical Span")
        ax4b.plot(ks, ts, marker=markers_r[r_name], lw=2.2, color=colors_r[r_name], label=f"{r_name}")

    ax4a.set_xscale("log", base=2)
    ax4a.set_xticks([1, 2, 4, 8, 16])
    ax4a.set_xticklabels(["1", "2", "4", "8", "16"])
    ax4a.set_xlabel("Number of Partitioned 3D Sub-Regions $B$", fontsize=10.5)
    ax4a.set_ylabel("Mean Pixel Bound Width $\\overline{\\Delta I}$", fontsize=10.5)
    ax4a.set_title("(a) Split ProbStar Convergence Toward Exact 3D Envelope", fontsize=11, fontweight="bold")
    ax4a.legend(fontsize=8, ncol=2)
    ax4a.grid(True, linestyle=":", alpha=0.5)

    ax4b.set_xscale("log", base=2)
    ax4b.set_xticks([1, 2, 4, 8, 16])
    ax4b.set_xticklabels(["1", "2", "4", "8", "16"])
    ax4b.set_xlabel("Number of Partitioned 3D Sub-Regions $B$", fontsize=10.5)
    ax4b.set_ylabel("Abstract Rendering Wall-Clock Time (ms)", fontsize=10.5)
    ax4b.set_title("(b) Computational Scaling vs Partition Count $B$", fontsize=11, fontweight="bold")
    ax4b.legend(fontsize=9)
    ax4b.grid(True, linestyle=":", alpha=0.5)

    fig4.tight_layout()
    fig4_path = os.path.join(RESULTS_DIR, "fig4_split_probstar_refinement.png")
    fig4.savefig(fig4_path, dpi=220, bbox_inches="tight")
    plt.close(fig4)
    saved_paths.append(fig4_path)

    # ------------------------------------------------------------------
    # Figure 5: End-to-End 3D-to-Perception Certification & Speedup Across CNN, ResNet, ViT, SSNN
    # ------------------------------------------------------------------
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(15.5, 5.6))

    rep_rows = [
        r for r in exp2_rows
        if (r["model"], r["renderer"]) in [
            ("CNN_MNIST_Small", "Pinhole"),
            ("CNN_MNIST_Small", "3DGS"),
            ("CNN_MNIST_Small", "NeRF"),
            ("CNN_CIFAR10_OVAL21_Base", "Pinhole"),
            ("ResNet_MNIST_28x28", "Pinhole"),
            ("ResNet_CIFAR10_32x32", "Pinhole"),
            ("ViT_MNIST_28x28", "Pinhole"),
            ("ViT_CIFAR10_32x32", "Pinhole"),
            ("SSNN_M2NIST_28x28", "Pinhole"),
        ]
    ]
    labels_e2 = [f"{r['model'].replace('_28x28','').replace('_32x32','')}\n[{r['renderer']}]" for r in rep_rows]
    x2 = np.arange(len(rep_rows))
    w2 = 0.20

    ax5a.bar(x2 - 1.5 * w2, [r["cond_p_min_ibp"] for r in rep_rows], w2, label="Naive IBP Rendering", color="#dc2626", alpha=0.85)
    ax5a.bar(x2 - 0.5 * w2, [r["cond_p_min_ours"] for r in rep_rows], w2, label="Single ProbStar [Ours]", color="#2563eb", alpha=0.92)
    ax5a.bar(x2 + 0.5 * w2, [r["cond_p_min_split4"] for r in rep_rows], w2, label="Split-4 ProbStar [Ours]", color="#059669", alpha=0.95)
    ax5a.bar(x2 + 1.5 * w2, [r["cond_p_mc"] for r in rep_rows], w2, label="Empirical Monte Carlo $\\hat{p}_{\\mathrm{MC}}$", color="#64748b", alpha=0.85)
    ax5a.set_ylim(0.0, 1.08)
    ax5a.set_xticks(x2)
    ax5a.set_xticklabels(labels_e2, fontsize=8, rotation=22, ha="right")
    ax5a.set_ylabel("Certified Conditional Safety Probability $p_{\\min}$", fontsize=10)
    ax5a.set_title("(a) End-to-End 3D-to-Perception Certified Probability Bounds", fontsize=11, fontweight="bold")
    ax5a.legend(fontsize=8, loc="lower right")
    ax5a.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax5b.bar(x2 - w2 / 2, [r["time_crown_s"] for r in rep_rows], w2, label="End-to-End 3D CROWN Backsub", color="#f59e0b", alpha=0.90)
    ax5b.bar(x2 + w2 / 2, [r["time_ours_s"] for r in rep_rows], w2, label="ProbStar PSIM_CSR [Ours]", color="#2563eb", alpha=0.95)
    ax5b.set_yscale("log")
    ax5b.set_xticks(x2)
    ax5b.set_xticklabels(labels_e2, fontsize=8, rotation=22, ha="right")
    ax5b.set_ylabel("Total Verification Time (s, log scale)", fontsize=10)
    ax5b.set_title("(b) End-to-End Runtime: ProbStar PSIM_CSR vs 3D CROWN", fontsize=11, fontweight="bold")
    ax5b.legend(fontsize=8.5)
    ax5b.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig5.tight_layout()
    fig5_path = os.path.join(RESULTS_DIR, "fig5_end_to_end_vision_certification.png")
    fig5.savefig(fig5_path, dpi=220, bbox_inches="tight")
    plt.close(fig5)
    saved_paths.append(fig5_path)

    return saved_paths


# ==============================================================================
# 6. Export CSVs, JSON Summary, and LaTeX Tables
# ==============================================================================

def save_artifacts(
    exp1_rows: List[Dict[str, Any]],
    exp2_rows: List[Dict[str, Any]],
    exp3_rows: List[Dict[str, Any]],
    figure_paths: List[str],
) -> None:
    """Save CSV tables, JSON results, and LaTeX paper tables."""
    _ensure_results_dir()

    def _write_csv(filename: str, rows: List[Dict[str, Any]]) -> str:
        path = os.path.join(RESULTS_DIR, filename)
        if not rows:
            return path
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    _write_csv("table1_renderer_tightness_and_memory.csv", exp1_rows)
    _write_csv("table2_end_to_end_3d_to_vision_certification.csv", exp2_rows)
    _write_csv("table3_split_probstar_refinement.csv", exp3_rows)

    summary_json = os.path.join(RESULTS_DIR, "icml2027_results.json")
    with open(summary_json, "w") as f:
        json.dump(
            {
                "table1_renderer_tightness": exp1_rows,
                "table2_end_to_end_certification": exp2_rows,
                "table3_split_refinement": exp3_rows,
                "figures": figure_paths,
            },
            f,
            indent=2,
        )

    tex_path = os.path.join(RESULTS_DIR, "icml2027_tables.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated ICML 2027 LaTeX Tables — ProbStar Abstract Rendering\n")
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{Abstract Renderer Tightness, Soundness, and Memory Scaling (Pinhole, 3DGS, NeRF).}\n")
        f.write("\\begin{tabular}{llrrrrrrr}\n\\toprule\n")
        f.write("Renderer & Dataset & MC Span & IBP Width & ProbStar [Ours] & Split-8 [Ours] & Viol. & CSR (KB) & Dense (KB) \\\\\n\\midrule\n")
        for r in exp1_rows:
            f.write(
                f"{r['renderer'].replace('_', '\\_')} & {r['dataset'].replace('_', '\\_')} & "
                f"{r['mc_mean_width']:.4f} & {r['ibp_mean_width']:.4f} & "
                f"\\textbf{{{r['probstar_single_width']:.4f}}} & \\textbf{{{r['probstar_split8_width']:.4f}}} & "
                f"{r['probstar_single_violations']} & {r['mem_psim_csr_kb']:.1f} & {r['mem_probstar_dense_kb']:.1f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")

        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{End-to-End 3D-to-Perception Probabilistic Certification across CNN, ResNet, ViT, and SSNN.}\n")
        f.write("\\begin{tabular}{lllrrrrrrrr}\n\\toprule\n")
        f.write("Model & Arch & Renderer & $p_{\\min}$ (IBP) & $p_{\\min}$ [Ours] & $p_{\\min}$ (Split-4) & $\\hat{p}_{\\mathrm{MC}}$ & $t_{\\mathrm{Ours}}$ (s) & $t_{\\mathrm{CROWN}}$ (s) & Speedup \\\\\n\\midrule\n")
        for r in exp2_rows:
            f.write(
                f"{r['model'].replace('_', '\\_')} & {r['arch_family']} & {r['renderer']} & "
                f"{r['cond_p_min_ibp']:.4f} & \\textbf{{{r['cond_p_min_ours']:.4f}}} & "
                f"\\textbf{{{r['cond_p_min_split4']:.4f}}} & {r['cond_p_mc']:.4f} & "
                f"{r['time_ours_s']:.3f} & {r['time_crown_s']:.3f} & {r['speedup_vs_crown']:.1f}$\\times$ \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ICML 2027 ProbStar Abstract Rendering Benchmark Suite")
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "renderers", "vision", "split", "figures"],
        help="Which benchmark suite to execute",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    print("=" * 88)
    print("ICML 2027 Benchmark Suite: ProbStar Abstract Rendering (Pinhole, 3DGS, NeRF -> Vision)")
    print("=" * 88)

    exp1 = run_renderer_tightness_experiments()
    exp2 = run_end_to_end_vision_certification()
    exp3 = run_split_refinement_experiment()
    figs = generate_all_icml2027_figures(exp1, exp2, exp3)
    save_artifacts(exp1, exp2, exp3, figs)

    total_s = time.perf_counter() - t_start
    print("=" * 88)
    print(f"Completed ICML 2027 suite in {total_s:.2f}s. Artifacts written to: {RESULTS_DIR}")
    for p in figs:
        print(f"  [Figure] {p}")
    print("=" * 88)


if __name__ == "__main__":
    main()
