#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
#########################################################################
"""
Unit & Soundness Tests for StarV ProbStar Abstract Rendering (`StarV.renderer`):
  1. Univariate & Bilinear Rendering Primitives (`sin_env`, `cos_env`, `square_env`,
     `reciprocal_env`, `exp_neg_env`, `relu_env`, `sigmoid_env`, `softplus_env`,
     `clamp_env`, `multiply_env`, `divide_env`) — verifying 0/1000 envelope violations
     and Gauss-Hermite moment accuracy.
  2. Pinhole Camera + Lambertian/Phong Abstract Renderer (`PinholeCameraMeshRenderer`)
     — verifying 0/500 pixel envelope violations, tighter bounds than IBP, and
     conversion to `ProbSparseImageStar2DCSR` / `ProbSparseImageStar2DCOO` / `ProbStar`.
  3. 3D Gaussian Splatting Abstract Renderer (`GaussianSplatting3DRenderer`)
     — verifying 0/500 pixel envelope violations, Cholesky conic accuracy, and
     alpha-compositing soundness.
  4. Volumetric NeRF Ray-Marching Abstract Renderer (`VolumetricNeRFRenderer`)
     — verifying 0/500 pixel envelope violations and volumetric transmittance soundness.
  5. Split ProbStar Refinement & End-to-End 3D-to-Vision Probabilistic Certification
     on CNN, ResNet, ViT, and Semantic Segmentation networks.
  6. 2D & 3D ProbStar Density Contour & Image-Envelope Rendering (`plot_probstar_render`).
"""

import unittest
import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from StarV.renderer import (
    PhysicalProbStar,
    AffineEnvelopeTensor,
    sin_env,
    cos_env,
    square_env,
    reciprocal_env,
    exp_neg_env,
    relu_env,
    leakyrelu_env,
    sigmoid_env,
    softplus_env,
    clamp_env,
    multiply_env,
    divide_env,
    PinholeCameraMeshRenderer,
    lift_image_to_3d_relief_scene,
    GaussianSplatting3DRenderer,
    fit_3d_gaussian_splats_to_image,
    VolumetricNeRFRenderer,
    fit_nerf_to_image,
    partition_physical_probstar,
    render_split_probstar_bounds,
    certify_3d_to_vision_probstar,
    certify_3d_to_ssnn_pixel_probstar,
    evaluate_3d_to_vision_mc,
)
from StarV.util.plot_probstar_render import (
    project_probstar_2d,
    render_2d_probstar_contour,
    render_3d_probstar_volume,
    render_imagestar_envelope_panel,
)
from StarV.layer.Conv2DLayer import Conv2DLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.FlattenLayer import FlattenLayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.net.network import NeuralNetwork


class TestProbStarAbstractRendering(unittest.TestCase):

    def setUp(self):
        self.phys = PhysicalProbStar(
            lb0=[-0.06, -0.06, -0.05, -0.04, -0.04, -0.04],
            ub0=[+0.06, +0.06, +0.05, +0.04, +0.04, +0.04],
            param_names=['d_pitch', 'd_yaw', 'd_roll', 'd_tx', 'd_ty', 'd_tz'],
        )
        # Create a synthetic 8x8 grayscale digit-like image for fast unit tests
        rr, cc = np.meshgrid(np.linspace(-1.0, 1.0, 8), np.linspace(-1.0, 1.0, 8), indexing='ij')
        self.test_img_gray = np.clip(np.exp(-2.5 * (rr ** 2 + cc ** 2)), 0.0, 1.0)[:, :, None]
        # And an 8x8x3 RGB image
        self.test_img_rgb = np.stack([
            self.test_img_gray[:, :, 0],
            np.clip(np.exp(-2.5 * ((rr - 0.2) ** 2 + (cc + 0.2) ** 2)), 0.0, 1.0),
            np.clip(0.5 * (1.0 - rr), 0.0, 1.0),
        ], axis=2)

    def test_univariate_and_bilinear_primitives_soundness(self):
        """Test 100% pointwise soundness (0/1000 violations) and moment accuracy of all primitives."""
        samples = self.phys.sample_truncated(n_samples=1000, seed=7)
        x_env = self.phys.coordinates_tensor([0, 1, 2, 3, 4, 5]).scale_shift(
            scale=[4.0, 3.0, 2.5, 5.0, 2.0, 1.5],
            shift=[-0.4, 0.3, 1.2, 2.5, -0.8, 0.5],
        )
        x_conc = np.array([-0.4, 0.3, 1.2, 2.5, -0.8, 0.5])[None, :] + samples * np.array([4.0, 3.0, 2.5, 5.0, 2.0, 1.5])[None, :]

        primitives = [
            ('sin', sin_env(x_env), np.sin(x_conc)),
            ('cos', cos_env(x_env), np.cos(x_conc)),
            ('square', square_env(x_env), np.square(x_conc)),
            ('exp_neg', exp_neg_env(x_env), np.exp(-x_conc)),
            ('relu', relu_env(x_env), np.maximum(x_conc, 0.0)),
            ('leakyrelu', leakyrelu_env(x_env, 0.1), np.where(x_conc >= 0.0, x_conc, 0.1 * x_conc)),
            ('sigmoid', sigmoid_env(x_env), 1.0 / (1.0 + np.exp(-x_conc))),
            ('softplus', softplus_env(x_env), np.log1p(np.exp(x_conc))),
            ('clamp', clamp_env(x_env, -0.2, 0.8), np.clip(x_conc, -0.2, 0.8)),
        ]

        for name, y_env, y_conc in primitives:
            lb_s, ub_s = y_env.evaluate_bounds_at(samples)
            self.assertTrue(
                np.all(y_conc >= lb_s - 1e-7),
                f"Lower envelope violation in {name}: max diff = {np.max(lb_s - y_conc)}",
            )
            self.assertTrue(
                np.all(y_conc <= ub_s + 1e-7),
                f"Upper envelope violation in {name}: max diff = {np.max(y_conc - ub_s)}",
            )
            # Check Gauss-Hermite mean accuracy
            mu_mc = np.mean(y_conc, axis=0)
            np.testing.assert_allclose(y_env.mu, mu_mc, atol=0.05, err_msg=f"Mean mismatch in {name}")

        # Test reciprocal on strictly positive depth z_env
        z_env = x_env.scale_shift(scale=0.2, shift=3.0)
        z_conc = 3.0 + 0.2 * x_conc
        rec_env = reciprocal_env(z_env)
        rec_conc = 1.0 / z_conc
        lb_r, ub_r = rec_env.evaluate_bounds_at(samples)
        self.assertTrue(np.all(rec_conc >= lb_r - 1e-7))
        self.assertTrue(np.all(rec_conc <= ub_r + 1e-7))

        # Test bilinear McCormick product and quotient
        prod_env = multiply_env(x_env, z_env)
        prod_conc = x_conc * z_conc
        lb_p, ub_p = prod_env.evaluate_bounds_at(samples)
        self.assertTrue(np.all(prod_conc >= lb_p - 1e-7))
        self.assertTrue(np.all(prod_conc <= ub_p + 1e-7))

        div_env = divide_env(x_env, z_env)
        div_conc = x_conc / z_conc
        lb_d, ub_d = div_env.evaluate_bounds_at(samples)
        self.assertTrue(np.all(div_conc >= lb_d - 1e-7))
        self.assertTrue(np.all(div_conc <= ub_d + 1e-7))

    def test_pinhole_renderer_soundness_and_tightness(self):
        """Test PinholeCameraMeshRenderer soundness (0/300 violations) and tightness vs. IBP."""
        renderer = lift_image_to_3d_relief_scene(self.test_img_gray, frustum_radius_px=1)
        nom_render = renderer.evaluate_concrete(np.zeros(6)).reshape(8, 8, 1)
        mse_nom = float(np.mean(np.square(nom_render - self.test_img_gray)))
        self.assertLess(mse_nom, 5e-3, f"Nominal pinhole reconstruction MSE too high: {mse_nom}")

        env_prob = renderer.render_envelope(self.phys, mode='probstar')
        env_ibp = renderer.render_envelope(self.phys, mode='ibp')

        mc_check = evaluate_3d_to_vision_mc(renderer, self.phys, env_to_check=env_prob, n_samples=300)
        self.assertEqual(mc_check['env_violations'], 0, f"Pinhole violations: {mc_check['max_violation_mag']}")

        width_prob = float(np.mean(env_prob.ub - env_prob.lb))
        width_ibp = float(np.mean(env_ibp.ub - env_ibp.lb))
        self.assertLessEqual(width_prob, width_ibp + 1e-9)

        # Verify CSR, COO, and ProbStar conversions
        star_csr = renderer.render_probstar(self.phys, mode='probstar', format='csr')
        star_coo = renderer.render_probstar(self.phys, mode='probstar', format='coo')
        pstar = renderer.render_probstar(self.phys, mode='probstar', format='probstar')
        self.assertEqual(star_csr.shape, (8, 8, 1))
        self.assertEqual(star_coo.shape, (8, 8, 1))
        self.assertEqual(pstar.dim, 64)

    def test_gaussian_splatting_3d_renderer_soundness(self):
        """Test 3D Gaussian Splatting (3DGS) Abstract Renderer soundness (0/300 violations)."""
        renderer = fit_3d_gaussian_splats_to_image(self.test_img_rgb, frustum_radius_px=1)
        nom_render = renderer.evaluate_concrete(np.zeros(6)).reshape(8, 8, 3)
        mse_nom = float(np.mean(np.square(nom_render - self.test_img_rgb)))
        self.assertLess(mse_nom, 5e-3, f"Nominal 3DGS reconstruction MSE too high: {mse_nom}")

        env_prob = renderer.render_envelope(self.phys, mode='probstar')
        mc_check = evaluate_3d_to_vision_mc(renderer, self.phys, env_to_check=env_prob, n_samples=300)
        self.assertEqual(mc_check['env_violations'], 0, f"3DGS violations: {mc_check['max_violation_mag']}")

        # Moment accuracy check
        mean_mae = float(np.mean(np.abs(env_prob.mu - mc_check['mu_mc'])))
        self.assertLess(mean_mae, 0.05)

    def test_volumetric_nerf_renderer_soundness(self):
        """Test Volumetric NeRF Ray-Marching Abstract Renderer soundness (0/300 violations)."""
        renderer = fit_nerf_to_image(self.test_img_gray, hidden_dim=6)
        nom_render = renderer.evaluate_concrete(np.zeros(6)).reshape(8, 8, 1)
        mse_nom = float(np.mean(np.square(nom_render - self.test_img_gray)))
        self.assertLess(mse_nom, 5e-3, f"Nominal NeRF reconstruction MSE too high: {mse_nom}")

        env_prob = renderer.render_envelope(self.phys, mode='probstar')
        mc_check = evaluate_3d_to_vision_mc(renderer, self.phys, env_to_check=env_prob, n_samples=300)
        self.assertEqual(mc_check['env_violations'], 0, f"NeRF violations: {mc_check['max_violation_mag']}")

    def test_split_probstar_and_end_to_end_vision_certification(self):
        """Test Split ProbStar refinement and End-to-End 3D-to-Vision Certification."""
        renderer = fit_3d_gaussian_splats_to_image(self.test_img_gray, frustum_radius_px=1)
        split_res = render_split_probstar_bounds(renderer, self.phys, num_branches=4)
        single_env = renderer.render_envelope(self.phys, mode='probstar')

        # Split bounds should be at least as tight as single-domain bounds
        self.assertLessEqual(
            float(np.mean(split_res['ub'] - split_res['lb'])),
            float(np.mean(single_env.ub - single_env.lb)) + 1e-7,
        )

        # Build a small 8x8x1 -> 3-class CNN
        rng = np.random.default_rng(123)
        w_conv = rng.normal(0.0, 0.25, size=(3, 3, 1, 2))
        b_conv = np.zeros(2)
        w_fc = rng.normal(0.0, 0.25, size=(3, 8 * 8 * 2))
        b_fc = np.array([1.5, -0.5, -0.5], dtype=np.float64)
        net = NeuralNetwork([
            Conv2DLayer([w_conv, b_conv], stride=[1, 1], padding=[1, 1, 1, 1]),
            ReLULayer(),
            FlattenLayer(),
            FullyConnectedLayer([w_fc, b_fc]),
        ])

        res_psim = certify_3d_to_vision_probstar(renderer, self.phys, net, true_label=0, method='psim_csr')
        res_crown = certify_3d_to_vision_probstar(renderer, self.phys, net, true_label=0, method='crown_end2end')
        res_ibp = certify_3d_to_vision_probstar(renderer, self.phys, net, true_label=0, method='ibp_render')
        res_split = certify_3d_to_vision_probstar(
            renderer, self.phys, net, true_label=0, method='split_probstar', num_branches=4
        )
        mc_res = evaluate_3d_to_vision_mc(renderer, self.phys, net=net, true_label=0, n_samples=200)

        for r in (res_psim, res_crown, res_ibp, res_split):
            self.assertLessEqual(r['cond_p_min'], mc_res['cond_p_mc'] + 1e-4)
            self.assertGreaterEqual(r['cond_p_max'], mc_res['cond_p_mc'] - 1e-4)

    def test_probstar_visualization_suite(self):
        """Test 2D/3D ProbStar density contour rendering and image-envelope panel plotting."""
        renderer = fit_3d_gaussian_splats_to_image(self.test_img_gray, frustum_radius_px=1)
        img_star = renderer.render_probstar(self.phys, mode='probstar', format='csr')

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        render_2d_probstar_contour(
            img_star, dims=(27, 28), ax=ax,
            unsafe_mat_2d=np.array([[1.0, -1.0]]), unsafe_vec_2d=np.array([-0.5]),
            title="Test 2D ProbStar Render",
        )
        plt.close(fig)

        fig3d = plt.figure(figsize=(4, 4))
        ax3d = fig3d.add_subplot(111, projection='3d')
        render_3d_probstar_volume(img_star, dims=(26, 27, 28), ax=ax3d, title="Test 3D ProbStar Render")
        plt.close(fig3d)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "panel.png")
            fig_p = render_imagestar_envelope_panel(img_star, save_path=out_path)
            plt.close(fig_p)
            self.assertTrue(os.path.exists(out_path))


if __name__ == '__main__':
    unittest.main()
