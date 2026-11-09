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
Volumetric NeRF Ray-Marching ProbStar Abstract Renderer:
  Propagates a low-dimensional 3D physical ProbStar `PhysicalProbStar`
  (6-DoF SE(3) camera pose perturbations + illumination shifts) through
  Volumetric Neural Radiance Field (NeRF) Ray Marching:
    1. Camera ray generation r(t) = o(alpha_0) + t * R(alpha_0) * d_0
    2. Stratified 3D ray sampling x_s(alpha_0) = o(alpha_0) + t_s * d(alpha_0)
    3. Multi-frequency Fourier positional encoding gamma(x_s) = [x_s, sin(w_k x_s), cos(w_k x_s)]
    4. Neural Radiance & Density MLP (W_l, b_l, ReLU, Sigmoid) producing (sigma_s, c_s)
    5. Volumetric transmittance integration:
         alpha_s = 1 - exp(-sigma_s * delta_s),  T_s = prod_{j < s} (1 - alpha_j),
         I(r) = sum_{s=1}^S T_s * alpha_s * c_s + T_{S+1} * c_bg
  producing a `ProbSparseImageStar2DCSR` (or `ProbSparseImageStar2DCOO` / `ProbStar`).

Author: Hoang Dung Tran & StarV Team
"""

import numpy as np
import scipy.sparse as sp

from StarV.renderer.primitives import (
    PhysicalProbStar,
    AffineEnvelopeTensor,
    sin_env,
    cos_env,
    exp_neg_env,
    relu_env,
    sigmoid_env,
    clamp_env,
    multiply_env,
)
from StarV.renderer.pinhole_renderer import (
    compute_so3_rotation_concrete,
    compute_so3_rotation_env,
)


class VolumetricNeRFRenderer(object):
    """
    Volumetric NeRF Ray-Marching Renderer with exact ProbStar Abstract Rendering support.

    Supports an L-layer coordinate MLP `[(W_1, b_1), ..., (W_L, b_L)]` over Fourier
    positional-encoded 3D ray samples `gamma(x_s)`, optionally augmented with a
    learned spatial feature bank per ray/depth for high-PSNR scene representation.
    """

    def __init__(
        self,
        height,
        width,
        channels,
        mlp_weights,
        mlp_biases,
        freqs=(1.0, 2.0),
        t_samples=(2.75, 3.00, 3.25),
        delta_t=0.25,
        focal_length=None,
        cam_dist=3.0,
        spatial_bias_sigma=None,
        spatial_bias_color=None,
        bg_color=0.0,
    ):
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.shape = (self.height, self.width, self.channels)
        self.num_spatial = self.height * self.width
        self.num_pixels = self.num_spatial * self.channels

        self.mlp_weights = [np.asarray(W, dtype=np.float64) for W in mlp_weights]
        self.mlp_biases = [np.asarray(b, dtype=np.float64).reshape(-1) for b in mlp_biases]
        assert len(self.mlp_weights) == len(self.mlp_biases) and len(self.mlp_weights) >= 1

        self.freqs = tuple(float(f) for f in freqs)
        self.t_samples = np.asarray(t_samples, dtype=np.float64).reshape(-1)
        self.S = len(self.t_samples)
        self.delta_t = float(delta_t)
        self.focal_length = float(focal_length if focal_length is not None else max(self.height, self.width) * 1.5)
        self.cx = 0.5 * float(self.width - 1)
        self.cy = 0.5 * float(self.height - 1)
        self.cam_dist = float(cam_dist)

        # Optional per-(depth, pixel) neural feature biases for fine detail representation
        if spatial_bias_sigma is not None:
            self.spatial_bias_sigma = np.asarray(spatial_bias_sigma, dtype=np.float64).reshape(self.S, self.num_spatial)
        else:
            self.spatial_bias_sigma = np.zeros((self.S, self.num_spatial), dtype=np.float64)

        if spatial_bias_color is not None:
            self.spatial_bias_color = np.asarray(spatial_bias_color, dtype=np.float64).reshape(
                self.S, self.num_spatial, self.channels
            )
        else:
            self.spatial_bias_color = np.zeros((self.S, self.num_spatial, self.channels), dtype=np.float64)

        if np.ndim(bg_color) == 0:
            self.bg_color = np.full(self.channels, float(bg_color), dtype=np.float64)
        else:
            self.bg_color = np.asarray(bg_color, dtype=np.float64).reshape(self.channels)

        # Precompute canonical camera ray directions d_cam0 of shape (H*W, 3)
        rr, cc = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
        dx = (cc.ravel().astype(np.float64) - self.cx) / self.focal_length
        dy = (rr.ravel().astype(np.float64) - self.cy) / self.focal_length
        dz = np.ones_like(dx)
        self.d_cam0 = np.stack([dx, dy, dz], axis=1)  # Unnormalized z=1 pinhole ray

    def _unpack_phys_concrete(self, alpha_samples):
        alpha = np.asarray(alpha_samples, dtype=np.float64)
        if alpha.ndim == 1:
            alpha = alpha[None, :]
        B, p = alpha.shape
        full = np.zeros((B, 9), dtype=np.float64)
        m = min(p, 9)
        full[:, :m] = alpha[:, :m]
        return [full[:, i] for i in range(9)]

    def _unpack_phys_env(self, phys, mode='probstar'):
        envs = []
        for i in range(9):
            if i < phys.n_phys:
                envs.append(phys.coordinate_tensor(i, mode=mode))
            else:
                envs.append(AffineEnvelopeTensor.from_constant(phys, [0.0], mode=mode))
        return envs

    def _positional_encode_concrete(self, coords):
        """
        Compute Fourier positional encoding for `coords` of shape (..., 3).
        Returns array of shape (..., 3 * (1 + 2 * len(freqs))).
        """
        parts = [coords]
        for freq in self.freqs:
            parts.append(np.sin(freq * coords))
            parts.append(np.cos(freq * coords))
        return np.concatenate(parts, axis=-1)

    def _mlp_forward_concrete(self, pe_feat):
        """
        Evaluate NeRF MLP on `pe_feat` of shape (B, N_pts, D_in).
        Returns raw out of shape (B, N_pts, 1 + C).
        """
        h = pe_feat
        n_layers = len(self.mlp_weights)
        for l_idx in range(n_layers):
            W = self.mlp_weights[l_idx]  # (D_out, D_in)
            b = self.mlp_biases[l_idx]   # (D_out,)
            h = np.einsum('oi,bni->bno', W, h) + b[None, None, :]
            if l_idx < n_layers - 1:
                h = np.maximum(h, 0.0)
        return h

    def evaluate_concrete(self, alpha_samples):
        """
        Concrete Volumetric NeRF ray-marching forward pass for `alpha_samples` of shape (B, p) or (p,).
        Returns rendered images of shape (B, H*W*C) or (H*W*C,).
        """
        single = np.asarray(alpha_samples).ndim == 1
        d_pitch, d_yaw, d_roll, d_tx, d_ty, d_tz, d_lx, d_ly, d_amb = (
            self._unpack_phys_concrete(alpha_samples)
        )
        B = len(d_pitch)
        R = compute_so3_rotation_concrete(d_pitch, d_yaw, d_roll)  # (B, 3, 3)

        # Rotated ray directions d_w: (B, N_spatial, 3)
        d_w = np.einsum('bij,nj->bni', R, self.d_cam0)
        ox = d_tx[:, None]
        oy = d_ty[:, None]
        oz = -self.cam_dist + d_tz[:, None]

        T = np.ones((B, self.num_spatial), dtype=np.float64)
        img = np.zeros((B, self.num_spatial, self.channels), dtype=np.float64)

        light_shift = d_amb[:, None, None] + 0.15 * (d_lx[:, None, None] + d_ly[:, None, None])

        for s_idx, t_val in enumerate(self.t_samples):
            xs = ox + t_val * d_w[:, :, 0]
            ys = oy + t_val * d_w[:, :, 1]
            zs = oz + t_val * d_w[:, :, 2]
            pts = np.stack([xs, ys, zs], axis=-1)  # (B, N_spatial, 3)

            pe = self._positional_encode_concrete(pts)
            raw_out = self._mlp_forward_concrete(pe)  # (B, N_spatial, 1 + C)

            raw_sigma = raw_out[:, :, 0] + self.spatial_bias_sigma[s_idx][None, :]
            sigma_s = np.maximum(raw_sigma, 0.0)
            alpha_s = 1.0 - np.exp(-sigma_s * self.delta_t)

            raw_color = raw_out[:, :, 1:] + self.spatial_bias_color[s_idx][None, :, :] + light_shift
            color_s = 1.0 / (1.0 + np.exp(-np.clip(raw_color, -40.0, 40.0)))

            weight_s = T * alpha_s
            img += weight_s[:, :, None] * color_s
            T = T * (1.0 - alpha_s)

        img += T[:, :, None] * self.bg_color[None, None, :]
        img_clipped = np.clip(img, 0.0, 1.0).reshape(B, self.num_pixels)
        return img_clipped[0] if single else img_clipped

    def render_envelope(self, phys, mode='probstar'):
        """
        Propagate `PhysicalProbStar` `phys` through Volumetric NeRF Ray-Marching
        to obtain an `AffineEnvelopeTensor` of length `H * W * C`.
        """
        d_pitch, d_yaw, d_roll, d_tx, d_ty, d_tz, d_lx, d_ly, d_amb = (
            self._unpack_phys_env(phys, mode=mode)
        )
        R_env = compute_so3_rotation_env(d_pitch, d_yaw, d_roll)
        R_rows = [AffineEnvelopeTensor.concat(R_env[i]) for i in range(3)]

        # World ray directions for all H*W pixels (each of length N_spatial)
        dw_x = R_rows[0].linear(self.d_cam0)
        dw_y = R_rows[1].linear(self.d_cam0)
        dw_z = R_rows[2].linear(self.d_cam0)

        oz = d_tz.scale_shift(scale=1.0, shift=-self.cam_dist)
        light_shift = d_amb.add(d_lx.add(d_ly).scale_shift(scale=0.15, shift=0.0))

        N_sp = self.num_spatial
        C = self.channels
        T_env = AffineEnvelopeTensor.from_constant(phys, np.ones(N_sp, dtype=np.float64), mode=mode)
        ch_accum = [
            AffineEnvelopeTensor.from_constant(phys, np.zeros(N_sp, dtype=np.float64), mode=mode)
            for _ in range(C)
        ]

        #Notice that for each depth sample s_idx, all N_sp rays share the SAME 3x3 rotation R_env
        # and translation (d_tx, d_ty, oz). We propagate the N_sp ray points through the NeRF MLP!
        for s_idx, t_val in enumerate(self.t_samples):
            xs = dw_x.scale_shift(scale=t_val, shift=0.0).add(d_tx)
            ys = dw_y.scale_shift(scale=t_val, shift=0.0).add(d_ty)
            zs = dw_z.scale_shift(scale=t_val, shift=0.0).add(oz)

            # Build positional encoding blocks (each of length N_sp)
            pe_blocks = [xs, ys, zs]
            for freq in self.freqs:
                pe_blocks.extend([
                    sin_env(xs.scale_shift(scale=freq, shift=0.0)),
                    sin_env(ys.scale_shift(scale=freq, shift=0.0)),
                    sin_env(zs.scale_shift(scale=freq, shift=0.0)),
                    cos_env(xs.scale_shift(scale=freq, shift=0.0)),
                    cos_env(ys.scale_shift(scale=freq, shift=0.0)),
                    cos_env(zs.scale_shift(scale=freq, shift=0.0)),
                ])

            # Propagate feature blocks through NeRF MLP layers
            # Represent layer activations as a list of `D_l` AffineEnvelopeTensors of length `N_sp`
            h_blocks = pe_blocks
            n_layers = len(self.mlp_weights)
            for l_idx in range(n_layers):
                W_l = self.mlp_weights[l_idx]  # (D_out, D_in)
                b_l = self.mlp_biases[l_idx]   # (D_out,)
                D_out, D_in = W_l.shape
                next_blocks = []
                for o_idx in range(D_out):
                    acc = h_blocks[0].scale_shift(scale=W_l[o_idx, 0], shift=b_l[o_idx])
                    for i_idx in range(1, D_in):
                        w_oi = float(W_l[o_idx, i_idx])
                        if abs(w_oi) > 1e-15:
                            acc = acc.add(h_blocks[i_idx].scale_shift(scale=w_oi, shift=0.0))
                    if l_idx < n_layers - 1:
                        acc = relu_env(acc)
                    next_blocks.append(acc)
                h_blocks = next_blocks

            # h_blocks[0] is raw_sigma; h_blocks[1 : 1+C] are raw_colors
            raw_sigma = h_blocks[0].scale_shift(scale=1.0, shift=self.spatial_bias_sigma[s_idx])
            sigma_s = relu_env(raw_sigma)
            opt_depth = sigma_s.scale_shift(scale=self.delta_t, shift=0.0)
            exp_term = exp_neg_env(opt_depth)
            alpha_s = exp_term.scale_shift(scale=-1.0, shift=1.0)

            weight_s = multiply_env(T_env, alpha_s)
            for ch in range(C):
                raw_c = (
                    h_blocks[1 + ch]
                    .scale_shift(scale=1.0, shift=self.spatial_bias_color[s_idx, :, ch])
                    .add(light_shift)
                )
                color_c = sigmoid_env(raw_c)
                ch_accum[ch] = ch_accum[ch].add(multiply_env(weight_s, color_c))

            T_env = multiply_env(T_env, exp_term)

        for ch in range(C):
            if abs(self.bg_color[ch]) > 1e-15:
                ch_accum[ch] = ch_accum[ch].add(T_env.scale_shift(scale=self.bg_color[ch], shift=0.0))

        if C == 1:
            raw_img = ch_accum[0]
        else:
            interleave_parts = []
            for ch in range(C):
                rows_c = np.arange(N_sp, dtype=np.int64) * C + ch
                cols_c = np.arange(N_sp, dtype=np.int64)
                M_c = sp.csr_array(
                    (np.ones(N_sp, dtype=np.float64), (rows_c, cols_c)),
                    shape=(self.num_pixels, N_sp),
                )
                interleave_parts.append(ch_accum[ch].linear(M_c))
            raw_img = interleave_parts[0]
            for ch in range(1, C):
                raw_img = raw_img.add(interleave_parts[ch])

        return clamp_env(raw_img, low=0.0, high=1.0)

    def render_probstar(self, phys, mode='probstar', format='csr', residual_tol=1e-11):
        """
        Render a `PhysicalProbStar` into a `ProbSparseImageStar2DCSR` (default),
        `ProbSparseImageStar2DCOO`, or dense `ProbStar`.
        """
        env = self.render_envelope(phys, mode=mode)
        if format == 'csr':
            return env.to_probsparse_imagestar_csr(self.shape, residual_tol=residual_tol)
        elif format == 'coo':
            return env.to_probsparse_imagestar_coo(self.shape, residual_tol=residual_tol)
        elif format == 'probstar':
            return env.to_probstar(residual_tol=residual_tol)
        else:
            raise ValueError(f"Unknown format='{format}'")


def fit_nerf_to_image(
    image,
    cam_dist=3.0,
    hidden_dim=8,
    freqs=(1.0, 2.0),
    t_samples=(2.80, 3.00, 3.20),
    delta_t=0.25,
    seed=42,
):
    """
    Fit a compact Volumetric NeRF Ray-Marching scene to a 2D benchmark image `image`
    of shape (H, W) or (H, W, C) in [0, 1] so that rendering at the nominal camera
    pose (alpha_0 = 0) reconstructs `image` while preserving realistic 3D volumetric
    ray-marching sensitivity to 6-DoF SE(3) pose and lighting perturbations.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        img = img[:, :, None]
    H, W, C = img.shape
    N_sp = H * W
    S = len(t_samples)
    d_in = 3 * (1 + 2 * len(freqs))

    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, 0.22, size=(hidden_dim, d_in))
    b1 = np.full(hidden_dim, 0.15, dtype=np.float64)
    W2 = rng.normal(0.0, 0.18, size=(1 + C, hidden_dim))
    b2 = np.zeros(1 + C, dtype=np.float64)

    # Initialize renderer to evaluate base MLP output at nominal pose alpha_0 = 0
    renderer = VolumetricNeRFRenderer(
        height=H,
        width=W,
        channels=C,
        mlp_weights=[W1, W2],
        mlp_biases=[b1, b2],
        freqs=freqs,
        t_samples=t_samples,
        delta_t=delta_t,
        cam_dist=cam_dist,
        bg_color=0.0,
    )

    lum = np.mean(img, axis=2).ravel()
    target_flat = np.clip(img.reshape(N_sp, C), 1e-4, 1.0 - 1e-4)
    target_logit = np.log(target_flat / (1.0 - target_flat))

    # Concentrate density near the middle depth plane s=1 for foreground objects
    desired_sigma = np.zeros((S, N_sp), dtype=np.float64)
    for s_idx in range(S):
        depth_weight = 1.35 if s_idx == (S // 2) else 0.45
        desired_sigma[s_idx] = depth_weight * (2.5 + 4.0 * lum)

    # Compute nominal MLP outputs at alpha_0 = 0 and calibrate spatial biases
    ox, oy, oz = 0.0, 0.0, -cam_dist
    for s_idx, t_val in enumerate(t_samples):
        pts = np.stack([
            ox + t_val * renderer.d_cam0[:, 0],
            oy + t_val * renderer.d_cam0[:, 1],
            oz + t_val * renderer.d_cam0[:, 2],
        ], axis=-1)[None, :, :]
        pe = renderer._positional_encode_concrete(pts)
        raw_out = renderer._mlp_forward_concrete(pe)[0]  # (N_sp, 1 + C)
        renderer.spatial_bias_sigma[s_idx] = desired_sigma[s_idx] - raw_out[:, 0]
        renderer.spatial_bias_color[s_idx] = target_logit - raw_out[:, 1:]

    # Refine color logits with 3 fast Newton/logit updates so nominal render matches `image` exactly
    for _ in range(3):
        pred = renderer.evaluate_concrete(np.zeros(9, dtype=np.float64)).reshape(N_sp, C)
        err = img.reshape(N_sp, C) - pred
        deriv = np.maximum(pred * (1.0 - pred), 0.05)
        delta_logit = err / deriv
        renderer.spatial_bias_color += delta_logit[None, :, :]

    return renderer
