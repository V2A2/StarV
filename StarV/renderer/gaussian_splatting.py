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
3D Gaussian Splatting (3DGS) ProbStar Abstract Renderer:
  Propagates a low-dimensional 3D physical ProbStar `PhysicalProbStar`
  (6-DoF SE(3) camera pose perturbations + spherical-harmonic / lighting variations)
  through anisotropic 3D Gaussian Splatting:
    1. SE(3) transformation of 3D Gaussian centers mu_w -> mu_c(alpha_0)
    2. Perspective projection mu_2d(alpha_0) = (f_x X_c / Z_c + c_x, f_y Y_c / Z_c + c_y)
    3. EWA splatting 2D covariance projection Sigma_2D = J W Sigma_3D W^T J^T + eps I
       and Cholesky-factored conic evaluation q = ||L^T (x - mu_2d(alpha_0))||_2^2
    4. Radial Gaussian opacity evaluation alpha_k = o_k * exp(-q_k)
    5. Depth-sorted front-to-back volumetric alpha-compositing:
         T_k = prod_{j < k} (1 - alpha_j),   I = sum_k T_k * alpha_k * c_k + T_end * c_bg
  producing a `ProbSparseImageStar2DCSR` (or `ProbSparseImageStar2DCOO` / `ProbStar`).

Author: Hoang Dung Tran & StarV Team
"""

import numpy as np
import scipy.sparse as sp

from StarV.renderer.primitives import (
    PhysicalProbStar,
    AffineEnvelopeTensor,
    square_env,
    reciprocal_env,
    exp_neg_env,
    gaussian_rbf_env,
    clamp_env,
    multiply_env,
)
from StarV.renderer.pinhole_renderer import (
    compute_so3_rotation_concrete,
    compute_so3_rotation_env,
)


class GaussianSplatting3DRenderer(object):
    """
    3D Gaussian Splatting (3DGS) Renderer with exact ProbStar Abstract Rendering support.

    Scene Parameters:
      - `means_3d`: (K, 3) 3D Gaussian centers in world coordinates
      - `covs_3d`: (K, 3, 3) 3D covariance matrices Sigma_w,k in world coordinates
      - `opacities`: (K,) base opacities o_k in (0, 1)
      - `colors`: (K, C) base RGB/grayscale colors c_k in [0, 1]
      - `sh_dirs`: (K, 3) directional sensitivity vectors for degree-1 view/lighting SH modulation
    """

    def __init__(
        self,
        height,
        width,
        channels,
        means_3d,
        covs_3d,
        opacities,
        colors,
        sh_dirs=None,
        focal_length=None,
        cam_dist=3.0,
        lowpass_eps=0.30,
        bg_color=0.0,
        frustum_radius_px=1,
        sh_scale=0.25,
    ):
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.shape = (self.height, self.width, self.channels)
        self.num_pixels = self.height * self.width * self.channels

        self.means_3d = np.asarray(means_3d, dtype=np.float64)
        self.K = self.means_3d.shape[0]
        assert self.means_3d.shape == (self.K, 3)

        self.covs_3d = np.asarray(covs_3d, dtype=np.float64).reshape(self.K, 3, 3)
        self.opacities = np.clip(np.asarray(opacities, dtype=np.float64).reshape(self.K), 1e-4, 0.99)
        self.colors = np.asarray(colors, dtype=np.float64).reshape(self.K, self.channels)

        if sh_dirs is None:
            # Default normal/view direction pointing toward camera
            sh_d = np.zeros((self.K, 3), dtype=np.float64)
            sh_d[:, 2] = 1.0
            self.sh_dirs = sh_d
        else:
            sh_d = np.asarray(sh_dirs, dtype=np.float64).reshape(self.K, 3)
            norms = np.maximum(np.linalg.norm(sh_d, axis=1, keepdims=True), 1e-12)
            self.sh_dirs = sh_d / norms

        self.focal_length = float(focal_length if focal_length is not None else max(self.height, self.width) * 1.5)
        self.cx = 0.5 * float(self.width - 1)
        self.cy = 0.5 * float(self.height - 1)
        self.cam_dist = float(cam_dist)
        self.lowpass_eps = float(lowpass_eps)
        self.frustum_radius_px = int(frustum_radius_px)
        self.sh_scale = float(sh_scale)

        if np.ndim(bg_color) == 0:
            self.bg_color = np.full(self.channels, float(bg_color), dtype=np.float64)
        else:
            self.bg_color = np.asarray(bg_color, dtype=np.float64).reshape(self.channels)

        # Precompute EWA 2D covariance projection, Cholesky conic factors, and depth-sorted pixel lists
        self._precompute_ewa_conics_and_tiles()

    def _precompute_ewa_conics_and_tiles(self):
        """
        Precompute EWA projected 2D covariance matrices Sigma_2D,k = J_k Sigma_3D,k J_k^T + eps I_2,
        Cholesky factors L_k of 0.5 * Sigma_2D,k^{-1} = L_k L_k^T so that
            0.5 * [du, dv] Sigma_2D,k^{-1} [du, dv]^T = (l11*du + l21*dv)^2 + (l22*dv)^2,
        and depth-sorted per-pixel Gaussian contributor lists.
        """
        X0 = self.means_3d[:, 0]
        Y0 = self.means_3d[:, 1]
        Z0 = np.maximum(self.means_3d[:, 2] + self.cam_dist, 1e-3)

        # Depth-sort Gaussians front-to-back (smallest Z0 first)
        self.depth_order = np.argsort(Z0)

        self.l11 = np.zeros(self.K, dtype=np.float64)
        self.l21 = np.zeros(self.K, dtype=np.float64)
        self.l22 = np.zeros(self.K, dtype=np.float64)

        f = self.focal_length
        for k in range(self.K):
            zk = Z0[k]
            J_k = np.array([
                [f / zk, 0.0, -f * X0[k] / (zk * zk)],
                [0.0, f / zk, -f * Y0[k] / (zk * zk)],
            ], dtype=np.float64)
            cov2d = J_k @ self.covs_3d[k] @ J_k.T + self.lowpass_eps * np.eye(2, dtype=np.float64)
            half_conic = 0.5 * np.linalg.inv(cov2d)
            # Cholesky: half_conic = L @ L.T with L lower-triangular [[l11, 0], [l21, l22]]
            L = np.linalg.cholesky(half_conic)
            self.l11[k] = L[0, 0]
            self.l21[k] = L[1, 0]
            self.l22[k] = L[1, 1]

        # Nominal 2D centers
        u0 = f * (X0 / Z0) + self.cx
        v0 = f * (Y0 / Z0) + self.cy
        u0_int = np.rint(u0).astype(np.int64)
        v0_int = np.rint(v0).astype(np.int64)

        H, W = self.height, self.width
        num_spatial = H * W
        per_px_lists = [[] for _ in range(num_spatial)]
        rad = self.frustum_radius_px

        for k in self.depth_order:
            uk, vk = int(u0_int[k]), int(v0_int[k])
            for dh in range(-rad, rad + 1):
                r = vk + dh
                if r < 0 or r >= H:
                    continue
                for dw in range(-rad, rad + 1):
                    c = uk + dw
                    if c < 0 or c >= W:
                        continue
                    per_px_lists[r * W + c].append(int(k))

        self.max_overlap = max((len(lst) for lst in per_px_lists), default=0)
        # Organize into depth-layer batches `layer_batches[s] = (px_indices, k_indices, rows, cols)`
        self.layer_batches = []
        for s in range(self.max_overlap):
            px_idx = [p_i for p_i, lst in enumerate(per_px_lists) if len(lst) > s]
            k_idx = [per_px_lists[p_i][s] for p_i in px_idx]
            px_arr = np.asarray(px_idx, dtype=np.int64)
            k_arr = np.asarray(k_idx, dtype=np.int64)
            r_arr = (px_arr // W).astype(np.float64)
            c_arr = (px_arr % W).astype(np.float64)
            self.layer_batches.append((px_arr, k_arr, r_arr, c_arr))

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

    def evaluate_concrete(self, alpha_samples):
        """
        Concrete 3D Gaussian Splatting forward pass for `alpha_samples` of shape (B, p) or (p,).
        Returns rendered images of shape (B, H*W*C) or (H*W*C,).
        """
        single = np.asarray(alpha_samples).ndim == 1
        d_pitch, d_yaw, d_roll, d_tx, d_ty, d_tz, d_lx, d_ly, d_amb = (
            self._unpack_phys_concrete(alpha_samples)
        )
        B = len(d_pitch)
        R = compute_so3_rotation_concrete(d_pitch, d_yaw, d_roll)

        # Transform 3D Gaussian means
        P_rot = np.einsum('bij,kj->bki', R, self.means_3d)
        X_c = P_rot[:, :, 0] + d_tx[:, None]
        Y_c = P_rot[:, :, 1] + d_ty[:, None]
        Z_c = np.maximum(P_rot[:, :, 2] + self.cam_dist + d_tz[:, None], 1e-3)

        inv_Z = 1.0 / Z_c
        u_proj = self.focal_length * (X_c * inv_Z) + self.cx  # (B, K)
        v_proj = self.focal_length * (Y_c * inv_Z) + self.cy  # (B, K)

        # Degree-1 SH / directional color modulation
        sh_rot = np.einsum('bij,kj->bki', R, self.sh_dirs)
        sh_mod = (
            1.0 + d_amb[:, None] +
            self.sh_scale * (sh_rot[:, :, 0] * d_lx[:, None] + sh_rot[:, :, 1] * d_ly[:, None] + (sh_rot[:, :, 2] - 1.0))
        )  # (B, K)

        H, W, C = self.height, self.width, self.channels
        num_spatial = H * W
        T = np.ones((B, num_spatial), dtype=np.float64)
        img = np.zeros((B, num_spatial, C), dtype=np.float64)

        for px_arr, k_arr, r_arr, c_arr in self.layer_batches:
            du = c_arr[None, :] - u_proj[:, k_arr]
            dv = r_arr[None, :] - v_proj[:, k_arr]
            z1 = self.l11[k_arr][None, :] * du + self.l21[k_arr][None, :] * dv
            z2 = self.l22[k_arr][None, :] * dv
            q = np.square(z1) + np.square(z2)
            alpha_k = self.opacities[k_arr][None, :] * np.exp(-q)

            T_cur = T[:, px_arr]
            weight = T_cur * alpha_k  # (B, len(px_arr))
            mod_k = weight * sh_mod[:, k_arr]
            img[:, px_arr, :] += mod_k[:, :, None] * self.colors[k_arr][None, :, :]
            T[:, px_arr] = T_cur * (1.0 - alpha_k)

        img += T[:, :, None] * self.bg_color[None, None, :]
        img_clipped = np.clip(img, 0.0, 1.0).reshape(B, self.num_pixels)
        return img_clipped[0] if single else img_clipped

    def render_envelope(self, phys, mode='probstar'):
        """
        Propagate `PhysicalProbStar` `phys` through the 3D Gaussian Splatting
        renderer to obtain an `AffineEnvelopeTensor` of length `H * W * C`.
        """
        d_pitch, d_yaw, d_roll, d_tx, d_ty, d_tz, d_lx, d_ly, d_amb = (
            self._unpack_phys_env(phys, mode=mode)
        )
        R_env = compute_so3_rotation_env(d_pitch, d_yaw, d_roll)
        R_rows = [AffineEnvelopeTensor.concat(R_env[i]) for i in range(3)]

        # 3D Gaussian centers in camera coordinates
        X_c = R_rows[0].linear(self.means_3d).add(d_tx)
        Y_c = R_rows[1].linear(self.means_3d).add(d_ty)
        Z_c = R_rows[2].linear(self.means_3d, b=np.full(self.K, self.cam_dist)).add(d_tz)

        inv_Z = reciprocal_env(Z_c, min_depth=1e-3)
        u_proj = multiply_env(X_c, inv_Z).scale_shift(scale=self.focal_length, shift=self.cx)
        v_proj = multiply_env(Y_c, inv_Z).scale_shift(scale=self.focal_length, shift=self.cy)

        # Degree-1 SH / directional modulation
        sh_x = R_rows[0].linear(self.sh_dirs)
        sh_y = R_rows[1].linear(self.sh_dirs)
        sh_z = R_rows[2].linear(self.sh_dirs).scale_shift(scale=1.0, shift=-1.0)
        sh_term = (
            multiply_env(sh_x, d_lx)
            .add(multiply_env(sh_y, d_ly))
            .add(sh_z)
            .scale_shift(scale=self.sh_scale, shift=0.0)
        )
        sh_mod = sh_term.add(d_amb.scale_shift(scale=1.0, shift=1.0))

        H, W, C = self.height, self.width, self.channels
        num_spatial = H * W

        T_env = AffineEnvelopeTensor.from_constant(phys, np.ones(num_spatial, dtype=np.float64), mode=mode)
        ch_accum = [
            AffineEnvelopeTensor.from_constant(phys, np.zeros(num_spatial, dtype=np.float64), mode=mode)
            for _ in range(C)
        ]

        for px_arr, k_arr, r_arr, c_arr in self.layer_batches:
            m_b = len(px_arr)
            u_k = u_proj.slice(k_arr)
            v_k = v_proj.slice(k_arr)
            du = u_k.scale_shift(scale=-1.0, shift=c_arr)
            dv = v_k.scale_shift(scale=-1.0, shift=r_arr)

            # Cholesky-factored conic quadratic form: exp(-(z1^2 + z2^2)) = exp(-z1^2) * exp(-z2^2)
            z1 = du.scale_shift(scale=self.l11[k_arr], shift=0.0).add(
                dv.scale_shift(scale=self.l21[k_arr], shift=0.0)
            )
            z2 = dv.scale_shift(scale=self.l22[k_arr], shift=0.0)
            k1 = gaussian_rbf_env(z1)
            k2 = gaussian_rbf_env(z2)
            alpha_k = multiply_env(k1, k2).scale_shift(scale=self.opacities[k_arr], shift=0.0)

            T_sub = T_env.slice(px_arr)
            weight = multiply_env(T_sub, alpha_k)
            mod_k = multiply_env(weight, sh_mod.slice(k_arr))

            # Scatter-add mod_k * color into ch_accum on px_arr
            for ch in range(C):
                col_ch = self.colors[k_arr, ch]
                if np.all(np.abs(col_ch) <= 1e-15):
                    continue
                M_sc = sp.csr_array(
                    (col_ch, (px_arr, np.arange(m_b, dtype=np.int64))),
                    shape=(num_spatial, m_b),
                )
                ch_accum[ch] = ch_accum[ch].add(mod_k.linear(M_sc))

            # Update transmittance T_{s+1} = T_s * (1 - alpha_k) on px_arr
            one_minus_alpha = alpha_k.scale_shift(scale=-1.0, shift=1.0)
            T_next_sub = multiply_env(T_sub, one_minus_alpha)

            # Replace px_arr entries in T_env via linear projection + scatter
            keep_mask = np.ones(num_spatial, dtype=np.float64)
            keep_mask[px_arr] = 0.0
            M_put = sp.csr_array(
                (np.ones(m_b, dtype=np.float64), (px_arr, np.arange(m_b, dtype=np.int64))),
                shape=(num_spatial, m_b),
            )
            T_env = T_env.scale_shift(scale=keep_mask, shift=0.0).add(T_next_sub.linear(M_put))

        # Add background contribution T_final * bg_color
        for ch in range(C):
            if abs(self.bg_color[ch]) > 1e-15:
                ch_accum[ch] = ch_accum[ch].add(T_env.scale_shift(scale=self.bg_color[ch], shift=0.0))

        if C == 1:
            raw_img = ch_accum[0]
        else:
            interleave_parts = []
            for ch in range(C):
                rows_c = np.arange(num_spatial, dtype=np.int64) * C + ch
                cols_c = np.arange(num_spatial, dtype=np.int64)
                M_c = sp.csr_array(
                    (np.ones(num_spatial, dtype=np.float64), (rows_c, cols_c)),
                    shape=(self.num_pixels, num_spatial),
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


def fit_3d_gaussian_splats_to_image(
    image,
    cam_dist=3.0,
    depth_relief=0.22,
    base_opacity=0.82,
    cov_scale_3d=0.0012,
    frustum_radius_px=1,
    calibrate_colors=True,
):
    """
    Convert a 2D benchmark image `image` of shape (H, W) or (H, W, C) in [0, 1]
    into an anisotropic 3D Gaussian Splatting (3DGS) scene calibrated so that
    rendering at the nominal camera pose (alpha_0 = 0) reconstructs `image`.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        img = img[:, :, None]
    H, W, C = img.shape
    focal_length = float(max(H, W) * 1.5)
    cx = 0.5 * float(W - 1)
    cy = 0.5 * float(H - 1)

    lum = np.mean(img, axis=2)
    z_map = -depth_relief * (lum - 0.5)

    dz_du = np.gradient(z_map, axis=1)
    dz_dv = np.gradient(z_map, axis=0)
    nx, ny = -dz_du, -dz_dv
    nz = np.ones_like(z_map)
    nlen = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx, ny, nz = nx / nlen, ny / nlen, nz / nlen

    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    rr = rr.ravel()
    cc = cc.ravel()
    K = len(rr)

    Z_k = z_map[rr, cc]
    Z_cam0 = Z_k + cam_dist
    X_k = (cc.astype(np.float64) - cx) * Z_cam0 / focal_length
    Y_k = (rr.astype(np.float64) - cy) * Z_cam0 / focal_length
    means_3d = np.stack([X_k, Y_k, Z_k], axis=1)

    # Anisotropic 3D covariance slightly elongated along local tangent plane
    covs_3d = np.zeros((K, 3, 3), dtype=np.float64)
    covs_3d[:, 0, 0] = cov_scale_3d
    covs_3d[:, 1, 1] = cov_scale_3d
    covs_3d[:, 2, 2] = 0.5 * cov_scale_3d

    opacities = np.full(K, float(base_opacity), dtype=np.float64)
    colors = img.reshape(K, C).copy()
    sh_dirs = np.stack([nx[rr, cc], ny[rr, cc], nz[rr, cc]], axis=1)

    renderer = GaussianSplatting3DRenderer(
        height=H,
        width=W,
        channels=C,
        means_3d=means_3d,
        covs_3d=covs_3d,
        opacities=opacities,
        colors=colors,
        sh_dirs=sh_dirs,
        focal_length=focal_length,
        cam_dist=cam_dist,
        frustum_radius_px=frustum_radius_px,
    )

    if calibrate_colors:
        # 2-step iterative Jacobi color calibration so front-to-back alpha-blending at alpha_0=0 matches `image`
        target_flat = img.reshape(K, C)
        for _ in range(3):
            pred_flat = renderer.evaluate_concrete(np.zeros(9, dtype=np.float64)).reshape(K, C)
            err = target_flat - pred_flat
            renderer.colors = np.clip(renderer.colors + 0.95 * err, 0.0, 2.0)

    return renderer
