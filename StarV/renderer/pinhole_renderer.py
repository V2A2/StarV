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
Pinhole Camera + 3D Surface Relief + Lambertian/Phong Abstract Renderer:
  Propagates a low-dimensional 3D physical ProbStar `PhysicalProbStar`
  (6-DoF SE(3) camera pose perturbations + 3D directional/ambient illumination)
  through:
    1. Exact SO(3) trigonometric rotation envelopes (roll, pitch, yaw) + translation
    2. Perspective division (1 / Z_c) and pinhole projection (u = f_x X_c / Z_c + c_x)
    3. Lambertian diffuse max(0, N_c . L) + Blinn-Phong specular (N_c . L)^2 reflectance
    4. Differentiable anti-aliased soft kernel rasterization onto (H, W, C) image space
  producing a `ProbSparseImageStar2DCSR` (or `ProbSparseImageStar2DCOO` / `ProbStar`)
  for end-to-end 3D-to-perception probabilistic verification.

Author: Hoang Dung Tran & StarV Team
"""

import time
import numpy as np

from StarV.renderer.primitives import (
    PhysicalProbStar,
    AffineEnvelopeTensor,
    sin_env,
    cos_env,
    square_env,
    reciprocal_env,
    exp_neg_env,
    gaussian_rbf_env,
    relu_env,
    clamp_env,
    multiply_env,
)


def compute_so3_rotation_concrete(pitch, yaw, roll):
    """
    Compute 3x3 rotation matrices R = R_z(roll) @ R_y(yaw) @ R_x(pitch)
    for 1D arrays pitch, yaw, roll of length B.
    Returns R of shape (B, 3, 3).
    """
    sp_, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    sr, cr = np.sin(roll), np.cos(roll)

    B = len(pitch)
    R = np.zeros((B, 3, 3), dtype=np.float64)
    R[:, 0, 0] = cr * cy
    R[:, 0, 1] = cr * sy * sp_ - sr * cp
    R[:, 0, 2] = cr * sy * cp + sr * sp_
    R[:, 1, 0] = sr * cy
    R[:, 1, 1] = sr * sy * sp_ + cr * cp
    R[:, 1, 2] = sr * sy * cp - cr * sp_
    R[:, 2, 0] = -sy
    R[:, 2, 1] = cy * sp_
    R[:, 2, 2] = cy * cp
    return R


def compute_so3_rotation_env(pitch_env, yaw_env, roll_env):
    """
    Compute sound AffineEnvelopeTensor entries R[i][j] (each of length 1)
    for the 3x3 rotation matrix R = R_z(roll) @ R_y(yaw) @ R_x(pitch).
    """
    sp_ = sin_env(pitch_env)
    cp = cos_env(pitch_env)
    sy = sin_env(yaw_env)
    cy = cos_env(yaw_env)
    sr = sin_env(roll_env)
    cr = cos_env(roll_env)

    cr_cy = multiply_env(cr, cy)
    cr_sy = multiply_env(cr, sy)
    sr_cy = multiply_env(sr, cy)
    sr_sy = multiply_env(sr, sy)

    R00 = cr_cy
    R01 = multiply_env(cr_sy, sp_).sub(multiply_env(sr, cp))
    R02 = multiply_env(cr_sy, cp).add(multiply_env(sr, sp_))

    R10 = sr_cy
    R11 = multiply_env(sr_sy, sp_).add(multiply_env(cr, cp))
    R12 = multiply_env(sr_sy, cp).sub(multiply_env(cr, sp_))

    R20 = sy.scale_shift(scale=-1.0, shift=0.0)
    R21 = multiply_env(cy, sp_)
    R22 = multiply_env(cy, cp)

    return [[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]


class PinholeCameraMeshRenderer(object):
    """
    Differentiable / Analytical Pinhole Camera + Lambertian/Phong Surface Renderer
    with exact ProbStar Abstract Rendering support.

    Physical Parameter Convention (up to 9 parameters in `PhysicalProbStar`):
      - Index 0: `d_pitch` (camera rotation around X axis, radians)
      - Index 1: `d_yaw`   (camera rotation around Y axis, radians)
      - Index 2: `d_roll`  (camera rotation around Z axis, radians)
      - Index 3: `d_tx`    (camera translation along X)
      - Index 4: `d_ty`    (camera translation along Y)
      - Index 5: `d_tz`    (camera translation along Z / depth)
      - Index 6: `d_lx`    (light direction X perturbation)
      - Index 7: `d_ly`    (light direction Y perturbation)
      - Index 8: `d_amb`   (ambient illumination intensity perturbation)
    If `phys.n_phys < 9`, any omitted trailing coordinates default to 0.
    """

    def __init__(
        self,
        height,
        width,
        channels,
        points_3d,
        normals_3d,
        albedos,
        weights=None,
        focal_length=None,
        cam_dist=3.0,
        splat_sigma=0.75,
        light_dir0=(0.2, -0.2, 0.9591663),
        ambient0=0.45,
        diffuse0=0.55,
        specular0=0.10,
        bg_color=0.0,
        frustum_radius_px=2,
    ):
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.shape = (self.height, self.width, self.channels)
        self.num_pixels = self.height * self.width * self.channels

        self.points_3d = np.asarray(points_3d, dtype=np.float64)
        self.K = self.points_3d.shape[0]
        assert self.points_3d.shape == (self.K, 3)

        normals = np.asarray(normals_3d, dtype=np.float64)
        assert normals.shape == (self.K, 3)
        norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        self.normals_3d = normals / norms

        self.albedos = np.asarray(albedos, dtype=np.float64).reshape(self.K, self.channels)
        self.weights = (
            np.ones(self.K, dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64).reshape(self.K)
        )

        self.focal_length = float(focal_length if focal_length is not None else max(self.height, self.width) * 1.5)
        self.cx = 0.5 * float(self.width - 1)
        self.cy = 0.5 * float(self.height - 1)
        self.cam_dist = float(cam_dist)
        self.splat_sigma = float(splat_sigma)
        self.beta_kernel = 1.0 / (2.0 * (self.splat_sigma ** 2))
        self.frustum_radius_px = int(frustum_radius_px)

        ld = np.asarray(light_dir0, dtype=np.float64).reshape(3)
        self.light_dir0 = ld / max(np.linalg.norm(ld), 1e-12)
        self.ambient0 = float(ambient0)
        self.diffuse0 = float(diffuse0)
        self.specular0 = float(specular0)
        if np.ndim(bg_color) == 0:
            self.bg_color = np.full(self.channels, float(bg_color), dtype=np.float64)
        else:
            self.bg_color = np.asarray(bg_color, dtype=np.float64).reshape(self.channels)

    def _unpack_phys_params_concrete(self, alpha_samples):
        """Extract 9 physical parameters of shape (B,) from (B, p) array."""
        alpha = np.asarray(alpha_samples, dtype=np.float64)
        if alpha.ndim == 1:
            alpha = alpha[None, :]
        B, p = alpha.shape
        full = np.zeros((B, 9), dtype=np.float64)
        m = min(p, 9)
        full[:, :m] = alpha[:, :m]
        return [full[:, i] for i in range(9)]

    def _unpack_phys_params_env(self, phys, mode='probstar'):
        """Extract 9 AffineEnvelopeTensors of length 1 from PhysicalProbStar `phys`."""
        envs = []
        for i in range(9):
            if i < phys.n_phys:
                envs.append(phys.coordinate_tensor(i, mode=mode))
            else:
                envs.append(AffineEnvelopeTensor.from_constant(phys, [0.0], mode=mode))
        return envs

    def evaluate_concrete(self, alpha_samples):
        """
        Concrete differentiable pinhole + Lambertian/Phong rendering for a batch
        of physical parameters `alpha_samples` of shape (B, p) or (p,).
        Returns rendered images of shape (B, H*W*C) or (H*W*C,) if 1D input.
        """
        single = np.asarray(alpha_samples).ndim == 1
        d_pitch, d_yaw, d_roll, d_tx, d_ty, d_tz, d_lx, d_ly, d_amb = (
            self._unpack_phys_params_concrete(alpha_samples)
        )
        B = len(d_pitch)
        R = compute_so3_rotation_concrete(d_pitch, d_yaw, d_roll)  # (B, 3, 3)

        # Rotate 3D points: P_c = R @ P^T + t  -> (B, K, 3)
        P_rot = np.einsum('bij,kj->bki', R, self.points_3d)
        X_c = P_rot[:, :, 0] + d_tx[:, None]
        Y_c = P_rot[:, :, 1] + d_ty[:, None]
        Z_c = np.maximum(P_rot[:, :, 2] + self.cam_dist + d_tz[:, None], 1e-3)

        inv_Z = 1.0 / Z_c
        u_proj = self.focal_length * (X_c * inv_Z) + self.cx  # (B, K)
        v_proj = self.focal_length * (Y_c * inv_Z) + self.cy  # (B, K)

        # Rotate surface normals: N_c = R @ N^T -> (B, K, 3)
        N_c = np.einsum('bij,kj->bki', R, self.normals_3d)
        Lx = self.light_dir0[0] + d_lx[:, None]
        Ly = self.light_dir0[1] + d_ly[:, None]
        Lz = np.full((B, 1), self.light_dir0[2], dtype=np.float64)

        dot_nl = N_c[:, :, 0] * Lx + N_c[:, :, 1] * Ly + N_c[:, :, 2] * Lz
        diff = np.maximum(dot_nl, 0.0)
        spec = np.square(diff)
        shade = (self.ambient0 + d_amb[:, None]) + self.diffuse0 * diff + self.specular0 * spec  # (B, K)

        # Nominal projection centers (at alpha=0) define the active local rasterization stencil
        u0 = self.focal_length * (self.points_3d[:, 0] / np.maximum(self.points_3d[:, 2] + self.cam_dist, 1e-3)) + self.cx
        v0 = self.focal_length * (self.points_3d[:, 1] / np.maximum(self.points_3d[:, 2] + self.cam_dist, 1e-3)) + self.cy
        u0_int = np.rint(u0).astype(np.int64)
        v0_int = np.rint(v0).astype(np.int64)

        H, W, C = self.height, self.width, self.channels
        img = np.broadcast_to(self.bg_color[None, None, None, :], (B, H, W, C)).copy()

        rad = self.frustum_radius_px
        for dh in range(-rad, rad + 1):
            rows = v0_int + dh
            for dw in range(-rad, rad + 1):
                cols = u0_int + dw
                valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
                if not np.any(valid):
                    continue
                k_idx = np.where(valid)[0]
                r_idx = rows[k_idx]
                c_idx = cols[k_idx]

                du = c_idx[None, :].astype(np.float64) - u_proj[:, k_idx]
                dv = r_idx[None, :].astype(np.float64) - v_proj[:, k_idx]
                q = self.beta_kernel * (np.square(du) + np.square(dv))
                kern = self.weights[k_idx][None, :] * np.exp(-q)
                contrib = (kern * shade[:, k_idx])[:, :, None] * self.albedos[k_idx][None, :, :]  # (B, len(k), C)
                for b_i in range(B):
                    np.add.at(img[b_i], (r_idx, c_idx), contrib[b_i])

        img_clipped = np.clip(img, 0.0, 1.0).reshape(B, self.num_pixels)
        return img_clipped[0] if single else img_clipped

    def render_envelope(self, phys, mode='probstar'):
        """
        Propagate `PhysicalProbStar` `phys` through the Pinhole + Lambertian/Phong
        renderer to obtain an `AffineEnvelopeTensor` of length `H * W * C`.
        """
        d_pitch, d_yaw, d_roll, d_tx, d_ty, d_tz, d_lx, d_ly, d_amb = (
            self._unpack_phys_params_env(phys, mode=mode)
        )
        R_env = compute_so3_rotation_env(d_pitch, d_yaw, d_roll)

        # Build 3x3 rotation row vectors as AffineEnvelopeTensors of length 3
        R_rows = [AffineEnvelopeTensor.concat(R_env[i]) for i in range(3)]

        # Rotate all K 3D points: X_c, Y_c, Z_c each of length K
        X_c = R_rows[0].linear(self.points_3d).add(d_tx)
        Y_c = R_rows[1].linear(self.points_3d).add(d_ty)
        Z_c = R_rows[2].linear(self.points_3d, b=np.full(self.K, self.cam_dist)).add(d_tz)

        inv_Z = reciprocal_env(Z_c, min_depth=1e-3)
        u_proj = multiply_env(X_c, inv_Z).scale_shift(scale=self.focal_length, shift=self.cx)
        v_proj = multiply_env(Y_c, inv_Z).scale_shift(scale=self.focal_length, shift=self.cy)

        # Rotate normals and compute Lambertian + Phong shading for all K primitives
        N_cx = R_rows[0].linear(self.normals_3d)
        N_cy = R_rows[1].linear(self.normals_3d)
        N_cz = R_rows[2].linear(self.normals_3d)

        Lx = d_lx.scale_shift(scale=1.0, shift=self.light_dir0[0])
        Ly = d_ly.scale_shift(scale=1.0, shift=self.light_dir0[1])

        dot_nl = (
            multiply_env(N_cx, Lx)
            .add(multiply_env(N_cy, Ly))
            .add(N_cz.scale_shift(scale=self.light_dir0[2], shift=0.0))
        )
        diff = relu_env(dot_nl)
        spec = square_env(diff)
        shade = (
            d_amb.scale_shift(scale=1.0, shift=self.ambient0)
            .add(diff.scale_shift(scale=self.diffuse0, shift=0.0))
            .add(spec.scale_shift(scale=self.specular0, shift=0.0))
        )

        # Rasterize across stencil shifts (dh, dw)
        u0 = self.focal_length * (self.points_3d[:, 0] / np.maximum(self.points_3d[:, 2] + self.cam_dist, 1e-3)) + self.cx
        v0 = self.focal_length * (self.points_3d[:, 1] / np.maximum(self.points_3d[:, 2] + self.cam_dist, 1e-3)) + self.cy
        u0_int = np.rint(u0).astype(np.int64)
        v0_int = np.rint(v0).astype(np.int64)

        H, W, C = self.height, self.width, self.channels
        num_px_spatial = H * W

        # Accumulate per-channel envelopes via sparse selection matrices
        ch_accum = [
            AffineEnvelopeTensor.from_constant(
                phys, np.full(num_px_spatial, self.bg_color[ch], dtype=np.float64), mode=mode
            )
            for ch in range(C)
        ]

        rad = self.frustum_radius_px
        for dh in range(-rad, rad + 1):
            rows = v0_int + dh
            for dw in range(-rad, rad + 1):
                cols = u0_int + dw
                valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
                if not np.any(valid):
                    continue
                k_idx = np.where(valid)[0]
                r_idx = rows[k_idx]
                c_idx = cols[k_idx]
                flat_px = r_idx * W + c_idx

                u_k = u_proj.slice(k_idx)
                v_k = v_proj.slice(k_idx)
                sqrt_beta = np.sqrt(self.beta_kernel)
                zu = u_k.scale_shift(scale=-sqrt_beta, shift=sqrt_beta * c_idx.astype(np.float64))
                zv = v_k.scale_shift(scale=-sqrt_beta, shift=sqrt_beta * r_idx.astype(np.float64))

                ku = gaussian_rbf_env(zu)
                kv = gaussian_rbf_env(zv)
                kern = multiply_env(ku, kv).scale_shift(scale=self.weights[k_idx], shift=0.0)
                ks = multiply_env(kern, shade.slice(k_idx))

                # Scatter-add from k_idx to flat_px
                import scipy.sparse as sp
                m_k = len(k_idx)
                for ch in range(C):
                    alb_ch = self.albedos[k_idx, ch]
                    if np.all( np.abs(alb_ch) <= 1e-15 ):
                        continue
                    M_scatter = sp.csr_array(
                        (alb_ch, (flat_px, np.arange(m_k, dtype=np.int64))),
                        shape=(num_px_spatial, m_k),
                    )
                    ch_contrib = ks.linear(M_scatter)
                    ch_accum[ch] = ch_accum[ch].add(ch_contrib)

        # Interleave channels into HWC flat order
        if C == 1:
            raw_img = ch_accum[0]
        else:
            # Interleave (H*W,) per channel into (H*W*C,)
            import scipy.sparse as sp
            interleave_parts = []
            for ch in range(C):
                rows_c = np.arange(num_px_spatial, dtype=np.int64) * C + ch
                cols_c = np.arange(num_px_spatial, dtype=np.int64)
                M_c = sp.csr_array(
                    (np.ones(num_px_spatial, dtype=np.float64), (rows_c, cols_c)),
                    shape=(self.num_pixels, num_px_spatial),
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


def lift_image_to_3d_relief_scene(
    image,
    cam_dist=3.0,
    depth_relief=0.25,
    splat_sigma=0.68,
    frustum_radius_px=1,
    intensity_threshold=0.0,
    calibrate_albedo=True,
):
    """
    Lift a 2D benchmark image `image` of shape (H, W) or (H, W, C) in [0, 1]
    into a calibrated 3D Pinhole + Lambertian/Phong surface relief scene such that
    rendering at the nominal camera pose (alpha_0 = 0) accurately reconstructs `image`.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        img = img[:, :, None]
    H, W, C = img.shape
    focal_length = float(max(H, W) * 1.5)
    cx = 0.5 * float(W - 1)
    cy = 0.5 * float(H - 1)

    # Luminance drives 3D relief height Z(h, w) and surface normals N(h, w)
    lum = np.mean(img, axis=2)
    z_map = -depth_relief * (lum - 0.5)  # Foreground slightly closer to camera

    # Finite-difference surface gradients for 3D normals pointing toward camera (+Z in view)
    dz_du = np.gradient(z_map, axis=1)
    dz_dv = np.gradient(z_map, axis=0)
    nx = -dz_du
    ny = -dz_dv
    nz = np.ones_like(z_map)
    norm_len = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx, ny, nz = nx / norm_len, ny / norm_len, nz / norm_len

    # Select active pixels (all pixels if intensity_threshold <= 0)
    mask = lum >= intensity_threshold
    rr, cc = np.where(mask)
    Z_k = z_map[rr, cc]
    Z_cam0 = Z_k + cam_dist
    X_k = (cc.astype(np.float64) - cx) * Z_cam0 / focal_length
    Y_k = (rr.astype(np.float64) - cy) * Z_cam0 / focal_length

    points_3d = np.stack([X_k, Y_k, Z_k], axis=1)
    normals_3d = np.stack([nx[rr, cc], ny[rr, cc], nz[rr, cc]], axis=1)
    albedos = img[rr, cc, :].copy()

    renderer = PinholeCameraMeshRenderer(
        height=H,
        width=W,
        channels=C,
        points_3d=points_3d,
        normals_3d=normals_3d,
        albedos=albedos,
        focal_length=focal_length,
        cam_dist=cam_dist,
        splat_sigma=splat_sigma,
        frustum_radius_px=frustum_radius_px,
    )

    if calibrate_albedo:
        # Calibrate weights/albedos so nominal render at alpha=0 matches target image
        unit_renderer = PinholeCameraMeshRenderer(
            height=H,
            width=W,
            channels=1,
            points_3d=points_3d,
            normals_3d=normals_3d,
            albedos=np.ones((len(rr), 1), dtype=np.float64),
            focal_length=focal_length,
            cam_dist=cam_dist,
            splat_sigma=splat_sigma,
            frustum_radius_px=frustum_radius_px,
            bg_color=0.0,
        )
        # Compute unclipped unit response at each pixel
        alpha0 = np.zeros(9, dtype=np.float64)
        R0 = compute_so3_rotation_concrete(alpha0[:1], alpha0[1:2], alpha0[2:3])
        N_c0 = np.einsum('bij,kj->bki', R0, renderer.normals_3d)[0]
        dot_nl0 = N_c0 @ renderer.light_dir0
        diff0 = np.maximum(dot_nl0, 0.0)
        shade0 = renderer.ambient0 + renderer.diffuse0 * diff0 + renderer.specular0 * (diff0 ** 2)

        # Kernel sum per pixel
        kern_sum = np.zeros((H, W), dtype=np.float64)
        rad = frustum_radius_px
        for dh in range(-rad, rad + 1):
            for dw in range(-rad, rad + 1):
                r_p = rr + dh
                c_p = cc + dw
                v_m = (r_p >= 0) & (r_p < H) & (c_p >= 0) & (c_p < W)
                q0 = renderer.beta_kernel * (dh ** 2 + dw ** 2)
                w0 = np.exp(-q0) * shade0[v_m]
                np.add.at(kern_sum, (r_p[v_m], c_p[v_m]), w0)

        denom = np.maximum(kern_sum[rr, cc], 1e-6)[:, None]
        renderer.albedos = np.clip(img[rr, cc, :] / denom, 0.0, 2.5)

    return renderer
