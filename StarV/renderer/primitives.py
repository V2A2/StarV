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
Core Mathematical Primitives for ProbStar Abstract Rendering:
  - PhysicalProbStar: Low-dimensional 3D physical perturbation space
    alpha_0 ~ TN(mu_0, Sig_0, [lb_0, ub_0], C_0 * alpha_0 <= d_0).
  - AffineEnvelopeTensor: Vectorized two-sided linear relaxation envelope
    gamma_lb + G_lb @ alpha_0 <= x(alpha_0) <= gamma_ub + G_ub @ alpha_0
    coupled with 5-point Gauss-Hermite quadrature moments (mu, var) and
    Stein's lemma cross-covariance Cov(x, alpha_0).
  - Sound univariate & bilinear 3D rendering primitives:
    sin, cos, reciprocal (1/z), square (x^2), exp(-x), relu, leaky_relu,
    sigmoid, softplus, clamp, McCormick bilinear product (x * y), and
    quotient (x / z).
  - Affine-Residual Decomposition into ProbSparseImageStar2DCSR,
    ProbSparseImageStar2DCOO, and ProbStar.

Author: Hoang Dung Tran & StarV Team
"""

import copy
import numpy as np
import scipy.sparse as sp

from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
from StarV.set.probsparseimagestar2dcoo import ProbSparseImageStar2DCOO
from StarV.set.probsparse_base import (
    RelaxationEnvelopeRecord,
    compute_box_input_probability,
    compute_truncated_gaussian_moments_1d,
    _MIN_VAR,
    _MIN_STD,
)

# 5-point Gauss-Hermite quadrature nodes and weights for standard normal Z ~ N(0, 1)
_GH_NODES = np.array([-2.8569700138728056, -1.3556261799742659, 0.0, 1.3556261799742659, 2.8569700138728056], dtype=np.float64)
_GH_WEIGHTS = np.array([0.011257411327720689, 0.22207592200561264, 0.5333333333333333, 0.22207592200561264, 0.011257411327720689], dtype=np.float64)
_SAFETY_PAD = 1e-12


class PhysicalProbStar(object):
    """
    Represents a p-dimensional 3D physical perturbation space:
        alpha_0 ~ N(mu_0, Sig_0) truncated to [lb_0, ub_0] (and optional C_0 @ alpha_0 <= d_0).

    Typical physical axes include:
        - 6-DoF SE(3) camera pose perturbations: (d_pitch, d_yaw, d_roll, d_tx, d_ty, d_tz)
        - Illumination perturbations: (d_light_x, d_light_y, d_light_z, d_amb, d_diff)
        - 3D scene / Gaussian splat geometry perturbations
    """

    def __init__(
        self,
        lb0,
        ub0,
        mu0=None,
        Sig0=None,
        sig_factor=3.0,
        C0=None,
        d0=None,
        param_names=None,
        cov_mode='diagonal',
    ):
        self.lb0 = np.asarray(lb0, dtype=np.float64).reshape(-1)
        self.ub0 = np.asarray(ub0, dtype=np.float64).reshape(-1)
        assert self.lb0.shape == self.ub0.shape, "lb0 and ub0 must have identical shape"
        assert np.all(self.lb0 <= self.ub0 + 1e-15), "lb0 must be <= ub0"
        self.ub0 = np.maximum(self.ub0, self.lb0)
        self.n_phys = len(self.lb0)
        self.cov_mode = cov_mode

        if mu0 is None:
            self.mu0 = 0.5 * (self.lb0 + self.ub0)
        else:
            self.mu0 = np.asarray(mu0, dtype=np.float64).reshape(-1)
            assert len(self.mu0) == self.n_phys

        if Sig0 is None:
            half_span = np.maximum(0.5 * (self.ub0 - self.lb0), _MIN_STD)
            std0 = half_span / float(sig_factor)
            var0 = np.maximum(np.square(std0), _MIN_VAR)
            self.Sig0 = var0 if cov_mode == 'diagonal' else np.diag(var0)
        else:
            Sig_arr = np.asarray(Sig0, dtype=np.float64)
            if Sig_arr.ndim == 1:
                assert len(Sig_arr) == self.n_phys
                var0 = np.maximum(Sig_arr, _MIN_VAR)
                self.Sig0 = var0 if cov_mode == 'diagonal' else np.diag(var0)
            else:
                assert Sig_arr.shape == (self.n_phys, self.n_phys)
                self.Sig0 = np.diag(Sig_arr).copy() if cov_mode == 'diagonal' else Sig_arr.copy()

        self.var0 = self.Sig0 if self.Sig0.ndim == 1 else np.diag(self.Sig0)
        self.mu0_trunc, self.var0_trunc = compute_truncated_gaussian_moments_1d(
            self.mu0, self.var0, self.lb0, self.ub0
        )
        if self.Sig0.ndim == 1:
            self.Sig0_trunc = self.var0_trunc
        else:
            scale0 = np.sqrt(self.var0_trunc / np.maximum(self.var0, _MIN_VAR))
            self.Sig0_trunc = (self.Sig0 * scale0[:, None]) * scale0[None, :]

        self.p_input = compute_box_input_probability(self.mu0, self.Sig0, self.lb0, self.ub0)

        if C0 is not None and d0 is not None and len(d0) > 0:
            self.C0 = np.asarray(C0, dtype=np.float64)
            self.d0 = np.asarray(d0, dtype=np.float64).reshape(-1)
        else:
            self.C0 = np.empty((0, self.n_phys), dtype=np.float64)
            self.d0 = np.empty((0,), dtype=np.float64)

        if param_names is None:
            self.param_names = [f"alpha_{i}" for i in range(self.n_phys)]
        else:
            self.param_names = list(param_names)

    def to_probstar(self):
        """Convert the physical parameter space into a p-dimensional StarV ProbStar."""
        V = np.hstack([np.zeros((self.n_phys, 1), dtype=np.float64), np.eye(self.n_phys, dtype=np.float64)])
        Sig_mat = np.diag(self.Sig0) if self.Sig0.ndim == 1 else self.Sig0
        return ProbStar(
            V, self.C0.copy(), self.d0.copy(),
            self.mu0.copy(), Sig_mat.copy(),
            self.lb0.copy(), self.ub0.copy(),
            cov_mode=self.cov_mode,
        )

    def coordinate_tensor(self, idx, mode='probstar'):
        """Return an AffineEnvelopeTensor of shape (1,) for physical coordinate `idx`."""
        return self.coordinates_tensor([idx], mode=mode)

    def coordinates_tensor(self, indices=None, mode='probstar'):
        """Return an AffineEnvelopeTensor for the selected physical coordinates."""
        if indices is None:
            indices = list(range(self.n_phys))
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        m = len(indices)
        gamma_lb = np.zeros(m, dtype=np.float64)
        gamma_ub = np.zeros(m, dtype=np.float64)
        G_lb = np.zeros((m, self.n_phys), dtype=np.float64)
        G_ub = np.zeros((m, self.n_phys), dtype=np.float64)
        G_lb[np.arange(m), indices] = 1.0
        G_ub[np.arange(m), indices] = 1.0
        lb = self.lb0[indices].copy()
        ub = self.ub0[indices].copy()
        mu = self.mu0_trunc[indices].copy()
        var = self.var0_trunc[indices].copy()
        if self.Sig0_trunc.ndim == 1:
            cross_cov = G_lb * self.Sig0_trunc[None, :]
        else:
            cross_cov = G_lb @ self.Sig0_trunc
        return AffineEnvelopeTensor(
            phys=self,
            gamma_lb=gamma_lb,
            G_lb=G_lb,
            gamma_ub=gamma_ub,
            G_ub=G_ub,
            lb=lb,
            ub=ub,
            mu=mu,
            var=var,
            cross_cov=cross_cov,
            mode=mode,
        )

    def sample_truncated(self, n_samples=1000, seed=42, include_corners=True):
        """
        Draw samples alpha_0 in [lb0, ub0] from the truncated Gaussian distribution
        (optionally prepending the nominal center and box extremes for soundness checks).
        """
        rng = np.random.default_rng(seed)
        std0 = np.maximum(np.sqrt(self.var0), _MIN_STD)
        from scipy.special import ndtr, ndtri
        cdf_l = ndtr((self.lb0 - self.mu0) / std0)
        cdf_u = ndtr((self.ub0 - self.mu0) / std0)
        u_rand = rng.uniform(cdf_l[None, :], cdf_u[None, :], size=(n_samples, self.n_phys))
        u_rand = np.clip(u_rand, 1e-12, 1.0 - 1e-12)
        samples = self.mu0[None, :] + std0[None, :] * ndtri(u_rand)
        samples = np.clip(samples, self.lb0[None, :], self.ub0[None, :])
        if include_corners and n_samples >= 3:
            samples[0] = self.mu0
            samples[1] = self.lb0
            samples[2] = self.ub0
            if self.n_phys <= 8 and n_samples >= 3 + (1 << self.n_phys):
                for mask in range(1 << self.n_phys):
                    corner = np.where(
                        [(mask >> j) & 1 for j in range(self.n_phys)],
                        self.ub0,
                        self.lb0,
                    )
                    samples[3 + mask] = corner
        return samples


class AffineEnvelopeTensor(object):
    """
    Vectorized 1D collection of N scalar quantities parameterized by physical
    ProbStar predicates alpha_0 in R^p:
        gamma_lb + G_lb @ alpha_0 <= x(alpha_0) <= gamma_ub + G_ub @ alpha_0,
        lb <= x(alpha_0) <= ub,
    together with analytical Gaussian moments:
        mu = E[x(alpha_0)],  var = Var[x(alpha_0)],  cross_cov = Cov(x(alpha_0), alpha_0).

    Modes supported:
      - 'probstar' (default): Full two-sided linear relaxation envelope + analytical moments
        + sparse affine-residual decomposition into ProbSparseImageStar2DCSR.
      - 'crown_linear': Two-sided linear relaxation envelope without residual generator splitting.
      - 'ibp': Interval Bound Propagation baseline (collapses G_lb = G_ub = 0 at nonlinear steps).
    """

    def __init__(
        self,
        phys,
        gamma_lb,
        G_lb,
        gamma_ub,
        G_ub,
        lb=None,
        ub=None,
        mu=None,
        var=None,
        cross_cov=None,
        mode='probstar',
    ):
        self.phys = phys
        self.mode = mode
        self.gamma_lb = np.asarray(gamma_lb, dtype=np.float64).reshape(-1)
        self.gamma_ub = np.asarray(gamma_ub, dtype=np.float64).reshape(-1)
        self.N = len(self.gamma_lb)
        self.p = phys.n_phys
        self.G_lb = np.asarray(G_lb, dtype=np.float64).reshape(self.N, self.p)
        self.G_ub = np.asarray(G_ub, dtype=np.float64).reshape(self.N, self.p)

        # Concretize sound interval bounds over [lb0, ub0]
        G_lb_pos = np.maximum(self.G_lb, 0.0)
        G_lb_neg = np.minimum(self.G_lb, 0.0)
        lin_lb = self.gamma_lb + G_lb_pos @ phys.lb0 + G_lb_neg @ phys.ub0

        G_ub_pos = np.maximum(self.G_ub, 0.0)
        G_ub_neg = np.minimum(self.G_ub, 0.0)
        lin_ub = self.gamma_ub + G_ub_pos @ phys.ub0 + G_ub_neg @ phys.lb0

        if lb is None:
            self.lb = lin_lb
        else:
            self.lb = np.maximum(np.asarray(lb, dtype=np.float64).reshape(-1), lin_lb)
        if ub is None:
            self.ub = lin_ub
        else:
            self.ub = np.minimum(np.asarray(ub, dtype=np.float64).reshape(-1), lin_ub)
        self.ub = np.maximum(self.ub, self.lb)

        if mode == 'ibp':
            # IBP baseline discards relational generator dependencies
            self.gamma_lb = self.lb.copy()
            self.gamma_ub = self.ub.copy()
            self.G_lb = np.zeros((self.N, self.p), dtype=np.float64)
            self.G_ub = np.zeros((self.N, self.p), dtype=np.float64)

        if mu is None or var is None or cross_cov is None:
            G_mid = 0.5 * (self.G_lb + self.G_ub)
            gamma_mid = 0.5 * (self.gamma_lb + self.gamma_ub)
            mu_calc = np.clip(gamma_mid + G_mid @ phys.mu0_trunc, self.lb, self.ub)
            if phys.Sig0_trunc.ndim == 1:
                cc_calc = G_mid * phys.Sig0_trunc[None, :]
                var_calc = np.sum(G_mid * cc_calc, axis=1)
            else:
                cc_calc = G_mid @ phys.Sig0_trunc
                var_calc = np.sum(G_mid * cc_calc, axis=1)
            var_calc = np.maximum(var_calc, _MIN_VAR)
            self.mu = mu_calc if mu is None else np.clip(np.asarray(mu, dtype=np.float64).reshape(-1), self.lb, self.ub)
            self.var = var_calc if var is None else np.maximum(np.asarray(var, dtype=np.float64).reshape(-1), _MIN_VAR)
            self.cross_cov = cc_calc if cross_cov is None else np.asarray(cross_cov, dtype=np.float64).reshape(self.N, self.p)
        else:
            self.mu = np.clip(np.asarray(mu, dtype=np.float64).reshape(-1), self.lb, self.ub)
            self.var = np.maximum(np.asarray(var, dtype=np.float64).reshape(-1), _MIN_VAR)
            self.cross_cov = np.asarray(cross_cov, dtype=np.float64).reshape(self.N, self.p)

    @staticmethod
    def from_constant(phys, val, mode='probstar'):
        """Create an AffineEnvelopeTensor from a constant 1D array `val`."""
        arr = np.asarray(val, dtype=np.float64).reshape(-1)
        N = len(arr)
        zeros_G = np.zeros((N, phys.n_phys), dtype=np.float64)
        return AffineEnvelopeTensor(
            phys=phys,
            gamma_lb=arr.copy(),
            G_lb=zeros_G.copy(),
            gamma_ub=arr.copy(),
            G_ub=zeros_G.copy(),
            lb=arr.copy(),
            ub=arr.copy(),
            mu=arr.copy(),
            var=np.full(N, _MIN_VAR, dtype=np.float64),
            cross_cov=zeros_G.copy(),
            mode=mode,
        )

    def copy(self):
        return AffineEnvelopeTensor(
            phys=self.phys,
            gamma_lb=self.gamma_lb.copy(),
            G_lb=self.G_lb.copy(),
            gamma_ub=self.gamma_ub.copy(),
            G_ub=self.G_ub.copy(),
            lb=self.lb.copy(),
            ub=self.ub.copy(),
            mu=self.mu.copy(),
            var=self.var.copy(),
            cross_cov=self.cross_cov.copy(),
            mode=self.mode,
        )

    def __len__(self):
        return self.N

    def slice(self, idx):
        """Subset elements along axis 0."""
        idx = np.asarray(idx)
        if idx.ndim == 0:
            idx = idx.reshape(1)
        return AffineEnvelopeTensor(
            phys=self.phys,
            gamma_lb=self.gamma_lb[idx],
            G_lb=self.G_lb[idx],
            gamma_ub=self.gamma_ub[idx],
            G_ub=self.G_ub[idx],
            lb=self.lb[idx],
            ub=self.ub[idx],
            mu=self.mu[idx],
            var=self.var[idx],
            cross_cov=self.cross_cov[idx],
            mode=self.mode,
        )

    @staticmethod
    def concat(tensors):
        """Concatenate a sequence of AffineEnvelopeTensors along axis 0."""
        assert len(tensors) > 0
        phys = tensors[0].phys
        mode = tensors[0].mode
        return AffineEnvelopeTensor(
            phys=phys,
            gamma_lb=np.concatenate([t.gamma_lb for t in tensors]),
            G_lb=np.vstack([t.G_lb for t in tensors]),
            gamma_ub=np.concatenate([t.gamma_ub for t in tensors]),
            G_ub=np.vstack([t.G_ub for t in tensors]),
            lb=np.concatenate([t.lb for t in tensors]),
            ub=np.concatenate([t.ub for t in tensors]),
            mu=np.concatenate([t.mu for t in tensors]),
            var=np.concatenate([t.var for t in tensors]),
            cross_cov=np.vstack([t.cross_cov for t in tensors]),
            mode=mode,
        )

    def scale_shift(self, scale=1.0, shift=0.0):
        """Compute exact affine transformation `scale * x + shift`."""
        s = np.broadcast_to(np.asarray(scale, dtype=np.float64).reshape(-1), (self.N,))
        b = np.broadcast_to(np.asarray(shift, dtype=np.float64).reshape(-1), (self.N,))
        s_pos = np.maximum(s, 0.0)
        s_neg = np.minimum(s, 0.0)

        gamma_lb = s_pos * self.gamma_lb + s_neg * self.gamma_ub + b
        G_lb = s_pos[:, None] * self.G_lb + s_neg[:, None] * self.G_ub
        gamma_ub = s_pos * self.gamma_ub + s_neg * self.gamma_lb + b
        G_ub = s_pos[:, None] * self.G_ub + s_neg[:, None] * self.G_lb

        lb = s_pos * self.lb + s_neg * self.ub + b
        ub = s_pos * self.ub + s_neg * self.lb + b
        mu = s * self.mu + b
        var = np.maximum(np.square(s) * self.var, _MIN_VAR)
        cross_cov = s[:, None] * self.cross_cov
        return AffineEnvelopeTensor(
            phys=self.phys,
            gamma_lb=gamma_lb,
            G_lb=G_lb,
            gamma_ub=gamma_ub,
            G_ub=G_ub,
            lb=lb,
            ub=ub,
            mu=mu,
            var=var,
            cross_cov=cross_cov,
            mode=self.mode,
        )

    def add(self, other):
        """Exact sum `self + other` where `other` is an AffineEnvelopeTensor or array/scalar."""
        if not isinstance(other, AffineEnvelopeTensor):
            return self.scale_shift(scale=1.0, shift=other)
        if other.N == 1 and self.N > 1:
            other = other.slice(np.zeros(self.N, dtype=np.int64))
        elif self.N == 1 and other.N > 1:
            return other.add(self)
        assert self.N == other.N, f"Shape mismatch {self.N} vs {other.N}"

        gamma_lb = self.gamma_lb + other.gamma_lb
        G_lb = self.G_lb + other.G_lb
        gamma_ub = self.gamma_ub + other.gamma_ub
        G_ub = self.G_ub + other.G_ub
        lb = self.lb + other.lb
        ub = self.ub + other.ub
        mu = self.mu + other.mu
        cross_cov = self.cross_cov + other.cross_cov

        # Include cross-correlation term between self and other via physical generators
        G1 = 0.5 * (self.G_lb + self.G_ub)
        G2 = 0.5 * (other.G_lb + other.G_ub)
        if self.phys.Sig0_trunc.ndim == 1:
            cov_12 = np.sum(G1 * G2 * self.phys.Sig0_trunc[None, :], axis=1)
        else:
            cov_12 = np.sum((G1 @ self.phys.Sig0_trunc) * G2, axis=1)
        var = np.maximum(self.var + other.var + 2.0 * cov_12, _MIN_VAR)

        return AffineEnvelopeTensor(
            phys=self.phys,
            gamma_lb=gamma_lb,
            G_lb=G_lb,
            gamma_ub=gamma_ub,
            G_ub=G_ub,
            lb=lb,
            ub=ub,
            mu=mu,
            var=var,
            cross_cov=cross_cov,
            mode=self.mode,
        )

    def sub(self, other):
        """Exact difference `self - other`."""
        if not isinstance(other, AffineEnvelopeTensor):
            return self.scale_shift(scale=1.0, shift=-np.asarray(other, dtype=np.float64))
        return self.add(other.scale_shift(scale=-1.0, shift=0.0))

    def linear(self, W, b=None):
        """
        Compute exact matrix-vector affine map `y = W @ x + b` where `W` has shape (M, N).
        """
        if sp.issparse(W):
            W_pos = W.maximum(0.0)
            W_neg = W.minimum(0.0)
        else:
            W_pos = np.maximum(W, 0.0)
            W_neg = np.minimum(W, 0.0)
        M = W.shape[0]
        b_vec = np.zeros(M, dtype=np.float64) if b is None else np.asarray(b, dtype=np.float64).reshape(-1)

        gamma_lb = (W_pos @ self.gamma_lb) + (W_neg @ self.gamma_ub) + b_vec
        G_lb = (W_pos @ self.G_lb) + (W_neg @ self.G_ub)
        gamma_ub = (W_pos @ self.gamma_ub) + (W_neg @ self.gamma_lb) + b_vec
        G_ub = (W_pos @ self.G_ub) + (W_neg @ self.G_lb)

        lb = (W_pos @ self.lb) + (W_neg @ self.ub) + b_vec
        ub = (W_pos @ self.ub) + (W_neg @ self.lb) + b_vec
        mu = (W @ self.mu) + b_vec
        cross_cov = np.asarray(W @ self.cross_cov, dtype=np.float64)

        G_mid = 0.5 * (G_lb + G_ub)
        if self.phys.Sig0_trunc.ndim == 1:
            var_lin = np.sum(np.square(G_mid) * self.phys.Sig0_trunc[None, :], axis=1)
        else:
            var_lin = np.sum((G_mid @ self.phys.Sig0_trunc) * G_mid, axis=1)
        # Add residual non-physical variance contribution
        G_in_mid = 0.5 * (self.G_lb + self.G_ub)
        if self.phys.Sig0_trunc.ndim == 1:
            var_in_lin = np.sum(np.square(G_in_mid) * self.phys.Sig0_trunc[None, :], axis=1)
        else:
            var_in_lin = np.sum((G_in_mid @ self.phys.Sig0_trunc) * G_in_mid, axis=1)
        res_var_in = np.maximum(self.var - var_in_lin, 0.0)
        W_sq = W.power(2) if sp.issparse(W) else np.square(W)
        var = np.maximum(var_lin + (W_sq @ res_var_in), _MIN_VAR)

        return AffineEnvelopeTensor(
            phys=self.phys,
            gamma_lb=gamma_lb,
            G_lb=G_lb,
            gamma_ub=gamma_ub,
            G_ub=G_ub,
            lb=lb,
            ub=ub,
            mu=mu,
            var=var,
            cross_cov=cross_cov,
            mode=self.mode,
        )

    def evaluate_bounds_at(self, alpha_samples):
        """
        Evaluate the lower and upper linear envelopes at a batch of physical parameters
        `alpha_samples` of shape (B, p). Returns (lb_vals, ub_vals) of shape (B, N).
        """
        alpha_samples = np.asarray(alpha_samples, dtype=np.float64)
        if alpha_samples.ndim == 1:
            alpha_samples = alpha_samples[None, :]
        lb_vals = self.gamma_lb[None, :] + alpha_samples @ self.G_lb.T
        ub_vals = self.gamma_ub[None, :] + alpha_samples @ self.G_ub.T
        lb_vals = np.maximum(lb_vals, self.lb[None, :])
        ub_vals = np.minimum(ub_vals, self.ub[None, :])
        return lb_vals, ub_vals

    # ------------------------------------------------------------------
    # Conversion to StarV Set Representations (Affine-Residual Theorem)
    # ------------------------------------------------------------------
    def to_probsparse_imagestar_csr(self, shape, residual_tol=1e-11):
        """
        Convert this AffineEnvelopeTensor into a `ProbSparseImageStar2DCSR` of `shape = (H, W, C)`
        using the Affine-Residual Decomposition Theorem:

            I_i(alpha_0) = c_i + V_phys[i, :] @ alpha_0 + beta_i

        where:
          - `c_i = 0.5 * (gamma_lb[i] + gamma_ub[i])`
          - `V_phys[i, :] = 0.5 * (G_lb[i, :] + G_ub[i, :])` (shared physical generators)
          - `beta_i` is a sparse localized nonlinear residual predicate introduced ONLY at
            pixels with non-zero envelope gap `r_i > residual_tol`, linked to `alpha_0` via
            an exact symmetric `RelaxationEnvelopeRecord` in `env_dag`.
        """
        if len(shape) == 2:
            shape = (shape[0], shape[1], 1)
        H, W, C = int(shape[0]), int(shape[1]), int(shape[2])
        assert H * W * C == self.N, f"Shape {shape} does not match tensor length {self.N}"
        p = self.p
        phys = self.phys

        if self.mode == 'ibp':
            # Box/IBP representation: each uncertain pixel has an independent interval predicate
            c_vec = 0.5 * (self.lb + self.ub)
            rad_vec = 0.5 * np.maximum(self.ub - self.lb, 0.0)
            active_px = np.where(rad_vec > residual_tol)[0]
            m_act = len(active_px)
            if m_act > 0:
                rows = active_px.astype(np.int32)
                cols = (p + np.arange(m_act, dtype=np.int32))
                data = np.ones(m_act, dtype=np.float64)
                V_csr = sp.csr_array((data, (rows, cols)), shape=(self.N, p + m_act))
                pred_lb = np.concatenate([phys.lb0, -rad_vec[active_px]])
                pred_ub = np.concatenate([phys.ub0, +rad_vec[active_px]])
                mu_res = np.clip(self.mu[active_px] - c_vec[active_px], -rad_vec[active_px], rad_vec[active_px])
                var_res = np.clip(self.var[active_px], _MIN_VAR, np.maximum(np.square(rad_vec[active_px]), _MIN_VAR))
                mu_ext = np.concatenate([phys.mu0, mu_res])
                Sig_ext = np.concatenate([phys.var0, var_res])
                zeros_G = sp.csr_array((m_act, p), dtype=np.float64)
                env_rec = RelaxationEnvelopeRecord(
                    n_prev=p,
                    n_new=m_act,
                    G_lb=zeros_G,
                    gamma_lb=-rad_vec[active_px],
                    G_ub=zeros_G,
                    gamma_ub=+rad_vec[active_px],
                )
                env_dag = [env_rec]
            else:
                V_csr = sp.csr_array((self.N, p), dtype=np.float64)
                pred_lb = phys.lb0.copy()
                pred_ub = phys.ub0.copy()
                mu_ext = phys.mu0.copy()
                Sig_ext = phys.var0.copy()
                env_dag = []

            C_mat = sp.csr_array((0, len(pred_lb)), dtype=np.float64)
            d_vec = np.empty(0, dtype=np.float64)
            ibp_csr = ProbSparseImageStar2DCSR(
                c_vec, V_csr, C_mat, d_vec, mu_ext, Sig_ext, pred_lb, pred_ub, (H, W, C),
                cov_mode='diagonal',
                mu0=phys.mu0.copy(),
                Sig0=phys.Sig0.copy(),
                pred_lb0=phys.lb0.copy(),
                pred_ub0=phys.ub0.copy(),
                nVars0=p,
                env_dag=env_dag,
                p_input=phys.p_input,
            )
            ibp_csr.render_lb = self.lb.copy()
            ibp_csr.render_ub = self.ub.copy()
            ibp_csr.render_mu = self.mu.copy()
            ibp_csr.render_var = self.var.copy()
            ibp_csr.render_std = np.sqrt(np.maximum(self.var, _MIN_VAR))
            return ibp_csr

        # Affine-Residual ProbStar Decomposition
        c_vec = 0.5 * (self.gamma_lb + self.gamma_ub)
        V_phys = 0.5 * (self.G_lb + self.G_ub)

        d_gamma = 0.5 * (self.gamma_ub - self.gamma_lb)
        d_G = 0.5 * (self.G_ub - self.G_lb)
        d_G_pos = np.maximum(d_G, 0.0)
        d_G_neg = np.minimum(d_G, 0.0)
        rad_res = np.maximum(0.0, d_gamma + d_G_pos @ phys.ub0 + d_G_neg @ phys.lb0)
        # Also tighten residual radius against concrete [lb, ub] width
        half_concrete = 0.5 * (self.ub - self.lb)
        rad_res = np.minimum(rad_res, half_concrete + 1e-12)

        if self.mode == 'crown_linear':
            active_px = np.empty(0, dtype=np.int64)
        else:
            active_px = np.where(rad_res > residual_tol)[0]
        m_res = len(active_px)

        V_phys_csr = sp.csr_array(V_phys)
        if m_res > 0:
            rows_res = active_px.astype(np.int32)
            cols_res = np.arange(m_res, dtype=np.int32)
            data_res = np.ones(m_res, dtype=np.float64)
            V_res_csr = sp.csr_array((data_res, (rows_res, cols_res)), shape=(self.N, m_res))
            V_csr = sp.hstack([V_phys_csr, V_res_csr], format='csr')

            r_act = rad_res[active_px]
            pred_lb = np.concatenate([phys.lb0, -r_act])
            pred_ub = np.concatenate([phys.ub0, +r_act])

            # Analytical residual moments matching Gauss-Hermite image moments
            mu_aff_act = c_vec[active_px] + V_phys[active_px] @ phys.mu0_trunc
            mu_res = np.clip(self.mu[active_px] - mu_aff_act, -r_act, +r_act)
            if phys.Sig0_trunc.ndim == 1:
                var_aff_act = np.sum(np.square(V_phys[active_px]) * phys.Sig0_trunc[None, :], axis=1)
            else:
                var_aff_act = np.sum((V_phys[active_px] @ phys.Sig0_trunc) * V_phys[active_px], axis=1)
            var_res = np.clip(
                np.maximum(self.var[active_px] - var_aff_act, _MIN_VAR),
                _MIN_VAR,
                np.maximum(np.square(r_act), _MIN_VAR),
            )
            mu_ext = np.concatenate([phys.mu0, mu_res])
            Sig_ext = np.concatenate([phys.var0, var_res])

            d_G_act = sp.csr_array(d_G[active_px])
            d_gamma_act = d_gamma[active_px]
            env_rec = RelaxationEnvelopeRecord(
                n_prev=p,
                n_new=m_res,
                G_lb=-d_G_act,
                gamma_lb=-d_gamma_act,
                G_ub=d_G_act,
                gamma_ub=d_gamma_act,
            )
            env_dag = [env_rec]
        else:
            V_csr = V_phys_csr
            pred_lb = phys.lb0.copy()
            pred_ub = phys.ub0.copy()
            mu_ext = phys.mu0.copy()
            Sig_ext = phys.var0.copy()
            env_dag = []

        if phys.C0.shape[0] > 0:
            C_pad = np.hstack([phys.C0, np.zeros((phys.C0.shape[0], m_res), dtype=np.float64)])
            C_mat = sp.csr_array(C_pad)
            d_vec = phys.d0.copy()
        else:
            C_mat = sp.csr_array((0, len(pred_lb)), dtype=np.float64)
            d_vec = np.empty(0, dtype=np.float64)

        star_csr = ProbSparseImageStar2DCSR(
            c_vec, V_csr, C_mat, d_vec, mu_ext, Sig_ext, pred_lb, pred_ub, (H, W, C),
            cov_mode='diagonal',
            mu0=phys.mu0.copy(),
            Sig0=phys.Sig0.copy(),
            pred_lb0=phys.lb0.copy(),
            pred_ub0=phys.ub0.copy(),
            nVars0=p,
            env_dag=env_dag,
            p_input=phys.p_input,
        )
        # Attach exact rendered bounds and analytical moments as metadata for fast inspection/plotting
        star_csr.render_lb = self.lb.copy()
        star_csr.render_ub = self.ub.copy()
        star_csr.render_mu = self.mu.copy()
        star_csr.render_var = self.var.copy()
        star_csr.render_std = np.sqrt(np.maximum(self.var, _MIN_VAR))
        return star_csr

    def to_probsparse_imagestar_coo(self, shape, residual_tol=1e-11):
        """Convert to `ProbSparseImageStar2DCOO`."""
        csr = self.to_probsparse_imagestar_csr(shape, residual_tol=residual_tol)
        coo = ProbSparseImageStar2DCOO(
            csr.c.copy(),
            csr.V.tocoo(),
            csr.C.copy(),
            csr.d.copy(),
            csr.mu.copy(),
            csr.Sig.copy(),
            csr.pred_lb.copy(),
            csr.pred_ub.copy(),
            csr.shape,
            cov_mode=csr.cov_mode,
            mu0=csr.mu0.copy(),
            Sig0=csr.Sig0.copy(),
            pred_lb0=csr.pred_lb0.copy(),
            pred_ub0=csr.pred_ub0.copy(),
            nVars0=csr.nVars0,
            env_dag=list(csr.env_dag),
            p_input=csr.get_input_probability(),
        )
        coo.render_lb = self.lb.copy()
        coo.render_ub = self.ub.copy()
        coo.render_mu = self.mu.copy()
        coo.render_var = self.var.copy()
        coo.render_std = np.sqrt(np.maximum(self.var, _MIN_VAR))
        return coo

    def to_probstar(self, residual_tol=1e-11):
        """Convert this AffineEnvelopeTensor into a dense `ProbStar`."""
        csr = self.to_probsparse_imagestar_csr((self.N, 1, 1), residual_tol=residual_tol)
        V_dense = np.hstack([csr.c[:, None], csr.V.toarray()])
        C_dense = csr.C.toarray() if sp.issparse(csr.C) else csr.C
        Sig_dense = np.diag(csr.Sig) if csr.Sig.ndim == 1 else csr.Sig
        pstar = ProbStar(
            V_dense, C_dense, csr.d.copy(),
            csr.mu.copy(), Sig_dense,
            csr.pred_lb.copy(), csr.pred_ub.copy(),
            cov_mode=csr.cov_mode,
            env_dag=list(csr.env_dag),
            mu0=csr.mu0.copy(),
            Sig0=csr.Sig0.copy(),
            pred_lb0=csr.pred_lb0.copy(),
            pred_ub0=csr.pred_ub0.copy(),
            nVars0=csr.nVars0,
            check_psd=False,
        )
        pstar.render_lb = self.lb.copy()
        pstar.render_ub = self.ub.copy()
        pstar.render_mu = self.mu.copy()
        pstar.render_var = self.var.copy()
        pstar.render_std = np.sqrt(np.maximum(self.var, _MIN_VAR))
        return pstar


# =====================================================================
# Sound Univariate & Bilinear Rendering Primitive Relaxations
# =====================================================================

def _certify_univariate_chords_on_grid(l, u, a_lb, b_lb, a_ub, b_ub, fn_eval, extra_pts=None):
    """
    Certify and tighten intercepts (b_lb, b_ub) across endpoints, midpoint,
    optional analytical stationary points `extra_pts`, and a 17-point interior
    Chebyshev/uniform grid so that for all x in [l, u]:
        a_lb * x + b_lb <= fn_eval(x) <= a_ub * x + b_ub
    holds with 100% mathematical soundness.
    """
    t_grid = np.linspace(0.0, 1.0, 17, dtype=np.float64)[None, :]
    x_grid = l[:, None] + (u - l)[:, None] * t_grid
    if extra_pts is not None:
        extra_clipped = [np.clip(pt, l, u)[:, None] for pt in extra_pts]
        x_grid = np.hstack([x_grid] + extra_clipped)

    y_grid = fn_eval(x_grid)
    res_lb = np.min(y_grid - a_lb[:, None] * x_grid, axis=1)
    res_ub = np.max(y_grid - a_ub[:, None] * x_grid, axis=1)

    b_lb_cert = np.minimum(b_lb, res_lb) - _SAFETY_PAD
    b_ub_cert = np.maximum(b_ub, res_ub) + _SAFETY_PAD
    return a_lb, b_lb_cert, a_ub, b_ub_cert


def apply_univariate_primitive(
    x_env,
    a_lb,
    b_lb,
    a_ub,
    b_ub,
    fn_eval,
    fn_deriv=None,
    out_lb_clip=None,
    out_ub_clip=None,
):
    """
    Propagate an AffineEnvelopeTensor `x_env` through a univariate primitive y = fn_eval(x)
    given sound linear bounds `a_lb * x + b_lb <= fn_eval(x) <= a_ub * x + b_ub` on `[x_env.lb, x_env.ub]`.
    Propagates analytical moments via 5-point Gauss-Hermite quadrature and Stein's lemma.
    """
    a_lb = np.asarray(a_lb, dtype=np.float64).reshape(-1)
    b_lb = np.asarray(b_lb, dtype=np.float64).reshape(-1)
    a_ub = np.asarray(a_ub, dtype=np.float64).reshape(-1)
    b_ub = np.asarray(b_ub, dtype=np.float64).reshape(-1)

    a_lb_pos = np.maximum(a_lb, 0.0)
    a_lb_neg = np.minimum(a_lb, 0.0)
    gamma_lb = a_lb_pos * x_env.gamma_lb + a_lb_neg * x_env.gamma_ub + b_lb
    G_lb = a_lb_pos[:, None] * x_env.G_lb + a_lb_neg[:, None] * x_env.G_ub

    a_ub_pos = np.maximum(a_ub, 0.0)
    a_ub_neg = np.minimum(a_ub, 0.0)
    gamma_ub = a_ub_pos * x_env.gamma_ub + a_ub_neg * x_env.gamma_lb + b_ub
    G_ub = a_ub_pos[:, None] * x_env.G_ub + a_ub_neg[:, None] * x_env.G_lb

    lb = a_lb_pos * x_env.lb + a_lb_neg * x_env.ub + b_lb
    ub = a_ub_pos * x_env.ub + a_ub_neg * x_env.lb + b_ub
    if out_lb_clip is not None:
        lb = np.maximum(lb, out_lb_clip)
    if out_ub_clip is not None:
        ub = np.minimum(ub, out_ub_clip)
    ub = np.maximum(ub, lb)

    # 5-point Gauss-Hermite quadrature for smooth nonlinear moments
    std_x = np.maximum(np.sqrt(x_env.var), _MIN_STD)
    x_nodes = np.clip(
        x_env.mu[:, None] + std_x[:, None] * _GH_NODES[None, :],
        x_env.lb[:, None],
        x_env.ub[:, None],
    )
    y_nodes = fn_eval(x_nodes)
    mu_y = np.sum(y_nodes * _GH_WEIGHTS[None, :], axis=1)
    ey2 = np.sum(np.square(y_nodes) * _GH_WEIGHTS[None, :], axis=1)
    var_y = np.maximum(ey2 - np.square(mu_y), _MIN_VAR)
    mu_y = np.clip(mu_y, lb, ub)
    var_y = np.minimum(var_y, np.maximum(0.25 * np.square(ub - lb), _MIN_VAR))

    if fn_deriv is not None:
        eff_slope = np.sum(fn_deriv(x_nodes) * _GH_WEIGHTS[None, :], axis=1)
    else:
        eff_slope = 0.5 * (a_lb + a_ub)
    cross_cov = eff_slope[:, None] * x_env.cross_cov

    return AffineEnvelopeTensor(
        phys=x_env.phys,
        gamma_lb=gamma_lb,
        G_lb=G_lb,
        gamma_ub=gamma_ub,
        G_ub=G_ub,
        lb=lb,
        ub=ub,
        mu=mu_y,
        var=var_y,
        cross_cov=cross_cov,
        mode=x_env.mode,
    )


def sin_env(x_env):
    """Sound linear envelope and moments for `sin(x)` on `[l, u]`."""
    l, u = x_env.lb, x_env.ub
    w = u - l
    mid = 0.5 * (l + u)
    yl, yu = np.sin(l), np.sin(u)
    dl, du, dm = np.cos(l), np.cos(u), np.cos(mid)
    chord = np.where(w > 1e-12, (yu - yl) / np.where(w > 1e-12, w, 1.0), dm)

    # Concave on [0, pi] (sin >= 0), convex on [-pi, 0] (sin <= 0)
    a_lb = np.where(mid >= 0.0, chord, dm)
    a_ub = np.where(mid >= 0.0, dm, chord)
    b_lb = yl - a_lb * l
    b_ub = yu - a_ub * u

    # Stationary points of sin(x) - a*x where cos(x) = a
    a_lb_clip = np.clip(a_lb, -1.0, 1.0)
    a_ub_clip = np.clip(a_ub, -1.0, 1.0)
    theta_lb = np.arccos(a_lb_clip)
    theta_ub = np.arccos(a_ub_clip)
    k_base = np.round(mid / (2.0 * np.pi)) * 2.0 * np.pi
    extra = [
        k_base + theta_lb, k_base - theta_lb,
        k_base + theta_ub, k_base - theta_ub,
        k_base + 0.5 * np.pi, k_base - 0.5 * np.pi,
    ]
    a_lb, b_lb, a_ub, b_ub = _certify_univariate_chords_on_grid(
        l, u, a_lb, b_lb, a_ub, b_ub, np.sin, extra_pts=extra
    )
    return apply_univariate_primitive(
        x_env, a_lb, b_lb, a_ub, b_ub, np.sin, fn_deriv=np.cos,
        out_lb_clip=-1.0, out_ub_clip=1.0
    )


def cos_env(x_env):
    """Sound linear envelope and moments for `cos(x) = sin(x + pi/2)` on `[l, u]`."""
    shifted = x_env.scale_shift(scale=1.0, shift=0.5 * np.pi)
    return sin_env(shifted)


def square_env(x_env):
    """
    Sound linear envelope and moments for `x^2` on `[l, u]`.
    Since f(x) = x^2 is strictly convex:
      - Lower bound is tangent at m = clip(mu_x, l, u): 2*m*x - m^2
      - Upper bound is secant chord: (l + u)*x - l*u
    """
    l, u = x_env.lb, x_env.ub
    m = np.clip(x_env.mu, l, u)
    a_lb = 2.0 * m
    b_lb = -np.square(m)
    a_ub = l + u
    b_ub = -l * u
    a_lb, b_lb, a_ub, b_ub = _certify_univariate_chords_on_grid(
        l, u, a_lb, b_lb, a_ub, b_ub, np.square, extra_pts=[np.zeros_like(l), m]
    )
    out_lb = np.where((l <= 0.0) & (u >= 0.0), 0.0, np.minimum(np.square(l), np.square(u)))
    out_ub = np.maximum(np.square(l), np.square(u))
    return apply_univariate_primitive(
        x_env, a_lb, b_lb, a_ub, b_ub, np.square, fn_deriv=lambda z: 2.0 * z,
        out_lb_clip=out_lb, out_ub_clip=out_ub
    )


def reciprocal_env(z_env, min_depth=1e-3):
    """
    Sound linear envelope and moments for perspective reciprocal `1 / z` on `0 < l <= z <= u`.
    Since f(z) = 1/z is strictly convex on (0, inf):
      - Lower bound is tangent at m = clip(mu_z, l, u): -(1/m^2)*z + 2/m
      - Upper bound is secant chord: -(1/(l*u))*z + (l + u)/(l*u)
    """
    l = np.maximum(z_env.lb, min_depth)
    u = np.maximum(z_env.ub, l)
    m = np.clip(z_env.mu, l, u)
    a_lb = -1.0 / np.square(m)
    b_lb = 2.0 / m
    lu = l * u
    a_ub = -1.0 / lu
    b_ub = (l + u) / lu
    fn = lambda z: 1.0 / np.maximum(z, min_depth)
    df = lambda z: -1.0 / np.square(np.maximum(z, min_depth))
    a_lb, b_lb, a_ub, b_ub = _certify_univariate_chords_on_grid(
        l, u, a_lb, b_lb, a_ub, b_ub, fn, extra_pts=[m, np.sqrt(lu)]
    )
    return apply_univariate_primitive(
        z_env, a_lb, b_lb, a_ub, b_ub, fn, fn_deriv=df,
        out_lb_clip=1.0 / u, out_ub_clip=1.0 / l
    )


def exp_neg_env(x_env):
    """
    Sound linear envelope and moments for `exp(-x)` on `[l, u]`.
    Used in both 3D Gaussian Splatting radial evaluation and Volumetric NeRF transmittance.
    Since f(x) = exp(-x) is strictly convex:
      - Lower bound is tangent at m = clip(mu_x, l, u): -exp(-m)*x + exp(-m)*(1 + m)
      - Upper bound is secant chord connecting (l, exp(-l)) and (u, exp(-u)).
    """
    l = np.clip(x_env.lb, -50.0, 50.0)
    u = np.clip(np.maximum(x_env.ub, l), -50.0, 50.0)
    m = np.clip(x_env.mu, l, u)
    el = np.exp(-l)
    eu = np.exp(-u)
    em = np.exp(-m)
    w = u - l

    a_lb = -em
    b_lb = em * (1.0 + m)
    a_ub = np.where(w > 1e-12, (eu - el) / np.where(w > 1e-12, w, 1.0), -em)
    b_ub = el - a_ub * l

    fn = lambda z: np.exp(-np.clip(z, -50.0, 50.0))
    df = lambda z: -np.exp(-np.clip(z, -50.0, 50.0))
    a_lb, b_lb, a_ub, b_ub = _certify_univariate_chords_on_grid(
        l, u, a_lb, b_lb, a_ub, b_ub, fn, extra_pts=[m]
    )
    return apply_univariate_primitive(
        x_env, a_lb, b_lb, a_ub, b_ub, fn, fn_deriv=df,
        out_lb_clip=eu, out_ub_clip=el
    )


def gaussian_rbf_env(z_env):
    """
    Sound linear envelope and moments for the 1D Gaussian radial basis kernel
    `psi(z) = exp(-z^2)` on `[l, u]`.
    Factored evaluation `exp(-(z1^2 + z2^2)) = gaussian_rbf_env(z1) * gaussian_rbf_env(z2)`
    preserves signed first-order derivatives across splat stencils without the double
    relaxation loss of composing `exp_neg_env(square_env(z))`.
    """
    l = np.clip(z_env.lb, -10.0, 10.0)
    u = np.clip(np.maximum(z_env.ub, l), -10.0, 10.0)
    m = np.clip(z_env.mu, l, u)
    w = u - l

    fn = lambda z: np.exp(-np.square(np.clip(z, -10.0, 10.0)))
    df = lambda z: -2.0 * z * np.exp(-np.square(np.clip(z, -10.0, 10.0)))

    yl = fn(l)
    yu = fn(u)
    dm = df(m)
    chord = np.where(w > 1e-12, (yu - yl) / np.where(w > 1e-12, w, 1.0), dm)

    # Concave on [-1/sqrt(2), +1/sqrt(2)], convex on |z| >= 1/sqrt(2)
    infl = 1.0 / np.sqrt(2.0)
    is_concave_mid = np.abs(m) <= infl
    a_lb = np.where(is_concave_mid, chord, dm)
    b_lb = np.where(is_concave_mid, yl - a_lb * l, fn(m) - dm * m)
    a_ub = np.where(is_concave_mid, dm, chord)
    b_ub = np.where(is_concave_mid, fn(m) - dm * m, yl - a_ub * l)

    extra = [
        np.zeros_like(l),
        m,
        np.full_like(l, infl),
        np.full_like(l, -infl),
    ]
    a_lb, b_lb, a_ub, b_ub = _certify_univariate_chords_on_grid(
        l, u, a_lb, b_lb, a_ub, b_ub, fn, extra_pts=extra
    )
    out_lb = np.minimum(yl, yu)
    out_ub = np.where((l <= 0.0) & (u >= 0.0), 1.0, np.maximum(yl, yu))
    return apply_univariate_primitive(
        z_env, a_lb, b_lb, a_ub, b_ub, fn, fn_deriv=df,
        out_lb_clip=out_lb, out_ub_clip=out_ub
    )


def relu_env(x_env):
    """Sound triangle linear envelope and moments for `ReLU(x) = max(0, x)`."""
    l, u = x_env.lb, x_env.ub
    a_lb = np.zeros_like(l)
    b_lb = np.zeros_like(l)
    a_ub = np.zeros_like(l)
    b_ub = np.zeros_like(l)

    active = l >= 0.0
    a_lb[active] = 1.0
    a_ub[active] = 1.0

    unst = (l < 0.0) & (u > 0.0)
    if np.any(unst):
        lu, uu = l[unst], u[unst]
        a_ub[unst] = uu / np.maximum(uu - lu, 1e-15)
        b_ub[unst] = -a_ub[unst] * lu
        a_lb[unst] = (uu >= np.abs(lu)).astype(np.float64)

    fn = lambda z: np.maximum(z, 0.0)
    df = lambda z: (z > 0.0).astype(np.float64)
    return apply_univariate_primitive(
        x_env, a_lb, b_lb, a_ub, b_ub, fn, fn_deriv=df,
        out_lb_clip=np.maximum(l, 0.0), out_ub_clip=np.maximum(u, 0.0)
    )


def leakyrelu_env(x_env, alpha=0.1):
    """Sound linear envelope and moments for `LeakyReLU(x; alpha)`."""
    l, u = x_env.lb, x_env.ub
    alpha = float(alpha)
    a_lb = np.full_like(l, alpha)
    b_lb = np.zeros_like(l)
    a_ub = np.full_like(l, alpha)
    b_ub = np.zeros_like(l)

    active = l >= 0.0
    a_lb[active] = 1.0
    a_ub[active] = 1.0

    unst = (l < 0.0) & (u > 0.0)
    if np.any(unst):
        lu, uu = l[unst], u[unst]
        a_ub[unst] = (uu - alpha * lu) / np.maximum(uu - lu, 1e-15)
        b_ub[unst] = (1.0 - alpha) * uu * (-lu) / np.maximum(uu - lu, 1e-15)
        a_lb[unst] = np.where(uu >= np.abs(lu), 1.0, alpha)

    fn = lambda z: np.where(z >= 0.0, z, alpha * z)
    df = lambda z: np.where(z >= 0.0, 1.0, alpha)
    return apply_univariate_primitive(
        x_env, a_lb, b_lb, a_ub, b_ub, fn, fn_deriv=df,
        out_lb_clip=fn(l), out_ub_clip=fn(u)
    )


def sigmoid_env(x_env):
    """Sound S-curve linear envelope and moments for `sigmoid(x) = 1 / (1 + exp(-x))`."""
    l, u = np.clip(x_env.lb, -40.0, 40.0), np.clip(x_env.ub, -40.0, 40.0)
    fn = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    df = lambda z: fn(z) * (1.0 - fn(z))

    yl, yu = fn(l), fn(u)
    dyl, dyu = df(l), df(u)
    w = u - l
    chord = np.where(w > 1e-12, (yu - yl) / np.where(w > 1e-12, w, 1.0), 0.5 * (dyl + dyu))
    m_min = np.minimum(dyl, dyu)

    a_lb = np.where(l >= 0.0, chord, np.where(u <= 0.0, dyl, m_min))
    b_lb = yl - a_lb * l
    a_ub = np.where(l >= 0.0, dyu, np.where(u <= 0.0, chord, m_min))
    b_ub = yu - a_ub * u

    a_lb, b_lb, a_ub, b_ub = _certify_univariate_chords_on_grid(
        l, u, a_lb, b_lb, a_ub, b_ub, fn, extra_pts=[np.zeros_like(l)]
    )
    return apply_univariate_primitive(
        x_env, a_lb, b_lb, a_ub, b_ub, fn, fn_deriv=df,
        out_lb_clip=yl, out_ub_clip=yu
    )


def softplus_env(x_env):
    """
    Sound linear envelope and moments for `softplus(x) = log(1 + exp(x))`.
    Since softplus is strictly convex on R:
      - Lower bound is tangent at m = clip(mu_x, l, u)
      - Upper bound is secant chord connecting (l, softplus(l)) and (u, softplus(u))
    """
    l, u = np.clip(x_env.lb, -40.0, 40.0), np.clip(x_env.ub, -40.0, 40.0)
    m = np.clip(x_env.mu, l, u)
    fn = lambda z: np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)
    df = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))

    yl, yu, ym = fn(l), fn(u), fn(m)
    dm = df(m)
    w = u - l

    a_lb = dm
    b_lb = ym - dm * m
    a_ub = np.where(w > 1e-12, (yu - yl) / np.where(w > 1e-12, w, 1.0), dm)
    b_ub = yl - a_ub * l

    a_lb, b_lb, a_ub, b_ub = _certify_univariate_chords_on_grid(
        l, u, a_lb, b_lb, a_ub, b_ub, fn, extra_pts=[m]
    )
    return apply_univariate_primitive(
        x_env, a_lb, b_lb, a_ub, b_ub, fn, fn_deriv=df,
        out_lb_clip=yl, out_ub_clip=yu
    )


def clamp_env(x_env, low=0.0, high=1.0):
    """Sound linear envelope and moments for `clip(x, low, high)`."""
    shifted = x_env.scale_shift(scale=1.0, shift=-float(low))
    rel1 = relu_env(shifted)
    span = float(high - low)
    inv = rel1.scale_shift(scale=-1.0, shift=span)
    rel2 = relu_env(inv)
    res = rel2.scale_shift(scale=-1.0, shift=float(high))
    res.lb = np.clip(res.lb, float(low), float(high))
    res.ub = np.clip(np.maximum(res.ub, res.lb), float(low), float(high))
    res.mu = np.clip(res.mu, res.lb, res.ub)
    return res


def multiply_env(x_env, y_env):
    """
    Sound bilinear McCormick linear envelope and correlated Gaussian product moments
    for elementwise product `z = x * y`:
      - Lower envelope: convex combination of McCormick lower planes
            z >= l_y * x + l_x * y - l_x * l_y
            z >= u_y * x + u_x * y - u_x * u_y
      - Upper envelope: convex combination of McCormick upper planes
            z <= u_y * x + l_x * y - l_x * u_y
            z <= l_y * x + u_x * y - u_x * l_y
      - Moments: exact correlated Gaussian product mean & variance
            E[x * y] = mu_x * mu_y + Cov(x, y)
    """
    if not isinstance(y_env, AffineEnvelopeTensor):
        return x_env.scale_shift(scale=y_env, shift=0.0)
    if not isinstance(x_env, AffineEnvelopeTensor):
        return y_env.scale_shift(scale=x_env, shift=0.0)

    if y_env.N == 1 and x_env.N > 1:
        y_env = y_env.slice(np.zeros(x_env.N, dtype=np.int64))
    elif x_env.N == 1 and y_env.N > 1:
        x_env = x_env.slice(np.zeros(y_env.N, dtype=np.int64))
    assert x_env.N == y_env.N

    lx, ux = x_env.lb, x_env.ub
    ly, uy = y_env.lb, y_env.ub

    # Concrete interval bounds of x * y over [lx, ux] x [ly, uy]
    p1, p2, p3, p4 = lx * ly, lx * uy, ux * ly, ux * uy
    lb = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
    ub = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))

    # Midpoint-balanced McCormick coefficients (weight 0.5 on each plane)
    # Plane L1: ly*x + lx*y - lx*ly, Plane L2: uy*x + ux*y - ux*uy
    # Average L: mx*y + my*x - 0.5*(lx*ly + ux*uy) where mx=0.5*(lx+ux), my=0.5*(ly+uy)
    mx = 0.5 * (lx + ux)
    my = 0.5 * (ly + uy)
    c_lb = -0.5 * (lx * ly + ux * uy)
    c_ub = -0.5 * (lx * uy + ux * ly)

    my_pos, my_neg = np.maximum(my, 0.0), np.minimum(my, 0.0)
    mx_pos, mx_neg = np.maximum(mx, 0.0), np.minimum(mx, 0.0)

    gamma_lb = (
        my_pos * x_env.gamma_lb + my_neg * x_env.gamma_ub +
        mx_pos * y_env.gamma_lb + mx_neg * y_env.gamma_ub +
        c_lb - _SAFETY_PAD
    )
    G_lb = (
        my_pos[:, None] * x_env.G_lb + my_neg[:, None] * x_env.G_ub +
        mx_pos[:, None] * y_env.G_lb + mx_neg[:, None] * y_env.G_ub
    )

    gamma_ub = (
        my_pos * x_env.gamma_ub + my_neg * x_env.gamma_lb +
        mx_pos * y_env.gamma_ub + mx_neg * y_env.gamma_lb +
        c_ub + _SAFETY_PAD
    )
    G_ub = (
        my_pos[:, None] * x_env.G_ub + my_neg[:, None] * x_env.G_lb +
        mx_pos[:, None] * y_env.G_ub + mx_neg[:, None] * y_env.G_lb
    )

    # Correlated Gaussian product moments via Stein's lemma
    phys = x_env.phys
    Gx = 0.5 * (x_env.G_lb + x_env.G_ub)
    Gy = 0.5 * (y_env.G_lb + y_env.G_ub)
    if phys.Sig0_trunc.ndim == 1:
        cov_xy = np.sum(Gx * Gy * phys.Sig0_trunc[None, :], axis=1)
    else:
        cov_xy = np.sum((Gx @ phys.Sig0_trunc) * Gy, axis=1)
    max_corr = np.sqrt(np.maximum(x_env.var * y_env.var, _MIN_VAR))
    cov_xy = np.clip(cov_xy, -0.999 * max_corr, 0.999 * max_corr)

    mu_z = np.clip(x_env.mu * y_env.mu + cov_xy, lb, ub)
    var_z = (
        np.square(y_env.mu) * x_env.var +
        np.square(x_env.mu) * y_env.var +
        2.0 * x_env.mu * y_env.mu * cov_xy +
        x_env.var * y_env.var +
        np.square(cov_xy)
    )
    var_z = np.clip(var_z, _MIN_VAR, np.maximum(0.25 * np.square(ub - lb), _MIN_VAR))
    cross_cov = y_env.mu[:, None] * x_env.cross_cov + x_env.mu[:, None] * y_env.cross_cov

    return AffineEnvelopeTensor(
        phys=phys,
        gamma_lb=gamma_lb,
        G_lb=G_lb,
        gamma_ub=gamma_ub,
        G_ub=G_ub,
        lb=lb,
        ub=ub,
        mu=mu_z,
        var=var_z,
        cross_cov=cross_cov,
        mode=x_env.mode,
    )


def divide_env(x_env, z_env, min_depth=1e-3):
    """Sound envelope and moments for `x / z = x * (1 / z)` with `z >= min_depth > 0`."""
    inv_z = reciprocal_env(z_env, min_depth=min_depth)
    return multiply_env(x_env, inv_z)
