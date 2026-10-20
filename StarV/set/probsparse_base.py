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
Shared mathematical engine for Approximate ProbStar / ProbSparseImageStar reachability:
  1. Analytic Moment Propagation (Rectified Gaussian, LeakyReLU, SatLin, SatLins,
     GELU, MaxPool order statistics, bilinear McCormick moments) with both
     O(n) diagonal variance ('diagonal') and Stein's-lemma correlated covariance ('full').
  2. Sound Linear Relaxation Envelope DAG (RelaxationEnvelopeRecord) and O(nnz)
     backward substitution to original input Gaussian predicates alpha_0 ~ N(mu_0, Sig_0).
  3. Two-Stage LP + Dual/Moment Quantitative Verification & Robustness Certification
     providing provably sound probability intervals [p_min, p_max] and moment-matched p_hat.

Author: Hoang Dung Tran & StarV Team
"""

import copy
import numpy as np
import scipy.sparse as sp
from scipy.special import erf, ndtr
from scipy.stats import norm, multivariate_normal
from scipy.linalg import block_diag

_INV_SQRT2 = 1.0 / np.sqrt(2.0)
_INV_SQRT2PI = 1.0 / np.sqrt(2.0 * np.pi)
_MIN_VAR = 1e-16
_MIN_STD = 1e-8


def std_normal_pdf(x):
    """Vectorized standard normal PDF phi(x)."""
    x = np.asarray(x, dtype=np.float64)
    return _INV_SQRT2PI * np.exp(-0.5 * np.clip(x * x, 0.0, 700.0))


def std_normal_cdf(x):
    """Vectorized standard normal CDF Phi(x)."""
    x = np.asarray(x, dtype=np.float64)
    return ndtr(x)


def mvn_cdf_lowdim(lb, ub, mu, Sig):
    """Compute multivariate normal probability P(lb <= Z <= ub) for Z ~ N(mu, Sig)."""
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    k = len(mu)
    if k == 0:
        return 1.0
    if np.any(ub < lb):
        return 0.0
    if k == 1:
        std = max(float(np.sqrt(max(np.asarray(Sig).item(), _MIN_VAR))), _MIN_STD)
        z_u = (ub[0] - mu[0]) / std
        z_l = (lb[0] - mu[0]) / std
        return float(np.clip(ndtr(z_u) - ndtr(z_l), 0.0, 1.0))

    Sig = np.asarray(Sig, dtype=np.float64)
    if Sig.ndim == 1:
        stds = np.maximum(np.sqrt(np.maximum(Sig, _MIN_VAR)), _MIN_STD)
        probs = np.clip(ndtr((ub - mu) / stds) - ndtr((lb - mu) / stds), 0.0, 1.0)
        return float(np.prod(probs))

    # Check if Sig is effectively diagonal
    off_diag = Sig - np.diag(np.diag(Sig))
    if np.max(np.abs(off_diag)) <= 1e-12 * max(1.0, float(np.max(np.abs(np.diag(Sig))))):
        stds = np.maximum(np.sqrt(np.maximum(np.diag(Sig), _MIN_VAR)), _MIN_STD)
        probs = np.clip(ndtr((ub - mu) / stds) - ndtr((lb - mu) / stds), 0.0, 1.0)
        return float(np.prod(probs))

    # Regularize covariance slightly for numerical stability
    Sig_reg = 0.5 * (Sig + Sig.T) + _MIN_VAR * np.eye(k)
    try:
        prob = multivariate_normal.cdf(
            ub, mean=mu, cov=Sig_reg, lower_limit=lb, allow_singular=True
        )
        if np.isnan(prob):
            raise ValueError("NaN from multivariate_normal.cdf")
        return float(np.clip(prob, 0.0, 1.0))
    except Exception:
        # Fallback to diagonal approximation or Monte Carlo if Genz fails
        rng = np.random.default_rng(42)
        samples = rng.multivariate_normal(mu, Sig_reg, size=20000)
        inside = np.all((samples >= lb[None, :]) & (samples <= ub[None, :]), axis=1)
        return float(np.mean(inside))


def compute_box_input_probability(mu0, Sig0, pred_lb0, pred_ub0):
    """Compute P(pred_lb0 <= alpha_0 <= pred_ub0) for alpha_0 ~ N(mu0, Sig0)."""
    mu0 = np.asarray(mu0, dtype=np.float64).reshape(-1)
    lb0 = np.asarray(pred_lb0, dtype=np.float64).reshape(-1)
    ub0 = np.asarray(pred_ub0, dtype=np.float64).reshape(-1)
    Sig0_arr = np.asarray(Sig0, dtype=np.float64)
    if Sig0_arr.ndim == 1:
        stds = np.maximum(np.sqrt(np.maximum(Sig0_arr, _MIN_VAR)), _MIN_STD)
        log_p = np.sum(np.log(np.maximum(ndtr((ub0 - mu0) / stds) - ndtr((lb0 - mu0) / stds), 1e-300)))
        return float(np.clip(np.exp(log_p), 0.0, 1.0))
    off_diag = Sig0_arr - np.diag(np.diag(Sig0_arr))
    if np.max(np.abs(off_diag)) <= 1e-12 * max(1.0, float(np.max(np.abs(np.diag(Sig0_arr))))):
        stds = np.maximum(np.sqrt(np.maximum(np.diag(Sig0_arr), _MIN_VAR)), _MIN_STD)
        log_p = np.sum(np.log(np.maximum(ndtr((ub0 - mu0) / stds) - ndtr((lb0 - mu0) / stds), 1e-300)))
        return float(np.clip(np.exp(log_p), 0.0, 1.0))
    return mvn_cdf_lowdim(lb0, ub0, mu0, Sig0_arr)


def compute_truncated_gaussian_moments_1d(mu0, var0, lb0, ub0):
    """
    Compute exact mean and variance of independent 1D truncated Gaussians
    alpha_0 | alpha_0 in [lb0, ub0] where alpha_{0,j} ~ N(mu0_j, var0_j).
    Because the box [lb0, ub0] is a Cartesian product, for diagonal covariance
    the conditional distribution on the box factors EXACTLY into independent
    1D truncated normal distributions with zero curse of dimensionality.
    """
    mu0 = np.asarray(mu0, dtype=np.float64).reshape(-1)
    var0 = np.asarray(var0, dtype=np.float64).reshape(-1)
    lb0 = np.asarray(lb0, dtype=np.float64).reshape(-1)
    ub0 = np.asarray(ub0, dtype=np.float64).reshape(-1)

    std0 = np.maximum(np.sqrt(np.maximum(var0, _MIN_VAR)), _MIN_STD)
    xi_l = np.clip((lb0 - mu0) / std0, -35.0, 35.0)
    xi_u = np.clip((ub0 - mu0) / std0, -35.0, 35.0)

    phi_l = std_normal_pdf(xi_l)
    phi_u = std_normal_pdf(xi_u)
    Z = np.maximum(std_normal_cdf(xi_u) - std_normal_cdf(xi_l), 1e-15)

    ratio_mean = (phi_l - phi_u) / Z
    mu_trunc = np.clip(mu0 + std0 * ratio_mean, lb0, ub0)

    term2 = (xi_l * phi_l - xi_u * phi_u) / Z
    var_factor = np.clip(1.0 + term2 - np.square(ratio_mean), 1e-6, 1.0)
    var_trunc = np.maximum(var0 * var_factor, _MIN_VAR)
    return mu_trunc, var_trunc


class RelaxationEnvelopeRecord(object):
    """
    Stores sound linear lower and upper bounds for a batch of newly introduced
    predicate variables beta in terms of the preceding predicate vector alpha_prev:

        gamma_lb + G_lb @ alpha_prev <= beta <= gamma_ub + G_ub @ alpha_prev

    To avoid duplicating sparse matrices, when the bounds are of the neuron-wise
    scaling form:
        G_lb = diag(a_lb) @ V_pre,   gamma_lb = a_lb * c_pre + b_lb
        G_ub = diag(a_ub) @ V_pre,   gamma_ub = a_ub * c_pre + b_ub
    we store (V_pre, c_pre, a_lb, b_lb, a_ub, b_ub) directly.
    Alternatively, explicit (G_lb, gamma_lb, G_ub, gamma_ub) can be stored
    (e.g., for MaxPool2D where lower and upper rows reference different candidates).
    """

    __slots__ = (
        'n_prev', 'n_new', 'V_pre', 'c_pre',
        'a_lb', 'b_lb', 'a_ub', 'b_ub',
        'G_lb', 'gamma_lb', 'G_ub', 'gamma_ub', 'is_explicit'
    )

    def __init__(
        self,
        n_prev,
        n_new,
        V_pre=None,
        c_pre=None,
        a_lb=None,
        b_lb=None,
        a_ub=None,
        b_ub=None,
        G_lb=None,
        gamma_lb=None,
        G_ub=None,
        gamma_ub=None,
    ):
        self.n_prev = int(n_prev)
        self.n_new = int(n_new)
        if G_lb is not None and G_ub is not None:
            self.is_explicit = True
            self.G_lb = G_lb
            self.gamma_lb = np.asarray(gamma_lb, dtype=np.float64).reshape(-1)
            self.G_ub = G_ub
            self.gamma_ub = np.asarray(gamma_ub, dtype=np.float64).reshape(-1)
            self.V_pre = None
            self.c_pre = None
            self.a_lb = None
            self.b_lb = None
            self.a_ub = None
            self.b_ub = None
        else:
            self.is_explicit = False
            self.V_pre = V_pre
            self.c_pre = np.asarray(c_pre, dtype=np.float64).reshape(-1)
            self.a_lb = np.asarray(a_lb, dtype=np.float64).reshape(-1)
            self.b_lb = np.asarray(b_lb, dtype=np.float64).reshape(-1)
            self.a_ub = np.asarray(a_ub, dtype=np.float64).reshape(-1)
            self.b_ub = np.asarray(b_ub, dtype=np.float64).reshape(-1)
            self.G_lb = None
            self.gamma_lb = self.a_lb * self.c_pre + self.b_lb
            self.G_ub = None
            self.gamma_ub = self.a_ub * self.c_pre + self.b_ub

    def nbytes(self):
        total = 0
        if self.is_explicit:
            for M in (self.G_lb, self.G_ub):
                if sp.issparse(M):
                    total += M.data.nbytes + M.indices.nbytes + M.indptr.nbytes
                elif isinstance(M, np.ndarray):
                    total += M.nbytes
            total += self.gamma_lb.nbytes + self.gamma_ub.nbytes
        else:
            if sp.issparse(self.V_pre):
                total += self.V_pre.data.nbytes
                if hasattr(self.V_pre, 'indices'):
                    total += self.V_pre.indices.nbytes + self.V_pre.indptr.nbytes
                elif hasattr(self.V_pre, 'row'):
                    total += self.V_pre.row.nbytes + self.V_pre.col.nbytes
            elif isinstance(self.V_pre, np.ndarray):
                total += self.V_pre.nbytes
            total += (
                self.c_pre.nbytes + self.a_lb.nbytes + self.b_lb.nbytes +
                self.a_ub.nbytes + self.b_ub.nbytes
            )
        return total


def compute_preactivation_moments(V_pre, c_pre, mu, Sig, cov_mode='diagonal'):
    """
    Compute marginal mean mu_x and variance var_x of pre-activations
    x = c_pre + V_pre @ alpha for alpha ~ N(mu, Sig).

    Returns:
        mu_x: 1D array of shape (m,)
        var_x: 1D array of shape (m,)
        V_Sig: (m, n) array if cov_mode == 'full' else None
    """
    c_pre = np.asarray(c_pre, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    mu_x = c_pre + (V_pre @ mu)

    if cov_mode == 'diagonal' or Sig.ndim == 1:
        var_vec = np.asarray(Sig, dtype=np.float64).reshape(-1) if Sig.ndim == 1 else np.diag(Sig)
        if sp.issparse(V_pre):
            V_sq = V_pre.power(2)
            var_x = np.asarray(V_sq @ var_vec, dtype=np.float64).reshape(-1)
        else:
            var_x = np.square(V_pre) @ var_vec
        var_x = np.maximum(var_x, _MIN_VAR)
        return mu_x, var_x, None
    else:
        Sig_2d = np.asarray(Sig, dtype=np.float64)
        V_Sig = V_pre @ Sig_2d  # shape (m, n)
        if sp.issparse(V_pre):
            var_x = np.asarray(V_pre.multiply(V_Sig).sum(axis=1)).reshape(-1)
        else:
            var_x = np.sum(V_pre * V_Sig, axis=1)
        var_x = np.maximum(var_x, _MIN_VAR)
        return mu_x, var_x, V_Sig


def propagate_relu_moments(
    V_pre, c_pre, l, u, mu, Sig,
    cov_mode='diagonal',
    relu_approx_mode='analytic_gaussian',
    dtype=np.float64
):
    """
    Compute extended mean new_mu and covariance new_Sig after introducing m = len(l)
    new predicate variables beta = ReLU(x_map) for ambiguous neurons with l < 0 < u,
    and construct the sound RelaxationEnvelopeRecord.

    Supports:
      - 'analytic_gaussian' (default): Exact rectified-Gaussian mean/variance
        + Stein's lemma cross-covariance when cov_mode == 'full'.
      - 'heuristic_gaussian': Box heuristic mu_new = 0.5 * u, sigma_new = u / 6.
      - 'constraints_only': Same as heuristic_gaussian with flag.
    """
    l = np.asarray(l, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    m = len(l)
    n_prev = len(mu)

    if m == 0:
        return mu.copy(), Sig.copy(), None

    if relu_approx_mode == 'analytic_gaussian':
        mu_x, var_x, V_Sig = compute_preactivation_moments(V_pre, c_pre, mu, Sig, cov_mode=cov_mode)
        std_x = np.maximum(np.sqrt(var_x), _MIN_STD)
        gamma = mu_x / std_x
        Phi = std_normal_cdf(gamma)
        phi = std_normal_pdf(gamma)

        raw_mu_beta = mu_x * Phi + std_x * phi
        mu_beta = np.clip(raw_mu_beta, 0.0, u)
        second_moment = ( np.square(mu_x) + var_x ) * Phi + mu_x * std_x * phi
        var_beta = np.maximum(second_moment - np.square(raw_mu_beta), _MIN_VAR)
        # Ensure variance is also consistent with [0, u] Popoviciu bound u^2 / 4
        var_beta = np.minimum(var_beta, np.maximum(0.25 * np.square(u), _MIN_VAR))
    elif relu_approx_mode in ('heuristic_gaussian', 'constraints_only'):
        mu_beta = 0.5 * u
        sig_beta = np.maximum(u / 6.0, _MIN_STD)
        var_beta = np.square(sig_beta)
        Phi = None
        V_Sig = None
    else:
        raise ValueError(
            f"Unknown relu_approx_mode='{relu_approx_mode}'. "
            "Expected 'analytic_gaussian', 'heuristic_gaussian', or 'constraints_only'."
        )

    new_mu = np.concatenate([mu, mu_beta]).astype(dtype, copy=False)

    if cov_mode == 'diagonal' or Sig.ndim == 1:
        var_prev = Sig if Sig.ndim == 1 else np.diag(Sig)
        new_Sig = np.concatenate([var_prev, var_beta]).astype(dtype, copy=False)
    else:
        if V_Sig is not None and Phi is not None:
            # Stein's Lemma: Cov(alpha, ReLU(x_j)) = Phi(gamma_j) * Sig @ v_j
            cross_cov = V_Sig.T * Phi[None, :]  # shape (n_prev, m)
            # Ensure Schur complement positivity: var_beta >= diag(cross_cov^T Sig^-1 cross_cov) + eps
            schur_diag = np.square(Phi) * var_x
            diag_beta = np.maximum(var_beta, schur_diag + 1e-10)
            top = np.hstack([Sig, cross_cov])
            bot = np.hstack([cross_cov.T, np.diag(diag_beta)])
            new_Sig = np.vstack([top, bot]).astype(dtype, copy=False)
        else:
            new_Sig = block_diag(Sig, np.diag(var_beta)).astype(dtype, copy=False)

    # Sound linear relaxation slopes for ReLU on [l, u] with l < 0 < u:
    # Upper chord: y <= (u / (u - l)) * (x - l) = a_ub * x + b_ub
    # Lower slope: y >= a_lb * x where a_lb = 1 if u >= |l| else 0 (minimum area sound lower bound)
    denom = np.maximum(u - l, 1e-15)
    a_ub = u / denom
    b_ub = -a_ub * l
    a_lb = (u >= np.abs(l)).astype(np.float64)
    b_lb = np.zeros(m, dtype=np.float64)

    env_record = RelaxationEnvelopeRecord(
        n_prev=n_prev,
        n_new=m,
        V_pre=V_pre,
        c_pre=c_pre,
        a_lb=a_lb,
        b_lb=b_lb,
        a_ub=a_ub,
        b_ub=b_ub,
    )

    return new_mu, new_Sig, env_record


def propagate_leakyrelu_moments(
    V_pre, c_pre, l, u, gamma_leak, mu, Sig,
    cov_mode='diagonal',
    dtype=np.float64
):
    """
    Analytic moment propagation and sound linear envelope for LeakyReLU(x; gamma_leak)
    on ambiguous neurons l < 0 < u:
        y = gamma_leak * x + (1 - gamma_leak) * max(0, x)
    """
    l = np.asarray(l, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    m = len(l)
    n_prev = len(mu)
    if m == 0:
        return mu.copy(), Sig.copy(), None

    mu_x, var_x, V_Sig = compute_preactivation_moments(V_pre, c_pre, mu, Sig, cov_mode=cov_mode)
    std_x = np.maximum(np.sqrt(var_x), _MIN_STD)
    z = mu_x / std_x
    Phi = std_normal_cdf(z)
    phi = std_normal_pdf(z)

    # E[max(0, x)] and E[LeakyReLU(x)]
    mu_pos = mu_x * Phi + std_x * phi
    mu_beta = gamma_leak * mu_x + (1.0 - gamma_leak) * mu_pos
    mu_beta = np.clip(mu_beta, gamma_leak * l, u)

    # E[y^2] = E[x^2] * Phi + gamma_leak^2 * E[x^2] * (1 - Phi) + (1 - gamma_leak^2) * mu_x * std_x * phi
    ex2 = np.square(mu_x) + var_x
    ey2 = ex2 * (Phi + (gamma_leak ** 2) * (1.0 - Phi)) + (1.0 - gamma_leak ** 2) * mu_x * std_x * phi
    var_beta = np.maximum(ey2 - np.square(mu_beta), _MIN_VAR)

    new_mu = np.concatenate([mu, mu_beta]).astype(dtype, copy=False)
    if cov_mode == 'diagonal' or Sig.ndim == 1:
        var_prev = Sig if Sig.ndim == 1 else np.diag(Sig)
        new_Sig = np.concatenate([var_prev, var_beta]).astype(dtype, copy=False)
    else:
        eff_slope = gamma_leak + (1.0 - gamma_leak) * Phi
        cross_cov = V_Sig.T * eff_slope[None, :]
        schur_diag = np.square(eff_slope) * var_x
        diag_beta = np.maximum(var_beta, schur_diag + 1e-10)
        new_Sig = np.vstack([
            np.hstack([Sig, cross_cov]),
            np.hstack([cross_cov.T, np.diag(diag_beta)])
        ]).astype(dtype, copy=False)

    denom = np.maximum(u - l, 1e-15)
    a_ub = (u - gamma_leak * l) / denom
    b_ub = (1.0 - gamma_leak) * (u * (-l)) / denom
    a_lb = np.where(u >= np.abs(l), 1.0, float(gamma_leak))
    b_lb = np.zeros(m, dtype=np.float64)

    env_record = RelaxationEnvelopeRecord(
        n_prev=n_prev, n_new=m, V_pre=V_pre, c_pre=c_pre,
        a_lb=a_lb, b_lb=b_lb, a_ub=a_ub, b_ub=b_ub
    )
    return new_mu, new_Sig, env_record


def propagate_general_elementwise_moments(
    V_pre, c_pre, a_lb, b_lb, a_ub, b_ub, pred_lb_new, pred_ub_new,
    mu, Sig, cov_mode='diagonal', fn_eval=None, fn_deriv=None, dtype=np.float64
):
    """
    Moment propagation + sound linear envelope for general elementwise nonlinearities
    (SatLin, SatLins, Sigmoid, Tanh, GELU, LayerNorm scaling) given sound linear chords:
        a_lb * x + b_lb <= y <= a_ub * x + b_ub
    """
    m = len(a_lb)
    n_prev = len(mu)
    if m == 0:
        return mu.copy(), Sig.copy(), None

    mu_x, var_x, V_Sig = compute_preactivation_moments(V_pre, c_pre, mu, Sig, cov_mode=cov_mode)
    std_x = np.maximum(np.sqrt(var_x), _MIN_STD)

    if fn_eval is not None:
        # 5-point Gauss-Hermite quadrature for smooth nonlinear activation moments
        gh_pts = np.array([-2.85697001, -1.35562618, 0.0, 1.35562618, 2.85697001])
        gh_wts = np.array([0.01125741, 0.22207592, 0.53333333, 0.22207592, 0.01125741])
        x_nodes = mu_x[:, None] + std_x[:, None] * gh_pts[None, :]
        y_nodes = fn_eval(x_nodes)
        mu_beta = np.sum(y_nodes * gh_wts[None, :], axis=1)
        ey2 = np.sum(np.square(y_nodes) * gh_wts[None, :], axis=1)
        var_beta = np.maximum(ey2 - np.square(mu_beta), _MIN_VAR)
        if fn_deriv is not None:
            eff_slope = np.sum(fn_deriv(x_nodes) * gh_wts[None, :], axis=1)
        else:
            eff_slope = 0.5 * (np.asarray(a_lb) + np.asarray(a_ub))
    else:
        eff_slope = 0.5 * (np.asarray(a_lb) + np.asarray(a_ub))
        eff_off = 0.5 * (np.asarray(b_lb) + np.asarray(b_ub))
        mu_beta = eff_slope * mu_x + eff_off
        var_beta = np.maximum(np.square(eff_slope) * var_x, _MIN_VAR)

    mu_beta = np.clip(mu_beta, pred_lb_new, pred_ub_new)
    new_mu = np.concatenate([mu, mu_beta]).astype(dtype, copy=False)

    if cov_mode == 'diagonal' or Sig.ndim == 1:
        var_prev = Sig if Sig.ndim == 1 else np.diag(Sig)
        new_Sig = np.concatenate([var_prev, var_beta]).astype(dtype, copy=False)
    else:
        cross_cov = V_Sig.T * eff_slope[None, :]
        schur_diag = np.square(eff_slope) * var_x
        diag_beta = np.maximum(var_beta, schur_diag + 1e-10)
        new_Sig = np.vstack([
            np.hstack([Sig, cross_cov]),
            np.hstack([cross_cov.T, np.diag(diag_beta)])
        ]).astype(dtype, copy=False)

    env_record = RelaxationEnvelopeRecord(
        n_prev=n_prev, n_new=m, V_pre=V_pre, c_pre=c_pre,
        a_lb=a_lb, b_lb=b_lb, a_ub=a_ub, b_ub=b_ub
    )
    return new_mu, new_Sig, env_record


def propagate_maxpool_moments(
    V, center, state_lb, state_ub, max_index_list, lb_new, ub_new,
    mu, Sig, cov_mode='diagonal', dtype=np.float64
):
    """
    Analytic moment propagation (Clark's pairwise max-of-Gaussians) and sound
    linear envelope for MaxPool2D over-approximation with nPred = len(max_index_list)
    ambiguous pooling windows.
    """
    nPred = len(max_index_list)
    n_prev = len(mu)
    if nPred == 0:
        return mu.copy(), Sig.copy(), None

    # Gather all unique candidate indices needed to compute pre-activation moments
    all_cands = np.unique(np.hstack(max_index_list))
    cand_pos = {int(idx): pos for pos, idx in enumerate(all_cands)}
    V_cands = V[all_cands]
    c_cands = center[all_cands]
    mu_cands, var_cands, V_Sig_cands = compute_preactivation_moments(
        V_cands, c_cands, mu, Sig, cov_mode=cov_mode
    )

    mu_beta = np.zeros(nPred, dtype=np.float64)
    var_beta = np.zeros(nPred, dtype=np.float64)
    best_lb_idx = np.zeros(nPred, dtype=np.int64)
    best_ub_idx = np.zeros(nPred, dtype=np.int64)
    ub_slack = np.zeros(nPred, dtype=np.float64)

    for i, cands in enumerate(max_index_list):
        cands = np.asarray(cands, dtype=np.int64)
        # Sound lower bound candidate: r* = argmax_{r in cands} state_lb[r]
        # Since max_{r in cands} x_r >= x_{r*}, x_{r*} is a 100% sound linear lower bound!
        r_lb = cands[int(np.argmax(state_lb[cands]))]
        best_lb_idx[i] = r_lb

        # Sound upper bound: for r** = argmax_{r in cands} state_ub[r],
        # notice that for all r in cands, x_r <= x_{r**} + max_{r' in cands} (u_{r'} - l_{r**})_+
        # and also max_{r} x_r <= ub_new[i]. We pick whichever upper bound has smaller mean!
        r_ub = cands[int(np.argmax(state_ub[cands]))]
        best_ub_idx[i] = r_ub
        other_cands = cands[cands != r_ub]
        if len(other_cands) > 0:
            slack = max(0.0, float(np.max(state_ub[other_cands]) - state_lb[r_ub]))
        else:
            slack = 0.0
        ub_slack[i] = slack

        # Clark's recursive max-of-Gaussians moment approximation over candidates
        pos0 = cand_pos[int(cands[0])]
        m_cur = float(mu_cands[pos0])
        v_cur = float(var_cands[pos0])
        for c_idx in cands[1:]:
            pos_k = cand_pos[int(c_idx)]
            m_k = float(mu_cands[pos_k])
            v_k = float(var_cands[pos_k])
            spread = np.sqrt(max(v_cur + v_k, _MIN_VAR))
            alpha_c = (m_cur - m_k) / spread
            Phi_c = float(std_normal_cdf(alpha_c))
            phi_c = float(std_normal_pdf(alpha_c))
            m_next = m_cur * Phi_c + m_k * (1.0 - Phi_c) + spread * phi_c
            ex2_next = (
                (m_cur ** 2 + v_cur) * Phi_c +
                (m_k ** 2 + v_k) * (1.0 - Phi_c) +
                (m_cur + m_k) * spread * phi_c
            )
            v_next = max(ex2_next - m_next ** 2, _MIN_VAR)
            m_cur, v_cur = m_next, v_next

        mu_beta[i] = np.clip(m_cur, lb_new[i], ub_new[i])
        var_beta[i] = min(v_cur, max(0.25 * (ub_new[i] - lb_new[i]) ** 2, _MIN_VAR))

    new_mu = np.concatenate([mu, mu_beta]).astype(dtype, copy=False)
    if cov_mode == 'diagonal' or Sig.ndim == 1:
        var_prev = Sig if Sig.ndim == 1 else np.diag(Sig)
        new_Sig = np.concatenate([var_prev, var_beta]).astype(dtype, copy=False)
    else:
        # Cross-covariance approximated along dominant candidate
        dom_pos = [cand_pos[int(idx)] for idx in best_ub_idx]
        cross_cov = V_Sig_cands[dom_pos, :].T
        dom_var = var_cands[dom_pos]
        diag_beta = np.maximum(var_beta, dom_var + 1e-10)
        new_Sig = np.vstack([
            np.hstack([Sig, cross_cov]),
            np.hstack([cross_cov.T, np.diag(diag_beta)])
        ]).astype(dtype, copy=False)

    # Construct explicit sound lower and upper linear bounds in terms of preceding predicates:
    # Lower bound: beta_i >= c[r_lb] + V[r_lb, :] @ alpha_prev
    # Upper bound: either constant beta_i <= ub_new[i] (0 * alpha_prev + ub_new[i])
    # or shifted dominant candidate beta_i <= (c[r_ub] + slack) + V[r_ub, :] @ alpha_prev
    G_lb = V[best_lb_idx]
    gamma_lb = center[best_lb_idx].astype(np.float64)

    # Use constant upper bound ub_new[i] when slack makes shifted candidate looser than ub_new[i]
    dom_pos_arr = np.array([cand_pos[int(idx)] for idx in best_ub_idx], dtype=np.int64)
    use_shifted = (mu_cands[dom_pos_arr] + ub_slack) < ub_new
    G_ub_raw = V[best_ub_idx]
    if sp.issparse(G_ub_raw):
        G_ub = G_ub_raw.multiply(use_shifted.astype(dtype)[:, None]).tocsr()
    else:
        G_ub = G_ub_raw * use_shifted.astype(dtype)[:, None]
    gamma_ub = np.where(use_shifted, center[best_ub_idx] + ub_slack, ub_new).astype(np.float64)

    env_record = RelaxationEnvelopeRecord(
        n_prev=n_prev,
        n_new=nPred,
        G_lb=G_lb,
        gamma_lb=gamma_lb,
        G_ub=G_ub,
        gamma_ub=gamma_ub,
    )
    return new_mu, new_Sig, env_record


def backsubstitute_linear_bounds(c_out, V_out, env_dag, nVars0, U=None, u_spec=None):
    """
    Propagate linear specification z = U @ (c_out + V_out @ alpha) - u_spec
    backward through the RelaxationEnvelopeRecord DAG `env_dag` down to the
    original input Gaussian predicate variables alpha_0 in R^{nVars0}.

    Guarantees point-wise soundness for every alpha_0 in [pred_lb0, pred_ub0]:
        d_lb + W_lb @ alpha_0 <= U @ f(x(alpha_0)) - u_spec <= d_ub + W_ub @ alpha_0

    Args:
        c_out: 1D array of shape (dim,)
        V_out: 2D array or sparse matrix of shape (dim, nVars)
        env_dag: list of RelaxationEnvelopeRecord
        nVars0: number of original input predicates
        U: optional 2D array (K_spec, dim); if None, uses identity
        u_spec: optional 1D array (K_spec,); if None, uses zeros

    Returns:
        d_lb, W_lb, d_ub, W_ub
        where d_lb, d_ub have shape (K_spec,) and W_lb, W_ub have shape (K_spec, nVars0).
    """
    c_out = np.asarray(c_out, dtype=np.float64).reshape(-1)
    if U is not None:
        U = np.asarray(U, dtype=np.float64)
        if U.ndim == 1:
            U = U.reshape(1, -1)
        d_cur = U @ c_out
        if sp.issparse(V_out):
            W_cur = np.asarray(U @ V_out, dtype=np.float64)
        else:
            W_cur = np.asarray(U @ V_out, dtype=np.float64)
    else:
        d_cur = c_out.copy()
        W_cur = V_out.toarray().astype(np.float64) if sp.issparse(V_out) else np.asarray(V_out, dtype=np.float64).copy()

    if u_spec is not None:
        d_cur = d_cur - np.asarray(u_spec, dtype=np.float64).reshape(-1)

    d_lb = d_cur.copy()
    W_lb = W_cur.copy()
    d_ub = d_cur.copy()
    W_ub = W_cur.copy()

    for rec in reversed(env_dag):
        n_prev = rec.n_prev
        n_new = rec.n_new
        if W_lb.shape[1] < n_prev + n_new:
            continue

        # Slice out coefficients on beta (columns n_prev : n_prev + n_new)
        W_lb_prev = W_lb[:, :n_prev]
        W_lb_beta = W_lb[:, n_prev:n_prev + n_new]
        W_lb_pos = np.maximum(W_lb_beta, 0.0)
        W_lb_neg = np.minimum(W_lb_beta, 0.0)

        W_ub_prev = W_ub[:, :n_prev]
        W_ub_beta = W_ub[:, n_prev:n_prev + n_new]
        W_ub_pos = np.maximum(W_ub_beta, 0.0)
        W_ub_neg = np.minimum(W_ub_beta, 0.0)

        d_lb = d_lb + W_lb_pos @ rec.gamma_lb + W_lb_neg @ rec.gamma_ub
        d_ub = d_ub + W_ub_pos @ rec.gamma_ub + W_ub_neg @ rec.gamma_lb

        if rec.is_explicit:
            W_lb = W_lb_prev + (W_lb_pos @ rec.G_lb) + (W_lb_neg @ rec.G_ub)
            W_ub = W_ub_prev + (W_ub_pos @ rec.G_ub) + (W_ub_neg @ rec.G_lb)
        else:
            # G_lb = diag(a_lb) @ V_pre, G_ub = diag(a_ub) @ V_pre
            coeff_lb = W_lb_pos * rec.a_lb[None, :] + W_lb_neg * rec.a_ub[None, :]
            coeff_ub = W_ub_pos * rec.a_ub[None, :] + W_ub_neg * rec.a_lb[None, :]
            W_lb = W_lb_prev + (coeff_lb @ rec.V_pre)
            W_ub = W_ub_prev + (coeff_ub @ rec.V_pre)

    # Ensure final width matches nVars0
    if W_lb.shape[1] > nVars0:
        W_lb = W_lb[:, :nVars0]
        W_ub = W_ub[:, :nVars0]

    return d_lb, W_lb, d_ub, W_ub


def verify_prob_spec(
    star_set,
    unsafe_mat,
    unsafe_vec,
    lp_solver='gurobi',
    use_lp_tightening=True
):
    """
    Two-Stage Quantitative Verification of a specification:
        Unsafe region U = { y | unsafe_mat @ y <= unsafe_vec }
    on a ProbSparseImageStar2DCSR, ProbSparseImageStar2DCOO, or ProbStar `star_set`.

    Returns dict with:
        - 'status': 'UNSAT' (provably safe, p=0), 'SAT_ALL' (all points unsafe, p=p_in),
                    or 'AMBIGUOUS' (partially unsafe)
        - 'p_min': provably sound lower bound on P(f(x) in U and x in InputBox)
        - 'p_max': provably sound upper bound on P(f(x) in U and x in InputBox)
        - 'p_hat': analytic moment-matched estimate of P(f(x) in U and x in InputBox)
        - 'p_input': probability mass of the input box P(x in InputBox)
        - 'cond_p_min': conditional lower bound P(f(x) in U | x in InputBox)
        - 'cond_p_max': conditional upper bound P(f(x) in U | x in InputBox)
        - 'cond_p_hat': conditional moment-matched estimate P(f(x) in U | x in InputBox)
    """
    U = np.asarray(unsafe_mat, dtype=np.float64)
    if U.ndim == 1:
        U = U.reshape(1, -1)
    u_vec = np.asarray(unsafe_vec, dtype=np.float64).reshape(-1)
    K_spec = U.shape[0]

    c_out, V_out = star_set.get_c_V()
    p_input = star_set.get_input_probability()

    # Specification functional z = U @ y - u_vec <= 0
    d_spec = U @ c_out - u_vec
    W_spec = np.asarray(U @ V_out, dtype=np.float64)

    # Stage 1a: Fast interval estimate on z over pred_lb <= alpha <= pred_ub
    W_pos = np.maximum(W_spec, 0.0)
    W_neg = np.minimum(W_spec, 0.0)
    z_lb_est = d_spec + W_pos @ star_set.pred_lb + W_neg @ star_set.pred_ub
    z_ub_est = d_spec + W_pos @ star_set.pred_ub + W_neg @ star_set.pred_lb

    # If any conjunctive row k has z_lb_est[k] > 0, the unsafe region is unreachable!
    if np.any(z_lb_est > 0.0):
        return {
            'status': 'UNSAT',
            'p_min': 0.0, 'p_max': 0.0, 'p_hat': 0.0, 'p_input': p_input,
            'cond_p_min': 0.0, 'cond_p_max': 0.0, 'cond_p_hat': 0.0,
        }

    # If all conjunctive rows have z_ub_est[k] <= 0, the entire input set is unsafe!
    if np.all(z_ub_est <= 0.0):
        return {
            'status': 'SAT_ALL',
            'p_min': p_input, 'p_max': p_input, 'p_hat': p_input, 'p_input': p_input,
            'cond_p_min': 1.0, 'cond_p_max': 1.0, 'cond_p_hat': 1.0,
        }

    # Stage 1b: LP tightening on the Star polytope C @ alpha <= d (if constraints exist)
    z_lb = z_lb_est.copy()
    z_ub = z_ub_est.copy()
    if use_lp_tightening and lp_solver != 'estimate' and star_set.has_constraints():
        from StarV.set.star import _solve_lp, _resolve_lp_solver
        solver_name = _resolve_lp_solver(lp_solver)
        for k in range(K_spec):
            if z_lb[k] <= 0.0 < z_ub[k]:
                f_k = W_spec[k, :]
                if np.any(f_k != 0.0):
                    try:
                        z_lb[k] = _solve_lp(
                            f_k, d_spec[k], star_set.C, star_set.d,
                            star_set.pred_lb, star_set.pred_ub, star_set.num_pred,
                            sense='min', lp_solver=solver_name, gurobi_opt_tol=1e-6
                        )
                        if z_lb[k] > 0.0:
                            return {
                                'status': 'UNSAT',
                                'p_min': 0.0, 'p_max': 0.0, 'p_hat': 0.0, 'p_input': p_input,
                                'cond_p_min': 0.0, 'cond_p_max': 0.0, 'cond_p_hat': 0.0,
                            }
                        z_ub[k] = _solve_lp(
                            f_k, d_spec[k], star_set.C, star_set.d,
                            star_set.pred_lb, star_set.pred_ub, star_set.num_pred,
                            sense='max', lp_solver=solver_name, gurobi_opt_tol=1e-6
                        )
                    except Exception:
                        pass

        if np.all(z_ub <= 0.0):
            return {
                'status': 'SAT_ALL',
                'p_min': p_input, 'p_max': p_input, 'p_hat': p_input, 'p_input': p_input,
                'cond_p_min': 1.0, 'cond_p_max': 1.0, 'cond_p_hat': 1.0,
            }

    # Stage 2a: Sound Dual Back-Substitution Bounds [p_min, p_max] in alpha_0 ~ N(mu0, Sig0)
    d_lb0, W_lb0, d_ub0, W_ub0 = backsubstitute_linear_bounds(
        c_out, V_out, star_set.env_dag, star_set.nVars0, U=U, u_spec=u_vec
    )
    mu0 = star_set.mu0
    Sig0 = star_set.Sig0
    lb0 = star_set.pred_lb0
    ub0 = star_set.pred_ub0

    # Interval check on back-substituted forms over [lb0, ub0]
    W_lb0_pos = np.maximum(W_lb0, 0.0)
    W_lb0_neg = np.minimum(W_lb0, 0.0)
    min_z_under = np.maximum(z_lb, d_lb0 + W_lb0_pos @ lb0 + W_lb0_neg @ ub0)
    max_z_under = d_lb0 + W_lb0_pos @ ub0 + W_lb0_neg @ lb0

    W_ub0_pos = np.maximum(W_ub0, 0.0)
    W_ub0_neg = np.minimum(W_ub0, 0.0)
    min_z_over = d_ub0 + W_ub0_pos @ lb0 + W_ub0_neg @ ub0
    max_z_over = np.minimum(z_ub, d_ub0 + W_ub0_pos @ ub0 + W_ub0_neg @ lb0)

    if np.any(min_z_under > 0.0):
        return {
            'status': 'UNSAT',
            'p_min': 0.0, 'p_max': 0.0, 'p_hat': 0.0, 'p_input': p_input,
            'cond_p_min': 0.0, 'cond_p_max': 0.0, 'cond_p_hat': 0.0,
        }
    if np.all(max_z_over <= 0.0):
        return {
            'status': 'SAT_ALL',
            'p_min': p_input, 'p_max': p_input, 'p_hat': p_input, 'p_input': p_input,
            'cond_p_min': 1.0, 'cond_p_max': 1.0, 'cond_p_hat': 1.0,
        }

    # Exact conditional moments of alpha_0 | alpha_0 in [lb0, ub0]
    var0_vec = Sig0 if Sig0.ndim == 1 else np.diag(Sig0)
    mu0_trunc, var0_trunc = compute_truncated_gaussian_moments_1d(mu0, var0_vec, lb0, ub0)
    if Sig0.ndim == 1:
        Sig0_trunc = var0_trunc
        cov_mode0 = 'diagonal'
    else:
        scale0 = np.sqrt(var0_trunc / np.maximum(var0_vec, _MIN_VAR))
        Sig0_trunc = (Sig0 * scale0[:, None]) * scale0[None, :]
        cov_mode0 = 'full'

    # Conditional moments of under-approximator z_lb0(alpha_0) and over-approximator z_ub0(alpha_0)
    mu_z_lb, var_z_lb, V_Sig_lb = compute_preactivation_moments(
        W_lb0, d_lb0, mu0_trunc, Sig0_trunc, cov_mode=cov_mode0
    )
    if K_spec == 1:
        Sig_z_lb = var_z_lb
    elif V_Sig_lb is not None:
        Sig_z_lb = V_Sig_lb @ W_lb0.T
    else:
        Sig_z_lb = (W_lb0 * Sig0_trunc[None, :]) @ W_lb0.T

    mu_z_ub, var_z_ub, V_Sig_ub = compute_preactivation_moments(
        W_ub0, d_ub0, mu0_trunc, Sig0_trunc, cov_mode=cov_mode0
    )
    if K_spec == 1:
        Sig_z_ub = var_z_ub
    elif V_Sig_ub is not None:
        Sig_z_ub = V_Sig_ub @ W_ub0.T
    else:
        Sig_z_ub = (W_ub0 * Sig0_trunc[None, :]) @ W_ub0.T

    neg_inf = np.full(K_spec, -np.inf, dtype=np.float64)
    zeros_k = np.zeros(K_spec, dtype=np.float64)

    # Conditional upper bound P(z(alpha_0) <= 0 | alpha_0 in B) <= P(z_lb0(alpha_0) <= 0 | alpha_0 in B)
    if np.all(max_z_under <= 0.0):
        cond_p_max = 1.0
    elif np.any(min_z_under > 0.0):
        cond_p_max = 0.0
    else:
        tot_under = mvn_cdf_lowdim(min_z_under, max_z_under, mu_z_lb, Sig_z_lb)
        uns_under = mvn_cdf_lowdim(min_z_under, np.minimum(zeros_k, max_z_under), mu_z_lb, Sig_z_lb)
        cond_under = uns_under / tot_under if tot_under > 1e-12 else mvn_cdf_lowdim(neg_inf, zeros_k, mu_z_lb, Sig_z_lb)
        # Also compute untruncated Boole-Frechet upper bound when p_input is close to 1
        mu_z_lb_raw, var_z_lb_raw, V_Sig_lb_raw = compute_preactivation_moments(
            W_lb0, d_lb0, mu0, Sig0, cov_mode=cov_mode0
        )
        Sig_z_lb_raw = var_z_lb_raw if K_spec == 1 else (
            V_Sig_lb_raw @ W_lb0.T if V_Sig_lb_raw is not None else (W_lb0 * Sig0[None, :]) @ W_lb0.T
        )
        bf_cond_max = mvn_cdf_lowdim(neg_inf, zeros_k, mu_z_lb_raw, Sig_z_lb_raw) / max(p_input, 1e-15)
        cond_p_max = float(np.clip(max(cond_under, min(1.0, bf_cond_max)), 0.0, 1.0))

    # Conditional lower bound P(z(alpha_0) <= 0 | alpha_0 in B) >= P(z_ub0(alpha_0) <= 0 | alpha_0 in B)
    if np.any(min_z_over > 0.0):
        cond_p_min = 0.0
    elif np.all(max_z_over <= 0.0):
        cond_p_min = 1.0
    else:
        tot_over = mvn_cdf_lowdim(min_z_over, max_z_over, mu_z_ub, Sig_z_ub)
        uns_over = mvn_cdf_lowdim(min_z_over, np.minimum(zeros_k, max_z_over), mu_z_ub, Sig_z_ub)
        cond_over = uns_over / tot_over if tot_over > 1e-12 else mvn_cdf_lowdim(neg_inf, zeros_k, mu_z_ub, Sig_z_ub)
        mu_z_ub_raw, var_z_ub_raw, V_Sig_ub_raw = compute_preactivation_moments(
            W_ub0, d_ub0, mu0, Sig0, cov_mode=cov_mode0
        )
        Sig_z_ub_raw = var_z_ub_raw if K_spec == 1 else (
            V_Sig_ub_raw @ W_ub0.T if V_Sig_ub_raw is not None else (W_ub0 * Sig0[None, :]) @ W_ub0.T
        )
        bf_cond_min = max(0.0, mvn_cdf_lowdim(neg_inf, zeros_k, mu_z_ub_raw, Sig_z_ub_raw) - (1.0 - p_input)) / max(p_input, 1e-15)
        cond_p_min = float(np.clip(min(cond_over, bf_cond_min) if p_input > 0.5 else cond_over, 0.0, cond_p_max))

    p_min = float(np.clip(p_input * cond_p_min, 0.0, p_input))
    p_max = float(np.clip(p_input * cond_p_max, p_min, p_input))

    # Stage 2b: Extended-Predicate Analytic Moment-Matched Probability Estimate p_hat
    mu_z_ext, var_z_ext, V_Sig_ext = compute_preactivation_moments(
        W_spec, d_spec, star_set.mu, star_set.Sig, cov_mode=star_set.cov_mode
    )
    if K_spec == 1:
        Sig_z_ext = var_z_ext
    elif V_Sig_ext is not None:
        Sig_z_ext = V_Sig_ext @ W_spec.T
    else:
        Sig_z_ext = (W_spec * star_set.Sig[None, :]) @ W_spec.T

    # Truncate extended output Gaussian to LP/interval range [z_lb, z_ub]
    total_ext_box = mvn_cdf_lowdim(z_lb, z_ub, mu_z_ext, Sig_z_ext)
    unsafe_ext_box = mvn_cdf_lowdim(z_lb, np.minimum(zeros_k, z_ub), mu_z_ext, Sig_z_ext)
    if total_ext_box > 1e-12:
        cond_ext = float(np.clip(unsafe_ext_box / total_ext_box, 0.0, 1.0))
    else:
        cond_ext = float(np.clip(mvn_cdf_lowdim(neg_inf, zeros_k, mu_z_ext, Sig_z_ext), 0.0, 1.0))

    cond_p_hat = float(np.clip(cond_ext, cond_p_min, cond_p_max))
    p_hat = float(np.clip(p_input * cond_p_hat, p_min, p_max))

    return {
        'status': 'AMBIGUOUS',
        'p_min': p_min,
        'p_max': p_max,
        'p_hat': p_hat,
        'p_input': p_input,
        'cond_p_min': cond_p_min,
        'cond_p_max': cond_p_max,
        'cond_p_hat': cond_p_hat,
    }
