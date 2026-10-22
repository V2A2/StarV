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
CROWN, alpha-CROWN, and alpha,beta-CROWN Verifier & Reachability Engine for StarV

Implements Linear Relaxation based Perturbation Analysis (LiRPA) and Branch-and-Bound:
  1. Classic CROWN:
     Backward linear bound propagation with closed-form adaptive activation relaxations
     across feedforward, convolutional, pooling, normalization, residual (DAG), and
     Vision Transformer (LayerNorm, GELU, Multi-Head Self-Attention) architectures.
  2. alpha-CROWN:
     Gradient-optimized activation relaxation slopes (specification-dependent alpha in [0, 1]
     for ReLU/LeakyReLU and simplex weights for MaxPool2D) with intermediate bound tightening.
  3. alpha,beta-CROWN:
     Complete verification combining alpha-CROWN with Branch-and-Bound (BaB), BaBSR smart
     branching over unstable ReLU neurons (plus input-split fallback), Lagrangian multipliers
     beta >= 0 for split constraints, and PGD counterexample search.
  4. StarV Set & Quantitative Probabilistic Integration:
     - Constructs exact Star, ImageStar, and ProbStar reachable sets encoding CROWN's
       end-to-end linear bounds A_lb * alpha_0 + b_lb <= f(x(alpha_0)) <= A_ub * alpha_0 + b_ub.
     - Propagates truncated Gaussian input distributions alpha_0 ~ N(mu_0, Sig_0) through
       CROWN / alpha-CROWN / alpha,beta-CROWN linear bounds to compute provably sound
       quantitative probability intervals [p_min, p_max] and moment-matched estimates p_hat.

Author: Hoang Dung Tran & StarV Team
"""

import time
import copy
import heapq
import numpy as np
import scipy.sparse as sp
import torch

from StarV.set.star import Star, _resolve_lp_solver, _solve_lp
from StarV.set.probstar import ProbStar
from StarV.set.sparsestar import SparseStar
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
from StarV.set.probsparseimagestar2dcoo import ProbSparseImageStar2DCOO
from StarV.set.probsparse_base import (
    RelaxationEnvelopeRecord,
    compute_box_input_probability,
    compute_truncated_gaussian_moments_1d,
    compute_preactivation_moments,
    mvn_cdf_lowdim,
    verify_prob_spec,
)

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
from StarV.layer.GELULayer import GELULayer, compute_gelu_linear_bounds
from StarV.layer.MultiHeadAttentionLayer import MultiHeadAttentionLayer, _softmax_last_axis
from StarV.fun.logsig import LogSig
from StarV.fun.tansig import TanSig


def _normalize_crown_mode(mode):
    """Normalize CROWN mode string into 'crown', 'alpha-crown', or 'alpha-beta-crown'."""
    if mode is None:
        return 'crown'
    m = str(mode).strip().lower().replace('_', '-')
    if m in ('crown', 'classic-crown', 'lirpa', 'backward'):
        return 'crown'
    if m in ('alpha-crown', 'acrown', 'a-crown'):
        return 'alpha-crown'
    if m in ('alpha-beta-crown', 'ab-crown', 'abcrown', 'beta-crown', 'bcrown', 'b-crown'):
        return 'alpha-beta-crown'
    return 'crown'


# =====================================================================
# 1. Sound Elementwise Linear Relaxations
# =====================================================================

def _relu_linear_relaxation(l, u, slope_mode='adaptive', custom_alpha_l=None):
    """
    Compute sound linear bounds a_lb * x + b_lb <= ReLU(x) <= a_ub * x + b_ub on [l, u].
    """
    l = np.asarray(l, dtype=np.float64).ravel()
    u = np.asarray(u, dtype=np.float64).ravel()
    n = len(l)

    a_lb = np.zeros(n, dtype=np.float64)
    b_lb = np.zeros(n, dtype=np.float64)
    a_ub = np.zeros(n, dtype=np.float64)
    b_ub = np.zeros(n, dtype=np.float64)

    active = l >= 0.0
    a_lb[active] = 1.0
    a_ub[active] = 1.0

    unstable = (l < 0.0) & (u > 0.0)
    if np.any(unstable):
        lu = l[unstable]
        uu = u[unstable]
        denom = np.maximum(uu - lu, 1e-15)
        slope_u = uu / denom
        a_ub[unstable] = slope_u
        b_ub[unstable] = -slope_u * lu

        if custom_alpha_l is not None:
            a_lb[unstable] = np.clip(np.asarray(custom_alpha_l, dtype=np.float64).ravel(), 0.0, 1.0)
        elif slope_mode == 'zero':
            a_lb[unstable] = 0.0
        elif slope_mode == 'one':
            a_lb[unstable] = 1.0
        else:
            a_lb[unstable] = (uu >= np.abs(lu)).astype(np.float64)

    return a_lb, b_lb, a_ub, b_ub


def _leaky_relu_linear_relaxation(l, u, gamma=0.1, custom_alpha_l=None):
    """
    Compute sound linear bounds for LeakyReLU(x; gamma) = max(gamma * x, x) on [l, u].
    Assumes 0 <= gamma <= 1.
    """
    l = np.asarray(l, dtype=np.float64).ravel()
    u = np.asarray(u, dtype=np.float64).ravel()
    gamma = float(gamma)
    n = len(l)

    a_lb = np.full(n, gamma, dtype=np.float64)
    b_lb = np.zeros(n, dtype=np.float64)
    a_ub = np.full(n, gamma, dtype=np.float64)
    b_ub = np.zeros(n, dtype=np.float64)

    active = l >= 0.0
    a_lb[active] = 1.0
    a_ub[active] = 1.0

    unstable = (l < 0.0) & (u > 0.0)
    if np.any(unstable):
        lu = l[unstable]
        uu = u[unstable]
        denom = np.maximum(uu - lu, 1e-15)
        a_ub[unstable] = (uu - gamma * lu) / denom
        b_ub[unstable] = (1.0 - gamma) * uu * (-lu) / denom
        if custom_alpha_l is not None:
            a_lb[unstable] = np.clip(np.asarray(custom_alpha_l, dtype=np.float64).ravel(), gamma, 1.0)
        else:
            a_lb[unstable] = np.where(uu >= np.abs(lu), 1.0, gamma)

    return a_lb, b_lb, a_ub, b_ub


def _satlin_linear_relaxation(l, u):
    """
    Compute sound linear bounds for SatLin(x) = min(1, max(0, x)) on [l, u].
    """
    l = np.asarray(l, dtype=np.float64).ravel()
    u = np.asarray(u, dtype=np.float64).ravel()
    n = len(l)

    a_lb = np.zeros(n, dtype=np.float64)
    b_lb = np.zeros(n, dtype=np.float64)
    a_ub = np.zeros(n, dtype=np.float64)
    b_ub = np.zeros(n, dtype=np.float64)

    # Case 1: l >= 1 -> y = 1
    c1 = l >= 1.0
    b_lb[c1] = 1.0
    b_ub[c1] = 1.0

    # Case 2: 0 <= l and u <= 1 -> y = x
    c2 = (l >= 0.0) & (u <= 1.0)
    a_lb[c2] = 1.0
    a_ub[c2] = 1.0

    # Case 3: l < 0 < u <= 1 -> ReLU on [l, u]
    c3 = (l < 0.0) & (u > 0.0) & (u <= 1.0)
    if np.any(c3):
        lu, uu = l[c3], u[c3]
        denom = np.maximum(uu - lu, 1e-15)
        a_ub[c3] = uu / denom
        b_ub[c3] = -a_ub[c3] * lu
        a_lb[c3] = (uu >= np.abs(lu)).astype(np.float64)

    # Case 4: 0 <= l < 1 < u -> concave
    c4 = (l >= 0.0) & (l < 1.0) & (u > 1.0)
    if np.any(c4):
        lu, uu = l[c4], u[c4]
        denom = np.maximum(uu - lu, 1e-15)
        a_lb[c4] = (1.0 - lu) / denom
        b_lb[c4] = lu - a_lb[c4] * lu
        use_id = (1.0 - lu) >= (uu - 1.0)
        a_ub[c4] = np.where(use_id, 1.0, 0.0)
        b_ub[c4] = np.where(use_id, 0.0, 1.0)

    # Case 5: l < 0 < 1 < u
    c5 = (l < 0.0) & (u > 1.0)
    if np.any(c5):
        lu, uu = l[c5], u[c5]
        a_ub[c5] = 1.0 / np.maximum(1.0 - lu, 1e-15)
        b_ub[c5] = -lu * a_ub[c5]
        a_lb[c5] = 1.0 / np.maximum(uu, 1e-15)
        b_lb[c5] = 0.0

    return a_lb, b_lb, a_ub, b_ub


def _satlins_linear_relaxation(l, u):
    """
    Compute sound linear bounds for SatLins(x) = min(1, max(-1, x)) on [l, u].
    """
    l = np.asarray(l, dtype=np.float64).ravel()
    u = np.asarray(u, dtype=np.float64).ravel()
    n = len(l)

    a_lb = np.zeros(n, dtype=np.float64)
    b_lb = np.zeros(n, dtype=np.float64)
    a_ub = np.zeros(n, dtype=np.float64)
    b_ub = np.zeros(n, dtype=np.float64)

    # Case 0: u <= -1 -> y = -1
    c0 = u <= -1.0
    b_lb[c0] = -1.0
    b_ub[c0] = -1.0

    # Case 1: l >= 1 -> y = 1
    c1 = l >= 1.0
    b_lb[c1] = 1.0
    b_ub[c1] = 1.0

    # Case 2: -1 <= l and u <= 1 -> y = x
    c2 = (l >= -1.0) & (u <= 1.0)
    a_lb[c2] = 1.0
    a_ub[c2] = 1.0

    # Case 3: l < -1 < u <= 1 -> convex
    c3 = (l < -1.0) & (u > -1.0) & (u <= 1.0)
    if np.any(c3):
        lu, uu = l[c3], u[c3]
        denom = np.maximum(uu - lu, 1e-15)
        a_ub[c3] = (uu + 1.0) / denom
        b_ub[c3] = -1.0 - a_ub[c3] * lu
        use_id = (uu + 1.0) >= (-1.0 - lu)
        a_lb[c3] = np.where(use_id, 1.0, 0.0)
        b_lb[c3] = np.where(use_id, 0.0, -1.0)

    # Case 4: -1 <= l < 1 < u -> concave
    c4 = (l >= -1.0) & (l < 1.0) & (u > 1.0)
    if np.any(c4):
        lu, uu = l[c4], u[c4]
        denom = np.maximum(uu - lu, 1e-15)
        a_lb[c4] = (1.0 - lu) / denom
        b_lb[c4] = lu - a_lb[c4] * lu
        use_id = (1.0 - lu) >= (uu - 1.0)
        a_ub[c4] = np.where(use_id, 1.0, 0.0)
        b_ub[c4] = np.where(use_id, 0.0, 1.0)

    # Case 5: l < -1 < 1 < u
    c5 = (l < -1.0) & (u > 1.0)
    if np.any(c5):
        lu, uu = l[c5], u[c5]
        a_ub[c5] = 2.0 / np.maximum(1.0 - lu, 1e-15)
        b_ub[c5] = 1.0 - a_ub[c5]
        a_lb[c5] = 2.0 / np.maximum(uu + 1.0, 1e-15)
        b_lb[c5] = -1.0 + a_lb[c5]

    return a_lb, b_lb, a_ub, b_ub


def _s_curve_linear_relaxation(l, u, fn_f, fn_df):
    """
    Sound linear relaxation for S-shaped activations (Sigmoid, Tanh) whose
    second derivative is >= 0 for x <= 0 (convex) and <= 0 for x >= 0 (concave).
    Matches StarV's LogSig / TanSig primary relaxation in vectorized 1D form.
    """
    l = np.asarray(l, dtype=np.float64).ravel()
    u = np.asarray(u, dtype=np.float64).ravel()
    yl = fn_f(l)
    yu = fn_f(u)
    dyl = fn_df(l)
    dyu = fn_df(u)

    n = len(l)
    a_lb = np.zeros(n, dtype=np.float64)
    a_ub = np.zeros(n, dtype=np.float64)
    b_lb = yl.copy()
    b_ub = yu.copy()

    width = u - l
    non_deg = width > 1e-12

    # Region 1: l >= 0 (concave) -> lower bound is chord, upper bound is tangent
    m1 = (l >= 0.0) & non_deg
    if np.any(m1):
        a_lb[m1] = (yu[m1] - yl[m1]) / width[m1]
        a_ub[m1] = dyu[m1]
        b_lb[m1] = yl[m1] - a_lb[m1] * l[m1]
        b_ub[m1] = yu[m1] - a_ub[m1] * u[m1]

    # Region 2: u <= 0 (convex) -> lower bound is tangent, upper bound is chord
    m2 = (u <= 0.0) & non_deg
    if np.any(m2):
        a_lb[m2] = dyl[m2]
        a_ub[m2] = (yu[m2] - yl[m2]) / width[m2]
        b_lb[m2] = yl[m2] - a_lb[m2] * l[m2]
        b_ub[m2] = yu[m2] - a_ub[m2] * u[m2]

    # Region 3: l < 0 < u -> min endpoint slope is sound at both endpoints
    m3 = (l < 0.0) & (u > 0.0)
    if np.any(m3):
        m_slope = np.minimum(dyl[m3], dyu[m3])
        a_lb[m3] = m_slope
        a_ub[m3] = m_slope
        b_lb[m3] = yl[m3] - m_slope * l[m3]
        b_ub[m3] = yu[m3] - m_slope * u[m3]

    return a_lb, b_lb, a_ub, b_ub


# =====================================================================
# 2. Input Representation Extraction & Star/ImageStar/ProbStar Builder
# =====================================================================

class _InputSpec(object):
    """Normalized representation of a StarV input set for CROWN."""

    def __init__(self, in_set, in_shape_hint=None):
        self.orig_set = in_set
        self.is_prob = isinstance(
            in_set, (ProbStar, ProbSparseImageStar2DCSR, ProbSparseImageStar2DCOO)
        )
        self.is_image = isinstance(
            in_set,
            (
                ImageStar,
                SparseImageStar2DCSR,
                SparseImageStar2DCOO,
                ProbSparseImageStar2DCSR,
                ProbSparseImageStar2DCOO,
            ),
        )

        if isinstance(in_set, ImageStar):
            self.in_shape = (in_set.height, in_set.width, in_set.num_channel)
            V_flat = in_set.V.reshape(in_set.num_pixel, in_set.num_pred + 1).astype(np.float64)
            self.c0_full = V_flat[:, 0].copy()
            self.V0_full = V_flat[:, 1:].copy()
            self.pred_lb_full = np.asarray(in_set.pred_lb, dtype=np.float64).ravel()
            self.pred_ub_full = np.asarray(in_set.pred_ub, dtype=np.float64).ravel()
            self.C0_full = in_set.C
            self.d0_full = in_set.d
            self.mu0_full = None
            self.Sig0_full = None
            self.cov_mode = None

        elif isinstance(
            in_set,
            (
                SparseImageStar2DCSR,
                SparseImageStar2DCOO,
                ProbSparseImageStar2DCSR,
                ProbSparseImageStar2DCOO,
            ),
        ):
            self.in_shape = tuple(in_set.shape) if in_set.shape is not None else in_shape_hint
            if self.in_shape is not None and len(self.in_shape) == 2:
                self.in_shape = (self.in_shape[0], self.in_shape[1], 1)
            c0, V0 = in_set.get_c_V() if hasattr(in_set, 'get_c_V') else (in_set.c, in_set.V)
            self.c0_full = np.asarray(c0, dtype=np.float64).ravel()
            self.V0_full = V0.tocsr() if sp.issparse(V0) else np.asarray(V0, dtype=np.float64)
            self.pred_lb_full = np.asarray(in_set.pred_lb, dtype=np.float64).ravel()
            self.pred_ub_full = np.asarray(in_set.pred_ub, dtype=np.float64).ravel()
            self.C0_full = in_set.C
            self.d0_full = in_set.d
            if self.is_prob:
                self.mu0_full = np.asarray(in_set.mu0, dtype=np.float64).ravel()
                self.Sig0_full = np.asarray(in_set.Sig0, dtype=np.float64)
                self.cov_mode = getattr(in_set, 'cov_mode', 'diagonal')
            else:
                self.mu0_full = None
                self.Sig0_full = None
                self.cov_mode = None

        elif isinstance(in_set, SparseStar):
            self.in_shape = in_shape_hint if in_shape_hint is not None else (in_set.dim,)
            self.c0_full = np.asarray(in_set.A[:, 0], dtype=np.float64).ravel()
            self.V0_full = np.asarray(in_set.A[:, 1:], dtype=np.float64)
            nI = in_set.nIVars
            self.pred_lb_full = np.asarray(in_set.pred_lb[-nI:], dtype=np.float64).ravel()
            self.pred_ub_full = np.asarray(in_set.pred_ub[-nI:], dtype=np.float64).ravel()
            self.C0_full = in_set.C[:, -nI:] if (in_set.C is not None and len(in_set.d) > 0) else np.empty((0, nI))
            self.d0_full = in_set.d
            self.mu0_full = None
            self.Sig0_full = None
            self.cov_mode = None

        elif isinstance(in_set, (Star, ProbStar)):
            self.in_shape = in_shape_hint if in_shape_hint is not None else (in_set.dim,)
            self.c0_full = np.asarray(in_set.V[:, 0], dtype=np.float64).ravel()
            self.V0_full = np.asarray(in_set.V[:, 1 : in_set.nVars + 1], dtype=np.float64)
            self.pred_lb_full = np.asarray(in_set.pred_lb, dtype=np.float64).ravel()
            self.pred_ub_full = np.asarray(in_set.pred_ub, dtype=np.float64).ravel()
            self.C0_full = in_set.C
            self.d0_full = in_set.d
            if isinstance(in_set, ProbStar):
                self.mu0_full = np.asarray(
                    in_set.mu0 if getattr(in_set, 'mu0', None) is not None else in_set.mu,
                    dtype=np.float64,
                ).ravel()
                self.Sig0_full = np.asarray(
                    in_set.Sig0 if getattr(in_set, 'Sig0', None) is not None else in_set.Sig,
                    dtype=np.float64,
                )
                self.cov_mode = getattr(in_set, 'cov_mode', 'full')
            else:
                self.mu0_full = None
                self.Sig0_full = None
                self.cov_mode = None

        elif isinstance(in_set, (tuple, list)) and len(in_set) == 2:
            lb, ub = np.asarray(in_set[0], dtype=np.float64), np.asarray(in_set[1], dtype=np.float64)
            self.in_shape = lb.shape if lb.ndim > 1 else (in_shape_hint or (lb.size,))
            self.orig_set = Star(lb.ravel(), ub.ravel())
            self.c0_full = np.asarray(self.orig_set.V[:, 0], dtype=np.float64).ravel()
            self.V0_full = np.asarray(self.orig_set.V[:, 1:], dtype=np.float64)
            self.pred_lb_full = np.asarray(self.orig_set.pred_lb, dtype=np.float64).ravel()
            self.pred_ub_full = np.asarray(self.orig_set.pred_ub, dtype=np.float64).ravel()
            self.C0_full = self.orig_set.C
            self.d0_full = self.orig_set.d
            self.mu0_full = None
            self.Sig0_full = None
            self.cov_mode = None
        else:
            raise TypeError(f"Unsupported inputSet type for CROWN: {type(in_set)}")

        self.dim = len(self.c0_full)
        self.nVars_full = len(self.pred_lb_full)

        # Identify active predicate variables (non-degenerate interval & non-zero column)
        width_full = self.pred_ub_full - self.pred_lb_full
        if sp.issparse(self.V0_full):
            col_nnz = np.diff(self.V0_full.tocsc().indptr) > 0
        else:
            col_nnz = np.any(np.abs(self.V0_full) > 0.0, axis=0)

        has_c0 = (
            self.C0_full is not None
            and len(self.d0_full) > 0
            and (self.C0_full.nnz > 0 if sp.issparse(self.C0_full) else self.C0_full.size > 0)
        )
        self.has_constraints = bool(has_c0)

        if not self.has_constraints:
            active_mask = (width_full > 1e-15) & col_nnz
        else:
            active_mask = np.ones(self.nVars_full, dtype=bool)

        self.active_idx = np.where(active_mask)[0]
        self.fixed_idx = np.where(~active_mask)[0]
        self.nVars = len(self.active_idx)

        self.c0 = self.c0_full.copy()
        if len(self.fixed_idx) > 0:
            fixed_vals = 0.5 * (self.pred_lb_full[self.fixed_idx] + self.pred_ub_full[self.fixed_idx])
            if sp.issparse(self.V0_full):
                self.c0 = self.c0 + np.asarray(self.V0_full[:, self.fixed_idx] @ fixed_vals).ravel()
            else:
                self.c0 = self.c0 + self.V0_full[:, self.fixed_idx] @ fixed_vals

        if sp.issparse(self.V0_full):
            self.V0 = self.V0_full[:, self.active_idx].toarray().astype(np.float64)
        else:
            self.V0 = np.asarray(self.V0_full[:, self.active_idx], dtype=np.float64)

        self.pred_lb = self.pred_lb_full[self.active_idx].copy()
        self.pred_ub = self.pred_ub_full[self.active_idx].copy()

        # Check if V0 is identity (common case)
        self.is_identity_V0 = (
            self.V0.shape[0] == self.V0.shape[1]
            and np.max(np.abs(self.V0 - np.eye(self.dim))) <= 1e-14
        )

    def expand_to_full_predicates(self, A_active, b_active):
        """Map linear form A_active @ alpha_active + b_active back to full nVars_full predicates."""
        K = A_active.shape[0]
        if self.nVars == self.nVars_full:
            return A_active, b_active
        A_full = np.zeros((K, self.nVars_full), dtype=np.float64)
        if self.nVars > 0:
            A_full[:, self.active_idx] = A_active
        return A_full, b_active


# =====================================================================
# 3. Topological Computation Graph (DAG) for StarV Networks
# =====================================================================

class _GraphNode(object):
    """Single primitive node in a compiled CROWNNetworkGraph."""

    def __init__(
        self,
        node_id,
        op,
        inputs,
        out_dim,
        out_shape,
        W=None,
        b=None,
        scale=None,
        shift=None,
        perm=None,
        inv_perm=None,
        gamma=None,
        layer_ref=None,
        window_indices=None,
    ):
        self.id = int(node_id)
        self.op = str(op)
        self.inputs = list(inputs)
        self.out_dim = int(out_dim)
        self.out_shape = tuple(out_shape) if out_shape is not None else (int(out_dim),)
        self.W = W
        self.b = b
        self.scale = scale
        self.shift = shift
        self.perm = perm
        self.inv_perm = inv_perm
        self.gamma = gamma
        self.layer_ref = layer_ref
        self.window_indices = window_indices


def _precompute_maxpool_windows(layer, in_shape):
    """
    Precompute pooling window input flat indices (in StarV HWC layout) for MaxPool2DLayer.
    Returns (window_indices, out_shape) where window_indices is a list of 1D int64 arrays
    of length mo * no * c. If all windows have identical size (no zero-padding truncation),
    also returns a 2D array `window_mat` of shape (ko, p*q) for vectorized indexing.
    """
    m, n, c = in_shape
    p, q = int(layer.kernel_size[0]), int(layer.kernel_size[1])
    sh, sw = int(layer.stride[0]), int(layer.stride[1])
    pad = layer.padding
    if np.ndim(pad) == 0:
        pt = pb = pl = pr = int(pad)
    elif len(pad) == 2:
        pt = pb = int(pad[0])
        pl = pr = int(pad[1])
    else:
        pt, pb, pl, pr = int(pad[0]), int(pad[1]), int(pad[2]), int(pad[3])

    mo, no = MaxPool2DLayer.get_output_size(m, n, layer.kernel_size, layer.stride, layer.padding)
    mo, no = int(mo), int(no)
    out_shape = (mo, no, c)

    # Create padded index grid where -1 denotes outside image
    grid = np.full((m + pt + pb, n + pl + pr, c), -1, dtype=np.int64)
    grid[pt : pt + m, pl : pl + n, :] = np.arange(m * n * c, dtype=np.int64).reshape(m, n, c)

    # Vectorized extraction when pt == pb == pl == pr == 0
    if pt == 0 and pb == 0 and pl == 0 and pr == 0:
        # Output index (i, j, ch) looks at rows i*sh : i*sh+p, cols j*sw : j*sw+q, channel ch
        rows = np.arange(mo, dtype=np.int64)[:, None, None, None, None] * sh + np.arange(p, dtype=np.int64)[None, None, None, :, None]
        cols = np.arange(no, dtype=np.int64)[None, :, None, None, None] * sw + np.arange(q, dtype=np.int64)[None, None, None, None, :]
        chs = np.arange(c, dtype=np.int64)[None, None, :, None, None]
        win_mat = (rows * (n * c) + cols * c + chs).reshape(mo * no * c, p * q)
        win_list = [win_mat[k] for k in range(mo * no * c)]
        return (win_list, win_mat), out_shape

    win_list = []
    same_len = True
    target_len = p * q
    for i in range(mo):
        r0 = i * sh
        for j in range(no):
            c0 = j * sw
            patch = grid[r0 : r0 + p, c0 : c0 + q, :]  # (p, q, c)
            for ch in range(c):
                idxs = patch[:, :, ch].ravel()
                valid = idxs[idxs >= 0]
                if len(valid) != target_len:
                    same_len = False
                win_list.append(valid)
    win_mat = np.stack(win_list, axis=0) if same_len else None
    return (win_list, win_mat), out_shape


class CROWNNetworkGraph(object):
    """
    Topological computation graph compiled from a StarV NeuralNetwork.
    Supports sequential and DAG (ResidualLayer) architectures across all StarV layer types.
    """

    def __init__(self, net, in_shape):
        self.net = net
        if isinstance(in_shape, int):
            in_shape = (in_shape,)
        self.in_shape = tuple(in_shape)
        self.in_dim = int(np.prod(self.in_shape))
        self.nodes = []
        self.has_pixel_classification = False
        self.num_pix_classes = None

        # Node 0: input
        self.nodes.append(
            _GraphNode(
                node_id=0,
                op='input',
                inputs=[],
                out_dim=self.in_dim,
                out_shape=self.in_shape,
            )
        )

        layers_to_compile = list(net.layers)
        if len(layers_to_compile) > 0 and isinstance(layers_to_compile[-1], PixelClassificationLayer):
            self.has_pixel_classification = True
            self.num_pix_classes = layers_to_compile[-1].classes
            layers_to_compile = layers_to_compile[:-1]

        last_id = 0
        for layer in layers_to_compile:
            last_id = self._append_layer(layer, last_id)

        self.output_node_id = last_id
        self.out_dim = self.nodes[last_id].out_dim
        self.out_shape = self.nodes[last_id].out_shape

    def _append_layer(self, layer, parent_id):
        parent = self.nodes[parent_id]
        cur_dim = parent.out_dim
        cur_shape = parent.out_shape
        nid = len(self.nodes)

        if isinstance(layer, FullyConnectedLayer):
            if layer.W is None:
                if layer.b is None:
                    node = _GraphNode(
                        node_id=nid,
                        op='identity',
                        inputs=[parent_id],
                        out_dim=cur_dim,
                        out_shape=cur_shape,
                        layer_ref=layer,
                    )
                else:
                    scale = np.ones(cur_dim, dtype=np.float64)
                    shift = np.asarray(layer.b, dtype=np.float64).ravel()
                    if shift.size == 1 and cur_dim > 1:
                        shift = np.full(cur_dim, float(shift[0]), dtype=np.float64)
                    node = _GraphNode(
                        node_id=nid,
                        op='scale_shift',
                        inputs=[parent_id],
                        out_dim=cur_dim,
                        out_shape=cur_shape,
                        scale=scale,
                        shift=shift,
                        layer_ref=layer,
                    )
                self.nodes.append(node)
                return nid
            W = np.asarray(layer.W, dtype=np.float64)
            out_dim = W.shape[0]
            b = np.zeros(out_dim, dtype=np.float64) if layer.b is None else np.asarray(layer.b, dtype=np.float64).ravel()
            node = _GraphNode(
                node_id=nid,
                op='affine',
                inputs=[parent_id],
                out_dim=out_dim,
                out_shape=(out_dim,),
                W=W,
                b=b,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, Conv2DLayer):
            assert len(cur_shape) == 3, f"Conv2DLayer requires 3D (H, W, C) input shape, got {cur_shape}"
            eye_csr = sp.eye(cur_dim, dtype=np.float64, format='csr')
            if layer.sparse:
                W_csr, out_shape = layer.fconv2d_csr(eye_csr, cur_shape)
            else:
                W_csr, out_shape = layer.fconv2d_csr_co_loop(eye_csr, cur_shape)
            W_csr = W_csr.tocsr().astype(np.float64)
            b = layer.conv2d(np.zeros(cur_shape, dtype=np.float64), bias=True).astype(np.float64).reshape(-1)
            out_dim = int(np.prod(out_shape))
            node = _GraphNode(
                node_id=nid,
                op='affine',
                inputs=[parent_id],
                out_dim=out_dim,
                out_shape=out_shape,
                W=W_csr,
                b=b,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, ConvTranspose2DLayer):
            assert len(cur_shape) == 3, f"ConvTranspose2DLayer requires 3D (H, W, C) input shape, got {cur_shape}"
            eye_csr = sp.eye(cur_dim, dtype=np.float64, format='csr')
            W_csr, out_shape = layer.fconvtrans2d_csr2(eye_csr, cur_shape)
            W_csr = W_csr.tocsr().astype(np.float64)
            b = layer.convtrans2d(np.zeros(cur_shape, dtype=np.float64), bias=True).astype(np.float64).reshape(-1)
            out_dim = int(np.prod(out_shape))
            node = _GraphNode(
                node_id=nid,
                op='affine',
                inputs=[parent_id],
                out_dim=out_dim,
                out_shape=out_shape,
                W=W_csr,
                b=b,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, BatchNorm2DLayer):
            c = cur_shape[2] if len(cur_shape) == 3 else layer.num_features
            spatial = cur_dim // c
            a_ch = np.asarray(layer.gamma, dtype=np.float64) / np.sqrt(
                np.asarray(layer.var, dtype=np.float64) + float(layer.eps)
            )
            b_ch = np.asarray(layer.beta, dtype=np.float64) - a_ch * np.asarray(layer.mean, dtype=np.float64)
            scale = np.tile(a_ch, spatial)
            shift = np.tile(b_ch, spatial)
            node = _GraphNode(
                node_id=nid,
                op='scale_shift',
                inputs=[parent_id],
                out_dim=cur_dim,
                out_shape=cur_shape,
                scale=scale,
                shift=shift,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, AvgPool2DLayer):
            assert len(cur_shape) == 3, f"AvgPool2DLayer requires 3D (H, W, C) input shape, got {cur_shape}"
            eye_csr = sp.eye(cur_dim, dtype=np.float64, format='csr')
            W_csr, out_shape = layer.favgpool2d_csr2(eye_csr, cur_shape)
            W_csr = W_csr.tocsr().astype(np.float64)
            out_dim = int(np.prod(out_shape))
            b = np.zeros(out_dim, dtype=np.float64)
            node = _GraphNode(
                node_id=nid,
                op='affine',
                inputs=[parent_id],
                out_dim=out_dim,
                out_shape=out_shape,
                W=W_csr,
                b=b,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, FlattenLayer):
            if not layer.channel_last and len(cur_shape) == 3:
                h, w, ch = cur_shape
                perm = np.arange(h * w * ch, dtype=np.int64).reshape(h, w, ch).transpose(2, 0, 1).ravel()
                inv_perm = np.argsort(perm)
                node = _GraphNode(
                    node_id=nid,
                    op='permute',
                    inputs=[parent_id],
                    out_dim=cur_dim,
                    out_shape=(cur_dim,),
                    perm=perm,
                    inv_perm=inv_perm,
                    layer_ref=layer,
                )
            else:
                node = _GraphNode(
                    node_id=nid,
                    op='identity',
                    inputs=[parent_id],
                    out_dim=cur_dim,
                    out_shape=(cur_dim,),
                    layer_ref=layer,
                )
            self.nodes.append(node)
            return nid

        if isinstance(layer, ReshapeLayer):
            out_shape = tuple(layer.out_shape)
            node = _GraphNode(
                node_id=nid,
                op='identity',
                inputs=[parent_id],
                out_dim=cur_dim,
                out_shape=out_shape,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, ReLULayer):
            node = _GraphNode(
                node_id=nid,
                op='relu',
                inputs=[parent_id],
                out_dim=cur_dim,
                out_shape=cur_shape,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, LeakyReLULayer):
            node = _GraphNode(
                node_id=nid,
                op='leaky_relu',
                inputs=[parent_id],
                out_dim=cur_dim,
                out_shape=cur_shape,
                gamma=float(getattr(layer, 'gamma', 0.1)),
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, SatLinLayer):
            node = _GraphNode(
                node_id=nid,
                op='satlin',
                inputs=[parent_id],
                out_dim=cur_dim,
                out_shape=cur_shape,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, SatLinsLayer):
            node = _GraphNode(
                node_id=nid,
                op='satlins',
                inputs=[parent_id],
                out_dim=cur_dim,
                out_shape=cur_shape,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, LogSigLayer):
            node = _GraphNode(
                node_id=nid,
                op='sigmoid',
                inputs=[parent_id],
                out_dim=cur_dim,
                out_shape=cur_shape,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, TanSigLayer):
            node = _GraphNode(
                node_id=nid,
                op='tanh',
                inputs=[parent_id],
                out_dim=cur_dim,
                out_shape=cur_shape,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, GELULayer):
            node = _GraphNode(
                node_id=nid,
                op='gelu',
                inputs=[parent_id],
                out_dim=cur_dim,
                out_shape=cur_shape,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, MaxPool2DLayer):
            assert len(cur_shape) == 3, f"MaxPool2DLayer requires 3D (H, W, C) input shape, got {cur_shape}"
            win_info, out_shape = _precompute_maxpool_windows(layer, cur_shape)
            out_dim = int(np.prod(out_shape))
            node = _GraphNode(
                node_id=nid,
                op='maxpool2d',
                inputs=[parent_id],
                out_dim=out_dim,
                out_shape=out_shape,
                window_indices=win_info,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, LayerNormLayer):
            node = _GraphNode(
                node_id=nid,
                op='layernorm',
                inputs=[parent_id],
                out_dim=cur_dim,
                out_shape=cur_shape,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, MultiHeadAttentionLayer):
            node = _GraphNode(
                node_id=nid,
                op='mhsa',
                inputs=[parent_id],
                out_dim=cur_dim,
                out_shape=cur_shape,
                layer_ref=layer,
            )
            self.nodes.append(node)
            return nid

        if isinstance(layer, ResidualLayer):
            main_id = parent_id
            for sub_l in layer.main_layers:
                main_id = self._append_layer(sub_l, main_id)
            short_id = parent_id
            for sub_l in layer.shortcut_layers:
                short_id = self._append_layer(sub_l, short_id)
            assert self.nodes[main_id].out_dim == self.nodes[short_id].out_dim, (
                f"ResidualLayer branch dimension mismatch: "
                f"{self.nodes[main_id].out_dim} vs {self.nodes[short_id].out_dim}"
            )
            add_id = len(self.nodes)
            self.nodes.append(
                _GraphNode(
                    node_id=add_id,
                    op='add',
                    inputs=[main_id, short_id],
                    out_dim=self.nodes[main_id].out_dim,
                    out_shape=self.nodes[main_id].out_shape,
                    layer_ref=layer,
                )
            )
            cur_end = add_id
            if layer.post_activation is not None:
                post_layers = (
                    layer.post_activation
                    if isinstance(layer.post_activation, list)
                    else [layer.post_activation]
                )
                for pl in post_layers:
                    cur_end = self._append_layer(pl, cur_end)
            return cur_end

        raise TypeError(f"Unsupported layer type in CROWNNetworkGraph: {type(layer)}")

    def evaluate_flat(self, x_flat):
        """
        Evaluate the compiled graph on a 1D vector (in_dim,) or 2D batch (in_dim, B).
        """
        x_arr = np.asarray(x_flat, dtype=np.float64)
        is_1d = x_arr.ndim == 1
        if is_1d:
            x_arr = x_arr[:, None]
        vals = {0: x_arr}
        for node in self.nodes[1:]:
            op = node.op
            if op == 'identity':
                vals[node.id] = vals[node.inputs[0]]
            elif op == 'permute':
                vals[node.id] = vals[node.inputs[0]][node.perm, :]
            elif op == 'scale_shift':
                vals[node.id] = node.scale[:, None] * vals[node.inputs[0]] + node.shift[:, None]
            elif op == 'affine':
                vals[node.id] = (node.W @ vals[node.inputs[0]]) + node.b[:, None]
            elif op == 'add':
                vals[node.id] = vals[node.inputs[0]] + vals[node.inputs[1]]
            elif op == 'relu':
                vals[node.id] = np.maximum(vals[node.inputs[0]], 0.0)
            elif op == 'leaky_relu':
                u = vals[node.inputs[0]]
                vals[node.id] = np.where(u >= 0.0, u, node.gamma * u)
            elif op == 'satlin':
                vals[node.id] = np.clip(vals[node.inputs[0]], 0.0, 1.0)
            elif op == 'satlins':
                vals[node.id] = np.clip(vals[node.inputs[0]], -1.0, 1.0)
            elif op == 'sigmoid':
                vals[node.id] = LogSig.f(vals[node.inputs[0]])
            elif op == 'tanh':
                vals[node.id] = TanSig.f(vals[node.inputs[0]])
            elif op == 'gelu':
                vals[node.id] = GELULayer.evaluate(vals[node.inputs[0]])
            elif op == 'maxpool2d':
                u = vals[node.inputs[0]]
                win_list, win_mat = node.window_indices
                if win_mat is not None:
                    vals[node.id] = np.max(u[win_mat, :], axis=1)
                else:
                    out = np.empty((node.out_dim, u.shape[1]), dtype=np.float64)
                    for r, idxs in enumerate(win_list):
                        out[r, :] = np.max(u[idxs, :], axis=0)
                    vals[node.id] = out
            elif op == 'layernorm':
                vals[node.id] = node.layer_ref.evaluate(vals[node.inputs[0]])
            elif op == 'mhsa':
                vals[node.id] = node.layer_ref.evaluate(vals[node.inputs[0]])
            else:
                raise ValueError(f"Unknown op {op}")
        res = vals[self.output_node_id]
        return res[:, 0] if is_1d else res


# =====================================================================
# 4. LayerNorm & MultiHeadAttention CROWN Envelope Builders
# =====================================================================

def _build_layernorm_crown_envelope(layer, A_lb_in, b_lb_in, A_ub_in, b_ub_in, pred_lb, pred_ub):
    """
    Construct sound linear relaxation envelope for LayerNormLayer:
        W_ln @ x + b_ln_lb <= LayerNorm(x) <= W_ln @ x + b_ln_ub
    using exact mean-centering projection P and coupled reciprocal-std bounds.
    """
    D = layer.normalized_dim
    N = A_lb_in.shape[0]
    n = len(pred_lb)
    assert N % D == 0
    T = N // D

    # Form midpoint affine generator V_x, c_x plus interval remainder rad_x
    V_x = 0.5 * (A_lb_in + A_ub_in)
    c_x = 0.5 * (b_lb_in + b_ub_in)
    dA = 0.5 * (A_ub_in - A_lb_in)
    db = 0.5 * (b_ub_in - b_lb_in)
    rad_x = np.maximum(
        0.0,
        np.maximum(dA, 0.0) @ pred_ub + np.minimum(dA, 0.0) @ pred_lb + db,
    )

    # Mean-center V_x and c_x across each token
    V_reshaped = V_x.reshape(T, D, n)
    V_z = (V_reshaped - np.mean(V_reshaped, axis=1, keepdims=True)).reshape(N, n)
    c_reshaped = c_x.reshape(T, D)
    c_z = (c_reshaped - np.mean(c_reshaped, axis=1, keepdims=True)).reshape(N)

    # Centered remainder: |z_{t,d} - z_mid_{t,d}| <= (1 - 1/D)*rad_{t,d} + (1/D)*sum_{j!=d} rad_{t,j}
    rad_tok = rad_x.reshape(T, D)
    rad_z = ((1.0 - 2.0 / float(D)) * rad_tok + np.mean(rad_tok, axis=1, keepdims=True)).reshape(N)
    rad_z = np.maximum(rad_z, 0.0)

    Vz_pos = np.maximum(V_z, 0.0)
    Vz_neg = np.minimum(V_z, 0.0)
    l_z = c_z + Vz_pos @ pred_lb + Vz_neg @ pred_ub - rad_z
    u_z = c_z + Vz_pos @ pred_ub + Vz_neg @ pred_lb + rad_z

    l_z_tok = l_z.reshape(T, D)
    u_z_tok = u_z.reshape(T, D)
    cross_zero = (l_z_tok <= 0.0) & (u_z_tok >= 0.0)
    q_lb = np.where(cross_zero, 0.0, np.minimum(np.square(l_z_tok), np.square(u_z_tok)))
    q_ub = np.maximum(np.square(l_z_tok), np.square(u_z_tok))

    sum_q_lb = np.sum(q_lb, axis=1, keepdims=True)
    sum_q_ub = np.sum(q_ub, axis=1, keepdims=True)

    # Cauchy-Schwarz leave-one-out lower bound
    z_mid = 0.5 * (l_z_tok + u_z_tok)
    Vz_tok = V_z.reshape(T, D, n)
    cz_tok = c_z.reshape(T, D)
    rz_tok = rad_z.reshape(T, D)
    norm_sq_all = np.sum(np.square(z_mid), axis=1, keepdims=True)
    norm_loo = np.sqrt(np.maximum(norm_sq_all - np.square(z_mid), 0.0))
    inv_norm_loo = np.where(norm_loo > 1e-12, 1.0 / np.where(norm_loo > 1e-12, norm_loo, 1.0), 0.0)

    dot_c_all = np.sum(z_mid * cz_tok, axis=1, keepdims=True)
    proj_c = (dot_c_all - z_mid * cz_tok) * inv_norm_loo
    dot_V_all = np.einsum('td,tdn->tn', z_mid, Vz_tok)[:, None, :]
    proj_V = (dot_V_all - z_mid[:, :, None] * Vz_tok) * inv_norm_loo[:, :, None]
    dot_r_all = np.sum(np.abs(z_mid) * rz_tok, axis=1, keepdims=True)
    proj_rad = (dot_r_all - np.abs(z_mid) * rz_tok) * inv_norm_loo

    pV_pos = np.maximum(proj_V, 0.0)
    pV_neg = np.minimum(proj_V, 0.0)
    l_proj = proj_c + np.einsum('tdn,n->td', pV_pos, pred_lb) + np.einsum('tdn,n->td', pV_neg, pred_ub) - proj_rad
    u_proj = proj_c + np.einsum('tdn,n->td', pV_pos, pred_ub) + np.einsum('tdn,n->td', pV_neg, pred_lb) + proj_rad
    cross_proj = (l_proj <= 0.0) & (u_proj >= 0.0)
    q_cs_loo = np.where(cross_proj, 0.0, np.minimum(np.square(l_proj), np.square(u_proj)))

    S_min = np.maximum(sum_q_lb - q_lb, q_cs_loo) / float(D) + layer.eps
    S_max = (sum_q_ub - q_ub) / float(D) + layer.eps
    S_mid = 0.5 * (S_min + S_max)

    def _g_norm(z_val, S_val):
        return z_val / np.sqrt(np.square(z_val) / float(D) + S_val)

    dz = u_z_tok - l_z_tok
    a_tan = S_mid / np.power(np.square(z_mid) / float(D) + S_mid, 1.5)
    a_chord = np.where(
        dz > 1e-12,
        (_g_norm(u_z_tok, S_mid) - _g_norm(l_z_tok, S_mid)) / np.where(dz > 1e-12, dz, 1.0),
        a_tan,
    )
    a_hat = np.maximum(a_chord, 1e-12)

    res_cands = []
    for S_bound in (S_min, S_max):
        z_star = np.sqrt(float(D) * np.maximum(0.0, np.power(S_bound / a_hat, 2.0 / 3.0) - S_bound))
        z_p = np.clip(z_star, l_z_tok, u_z_tok)
        z_m = np.clip(-z_star, l_z_tok, u_z_tok)
        for z_cand in (l_z_tok, u_z_tok, z_p, z_m):
            res_cands.append(_g_norm(z_cand, S_bound) - a_hat * z_cand)
    res_stack = np.stack(res_cands, axis=0)
    b_hat_lb = np.min(res_stack, axis=0)
    b_hat_ub = np.max(res_stack, axis=0)

    gamma_tok = layer.gamma[None, :]
    beta_tok = layer.beta[None, :]
    a_eff = (gamma_tok * a_hat).reshape(N)
    b_lb = np.where(gamma_tok >= 0.0, gamma_tok * b_hat_lb + beta_tok, gamma_tok * b_hat_ub + beta_tok).reshape(N)
    b_ub = np.where(gamma_tok >= 0.0, gamma_tok * b_hat_ub + beta_tok, gamma_tok * b_hat_lb + beta_tok).reshape(N)

    P_op = layer._build_centering_block_diag(N, dtype=np.float64)
    if sp.issparse(P_op):
        W_ln = P_op.multiply(a_eff[:, None]).tocsr()
    else:
        W_ln = a_eff[:, None] * P_op

    return W_ln, b_lb, b_ub


def _build_mhsa_crown_envelope(layer, A_lb_in, b_lb_in, A_ub_in, b_ub_in, pred_lb, pred_ub):
    """
    Construct sound linear relaxation envelope for MultiHeadAttentionLayer:
        W_mhsa @ x + b_mhsa_lb <= MHSA(x) <= W_mhsa @ x + b_mhsa_ub
    using McCormick score bounds, difference-generator Softmax bounds, and
    output-projected box-simplex water-filling along the value path V(x).
    """
    D = layer.embed_dim
    H = layer.num_heads
    dk = layer.head_dim
    N = A_lb_in.shape[0]
    assert N % D == 0
    T = N // D

    # Wrap current affine bounds into a temporary Star to reuse MultiHeadAttentionLayer's exact logic
    V_mid = 0.5 * (A_lb_in + A_ub_in)
    c_mid = 0.5 * (b_lb_in + b_ub_in)
    dA = 0.5 * (A_ub_in - A_lb_in)
    db = 0.5 * (b_ub_in - b_lb_in)
    rad_in = np.maximum(
        0.0,
        np.maximum(dA, 0.0) @ pred_ub + np.minimum(dA, 0.0) @ pred_lb + db,
    )

    tol = 1e-12
    amb = np.where(rad_in > tol)[0]
    if len(amb) > 0:
        V_ext = np.zeros((N, len(amb)), dtype=np.float64)
        V_ext[amb, np.arange(len(amb))] = rad_in[amb]
        V_x = np.hstack([V_mid, V_ext])
        lb_ext = np.hstack([pred_lb, -np.ones(len(amb), dtype=np.float64)])
        ub_ext = np.hstack([pred_ub, np.ones(len(amb), dtype=np.float64)])
    else:
        V_x = V_mid
        lb_ext = pred_lb
        ub_ext = pred_ub

    n_ext = len(lb_ext)
    (l_Q, u_Q), (l_K, u_K), (l_V, u_V), c_Q, VQ, c_K, VK, c_V, VV = layer._compute_qkv_bounds(
        c_mid, V_x, lb_ext, ub_ext, T
    )

    q_l = l_Q[:, :, None, :]
    q_u = u_Q[:, :, None, :]
    k_l = l_K[:, None, :, :]
    k_u = u_K[:, None, :, :]
    p1, p2, p3, p4 = q_l * k_l, q_l * k_u, q_u * k_l, q_u * k_u
    prod_lb = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
    prod_ub = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
    l_S_box = np.sum(prod_lb, axis=-1) * layer.scale
    u_S_box = np.sum(prod_ub, axis=-1) * layer.scale

    Q0 = 0.5 * (l_Q + u_Q)
    dQ = 0.5 * np.maximum(u_Q - l_Q, 0.0)
    K0 = 0.5 * (l_K + u_K)
    dK = 0.5 * np.maximum(u_K - l_K, 0.0)

    VQ_dense = VQ.toarray().reshape(T, H, dk, n_ext).transpose(1, 0, 2, 3)
    VK_dense = VK.toarray().reshape(T, H, dk, n_ext).transpose(1, 0, 2, 3)
    cQ_arr = c_Q.reshape(T, H, dk).transpose(1, 0, 2)
    cK_arr = c_K.reshape(T, H, dk).transpose(1, 0, 2)

    c_S = (
        np.einsum('hid,hjd->hij', Q0, cK_arr)
        + np.einsum('hjd,hid->hij', K0, cQ_arr)
        - np.einsum('hid,hjd->hij', Q0, K0)
    ) * layer.scale
    V_S = (
        np.einsum('hid,hjdn->hijn', Q0, VK_dense)
        + np.einsum('hjd,hidn->hijn', K0, VQ_dense)
    ) * layer.scale
    rem_S = np.einsum('hid,hjd->hij', dQ, dK) * layer.scale

    VS_pos = np.maximum(V_S, 0.0)
    VS_neg = np.minimum(V_S, 0.0)
    l_S_aff = c_S + np.einsum('hijn,n->hij', VS_pos, lb_ext) + np.einsum('hijn,n->hij', VS_neg, ub_ext) - rem_S
    u_S_aff = c_S + np.einsum('hijn,n->hij', VS_pos, ub_ext) + np.einsum('hijn,n->hij', VS_neg, lb_ext) + rem_S
    l_S = np.maximum(l_S_box, l_S_aff)
    u_S = np.minimum(u_S_box, u_S_aff)

    l_A = np.zeros((H, T, T), dtype=np.float64)
    u_A = np.zeros((H, T, T), dtype=np.float64)
    for h in range(H):
        for i in range(T):
            for j in range(T):
                d_c = c_S[h, i, :] - c_S[h, i, j]
                d_V = V_S[h, i, :, :] - V_S[h, i, j, None, :]
                d_rem = rem_S[h, i, :] + rem_S[h, i, j]
                d_rem[j] = 0.0
                dV_pos = np.maximum(d_V, 0.0)
                dV_neg = np.minimum(d_V, 0.0)
                diff_u_aff = d_c + dV_pos @ ub_ext + dV_neg @ lb_ext + d_rem
                diff_l_aff = d_c + dV_pos @ lb_ext + dV_neg @ ub_ext - d_rem

                diff_u = np.minimum(u_S[h, i, :] - l_S[h, i, j], diff_u_aff)
                diff_u[j] = 0.0
                l_A[h, i, j] = 1.0 / np.sum(np.exp(np.clip(diff_u, -60.0, 60.0)))

                diff_l = np.maximum(l_S[h, i, :] - u_S[h, i, j], diff_l_aff)
                diff_l[j] = 0.0
                u_A[h, i, j] = 1.0 / np.sum(np.exp(np.clip(diff_l, -60.0, 60.0)))

    span_A = np.maximum(u_A - l_A, 0.0)
    sum_lA = np.sum(l_A, axis=-1, keepdims=True)
    sum_spanA = np.sum(span_A, axis=-1, keepdims=True)
    theta_A = np.where(
        sum_spanA > 1e-15,
        np.clip((1.0 - sum_lA) / np.where(sum_spanA > 1e-15, sum_spanA, 1.0), 0.0, 1.0),
        0.5,
    )
    A0 = l_A + theta_A * span_A
    dA_attn = np.maximum(A0 - l_A, u_A - A0)

    def _box_simplex_extrema(l_vec, u_vec, v_vec):
        rem = max(0.0, 1.0 - float(np.sum(l_vec)))
        cap = np.maximum(u_vec - l_vec, 0.0)
        base = float(np.dot(l_vec, v_vec))
        order_min = np.argsort(v_vec)
        rem_min = rem
        val_min = base
        for idx in order_min:
            take = min(rem_min, float(cap[idx]))
            val_min += take * float(v_vec[idx])
            rem_min -= take
            if rem_min <= 1e-15:
                break
        order_max = order_min[::-1]
        rem_max = rem
        val_max = base
        for idx in order_max:
            take = min(rem_max, float(cap[idx]))
            val_max += take * float(v_vec[idx])
            rem_max -= take
            if rem_max <= 1e-15:
                break
        return val_min, val_max

    W_lin = np.zeros((T * D, T * D), dtype=np.float64)
    rad_Y_mat = np.zeros((T, D), dtype=np.float64)
    shift_Y_mat = np.zeros((T, D), dtype=np.float64)
    VV_dense = VV.toarray().reshape(T, H, dk, n_ext)
    cV_arr = c_V.reshape(T, H, dk)

    for h in range(H):
        W_O_h = layer.W_O[:, h * dk : (h + 1) * dk]
        cU_h = cV_arr[:, h, :] @ W_O_h.T
        VU_h = np.einsum('od,tdn->ton', W_O_h, VV_dense[:, h, :, :])
        VUp = np.maximum(VU_h, 0.0)
        VUn = np.minimum(VU_h, 0.0)
        lU_h = cU_h + np.einsum('ton,n->to', VUp, lb_ext) + np.einsum('ton,n->to', VUn, ub_ext)
        uU_h = cU_h + np.einsum('ton,n->to', VUp, ub_ext) + np.einsum('ton,n->to', VUn, lb_ext)
        U0_h = 0.5 * (lU_h + uU_h)
        dU_h = 0.5 * np.maximum(uU_h - lU_h, 0.0)
        quad_rem_h = dA_attn[h] @ dU_h
        for i in range(T):
            for d_out in range(D):
                u_col = U0_h[:, d_out]
                umin, umax = _box_simplex_extrema(l_A[h, i, :], u_A[h, i, :], u_col)
                u_ref = float(np.dot(A0[h, i, :], u_col))
                d_min = umin - u_ref
                d_max = umax - u_ref
                shift_Y_mat[i, d_out] += 0.5 * (d_min + d_max)
                rad_Y_mat[i, d_out] += 0.5 * max(0.0, d_max - d_min) + quad_rem_h[i, d_out]

        for i in range(T):
            row_slice = slice(i * D, (i + 1) * D)
            for j in range(T):
                col_slice = slice(j * D + h * dk, j * D + (h + 1) * dk)
                W_lin[row_slice, col_slice] += A0[h, i, j] * W_O_h

    rad_Y = rad_Y_mat.reshape(N)
    shift_Y = shift_Y_mat.reshape(N)

    # Express V(x) = M_V @ x + b_V_tiled so Y_0(x) = (W_lin @ M_V) @ x + (W_lin @ b_V_tiled + b_O_tiled + shift_Y)
    I_T = sp.eye(T, dtype=np.float64, format='csr')
    M_V = sp.kron(I_T, sp.csr_array(layer.W_V), format='csr')
    b_V_tiled = np.tile(layer.b_V, T)
    b_O_tiled = np.tile(layer.b_O, T)

    W_mhsa = W_lin @ M_V.toarray()
    b_center = W_lin @ b_V_tiled + b_O_tiled + shift_Y
    b_lb = b_center - rad_Y
    b_ub = b_center + rad_Y
    return W_mhsa, b_lb, b_ub


# =====================================================================
# 5. Core Backward CROWN, alpha-CROWN, and alpha,beta-CROWN Engine
# =====================================================================

class CROWNVerifier(object):
    """
    Unified CROWN / alpha-CROWN / alpha,beta-CROWN bound propagation and verification engine.

    Args:
        mode: 'crown', 'alpha-crown', or 'alpha-beta-crown'
        slope_mode: 'adaptive' (default), 'zero', or 'one' for classic CROWN ReLU slopes
        alpha_iters: number of Adam optimization steps for alpha-CROWN (default: 30)
        alpha_lr: learning rate for alpha-CROWN slopes (default: 0.15)
        beta_iters: number of Adam optimization steps per BaB branch in alpha,beta-CROWN (default: 20)
        beta_lr: learning rate for beta multipliers (default: 0.1)
        max_bab_branches: maximum number of BaB branches to explore in alpha,beta-CROWN (default: 200)
        bab_timeout: timeout in seconds for BaB loop (default: 60.0)
        use_input_constraints: whether to tighten input-space concretization using LP when C0 * alpha_0 <= d0 exists
        lp_solver: LP solver name ('gurobi', 'linprog', 'glpk', 'estimate')
        show: verbose logging flag
    """

    def __init__(
        self,
        mode='crown',
        slope_mode='adaptive',
        alpha_iters=30,
        alpha_lr=0.15,
        beta_iters=20,
        beta_lr=0.1,
        max_bab_branches=200,
        bab_timeout=60.0,
        use_input_constraints=True,
        lp_solver='gurobi',
        show=False,
    ):
        self.mode = _normalize_crown_mode(mode)
        self.slope_mode = slope_mode
        self.alpha_iters = int(alpha_iters)
        self.alpha_lr = float(alpha_lr)
        self.beta_iters = int(beta_iters)
        self.beta_lr = float(beta_lr)
        self.max_bab_branches = int(max_bab_branches)
        self.bab_timeout = float(bab_timeout)
        self.use_input_constraints = bool(use_input_constraints)
        self.lp_solver = lp_solver
        self.show = bool(show)

    def _concretize_bounds(self, A_lb, b_lb, A_ub, b_ub, pred_lb, pred_ub, inp_spec=None, use_lp=False):
        """
        Concretize linear bounds A_lb @ alpha_0 + b_lb <= z <= A_ub @ alpha_0 + b_ub
        over alpha_0 in [pred_lb, pred_ub] (and optionally C0 @ alpha_0 <= d0).
        """
        A_lb_pos = np.maximum(A_lb, 0.0)
        A_lb_neg = np.minimum(A_lb, 0.0)
        z_lb = b_lb + A_lb_pos @ pred_lb + A_lb_neg @ pred_ub

        if A_ub is None:
            return z_lb, None

        A_ub_pos = np.maximum(A_ub, 0.0)
        A_ub_neg = np.minimum(A_ub, 0.0)
        z_ub = b_ub + A_ub_pos @ pred_ub + A_ub_neg @ pred_lb

        if (
            use_lp
            and inp_spec is not None
            and inp_spec.has_constraints
            and self.lp_solver != 'estimate'
        ):
            solver = _resolve_lp_solver(self.lp_solver)
            K = A_lb.shape[0]
            for k in range(K):
                try:
                    z_lb[k] = _solve_lp(
                        A_lb[k], b_lb[k], inp_spec.C0_full, inp_spec.d0_full,
                        pred_lb, pred_ub, len(pred_lb), sense='min', lp_solver=solver
                    )
                    z_ub[k] = _solve_lp(
                        A_ub[k], b_ub[k], inp_spec.C0_full, inp_spec.d0_full,
                        pred_lb, pred_ub, len(pred_lb), sense='max', lp_solver=solver
                    )
                except Exception:
                    pass
        return z_lb, z_ub

    def compute_node_relaxations_and_pre_bounds(
        self,
        graph,
        inp_spec,
        pred_lb=None,
        pred_ub=None,
        split_constraints=None,
        initial_pre_bounds=None,
        custom_alphas=None,
    ):
        """
        Forward-scheduled backward CROWN pass:
        Computes pre-activation bounds [l, u] for every nonlinear node in `graph`
        by backward propagation to the input predicates alpha_0, and records the
        sound linear relaxation parameters for every node.

        Args:
            graph: CROWNNetworkGraph
            inp_spec: _InputSpec
            pred_lb, pred_ub: input predicate bounds (defaults to inp_spec.pred_lb, inp_spec.pred_ub)
            split_constraints: optional dict {relu_node_id: {neuron_idx: sign}} where
                               sign = -1 means x >= 0 (active) and sign = +1 means x <= 0 (inactive)
            initial_pre_bounds: optional dict {node_id: (l, u)} to intersect with computed bounds
            custom_alphas: optional dict {relu_node_id: 1D array of slopes}

        Returns:
            node_bounds: dict {node_id: (l, u)}
            node_relax: dict {node_id: relaxation_dict}
        """
        if pred_lb is None:
            pred_lb = inp_spec.pred_lb
        if pred_ub is None:
            pred_ub = inp_spec.pred_ub
        split_constraints = split_constraints or {}
        initial_pre_bounds = initial_pre_bounds or {}
        custom_alphas = custom_alphas or {}

        node_bounds = {}
        node_relax = {}

        # Input node 0 bounds
        l0, u0 = self._concretize_bounds(
            inp_spec.V0, inp_spec.c0, inp_spec.V0, inp_spec.c0, pred_lb, pred_ub
        )
        node_bounds[0] = (l0, u0)

        # Determine which nodes need explicit bounds (parents of nonlinear nodes)
        nonlinear_ops = {
            'relu', 'leaky_relu', 'satlin', 'satlins',
            'sigmoid', 'tanh', 'gelu', 'maxpool2d', 'layernorm', 'mhsa',
        }
        needs_linear_bounds = {'layernorm', 'mhsa'}

        parent_needs_bounds = set()
        parent_needs_linear = set()
        for node in graph.nodes[1:]:
            if node.op in nonlinear_ops:
                parent_needs_bounds.add(node.inputs[0])
            if node.op in needs_linear_bounds:
                parent_needs_linear.add(node.inputs[0])

        node_linear_bounds = {
            0: (inp_spec.V0, inp_spec.c0, inp_spec.V0, inp_spec.c0)
        }

        for node in graph.nodes[1:]:
            nid = node.id
            op = node.op
            p_id = node.inputs[0]

            # Ensure parent bounds are available if node is nonlinear
            if op in nonlinear_ops and p_id not in node_bounds:
                A_lb_p, b_lb_p, A_ub_p, b_ub_p = self.backward_propagate(
                    graph, inp_spec, target_node_id=p_id, C_spec=None,
                    node_relax=node_relax, split_constraints=split_constraints
                )
                lp, up = self._concretize_bounds(A_lb_p, b_lb_p, A_ub_p, b_ub_p, pred_lb, pred_ub)
                if p_id in initial_pre_bounds:
                    lp0, up0 = initial_pre_bounds[p_id]
                    lp = np.maximum(lp, lp0)
                    up = np.minimum(up, up0)
                lp = np.minimum(lp, up)
                node_bounds[p_id] = (lp, up)
                node_linear_bounds[p_id] = (A_lb_p, b_lb_p, A_ub_p, b_ub_p)

            elif op in needs_linear_bounds and p_id not in node_linear_bounds:
                A_lb_p, b_lb_p, A_ub_p, b_ub_p = self.backward_propagate(
                    graph, inp_spec, target_node_id=p_id, C_spec=None,
                    node_relax=node_relax, split_constraints=split_constraints
                )
                node_linear_bounds[p_id] = (A_lb_p, b_lb_p, A_ub_p, b_ub_p)

            # Build relaxation for current node if nonlinear
            if op == 'relu':
                lp, up = node_bounds[p_id]
                lp = lp.copy()
                up = up.copy()
                # Enforce any split constraints on this ReLU node
                if nid in split_constraints:
                    for j_idx, s_val in split_constraints[nid].items():
                        if s_val == -1:  # x_j >= 0
                            lp[j_idx] = max(lp[j_idx], 0.0)
                            up[j_idx] = max(up[j_idx], lp[j_idx])
                        elif s_val == 1:  # x_j <= 0
                            up[j_idx] = min(up[j_idx], 0.0)
                            lp[j_idx] = min(lp[j_idx], up[j_idx])
                    node_bounds[p_id] = (lp, up)

                custom_a = custom_alphas.get(nid, None)
                a_lb, b_lb, a_ub, b_ub = _relu_linear_relaxation(
                    lp, up, slope_mode=self.slope_mode, custom_alpha_l=custom_a
                )
                node_relax[nid] = {
                    'type': 'elementwise',
                    'a_lb': a_lb, 'b_lb': b_lb, 'a_ub': a_ub, 'b_ub': b_ub,
                    'l_pre': lp, 'u_pre': up,
                }
                node_bounds[nid] = (np.maximum(lp, 0.0), np.maximum(up, 0.0))

            elif op == 'leaky_relu':
                lp, up = node_bounds[p_id]
                custom_a = custom_alphas.get(nid, None)
                a_lb, b_lb, a_ub, b_ub = _leaky_relu_linear_relaxation(
                    lp, up, gamma=node.gamma, custom_alpha_l=custom_a
                )
                node_relax[nid] = {
                    'type': 'elementwise',
                    'a_lb': a_lb, 'b_lb': b_lb, 'a_ub': a_ub, 'b_ub': b_ub,
                    'l_pre': lp, 'u_pre': up,
                }

            elif op == 'satlin':
                lp, up = node_bounds[p_id]
                a_lb, b_lb, a_ub, b_ub = _satlin_linear_relaxation(lp, up)
                node_relax[nid] = {
                    'type': 'elementwise',
                    'a_lb': a_lb, 'b_lb': b_lb, 'a_ub': a_ub, 'b_ub': b_ub,
                    'l_pre': lp, 'u_pre': up,
                }

            elif op == 'satlins':
                lp, up = node_bounds[p_id]
                a_lb, b_lb, a_ub, b_ub = _satlins_linear_relaxation(lp, up)
                node_relax[nid] = {
                    'type': 'elementwise',
                    'a_lb': a_lb, 'b_lb': b_lb, 'a_ub': a_ub, 'b_ub': b_ub,
                    'l_pre': lp, 'u_pre': up,
                }

            elif op == 'sigmoid':
                lp, up = node_bounds[p_id]
                a_lb, b_lb, a_ub, b_ub = _s_curve_linear_relaxation(lp, up, LogSig.f, LogSig.df)
                node_relax[nid] = {
                    'type': 'elementwise',
                    'a_lb': a_lb, 'b_lb': b_lb, 'a_ub': a_ub, 'b_ub': b_ub,
                    'l_pre': lp, 'u_pre': up,
                }

            elif op == 'tanh':
                lp, up = node_bounds[p_id]
                a_lb, b_lb, a_ub, b_ub = _s_curve_linear_relaxation(lp, up, TanSig.f, TanSig.df)
                node_relax[nid] = {
                    'type': 'elementwise',
                    'a_lb': a_lb, 'b_lb': b_lb, 'a_ub': a_ub, 'b_ub': b_ub,
                    'l_pre': lp, 'u_pre': up,
                }

            elif op == 'gelu':
                lp, up = node_bounds[p_id]
                a_lb, b_lb, a_ub, b_ub, y_min, y_max = compute_gelu_linear_bounds(lp, up)
                node_relax[nid] = {
                    'type': 'elementwise',
                    'a_lb': a_lb, 'b_lb': b_lb, 'a_ub': a_ub, 'b_ub': b_ub,
                    'l_pre': lp, 'u_pre': up,
                }

            elif op == 'maxpool2d':
                lp, up = node_bounds[p_id]
                win_list, win_mat = node.window_indices
                ko = node.out_dim
                in_dim_p = len(lp)

                if win_mat is not None:
                    l_win = lp[win_mat]  # (ko, ws)
                    u_win = up[win_mat]  # (ko, ws)
                    best_lb_pos = np.argmax(l_win, axis=1)
                    best_ub_pos = np.argmax(u_win, axis=1)
                    rows_arange = np.arange(ko, dtype=np.int64)
                    r_lb = win_mat[rows_arange, best_lb_pos]
                    r_ub = win_mat[rows_arange, best_ub_pos]
                    u_max = u_win[rows_arange, best_ub_pos]

                    # Mask out best_ub_pos to compute second highest upper bound
                    u_win_other = u_win.copy()
                    u_win_other[rows_arange, best_ub_pos] = -np.inf
                    max_other_u = np.max(u_win_other, axis=1)
                    slack = np.maximum(0.0, max_other_u - lp[r_ub])
                else:
                    r_lb = np.zeros(ko, dtype=np.int64)
                    r_ub = np.zeros(ko, dtype=np.int64)
                    u_max = np.zeros(ko, dtype=np.float64)
                    slack = np.zeros(ko, dtype=np.float64)
                    for o_idx, cands in enumerate(win_list):
                        r_l = cands[int(np.argmax(lp[cands]))]
                        r_u = cands[int(np.argmax(up[cands]))]
                        r_lb[o_idx] = r_l
                        r_ub[o_idx] = r_u
                        u_max[o_idx] = up[r_u]
                        others = cands[cands != r_u]
                        if len(others) > 0:
                            slack[o_idx] = max(0.0, float(np.max(up[others]) - lp[r_u]))

                mid_dom = 0.5 * (lp[r_ub] + up[r_ub])
                use_shifted = (mid_dom + slack) < u_max

                G_lb = sp.csr_array(
                    (np.ones(ko, dtype=np.float64), (np.arange(ko, dtype=np.int32), r_lb.astype(np.int32))),
                    shape=(ko, in_dim_p),
                )
                gamma_lb = np.zeros(ko, dtype=np.float64)

                ub_weights = use_shifted.astype(np.float64)
                G_ub = sp.csr_array(
                    (ub_weights, (np.arange(ko, dtype=np.int32), r_ub.astype(np.int32))),
                    shape=(ko, in_dim_p),
                )
                gamma_ub = np.where(use_shifted, slack, u_max)

                node_relax[nid] = {
                    'type': 'explicit_matrix',
                    'G_lb': G_lb, 'gamma_lb': gamma_lb,
                    'G_ub': G_ub, 'gamma_ub': gamma_ub,
                    'r_lb': r_lb, 'r_ub': r_ub,
                    'use_shifted': use_shifted, 'slack': slack, 'u_max': u_max,
                    'l_pre': lp, 'u_pre': up,
                }

            elif op == 'layernorm':
                A_lb_p, b_lb_p, A_ub_p, b_ub_p = node_linear_bounds[p_id]
                W_ln, b_ln_lb, b_ln_ub = _build_layernorm_crown_envelope(
                    node.layer_ref, A_lb_p, b_lb_p, A_ub_p, b_ub_p, pred_lb, pred_ub
                )
                node_relax[nid] = {
                    'type': 'shared_matrix',
                    'W': W_ln, 'b_lb': b_ln_lb, 'b_ub': b_ln_ub,
                }

            elif op == 'mhsa':
                A_lb_p, b_lb_p, A_ub_p, b_ub_p = node_linear_bounds[p_id]
                W_mhsa, b_mhsa_lb, b_mhsa_ub = _build_mhsa_crown_envelope(
                    node.layer_ref, A_lb_p, b_lb_p, A_ub_p, b_ub_p, pred_lb, pred_ub
                )
                node_relax[nid] = {
                    'type': 'shared_matrix',
                    'W': W_mhsa, 'b_lb': b_mhsa_lb, 'b_ub': b_mhsa_ub,
                }

            # If this node itself is a parent of a future nonlinear node, compute its bounds now
            if (nid in parent_needs_bounds and nid not in node_bounds) or nid in parent_needs_linear:
                A_lb_n, b_lb_n, A_ub_n, b_ub_n = self.backward_propagate(
                    graph, inp_spec, target_node_id=nid, C_spec=None,
                    node_relax=node_relax, split_constraints=split_constraints
                )
                ln, un = self._concretize_bounds(A_lb_n, b_lb_n, A_ub_n, b_ub_n, pred_lb, pred_ub)
                if nid in initial_pre_bounds:
                    ln0, un0 = initial_pre_bounds[nid]
                    ln = np.maximum(ln, ln0)
                    un = np.minimum(un, un0)
                if op == 'relu':
                    ln = np.maximum(ln, 0.0)
                    un = np.maximum(un, 0.0)
                ln = np.minimum(ln, un)
                node_bounds[nid] = (ln, un)
                node_linear_bounds[nid] = (A_lb_n, b_lb_n, A_ub_n, b_ub_n)

        return node_bounds, node_relax

    def backward_propagate(
        self,
        graph,
        inp_spec,
        target_node_id=None,
        C_spec=None,
        d_spec=None,
        node_relax=None,
        split_constraints=None,
        compute_upper=True,
    ):
        """
        Propagate linear specification C_spec @ z^{(target_node_id)} - d_spec
        backward through the topological DAG down to the active input predicates alpha_0:
            A_lb @ alpha_0 + b_lb <= C_spec @ z^{(target)} - d_spec <= A_ub @ alpha_0 + b_ub.
        """
        if target_node_id is None:
            target_node_id = graph.output_node_id
        target_dim = graph.nodes[target_node_id].out_dim

        if C_spec is None:
            A_L_init = np.eye(target_dim, dtype=np.float64)
        else:
            A_L_init = np.asarray(C_spec, dtype=np.float64)
            if A_L_init.ndim == 1:
                A_L_init = A_L_init.reshape(1, -1)

        K = A_L_init.shape[0]
        if d_spec is None:
            b_L = np.zeros(K, dtype=np.float64)
            b_U = np.zeros(K, dtype=np.float64) if compute_upper else None
        else:
            d_vec = np.asarray(d_spec, dtype=np.float64).ravel()
            b_L = -d_vec.copy()
            b_U = -d_vec.copy() if compute_upper else None

        A_L_map = {target_node_id: A_L_init.copy()}
        A_U_map = {target_node_id: A_L_init.copy()} if compute_upper else {}

        def _add_to_map(m_dict, nid_key, mat):
            if nid_key in m_dict:
                m_dict[nid_key] = m_dict[nid_key] + mat
            else:
                m_dict[nid_key] = mat

        for nid in range(target_node_id, 0, -1):
            if nid not in A_L_map:
                continue
            node = graph.nodes[nid]
            op = node.op
            A_L = A_L_map.pop(nid)
            A_U = A_U_map.pop(nid) if compute_upper else None

            if op == 'identity':
                p_id = node.inputs[0]
                _add_to_map(A_L_map, p_id, A_L)
                if compute_upper:
                    _add_to_map(A_U_map, p_id, A_U)

            elif op == 'permute':
                p_id = node.inputs[0]
                _add_to_map(A_L_map, p_id, A_L[:, node.inv_perm])
                if compute_upper:
                    _add_to_map(A_U_map, p_id, A_U[:, node.inv_perm])

            elif op == 'add':
                p0, p1 = node.inputs[0], node.inputs[1]
                _add_to_map(A_L_map, p0, A_L)
                _add_to_map(A_L_map, p1, A_L)
                if compute_upper:
                    _add_to_map(A_U_map, p0, A_U)
                    _add_to_map(A_U_map, p1, A_U)

            elif op == 'scale_shift':
                p_id = node.inputs[0]
                b_L = b_L + A_L @ node.shift
                _add_to_map(A_L_map, p_id, A_L * node.scale[None, :])
                if compute_upper:
                    b_U = b_U + A_U @ node.shift
                    _add_to_map(A_U_map, p_id, A_U * node.scale[None, :])

            elif op == 'affine':
                p_id = node.inputs[0]
                b_L = b_L + A_L @ node.b
                A_L_prev = np.asarray(A_L @ node.W, dtype=np.float64)
                _add_to_map(A_L_map, p_id, A_L_prev)
                if compute_upper:
                    b_U = b_U + A_U @ node.b
                    A_U_prev = np.asarray(A_U @ node.W, dtype=np.float64)
                    _add_to_map(A_U_map, p_id, A_U_prev)

            elif op in ('relu', 'leaky_relu', 'satlin', 'satlins', 'sigmoid', 'tanh', 'gelu'):
                p_id = node.inputs[0]
                rel = node_relax[nid]
                a_lb, beta_lb = rel['a_lb'], rel['b_lb']
                a_ub, beta_ub = rel['a_ub'], rel['b_ub']

                A_L_pos = np.maximum(A_L, 0.0)
                A_L_neg = np.minimum(A_L, 0.0)
                b_L = b_L + A_L_pos @ beta_lb + A_L_neg @ beta_ub
                if a_lb.ndim == 2:
                    A_L_prev = A_L_pos * a_lb + A_L_neg * a_ub[None, :]
                else:
                    A_L_prev = A_L_pos * a_lb[None, :] + A_L_neg * a_ub[None, :]
                _add_to_map(A_L_map, p_id, A_L_prev)

                if compute_upper:
                    A_U_pos = np.maximum(A_U, 0.0)
                    A_U_neg = np.minimum(A_U, 0.0)
                    b_U = b_U + A_U_pos @ beta_ub + A_U_neg @ beta_lb
                    if a_lb.ndim == 2:
                        A_U_prev = A_U_pos * a_ub[None, :] + A_U_neg * a_lb
                    else:
                        A_U_prev = A_U_pos * a_ub[None, :] + A_U_neg * a_lb[None, :]
                    _add_to_map(A_U_map, p_id, A_U_prev)

            elif op == 'maxpool2d':
                p_id = node.inputs[0]
                rel = node_relax[nid]
                G_lb, gamma_lb = rel['G_lb'], rel['gamma_lb']
                G_ub, gamma_ub = rel['G_ub'], rel['gamma_ub']

                A_L_pos = np.maximum(A_L, 0.0)
                A_L_neg = np.minimum(A_L, 0.0)
                b_L = b_L + A_L_pos @ gamma_lb + A_L_neg @ gamma_ub
                A_L_prev = np.asarray(A_L_pos @ G_lb + A_L_neg @ G_ub, dtype=np.float64)
                _add_to_map(A_L_map, p_id, A_L_prev)

                if compute_upper:
                    A_U_pos = np.maximum(A_U, 0.0)
                    A_U_neg = np.minimum(A_U, 0.0)
                    b_U = b_U + A_U_pos @ gamma_ub + A_U_neg @ gamma_lb
                    A_U_prev = np.asarray(A_U_pos @ G_ub + A_U_neg @ G_lb, dtype=np.float64)
                    _add_to_map(A_U_map, p_id, A_U_prev)

            elif op in ('layernorm', 'mhsa'):
                p_id = node.inputs[0]
                rel = node_relax[nid]
                W_mat = rel['W']
                beta_lb, beta_ub = rel['b_lb'], rel['b_ub']

                A_L_pos = np.maximum(A_L, 0.0)
                A_L_neg = np.minimum(A_L, 0.0)
                b_L = b_L + A_L_pos @ beta_lb + A_L_neg @ beta_ub
                A_L_prev = np.asarray(A_L @ W_mat, dtype=np.float64)
                _add_to_map(A_L_map, p_id, A_L_prev)

                if compute_upper:
                    A_U_pos = np.maximum(A_U, 0.0)
                    A_U_neg = np.minimum(A_U, 0.0)
                    b_U = b_U + A_U_pos @ beta_ub + A_U_neg @ beta_lb
                    A_U_prev = np.asarray(A_U @ W_mat, dtype=np.float64)
                    _add_to_map(A_U_map, p_id, A_U_prev)
            else:
                raise ValueError(f"Unsupported op {op} in backward_propagate")

        A_x_L = A_L_map.get(0, np.zeros((K, graph.in_dim), dtype=np.float64))
        b_alpha_L = b_L + A_x_L @ inp_spec.c0
        A_alpha_L = A_x_L if inp_spec.is_identity_V0 else (A_x_L @ inp_spec.V0)

        if not compute_upper:
            return A_alpha_L, b_alpha_L, None, None

        A_x_U = A_U_map.get(0, np.zeros((K, graph.in_dim), dtype=np.float64))
        b_alpha_U = b_U + A_x_U @ inp_spec.c0
        A_alpha_U = A_x_U if inp_spec.is_identity_V0 else (A_x_U @ inp_spec.V0)
        return A_alpha_L, b_alpha_L, A_alpha_U, b_alpha_U

    # -----------------------------------------------------------------
    # PyTorch Differentiable alpha-CROWN & alpha,beta-CROWN Optimizer
    # -----------------------------------------------------------------
    def _optimize_alpha_beta_crown_torch(
        self,
        graph,
        inp_spec,
        node_relax,
        C_spec=None,
        d_spec=None,
        pred_lb=None,
        pred_ub=None,
        split_constraints=None,
        optimize_upper=True,
        num_iters=None,
        lr_alpha=None,
        lr_beta=None,
    ):
        """
        Differentiable PyTorch optimization of activation slopes alpha in [0, 1]
        (for ReLU and LeakyReLU) and Lagrangian multipliers beta >= 0 (for split constraints).

        Tracks the best sound linear lower bound (A_lb*, b_lb*, z_lb*) and best sound
        linear upper bound (A_ub*, b_ub*, z_ub*) across all iterations per specification row k,
        guaranteeing strict monotonic improvement over classic CROWN.
        """
        if pred_lb is None:
            pred_lb = inp_spec.pred_lb
        if pred_ub is None:
            pred_ub = inp_spec.pred_ub
        split_constraints = split_constraints or {}

        # First compute baseline classic CROWN bounds
        A_lb_best, b_lb_best, A_ub_best, b_ub_best = self.backward_propagate(
            graph, inp_spec, target_node_id=graph.output_node_id,
            C_spec=C_spec, d_spec=d_spec, node_relax=node_relax,
            split_constraints=split_constraints, compute_upper=optimize_upper
        )
        z_lb_best, z_ub_best = self._concretize_bounds(
            A_lb_best, b_lb_best, A_ub_best, b_ub_best, pred_lb, pred_ub
        )

        # Check if there are any optimizable unstable ReLUs/LeakyReLUs or active beta splits
        unstable_relus = {}
        for node in graph.nodes[1:]:
            if node.op in ('relu', 'leaky_relu') and node.id in node_relax:
                rel = node_relax[node.id]
                lp, up = rel['l_pre'], rel['u_pre']
                unst_idx = np.where((lp < 0.0) & (up > 0.0))[0]
                if len(unst_idx) > 0:
                    unstable_relus[node.id] = unst_idx

        has_splits = any(len(v) > 0 for v in split_constraints.values())
        if len(unstable_relus) == 0 and not has_splits:
            return A_lb_best, b_lb_best, z_lb_best, A_ub_best, b_ub_best, z_ub_best, {}

        target_id = graph.output_node_id
        target_dim = graph.nodes[target_id].out_dim
        if C_spec is None:
            C_mat = np.eye(target_dim, dtype=np.float64)
        else:
            C_mat = np.asarray(C_spec, dtype=np.float64)
            if C_mat.ndim == 1:
                C_mat = C_mat.reshape(1, -1)
        K = C_mat.shape[0]
        d_vec = np.zeros(K, dtype=np.float64) if d_spec is None else np.asarray(d_spec, dtype=np.float64).ravel()

        if num_iters is None:
            num_iters = self.beta_iters if has_splits else self.alpha_iters
        if lr_alpha is None:
            lr_alpha = self.alpha_lr
        if lr_beta is None:
            lr_beta = self.beta_lr

        if num_iters <= 0:
            return A_lb_best, b_lb_best, z_lb_best, A_ub_best, b_ub_best, z_ub_best, {}

        dtype = torch.float64
        C_t = torch.tensor(C_mat, dtype=dtype)
        d_t = torch.tensor(d_vec, dtype=dtype)
        c0_t = torch.tensor(inp_spec.c0, dtype=dtype)
        V0_t = None if inp_spec.is_identity_V0 else torch.tensor(inp_spec.V0, dtype=dtype)
        lb0_t = torch.tensor(pred_lb, dtype=dtype)
        ub0_t = torch.tensor(pred_ub, dtype=dtype)
        mu_box_t = 0.5 * (lb0_t + ub0_t)
        rad_box_t = 0.5 * (ub0_t - lb0_t)

        # Prepare torch operators for each node
        torch_nodes = {}
        for node in graph.nodes[1:]:
            nid = node.id
            op = node.op
            if op == 'affine':
                if sp.issparse(node.W):
                    if node.W.shape[0] * node.W.shape[1] <= 4_000_000:
                        W_t = torch.tensor(node.W.toarray(), dtype=dtype)
                        is_sp = False
                    else:
                        W_coo = node.W.T.tocoo()
                        indices = torch.tensor(np.vstack([W_coo.row, W_coo.col]), dtype=torch.long)
                        values = torch.tensor(W_coo.data, dtype=dtype)
                        W_t = torch.sparse_coo_tensor(indices, values, size=W_coo.shape, dtype=dtype).coalesce()
                        is_sp = True
                else:
                    W_t = torch.tensor(node.W, dtype=dtype)
                    is_sp = False
                b_t = torch.tensor(node.b, dtype=dtype)
                torch_nodes[nid] = {'W': W_t, 'b': b_t, 'is_sp': is_sp}
            elif op == 'scale_shift':
                torch_nodes[nid] = {
                    'scale': torch.tensor(node.scale, dtype=dtype),
                    'shift': torch.tensor(node.shift, dtype=dtype),
                }
            elif op == 'permute':
                torch_nodes[nid] = {
                    'inv_perm': torch.tensor(node.inv_perm, dtype=torch.long)
                }
            elif op in ('relu', 'leaky_relu', 'satlin', 'satlins', 'sigmoid', 'tanh', 'gelu'):
                rel = node_relax[nid]
                torch_nodes[nid] = {
                    'a_lb_base': torch.tensor(rel['a_lb'], dtype=dtype),
                    'b_lb': torch.tensor(rel['b_lb'], dtype=dtype),
                    'a_ub': torch.tensor(rel['a_ub'], dtype=dtype),
                    'b_ub': torch.tensor(rel['b_ub'], dtype=dtype),
                }
            elif op == 'maxpool2d':
                rel = node_relax[nid]
                torch_nodes[nid] = {
                    'G_lb': torch.tensor(rel['G_lb'].toarray(), dtype=dtype),
                    'gamma_lb': torch.tensor(rel['gamma_lb'], dtype=dtype),
                    'G_ub': torch.tensor(rel['G_ub'].toarray(), dtype=dtype),
                    'gamma_ub': torch.tensor(rel['gamma_ub'], dtype=dtype),
                }
            elif op in ('layernorm', 'mhsa'):
                rel = node_relax[nid]
                W_m = rel['W'].toarray() if sp.issparse(rel['W']) else rel['W']
                torch_nodes[nid] = {
                    'W': torch.tensor(W_m, dtype=dtype),
                    'b_lb': torch.tensor(rel['b_lb'], dtype=dtype),
                    'b_ub': torch.tensor(rel['b_ub'], dtype=dtype),
                }

        # Create learnable parameters for lower and (optionally) upper bound passes
        alpha_params_lb = {}
        alpha_params_ub = {}
        param_groups = []

        for nid, unst_idx in unstable_relus.items():
            base_a = node_relax[nid]['a_lb'][unst_idx]
            init_t = torch.tensor(np.tile(base_a[None, :], (K, 1)), dtype=dtype, requires_grad=True)
            alpha_params_lb[nid] = (torch.tensor(unst_idx, dtype=torch.long), init_t)
            param_groups.append({'params': [init_t], 'lr': lr_alpha})
            if optimize_upper:
                init_u = torch.tensor(np.tile(base_a[None, :], (K, 1)), dtype=dtype, requires_grad=True)
                alpha_params_ub[nid] = (torch.tensor(unst_idx, dtype=torch.long), init_u)
                param_groups.append({'params': [init_u], 'lr': lr_alpha})

        # Create learnable beta parameters for split constraints (attached at ReLU node nid)
        beta_params_lb = {}
        beta_params_ub = {}
        for nid, splits_dict in split_constraints.items():
            if len(splits_dict) == 0:
                continue
            s_idxs = sorted(splits_dict.keys())
            s_signs = [float(splits_dict[j]) for j in s_idxs]
            idx_t = torch.tensor(s_idxs, dtype=torch.long)
            sign_t = torch.tensor(s_signs, dtype=dtype)
            b_lb_p = torch.zeros((K, len(s_idxs)), dtype=dtype, requires_grad=True)
            beta_params_lb[nid] = (idx_t, sign_t, b_lb_p)
            param_groups.append({'params': [b_lb_p], 'lr': lr_beta})
            if optimize_upper:
                b_ub_p = torch.zeros((K, len(s_idxs)), dtype=dtype, requires_grad=True)
                beta_params_ub[nid] = (idx_t, sign_t, b_ub_p)
                param_groups.append({'params': [b_ub_p], 'lr': lr_beta})

        optimizer = torch.optim.Adam(param_groups)

        def _run_backward_torch(is_lower):
            A_map = {target_id: C_t}
            b_acc = -d_t.clone()
            a_params = alpha_params_lb if is_lower else alpha_params_ub
            b_params = beta_params_lb if is_lower else beta_params_ub

            for nid in range(target_id, 0, -1):
                if nid not in A_map:
                    continue
                node = graph.nodes[nid]
                op = node.op
                A_cur = A_map.pop(nid)
                tn = torch_nodes.get(nid)

                if op == 'identity':
                    p_id = node.inputs[0]
                    A_map[p_id] = A_map[p_id] + A_cur if p_id in A_map else A_cur

                elif op == 'permute':
                    p_id = node.inputs[0]
                    A_prev = A_cur[:, tn['inv_perm']]
                    A_map[p_id] = A_map[p_id] + A_prev if p_id in A_map else A_prev

                elif op == 'add':
                    p0, p1 = node.inputs[0], node.inputs[1]
                    A_map[p0] = A_map[p0] + A_cur if p0 in A_map else A_cur
                    A_map[p1] = A_map[p1] + A_cur if p1 in A_map else A_cur

                elif op == 'scale_shift':
                    p_id = node.inputs[0]
                    b_acc = b_acc + torch.mv(A_cur, tn['shift'])
                    A_prev = A_cur * tn['scale'].unsqueeze(0)
                    A_map[p_id] = A_map[p_id] + A_prev if p_id in A_map else A_prev

                elif op == 'affine':
                    p_id = node.inputs[0]
                    b_acc = b_acc + torch.mv(A_cur, tn['b'])
                    if tn['is_sp']:
                        A_prev = torch.sparse.mm(tn['W'], A_cur.T).T
                    else:
                        A_prev = torch.matmul(A_cur, tn['W'])
                    A_map[p_id] = A_map[p_id] + A_prev if p_id in A_map else A_prev

                elif op in ('relu', 'leaky_relu', 'satlin', 'satlins', 'sigmoid', 'tanh', 'gelu'):
                    p_id = node.inputs[0]
                    A_pos = torch.clamp(A_cur, min=0.0)
                    A_neg = torch.clamp(A_cur, max=0.0)
                    a_ub_v = tn['a_ub'].unsqueeze(0)
                    b_lb_v = tn['b_lb']
                    b_ub_v = tn['b_ub']

                    if nid in a_params:
                        unst_idx_t, raw_p = a_params[nid]
                        a_lb_mat = tn['a_lb_base'].unsqueeze(0).expand(K, -1).clone()
                        if op == 'relu':
                            clamped = torch.clamp(raw_p, 0.0, 1.0)
                        else:
                            g_val = float(node.gamma)
                            clamped = torch.clamp(raw_p, g_val, 1.0)
                        a_lb_mat[:, unst_idx_t] = clamped
                    else:
                        a_lb_mat = tn['a_lb_base'].unsqueeze(0)

                    if is_lower:
                        b_acc = b_acc + torch.mv(A_pos, b_lb_v) + torch.mv(A_neg, b_ub_v)
                        A_prev = A_pos * a_lb_mat + A_neg * a_ub_v
                    else:
                        b_acc = b_acc + torch.mv(A_pos, b_ub_v) + torch.mv(A_neg, b_lb_v)
                        A_prev = A_pos * a_ub_v + A_neg * a_lb_mat

                    # Inject Lagrangian multiplier beta >= 0 for split constraints
                    if nid in b_params:
                        s_idx_t, s_sign_t, raw_beta = b_params[nid]
                        beta_pos = torch.clamp(raw_beta, min=0.0)
                        # On feasible branch s_j * x_j <= 0:
                        # Lower bound adds +beta * s_j * x_j; Upper bound adds -beta * s_j * x_j
                        delta = beta_pos * s_sign_t.unsqueeze(0)
                        A_prev = A_prev.clone()
                        if is_lower:
                            A_prev[:, s_idx_t] = A_prev[:, s_idx_t] + delta
                        else:
                            A_prev[:, s_idx_t] = A_prev[:, s_idx_t] - delta

                    A_map[p_id] = A_map[p_id] + A_prev if p_id in A_map else A_prev

                elif op == 'maxpool2d':
                    p_id = node.inputs[0]
                    A_pos = torch.clamp(A_cur, min=0.0)
                    A_neg = torch.clamp(A_cur, max=0.0)
                    if is_lower:
                        b_acc = b_acc + torch.mv(A_pos, tn['gamma_lb']) + torch.mv(A_neg, tn['gamma_ub'])
                        A_prev = torch.matmul(A_pos, tn['G_lb']) + torch.matmul(A_neg, tn['G_ub'])
                    else:
                        b_acc = b_acc + torch.mv(A_pos, tn['gamma_ub']) + torch.mv(A_neg, tn['gamma_lb'])
                        A_prev = torch.matmul(A_pos, tn['G_ub']) + torch.matmul(A_neg, tn['G_lb'])
                    A_map[p_id] = A_map[p_id] + A_prev if p_id in A_map else A_prev

                elif op in ('layernorm', 'mhsa'):
                    p_id = node.inputs[0]
                    A_pos = torch.clamp(A_cur, min=0.0)
                    A_neg = torch.clamp(A_cur, max=0.0)
                    if is_lower:
                        b_acc = b_acc + torch.mv(A_pos, tn['b_lb']) + torch.mv(A_neg, tn['b_ub'])
                    else:
                        b_acc = b_acc + torch.mv(A_pos, tn['b_ub']) + torch.mv(A_neg, tn['b_lb'])
                    A_prev = torch.matmul(A_cur, tn['W'])
                    A_map[p_id] = A_map[p_id] + A_prev if p_id in A_map else A_prev

            A_x = A_map[0]
            b_alpha = b_acc + torch.mv(A_x, c0_t)
            A_alpha = A_x if V0_t is None else torch.matmul(A_x, V0_t)
            if is_lower:
                z_bound = b_alpha + torch.mv(A_alpha, mu_box_t) - torch.mv(torch.abs(A_alpha), rad_box_t)
            else:
                z_bound = b_alpha + torch.mv(A_alpha, mu_box_t) + torch.mv(torch.abs(A_alpha), rad_box_t)
            return A_alpha, b_alpha, z_bound

        for _ in range(num_iters):
            optimizer.zero_grad()
            A_lb_t, b_lb_t, z_lb_t = _run_backward_torch(is_lower=True)
            loss = -torch.sum(z_lb_t)

            if optimize_upper:
                A_ub_t, b_ub_t, z_ub_t = _run_backward_torch(is_lower=False)
                loss = loss + torch.sum(z_ub_t)

            # Update row-wise best sound bounds before stepping
            z_lb_np = z_lb_t.detach().numpy()
            improved_lb = z_lb_np > z_lb_best
            if np.any(improved_lb):
                A_lb_np = A_lb_t.detach().numpy()
                b_lb_np = b_lb_t.detach().numpy()
                z_lb_best[improved_lb] = z_lb_np[improved_lb]
                A_lb_best[improved_lb] = A_lb_np[improved_lb]
                b_lb_best[improved_lb] = b_lb_np[improved_lb]

            if optimize_upper:
                z_ub_np = z_ub_t.detach().numpy()
                improved_ub = z_ub_np < z_ub_best
                if np.any(improved_ub):
                    A_ub_np = A_ub_t.detach().numpy()
                    b_ub_np = b_ub_t.detach().numpy()
                    z_ub_best[improved_ub] = z_ub_np[improved_ub]
                    A_ub_best[improved_ub] = A_ub_np[improved_ub]
                    b_ub_best[improved_ub] = b_ub_np[improved_ub]

            loss.backward()
            optimizer.step()

            # Clamp alpha parameters in-place
            with torch.no_grad():
                for nid, (_, p_t) in alpha_params_lb.items():
                    low_val = 0.0 if graph.nodes[nid].op == 'relu' else float(graph.nodes[nid].gamma)
                    p_t.clamp_(low_val, 1.0)
                for nid, (_, p_t) in alpha_params_ub.items():
                    low_val = 0.0 if graph.nodes[nid].op == 'relu' else float(graph.nodes[nid].gamma)
                    p_t.clamp_(low_val, 1.0)
                for _, (_, _, b_t) in beta_params_lb.items():
                    b_t.clamp_(min=0.0)
                for _, (_, _, b_t) in beta_params_ub.items():
                    b_t.clamp_(min=0.0)

        # Final evaluation after last step
        with torch.no_grad():
            A_lb_t, b_lb_t, z_lb_t = _run_backward_torch(is_lower=True)
            z_lb_np = z_lb_t.numpy()
            improved_lb = z_lb_np > z_lb_best
            if np.any(improved_lb):
                z_lb_best[improved_lb] = z_lb_np[improved_lb]
                A_lb_best[improved_lb] = A_lb_t.numpy()[improved_lb]
                b_lb_best[improved_lb] = b_lb_t.numpy()[improved_lb]

            if optimize_upper:
                A_ub_t, b_ub_t, z_ub_t = _run_backward_torch(is_lower=False)
                z_ub_np = z_ub_t.numpy()
                improved_ub = z_ub_np < z_ub_best
                if np.any(improved_ub):
                    z_ub_best[improved_ub] = z_ub_np[improved_ub]
                    A_ub_best[improved_ub] = A_ub_t.numpy()[improved_ub]
                    b_ub_best[improved_ub] = b_ub_t.numpy()[improved_ub]

        # Extract mean optimized alpha per ReLU node for intermediate bound reuse
        learned_alphas = {}
        with torch.no_grad():
            for nid, (unst_idx_t, p_t) in alpha_params_lb.items():
                full_a = node_relax[nid]['a_lb'].copy()
                full_a[unst_idx_t.numpy()] = p_t.mean(dim=0).numpy()
                learned_alphas[nid] = full_a

        if self.use_input_constraints and inp_spec.has_constraints:
            z_lb_best, z_ub_best = self._concretize_bounds(
                A_lb_best, b_lb_best, A_ub_best, b_ub_best,
                pred_lb, pred_ub, inp_spec=inp_spec, use_lp=True
            )

        return A_lb_best, b_lb_best, z_lb_best, A_ub_best, b_ub_best, z_ub_best, learned_alphas

    def compute_bounds(
        self,
        net_or_graph,
        in_set,
        C_spec=None,
        d_spec=None,
        pred_lb=None,
        pred_ub=None,
        split_constraints=None,
        initial_pre_bounds=None,
        optimize_upper=True,
    ):
        """
        Compute end-to-end linear bounds and concretized interval bounds:
            A_lb @ alpha_0 + b_lb <= C_spec @ f(x(alpha_0)) - d_spec <= A_ub @ alpha_0 + b_ub
            z_lb <= C_spec @ f(x(alpha_0)) - d_spec <= z_ub
        using CROWN or alpha-CROWN.
        """
        inp_spec = in_set if isinstance(in_set, _InputSpec) else _InputSpec(in_set)
        graph = (
            net_or_graph
            if isinstance(net_or_graph, CROWNNetworkGraph)
            else CROWNNetworkGraph(net_or_graph, inp_spec.in_shape)
        )
        if pred_lb is None:
            pred_lb = inp_spec.pred_lb
        if pred_ub is None:
            pred_ub = inp_spec.pred_ub

        node_bounds, node_relax = self.compute_node_relaxations_and_pre_bounds(
            graph, inp_spec, pred_lb=pred_lb, pred_ub=pred_ub,
            split_constraints=split_constraints, initial_pre_bounds=initial_pre_bounds
        )

        has_splits = bool(split_constraints and any(len(v) > 0 for v in split_constraints.values()))
        if self.mode in ('alpha-crown', 'alpha-beta-crown') or has_splits:
            # Optional intermediate pre-activation bound tightening for multi-ReLU networks
            relu_node_ids = [
                n.id for n in graph.nodes[1:]
                if n.op in ('relu', 'leaky_relu') and n.id in node_relax
            ]
            if len(relu_node_ids) >= 2 and not has_splits and self.alpha_iters > 0:
                tight_pre = dict(initial_pre_bounds) if initial_pre_bounds else {}
                for r_idx, r_nid in enumerate(relu_node_ids):
                    if r_idx == 0:
                        continue
                    p_id = graph.nodes[r_nid].inputs[0]
                    lp_cur, up_cur = node_bounds[p_id]
                    unst = np.where((lp_cur < 0.0) & (up_cur > 0.0))[0]
                    if 0 < len(unst) <= 256:
                        C_inter = np.zeros((len(unst), graph.nodes[p_id].out_dim), dtype=np.float64)
                        C_inter[np.arange(len(unst)), unst] = 1.0
                        sub_graph = copy.copy(graph)
                        sub_graph.output_node_id = p_id
                        _, _, l_opt, _, _, u_opt, _ = self._optimize_alpha_beta_crown_torch(
                            sub_graph, inp_spec, node_relax, C_spec=C_inter,
                            pred_lb=pred_lb, pred_ub=pred_ub,
                            optimize_upper=True, num_iters=max(8, self.alpha_iters // 3),
                            lr_alpha=self.alpha_lr
                        )
                        lp_new = lp_cur.copy()
                        up_new = up_cur.copy()
                        lp_new[unst] = np.maximum(lp_new[unst], l_opt)
                        up_new[unst] = np.minimum(up_new[unst], u_opt)
                        lp_new = np.minimum(lp_new, up_new)
                        tight_pre[p_id] = (lp_new, up_new)
                        node_bounds, node_relax = self.compute_node_relaxations_and_pre_bounds(
                            graph, inp_spec, pred_lb=pred_lb, pred_ub=pred_ub,
                            split_constraints=split_constraints, initial_pre_bounds=tight_pre
                        )

            A_lb, b_lb, z_lb, A_ub, b_ub, z_ub, _ = self._optimize_alpha_beta_crown_torch(
                graph, inp_spec, node_relax, C_spec=C_spec, d_spec=d_spec,
                pred_lb=pred_lb, pred_ub=pred_ub, split_constraints=split_constraints,
                optimize_upper=optimize_upper
            )
            return A_lb, b_lb, z_lb, A_ub, b_ub, z_ub, node_bounds, node_relax

        A_lb, b_lb, A_ub, b_ub = self.backward_propagate(
            graph, inp_spec, target_node_id=graph.output_node_id,
            C_spec=C_spec, d_spec=d_spec, node_relax=node_relax,
            split_constraints=split_constraints, compute_upper=optimize_upper
        )
        z_lb, z_ub = self._concretize_bounds(
            A_lb, b_lb, A_ub, b_ub, pred_lb, pred_ub,
            inp_spec=inp_spec, use_lp=self.use_input_constraints
        )
        return A_lb, b_lb, z_lb, A_ub, b_ub, z_ub, node_bounds, node_relax

    # -----------------------------------------------------------------
    # Branch-and-Bound Smart Branching Heuristic (BaBSR) & Counterexample Search
    # -----------------------------------------------------------------
    def _sample_and_pgd_counterexample(
        self, graph, inp_spec, C_safety, d_safety, pred_lb, pred_ub, num_random=32, pgd_steps=15
    ):
        """
        Search for a concrete counterexample alpha_0 in [pred_lb, pred_ub] satisfying
        all rows of C_safety @ f(x(alpha_0)) <= d_safety (i.e. max_k (C_k y - d_k) <= 0).
        Uses center + corner + random samples + finite-difference / linear-bound guided step.
        """
        C_safety = np.asarray(C_safety, dtype=np.float64)
        if C_safety.ndim == 1:
            C_safety = C_safety.reshape(1, -1)
        d_safety = np.asarray(d_safety, dtype=np.float64).ravel()

        if inp_spec.has_constraints:
            return None, None

        rng = np.random.default_rng(42)
        mid = 0.5 * (pred_lb + pred_ub)
        cands = [mid, pred_lb, pred_ub]
        if inp_spec.nVars > 0 and num_random > 0:
            rand_pts = rng.uniform(pred_lb[:, None], pred_ub[:, None], size=(inp_spec.nVars, num_random))
            cands.append(rand_pts)
        alpha_batch = np.column_stack(cands) if len(cands) > 0 else mid[:, None]
        x_batch = inp_spec.c0[:, None] + inp_spec.V0 @ alpha_batch
        y_batch = graph.evaluate_flat(x_batch)
        if y_batch.ndim == 1:
            y_batch = y_batch[:, None]

        lhs = C_safety @ y_batch - d_safety[:, None]  # (K, B)
        max_violation = np.max(lhs, axis=0)           # <= 0 means ALL conjunctive rows satisfied!
        sat_idx = np.where(max_violation <= 1e-12)[0]
        if len(sat_idx) > 0:
            best_i = int(sat_idx[0])
            return x_batch[:, best_i], y_batch[:, best_i]

        # Guided PGD from best candidate if nVars <= 1024
        if pgd_steps > 0 and 0 < inp_spec.nVars <= 1024:
            best_i = int(np.argmin(max_violation))
            alpha_cur = alpha_batch[:, best_i].copy()
            step_size = 0.25 * (pred_ub - pred_lb)
            for _ in range(pgd_steps):
                x_cur = inp_spec.c0 + inp_spec.V0 @ alpha_cur
                y_cur = graph.evaluate_flat(x_cur)
                viol = C_safety @ y_cur - d_safety
                if np.all(viol <= 1e-12):
                    return x_cur, y_cur
                # Estimate gradient of active row via SPSA (2 evaluations per step)
                active_k = int(np.argmax(viol))
                delta = rng.choice([-1.0, 1.0], size=inp_spec.nVars) * np.maximum(1e-5 * (pred_ub - pred_lb), 1e-7)
                x_plus = inp_spec.c0 + inp_spec.V0 @ np.clip(alpha_cur + delta, pred_lb, pred_ub)
                x_minus = inp_spec.c0 + inp_spec.V0 @ np.clip(alpha_cur - delta, pred_lb, pred_ub)
                f_plus = float(C_safety[active_k] @ graph.evaluate_flat(x_plus) - d_safety[active_k])
                f_minus = float(C_safety[active_k] @ graph.evaluate_flat(x_minus) - d_safety[active_k])
                grad_sign = np.sign(f_plus - f_minus) * np.sign(delta)
                alpha_cur = np.clip(alpha_cur - step_size * grad_sign, pred_lb, pred_ub)
                step_size *= 0.85

            x_cur = inp_spec.c0 + inp_spec.V0 @ alpha_cur
            y_cur = graph.evaluate_flat(x_cur)
            if np.all(C_safety @ y_cur - d_safety <= 1e-12):
                return x_cur, y_cur

        return None, None

    def _select_babsr_relu_split(self, graph, C_spec_row, node_relax, split_constraints):
        """
        Select the best unstable ReLU neuron (relu_node_id, neuron_idx) using the
        BaBSR (Branch-and-Bound Smart Branching) heuristic:
        Computes backward linear sensitivity A^{(nid)} from C_spec_row and scores each
        unstable neuron j by |A^{(nid)}_j| * (-l_j * u_j) / (u_j - l_j).
        """
        target_id = graph.output_node_id
        A_map = {target_id: np.asarray(C_spec_row, dtype=np.float64).reshape(1, -1)}

        best_score = -1.0
        best_choice = None

        for nid in range(target_id, 0, -1):
            if nid not in A_map:
                continue
            node = graph.nodes[nid]
            op = node.op
            A_cur = A_map.pop(nid)
            p_id = node.inputs[0] if len(node.inputs) > 0 else None

            if op == 'relu':
                rel = node_relax[nid]
                lp, up = rel['l_pre'], rel['u_pre']
                already_split = split_constraints.get(nid, {})
                unst_mask = (lp < -1e-9) & (up > 1e-9)
                for j_fixed in already_split.keys():
                    unst_mask[j_fixed] = False
                if np.any(unst_mask):
                    unst_idxs = np.where(unst_mask)[0]
                    lu = lp[unst_idxs]
                    uu = up[unst_idxs]
                    intercept = (-lu * uu) / np.maximum(uu - lu, 1e-15)
                    sens = np.abs(A_cur[0, unst_idxs])
                    scores = (sens + 1e-6) * intercept
                    top_pos = int(np.argmax(scores))
                    if float(scores[top_pos]) > best_score:
                        best_score = float(scores[top_pos])
                        best_choice = (nid, int(unst_idxs[top_pos]))

                a_lb, a_ub = rel['a_lb'], rel['a_ub']
                A_pos = np.maximum(A_cur, 0.0)
                A_neg = np.minimum(A_cur, 0.0)
                A_prev = A_pos * a_lb[None, :] + A_neg * a_ub[None, :]
                A_map[p_id] = A_map.get(p_id, 0.0) + A_prev

            elif op == 'identity':
                A_map[p_id] = A_map.get(p_id, 0.0) + A_cur
            elif op == 'permute':
                A_map[p_id] = A_map.get(p_id, 0.0) + A_cur[:, node.inv_perm]
            elif op == 'add':
                p0, p1 = node.inputs[0], node.inputs[1]
                A_map[p0] = A_map.get(p0, 0.0) + A_cur
                A_map[p1] = A_map.get(p1, 0.0) + A_cur
            elif op == 'scale_shift':
                A_map[p_id] = A_map.get(p_id, 0.0) + A_cur * node.scale[None, :]
            elif op == 'affine':
                A_map[p_id] = A_map.get(p_id, 0.0) + np.asarray(A_cur @ node.W, dtype=np.float64)
            elif op in ('leaky_relu', 'satlin', 'satlins', 'sigmoid', 'tanh', 'gelu'):
                rel = node_relax[nid]
                A_pos = np.maximum(A_cur, 0.0)
                A_neg = np.minimum(A_cur, 0.0)
                A_prev = A_pos * rel['a_lb'][None, :] + A_neg * rel['a_ub'][None, :]
                A_map[p_id] = A_map.get(p_id, 0.0) + A_prev
            elif op == 'maxpool2d':
                rel = node_relax[nid]
                A_pos = np.maximum(A_cur, 0.0)
                A_neg = np.minimum(A_cur, 0.0)
                A_prev = np.asarray(A_pos @ rel['G_lb'] + A_neg @ rel['G_ub'], dtype=np.float64)
                A_map[p_id] = A_map.get(p_id, 0.0) + A_prev
            elif op in ('layernorm', 'mhsa'):
                rel = node_relax[nid]
                A_map[p_id] = A_map.get(p_id, 0.0) + np.asarray(A_cur @ rel['W'], dtype=np.float64)

        return best_choice

    def verify_unsafe_polytope(self, net_or_graph, in_set, unsafe_mat, unsafe_vec):
        """
        Verify whether the unsafe region U = { y | unsafe_mat @ y <= unsafe_vec }
        is reachable from `in_set` using CROWN, alpha-CROWN, or alpha,beta-CROWN.

        Note on sign convention:
          Unsafe condition is: for ALL row k, unsafe_mat[k] @ y - unsafe_vec[k] <= 0.
          Equivalently, a domain is PROVABLY SAFE (UNSAT) if there exists row k such that
              min_{x in Domain} (d_k - u_k^T y) < 0  <=>  max_{x in Domain} (u_k^T y - d_k) < 0,
          or in minimization form:
              max_k  min_{x in Domain} (u_k^T y - d_k) > 0!
          Let z(x) = unsafe_mat @ y(x) - unsafe_vec.
          - If max_k z_lb[k] > 0: row k cannot be <= 0 anywhere in Domain -> UNSAT (Safe)!
          - If all_k z_ub[k] <= 0: every point in Domain satisfies U y <= u -> SAT (All Unsafe)!
          - If we find a concrete sample x* in Domain with U y(x*) <= u -> SAT (Counterexample)!

        Returns dict with:
          'status': 'UNSAT' | 'SAT' | 'UNKNOWN'
          'z_lb', 'z_ub', 'A_lb', 'b_lb', 'A_ub', 'b_ub',
          'counterexample_x', 'counterexample_y', 'branches_explored'
        """
        inp_spec = in_set if isinstance(in_set, _InputSpec) else _InputSpec(in_set)
        graph = (
            net_or_graph
            if isinstance(net_or_graph, CROWNNetworkGraph)
            else CROWNNetworkGraph(net_or_graph, inp_spec.in_shape)
        )
        U = np.asarray(unsafe_mat, dtype=np.float64)
        if U.ndim == 1:
            U = U.reshape(1, -1)
        u_vec = np.asarray(unsafe_vec, dtype=np.float64).ravel()

        # Step 1: Quick PGD / sampling counterexample check
        ce_x, ce_y = self._sample_and_pgd_counterexample(
            graph, inp_spec, U, u_vec, inp_spec.pred_lb, inp_spec.pred_ub
        )
        if ce_x is not None:
            A_lb, b_lb, z_lb, A_ub, b_ub, z_ub, _, _ = self.compute_bounds(
                graph, inp_spec, C_spec=U, d_spec=u_vec
            )
            return {
                'status': 'SAT',
                'z_lb': z_lb, 'z_ub': z_ub,
                'A_lb': A_lb, 'b_lb': b_lb, 'A_ub': A_ub, 'b_ub': b_ub,
                'counterexample_x': ce_x, 'counterexample_y': ce_y,
                'branches_explored': 1,
            }

        # Step 2: Initial CROWN / alpha-CROWN pass on full domain
        A_lb, b_lb, z_lb, A_ub, b_ub, z_ub, node_bounds, node_relax = self.compute_bounds(
            graph, inp_spec, C_spec=U, d_spec=u_vec
        )

        # Safe if ANY conjunctive row k has lower bound z_lb[k] > 0
        if np.any(z_lb > 0.0):
            return {
                'status': 'UNSAT',
                'z_lb': z_lb, 'z_ub': z_ub,
                'A_lb': A_lb, 'b_lb': b_lb, 'A_ub': A_ub, 'b_ub': b_ub,
                'counterexample_x': None, 'counterexample_y': None,
                'branches_explored': 1,
            }

        # All unsafe if ALL conjunctive rows have upper bound z_ub[k] <= 0
        if z_ub is not None and np.all(z_ub <= 0.0):
            mid_alpha = 0.5 * (inp_spec.pred_lb + inp_spec.pred_ub)
            ce_x = inp_spec.c0 + inp_spec.V0 @ mid_alpha
            ce_y = graph.evaluate_flat(ce_x)
            return {
                'status': 'SAT',
                'z_lb': z_lb, 'z_ub': z_ub,
                'A_lb': A_lb, 'b_lb': b_lb, 'A_ub': A_ub, 'b_ub': b_ub,
                'counterexample_x': ce_x, 'counterexample_y': ce_y,
                'branches_explored': 1,
            }

        if self.mode != 'alpha-beta-crown' or self.max_bab_branches <= 1:
            return {
                'status': 'UNKNOWN',
                'z_lb': z_lb, 'z_ub': z_ub,
                'A_lb': A_lb, 'b_lb': b_lb, 'A_ub': A_ub, 'b_ub': b_ub,
                'counterexample_x': None, 'counterexample_y': None,
                'branches_explored': 1,
            }

        # Step 3: alpha,beta-CROWN Branch-and-Bound (ReLU splitting + beta multipliers + input-split fallback)
        t_start = time.perf_counter()
        best_row = int(np.argmax(z_lb))
        root_margin = float(z_lb[best_row])

        # Priority queue ordered by smallest margin (hardest unverified branch first)
        # Entry: (margin, tie_breaker, pred_lb_b, pred_ub_b, splits_b, pre_bounds_b, node_relax_b, A_lb_b)
        pq = []
        tie_id = 0
        heapq.heappush(
            pq,
            (
                root_margin,
                tie_id,
                inp_spec.pred_lb.copy(),
                inp_spec.pred_ub.copy(),
                {},
                {k: (v[0].copy(), v[1].copy()) for k, v in node_bounds.items()},
                node_relax,
                A_lb.copy(),
            ),
        )
        branches_explored = 1

        while len(pq) > 0 and branches_explored < self.max_bab_branches:
            if (time.perf_counter() - t_start) >= self.bab_timeout:
                break

            margin_b, _, lb_b, ub_b, splits_b, pre_b, relax_b, A_lb_b = heapq.heappop(pq)
            if margin_b > 0.0:
                continue

            # Pick row closest to being verified > 0
            target_row = best_row
            split_choice = self._select_babsr_relu_split(
                graph, U[target_row], relax_b, splits_b
            )

            children = []
            if split_choice is not None:
                relu_nid, neuron_idx = split_choice
                p_nid = graph.nodes[relu_nid].inputs[0]
                for sign_val in (-1, +1):  # -1: x >= 0 (active), +1: x <= 0 (inactive)
                    c_splits = {k: dict(v) for k, v in splits_b.items()}
                    c_splits.setdefault(relu_nid, {})[neuron_idx] = sign_val
                    c_pre = {k: (v[0].copy(), v[1].copy()) for k, v in pre_b.items()}
                    if p_nid in c_pre:
                        lp_c, up_c = c_pre[p_nid]
                        if sign_val == -1:
                            lp_c[neuron_idx] = max(lp_c[neuron_idx], 0.0)
                        else:
                            up_c[neuron_idx] = min(up_c[neuron_idx], 0.0)
                        c_pre[p_nid] = (lp_c, up_c)
                    children.append((lb_b.copy(), ub_b.copy(), c_splits, c_pre))
            elif inp_spec.nVars > 0 and not inp_spec.has_constraints:
                # Fallback to input coordinate splitting along highest sensitivity * width
                widths = ub_b - lb_b
                if np.max(widths) <= 1e-12:
                    return {
                        'status': 'UNKNOWN',
                        'z_lb': z_lb, 'z_ub': z_ub,
                        'A_lb': A_lb, 'b_lb': b_lb, 'A_ub': A_ub, 'b_ub': b_ub,
                        'counterexample_x': None, 'counterexample_y': None,
                        'branches_explored': branches_explored,
                    }
                sens_in = np.abs(A_lb_b[target_row]) * widths
                split_dim = int(np.argmax(sens_in)) if np.max(sens_in) > 0 else int(np.argmax(widths))
                mid_val = 0.5 * (lb_b[split_dim] + ub_b[split_dim])
                lb1, ub1 = lb_b.copy(), ub_b.copy()
                ub1[split_dim] = mid_val
                lb2, ub2 = lb_b.copy(), ub_b.copy()
                lb2[split_dim] = mid_val
                children.append((lb1, ub1, splits_b, {}))
                children.append((lb2, ub2, splits_b, {}))
            else:
                # No further splits possible
                return {
                    'status': 'UNKNOWN',
                    'z_lb': z_lb, 'z_ub': z_ub,
                    'A_lb': A_lb, 'b_lb': b_lb, 'A_ub': A_ub, 'b_ub': b_ub,
                    'counterexample_x': None, 'counterexample_y': None,
                    'branches_explored': branches_explored,
                }

            for c_lb, c_ub, c_splits, c_pre in children:
                branches_explored += 1
                # Check midpoint of child input box for counterexample
                ce_x, ce_y = self._sample_and_pgd_counterexample(
                    graph, inp_spec, U, u_vec, c_lb, c_ub, num_random=4, pgd_steps=3
                )
                if ce_x is not None:
                    return {
                        'status': 'SAT',
                        'z_lb': z_lb, 'z_ub': z_ub,
                        'A_lb': A_lb, 'b_lb': b_lb, 'A_ub': A_ub, 'b_ub': b_ub,
                        'counterexample_x': ce_x, 'counterexample_y': ce_y,
                        'branches_explored': branches_explored,
                    }

                # Check if split made pre-activation bounds infeasible (l > u)
                infeasible = False
                for lp_c, up_c in c_pre.values():
                    if np.any(lp_c > up_c + 1e-12):
                        infeasible = True
                        break
                if infeasible:
                    continue

                A_lb_c, b_lb_c, z_lb_c, _, _, z_ub_c, bounds_c, relax_c = self.compute_bounds(
                    graph, inp_spec, C_spec=U, d_spec=u_vec,
                    pred_lb=c_lb, pred_ub=c_ub,
                    split_constraints=c_splits, initial_pre_bounds=c_pre,
                    optimize_upper=False
                )
                c_margin = float(np.max(z_lb_c))
                if c_margin > 0.0:
                    # Pruned (verified safe on this branch!)
                    continue

                tie_id += 1
                heapq.heappush(
                    pq,
                    (c_margin, tie_id, c_lb, c_ub, c_splits, bounds_c, relax_c, A_lb_c),
                )

        if len(pq) == 0:
            # All branches verified safe!
            z_lb_verified = z_lb.copy()
            z_lb_verified[best_row] = max(float(z_lb_verified[best_row]), 1e-6)
            return {
                'status': 'UNSAT',
                'z_lb': z_lb_verified, 'z_ub': z_ub,
                'A_lb': A_lb, 'b_lb': b_lb, 'A_ub': A_ub, 'b_ub': b_ub,
                'counterexample_x': None, 'counterexample_y': None,
                'branches_explored': branches_explored,
            }

        # Update global lower bound from remaining worst branch in queue
        worst_remaining_margin = float(pq[0][0])
        z_lb_out = z_lb.copy()
        z_lb_out[best_row] = max(float(z_lb_out[best_row]), worst_remaining_margin)
        return {
            'status': 'UNKNOWN',
            'z_lb': z_lb_out, 'z_ub': z_ub,
            'A_lb': A_lb, 'b_lb': b_lb, 'A_ub': A_ub, 'b_ub': b_ub,
            'counterexample_x': None, 'counterexample_y': None,
            'branches_explored': branches_explored,
        }


# =====================================================================
# 6. Reachability Set Construction: Star, ImageStar, ProbStar
# =====================================================================

def _build_crown_output_set(
    inp_spec,
    out_shape,
    A_lb_active,
    b_lb,
    z_lb,
    A_ub_active,
    b_ub,
    z_ub,
    tol=1e-10,
):
    """
    Construct a Star, ImageStar, or ProbStar encoding CROWN's linear and interval bounds:
        A_lb_full @ alpha_0 + b_lb <= y <= A_ub_full @ alpha_0 + b_ub,
        z_lb <= y <= z_ub.

    Properties of the returned set S_out:
      - S_out.estimateRanges() returns (z_lb, z_ub) in O(1) time.
      - S_out.C @ alpha <= S_out.d encodes the relational CROWN linear constraints
        in terms of the original input predicates alpha_0 and output variables.
      - When input is a ProbStar (or ProbSparseImageStar), S_out is a ProbStar with
        a RelaxationEnvelopeRecord in `env_dag` and moment-matched Gaussian parameters
        (mu, Sig), so S_out.verifySpec(), S_out.certifyLabel(), S_out.getSoundInputBounds(),
        and S_out.getOutputMoments() work out of the box!
    """
    A_lb_full, b_lb_full = inp_spec.expand_to_full_predicates(A_lb_active, b_lb)
    A_ub_full, b_ub_full = inp_spec.expand_to_full_predicates(A_ub_active, b_ub)

    M = len(z_lb)
    n0 = inp_spec.nVars_full
    lb0 = inp_spec.pred_lb_full
    ub0 = inp_spec.pred_ub_full

    gap = np.maximum(z_ub - z_lb, 0.0)
    diff_A = np.max(np.abs(A_ub_full - A_lb_full), axis=1) if n0 > 0 else np.zeros(M)
    diff_b = np.abs(b_ub_full - b_lb_full)
    is_exact_affine = (diff_A <= tol) & (diff_b <= tol)

    det_idx = np.where(is_exact_affine | (gap <= tol))[0]
    amb_idx = np.where(~(is_exact_affine | (gap <= tol)))[0]
    m_amb = len(amb_idx)

    V1 = np.zeros((M, n0 + 1), dtype=np.float64)
    if len(det_idx) > 0:
        V1[det_idx, 0] = 0.5 * (b_lb_full[det_idx] + b_ub_full[det_idx])
        if n0 > 0:
            V1[det_idx, 1 : n0 + 1] = 0.5 * (A_lb_full[det_idx, :] + A_ub_full[det_idx, :])

    if m_amb > 0:
        V2 = np.zeros((M, m_amb), dtype=np.float64)
        V2[amb_idx, np.arange(m_amb)] = 1.0
        new_V = np.hstack([V1, V2])

        A_l_amb = A_lb_full[amb_idx, :]
        b_l_amb = b_lb_full[amb_idx]
        A_u_amb = A_ub_full[amb_idx, :]
        b_u_amb = b_ub_full[amb_idx]
        E = np.eye(m_amb, dtype=np.float64)

        # Constraints:
        # 1) A_l_amb @ alpha_0 - y_amb <= -b_l_amb  (y_amb >= A_l_amb @ alpha_0 + b_l_amb)
        # 2) -A_u_amb @ alpha_0 + y_amb <= b_u_amb  (y_amb <= A_u_amb @ alpha_0 + b_u_amb)
        C1 = np.hstack([A_l_amb, -E])
        d1 = -b_l_amb
        C2 = np.hstack([-A_u_amb, E])
        d2 = b_u_amb

        if inp_spec.has_constraints:
            C0_dense = inp_spec.C0_full.toarray() if sp.issparse(inp_spec.C0_full) else inp_spec.C0_full
            C0_pad = np.hstack([C0_dense, np.zeros((C0_dense.shape[0], m_amb), dtype=np.float64)])
            new_C = np.vstack([C0_pad, C1, C2])
            new_d = np.hstack([inp_spec.d0_full, d1, d2])
        else:
            new_C = np.vstack([C1, C2])
            new_d = np.hstack([d1, d2])

        new_lb = np.hstack([lb0, z_lb[amb_idx]])
        new_ub = np.hstack([ub0, z_ub[amb_idx]])
    else:
        new_V = V1
        if inp_spec.has_constraints:
            new_C = inp_spec.C0_full.toarray() if sp.issparse(inp_spec.C0_full) else inp_spec.C0_full.copy()
            new_d = inp_spec.d0_full.copy()
        else:
            new_C = np.empty((0, n0), dtype=np.float64)
            new_d = np.empty((0,), dtype=np.float64)
        new_lb = lb0.copy()
        new_ub = ub0.copy()

    if inp_spec.is_prob:
        mu0 = inp_spec.mu0_full
        Sig0 = inp_spec.Sig0_full
        cov_mode = inp_spec.cov_mode or ('diagonal' if Sig0.ndim == 1 else 'full')

        if m_amb > 0:
            A_mid_amb = 0.5 * (A_l_amb + A_u_amb)
            b_mid_amb = 0.5 * (b_l_amb + b_u_amb)
            mu_amb, var_amb, V_Sig_amb = compute_preactivation_moments(
                A_mid_amb, b_mid_amb, mu0, Sig0, cov_mode=cov_mode
            )
            mu_amb = np.clip(mu_amb, z_lb[amb_idx], z_ub[amb_idx])
            new_mu = np.concatenate([mu0, mu_amb])
            if cov_mode == 'diagonal' or Sig0.ndim == 1:
                var0 = Sig0 if Sig0.ndim == 1 else np.diag(Sig0)
                new_Sig = np.concatenate([var0, var_amb])
            else:
                cross_cov = V_Sig_amb.T
                diag_amb = np.maximum(var_amb, np.sum(A_mid_amb * V_Sig_amb, axis=1) + 1e-10)
                new_Sig = np.vstack([
                    np.hstack([Sig0, cross_cov]),
                    np.hstack([cross_cov.T, np.diag(diag_amb)]),
                ])
            env_rec = RelaxationEnvelopeRecord(
                n_prev=n0,
                n_new=m_amb,
                G_lb=A_l_amb,
                gamma_lb=b_l_amb,
                G_ub=A_u_amb,
                gamma_ub=b_u_amb,
            )
            env_dag = [env_rec]
        else:
            new_mu = mu0.copy()
            new_Sig = Sig0.copy()
            env_dag = []

        pstar_out = ProbStar(
            new_V, new_C, new_d, new_mu, new_Sig, new_lb, new_ub,
            copy_=False, cov_mode=cov_mode, env_dag=env_dag,
            mu0=mu0, Sig0=Sig0, pred_lb0=lb0, pred_ub0=ub0,
            nVars0=n0, check_psd=False,
        )
        pstar_out.shape = out_shape
        pstar_out.num_pred = pstar_out.nVars
        return pstar_out

    if inp_spec.is_image and len(out_shape) == 3:
        H_out, W_out, C_out = out_shape
        V_img = new_V.reshape(H_out, W_out, C_out, new_V.shape[1])
        return ImageStar(V_img, new_C, new_d, new_lb, new_ub)

    star_out = Star(new_V, new_C, new_d, new_lb, new_ub)
    star_out.shape = out_shape
    star_out.num_pred = star_out.nVars
    return star_out


# =====================================================================
# 7. Public StarV Reachability, Verification, and Certification APIs
# =====================================================================

def reachCROWN(
    net,
    inputSet,
    mode='crown',
    slope_mode='adaptive',
    alpha_iters=30,
    alpha_lr=0.15,
    lp_solver='gurobi',
    show=False,
):
    """
    Compute the CROWN or alpha-CROWN reachable set of `net` on `inputSet`.
    Supports single input set or list of input sets (Star, ProbStar, SparseStar,
    ImageStar, SparseImageStar2DCSR/COO, ProbSparseImageStar2DCSR/COO).

    Returns:
        Star, ImageStar, or ProbStar (or list thereof if `inputSet` is a list).
    """
    if isinstance(inputSet, list):
        return [
            reachCROWN(
                net, s, mode=mode, slope_mode=slope_mode,
                alpha_iters=alpha_iters, alpha_lr=alpha_lr,
                lp_solver=lp_solver, show=show,
            )
            for s in inputSet
        ]

    verifier = CROWNVerifier(
        mode=mode,
        slope_mode=slope_mode,
        alpha_iters=alpha_iters,
        alpha_lr=alpha_lr,
        lp_solver=lp_solver,
        show=show,
    )
    inp_spec = _InputSpec(inputSet)
    graph = CROWNNetworkGraph(net, inp_spec.in_shape)

    A_lb, b_lb, z_lb, A_ub, b_ub, z_ub, _, _ = verifier.compute_bounds(
        graph, inp_spec, C_spec=None, d_spec=None, optimize_upper=True
    )
    return _build_crown_output_set(
        inp_spec, graph.out_shape, A_lb, b_lb, z_lb, A_ub, b_ub, z_ub
    )


def qualiVerifyCROWN(
    net,
    inputSet,
    unsafe_mat,
    unsafe_vec,
    mode='alpha-beta-crown',
    slope_mode='adaptive',
    alpha_iters=30,
    alpha_lr=0.15,
    beta_iters=20,
    beta_lr=0.1,
    max_bab_branches=200,
    bab_timeout=60.0,
    lp_solver='gurobi',
    show=False,
    return_info=False,
):
    """
    Qualitative verification of specification `unsafe_mat @ y <= unsafe_vec`
    using CROWN, alpha-CROWN, or alpha,beta-CROWN.

    Returns:
        S_out: Output Star / ImageStar set
        P: Unsafe output set (or [] if UNSAT)
        Res: 'UNSAT' (safe), 'SAT' (unsafe with counterexample), or 'UNKNOWN'
        (and `info` dict if `return_info=True`)
    """
    single_set = inputSet[0] if isinstance(inputSet, list) else inputSet
    t0 = time.perf_counter()

    verifier = CROWNVerifier(
        mode=mode,
        slope_mode=slope_mode,
        alpha_iters=alpha_iters,
        alpha_lr=alpha_lr,
        beta_iters=beta_iters,
        beta_lr=beta_lr,
        max_bab_branches=max_bab_branches,
        bab_timeout=bab_timeout,
        lp_solver=lp_solver,
        show=show,
    )
    inp_spec = _InputSpec(single_set)
    graph = CROWNNetworkGraph(net, inp_spec.in_shape)

    ver_res = verifier.verify_unsafe_polytope(graph, inp_spec, unsafe_mat, unsafe_vec)

    # Also construct output Star set S_out
    A_out_lb, b_out_lb, y_lb, A_out_ub, b_out_ub, y_ub, _, _ = verifier.compute_bounds(
        graph, inp_spec, C_spec=None, d_spec=None, optimize_upper=True
    )
    S_out = _build_crown_output_set(
        inp_spec, graph.out_shape, A_out_lb, b_out_lb, y_lb, A_out_ub, b_out_ub, y_ub
    )

    status = ver_res['status']
    if status == 'UNSAT':
        P = []
    else:
        P = S_out

    ver_res['verification_time'] = time.perf_counter() - t0
    if return_info:
        return S_out, P, status, ver_res
    return S_out, P, status


def _compute_gaussian_prob_from_crown_bounds(
    inp_spec,
    pred_lb_sub,
    pred_ub_sub,
    A_lb,
    b_lb,
    z_lb,
    A_ub,
    b_ub,
    z_ub,
):
    """
    Compute provably sound quantitative probability bounds [p_min, p_max] and
    moment-matched estimate p_hat for specification z(alpha_0) <= 0 on a sub-box
    [pred_lb_sub, pred_ub_sub] using CROWN / alpha-CROWN linear bounds:
        A_lb @ alpha_0 + b_lb <= z(alpha_0) <= A_ub @ alpha_0 + b_ub.
    """
    K_spec = A_lb.shape[0]
    mu0_full = inp_spec.mu0_full
    Sig0_full = inp_spec.Sig0_full

    # Construct full sub-box bounds
    lb0_full = inp_spec.pred_lb_full.copy()
    ub0_full = inp_spec.pred_ub_full.copy()
    if inp_spec.nVars > 0:
        lb0_full[inp_spec.active_idx] = pred_lb_sub
        ub0_full[inp_spec.active_idx] = pred_ub_sub

    p_input = compute_box_input_probability(mu0_full, Sig0_full, lb0_full, ub0_full)
    if p_input <= 0.0:
        return {
            'status': 'UNSAT',
            'p_min': 0.0, 'p_max': 0.0, 'p_hat': 0.0, 'p_input': 0.0,
            'cond_p_min': 0.0, 'cond_p_max': 0.0, 'cond_p_hat': 0.0,
        }

    if np.any(z_lb > 0.0):
        return {
            'status': 'UNSAT',
            'p_min': 0.0, 'p_max': 0.0, 'p_hat': 0.0, 'p_input': p_input,
            'cond_p_min': 0.0, 'cond_p_max': 0.0, 'cond_p_hat': 0.0,
        }

    if np.all(z_ub <= 0.0):
        return {
            'status': 'SAT_ALL',
            'p_min': p_input, 'p_max': p_input, 'p_hat': p_input, 'p_input': p_input,
            'cond_p_min': 1.0, 'cond_p_max': 1.0, 'cond_p_hat': 1.0,
        }

    W_lb0, d_lb0 = inp_spec.expand_to_full_predicates(A_lb, b_lb)
    W_ub0, d_ub0 = inp_spec.expand_to_full_predicates(A_ub, b_ub)

    var0_vec = Sig0_full if Sig0_full.ndim == 1 else np.diag(Sig0_full)
    mu0_trunc, var0_trunc = compute_truncated_gaussian_moments_1d(mu0_full, var0_vec, lb0_full, ub0_full)
    if Sig0_full.ndim == 1:
        Sig0_trunc = var0_trunc
        cov_mode0 = 'diagonal'
    else:
        scale0 = np.sqrt(var0_trunc / np.maximum(var0_vec, 1e-16))
        Sig0_trunc = (Sig0_full * scale0[:, None]) * scale0[None, :]
        cov_mode0 = 'full'

    # Under-approximator z_lb0(alpha_0) = W_lb0 @ alpha_0 + d_lb0 <= z(alpha_0)
    # -> P(z <= 0 | B) <= P(z_lb0 <= 0 | B) = cond_p_max
    mu_z_lb, var_z_lb, V_Sig_lb = compute_preactivation_moments(
        W_lb0, d_lb0, mu0_trunc, Sig0_trunc, cov_mode=cov_mode0
    )
    Sig_z_lb = var_z_lb if K_spec == 1 else (
        V_Sig_lb @ W_lb0.T if V_Sig_lb is not None else (W_lb0 * Sig0_trunc[None, :]) @ W_lb0.T
    )

    # Over-approximator z_ub0(alpha_0) = W_ub0 @ alpha_0 + d_ub0 >= z(alpha_0)
    # -> P(z <= 0 | B) >= P(z_ub0 <= 0 | B) = cond_p_min
    mu_z_ub, var_z_ub, V_Sig_ub = compute_preactivation_moments(
        W_ub0, d_ub0, mu0_trunc, Sig0_trunc, cov_mode=cov_mode0
    )
    Sig_z_ub = var_z_ub if K_spec == 1 else (
        V_Sig_ub @ W_ub0.T if V_Sig_ub is not None else (W_ub0 * Sig0_trunc[None, :]) @ W_ub0.T
    )

    neg_inf = np.full(K_spec, -np.inf, dtype=np.float64)
    zeros_k = np.zeros(K_spec, dtype=np.float64)

    W_lb0_pos, W_lb0_neg = np.maximum(W_lb0, 0.0), np.minimum(W_lb0, 0.0)
    min_z_under = np.maximum(z_lb, d_lb0 + W_lb0_pos @ lb0_full + W_lb0_neg @ ub0_full)
    max_z_under = d_lb0 + W_lb0_pos @ ub0_full + W_lb0_neg @ lb0_full

    W_ub0_pos, W_ub0_neg = np.maximum(W_ub0, 0.0), np.minimum(W_ub0, 0.0)
    min_z_over = d_ub0 + W_ub0_pos @ lb0_full + W_ub0_neg @ ub0_full
    max_z_over = np.minimum(z_ub, d_ub0 + W_ub0_pos @ ub0_full + W_ub0_neg @ lb0_full)

    if np.all(max_z_under <= 0.0):
        cond_p_max = 1.0
    elif np.any(min_z_under > 0.0):
        cond_p_max = 0.0
    else:
        tot_under = mvn_cdf_lowdim(min_z_under, max_z_under, mu_z_lb, Sig_z_lb)
        uns_under = mvn_cdf_lowdim(min_z_under, np.minimum(zeros_k, max_z_under), mu_z_lb, Sig_z_lb)
        cond_under = uns_under / tot_under if tot_under > 1e-12 else mvn_cdf_lowdim(neg_inf, zeros_k, mu_z_lb, Sig_z_lb)
        mu_z_lb_raw, var_z_lb_raw, V_Sig_lb_raw = compute_preactivation_moments(
            W_lb0, d_lb0, mu0_full, Sig0_full, cov_mode=cov_mode0
        )
        Sig_z_lb_raw = var_z_lb_raw if K_spec == 1 else (
            V_Sig_lb_raw @ W_lb0.T if V_Sig_lb_raw is not None else (W_lb0 * Sig0_full[None, :]) @ W_lb0.T
        )
        bf_cond_max = mvn_cdf_lowdim(neg_inf, zeros_k, mu_z_lb_raw, Sig_z_lb_raw) / max(p_input, 1e-15)
        cond_p_max = float(np.clip(max(cond_under, min(1.0, bf_cond_max)), 0.0, 1.0))

    if np.any(min_z_over > 0.0):
        cond_p_min = 0.0
    elif np.all(max_z_over <= 0.0):
        cond_p_min = 1.0
    else:
        tot_over = mvn_cdf_lowdim(min_z_over, max_z_over, mu_z_ub, Sig_z_ub)
        uns_over = mvn_cdf_lowdim(min_z_over, np.minimum(zeros_k, max_z_over), mu_z_ub, Sig_z_ub)
        cond_over = uns_over / tot_over if tot_over > 1e-12 else mvn_cdf_lowdim(neg_inf, zeros_k, mu_z_ub, Sig_z_ub)
        mu_z_ub_raw, var_z_ub_raw, V_Sig_ub_raw = compute_preactivation_moments(
            W_ub0, d_ub0, mu0_full, Sig0_full, cov_mode=cov_mode0
        )
        Sig_z_ub_raw = var_z_ub_raw if K_spec == 1 else (
            V_Sig_ub_raw @ W_ub0.T if V_Sig_ub_raw is not None else (W_ub0 * Sig0_full[None, :]) @ W_ub0.T
        )
        bf_cond_min = max(0.0, mvn_cdf_lowdim(neg_inf, zeros_k, mu_z_ub_raw, Sig_z_ub_raw) - (1.0 - p_input)) / max(p_input, 1e-15)
        cond_p_min = float(np.clip(min(cond_over, bf_cond_min) if p_input > 0.5 else cond_over, 0.0, cond_p_max))

    # Midpoint linear form for moment-matched estimate p_hat
    W_mid0 = 0.5 * (W_lb0 + W_ub0)
    d_mid0 = 0.5 * (d_lb0 + d_ub0)
    mu_z_mid, var_z_mid, V_Sig_mid = compute_preactivation_moments(
        W_mid0, d_mid0, mu0_trunc, Sig0_trunc, cov_mode=cov_mode0
    )
    Sig_z_mid = var_z_mid if K_spec == 1 else (
        V_Sig_mid @ W_mid0.T if V_Sig_mid is not None else (W_mid0 * Sig0_trunc[None, :]) @ W_mid0.T
    )
    tot_mid = mvn_cdf_lowdim(z_lb, z_ub, mu_z_mid, Sig_z_mid)
    uns_mid = mvn_cdf_lowdim(z_lb, np.minimum(zeros_k, z_ub), mu_z_mid, Sig_z_mid)
    cond_mid = uns_mid / tot_mid if tot_mid > 1e-12 else mvn_cdf_lowdim(neg_inf, zeros_k, mu_z_mid, Sig_z_mid)
    cond_p_hat = float(np.clip(cond_mid, cond_p_min, cond_p_max))

    p_min = float(np.clip(p_input * cond_p_min, 0.0, p_input))
    p_max = float(np.clip(p_input * cond_p_max, p_min, p_input))
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


def quantiVerifyCROWN(
    net,
    inputSet,
    unsafe_mat,
    unsafe_vec,
    mode='alpha-crown',
    slope_mode='adaptive',
    alpha_iters=30,
    alpha_lr=0.15,
    max_branches=1,
    lp_solver='gurobi',
    show=False,
):
    """
    Quantitative / probabilistic verification of `unsafe_mat @ y <= unsafe_vec`
    on a ProbStar, ProbSparseImageStar2DCSR, or ProbSparseImageStar2DCOO input set
    using CROWN, alpha-CROWN, or alpha,beta-CROWN (with probability-guided input BaB).

    Returns:
        S_out: Output ProbStar encoding the CROWN linear bounds
        res: dict with keys:
             'status', 'p_min', 'p_max', 'p_hat', 'p_input',
             'cond_p_min', 'cond_p_max', 'cond_p_hat',
             'total_time', 'verification_time', 'branches_explored'
    """
    if isinstance(inputSet, list):
        assert len(inputSet) == 1, "quantiVerifyCROWN expects a single probabilistic input set"
        inputSet = inputSet[0]

    t0 = time.perf_counter()
    norm_mode = _normalize_crown_mode(mode)
    if norm_mode == 'alpha-beta-crown' and max_branches <= 1:
        max_branches = 16

    verifier = CROWNVerifier(
        mode='alpha-crown' if norm_mode in ('alpha-crown', 'alpha-beta-crown') else 'crown',
        slope_mode=slope_mode,
        alpha_iters=alpha_iters,
        alpha_lr=alpha_lr,
        lp_solver=lp_solver,
        show=show,
    )
    inp_spec = _InputSpec(inputSet)
    assert inp_spec.is_prob, "quantiVerifyCROWN requires a ProbStar or ProbSparseImageStar input set"
    graph = CROWNNetworkGraph(net, inp_spec.in_shape)

    U = np.asarray(unsafe_mat, dtype=np.float64)
    if U.ndim == 1:
        U = U.reshape(1, -1)
    u_vec = np.asarray(unsafe_vec, dtype=np.float64).ravel()

    # Compute specification CROWN / alpha-CROWN bounds directly on z = U @ y - u_vec
    A_lb, b_lb, z_lb, A_ub, b_ub, z_ub, _, _ = verifier.compute_bounds(
        graph, inp_spec, C_spec=U, d_spec=u_vec, optimize_upper=True
    )

    root_prob = _compute_gaussian_prob_from_crown_bounds(
        inp_spec, inp_spec.pred_lb, inp_spec.pred_ub,
        A_lb, b_lb, z_lb, A_ub, b_ub, z_ub
    )
    p_input_total = root_prob['p_input']

    branches_explored = 1
    if (
        max_branches > 1
        and root_prob['status'] == 'AMBIGUOUS'
        and (root_prob['p_max'] - root_prob['p_min']) > 1e-6
        and inp_spec.nVars > 0
        and not inp_spec.has_constraints
    ):
        # Probability-gap guided Branch-and-Bound over disjoint input sub-boxes
        pq = []
        tie = 0
        gap0 = root_prob['p_max'] - root_prob['p_min']
        heapq.heappush(
            pq,
            (-gap0, tie, inp_spec.pred_lb.copy(), inp_spec.pred_ub.copy(), root_prob, A_lb, A_ub),
        )
        p_min_acc = 0.0
        p_max_acc = 0.0
        p_hat_acc = 0.0

        while len(pq) > 0 and branches_explored < max_branches:
            neg_gap, _, lb_b, ub_b, prob_b, A_lb_b, A_ub_b = heapq.heappop(pq)
            if (-neg_gap) <= 1e-6 or prob_b['status'] != 'AMBIGUOUS':
                p_min_acc += prob_b['p_min']
                p_max_acc += prob_b['p_max']
                p_hat_acc += prob_b['p_hat']
                continue

            # Split along input coordinate with largest sensitivity * width
            widths = ub_b - lb_b
            sens = (np.sum(np.abs(A_lb_b), axis=0) + np.sum(np.abs(A_ub_b), axis=0)) * widths
            split_dim = int(np.argmax(sens)) if np.max(sens) > 0 else int(np.argmax(widths))
            mid_v = 0.5 * (lb_b[split_dim] + ub_b[split_dim])

            lb1, ub1 = lb_b.copy(), ub_b.copy()
            ub1[split_dim] = mid_v
            lb2, ub2 = lb_b.copy(), ub_b.copy()
            lb2[split_dim] = mid_v

            for c_lb, c_ub in ((lb1, ub1), (lb2, ub2)):
                branches_explored += 1
                A_l_c, b_l_c, z_l_c, A_u_c, b_u_c, z_u_c, _, _ = verifier.compute_bounds(
                    graph, inp_spec, C_spec=U, d_spec=u_vec,
                    pred_lb=c_lb, pred_ub=c_ub, optimize_upper=True
                )
                prob_c = _compute_gaussian_prob_from_crown_bounds(
                    inp_spec, c_lb, c_ub, A_l_c, b_l_c, z_l_c, A_u_c, b_u_c, z_u_c
                )
                if prob_c['status'] != 'AMBIGUOUS' or (prob_c['p_max'] - prob_c['p_min']) <= 1e-6:
                    p_min_acc += prob_c['p_min']
                    p_max_acc += prob_c['p_max']
                    p_hat_acc += prob_c['p_hat']
                else:
                    tie += 1
                    c_gap = prob_c['p_max'] - prob_c['p_min']
                    heapq.heappush(
                        pq, (-c_gap, tie, c_lb, c_ub, prob_c, A_l_c, A_u_c)
                    )

        for _, _, _, _, prob_rem, _, _ in pq:
            p_min_acc += prob_rem['p_min']
            p_max_acc += prob_rem['p_max']
            p_hat_acc += prob_rem['p_hat']

        p_min_final = float(np.clip(max(root_prob['p_min'], p_min_acc), 0.0, p_input_total))
        p_max_final = float(np.clip(min(root_prob['p_max'], p_max_acc), p_min_final, p_input_total))
        p_hat_final = float(np.clip(p_hat_acc, p_min_final, p_max_final))
        denom_p = max(p_input_total, 1e-15)
        cond_min = float(np.clip(p_min_final / denom_p, 0.0, 1.0))
        cond_max = float(np.clip(p_max_final / denom_p, cond_min, 1.0))
        cond_hat = float(np.clip(p_hat_final / denom_p, cond_min, cond_max))
        status_final = (
            'UNSAT' if p_max_final <= 0.0
            else ('SAT_ALL' if p_min_final >= p_input_total - 1e-15 else 'AMBIGUOUS')
        )
        root_prob = {
            'status': status_final,
            'p_min': p_min_final,
            'p_max': p_max_final,
            'p_hat': p_hat_final,
            'p_input': p_input_total,
            'cond_p_min': cond_min,
            'cond_p_max': cond_max,
            'cond_p_hat': cond_hat,
        }

    # Build output ProbStar
    A_out_lb, b_out_lb, y_lb, A_out_ub, b_out_ub, y_ub, _, _ = verifier.compute_bounds(
        graph, inp_spec, C_spec=None, d_spec=None, optimize_upper=True
    )
    S_out = _build_crown_output_set(
        inp_spec, graph.out_shape, A_out_lb, b_out_lb, y_lb, A_out_ub, b_out_ub, y_ub
    )

    dt = time.perf_counter() - t0
    root_prob['total_time'] = dt
    root_prob['verification_time'] = dt
    root_prob['branches_explored'] = branches_explored
    return S_out, root_prob


def certifyRobustness_single_input_CROWN(
    net,
    in_set,
    label=None,
    mode='alpha-beta-crown',
    slope_mode='adaptive',
    alpha_iters=30,
    alpha_lr=0.15,
    beta_iters=20,
    beta_lr=0.1,
    max_bab_branches=200,
    bab_timeout=60.0,
    lp_solver='gurobi',
    show=False,
):
    """
    Certify classification robustness for a single input set using
    CROWN, alpha-CROWN, or alpha,beta-CROWN.

    Propagates difference specifications z_c = y_{true_label} - y_c directly from
    the output layer so shared correlations between logits cancel before relaxation!

    Returns:
        rb: 1 (ROBUST), 0 (UNROBUST / counterexample found), or 2 (UNKNOWN)
        vt: verification time in seconds
        Y: output Star / ImageStar / ProbStar set
    """
    t0 = time.perf_counter()
    inp_spec = _InputSpec(in_set)
    graph = CROWNNetworkGraph(net, inp_spec.in_shape)

    if label is None:
        mid_alpha = 0.5 * (inp_spec.pred_lb + inp_spec.pred_ub)
        x_center = inp_spec.c0 + inp_spec.V0 @ mid_alpha
        y_center = graph.evaluate_flat(x_center)
        true_label = int(np.argmax(y_center))
    else:
        true_label = int(np.asarray(label).ravel()[0])

    num_classes = graph.out_dim
    other_classes = [c for c in range(num_classes) if c != true_label]

    verifier = CROWNVerifier(
        mode=mode,
        slope_mode=slope_mode,
        alpha_iters=alpha_iters,
        alpha_lr=alpha_lr,
        beta_iters=beta_iters,
        beta_lr=beta_lr,
        max_bab_branches=max_bab_branches,
        bab_timeout=bab_timeout,
        lp_solver=lp_solver,
        show=show,
    )

    # First compute output reachable set via classic/alpha CROWN
    A_out_lb, b_out_lb, y_lb, A_out_ub, b_out_ub, y_ub, node_bounds, node_relax = verifier.compute_bounds(
        graph, inp_spec, C_spec=None, d_spec=None, optimize_upper=True
    )
    Y = _build_crown_output_set(
        inp_spec, graph.out_shape, A_out_lb, b_out_lb, y_lb, A_out_ub, b_out_ub, y_ub
    )

    # Check if center / corners already misclassify
    mid_alpha = 0.5 * (inp_spec.pred_lb + inp_spec.pred_ub)
    for alpha_test in (mid_alpha, inp_spec.pred_lb, inp_spec.pred_ub):
        x_test = inp_spec.c0 + inp_spec.V0 @ alpha_test
        y_test = graph.evaluate_flat(x_test)
        if int(np.argmax(y_test)) != true_label:
            return 0, time.perf_counter() - t0, Y

    # Filter candidate rival classes where y_ub[c] >= y_lb[true_label]
    cand_classes = [c for c in other_classes if y_ub[c] >= y_lb[true_label] - 1e-12]
    if len(cand_classes) == 0:
        return 1, time.perf_counter() - t0, Y

    # Build difference specification matrix C_diff for all candidate classes:
    # row k: y_{true_label} - y_{cand_k} (robust iff min >= 0 for all k)
    C_diff = np.zeros((len(cand_classes), num_classes), dtype=np.float64)
    for k, c in enumerate(cand_classes):
        C_diff[k, true_label] = 1.0
        C_diff[k, c] = -1.0

    # Fast classic CROWN difference pass reusing node_relax
    A_diff_lb, b_diff_lb, _, _ = verifier.backward_propagate(
        graph, inp_spec, target_node_id=graph.output_node_id,
        C_spec=C_diff, node_relax=node_relax, compute_upper=False
    )
    diff_lb, _ = verifier._concretize_bounds(
        A_diff_lb, b_diff_lb, None, None, inp_spec.pred_lb, inp_spec.pred_ub
    )

    remaining_cands = [cand_classes[k] for k in range(len(cand_classes)) if diff_lb[k] <= 0.0]
    if len(remaining_cands) == 0:
        return 1, time.perf_counter() - t0, Y

    if verifier.mode == 'crown':
        # Check PGD counterexample on remaining candidates before returning UNKNOWN (2)
        for c in remaining_cands:
            u_row = np.zeros((1, num_classes), dtype=np.float64)
            u_row[0, true_label] = 1.0
            u_row[0, c] = -1.0
            ce_x, _ = verifier._sample_and_pgd_counterexample(
                graph, inp_spec, u_row, np.zeros(1), inp_spec.pred_lb, inp_spec.pred_ub
            )
            if ce_x is not None:
                return 0, time.perf_counter() - t0, Y
        return 2, time.perf_counter() - t0, Y

    # Run alpha-CROWN or alpha,beta-CROWN on each remaining rival class c
    rb = 1
    for c in remaining_cands:
        # Unsafe condition for rival class c: y_{true_label} - y_c <= 0
        u_row = np.zeros((1, num_classes), dtype=np.float64)
        u_row[0, true_label] = 1.0
        u_row[0, c] = -1.0
        ver_c = verifier.verify_unsafe_polytope(graph, inp_spec, u_row, np.zeros(1))
        if ver_c['status'] == 'SAT':
            return 0, time.perf_counter() - t0, Y
        elif ver_c['status'] != 'UNSAT':
            rb = 2

    return rb, time.perf_counter() - t0, Y


def certifyRobustness_CROWN(
    net,
    inputs,
    labels=None,
    mode='alpha-beta-crown',
    slope_mode='adaptive',
    alpha_iters=30,
    alpha_lr=0.15,
    beta_iters=20,
    beta_lr=0.1,
    max_bab_branches=200,
    bab_timeout=60.0,
    lp_solver='gurobi',
    return_output=False,
    show=False,
):
    """
    Certify classification robustness of `net` on a single input set or list of input sets
    using CROWN, alpha-CROWN, or alpha,beta-CROWN.
    Matches the return signature `(RB, VT, vt_total, Y)` of `StarV.verifier.certifier.certifyRobustness`.
    """
    start = time.perf_counter()
    if not (
        (isinstance(inputs, list) and len(inputs) > 1)
        or (isinstance(labels, np.ndarray) and len(labels) > 1)
    ):
        inp = inputs[0] if isinstance(inputs, list) else inputs
        lbl = labels[0] if isinstance(labels, (list, np.ndarray)) else labels
        rb, vt, Y = certifyRobustness_single_input_CROWN(
            net, inp, label=lbl, mode=mode, slope_mode=slope_mode,
            alpha_iters=alpha_iters, alpha_lr=alpha_lr,
            beta_iters=beta_iters, beta_lr=beta_lr,
            max_bab_branches=max_bab_branches, bab_timeout=bab_timeout,
            lp_solver=lp_solver, show=show,
        )
        vt_total = time.perf_counter() - start
        return rb, vt, vt_total, Y

    N = len(inputs)
    RB = np.zeros(N)
    VT = np.zeros(N)
    Y_list = []

    if labels is None:
        labels = [None] * N

    for i, (inp, lbl) in enumerate(zip(inputs, labels)):
        RB[i], VT[i], O = certifyRobustness_single_input_CROWN(
            net, inp, label=lbl, mode=mode, slope_mode=slope_mode,
            alpha_iters=alpha_iters, alpha_lr=alpha_lr,
            beta_iters=beta_iters, beta_lr=beta_lr,
            max_bab_branches=max_bab_branches, bab_timeout=bab_timeout,
            lp_solver=lp_solver, show=show,
        )
        if return_output:
            Y_list.append(O)

    vt_total = time.perf_counter() - start
    return RB, VT, vt_total, Y_list


def certifyProbabilisticRobustness_single_input_CROWN(
    net,
    in_set,
    label,
    mode='alpha-crown',
    slope_mode='adaptive',
    alpha_iters=30,
    alpha_lr=0.15,
    max_branches=1,
    lp_solver='gurobi',
    use_lp=False,
    show=False,
):
    """
    Certify quantitative probabilistic classification robustness on a single
    ProbStar / ProbSparseImageStar input set using CROWN / alpha-CROWN / alpha,beta-CROWN.

    Returns `(res, Y)` matching `certifyProbabilisticRobustness_single_input`.
    """
    t0 = time.perf_counter()
    inp_spec = _InputSpec(in_set)
    assert inp_spec.is_prob, "certifyProbabilisticRobustness_CROWN requires a probabilistic input set"
    graph = CROWNNetworkGraph(net, inp_spec.in_shape)

    verifier = CROWNVerifier(
        mode='alpha-crown' if _normalize_crown_mode(mode) in ('alpha-crown', 'alpha-beta-crown') else 'crown',
        slope_mode=slope_mode,
        alpha_iters=alpha_iters,
        alpha_lr=alpha_lr,
        lp_solver=lp_solver,
        show=show,
    )

    t_reach0 = time.perf_counter()
    A_out_lb, b_out_lb, y_lb, A_out_ub, b_out_ub, y_ub, node_bounds, node_relax = verifier.compute_bounds(
        graph, inp_spec, C_spec=None, d_spec=None, optimize_upper=True
    )
    Y = _build_crown_output_set(
        inp_spec, graph.out_shape, A_out_lb, b_out_lb, y_lb, A_out_ub, b_out_ub, y_ub
    )
    reach_time = time.perf_counter() - t_reach0

    t_ver0 = time.perf_counter()
    true_label = int(label)
    num_classes = graph.out_dim
    other_classes = [c for c in range(num_classes) if c != true_label]
    p_input = Y.get_input_probability()

    # Propagate pairwise difference specifications z_c = y_c - y_{true_label}
    # A point is SAFE (correctly classified) iff for ALL c != true_label: y_c - y_{true_label} <= 0.
    C_safe = np.zeros((len(other_classes), num_classes), dtype=np.float64)
    for k, c in enumerate(other_classes):
        C_safe[k, c] = 1.0
        C_safe[k, true_label] = -1.0

    if verifier.mode == 'crown':
        A_s_lb, b_s_lb, A_s_ub, b_s_ub = verifier.backward_propagate(
            graph, inp_spec, target_node_id=graph.output_node_id,
            C_spec=C_safe, node_relax=node_relax, compute_upper=True
        )
        z_s_lb, z_s_ub = verifier._concretize_bounds(
            A_s_lb, b_s_lb, A_s_ub, b_s_ub, inp_spec.pred_lb, inp_spec.pred_ub
        )
    else:
        A_s_lb, b_s_lb, z_s_lb, A_s_ub, b_s_ub, z_s_ub, _, _ = verifier.compute_bounds(
            graph, inp_spec, C_spec=C_safe, d_spec=None,
            initial_pre_bounds=node_bounds, optimize_upper=True
        )

    # Check if provably robust (all y_c - y_true <= 0, i.e. z_s_ub <= 0)
    if np.all(z_s_ub <= 0.0):
        spec_res = {
            'status': 'ROBUST',
            'p_min': p_input, 'p_max': p_input, 'p_hat': p_input, 'p_input': p_input,
            'cond_p_min': 1.0, 'cond_p_max': 1.0, 'cond_p_hat': 1.0,
            'rival_classes': [],
        }
    elif np.any(z_s_lb > 0.0):
        # There exists a rival class c where y_c - y_true > 0 everywhere in the input box!
        rivals = [other_classes[k] for k in range(len(other_classes)) if z_s_ub[k] > 0.0]
        spec_res = {
            'status': 'VIOLATED',
            'p_min': 0.0, 'p_max': 0.0, 'p_hat': 0.0, 'p_input': p_input,
            'cond_p_min': 0.0, 'cond_p_max': 0.0, 'cond_p_hat': 0.0,
            'rival_classes': rivals,
        }
    else:
        # Filter active rival classes where z_s_ub[k] > 0
        active_k = [k for k in range(len(other_classes)) if z_s_ub[k] > 0.0]
        rivals = [other_classes[k] for k in active_k]
        prob_res = _compute_gaussian_prob_from_crown_bounds(
            inp_spec,
            inp_spec.pred_lb,
            inp_spec.pred_ub,
            A_s_lb[active_k],
            b_s_lb[active_k],
            z_s_lb[active_k],
            A_s_ub[active_k],
            b_s_ub[active_k],
            z_s_ub[active_k],
        )
        st = prob_res['status']
        if st == 'SAT_ALL':
            status_lbl = 'ROBUST'
        elif st == 'UNSAT':
            status_lbl = 'VIOLATED'
        else:
            status_lbl = 'UNKNOWN'
        spec_res = dict(prob_res)
        spec_res['status'] = status_lbl
        spec_res['rival_classes'] = rivals

    verify_time = time.perf_counter() - t_ver0
    total_time = time.perf_counter() - t0

    status_str = spec_res['status']
    rb = 1 if status_str == 'ROBUST' else (0 if status_str == 'VIOLATED' else 2)
    mem_bytes = Y.V.nbytes + Y.mu.nbytes + Y.Sig.nbytes

    res = dict(spec_res)
    res.update({
        'rb': rb,
        'reach_time': reach_time,
        'verify_time': verify_time,
        'total_time': total_time,
        'mem_bytes': int(mem_bytes),
        'num_pred': int(Y.nVars),
    })
    return res, Y


def certifyProbabilisticRobustness_CROWN(
    net,
    inputs,
    labels,
    mode='alpha-crown',
    slope_mode='adaptive',
    alpha_iters=30,
    alpha_lr=0.15,
    max_branches=1,
    lp_solver='gurobi',
    use_lp=False,
    return_output=False,
    show=False,
):
    """
    Certify quantitative probabilistic classification robustness over a single input set
    or list of input sets using CROWN / alpha-CROWN / alpha,beta-CROWN.
    """
    if not isinstance(inputs, list):
        lbl = labels[0] if isinstance(labels, (list, np.ndarray)) else labels
        res, Y = certifyProbabilisticRobustness_single_input_CROWN(
            net, inputs, lbl, mode=mode, slope_mode=slope_mode,
            alpha_iters=alpha_iters, alpha_lr=alpha_lr,
            max_branches=max_branches, lp_solver=lp_solver, use_lp=use_lp, show=show,
        )
        return res, (Y if return_output else None)

    results = []
    outputs = []
    for inp, lbl in zip(inputs, labels):
        res, Y = certifyProbabilisticRobustness_single_input_CROWN(
            net, inp, int(lbl), mode=mode, slope_mode=slope_mode,
            alpha_iters=alpha_iters, alpha_lr=alpha_lr,
            max_branches=max_branches, lp_solver=lp_solver, use_lp=use_lp, show=show,
        )
        results.append(res)
        if return_output:
            outputs.append(Y)
    return results, outputs
