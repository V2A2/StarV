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
##   This program is licensed under the BSD 3‑Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
GELU Layer Class for Vision Transformers (ViTs) and Modern ConvNets
Supports Star, ProbStar, ImageStar, SparseImageStar2DCSR/COO, and
ProbSparseImageStar2DCSR/COO with provably sound linear relaxation envelopes
and 5-point Gauss-Hermite + Stein moment matching.
"""

import copy
import numpy as np
import scipy.sparse as sp
from scipy.special import erf

from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
from StarV.set.probsparseimagestar2dcoo import ProbSparseImageStar2DCOO
from StarV.set.probsparse_base import propagate_general_elementwise_moments

_INV_SQRT2 = 1.0 / np.sqrt(2.0)
_INV_SQRT2PI = 1.0 / np.sqrt(2.0 * np.pi)
# Global upper bound on |GELU''(x)| = |(2 - x^2) * phi(x)| <= 2 * phi(0) < 0.798
_MAX_GELU_D2 = 0.80


def _gelu(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + erf(x * _INV_SQRT2))


def _gelu_deriv(x):
    x = np.asarray(x, dtype=np.float64)
    cdf = 0.5 * (1.0 + erf(x * _INV_SQRT2))
    pdf = _INV_SQRT2PI * np.exp(-0.5 * np.square(x))
    return cdf + x * pdf


def _gelu_d2(x):
    x = np.asarray(x, dtype=np.float64)
    pdf = _INV_SQRT2PI * np.exp(-0.5 * np.square(x))
    return (2.0 - np.square(x)) * pdf


def compute_gelu_linear_bounds(l, u, num_grid=33):
    """
    Compute provably sound linear lower and upper bounds on [l, u]:
      a_lb * x + b_lb <= GELU(x) <= a_ub * x + b_ub  for all x in [l, u].
    Uses tangent vs. chord slope selection based on midpoint curvature, followed
    by vectorized grid extremum + exact Taylor remainder padding.
    """
    l = np.asarray(l, dtype=np.float64).ravel()
    u = np.asarray(u, dtype=np.float64).ravel()
    width = np.maximum(u - l, 0.0)
    mid = 0.5 * (l + u)

    gl = _gelu(l)
    gu = _gelu(u)
    dmid = _gelu_deriv(mid)
    d2mid = _gelu_d2(mid)

    chord = np.where(width > 1e-9, (gu - gl) / np.maximum(width, 1e-9), dmid)
    convex_mid = d2mid >= 0.0
    a_lb = np.where(convex_mid, dmid, chord)
    a_ub = np.where(convex_mid, chord, dmid)

    # Vectorized grid over [l, u] + Taylor second-derivative bound
    t = np.linspace(0.0, 1.0, num_grid)[None, :]  # (1, K)
    x_grid = l[:, None] + width[:, None] * t      # (m, K)
    g_grid = _gelu(x_grid)                        # (m, K)
    dx = width / float(num_grid - 1)
    taylor_pad = 0.125 * _MAX_GELU_D2 * np.square(dx)

    res_lb = g_grid - a_lb[:, None] * x_grid
    res_ub = g_grid - a_ub[:, None] * x_grid
    b_lb = np.min(res_lb, axis=1) - taylor_pad
    b_ub = np.max(res_ub, axis=1) + taylor_pad

    # Range of GELU(x) on [l, u]
    y_min = np.min(g_grid, axis=1) - taylor_pad
    y_max = np.max(g_grid, axis=1) + taylor_pad
    return a_lb, b_lb, a_ub, b_ub, y_min, y_max


class GELULayer(object):
    """
    Gaussian Error Linear Unit (GELU) layer for qualitative and quantitative reachability.
    """

    def __init__(self, tol=1e-6):
        self.tol = float(tol)

    def __str__(self):
        print(f"Layer type: {self.__class__.__name__}")
        return '\n'

    def info(self):
        print(self)

    @staticmethod
    def evaluate(x):
        return _gelu(x)

    @staticmethod
    def derivative(x):
        return _gelu_deriv(x)

    def reachSingleInput(self, In, lp_solver='estimate', RF=0.0, show=False):
        is_prob = isinstance(In, (ProbStar, ProbSparseImageStar2DCSR, ProbSparseImageStar2DCOO))
        l_all, u_all = In.estimateRanges()
        width = u_all - l_all

        # Neurons with negligible interval width are mapped deterministically
        nonlin_mask = width > self.tol
        det_idx = np.where(~nonlin_mask)[0]
        amb_idx = np.where(nonlin_mask)[0]

        a_lb_all, b_lb_all, a_ub_all, b_ub_all, y_min_all, y_max_all = compute_gelu_linear_bounds(l_all, u_all)

        if isinstance(In, (Star, ProbStar)):
            N = In.dim
            n = In.nVars
            dtype = In.V.dtype
            c_orig = In.V[:, 0].copy()
            V_orig = In.V[:, 1:n + 1].copy()

            m = len(amb_idx)
            V1 = np.zeros((N, n + 1), dtype=dtype)
            if len(det_idx) > 0:
                # Affine tangent map for deterministic / tiny-width neurons
                V1[det_idx, 0] = a_lb_all[det_idx] * c_orig[det_idx] + b_lb_all[det_idx]
                V1[det_idx, 1:n + 1] = a_lb_all[det_idx, None] * V_orig[det_idx, :]

            if m == 0:
                if isinstance(In, ProbStar):
                    return In.with_updated_state(V=V1)
                return Star(V1, In.C, In.d, In.pred_lb, In.pred_ub)

            V2 = np.zeros((N, m), dtype=dtype)
            V2[amb_idx, np.arange(m)] = 1.0
            new_V = np.hstack([V1, V2])

            V_amb = V_orig[amb_idx, :]
            c_amb = c_orig[amb_idx]
            a_l, b_l = a_lb_all[amb_idx], b_lb_all[amb_idx]
            a_u, b_u = a_ub_all[amb_idx], b_ub_all[amb_idx]
            y_lb, y_ub = y_min_all[amb_idx], y_max_all[amb_idx]

            E = np.eye(m, dtype=dtype)
            # y >= a_l * x + b_l  =>  a_l * V_amb * alpha - y <= -a_l * c_amb - b_l
            C1 = np.hstack([a_l[:, None] * V_amb, -E])
            d1 = -a_l * c_amb - b_l
            # y <= a_u * x + b_u  =>  -a_u * V_amb * alpha + y <= a_u * c_amb + b_u
            C2 = np.hstack([-a_u[:, None] * V_amb, E])
            d2 = a_u * c_amb + b_u

            if len(In.C) == 0:
                new_C = np.vstack([C1, C2])
                new_d = np.hstack([d1, d2])
            else:
                C0 = np.hstack([In.C, np.zeros((In.C.shape[0], m), dtype=dtype)])
                new_C = np.vstack([C0, C1, C2])
                new_d = np.hstack([In.d, d1, d2])

            new_lb = np.hstack([In.pred_lb, y_lb])
            new_ub = np.hstack([In.pred_ub, y_ub])

            if isinstance(In, ProbStar):
                mu_new, Sig_new, env_rec = propagate_general_elementwise_moments(
                    V_pre=V_amb, c_pre=c_amb,
                    a_lb=a_l, b_lb=b_l, a_ub=a_u, b_ub=b_u,
                    pred_lb_new=y_lb, pred_ub_new=y_ub,
                    mu=In.mu, Sig=In.Sig,
                    fn_eval=_gelu, fn_deriv=_gelu_deriv,
                    cov_mode=In.cov_mode, dtype=dtype
                )
                return In.with_updated_state(
                    V=new_V, C=new_C, d=new_d, pred_lb=new_lb, pred_ub=new_ub,
                    mu=mu_new, Sig=Sig_new, new_env_record=env_rec
                )
            return Star(new_V, new_C, new_d, new_lb, new_ub)

        elif isinstance(In, ImageStar):
            S = In.toStar(copy_=False)
            S_out = self.reachSingleInput(S, lp_solver=lp_solver, RF=RF, show=show)
            new_V = S_out.V.reshape(In.height, In.width, In.num_channel, S_out.nVars + 1)
            return ImageStar(new_V, S_out.C, S_out.d, S_out.pred_lb, S_out.pred_ub)

        elif isinstance(In, (SparseImageStar2DCSR, SparseImageStar2DCOO)):
            c_orig, V_orig = In.get_c_V() if hasattr(In, 'get_c_V') else (In.c, In.V)
            c_orig = c_orig.copy()
            N = In.dim
            n = In.num_pred
            dtype = V_orig.dtype
            m = len(amb_idx)
            amb_idx32 = amb_idx.astype(np.int32)

            new_c = np.zeros(N, dtype=dtype)
            if len(det_idx) > 0:
                new_c[det_idx] = a_lb_all[det_idx] * c_orig[det_idx] + b_lb_all[det_idx]

            V_csr = V_orig.tocsr()
            if m == N:
                # All neurons introduce new relaxation variables
                row_new = np.arange(N, dtype=np.int32)
                col_new = np.arange(N, dtype=np.int32) + n
                data_new = np.ones(N, dtype=dtype)
                new_V = sp.csr_array((data_new, (row_new, col_new)), shape=(N, n + N))
                if not isinstance(In, SparseImageStar2DCSR):
                    new_V = new_V.tocoo()
            else:
                scale = np.zeros(N, dtype=dtype)
                if len(det_idx) > 0:
                    scale[det_idx] = a_lb_all[det_idx]
                V_rem = V_csr.multiply(scale[:, None]).tocoo()
                V_rem.data = np.hstack([V_rem.data, np.ones(m, dtype=dtype)])
                V_rem.row = np.hstack([V_rem.row.astype(np.int32), amb_idx32])
                V_rem.col = np.hstack([V_rem.col.astype(np.int32), np.arange(m, dtype=np.int32) + n])
                V_rem._shape = (N, n + m)
                new_V = V_rem.tocsr() if isinstance(In, SparseImageStar2DCSR) else V_rem

            if m == 0:
                if is_prob:
                    return In.with_updated_state(c=new_c, V=new_V, shape=In.shape)
                cls = SparseImageStar2DCSR if isinstance(In, SparseImageStar2DCSR) else SparseImageStar2DCOO
                return cls(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub, In.shape, copy_=False)

            V_amb = V_csr[amb_idx32, :]
            c_amb = c_orig[amb_idx32]
            a_l, b_l = a_lb_all[amb_idx32], b_lb_all[amb_idx32]
            a_u, b_u = a_ub_all[amb_idx32], b_ub_all[amb_idx32]
            y_lb, y_ub = y_min_all[amb_idx32], y_max_all[amb_idx32]

            V_l = V_amb.multiply(a_l[:, None]).tocoo()
            V_u = V_amb.multiply(-a_u[:, None]).tocoo()
            eye_data = np.ones(m, dtype=dtype)
            eye_col = np.arange(m, dtype=np.int32) + n
            eye_row = np.arange(m, dtype=np.int32)

            d1 = -a_l * c_amb - b_l
            d2 = a_u * c_amb + b_u

            data = np.hstack([V_l.data, -eye_data, V_u.data, eye_data])
            row = np.hstack([V_l.row, eye_row, V_u.row + m, eye_row + m])
            col = np.hstack([V_l.col, eye_col, V_u.col, eye_col])
            C = sp.csr_array((data, (row, col)), shape=(2 * m, n + m), copy=False)

            if In.C.nnz > 0:
                data_c = np.hstack([In.C.data, C.data])
                indices_c = np.hstack([In.C.indices, C.indices])
                indptr_c = np.hstack([In.C.indptr, C.indptr[1:] + In.C.nnz])
                new_C = sp.csr_array(
                    (data_c, indices_c, indptr_c),
                    shape=(In.C.shape[0] + C.shape[0], C.shape[1]),
                    copy=False
                )
                new_d = np.hstack([In.d, d1, d2])
            else:
                new_C = C
                new_d = np.hstack([d1, d2])

            new_lb = np.hstack([In.pred_lb, y_lb])
            new_ub = np.hstack([In.pred_ub, y_ub])

            if is_prob:
                mu_new, Sig_new, env_rec = propagate_general_elementwise_moments(
                    V_pre=V_amb, c_pre=c_amb,
                    a_lb=a_l, b_lb=b_l, a_ub=a_u, b_ub=b_u,
                    pred_lb_new=y_lb, pred_ub_new=y_ub,
                    mu=In.mu, Sig=In.Sig,
                    fn_eval=_gelu, fn_deriv=_gelu_deriv,
                    cov_mode=In.cov_mode, dtype=dtype
                )
                return In.with_updated_state(
                    c=new_c, V=new_V, C=new_C, d=new_d,
                    pred_lb=new_lb, pred_ub=new_ub, shape=In.shape,
                    mu=mu_new, Sig=Sig_new, new_env_record=env_rec
                )
            cls = SparseImageStar2DCSR if isinstance(In, SparseImageStar2DCSR) else SparseImageStar2DCOO
            return cls(new_c, new_V, new_C, new_d, new_lb, new_ub, In.shape, copy_=False)

        raise Exception(f"Unsupported input set type in GELULayer: {type(In)}")

    def reach(self, inputSet, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False, **kwargs):
        if isinstance(inputSet, list):
            return [self.reachSingleInput(s, lp_solver=lp_solver, RF=RF, show=show) for s in inputSet]
        return self.reachSingleInput(inputSet, lp_solver=lp_solver, RF=RF, show=show)
