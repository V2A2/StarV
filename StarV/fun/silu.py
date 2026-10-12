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
SiLU (Swish-1) Class: f(x) = x * sigmoid(x) = x / (1 + e^-x)
Reachability for Star and SparseStar sets.
"""

import copy
import numpy as np
import scipy.sparse as sp
from StarV.set.star import Star
from StarV.set.sparsestar import SparseStar


class SiLU(object):
    """
    SiLU (Swish-1) Class for reachability analysis.
    f(x) = x * sigmoid(x)
    Global minimum: x* = -1.27846454, f(x*) = -0.27846454
    Inflection points: x = +/- 2.399357
    """

    SILU_MIN_VAL = -0.27846454276107363

    @staticmethod
    def evaluate(x):
        return SiLU.f(x)

    @staticmethod
    def f(x):
        """Numerically stable SiLU"""
        x_arr = np.asarray(x, dtype=np.float64)
        sig = 1.0 / (1.0 + np.exp(-np.clip(x_arr, -50.0, 50.0)))
        return x_arr * sig

    @staticmethod
    def df(x):
        """Derivative of SiLU: f'(x) = sig(x) * (1 + x * (1 - sig(x)))"""
        x_arr = np.asarray(x, dtype=np.float64)
        sig = 1.0 / (1.0 + np.exp(-np.clip(x_arr, -50.0, 50.0)))
        return sig * (1.0 + x_arr * (1.0 - sig))

    @staticmethod
    def get_convex_bounds(l, u, K=100):
        """
        Compute provably sound upper and lower linear bounding planes for SiLU
        over intervals [l, u].
        Uses secant line with Taylor-bounded envelope:
        Since |f''(x)| <= 0.5 everywhere, grid interpolation error is <= h^2 / 16.
        """
        yl = SiLU.f(l)
        yu = SiLU.f(u)
        m_sec = (yu - yl) / (u - l)

        h = (u - l) / float(K)
        t = np.linspace(0.0, 1.0, K + 1)
        grid = l[:, None] + t[None, :] * (u[:, None] - l[:, None])
        y_grid = SiLU.f(grid)
        y_sec = m_sec[:, None] * (grid - l[:, None]) + yl[:, None]
        diff = y_grid - y_sec

        margin = (h**2) / 16.0
        d_max = np.maximum(0.0, np.max(diff, axis=1) + margin)
        d_min = np.minimum(0.0, np.min(diff, axis=1) - margin)

        return m_sec, yl, yu, d_min, d_max

    @staticmethod
    def reachApprox_star(I, lp_solver='gurobi', RF=0.0, K=100):
        """
        Approximate reachability of SiLU for Star set.
        """
        assert isinstance(I, Star), 'error: input set is not a Star set'

        N = I.dim
        l, u = I.getRanges(lp_solver=lp_solver, RF=RF)

        map0 = np.where(l != u)[0]
        m = len(map0)

        V0 = np.zeros((N, m))
        for i in range(m):
            V0[map0[i], i] = 1.0

        new_V = np.hstack([np.zeros([N, I.nVars + 1]), V0])

        map1 = np.where(l == u)[0]
        if len(map1) > 0:
            new_V[map1, 0] = SiLU.f(l[map1])
            new_V[map1, 1:m + 1] = 0.0

        if m > 0:
            map_ = map0
            l_, u_ = l[map_], u[map_]

            m_sec, yl_, yu_, d_min, d_max = SiLU.get_convex_bounds(l_, u_, K=K)

            c1 = I.V[map_, 0]
            V1 = I.V[map_, 1:I.nVars + 1]
            V2 = V0[map_, :]

            # 1. Upper bound constraint:
            # y <= m_sec * (x - l) + yl + d_max
            # <=> -m_sec * V1 * alpha + V2 * y <= m_sec * (c1 - l) + yl + d_max
            C_up = np.hstack([-m_sec[:, None] * V1, V2])
            d_up = m_sec * (c1 - l_) + yl_ + d_max

            # 2. Lower bound constraint:
            # y >= m_sec * (x - l) + yl + d_min
            # <=> m_sec * V1 * alpha - V2 * y <= -m_sec * (c1 - l) - yl - d_min
            C_low = np.hstack([m_sec[:, None] * V1, -V2])
            d_low = -m_sec * (c1 - l_) - yl_ - d_min

            new_C_added = np.vstack([C_up, C_low])
            new_d_added = np.hstack([d_up, d_low])

            if len(I.C) > 0:
                C0 = np.hstack([I.C, np.zeros([I.C.shape[0], m])])
                new_C = np.vstack([C0, new_C_added])
                new_d = np.hstack([I.d, new_d_added])
            else:
                new_C = new_C_added
                new_d = new_d_added

            # Predicate bounds: y in [min(yl, yu, SiLU_MIN), max(yl, yu)]
            y_pred_lb = np.maximum(SiLU.SILU_MIN_VAL, np.minimum(yl_, yu_) + d_min)
            y_pred_ub = np.maximum(yl_, yu_) + d_max

            new_pred_lb = np.hstack([I.pred_lb, y_pred_lb])
            new_pred_ub = np.hstack([I.pred_ub, y_pred_ub])
        else:
            if len(I.C) > 0:
                new_C = I.C
                new_d = I.d
            else:
                new_C = np.empty([0, I.nVars])
                new_d = np.empty(0)
            new_pred_lb = I.pred_lb
            new_pred_ub = I.pred_ub

        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)

    @staticmethod
    def reachApprox_sparse(I, lp_solver='gurobi', RF=0.0, DR=0, K=100):
        """
        Approximate reachability of SiLU for SparseStar set.
        """
        assert isinstance(I, SparseStar), 'error: input set is not a SparseStar set'

        N = I.dim
        l, u = I.getRanges(lp_solver=lp_solver, RF=RF)

        map0 = np.where(l != u)[0]
        m = len(map0)

        A0 = np.zeros((N, m))
        for i in range(m):
            A0[map0[i], i] = 1.0

        new_A = np.hstack([np.zeros((N, 1)), A0])

        map1 = np.where(l == u)[0]
        if len(map1) > 0:
            new_A[map1, 0] = SiLU.f(l[map1])
            new_A[map1, 1:m + 1] = 0.0

        if m > 0:
            map_ = map0
            l_, u_ = l[map_], u[map_]

            m_sec, yl_, yu_, d_min, d_max = SiLU.get_convex_bounds(l_, u_, K=K)

            Z = sp.csc_matrix((m, I.nZVars))
            X_map = I.X(map_)
            c_map = I.c(map_).reshape(-1)

            # Upper bound:
            C_up = sp.hstack([Z, -m_sec[:, None] * X_map, A0[map_, :]])
            d_up = (m_sec * (c_map - l_) + yl_ + d_max).flatten()

            # Lower bound:
            C_low = sp.hstack([Z, m_sec[:, None] * X_map, -A0[map_, :]])
            d_low = (-m_sec * (c_map - l_) - yl_ - d_min).flatten()

            C_added = sp.vstack([C_up, C_low]).tocsc()
            d_added = np.hstack([d_up, d_low])

            if len(I.d) > 0:
                C0 = sp.hstack([I.C, sp.csc_matrix((I.C.shape[0], m))])
                new_C = sp.vstack([C0, C_added]).tocsc()
                new_d = np.hstack([I.d, d_added])
            else:
                new_C = C_added
                new_d = d_added

            y_pred_lb = np.maximum(SiLU.SILU_MIN_VAL, np.minimum(yl_, yu_) + d_min)
            y_pred_ub = np.maximum(yl_, yu_) + d_max

            new_pred_lb = np.hstack([I.pred_lb, y_pred_lb])
            new_pred_ub = np.hstack([I.pred_ub, y_pred_ub])
            new_pred_depth = np.hstack([I.pred_depth + 1, np.zeros(m)])
        else:
            new_C = I.C
            new_d = I.d
            new_pred_lb = I.pred_lb
            new_pred_ub = I.pred_ub
            new_pred_depth = I.pred_depth

        S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
        if DR > 0:
            S = S.depthReduction(DR)
        return S

    @staticmethod
    def reach(I, lp_solver='gurobi', pool=None, RF=0.0, DR=0, K=100):
        if isinstance(I, Star):
            return SiLU.reachApprox_star(I, lp_solver=lp_solver, RF=RF, K=K)
        elif isinstance(I, SparseStar):
            return SiLU.reachApprox_sparse(I, lp_solver=lp_solver, RF=RF, DR=DR, K=K)
        elif isinstance(I, list):
            res = []
            for item in I:
                res.append(SiLU.reach(item, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, K=K))
            return res
        else:
            raise ValueError(f'Unsupported set type: {type(I)}')
