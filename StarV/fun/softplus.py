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
Softplus Class (Softplus activation function: ln(1 + e^x))
Reachability for Star and SparseStar sets.
"""

import copy
import numpy as np
import scipy.sparse as sp
from StarV.set.star import Star
from StarV.set.sparsestar import SparseStar


class Softplus(object):
    """
    Softplus Class for reachability analysis.
    f(x) = ln(1 + e^x)
    Strictly convex everywhere on R.
    """

    @staticmethod
    def evaluate(x):
        return Softplus.f(x)

    @staticmethod
    def f(x):
        """Numerically stable softplus: ln(1 + e^x)"""
        x_arr = np.asarray(x, dtype=np.float64)
        return np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0.0)

    @staticmethod
    def df(x):
        """Derivative of softplus(x) is sigmoid(x) = 1 / (1 + e^-x)"""
        x_arr = np.asarray(x, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(x_arr, -50.0, 50.0)))

    @staticmethod
    def reachApprox_star(I, lp_solver='gurobi', RF=0.0):
        """
        Approximate reachability of Softplus for Star set.
        Since softplus is strictly convex everywhere:
        - Upper bound is the secant line between (l, f(l)) and (u, f(u)).
        - Lower bounds are the tangent lines at l, u, and midpoint xm = (l+u)/2.
        """
        assert isinstance(I, Star), 'error: input set is not a Star set'

        N = I.dim
        l, u = I.getRanges(lp_solver=lp_solver, RF=RF)
        yl, yu = Softplus.f(l), Softplus.f(u)
        dyl, dyu = Softplus.df(l), Softplus.df(u)

        map0 = np.where(l != u)[0]
        m = len(map0)

        V0 = np.zeros((N, m))
        for i in range(m):
            V0[map0[i], i] = 1.0

        new_V = np.hstack([np.zeros([N, I.nVars + 1]), V0])

        map1 = np.where(l == u)[0]
        if len(map1) > 0:
            new_V[map1, 0] = yl[map1]
            new_V[map1, 1:m + 1] = 0.0

        nv = I.nVars + m

        if m > 0:
            map_ = map0
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]

            c1 = I.V[map_, 0]
            V1 = I.V[map_, 1:I.nVars + 1]
            V2 = V0[map_, :]

            xm_ = 0.5 * (l_ + u_)
            yxm_ = Softplus.f(xm_)
            dyxm_ = Softplus.df(xm_)

            # 1. Upper bound (Secant line):
            # y <= m_sec * (x - l) + yl  <=>  -m_sec * V1 * alpha + V2 * y <= m_sec * c1 + yl - m_sec * l
            m_sec = (yu_ - yl_) / (u_ - l_)
            C_up = np.hstack([-m_sec[:, None] * V1, V2])
            d_up = m_sec * (c1 - l_) + yl_

            # 2. Lower bound at l (Tangent at l):
            # y >= dyl * (x - l) + yl  <=>  dyl * V1 * alpha - V2 * y <= -dyl * (c1 - l) - yl
            C_low_l = np.hstack([dyl_[:, None] * V1, -V2])
            d_low_l = -dyl_ * (c1 - l_) - yl_

            # 3. Lower bound at u (Tangent at u):
            C_low_u = np.hstack([dyu_[:, None] * V1, -V2])
            d_low_u = -dyu_ * (c1 - u_) - yu_

            # 4. Lower bound at xm (Tangent at midpoint):
            C_low_m = np.hstack([dyxm_[:, None] * V1, -V2])
            d_low_m = -dyxm_ * (c1 - xm_) - yxm_

            new_C_added = np.vstack([C_up, C_low_l, C_low_u, C_low_m])
            new_d_added = np.hstack([d_up, d_low_l, d_low_u, d_low_m])

            if len(I.C) > 0:
                C0 = np.hstack([I.C, np.zeros([I.C.shape[0], m])])
                new_C = np.vstack([C0, new_C_added])
                new_d = np.hstack([I.d, new_d_added])
            else:
                new_C = new_C_added
                new_d = new_d_added

            new_pred_lb = np.hstack([I.pred_lb, np.maximum(0.0, yl_)])
            new_pred_ub = np.hstack([I.pred_ub, yu_])
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
    def reachApprox_sparse(I, lp_solver='gurobi', RF=0.0, DR=0):
        """
        Approximate reachability of Softplus for SparseStar set.
        """
        assert isinstance(I, SparseStar), 'error: input set is not a SparseStar set'

        N = I.dim
        l, u = I.getRanges(lp_solver=lp_solver, RF=RF)
        yl, yu = Softplus.f(l), Softplus.f(u)
        dyl, dyu = Softplus.df(l), Softplus.df(u)

        map0 = np.where(l != u)[0]
        m = len(map0)

        A0 = np.zeros((N, m))
        for i in range(m):
            A0[map0[i], i] = 1.0

        new_A = np.hstack([np.zeros((N, 1)), A0])

        map1 = np.where(l == u)[0]
        if len(map1) > 0:
            new_A[map1, 0] = yl[map1]
            new_A[map1, 1:m + 1] = 0.0

        nv = I.nVars + m

        if m > 0:
            map_ = map0
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]

            xm_ = 0.5 * (l_ + u_)
            yxm_ = Softplus.f(xm_)
            dyxm_ = Softplus.df(xm_)

            m_sec = (yu_ - yl_) / (u_ - l_)

            Z = sp.csc_matrix((m, I.nZVars))
            X_map = I.X(map_)
            c_map = I.c(map_).reshape(-1)

            # 1. Upper bound (Secant):
            C1 = sp.hstack([Z, -m_sec[:, None] * X_map, A0[map_, :]])
            d1 = (m_sec * (c_map - l_) + yl_).flatten()

            # 2. Lower bound at l:
            C2 = sp.hstack([Z, dyl_[:, None] * X_map, -A0[map_, :]])
            d2 = (-dyl_ * (c_map - l_) - yl_).flatten()

            # 3. Lower bound at u:
            C3 = sp.hstack([Z, dyu_[:, None] * X_map, -A0[map_, :]])
            d3 = (-dyu_ * (c_map - u_) - yu_).flatten()

            # 4. Lower bound at midpoint:
            C4 = sp.hstack([Z, dyxm_[:, None] * X_map, -A0[map_, :]])
            d4 = (-dyxm_ * (c_map - xm_) - yxm_).flatten()

            C_added = sp.vstack([C1, C2, C3, C4]).tocsc()
            d_added = np.hstack([d1, d2, d3, d4])

            if len(I.d) > 0:
                C0 = sp.hstack([I.C, sp.csc_matrix((I.C.shape[0], m))])
                new_C = sp.vstack([C0, C_added]).tocsc()
                new_d = np.hstack([I.d, d_added])
            else:
                new_C = C_added
                new_d = d_added

            new_pred_lb = np.hstack([I.pred_lb, np.maximum(0.0, yl_)])
            new_pred_ub = np.hstack([I.pred_ub, yu_])
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
    def reach(I, lp_solver='gurobi', pool=None, RF=0.0, DR=0):
        if isinstance(I, Star):
            return Softplus.reachApprox_star(I, lp_solver=lp_solver, RF=RF)
        elif isinstance(I, SparseStar):
            return Softplus.reachApprox_sparse(I, lp_solver=lp_solver, RF=RF, DR=DR)
        elif isinstance(I, list):
            res = []
            for item in I:
                res.append(Softplus.reach(item, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR))
            return res
        else:
            raise ValueError(f'Unsupported set type: {type(I)}')
