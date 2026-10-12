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
ExpDecay Class for Mamba Discretization: f(delta) = exp(delta * A), where A <= 0.
Reachability for Star and SparseStar sets.
"""

import copy
import numpy as np
import scipy.sparse as sp
from StarV.set.star import Star
from StarV.set.sparsestar import SparseStar


class ExpDecay(object):
    """
    ExpDecay Class for reachability analysis of discretized state transitions.
    Computes reachable sets for y = exp(delta * A), where A <= 0 is a constant diagonal parameter
    and delta is an input-dependent timescale Star set.
    Since A <= 0, f(delta) is strictly convex over [delta_l, delta_u].
    """

    @staticmethod
    def evaluate(delta, A):
        """
        Evaluate y = exp(delta * A).
        delta: array or scalar
        A: negative array or scalar of same shape or broadcastable
        """
        return np.exp(delta * A)

    @staticmethod
    def reachApprox_star(I, A, lp_solver='gurobi', RF=0.0):
        """
        Approximate reachability of y = exp(delta * A) for Star set.
        I: Star set representing delta (dim = D)
        A: numpy array of decay parameters (dim = D), all A <= 0
        """
        assert isinstance(I, Star), 'error: input set is not a Star set'
        assert isinstance(A, np.ndarray), 'error: A must be a numpy array'
        assert np.all(A <= 0.0), 'error: all elements of A must be <= 0 for ExpDecay'

        A = A.reshape(I.dim)
        N = I.dim
        l, u = I.getRanges(lp_solver=lp_solver, RF=RF)

        yl = np.exp(l * A)
        yu = np.exp(u * A)

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

        if m > 0:
            map_ = map0
            l_, u_ = l[map_], u[map_]
            A_ = A[map_]
            yl_, yu_ = yl[map_], yu[map_]

            c1 = I.V[map_, 0]
            V1 = I.V[map_, 1:I.nVars + 1]
            V2 = V0[map_, :]

            xm_ = 0.5 * (l_ + u_)
            yxm_ = np.exp(xm_ * A_)

            # Derivatives at l, u, xm
            dyl_ = A_ * yl_
            dyu_ = A_ * yu_
            dyxm_ = A_ * yxm_

            # 1. Upper bound (Secant line):
            m_sec = (yu_ - yl_) / (u_ - l_)
            C_up = np.hstack([-m_sec[:, None] * V1, V2])
            d_up = m_sec * (c1 - l_) + yl_

            # 2. Lower bound at l (Tangent):
            C_low_l = np.hstack([dyl_[:, None] * V1, -V2])
            d_low_l = -dyl_ * (c1 - l_) - yl_

            # 3. Lower bound at u (Tangent):
            C_low_u = np.hstack([dyu_[:, None] * V1, -V2])
            d_low_u = -dyu_ * (c1 - u_) - yu_

            # 4. Lower bound at midpoint (Tangent):
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

            # Since A <= 0, exp(delta * A) in (0, 1] if delta >= 0
            new_pred_lb = np.hstack([I.pred_lb, np.minimum(yl_, yu_)])
            new_pred_ub = np.hstack([I.pred_ub, np.maximum(yl_, yu_)])
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
    def reach(I, A, lp_solver='gurobi', pool=None, RF=0.0):
        if isinstance(I, Star):
            return ExpDecay.reachApprox_star(I, A, lp_solver=lp_solver, RF=RF)
        elif isinstance(I, list):
            res = []
            for item in I:
                res.append(ExpDecay.reach(item, A, lp_solver=lp_solver, pool=pool, RF=RF))
            return res
        else:
            raise ValueError(f'Unsupported set type: {type(I)}')
