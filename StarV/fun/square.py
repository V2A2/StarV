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
Square Class
Sung Woo Choi, 06/12/2023

"""

# !/usr/bin/python3
import copy
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.linalg import block_diag
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star

class Square(object):
    """
    Square Class for reachability
    Author: Sung Woo Choi
    Date: 06/12/2023
    
    """

    @staticmethod
    def f(x):
        """square(x)"""
        return x**2
    
    @staticmethod
    def df(x):
        """Derivative of square(x)"""
        return 2*x
    
    def reachApprox_sparse(I, lp_solver='gurobi', RF=0.0, DR=0):
        
        assert isinstance(I, SparseStar), 'error: input set is not a SparseStar set'

        N = I.dim

        l, u = I.getRanges(lp_solver=lp_solver, RF=RF)
        l = l.reshape(N, 1)
        u = u.reshape(N, 1)
        yl = Square.f(l)
        yu = Square.f(u)

        ## l != u
        map0 = np.where(l != u)[0]
        m = len(map0)
        A0 = np.zeros((N, m))
        for i in range(m):
            A0[map0[i], i] = 1
        new_A = np.hstack((np.zeros((N, 1)), A0))

        ## l == u
        map1 = np.where(l == u)[0]
        if len(map1):
            new_A[map1, 0] = yl[map1]
            new_A[map1, 1:m+1] = 0

        nv = I.nVars + m

        plb = np.zeros(m)
        pub = np.zeros(m)

        if len(map0):
            Z = sp.csc_matrix((len(map0), I.nZVars))

            a = (u[map0] + l[map0])
            b = l[map0]*u[map0]

            # constraint 1: y <= (u + l)*x - l*u
            C1 = sp.hstack((Z, -a*I.X(map0), A0[map0, :])).tocsc()
            d1 = (a*I.c(map0) - b).flatten()
        else:
            C1 = sp.csc_matrix((0, nv))
            d1 = np.empty((0))

        ## ((u <= 0) | (l >= 0)) & l != u
        map1 = np.where((u[map0] <= 0) | (l[map0] >= 0))[0]
        if len(map1):
            map_ = map0[map1]
            l_ = l[map_]
            u_ = u[map_]
            yl_ = yl[map_]
            yu_ = yu[map_]

            Z = sp.csc_matrix((len(map_), I.nZVars))

            # constraint 1: y >= (l + u) * x - 0.25 * (l + u)**2
            a = l_ + u_
            C21 = sp.hstack((Z, a*I.X(map_), -A0[map_, :]))
            d21 = -a*I.c(map_) + 0.25 * Square.f(a)

            # constraint 2: y >= 2.0 * l_ * x - l_**2
            a = 2.0 * l_
            C22 = sp.hstack((Z, a*I.X(map_), -A0[map_, :]))
            d22 = -a*l_*I.c(map_) + yl_

            # constraint 3: y >= 2.0 * u_ * x - u_**2
            a = 2.0 * u_
            C23 = sp.hstack((Z, a*I.X(map_), -A0[map_, :]))
            d23 = -a*I.c(map_) + yu_

            C2 = sp.vstack((C21, C22, C23)).tocsc()
            d2 = np.vstack((d21, d22, d23)).flatten()

            plb[map_] = np.minimum(yl_, yu_).flatten()
            pub[map_] = np.maximum(yl_, yu_).flatten()
        else:
            C2 = sp.csc_matrix((0, nv))
            d2 = np.empty((0))

        ## l < 0 & u > 0
        map1 = np.where((l[map0] < 0) & (u[map0] > 0))[0]
        if len(map1):
            map_ = map0[map1]
            l_ = l[map_]
            u_ = u[map_]
            yl_ = yl[map_]
            yu_ = yu[map_]

            Z = sp.csc_matrix((len(map_), I.nZVars))

            # constraint 1: y = x**2 >= 0
            C31 = sp.hstack((Z, np.zeros(I.X(map_).shape), -A0[map_, :]))
            d31 = np.zeros((len(map_), 1))

            # constraint 2: y >= u * x - 0.25 * u**2
            C32 = sp.hstack((Z, u_*I.X(map_), -A0[map_, :]))
            d32 = -u_*I.c(map_) + 0.25 * yu_

            # constraint 3: y >= l * x - 0.25 * l**2
            C33 = sp.hstack((Z, l_*I.X(map_), -A0[map_, :]))
            d33 = -l_*I.c(map_) + 0.25 * yl_

            # constraint 4: y >= 2.0 * u * x - u**2
            a = 2.0 * u_
            C34 = sp.hstack((Z, a*I.X(map_), -A0[map_, :]))
            d34 = -a*I.c(map_) + yu_

            # constraint 5: y >= 2.0 * l * x - l**2
            a = 2.0 * l_
            C35 = sp.hstack((Z, a*I.X(map_), -A0[map_, :]))
            d35 = -a*I.c(map_) + yl_

            C3 = sp.vstack((C31, C32, C33, C34, C35)).tocsc()
            d3 = np.vstack((d31, d32, d33, d34, d35)).flatten()

            pub[map_] = np.maximum(yl_, yu_).flatten()
        else:
            C3 = sp.csc_matrix((0, nv))
            d3 = np.empty((0))

        n = I.C.shape[0]
        if len(I.d):
            C0 = sp.hstack((I.C, sp.csc_matrix((n, m)))) 
            d0 = I.d
        else:
            C0 = sp.csc_matrix((0, I.nVars+m))
            d0 = np.empty((0))

        new_C = sp.vstack((C0, C1, C2, C3))
        new_d = np.hstack((d0, d1, d2, d3))

        new_pred_lb = np.hstack((I.pred_lb, plb))
        new_pred_ub = np.hstack((I.pred_ub, pub))
        new_pred_depth = np.hstack((I.pred_depth+1, np.zeros(m)))
        
        S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
        if DR > 0:
            S = S.depthReduction(DR=DR)
        return S

    def reach(I, lp_solver='gurobi', pool=None, RF=0.0, DR=0):
        if isinstance(I, SparseStar):
            return Square.reachApprox_sparse(I, lp_solver=lp_solver, RF=RF, DR=DR)
        elif isinstance(I, Star):
            # return Square.reachApproxStar(I, lp_solver, RF)
            raise Exception('error: under development')
        else:
            raise Exception('error: unknown input set')




