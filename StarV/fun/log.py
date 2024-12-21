"""
Logarithm Class
Sung Woo Choi, 06/24/2023

"""

# !/usr/bin/python3
import copy
import numpy as np
import scipy.sparse as sp
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star

class Log(object):
    """
    Logarithm Class for reachability
    Author: Sung Woo Choi
    Date: 06/24//2023

    """
    
    @staticmethod
    def f(x):
        """log(x)"""
        return np.log(x)
    
    @staticmethod
    def df(x):
        """Derivative of log(x)"""
        return 1.0/x
    
    def reachApprox_sparse(I, lp_solver='gurobi', RF=0.0, DR=0):
        
        assert isinstance(I, SparseStar), 'error: input set is not a SparseStar set'

        N = I.dim

        l, u = I.getRanges(lp_solver=lp_solver, RF=RF)

        assert (l > 0.0).all(), 'error: log(x) is concave for x > 0 and l <= 0 cannot be bounded due to log(0) = - inf'

        l = l.reshape(N, 1)
        u = u.reshape(N, 1)
        yl = Log.f(l)
        yu = Log.f(u)

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

        if len(map0):
            Z = sp.csc_matrix((len(map0), I.nZVars))

            # constraint 1: y <= 2.0 * x / a + log(0.5*a) - 1
            a = l[map0] + u[map0]
            b = 2.0/a
            C11 = sp.hstack((Z, -b*I.X(map0), A0[map0, :]))
            d11 = b*I.c(map0) + Log.f(0.5*a) - 1.0

            # constraint 2: y >= a * x + log(u) - u*a
            a = Log.f(u[map0] / l[map0]) / (u[map0] - l[map0])
            C12 = sp.hstack((Z, a*I.X(map0), -A0[map0, :]))
            d12 = -a*I.c(map0) - yu[map0] + u[map0]*a

            # constraint 3: y <= x / u + log(u) - 1.0
            a = 1 / u[map0]
            C13 = sp.hstack((Z, -a*I.X(map0), A0[map0, :]))
            d13 = a*I.c(map0) + Log.f(u[map0]) - 1.0

            # constraint 4: y <= x / l + log(l) - 1.0
            a = 1 / l[map0]
            C14 = sp.hstack((Z, -a*I.X(map0), A0[map0, :]))
            d14 = a*I.c(map0) + Log.f(l[map0]) - 1.0
 
            C1 = sp.vstack((C11, C12, C13, C14)).tocsc()
            d1 = np.vstack((d11, d12, d13, d14)).flatten()
        else:
            C1 = sp.csc_matrix((0, nv))
            d1 = np.empty((0))
        
        n = I.C.shape[0]
        if len(I.d):
            C0 = sp.hstack((I.C, sp.csc_matrix((n, m)))) 
            d0 = I.d
        else:
            C0 = sp.csc_matrix((0, I.nVars+m))
            d0 = np.empty((0))

        new_C = sp.vstack((C0, C1))
        new_d = np.hstack((d0, d1))

        new_pred_lb = np.hstack((I.pred_lb, yl[map0].flatten()))
        new_pred_ub = np.hstack((I.pred_ub, yu[map0].flatten()))
        new_pred_depth = np.hstack((I.pred_depth+1, np.zeros(m)))
        
        S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
        if DR > 0:
            S = S.depthReduction(DR=DR)
        return S
    
    def reach(I, lp_solver='gurobi', pool=None, RF=0.0, DR=0):
        if isinstance(I, SparseStar):
            return Log.reachApprox_sparse(I, lp_solver=lp_solver, RF=RF, DR=DR)
        elif isinstance(I, Star):
            # return Square.reachApproxStar(I, lp_solver, RF)
            raise Exception('error: under development')
        else:
            raise Exception('error: unknown input set')