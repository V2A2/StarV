"""
Even power operater
Zhuoyang Zhou, 02/05/2026
"""

import numpy as np
import scipy.sparse as sp
from StarV.set.star import Star

class PowerEven(object):
    """
    PowerEven Class for reachability   
    Author: Zhuoyang Zhou
    Date: 02/05/2026 Update: 02/12/2026

    """
    @staticmethod
    def f(x, n):
        """PowerEven function"""
        return np.power(x, n)
    
    @staticmethod
    def df(x, n):
        """Derivative of PowerEven function"""
        return n * np.power(x, n - 1)
    
    @staticmethod
    def reachApprox_star(I, n, opt=True, lp_solver='gurobi', RF=0.0):
        assert isinstance(I, Star), 'error: input set must be a Star set'
        
        N = I.dim
        l, u = I.getRanges(lp_solver=lp_solver, RF=RF)
        yl, yu = PowerEven.f(l, n), PowerEven.f(u, n)
        dyl, dyu = PowerEven.df(l, n), PowerEven.df(u, n)  

        ## l != u
        map0 = np.where(l != u)[0]
        m = len(map0)
        V0 = np.zeros((N, m))
        for i in range(m):
            V0[map0[i], i] = 1
            
        new_V = np.hstack([np.zeros([N, 1]), np.zeros([N, I.nVars]), V0])

        map1 = np.where(l == u)[0]
        if len(map1):
            new_V[map1, 0] = yl[map1]
            new_V[map1, 1:] = 0

        nv = I.nVars + m
        
        ## u < 0, l > 0 & l != u            
        map1 = np.where(((u[map0] <= 0) | (l[map0] >= 0)))[0]
        if len(map1):
            map_ = map0[map1] 
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_] 
            dyl_, dyu_ = dyl[map_], dyu[map_] 
            
            c1, V1 = I.V[map_, 0], I.V[map_, 1:] 
            V2 = V0[map_, :] 

            # broadcasting-safe diagonals
            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # constraint 1: y >= y'(l) * (x - l) + y(l)
            C11 = np.hstack([dyl_diag @ V1, -V2])
            d11 = -dyl_ * (c1 - l_) - yl_

            # constraint 2: y >= y'(u) * (x - u) + y(u)
            C12 = np.hstack([dyu_diag @ V1, -V2])
            d12 = -dyu_ * (c1 - u_) - yu_

            # constraint 3: y <= (y(u) - y(l)) * (x - l) / (u - l) + y(l)
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C13 = np.hstack([-g_diag @ V1, V2])
            d13 = g * (c1 - l_) + yl_

            # constraint 4: y >= y'(xo) * (x - xo) + y(xo)
            xo = 0.5 * (u_ + l_)
            dyo = PowerEven.df(xo, n)          # <-- if your df needs n, use: PowerEven.df(xo, n)
            dyo_diag = np.diag(dyo.flatten())
            C14 = np.hstack([dyo_diag @ V1, -V2])
            d14 = -dyo * (c1 - xo) - PowerEven.f(xo, n)   # <-- if your f needs n, use: PowerEven.f(xo, n)

            C1 = np.vstack((C11, C12, C13, C14))
            d1 = np.hstack((d11, d12, d13, d14))
        else:
            C1 = np.empty((0, nv))
            d1 = np.empty((0,))
        
        # l < 0 < u & and l != u
        map1 = np.where((l[map0] < 0) & (u[map0] > 0))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_   = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]

            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            # broadcasting-safe diagonals
            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # constraint 1: y >= y'(l) * (x - l) + y(l)   (tangent at l, lower bound)
            C21 = np.hstack([dyl_diag @ V1, -V2])
            d21 = -dyl_ * (c1 - l_) - yl_

            # constraint 2: y >= y'(u) * (x - u) + y(u)   (tangent at u, lower bound)
            C22 = np.hstack([dyu_diag @ V1, -V2])
            d22 = -dyu_ * (c1 - u_) - yu_

            # constraint 3: y <= (y(u) - y(l)) * (x - l) / (u - l) + y(l)   (secant, upper bound)
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C23 = np.hstack([-g_diag @ V1, V2])
            d23 = g * (c1 - l_) + yl_

            # constraint 4 (REPLACEMENT): y >= 0
            # In Star constraint form: y >= 0  <=>  -y <= 0
            # If y has no separate center term (common in these activation encodings), this is:
            #   -V2 * a_y <= 0
            Z = np.zeros_like(V1)
            C24 = np.hstack([Z, -V2])
            d24 = np.zeros_like(l_.flatten())

            C2 = np.vstack((C21, C22, C23, C24))
            d2 = np.hstack((d21, d22, d23, d24))
        else:
            C2 = np.empty((0, nv))
            d2 = np.empty((0,))
            
        n_constraints = I.C.shape[0]
        if len(I.d):
            C0 = np.hstack([I.C, np.zeros([n_constraints, m])])
            d0 = I.d
        else:
            C0 = np.empty([0, I.nVars + m])
            d0 = np.empty([0])

        new_C = np.vstack((C0, C1, C2))
        new_d = np.hstack((d0, d1, d2))

        # new_pred_lb = np.hstack((I.pred_lb, yl[map0]))
        # new_pred_ub = np.hstack((I.pred_ub, yu[map0]))
        
        l0 = l[map0]
        u0 = u[map0]

        # endpoint values (keep these for convenience)
        fl = PowerEven.f(l0, n)
        fu = PowerEven.f(u0, n)

        # init
        y_lb = np.minimum(fl, fu)
        y_ub = np.maximum(fl, fu)

        # crossing zero -> min is 0, max is max(|l|^n, |u|^n)
        cross = (l0 <= 0) & (u0 >= 0)
        if np.any(cross):
            y_lb[cross] = 0.0
            y_ub[cross] = np.maximum(PowerEven.f(np.abs(l0[cross]), n),
                                    PowerEven.f(np.abs(u0[cross]), n))

        new_pred_lb = np.hstack((I.pred_lb, y_lb))
        new_pred_ub = np.hstack((I.pred_ub, y_ub))

        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)
    
    @staticmethod
    def reach(I, n, opt=False, delta=0.98, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        if isinstance(I, Star):
            return PowerEven.reachApprox_star(I, n=n, opt=opt, lp_solver=lp_solver, RF=RF)
        # elif isinstance(I, SparseStar):
        #     return PowerEven.reachApprox_sparse(I=I, n=n, opt=opt, delta=delta, lp_solver=lp_solver, RF=RF, DR=DR, show=show)
        # elif isinstance(I, ImageStar):
        #     shape = I.shape()
        #     S = PowerEven.reachApprox_star(I.toStar(), n=n, opt=opt, lp_solver=lp_solver, RF=RF)
        #     return S.toImageStar(image_shape=shape, copy_=False)
        else:
            raise Exception('error: unknown input set')