"""
IdentityXIdentity Class (x * y, where x \in X, y \in Y, X and Y are star sets)
Sung Woo Choi, 06/18/2023

"""

# !/usr/bin/python3
import copy
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.linalg import block_diag
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star

class IdentityXIdentity(object):
    """
    IdentityXIdentity Class for reachability
    Author: Sung Woo Choi
    Date: 06/18/2023

    """

    u_ = 0 # index for upper bound case
    l_ = 1 # index for lower bound case

    x_ = 0 # index for x-coordinate constraint
    y_ = 1 # index for y-coordinate constraint
    z_ = 2 # index for z-coordinate constraint

    iux_ = 3 # intersection line on x-coordinate for upper bound case
    iuy_ = 4 # intersection line on y-coordinate for upper bound case

    ilx_ = 5 # intersection line on x-coordinate for lower bound case
    ily_ = 6 # intersection line on y-coordinate for lower bound case
    
    num_of_points = 4

    z_max = num_of_points-1   
    z_min = 0 

    @staticmethod
    def evaluate(x, y):
        return x * y

    @staticmethod
    def f(x, y):
        return x * y
    
    @staticmethod
    def gf(x, y):
        """Gradient of x*y"""
        return x, y
    
    @staticmethod
    def getX(p, bound):
        len_ = len(p.shape)
        if len_ == 2:
            if bound == IdentityXIdentity.u_:
                for i in range(IdentityXIdentity.z_max-1, IdentityXIdentity.z_min-1, -1):
                    if p[i, IdentityXIdentity.iuy_] == np.inf:
                        return copy.deepcopy(p[i, IdentityXIdentity.x_ : IdentityXIdentity.z_+1])
            elif bound == IdentityXIdentity.l_:
                for i in range(IdentityXIdentity.z_min+1, IdentityXIdentity.z_max+1):
                    if p[i, IdentityXIdentity.ily_] == np.inf:
                        return copy.deepcopy(p[i, IdentityXIdentity.x_ : IdentityXIdentity.z_+1])
            else:
                raise Exception('error: unknown bound; should be u_ or l_')
            
        elif len_ == 3:
            dim = p.shape[0]
            px = np.zeros((dim, IdentityXIdentity.z_+1))
            for k in range(dim):
                if bound == IdentityXIdentity.u_:
                    for i in range(IdentityXIdentity.z_max-1, IdentityXIdentity.z_min-1, -1):
                        if p[k, i, IdentityXIdentity.iuy_] == np.inf:
                            px[k, :] = p[k, i, IdentityXIdentity.x_ : IdentityXIdentity.z_+1]
                            break
                elif bound == IdentityXIdentity.l_:
                    for i in range(IdentityXIdentity.z_min+1, IdentityXIdentity.z_max+1):
                        if p[k, i, IdentityXIdentity.ily_] == np.inf:
                            px[k, :] = p[k, i, IdentityXIdentity.x_ : IdentityXIdentity.z_+1]
                            break
                else:
                    raise Exception('error: unknown bound; should be u_ or l_')
            return copy.deepcopy(px)
        
        else:
            raise Exception('error: supports 2D and 3D matrices')

    @staticmethod
    def getY(p, bound):
        len_ = len(p.shape)
        if len_ == 2:
            if bound == IdentityXIdentity.u_:
                for i in range(IdentityXIdentity.z_max-1, IdentityXIdentity.z_min-1, -1):
                    if p[i, IdentityXIdentity.iux_] == np.inf:
                        return copy.deepcopy(p[i, IdentityXIdentity.x_ : IdentityXIdentity.z_+1])
            elif bound == IdentityXIdentity.l_:
                for i in range(IdentityXIdentity.z_min+1, IdentityXIdentity.z_max+1):
                    if p[i, IdentityXIdentity.ilx_] == np.inf:
                        return copy.deepcopy(p[i, IdentityXIdentity.x_ : IdentityXIdentity.z_+1])
            else:
                raise Exception('error: unknown bound; should be u_ or l_')
            
        elif len_ == 3:
            dim = p.shape[0]
            py = np.zeros((dim, IdentityXIdentity.z_+1))
            for k in range(dim):
                if bound == IdentityXIdentity.u_:
                    for i in range(IdentityXIdentity.z_max-1, IdentityXIdentity.z_min-1, -1):
                        if p[k, i, IdentityXIdentity.iux_] == np.inf:
                            py[k, :] = p[k, i, IdentityXIdentity.x_ : IdentityXIdentity.z_+1]
                            break
                elif bound == IdentityXIdentity.l_:
                    for i in range(IdentityXIdentity.z_min+1, IdentityXIdentity.z_max+1):
                        if p[k, i, IdentityXIdentity.ilx_] == np.inf:
                            py[k, :] = p[k, i, IdentityXIdentity.x_ : IdentityXIdentity.z_+1]
                            break
                else:
                    raise Exception('error: unknown bound; should be u_ or l_')
            return copy.deepcopy(py)
        
        else:
            raise Exception('error: supports 2D and 3D matrices')
        
    @staticmethod
    def getO(p, index):
        # get opposite point
        len_ = len(p.shape)
        if len_ == 2:
            for i in range(IdentityXIdentity.num_of_points):
                if p[i, IdentityXIdentity.x_] != p[index, IdentityXIdentity.x_] and p[i, IdentityXIdentity.y_] != p[index, IdentityXIdentity.y_]:
                    return copy.deepcopy(p[i, IdentityXIdentity.x_ : IdentityXIdentity.z_+1])
                
        elif len_ == 3:
            dim = p.shape[0]
            po = np.zeros((dim, IdentityXIdentity.z_+1))
            for k in range(dim):
                for i in range(IdentityXIdentity.num_of_points):
                    if p[k, i, IdentityXIdentity.x_] != p[k, index, IdentityXIdentity.x_] and p[k, i, IdentityXIdentity.y_] != p[k, index, IdentityXIdentity.y_]:
                        po[k, :] = p[k, i, IdentityXIdentity.x_ : IdentityXIdentity.z_+1]
                        break
            return copy.deepcopy(po)
        
        else:
            raise Exception('error: supports 2D and 3D matrices')

    @staticmethod
    def getConstraints(xl, xu, yl, yu):

        assert isinstance(xl, float), 'error: lower bound, xl, should be a floating point number'
        assert isinstance(xu, float), 'error: upper bound, xu, should be a floating point number'
        assert isinstance(yl, float), 'error: lower bound, yl, should be a floating point number'
        assert isinstance(yu, float), 'error: upper bound, yu, should be a floating point number'

        u_ = IdentityXIdentity.u_
        l_ = IdentityXIdentity.l_
    
        x_ = IdentityXIdentity.x_
        y_ = IdentityXIdentity.y_
        z_ = IdentityXIdentity.z_
        iux_ = IdentityXIdentity.iux_
        iuy_ = IdentityXIdentity.iuy_
        ilx_ = IdentityXIdentity.ilx_
        ily_ = IdentityXIdentity.ily_
        
        z_max = IdentityXIdentity.z_max
        z_min = IdentityXIdentity.z_min
    
        num_of_points = IdentityXIdentity.num_of_points

        pzs = np.zeros((num_of_points, 3))

        pzs[0, x_:y_+1] = [xl, yl]
        pzs[1, x_:y_+1] = [xu, yu]
        pzs[2, x_:y_+1] = [xu, yl]
        pzs[3, x_:y_+1] = [xl, yu]

        pzs[:,  z_] = IdentityXIdentity.f(pzs[:, x_], pzs[:,y_])

        # sort z-coordinate points in ascending order
        z_sorted_i = np.argsort(pzs[:, z_])
        pzs = np.hstack([pzs[z_sorted_i, :], np.full((num_of_points, num_of_points), np.nan)])

        #####################################################################
        #                       Upper bound case
        #####################################################################
        # upper bound case for finding slopes from z_max point to other points
        z_max_range = pzs[z_max, z_] - pzs[:, z_] # slope in z-coordinate
        with np.errstate(divide='ignore', invalid='ignore'):
            pzs[:, iux_] = z_max_range / (pzs[z_max, x_] - pzs[:, x_]) # slope in x-coordinate
            pzs[:, iuy_] = z_max_range / (pzs[z_max, y_] - pzs[:, y_]) # slope in y-coordinate

        px = IdentityXIdentity.getX(pzs, u_)
        py = IdentityXIdentity.getY(pzs, u_)
        po = IdentityXIdentity.getO(pzs, z_max)

        gammaU = (pzs[z_max, z_] - px[z_]) / (pzs[z_max, x_] - px[x_])
        alphaU = (px[z_] - po[z_]) / (px[y_] - po[y_])

        muU = (pzs[z_max, z_] - py[z_]) / (pzs[z_max, y_] - py[y_])
        betaU = (py[z_] - po[z_]) / (py[x_] - po[x_])

        z_zmax = z_sorted_i[z_max]
        # upper hyper-plane constraints on x-coordinate and y-coordinate
        if z_zmax == 0 or z_zmax == 1:
            #splitted
            Ux = np.hstack([gammaU, betaU])
            Uy = np.hstack([alphaU, muU])
            Ub = Ux.reshape(-1)*pzs[z_max, x_] + Uy.reshape(-1)*pzs[z_max, y_] - pzs[z_max, z_]

        elif z_zmax == 2 or z_zmax == 3:
            Ux = np.hstack([gammaU, betaU])
            Uy = np.hstack([muU, alphaU])
            Ub = Ux.reshape(-1)*pzs[z_max, x_] + Uy.reshape(-1)*pzs[z_max, y_] - pzs[z_max, z_]
            Ub1 = Ux[0]*pzs[z_max, x_] + Uy[0]*pzs[z_max, y_] - pzs[z_max, z_]
            Ub2 = Ux[1]*pzs[z_min, x_] + Uy[1]*pzs[z_min, y_] - pzs[z_min, z_]
            Ub = np.hstack((Ub1, Ub2))

        else:
            raise Exception('error: unknown point')

        #####################################################################
        #                       Lower bound case
        #####################################################################
        # combine lower bound pzs with upper bound pzs
        # lower bound case for finding slopes from z_min point to other points
        z_min_range = pzs[:, z_] - pzs[z_min, z_] # slope in z-coordinate
        with np.errstate(divide='ignore', invalid='ignore'):
            pzs[:, ilx_] = z_min_range / (pzs[:, x_] - pzs[z_min, x_]) # slope in x-coordinate
            pzs[:, ily_] = z_min_range / (pzs[:, y_] - pzs[z_min, y_]) # slope in y-coordinate

        px = IdentityXIdentity.getX(pzs, l_)
        py = IdentityXIdentity.getY(pzs, l_)
        po = IdentityXIdentity.getO(pzs, z_min)

        gammaL = (pzs[z_min, z_] - px[z_]) / (pzs[z_min, x_] - px[x_])
        alphaL = (px[z_] - po[z_]) / (px[y_] - po[y_])

        muL = (pzs[z_min, z_] - py[z_]) / (pzs[z_min, y_] - py[y_])
        betaL = (py[z_] - po[z_]) / (py[x_] - po[x_])

        z_zmin = z_sorted_i[z_min]
        # lower hyper-plane constraints on x-coordinate and y-coordinate
        if  z_zmin == 0 or z_zmin == 1:
            #splitted
            Lx = np.hstack([gammaL, betaL])
            Ly = np.hstack([muL, alphaL])

            Lb1 = Lx[0]*pzs[z_min, x_] + Ly[0]*pzs[z_min, y_] - pzs[z_min, z_]
            Lb2 = Lx[1]*pzs[z_max, x_] + Ly[1]*pzs[z_max, y_] - pzs[z_max, z_]
            Lb = np.hstack((Lb1, Lb2))

        elif z_zmin == 2 or z_zmin == 3:
            Lx = np.hstack([gammaL, betaL])
            Ly = np.hstack([alphaL, muL])
            Lb = Lx.reshape(-1)*pzs[z_min, x_] + Ly.reshape(-1)*pzs[z_min, y_] - pzs[z_min, z_]

        else:
            raise Exception('error: unknown point')

        return Ux, Uy, Ub, Lx, Ly, Lb, pzs[z_max, z_], pzs[z_min, z_]
    
    @staticmethod
    def getMultiConstraints(xl, xu, yl, yu):
        """
            Get four hyper-plane linear constraints thatt constraints x*y function.
            The method is equivalent to convexull of four vertices
            
            Args:
                @xl: lower bound vector of an input set, X
                @xu: upper bound vector of an input set, X
                @yl: lower bound vector of an input set, Y
                @yu: upper bound vector of an input set, Y
                
            Returns:
                @Ux = [ux1, ux2], where ux1, ux2 are column vector
                @Uy = [uy1, uy2], where uy1, uy2 are column vector
                @Ub = [ub1, ub2], where ub1, ub2 are column vector
                @Lx = [lx1, lx2], where lx1, lx2 are column vector
                @Ly = [ly1, ly2], where ly1, ly2 are column vector
                @Lb = [lb1, lb2], where lb1, lb2 are column vector
                
                Hyper-planes:
                    1. z <= ux1 * x + uy1 * y + ub1
                    2. z <= ux2 * x + uy2 * y + ub2
                    3. z >= lx1 * x + ly1 * y + lb1
                    4. z >= lx2 * x + ly2 * y + lb2
            
            Author: Sung Woo Choi
            Date: 06/23/2023
        """

        assert isinstance(xl, np.ndarray), 'error: lower bound, xl, should be a 1D numpy array'
        assert isinstance(xu, np.ndarray), 'error: upper bound, xu, should be a 1D numpy array'
        assert isinstance(yl, np.ndarray), 'error: lower bound, yl, should be a 1D numpy array'
        assert isinstance(yu, np.ndarray), 'error: upper bound, yu, should be a 1D numpy array'

        u_ = IdentityXIdentity.u_
        l_ = IdentityXIdentity.l_

        x_ = IdentityXIdentity.x_
        y_ = IdentityXIdentity.y_
        z_ = IdentityXIdentity.z_
        iux_ = IdentityXIdentity.iux_
        iuy_ = IdentityXIdentity.iuy_
        ilx_ = IdentityXIdentity.ilx_
        ily_ = IdentityXIdentity.ily_

        z_max = IdentityXIdentity.z_max
        z_min = IdentityXIdentity.z_min

        num_of_points = IdentityXIdentity.num_of_points

        dim = len(xl)

        pzs = np.zeros((dim, num_of_points, 3))
        
        pzs[:, 0, x_:y_+1] = np.vstack([xl, yl]).T
        pzs[:, 1, x_:y_+1] = np.vstack([xu, yu]).T
        pzs[:, 2, x_:y_+1] = np.vstack([xu, yl]).T
        pzs[:, 3, x_:y_+1] = np.vstack([xl, yu]).T
        
        pzs[:, :, z_] = IdentityXIdentity.f(pzs[:, :, x_], pzs[:, :, y_])
        
        # sort z-coordinate points in ascending order
        z_sorted_i = pzs[:, :, z_].argsort()
        pzs = [pzs[i, z_sorted_i[i]] for i in range(dim)]
        
        pzs = np.concatenate((pzs, np.full((dim, num_of_points, num_of_points), np.nan)), axis=2)
        
        #####################################################################
        #                       Upper bound case
        #####################################################################
        # upper bound case for finding slopes from z_max point to other points
        z_max_range = pzs[:, z_max, z_, np.newaxis] - pzs[:, :, z_] # slope in z-coordinate
        with np.errstate(divide='ignore', invalid='ignore'):
            pzs[:, :, iux_] = z_max_range / (pzs[:, z_max, x_, np.newaxis] - pzs[:, :, x_]) # slope in x-coordinate
            pzs[:, :, iuy_] = z_max_range / (pzs[:, z_max, y_, np.newaxis] - pzs[:, :, y_]) # slope in y-coordinate

        px = IdentityXIdentity.getX(pzs, u_)
        py = IdentityXIdentity.getY(pzs, u_)
        po = IdentityXIdentity.getO(pzs, z_max)
        
        gammaU = (pzs[:, z_max, z_] - px[:, z_]) / (pzs[:, z_max, x_] - px[:, x_])
        alphaU = (px[:, z_] - po[:, z_]) / (px[:, y_] - po[:, y_])

        muU = (pzs[:, z_max, z_] - py[:, z_]) / (pzs[:, z_max, y_] - py[:, y_])
        betaU = (py[:, z_] - po[:, z_]) / (py[:, x_] - po[:, x_])

        Ux = np.zeros((2, dim))
        Uy = np.zeros((2, dim))
        Ub = np.zeros((2, dim))

        z_zmax = z_sorted_i[:, z_max]
        # upper hyper-plane constraints on x-coordinate and y-coordinate
        map1 = np.where((z_zmax == 0) | (z_zmax == 1))[0] #splitted
        if len(map1):
            Ux[:, map1] = np.row_stack([gammaU[map1], betaU[map1]])
            Uy[:, map1] = np.row_stack([alphaU[map1], muU[map1]])
            Ub[:, map1] = Ux[:, map1]*pzs[map1, z_max, x_] + Uy[:, map1]*pzs[map1, z_max, y_] - pzs[map1, z_max, z_]
            
        
        map1 = np.where((z_zmax == 2) | (z_zmax == 3))[0]
        if len(map1):
            Ux[:, map1] = np.row_stack([gammaU[map1], betaU[map1]])
            Uy[:, map1]= np.row_stack([muU[map1], alphaU[map1]])
            Ub1 = Ux[0, map1] * pzs[map1, z_max, x_] + Uy[0, map1] * pzs[map1, z_max, y_] - pzs[map1, z_max, z_]
            Ub2 = Ux[1, map1] * pzs[map1, z_min, x_] + Uy[1, map1] * pzs[map1, z_min, y_] - pzs[map1, z_min, z_]
            Ub[:, map1] = np.vstack((Ub1, Ub2))

        #####################################################################
        #                       Lower bound case
        #####################################################################
        # combine lower bound pzs with upper bound pzs
        # lower bound case for finding slopes from z_min point to other points
        z_min_range = pzs[:, :, z_] - pzs[:, z_min, z_, np.newaxis] # slope in z-coordinate
        with np.errstate(divide='ignore', invalid='ignore'):
            pzs[:, :, ilx_] = z_min_range / (pzs[:, :, x_] - pzs[:, z_min, x_, np.newaxis]) # slope in x-coordinate
            pzs[:, :, ily_] = z_min_range / (pzs[:, :, y_] - pzs[:, z_min, y_, np.newaxis]) # slope in y-coordinate

        px = IdentityXIdentity.getX(pzs, l_)
        py = IdentityXIdentity.getY(pzs, l_)
        po = IdentityXIdentity.getO(pzs, z_min)
        
        gammaL = (pzs[:, z_min, z_] - px[:, z_]) / (pzs[:, z_min, x_] - px[:, x_])
        alphaL = (px[:, z_] - po[:, z_]) / (px[:, y_] - po[:, y_])

        muL = (pzs[:, z_min, z_] - py[:, z_]) / (pzs[:, z_min, y_] - py[:, y_])
        betaL = (py[:, z_] - po[:, z_]) / (py[:, x_] - po[:, x_])

        Lx = np.zeros((2, dim))
        Ly = np.zeros((2, dim))
        Lb = np.zeros((2, dim))

        z_zmin = z_sorted_i[:, z_min]
        # lower hyper-plane constraints on x-coordinate and y-coordinate
        map1 = np.where((z_zmin == 0) | (z_zmin == 1))[0]
        if len(map1):
            Lx[:, map1] = np.row_stack([gammaL[map1], betaL[map1]])
            Ly[:, map1] = np.row_stack([muL[map1], alphaL[map1]])
            Lb1 = Lx[0, map1] * pzs[map1, z_min, x_] + Ly[0, map1]*pzs[map1, z_min, y_] - pzs[map1, z_min, z_]
            Lb2 = Lx[1, map1] * pzs[map1, z_max, x_] + Ly[1, map1]*pzs[map1, z_max, y_] - pzs[map1, z_max, z_]
            Lb[:, map1] = np.vstack((Lb1, Lb2))

        map1 = np.where((z_zmin == 2) | (z_zmin == 3))[0]
        if len(map1):
            Lx[:, map1] = np.row_stack([gammaL[map1], betaL[map1]])
            Ly[:, map1] = np.row_stack([alphaL[map1], muL[map1]])
            Lb[:, map1] = Lx[:, map1]*pzs[map1, z_min, x_] + Ly[:, map1]*pzs[map1, z_min, y_] - pzs[map1, z_min, z_]

        return Ux, Uy, Ub, Lx, Ly, Lb, pzs[:, z_max, z_], pzs[:, z_min, z_]
    
    def multiStep_sparse(X, H, DR=0, RF=0.0, lp_solver='gurobi', pool=None):

        assert isinstance(X, SparseStar) and isinstance(H, SparseStar), 'error: \
        both input sets should be SparseStar sets'
        
        assert X.dim == H.dim, 'error: dimension of two input sets should be equivalent'
        N = X.dim

        xl, xu = X.getRanges(lp_solver=lp_solver, RF=RF)
        hl, hu = H.getRanges(lp_solver=lp_solver, RF=RF)

        map0 = np.where((xl != xu) & (hl != hu))[0]
        m = len(map0)

        nVars = X.nVars + H.nVars
        nZVars = X.nZVars + H.nZVars
        nIVars = X.nIVars + H.nIVars
        nv = nVars + m

        # X1 = np.hstack((X.X(), H.X()))
        X1 = np.hstack((H.c()*X.X(), X.c()*H.X()))
        c1 = X.c() * H.c()
        A1 = np.hstack((c1, X1))

        A0 = np.zeros((N, m))
        for i in range(m):
            A0[map0[i], i] = 1
        A1[map0, :] = 0
        A1 = np.hstack((A1, A0))

        if m != N:

            map1 = np.where((xl == xu) & (hl == hu))[0]
            m1 = len(map1)
            if m1 == N:
                if len(X.d):
                    new_A = IdentityXIdentity.f(xl, hl)[:, None]
                    return SparseStar(new_A, X.C, X.d, X.pred_lb, X.pred_ub, X.pred_depth)
                
                else:
                    new_A = IdentityXIdentity.f(xl, hl)[:, None]
                    return SparseStar(new_A, H.C, H.d, H.pred_lb, H.pred_ub, H.pred_depth)

                # if len(H.d) > 0:
                #     # consider X as a point
                #     # x_c = np.diag(0.5*(xl + xu))
                #     x_c = np.diag(xl)
                #     new_A = np.matmul(x_c, H.A)
                #     return SparseStar(new_A, H.C, H.d, H.pred_lb, H.pred_ub, H.pred_depth)

                # else:
                #     # cosider H as a point
                #     # h_c = np.diag(0.5*(hl + hu))
                #     h_c = np.diag(hl)
                #     new_A = np.matmul(h_c, X.A)
                #     return SparseStar(new_A, X.C, X.d, X.pred_lb, X.pred_ub, X.pred_depth)

            map2 = np.where((xl == xu) & (hl != hu))[0]
            m2 = len(map2)
            if m2 == N:
                # x_c = np.diag(0.5*(xl + xu))
                # new_A = np.matmul(x_c, H.A)
                new_A = xl[:, None] * H.A
                return SparseStar(new_A, H.C, H.d, H.pred_lb, H.pred_ub, H.pred_depth)

            map3 = np.where((xl != xu) & (hl == hu))[0]
            m3 = len(map3)
            if m3 == N:
                # h_c = np.diag(0.5*(hl + hu))
                # new_A = np.matmul(h_c, X.A)
                new_A = hl[:, None] * X.A
                return SparseStar(new_A, X.C, X.d, X.pred_lb, X.pred_ub, X.pred_depth)
            
            if m1:
                A1[map1, :] = 0
                A1[map1, 0] = xl[map1] * hl[map1]

            if m2: #(xl == xu) & (hl != hu)
                #consider x[i] as a point, where i \in map2
                A1[map2, 1:] = 0
                A1[map2, 0] = xl[map2] * H.c(map2).reshape(-1) 
                A1[map2, 1+X.nIVars:nIVars+1] = xl[map2, None] * H.X(map2)
            
            if m3: #(xl != xu) & (hl == hu)
                #consider h[i] as a point, where i \in map3
                # h_c = 0.5*(hl[map3] + hu[map3])
                A1[map3, 1:] = 0
                A1[map3, 0] = hl[map3] * X.c(map3).reshape(-1)
                A1[map3, 1:X.nIVars+1] = hl[map3, None] * X.X(map3)
            
        XC1 = X.C[:, 0:X.nZVars]
        XC2 = X.C[:, X.nZVars:X.nVars]

        HC1 = H.C[:, 0:H.nZVars]
        HC2 = H.C[:, H.nZVars:H.nVars]

        C1_ = sp.block_diag((XC1, HC1)).tocsc()
        C2_ = sp.block_diag((XC2, HC2)).tocsc()

        d1_ = np.concatenate([X.d[0:X.nZVars], H.d[0:H.nZVars]])
        d2_ = np.concatenate([X.d[X.nZVars:X.nVars], H.d[H.nZVars:H.nVars]])
        
        C0 = sp.hstack([C1_, C2_, sp.csc_matrix((C2_.shape[0], m))]).tocsc()
        d0 = np.concatenate([d1_, d2_])

        # if C1_.nnz > 0:
        #     C_ = sp.hstack([C1_, C2_]).tocsc()
        # else:
        #     C_ = C2_.tocsc()
        
        # C0 = sp.csc_matrix((0, nv))s
        # if C1_.nnz > 0:
        #     C1_ = sp.hstack([C1_, sp.csc_matrix((C1_.shape[0], m))])
        #     C0 = sp.vstack([C0, C1_]).tocsc()

        # if C2_.nnz > 0:
        #     C2_ = sp.hstack([C2_, sp.csc_matrix((C2_.shape[0], m))])
        #     C0 = sp.vstack([C0, C2_]).tocsc()

        
        if m:
            Z = sp.csc_matrix((m, nZVars))

            Ux, Uh, Ub, Lx, Lh, Lb, zmax, zmin = IdentityXIdentity.getMultiConstraints(xl[map0], xu[map0], hl[map0], hu[map0])

            #1. z <= ux1 * x + uy1 * y + ub1
            C11 = sp.hstack((Z, -Ux[0, :, np.newaxis]*X.X(map0), -Uh[0, :, np.newaxis]*H.X(map0), A0[map0]))
            d11 = Ux[0, :, np.newaxis]*X.c(map0) + Uh[0, :, np.newaxis]*H.c(map0) - Ub[0, :, np.newaxis]

            #2. z <= ux2 * x + uy2 * y + ub2
            C12 = sp.hstack((Z, -Ux[1, :, np.newaxis]*X.X(map0), -Uh[1, :, np.newaxis]*H.X(map0), A0[map0]))
            d12 = Ux[1, :, np.newaxis]*X.c(map0) + Uh[1, :, np.newaxis]*H.c(map0) - Ub[1, :, np.newaxis]

            #3. z >= lx1 * x + ly1 * y + lb1
            C13 = sp.hstack((Z, Lx[0, :, np.newaxis]*X.X(map0), Lh[0, :, np.newaxis]*H.X(map0), -A0[map0]))
            d13 = -Lx[0, : ,np.newaxis]*X.c(map0) - Lh[0, : ,np.newaxis]*H.c(map0) + Lb[0, : ,np.newaxis]

            #4. z >= lx2 * x + ly2 * y + lb2
            C14 = sp.hstack((Z, Lx[1, :, np.newaxis]*X.X(map0), Lh[1, :, np.newaxis]*H.X(map0), -A0[map0]))
            d14 = -Lx[1, :, np.newaxis]*X.c(map0) - Lh[1, :, np.newaxis]*H.c(map0) + Lb[1, :, np.newaxis]

            C1 = sp.vstack((C11, C12, C13, C14)).tocsc()
            d1 = np.vstack((d11, d12, d13, d14)).reshape(-1)
        else:
            C1 = sp.csc_matrix((0, nv))
            d1 = np.empty((0))
            zmin = np.empty((0))
            zmax = np.empty((0))
        
        new_A = A1
        new_C = sp.vstack((C0, C1))
        new_d = np.hstack((d0, d1))

        new_pred_lb =    np.hstack((X.pred_lb[0:X.nZVars],          H.pred_lb[0:H.nZVars],
                                    X.pred_lb[X.nZVars:X.nVars],    H.pred_lb[H.nZVars:H.nVars], zmin))
        new_pred_ub =    np.hstack((X.pred_ub[0:X.nZVars],          H.pred_ub[0:H.nZVars],
                                    X.pred_ub[X.nZVars:X.nVars],    H.pred_ub[H.nZVars:H.nVars], zmax))
        new_pred_depth = np.hstack((X.pred_depth[0:X.nZVars],       H.pred_depth[0:H.nZVars],
                                    X.pred_depth[X.nZVars:X.nVars], H.pred_depth[H.nZVars:H.nVars])) + 1
        new_pred_depth = np.hstack([new_pred_depth, np.zeros(m)])

        S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
        if DR > 0:
            S = S.depthReduction(DR)
        return S

    # def multiStep_sparse(I, X, H, DR=0, RF=0.0, lp_solver='gurobi', pool=None):

    #     assert isinstance(I, SparseStar) and isinstance(X, SparseStar) and isinstance(H, SparseStar), 'error: \
    #     both input sets should be SparseStar sets'

    #     N = I.dim

    #     xl, xu = X.getRanges(lp_solver=lp_solver, RF=RF)
    #     hl, hu = H.getRanges(lp_solver=lp_solver, RF=RF)

    #     ## l != u
    #     map0 = np.where((xl != xu) & (hl != hu))[0]
    #     m = len(map0)
    #     A0 = np.zeros((N, m))
    #     for i in range(m):
    #         A0[map0[i], i] = 1
    #     new_A = np.hstack((np.zeros((N, 1)), A0))

    #     map1 = np.where((xl == xu) & (hl == hu))[0]
    #     if len(map1):
    #         zl = IdentityXIdentity.f(xl[map1], hl[map1]).reshape(-1)
    #         new_A[map1, 0] = zl
    #         new_A[map1, 1:m+1] = 0

    #     map1 = np.where((xl == xu) & (hl != hu))[0]
    #     if len(map1):
    #         new_A[map1, 0] = xl[map1] * H.c(map1).reshape(-1)
    #         new_A[map1, 1:m+1] = xl[map1] * I.X(map1)

    #     map1 = np.where((xl != xu) & (hl == hu))[0]
    #     if len(map1):
    #         new_A[map1, 0] = hl[map1] * X.c(map1).reshape(-1)
    #         new_A[map1, 1:m+1] = hl[map1] * I.X(map1)

    #     nv = I.nVars + m
    #     if len(map0):
    #         Z = sp.csc_matrix((len(map0), I.nZVars))

    #         Ux, Uh, Ub, Lx, Lh, Lb, zmax, zmin = IdentityXIdentity.getMultiConstraints(xl[map0], xu[map0], hl[map0], hu[map0])

    #         #1. z <= ux1 * x + uy1 * y + ub1
    #         C11 = sp.hstack((Z, -Ux[0, :, np.newaxis]*X.X(map0), -Uh[0, :, np.newaxis]*H.X(map0), A0[map0]))
    #         d11 = Ux[0, :, np.newaxis]*X.c(map0) + Uh[0, :, np.newaxis]*H.c(map0) - Ub[0, :, np.newaxis]

    #         #2. z <= ux2 * x + uy2 * y + ub2
    #         C12 = sp.hstack((Z, -Ux[1, :, np.newaxis]*X.X(map0), -Uh[1, :, np.newaxis]*H.X(map0), A0[map0]))
    #         d12 = Ux[1, :, np.newaxis]*X.c(map0) + Uh[1, :, np.newaxis]*H.c(map0) - Ub[1, :, np.newaxis]

    #         #3. z >= lx1 * x + ly1 * y + lb1
    #         C13 = sp.hstack((Z, Lx[0, :, np.newaxis]*X.X(map0), Lh[0, :, np.newaxis]*H.X(map0), -A0[map0]))
    #         d13 = -Lx[0, : ,np.newaxis]*X.c(map0) - Lh[0, : ,np.newaxis]*H.c(map0) + Lb[0, : ,np.newaxis]

    #         #4. z >= lx2 * x + ly2 * y + lb2
    #         C14 = sp.hstack((Z, Lx[1, :, np.newaxis]*X.X(map0), Lh[1, :, np.newaxis]*H.X(map0), -A0[map0]))
    #         d14 = -Lx[1, :, np.newaxis]*X.c(map0) - Lh[1, :, np.newaxis]*H.c(map0) + Lb[1, :, np.newaxis]

    #         C1 = sp.vstack((C11, C12, C13, C14)).tocsc()
    #         d1 = np.vstack((d11, d12, d13, d14)).reshape(-1)
    #     else:
    #         C1 = sp.csc_matrix((0, nv))
    #         d1 = np.empty((0))
    #         Zl = np.empty((0))
    #         Zu = np.empty((0))
        
    #     n = I.C.shape[0]
    #     if len(I.d):
    #         C0 = sp.hstack((I.C, sp.csc_matrix((n, m)))) 
    #         d0 = I.d
    #     else:
    #         C0 = sp.csc_matrix((0, I.nVars+m))
    #         d0 = np.empty((0))
            
    #     new_C = sp.vstack((C0, C1))
    #     new_d = np.hstack((d0, d1))

    #     new_pred_lb = np.hstack((I.pred_lb, zmin))
    #     new_pred_ub = np.hstack((I.pred_ub, zmax))
    #     new_pred_depth = np.hstack((I.pred_depth+1, np.zeros(m)))
        
    #     S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
    #     if DR > 0:
    #         S = S.depthReduction(DR)
    #     return S

    def reachApprox_sparse(X, H, lp_solver='gurobi', RF=0.0, DR=0):
        assert isinstance(X, SparseStar) and isinstance(H, SparseStar), 'error: both or one of input sets are not SparseStar sets'
        assert X.dim == H.dim, 'error: dimension of input sets does not match'

        return IdentityXIdentity.multiStep_sparse(X, H, lp_solver=lp_solver, RF=RF, DR=DR)

    def reach(X, H, lp_solver='gurobi', pool=None, RF=0.0, DR=0):
        if isinstance(X, SparseStar) and isinstance(H, SparseStar):
            return IdentityXIdentity.reachApprox_sparse(X, H, lp_solver=lp_solver, RF=RF, DR=DR)
        elif isinstance(X, Star) and isinstance(H, Star):
            # return IdentityXIdentity.reachApproxStar(X, H, lp_solver RF, DR)
            raise Exception('error: under development')
        else:
            raise Exception('error: both input sets (X, H) should be SparseStar or Star')