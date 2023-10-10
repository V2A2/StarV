"""
IdentityXIdentityOLD Class (x * y, where x \in X, y \in Y, X and Y are star sets)
Sung Woo Choi, 04/11/2023

"""

# !/usr/bin/python3
import copy
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.linalg import block_diag
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star

class IdentityXIdentityOLD(object):
    """
    IdentityXIdentityOLD Class for reachability
    Author: Sung Woo Choi
    Date: 04/11/2023

    """
    u_ = 0 # index for upper bound case
    l_ = 1 # index for lower bound case

    x_ = 0 # index for x-coordinate constraint
    y_ = 1 # index for y-coordinate constraint
    z_ = 2 # index for z-coordinate constraint

    dzx_ = 3 # grandient on x-coordinate
    dzy_ = 4 # grandient on y-coordinate

    iux_ = 5 # intersection line on x-coordinate for upper bound case
    iuy_ = 6 # intersection line on y-coordinate for upper bound case

    ilx_ = 7 # intersection line on x-coordinate for lower bound case
    ily_ = 8 # intersection line on y-coordinate for lower bound case
    
    num_of_points = 4

    z_max = num_of_points-1   
    z_min = 0 

    @staticmethod
    def f(x, y):
        return x * y
    
    @staticmethod
    def gf(x, y):
        """Gradient of x*y"""
        return x, y
    
    @staticmethod
    def getX(p, bound):
        if bound == IdentityXIdentityOLD.u_:
            for i in range(IdentityXIdentityOLD.z_max-1, IdentityXIdentityOLD.z_min-1, -1):
                if p[i, IdentityXIdentityOLD.iuy_] == np.inf:
                    return copy.deepcopy(p[i, IdentityXIdentityOLD.x_ : IdentityXIdentityOLD.dzy_+1])
        elif bound == IdentityXIdentityOLD.l_:
            for i in range(IdentityXIdentityOLD.z_min+1, IdentityXIdentityOLD.z_max+1):
                if p[i, IdentityXIdentityOLD.ily_] == np.inf:
                    return copy.deepcopy(p[i, IdentityXIdentityOLD.x_ : IdentityXIdentityOLD.dzy_+1])
        else:
            raise Exception('error: unknown bound; should be u_ or l_')

    @staticmethod
    def getY(p, bound):
        if bound == IdentityXIdentityOLD.u_:
            for i in range(IdentityXIdentityOLD.z_max-1, IdentityXIdentityOLD.z_min-1, -1):
                if p[i, IdentityXIdentityOLD.iux_] == np.inf:
                    return copy.deepcopy(p[i, IdentityXIdentityOLD.x_ : IdentityXIdentityOLD.dzy_+1])
        elif bound == IdentityXIdentityOLD.l_:
            for i in range(IdentityXIdentityOLD.z_min+1, IdentityXIdentityOLD.z_max+1):
                if p[i, IdentityXIdentityOLD.ilx_] == np.inf:
                    return copy.deepcopy(p[i, IdentityXIdentityOLD.x_ : IdentityXIdentityOLD.dzy_+1])
        else:
            raise Exception('error: unknown bound; should be u_ or l_')
        
    @staticmethod
    def getO(p, index, rtype='index'):
        # get opposite point
        for i in range(IdentityXIdentityOLD.num_of_points):
            if p[i, IdentityXIdentityOLD.x_] != p[index, IdentityXIdentityOLD.x_] and p[i, IdentityXIdentityOLD.y_] != p[index, IdentityXIdentityOLD.y_]:
                if rtype == 'index':
                    return copy.deepcopy(i)
                elif rtype == 'row':
                    return copy.deepcopy(p[i, :])
    
    @staticmethod
    def getConstraints(xl, xu, yl, yu):
        
        assert isinstance(xl, float), 'error: \
        lower bound, xl, should be a floating point number'
        assert isinstance(xu, float), 'error: \
        upper bound, xu, should be a floating point number'
        assert isinstance(yl, float), 'error: \
        lower bound, yl, should be a floating point number'
        assert isinstance(yu, float), 'error: \
        upper bound, yu, should be a floating point number'

        u_ = IdentityXIdentityOLD.u_
        l_ = IdentityXIdentityOLD.l_
    
        x_ = IdentityXIdentityOLD.x_
        y_ = IdentityXIdentityOLD.y_
        z_ = IdentityXIdentityOLD.z_
        dzx_ = IdentityXIdentityOLD.dzx_
        dzy_ = IdentityXIdentityOLD.dzy_
        iux_ = IdentityXIdentityOLD.iux_
        iuy_ = IdentityXIdentityOLD.iuy_
        ilx_ = IdentityXIdentityOLD.ilx_
        ily_ = IdentityXIdentityOLD.ily_
        
        z_max = IdentityXIdentityOLD.z_max
        z_min = IdentityXIdentityOLD.z_min
    
        num_of_points = IdentityXIdentityOLD.num_of_points

        pz = np.zeros((num_of_points, num_of_points+1))

        pz[0, x_:y_+1] = [xl, yl]
        pz[1, x_:y_+1] = [xu, yu]
        pz[2, x_:y_+1] = [xu, yl]
        pz[3, x_:y_+1] = [xl, yu]

        pz[:,  z_] = IdentityXIdentityOLD.f(pz[:, x_], pz[:,y_])
        pz[:, dzx_:dzy_+1] = np.column_stack((IdentityXIdentityOLD.gf(pz[:, x_], pz[:, y_])))

        # sort z-coordinate points in ascending order
        z_sorted_i = np.argsort(pz[:, z_])
        pzs = np.hstack([pz[z_sorted_i, :], np.full((num_of_points, num_of_points), np.nan)])
        print('pzs: \n', pzs)

        tan_i = 3  # index for tangent line
        #####################################################################
        #                       Upper bound case
        #####################################################################

        max2min_i = np.nan
        # upper bound case for finding slopes from global maximum to other points
        # for i in range(z_max-1, z_min-1, -1):
        for i in range(z_min, z_max):
            z_max_range = pzs[z_max, z_] - pzs[i, z_] # slope in z-coordinate
            x_max_range = pzs[z_max, x_] - pzs[i, x_]
            y_max_range = pzs[z_max, y_] - pzs[i, y_]
            pzs[i, iux_] = z_max_range / x_max_range if x_max_range else np.inf # slope in x-coordinate
            pzs[i, iuy_] = z_max_range / y_max_range if y_max_range else np.inf # slope in y-coordinate

            # pzs[i, iux_] = z_max_range / x_max_range # slope in x-coordinate
            # pzs[i, iuy_] = z_max_range / y_max_range # slope in y-coordinate
            if x_max_range != 0 and y_max_range != 0:
                pzs[i, iux_] /= 2.0
                pzs[i, iuy_] /= 2.0
                max2min_i = i # index for intersection between z_max and z_min

        # finding minimal slopes among tangent lines and intersection lines
        ux = abs(np.hstack((pzs[z_min:z_max, iux_], pzs[z_max, dzx_])))
        uy = abs(np.hstack((pzs[z_min:z_max, iuy_], pzs[z_max, dzy_])))
        us = [ux, uy] # upper bound slope [x-; y-] coordinates
        us_i = [np.argmin(ux), np.argmin(uy)]
        
        if (us[x_][us_i[x_]] == abs(pzs[max2min_i, iux_])) & (us[y_][us_i[y_]] < abs(pzs[max2min_i, iuy_])):
            us[x_][us_i[x_]] *= 2.0
            us_i[x_] = np.argmin(us[x_])

            # In x-axis, if optimal slope is not tangent line
            if us_i[x_] != tan_i:
                print('U case 1 (a) --------------------------------------\n')
                px = IdentityXIdentityOLD.getX(pzs, u_)
                px[z_] = pzs[z_max, z_] - us[x_][us_i[x_]] * (xu - xl)
                po = pzs[max2min_i, x_:z_+1]

                px_inter_y = [(px[z_] - po[z_]) / (yu - yl)]
                us[y_] = abs(np.hstack((us[y_][us_i[y_]], px_inter_y, px[dzy_])))
                us_i[y_] = np.argmin(us[y_])
            else:
                print('U case 1 (b) --------------------------------------\n')
                py = IdentityXIdentityOLD.getY(pzs, u_)
                po = pzs[max2min_i, x_:z_+1]
                new_z = pzs[z_max, z_] - us[x_][us_i[x_]] * (xu -xl) - us[y_][us_i[y_]] * (yu - yl)

                if new_z < po[z_]:
                    z_range = po[z_] - pzs[z_max, z_]
                    x_slope = -(z_range + us[y_][us_i[y_]]*(yu - yl)) / (xu - xl)
                    us[x_] = abs(np.hstack((us[x_][us_i[x_]], x_slope, py[dzx_])))
                    us_i[x_] = np.argmin(us[x_])
                else:
                    us[x_] = abs(np.hstack((us[x_][us_i[x_]], py[dzx_])))
                    us_i[x_] = np.argmin(us[x_])

        elif (us[x_][us_i[x_]] < abs(pzs[max2min_i, iux_])) & (us[y_][us_i[y_]] == abs(pzs[max2min_i, iuy_])):
            us[y_][us_i[y_]] *= 2.0
            us_i[y_] = np.argmin(us[y_])

            # In y-axis, if optimal slope is not tangent line
            if us_i[y_] != tan_i:
                print('U case 2 (a) -------------------------------------- +\n')
                py = IdentityXIdentityOLD.getY(pzs, u_)
                py[z_] = pzs[z_max, z_] - us[y_][us_i[y_]] * (yu - yl)
                po = pzs[max2min_i, x_:z_+1]

                py_inter_x = [(py[z_] - po[z_]) / (xu - xl)]
                us[x_] = abs(np.hstack((us[x_][us_i[x_]], py_inter_x, py[dzx_])))
                us_i[x_] = np.argmin(us[x_])
            
            else:
                print('U case 2 (b) -------------------------------------- +\n')
                px = IdentityXIdentityOLD.getX(pzs, u_)
                po = pzs[max2min_i, x_:z_+1]
                new_z = pzs[z_max, z_] - us[x_][us_i[x_]] * (xu - xl) - us[y_][us_i[y_]] * (yu - yl)

                if new_z < po[z_]:
                    z_range = po[z_] - pzs[z_max, z_]
                    y_slope = -(z_range + us[x_][us_i[x_]]*(xu - xl)) / (yu - yl)
                    us[y_] = abs(np.hstack((us[y_][us_i[y_]], y_slope, px[dzy_])))
                    us_i[y_] = np.argmin(us[y_])

                else:
                    us[y_] = abs(np.hstack((us[y_][us_i[y_]], px[dzy_])))
                    us_i[y_] = np.argmin(us[y_])

        elif (us_i[x_] != tan_i) & (us_i[x_] != max2min_i):
            print('U case 3 ------------------------------------------\n')
            px = IdentityXIdentityOLD.getX(pzs, u_)
            us[y_] = abs(np.hstack((us[y_][us_i[y_]], px[dzy_])))
            us_i[y_] = np.argmin(us[y_])

        elif (us_i[y_] != tan_i) & (us_i[y_] != max2min_i):
            print('U case 4 ------------------------------------------ +\n')
            py = IdentityXIdentityOLD.getY(pzs, u_)
            us[x_] = abs(np.hstack((us[x_][us_i[x_]], py[dzx_])))
            us_i[x_] = np.argmin(us[x_])
        else:
            print('U case 5 ------------------------------------------ \n')

        #####################################################################
        #                       Lower bound case
        #####################################################################
        min2max_i = np.nan
        # combine lower bound pzs with upper bound pzs
        # lower bound case for finding slopes from global minimum to other points
        for i in range(z_min+1, z_max+1):
#            print('i: ', i)
            z_min_range = pzs[i, z_] - pzs[z_min, z_] # slope in z-coordinate
            x_min_range = pzs[i, x_] - pzs[z_min, x_]
            y_min_range = pzs[i, y_] - pzs[z_min, y_]
            pzs[i, ilx_] = z_min_range / x_min_range if x_min_range else np.inf # slope in x-coordinate
            pzs[i, ily_] = z_min_range / y_min_range if y_min_range else np.inf # slope in y-coordinate
            if x_min_range != 0 and y_min_range != 0:
                pzs[i, ilx_] /= 2.0
                pzs[i, ily_] /= 2.0
                min2max_i = i

        # finding minimal slopes among tangent lines and intersection lines
        lx = abs(np.hstack((pzs[z_min+1:z_max+1, ilx_], pzs[z_min, dzx_])))
        ly = abs(np.hstack((pzs[z_min+1:z_max+1, ily_], pzs[z_min, dzy_])))
        ls = [lx, ly] # lower bound slope [x-; y-] coordinates
        ls_i = [np.argmin(lx), np.argmin(ly)]

        if (abs(ls[x_][ls_i[x_]]) == abs(pzs[min2max_i, ilx_])) & (abs(ls[y_][ls_i[y_]]) < abs(pzs[min2max_i, ily_])):
            ls[x_][ls_i[x_]] *= 2.0
            ls_i[x_] = np.argmin(ls[x_])

            # In x-axis, if optimal slope is not tangent line
            if ls_i[x_] != tan_i:
                print('L case 1 (a) --------------------------------------\n')
                px = IdentityXIdentityOLD.getX(pzs, l_)
                px[z_] = pzs[z_min, z_] + ls[x_][ls_i[x_]] * (xu - xl)
                po = pzs[min2max_i, x_:z_+1]

                px_inter_y = [(px[z_] - po[z_]) / (yu - yl)]
                ls[y_] = abs(np.hstack((ls[y_][ls_i[y_]], px_inter_y, px[dzy_])))
                ls_i[y_] = np.argmin(ls[y_])
            
            else:
                print('L case 1 (b) --------------------------------------\n')
                py = IdentityXIdentityOLD.getY(pzs, l_)
                po = pzs[min2max_i, x_:z_+1]
                new_z = pzs[z_min, z_] + ls[x_][ls_i[x_]] * (xu - xl) + ls[y_][ls_i[y_]] * (yu - yl)

                if new_z > po[z_]:
                    z_range = po[z_] - pzs[z_min, z_]
                    x_slope = -(z_range + ls[y_][ls_i[y_]] * (yu - yl)) / (xu - xl)
                    ls[x_] = abs(np.hstack((ls[x_][ls_i[x_]], x_slope, py[dzx_])))
                    ls_i[x_] = np.argmin(ls[x_])

                else:
                    ls[x_] = abs(np.hstack((ls[x_][ls_i[x_]], py[dzx_])))
                    ls_i[x_] = np.argmin(ls[x_])

        elif (abs(ls[x_][ls_i[x_]]) < abs(pzs[min2max_i, ilx_])) & (abs(ls[y_][ls_i[y_]]) == abs(pzs[min2max_i, ily_])):
            ls[y_][ls_i[y_]] *= 2.0
            ls_i[y_] = np.argmin(ls[y_])
                                 
            # In y-axis, if optimal solution is not tangent line
            if ls_i[y_] != tan_i:
                print('L case 2 (a) -------------------------------------- +\n')
                py = IdentityXIdentityOLD.getY(pzs, l_)
                py[z_] = pzs[z_min, z_] + ls[y_][ls_i[y_]] * (yu - yl)
                po = pzs[min2max_i, x_:z_+1]

                py_inter_x = (py[z_] - po[z_]) / (xu - xl)
                ls[x_] = abs(np.hstack((ls[x_][ls_i[x_]], py_inter_x, py[dzx_])))
                ls_i[x_] = np.argmin(ls[x_])
            else:
                print('L case 2 (b) -------------------------------------- +\n')
                px = IdentityXIdentityOLD.getX(pzs, l_)
                po = pzs[min2max_i, x_:z_+1]
                new_z = pzs[z_min, z_] + ls[x_][ls_i[x_]] * (xu -xl) + ls[y_][ls_i[y_]] * (yu - yl)

                if new_z > po[z_]:
                    z_range = po[z_] - pzs[z_min, z_]
                    y_slope = -(z_range + ls[x_][ls_i[x_]] * (xu - xl)) / (yu - yl)
                    ls[y_] = abs(np.hstack((ls[y_][ls_i[y_]], y_slope, px[dzy_])))
                    ls_i[y_] = np.argmin(ls[y_])

                else:
                    ls[y_] = abs(np.hstack((ls[y_][ls_i[y_]], px[dzy_])))
                    ls_i[y_] = np.argmin(ls[y_])

        elif (ls_i[x_] != tan_i) & (ls_i[x_] != min2max_i):
            print('L case 3 ------------------------------------------ +\n')
            ## add two tight hyper-plane constraints instead of one hyper-plane constraint
            px = IdentityXIdentityOLD.getX(pzs, l_)
            ls[y_] = abs(np.hstack((ls[y_][ls_i[y_]], px[dzy_])))
            ls_i[y_] = np.argmin(ls[y_])

        elif (ls_i[y_] != tan_i) & (ls_i[y_] != min2max_i):
            print('L case 4 ------------------------------------------\n')
            py = IdentityXIdentityOLD.getY(pzs, l_)
            ls[x_] = abs(np.hstack((ls[x_][ls_i[x_]], py[dzx_])))
            ls_i[x_] = np.argmin(ls[x_])
        else:
            print('L case 5 ------------------------------------------ \n')

        # upper linear constraints on x-coordinate and y-coordinate
        if z_sorted_i[z_max] == 0:
            Ux = us[x_][us_i[x_]]
            Uy = us[y_][us_i[y_]]
        elif z_sorted_i[z_max] == 1:
            Ux = -us[x_][us_i[x_]]
            Uy = -us[y_][us_i[y_]]
        elif z_sorted_i[z_max] == 2:
            Ux = -us[x_][us_i[x_]]
            Uy = us[y_][us_i[y_]]
        elif z_sorted_i[z_max] == 3:
            Ux = us[x_][us_i[x_]]
            Uy = -us[y_][us_i[y_]]
        else:
            raise Exception('error: unknown point')
        Ub = pzs[z_max, z_] + Ux*pzs[z_max, x_] + Uy*pzs[z_max, y_]
    
        # lower linear constraints on x-coordinate and y-coordinate
        if z_sorted_i[z_min] == 0:
            Lx = -ls[x_][ls_i[x_]]
            Ly = -ls[y_][ls_i[y_]]
        elif z_sorted_i[z_min] == 1:
            Lx = ls[x_][ls_i[x_]]
            Ly = ls[y_][ls_i[y_]]
        elif z_sorted_i[z_min] == 2:
            Lx = ls[x_][ls_i[x_]]
            Ly = -ls[y_][ls_i[y_]]
        elif z_sorted_i[z_min] == 3:
            Lx = -ls[x_][ls_i[x_]]
            Ly = ls[y_][ls_i[y_]]
        else:
            raise Exception('error: unknown point')
        Lb = pzs[z_min, z_] + Lx*pzs[z_min, x_] + Ly*pzs[z_min, y_]

        zmax = pzs[z_max, z_]
        zmin = pzs[z_min, z_]

        return Ux, Uy, Ub, Lx, Ly, Lb, zmax, zmin

    def multiStepIdentityXIdentity_sparse(I, X, H, DR=0, RF=0.0, lp_solver='gurobi', pool=None):

        assert isinstance(I, SparseStar) and isinstance(X, SparseStar) and isinstance(H, SparseStar), 'error: \
        both input sets should be SparseStar sets'

        N = I.dim

        xl, xu = X.getRanges(lp_solver=lp_solver, RF=RF)
        hl, hu = H.getRanges(lp_solver=lp_solver, RF=RF)
        xl = xl.reshape(N, 1)
        xu = xu.reshape(N, 1)
        hl = hl.reshape(N, 1)
        hu = hu.reshape(N, 1)

        ## l != u
        map0 = np.where((xl != xu) & (hl != hu))[0]
        m = len(map0)
        A0 = np.zeros((N, m))
        for i in range(m):
            A0[map0[i], i] = 1
        new_A = np.hstack((np.zeros((N, 1)), A0))

        map1 = np.where((xl == xu) & (hl == hu))[0]
        if len(map1):
            zl = IdentityXIdentityOLD.f(xl[map1], hl[map1])
            new_A[map1, 0] = zl
            new_A[map1, 1:m+1] = 0

        map1 = np.where((xl == xu) & (hl != hu))[0]
        if len(map1):
            new_A[map1, 0] = xl[map1] * H.c(map1)
            new_A[map1, 1:m+1] = xl[map1] * I.X(map1)

        map1 = np.where((xl != xu) & (hl == hu))[0]
        if len(map1):
            new_A[map1, 0] = hl[map1] * X.c(map1)
            new_A[map1, 1:m+1] = hl[map1] * I.X(map1)

        nv = I.nVars + m
        if len(map0):
            Z = sp.csc_matrix((len(map0), I.nZVars))

            # Ux, Uh, Ub = np.zeros((m, 1)), np.zeros((m, 1)), np.zeros((m, 1))
            # Lx, Lh, Lb = np.zeros((m, 1)), np.zeros((m, 1)), np.zeros((m, 1))
            # Zl, Zu = np.zeros((m, 1)), np.zeros((m, 1))

            Ux, Uy, Ub, Lx, Ly, Lb, Zu, Zl = [np.zeros((m, 1)) for _ in range(8)]
            
            for i in range(m):
                Ux[i], Uh[i], Ub[i], Lx[i], Lh[i], Lb[i], Zu[i], Zl[i] = IdentityXIdentityOLD.getConstraints(xl[map0[i]].item(), xu[map0[i]].item(), hl[map0[i]].item(), hu[map0[i]].item())

            C11 = sp.hstack((Z, -Lx*X.X(map0), -Lh*H.X(map0), -A0[map0]))
            d11 = Lx*X.c(map0) + Lh*H.c(map0) - Lb

            C12 = sp.hstack((Z, Ux*X.X(map0), Uh*H.X(map0), A0[map0]))
            d12 = -(Ux*X.c(map0) + Uh*H.c(map0)) + Ub

            C1 = sp.vstack((C11, C12)).tocsc()
            d1 = np.vstack((d11, d12)).flatten()
        else:
            C1 = sp.csc_matrix((0, nv))
            d1 = np.empty((0))
            Zl = np.empty((0))
            Zu = np.empty((0))
        
        n = I.C.shape[0]
        if len(I.d):
            C0 = sp.hstack((I.C, sp.csc_matrix((n, m)))) 
            d0 = I.d
        else:
            C0 = sp.csc_matrix((0, I.nVars+m))
            d0 = np.empty((0))
            
        new_C = sp.vstack((C0, C1))
        new_d = np.hstack((d0, d1))

        new_pred_lb = np.hstack((I.pred_lb, Zl.flatten()))
        new_pred_ub = np.hstack((I.pred_ub, Zu.flatten()))
        new_pred_depth = np.hstack((I.pred_depth+1, np.zeros(m)))
        
        S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
        if DR > 0:
            S = S.depthReduction(DR)
        return S

    def reachApproxSparse(X, H, lp_solver='gurobi', RF=0.0, DR=0):
        assert isinstance(X, SparseStar) and isinstance(H, SparseStar), 'error: both or one of input sets are not SparseStar sets'
        assert X.dim == H.dim, 'error: dimension of input sets does not match'

        n = X.dim
        S = X.minKowskiSum(H)
        return IdentityXIdentityOLD.multiStepIdentityXIdentity_sparse(S, X, H, lp_solver=lp_solver, RF=RF, DR=DR)

    def reach(X, H, lp_solver='gurobi', pool=None, RF=0.0, DR=0):
        if isinstance(X, SparseStar) and isinstance(H, SparseStar):
            return IdentityXIdentityOLD.reachApproxSparse(X, H, lp_solver=lp_solver, RF=RF, DR=DR)
        elif isinstance(X, Star) and isinstance(H, Star):
            # return IdentityXIdentityOLD.reachApproxStar(X, H, lp_solver RF, DR)
            raise Exception('error: under development')
        else:
            raise Exception('error: both input sets (X, H) should be SparseStar or Star')
        



    

    def get4Constraints(xl, xu, yl, yu):
        
        assert isinstance(xl, float), 'error: \
        lower bound, xl, should be a floating point number'
        assert isinstance(xu, float), 'error: \
        upper bound, xu, should be a floating point number'
        assert isinstance(yl, float), 'error: \
        lower bound, yl, should be a floating point number'
        assert isinstance(yu, float), 'error: \
        upper bound, yu, should be a floating point number'

        print('pz(1): [x, \t y, \t z1_min, \t dz1/dx, \t dz1/dy \t intl_x, \t intl_y]\n')
        print('pz(2): [x, \t y, \t z2, \t\t dz2/dx, \t dz1/dy \t intl_x, \t intl_y]\n')
        print('pz(3): [x, \t y, \t z3, \t\t dz3/dx, \t dz1/dy \t intl_x, \t intl_y]\n')
        print('pz(4): [x, \t y, \t z4_max, \t dz4/dx, \t dz1/dy \t intl_x, \t intl_y]\n')

        u_ = IdentityXIdentityOLD.u_
        l_ = IdentityXIdentityOLD.l_
    
        x_ = IdentityXIdentityOLD.x_
        y_ = IdentityXIdentityOLD.y_
        z_ = IdentityXIdentityOLD.z_
        dzx_ = IdentityXIdentityOLD.dzx_
        dzy_ = IdentityXIdentityOLD.dzy_
        iux_ = IdentityXIdentityOLD.iux_
        iuy_ = IdentityXIdentityOLD.iuy_
        ilx_ = IdentityXIdentityOLD.ilx_
        ily_ = IdentityXIdentityOLD.ily_
        
        z_max = IdentityXIdentityOLD.z_max
        z_min = IdentityXIdentityOLD.z_min
    
        num_of_points = IdentityXIdentityOLD.num_of_points

        pz = np.zeros((num_of_points, num_of_points+1))

        pz[0, x_:y_+1] = [xl, yl]
        pz[1, x_:y_+1] = [xu, yu]
        pz[2, x_:y_+1] = [xu, yl]
        pz[3, x_:y_+1] = [xl, yu]

        pz[:,  z_] = IdentityXIdentityOLD.f(pz[:, x_], pz[:,y_])
        pz[:, dzx_:dzy_+1] = np.column_stack((IdentityXIdentityOLD.gf(pz[:, x_], pz[:, y_])))

        # sort z-coordinate points in ascending order
        z_sorted_i = np.argsort(pz[:, z_])
        pzs = np.hstack([pz[z_sorted_i, :], np.full((num_of_points, num_of_points), np.nan)])
        print('pzs: \n', pzs)

        tan_i = 3  # index for tangent line
        #####################################################################
        #                       Upper bound case
        #####################################################################

        max2min_i = np.nan
        # upper bound case for finding slopes from global maximum to other points
        # for i in range(z_max-1, z_min-1, -1):
        for i in range(z_min, z_max):
            z_max_range = pzs[z_max, z_] - pzs[i, z_] # slope in z-coordinate
            x_max_range = pzs[z_max, x_] - pzs[i, x_]
            y_max_range = pzs[z_max, y_] - pzs[i, y_]
            pzs[i, iux_] = z_max_range / x_max_range if x_max_range else np.inf # slope in x-coordinate
            pzs[i, iuy_] = z_max_range / y_max_range if y_max_range else np.inf # slope in y-coordinate

            # pzs[i, iux_] = z_max_range / x_max_range # slope in x-coordinate
            # pzs[i, iuy_] = z_max_range / y_max_range # slope in y-coordinate
            if x_max_range != 0 and y_max_range != 0:
                pzs[i, iux_] /= 2.0
                pzs[i, iuy_] /= 2.0
                max2min_i = i # index for intersection between z_max and z_min

        # finding minimal slopes among tangent lines and intersection lines
        ux = abs(np.hstack((pzs[z_min:z_max, iux_], pzs[z_max, dzx_])))
        uy = abs(np.hstack((pzs[z_min:z_max, iuy_], pzs[z_max, dzy_])))
        us = [ux, uy] # upper bound slope [x-; y-] coordinates
        us_i = [np.argmin(ux), np.argmin(uy)]
        
        if (us[x_][us_i[x_]] == abs(pzs[max2min_i, iux_])) & (us[y_][us_i[y_]] < abs(pzs[max2min_i, iuy_])):
            us[x_][us_i[x_]] *= 2.0
            us_i[x_] = np.argmin(us[x_])

            # In x-axis, if optimal slope is not tangent line
            if us_i[x_] != tan_i:
                print('U case 1 (a) --------------------------------------\n')
                px = IdentityXIdentityOLD.getX(pzs, u_)
                px[z_] = pzs[z_max, z_] - us[x_][us_i[x_]] * (xu - xl)
                po = pzs[max2min_i, x_:z_+1]

                px_inter_y = [(px[z_] - po[z_]) / (yu - yl)]
                us[y_] = abs(np.hstack((us[y_][us_i[y_]], px_inter_y, px[dzy_])))
                us_i[y_] = np.argmin(us[y_])
            else:
                print('U case 1 (b) --------------------------------------\n')
                py = IdentityXIdentityOLD.getY(pzs, u_)
                po = pzs[max2min_i, x_:z_+1]
                new_z = pzs[z_max, z_] - us[x_][us_i[x_]] * (xu -xl) - us[y_][us_i[y_]] * (yu - yl)

                if new_z < po[z_]:
                    z_range = po[z_] - pzs[z_max, z_]
                    x_slope = -(z_range + us[y_][us_i[y_]]*(yu - yl)) / (xu - xl)
                    us[x_] = abs(np.hstack((us[x_][us_i[x_]], x_slope, py[dzx_])))
                    us_i[x_] = np.argmin(us[x_])
                else:
                    us[x_] = abs(np.hstack((us[x_][us_i[x_]], py[dzx_])))
                    us_i[x_] = np.argmin(us[x_])

        elif (us[x_][us_i[x_]] < abs(pzs[max2min_i, iux_])) & (us[y_][us_i[y_]] == abs(pzs[max2min_i, iuy_])):
            us[y_][us_i[y_]] *= 2.0
            us_i[y_] = np.argmin(us[y_])

            # In y-axis, if optimal slope is not tangent line
            if us_i[y_] != tan_i:
                print('U case 2 (a) -------------------------------------- +\n')
                py = IdentityXIdentityOLD.getY(pzs, u_)
                py[z_] = pzs[z_max, z_] - us[y_][us_i[y_]] * (yu - yl)
                po = pzs[max2min_i, x_:z_+1]

                py_inter_x = [(py[z_] - po[z_]) / (xu - xl)]
                us[x_] = abs(np.hstack((us[x_][us_i[x_]], py_inter_x, py[dzx_])))
                us_i[x_] = np.argmin(us[x_])
            
            else:
                print('U case 2 (b) -------------------------------------- +\n')
                px = IdentityXIdentityOLD.getX(pzs, u_)
                po = pzs[max2min_i, x_:z_+1]
                new_z = pzs[z_max, z_] - us[x_][us_i[x_]] * (xu - xl) - us[y_][us_i[y_]] * (yu - yl)

                if new_z < po[z_]:
                    z_range = po[z_] - pzs[z_max, z_]
                    y_slope = -(z_range + us[x_][us_i[x_]]*(xu - xl)) / (yu - yl)
                    us[y_] = abs(np.hstack((us[y_][us_i[y_]], y_slope, px[dzy_])))
                    us_i[y_] = np.argmin(us[y_])

                else:
                    us[y_] = abs(np.hstack((us[y_][us_i[y_]], px[dzy_])))
                    us_i[y_] = np.argmin(us[y_])

        elif (us_i[x_] != tan_i) & (us_i[x_] != max2min_i):
            print('U case 3 ------------------------------------------\n')
            px = IdentityXIdentityOLD.getX(pzs, u_)
            us[y_] = abs(np.hstack((us[y_][us_i[y_]], px[dzy_])))
            us_i[y_] = np.argmin(us[y_])

        elif (us_i[y_] != tan_i) & (us_i[y_] != max2min_i):
            print('U case 4 ------------------------------------------ +\n')
            py = IdentityXIdentityOLD.getY(pzs, u_)
            us[x_] = abs(np.hstack((us[x_][us_i[x_]], py[dzx_])))
            us_i[x_] = np.argmin(us[x_])
        else:
            print('U case 5 ------------------------------------------ \n')
            ## add two tight constraints
            
            # pzs[max2min_i, ilx_] /= 2.0
            # pzs[max2min_i, ily_] /= 2.0
            # ## add two tight constraints

            # us[x_][us_i[x_]] = pzs[max2min_i, iux_]*4.0
            # us[y_][us_i[y_]] = pzs[max2min_i, iuy_]*0.5

            # us[x_][us_i[x_]] *= 0.0
            # us[y_][us_i[y_]] *= 2.0

            us[x_][us_i[x_]] *= 3.45
            us[y_][us_i[y_]] *= -1.5

        #####################################################################
        #                       Lower bound case
        #####################################################################
        min2max_i = np.nan
        # combine lower bound pzs with upper bound pzs
        # lower bound case for finding slopes from global minimum to other points
        for i in range(z_min+1, z_max+1):
#            print('i: ', i)
            z_min_range = pzs[i, z_] - pzs[z_min, z_] # slope in z-coordinate
            x_min_range = pzs[i, x_] - pzs[z_min, x_]
            y_min_range = pzs[i, y_] - pzs[z_min, y_]
            pzs[i, ilx_] = z_min_range / x_min_range if x_min_range else np.inf # slope in x-coordinate
            pzs[i, ily_] = z_min_range / y_min_range if y_min_range else np.inf # slope in y-coordinate
            if x_min_range != 0 and y_min_range != 0:
                pzs[i, ilx_] /= 2.0
                pzs[i, ily_] /= 2.0
                min2max_i = i

        print('pzs: \n', pzs)
#        print('getX(u_): ', IdentityXIdentityOLD.getX(pzs, u_))
#        print('getY(u_): ', IdentityXIdentityOLD.getY(pzs, u_))
#        print('getX(l_): ', IdentityXIdentityOLD.getX(pzs, l_))
#        print('getY(l_): ', IdentityXIdentityOLD.getY(pzs, l_))

        # finding minimal slopes among tangent lines and intersection lines
        lx = abs(np.hstack((pzs[z_min+1:z_max+1, ilx_], pzs[z_min, dzx_])))
        ly = abs(np.hstack((pzs[z_min+1:z_max+1, ily_], pzs[z_min, dzy_])))
        ls = [lx, ly] # lower bound slope [x-; y-] coordinates
        ls_i = [np.argmin(lx), np.argmin(ly)]

        if (abs(ls[x_][ls_i[x_]]) == abs(pzs[min2max_i, ilx_])) & (abs(ls[y_][ls_i[y_]]) < abs(pzs[min2max_i, ily_])):
            ls[x_][ls_i[x_]] *= 2.0
            ls_i[x_] = np.argmin(ls[x_])

            # In x-axis, if optimal slope is not tangent line
            if ls_i[x_] != tan_i:
                print('L case 1 (a) --------------------------------------\n')
                px = IdentityXIdentityOLD.getX(pzs, l_)
                px[z_] = pzs[z_min, z_] + ls[x_][ls_i[x_]] * (xu - xl)
                po = pzs[min2max_i, x_:z_+1]

                px_inter_y = [(px[z_] - po[z_]) / (yu - yl)]
                ls[y_] = abs(np.hstack((ls[y_][ls_i[y_]], px_inter_y, px[dzy_])))
                ls_i[y_] = np.argmin(ls[y_])
            
            else:
                print('L case 1 (b) --------------------------------------\n')
                py = IdentityXIdentityOLD.getY(pzs, l_)
                po = pzs[min2max_i, x_:z_+1]
                new_z = pzs[z_min, z_] + ls[x_][ls_i[x_]] * (xu - xl) + ls[y_][ls_i[y_]] * (yu - yl)

                if new_z > po[z_]:
                    z_range = po[z_] - pzs[z_min, z_]
                    x_slope = -(z_range + ls[y_][ls_i[y_]] * (yu - yl)) / (xu - xl)
                    ls[x_] = abs(np.hstack((ls[x_][ls_i[x_]], x_slope, py[dzx_])))
                    ls_i[x_] = np.argmin(ls[x_])

                else:
                    ls[x_] = abs(np.hstack((ls[x_][ls_i[x_]], py[dzx_])))
                    ls_i[x_] = np.argmin(ls[x_])

        elif (abs(ls[x_][ls_i[x_]]) < abs(pzs[min2max_i, ilx_])) & (abs(ls[y_][ls_i[y_]]) == abs(pzs[min2max_i, ily_])):
            ls[y_][ls_i[y_]] *= 2.0
            ls_i[y_] = np.argmin(ls[y_])
                                 
            # In y-axis, if optimal solution is not tangent line
            if ls_i[y_] != tan_i:
                print('L case 2 (a) -------------------------------------- +\n')
                py = IdentityXIdentityOLD.getY(pzs, l_)
                py[z_] = pzs[z_min, z_] + ls[y_][ls_i[y_]] * (yu - yl)
                po = pzs[min2max_i, x_:z_+1]

                py_inter_x = (py[z_] - po[z_]) / (xu - xl)
                ls[x_] = abs(np.hstack((ls[x_][ls_i[x_]], py_inter_x, py[dzx_])))
                ls_i[x_] = np.argmin(ls[x_])
            else:
                print('L case 2 (b) -------------------------------------- +\n')
                px = IdentityXIdentityOLD.getX(pzs, l_)
                po = pzs[min2max_i, x_:z_+1]
                new_z = pzs[z_min, z_] + ls[x_][ls_i[x_]] * (xu -xl) + ls[y_][ls_i[y_]] * (yu - yl)

                if new_z > po[z_]:
                    z_range = po[z_] - pzs[z_min, z_]
                    y_slope = -(z_range + ls[x_][ls_i[x_]] * (xu - xl)) / (yu - yl)
                    ls[y_] = abs(np.hstack((ls[y_][ls_i[y_]], y_slope, px[dzy_])))
                    ls_i[y_] = np.argmin(ls[y_])

                else:
                    ls[y_] = abs(np.hstack((ls[y_][ls_i[y_]], px[dzy_])))
                    ls_i[y_] = np.argmin(ls[y_])

        elif (ls_i[x_] != tan_i) & (ls_i[x_] != min2max_i):
            print('L case 3 ------------------------------------------ +\n')
            ## add two tight hyper-plane constraints instead of one hyper-plane constraint
            px = IdentityXIdentityOLD.getX(pzs, l_)
            ls[y_] = abs(np.hstack((ls[y_][ls_i[y_]], px[dzy_])))
            ls_i[y_] = np.argmin(ls[y_])

        elif (ls_i[y_] != tan_i) & (ls_i[y_] != min2max_i):
            print('L case 4 ------------------------------------------\n')
            py = IdentityXIdentityOLD.getY(pzs, l_)
            ls[x_] = abs(np.hstack((ls[x_][ls_i[x_]], py[dzx_])))
            ls_i[x_] = np.argmin(ls[x_])
        else:
            print('L case 5 ------------------------------------------ \n')

        # upper linear constraints on x-coordinate and y-coordinate
        if z_sorted_i[z_max] == 0:
            Ux = us[x_][us_i[x_]]
            Uy = us[y_][us_i[y_]]
        elif z_sorted_i[z_max] == 1:
            Ux = -us[x_][us_i[x_]]
            Uy = -us[y_][us_i[y_]]
        elif z_sorted_i[z_max] == 2:
            Ux = -us[x_][us_i[x_]]
            Uy = us[y_][us_i[y_]]
        elif z_sorted_i[z_max] == 3:
            Ux = us[x_][us_i[x_]]
            Uy = -us[y_][us_i[y_]]
        else:
            raise Exception('error: unknown point')
        Ub = pzs[z_max, z_] + Ux*pzs[z_max, x_] + Uy*pzs[z_max, y_]
    
        # lower linear constraints on x-coordinate and y-coordinate
        if z_sorted_i[z_min] == 0:
            Lx = -ls[x_][ls_i[x_]]
            Ly = -ls[y_][ls_i[y_]]
        elif z_sorted_i[z_min] == 1:
            Lx = ls[x_][ls_i[x_]]
            Ly = ls[y_][ls_i[y_]]
        elif z_sorted_i[z_min] == 2:
            Lx = ls[x_][ls_i[x_]]
            Ly = -ls[y_][ls_i[y_]]
        elif z_sorted_i[z_min] == 3:
            Lx = -ls[x_][ls_i[x_]]
            Ly = ls[y_][ls_i[y_]]
        else:
            raise Exception('error: unknown point')
        Lb = pzs[z_min, z_] + Lx*pzs[z_min, x_] + Ly*pzs[z_min, y_]

        zmax = pzs[z_max, z_]
        zmin = pzs[z_min, z_]

        return Ux, Uy, Ub, Lx, Ly, Lb, zmax, zmin