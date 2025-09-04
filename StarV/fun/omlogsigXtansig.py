"""
onlogsigXtansig Class ( (1 - logsig(x)) * tansig(y), where x in X, y in Y, X and Y are star sets)
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
from StarV.fun.logsig import LogSig
from StarV.fun.tansig import TanSig

class OmLogsigXTansig(object):
    """
    OmLogsigXTansig Class for reachability
    Author: Sung Woo Choi
    Date: 01/12/2024

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
    def evaluate(x, y):
        return (1 - LogSig.f(x)) * TanSig.f(y)
		
    def f(x, y):
        return (1 - LogSig.f(x)) * TanSig.f(y)
    
    @staticmethod
    def gf(x, y):
        """Gradient of x*y"""
        return -LogSig.df(x)*TanSig.f(y), (1-LogSig.f(x))*TanSig.df(y) 
    #     return LogSig.df(x)*TanSig.f(y), LogSig.f(x)*TanSig.df(y) 
    
    @staticmethod
    def getX(p, bound):
        if bound == OmLogsigXTansig.u_:
            for i in range(OmLogsigXTansig.z_max-1, OmLogsigXTansig.z_min-1, -1):
                if p[i, OmLogsigXTansig.iuy_] == np.inf:
                    return copy.deepcopy(p[i, OmLogsigXTansig.x_ : OmLogsigXTansig.dzy_+1])
        elif bound == OmLogsigXTansig.l_:
            for i in range(OmLogsigXTansig.z_min+1, OmLogsigXTansig.z_max+1):
                if p[i, OmLogsigXTansig.ily_] == np.inf:
                    return copy.deepcopy(p[i, OmLogsigXTansig.x_ : OmLogsigXTansig.dzy_+1])
        else:
            raise Exception('error: unknown bound; should be u_ or l_')

    @staticmethod
    def getY(p, bound):
        if bound == OmLogsigXTansig.u_:
            for i in range(OmLogsigXTansig.z_max-1, OmLogsigXTansig.z_min-1, -1):
                if p[i, OmLogsigXTansig.iux_] == np.inf:
                    return copy.deepcopy(p[i, OmLogsigXTansig.x_ : OmLogsigXTansig.dzy_+1])
        elif bound == OmLogsigXTansig.l_:
            for i in range(OmLogsigXTansig.z_min+1, OmLogsigXTansig.z_max+1):
                if p[i, OmLogsigXTansig.ilx_] == np.inf:
                    return copy.deepcopy(p[i, OmLogsigXTansig.x_ : OmLogsigXTansig.dzy_+1])
        else:
            raise Exception('error: unknown bound; should be u_ or l_')
        
    @staticmethod
    def getO(p, index, rtype='index'):
        # get opposite point
        for i in range(OmLogsigXTansig.num_of_points):
            if p[i, OmLogsigXTansig.x_] != p[index, OmLogsigXTansig.x_] and p[i, OmLogsigXTansig.y_] != p[index, OmLogsigXTansig.y_]:
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

        u_ = OmLogsigXTansig.u_
        l_ = OmLogsigXTansig.l_
    
        x_ = OmLogsigXTansig.x_
        y_ = OmLogsigXTansig.y_
        z_ = OmLogsigXTansig.z_
        dzx_ = OmLogsigXTansig.dzx_
        dzy_ = OmLogsigXTansig.dzy_
        iux_ = OmLogsigXTansig.iux_
        iuy_ = OmLogsigXTansig.iuy_
        ilx_ = OmLogsigXTansig.ilx_
        ily_ = OmLogsigXTansig.ily_
        
        z_max = OmLogsigXTansig.z_max
        z_min = OmLogsigXTansig.z_min
    
        num_of_points = OmLogsigXTansig.num_of_points

        pz = np.zeros((num_of_points, num_of_points+1))

        pz[0, x_:y_+1] = [xl, yl]
        pz[1, x_:y_+1] = [xu, yu]
        pz[2, x_:y_+1] = [xu, yl]
        pz[3, x_:y_+1] = [xl, yu]

        pz[:,  z_] = OmLogsigXTansig.f(pz[:, x_], pz[:,y_])
        pz[:, dzx_:dzy_+1] = np.column_stack((OmLogsigXTansig.gf(pz[:, x_], pz[:, y_])))

        # sort z-coordinate points in ascending order
        z_sorted_i = np.argsort(pz[:, z_])
        pzs = np.hstack([pz[z_sorted_i, :], np.full((num_of_points, num_of_points), np.nan)])

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
                # print('U case 1 (a) --------------------------------------\n')
                px = OmLogsigXTansig.getX(pzs, u_)
                px[z_] = pzs[z_max, z_] - us[x_][us_i[x_]] * (xu - xl)
                po = pzs[max2min_i, x_:z_+1]

                px_inter_y = [(px[z_] - po[z_]) / (yu - yl)]
                us[y_] = abs(np.hstack((us[y_][us_i[y_]], px_inter_y, px[dzy_])))
                us_i[y_] = np.argmin(us[y_])
            else:
                # print('U case 1 (b) --------------------------------------\n')
                py = OmLogsigXTansig.getY(pzs, u_)
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
                # print('U case 2 (a) -------------------------------------- \n') #+
                py = OmLogsigXTansig.getY(pzs, u_)
                py[z_] = pzs[z_max, z_] - us[y_][us_i[y_]] * (yu - yl)
                po = pzs[max2min_i, x_:z_+1]

                py_inter_x = [(py[z_] - po[z_]) / (xu - xl)]
                us[x_] = abs(np.hstack((us[x_][us_i[x_]], py_inter_x, py[dzx_])))
                us_i[x_] = np.argmin(us[x_])
            
            else:
                # print('U case 2 (b) -------------------------------------- \n') #+
                px = OmLogsigXTansig.getX(pzs, u_)
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
            # print('U case 3 ------------------------------------------\n')
            px = OmLogsigXTansig.getX(pzs, u_)
            us[y_] = abs(np.hstack((us[y_][us_i[y_]], px[dzy_])))
            us_i[y_] = np.argmin(us[y_])

        elif (us_i[y_] != tan_i) & (us_i[y_] != max2min_i):
            # print('U case 4 ------------------------------------------ +\n') #+
            py = OmLogsigXTansig.getY(pzs, u_)
            us[x_] = abs(np.hstack((us[x_][us_i[x_]], py[dzx_])))
            us_i[x_] = np.argmin(us[x_])
        else:
            # print('U case 5 ------------------------------------------ \n')
            pass

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
                # print('L case 1 (a) -------------------------------------- +\n')
                px = OmLogsigXTansig.getX(pzs, l_)
                px[z_] = pzs[z_min, z_] + ls[x_][ls_i[x_]] * (xu - xl)
                po = pzs[min2max_i, x_:z_+1]

                px_inter_y = [(px[z_] - po[z_]) / (yu - yl)]
                ls[y_] = abs(np.hstack((ls[y_][ls_i[y_]], px_inter_y, px[dzy_])))
                ls_i[y_] = np.argmin(ls[y_])
            
            else:
                # print('L case 1 (b) --------------------------------------\n')
                py = OmLogsigXTansig.getY(pzs, l_)
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
                # print('L case 2 (a) -------------------------------------- \n') #+
                py = OmLogsigXTansig.getY(pzs, l_)
                py[z_] = pzs[z_min, z_] + ls[y_][ls_i[y_]] * (yu - yl)
                po = pzs[min2max_i, x_:z_+1]

                py_inter_x = (py[z_] - po[z_]) / (xu - xl)
                ls[x_] = abs(np.hstack((ls[x_][ls_i[x_]], py_inter_x, py[dzx_])))
                ls_i[x_] = np.argmin(ls[x_])
            else:
                # print('L case 2 (b) -------------------------------------- \n') #+
                px = OmLogsigXTansig.getX(pzs, l_)
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
            # print('L case 3 ------------------------------------------ \n') #+
            px = OmLogsigXTansig.getX(pzs, l_)
            ls[y_] = abs(np.hstack((ls[y_][ls_i[y_]], px[dzy_])))
            ls_i[y_] = np.argmin(ls[y_])

        elif (ls_i[y_] != tan_i) & (ls_i[y_] != min2max_i):
            # print('L case 4 ------------------------------------------\n')
            py = OmLogsigXTansig.getY(pzs, l_)
            ls[x_] = abs(np.hstack((ls[x_][ls_i[x_]], py[dzx_])))
            ls_i[x_] = np.argmin(ls[x_])
        else:
            # print('L case 5 ------------------------------------------ \n')
            pass

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



    def multiStep_sparse(X, H, DR=0, RF=0.0, lp_solver='gurobi', pool=None):

        assert isinstance(X, SparseStar) and isinstance(H, SparseStar), 'error: \
        both input sets should be SparseStar sets'
        
        assert X.dim == H.dim, 'error: dimension of two input sets should be equivalent'
        N = X.dim

        nVars = X.nVars + H.nVars
        nZVars = X.nZVars + H.nZVars     

        xl, xu = X.getRanges(lp_solver=lp_solver, RF=RF)
        hl, hu = H.getRanges(lp_solver=lp_solver, RF=RF)

        map0 = np.where((xl != xu) & (hl != hu))[0]
        m0 = len(map0)

        zmin0 = np.empty([0])
        zmax0 = np.empty([0])

        if m0 == N:
            A0 = np.eye(N)
            A1 = np.hstack([np.zeros([N, 1]), A0])        
            m = N
            nv = nVars + m
            
        else:               

            map1 = np.where((xl == xu) & (hl == hu))[0]
            m1 = len(map1)
            if m1 == N:
                if len(H.d):
                    new_A = OmLogsigXTansig.f(xl, hl)[:, None]
                    return SparseStar(new_A, H.C, H.d, H.pred_lb, H.pred_ub, H.pred_depth)
                else:
                    new_A = OmLogsigXTansig.f(xl, hl)[:, None]
                    return SparseStar(new_A, X.C, X.d, X.pred_lb, X.pred_ub, X.pred_depth)
                    
            map2 = np.where((xl == xu) & (hl != hu))[0]
            m2 = len(map2)
            if m2 == N:
                TH = TanSig.reach(H, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                new_A = (1-LogSig.f(xl[:, None])) * TH.A
                return SparseStar(new_A, TH.C, TH.d, TH.pred_lb, TH.pred_ub, TH.pred_depth)

            map3 = np.where((xl != xu) & (hl == hu))[0]
            m3 = len(map3)
            if m3 == N:
                #(1-sigmoid(X)) * tanh(H)
                SX = LogSig.reach(X, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                SX = SX.affineMap(-np.eye(m3), np.ones(m3))
                new_A = TanSig.f(hl)[:, None] * SX.A
                return SparseStar(new_A, SX.C, SX.d, SX.pred_lb, SX.pred_ub, SX.pred_depth)
            
                
            map4 = np.concatenate([map0, map2, map3])
            m = len(map4)
            nv = nVars + m
            
            A1 = np.zeros((N, 1))
            if m:
                A0 = np.zeros((N, m))   
                for i in range(m):
                    A0[map4[i], i] = 1
                A1 = np.hstack((A1, A0))

            if m1:
                A1[map1, 0] = OmLogsigXTansig.f(xl[map1], hl[map1])[:, None]
                
            if m2:
                X02 = np.hstack([np.zeros([m2, X.nVars]), H.X(map2)])
                C02, d02, yl02, yu02 = TanSig.getConstraints(H.c(map2), X02, A1[map2, 1:], hl[map2], hu[map2], nZVars)
                A1[map2, :] *= (1-LogSig.f(xl[map2])[:, None])
                zmin0 = np.hstack([zmin0, yl02.reshape(-1)])
                zmax0 = np.hstack([zmax0, yu02.reshape(-1)])
                            
            if m3:
                #(xl != xu) & (hl == hu)
                #(1-sigmoid(X)) * tanh(H) = tanh(H) - sigmoid(X)*tanh(H)
                X03 = np.hstack([X.X(map3), np.zeros([m3, H.nVars])])
                C03, d03, yl03, yu03 = LogSig.getConstraints(X.c(map3), X03, A1[map3, 1:], xl[map3], xu[map3], nZVars)
                
                t = TanSig.f(hl[map3])
                A1[map3, :] *= -t[:, None]
                A1[map3, 0] += t
                zmin0 = np.hstack([zmin0, yl03.reshape(-1)])
                zmax0 = np.hstack([zmax0, yu03.reshape(-1)])         
                        
        XC1 = X.C[:, 0:X.nZVars]
        XC2 = X.C[:, X.nZVars:X.nVars]

        HC1 = H.C[:, 0:H.nZVars]
        HC2 = H.C[:, H.nZVars:H.nVars]

        C1_ = sp.block_diag((XC1, HC1)).tocsc()
        C2_ = sp.block_diag((XC2, HC2)).tocsc()
        
        C0 = sp.hstack([C1_, C2_, sp.csc_matrix((C2_.shape[0], m))]).tocsc()
        d0 = np.concatenate((X.d, H.d))
            
        if m0 != N:
            C0 = sp.vstack([C0, C02, C03]).tocsc()
            d0 = np.concatenate((d0, d02, d03))
        
        if m0:
            Z = sp.csc_matrix((m0, nZVars))

            Ux, Uh, Ub, Lx, Lh, Lb, zmax, zmin = [np.zeros((m0, 1)) for _ in range(8)]

            for i in range(m0):
                Ux[i], Uh[i], Ub[i], Lx[i], Lh[i], Lb[i], zmax[i], zmin[i] = OmLogsigXTansig.getConstraints(xl[map0[i]].item(), xu[map0[i]].item(), hl[map0[i]].item(), hu[map0[i]].item())

            C11 = sp.hstack((Z, -Lx*X.X(map0), -Lh*H.X(map0), -A0[map0]))
            d11 = Lx*X.c(map0) + Lh*H.c(map0) - Lb

            C12 = sp.hstack((Z, Ux*X.X(map0), Uh*H.X(map0), A0[map0]))
            d12 = -(Ux*X.c(map0) + Uh*H.c(map0)) + Ub

            C1 = sp.vstack((C11, C12)).tocsc()
            d1 = np.vstack((d11, d12)).reshape(-1)
        else:
            C1 = sp.csc_matrix((0, nv))
            d1 = np.empty((0))
            zmin = np.empty((0))
            zmax = np.empty((0))

        new_A = A1
        new_C = sp.vstack((C0, C1))
        new_d = np.hstack((d0, d1))

        new_pred_lb =    np.hstack((X.pred_lb[0:X.nZVars],          H.pred_lb[0:H.nZVars],
                                    X.pred_lb[X.nZVars:X.nVars],    H.pred_lb[H.nZVars:H.nVars]))
        new_pred_ub =    np.hstack((X.pred_ub[0:X.nZVars],          H.pred_ub[0:H.nZVars],
                                    X.pred_ub[X.nZVars:X.nVars],    H.pred_ub[H.nZVars:H.nVars]))
        new_pred_depth = np.hstack((X.pred_depth[0:X.nZVars],       H.pred_depth[0:H.nZVars],
                                    X.pred_depth[X.nZVars:X.nVars], H.pred_depth[H.nZVars:H.nVars])) + 1
        
        new_pred_lb = np.hstack([new_pred_lb, zmin.reshape(-1), zmin0])
        new_pred_ub = np.hstack([new_pred_ub, zmax.reshape(-1), zmax0])
        new_pred_depth = np.hstack([new_pred_depth, np.zeros(m)])

        S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
        if DR > 0:
            S = S.depthReduction(DR)
        return S




    # def multiStepOmLogsigXTansig_sparse(I, X, H, lp_solver='gurobi', DR=0, RF=0.0):

    #     assert isinstance(I, SparseStar) and isinstance(X, SparseStar) and isinstance(H, SparseStar), 'error: \
    #     both input sets should be SparseStar sets'

    #     N = I.dim

    #     xl, xu = X.getRanges(lp_solver=lp_solver, RF=RF)
    #     hl, hu = H.getRanges(lp_solver=lp_solver, RF=RF)
    #     xl = xl.reshape(N, 1)
    #     xu = xu.reshape(N, 1)
    #     hl = hl.reshape(N, 1)
    #     hu = hu.reshape(N, 1)

    #     ## l != u
    #     map0 = np.where((xl != xu) & (hl != hu))[0]
    #     m = len(map0)
    #     A0 = np.zeros((N, m))
    #     for i in range(m):
    #         A0[map0[i], i] = 1
    #     new_A = np.hstack((np.zeros((N, 1)), A0))

    #     map1 = np.where((xl == xu) & (hl == hu))[0]
    #     if len(map1):
    #         zl = OmLogsigXTansig.f(xl[map1], hl[map1]).reshape(-1)
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

    #         # Ux, Uh, Ub = np.zeros((m, 1)), np.zeros((m, 1)), np.zeros((m, 1))
    #         # Lx, Lh, Lb = np.zeros((m, 1)), np.zeros((m, 1)), np.zeros((m, 1))
    #         # Zl, Zu = np.zeros((m, 1)), np.zeros((m, 1))

    #         Ux, Uh, Ub, Lx, Lh, Lb, Zu, Zl = [np.zeros((m, 1)) for _ in range(8)]
            
    #         for i in range(m):
    #             Ux[i], Uh[i], Ub[i], Lx[i], Lh[i], Lb[i], Zu[i], Zl[i] = OmLogsigXTansig.getConstraints(xl[map0[i]].item(), xu[map0[i]].item(), hl[map0[i]].item(), hu[map0[i]].item())

    #         C11 = sp.hstack((Z, -Lx*X.X(map0), -Lh*H.X(map0), -A0[map0]))
    #         d11 = Lx*X.c(map0) + Lh*H.c(map0) - Lb

    #         C12 = sp.hstack((Z, Ux*X.X(map0), Uh*H.X(map0), A0[map0]))
    #         d12 = -(Ux*X.c(map0) + Uh*H.c(map0)) + Ub

    #         C1 = sp.vstack((C11, C12)).tocsc()
    #         d1 = np.vstack((d11, d12)).reshape(-1)
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

    #     new_pred_lb = np.hstack((I.pred_lb, Zl.reshape(-1)))
    #     new_pred_ub = np.hstack((I.pred_ub, Zu.reshape(-1)))
    #     new_pred_depth = np.hstack((I.pred_depth+1, np.zeros(m)))
        
    #     S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
    #     if DR > 0:
    #         S = S.depthReduction(DR)
    #     return S

    
    def reachApproxSparse(X, H, lp_solver='gurobi', RF=0.0, DR=0):

        assert isinstance(X, SparseStar) and isinstance(H, SparseStar), 'error: both or one of input sets are not SparseStar sets'
        assert X.dim == H.dim, 'error: dimension of input sets does not match'

        return OmLogsigXTansig.multiStep_sparse(X, H, lp_solver=lp_solver, RF=RF, DR=DR)
    
        
    def reach(X, H, lp_solver='gurobi', pool=None, RF=0.0, DR=0):
        if isinstance(X, SparseStar) and isinstance(H, SparseStar):

            return OmLogsigXTansig.reachApproxSparse(X, H, lp_solver, RF, DR)

        elif isinstance(X, Star) and isinstance(H, Star):
            #return LogsigXTansig.reachApproxStar(X, H, depthReduct, relaxFactor, lp_solver)
            raise Exception('error: under development')
        else:
            raise Exception('error: both input sets (X, H) should be SparseStar or Star')