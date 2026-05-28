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
PosLin Class
Dung Tran, 8/29/2022
Update: 12/20/2024 (Sung Woo Choi, merging)
Update: Yuntao Li, Date: 09/16/2025
Update: 03/07/2026 (Sung Woo Choi, support milp; csr now operates in csr format instead of changing to coo)
"""

# !/usr/bin/python3
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
from StarV.set.imagestar import ImageStar
from StarV.set.sparsestar import SparseStar
from StarV.set.sparseimagestar import *
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.set.predicate_layout import PredicateLayout

import numpy as np
import scipy.sparse as sp
import copy
import multiprocessing
import ipyparallel


class PosLin(object):
    """
    PosLin Class for qualitative and quantitative reachability
    Author: Dung Tran
    Date: 8/29/2022

    """

    @staticmethod
    def evaluate(x):
        """
        Evaluate method
        Args: @x = np.array()
        Returns:
            0, if x < 0
            x, if x >= 0
        """

        return np.maximum(x, 0)
    
    @staticmethod
    def f(x):
        return np.maximum(x, 0)
    
    @staticmethod
    def multiStepReach(I, lp_solver='gurobi'):
        """
        multiStepReach method, compute reachable set for a multiple steps

        Args:
            @I: a single input set (i.e. ProbStar, Star, ImageStar, SparseStar, SparseImageStar)
        """

        assert isinstance(I, ProbStar) or isinstance(I, Star) or isinstance(I, SparseStar) or \
        isinstance(I, ImageStar) or isinstance(I, SparseImageStar), \
        'error: input set is not supported, type of input is {}'.format(type(I))

        xmin, xmax = I.estimateRanges()
        

    @staticmethod
    def stepReach(*args):

        """
        stepReach method, compute reachable set for a single step

        Args:
            @I: single star set input
            @index: index of current x[index] of current step

        Returns:
            @S: star output set

        Update: Yuntao Li, Date: 09/16/2025
        """

        len_args = len(args)
        if len_args == 2:  # 2 arguments
            [I, index] = args
            lp_solver = 'gurobi'
        elif len_args == 3:  # 3 arguments
            [I, index, lp_solver] = args
        else:
            raise Exception('error: \
            Invalid number of input arguments, should be 2 or 3')

        assert isinstance(I, ProbStar) or isinstance(I, Star), \
        'error: input is not a Star or ProbStar, type of input is {}'.format(type(I))

        xmin, xmax = I.estimateRange(index)
        if xmin >= 0:
            S = []
            S.append(I)
        elif xmax <= 0:
            S = []
            S.append(I.resetRow(index))
        else:
            xmax = I.getMax(index, lp_solver)
            if xmax <= 0:
                S = []
                S.append(I.resetRow(index))
            else:
                xmin = I.getMin(index, lp_solver)
                if xmin >= 0:
                    S = []
                    S.append(I)
                else:
                    C = np.zeros(I.dim,)
                    C[index] = 1.0
                    d = np.zeros(1,)
                    S1 = copy.deepcopy(I)
                    S2 = copy.deepcopy(I)
                    S1 = S1.addConstraint(C, d, copy_=False)  # x <= 0
                    S1 = S1.resetRow(index, copy_=False)
                    S2 = S2.addConstraint(-C, d, copy_=False)  # x >= 0
                    S = []
                    S.append(S1)
                    S.append(S2)
        return S

    @staticmethod
    def stepReachMultiInputs(*args):
        """
        stepReach with multiple inputs
        Args:
            @I: a list of input set
            @lp_solver: lp_solver

        Returns:
            @S: a list of output set

        Author: Dung Tran, Date: 8/30/2022
        """
        if len(args) == 2:
            [I, index] = args
            lp_solver = 'gurobi'
        elif len(args) == 3:
            [I, index, lp_solver] = args
        else:
            raise Exception('error: \
            Invalid number of input arguments, should be 2 or 3 ')

        assert isinstance(I, list), 'error: input is not a list, \
        type of input is {}'.format(type(I))

        S = []
        for i in range(0, len(I)):
            S1 = PosLin.stepReach(I[i], index, lp_solver)
            S.extend(S1)
        return S

    @staticmethod
    def reachExactSingleInput(*args):
        """
        Exact reachability using stepReach
        Args:
            @I: a single input set
            @lp_solver: lp_solver

        Returns:
            @S: output set

        Author: Dung Tran, Date: 8/30/2022
        """

        if isinstance(args[0], tuple):  # when this method is called in parallel
            args1 = list(args[0])
        else:
            args1 = args
        if len(args1) == 1:
            In = args1
            lp_solver = 'gurobi'
        elif len(args1) == 2:
            [In, lp_solver] = args1
        else:
            raise Exception('error: Invalid \
            number of input arguments, should be 1 or 2')

        assert isinstance(In, ProbStar) or isinstance(In, Star), \
        'error: input is not a Star or ProbStar, type of input is {}'.format(type(In))

        S = []
        S1 = [In]
        for i in range(0, In.dim):
            S1 = PosLin.stepReachMultiInputs(S1, i, lp_solver)

        S.extend(S1)

        return S

    def reachExactMultiInputs(*args):
        """
        Exact reachability with multiple inputs
        Work with bread-first-search verification

        Args:
            @I: a single input set
            @lp_solver: lp_solver ('gurobi' or 'glpk' or 'linprog')
            @pool: pool for parallel computation
        Returns:
            @S: output set

        Author: Dung Tran, Date: 8/30/2022
        """
        lp_solver_default = 'gurobi'
        
        if len(args) == 1:
            In = args
            lp_solver = lp_solver_default
            pool = None
        elif len(args) == 2:
            [In, lp_solver] = args
            pool = None
        elif len(args) == 3:
            [In, lp_solver, pool] = args
       
        else:
            raise Exception('error: Invalid \
            number of input arguments, should be 1, 2 or 3')

        assert isinstance(In, list), 'error: input sets should be in a list'
        S = []
        if pool is None:
            for i in range(0, len(In)):
                S.extend(PosLin.reachExactSingleInput(In[i], lp_solver))
        elif isinstance(pool, multiprocessing.pool.Pool):
            S1 = []
            S1 = S1 + pool.map(PosLin.reachExactSingleInput, zip(In, [lp_solver]*len(In)))
            for i in range(0, len(S1)):
                S.extend(S1[i])
        elif isinstance(pool, ipyparallel.client.view.DirectView):
            # S1 = pool.map(PosLin.reachExactSingleInput, zip(In, [lp_solver]*len(In)))
            # print('S1 = {}'.format(S1))
            raise Exception('error: ipyparallel option is under testing...')
        else:
            raise Exception('error: unknown/unsupport pool type')    
        return S

    
    def relax_by_area(I, l, u, lp_solver='gurobi', RF=0.0, show=False):

        # if RF == 0.0:
        #     if show:
        #         print('No relaxation by triangular area applied due to RF = {}'.format(RF))
        #     return l, u
        
        # elif RF == 0.0:
        #     if show:
        #         print('Applying full relaxation (RF = {})'.format(RF))

        mapL = np.argwhere(u <= 0).reshape(-1)
        mapM = np.argwhere((l < 0) & (u > 0)).reshape(-1)

        n1 = round((1 - RF) * len(mapM)) #number of LP need to solve
        
        if show:
            print('Applying relaxation by triangular area with RF = {}'.format(RF))
            print('{} neurons with (ub <= 0) case are found by bound estimation'.format(len(mapL)))
            print('(1 - {}) x {} = {} neurons with (l < 0) & (u > 0) case are found by LP solver'.format(RF, len(mapM), n1))
        
        areas = 0.5 * abs(u[mapM] * l[mapM])
        midx = np.argsort(areas)[::-1] #sort in descending order

        mapO = mapM[midx[:n1]] # neurons with LP optimized ranges
        mapN = mapM[midx[n1:]] # neurons estimation ranges

        lN = l[mapN]
        uN = u[mapN]

        if show:
            print('Optimize upper bounds of {} neurons'.format(len(mapO)))
        xmax = I.getMaxs(mapO, lp_solver)
        
        map_ = np.argwhere(xmax <= 0).reshape(-1)
        mapOL = mapO[map_]

        # case ub <= 0
        mapL = np.concatenate([mapL, mapOL])
        In = I.resetRows(mapL) # reset to zero at the neuron having ub <= 0
        
        # case lb < 0 & ub > 0
        map1_ = np.argwhere(xmax > 0).reshape(-1)
        map1 = mapO[map1_] # all indexes having ub > 0

        xmax1 = xmax[map1_] # upper bound of all neurons having ub > 0

        if show:
            print('Optimize lower bounds of {} neurons'.format(len(mapO)))
        xmin = I.getMins(map1, lp_solver)

        map2_ = np.argwhere(xmin < 0).reshape(-1)
        map2 = map1[map2_]
                    
        lO = xmin[map2_]
        uO = xmax1[map2_]

        MAP = np.concatenate([mapN, map2])
        lb = np.concatenate([lN, lO])
        ub = np.concatenate([uN, uO])
        return In, lb, ub, MAP


    def approx(I, l, u, lp_solver='gurobi', show=False):
        """
        updated: 
            Yuntao, 07/31/2025
            Sung Woo Choi, 09/13/2025
            Yuntao, 09/16/2025
        """

        if show:
            print('Internediate reachable set has {} neurons'.format(len(l)))

        map1 = np.argwhere(u <= 0).reshape(-1)
        if show:
            print('Ranges of {} neurons with (ub <= 0) are found by estimation initially'.format(len(map1)))

        map2 = np.argwhere((l < 0) & (u > 0)).reshape(-1)
        if show:
            print('Ranges of {} neurons with (lb < 0) and (ub > 0) are found by LP solver'.format(len(map2)))

        xmax = I.getMaxs(map2, lp_solver)
        map3 = np.argwhere(xmax <= 0).reshape(-1)
        if show:
            print('Ranges of {} neurons with (ub <= 0) are found by LP solver'.format(len(map3)))

        map4 = map2[map3]
        map11 = np.concatenate([map1, map4])

        # updated: Yuntao, 09/16/2025
        In = I.resetRows(map11) if len(map11) > 0 else I

        if show:
            print('({} + {} = {}) / {} neurons have ub < = 0'.format(len(map1), len(map3), len(map11), len(u)))

        # find all indexes that have (l < 0) and (u > 0), then,
        # apply the over-approximation rule for ReLU

        if show:
            print('Finding all neurons with (lb < 0) and (ub > 0)')
 
        map5 = np.argwhere(xmax > 0).reshape(-1)
        map6 = map2[map5] # all indexes having ub > 0
        xmax1 = xmax[map5] # upper bound of all neurons having ub > 0

        xmin = I.getMins(map6, lp_solver)
        map7 = np.argwhere(xmin < 0).reshape(-1)
        map8 = map6[map7]

        lb = xmin[map7]
        ub = xmax1[map7]

        # return PosLin.addConstraints(I=I, map=map8, l=lb, u=ub)
        return In, lb, ub, map8

        
    def addConstraints(I, map, l, u, milp=False):
        """
        Over-approximate the ReLU function by linear constraints for neurons with (l < 0) and (u > 0)
        Args:
        - I: input star set before ReLU layer
        - map: indices of neurons with (l < 0) and (u > 0)
        - l: lower bound vector of the neurons in map
        - u: upper bound vector of the neurons in map
        - milp: whether to apply MILP constraints (big-M method) for over-approximation (default: False)
        Return:
        - output star set after adding constraints for ReLU layer
        """
        m = len(map) # number of neurons invovled
        if m == 0:
            return I

        N = I.dim
        n = I.nVars # number of predicate variables in the reachable set before ReLU layer
        dtype = I.V.dtype        
        
        # New predicate variables for the output of ReLU layer
        V1 = copy.deepcopy(I.V)
        V1[map, :] = 0
        
        V2 = np.zeros([N, m], dtype=dtype)
        for i in range(m):
            V2[map[i], i] = 1
        
        # x = cx + Vx*alpha 
        cx = I.V[map, 0]
        Vx = I.V[map, 1:n+1]
        
        # Prepare constraint matrices
        Z_mn = np.zeros([m, n], dtype=dtype)
        I_m  = np.eye(m, dtype=dtype)
        Z_mm = np.zeros([m, m], dtype=dtype)
        z_m  = np.zeros(m, dtype=dtype)
            
        if milp:
            # Existing constraints need both y and z variable blocks.
            if len(I.C) == 0:
                C0 = np.empty([0, n + 2 * m], dtype=dtype)
                d0 = np.empty([0], dtype=dtype)
            else:
                C0 = np.hstack([I.C, np.zeros([I.C.shape[0], 2 * m], dtype=dtype)])
                d0 = I.d

            # over-approximate the ReLU function by MILP constraints (big-M method)
            # Apply 4 linear constraints for each neuron with (l < 0) and (u > 0)
            # case 1: y[index] >= 0
            # case 2: y[index] >= x[index]
            # case 3: y[index] <= u * z
            # case 4: y[index] <= x[index] * (1 - z), where z is a binary variable (z in {0, 1})
            
            # V2 is the coefficient matrix for new predicate variables corresponding to the output of ReLU layer (y cols)
            # V3 is the coefficient matrix for new binary variables (z cols)
            V3 = np.zeros([N, m], dtype=dtype)
            new_V = np.hstack([V1, V2, V3])
    
            # case 1: y[index] >= 0 <=> -y[index] <= 0
            C1 = np.hstack([Z_mn, -I_m, Z_mm])
            d1 = z_m
            
            # case 2: y[index] >= x[index] <=> Vx*alpha -y[index] <= -cx
            C2 = np.hstack([Vx, -I_m, Z_mm])
            d2 = -cx
            
            # case 3: y[index] <= u * z <=> y[index] - u * z <= 0
            C3 = np.hstack([Z_mn, I_m, -np.diag(u)])
            d3 = z_m
            
            # case 4: y[index] <= x[index] - l[index] * (1 - z) <=> -Vx*alpha + y[index] - l * z <= cx - l
            C4 = np.hstack([-Vx, I_m, -np.diag(l)])
            d4 = cx - l
            
            # Combine all constraints
            new_C = np.vstack([C0, C1, C2, C3, C4])
            new_d = np.hstack([d0, d1, d2, d3, d4])
            
            # Extend predicate bounds for new variables: y in [0, u], z in {0, 1}
            new_pred_lb = np.hstack([I.pred_lb, z_m, z_m])
            new_pred_ub = np.hstack([I.pred_ub, u, np.ones(m, dtype=dtype)])
            
            layout = copy.deepcopy(getattr(I, "pred_layout", None))
            if layout is None:
                layout = PredicateLayout(n_base=I.nVars)
            y_sl, z_sl = layout.add_relu_bigM_block(m) # add new (y,z) block for ReLU layer
            
            return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub, layout)

        # Extend existing predicate constraints C alpha <= d with zeros for new y vars
        if len(I.C) == 0:
            C0 = np.empty([0, n + m], dtype=dtype)
            d0 = np.empty([0], dtype=dtype)
        else:
            C0 = np.hstack([I.C, np.zeros([I.C.shape[0], m], dtype=dtype)])
            d0 = I.d
            
        # Triangular area relaxation for ReLU function
        # Apply 3 linear constraints for each neuron with (l < 0) and (u > 0)
        # case 1: y[index] >= 0
        # case 2: y[index] >= x[index]
        # case 3: y[index] <= (u / (u - l)) * (x - l)
        
        new_V = np.hstack([V1, V2])
        
        # case 1: y[index] >= 0
        C1 = np.hstack([Z_mn, -I_m])
        d1 = z_m

        # case 2: y[index] >= x[index]
        C2 = np.hstack([Vx, -I_m])
        d2 = -cx

        # case 3: y[index] <= (u / (u - l)) * (x - l)
        a = u / (u - l)
        b = a * l

        C3 = np.hstack([(-a[:, None] * Vx), I_m])
        d3 = a * cx - b

        # Combine all constraints
        new_C = np.vstack([C0, C1, C2, C3])
        new_d = np.hstack([d0, d1, d2, d3])

        # Extend predicate bounds for new variables: y in [0, u]
        new_pred_lb = np.hstack([I.pred_lb, z_m])
        new_pred_ub = np.hstack([I.pred_ub, u])
        
        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)


    def addConstraints_sparse(I, map, l, u, milp=False):
        N = I.dim
        m = len(map) # number of neurons invovled
        dtype = I.V.dtype

        A1 = copy.deepcopy(I.A)
        A1[map, :] = 0

        A2 = np.zeros([N, m], dtype=dtype)
        for i in range(m):
            A2[map[i], i] = 1

        if (A1[:, 1:] == 0).all():
            A = np.hstack((A1[:, 0, None], A2))
        else:
            A = np.hstack((A1, A2))

        n = I.nVars
        Z = sp.csc_array((m, I.nZVars))

        C0 = sp.hstack((I.C, sp.csc_array((I.C.shape[0], m)))) 
        d0 = I.d

        # case 1: y[index] >= 0
        # C1 = sp.hstack((Z, sp.csc_matrix((m, n)), -np.identity(m, dtype=dtype)))
        C1 = sp.hstack((sp.csc_array((m, n+I.nZVars), dtype=dtype), -np.identity(m, dtype=dtype)))
        d1 = np.zeros(m, dtype=dtype)

        # case 2: y[index] >= x[index]
        C2 = sp.hstack((Z, I.X(map), -A2[map, :]))
        d2 = -I.c(map).reshape(-1)

        # case 3: y[index] <= (u / (u - l)) * (x - l)
        a = u / (u - l)
        b = a * l

        C3 = sp.hstack((Z, -a.reshape(-1, 1) * I.X(map), A2[map, :]))
        d3 = a * I.c(map).reshape(-1) - b

        new_A = A
        new_C = sp.vstack([C0, C1, C2, C3]).tocsc()
        new_d = np.hstack([d0, d1, d2, d3])

        new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype)])
        new_pred_ub = np.hstack([I.pred_ub, u])
        new_pred_depth = np.hstack((I.pred_depth+1, np.zeros(m, dtype=dtype)))

        return SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
    

    def addConstraints_sparseimagestar(I, map, l, u, milp=False):

        N = I.num_pixel
        m = len(map)
        h, w, c, n = I.height, I.width, I.num_channel, I.num_pred
        dtype = I.V.dtype

        h_map, w_map, c_map = I.V.index_to3D(map)
        
        new_c = copy.deepcopy(I.c)
        Ic = np.zeros(m, dtype=dtype)
        for i in range(m):
            Ic[i] = I.c[h_map[i], w_map[i], c_map[i]]
            new_c[h_map[i], w_map[i], c_map[i]] = 0

        new_V = I.V.resetRows_hwc(h_map, w_map, c_map)
        new_V.num_pred += m
        n_ = n
        for i in range(m):
            im3d = SparseImage3D(h, w, c, n_)

            row = np.array([h_map[i]]).astype(np.ushort)
            col = np.array([w_map[i]]).astype(np.ushort)
            data = np.ones(1, dtype=dtype)

            im2d = sp.coo_array(
                (data, (row, col)), shape=(h, w)
            )
            im3d.append(im2d, c_map[i], n_)
            new_V.append(im3d)
            n_ += 1

        E = np.identity(m, dtype=dtype)
        # E = sp.eye(m, dtype=dtype)
        V1 = I.V.getRows_2D(map, n)

        # case 1: y[index] >= 0
        C1 = sp.hstack([sp.csr_matrix((m, n),dtype=dtype), -E])
        d1 = np.zeros(m, dtype=dtype)

        # case 2: y[index] >= x[index]
        C2 = sp.hstack([V1, -E])
        d2 = -Ic

        # case 3: y[index] <= (u / (u - l)) * (x - l)
        a = u / (u - l)
        b = a * l

        # C3 = sp.hstack([(-a[:, None] * V1), E])
        C3 = sp.hstack([V1.multiply(-a[:, None]), E])
        d3 = a * Ic - b

        if I.C.nnz > 0:
            C0 = sp.hstack((I.C, sp.csr_matrix((I.C.shape[0], m)))) 
            d0 = I.d

            new_C = sp.vstack([C0, C1, C2, C3]).tocsr()
            new_d = np.hstack([d0, d1, d2, d3])
        else:
            new_C = sp.vstack([C1, C2, C3]).tocsr()
            new_d = np.hstack([d1, d2, d3])

        new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype)])
        new_pred_ub = np.hstack([I.pred_ub, u])

        return SparseImageStar(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub)
    
    # def addConstraints_sparseimagestar_by_dense(I, map, l, u):

    #     N = I.num_pixel
    #     m = len(map) 
    #     n = I.num_pred
    #     dtype = I.V.dtype
        
    #     Ic = I.c.reshape(-1)
    #     new_c = Ic.copy() #copy.deepcopy(Ic)
    #     new_c[map] = 0
    #     new_c.reshape(I.height, I.weight, I.num_channel)

    #     VD = I.V.to_dense()
    #     V1 = VD.reshape(N, n ).copy #copy.deepcopy(VD).reshape(N, n)

    #     V2 = np.zeros([N, m], dtype=dtype)
    #     for i in range(m):
    #         V2[map[i], i] = 1
    #     new_V = np.hstack([V1, V2])

    #     C0 = sp.hstack((I.C, sp.csc_matrix((I.C.shape[0], m)))) 
    #     d0 = I.d

    #     # case 1: y[index] >= 0
    #     C1 = sp.hstack([sp.csc_matrix((m, n)), -np.identity(m, dtype=dtype)])
    #     d1 = np.zeros(m, dtype=dtype)

    #     # case 2: y[index] >= x[index]
    #     C2 = sp.hstack([VD[map, :], -V2[map, :]])
    #     d2 = -Ic[map].reshape(-1)

    #     # case 3: y[index] <= (u / (u - l)) * (x - l)
    #     a = u / (u - l)
    #     b = a * l

    #     # C3 = sp.hstack([(-a[:, None] * VD[map, :]), V2[map, :]])
    #     C3 = sp.hstack([(-a[:, None] * VD[map, :]), V2[map, :]])
    #     d3 = a * Ic[map] - b

    #     new_C = sp.vstack([C0, C1, C2, C3]).tocsc()
    #     new_d = np.hstack([d0, d1, d2, d3])

    #     new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype)])
    #     new_pred_ub = np.hstack([I.pred_ub, u])

    #     return SparseImageStar(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub)
    
    def addConstraints_sparseimagestar2d_coo(I, map, l, u, milp=False):
        m = len(map)
        if m == 0:
            return I

        N = I.V.shape[0]
        n = I.num_pred
        dtype = I.V.dtype
        out_shape = copy.deepcopy(I.shape)
        is_dense = isinstance(I.V, np.ndarray)

        if is_dense:
            V1 = copy.deepcopy(I.V)
            V1[map, :] = 0
            new_V = np.hstack([V1, np.zeros((N, m), dtype=dtype)])
            new_V[map, n:] = np.eye(m, dtype=dtype)
            Ic = I.V[map, 0]
            V1 = I.V[map, 1:]
        else:
            Ic = I.c[map]
            new_c = copy.deepcopy(I.c)
            new_c[map] = 0
            V = I.reset_rows_coo(map)
            V.data = np.hstack([V.data, np.ones(m, dtype=dtype)])
            V.row = np.hstack([V.row, map])
            V.col = np.hstack([V.col, np.arange(m, dtype=np.int32)+V.shape[1]])
            V._shape = (N, n+m)
            new_V = V
            V1 = I.getRows(map).tocoo(copy=False)

        # Triangular area relaxation for ReLU function
        # Apply 3 linear constraints for each neuron with (l < 0) and (u > 0)
        # case 1: y[map] >= 0
        # case 2: y[map] >= x[map]
        # case 3: y[map] <= (u / (u - l)) * (x - l)
        d1 = np.zeros(m, dtype=dtype)
        d2 = -Ic
        a = u / (u - l)
        b = a * l
        d3 = a * Ic - b

        eye_data = np.ones(m, dtype=dtype)
        eye_col = np.arange(m, dtype=np.int32) + n
        eye_row = np.arange(m, dtype=np.int32)

        if is_dense:
            V1 = sp.coo_array(V1)
            V3 = sp.coo_array(np.multiply(V1, -a[:, None]))
        else:
            V3 = sp.coo_array((-a[V1.row] * V1.data, (V1.row, V1.col)), shape=V1.shape)

        data = np.hstack([-eye_data, V1.data, -eye_data, V3.data, eye_data])
        row = np.hstack([eye_row, V1.row + m, eye_row + m, V3.row + 2*m, eye_row + 2*m])
        col = np.hstack([eye_col, V1.col, eye_col, V3.col, eye_col])
        C = sp.csr_array((data, (row, col)), shape=(3*m, n+m), copy=False)

        if I.C.nnz > 0:
            data = np.hstack([I.C.data, C.data])
            indices = np.hstack([I.C.indices, C.indices])
            indptr = np.hstack([I.C.indptr, C.indptr[1:]+I.C.nnz])
            new_C = sp.csr_array((data, indices, indptr), shape=(I.C.shape[0]+C.shape[0], C.shape[1]), copy=False)
            new_d = np.hstack([I.d, d1, d2, d3])
        else:
            new_C = C
            new_d = np.hstack([d1, d2, d3])

        new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype)])
        new_pred_ub = np.hstack([I.pred_ub, u])

        if is_dense:
            return SparseImageStar2DCOO(new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)
        return SparseImageStar2DCOO(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)
    
    def addConstraints_sparseimagestar2d_coo2(I, map, l, u, milp=False):
        """
        COO-native implementation with the same logic as addConstraints_sparseimagestar2d_csr2.
        Includes MILP support and returns SparseImageStar2DCOO.
        """
        assert isinstance(I, SparseImageStar2DCOO), 'error: Input set must be of type SparseImageStar2DCOO'

        m = len(map)
        if m == 0:
            return I

        N = I.V.shape[0]
        n = I.num_pred
        dtype = I.V.dtype
        out_shape = copy.deepcopy(I.shape)
        is_dense = isinstance(I.V, np.ndarray)

        def _merge_existing_constraints(C_add, d_add, n_new_pred):
            if I.C.nnz == 0:
                return C_add, d_add
            C0 = sp.csr_array((I.C.data, I.C.indices, I.C.indptr), shape=(I.C.shape[0], n_new_pred))
            C_new = SparseImageStar2DCSR.vstack_csr(A=C0, B=C_add, A_shape=C0.shape, B_shape=C_add.shape)
            d_new = np.hstack([I.d, d_add])
            return C_new, d_new

        def _update_predicate_layout(S):
            pred_layout = getattr(I, "pred_layout", None)
            if pred_layout is None:
                return S
            pred_layout = copy.deepcopy(pred_layout)
            if milp:
                pred_layout.add_relu_bigM_block(m)
            else:
                pred_layout.add_y_block(m)
            S.pred_layout = pred_layout
            return S

        # -------------------------
        # For Dense Case
        # -------------------------
        if is_dense:
            # Keep the same dense handling pattern as coo/csr2
            new_V = copy.deepcopy(I.V)
            new_V[map, :] = 0 # reset rows for neurons in map

            if milp:
                new_V = np.hstack([new_V, np.zeros((N, m + m), dtype=dtype)])
                y_col_start = 1 + n
                for i in range(m):
                    new_V[map[i], y_col_start + i] = 1.0

                cx = I.V[map, 0]
                Vx = I.V[map, 1:1 + n]
                new_npred = n + 2 * m

                Z_mn = sp.csr_array((m, n), dtype=dtype)
                I_m = SparseImageStar2DCSR.csr_eye(m, dtype=dtype)
                Z_mm = sp.csr_array((m, m), dtype=dtype)

                C1 = SparseImageStar2DCSR.hstack_csr(A=Z_mn, B=-I_m, A_shape=Z_mn.shape, B_shape=I_m.shape)
                C1 = SparseImageStar2DCSR.hstack_csr(A=C1, B=Z_mm, A_shape=C1.shape, B_shape=Z_mm.shape)
                d1 = np.zeros(m, dtype=dtype)

                Vx_csr = sp.csr_array(Vx, dtype=dtype)
                C2 = SparseImageStar2DCSR.hstack_csr(
                    A=Vx_csr,
                    B=-I_m,
                    A_shape=Vx_csr.shape,
                    B_shape=I_m.shape
                )
                C2 = SparseImageStar2DCSR.hstack_csr(A=C2, B=Z_mm, A_shape=C2.shape, B_shape=Z_mm.shape)
                d2 = -cx

                Zu = sp.diags(u, 0, format='csr', dtype=dtype)
                C3 = SparseImageStar2DCSR.hstack_csr(A=Z_mn, B=I_m, A_shape=Z_mn.shape, B_shape=I_m.shape)
                C3 = SparseImageStar2DCSR.hstack_csr(A=C3, B=-Zu, A_shape=C3.shape, B_shape=Zu.shape)
                d3 = np.zeros(m, dtype=dtype)

                Zl = sp.diags(l, 0, format='csr', dtype=dtype)
                C4 = SparseImageStar2DCSR.hstack_csr(
                    A=-Vx_csr,
                    B=I_m,
                    A_shape=Vx_csr.shape,
                    B_shape=I_m.shape
                )
                C4 = SparseImageStar2DCSR.hstack_csr(A=C4, B=-Zl, A_shape=C4.shape, B_shape=Zl.shape)
                d4 = cx - l

                C_add = SparseImageStar2DCSR.vstack_csr(
                    A=SparseImageStar2DCSR.vstack_csr(A=C1, B=C2, A_shape=C1.shape, B_shape=C2.shape),
                    B=SparseImageStar2DCSR.vstack_csr(A=C3, B=C4, A_shape=C3.shape, B_shape=C4.shape),
                    A_shape=(C1.shape[0] + C2.shape[0], new_npred),
                    B_shape=(C3.shape[0] + C4.shape[0], new_npred)
                )
                d_add = np.hstack([d1, d2, d3, d4])
                new_C, new_d = _merge_existing_constraints(C_add, d_add, new_npred)

                new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype), np.zeros(m, dtype=dtype)])
                new_pred_ub = np.hstack([I.pred_ub, u, np.ones(m, dtype=dtype)])

                S = SparseImageStar2DCOO(new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)
                return _update_predicate_layout(S)

            d1 = np.zeros(m, dtype=dtype)
            d2 = -I.V[map, 0]
            a = u / (u - l)
            b = a * l
            d3 = a * I.V[map, 0] - b

            new_V = np.hstack([new_V, np.zeros((N, m), dtype=dtype)])
            new_V[map, n:] = np.eye(m, dtype=dtype)
            V1 = I.V[map, 1:]

            C1 = np.hstack([np.zeros((m, n), dtype=dtype), -np.eye(m, dtype=dtype)])
            C2 = np.hstack([V1, -np.eye(m, dtype=dtype)])
            C3 = np.hstack([(-a[:, None] * V1), np.eye(m, dtype=dtype)])
            C_new = sp.csr_array(np.vstack([C1, C2, C3]), dtype=dtype)
            d_new = np.hstack([d1, d2, d3])

            new_C, new_d = _merge_existing_constraints(C_new, d_new, C_new.shape[1])

            new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype)])
            new_pred_ub = np.hstack([I.pred_ub, u])
            S = SparseImageStar2DCOO(new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)
            return _update_predicate_layout(S)

        # -------------------------
        # For Sparse Case
        # -------------------------
        new_c = copy.deepcopy(I.c)
        new_c[map] = 0

        V0 = I.reset_rows_coo(map)
        Y = sp.coo_array(
            (np.ones(m, dtype=dtype), (np.asarray(map, dtype=np.int32), np.arange(m, dtype=np.int32))),
            shape=(N, m)
        )
        new_V = sp.coo_array(
            (np.hstack([V0.data, Y.data]), (np.hstack([V0.row, Y.row]), np.hstack([V0.col, Y.col]))),
            shape=(N, n + m),
            dtype=dtype
        )

        if milp:
            new_npred = n + 2 * m
            new_V = sp.coo_array((new_V.data, (new_V.row, new_V.col)), shape=(N, new_npred), dtype=dtype)
        else:
            new_npred = n + m

        Ic = I.c[map]
        # scipy.sparse.coo_array does not support direct row slicing
        Vx = I.V.tocsr(copy=False)[map, :]

        Zmn = sp.csr_array((m, n), dtype=dtype)
        Im = SparseImageStar2DCSR.csr_eye(m, dtype=dtype)
        Zmm = sp.csr_array((m, m), dtype=dtype)

        if not milp:
            a = u / (u - l)

            C1 = SparseImageStar2DCSR.hstack_csr(A=Zmn, B=-Im, A_shape=Zmn.shape, B_shape=Im.shape)
            C2 = SparseImageStar2DCSR.hstack_csr(A=Vx, B=-Im, A_shape=Vx.shape, B_shape=Im.shape)
            V3 = Vx.multiply((-a).reshape(-1, 1)).tocsr(copy=False)
            C3 = SparseImageStar2DCSR.hstack_csr(A=V3, B=Im, A_shape=V3.shape, B_shape=Im.shape)

            C_add = SparseImageStar2DCSR.vstack_csr(
                A=SparseImageStar2DCSR.vstack_csr(A=C1, B=C2, A_shape=C1.shape, B_shape=C2.shape),
                B=C3,
                A_shape=(C1.shape[0] + C2.shape[0], new_npred),
                B_shape=C3.shape
            )
            d_add = np.hstack([
                np.zeros(m, dtype=dtype),
                -Ic,
                a * Ic - a * l
            ])

            new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype)])
            new_pred_ub = np.hstack([I.pred_ub, u])
        else:
            Zu = SparseImageStar2DCSR.csr_diag(u, dtype=dtype)
            Zl = SparseImageStar2DCSR.csr_diag(l, dtype=dtype)

            C1 = SparseImageStar2DCSR.hstack_csr(A=Zmn, B=-Im, A_shape=Zmn.shape, B_shape=Im.shape)
            C1 = SparseImageStar2DCSR.hstack_csr(A=C1, B=Zmm, A_shape=C1.shape, B_shape=Zmm.shape)
            C2 = SparseImageStar2DCSR.hstack_csr(A=Vx, B=-Im, A_shape=Vx.shape, B_shape=Im.shape)
            C2 = SparseImageStar2DCSR.hstack_csr(A=C2, B=Zmm, A_shape=C2.shape, B_shape=Zmm.shape)
            C3 = SparseImageStar2DCSR.hstack_csr(A=Zmn, B=Im, A_shape=Zmn.shape, B_shape=Im.shape)
            C3 = SparseImageStar2DCSR.hstack_csr(A=C3, B=-Zu, A_shape=C3.shape, B_shape=Zu.shape)
            C4 = SparseImageStar2DCSR.hstack_csr(A=-Vx, B=Im, A_shape=(-Vx).shape, B_shape=Im.shape)
            C4 = SparseImageStar2DCSR.hstack_csr(A=C4, B=-Zl, A_shape=C4.shape, B_shape=Zl.shape)

            C_add = SparseImageStar2DCSR.vstack_csr(
                A=SparseImageStar2DCSR.vstack_csr(A=C1, B=C2, A_shape=C1.shape, B_shape=C2.shape),
                B=SparseImageStar2DCSR.vstack_csr(A=C3, B=C4, A_shape=C3.shape, B_shape=C4.shape),
                A_shape=(C1.shape[0] + C2.shape[0], new_npred),
                B_shape=(C3.shape[0] + C4.shape[0], new_npred)
            )
            d_add = np.hstack([
                np.zeros(m, dtype=dtype),
                -Ic,
                np.zeros(m, dtype=dtype),
                Ic - l
            ])

            new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype), np.zeros(m, dtype=dtype)])
            new_pred_ub = np.hstack([I.pred_ub, u, np.ones(m, dtype=dtype)])

        new_C, new_d = _merge_existing_constraints(C_add, d_add, new_npred)
        S = SparseImageStar2DCOO(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)
        return _update_predicate_layout(S)
    
    def addConstraints_sparseimagestar2d_csr2(I, map, l, u, milp=False):
        """
        Over-approximate the ReLU function by linear constraints for neurons with (l < 0) and (u > 0) in SparseImageStar2DCSR format.
        This method is similar to addConstraints_sparseimagestar2d_csr but it does not convert to COO format for V1 and C3, 
        which can be more efficient for large sparse matrices. It directly constructs the CSR format for the new constraints.
        Args:
        - I: input star set before ReLU layer
        - map: indices of neurons with (l < 0) and (u > 0)
        - l: lower bound vector of the neurons in map
        - u: upper bound vector of the neurons in map
        - milp: whether to apply MILP constraints (big-M method) for over-approximation (default: False)
        Return:
        - output star set after adding constraints for ReLU layer
        """
        assert isinstance(I, SparseImageStar2DCSR), "error: Input set must be of type SparseImageStar2DCSR"

        m = len(map)
        if m == 0:
            return I

        N = I.V.shape[0]

        n = I.num_pred
        dtype = I.V.dtype
        out_shape = copy.deepcopy(I.shape)
        is_dense = isinstance(I.V, np.ndarray)

        def _merge_existing_constraints(C_add, d_add, n_new_pred):
            if I.C.nnz == 0:
                return C_add, d_add
            C0 = SparseImageStar2DCSR.csr_extend_ncols(I.C.tocsr(copy=False), n_new_pred, dtype=dtype)
            C_new = SparseImageStar2DCSR.vstack_csr(A=C0, B=C_add, A_shape=C0.shape, B_shape=C_add.shape)
            d_new = np.hstack([I.d, d_add])
            return C_new, d_new

        # -------------------------
        # For Dense Case
        # -------------------------
        if is_dense:
            # Keep the same as addConstraints_sparseimagestar2d_csr for dense case
            new_V = copy.deepcopy(I.V)
            new_V[map, :] = 0

            if milp:
                # MILP constraints (big-M method) for over-approximation
                # Apply 4 linear constraints for each neuron with (l < 0) and (u > 0)
                # case 1: y[map] >= 0 <=> -y[map] <= 0
                # case 2: y[map] >= x[map] <=> V1*alpha - y[map] <= -Ic
                # case 3: y[map] <= u * z <=> y[map] - u * z <= 0
                # case 4: y[map] <= x[map] - l * (1 - z) <=> -V1*alpha + y[map] - l * z <= Ic - l, where z is a binary variable (z in {0, 1})

                # Build new_V: add y (m) and z (m) predicate vars
                new_V = np.hstack([new_V, np.zeros((N, m + m), dtype=dtype)])  # add y,z predicate columns
                y_col_start = 1 + n
                for i in range(m):
                    new_V[map[i], y_col_start + i] = 1.0

                # Pre-activation x at mapped indices: x = cx + Vx*alpha
                cx = I.V[map, 0]           # (m,)
                Vx = I.V[map, 1:1+n]       # (m,n) dense here
                new_npred = n + 2*m

                # Build MILP constraint blocks in sparse CSR
                # Variables ordering in constraints is: [alpha (n) | y (m) | z (m)]
                Z_mn = sp.csr_array((m, n), dtype=dtype)
                I_m = SparseImageStar2DCSR.csr_eye(m, dtype=dtype)
                Z_mm = sp.csr_array((m, m), dtype=dtype)

                # case 1: y >= 0  <=>  -y <= 0
                C1 = SparseImageStar2DCSR.hstack_csr(A=Z_mn, B=-I_m, A_shape=Z_mn.shape, B_shape=I_m.shape)
                C1 = SparseImageStar2DCSR.hstack_csr(A=C1, B=Z_mm, A_shape=C1.shape, B_shape=Z_mm.shape)
                d1 = np.zeros(m, dtype=dtype)

                # case 2: y >= x  <=>  Vx*alpha - y <= -cx
                Vx_csr = sp.csr_array(Vx, dtype=dtype)
                C2 = SparseImageStar2DCSR.hstack_csr(
                    A=Vx_csr,
                    B=-I_m,
                    A_shape=Vx_csr.shape,
                    B_shape=I_m.shape
                )
                C2 = SparseImageStar2DCSR.hstack_csr(A=C2, B=Z_mm, A_shape=C2.shape, B_shape=Z_mm.shape)
                d2 = -cx

                # case 3: y <= u z  <=>  y - u z <= 0
                Zu = sp.diags(u, 0, format="csr", dtype=dtype)
                C3 = SparseImageStar2DCSR.hstack_csr(A=Z_mn, B=I_m, A_shape=Z_mn.shape, B_shape=I_m.shape)
                C3 = SparseImageStar2DCSR.hstack_csr(A=C3, B=-Zu, A_shape=C3.shape, B_shape=Zu.shape)
                d3 = np.zeros(m, dtype=dtype)

                # case 4: y <= x - l(1-z)  <=>  -Vx*alpha + y - l z <= cx - l
                Zl = sp.diags(l, 0, format="csr", dtype=dtype)
                C4 = SparseImageStar2DCSR.hstack_csr(
                    A=-Vx_csr,
                    B=I_m,
                    A_shape=Vx_csr.shape,
                    B_shape=I_m.shape
                )
                C4 = SparseImageStar2DCSR.hstack_csr(A=C4, B=-Zl, A_shape=C4.shape, B_shape=Zl.shape)
                d4 = cx - l

                C_add = SparseImageStar2DCSR.vstack_csr(
                    A=SparseImageStar2DCSR.vstack_csr(A=C1, B=C2, A_shape=C1.shape, B_shape=C2.shape),
                    B=SparseImageStar2DCSR.vstack_csr(A=C3, B=C4, A_shape=C3.shape, B_shape=C4.shape),
                    A_shape=(C1.shape[0] + C2.shape[0], new_npred),
                    B_shape=(C3.shape[0] + C4.shape[0], new_npred)
                )
                d_add = np.hstack([d1, d2, d3, d4])
                new_C, new_d = _merge_existing_constraints(C_add, d_add, new_npred)

                # predicate bounds: alpha as-is, y in [0,u], z in [0,1] (binary)
                new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype), np.zeros(m, dtype=dtype)])
                new_pred_ub = np.hstack([I.pred_ub, u, np.ones(m, dtype=dtype)])

                # IMPORTANT: store which variables are binary (the last m)
                # e.g., pred_layout.add_relu_bigM_block(m) or keep bin_idx = range(n+m, n+2m)
                pred_layout = copy.deepcopy(getattr(I, "pred_layout", None))
                if pred_layout is not None:
                    pred_layout.add_relu_bigM_block(m)

                # Dense constructor form does not carry pred_layout.
                return SparseImageStar2DCSR(new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)

            # Triangular area relaxation for ReLU function
            # Apply 3 linear constraints for each neuron with (l < 0) and (u > 0)
            # case 1: y[map] >= 0
            # case 2: y[map] >= x[map]
            # case 3: y[map] <= (u / (u - l)) * (x - l)
            new_V = np.hstack([new_V, np.zeros((N, m), dtype=dtype)])
            new_V[map, n:] = np.eye(m, dtype=dtype)  # (as in your code)
            Ic = I.V[map, 0]
            V1 = I.V[map, 1:]

            d1 = np.zeros(m, dtype=dtype)
            d2 = -Ic
            a = u / (u - l)
            b = a * l
            d3 = a * Ic - b

            # build C as CSR from dense blocks (still fine for dense mode)
            Z_mn = np.zeros((m, n), dtype=dtype)
            I_m = np.eye(m, dtype=dtype)

            C1 = np.hstack([Z_mn, -I_m])    # case 1: y[map] >= 0
            C2 = np.hstack([V1,  -I_m])     # case 2: y[map] >= x[map]
            C3 = np.hstack([(-a[:, None] * V1), I_m]) # case 3: y[map] <= (u / (u - l)) * (x - l)

            C_new = sp.csr_array(np.vstack([C1, C2, C3]), dtype=dtype)
            d_new = np.hstack([d1, d2, d3])
            if I.C.nnz > 0:
                C_new, d_new = _merge_existing_constraints(C_new, d_new, C_new.shape[1])

            new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype)])
            new_pred_ub = np.hstack([I.pred_ub, u])

            return SparseImageStar2DCSR(new_V, C_new, d_new, new_pred_lb, new_pred_ub, out_shape, copy_=False)

        # -------------------------
        # For Sparse Case
        # -------------------------
        new_c = copy.deepcopy(I.c)
        new_c[map] = 0

        # V0: generators with rows `map` zeroed
        V0 = I.reset_rows_csr(map)  # (N, n)

        # Append y-columns: one 1 per selected row; y_i is column i (local)
        Y = sp.csr_array(
            (np.ones(m, dtype=dtype), (np.asarray(map, dtype=np.int32), np.arange(m, dtype=np.int32))),
            shape=(N, m)
        )  # (N, m)

        if not milp:
            new_V = SparseImageStar2DCSR.hstack_csr(A=V0, B=Y, A_shape=V0.shape, B_shape=Y.shape)      # (N, n+m)
            new_npred = n + m
        else:
            # add z columns (all zero in V)
            Z0 = sp.csr_array((N, m), dtype=dtype)
            new_V = SparseImageStar2DCSR.hstack_csr(A=V0, B=Y, A_shape=V0.shape, B_shape=Y.shape)
            new_V = SparseImageStar2DCSR.hstack_csr(A=new_V, B=Z0, A_shape=new_V.shape, B_shape=Z0.shape)  # (N, n+2m)
            new_npred = n + 2*m

        Ic = I.c[map]                       # (m,)
        Vx = I.V.tocsr(copy=False)[map, :]  # (m, n)

        Zmn = sp.csr_array((m, n), dtype=dtype)
        Im = SparseImageStar2DCSR.csr_eye(m, dtype=dtype)
        Zmm = sp.csr_array((m, m), dtype=dtype)

        if not milp:
            # Triangle relaxation:
            # 1) -y <= 0
            # 2) Vx*alpha - y <= -Ic
            # 3) (-a*Vx)*alpha + y <= a*Ic - a*l
            a = u / (u - l)

            C1 = SparseImageStar2DCSR.hstack_csr(A=Zmn, B=-Im, A_shape=Zmn.shape, B_shape=Im.shape)
            C2 = SparseImageStar2DCSR.hstack_csr(A=Vx, B=-Im, A_shape=Vx.shape, B_shape=Im.shape)
            V3 = Vx.multiply((-a).reshape(-1, 1)).tocsr(copy=False)  # scales each row i by -a[i]
            C3 = SparseImageStar2DCSR.hstack_csr(A=V3, B=Im, A_shape=V3.shape, B_shape=Im.shape)

            C_add = SparseImageStar2DCSR.vstack_csr(
                A=SparseImageStar2DCSR.vstack_csr(A=C1, B=C2, A_shape=C1.shape, B_shape=C2.shape),
                B=C3,
                A_shape=(C1.shape[0] + C2.shape[0], new_npred),
                B_shape=C3.shape
            )
            d_add = np.hstack([np.zeros(m, dtype=dtype), -Ic, a * Ic - a * l])

            new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype)])
            new_pred_ub = np.hstack([I.pred_ub, u])

        else:
            # Big-M MILP:
            # 1) -y <= 0
            # 2) Vx*alpha - y <= -Ic
            # 3) y - u*z <= 0
            # 4) -Vx*alpha + y - l*z <= Ic - l
            Zu = SparseImageStar2DCSR.csr_diag(u, dtype=dtype)
            Zl = SparseImageStar2DCSR.csr_diag(l, dtype=dtype)

            C1 = SparseImageStar2DCSR.hstack_csr(A=Zmn, B=-Im, A_shape=Zmn.shape, B_shape=Im.shape)
            C1 = SparseImageStar2DCSR.hstack_csr(A=C1, B=Zmm, A_shape=C1.shape, B_shape=Zmm.shape)
            C2 = SparseImageStar2DCSR.hstack_csr(A=Vx, B=-Im, A_shape=Vx.shape, B_shape=Im.shape)
            C2 = SparseImageStar2DCSR.hstack_csr(A=C2, B=Zmm, A_shape=C2.shape, B_shape=Zmm.shape)
            C3 = SparseImageStar2DCSR.hstack_csr(A=Zmn, B=Im, A_shape=Zmn.shape, B_shape=Im.shape)
            C3 = SparseImageStar2DCSR.hstack_csr(A=C3, B=-Zu, A_shape=C3.shape, B_shape=Zu.shape)
            C4 = SparseImageStar2DCSR.hstack_csr(A=-Vx, B=Im, A_shape=(-Vx).shape, B_shape=Im.shape)
            C4 = SparseImageStar2DCSR.hstack_csr(A=C4, B=-Zl, A_shape=C4.shape, B_shape=Zl.shape)

            C_add = SparseImageStar2DCSR.vstack_csr(
                A=SparseImageStar2DCSR.vstack_csr(A=C1, B=C2, A_shape=C1.shape, B_shape=C2.shape),
                B=SparseImageStar2DCSR.vstack_csr(A=C3, B=C4, A_shape=C3.shape, B_shape=C4.shape),
                A_shape=(C1.shape[0] + C2.shape[0], new_npred),
                B_shape=(C3.shape[0] + C4.shape[0], new_npred)
            )
            d_add = np.hstack([np.zeros(m, dtype=dtype), -Ic, np.zeros(m, dtype=dtype), Ic - l])

            new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype), np.zeros(m, dtype=dtype)])
            new_pred_ub = np.hstack([I.pred_ub, u, np.ones(m, dtype=dtype)])

            # IMPORTANT: you must mark these last m vars as binary using your pred_layout.
            # (If you keep pred_layout inside the set, call pred_layout.add_relu_bigM_block(m).)

        new_C, new_d = _merge_existing_constraints(C_add, d_add, new_npred)

        # Update pred_layout if present
        pred_layout = getattr(I, "pred_layout", None)
        if pred_layout is not None:
            pred_layout = copy.deepcopy(pred_layout)
            if not milp:
                pred_layout.add_y_block(m)
            else:
                pred_layout.add_relu_bigM_block(m)

        if pred_layout is None:
            return SparseImageStar2DCSR(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)
        return SparseImageStar2DCSR(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub, pred_layout, out_shape, copy_=False)

    def addConstraints_sparseimagestar2d_csr(I, map, l, u):
        """
        Over-approximate the ReLU function by linear constraints for neurons with (l < 0) and (u > 0) in SparseImageStar2DCSR format.
        No MILP constraints are supported in this method, it directly applies the triangular area relaxation for ReLU function.
        Args:
        - I: input star set before ReLU layer
        - map: indices of neurons with (l < 0) and (u > 0)
        - l: lower bound vector of the neurons in map
        - u: upper bound vector of the neurons in map
        - milp: whether to apply MILP constraints (big-M method) for over-approximation (default: False)
        Return:
        - output star set after adding constraints for ReLU layer
        """
        assert isinstance(I, SparseImageStar2DCSR), "error: Input set must be of type SparseImageStar2DCSR"
        
        m = len(map)
        if m == 0:
            return I
        
        N = I.V.shape[0]
        n = I.num_pred
        dtype = I.V.dtype
        out_shape = copy.deepcopy(I.shape)
        is_dense = isinstance(I.V, np.ndarray)

        if is_dense:
            new_V = copy.deepcopy(I.V)
            new_V[map, :] = 0
            new_V = np.hstack([new_V, np.zeros((N, m), dtype=dtype)])
            new_V[map, n:] = np.eye(m, dtype=dtype)
            Ic = I.V[map, 0]
            V1 = I.V[map, 1:]
        else:
            new_c = copy.deepcopy(I.c)
            new_c[map] = 0
            V = I.reset_rows_csr(map).tocoo(copy=False)
            V.data = np.hstack([V.data, np.ones(m, dtype=dtype)])
            V.row = np.hstack([V.row, map])
            V.col = np.hstack([V.col, np.arange(m, dtype=np.int32) + V.shape[1]])
            V._shape = (N, n + m)
            new_V = V.tocsr(copy=False)
            Ic = I.c[map]
            V1 = I.V[map, :].tocoo(copy=False)

        # Triangular area relaxation for ReLU function
        # Apply 3 linear constraints for each neuron with (l < 0) and (u > 0)
        # case 1: y[index] >= 0
        # case 2: y[index] >= x[index]
        # case 3: y[index] <= (u / (u - l)) * (x - l)
        d1 = np.zeros(m, dtype=dtype)
        d2 = -Ic
        a = u / (u - l)
        b = a * l
        d3 = a * Ic - b
        if not is_dense:
            V1 = V1.tocoo(copy=False)
            V3 = sp.coo_array((-a[V1.row] * V1.data,
                               (V1.row, V1.col)),
                               shape=V1.shape)
        else:
            V1 = sp.coo_array(V1)
            V3 = sp.coo_array(np.multiply(V1, -a[:, None]))

        eye_data = np.ones(m, dtype=dtype)
        eye_col = np.arange(m, dtype=np.int32) + n
        eye_row = np.arange(m, dtype=np.int32)

        data = np.hstack([-eye_data, V1.data, -eye_data, V3.data, eye_data])
        row = np.hstack([eye_row, V1.row + m, eye_row + m, V3.row + 2*m, eye_row + 2*m])
        col = np.hstack([eye_col, V1.col, eye_col, V3.col, eye_col])
        C = sp.csr_array((data, (row, col)), shape=(3*m, n+m), copy=False)

        if I.C.nnz > 0:
            data = np.hstack([I.C.data, C.data])
            indices = np.hstack([I.C.indices, C.indices])
            indptr = np.hstack([I.C.indptr, C.indptr[1:] + I.C.nnz])
            new_C = sp.csr_array((data, indices, indptr), shape=(I.C.shape[0] + C.shape[0], C.shape[1]), copy=False)
            new_d = np.hstack([I.d, d1, d2, d3])
        else:
            new_C = C
            new_d = np.hstack([d1, d2, d3])

        new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype)])
        new_pred_ub = np.hstack([I.pred_ub, u])

        if is_dense:
            return SparseImageStar2DCSR(new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)
        return SparseImageStar2DCSR(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)
    
    def stepReachApprox(In, lp_solver='gurobi', RF=0.0, DR=0, milp=False, show=False):
        """
        Approx reachability using multi stepReachApprox
        Args:
            @In: a single input set
            @lp_solver: lp_solver

        Returns:
            output set

        Author: Sung Woo Choi, Date: 09/12/2023
        """

        I = copy.deepcopy(In)

        if isinstance(I, ImageStar):
            I = I.toStar(copy_=False)

        if show:
            print('Applying approximate reachability on \'poslin\' or \'relu\' activation function')

        l, u = I.estimateRanges()

        if (l > 0).all():
            return I
        elif (u < 0).all():
            return I.resetRows(np.arange(I.dim))
        
        if lp_solver == 'estimate':
            # approximation with estimate methods
            I, l, u, map = PosLin.relax_by_area(I=I, l=l, u=u, lp_solver=lp_solver, RF=1.0, show=show)

        # no relaxation; get bounds using LP solver
        elif RF == 0.0: # and lp_solver != 'estimate':
            if show:
                print('Finding lower and upper bounds of neurons with LP solver')
            I, l, u, map = PosLin.approx(I=I, l=l, u=u, lp_solver=lp_solver, show=show)

        # applying relaxation
        else:
            if show:
                print('Estimating lower and upper bounds of neurons')
 
            assert RF >= 0.0 and RF <= 1.0, \
            'error: relaxation factor should be between 0.0 and 1.0, i.e. RF in [0.0, 1.0]' 

            # applying partial relaxation and partial LP solver
            I, l, u, map = PosLin.relax_by_area(I=I, l=l, u=u, lp_solver=lp_solver, RF=RF, show=show)

        if isinstance(In, Star):
            return PosLin.addConstraints(I=I, map=map, l=l, u=u, milp=milp)

        elif isinstance(In, ImageStar):
            S = PosLin.addConstraints(I=I, map=map, l=l, u=u, milp=milp)
            if In.V.ndim == 4:
                new_V = S.V.reshape(In.height, In.width, In.num_channel, S.nVars + 1)
            else:
                new_V = S.V
            return ImageStar(new_V, S.C, S.d, S.pred_lb, S.pred_ub)

        elif isinstance(In, SparseStar):
            S = PosLin.addConstraints_sparse(I=I, map=map, l=l, u=u, milp=milp)
            if DR > 0:
                if show:
                    if (S.pred_depth >= DR).any():
                        print('Applying depth reduction {}'.format(DR))
                S = S.depthReduction(DR=DR)
            return S
        
        elif isinstance(In, SparseImageStar):
            return PosLin.addConstraints_sparseimagestar(I=I, map=map, l=l, u=u, milp=milp)
        
        elif isinstance(In, SparseImageStar2DCOO):
            return PosLin.addConstraints_sparseimagestar2d_coo2(I=I, map=map, l=l, u=u, milp=milp)
        
        elif isinstance(In, SparseImageStar2DCSR):
            return PosLin.addConstraints_sparseimagestar2d_csr2(I=I, map=map, l=l, u=u, milp=milp)
        
        else:
            raise Exception(
                    'error: approximate reachaiblity of \'relu\' or \'poslin\' supports Star, SparseStar'
                )

    @staticmethod
    def reachApproxSingleInput(In, lp_solver='gurobi', RF=0.0, DR=0.0, milp=False, show=False):
        """
        Approx reachability using stepReach
        Args:
            @In: a single input set
            @lp_solver: lp_solver

        Returns:
            output set

        Author: Sung Woo Choi, Date: 09/12/2023
        """
        assert isinstance(In, Star) or isinstance(In, ImageStar) or \
            isinstance(In, SparseStar) or isinstance(In, SparseImageStar) or \
            isinstance(In, SparseImageStar2DCOO) or isinstance(In, SparseImageStar2DCSR), \
            f"error: approximate reachaiblity of \'relu\' or \'poslin\' supports Star, ImageStar, SparseStar, SparseImageStar but received In={type(In)}"

        return PosLin.stepReachApprox(In=In, lp_solver=lp_solver, RF=RF, DR=DR, milp=milp, show=show)

                
    @staticmethod
    def stepReachStarApprox(In, index, lp_solver='gurobi', show=False):

        """
        Approx reachability using stepReach
        Args:
            @In: a single input set
            @index: index of a neuron
            @lp_solver: lp_solver

        Returns:
            output set

        Author: Sung Woo Choi, Date: 09/12/2023
        """

        assert isinstance(In, Star), 'Input is not a Star set'

        l = In.getMin(index=index, lp_solver=lp_solver)
        # If the lower bound is greater than 0, then the ReLU function is linear and we can return the input set as it is.
        if l > 0:
            return In

        u = In.getMax(index=index, lp_solver=lp_solver)
        # If the upper bound is less than or equal to 0, then the ReLU function is linear and we can return the input set with the corresponding row zeroed out.
        if u <= 0:
            V = copy.deepcopy(In.V)
            V[index, :] = 0
            return Star(V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        if show:
            print('Add a new predicate variables at index = {}'.format(index))

        # Case when l < 0 and u > 0, we need to add new predicate variable y[index] for ReLU output and 
        # add linear constraints for the relationship between x[index] and y[index].
        n = In.nVars + 1
        dtype = In.V.dtype

        # y[index] >= 0
        C1 = np.zeros([1, n], dtype=dtype)
        C1[0, n-1] = -1
        d1 = 0

        # y[index] >= x[index]
        C2 = np.hstack([In.V[index, 1:n], -1]).reshape(1, -1)
        d2 = -In.V[index, 0]

        # y[index] <= ub * (x[index] - lb) / (ub - lb)
        a = -u / (u - l)
        C3 = np.hstack([a * In.V[index, 1:n], 1]).reshape(1, -1)
        d3 = a*(l - In.V[index, 0])

        if len(In.d) == 0: # for Star set as initial C and d might be []
            C0 = np.empty([0, n], dtype=dtype)
            d0 = np.empty([0], dtype=dtype)
        else:
            m = In.C.shape[0]
            C0 = np.hstack([In.C, np.zeros([m, 1], dtype=dtype)])
            d0 = In.d

        C = np.vstack([C0, C1, C2, C3])
        d = np.hstack([d0, d1, d2, d3])
        V = np.hstack([In.V, np.zeros([In.dim, 1], dtype=dtype)])
        V[index, :] = 0
        V[index, n] = 1
        pred_lb = np.hstack([In.pred_lb, 0])
        pred_ub = np.hstack([In.pred_ub, u])
        return Star(V, C, d, pred_lb, pred_ub)


    @staticmethod
    def reachApprox(In, lp_solver='gurobi', show=False):
        """
        Approximate reachability for Star set
        Args:
            @In: a single input set
            @lp_solver: lp_solver

        Returns:
            output set

        Author: Sung Woo Choi, Date: 09/12/2023
        """

        assert isinstance(In, Star), 'Input is not a Star set'

        l, u = In.estimateRanges()

        map = np.argwhere(u <= 0).reshape(-1)
        V = copy.deepcopy(In.V)
        V[map, :] = 0
        I = Star(V, In.C, In.d, In.pred_lb, In.pred_ub)

        map = np.argwhere((l < 0) & (u > 0)).reshape(-1)
        for i in range(len(map)):
            if show:
                print('Performing approximate PosLin operation on {} neuron'.format(map[i]))
            I = PosLin.stepReachStarApprox(In=I, index=int(map[i]), lp_solver=lp_solver, show=show)
        
        return I


        
