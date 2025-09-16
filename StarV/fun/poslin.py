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
"""

# !/usr/bin/python3
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
from StarV.set.imagestar import ImageStar
from StarV.set.sparsestar import SparseStar
from StarV.set.sparseimagestar import *
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR

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
                    S1 = S1.addConstraint(C, d)  # x <= 0
                    S1 = S1.resetRow(index)
                    S2 = S2.addConstraint(-C, d)  # x >= 0
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
        
        # else if RF == 0.0:
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

        
    def addConstraints(I, map, l, u):

        N = I.dim
        m = len(map) # number of neurons invovled
        dtype = I.V.dtype

        V1 = copy.deepcopy(I.V)
        V1[map, :] = 0
        V2 = np.zeros([N, m], dtype=dtype)
        for i in range(m):
            V2[map[i], i] = 1
        new_V = np.hstack([V1, V2])

        n = I.nVars
        if len(I.C) == 0:
            C0 = np.empty([0, n+m], dtype=dtype)
            d0 = np.empty([0], dtype=dtype)
        else:
            C0 = np.hstack([I.C, np.zeros([I.C.shape[0], m], dtype=dtype)])
            d0 = I.d

        # case 1: y[index] >= 0
        C1 = np.hstack([np.zeros([m, n], dtype=dtype), -np.identity(m, dtype=dtype)])
        d1 = np.zeros(m, dtype=dtype)

        # case 2: y[index] >= x[index]
        C2 = np.hstack([I.V[map, 1:n + 1], -V2[map, 0:m]])
        d2 = -I.V[map, 0] 

        # case 3: y[index] <= (u / (u - l)) * (x - l)
        a = u / (u - l)
        b = a * l

        C3 = np.hstack([(-a[:, None] * I.V[map, 1:n + 1]), V2[map, 0:m]])
        d3 = a * I.V[map, 0] - b

        new_C = np.vstack([C0, C1, C2, C3])
        new_d = np.hstack([d0, d1, d2, d3])

        new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype)])
        new_pred_ub = np.hstack([I.pred_ub, u])
        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)


    def addConstraints_sparse(I, map, l, u):
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
    

    def addConstraints_sparseimagestar(I, map, l, u):

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
    
    def addConstraints_sparseimagestar2d_coo(I, map, l, u):
        map = map.astype(np.int32)

        N = I.V.shape[0]
        m = len(map)
        n = I.num_pred
        dtype = I.V.dtype

        if isinstance(I.V, np.ndarray):
            V1 = copy.deepcopy(I.V)
            V1[map, :] = 0
            V2 = np.zeros([N, m], dtype=dtype)
            # V2[map, np.arange(m, dtype=np.int32)] = 1
            for i in range(m):
                V2[map[i], i] = 1
            new_V = np.hstack([V1, V2])

            # case 1: y[map] >= 0
            # case 2: y[map] >= x[map]
            # case 3: y[map] <= (u / (u - l)) * (x - l)
            Ic = I.V[map, 0]
            d1 = np.zeros(m, dtype=dtype)
            d2 = -Ic
            a = u / (u - l)
            b = a * l

            V1 = I.V[map, 1:]
            V3 = np.multiply(V1, -a[:, None])
            d3 = a * Ic - b

            V1 = sp.coo_array(V1)
            V3 = sp.coo_array(V3)

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
                indptr = np.hstack([I.C.indptr, C.indptr[1:]+I.C.nnz])
                new_C = sp.csr_array((data, indices, indptr), shape=(I.C.shape[0]+C.shape[0], C.shape[1]), copy=False)
                new_d = np.hstack([I.d, d1, d2, d3])
            else:
                new_C = C
                new_d = np.hstack([d1, d2, d3])

            new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype)])
            new_pred_ub = np.hstack([I.pred_ub, u])
            out_shape = copy.deepcopy(I.shape)

            return SparseImageStar2DCOO(new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)

        else:
        
            Ic = I.c[map]
            new_c = copy.deepcopy(I.c)
            new_c[map] = 0
            V = I.resetRows_V(map)

            V.data = np.hstack([V.data, np.ones(m, dtype=dtype)])
            V.row = np.hstack([V.row, map])
            V.col = np.hstack([V.col, np.arange(m, dtype=np.int32)+V.shape[1]])
            V._shape = (N, n+m)
            new_V = V

            V1 = I.getRows(map)
            eye_data = np.ones(m, dtype=dtype)
            eye_col = np.arange(m, dtype=np.int32) + n
            eye_row = np.arange(m, dtype=np.int32)

            # case 1: y[map] >= 0
            # case 2: y[map] >= x[map]
            # case 3: y[map] <= (u / (u - l)) * (x - l)
            d1 = np.zeros(m, dtype=dtype)
            d2 = -Ic
            a = u / (u - l)
            b = a * l
            V3 = V1.multiply(-a[:, None]).tocoo() # returns csr format
            d3 = a * Ic - b

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
            out_shape = copy.deepcopy(I.shape)

            return SparseImageStar2DCOO(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)
    
    # def addConstraints_sparseimagestar2d_coo(I, map, l, u):

    #     N = I.V.shape[0]
    #     m = len(map)
    #     n = I.num_pred
    #     dtype = l.dtype
        
    #     Ic = I.c[map]
    #     new_c = copy.deepcopy(I.c)
    #     new_c[map] = 0
    #     V1 = I.resetRows_V(map)
    #     V2 = np.zeros([N, m])
    #     for i in range(m):
    #         V2[map[i], i] = 1
    #     new_V = sp.hstack([V1, V2])

    #     E = sp.eye(m, dtype=dtype)
    #     V3 = I.getRows(map)

    #     # case 1: y[index] >= 0
    #     C1 = sp.hstack([sp.csr_matrix((m, n),dtype=dtype), -E])
    #     d1 = np.zeros(m)

    #     # case 2: y[index] >= x[index]
    #     C2 = sp.hstack([V3, -E])
    #     d2 = -Ic

    #     # case 3: y[index] <= (u / (u - l)) * (x - l)
    #     a = u / (u - l)
    #     b = a * l

    #     # C3 = sp.hstack([(-a[:, None] * V1), E])
    #     C3 = sp.hstack([V3.multiply(-a[:, None]), E])
    #     d3 = a * Ic - b

    #     if I.C.nnz > 0:
    #         C0 = sp.hstack((I.C, sp.csr_matrix((I.C.shape[0], m)))) 
    #         d0 = I.d

    #         new_C = sp.vstack([C0, C1, C2, C3]).tocsr()
    #         new_d = np.hstack([d0, d1, d2, d3])
    #     else:
    #         new_C = sp.vstack([C1, C2, C3]).tocsr()
    #         new_d = np.hstack([d1, d2, d3])

    #     new_pred_lb = np.hstack([I.pred_lb, np.zeros(m)])
    #     new_pred_ub = np.hstack([I.pred_ub, u])

    #     return SparseImageStar2DCOO(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub, I.shape)
    

    def addConstraints_sparseimagestar2d_csr(I, map, l, u):
        map = map.astype(np.int32)

        N = I.V.shape[0]
        m = len(map)
        n = I.num_pred
        dtype = I.V.dtype
        
        if isinstance(I.V, np.ndarray):
            V1 = copy.deepcopy(I.V)
            V1[map, :] = 0
            V2 = np.zeros([N, m], dtype=dtype)
            # V2[map, np.arange(m, dtype=np.int32)] = 1
            for i in range(m):
                V2[map[i], i] = 1
            new_V = np.hstack([V1, V2])

            # case 1: y[map] >= 0
            # case 2: y[map] >= x[map]
            # case 3: y[map] <= (u / (u - l)) * (x - l)
            Ic = I.V[map, 0]
            d1 = np.zeros(m, dtype=dtype)
            d2 = -Ic
            a = u / (u - l)
            b = a * l

            V1 = I.V[map, 1:]
            V3 = np.multiply(V1, -a[:, None])
            d3 = a * Ic - b

            V1 = sp.coo_array(V1)
            V3 = sp.coo_array(V3)

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
                indptr = np.hstack([I.C.indptr, C.indptr[1:]+I.C.nnz])
                new_C = sp.csr_array((data, indices, indptr), shape=(I.C.shape[0]+C.shape[0], C.shape[1]), copy=False)
                new_d = np.hstack([I.d, d1, d2, d3])
            else:
                new_C = C
                new_d = np.hstack([d1, d2, d3])

            new_pred_lb = np.hstack([I.pred_lb, np.zeros(m, dtype=dtype)])
            new_pred_ub = np.hstack([I.pred_ub, u])
            out_shape = copy.deepcopy(I.shape)

            return SparseImageStar2DCSR(new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)
        
        else:
            Ic = I.c[map]
            new_c = copy.deepcopy(I.c)
            new_c[map] = 0
            V = I.resetRows_V(map).tocoo(copy=False)

            V.data = np.hstack([V.data, np.ones(m, dtype=dtype)])
            V.row = np.hstack([V.row, map])
            V.col = np.hstack([V.col, np.arange(m, dtype=np.int32)+V.shape[1]])
            V._shape = (N, n+m)
            new_V = V.tocsr(copy=False)

            V1 = I.V[map, :].tocoo()
            eye_data = np.ones(m, dtype=dtype)
            eye_col = np.arange(m, dtype=np.int32) + n
            eye_row = np.arange(m, dtype=np.int32)

            # case 1: y[map] >= 0
            # case 2: y[map] >= x[map]
            # case 3: y[map] <= (u / (u - l)) * (x - l)
            d1 = np.zeros(m, dtype=dtype)
            d2 = -Ic
            a = u / (u - l)
            b = a * l
            V3 = I.V[map, :].multiply(-a[:, None]).tocoo()
            d3 = a * Ic - b

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
            out_shape = copy.deepcopy(I.shape)

            return SparseImageStar2DCSR(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape, copy_=False)
    
    def stepReachApprox(In, lp_solver='gurobi', RF=0.0, DR=0, show=False):
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

        if isinstance(I, Star):
            return PosLin.addConstraints(I=I, map=map, l=l, u=u)

        
        elif isinstance(I, ImageStar):
            S = I.toStar(copy_=False)
            S = PosLin.addConstraints(I=S, map=map, l=l, u=u)
            if I.V.ndim == 4:
                new_V = S.V.reshape(I.height, I.width, I.num_channel, S.nVars + 1)
            else:
                new_V = S.V
            return ImageStar(new_V, S.C, S.d, S.pred_lb, S.pred_ub)

        elif isinstance(I, SparseStar):
            S = PosLin.addConstraints_sparse(I=I, map=map, l=l, u=u)
            if DR > 0:
                if show:
                    if (S.pred_depth >= DR).any():
                        print('Applying depth reduction {}'.format(DR))
                S = S.depthReduction(DR=DR)
            return S
        
        elif isinstance(I, SparseImageStar):
            return PosLin.addConstraints_sparseimagestar(I=I, map=map, l=l, u=u)
        
        elif isinstance(I, SparseImageStar2DCOO):
            return PosLin.addConstraints_sparseimagestar2d_coo(I=I, map=map, l=l, u=u)
        
        elif isinstance(I, SparseImageStar2DCSR):
            return PosLin.addConstraints_sparseimagestar2d_csr(I=I, map=map, l=l, u=u)
        
        else:
            raise Exception(
                    'error: approximate reachaiblity of \'relu\' or \'poslin\' supports Star, SparseStar'
                )

    @staticmethod
    def reachApproxSingleInput(In, lp_solver='gurobi', RF=0.0, DR=0.0, show=False):
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

        return PosLin.stepReachApprox(In=In, lp_solver=lp_solver, RF=RF, DR=0, show=show)

                
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

        if l > 0:
            return In
        
        else:
            u = In.getMax(index=index, lp_solver=lp_solver)
            if u <= 0:
                V = copy.deepcopy(In.V)
                V[index, :] = 0
                return Star(V, In.C, In.d, In.pred_lb, In.pred_ub)
            
            else:
                if show:
                    print('Add a new predicate variables at index = {}'.format(index))

                n = In.nVars + 1
                dtype = In.V.dtype

                # y[index] >= 0
                C1 = np.zeros([1, n], dtype=dtype)
                C1[0, n-1] = -1
                d1 = 0

                # y[index] >= x[index]
                C2 = np.column_stack([In.V[index, 1:n], -1])
                d2 = -In.V[index, 0]

                # y[index] <= ub * (x[index] - lb) / (ub - lb)
                a = -u / (u - l)
                C3 = np.column_stack([a*In.V[index, 1:n], 1])
                d3 = a*(l - In.V[index, 0])

                if len(In.d) == 0: # for Star set as initial C and d might be []
                    C0 = np.empty([0, n], dtype=dtype)
                    d0 = np.empty([0], dtype=dtype)
                else:
                    m = In.C.shape[0]
                    C0 = np.column_stack([In.C, np.zeros([m, 1], dtype=dtype)])
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

        map = np.argwhere(u <= 0)
        V = copy.deepcopy(In.V)
        V[map, :] = 0
        I = Star(V, In.C, In.d, In.pred_lb, In.pred_ub)

        map = np.argwhere((l < 0) & (u > 0))
        for i in range(len(map)):
            if show:
                print('Performing approximate PosLin operation on {} neuron'.format(map[i]))
            I = PosLin.stepReachStarApprox(In=I, index=map[i], lp_solver=lp_solver, show=show)
        
        return I


        