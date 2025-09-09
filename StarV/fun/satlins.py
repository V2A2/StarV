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
SatLins Class
Yuntao Li, 1/18/2024
"""

from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
import copy
import multiprocessing
import ipyparallel

class SatLins(object):
    """
    SatLins class for computing reachable set of SatLins Transfer Function
    Yuntao Li, 1/18/2024
    """

    @staticmethod
    def evaluate(x):
        """
        evaluate method and reachability analysis with stars

        Args:
            @x = np.arrays

        Returns:
            -1, if n <= -1
            n, if -1 <= n <= 1
            1, if 1 <= n

        Author: Yuntao Li, Date: 1/18/2024
        """

        a = np.maximum(x, -1)
        b = np.minimum(a, 1)
        # print("\n b ------------------------ \n", b)
        return b


    @staticmethod
    def stepReach(*args):
        """
        stepReach method, compute reachable set for a single step

        Args:
            @I: single star set input
            @index: index of current x[index] of current step (should be the number from matlab - 1)

        Others:
            @xmin: min of x[index]
            @xmax: max of x[index]

        Returns:
            @S: star output set

        Author: Yuntao Li, Date: 1/18/2024
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

        if not isinstance(I, ProbStar) and not isinstance(I, Star):
            raise Exception('error: input is not a Star or ProbStar set, \
            type of input = {}'.format(type(I)))

        xmin = I.getMin(index, lp_solver)
        xmax = I.getMax(index, lp_solver)

        # ------------- case 1) only single set -------------
        if -1 <= xmin and xmax <= 1:
            S = []
            S.append(I) # -1 <= x <= 1
            return S

        # ------------- case 2) -------------
        if -1 <= xmin < 1 < xmax:
            C = np.zeros(I.dim,)
            C[index] = 1.0
            d = np.ones(1,)
            S1 = copy.deepcopy(I) 
            S2 = copy.deepcopy(I)

            # -1 <= x <= 1
            S1.addConstraint(-C, d) # x >= -1 ---> -x <= 1
            S1.addConstraint(C, d) # x <= 1

            S2.addConstraint(-C, -d) # x >= 1 ---> -x <= -1
            S2.resetRowWithUpdatedCenter(index, 1.0)

            S = []
            S.append(S1)
            S.append(S2)
            return S

        # ------------- case 3) -------------
        if xmin < -1 < xmax <= 1:
            C = np.zeros(I.dim,)
            C[index] = 1.0
            d = np.ones(1,)
            S1 = copy.deepcopy(I)
            S2 = copy.deepcopy(I)

            S1.addConstraint(C, -d)  # x <= -1
            S1.resetRowWithUpdatedCenter(index, -1.0)

            S2.addConstraint(-C, d) # x >= -1 -> -x <= 1

            S = []
            S.append(S1)
            S.append(S2)
            return S

        # ------------- case 4) -------------
        if xmin < -1 and xmax > 1:
            C = np.zeros(I.dim,)
            C[index] = 1.0
            d = np.ones(1,)
            S1 = copy.deepcopy(I)
            S2 = copy.deepcopy(I)
            S3 = copy.deepcopy(I)

            S1.addConstraint(C, -d)  # x <= -1
            S1.resetRowWithUpdatedCenter(index, -1.0)

            # -1 <= x <= 1
            S2.addConstraint(-C, d) # x >= -1 -> -x <= 1
            S2.addConstraint(C, d) # x <= 1

            S3.addConstraint(-C, -d) # x >= 1 -> -x <= -1
            S3.resetRowWithUpdatedCenter(index, 1.0)

            S = []
            S.append(S1)
            S.append(S2)
            S.append(S3)
            return S

        # ------------- case 5) -------------
        if xmin >= 1:
            S1 = copy.deepcopy(I)
            S1.resetRowWithUpdatedCenter(index, 1.0) # x >= 1 -> -x <= -1
            S = []
            S.append(S1)
            return S

        # ------------- case 6) -------------
        if xmax <= -1:
            S = []
            S.append(I.resetRowWithUpdatedCenter(index, -1.0)) # x <= -1
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

        Author: Yuntao Li, Date: 1/18/2024
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
            S1 = SatLins.stepReach(I[i], index, lp_solver)
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

        Author: Yuntao Li, Date: 1/18/2024
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

        if not isinstance(In, ProbStar) and not isinstance(In, Star):
            raise Exception('error: input is not a Star or ProbStar, \
            type of input is {}'.format(type(In)))

        lb, ub = In.estimateRanges()
        map_n1 = np.argwhere(ub <= -1).reshape(-1)
        map_1 = np.argwhere(lb >= 1).reshape(-1)
        
        I = copy.deepcopy(In)
        if len(map_n1) > 0:
            I.resetRowsWithUpdatedCenter(map_n1, -1.0)  # x <= -1
        if len(map_1) > 0:
            I.resetRowsWithUpdatedCenter(map_1, 1.0) # x >= 1

        map_1_n1 = np.argwhere((lb < 1) & (ub > -1)).reshape(-1)

        S = []
        S1 = [I]

        for i in range(len(map_1_n1)):
            S1 = SatLins.stepReachMultiInputs(S1, map_1_n1[i], lp_solver)
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

        Author: Yuntao Li, Date: 1/18/2024
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

        assert isinstance(In, list), 'error: inputsets should be in a list'
        S = []
        if pool is None:
            for i in range(0, len(In)):
                S.extend(SatLins.reachExactSingleInput(In[i], lp_solver))
        elif isinstance(pool, multiprocessing.pool.Pool):
            S1 = []
            S1 = S1 + pool.map(SatLins.reachExactSingleInput, zip(In, [lp_solver]*len(In)))
            for i in range(0, len(S1)):
                S.extend(S1[i])
        elif isinstance(pool, ipyparallel.client.view.DirectView):
            # S1 = pool.map(SatLins.reachExactSingleInput, zip(In, [lp_solver]*len(In)))
            # print('S1 = {}'.format(S1))
            raise Exception('error: ipyparallel option is under testing...')
        else:
            raise Exception('error: unknown/unsupport pool type')    
        return S

        
    @staticmethod
    def stepReachStarApprox(In, index, lp_solver='gurobi', show=False):

        """
        Approx reachability for satlin using stepReach
        Args:
            @In: a single input set
            @index: index of a neuron
            @lp_solver: lp_solver

        Returns:
            output set

        Author: Yuntao Li, Date: 07/31/2025
        """

        assert isinstance(In, Star), 'Input is not a Star set'

        lb = In.getMin(index, lp_solver)
        ub = In.getMax(index, lp_solver)

        n = In.nVars + 1
        dtype = In.V.dtype

        # ------------- case 1) -------------
        if lb >= -1 and ub <= 1:
            if show:
                print('\nPerforming SatLins stepReachStarApprox for case 1...')
            return In # -1 <= x <= 1

        # ------------- case 2) -------------
        if -1 <= lb < 1 < ub:
            if show:
                print('\nPerforming SatLins stepReachStarApprox for case 2...')
                print('Add a new predicate variable with three more constraints at index = {}'.format(index))

            # constraint 1: y(index) <= x[index]
            C1 = np.hstack([-In.V[index, 1:n], 1])
            d1 = In.V[index, 0]

            # constraint 2: y[index] <= 1
            C2 = np.zeros((1, n), dtype=dtype)
            C2[0, n-1] = 1
            d2 = 1

            # constraint 3: y[index] >= ((1-lb)/(ub-lb))(x-lb) + lb
            a = (1 - lb) / (ub - lb)
            C3 = np.hstack([a* In.V[index, 1:n], -1])
            d3 = -lb + a * lb - a * In.V[index, 0]

            if len(In.d) == 0:
                C0 = np.empty([0, n], dtype=dtype)
                d0 = np.empty([0], dtype=dtype)
            else:
                m = In.C.shape[0]
                C0 = np.hstack([In.C, np.zeros((m, 1), dtype=dtype)])
                d0 = In.d

            new_C = np.vstack([C0, C1, C2, C3])
            new_d = np.hstack([d0, d1, d2, d3])
            new_V = np.hstack([In.V, np.zeros((In.dim, 1), dtype=dtype)])
            new_V[index, :] = 0
            new_V[index, n] = 1
            new_lb = np.hstack([In.pred_lb, lb])
            new_ub = np.hstack([In.pred_ub, 1.0])
            return Star(new_V, new_C, new_d, new_lb, new_ub)

        # ------------- case 3) -------------
        if lb < -1 < ub <= 1:
            if show:
                print('\nPerforming SatLins stepReachStarApprox for case 3...')
                print('Add a new predicate variable with three more constraints at index = {}'.format(index))
                
            # constraint 1: y[index] >= -1
            C1 = np.zeros([1, n], dtype=dtype)
            C1[0, n-1] = -1
            d1 = 1

            # constraint 2: y[index] >= x[index]
            C2 = np.hstack([In.V[index, 1:n], -1])
            d2 = -In.V[index, 0]

            # constraint 3: y[index] <= (1+ub)(x-lb)/(ub - lb) -1
            a = (1 + ub) / (ub - lb)
            C3 = np.hstack([-a*In.V[index, 1:n], 1])
            d3 = -1 + a * In.V[index, 0] - a * lb

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
            pred_lb = np.hstack([In.pred_lb, -1])
            pred_ub = np.hstack([In.pred_ub, ub])
            return Star(V, C, d, pred_lb, pred_ub)

        # ------------- case 4) -------------
        if lb < -1 < 1 < ub:
            if show:
                print('\nPerforming SatLins stepReachStarApprox for case 4...')
                print('Add a new predicate variable with four more constraints at index = {}'.format(index))
            # constraint 1: y[index] >= -1
            C1 = np.zeros([1, n], dtype=dtype)
            C1[0, n-1] = -1
            d1 = 1

            # constraint 2: y[index] <= 1
            C2 = np.zeros((1, n), dtype=dtype)
            C2[0, n-1] = 1
            d2 = 1

            # constraint 3: y[index] <= 2(x-lb)/(1-lb) - 1
            # need to change
            a = 2 / (1 - lb)
            C3 = np.hstack([-a * In.V[index, 1:n], 1])
            d3 = a * In.V[index, 0] - a * lb - 1

            # constraint 4: y[index] >=  2(x+1)/(ub+1) - 1
            # need to change
            b = 2 / (ub + 1)
            C4 = np.hstack([b * In.V[index, 1:n], -1])
            d4 = -b * In.V[index, 0] - b + 1

            if len(In.d) == 0: # for Star set as initial C and d might be []
                C0 = np.empty([0, n], dtype=dtype)
                d0 = np.empty([0], dtype=dtype)
            else:
                m = In.C.shape[0]
                C0 = np.hstack([In.C, np.zeros((m, 1), dtype=dtype)])
                d0 = In.d

            new_C = np.vstack([C0, C1, C2, C3, C4])
            new_d = np.hstack([d0, d1, d2, d3, d4])
            new_V = np.hstack([In.V, np.zeros((In.dim, 1), dtype=dtype)])
            new_V[index, :] = 0
            new_V[index, n] = 1
            new_lb = np.hstack([In.pred_lb, -1.0])
            new_ub = np.hstack([In.pred_ub, 1.0])
            return Star(new_V, new_C, new_d, new_lb, new_ub)

        # ------------- case 5) -------------
        if lb >= 1:
            if show:
                print('\nPerforming SatLins stepReachStarApprox for case 5...')
            S = copy.deepcopy(In)
            return S.resetRowWithUpdatedCenter(index, 1.0) # x >= 1 -> -x <= -1

        # ------------- case 6) -------------
        if ub <= -1:
            if show:
                print('\nPerforming SatLins stepReachStarApprox for case 6...')
            S = copy.deepcopy(In)
            return S.resetRowsWithUpdatedCenter(index, -1.0)  # x <= -1


    @staticmethod
    def reachApprox(In, lp_solver='gurobi', show=False):
        """
        Approximate reachability for Star set using stepReachStarApprox
        Args:
            @In: a single input set
            @lp_solver: lp_solver

        Returns:
            output set

        Author: Yuntao Li, Date: 07/31/2025
        """

        assert isinstance(In, Star), 'Input is not a Star set'
        
        lb, ub = In.estimateRanges()
        map_n1 = np.argwhere(ub <= -1).reshape(-1)
        map_1 = np.argwhere(lb >= 1).reshape(-1)
        
        I = copy.deepcopy(In)
        if len(map_n1) > 0:
            I.resetRowsWithUpdatedCenter(map_n1, -1.0)  # x <= -1
        if len(map_1) > 0:
            I.resetRowsWithUpdatedCenter(map_1, 1.0) # x >= 1

        map_1_n1 = np.argwhere((lb < 1) & (ub > -1)).reshape(-1)

        for i in range(len(map_1_n1)):
            if show:
                print('Performing approximate SatLin operation on {} neuron'.format(i))
            I = SatLins.stepReachStarApprox(In=I, index=map_1_n1[i], lp_solver=lp_solver, show=show)
        return I