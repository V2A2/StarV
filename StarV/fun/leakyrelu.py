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
LeakyReLU Class
Yuntao Li, 1/10/2024
Update: Yuntao Li, Date: 09/16/2025
"""

# !/usr/bin/python3
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
import copy
import multiprocessing
import ipyparallel

class LeakyReLU:
    """
    LeakyReLU Class contains method for reachability analysis for Layer with LeakyReLU activation function
    Yuntao Li, 1/10/2024
    """

    @staticmethod
    def evaluate(x, gamma):
        """
        Evaluate method for LeakyReLU
        Args:
            @x: input array
            @gamma: leaking factor
        Returns:
            Modified array with LeakyReLU applied

        Author: Yuntao Li, Date: 1/18/2024
        """
        y = np.array(x, dtype=float)  # Ensure the array is of float type for proper arithmetic operations
        negative_indices = y < 0
        y[negative_indices] = gamma * y[negative_indices]
        return y
    

    @staticmethod
    def stepReach(*args):
        """
        StepReach method, compute reachable set for a single step
        Args:
            @I: single star set input
            @index: index of the neuron performing stepLeakyReLU
            @gamma: leaking factor
            @lp_solver: LP solver method
        Returns:
            @S: star output set

        Author: Yuntao Li, Date: 1/18/2024
        Update: Yuntao Li, Date: 09/16/2025
        """
        len_args = len(args)
        if len_args == 3:  # 3 arguments
            [I, index, gamma] = args
            lp_solver = 'gurobi'
        elif len_args == 4:  # 4 arguments
            [I, index, gamma, lp_solver] = args
        else:
            raise Exception('error: \
            Invalid number of input arguments, should be 3 or 4')

        if not isinstance(I, ProbStar) and not isinstance(I, Star):
            raise Exception('error: input is not a Star or ProbStar set, \
            type of input = {}'.format(type(I)))

        # xmin = I.getMin(index, lp_solver) # Star
        xmin, xmax = I.estimateRange(index) # Prob Star
        if xmin >= 0:
            S = []
            S.append(I)
        else:
            xmax = I.getMax(index, lp_solver) # Star
            if xmax <= 0:
                S = []
                S.append(I.resetRowWithFactor(index, gamma))
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
                    S1 = S1.resetRowWithFactor(index, gamma, copy_=False)
                    S2 = S2.addConstraint(-C, d, copy_=False)  # x >= 0
                    S = []
                    S.append(S1)
                    S.append(S2)

        return S


    @staticmethod
    def stepReachMultipleInputs(*args):
        """
        StepReach with multiple inputs
        Args:
            @I: an array of stars
            @index: index where stepReach is performed
            @gamma: leaking factor
            @option: parallel computation option
            @lp_solver: LP solver method
        Returns:
            @S: a list of output set

        Author: Yuntao Li, Date: 1/18/2024
        """
        if len(args) == 3:
            [I, index, gamma] = args
            lp_solver = 'gurobi'
        elif len(args) == 4:
            [I, index, gamma, lp_solver] = args
        else:
            raise Exception('error: \
            Invalid number of input arguments, should be 3 or 4 ')

        assert isinstance(I, list), 'error: input is not a list, \
        type of input is {}'.format(type(I))

        S = []
        for i in range(0, len(I)):
            S1 = LeakyReLU.stepReach(I[i], index, gamma, lp_solver)
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
        if len(args1) == 2:
            [In, gamma] = args1
            lp_solver = 'gurobi'
        elif len(args1) == 3:
            [In, gamma, lp_solver] = args1
        else:
            raise Exception('error: Invalid \
            number of input arguments, should be 1 or 2')

        if not isinstance(In, ProbStar) and not isinstance(In, Star):
            raise Exception('error: input is not a Star or ProbStar, \
            type of input is {}'.format(type(In)))

        S = []
        S1 = [In]
        for i in range(0, In.dim):
            S1 = LeakyReLU.stepReachMultipleInputs(S1, i, gamma, lp_solver)

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
        
        if len(args) == 2:
            [In, gamma] = args
            lp_solver = lp_solver_default
            pool = None
        elif len(args) == 3:
            [In, gamma, lp_solver] = args
            pool = None
        elif len(args) == 4:
            [In, gamma, lp_solver, pool] = args
       
        else:
            raise Exception('error: Invalid \
            number of input arguments, should be 1, 2 or 3')

        assert isinstance(In, list), 'error: inputsets should be in a list'
        S = []
        if pool is None:
            for i in range(0, len(In)):
                S.extend(LeakyReLU.reachExactSingleInput(In[i], gamma, lp_solver))
        elif isinstance(pool, multiprocessing.pool.Pool):
            S1 = []
            S1 = S1 + pool.map(LeakyReLU.reachExactSingleInput, zip(In, [gamma]*len(In), [lp_solver]*len(In)))
            for i in range(0, len(S1)):
                S.extend(S1[i])
        elif isinstance(pool, ipyparallel.client.view.DirectView):
            # S1 = pool.map(LeakyReLU.reachExactSingleInput, zip(In, [lp_solver]*len(In)))
            # print('S1 = {}'.format(S1))
            raise Exception('error: ipyparallel option is under testing...')
        else:
            raise Exception('error: unknown/unsupport pool type')    
        return S
