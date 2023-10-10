"""
PosLin Class
Dung Tran, 8/29/2022

"""

# !/usr/bin/python3
from StarV.set.probstar import ProbStar
import numpy as np
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
    def stepReach(*args):

        """
        stepReach method, compute reachable set for a single step

        Args:
            @I: single star set input
            @index: index of current x[index] of current step

        Returns:
            @S: star output set
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

        if not isinstance(I, ProbStar):
            raise Exception('error: input is not a Star or ProbStar set, \
            type of input = {}'.format(type(I)))

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
                    S1.addConstraint(C, d)  # x <= 0
                    S1.resetRow(index)
                    S2.addConstraint(-C, d)  # x >= 0
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

        if not isinstance(In, ProbStar):
            raise Exception('error: input is not a Star or ProbStar, \
            type of input is {}'.format(type(In)))

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
        elif len(args) == 3:
            [In, lp_solver, pool] = args
       
        else:
            raise Exception('error: Invalid \
            number of input arguments, should be 1, 2 or 3')

        assert isinstance(In, list), 'error: inputsets should be in a list'
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
