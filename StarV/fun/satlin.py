"""
Satlin Class
Yuntao Li, 1/18/2024
"""

from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
import copy
import multiprocessing
import ipyparallel

class SatLin(object):
    """
    SATLIN class for computing reachable set of Satlin Transfer Function
    Yuntao Li, 1/18/2024
    """

    @staticmethod
    def evaluate(x):
        """
        evaluate method and reachability analysis with stars

        Args:
            @x = np.arrays

        Returns:
            0, if n <= 0
            n, if 0 <= n <= 1
            1, if 1 <= n

        Author: Yuntao Li, Date: 1/18/2024
        """

        a = np.maximum(x, 0)
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

        Author: Dung Tran, update 4/20/2024, fix some small errors
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

        # xmin, xmax = I.estimateRange(index)
        xmin = I.getMin(index, lp_solver)
        xmax = I.getMax(index, lp_solver)

        # ------------- case 1) only single set -------------
        if xmin >= 0 and xmax <= 1:
            S = []
            S.append(I) # 0 <= x <= 1
            return S

        # ------------- case 2) -------------
        if 0 <= xmin < 1 < xmax:
            C = np.zeros(I.dim,)
            C[index] = 1.0
            d = np.zeros(1,)
            d1 = np.ones(1,)
            S1 = copy.deepcopy(I) 
            S2 = copy.deepcopy(I)

            # 0 <= x <= 1
            S1.addConstraint(-C, d) # x >= 0 -> -x <= 0
            #S1.addConstraint(C, d) # x <= 1
            S1.addConstraint(C, d1) # x1 <= 1, Tran Fixed this
            
            S2.addConstraint(-C, -d1) # x >= 1 -> -x <= -1
            S2.resetRowWithUpdatedCenter(index, 1.0)

            S = []
            S.append(S1)
            S.append(S2)
            return S

        # ------------- case 3) -------------
        if xmin < 0 < xmax <= 1:
            C = np.zeros(I.dim,)
            C[index] = 1.0
            d = np.zeros(1,)
            S1 = copy.deepcopy(I)
            S2 = copy.deepcopy(I)

            S1.addConstraint(C, d)  # x <= 0
            S1.resetRow(index)

            S2.addConstraint(-C, d) # x >= 0 -> -x <= 0

            S = []
            S.append(S1)
            S.append(S2)
            return S

        # ------------- case 4) -------------
        if xmin < 0 < 1 < xmax:
            C = np.zeros(I.dim,)
            C[index] = 1.0
            d = np.zeros(1,)
            d1 = np.ones(1,)
            S1 = copy.deepcopy(I)
            S2 = copy.deepcopy(I)
            S3 = copy.deepcopy(I)

            S1.addConstraint(C, d)  # x <= 0
            S1.resetRow(index)

            # 0 <= x <= 1
            S2.addConstraint(-C, d) # x >= 0 -> -x <= 0
            S2.addConstraint(C, d1) # x <= 1

            S3.addConstraint(-C, -d1) # x >= 1 -> -x <= -1
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
        if xmax <= 0:
            S = []
            S.append(I.resetRow(index)) # x <= 0
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
            S1 = SatLin.stepReach(I[i], index, lp_solver)
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

        S = []
        S1 = [In]
        for i in range(0, In.dim):
            S1 = SatLin.stepReachMultiInputs(S1, i, lp_solver)

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
        print(pool)
        if pool is None:
            print("pool is none")
            for i in range(0, len(In)):
                S.extend(SatLin.reachExactSingleInput(In[i], lp_solver))
        elif isinstance(pool, multiprocessing.pool.Pool):
            S1 = []
            S1 = S1 + pool.map(SatLin.reachExactSingleInput, zip(In, [lp_solver]*len(In)))
            for i in range(0, len(S1)):
                S.extend(S1[i])
        elif isinstance(pool, ipyparallel.client.view.DirectView):
            # S1 = pool.map(SatLin.reachExactSingleInput, zip(In, [lp_solver]*len(In)))
            # print('S1 = {}'.format(S1))
            raise Exception('error: ipyparallel option is under testing...')
        else:
            raise Exception('error: unknown/unsupport pool type')    
        return S
