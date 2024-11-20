"""
LeakyReLU Class for reachability analysis of neural network layers with LeakyReLU activation function.
Author: Yuntao Li
Date: 1/10/2024
"""

from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
import copy
import multiprocessing
import ipyparallel

class LeakyReLU:
    """
    LeakyReLU Class contains methods for reachability analysis for Layer with LeakyReLU activation function.
    """

    @staticmethod
    def evaluate(x, gamma):
        """
        Evaluate method for LeakyReLU.

        Args:
            x: input array
            gamma: leaking factor

        Returns:
            Modified array with LeakyReLU applied

        The LeakyReLU function is defined as:
        f(x) = x if x >= 0
              = gamma * x if x < 0
        where gamma is a small positive value (e.g., 0.01).
        """
        return np.where(x >= 0, x, gamma * x)

    @staticmethod
    def stepReach(*args):
        """
        StepReach method, compute reachable set for a single step.

        Args:
            I: single star set input
            index: index of the neuron performing stepLeakyReLU
            gamma: leaking factor
            lp_solver: LP solver method (optional, default: 'gurobi')

        Returns:
            S: star output set

        This method computes the reachable set for a single neuron in the LeakyReLU layer.
        It handles three cases:
        1. The entire input set is non-negative
        2. The entire input set is negative
        3. The input set intersects both positive and negative regions
        """
        if len(args) == 3:
            I, index, gamma = args
            lp_solver = 'gurobi'
        elif len(args) == 4:
            I, index, gamma, lp_solver = args
        else:
            raise ValueError('Invalid number of input arguments, should be 3 or 4')

        if not isinstance(I, (ProbStar, Star)):
            raise TypeError(f'Input is not a Star or ProbStar set, type of input = {type(I)}')

        xmin, xmax = I.estimateRange(index)
        
        if xmin >= 0:
            return [I]
        
        xmax = I.getMax(index, lp_solver)
        if xmax <= 0:
            return [I.resetRowWithFactor(index, gamma)]
        
        xmin = I.getMin(index, lp_solver)
        if xmin >= 0:
            return [I]
        
        C = np.zeros(I.dim)
        C[index] = 1.0
        d = np.zeros(1)
        S1 = I.clone()
        S2 = I.clone()
        S1.addConstraint(C, d)  # x <= 0
        S1.resetRowWithFactor(index, gamma)
        S2.addConstraint(-C, d)  # x >= 0
        return [S1, S2]

    @staticmethod
    def stepReachMultipleInputs(*args):
        """
        StepReach with multiple inputs.

        Args:
            I: an array of stars
            index: index where stepReach is performed
            gamma: leaking factor
            lp_solver: LP solver method (optional, default: 'gurobi')

        Returns:
            S: a list of output sets

        This method applies stepReach to multiple input sets.
        """
        if len(args) == 3:
            I, index, gamma = args
            lp_solver = 'gurobi'
        elif len(args) == 4:
            I, index, gamma, lp_solver = args
        else:
            raise ValueError('Invalid number of input arguments, should be 3 or 4')

        if not isinstance(I, list):
            raise TypeError(f'Input is not a list, type of input is {type(I)}')

        S = []
        for input_set in I:
            S.extend(LeakyReLU.stepReach(input_set, index, gamma, lp_solver))
        return S

    @staticmethod
    def reachExactSingleInput(*args):
        """
        Exact reachability using stepReach for a single input set.

        Args:
            In: a single input set
            gamma: leaking factor
            lp_solver: LP solver method (optional, default: 'gurobi')

        Returns:
            S: output set

        This method computes the exact reachable set for a single input set
        by applying stepReach to each dimension sequentially.
        """
        if isinstance(args[0], tuple):
            args = list(args[0])
        
        if len(args) == 2:
            In, gamma = args
            lp_solver = 'gurobi'
        elif len(args) == 3:
            In, gamma, lp_solver = args
        else:
            raise ValueError('Invalid number of input arguments, should be 2 or 3')

        if not isinstance(In, (ProbStar, Star)):
            raise TypeError(f'Input is not a Star or ProbStar, type of input is {type(In)}')

        S = [In]
        for i in range(In.dim):
            S = LeakyReLU.stepReachMultipleInputs(S, i, gamma, lp_solver)

        return S

    @staticmethod
    def reachExactMultiInputs(*args):
        """
        Exact reachability with multiple inputs.
        Works with breadth-first-search verification.

        Args:
            In: a list of input sets
            gamma: leaking factor
            lp_solver: LP solver method (optional, default: 'gurobi')
            pool: pool for parallel computation (optional)

        Returns:
            S: output set

        This method computes the exact reachable set for multiple input sets,
        optionally using parallel computation.
        """
        if len(args) == 2:
            In, gamma = args
            lp_solver = 'gurobi'
            pool = None
        elif len(args) == 3:
            In, gamma, lp_solver = args
            pool = None
        elif len(args) == 4:
            In, gamma, lp_solver, pool = args
        else:
            raise ValueError('Invalid number of input arguments, should be 2, 3, or 4')

        if not isinstance(In, list):
            raise TypeError('Input sets should be in a list')

        if pool is None:
            S = []
            for input_set in In:
                S.extend(LeakyReLU.reachExactSingleInput(input_set, gamma, lp_solver))
        elif isinstance(pool, multiprocessing.pool.Pool):
            results = pool.starmap(LeakyReLU.reachExactSingleInput, 
                                   [(input_set, gamma, lp_solver) for input_set in In])
            S = [item for sublist in results for item in sublist]
        elif isinstance(pool, ipyparallel.client.view.DirectView):
            raise NotImplementedError('ipyparallel option is under testing')
        else:
            raise ValueError('Unknown/unsupported pool type')
        
        return S