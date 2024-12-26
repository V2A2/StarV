"""
SatLins Class for reachability analysis of neural network layers with SatLins activation function.
Author: Yuntao Li
Date: 1/10/2024
"""
from typing import List, Union, Optional
import numpy as np
from multiprocessing.pool import Pool
from StarV.set.probstar import ProbStar
from StarV.set.star import Star

class SatLins:
    """
    SatLins class for computing reachable set of Symmetric Saturating Linear Transfer Function.
    """

    @staticmethod
    def evaluate(x: np.ndarray) -> np.ndarray:
        """
        Evaluate the SatLins function element-wise on the input.

        The SatLins function is defined as:
        f(x) = -1 if x < -1
             = x if -1 <= x <= 1
             = 1 if x > 1

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Result of applying SatLins to each element.
        """
        return np.clip(x, -1, 1)

    @staticmethod
    def stepReach(*args) -> List[Union[ProbStar, Star]]:
        """
        Compute reachable set for a single step of SatLins.

        Args:
            I: Input set (ProbStar or Star).
            index: Index of the current dimension being processed.
            lp_solver (optional): LP solver to use. Defaults to 'gurobi'.

        Returns:
            List[Union[ProbStar, Star]]: List of output sets after applying SatLins.

        Raises:
            ValueError: If the input is not a ProbStar or Star set.
        """
        if len(args) == 2:
            I, index = args
            lp_solver = 'gurobi'
        elif len(args) == 3:
            I, index, lp_solver = args
        else:
            raise ValueError("Invalid number of arguments. Expected 2 or 3.")

        if not isinstance(I, (ProbStar, Star)):
            raise ValueError(f"Input must be a Star or ProbStar set, got {type(I)}")

        xmin = I.getMin(index, lp_solver)
        xmax = I.getMax(index, lp_solver)

        C = np.zeros(I.dim)
        C[index] = 1.0
        d = np.ones(1)

        if -1 <= xmin and xmax <= 1:
            return [I]
        elif -1 <= xmin < 1 < xmax:
            S1, S2 = I.clone(), I.clone()
            S1.addConstraintWithoutUpdateBounds(-C, d)
            S1.addConstraintWithoutUpdateBounds(C, d)
            S2.addConstraintWithoutUpdateBounds(-C, -d)
            S2.resetRowWithUpdatedCenter(index, 1.0)
            return [S1, S2]
        elif xmin < -1 < xmax <= 1:
            S1, S2 = I.clone(), I.clone()
            S1.addConstraintWithoutUpdateBounds(C, -d)
            S1.resetRowWithUpdatedCenter(index, -1.0)
            S2.addConstraintWithoutUpdateBounds(-C, d)
            return [S1, S2]
        elif xmin < -1 and xmax > 1:
            S1, S2, S3 = I.clone(), I.clone(), I.clone()
            S1.addConstraintWithoutUpdateBounds(C, -d)
            S1.resetRowWithUpdatedCenter(index, -1.0)
            S2.addConstraintWithoutUpdateBounds(-C, d)
            S2.addConstraintWithoutUpdateBounds(C, d)
            S3.addConstraintWithoutUpdateBounds(-C, -d)
            S3.resetRowWithUpdatedCenter(index, 1.0)
            return [S1, S2, S3]
        elif xmin >= 1:
            S = I.clone()
            S.resetRowWithUpdatedCenter(index, 1.0)
            return [S]
        else:  # xmax <= -1
            S = I.clone()
            S.resetRowWithUpdatedCenter(index, -1.0)
            return [S]

    @staticmethod
    def stepReachMultiInputs(*args) -> List[Union[ProbStar, Star]]:
        """
        Compute reachable set for a single step of SatLins with multiple inputs.

        Args:
            I: List of input sets (ProbStar or Star).
            index: Index of the current dimension being processed.
            lp_solver (optional): LP solver to use. Defaults to 'gurobi'.

        Returns:
            List[Union[ProbStar, Star]]: List of output sets after applying SatLins.
        """
        if len(args) == 2:
            I, index = args
            lp_solver = 'gurobi'
        elif len(args) == 3:
            I, index, lp_solver = args
        else:
            raise ValueError("Invalid number of arguments. Expected 2 or 3.")

        return [output for input_set in I for output in SatLins.stepReach(input_set, index, lp_solver)]

    @staticmethod
    def reachExactSingleInput(*args) -> List[Union[ProbStar, Star]]:
        """
        Perform exact reachability analysis for a single input set.

        Args:
            In: Input set (ProbStar or Star).
            lp_solver (optional): LP solver to use. Defaults to 'gurobi'.

        Returns:
            List[Union[ProbStar, Star]]: List of output sets after applying SatLins to all dimensions.

        Raises:
            ValueError: If the input is not a ProbStar or Star set.
        """
        if len(args) == 1:
            In = args[0]
            lp_solver = 'gurobi'
        elif len(args) == 2:
            In, lp_solver = args
        else:
            raise ValueError("Invalid number of arguments. Expected 1 or 2.")

        if not isinstance(In, (ProbStar, Star)):
            raise ValueError(f"Input must be a Star or ProbStar set, got {type(In)}")

        S = [In]
        for i in range(In.dim):
            S = SatLins.stepReachMultiInputs(S, i, lp_solver)

        return S

    @staticmethod
    def reachExactMultiInputs(*args) -> List[Union[ProbStar, Star]]:
        """
        Perform exact reachability analysis with multiple inputs.

        This method supports parallel computation if a pool is provided.

        Args:
            In: List of input sets (ProbStar or Star).
            lp_solver (optional): LP solver to use. Defaults to 'gurobi'.
            pool (optional): Pool for parallel computation. Defaults to None.

        Returns:
            List[Union[ProbStar, Star]]: List of output sets after applying SatLins to all inputs.

        Raises:
            ValueError: If the pool type is unknown or unsupported.
        """
        if len(args) == 1:
            In = args[0]
            lp_solver = 'gurobi'
            pool = None
        elif len(args) == 2:
            In, lp_solver = args
            pool = None
        elif len(args) == 3:
            In, lp_solver, pool = args
        else:
            raise ValueError("Invalid number of arguments. Expected 1, 2, or 3.")

        if pool is None:
            return [output for input_set in In for output in SatLins.reachExactSingleInput(input_set, lp_solver)]
        elif isinstance(pool, Pool):
            results = pool.starmap(SatLins.reachExactSingleInput, [(input_set, lp_solver) for input_set in In])
            return [output for result in results for output in result]
        else:
            raise ValueError(f"Unknown or unsupported pool type: {type(pool)}")