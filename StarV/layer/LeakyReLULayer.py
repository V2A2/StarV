"""
LeakyReLU layer class
Author: Yuntao Li
Date: 1/20/2024
"""
from StarV.fun.leakyrelu import LeakyReLU
from typing import List, Union
from StarV.set.probstar import ProbStar
from StarV.set.star import Star

class LeakyReLULayer:
    """
    LeakyReLULayer class for qualitative and quantitative reachability analysis.
    
    This class provides methods for evaluating LeakyReLU activation and
    computing reachable sets for LeakyReLU layers.
    """

    @staticmethod
    def evaluate(x, gamma: float = 0.1):
        """
        Evaluate the LeakyReLU function on input x.

        Args:
            x: Input data.
            gamma (float): The slope of the LeakyReLU function for negative inputs. Default is 0.1.

        Returns:
            Result of applying LeakyReLU to x.
        """
        return LeakyReLU.evaluate(x, gamma)
    
    @staticmethod
    # def reach(In: List[Union[Star, ProbStar]], method: str = 'exact', 
    #           lp_solver: str = 'gurobi', pool = None, RF: float = 0.0, 
    #           gamma: float = 0.1) -> List[Union[Star, ProbStar]]:
    def reach(In: List[Union[Star, ProbStar]], method: str = 'exact', 
              lp_solver: str = 'gurobi', pool = None, RF: float = 0.0) -> List[Union[Star, ProbStar]]:
        """
        Main reachability method for LeakyReLU layer.

        Args:
            In: A list of input sets (Star or ProbStar).
            method: Reachability method: 'exact', 'approx', or 'relax'.
            lp_solver: LP solver: 'gurobi' (default), 'glpk', or 'linprog'.
            pool: Parallel pool (None or multiprocessing.pool.Pool).
            RF: Relax-factor from 0 to 1 (0 by default).
            gamma: The slope of the LeakyReLU function for negative inputs. Default is 0.1.

        Returns:
            A list of reachable sets.

        Raises:
            ValueError: If an unknown reachability method is specified.
            NotImplementedError: If 'approx' or 'relax' methods are used (currently under development).
        """
        print("\nLeakyReLULayer reach function\n")
        
        if method == 'exact':
            gamma = 0.1
            return LeakyReLU.reachExactMultiInputs(In, gamma, lp_solver, pool)
        elif method in ['approx', 'relax']:
            raise NotImplementedError(f"The '{method}' method is currently under development.")
        else:
            raise ValueError(f"Unknown reachability method: {method}")