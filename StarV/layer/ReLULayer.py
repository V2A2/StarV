"""
ReLU layer class
Dung Tran, 9/10/2022
"""

from StarV.fun.poslin import PosLin


class ReLULayer(object):
    """ ReLULayer class for qualitative and quantitative reachability
        Author: Dung Tran
        Date: 9/10/2022
    """
    

    @staticmethod
    def evaluate(x):
        return PosLin.evaluate(x)
    
    @staticmethod
    def reach(In, method='exact', lp_solver='gurobi', pool=None, RF=0.0):
        """main reachability method
           Args:
               @I: a list of input set (Star or ProbStar)
               @method: method: 'exact', 'approx', or 'relax'
               @lp_solver: lp solver: 'gurobi' (default), 'glpk', or 'linprog'
               @pool: parallel pool: None or multiprocessing.pool.Pool
               @RF: relax-factor from 0 to 1 (0 by default)

            Return: 
               @R: a list of reachable set
        """

        print("\nReLULayer reach function\n")

        if method == 'exact':
            return PosLin.reachExactMultiInputs(In, lp_solver, pool)
        elif method == 'approx':
            raise Exception('error: under development')
        elif method == 'relax':
            raise Exception('error: under development')
        else:
            raise Exception('error: unknown reachability method')
