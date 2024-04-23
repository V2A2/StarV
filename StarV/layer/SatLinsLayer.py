"""
SatLins layer class
Yuntao Li, 1/20/2024
"""

from StarV.fun.satlins import SatLins


class SatLinsLayer(object):
    """ SatLinsLayer class for qualitative and quantitative reachability
        Author: Yuntao Li
        Date: 1/20/2024
    """

    @staticmethod
    def evaluate(x):
        return SatLins.evaluate(x)
    
    
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

        print("\nSatLinsLayer reach function\n")

        if method == 'exact':
            return SatLins.reachExactMultiInputs(In, lp_solver, pool)
        elif method == 'approx':
            raise Exception('error: under development')
        elif method == 'relax':
            raise Exception('error: under development')
        else:
            raise Exception('error: unknown reachability method')
