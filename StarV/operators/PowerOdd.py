"""
Odd power operater
Zhuoyang Zhou, 02/06/2026
"""

from StarV.dynamic.powerodd import PowerOdd

class PowerOddOperater(object):
    """ PowerOdd class for qualitative reachability
        Author: Zhuoyang Zhou
        Date: 02/08/2026
    """
    
    def __init__(self, opt=False, delta=0.98):
        self.opt = opt
        self.delta = delta
        
    def reach(self, In, n, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """main reachabilikty method
            Args:
                @I: an input set (Star, SparseStar, or ProbStar)
                @method: method: 'approx'
                @lp_solver: lp solver: 'gurobi' (default), 'glpk', or 'linprog'
                @pool: parallel pool: None or multiprocessing.pool.Pool
                @RF: relax-factor from 0 to 1 (0 by default)
                @DR: depth reduction from 1 to k-Layers (0 by default)

            Return:
                @R: a reachable set        
        """
        assert isinstance(n, int), "n must be an integer"
        assert n > 0 and n % 2 == 1, "n must be a positive odd integer"

        if method == 'exact':
            raise Exception('error: exact method for powerodd function is not supported')
        if method == 'approx':
            return PowerOdd.reach(I=In, n=n, opt=self.opt, delta=self.delta, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
        raise Exception('error: unknown reachability method')