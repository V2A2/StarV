"""
Even power operater
Zhuoyang Zhou, 02/05/2026
"""
from StarV.dynamic.powereven import PowerEven

class PowerEvenOperater(object):
    """ PowerEven class for qualitative reachability
        Author: Zhuoyang Zhou
        Date: 02/08/2026
    """
    
    def __init__(self, opt=False, delta=0.98):
        self.opt = opt
        self.delta = delta
        
    def reach(self, In, n, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """main reachability method
            Args:
                @In: an input set (Star, SparseStar, or ProbStar)
                @n: power exponent (positive even integer)
                @method: method: 'approx'
                @lp_solver: lp solver: 'gurobi' (default), 'glpk', or 'linprog'
                @pool: parallel pool: None or multiprocessing.pool.Pool
                @RF: relax-factor from 0 to 1 (0 by default)
                @DR: depth reduction from 1 to k-Layers (0 by default)
                @show: display computation progress

            Return:
                @R: a reachable set        
        """
        assert isinstance(n, int), "n must be an integer"
        assert n > 0 and n % 2 == 0, "n must be a positive even integer"
        
        if method == 'exact':
            raise Exception('error: exact method for powereven function is not supported')
        if method == 'approx':
            return PowerEven.reach(I=In, n=n, opt=self.opt, delta=self.delta, 
                                 lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
        raise Exception('error: unknown reachability method')