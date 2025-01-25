"""
TanSig layer class
Sung Woo Choi, 04/11/2023
"""

from StarV.fun.tansig import TanSig


class TanSigLayer(object):
    """ TanSigLayer class for qualitative reachability
        Author: Sung Woo Choi
        Date: 04/11/2023
    """

    def __init__(self, opt=False, delta=0.98):
        self.opt = opt
        self.delta = delta

    @staticmethod
    def evaluate(x):
        return TanSig.f(x)
    
    def reach(self, In, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """main reachability method
            Args:
                @In: an input set (Star, SparseStar, or ProbStar)
                @method: method: 'approx'
                @lp_solver: lp solver: 'gurobi' (default), 'glpk', or 'linprog'
                @pool: parallel pool: None or multiprocessing.pool.Pool
                @RF: relax-factor from 0 to 1 (0 by default)
                @DR: depth reduction from 1 to k-Layers (0 by default)

            Return:
                @R: a reachable set        
        """

        if method == 'exact':
            raise Exception('error: exact method for tansig or tanh layer is not supported')
        elif method == 'approx':
            return TanSig.reach(I=In, opt=self.opt, delta=self.delta, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
        raise Exception('error: unknown reachability method')