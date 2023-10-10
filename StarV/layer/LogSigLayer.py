"""
LogSig layer class
Sung Woo Choi, 04/11/2023
"""

from StarV.fun.logsig import LogSig


class LogSigLayer(object):
    """ LogSigLayer class for qualitative reachability
        Author: Sung Woo Choi
        Date: 04/11/2023
    """

    def __init__(self, opt=False, delta=0.98):
        self.opt = opt
        self.delta = delta

    @staticmethod
    def evaluate(x):
        return LogSig.f(x)
    
    def reach(self, In, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
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

        if method == 'exact':
            raise Exception('error: exact method for logsig or sigmoid function is not supported')
        if method == 'approx':
            return LogSig.reach(I=In, opt=self.opt, delta=self.delta, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
        raise Exception('error: unknown reachability method')