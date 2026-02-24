"""
Even power operater
Zhuoyang Zhou, 02/05/2026, Update: 02/19/2026
"""
from StarV.dynamic.powereven import PowerEven
from StarV.set.star import Star


class PowerEvenOperater(object):
    """ PowerEven class for qualitative reachability
        Author: Zhuoyang Zhou
        Date: 02/08/2026
    """

    def __init__(self, opt=False, delta=0.98):
        self.opt = opt
        self.delta = delta

    def reach(self, In, n, idx=None, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False, split=False, max_splits=0):
        """main reachability method
            Args:
                @In: an input set (Star or list of Stars)
                @n: power exponent (positive even integer)
                @idx: optional index for high-dimensional append mode
                @method: method: 'approx'
                @lp_solver: lp solver: 'gurobi' (default), 'glpk', or 'linprog'
                @pool: parallel pool: None or multiprocessing.pool.Pool
                @RF: relax-factor from 0 to 1 (0 by default)
                @DR: depth reduction from 1 to k-Layers (0 by default)
                @show: display computation progress
                @split: splitting flag (reserved)
                @max_splits: max split count (reserved)

            Return:
                @R: a reachable set
        """
        assert isinstance(n, int), "n must be an integer"
        assert n > 0 and n % 2 == 0, "n must be a positive even integer"

        if method == 'exact':
            raise Exception('error: exact method for powereven function is not supported')
        if method != 'approx':
            raise Exception('error: unknown reachability method')

        if isinstance(In, list):
            out = []
            for S in In:
                if not isinstance(S, Star):
                    raise Exception('error: list must contain Star elements only')
                if idx is not None:
                    assert isinstance(idx, int), 'error: idx must be an integer'
                    assert 0 <= idx < S.dim, f"idx {idx} is out of range [0, {S.dim-1}]"
                R = PowerEven.reach(
                    I=S,
                    n=n,
                    idx=idx,
                    opt=self.opt,
                    delta=self.delta,
                    lp_solver=lp_solver,
                    pool=pool,
                    RF=RF,
                    DR=DR,
                    show=show,
                    split=split,
                    max_splits=max_splits,
                )
                if isinstance(R, list):
                    out.extend(R)
                else:
                    out.append(R)
            return out

        if idx is not None and isinstance(In, Star):
            assert isinstance(idx, int), 'error: idx must be an integer'
            assert 0 <= idx < In.dim, f"idx {idx} is out of range [0, {In.dim-1}]"

        return PowerEven.reach(
            I=In,
            n=n,
            idx=idx,
            opt=self.opt,
            delta=self.delta,
            lp_solver=lp_solver,
            pool=pool,
            RF=RF,
            DR=DR,
            show=show,
            split=split,
            max_splits=max_splits,
        )
