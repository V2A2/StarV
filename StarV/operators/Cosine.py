# -*- coding: utf-8 -*-
"""
Cos layer class (y = cos(x)) Star-based approximate reachability
Author: Zhuoyang Zhou
Date: 02/09/2026 Update: 02/22/2026
"""

import numpy as np
from StarV.set.star import Star
from StarV.dynamic.cosine import Cosine


class CosLayer(object):
    """
    CosLayer for qualitative/approximate reachability with Star.
    """

    def __init__(self):
        pass

    @staticmethod
    def evaluate(x):
        """Pointwise evaluation (for testing / debugging)."""
        return np.cos(x)

    def reach(self, In, idx=None, method='approx', lp_solver='gurobi', RF=0.0, split=False, max_splits=0):
        """
        Main reachability method (approx only, Star only).

        Args:
            In         : Star or list[Star]
            idx        : None (old behavior, cos on all dims) or int (append cos(x_idx))
            method     : 'approx' (only)
            lp_solver  : 'gurobi' (default), 'glpk', or 'linprog'
            RF         : relax-factor from 0 to 1 (0 by default)
            split      : splitting flag (reserved)
            max_splits : max split count (reserved)

        Returns:
            Star or list[Star]
        """
        if method != 'approx':
            raise Exception("error: only 'approx' method is supported for cos")

        if split:
            raise NotImplementedError('error: split=True is not implemented for cosine operator yet')
        _ = max_splits

        # Accept a single Star or a list of Stars (propagate unions)
        if isinstance(In, Star):
            if idx is not None:
                assert isinstance(idx, int), 'error: idx must be an integer'
                assert 0 <= idx < In.dim, f"idx {idx} is out of range [0, {In.dim-1}]"
            return Cosine.reachApprox_star(In, idx=idx, lp_solver=lp_solver, RF=RF, split=split, max_splits=max_splits)

        if isinstance(In, list):
            out = []
            for S in In:
                if not isinstance(S, Star):
                    raise Exception('error: list must contain Star elements only')
                if idx is not None:
                    assert isinstance(idx, int), 'error: idx must be an integer'
                    assert 0 <= idx < S.dim, f"idx {idx} is out of range [0, {S.dim-1}]"

                R = Cosine.reachApprox_star(S, idx=idx, lp_solver=lp_solver, RF=RF, split=split, max_splits=max_splits)
                # R can be Star or list[Star]; normalize to list and extend
                if isinstance(R, list):
                    out.extend(R)
                else:
                    out.append(R)
            return out

        raise Exception('error: input must be a Star or a list of Star')

    def __str__(self):
        print('Layer type: {}'.format(self.__class__.__name__))
        print('')
        return '\n'

    def info(self):
        print(self)