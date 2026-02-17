# -*- coding: utf-8 -*-
"""
Sin layer class (y = sin(x)): Star-based approximate reachability
Minimal driver that calls Sin.reachApprox_star with 3 cases:
  (A) u <= 0, (B) l < 0 < u (split), (C) l >= 0

Author: Zhuoyang Zhou
Date: 02/09/2026
"""

import numpy as np
from StarV.set.star import Star
from StarV.fun.cosine import Cosine


class CosLayer(object):
    """
    CosLayer for qualitative/approximate reachability with Star only.

    Notes:
    - Input: Star or list[Star]
    - Output: Star or list[Star]
      * If any neuron interval crosses zero, Cosine.reachApprox_star may return a list[Star]
        representing a union (at most two for 1-D pendulum angle).
    - No SparseStar/ImageStar, no exact method, no pool/DR/show.
    """

    def __init__(self):
        pass

    @staticmethod
    def evaluate(x):
        """Pointwise evaluation (for testing / debugging)."""
        return np.cos(x)

    def reach(self, In, method='approx', lp_solver='gurobi', RF=0.0):
        """
        Main reachability method (approx only, Star only).

        Args:
            In        : Star or list[Star]
            method    : 'approx' (only)
            lp_solver : 'gurobi' (default), 'glpk', or 'linprog'
            RF        : relax-factor from 0 to 1 (0 by default)

        Returns:
            Star or list[Star]:
              - If no zero-crossing: a single Star
              - If zero-crossing: a list[Star] (union)
        """
        if method != 'approx':
            raise Exception("error: only 'approx' method is supported for cos")

        # Accept a single Star or a list of Stars (propagate unions)
        if isinstance(In, Star):
            return Cosine.reachApprox_star(In, lp_solver=lp_solver, RF=RF)

        if isinstance(In, list):
            out = []
            for S in In:
                if not isinstance(S, Star):
                    raise Exception('error: list must contain Star elements only')
                R = Sine.reachApprox_star(S, lp_solver=lp_solver, RF=RF)
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
