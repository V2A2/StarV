# -*- coding: utf-8 -*-
"""
Sin layer class (y = sin(x)) Star-based approximate reachability
Author: Zhuoyang Zhou
Date: 02/07/2026
"""

import numpy as np
from StarV.set.star import Star
from StarV.dynamic.sine import Sine


class SinLayer(object):
    """
    SinLayer for qualitative/approximate reachability with Star.
    """

    def __init__(self):
        pass

    @staticmethod
    def evaluate(x):
        """Pointwise evaluation (for testing / debugging)."""
        return np.sin(x)

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
            raise Exception("error: only 'approx' method is supported for sin")

        # Accept a single Star or a list of Stars (propagate unions)
        if isinstance(In, Star):
            return Sine.reachApprox_star(In, lp_solver=lp_solver, RF=RF)

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
