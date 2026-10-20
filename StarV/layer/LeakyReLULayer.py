#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
##                                                                     ##
##   Primary contacts: Hoang Dung Tran <dungtran@ufl.edu> (UF)         ##
##                     Sung Woo Choi <sungwoo.choi@ufl.edu> (UF)       ##
##                     Yuntao Li <yli17@ufl.edu> (UF)                  ##
##                     Qing Liu <qliu1@ufl.edu> (UF)                   ##
##                                                                     ##
##   See CONTRIBUTORS for full author contacts and affiliations.       ##
##   This program is licensed under the BSD 3‑Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
LeakyReLU layer class
Yuntao Li, 1/20/2024
"""

from StarV.fun.leakyrelu import LeakyReLU


class LeakyReLULayer(object):
    """ LeakyReLUayer class for qualitative and quantitative reachability
        Author: Yuntao Li
        Date: 1/20/2024
    """

    def __init__(self, gamma=0.1):
        self.gamma = float(gamma)

    def evaluate(self, x, gamma=None):
        g = self.gamma if gamma is None else gamma
        return LeakyReLU.evaluate(x, g)
    
    def reach(self, In, method='exact', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False, **kwargs):
        """main reachability method
           Args:
               @I: a list of input set (Star or ProbStar) or a single input set
               @method: method: 'exact', 'approx', or 'relax'
               @lp_solver: lp solver: 'gurobi' (default), 'glpk', or 'linprog'
               @pool: parallel pool: None or multiprocessing.pool.Pool
               @RF: relax-factor from 0 to 1 (0 by default)

            Return: 
               @R: a list of reachable set or single reachable set
        """

        if show:
            print("\nLeakyReLULayer reach function\n")

        gamma = getattr(self, 'gamma', 0.1)
        if method == 'exact':
            return LeakyReLU.reachExactMultiInputs(In, gamma, lp_solver, pool)
        elif method == 'approx':
            return LeakyReLU.reachApprox(In=In, gamma=gamma, lp_solver=lp_solver, RF=RF, show=show)
        elif method == 'relax':
            return LeakyReLU.reachApprox(In=In, gamma=gamma, lp_solver=lp_solver, RF=RF, show=show)
        else:
            raise Exception('error: unknown reachability method')

    def __str__(self):
        print('Layer type: {}'.format(self.__class__.__name__))
        print('')
        return '\n'

    def info(self):
        print(self)
