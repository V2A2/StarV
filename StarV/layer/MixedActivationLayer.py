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
Mixed Activaiton Layer Class
Sung Woo Choi, 10/29/2024
"""

import copy
import numpy as np
import multiprocessing
import ipyparallel

from StarV.set.probstar import ProbStar
from StarV.set.star import Star
from StarV.fun.poslin import PosLin
from StarV.fun.logsig import LogSig
from StarV.fun.tansig import TanSig
from StarV.fun.leakyrelu import LeakyReLU
from StarV.fun.satlin import SatLin
from StarV.fun.satlins import SatLins
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.LeakyReLULayer import LeakyReLULayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.SatLinsLayer import SatLinsLayer
# from StarV.layer.TanSigLayer import TanSigLayer
# from StarV.layer.LogSigLayer import LogSigLayer

class MixedActivationLayer(object):
    """ MixedActivationLayer class for qualitative and quantitative reachability
        Author: Sung Woo Choi
        Date: 10/29/2024
    """

    def __init__(self, actv_fun):
        self.actv_fun = actv_fun

    def evaluate(self, x):
        shape = x.shape
        dim = np.prod(shape)

        assert len(self.actv_fun) == dim,  'error: dimension of input should match number of activation functions lists'

        y = np.zeros(dim)
        for i in range(dim):
            actv_fun = self.actv_fun[i]
            if actv_fun == 'purelin':
                y[i] = x[i]
            elif actv_fun in ['poslin', 'relu']:
                y[i] = PosLin.evaluate(x[i])
            elif actv_fun in ['logsig', 'sigmoid']:
                y[i] = LogSig.evaluate(x[i])
            elif actv_fun in ['tansig', 'tanh']:
                y[i] = TanSig.evaluate(x[i])
            elif actv_fun == 'leakyrelu':
                y[i] = LeakyReLU.evaluate(x[i])
            elif actv_fun == 'satlin':
                y[i] = SatLin.evaluate(x[i])
            elif actv_fun == 'satlins':
                y[i] = SatLins.evaluate(x[i])
            else:
                raise Exception(f'error: {actv_fun} activation function is not supported')
        return y.reshape(shape)
            

    def reachExactMultiInputs(self, *args):
        """
        Exact reachability with multiple inputs
        Work with bread-first-search verification

        Args:
            @I: a single input set
            @lp_solver: lp_solver ('gurobi' or 'glpk' or 'linprog')
            @pool: pool for parallel computation
        Returns:
            @S: output set

        Author: Sung Woo Choi, Date: 10/29/2024
        """
        lp_solver_default = 'gurobi'
        
        if len(args) == 1:
            In = args
            lp_solver = lp_solver_default
            pool = None
        elif len(args) == 2:
            [In, lp_solver] = args
            pool = None
        elif len(args) == 3:
            [In, lp_solver, pool] = args
       
        else:
            raise Exception('error: Invalid \
            number of input arguments, should be 1, 2 or 3')

        assert isinstance(In, list), 'error: inputsets should be in a list'
        S = []
        if pool is None:
            for i in range(0, len(In)):
                S.extend(self.reachExactSingleInput(In[i], lp_solver))
        elif isinstance(pool, multiprocessing.pool.Pool):
            S1 = []
            S1 = S1 + pool.map(self.reachExactSingleInput, zip(In, [lp_solver]*len(In)))
            for i in range(0, len(S1)):
                S.extend(S1[i])
        elif isinstance(pool, ipyparallel.client.view.DirectView):
            raise Exception('error: ipyparallel option is under testing...')
        else:
            raise Exception('error: unknown/unsupport pool type')    
        return S
    
    def reachExactSingleInput(self, *args):
        """
        Exact reachability using stepReach
        Args:
            @I: a single input set
            @lp_solver: lp_solver

        Returns:
            @S: output set

        Author: Sung Woo Choi, Date: 10/29/2024
        """

        if isinstance(args[0], tuple):  # when this method is called in parallel
            args1 = list(args[0])
        else:
            args1 = args
        if len(args1) == 1:
            In = args1
            lp_solver = 'gurobi'
        elif len(args1) == 2:
            [In, lp_solver] = args1
        else:
            raise Exception('error: Invalid \
            number of input arguments, should be 1 or 2')

        if not isinstance(In, ProbStar) and not isinstance(In, Star):
            raise Exception('error: input is not a Star or ProbStar, \
            type of input is {}'.format(type(In)))

        S = []
        S1 = [In]
        for i in range(0, In.dim):
            actv_fun = self.actv_fun[i]
            if actv_fun == 'purelin':
                continue
            elif actv_fun in ['poslin', 'relu']:
                S1 = PosLin.stepReachMultiInputs(S1, i, lp_solver)
            # elif actv_fun == 'leakyrelu':
            #     S1 = LeakyReLU.stepReachMultipleInputs(S1, i, gamma, lp_solver)
            elif actv_fun == 'satlin':
                S1 = SatLin.stepReachMultiInputs(S1, i, lp_solver)
            elif actv_fun == 'satlins':
                S1 = SatLins.stepReachMultiInputs(S1, i, lp_solver)
            else:
                raise Exception(f'error: {actv_fun} activation function is not supported')
            
        S.extend(S1)

        return S
    
    def reachApproxSingleInput(self, In, lp_solver='gurobi', RF=0.0, relu_approx_mode='analytic_gaussian', show=False):
        """
        Approximate reachability on a single Star or ProbStar with mixed neuron activations.
        Groups neurons by activation type, applies vectorized stepReachApprox to each group,
        and reassembles the full state basis while preserving predicate extensions and env_dag.
        """
        if not isinstance(In, (ProbStar, Star)):
            raise Exception(f'error: input is not a Star or ProbStar, type of input is {type(In)}')

        cur = In.clone()
        dim = cur.dim
        assert len(self.actv_fun) == dim, 'error: dimension of input should match length of actv_fun'

        # Group indices by canonical activation name in deterministic order
        groups = [
            ('poslin', [i for i, a in enumerate(self.actv_fun) if a in ('poslin', 'relu')]),
            ('leakyrelu', [i for i, a in enumerate(self.actv_fun) if a == 'leakyrelu']),
            ('satlin', [i for i, a in enumerate(self.actv_fun) if a == 'satlin']),
            ('satlins', [i for i, a in enumerate(self.actv_fun) if a == 'satlins']),
            ('tansig', [i for i, a in enumerate(self.actv_fun) if a in ('tansig', 'tanh')]),
            ('logsig', [i for i, a in enumerate(self.actv_fun) if a in ('logsig', 'sigmoid')]),
        ]
        supported = {'purelin', 'poslin', 'relu', 'leakyrelu', 'satlin', 'satlins', 'tansig', 'tanh', 'logsig', 'sigmoid'}
        for a in self.actv_fun:
            if a not in supported:
                raise Exception(f'error: {a} activation function is not supported')

        for act_name, idx_list in groups:
            if not idx_list:
                continue
            idx_arr = np.asarray(idx_list, dtype=int)
            V_sub = cur.V[idx_arr, :].copy()
            if isinstance(cur, ProbStar):
                sub_set = cur.with_updated_state(V=V_sub)
            else:
                sub_set = Star(V_sub, cur.C, cur.d, cur.pred_lb, cur.pred_ub)

            if act_name == 'poslin':
                out_sub = PosLin.stepReachApprox(
                    In=sub_set, lp_solver=lp_solver, RF=RF,
                    relu_approx_mode=relu_approx_mode, show=show
                )
            elif act_name == 'leakyrelu':
                out_sub = LeakyReLU.stepReachApprox(
                    In=sub_set, lp_solver=lp_solver, RF=RF, show=show
                )
            elif act_name == 'satlin':
                out_sub = SatLin.reachApprox(
                    In=sub_set, lp_solver=lp_solver, show=show
                )
            elif act_name == 'satlins':
                out_sub = SatLins.reachApprox(
                    In=sub_set, lp_solver=lp_solver, show=show
                )
            elif act_name == 'tansig':
                out_sub = TanSig.reachApprox_star(
                    I=sub_set, lp_solver=lp_solver, RF=RF
                )
            elif act_name == 'logsig':
                out_sub = LogSig.reachApprox_star(
                    I=sub_set, lp_solver=lp_solver, RF=RF
                )

            V_full = cur.padBasisMatrix(out_sub.nVars)
            V_full[idx_arr, :] = out_sub.V
            if isinstance(out_sub, ProbStar):
                cur = out_sub.with_updated_state(V=V_full)
            else:
                cur = Star(V_full, out_sub.C, out_sub.d, out_sub.pred_lb, out_sub.pred_ub)

        return cur

    def reach(
        self, In, method='exact', lp_solver='gurobi', pool=None,
        RF=0.0, DR=0, show=False, relu_approx_mode='analytic_gaussian'
    ):
        """main reachability method
           Args:
               @I: a list of input set (Star or ProbStar) or a single Star/ProbStar
               @method: method: 'exact', 'approx', or 'relax'
               @lp_solver: lp solver: 'gurobi' (default), 'glpk', or 'linprog'
               @pool: parallel pool: None or multiprocessing.pool.Pool
               @RF: relax-factor from 0 to 1 (0 by default)

            Return: 
               @R: a list of reachable set or a single reachable set
        """

        if method == 'exact':
            return self.reachExactMultiInputs(In, lp_solver, pool)
        elif method in ('approx', 'relax'):
            if isinstance(In, list):
                return [
                    self.reachApproxSingleInput(
                        s, lp_solver=lp_solver, RF=RF,
                        relu_approx_mode=relu_approx_mode, show=show
                    )
                    for s in In
                ]
            else:
                return self.reachApproxSingleInput(
                    In, lp_solver=lp_solver, RF=RF,
                    relu_approx_mode=relu_approx_mode, show=show
                )
        else:
            raise Exception(f'error: {method} reachability method is not supported')
        
    
    
