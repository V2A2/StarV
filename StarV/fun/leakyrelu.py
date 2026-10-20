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
LeakyReLU Class
Yuntao Li, 1/10/2024
Update: Yuntao Li, Date: 09/16/2025
"""

# !/usr/bin/python3
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.set.probsparseimagestar2dcoo import ProbSparseImageStar2DCOO
from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
from StarV.set.probsparse_base import propagate_leakyrelu_moments
import numpy as np
import scipy.sparse as sp
import copy
import multiprocessing
import ipyparallel

class LeakyReLU:
    """
    LeakyReLU Class contains method for reachability analysis for Layer with LeakyReLU activation function
    Yuntao Li, 1/10/2024
    """

    @staticmethod
    def evaluate(x, gamma):
        """
        Evaluate method for LeakyReLU
        Args:
            @x: input array
            @gamma: leaking factor
        Returns:
            Modified array with LeakyReLU applied

        Author: Yuntao Li, Date: 1/18/2024
        """
        y = np.array(x, dtype=float)  # Ensure the array is of float type for proper arithmetic operations
        negative_indices = y < 0
        y[negative_indices] = gamma * y[negative_indices]
        return y
    

    @staticmethod
    def stepReach(*args):
        """
        StepReach method, compute reachable set for a single step
        Args:
            @I: single star set input
            @index: index of the neuron performing stepLeakyReLU
            @gamma: leaking factor
            @lp_solver: LP solver method
        Returns:
            @S: star output set

        Author: Yuntao Li, Date: 1/18/2024
        Update: Yuntao Li, Date: 09/16/2025
        """
        len_args = len(args)
        if len_args == 3:  # 3 arguments
            [I, index, gamma] = args
            lp_solver = 'gurobi'
        elif len_args == 4:  # 4 arguments
            [I, index, gamma, lp_solver] = args
        else:
            raise Exception('error: \
            Invalid number of input arguments, should be 3 or 4')

        if not isinstance(I, ProbStar) and not isinstance(I, Star):
            raise Exception('error: input is not a Star or ProbStar set, \
            type of input = {}'.format(type(I)))

        # xmin = I.getMin(index, lp_solver) # Star
        xmin, xmax = I.estimateRange(index) # Prob Star
        if xmin >= 0:
            S = []
            S.append(I)
        else:
            xmax = I.getMax(index, lp_solver) # Star
            if xmax <= 0:
                S = []
                S.append(I.resetRowWithFactor(index, gamma))
            else:
                xmin = I.getMin(index, lp_solver)
                if xmin >= 0:
                    S = []
                    S.append(I)
                else:
                    C = np.zeros(I.dim,)
                    C[index] = 1.0
                    d = np.zeros(1,)
                    S1 = copy.deepcopy(I)
                    S2 = copy.deepcopy(I)
                    S1 = S1.addConstraint(C, d, copy_=False)  # x <= 0
                    S1 = S1.resetRowWithFactor(index, gamma, copy_=False)
                    S2 = S2.addConstraint(-C, d, copy_=False)  # x >= 0
                    S = []
                    S.append(S1)
                    S.append(S2)

        return S


    @staticmethod
    def stepReachMultipleInputs(*args):
        """
        StepReach with multiple inputs
        Args:
            @I: an array of stars
            @index: index where stepReach is performed
            @gamma: leaking factor
            @option: parallel computation option
            @lp_solver: LP solver method
        Returns:
            @S: a list of output set

        Author: Yuntao Li, Date: 1/18/2024
        """
        if len(args) == 3:
            [I, index, gamma] = args
            lp_solver = 'gurobi'
        elif len(args) == 4:
            [I, index, gamma, lp_solver] = args
        else:
            raise Exception('error: \
            Invalid number of input arguments, should be 3 or 4 ')

        assert isinstance(I, list), 'error: input is not a list, \
        type of input is {}'.format(type(I))

        S = []
        for i in range(0, len(I)):
            S1 = LeakyReLU.stepReach(I[i], index, gamma, lp_solver)
            S.extend(S1)
        return S


    @staticmethod
    def reachExactSingleInput(*args):
        """
        Exact reachability using stepReach
        Args:
            @I: a single input set
            @lp_solver: lp_solver

        Returns:
            @S: output set

        Author: Yuntao Li, Date: 1/18/2024
        """

        if isinstance(args[0], tuple):  # when this method is called in parallel
            args1 = list(args[0])
        else:
            args1 = args
        if len(args1) == 2:
            [In, gamma] = args1
            lp_solver = 'gurobi'
        elif len(args1) == 3:
            [In, gamma, lp_solver] = args1
        else:
            raise Exception('error: Invalid \
            number of input arguments, should be 1 or 2')

        if not isinstance(In, ProbStar) and not isinstance(In, Star):
            raise Exception('error: input is not a Star or ProbStar, \
            type of input is {}'.format(type(In)))

        S = []
        S1 = [In]
        for i in range(0, In.dim):
            S1 = LeakyReLU.stepReachMultipleInputs(S1, i, gamma, lp_solver)

        S.extend(S1)

        return S


    def reachExactMultiInputs(*args):
        """
        Exact reachability with multiple inputs
        Work with bread-first-search verification

        Args:
            @I: a single input set
            @lp_solver: lp_solver ('gurobi' or 'glpk' or 'linprog')
            @pool: pool for parallel computation
        Returns:
            @S: output set

        Author: Yuntao Li, Date: 1/18/2024
        """
        lp_solver_default = 'gurobi'
        
        if len(args) == 2:
            [In, gamma] = args
            lp_solver = lp_solver_default
            pool = None
        elif len(args) == 3:
            [In, gamma, lp_solver] = args
            pool = None
        elif len(args) == 4:
            [In, gamma, lp_solver, pool] = args
       
        else:
            raise Exception('error: Invalid \
            number of input arguments, should be 1, 2 or 3')

        assert isinstance(In, list), 'error: inputsets should be in a list'
        S = []
        if pool is None:
            for i in range(0, len(In)):
                S.extend(LeakyReLU.reachExactSingleInput(In[i], gamma, lp_solver))
        elif isinstance(pool, multiprocessing.pool.Pool):
            S1 = []
            S1 = S1 + pool.map(LeakyReLU.reachExactSingleInput, zip(In, [gamma]*len(In), [lp_solver]*len(In)))
            for i in range(0, len(S1)):
                S.extend(S1[i])
        elif isinstance(pool, ipyparallel.client.view.DirectView):
            # S1 = pool.map(LeakyReLU.reachExactSingleInput, zip(In, [lp_solver]*len(In)))
            # print('S1 = {}'.format(S1))
            raise Exception('error: ipyparallel option is under testing...')
        else:
            raise Exception('error: unknown/unsupport pool type')    
        return S

    @staticmethod
    def stepReachApprox(In, gamma=0.01, lp_solver='gurobi', RF=0.0, show=False):
        """
        Approximate reachability for LeakyReLU(x) = max(gamma * x, x) (for 0 <= gamma < 1).
        Uses the exact triangle relaxation on ambiguous neurons l < 0 < u:
          y >= gamma * x, y >= x, y <= ((u - gamma * l) / (u - l)) * (x - l) + gamma * l.
        """
        is_prob = isinstance(In, (ProbStar, ProbSparseImageStar2DCSR, ProbSparseImageStar2DCOO))
        if is_prob:
            c_pre_orig, V_pre_orig = In.get_c_V()
            c_pre_orig = c_pre_orig.copy()
            V_pre_orig = V_pre_orig.copy()

        I = copy.deepcopy(In)
        l_all, u_all = I.estimateRanges()

        neg_idx = np.where(u_all <= 0)[0]
        amb_idx = np.where((l_all < 0) & (u_all > 0))[0]

        if len(neg_idx) > 0:
            if hasattr(I, 'resetRowsWithFactor'):
                I = I.resetRowsWithFactor(neg_idx, gamma)
            else:
                # SparseImageStar2DCSR / COO scale rows by gamma
                I.c[neg_idx] *= gamma
                if sp.issparse(I.V):
                    scale = np.ones(I.dim, dtype=I.V.dtype)
                    scale[neg_idx] = gamma
                    I.V = I.V.multiply(scale[:, None]).tocsr() if isinstance(I, SparseImageStar2DCSR) else I.V.multiply(scale[:, None]).tocoo()
                else:
                    I.V[neg_idx, :] *= gamma

        m = len(amb_idx)
        if is_prob:
            if m == 0:
                c_cur, V_cur = I.get_c_V()
                return In.with_updated_state(
                    c=c_cur, V=V_cur, C=I.C, d=I.d,
                    pred_lb=I.pred_lb, pred_ub=I.pred_ub,
                    mu=In.mu.copy(), Sig=In.Sig.copy(), new_env_record=None
                )
            c_amb = c_pre_orig[amb_idx]
            V_amb = V_pre_orig.tocsr()[amb_idx] if sp.issparse(V_pre_orig) else V_pre_orig[amb_idx]
            mu_new, Sig_new, env_rec = propagate_leakyrelu_moments(
                V_amb, c_amb, l_all[amb_idx], u_all[amb_idx],
                gamma, In.mu, In.Sig, cov_mode=In.cov_mode
            )
        elif m == 0:
            return I

        l_u = l_all[amb_idx]
        u_u = u_all[amb_idx]
        slope_u = (u_u - gamma * l_u) / np.maximum(u_u - l_u, 1e-12)
        int_u = gamma * l_u - slope_u * l_u  # ((1 - gamma) * u * l) / (u - l)

        if isinstance(I, (Star, ProbStar)):
            N = I.dim
            n = I.nVars
            dtype = I.V.dtype
            V1 = I.V.copy()
            V1[amb_idx, :] = 0.0
            V2 = np.zeros((N, m), dtype=dtype)
            V2[amb_idx, np.arange(m)] = 1.0
            new_V = np.hstack([V1, V2])
            if len(I.C) == 0:
                C0 = np.empty((0, n + m), dtype=dtype)
                d0 = np.empty((0,), dtype=dtype)
            else:
                C0 = np.hstack([I.C, np.zeros((I.C.shape[0], m), dtype=dtype)])
                d0 = I.d
            V_amb = I.V[amb_idx, 1:n + 1]
            c_amb = I.V[amb_idx, 0]
            E = np.eye(m, dtype=dtype)
            # y >= gamma * x  =>  gamma * V_amb * alpha - y <= -gamma * c_amb
            C1 = np.hstack([gamma * V_amb, -E])
            d1 = -gamma * c_amb
            # y >= x          =>  V_amb * alpha - y <= -c_amb
            C2 = np.hstack([V_amb, -E])
            d2 = -c_amb
            # y <= slope_u * x + int_u  =>  -slope_u * V_amb * alpha + y <= slope_u * c_amb + int_u
            C3 = np.hstack([-slope_u[:, None] * V_amb, E])
            d3 = slope_u * c_amb + int_u
            new_C = np.vstack([C0, C1, C2, C3])
            new_d = np.hstack([d0, d1, d2, d3])
            new_lb = np.hstack([I.pred_lb, gamma * l_u])
            new_ub = np.hstack([I.pred_ub, u_u])
            if isinstance(I, ProbStar):
                return In.with_updated_state(
                    V=new_V, C=new_C, d=new_d, pred_lb=new_lb, pred_ub=new_ub,
                    mu=mu_new, Sig=Sig_new, new_env_record=env_rec
                )
            return Star(new_V, new_C, new_d, new_lb, new_ub)

        elif isinstance(I, (SparseImageStar2DCSR, SparseImageStar2DCOO)):
            amb_idx = amb_idx.astype(np.int32)
            N = I.dim
            n = I.num_pred
            dtype = I.V.dtype
            Ic = I.c[amb_idx].copy()
            new_c = I.c.copy()
            new_c[amb_idx] = 0.0
            V_rem = I.resetRows_V(amb_idx).tocoo(copy=False)
            V_rem.data = np.hstack([V_rem.data, np.ones(m, dtype=dtype)])
            V_rem.row = np.hstack([V_rem.row, amb_idx])
            V_rem.col = np.hstack([V_rem.col, np.arange(m, dtype=np.int32) + n])
            V_rem._shape = (N, n + m)
            new_V = V_rem.tocsr(copy=False) if isinstance(I, SparseImageStar2DCSR) else V_rem

            V1 = I.V[amb_idx, :].tocoo() if isinstance(I, SparseImageStar2DCSR) else I.getRows(amb_idx)
            V_g = V1.multiply(gamma).tocoo()
            V3 = V1.multiply(-slope_u[:, None]).tocoo()
            eye_data = np.ones(m, dtype=dtype)
            eye_col = np.arange(m, dtype=np.int32) + n
            eye_row = np.arange(m, dtype=np.int32)

            d1 = -gamma * Ic
            d2 = -Ic
            d3 = slope_u * Ic + int_u

            data = np.hstack([V_g.data, -eye_data, V1.data, -eye_data, V3.data, eye_data])
            row = np.hstack([V_g.row, eye_row, V1.row + m, eye_row + m, V3.row + 2 * m, eye_row + 2 * m])
            col = np.hstack([V_g.col, eye_col, V1.col, eye_col, V3.col, eye_col])
            C = sp.csr_array((data, (row, col)), shape=(3 * m, n + m), copy=False)
            if I.C.nnz > 0:
                data_c = np.hstack([I.C.data, C.data])
                indices_c = np.hstack([I.C.indices, C.indices])
                indptr_c = np.hstack([I.C.indptr, C.indptr[1:] + I.C.nnz])
                new_C = sp.csr_array((data_c, indices_c, indptr_c), shape=(I.C.shape[0] + C.shape[0], C.shape[1]), copy=False)
                new_d = np.hstack([I.d, d1, d2, d3])
            else:
                new_C = C
                new_d = np.hstack([d1, d2, d3])
            new_lb = np.hstack([I.pred_lb, gamma * l_u])
            new_ub = np.hstack([I.pred_ub, u_u])
            if is_prob:
                return In.with_updated_state(
                    c=new_c, V=new_V, C=new_C, d=new_d,
                    pred_lb=new_lb, pred_ub=new_ub, shape=I.shape,
                    mu=mu_new, Sig=Sig_new, new_env_record=env_rec
                )
            cls = SparseImageStar2DCSR if isinstance(I, SparseImageStar2DCSR) else SparseImageStar2DCOO
            return cls(new_c, new_V, new_C, new_d, new_lb, new_ub, I.shape, copy_=False)
        else:
            raise Exception(f'Unsupported set type for LeakyReLU.stepReachApprox: {type(I)}')

    @staticmethod
    def reachApproxSingleInput(In, gamma=0.01, lp_solver='gurobi', RF=0.0, show=False):
        return LeakyReLU.stepReachApprox(In=In, gamma=gamma, lp_solver=lp_solver, RF=RF, show=show)

    @staticmethod
    def reachApprox(In, gamma=0.01, lp_solver='gurobi', RF=0.0, show=False):
        if isinstance(In, list):
            return [LeakyReLU.stepReachApprox(x, gamma=gamma, lp_solver=lp_solver, RF=RF, show=show) for x in In]
        return LeakyReLU.stepReachApprox(In, gamma=gamma, lp_solver=lp_solver, RF=RF, show=show)
