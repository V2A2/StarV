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
Main Verifier Class
Dung Tran, 9/10/2022
"""

from StarV.net.network import NeuralNetwork, reachExactBFS, reachApproxBFS
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
from StarV.set.probsparseimagestar2dcoo import ProbSparseImageStar2DCOO
from StarV.spec.dProbStarTL import Formula
from StarV.plant.dlode import DLODE
from StarV.plant.lode import LODE
import time
import copy
import multiprocessing
import numpy as np
from scipy.stats import beta as sp_beta
import polytope as pc
from StarV.util.print_util import print_util

class Verifier(object):
    """
       Verifier Class

       Properties: (Verification Settings)

        @lp_solver: lp solver: 'gurobi' (default), 'glpk', 'linprog'
        @method: verification method: 'BFS' ("bread-first-search"), 'exact', 'approx', or 'MC' ("Monte Carlo")
        @n_processes: number of processes used for verification
        @time_out: time_out for a single verification query (single input)

      Methods:
        @verify: main verification method (automatically dispatches qualitative/quantitative verification)
        @quantiVerify: quantitative verification for ProbStar input sets
        @qualiVerify: qualitative verification for Star input sets
        @quantiVerifyMC: Monte Carlo quantitative verification for a ProbStar input set
        @evaluate: evaluate network on a specific input or batch of samples
    """

    def __init__(self, lp_solver='gurobi', method='BFS', n_processes=1, time_out=None):
        self.lp_solver = lp_solver
        self.method = method
        self.n_processes = n_processes
        self.time_out = time_out

    def verify(self, net, inputSet, unsafe_mat, unsafe_vec, p_filter=0.0, reachMethod='exact', show=True):
        """Main verification method.

        Dispatches to quantitative verification if inputSet contains ProbStar(s)
        or ProbSparseImageStar2DCSR/COO, or qualitative verification if inputSet
        contains Star(s).
        """
        assert isinstance(net, NeuralNetwork), 'error: input is not a NeuralNetwork object'
        first_set = inputSet[0] if isinstance(inputSet, list) else inputSet
        if isinstance(first_set, (ProbStar, ProbSparseImageStar2DCSR, ProbSparseImageStar2DCOO)):
            if self.method == 'MC':
                return self.quantiVerifyMC(net, first_set, unsafe_mat, unsafe_vec)
            if reachMethod == 'approx' or self.method == 'approx' or isinstance(first_set, (ProbSparseImageStar2DCSR, ProbSparseImageStar2DCOO)):
                return quantiVerifyApproxBFS(
                    net=net, inputSet=first_set, unsafe_mat=unsafe_mat, unsafe_vec=unsafe_vec,
                    lp_solver=self.lp_solver, show=show
                )
            return self.quantiVerify(net, inputSet, unsafe_mat, unsafe_vec, p_filter=p_filter, show=show)
        elif isinstance(first_set, Star):
            return self.qualiVerify(net, inputSet, unsafe_mat, unsafe_vec, reachMethod=reachMethod, show=show)
        else:
            raise TypeError(f'Unsupported inputSet element type: {type(first_set)}')

    def quantiVerify(self, net, inputSet, unsafe_mat, unsafe_vec, p_filter=0.0, show=True):
        """Quantitative verification of a NeuralNetwork with ProbStar inputSet."""
        assert isinstance(net, NeuralNetwork), 'error: input is not a NeuralNetwork object'
        if isinstance(inputSet, ProbStar):
            inputSet = [inputSet]
        if p_filter == 0.0 and self.method == 'exact':
            return quantiVerifyExactBFS(
                net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, unsafe_vec=unsafe_vec,
                lp_solver=self.lp_solver, numCores=self.n_processes, show=show
            )
        return quantiVerifyBFS(
            net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, unsafe_vec=unsafe_vec,
            p_filter=p_filter, lp_solver=self.lp_solver, numCores=self.n_processes, show=show
        )

    def qualiVerify(self, net, inputSet, unsafe_mat, unsafe_vec, reachMethod='exact', show=True):
        """Qualitative verification of a NeuralNetwork with Star inputSet."""
        assert isinstance(net, NeuralNetwork), 'error: input is not a NeuralNetwork object'
        if reachMethod == 'approx' or self.method == 'approx':
            single_set = inputSet[0] if isinstance(inputSet, list) else inputSet
            return qualiVerifyApproxBFS(
                net=net, inputSet=single_set, unsafe_mat=unsafe_mat, unsafe_vec=unsafe_vec,
                lp_solver=self.lp_solver, show=show
            )
        if isinstance(inputSet, Star):
            inputSet = [inputSet]
        return qualiVerifyExactBFS(
            net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, unsafe_vec=unsafe_vec,
            lp_solver=self.lp_solver, numCores=self.n_processes, show=show
        )

    def quantiVerifyMC(self, net, inputSet, unsafe_mat, unsafe_vec, numSamples=100000, nTimes=10):
        """Monte Carlo quantitative verification of a NeuralNetwork with a ProbStar inputSet."""
        assert isinstance(net, NeuralNetwork), 'error: input is not a NeuralNetwork object'
        single_set = inputSet[0] if isinstance(inputSet, list) else inputSet
        return quantiVerifyMC(
            net=net, inputSet=single_set, unsafe_mat=unsafe_mat, unsafe_vec=unsafe_vec,
            numSamples=numSamples, nTimes=nTimes, numCores=self.n_processes
        )

    def evaluate(self, net, samples):
        """Evaluate network on a set of samples."""
        return evaluate(net, samples)


def checkSafetyStar(*args):
    """Intersect with unsafe region, can work in parallel"""

    if isinstance(args[0], tuple):
        args1 = args[0]
    else:
        args1 = args

    unsafe_mat = args1[0]
    unsafe_vec = args1[1]
    S = args1[2]
    C = unsafe_mat
    d = unsafe_vec
    assert isinstance(C, np.ndarray), 'error: constraint matrix should be a numpy array'
    assert isinstance(d, np.ndarray) and len(d.shape) == 1, 'error: constraint vector \
    should be a 1D numpy array'
    assert C.shape[0] == d.shape[0], 'error: inconsistency between constraint matrix and \
    constraint vector'

    P = copy.deepcopy(S)
    v = np.matmul(C, P.V)
    newC = v[:, 1:P.nVars+1]
    newd = d - v[:,0]

    if len(P.C) != 0:
        P.C = np.vstack((newC, P.C))
        P.d = np.concatenate([newd, P.d])
    else:
        if len(newC.shape) == 1:
            P.C = newC.reshape(1, P.nVars)
        else:
            P.C = newC
        P.d = newd

    if P.isEmptySet():
        P = []
    return P


def checkSafetyProbStar(*args):
    """Intersect with unsafe region, can work in parallel"""

    if isinstance(args[0], tuple):
        args1 = args[0]
    else:
        args1 = args
    unsafe_mat = args1[0]
    unsafe_vec = args1[1]
    S = args1[2]
    C = unsafe_mat
    d = unsafe_vec
    assert isinstance(C, np.ndarray), 'error: constraint matrix should be a numpy array'
    assert isinstance(d, np.ndarray) and len(d.shape) == 1, 'error: constraint vector \
    should be a 1D numpy array'
    assert C.shape[0] == d.shape[0], 'error: inconsistency between constraint matrix and \
    constraint vector'

    P = copy.deepcopy(S)
    v = np.matmul(C, P.V)
    newC = v[:, 1:P.nVars+1]
    newd = d - v[:,0]

    if len(P.C) != 0:
        P.C = np.vstack((newC, P.C))
        P.d = np.concatenate([newd, P.d])
    else:
        if len(newC.shape) == 1:
            P.C = newC.reshape(1, P.nVars)
        else:
            P.C = newC
        P.d = newd

    if P.isEmptySet():
        P = []
        prob = 0.0
    else:
        prob = P.estimateProbability()
        
    return P, prob


def filterProbStar(*args):
    """Filtering out some probstars"""

    if isinstance(args[0], tuple):
        args1 = args[0]
    else:
        args1 = args
    p_filter = args1[0]
    S = args1[1]
    assert isinstance(S, ProbStar), 'error: input is not a probstar'
    prob = S.estimateProbability()
    # print_util('h4')
    # print("prob = ", prob)
    # print_util('h4')
    if prob >= p_filter:
        P = S
        p_ignored = 0.0
    else:
        P = []
        p_ignored = prob

    return P, p_ignored
    
    
def quantiVerifyExactBFS(net, inputSet, unsafe_mat, unsafe_vec, lp_solver='gurobi', numCores=1, show=True):
    """Quantitative Verification of ReLU Networks using exact bread-first-search"""

    if numCores > 1:
        pool = multiprocessing.Pool(numCores)
    else:
        pool = None
    S = reachExactBFS(net, inputSet, lp_solver, pool, show)  # output set
    P = []  # unsafe output set
    prob = []  # probability of unsafe output set
    if pool is None:
        for S1 in S:
            P1, prob1 = checkSafetyProbStar(unsafe_mat, unsafe_vec, S1)
            if isinstance(P1, ProbStar):
                P.append(P1)
                prob.append(prob1)
    else:
        S1 = pool.map(checkSafetyProbStar, zip([unsafe_mat]*len(S), [unsafe_vec]*len(S), S))
        pool.close()
        pool.join()
        for S2 in S1:
            if isinstance(S2[0], ProbStar):
                P.append(S2[0])
                prob.append(S2[1])
          
    return S, P, sum(prob)


def quantiVerifyBFS(net, inputSet, unsafe_mat, unsafe_vec,  p_filter=0.0, lp_solver='gurobi', numCores=1, show=True):
    """ Overapproximate quantitative verification of ReLU network"""


    inputProb = inputSet[0].estimateProbability()

    assert p_filter >= 0.0, 'error: invalid filtering probability'

    if p_filter == 0.0:
        S, P, p_v = quantiVerifyExactBFS(net, inputSet, unsafe_mat, unsafe_vec, lp_solver, numCores, show)
        p_v_ub = p_v
        p_v_lb = p_v
    else:
        if numCores > 1:
            pool = multiprocessing.Pool(numCores)
        else:
            pool = None
        # compute and filter reachable sets
        I = copy.deepcopy(inputSet)
        p_ignored = 0.0
        for i in range(0, net.n_layers):
            if show:
                print('================ Layer {} ================='.format(i))
                print('Computing layer {} reachable set...'.format(i))
            S = net.layers[i].reach(I, method='exact', lp_solver=lp_solver, pool=pool)
            if show:
                print('Number of probstars: {}'.format(len(S)))
                print('Filtering probstars whose probabilities < {}...'.format(p_filter))
            P = []
            if pool is None:
                for S1 in S:
                    P1, prob1 = filterProbStar(p_filter, S1)
                    if isinstance(P1, ProbStar):
                        P.append(P1)
                    p_ignored = p_ignored + prob1  # update the total probability of ignored sets
            else:
                S1 = pool.map(filterProbStar, zip([p_filter]*len(S), S))
                for S2 in S1:
                    if isinstance(S2[0], ProbStar):
                        P.append(S2[0])
                    p_ignored = p_ignored + S2[1]
            I = P            
            if show:
                print('Number of ignored probstars: {}'.format(len(S) - len(I)))
                print('Number of remaining probstars: {}'.format(len(I)))

            if len(I) == 0:
                break
            
        if len(I) == 0:
            if pool is not None:
                pool.close()
                pool.join()
            p_v_lb = p_ignored
            p_v_ub = p_ignored
            S = [] # empty output set
            P = [] # empty unsafe outputset
        else:          
            # verify output reachable sets
            P = []
            prob = []
            if pool is None:
                for S1 in I:
                    P1, prob1 = checkSafetyProbStar(unsafe_mat, unsafe_vec, S1)
                    if isinstance(P1, ProbStar):
                        P.append(P1)
                        prob.append(prob1)
            else:           
                S1 = pool.map(checkSafetyProbStar, zip([unsafe_mat]*len(I), [unsafe_vec]*len(I), I))
                pool.close()
                pool.join()
                for S2 in S1:
                    if isinstance(S2[0], ProbStar):
                        P.append(S2[0])
                        prob.append(S2[1])

            p_v_lb = sum(prob)
            p_v_ub = p_v_lb + p_ignored
            S = I

    # estimate maximum and minimum of probability of violating for entire infinite input space
    p_max = p_v_ub + 1.0 - inputSet[0].estimateProbability()
    p_min = p_v_lb

    # obtain counterexample sets
    C = []
    if len(P) > 0:
        for P1 in P:
            C1 = ProbStar(inputSet[0].V, P1.C, P1.d, P1.mu, P1.Sig, P1.pred_lb, P1.pred_ub)
            C.append(C1)
        

    return S, P, C, min(inputProb, p_v_lb), min(inputProb, p_v_ub), min(inputProb, p_min), min(1.0, p_max)


def evaluate(*args):
    """evaluate the network on a set of samples"""
    
    if isinstance(args[0], tuple):
        args1 = args[0]
    else:
        args1 = args

    net = args1[0]
    samples = args1[1]
    
    assert isinstance(net, NeuralNetwork), 'error: net should be a NeuralNetwork object'
    x = samples
    for layer in net.layers:
        y = layer.evaluate(x)
        x = y

    return y


def checkSafetyPoints(*args):
    'check safety for a single point'

    if isinstance(args[0], tuple):
        args1 = args[0]
    else:
        args1 = args

    unsafe_mat = args1[0]
    unsafe_vec = args1[1]
    points = args1[2]
    
    P = pc.Polytope(unsafe_mat, unsafe_vec)

    n = points.shape[1]
    nSAT = 0
    for i in range(0,n):
        y1 = points[:, i]
        if y1 in P:
            nSAT = nSAT + 1
            
    return nSAT, n 


def quantiVerifyMC(net, inputSet, unsafe_mat, unsafe_vec, numSamples=100000, nTimes=10, numCores=1):
    'quantitative verification using traditional Monte Carlo sampling-based method'

    assert isinstance(inputSet, ProbStar), 'error: input set should be a probstar object'
    assert nTimes >= 1, 'error: invalid number of times for computing avarage probSAT'

    probSAT = 0
    for i in range(0, nTimes):
        
        samples = inputSet.sampling(numSamples)

        if numCores > 1:
            pool = multiprocessing.Pool(numCores)
            # divide samples into N batches, N = numCores
            nBatchs = numCores
            batchSize = int(np.floor(numSamples/numCores))
            I = []
            for j in range(0, nBatchs):
                if j == 0:
                    start_ID = 0
                else:
                    start_ID = start_ID + batchSize

                if j != nBatchs-1:
                    y1 = samples[:, start_ID:start_ID+batchSize]
                else:
                    y1 = samples[:, start_ID:samples.shape[1]]

                I.append(y1)

        else:
            pool = None

        if pool is None:
            y = evaluate(net, samples)
            nSAT, n = checkSafetyPoints(unsafe_mat, unsafe_vec, y)

        else:
            y = pool.map(evaluate, zip([net]*nBatchs, I))
            S = pool.map(checkSafetyPoints, zip([unsafe_mat]*nBatchs, [unsafe_vec]*nBatchs, y))
            pool.close()
            pool.join()

            nSAT = 0
            n = 0
            for S1 in S:
                nSAT = nSAT + S1[0]
                n = n + S1[1]

        probSAT = probSAT + float(nSAT/n)

    probSAT = probSAT/nTimes

    return probSAT


def qualiVerifyExactBFS(net, inputSet, unsafe_mat, unsafe_vec, lp_solver='gurobi', numCores=1, show=True):
    """Qualitative Verification of FFNN Networks using exact bread-first-search"""

    if numCores > 1:
        pool = multiprocessing.Pool(numCores)
    else:
        pool = None
    S = reachExactBFS(net, inputSet, lp_solver, pool, show)  # output set
    P = []  # unsafe output set
    Res = [] # Verification result

    if pool is None:
        for S1 in S:
            P1 = checkSafetyStar(unsafe_mat, unsafe_vec, S1)
            if isinstance(P1, Star):
                P.append(P1)
                Res.append('SAT')
            else:
                Res.append('UNSAT')
    else:
        S1 = pool.map(checkSafetyStar, zip([unsafe_mat]*len(S), [unsafe_vec]*len(S), S))
        pool.close()
        pool.join()
        for S2 in S1:
            if isinstance(S2, Star):
                P.append(S2)
                Res.append('SAT')
            else:
                Res.append('UNSAT')

    # obtain counterexample sets
    C = []
    if len(P) > 0:
        for P1 in P:
            C1 = Star(inputSet[0].V, P1.C, P1.d, P1.pred_lb, P1.pred_ub)
            C.append(C1)

    return S, P, C, Res

def qualiVerifyApproxBFS(net, inputSet, unsafe_mat, unsafe_vec, lp_solver='gurobi', show=True):
    """ Overapproximate qualitative verification of ReLU network"""

    S = reachApproxBFS(net, inputSet, lp_solver=lp_solver, show=show)  # output set

    P = checkSafetyStar(unsafe_mat, unsafe_vec, S)
    if not isinstance(P, Star):
        P = []
        Res = 'UNSAT'
    else:
        Res = 'UNKNOWN'

    return S, P, Res


def quantiVerifyApproxBFS(
    net, inputSet, unsafe_mat, unsafe_vec,
    lp_solver='gurobi', RF=0.0, DR=0,
    relu_approx_mode='analytic_gaussian', use_lp=False, show=False
):
    """
    Single-set approximate quantitative verification for ProbStar,
    ProbSparseImageStar2DCSR, and ProbSparseImageStar2DCOO.

    Propagates a single probabilistic star set through `net` using analytic
    moment matching and linear envelope DAG recording, then verifies the
    specification `unsafe_mat * y <= unsafe_vec` using Two-Stage LP +
    Dual Back-Substitution + Moment Estimation.

    Returns:
        S_out: Output ProbStar / ProbSparseImageStar2DCSR / ProbSparseImageStar2DCOO
        res: dict with keys:
          'status', 'p_min', 'p_max', 'p_hat', 'p_input',
          'cond_p_min', 'cond_p_max', 'cond_p_hat'
    """
    if isinstance(inputSet, list):
        assert len(inputSet) == 1, 'quantiVerifyApproxBFS expects a single probabilistic star set'
        inputSet = inputSet[0]

    t0 = time.perf_counter()
    S_out = reachApproxBFS(
        net, inputSet, lp_solver=lp_solver, show=show,
        RF=RF, DR=DR, relu_approx_mode=relu_approx_mode
    )
    res = S_out.verifySpec(unsafe_mat, unsafe_vec, lp_solver=lp_solver, use_lp=use_lp)
    dt = time.perf_counter() - t0
    res['total_time'] = dt
    res['verification_time'] = dt
    return S_out, res


def clopper_pearson_interval(k, n, confidence=0.99):
    """Exact Clopper-Pearson binomial confidence interval for k successes out of n."""
    alpha = 1.0 - float(confidence)
    if k <= 0:
        lo = 0.0
    else:
        lo = float(sp_beta.ppf(alpha / 2.0, k, n - k + 1))
    if k >= n:
        hi = 1.0
    else:
        hi = float(sp_beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def quantiVerifyVisionMC(
    net, inputSet, unsafe_mat=None, unsafe_vec=None, true_label=None,
    numSamples=10000, seed=42, confidence=0.99
):
    """
    Fast vectorized Monte Carlo quantitative verification for high-dimensional
    vision models on ProbStar, ProbSparseImageStar2DCSR, or ProbSparseImageStar2DCOO.

    Samples alpha_0 from the truncated Gaussian input distribution, evaluates `net`,
    and returns both conditional (given input lies in box) and unconditional
    probabilities along with exact Clopper-Pearson confidence intervals.
    """
    if isinstance(inputSet, list):
        inputSet = inputSet[0]

    t0 = time.perf_counter()
    p_input = inputSet.get_input_probability()
    if isinstance(inputSet, ProbStar):
        alpha0 = inputSet.sampling_alpha0(numSamples, seed=seed)
        c0, V0 = inputSet.get_c_V()
        samples = c0[:, None] + V0[:, :inputSet.nVars0] @ alpha0
        y_out = net.evaluate(samples)
    else:
        # ProbSparseImageStar2DCSR or ProbSparseImageStar2DCOO
        samples = inputSet.sampling(numSamples, seed=seed, reshape_4d=True)
        # Evaluate batch or sample-by-sample depending on network layer support
        try:
            y_out = net.evaluate(samples)
            if y_out.ndim > 2:
                y_out = y_out.reshape(-1, numSamples)
        except Exception:
            y_list = []
            for s_idx in range(numSamples):
                yi = net.evaluate(samples[:, :, :, s_idx]).ravel()
                y_list.append(yi)
            y_out = np.column_stack(y_list)

    if y_out.ndim == 1:
        y_out = y_out[:, None]
    if y_out.shape[1] != numSamples and y_out.shape[0] == numSamples:
        y_out = y_out.T

    if true_label is not None:
        # Certify classification robustness: y[k] - y[true_label] <= 0 for all k != true_label
        preds = np.argmax(y_out, axis=0)
        sat_mask = (preds == int(true_label))
    else:
        assert unsafe_mat is not None and unsafe_vec is not None
        U = np.asarray(unsafe_mat, dtype=np.float64)
        u_v = np.asarray(unsafe_vec, dtype=np.float64).ravel()
        lhs = U @ y_out  # (K, numSamples)
        sat_mask = np.all(lhs <= u_v[:, None] + 1e-12, axis=0)

    k_sat = int(np.sum(sat_mask))
    cond_p_mc = float(k_sat) / float(numSamples)
    cond_lo, cond_hi = clopper_pearson_interval(k_sat, numSamples, confidence=confidence)

    dt = time.perf_counter() - t0
    return {
        'k_sat': k_sat,
        'num_samples': int(numSamples),
        'p_input': float(p_input),
        'cond_p_mc': cond_p_mc,
        'cond_ci_low': cond_lo,
        'cond_ci_high': cond_hi,
        'p_mc': cond_p_mc * p_input,
        'ci_low': cond_lo * p_input,
        'ci_high': cond_hi * p_input,
        'verify_time': dt,
        'verification_time': dt,
    }


# def quantiVerifyProbStarTL(model, spec, timeStep, numSteps, X0=None, U=None):
#     'quantitative verification of probstar temporal logic specification'

    
#     assert isinstance(model, DLODE) or isinstance(model, LODE), 'error: model should be a linear ode or discrete linear ode object'
#     assert timeStep > 0, 'error: invalid timeStep'
#     assert numSteps >=1, 'error: invalid number of time steps'
#     assert isinstance(spec, Formula), 'error: specification should be a Formula object'


#     Xt = model.multiStepReach(timeStep, X0=X0, U=U, k=numSteps)
#     S  = spec.render(Xt)

#     probSAT = None
#     if spec.formula_type == 'ConjunctiveAlways':
#         probSAT = S.estimateProbability()
#     else:
#         raise RuntimeError('currently support only conjunctive always formula_type')
    

#     return probSAT, Xt
