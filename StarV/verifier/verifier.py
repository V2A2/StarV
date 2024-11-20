"""
Main Verifier Class
Dung Tran, 9/10/2022
"""

from StarV.net.network import NeuralNetwork, reachExactBFS
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
from StarV.spec.dProbStarTL import Formula
from StarV.plant.dlode import DLODE
from StarV.plant.lode import LODE
import copy
import multiprocessing
import numpy as np
import polytope as pc
from StarV.util.print_util import print_util
from typing import Union, Tuple, List


class Verifier(object):
    """
       Verifier Class

       Properties: (Verification Settings)

        @lp_solver: lp solver: 'gurobi' (default), 'glpk', 'linprog'
        @method: verification method: BFS "bread-first-search" or DFS "depth-first-search"
        @n_processes: number of processes used for verification
        @time_out: time_out for a single verification querry (single input)

      Methods:
        @verify: main verification method
        @evaluate: evaluate method on a specific input 

    """

    def __init__(self, lp_solver='gurobi', method='BFS', n_processes=1, time_out=None):
        self.lp_solver = lp_solver
        self.method = method
        self.n_processes = n_processes
        self.time_out = time_out

    def verify(self, net, inputSet):
        """main verification method"""

        assert isinstance(net, NeuralNetwork), 'error: input is not a NeuralNetwork object'
        pass

def checkSafetyStar(unsafe_mat: np.ndarray, unsafe_vec: np.ndarray, S: Star) -> Union[Star, List]:
    """
    Intersect Star with unsafe region.

    Args:
        unsafe_mat (np.ndarray): Constraint matrix.
        unsafe_vec (np.ndarray): Constraint vector.
        S (Star): Star object to check.

    Returns:
        Union[Star, List]: Intersected Star or empty list if no intersection.

    Raises:
        ValueError: If inputs are not of correct type or shape.
    """
    if not isinstance(unsafe_mat, np.ndarray) or not isinstance(unsafe_vec, np.ndarray):
        raise ValueError('Constraint matrix and vector should be numpy arrays')
    if unsafe_vec.ndim != 1 or unsafe_mat.shape[0] != unsafe_vec.shape[0]:
        raise ValueError('Inconsistency between constraint matrix and vector')

    P = S.clone()
    # v = np.matmul(unsafe_mat, P.V)
    v = unsafe_mat @ P.V
    newC = v[:, 1:P.nVars+1]
    newd = unsafe_vec - v[:, 0]

    if len(P.C) != 0:
        P.C = np.vstack((newC, P.C))
        P.d = np.concatenate([newd, P.d])
    else:
        P.C = newC.reshape(1, P.nVars) if newC.ndim == 1 else newC
        P.d = newd

    return P if not P.isEmptySet() else []

def checkSafetyProbStar(*args) -> Tuple[Union[ProbStar, List], float]:
    """
    Intersect ProbStar with unsafe region.

    Args:
        args: Tuple containing (unsafe_mat, unsafe_vec, S) or individual arguments.

    Returns:
        Tuple[Union[ProbStar, List], float]: Intersected ProbStar (or empty list) and probability.

    Raises:
        ValueError: If inputs are not of correct type or shape.
    """
    if isinstance(args[0], tuple):
        unsafe_mat, unsafe_vec, S = args[0]
    else:
        unsafe_mat, unsafe_vec, S = args

    if not isinstance(unsafe_mat, np.ndarray) or not isinstance(unsafe_vec, np.ndarray):
        raise ValueError('Constraint matrix and vector should be numpy arrays')
    if unsafe_vec.ndim != 1 or unsafe_mat.shape[0] != unsafe_vec.shape[0]:
        raise ValueError('Inconsistency between constraint matrix and vector')

    P = S.clone()
    v = unsafe_mat @ P.V
    newC = v[:, 1:P.nVars+1]
    newd = unsafe_vec - v[:, 0]

    if len(P.C) != 0:
        P.C = np.vstack((newC, P.C))
        P.d = np.concatenate([newd, P.d])
    else:
        P.C = newC.reshape(1, P.nVars) if newC.ndim == 1 else newC
        P.d = newd

    if P.isEmptySet():
        return [], 0.0
    else:
        prob = P.estimateProbability()
        return P, prob

def filterProbStar(*args) -> Tuple[Union[ProbStar, List], float]:
    """
    Filter out ProbStars based on probability threshold.

    Args:
        args: Tuple containing (p_filter, S) or individual arguments.

    Returns:
        Tuple[Union[ProbStar, List], float]: Filtered ProbStar (or empty list) and ignored probability.

    Raises:
        ValueError: If input is not a ProbStar.
    """
    if isinstance(args[0], tuple):
        p_filter, S = args[0]
    else:
        p_filter, S = args

    if not isinstance(S, ProbStar):
        raise ValueError('Input is not a ProbStar')

    prob = S.estimateProbability()
    return (S, 0.0) if prob >= p_filter else ([], prob)

def quantiVerifyExactBFS(net, inputSet, unsafe_mat, unsafe_vec, lp_solver='gurobi', numCores=1, show=True):
    """Quantitative Verification of ReLU Networks using exact bread-first-search"""
    
    pool = multiprocessing.Pool(numCores) if numCores > 1 else None
    S = reachExactBFS(net, inputSet, lp_solver, pool, show)  # output set
    P = []  # unsafe output set
    prob = []  # probability of unsafe output set
    
    if pool is None:
        for S1 in S:
            P1, prob1 = checkSafetyProbStar(unsafe_mat, unsafe_vec, S1)
            if isinstance(P1, ProbStar):
                print(P1.__str__())
                P.append(P1)
                prob.append(prob1)
    else:
        # S1 = pool.starmap(checkSafetyProbStar, [(unsafe_mat, unsafe_vec, s) for s in S])
        S1 = pool.map(checkSafetyProbStar, zip([unsafe_mat]*len(S), [unsafe_vec]*len(S), S))
        pool.close()
        for S2 in S1:
            if isinstance(S2[0], ProbStar):
                P.append(S2[0])
                prob.append(S2[1])

    print('length of unsafe sets: ', len(P))
          
    return S, P, sum(prob)

def quantiVerifyBFS(net, inputSet, unsafe_mat, unsafe_vec, p_filter=0.0, lp_solver='gurobi', numCores=1, show=True):
    """ Overapproximate quantitative verification of ReLU network"""

    inputProb = inputSet[0].estimateProbability()
    
    pool = multiprocessing.Pool(numCores) if numCores > 1 else None

    if p_filter < 0.0:
        raise ValueError('Invalid filtering probability')

    if p_filter == 0.0:
        S, P, p_v = quantiVerifyExactBFS(net, inputSet, unsafe_mat, unsafe_vec, lp_solver, numCores, show)
        p_v_ub = p_v_lb = p_v
    else:
        # compute and filter reachable sets
        I = [probstar.clone() for probstar in inputSet]
        p_ignored = 0.0
        for i in range(net.n_layers):
            if show:
                print(f'================ Layer {i} =================')
                print(f'Computing layer {i} reachable set...')
            S = net.layers[i].reach(I, method='exact', lp_solver=lp_solver, pool=pool)
            if show:
                print(f'Number of probstars: {len(S)}')
                print(f'Filtering probstars whose probabilities < {p_filter}...')
            P = []
            if pool is None:
                for S1 in S:
                    P1, prob1 = filterProbStar(p_filter, S1)
                    if isinstance(P1, ProbStar):
                        P.append(P1)
                    p_ignored += prob1  # update the total probability of ignored sets
            else:
                # S1 = pool.starmap(filterProbStar, [(p_filter, s) for s in S])
                S1 = pool.map(filterProbStar, zip([p_filter]*len(S), S))
                for S2 in S1:
                    if isinstance(S2[0], ProbStar):
                        P.append(S2[0])
                    p_ignored += S2[1]
            I = P            
            if show:
                print(f'Number of ignored probstars: {len(S) - len(I)}')
                print(f'Number of remaining probstars: {len(I)}')

            if len(I) == 0:
                break
            
        if len(I) == 0:
            p_v_lb = p_v_ub = p_ignored
            S = P = []  # empty output set and unsafe outputset
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
                # S1 = pool.starmap(checkSafetyProbStar, [(unsafe_mat, unsafe_vec, s) for s in I])
                S1 = pool.map(checkSafetyProbStar, zip([unsafe_mat]*len(I), [unsafe_vec]*len(I), I))
                pool.close()
                for S2 in S1:
                    if isinstance(S2[0], ProbStar):
                        P.append(S2[0])
                        prob.append(S2[1])

            p_v_lb = sum(prob)
            p_v_ub = p_v_lb + p_ignored
            S = I

    # estimate maximum and minimum of probability of violating for entire infinite input space
    p_max = p_v_ub + 1.0 - inputProb
    p_min = p_v_lb

    # obtain counterexample sets
    C = [ProbStar(inputSet[0].V, P1.C, P1.d, P1.mu, P1.Sig, P1.pred_lb, P1.pred_ub) 
         for P1 in P if not ProbStar(inputSet[0].V, P1.C, P1.d, P1.mu, P1.Sig, P1.pred_lb, P1.pred_ub).isEmptySet()]

    return S, P, C, min(inputProb, p_v_lb), min(inputProb, p_v_ub), min(inputProb, p_min), min(1.0, p_max)

def evaluate(*args) -> np.ndarray:
    """Evaluate the network on a set of samples"""
    args1 = args[0] if isinstance(args[0], tuple) else args
    net, samples = args1

    if not isinstance(net, NeuralNetwork):
        raise ValueError('net should be a NeuralNetwork object')

    x = samples
    for layer in net.layers:
        x = layer.evaluate(x)
    return x

def checkSafetyPoints(*args) -> Tuple[int, int]:
    """Check safety for a set of points"""
    args1 = args[0] if isinstance(args[0], tuple) else args
    unsafe_mat, unsafe_vec, points = args1

    P = pc.Polytope(unsafe_mat, unsafe_vec)
    n = points.shape[1]
    nSAT = sum(1 for i in range(n) if points[:, i] in P)

    return nSAT, n

def quantiVerifyMC(net: NeuralNetwork, inputSet: ProbStar, unsafe_mat: np.ndarray, unsafe_vec: np.ndarray, 
                   numSamples: int = 100000, nTimes: int = 10, numCores: int = 1) -> float:
    """Quantitative verification using traditional Monte Carlo sampling-based method"""
    if not isinstance(inputSet, ProbStar):
        raise ValueError('input set should be a ProbStar object')
    if nTimes < 1:
        raise ValueError('invalid number of times for computing average probSAT')

    probSAT = 0
    for _ in range(nTimes):
        samples = inputSet.sampling(numSamples)
        
        if numCores > 1:
            with multiprocessing.Pool(numCores) as pool:
                batchSize = numSamples // numCores
                batches = [samples[:, i:i+batchSize] for i in range(0, numSamples, batchSize)]
                y = pool.starmap(evaluate, [(net, batch) for batch in batches])
                results = pool.starmap(checkSafetyPoints, [(unsafe_mat, unsafe_vec, output) for output in y])
                nSAT = sum(result[0] for result in results)
                n = sum(result[1] for result in results)
        else:
            y = evaluate(net, samples)
            nSAT, n = checkSafetyPoints(unsafe_mat, unsafe_vec, y)

        probSAT += float(nSAT / n)

    return probSAT / nTimes

def quantiVerifyProbStarTL(model: Union[DLODE, LODE], spec: Formula, timeStep: float, numSteps: int, 
                           X0: Union[ProbStar, None] = None, U: Union[ProbStar, None] = None) -> Tuple[float, List[ProbStar]]:
    """Quantitative verification of ProbStar temporal logic specification"""
    if not isinstance(model, (DLODE, LODE)):
        raise ValueError('model should be a linear ODE or discrete linear ODE object')
    if timeStep <= 0:
        raise ValueError('invalid timeStep')
    if numSteps < 1:
        raise ValueError('invalid number of time steps')
    if not isinstance(spec, Formula):
        raise ValueError('specification should be a Formula object')

    Xt = model.multiStepReach(timeStep, X0=X0, U=U, k=numSteps)
    S = spec.render(Xt)

    if spec.formula_type == 'ConjunctiveAlways':
        probSAT = S.estimateProbability()
    else:
        raise RuntimeError('currently support only conjunctive always formula_type')

    return probSAT, Xt