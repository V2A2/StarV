"""
Verifier for star reachability
Yuntao Li, 3/22/2024
"""

from StarV.net.network import NeuralNetwork, reachExactBFS
from StarV.set.star import Star
import copy
import multiprocessing
import numpy as np
import polytope as pc
from StarV.util.print_util import print_util
from typing import Optional, Tuple, Union


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

def quantiVerifyExactBFS(net, inputSet, unsafe_mat, unsafe_vec, lp_solver='gurobi', numCores=1, show=True):
    """Quantitative Verification of ReLU Networks using exact bread-first-search"""
    
    pool = multiprocessing.Pool(numCores) if numCores > 1 else None
    S = reachExactBFS(net, inputSet, lp_solver, pool, show)  # output set
    P = []  # unsafe output set
    prob = []  # probability of unsafe output set
    
    if pool is None:
        for S1 in S:
            P1 = checkSafetyStar(unsafe_mat, unsafe_vec, S1)
            if isinstance(P1, Star):
                P.append(P1)
    else:
        # S1 = pool.starmap(checkSafetyStar, [(unsafe_mat, unsafe_vec, s) for s in S])
        S1 = pool.map(checkSafetyStar, zip([unsafe_mat]*len(S), [unsafe_vec]*len(S), S))
        pool.close()
        for S2 in S1:
            if isinstance(S2[0], Star):
                P.append(S2[0])

    print('length of unsafe sets: ', len(P))
          
    return S, P

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

def quantiVerifyMC(net: NeuralNetwork, inputSet: Star, unsafe_mat: np.ndarray, unsafe_vec: np.ndarray, 
                   numSamples: int = 100000, nTimes: int = 10, numCores: int = 1) -> float:
    """Quantitative verification using traditional Monte Carlo sampling-based method"""
    if not isinstance(inputSet, Star):
        raise ValueError('input set should be a Star object')
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