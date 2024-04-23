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

class Verifier_Star(object):
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


def checkSafetyStar(unsafe_mat, unsafe_vec, S):
    """Intersect with unsafe region, can work in parallel"""

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

def quanliVerifyExactBFS(net, inputSet, unsafe_mat, unsafe_vec, lp_solver='gurobi', numCores=1, show=True):
    """Quantitative Verification of ReLU Networks using exact bread-first-search"""

    if numCores > 1:
        pool = multiprocessing.Pool(numCores)
    else:
        pool = None
    S = reachExactBFS(net, inputSet, lp_solver, pool, show)  # output set
    P = []  # unsafe output set
    if pool is None:
        for S1 in S:
            P1 = checkSafetyStar(unsafe_mat, unsafe_vec, S1)
            if isinstance(P1, Star):
                P.append(P1)
    else:
        S1 = pool.map(checkSafetyStar, zip([unsafe_mat]*len(S), [unsafe_vec]*len(S), S))
        pool.close()
        for S2 in S1:
            if isinstance(S2[0], Star):
                P.append(S2[0])
          
    return S, P


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
            for i in (0, nBatchs):
                if i==0:
                    start_ID = 0
                else:
                    start_ID = start_ID + batchSize

                if i!= nBatchs-1:
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

            nSAT = 0
            n = 0
            for S1 in S:
                nSAT = nSAT + S1[0]
                n = n + S1[1]

        probSAT = probSAT + float(nSAT/n)

        probSAT = probSAT/nTimes

    return probSAT