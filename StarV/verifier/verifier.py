"""
Main Verifier Class
Dung Tran, 9/10/2022
"""

from StarV.net.network import NeuralNetwork, reachExactBFS
from StarV.set.probstar import ProbStar
from StarV.spec.dProbStarTL import Formula
from StarV.plant.dlode import DLODE
from StarV.plant.lode import LODE
import copy
import multiprocessing
import numpy as np
import polytope as pc
from StarV.util.print_util import print_util

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
        for S2 in S1:
            if isinstance(S2[0], ProbStar):
                P.append(S2[0])
                prob.append(S2[1])
          
    return S, P, sum(prob)


def quantiVerifyBFS(net, inputSet, unsafe_mat, unsafe_vec,  p_filter=0.0, lp_solver='gurobi', numCores=1, show=True):
    """ Overapproximate quantitative verification of ReLU network"""


    inputProb = inputSet[0].estimateProbability()
    
    if numCores > 1:
        pool = multiprocessing.Pool(numCores)
    else:
        pool = None

    assert p_filter >= 0.0, 'error: invalid filtering probability'

    if p_filter == 0.0:
        S, P, p_v = quantiVerifyExactBFS(net, inputSet, unsafe_mat, unsafe_vec, lp_solver, numCores, show)
        p_v_ub = p_v
        p_v_lb = p_v
    else:
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


def quantiVerifyProbStarTL(model, spec, timeStep, numSteps, X0=None, U=None):
    'quantitative verification of probstar temporal logic specification'

    
    assert isinstance(model, DLODE) or isinstance(model, LODE), 'error: model should be a linear ode or discrete linear ode object'
    assert timeStep > 0, 'error: invalid timeStep'
    assert numSteps >=1, 'error: invalid number of time steps'
    assert isinstance(spec, Formula), 'error: specification should be a Formula object'


    Xt = model.multiStepReach(timeStep, X0=X0, U=U, k=numSteps)
    S  = spec.render(Xt)

    probSAT = None
    if spec.formula_type == 'ConjunctiveAlways':
        probSAT = S.estimateProbability()
    else:
        raise RuntimeError('currently support only conjunctive always formula_type')
    

    return probSAT, Xt
