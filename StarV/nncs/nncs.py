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
  Generic Neural Network Control System Class
  
  Dung Tran, 8/14/2023
  Sung Woo Choi, 06/10/2025
  	- added DynNN_NNCS (dynamic neural network NNCS)
	- plant: dynamic NN
	- controller: NN controller
"""

from StarV.net.network import NeuralNetwork, reachExactBFS, reachApproxBFS
from StarV.plant.dlode import DLODE
from StarV.plant.lode import LODE
from StarV.set.probstar import ProbStar
from StarV.spec.dProbStarTL import Formula, DynamicFormula, AtomicPredicate
import multiprocessing
from multiprocessing import Process, Queue
import copy
import time
import numpy as np

class ReachPRM_NNCS(object):
    'reachability parameters for NNCS'

    def __init__(self):
        self.initSet = None
        self.numSteps = 1
        self.refInputs = []
        self.method = 'exact-probstar'
        self.filterProb = 0.0
        self.lpSolver = 'gurobi'
        self.show = True
        self.numCores = 1
        self.specs = None   # abstract specification used to guide reachability analysis, a Formula object

class ReachPRM_DYNNN_NNCS(object):
    'reachability parameters for NNCS'

    def __init__(self):
        self.initSet = None
        self.numSteps = 1
        self.refInputs = []
        self.method = 'exact-probstar'
        self.filterProb = 0.0
        self.lpSolver = 'gurobi'
        self.show = True
        self.numCores = 1
        self.abSpecs = None # abstract DNF specification used to guide reachability analysis, see spec/dProbStarTL.py

class VerifyPRM_NNCS(object):
    'reachability parameters for NNCS'

    def __init__(self):
        self.initSet = None
        self.numSteps = 1
        self.refInputs = []
        self.verifyMethod = 'Q2' # verification method, qualitative (Ql), quantitative (Qt) or Q2 (both Ql and Qt)
        self.pf = 0.0  # filtering probability if p_f> 0 -> approx-method is used
        self.lpSolver = 'gurobi'
        self.show = True
        self.numCores = 1
        self.unsafeSpec = None # unsafe constraints # for safety properties
        self.temporalSpecs = None # temporal specifications, a list of ProbStarTL formulas
        self.computeProbMethod = 'approx' # compute probability of satisfaction (exact) or approx
        #  self.computeProbMethod = 'exact' return exact probability of satisfaction of a CDNF
        #  self.computeProbMethod = 'approx' return the estimation of lower-bound of probability of satisfaction of a CDNF
        self.timeOut = np.inf # timeout for verification process
        self.MaxLengthCDNF = 11    # maximum allowable length for computing probability of satisfaction of a CDNF

class VerifyRes_NNCS(object):

    def __init__(self):

        self.RX = None # reachable set
        self.RY = None # output reachable set
        self.RU = None # control reachable set

        self.CeIn = None # counter initial set
        self.CeOut = None # counter output set
        self.Ql = None # qualitative result
        self.Qt = None # quantitative result
        self.Qt_lb = None # lower bound quantitative result
        self.Qt_ub = None # upper bound quantitative result
        self.Qt_min = None # minimum lower bound quantitative result (for unbounded input set)
        self.Qt_max = None # maximum upper bound quantitative result (for unbounded input set)
        self.p_ignored = None # total ignored probability
        
class NCS:
    def __init__(self):
        self.controller = None
        self.plant = None

class NNCS(object):
    """Generic neural network control system class

       % nerual network control system architecture
       %
       %              --->| plant ---> x(k+1)--------------->y(k+1) 
       %             |                                       |
       %             |                                       |
       %             u(k) <---- controller |<------ y(k)-----|--- (output feedback)
       %                                   |<--------------------- ref_inputs
       %                                                           
        
        
        % the input to neural net controller is grouped into 2 group
        % the first group contains all the reference inputs
           
        % the first layer weight matrix of the controller is decomposed into two
        % submatrices: W = [W1 W2] where
        %              W1 is conresponding to I1 = v[k] (the reference input)
        %              W2 is conresponding to I2 = y[k] (the feedback inputs)  
        
        % the reach set of the first layer of the controller is: 
        %              R = f(W1 * I1 + W2 * I2 + b), b is the bias vector of
        %              the first layer, f is the activation function

        nO = 0; % number of output
        nI = 0; % number of inputs = size(I1, 1) + size(I2, 1) to the controller
        nI_ref = 0; % number of reference inputs to the controller
        nI_fb = 0; % number of feedback inputs to the controller
        
        % for reachability analysis
        method = 'exact-star'; % by default
        plantReachSet = {};
        controllerReachSet = {};
        numCores = 1; % default setting, using single core for computation
        ref_I = []; % reference input set
        init_set = []; % initial set for the plant
        reachTime = 0;
        
        % for simulation
        simTraces = {}; % simulation trace
        controlTraces = {}; % control trace
        
        % use for falsification
        falsifyTraces = {};
        falsifyTime = 0;
        

       The controller network can be:

        * feedforward with ReLU
        * new activation functions will be added in future
       
       The dynamical system can be:
        * Linear ODE
        * Nonlinear ODE will be added in future
        * Dynamic Neural network

       Properties:
           @type: 1) linear-nncs: relu net + Linear ODEs
                  2) nonlinear-nncs: relu net + Nonlinear ODEs / sigmoid net + ODEs 
                  3) dynnn-nncs: relu net (controller) + relu net (plant) 
           @in_dim: input dimension
           @out_dim: output dimension

       Methods: 
           @reach: compute reachable set
    """

    def __init__(self, controller_net, plant, type=None):

        assert isinstance(controller_net, NeuralNetwork), 'error: net should be a Neural Network object'
        assert isinstance(plant, DLODE) or isinstance(plant, LODE) or \
            isinstance(plant, NeuralNetwork), 'error: plant should be a discrete ODE or neural network object'

        # TODO implement isReLUNetwork?

        # checking consistency
        # nI = plant.in_dim if isinstance(plant, NeuralNetwork) else plant.nI
        # assert nI == controller_net.out_dim, 'error: number of plant inputs \
        # does not equal to the number of controller outputs'

        self.ncs = NCS()
        self.ncs.controller = controller_net
        self.ncs.plant = plant

        if isinstance(plant, NeuralNetwork):
            self.nO = plant.out_dim
            self.nI_fb = plant.out_dim    # number of feedback inputs to the controller
        else:
            self.nO = plant.nO
            self.nI_fb = plant.nO    # number of feedback inputs to the controller

        self.nI = controller_net.in_dim
        self.nI_ref = controller_net.in_dim - self.nI_fb   # number of reference inputs to the controller
        self.type = type
        self.RX = None     # state reachable set
        self.RY = None     # output reachable set
        self.RU = None     # control set


    def __str__(self):
        """print information of the neural network control system"""

        print('\n=================NEURAL NETWORK CONTROL SYSTEM=================')
        print('\n nncs-type: {}'.format(self.type))
        print('\n number of outputs: {}'.format(self.nO))
        print('\n number of inputs: {}'.format(self.nI))
        print('\n number of feeback inputs: {}'.format(self.nI_fb))
        print('\n number of reference inputs: {}'.format(self.nI_ref))
        print('\n network controller & plant model information:')
        print(self.ncs.controller)
        print(self.ncs.plant)
        print('')
        return '\n'
        
    def info(self):
        print(self)

    def reach(self, reachPRM):
        'reachability analysis'

        # reachPRM: reachability parameters
        #   1) reachPRM.initSet
        #   2) reachPRM.refInputs
        #   3) reachPRM.numSteps
        #   4) reachPRM.method: exact-star or approx-star
        #   5) reachPRM.numCores
        #   6) reachPRM.lpSolver
        #   7) reachPRM.show

        if self.type == 'DLNNCS': # discrete linear NNCS
            self.RX, self.RY, self.RU = reachBFS_DLNNCS(self.ncs, reachPRM)
        elif self.type == 'DynNN-NNCS': # dynamic neural network NNCS
            RX, p_ignored = reachBFS_DynNN_NNCS(self.ncs, reachPRM)
            return RX, p_ignored
        else:
            raise RuntimeError('We are have not support \
            reachability analysis for type = {} yet'.format(self.type))

    def visualizeReachSet(self, rs='state', dis='box', dim=[0, 1]):
        'visualize reachable set'

        if self.RX is None:
            raise RuntimeError('No reachable set, please call reach function first')

        if rs == 'state':
            pass
        elif rs == 'output':
            pass
        elif rs == 'control':
            pass
        elif rs == 'all':
            pass
        else:
            raise RuntimeError('Unknown display option')

class AEBS_NNCS(object):

    # a special nncs with two neural networks inside

    def __init__(self, controller, transformer, norm_mat, scale_mat, plant):
        self.controller = controller
        self.transformer = transformer
        self.norm_mat = norm_mat
        self.scale_mat = scale_mat
        self.plant = plant

class DynNN_NNCS(object):

    # plant: a dynamic neural network
    # controller: a neural network controller

    def __init__(self, controller, plant):
        self.controller = controller
        self.plant = plant

def verify_DLNNCS_MonteCarlo(ncs, verifyPRM, N_samples):
    'Use MonteCarlo sampling method for verification'

    stateSamples = verifyPRM.initSet.sampling(N_samples)

def check_SAT_sim_trace(unsafe_mat, unsafe_vec, trace):
    'check satification of a single  simulation trace'
        

    n = len(trace)
    Ql = []
    for i in range(0, n):
        x = trace[i]
        y = np.matmul(unsafe_mat, x) - unsafe_vec
        ymax = y.max(axis=0)
        if ymax > 0:
            Ql.append(0)
        else:
            Ql.append(1)
        
    return Ql

def simulate_DLNNCS(ncs, state_vec, ref_input_vec, numSteps):
    'simulate a DLNCS with a set of input samples'


    x = state_vec
    X = []
    X.append(x)
    for i in range(0, numSteps + 1):
        x = stepSim_DLNNCS(ncs, x, ref_input_vec)
        X.append(x)

    return X


def stepSim_DLNNCS(ncs, state_vec, ref_input_vec):
    'step simulation of DLNNCS'

    net = ncs.controller
    plant = ncs.plant

    fb_y = np.matmul((plant.C, state_vec))
    fb_input = np.vstack(fb_y, ref_input_vec)
    u = net.evaluate(fb_input)
    x = np.matmul(plant.A, state_vec) + np.matmul(plant.B, u)

    return x



def verifyBFS_DLNNCS(ncs, verifyPRM):
    'Q^2 verification of DLNNCS'

    assert isinstance(verifyPRM, VerifyPRM_NNCS), 'error: second input should be a VerifyPRM_NNCS object'

    reachPRM = ReachPRM_NNCS()
    reachPRM.initSet = copy.deepcopy(verifyPRM.initSet)
    reachPRM.numSteps = copy.deepcopy(verifyPRM.numSteps)
    reachPRM.refInputs = copy.deepcopy(verifyPRM.refInputs)
    reachPRM.filterProb = copy.deepcopy(verifyPRM.pf)
    reachPRM.lpSolver = copy.deepcopy(verifyPRM.lpSolver)
    reachPRM.show = copy.deepcopy(verifyPRM.show)
    reachPRM.numCores = copy.deepcopy(verifyPRM.numCores)
    
   
    # perform reachability analysis to compute the reachable sets

    if isinstance(ncs, NNCS):  
        RX, p_ignored = reachBFS_DLNNCS(ncs, reachPRM)  # this is a traditional neural network control system
    elif isinstance(ncs, AEBS_NNCS):
        RX, p_ignored = reachBFS_AEBS(ncs, reachPRM)    # this is AEBS NNCS
    else:
        raise RuntimeError('Unknown system')
        
    # check the intersection of the reachable sets with unsafe region
    CeIn = []  # counterexample input set
    CeOut = [] # counterexample state set
    Ql = []  # qualitative results 0->UNSAT, 1->SAT, 2->Unknown
    Qt = []  # quantitative results, i.e., probability of violation (or SAT)
    Qt_ub = [] # upper bound of probability of violation, if pf > 0 is used
    Qt_max = [] # maximum lower bound of probability of violation for unbounded input set (handling the tail)

    pI = reachPRM.initSet.estimateProbability()

    i = 0

    n = len(RX)
    for k in range(0, n):
        Rk = RX[k]
        Cek = []
        Cok = []
        pk = 0.0
        Qlk = 0
        p_ign_k = p_ignored[k]

        for i in range(0, len(Rk)):
            Z1 = copy.deepcopy(Rk[i])
            Z1.addMultipleConstraintsWithoutUpdateBounds(verifyPRM.unsafeSpec[0], verifyPRM.unsafeSpec[1])
            Ceki = ProbStar(verifyPRM.initSet.V, Z1.C, Z1.d, Z1.mu, Z1.Sig, Z1.pred_lb, Z1.pred_ub)          
            if not Ceki.isEmptySet():
                Qlk = 1
                Cek.append(Ceki)
                Cok.append(Z1)
                pki = Ceki.estimateProbability()
                pk = pk + pki

        CeIn.append(Cek)
        CeOut.append(Cok)
        Ql.append(Qlk)
        Qt.append(pk)
        Qt_ub.append(min(pk + p_ign_k,1.0))
        Qt_max.append(min(pk + p_ign_k + 1 - pI, 1.0))


    res = VerifyRes_NNCS()
    res.RX = RX
    res.CeIn = CeIn
    res.CeOut = CeOut
    res.Ql = Ql
    res.Qt = Qt
    res.Qt_ub = Qt_ub
    res.Qt_max = Qt_max
    res.p_ignored = p_ignored

    return res

    

def reachBFS_DLNNCS(ncs, reachPRM):
    'breath first search reachability of DLNNCS'

    assert isinstance(reachPRM, ReachPRM_NNCS), 'error: reachability parameter should be an ReachPRM_NNCS object'
    assert reachPRM.initSet is not None, 'error: there is no initial set for reachability'
    assert reachPRM.numSteps >= 1, 'error: number of time steps should be >= 1'
    
    if reachPRM.numCores > 1:
        pool = multiprocessing.Pool(reachPRM.numCores)
    else:
        pool = None

    if reachPRM.show:
        print('\nReachability analysis of discrete linear NNCS for {} steps...'.format(reachPRM.numSteps))
                
    RX = []   # state reachable set RX = [RX1, ..., RYN], RXi = [RXi1, RXi2, ...RXiM]

    RX.append([reachPRM.initSet])
    p_ignored = [0.]
    for i in range(0, reachPRM.numSteps+1):
        if reachPRM.show:
            print('\nReachability analysis at step {}...'.format(i))

        X0 = RX[i]
        RXi = []
        p_ign_i = p_ignored[i]
        for j in range(0, len(X0)):            
            RXij, pij = stepReach_DLNNCS(ncs, X0[j], reachPRM)
            RXi.extend(RXij)
            p_ign_i = p_ign_i + pij

        RX.append(RXi)
        p_ignored.append(p_ign_i)

    return RX, p_ignored


def stepReach_DLNNCS(ncs, Xi, reachPRM):
    'one-step reachability analysis of Discrete linear NNCS, a normal NNCS with one controller and one plant'


    refInputs = reachPRM.refInputs
    filterProb = reachPRM.filterProb
    numCores = reachPRM.numCores
    lp_solver = reachPRM.lpSolver
    net = ncs.controller
    plant = ncs.plant

    # Xi: set of state of the plant at step i
    # Yi: feedback to the network controller at step i

    if filterProb < 0:
        raise RuntimeError('Invalid filtering probability')

    if numCores > 1:
        pool = multiprocessing.Pool(numCores)
    else:
        pool = None

    RX = []
    Yi = Xi.affineMap(plant.C)
    fb_I = Yi.concatenate_with_vector(refInputs)
    p_ig = 0.0

    RU = reachExactBFS(net, [fb_I], lp_solver, pool=pool, show=False)
    for U in RU:
        RX1, _ = plant.stepReach(X0=Xi, U=U, subSetPredicate=True)
        if filterProb == 0:
            RX.append(RX1)
        else:
            
            pi = RX1.estimateProbability()
            if pi > filterProb:
                RX.append(RX1)
            else:
                p_ig = p_ig + pi     

    return RX, p_ig


def stepReach_DLNNCS_extended(ncs, Xi, reachPRM):
    'one step reachability of discrete linear NNCS, extended to include the AEBS system'
    
    
    if isinstance(ncs, NNCS):  

        RX, p_ignored = stepReach_DLNNCS(ncs, Xi, reachPRM)   # traditional NNCS system, i.e., one controller, one plant                                     
    elif isinstance(ncs, AEBS_NNCS):   # AEBS system, two networks, one plant

        if reachPRM.numCores > 1:
            pool = multiprocessing.Pool(reachPRM.numCores)
        else:
            pool = None       

        Xi1 = stepReach_AEBS(ncs, Xi, pool)    # this is AEBS NNCS

        if reachPRM.filterProb == 0:
            RX = Xi1
            p_ignored = 0
        else:
            
            RX = []
            p_ignored = 0
            n = len(Xi1)
            for i in range(0, n):
                Xpf = Xi1[i].estimateProbability() 
                if Xpf > reachPRM.filterProb:
                    RX.append(Xi1[i])
                else:
                    p_ignored = p_ignored + Xpf

    elif isinstance(ncs, DynNN_NNCS):
        RX, p_ignored = stepReach_DynNN_NNCS(ncs, Xi, reachPRM)   # NNCS system with plant: Dynamic NN, and control: Control NN.

    else:
        raise RuntimeError('Unknown system')


    return RX, p_ignored
    

def reachBFS_AEBS(AEBS, reachPRM):
    'compute the reachable set of AEBS for multiple steps T, BFS reachability'
    
    X0 = copy.deepcopy(reachPRM.initSet)
    T = reachPRM.numSteps
    pf = reachPRM.filterProb
    if reachPRM.numCores > 1:
        pool = multiprocessing.Pool(reachPRM.numCores)
    else:
        pool = None
    
    if T < 1:
        raise RuntimeError('Invalid number of steps')

    X = []
    X.append([X0])
    p_ignored = [0.]
    for i in range(0, T+1):
        X1, p_ig1 = stepReach_AEBS_multipleInitSets(AEBS, X[i], pf, pool)         
        X.append(X1)
        p_ign1 = p_ig1 + p_ignored[i]
        p_ignored.append(p_ign1)

    return X, p_ignored

def stepReach_AEBS_multipleInitSets(AEBS, X0, pf, pool):
    'step reach of AEBS with multiple initial sets'

    n = len(X0)
    X1 = []
    p_ignored = 0
    for i in range(0, n):
        if pf == 0:
            X1i = stepReach_AEBS(AEBS, X0[i], pool)
            X1.extend(X1i)
        else:
            pX = X0[i].estimateProbability()
            if X0[i].estimateProbability() > pf:
                X1i = stepReach_AEBS(AEBS, X0[i], pool)
                X1.extend(X1i)

            else:
                p_ignored = p_ignored + pX

    return X1, p_ignored


def stepReach_AEBS(AEBS, X0, pool):
    'step reachability of AEBS system'
    

    # stepReach computation for AEBS

    # step 0: initial state of plant
    # step 1: normalizing state
    # step 2: compute brake output of RL controller
    # step 3: compose brake output and normalized speed
    # step 4: compute transformer output
    # step 5: get control output
    # step 6: scale control output
    # step 7: compute reachable set of the plannt with the new control input and initial state

    # step 8: go back to step 1, .... (another stepReach)

    # step 1: normalizing state

    controller = AEBS.controller
    transformer = AEBS.transformer
    norm_mat = AEBS.norm_mat
    scale_mat = AEBS.scale_mat
    plant = AEBS.plant
    

    
    norm_X = X0.affineMap(norm_mat)
    
    print('Computing reachable set of RL controller ...\n')

    # step 2: compute brake output of the RL controller
    brake = reachExactBFS(controller, [norm_X], pool=pool)
    
    m = len(brake)

    print('Geting exact input sets to transformer ...\n')
    # step 3: compose brake output and normalized speed to get exact inputs to transformer
    speed_brake = [] # exact inputs to transformer
    for i in range(0, m):
        V = np.vstack((norm_X.V[1, :], brake[i].V))
        speed_brake_i = ProbStar(V, brake[i].C, brake[i].d, brake[i].mu, brake[i].Sig, brake[i].pred_lb, brake[i].pred_ub)
        speed_brake.append(speed_brake_i)


    print('Computing exact transformer output ... \n')
    # step 4: get exact transformer output
    tf_outs = []
    for i in range(0, m):
        tf_out = reachExactBFS(transformer, [speed_brake[i]], pool=pool)
        tf_outs.extend(tf_out)

    print('Getting control input set to the plant and scale it ...\n')
    # step 5: get control input to the plant and scale it using scale matrix
    n = len(tf_outs)
    controls = []
    for i in range(0, n):
        V = np.vstack((norm_X.V[1, :], tf_outs[i].V))
        control = ProbStar(V, tf_outs[i].C, tf_outs[i].d, tf_outs[i].mu, tf_outs[i].Sig, tf_outs[i].pred_lb, tf_outs[i].pred_ub)
        controls.append(control.affineMap(scale_mat)) # scale the control inputs

    print('Compute the next step reachable set for the plant ...\n')
    # compute the next step reachable set for the plant
    X1 = []
    for i in range(0, n):
        X1i, _ = plant.stepReach(X0, controls[i], subSetPredicate=True)
        X1.append(X1i)

    return X1



def reachDFS_DLNNCS(ncs, reachPRM):
    'Depth First Search Reachability Analysis for Discrete Linear NNCS, extended to include AEBS system'

    assert isinstance(reachPRM, ReachPRM_NNCS), 'error: reachability parameter should be an ReachPRM_NNCS object'
    assert reachPRM.initSet is not None, 'error: there is no initial set for reachability'
    assert reachPRM.numSteps >= 1, 'error: number of time steps should be >= 1'

    k = reachPRM.numSteps  # number of reachability steps
    remains = [] # contains all remaining reachable set at all k steps, remains = [RM1, RM2, ..., RMk], RMi = [X1, X2, ..., Xm]

    # The algorithm works as follows:
    # step: t = 0, X0, U = 0
    # * strore X0 to a trace T <-X0
    # 
    # step:t = 1, X0, U = f(Y0) = f(CX0) = [U1, ..., Um] -> [X1, ....,Xm]
    #    * pop X0 from trace T: X0 <- T[0] = T[t-1]
    #    * get output feedback Y0 = CX0
    #    * compute control output U = F(Y0) = [U1, ..., Um]
    #    * get new state reachable set X1 = [X11, ..., X1m]
    #    * store the first X11 to trace T: T <- X11
    #    * store the remaining sets in a remain RM1 = [X12, ...., X1m]
    #    * store the remain RM1 to remains RMs: RMs <- RM1
    # step: t = 2:
    #    * pop the newest reach set in trace T, X11<- T[1] = T[t-1]
    #    * get output feedback Y1 = CX11
    #    * compute control output U = F(Y1) = [U1, ...., Un]
    #    * get new state reachable set X2 = [X21, X22, ..., X2n]
    #    * store the first X21 into trace T: T<-X21
    #    * store the remaining sets in a remain RM2 = [X22, ...., X2n]
    #    * store the remain RM2 to remain RMs:  RMs <- RM2 
    #    .
    #    .
    #    .
    # step: t = k:
    #    * pop the newest reach set in trace T, X[k-1,1] <- T[k-1]
    #    * (substep k1) get output feedback Y[k-1] = CX[k-1,1]
    #    * (substep k2) compute control output U = F(Y[k-1]) = [U1, ...., Up]
    #    * (substep k3) get new state reachable set Xk = [Xk1, Xk2, ..., Xkp]
    #    * (substep k4) get p copies of trace T: T1, T2, ... Tp
    #    * (substep k5) store Xki into trace Ti, Ti<-Xki  
    #    * (substep k6) store all traces into traces Ts: Ts <- [T1, T2, ..., Tp]
    #    * we finish construct a partial reachable set trace here, i.e., T1, T2, ..., Tp
    #    *** We can verify traces T1, ...., Tp and get one partial verification result here, and then ignore them
    #    *** LOOP here:
    #    *** get RM[k-1] from RMs, i.e., RM[k-1] <- RMs[k-2]
    #    *** if RM[k-1] is not empty,  
    #    *** (substep k9) pop the next newest reach set, e.g, X[k-1,2] <- RM[k-1]
    #    *** (substep k10) replace the newest reach set in trace T, X[k-1,1] by X[k-1,2]
    #    *** (substep k11) Recall substeps k1, k2, k3, k4, k5, k6
    #    *** repeat substeps k9, k10, k11 until RM[k-1] empty
    #    if length RM[k-1]== 0, i.e., RM[k-1] is empty,
    #    *** pop RM[k-2] <- RMs[k-3]
    #    *** delete the 2 newest reach set2 in trace T, i.e., X[k-1, 1], X[k-2,1]
    #    *** pop the next newest reach set X[k-2, 2]
    #    *** repeat ...
    
    
    X0 = copy.deepcopy(reachPRM.initSet)
    remains.append([X0])
    T = []
    p_ig = 0.0
    traces = []
    
    while True:
        i = len(remains)  # equal to the current considering time step
        if i != 0:
            Ri = remains[i-1]
        else:
            break
        
        if len(Ri) != 0:
            Xi = Ri.pop(0)
            T.append(Xi)
            Xi_plus_1, pig_i = stepReach_DLNNCS_extended(ncs, Xi, reachPRM)
            p_ig = p_ig + pig_i
            
            if i == k:
                
                for Xj in Xi_plus_1:
                    Tj = copy.deepcopy(T)
                    Tj.append(Xj)
                    traces.append(Tj)
                T.pop(len(T)-1)
            else:
                remains.append(Xi_plus_1)
                    
        else:
            
            remains.pop(i-1)
            if len(T) != 0:
                T.pop(len(T)-1)

    # a trace share the same predicate constraint P(alpha) with the final reach set in the trace
    # 

    traces1 = []
    for trace in traces:
        n = len(trace)
        trace1 = []
        for i in range(0, n):
            R = ProbStar(trace[i].V, trace[n-1].C, trace[n-1].d, trace[n-1].mu, \
                         trace[n-1].Sig, trace[n-1].pred_lb, trace[n-1].pred_ub)
            trace1.append(R)
        traces1.append(trace1)         
    
    return traces1, p_ig


def specGuidedReachDFS_DLNNCS(ncs, reachPRM):
    'specification guided reachability using abstract DNF specification to reduce the number of traces'
    # Dung Tran: 11/10/2025 updated date:

    pass

def singleSpecGuidedReachDFS_DLNNCS(ncs, initSet, spec):
    'a single abstract specification - guided reachability'

    # Dung Tran: 11/11/2025, update date:
    # spec = P = [C0, C1, ..., Cn], Ci is an AtomicPredicate
    # 
    pass

def specGuidedStepReach_DLNNCS(ncs, Xi, reachPRM, spec_i_mat, spec_i_vec):
    'a single step spec-guided reachability'

    # Dung Tran: 11/11/2025, update date:

    Xi_AND_spec_i = Xi.intersectHalfSpace(spec_i_mat, spec_i_vec)
    RX, p_ig = stepReach_DLNNCS_extended(ncs, Xi_AND_spec_i, reachPRM)

    return RX, p_ig
    
def getAbstractSpecConstraints(spec):
    'get constraint matrix and vector from an abstract specification'

    # Dung Tran: 11/11/2025, update date:
    # spec = [C0, C1, ..., Cn], Ci is an AtomicPredicate with time t
    # we collect all constraints for all time steps in the spec
    # for example: C0.t = 0, C1.t = 0, C2.t = 1, C3.t = 1, C4.t = 2, C5.t = 2
    # we will merge C0 and C1 together for step t = 0,
    # and then (C2, C3) for step t=1, and (C4, C5) for step t=2

    T = [] # list of time steps
    for i in range(0, len(spec)):
        Ci = spec[i]
        assert isinstance(Ci, AtomicPredicate), 'error: spec[{}] is not an AtomicPredicate object'.format(i)
        T.append(Ci.t)

    
    pass

def reachBFS_DynNN_NNCS(ncs, reachPRM):
    'breath first search reachability of dynamic neural network NNCS (DynNN_NNCS)'

    assert isinstance(reachPRM, ReachPRM_DYNNN_NNCS), 'error: reachability parameter should be an ReachPRM_NNCS object'
    assert reachPRM.initSet is not None, 'error: there is no initial set for reachability'
    assert reachPRM.numSteps >= 1, 'error: number of time steps should be >= 1'
    
    if reachPRM.numCores > 1:
        pool = multiprocessing.Pool(reachPRM.numCores)
    else:
        pool = None

    if reachPRM.show:
        print('\nReachability analysis of dynamic NN NNCS for {} steps...'.format(reachPRM.numSteps))
                
    RX = []   # state reachable set RX = [RX1, ..., RYN], RXi = [RXi1, RXi2, ...RXiM]

    RX.append([reachPRM.initSet])
    p_ignored = [0.]
    # for i in range(0, reachPRM.numSteps+1):
    for i in range(reachPRM.numSteps):
        if reachPRM.show:
            print('\nReachability analysis at step {}...'.format(i))

        X0 = RX[i]
        RXi = []
        p_ign_i = p_ignored[i]
        for j in range(0, len(X0)):            
            RXij, pij = stepReach_DynNN_NNCS(ncs, X0[j], reachPRM=reachPRM)
            RXi.extend(RXij)
            p_ign_i = p_ign_i + pij

        RX.append(RXi)
        p_ignored.append(p_ign_i)

    return RX, p_ignored


def stepReach_DynNN_NNCS(ncs, Xi, reachPRM):
    'one-step reachability analysis of dynamic NN NNCS, a normal NNCS with one controller and one plant'

    refInputs = reachPRM.refInputs
    filterProb = reachPRM.filterProb
    numCores = reachPRM.numCores
    lp_solver = reachPRM.lpSolver
    controller = ncs.controller
    plant = ncs.plant
    Ui = None
    # Xi: set of state of the plant at step i
    # Yi: feedback to the network controller at step i

    if filterProb < 0:
        raise RuntimeError('Invalid filtering probability')

    if numCores > 1:
        pool = multiprocessing.Pool(numCores)
    else:
        pool = None
    
    # if control input, Ui, is not provided, compute control input with NN controller, i.e, Ui = NN_control(Xi)
    if Ui is None or len(Ui) == 0:
        # achieve control input from reachability of NN controller
        fb_I = Xi.concatenate_with_vector(refInputs) if len(refInputs) > 0 else Xi
        Ui = reachExactBFS(controller, [fb_I], lp_solver, pool=pool, show=False)
    else:
        if not isinstance(Ui, list):
            Ui = [Ui]
    p_ig = 0.0

    print('Xi: \n')
    print(repr(Xi))
    print('prob: ', Xi.estimateProbability())
    print()

    print('fb_I: \n')
    print(repr(fb_I))
    print('prob: ', fb_I.estimateProbability())
    print()
    
    # concatenate state and control input to construct input for NN plant, i.e., XUi = [Xi, Ui]^T
    XUi = []
    for U in Ui:
        print('U: \n')
        print(repr(U))
        print('prob: ', U.estimateProbability())
        print()
        if filterProb > 0:
            # filter out control input if the probability of control input is less tahn filterProb
            pi = U.estimateProbability()
            if pi <= filterProb:
                p_ig += pi
                continue
 
        XUi.append(Xi.concatenate(U))
        
    
    # reachability of NN plant; yi = NeuralNetwork(xi, ui) = NeuralNetwork([xi, ui])
    Y = []
    for I in XUi:
        Y1 = reachExactBFS(plant, [I], lp_solver, pool=pool, show=False)
        for Yi in Y1:
            print('Yi: \n')
            print(repr(Yi))
            print('prob: ', Yi.estimateProbability())
            print()
            if filterProb > 0:
                pi = Yi.estimateProbability()
                if pi <= filterProb:
                    p_ig += pi
                    continue
        
        Y.append(Yi)

    return Y, p_ig


def check_sat_on_trace(*args):
    'verify a temporal property on a single trace'
    # Dung Tran: 1/14/2024


    if isinstance(args[0], tuple):
        args1 = args[0]
    else:
        args1 = args
    spec = args1[0]      # user-define spec
    trace = args1[1]     # ProbStar signal

    assert isinstance(spec, Formula), 'error: spec is not a Formula object'
    print('Transform spec to abstract dijunctive normal form (DNF)...')
    DNF_spec = spec.getDynamicFormula()
    print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
    sat, p_max, p_min, _ = DNF_spec.evaluate(trace)
    
    return p_max, p_min

def check_sat_on_trace_for_full_analysis(*args):
    'verify a temporal property on a single trace'
    # Dung Tran: 1/14/2024


    if isinstance(args[0], tuple):
        args1 = args[0]
    else:
        args1 = args
    spec = args1[0]      # user-define spec
    trace = args1[1]     # ProbStar signal

    assert isinstance(spec, Formula), 'error: spec is not a Formula object'
    print('Transform spec to abstract dijunctive normal form (DNF)...')
    DNF_spec = spec.getDynamicFormula()
    print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
    p_max, p_min, p_ig, cdnf_sat, cdnf_ig, sat_trace = DNF_spec.evaluate_for_full_analysis(trace)
    
    return p_max, p_min, p_ig, cdnf_sat, cdnf_ig, sat_trace
    

def verify_temporal_specs_DLNNCS_with_timeOut(ncs, verifyPRM):
    'verify temporal behaviors of DLNNCS'
    # Dung Tran: 1/14/2024
    # Note: this function somehow have poor performance compared to the one without timeOut
    # Under testing

    Q = Queue()
       
    def verify_temporal_specs():
        'trick to kill verification process when timeOut'

        assert isinstance(ncs, NNCS), 'error: ncs is not an NNCS object'
        assert isinstance(verifyPRM, VerifyPRM_NNCS), 'error: verifyPRM is not a VerifyPRM_NNCS object'

        reachPRM = ReachPRM_NNCS()
        reachPRM.initSet = copy.deepcopy(verifyPRM.initSet)
        reachPRM.numSteps = copy.deepcopy(verifyPRM.numSteps)
        reachPRM.refInputs = copy.deepcopy(verifyPRM.refInputs)
        reachPRM.filterProb = copy.deepcopy(verifyPRM.pf)
        reachPRM.lpSolver = copy.deepcopy(verifyPRM.lpSolver)
        reachPRM.show = copy.deepcopy(verifyPRM.show)
        reachPRM.numCores = copy.deepcopy(verifyPRM.numCores)

        # get traces using DFS algorithm
        print('Get ProbStar traces using DFS-reachability algorithm...')
        start = time.time()
        traces, p_ig = reachDFS_DLNNCS(ncs.controller, ncs.plant, reachPRM)
        end = time.time()
        reachTime = end-start  # reachability time in seconds
        print('Total ProbStar traces: {}'.format(len(traces)))

        DNF_transform_time = []  # get abstract DNF
        # get temporal specifications
        specs = verifyPRM.temporalSpecs

        if verifyPRM.numCores > 1:
            pool = multiprocessing.Pool(verifyPRM.numCores)
        else:
            pool = None


        print('Verifying traces against temporal specification ...')
        # verify temporal specifications
        p_SAT = []
        checking_time = [] # get checking time
        verifyTime = []  # total verification time
        for k in range(0, len(specs)):
            spec = specs[k]
            start = time.time()
            p_sat = []
            if pool is None:
                for i in range(0, len(traces)):
                    print('Verifying trace {} against spec {}...'.format(i, k))
                    trace = traces[i]
                    p_sat1 = check_sat_on_trace(spec, trace, verifyPRM.computeProbMethod)
                    p_sat.append(p_sat1)

            else:
                rs = pool.map(check_sat_on_trace, zip([spec]*len(traces), traces))
                for rs1 in rs:
                    p_sat.append(rs[0])

            if verifyPRM.computeProbMethod == 'exact':
                p = sum(p_sat)
            else:
                p = max(p_sat)

            p_SAT.append(p)
            end = time.time()
            ct = end-start

            checking_time.append(ct)
            verifyTime.append(ct + reachTime)

        Q.put([traces, p_SAT, reachTime, checking_time, verifyTime])
       
        # end of internal function
    
    p1 = Process(target= verify_temporal_specs, name='verify_temoral_specs')
    p1.start()
    p1.join(timeout=verifyPRM.timeOut)
    p1.terminate()

    if p1.exitcode is None:
        print('Verification process reach timeouts = {}'.format(verifyPRM.timeOut))
        traces = []
        p_SAT = []
        reachTime = []
        checking_time = []
        verifyTime = []
    else:
        print('Verificaion is done successfully!')
        res = Q.get()
        traces = res[0]
        p_SAT = res[1]
        reachTime = res[2]
        checking_time = res[3]
        verifyTime = res[4]
    return traces, p_SAT, reachTime, checking_time, verifyTime
    

def verify_temporal_specs_DLNNCS(ncs, verifyPRM):
    'verify temporal behaviors of DLNNCS'
    # Dung Tran: 1/14/2024, update 7/4/2024


    assert isinstance(ncs, NNCS) or isinstance(ncs, AEBS_NNCS) or isinstance(ncs, DynNN_NNCS), 'error: ncs is not an NNCS object'
    assert isinstance(verifyPRM, VerifyPRM_NNCS), 'error: verifyPRM is not a VerifyPRM_NNCS object'

    reachPRM = ReachPRM_NNCS()
    reachPRM.initSet = copy.deepcopy(verifyPRM.initSet)
    reachPRM.numSteps = copy.deepcopy(verifyPRM.numSteps)
    reachPRM.refInputs = copy.deepcopy(verifyPRM.refInputs)
    reachPRM.filterProb = copy.deepcopy(verifyPRM.pf)
    reachPRM.lpSolver = copy.deepcopy(verifyPRM.lpSolver)
    reachPRM.show = copy.deepcopy(verifyPRM.show)
    reachPRM.numCores = copy.deepcopy(verifyPRM.numCores)
    

    # get traces using DFS algorithm
    print('Get ProbStar traces using DFS-reachability algorithm...')
    start = time.time()
    traces, p_ig0 = reachDFS_DLNNCS(ncs, reachPRM)
    end = time.time()
    reachTime = end-start  # reachability time in seconds
    print('Total ProbStar traces: {}'.format(len(traces)))

    DNF_transform_time = []  # get abstract DNF
    # get temporal specifications
    specs = verifyPRM.temporalSpecs
    

    if verifyPRM.numCores > 1:
        pool = multiprocessing.Pool(verifyPRM.numCores)
    else:
        pool = None


    print('Verifying traces against temporal specification ...')
    # verify temporal specifications
    p_SAT = []
    checking_time = [] # get checking time
    verifyTime = []  # total verification time
    p_SAT_MIN = []
    p_SAT_MAX = []
    p_input = verifyPRM.initSet.estimateProbability()

    for k in range(0, len(specs)):
        spec = specs[k]
        start = time.time()
        p_sat_min = []
        p_sat_max = []
        p_ig = 0.0
        if pool is None:
            for i in range(0, len(traces)):
                print('Verifying trace {} against spec {}...'.format(i, k))
                trace = traces[i]
                p_max, p_min = check_sat_on_trace(spec, trace)
                p_sat_max.append(p_max)
                p_sat_min.append(p_min)
        else:
             RS = pool.map(check_sat_on_trace, zip([spec]*len(traces), traces))
             for S in RS:
                 p_sat_max.append(S[0])
                 p_sat_min.append(S[1])
             
        
        p_SAT_MAX.append(min(sum(p_sat_max) + p_ig0, p_input))   # give the upperbound of probability of satisfaction
        p_SAT_MIN.append(min(sum(p_sat_min), p_input))       # give the lower bound of probability of satisfaction
        end = time.time()
        ct = end-start

        checking_time.append(ct)
        verifyTime.append(ct + reachTime)

   
    return traces, p_SAT_MAX, p_SAT_MIN, reachTime, checking_time, verifyTime
    

def verify_temporal_specs_DLNNCS_for_full_analysis(ncs, verifyPRM):
    'verify temporal behaviors of DLNNCS'
    # Dung Tran: 1/14/2024, updated 7/4/2024
    # more analysis can be done using this verification function

    assert isinstance(ncs, NNCS), 'error: ncs is not an NNCS object'
    assert isinstance(verifyPRM, VerifyPRM_NNCS), 'error: verifyPRM is not a VerifyPRM_NNCS object'

    reachPRM = ReachPRM_NNCS()
    reachPRM.initSet = copy.deepcopy(verifyPRM.initSet)
    reachPRM.numSteps = copy.deepcopy(verifyPRM.numSteps)
    reachPRM.refInputs = copy.deepcopy(verifyPRM.refInputs)
    reachPRM.filterProb = copy.deepcopy(verifyPRM.pf)
    reachPRM.lpSolver = copy.deepcopy(verifyPRM.lpSolver)
    reachPRM.show = copy.deepcopy(verifyPRM.show)
    reachPRM.numCores = copy.deepcopy(verifyPRM.numCores)

    # get traces using DFS algorithm
    print('Get ProbStar traces using DFS-reachability algorithm...')
    start = time.time()
    traces, p_ig0 = reachDFS_DLNNCS(ncs, reachPRM)
    end = time.time()
    reachTime = end-start  # reachability time in seconds
    print('Total ProbStar traces: {}'.format(len(traces)))

    DNF_transform_time = []  # get abstract DNF
    # get temporal specifications
    specs = verifyPRM.temporalSpecs

    if verifyPRM.numCores > 1:
        pool = multiprocessing.Pool(verifyPRM.numCores)
    else:
        pool = None


    print('Verifying traces against temporal specification ...')
    # verify temporal specifications
    p_SAT = []
    checking_time = [] # get checking time
    verifyTime = []  # total verification time
    p_SAT_MIN = []
    p_SAT_MAX = []
    CDNF_SAT = []
    CDNF_IG = []
    p_IG = []
    SAT_traces = []
    conservativeness= [] # conservativeness of verification result in percentage (pmax - pmin)/pmax
    constitution = []    # the constitution of ignored trace in estimating the pmax
    p_input = verifyPRM.initSet.estimateProbability()
    # if constitution = 0 -> pmax is p_exact
    for k in range(0, len(specs)):
        spec = specs[k]
        start = time.time()
        p_sat_min = []
        p_sat_max = []
        p_ig = []
        cdnf_sat = []
        cdnf_ig = []
        conserv = []
        constit = []
        sat_trace = []
        if pool is None:
            for i in range(0, len(traces)):
                print('Verifying trace {} against spec {}...'.format(i, k))
                trace = traces[i]
                p_max, p_min, p_ig1, cdnf_sat1, cdnf_ig1, sat_trace1 = check_sat_on_trace_for_full_analysis(spec, trace)
                p_sat_max.append(p_max)
                p_sat_min.append(p_min)
                p_ig.append(p_ig1)
                cdnf_sat.append(cdnf_sat1)
                cdnf_ig.append(cdnf_ig1)
                sat_trace.append(sat_trace1)

        else:
             RS = pool.map(check_sat_on_trace_for_full_analysis, zip([spec]*len(traces), traces))
             for S in RS:
                 p_sat_max.append(S[0])
                 p_sat_min.append(S[1])
                 p_ig.append(S[2])
                 cdnf_sat.append(S[3])
                 cdnf_ig.append(S[4])
                 sat_trace.append(S[5])
        
        p_SAT_MAX.append(min(sum(p_sat_max) + p_ig0, p_input))   # give the upperbound of probability of satisfaction
        p_SAT_MIN.append(min(sum(p_sat_min), p_input)) # give the lower bound of probability of satisfaction
        p_IG.append(p_ig)
        CDNF_SAT.append(cdnf_sat)
        CDNF_IG.append(cdnf_ig)
        sat_trace_short = [ele for ele in sat_trace if ele != []]
        SAT_traces.append(sat_trace_short)


        if sum(p_sat_max) != 0:
            conserv1 = 100*(min(sum(p_sat_max) + p_ig0, p_input) - min(sum(p_sat_min), p_input))/(min(sum(p_sat_max) + p_ig0, p_input))
            constit1 = 100*(sum(p_ig) + p_ig0)/(min(sum(p_sat_max) + p_ig0, p_input))
        else:
            conserv1 = 0.0
            constit1 = 0.0
            
        conservativeness.append(conserv1)
        constitution.append(constit1)
        
        end = time.time()
        ct = end-start

        checking_time.append(ct)
        verifyTime.append(ct + reachTime)

        
   
    return traces, p_SAT_MAX, p_SAT_MIN, reachTime, checking_time, \
        verifyTime, p_IG, p_ig0, CDNF_SAT, CDNF_IG, conservativeness, constitution, SAT_traces
    
