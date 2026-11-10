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
        - 11/15/2025: specification-guided reachability
  Sung Woo Choi, 06/10/2025
  	- added DynNN_NNCS (dynamic neural network NNCS)
	- plant: dynamic NN
	- controller: NN controller
"""

from StarV.net.network import NeuralNetwork, reachExactBFS, reachApproxBFS
from StarV.plant.dlode import DLODE
from StarV.plant.lode import LODE
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
from StarV.spec.dProbStarTL import Formula, DynamicFormula, AtomicPredicate, Predicate
import multiprocessing
import warnings
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
        self.RF = 1.0
        self.DR = 0
        self.relu_approx_mode = 'analytic_gaussian'
        self.cov_mode = 'full'

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
        self.RF = 1.0
        self.DR = 0
        self.relu_approx_mode = 'analytic_gaussian'
        self.cov_mode = 'full'

class VerifyPRM_NNCS(object):
    'reachability parameters for NNCS'

    def __init__(self):
        self.initSet = None
        self.numSteps = 1
        self.refInputs = []
        self.method = 'exact-probstar'
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
        self.RF = 1.0
        self.DR = 0
        self.relu_approx_mode = 'analytic_gaussian'
        self.cov_mode = 'full'
        self.use_lp_tightening = False
        self.refine_max_splits = 0
        self.refine_tol = 0.01

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
        self.num_partitions = 1 # number of initial-space partitions used
        

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

        self.controller = controller_net
        self.plant = plant

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
        print(self.controller)
        print(self.plant)
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
            self.RX, p_ignored = reachBFS_DLNNCS(self, reachPRM)
            return self.RX, p_ignored
        elif self.type == 'DynNN-NNCS': # dynamic neural network NNCS
            self.RX, p_ignored = reachBFS_DynNN_NNCS(self, reachPRM)
            return self.RX, p_ignored
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


def stepSim_DLNNCS(ncs, state_vec, ref_input_vec=[]):
    'step simulation of DLNNCS'

    net = ncs.controller
    plant = ncs.plant

    x_vec = np.asarray(state_vec, dtype=np.float64).ravel()
    fb_y = np.matmul(plant.C, x_vec)
    if ref_input_vec is not None and len(ref_input_vec) > 0:
        ref_vec = np.asarray(ref_input_vec, dtype=np.float64).ravel()
        fb_input = np.concatenate([ref_vec, fb_y])
    else:
        fb_input = fb_y
    u = np.asarray(net.evaluate(fb_input), dtype=np.float64).ravel()
    A_mat = plant.A if hasattr(plant, 'A') and not hasattr(plant, 'gA') else (plant.gA if getattr(plant, 'gA', None) is not None else plant.A)
    B_mat = plant.B if hasattr(plant, 'B') and not hasattr(plant, 'gB') else (plant.gB if getattr(plant, 'gB', None) is not None else plant.B)
    x = np.matmul(A_mat, x_vec) + np.matmul(B_mat, u)

    return x



def _eval_single_partition_approx_safety(ncs, init_set, reachPRM_base, verifyPRM):
    """Run approximate reachability and two-stage spec verification on a single initial ProbStar partition."""
    reachPRM = copy.deepcopy(reachPRM_base)
    reachPRM.initSet = init_set
    if isinstance(ncs, NNCS):
        RX, p_ignored = reachBFS_DLNNCS(ncs, reachPRM)
    elif isinstance(ncs, AEBS_NNCS):
        RX, p_ignored = reachBFS_AEBS(ncs, reachPRM)
    elif isinstance(ncs, DynNN_NNCS):
        RX, p_ignored = reachBFS_DynNN_NNCS(ncs, reachPRM)
    else:
        raise RuntimeError('Unknown system')

    n = len(RX)
    p_min_vec = np.zeros(n, dtype=np.float64)
    p_max_vec = np.zeros(n, dtype=np.float64)
    p_hat_vec = np.zeros(n, dtype=np.float64)
    n0 = init_set.nVars0
    weights_acc = np.zeros(n0, dtype=np.float64)

    unsafe_mat = verifyPRM.unsafeSpec[0]
    unsafe_vec = verifyPRM.unsafeSpec[1]
    use_lp = getattr(verifyPRM, 'use_lp_tightening', True)

    for k in range(n):
        for r_set in RX[k]:
            spec_res = r_set.verifySpec(
                unsafe_mat, unsafe_vec,
                lp_solver=verifyPRM.lpSolver,
                use_lp_tightening=use_lp
            )
            p_min_vec[k] += spec_res['p_min']
            p_max_vec[k] += spec_res['p_max']
            p_hat_vec[k] += spec_res['p_hat']
            gap_k = max(0.0, spec_res['p_max'] - spec_res['p_min'])
            if 'sensitivity_w0' in spec_res and gap_k > 0.0:
                w0 = np.asarray(spec_res['sensitivity_w0'], dtype=np.float64).ravel()
                if len(w0) == n0:
                    weights_acc += gap_k * w0

    return {
        'init_set': init_set,
        'RX': RX,
        'p_ignored': p_ignored,
        'p_min_vec': p_min_vec,
        'p_max_vec': p_max_vec,
        'p_hat_vec': p_hat_vec,
        'gap': float(np.max(p_max_vec - p_min_vec)),
        'weights': weights_acc,
    }


def verifyBFS_DLNNCS(ncs, verifyPRM):
    'Q^2 verification of DLNNCS'

    assert isinstance(verifyPRM, VerifyPRM_NNCS), 'error: second input should be a VerifyPRM_NNCS object'

    reachPRM = ReachPRM_NNCS()
    reachPRM.initSet = copy.deepcopy(verifyPRM.initSet)
    reachPRM.numSteps = copy.deepcopy(verifyPRM.numSteps)
    reachPRM.refInputs = copy.deepcopy(verifyPRM.refInputs)
    reachPRM.method = copy.deepcopy(getattr(verifyPRM, 'method', 'exact-probstar'))
    reachPRM.filterProb = copy.deepcopy(verifyPRM.pf)
    reachPRM.lpSolver = copy.deepcopy(verifyPRM.lpSolver)
    reachPRM.show = copy.deepcopy(verifyPRM.show)
    reachPRM.numCores = copy.deepcopy(verifyPRM.numCores)
    reachPRM.RF = getattr(verifyPRM, 'RF', 0.0)
    reachPRM.DR = getattr(verifyPRM, 'DR', 0)
    reachPRM.relu_approx_mode = getattr(verifyPRM, 'relu_approx_mode', 'analytic_gaussian')
    reachPRM.cov_mode = getattr(verifyPRM, 'cov_mode', 'full')

    if reachPRM.method in ('approx-probstar', 'approx'):
        pI = reachPRM.initSet.get_input_probability()
        part0 = _eval_single_partition_approx_safety(ncs, reachPRM.initSet, reachPRM, verifyPRM)
        partitions = [part0]

        max_splits = int(getattr(verifyPRM, 'refine_max_splits', 0))
        refine_tol = float(getattr(verifyPRM, 'refine_tol', 0.01))
        splits_done = 0

        while splits_done < max_splits:
            total_p_min = sum(p['p_min_vec'] for p in partitions)
            total_p_max = sum(p['p_max_vec'] for p in partitions)
            max_total_gap = float(np.max(total_p_max - total_p_min))
            if max_total_gap <= refine_tol:
                break
            # Pick partition with largest uncertainty gap
            best_idx = int(np.argmax([p['gap'] for p in partitions]))
            if partitions[best_idx]['gap'] <= 1e-12:
                break
            target_part = partitions.pop(best_idx)
            children_init = target_part['init_set'].splitInitialPredicates(
                weights=target_part['weights']
            )
            if len(children_init) < 2:
                partitions.append(target_part)
                break
            for child_init in children_init:
                partitions.append(
                    _eval_single_partition_approx_safety(ncs, child_init, reachPRM, verifyPRM)
                )
            splits_done += 1

        n = len(partitions[0]['RX'])
        RX = []
        CeIn = []
        CeOut = []
        Ql = []
        Qt = []
        Qt_lb = []
        Qt_ub = []
        Qt_max = []
        p_ignored = [0.0] * n

        for k in range(n):
            Rk = []
            for p in partitions:
                Rk.extend(p['RX'][k])
            RX.append(Rk)
            pk_lb = float(np.clip(sum(p['p_min_vec'][k] for p in partitions), 0.0, pI))
            pk_ub = float(np.clip(sum(p['p_max_vec'][k] for p in partitions), pk_lb, pI))
            pk_hat = float(np.clip(sum(p['p_hat_vec'][k] for p in partitions), pk_lb, pk_ub))
            Qlk = 1 if pk_ub > 0.0 else 0
            CeIn.append([])
            CeOut.append([])
            Ql.append(Qlk)
            Qt_lb.append(pk_lb)
            Qt.append(pk_hat)
            Qt_ub.append(pk_ub)
            Qt_max.append(min(pk_ub + 1.0 - pI, 1.0))

        res = VerifyRes_NNCS()
        res.RX = RX
        res.CeIn = CeIn
        res.CeOut = CeOut
        res.Ql = Ql
        res.Qt = Qt
        res.Qt_lb = Qt_lb
        res.Qt_ub = Qt_ub
        res.Qt_max = Qt_max
        res.p_ignored = p_ignored
        res.num_partitions = len(partitions)
        return res

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
    Qt_lb = [] # lower bound of probability of violation
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
            Z1 = Rk[i].addMultipleConstraints(verifyPRM.unsafeSpec[0], verifyPRM.unsafeSpec[1], tighten_bounds=False)
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
        Qt_lb.append(pk)
        Qt.append(pk)
        Qt_ub.append(min(pk + p_ign_k,1.0))
        Qt_max.append(min(pk + p_ign_k + 1 - pI, 1.0))


    res = VerifyRes_NNCS()
    res.RX = RX
    res.CeIn = CeIn
    res.CeOut = CeOut
    res.Ql = Ql
    res.Qt = Qt
    res.Qt_lb = Qt_lb
    res.Qt_ub = Qt_ub
    res.Qt_max = Qt_max
    res.p_ignored = p_ignored
    res.num_partitions = 1

    return res

    

def reachBFS_DLNNCS(ncs, reachPRM):
    'breath first search reachability of DLNNCS'

    assert isinstance(reachPRM, ReachPRM_NNCS), 'error: reachability parameter should be an ReachPRM_NNCS object'
    assert reachPRM.initSet is not None, 'error: there is no initial set for reachability'
    assert reachPRM.numSteps >= 1, 'error: number of time steps should be >= 1'

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
    method = getattr(reachPRM, 'method', 'exact-probstar')
    RF = getattr(reachPRM, 'RF', 0.0)
    DR = getattr(reachPRM, 'DR', 0)
    relu_approx_mode = getattr(reachPRM, 'relu_approx_mode', 'analytic_gaussian')

    if hasattr(ncs, 'ncs'):
        net = ncs.ncs.controller
        plant = ncs.ncs.plant
    else:
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

    if method in ('approx-probstar', 'approx-star', 'approx'):
        U_approx = reachApproxBFS(
            net, fb_I, lp_solver=lp_solver, pool=pool, show=False,
            RF=RF, DR=DR, relu_approx_mode=relu_approx_mode
        )
        RU = U_approx if isinstance(U_approx, list) else [U_approx]
    else:
        RU = reachExactBFS(net, [fb_I], lp_solver, pool=pool, show=False)

    if pool is not None:
        pool.close()
        pool.join()
    for U in RU:
        RX1, _ = plant.stepReach(X0=Xi, U=U, subSetPredicate=True)
        if filterProb == 0 or method in ('approx-probstar', 'approx-star', 'approx'):
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
            with multiprocessing.Pool(reachPRM.numCores) as pool:
                Xi1 = stepReach_AEBS(ncs, Xi, pool, reachPRM=reachPRM)
        else:
            Xi1 = stepReach_AEBS(ncs, Xi, None, reachPRM=reachPRM)
      

        # Xi1 = stepReach_AEBS(ncs, Xi, pool)    # this is AEBS NNCS

        if reachPRM.filterProb == 0 or getattr(reachPRM, 'method', 'exact-probstar') in ('approx-probstar', 'approx-star', 'approx'):
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
    if T < 1:
        raise RuntimeError('Invalid number of steps')

    if reachPRM.numCores > 1:
        pool = multiprocessing.Pool(reachPRM.numCores)
    else:
        pool = None

    X = []
    X.append([X0])
    p_ignored = [0.]
    for i in range(0, T+1):
        X1, p_ig1 = stepReach_AEBS_multipleInitSets(AEBS, X[i], pf, pool, reachPRM=reachPRM)         
        X.append(X1)
        p_ign1 = p_ig1 + p_ignored[i]
        p_ignored.append(p_ign1)

    if pool is not None:
        pool.close()
        pool.join()

    return X, p_ignored

def stepReach_AEBS_multipleInitSets(AEBS, X0, pf, pool, reachPRM=None):
    'step reach of AEBS with multiple initial sets'

    n = len(X0)
    X1 = []
    p_ignored = 0
    is_approx = reachPRM is not None and getattr(reachPRM, 'method', 'exact-probstar') in ('approx-probstar', 'approx-star', 'approx')
    for i in range(0, n):
        if pf == 0 or is_approx:
            X1i = stepReach_AEBS(AEBS, X0[i], pool, reachPRM=reachPRM)
            X1.extend(X1i)
        else:
            pX = X0[i].estimateProbability()
            if pX > pf:
                X1i = stepReach_AEBS(AEBS, X0[i], pool, reachPRM=reachPRM)
                X1.extend(X1i)

            else:
                p_ignored = p_ignored + pX

    return X1, p_ignored


def stepReach_AEBS(AEBS, X0, pool, reachPRM=None):
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
    method = getattr(reachPRM, 'method', 'exact-probstar') if reachPRM is not None else 'exact-probstar'
    lp_solver = getattr(reachPRM, 'lpSolver', 'gurobi') if reachPRM is not None else 'gurobi'
    RF = getattr(reachPRM, 'RF', 0.0) if reachPRM is not None else 0.0
    DR = getattr(reachPRM, 'DR', 0) if reachPRM is not None else 0
    relu_approx_mode = getattr(reachPRM, 'relu_approx_mode', 'analytic_gaussian') if reachPRM is not None else 'analytic_gaussian'
    show = getattr(reachPRM, 'show', True) if reachPRM is not None else True

    norm_X = X0.affineMap(norm_mat)
    
    if show:
        print('Computing reachable set of RL controller ...\n')

    # step 2: compute brake output of the RL controller
    if method in ('approx-probstar', 'approx-star', 'approx'):
        b_out = reachApproxBFS(
            controller, norm_X, lp_solver=lp_solver, pool=pool, show=False,
            RF=RF, DR=DR, relu_approx_mode=relu_approx_mode
        )
        brake = b_out if isinstance(b_out, list) else [b_out]
    else:
        brake = reachExactBFS(controller, [norm_X], lp_solver=lp_solver, pool=pool)
    
    m = len(brake)

    if show:
        print('Geting input sets to transformer ...\n')
    # step 3: compose brake output and normalized speed to get inputs to transformer
    speed_brake = []
    for i in range(0, m):
        if isinstance(brake[i], (ProbStar, Star)):
            speed_brake_i = norm_X.stackWithSharedPrefix(brake[i], self_rows=[1])
        else:
            raise RuntimeError('unsupported brake set type {}'.format(type(brake[i])))
        speed_brake.append(speed_brake_i)


    if show:
        print('Computing transformer output ... \n')
    # step 4: get transformer output
    tf_outs = []
    for i in range(0, m):
        if method in ('approx-probstar', 'approx-star', 'approx'):
            tf_out = reachApproxBFS(
                transformer, speed_brake[i], lp_solver=lp_solver, pool=pool, show=False,
                RF=RF, DR=DR, relu_approx_mode=relu_approx_mode
            )
            tf_outs.extend(tf_out if isinstance(tf_out, list) else [tf_out])
        else:
            tf_out = reachExactBFS(transformer, [speed_brake[i]], lp_solver=lp_solver, pool=pool)
            tf_outs.extend(tf_out)

    if show:
        print('Getting control input set to the plant and scale it ...\n')
    # step 5: get control input to the plant and scale it using scale matrix
    n = len(tf_outs)
    controls = []
    for i in range(0, n):
        if isinstance(tf_outs[i], (ProbStar, Star)):
            control = norm_X.stackWithSharedPrefix(tf_outs[i], self_rows=[1])
        else:
            raise RuntimeError('unsupported transformer output set type {}'.format(type(tf_outs[i])))
        controls.append(control.affineMap(scale_mat)) # scale the control inputs

    if show:
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
        last_set = trace[n - 1]
        for i in range(0, n):
            V_pad = trace[i].padBasisMatrix(last_set.nVars)
            if isinstance(last_set, ProbStar):
                R = last_set.with_updated_state(V=V_pad, env_dag=trace[i].env_dag if len(trace[i].env_dag) > 0 else last_set.env_dag)
            elif isinstance(last_set, Star):
                R = Star(V_pad, last_set.C, last_set.d, last_set.pred_lb, last_set.pred_ub)
            else:
                raise RuntimeError('unsupported trace set type {}'.format(type(last_set)))
            trace1.append(R)
        traces1.append(trace1)         
    
    return traces1, p_ig

def specGuidedVerification_DLNNCS(ncs, verifyPRM):
    'specification-guided verification'

    # Dung Tran: 11/20/2025

    pass

def specGuidedReachDFS_DLNNCS(ncs, reachPRM, abSpec):
    'specification guided reachability using abstract DNF specification to reduce the number of traces'
    # Dung Tran: 11/10/2025 updated date: 11/20/2025

    assert isinstance(abSpec, DynamicFormula), 'error: abSpec (abstract DNF specification) should be a DynamicFormula object, not None'

    sat_traces = []
    p_ig = []
    for i in range(0, len(abSpec)):
        spec_i = abSpec[i]
        traces_i, p_ig_i = singleSpecGuidedReachDFS_DLNNCS(ncs, reachPRM, spec_i)
        sat_traces.append(traces_i)
        p_ig.append(p_ig_i)

    return sat_traces, p_ig

def singleSpecGuidedReachDFS_DLNNCS(ncs, reachPRM, spec_i):
    'a single abstract specification - guided reachability'

    # Dung Tran: 11/11/2025, update date: 11/20/2025
    # abSpec (DNF format) = P = [[P0], [P1], ..., [Pm]], spec_i = Pi = [C0, C1, ..., Cn], Ci is an AtomicPredicate
    
    assert isinstance(reachPRM, ReachPRM_NNCS), 'error: reachability parameter should be an ReachPRM_NNCS object'
    assert reachPRM.initSet is not None, 'error: there is no initial set for reachability'
    assert reachPRM.numSteps >= 1, 'error: number of time steps should be >= 1'
    
    # get number of time steps needed for checking spec_i, not neccessary equal to reachPRM.numSteps

    steps = []
    for i in range(len(spec_i)):
        Ci = spec_i[i]
        steps.append(Ci.t)

    k = max(steps)  # number of reachability steps
    remains = [] # contains all remaining reachable set at all k steps, remains = [RM1, RM2, ..., RMk], RMi = [X1, X2, ..., Xm]
    
    X0 = copy.deepcopy(reachPRM.initSet)
    remains.append([X0])
    T = []
    p_ig = 0.0
    traces = []
    
    while True:
        i = len(remains)  # equal to the current considering time step
        if i != 0:
            Ri = remains[i-1]
            current_step = i-1  # current time step in reachability analysis
            cur_step_pred = getCurrentSpecConstraints(spec_i, current_step)  # constraints corresponding to the current time step
        else:
            break
        
        if len(Ri) != 0:
            Xi = Ri.pop(0)

            # construct new initial set with new constraints for current time step
            newXi = Xi.addMultipleConstraints(cur_step_pred.A, cur_step_pred.b) if cur_step_pred.A is not None else Xi
            T.append(newXi)

            Xi_plus_1, pig_i = stepReach_DLNNCS_extended(ncs, newXi, reachPRM)
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
    traces1 = []
    for trace in traces:
        n = len(trace)
        trace1 = []
        last_set = trace[n - 1]
        for i in range(0, n):
            V_pad = trace[i].padBasisMatrix(last_set.nVars)
            if isinstance(last_set, ProbStar):
                R = last_set.with_updated_state(V=V_pad, env_dag=trace[i].env_dag if len(trace[i].env_dag) > 0 else last_set.env_dag)
            else:
                R = Star(V_pad, last_set.C, last_set.d, last_set.pred_lb, last_set.pred_ub)
            trace1.append(R)
        traces1.append(trace1)         
    
    return traces1, p_ig
    
def getCurrentSpecConstraints(spec, current_step):
    'get constraint matrix and vector from an abstract specification'

    # Dung Tran: 11/11/2025, update date: 11/18/2025
    # spec = [C0, C1, ..., Cn], Ci is an AtomicPredicate with time t
    # we collect all constraints for all time steps in the spec
    # for example: C0.t = 0, C1.t = 0, C2.t = 1, C3.t = 1, C4.t = 2, C5.t = 2
    # if current_step = 0, we return a composed predicate P = [C0 & C1]
    # if current_step = 1, we return a composed predicate P = [C2 & C3]

    assert current_step >= 0, 'error: invalid current_step, should be >= 0'
    P = Predicate(A=None,b=None,t=None)
    for i in range(0, len(spec)):
        Ci = spec[i]
        assert isinstance(Ci, AtomicPredicate), 'error: spec[{}] is not an AtomicPredicate object'.format(i)
        if Ci.t == current_step:
            P = P.compose(Ci)
    if P.A is None:
        warnings.warn('There are no constraints related to current_step = {}'.format(current_step))
    return P

def reachBFS_DynNN_NNCS(ncs, reachPRM):
    'breath first search reachability of dynamic neural network NNCS (DynNN_NNCS)'

    assert isinstance(reachPRM, ReachPRM_DYNNN_NNCS), 'error: reachability parameter should be an ReachPRM_NNCS object'
    assert reachPRM.initSet is not None, 'error: there is no initial set for reachability'
    assert reachPRM.numSteps >= 1, 'error: number of time steps should be >= 1'

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
    method = getattr(reachPRM, 'method', 'exact-probstar')
    RF = getattr(reachPRM, 'RF', 0.0)
    DR = getattr(reachPRM, 'DR', 0)
    relu_approx_mode = getattr(reachPRM, 'relu_approx_mode', 'analytic_gaussian')
    show = getattr(reachPRM, 'show', False)
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
        fb_I = Xi.concatenate_with_vector(refInputs) if len(refInputs) > 0 else Xi
        if method in ('approx-probstar', 'approx-star', 'approx'):
            U_approx = reachApproxBFS(
                controller, fb_I, lp_solver=lp_solver, pool=pool, show=False,
                RF=RF, DR=DR, relu_approx_mode=relu_approx_mode
            )
            Ui = U_approx if isinstance(U_approx, list) else [U_approx]
        else:
            Ui = reachExactBFS(controller, [fb_I], lp_solver, pool=pool, show=False)
    else:
        if not isinstance(Ui, list):
            Ui = [Ui]
    p_ig = 0.0

    # concatenate state and control input to construct input for NN plant, i.e., XUi = [Xi, Ui]^T
    XUi = []
    for U in Ui:
        if method in ('approx-probstar', 'approx-star', 'approx'):
            XUi.append(Xi.stackWithSharedPrefix(U))
        else:
            if filterProb > 0:
                pi = U.estimateProbability()
                if pi <= filterProb:
                    p_ig += pi
                    continue
            XUi.append(Xi.concatenate(U))
    
    # reachability of NN plant; yi = NeuralNetwork(xi, ui) = NeuralNetwork([xi, ui])
    Y = []
    for I in XUi:
        if method in ('approx-probstar', 'approx-star', 'approx'):
            Y_approx = reachApproxBFS(
                plant, I, lp_solver=lp_solver, pool=pool, show=False,
                RF=RF, DR=DR, relu_approx_mode=relu_approx_mode
            )
            Y1 = Y_approx if isinstance(Y_approx, list) else [Y_approx]
            Y.extend(Y1)
        else:
            Y1 = reachExactBFS(plant, [I], lp_solver, pool=pool, show=False)
            for Yi in Y1:
                if filterProb > 0:
                    pi = Yi.estimateProbability()
                    if pi <= filterProb:
                        p_ig += pi
                        continue
                Y.append(Yi)

    if pool is not None:
        pool.close()
        pool.join()

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

        if pool is not None:
            pool.close()
            pool.join()

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
    reachPRM.method = copy.deepcopy(getattr(verifyPRM, 'method', 'exact-probstar'))
    reachPRM.filterProb = copy.deepcopy(verifyPRM.pf)
    reachPRM.lpSolver = copy.deepcopy(verifyPRM.lpSolver)
    reachPRM.show = copy.deepcopy(verifyPRM.show)
    reachPRM.numCores = copy.deepcopy(verifyPRM.numCores)
    reachPRM.RF = getattr(verifyPRM, 'RF', 0.0)
    reachPRM.DR = getattr(verifyPRM, 'DR', 0)
    reachPRM.relu_approx_mode = getattr(verifyPRM, 'relu_approx_mode', 'analytic_gaussian')
    reachPRM.cov_mode = getattr(verifyPRM, 'cov_mode', 'full')

    if reachPRM.method in ('approx-probstar', 'approx') and int(getattr(verifyPRM, 'refine_max_splits', 0)) > 0:
        max_splits = int(verifyPRM.refine_max_splits)
        refine_tol = float(getattr(verifyPRM, 'refine_tol', 0.01))
        specs = verifyPRM.temporalSpecs
        dnf_specs = [s.getDynamicFormula() for s in specs]
        p_input = verifyPRM.initSet.get_input_probability()

        def _eval_part_temporal(init_s):
            r_prm = copy.deepcopy(reachPRM)
            r_prm.initSet = init_s
            t0 = time.time()
            tr_list, _ = reachDFS_DLNNCS(ncs, r_prm)
            r_time = time.time() - t0
            tr = tr_list[0]
            p_max_arr = np.zeros(len(specs), dtype=np.float64)
            p_min_arr = np.zeros(len(specs), dtype=np.float64)
            c_time_arr = np.zeros(len(specs), dtype=np.float64)
            w0_acc = np.zeros(init_s.nVars0, dtype=np.float64)
            for idx_s, dnf_s in enumerate(dnf_specs):
                ts0 = time.time()
                _, _, w0 = dnf_s.realization_initial_space_duals(tr)
                _, p_max_k, p_min_k, _ = dnf_s.evaluate_approx(tr)
                c_time_arr[idx_s] = time.time() - ts0
                p_max_arr[idx_s] = p_max_k
                p_min_arr[idx_s] = p_min_k
                gap_k = max(0.0, p_max_k - p_min_k)
                if len(w0) == init_s.nVars0:
                    w0_acc += gap_k * w0
            return {
                'init_set': init_s,
                'trace': tr,
                'reach_time': r_time,
                'check_time': c_time_arr,
                'p_max': p_max_arr,
                'p_min': p_min_arr,
                'gap': float(np.max(p_max_arr - p_min_arr)),
                'weights': w0_acc,
            }

        parts = [_eval_part_temporal(reachPRM.initSet)]
        splits_done = 0
        while splits_done < max_splits:
            tot_max = sum(p['p_max'] for p in parts)
            tot_min = sum(p['p_min'] for p in parts)
            if float(np.max(tot_max - tot_min)) <= refine_tol:
                break
            best_i = int(np.argmax([p['gap'] for p in parts]))
            if parts[best_i]['gap'] <= 1e-12:
                break
            target_p = parts.pop(best_i)
            children = target_p['init_set'].splitInitialPredicates(weights=target_p['weights'])
            if len(children) < 2:
                parts.append(target_p)
                break
            for c_init in children:
                parts.append(_eval_part_temporal(c_init))
            splits_done += 1

        traces = [p['trace'] for p in parts]
        reachTime = sum(p['reach_time'] for p in parts)
        checking_time = [float(sum(p['check_time'][k] for p in parts)) for k in range(len(specs))]
        verifyTime = [ct + reachTime for ct in checking_time]
        p_SAT_MAX = [float(np.clip(sum(p['p_max'][k] for p in parts), 0.0, p_input)) for k in range(len(specs))]
        p_SAT_MIN = [float(np.clip(sum(p['p_min'][k] for p in parts), 0.0, p_SAT_MAX[k])) for k in range(len(specs))]
        return traces, p_SAT_MAX, p_SAT_MIN, reachTime, checking_time, verifyTime
    

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

    if pool is not None:
        pool.close()
        pool.join()
   
    return traces, p_SAT_MAX, p_SAT_MIN, reachTime, checking_time, verifyTime
    

def verify_temporal_specs_DLNNCS_for_full_analysis(ncs, verifyPRM):
    'verify temporal behaviors of DLNNCS'
    # Dung Tran: 1/14/2024, updated 7/4/2024
    # more analysis can be done using this verification function

    assert isinstance(ncs, (NNCS, AEBS_NNCS, DynNN_NNCS)), 'error: ncs is not an NNCS object'
    assert isinstance(verifyPRM, VerifyPRM_NNCS), 'error: verifyPRM is not a VerifyPRM_NNCS object'

    reachPRM = ReachPRM_NNCS()
    reachPRM.initSet = copy.deepcopy(verifyPRM.initSet)
    reachPRM.numSteps = copy.deepcopy(verifyPRM.numSteps)
    reachPRM.refInputs = copy.deepcopy(verifyPRM.refInputs)
    reachPRM.method = copy.deepcopy(getattr(verifyPRM, 'method', 'exact-probstar'))
    reachPRM.filterProb = copy.deepcopy(verifyPRM.pf)
    reachPRM.lpSolver = copy.deepcopy(verifyPRM.lpSolver)
    reachPRM.show = copy.deepcopy(verifyPRM.show)
    reachPRM.numCores = copy.deepcopy(verifyPRM.numCores)
    reachPRM.RF = getattr(verifyPRM, 'RF', 0.0)
    reachPRM.DR = getattr(verifyPRM, 'DR', 0)
    reachPRM.relu_approx_mode = getattr(verifyPRM, 'relu_approx_mode', 'analytic_gaussian')
    reachPRM.cov_mode = getattr(verifyPRM, 'cov_mode', 'full')

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

    if pool is not None:
        pool.close()
        pool.join()
   
    return traces, p_SAT_MAX, p_SAT_MIN, reachTime, checking_time, \
        verifyTime, p_IG, p_ig0, CDNF_SAT, CDNF_IG, conservativeness, constitution, SAT_traces
    
