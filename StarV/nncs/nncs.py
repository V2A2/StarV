"""
  Generic Neural Network Control System Class
  
  Dung Tran, 8/14/2023
"""

from StarV.net.network import NeuralNetwork, reachExactBFS, reachApproxBFS
from StarV.plant.dlode import DLODE
from StarV.plant.lode import LODE
from StarV.set.probstar import ProbStar
from StarV.spec.dProbStarTL import Formula, DynamicFormula
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
        self.timeOut = np.Inf # timeout for verification process 


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

       Properties:
           @type: 1) linear-nncs: relu net + Linear ODEs
                  2) nonlinear-nncs: relu net + Nonlinear ODEs / sigmoid net + ODEs 
           @in_dim: input dimension
           @out_dim: output dimension

       Methods: 
           @reach: compute reachable set
    """

    def __init__(self, controller_net, plant, type=None):

        assert isinstance(controller_net, NeuralNetwork), 'error: net should be a Neural Network object'
        assert isinstance(plant, DLODE) or isinstance(plant, LODE), 'error: plant should be a discrete ODE object'

        # TODO implement isReLUNetwork?

        # checking consistency
        
        assert plant.nI == controller_net.out_dim, 'error: number of plant inputs \
        does not equal to the number of controller outputs'

        self.controller = controller_net
        self.plant = plant
        self.nO = plant.nO
        self.nI = controller_net.in_dim
        self.nI_fb = plant.nO    # number of feedback inputs to the controller
        self.nI_ref = controller_net.in_dim - self.nI_fb   # number of reference inputs to the controller
        self.type = type
        self.RX = None     # state reachable set
        self.RY = None     # output reachable set
        self.RU = None     # control set
        
    def info(self):
        """print information of the neural network control system"""

        print('\n=================NEURAL NETWORK CONTROL SYSTEM=================')
        print('\n nncs-type: {}'.format(self.type))
        print('\n number of outputs: {}'.format(self.nO))
        print('\n number of inputs: {}'.format(self.nI))
        print('\n number of feeback inputs: {}'.format(self.nI_fb))
        print('\n numher of reference inputs: {}'.format(self.nI_ref))
        print('\n network controller & plant model information:')
        self.controller.info()
        self.plant.info()

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
            self.RX, self.RY, self.RU = reachBFS_DLNNCS(self.controller, self.plant, reachPRM)
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
                         
        
def reachBFS_DLNNCS(net, plant, reachPRM):
    'reachability of discrete linear NNCS'

    assert isinstance(reachPRM, ReachPRM_NNCS), 'error: reachability parameter should be an ReachPRM_NNCS object'
    assert reachPRM.initSet is not None, 'error: there is no initial set for reachability'
    assert reachPRM.numSteps >= 1, 'error: number of time steps should be >= 1'
    
    if reachPRM.numCores > 1:
        pool = multiprocessing.Pool(reachPRM.numCores)
    else:
        pool = None

    p_ignored = 0.0   # total probability of ignored input subsets

    if reachPRM.show:
        print('\nReachability analysis of discrete linear NNCS for {} steps...'.format(reachPRM.numSteps))
                
    RX = []   # state reachable set RX = [RX1, ..., RYN], RXi = [RXi1, RXi2, ...RXiM]
    RY = []   # output reachable set RY = [RY1, RY2, ... RYN], RYi = [RYi1, RYi2, ..., RYiM]
    RU = []   # control set RU = [RU1, RU2, ..., RUN], RUi = [RUi1, RUi2, ...,RUiM]
    
    if reachPRM.method == 'exact-star' or reachPRM.method == 'exact-probstar':
        for i in range(0, reachPRM.numSteps+1):
            if reachPRM.show:
                print('\nReachability analysis at step {}...'.format(i))
                
            if i==0:
                RX.append([reachPRM.initSet])
                X0 = RX[0]
                Y0 = X0.affineMap(A=plant.C)
                RY.append([Y0])
            
            else:
                
                IS = RX[i-1]       # initSet at step i 
                FS = RY[i-2]       # feedback set at step i, len(FS) == len(IS)
                RXi = []           # state reachable set at step i
                RYi = []           # output reachable set at step i
                RUi = []
                for j in range(0, len(FS)):
                    fb_I = FS[j].concatenate_with_vector(reachPRM.refInputs) 
                    Ui = reachExactBFS(net, [fb_I], lp_solver=reachPRM.lpSolver, pool=pool, show=False)
                    for u in Ui:
                        RX1, RY1 = plant.stepReach(X0=IS[j], U=u, subSetPredicate=True)
                        RXi.append(RX1)
                        RYi.append(RY1)
                        RUi.append(u)
                
                RX.append(RXi)
                RY.append(RYi)
                RU.append(RUi)
                
        if reachPRM.show:
            print('\nReachability analysis is done successfully!')
        
    elif reachPRM.method == 'approx-probstar':
        for i in range(0, reachPRM.numSteps+1):
            if reachPRM.show:
                print('\nReachability analysis at step {}...'.format(i))
                
            if i==0:
                RX.append([reachPRM.initSet])
                X0 = RX[0]
                Y0 = X0.affineMap(A=plant.C)
                RY.append([Y0])
            else:
                
                IS = RX[i-1]       # initSet at step i 
                FS = RY[i-2]       # feedback set at step i, len(FS) == len(IS)
                RXi = []           # state reachable set at step i
                RYi = []           # output reachable set at step i
                RUi = []
                for j in range(0, len(FS)):
                    fb_I = FS[j].concatenate_with_vector(reachPRM.refInputs)
                    pi = fb_I.estimateProbability()
                    if pi >= reachPRM.filterProb:
                        
                        Ui = reachExactBFS(net, [fb_I], lp_solver=reachPRM.lpSolver, pool=pool, show=False)
                        for u in Ui:
                            RX1, RY1 = plant.stepReach(X0=IS[j], U=u, subSetPredicate=True)
                            RXi.append(RX1)
                            RYi.append(RY1)
                            RUi.append(u)
                    else:
                        p_ignored = p_ignored + pi  # update ignored probability
                
                RX.append(RXi)
                RY.append(RYi)
                RU.append(RUi)
                
        if reachPRM.show:
            print('\nReachability analysis is done successfully!')
        
    else:
        raise RuntimeError('Unknow/Unsupported reachability method')


    return RX, RY, RU, p_ignored

def verifyBFS_DLNNCS(ncs, verifyPRM):
    'Q^2 verification of DLNNCS'

    assert isinstance(ncs, NNCS), 'error: first input should be an NNCS object'
    assert isinstance(verifyPRM, VerifyPRM_NNCS), 'error: second input should be a VerifyPRM_NNCS object'

    reachPRM = ReachPRM_NNCS()
    reachPRM.initSet = copy.deepcopy(verifyPRM.initSet)
    reachPRM.numSteps = copy.deepcopy(verifyPRM.numSteps)
    reachPRM.refInputs = copy.deepcopy(verifyPRM.refInputs)
    reachPRM.filterProb = copy.deepcopy(verifyPRM.pf)
    reachPRM.lpSolver = copy.deepcopy(verifyPRM.lpSolver)
    reachPRM.show = copy.deepcopy(verifyPRM.show)
    reachPRM.numCores = copy.deepcopy(verifyPRM.numCores)
    
    if verifyPRM.pf == 0.0:
        reachPRM.method = 'exact-probstar'
    else:
        reachPRM.method = 'approx-probstar'

    # perform reachability analysis to compute the reachable sets
    RX, RY, RU, p_ignored = reachBFS_DLNNCS(ncs.controller, ncs.plant, reachPRM)
        
    # check the intersection of the reachable sets with unsafe region
    CeIn = []  # counterexample input set
    CeOut = [] # counterexample output set
    Ql = []  # qualitative results 0->UNSAT, 1->SAT, 2->Unknown
    Qt = []  # quantitative results, i.e., probability of violation (or SAT)
    Qt_lb = [] # lower bound of probability of violation, if pf > 0 is used
    Qt_ub = [] # upper bound of probability of violation, if pf > 0 is used

    i = 0
    for Rk in RX:
        Cek = []
        Cok = []
        pk = 0.0
        Qlk = 0
        i = i + 1
        j = 0
        for S in Rk:
            j = j + 1
            Z1 = copy.deepcopy(S)
            #Z1.addMultipleConstraints(verifyPRM.unsafeSpec[0], verifyPRM.unsafeSpec[1])
            Z1.addMultipleConstraintsWithoutUpdateBounds(verifyPRM.unsafeSpec[0], verifyPRM.unsafeSpec[1])
            Ceki = ProbStar(verifyPRM.initSet.V, Z1.C, Z1.d, Z1.mu, Z1.Sig, Z1.pred_lb, Z1.pred_ub)
            
            if verifyPRM.verifyMethod == 'Ql':

                if not Ceki.isEmptySet():
                    Qlk = 1
                    Cek.append(Ceki)
                    Cok.append(Z)


            elif verifyPRM.verifyMethod == 'Qt':

                pki = Ceki.estimateProbability()
                pk = pk + pki

            elif verifyPRM.verifyMethod == 'Q2':

                if not Ceki.isEmptySet():
                    Qlk = 1
                    Cek.append(Ceki)
                    Cok.append(Z1)
                pki = Ceki.estimateProbability()
                pk = pk + pki
                #print('j = {}, pkj = {}'.format(j,pki))

            else:
                raise RuntimeError('Unknown verification option')

            #print('i = {}, pk = {}'.format(i, pk))

        CeIn.append(Cek)
        CeOut.append(Cok)
        Ql.append(Qlk)
        Qt.append(pk)
        Qt_lb.append(max(0.0, pk-p_ignored))
        Qt_ub.append(min(pk+p_ignored,1.0))

    return RX, RY, RU, CeIn, CeOut, Ql, Qt, Qt_lb, Qt_ub, p_ignored
    
    

def stepReach_DLNNCS(net, plant, X0, refInputs, filterProb, numCores=1, lp_solver='Gurobi'):
    'one-step reachability analysis of Discrete linear NNCS'

    # X0: initial set of state of the plant
    # Y0: feedback to the network controller

    if filterProb < 0:
        raise RuntimeError('Invalid filtering probability')

    if numCores > 1:
        pool = multiprocessing.Pool(numCores)
    else:
        pool = None

    RX = []
    Y0 = X0.affineMap(plant.C)
    fb_I = Y0.concatenate_with_vector(refInputs)
    p_ig = 0.0
    if filterProb == 0:
        RU = reachExactBFS(net, [fb_I], lp_solver, pool=pool, show=False)
    else:
        RU, pi = reachApproxBFS(net, [fb_I], filterProb, lp_solver, pool=pool, show=False)
        p_ig = p_ig + pi
    for U in RU:
        RX1, _ = plant.stepReach(X0=X0, U=U, subSetPredicate=True)
        RX.append(RX1)
        

    return RX, p_ig

def reachDFS_DLNNCS(net, plant, reachPRM):
    'Depth First Search Reachability Analysis for Discrete Linear NNCS'

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
            Xi_plus_1, pig_i = stepReach_DLNNCS(net, plant, Xi, reachPRM.refInputs, reachPRM.filterProb,\
                                                reachPRM.numCores, reachPRM.lpSolver)
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
    p_max, p_min = DNF_spec.evaluate(trace)
    
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
    # Dung Tran: 1/14/2024


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
    traces, p_ig0 = reachDFS_DLNNCS(ncs.controller, ncs.plant, reachPRM)
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
             
        
        p_SAT_MAX.append(sum(p_sat_max) + p_ig0)   # give the upperbound of probability of satisfaction
        p_SAT_MIN.append(sum(p_sat_min))       # give the lower bound of probability of satisfaction
        end = time.time()
        ct = end-start

        checking_time.append(ct)
        verifyTime.append(ct + reachTime)

   
    return traces, p_SAT_MAX, p_SAT_MIN, reachTime, checking_time, verifyTime
    

def verify_temporal_specs_DLNNCS_for_full_analysis(ncs, verifyPRM):
    'verify temporal behaviors of DLNNCS'
    # Dung Tran: 1/14/2024
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
    traces, p_ig0 = reachDFS_DLNNCS(ncs.controller, ncs.plant, reachPRM)
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
        
        p_SAT_MAX.append(sum(p_sat_max) + p_ig0)   # give the upperbound of probability of satisfaction
        p_SAT_MIN.append(sum(p_sat_min)) # give the lower bound of probability of satisfaction
        p_IG.append(p_ig)
        CDNF_SAT.append(cdnf_sat)
        CDNF_IG.append(cdnf_ig)
        sat_trace_short = [ele for ele in sat_trace if ele != []]
        SAT_traces.append(sat_trace_short)


        if sum(p_sat_max) != 0:
            conserv1 = 100*(sum(p_sat_max) + p_ig0 - sum(p_sat_min))/(sum(p_sat_max) + p_ig0)
            constit1 = 100*(sum(p_ig) + p_ig0)/(sum(p_sat_max) + p_ig0)
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
    
def verify_traces(traces, p_ig0, verifyPRM):
    'This function is used when reachable traces are available'

    
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
        
        p_SAT_MAX.append(sum(p_sat_max) + p_ig0)   # give the upperbound of probability of satisfaction
        p_SAT_MIN.append(sum(p_sat_min)) # give the lower bound of probability of satisfaction
        p_IG.append(p_ig)
        CDNF_SAT.append(cdnf_sat)
        CDNF_IG.append(cdnf_ig)
        sat_trace_short = [ele for ele in sat_trace if ele != []]
        SAT_traces.append(sat_trace_short)


        if sum(p_sat_max) != 0:
            conserv1 = 100*(sum(p_sat_max) + p_ig0 - sum(p_sat_min))/(sum(p_sat_max) + p_ig0)
            constit1 = 100*(sum(p_ig) + p_ig0)/(sum(p_sat_max) + p_ig0)
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
