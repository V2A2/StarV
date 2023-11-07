"""
  Generic Neural Network Control System Class
  
  Dung Tran, 8/14/2023
"""

from StarV.net.network import NeuralNetwork, reachExactBFS
from StarV.plant.dlode import DLODE
from StarV.plant.lode import LODE
import multiprocessing

class ReachPRM_NNCS(object):
    'reachability parameters for NNCS'

    def __init__(self):
        self.initSet = None
        self.numSteps = 1
        self.refInput = None
        self.method = 'exact-probStar'
        self.lpSolver = 'gurobi'
        self.show = True
        self.numCores = 1


class NNCS(object):
    """Generic neural network control system class

       % nerual network control system architecture
       %
       %              --->| plant ---> x(k+1)--------------->y(k+1) 
       %             |                                       |
       %             |                                       |
       %             u(k) <---- controller |<------ y(k)-----|--- (output feedback) 
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
        assert plant.nO == controller_net.in_dim, 'error: the number of plant outputs \
        does not equal to the number of controller inputs'

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
        print('nncs-type: {}'.format(self.type))
        self.controller.info()
        self.plant.info()

    def reach(self, reachPRM):
        'reachability analysis'

        # reachPRM: reachability parameters
        #   1) reachPRM.initSet
        #   2) reachPRM.refInput
        #   3) reachPRM.numSteps
        #   4) reachPRM.method
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
            elif i==1:
                X0 = RX[0]
                X1, Y1 = plant.stepReach(X0[0], U=None, subSetPredicate=False)
                RX.append([X1])
                RY.append([Y1])
            else:
                
                IS = RX[i-1]       # initSet at step i 
                FS = RY[i-2]       # feedback set at step i, len(FS) == len(IS)
                RXi = []           # state reachable set at step i
                RYi = []           # output reachable set at step i
                RUi = []
                for j in range(0, len(FS)):
                    Ui = reachExactBFS(net, [FS[j]], lp_solver=reachPRM.lpSolver, pool=pool, show=False)
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
        
    elif reachPRM.method == 'approx-star':
        pass
    else:
        raise RuntimeError('Unknow/Unsupported reachability method')


    return RX, RY, RU


def stepReach_DLNNCS(net, plant, X0, Y0, numCores=1, lp_solver='Gurobi'):
    'one-step reachability analysis of Discrete linear NNCS'

    # X0: initial set of state of the plant
    # Y0: feedback to the network controller

    if numCores > 1:
        pool = multiprocessing.Pool(numCores)
    else:
        pool = None

    RX = []
    RY = []
    RU = reachExactBFS(net, Y0, lp_solver, pool=pool, show=False)
    for U in RU:
        RX1, RY1 = plant.stepReach(X0=X0, U=U, subSetPredicate=True)
        RX.append(RX1)
        RY.append(RY1)

    return RX, RY, RU

def reachDFS_DLNNCS(net, plant, reachPRM):
    'Depth First Search Reachability Analysis for Discrete Linear NNCS'

    assert isinstance(reachPRM, ReachPRM_NNCS), 'error: reachability parameter should be an ReachPRM_NNCS object'
    assert reachPRM.initSet is not None, 'error: there is no initial set for reachability'
    assert reachPRM.numSteps >= 1, 'error: number of time steps should be >= 1'

    k = reachPRM.numSteps  # number of reachability steps
    traces = []  # contains all reachable traces, traces = [T1, T2, ..., TN], Ti = [Xi0, Xi1, ..., Xik]
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
    
    
    
    
    
    pass
