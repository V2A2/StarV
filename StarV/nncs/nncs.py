"""
  Generic Neural Network Control System Class
  
  Dung Tran, 8/14/2023
"""

from StarV.net.network import NeuralNetwork
from StarV.plant.dlode import DLODE
from StarV.plant.lode import LODE

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
        
    def info(self):
        """print information of the neural network control system"""

        print('\n=================NEURAL NETWORK CONTROL SYSTEM=================')
        print('nncs-type: {}'.format(self.type))
        self.controller.info()
        self.plant.info()

    def reach(self, reachPRM):
        'reachability analysis'

        # reachPRM: reachability parameters
        #   1) reachPRM.init_set
        #   2) reachPRM.ref_input
        #   3) reachPRM.numSteps
        #   4) reachPRM.method
        #   5) reachPRM.numCores

        pass
