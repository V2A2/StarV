"""
load module, to load existing networks for testing/evaluation
Dung Tran, 9/12/2022
"""
import os
from scipy.io import loadmat
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.net.network import NeuralNetwork
from StarV.plant.lode import LODE
from StarV.plant.dlode import DLODE
import numpy as np
# import torch
import math


def load_ABES():

    """
    Load the advanced emergency braking system

    ref paper: Tran et al., "Safety Verification for Learning-enabled 
              Cyber-Physical Systems with Reinforcement Learning Control", EMSOFT 2019
    """

    cur_path = os.path.dirname(__file__)
    controller_path = cur_path + '/data/nets/AEBS/controller.mat'
    transform_path = cur_path + '/data/nets/AEBS/transform.mat'
    controller_contents = loadmat(controller_path)
    transform_contents = loadmat(transform_path)

    control_W = controller_contents['W']
    control_b = controller_contents['b']
    transform_W = transform_contents['W']
    transform_b = transform_contents['b']

    control_layers = []
    transform_layers = []

    # controller
    FC1 = fullyConnectedLayer(control_W[0, 0], control_b[0, 0])
    FC2 = fullyConnectedLayer(control_W[0, 1], control_b[0, 1])
    FC3 = fullyConnectedLayer(control_W[0, 2], control_b[0, 2])
    RL1 = ReLULayer()
    RL2 = ReLULayer()
    CLayers = [FC1, RL1, FC2, RL2, FC3]
    controller = NeuralNetwork(CLayers, net_type='controller')

    # transformer
    TFC1 = fullyConnectedLayer(transform_W[0, 0], transform_b[0, 0])
    TFC2 = fullyConnectedLayer(transform_W[0, 1], transform_b[0, 1])
    TFC3 = fullyConnectedLayer(transform_W[0, 2], transform_b[0, 2])
    TRL1 = ReLULayer()
    TRL2 = ReLULayer()

    TLayers = [TFC1, TRL1, TFC2, TRL2, TFC3]
    transformer = NeuralNetwork(TLayers, net_types='transformer')


    # normalization
    norm_mat = np.array([[1/250., 0., 0.], [0., 3.6/120., 0.], [0.,  0., 1/20.]])

    # control signal scale
    scale_mat = np.array([-15.0*120/3.6, 15.0*120/3.6])

    # plant matrices

    A = np.array([[1., -1/15., 0], [0., 1., 0.], [0., 0., 0.]])
    B = np.array([[0.], [1.0/15], [1.]])
    C = np.eye(3)
    
    plant = DLODE(A, B, C)

    # initial conditions

    lb1 = np.array([97., 25.2, 0.])
    ub1 = np.array([97.5, 25.5, 0.])

    lb2 = np.array([90., 27., 0.])
    ub2 = np.array([90.5, 27.2, 0.])

    lb3 = np.array([60., 30.2, 0.])
    ub3 = np.array([60.5, 30.4, 0.])

    lb4 = np.array([50., 32., 0.])
    ub4 = np.array([50.5, 32.2, 0.])

    
    

    

def load_2017_IEEE_TNNLS():
    """Load network from the IEEE TNNLS 2017 paper
       refs: https://arxiv.org/abs/1712.08163
    """
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/2017_IEEE_TNNLS.mat'
    mat_contents = loadmat(cur_path)
    W = mat_contents['W']
    b = mat_contents['b']
    layers = []
    for i in range(0, b.shape[1]-1):
        Wi = W[0, i]
        bi = b[0, i]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = ReLULayer()
        layers.append(L2)

    Wi = W[0, b.shape[1]-1]
    bi = b[0, b.shape[1]-1]
    bi = bi.reshape(bi.shape[0],)
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)
    net = NeuralNetwork(layers, net_type='ffnn_2017_IEEE_TNNLS')

    return net

def load_ACASXU(x, y, spec_id, actv='relu'):
    """Load ACASXU networks
       Args:
           @ network id (x,y)
           @ specification_id: 1, 2, 3, or 4

       Return
           @net: network
           @lb: normalized lower bound of inputs
           @ub: normalized upper bound of inputs
           @unsafe_mat: unsafe matrix, i.e., unsafe region of the outputs
           @unsafe_vec: unsafe vector: 
           ***unsafe regrion: (unsafe_mat * y <= unsafe_vec)
    """

    net_name = 'ACASXU_run2a_{}_{}_batch_2000'.format(x,y)
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/ACASXU/' + net_name
    mat_contents = loadmat(cur_path)
    W = mat_contents['W']
    b = mat_contents['b']
    means_for_scaling = mat_contents['means_for_scaling']
    means_for_scaling = means_for_scaling.reshape(6,)
    range_for_scaling = mat_contents['range_for_scaling']
    range_for_scaling = range_for_scaling.reshape(6,)
    
    layers = []
    for i in range(0, b.shape[1]-1):
        Wi = W[0, i]
        bi = b[0, i]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = ReLULayer()
        layers.append(L2)

    Wi = W[0, b.shape[1]-1]
    bi = b[0, b.shape[1]-1]
    bi = bi.reshape(bi.shape[0],)
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)
    net = NeuralNetwork(layers, net_type='ffnn_ACASXU_{}_{}'.format(x,y))

    # Get input constraints and specs (unsafe constraints on the outputs)
    # paper: https://arxiv.org/pdf/1702.01135.pdf
    
    if spec_id == 1 or spec_id == 2:

        # Input Constraints:
        #      55947.69 <= i1(\rho) <= 60760
        #
        # Input Constraints
        # 55947.69 <= i1(\rho) <= 60760,
        # -3.14 <= i2 (\theta) <= 3.14,
        # -3.14 <= i3 (\shi) <= -3.14
        #  1145 <= i4 (\v_own) <= 1200, 
        #  0 <= i5 (\v_in) <= 60
        lb = np.array([55947.69, -3.14, -3.14, 1145, 0])
        ub = np.array([60760, 3.14, 3.14, 1200, 60])

        # Output constraints (specifications)
        if spec_id == 1:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # verify safety: COC <= 1500 or x1 <= 1500 after normalization
        
            # safe region before normalization
            # x1' <= (1500 - 7.5189)/373.9499 = 3.9911, 373.9499 is from range_for_scaling[5]
            unsafe_mat = np.array([-1, 0, 0, 0, 0])
            unsafe_vec = np.array([-3.9911])  # unsafe region x1' > 3.9911
        if spec_id == 2:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # safety property: COC is not the maximal score
            # unsafe region: COC is the maximal score: x1 >= x2; x1 >= x3; x1 >= x4, x1 >= x5
            unsafe_mat = np.array([[-1.0, 1.0, 0., 0., 0.], [-1., 0., 1., 0., 0.], [-1., 0., 0., 1., 0.], [-1., 0., 0., 0., 1.,]])
            unsafe_vec = np.array([0., 0., 0., 0.])
        

    elif spec_id == 3:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # 3.1 <= i3 (\shi) <= 3.14
        # 980 <= i4 (\v_own) <= 1200, 
        # 960 <= i5 (\v_in) <= 1200
        # ****NOTE There was a slight mismatch of the ranges of
        # this i5 input for the conference paper, FM2019 "Star-based Reachability of DNNs"
        lb = np.array([1500, -0.06, 3.1, 980, 960])
        ub = np.array([1800, 0.06, 3.14, 1200, 1200])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 4:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # (\shi) = 0
        # 1000 <= i4 (\v_own) <= 1200, 
        # 700 <= i5 (\v_in) <= 800
        lb = np.array([1500, -0.06, 0, 1000, 700])
        ub = np.array([1800, 0.06, 0, 1200, 800])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    else:
        raise Exception('Invalide Specification ID')

    # Normalize input
    for i in range(0, 5):
        lb[i] = (lb[i] - means_for_scaling[i])/range_for_scaling[i]
        ub[i] = (ub[i] - means_for_scaling[i])/range_for_scaling[i]

   
    return net, lb, ub, unsafe_mat, unsafe_vec


def load_DRL(net_id, prob_id):
    
    """
    Load unsafe networks trained for controlling Rocket landing 
    paper: Xiaodong Yang, Neural Network Repair with Reachability Analysis, FORMATS 2022
    id: = 0, 1 or 2
    tool: veritex: https://github.com/Shaddadi/veritex/tree/master/examples/DRL/nets
    """
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/DRL/unsafe_agent' + str(net_id) + '.pt'
    net0 = torch.load(cur_path)
    net0.eval()
    layers = []
    for i in range(11):
        if (i % 2) == 0:
            W = net0[i].weight.detach().numpy()
            b = net0[i].bias.detach().numpy()
            L = fullyConnectedLayer(W, b)
        else:
            L = ReLULayer()

        layers.append(L)

    net = NeuralNetwork(layers, net_type='ffnn_DRL_{}_prob_{}'.format(net_id, prob_id))

    # property: https://github.com/Shaddadi/veritex/blob/master/examples/DRL/repair/agent_properties.py

    # This is the original input
    lb_p0 = np.array([-0.2, 0.02, -0.5, -1.0, -20 * math.pi / 180, -0.2, 0.0, 0.0, 0.0, -1.0, -15 * math.pi / 180])
    ub_p0 = np.array([0.2, 0.5, 0.5, 1.0, -6 * math.pi / 180, -0.0, 0.0, 0.0, 1.0, 0.0, 0 * math.pi / 180])
    lb_p1 = np.array([-0.2, 0.02, -0.5, -1.0, 6 * math.pi / 180, 0.0, 0.0, 0.0, 0.0, 0.0, 0 * math.pi / 180])
    ub_p1 = np.array([0.2, 0.5, 0.5, 1.0, 20 * math.pi / 180, 0.2, 0.0, 0.0, 1.0, 1.0, 15 * math.pi / 180])

    # This is a smaller input set # take about 505.389 seconds to verify, can find 28 counterexamples set but probability is ~ 0
  
    # lb_p0 = np.array([-0.02, 0.02, -0.05, -1.0, -20 * math.pi / 180, -0.2, 0.0, 0.0, 0.0, -1.0, -15 * math.pi / 180])
    # ub_p0 = np.array([0.02, 0.5, 0.05, 1.0, -6 * math.pi / 180, -0.0, 0.0, 0.0, 1.0, 0.0, 0 * math.pi / 180])
    # lb_p1 = np.array([-0.02, 0.02, -0.05, -1.0, 6 * math.pi / 180, 0.0, 0.0, 0.0, 0.0, 0.0, 0 * math.pi / 180])
    # ub_p1 = np.array([0.02, 0.5, 0.05, 1.0, 20 * math.pi / 180, 0.2, 0.0, 0.0, 1.0, 1.0, 15 * math.pi / 180])

    # This is a smaller input set: y = [0.4, 0.5], take about 80 seconds to verify, can find 28 counterexamples set but probability is ~ 0
    # lb_p0 = np.array([-0.02, 0.4, -0.05, -1.0, -20 * math.pi / 180, -0.2, 0.0, 0.0, 0.0, -1.0, -15 * math.pi / 180])
    # ub_p0 = np.array([0.02, 0.5, 0.05, 1.0, -6 * math.pi / 180, -0.0, 0.0, 0.0, 1.0, 0.0, 0 * math.pi / 180])
    # lb_p1 = np.array([-0.02, 0.4, -0.05, -1.0, 6 * math.pi / 180, 0.0, 0.0, 0.0, 0.0, 0.0, 0 * math.pi / 180])
    # ub_p1 = np.array([0.02, 0.5, 0.05, 1.0, 20 * math.pi / 180, 0.2, 0.0, 0.0, 1.0, 1.0, 15 * math.pi / 180])

    A_unsafe0 = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    d_unsafe0 = np.array([0.0, 0.0])
    A_unsafe1 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    d_unsafe1 = np.array([0.0, 0.0])

    if prob_id == 1:
        lb = lb_p0
        ub = ub_p0
        unsafe_mat = A_unsafe0
        unsafe_vec = d_unsafe0
    elif prob_id == 2:
        lb = lb_p1
        ub = ub_p1
        unsafe_mat = A_unsafe1
        unsafe_vec = d_unsafe1
    else:
        raise Exception('Invalid property id, shoule be 1 or 2')
        
    return net, lb, ub, unsafe_mat, unsafe_vec

def load_tiny_network():
    """Load a tiny 2-inputs 2-output network as a running example"""

    layers = []
    W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b1 = np.array([0.5, 1.0, -0.5])
    L1 = fullyConnectedLayer(W1, b1)
    L2 = ReLULayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    L3 = fullyConnectedLayer(W2,b2)
    
    layers.append(L1)
    layers.append(L2)
    layers.append(L3)

    net = NeuralNetwork(layers, 'ffnn_tiny_network')

    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])

    unsafe_mat = np.array([[1.0, 0]])
    unsafe_vec = np.array([-2.0])

    return net, lb, ub, unsafe_mat, unsafe_vec

def load_harmonic_oscillator_model():
    """Load LODE harmonic oscillator model"""

    # model: x' = y + u1, y' = -x + u2
    # input range: u1, u2 is in [-0.5, 0.5]
    # initial conditions: x in [-6, -5], y in [0, 1]
    # ref: Bak2017CAV: Simulation-Equivalent Reachability of Large Linear Systems with Inputs
    
    A = np.array([[0., 1.], [-1., 0]])  # system matrix
    B = np.eye(2)

    lb = np.array([-6., 0.])
    ub = np.array([-5., 1.])

    input_lb = np.array([-0.5, -0.5])
    input_ub = np.array([0.5, 0.5])

    plant = LODE(A, B)

    return plant, lb, ub, input_lb, input_ub


def load_building_model():
    """Load LODE building model"""

    pass

def load_iss_model():
    """Load LODE International State Space Model"""

    pass

def load_helicopter_model():
    """Load LODE helicopter model"""

    pass

def load_MNA5_model():
    """Load LODE MNA5 model"""

    pass
