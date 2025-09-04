"""
load module, to load existing networks for testing/evaluation
Dung Tran, 9/12/2022

Update: 12/20/2024 (Sung Woo Choi, merging)
"""
import os
import numpy as np
import torch
import math
import copy
import onnx
import onnx2pytorch
import csv

from scipy.io import loadmat
from scipy.sparse import csc_matrix
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.LogSigLayer import LogSigLayer
from StarV.layer.TanSigLayer import TanSigLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.LSTMLayer import LSTMLayer
from StarV.layer.GRULayer import GRULayer
from StarV.layer.Conv2DLayer import Conv2DLayer
from StarV.layer.ConvTranspose2DLayer import ConvTranspose2DLayer       
from StarV.layer.AvgPool2DLayer import AvgPool2DLayer
from StarV.layer.BatchNorm2DLayer import BatchNorm2DLayer
from StarV.layer.MaxPool2DLayer import MaxPool2DLayer
from StarV.layer.FlattenLayer import FlattenLayer
from StarV.layer.PixelClassificationLayer import PixelClassificationLayer
from StarV.layer.MixedActivationLayer import MixedActivationLayer
from StarV.net.network import NeuralNetwork
from StarV.nncs.nncs import NNCS
from StarV.plant.lode import LODE, DLODE
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_, _OR_
from StarV.spec.dProbStarTL import DynamicFormula

def convert_to_numpy(matrix):
    if matrix is None:
        return None
    elif isinstance(matrix, csc_matrix):
        return matrix.toarray()
    else:
        return matrix

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
        L1 = FullyConnectedLayer([Wi, bi])
        layers.append(L1)
        L2 = ReLULayer()
        layers.append(L2)

    Wi = W[0, b.shape[1]-1]
    bi = b[0, b.shape[1]-1]
    bi = bi.reshape(bi.shape[0],)
    L1 = FullyConnectedLayer([Wi, bi])
    layers.append(L1)
    net = NeuralNetwork(layers, net_type='ffnn_2017_IEEE_TNNLS')

    return net

def load_ACASXU(x,y,spec_id):
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
        L1 = FullyConnectedLayer([Wi, bi])
        layers.append(L1)
        L2 = ReLULayer()
        layers.append(L2)

    Wi = W[0, b.shape[1]-1]
    bi = b[0, b.shape[1]-1]
    bi = bi.reshape(bi.shape[0],)
    L1 = FullyConnectedLayer([Wi, bi])
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
            L = FullyConnectedLayer([W, b])
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
    L1 = FullyConnectedLayer([W1, b1])
    L2 = ReLULayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    L3 = FullyConnectedLayer([W2, b2])
    
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


def load_acc_model(netname='controller_5_20', plant='linear', spec_ids=None, initSet_id=None, T=None, t=None):
    'load advanced neural network-controlled adaptive cruise control system'

    # This is from the paper:
    # NNV: A Verification Tool for Deep Neural Networks and Learning-enabled Cyber-Physical Systems,
    # Tran et al, CAV 2020

    # System model
    # a_lead = -5 (m^2/s)
    # x1 = lead_car position
    # x2 = lead_car velocity
    # x3 = lead_car internal state
    # x4 = ego_car position
    # x5 = ego_car velocity
    # x6 = ego_car internal state

    # lead car dynamics
    # dx1 = x2
    # dx2 = x3
    # dx3 = -2x3 + 2*a_lead - mu*x2^2 (mu = 0 for linear case)
    
    # ego car dynamics
    # dx4 = x5
    # dx5 = x6
    # dx6 = -2x6 + 2*a_ego - mu*x5^2 (mu = 0 for linear case)

    # let x7 = -2x3 + 2*a_lead -> x7(0) = -2x3(0) + 2*a_lead (initial condition consistency)
    # dx7 = -2dx3
    # dx3 = x7 and dx7 = -2dx3 = -2x7 (There may be a mistake in the code for CAV2020???)

    # controller net option:
    # 1) controller_3_20, 2) controller_5_20, 3) controller_7_20, 4) controller_10_20

    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/ACC/' + netname
    mat_contents = loadmat(cur_path)
    W = mat_contents['W']
    b = mat_contents['b']

    n = W.shape[1]
    layers = []
    for i in range(0,n-1):
        Wi = W[0,i]
        bi = b[i,0]
        bi = bi.reshape(bi.shape[0],)
        L1 = FullyConnectedLayer([Wi, bi])
        L2 = ReLULayer()
        layers.append(L1)
        layers.append(L2)


    bi = b[n-1,0]
    bi = bi.reshape(bi.shape[0],)
    L1 = FullyConnectedLayer([W[0,n-1], bi])
    layers.append(L1)
    
    net = NeuralNetwork(layers, netname)
    # net.info()

    if plant=='linear':

        A = np.array([[0., 1., 0., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 1.],
                      [0., 0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., -2., 0.],
                      [0., 0., 0., 0., 0., 0., -2.]])
        B = np.array([[0.], [0.], [0.], [0.], [0.], [2.], [0.]])
        C = np.array([[1., 0., 0., -1., 0., 0., 0.],
                      [0., 1., 0., 0., -1., 0., 0.],
                      [0., 0., 0., 0., 1., 0., 0.]])
        # feedbacks:
        # 1) relative distance: x1 - x4
        # 2) relative velocity: x2 - x5
        # 3) longtitudinal velocity: x5

        D = np.array([[0.], [0.], [0.]])

        plant_model = LODE(A, B, C, D)
        dplant = plant_model.toDLODE(0.1)  # dt = 0.1
        
    else:
        raise RuntimeError("Unknown option: only have linear model for ACC for now")


    sys = NNCS(net, dplant, type='DLNNCS')
    #sys.info()

    # reference inputs
    refInputs = np.array([30., 1.4])

    # input sets (multiple input set - 6 individual depending on v_lead_0)
    x_lead_0 = [90., 92.]
    v_lead_0 = [[29., 30.], [28., 29.], [27., 28.], [26., 27.], [25., 26.], [20., 21.]]
    acc_lead_0 = [0., 0.]
    x_ego_0 = [30., 31.,]
    v_ego_0 = [30., 30.5]
    acc_ego_0 = [0., 0.]
    a_lead = -5.0
    x7_0 = [2*a_lead, 2*a_lead]

    initSets = []
    for i in range(0, 6):
        v_lead_0_i = v_lead_0[i]
        lb = np.array([x_lead_0[0], v_lead_0_i[0], acc_lead_0[0], x_ego_0[0], v_ego_0[0], acc_ego_0[0], x7_0[0]])
        ub = np.array([x_lead_0[1], v_lead_0_i[1], acc_lead_0[1], x_ego_0[1], v_ego_0[1], acc_ego_0[1], x7_0[1]])
        S = Star(lb, ub)
        mu = 0.5*(S.pred_lb + S.pred_ub)
        a = 2.5 # coefficience to adjust the distribution
        sig = (mu - S.pred_lb)/a
        Sig = np.diag(np.square(sig))
        I1 = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
        initSets.append(I1)


    # unsafe constraints
    # safety property: actual distance > alpha * safe distance <=> d = (x1 - x4) > alpha * d_safe = alpha * (1.4 * v_ego + 10)
    # unsafe region: x1 - x4 <= alpha * (1.4 * v_ego + 10)

    alpha = 1.0
    unsafe_mat = np.array([[1.0, 0., 0., -1., -alpha*1.4, 0., 0.]])
    unsafe_vec = np.array([alpha*10.0])

    if spec_ids is None: # return systems with unsafe properties
        
        return sys, initSets, refInputs, unsafe_mat, unsafe_vec

    else: # return system with temporal specifications

        # Temporal Specifications

        assert T is not None, 'error: T should be > 0'
        assert t is not None, 'error: t should be > 0'

        EV0T = _EVENTUALLY_(0,T)
        EV0t = _EVENTUALLY_(0,t)
        AND = _AND_()
        OR = _OR_()
        lb = _LeftBracket_()
        rb = _RightBracket_()
        AW0T = _ALWAYS_(0,T)
        AW0t = _ALWAYS_(0,t)

        # phi1 : eventually_[0, T](x_lead - x_ego <= D_safe = 10 + 1.4 v_ego) : A2x <= b2
        A1 = np.array([1., 0., 0., -1., -1.4, 0., 0.])
        b1 = np.array([10.])
        P1 = AtomicPredicate(A1,b1)
        phi1 = Formula([EV0T, lb, P1, rb])

        #phi1c always_[0, T] (x_lead - x_ego >= D_safe = 10 + 1.4 v_ego): A1x <= b1
        P1c = AtomicPredicate(-A1,-b1)
        phi1c = Formula([AW0T, lb, P1c, rb])

        # phi2 : eventually_[0, T](v_lead <= v_lead(0)_min - 0.1 OR v_ego <= v_ego(0)_min - 0.1): A3 <= b3

        # phi2 IS DIFFERENT FOR DIFFERENT INITIAL CONDITION

        A21 = np.array([0., 1., 0., 0., 0, 0., 0.])
        b21 = np.array([min(v_lead_0[initSet_id]) - 0.1])
        P21 = AtomicPredicate(A21,b21)

        A22 = np.array([0., 0., 0., 0., 1., 0., 0.])
        b22 = np.array([min(v_ego_0) - 0.1])
        P22 = AtomicPredicate(A22,b22)

        phi2= Formula([EV0T, lb, P21, OR, P22, rb]) #

        P21c = AtomicPredicate(-A21, -b21)
        P22c = AtomicPredicate(-A22, -b22)

        # complement properties
        phi2c = Formula([AW0T, lb, P21c, AND, P22c, rb])

        # phi3 : eventually_[0, T](v_lead <= v_lead(0)_min - 0.1  AND eventually_[0, 10](v_ego <= v_ego(0)_min - 0.1))

        # phi3 is different for different initial condition


        A31 = np.array([0., 1., 0., 0., 0, 0., 0.])
        b31 = np.array([min(v_lead_0[initSet_id]) - 0.1])
        P31 = AtomicPredicate(A31,b31)

        A32 = np.array([0., 0., 0., 0., 1., 0., 0.])
        b32 = np.array([min(v_ego_0) - 0.1])
        P32 = AtomicPredicate(A32,b32)

        phi3 = Formula([EV0T, lb, P31, AND, lb, EV0t, P32, rb, rb])

        # phi4 : always_[0, T](x_lead - x_ego <= D_safe -> eventually_[0,10](x_lead - x_ego >= D_safe))
        # equivalent to : always_[0, T](x_lead - x_ego >= D_safe OR eventually_[0,t](x_lead - x_ego >= D_safe))
        # P(phi5) = 1 - P(phi5')
        # P(always(A or B)) = 1 - P(eventually (not A AND not B))
        # not B = not eventually C = always not C

        # phi4' = eventually_[0,T](x_lead-x_ego <= D_safe AND always_[0,t](x_lead - x_ego <= D_safe))

        A41 = np.array([1., 0., 0., -1., -1.4, 0., 0.])
        b41 = np.array([10.])
        P41 = AtomicPredicate(A41,b41)
        P42 = AtomicPredicate(-A41,-b41)

        phi4 = Formula([AW0T, lb, P41, OR, lb, EV0t, P42 , rb, rb])

        phi4c = Formula([EV0T, lb, P41, AND, lb, AW0t, P41, rb, rb])

        phi = [phi1, phi1c, phi2, phi2c, phi3, phi4, phi4c]

        assert isinstance(spec_ids, list), 'Error: spec_ids should be a list'
        id_max = max(spec_ids)
        id_min = min(spec_ids)

        if id_min < 0 or id_max > 6:
            raise RuntimeError('Invalid spec_ids, id should be between 0 and 4')

        phi_v = []
        for id in spec_ids:
            phi_v.append(phi[id])
        
        return sys, phi_v, initSets[initSet_id], refInputs


def load_acc_trapezius(t=10):
    # load_Trapezius_network

    net_name = f'phi3_acc_network_t{t}'
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + f'/data/nets/ACC/Trapezius/{net_name}.mat'
    mat_file = loadmat(cur_path)
    Net = mat_file['Net']
    W = Net[0][0][0].ravel()
    b = Net[0][0][1].ravel()
    act_fun = Net[0][0][2].ravel() # containts activation list of each neurons
    
    n_weight = len(W)
    n_act_fun = len(act_fun)
    
    
    layers = []
    for i in range(n_weight):
        layers.append(FullyConnectedLayer([W[i].toarray(), b[i].toarray().ravel()]))
        if i < n_act_fun:
            layers.append(MixedActivationLayer(act_fun[i]))

    return NeuralNetwork(layers, net_type=net_name)



def load_acc_trapezius_model(netname='controller_3_20', plant='linear', spec_ids=None, T=None, t=None, a_lead=-2):
    'load advanced neural network-controlled adaptive cruise control system'

    # This is from the paper:
    # NNV: A Verification Tool for Deep Neural Networks and Learning-enabled Cyber-Physical Systems,
    # Tran et al, CAV 2020

    # System model
    # a_lead = -5 (m^2/s)
    # x1 = lead_car position
    # x2 = lead_car velocity
    # x3 = lead_car internal state
    # x4 = ego_car position
    # x5 = ego_car velocity
    # x6 = ego_car internal state

    # lead car dynamics
    # dx1 = x2
    # dx2 = x3
    # dx3 = -2x3 + 2*a_lead - mu*x2^2 (mu = 0 for linear case)
    
    # ego car dynamics
    # dx4 = x5
    # dx5 = x6
    # dx6 = -2x6 + 2*a_ego - mu*x5^2 (mu = 0 for linear case)

    # let x7 = -2x3 + 2*a_lead -> x7(0) = -2x3(0) + 2*a_lead (initial condition consistency)
    # dx7 = -2dx3
    # dx3 = x7 and dx7 = -2dx3 = -2x7 (There may be a mistake in the code for CAV2020???)

    # controller net option:
    # 1) controller_3_20, 2) controller_5_20, 3) controller_7_20, 4) controller_10_20

    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/ACC/' + netname
    mat_contents = loadmat(cur_path)
    W = mat_contents['W']
    b = mat_contents['b']

    n = W.shape[1]
    layers = []
    for i in range(0,n-1):
        Wi = W[0,i]
        bi = b[i,0]
        bi = bi.reshape(bi.shape[0],)
        L1 = FullyConnectedLayer([Wi, bi])
        L2 = ReLULayer()
        layers.append(L1)
        layers.append(L2)


    bi = b[n-1,0]
    bi = bi.reshape(bi.shape[0],)
    L1 = FullyConnectedLayer([W[0,n-1], bi])
    layers.append(L1)
    
    net = NeuralNetwork(layers, netname)
    net.info()

    if plant=='linear':

        A = np.array([[0., 1., 0., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0., 0.],
                      [0., 0.,-2., 0., 0., 0., 1.],
                      [0., 0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., -2, 0.],
                      [0., 0., 0., 0., 0., 0., -2.]])
        B = np.array([[0.], [0.], [0.], [0.], [0.], [2.], [0.]])
        C = np.array([[1., 0., 0., -1., 0., 0., 0.],
                      [0., 1., 0., 0., -1., 0., 0.],
                      [0., 0., 0., 0., 1., 0., 0.]])
        
        # feedbacks:
        # 1) relative distance: x1 - x4
        # 2) relative velocity: x2 - x5
        # 3) longtitudinal velocity: x5 

        D = np.array([[0.], [0.], [0.]])

        plant_model = LODE(A, B, C, D)
        dplant = plant_model.toDLODE(0.1)  # dt = 0.1
        
    else:
        raise RuntimeError("Unknown option: only have linear model for ACC for now")


    sys = NNCS(net, dplant, type='DLNNCS')
    sys.info()

    # reference inputs
    refInputs = np.array([30., 1.4])

    # input sets (multiple input set - 6 individual depending on v_lead_0)
    x_lead_0 = [90., 110.]
    v_lead_0 = [32.0, 32.2]
    acc_lead_0 = [0., 0.]

    x_ego_0 = [10., 11.]
    v_ego_0 = [30., 30.2]
    acc_ego_0 = [0., 0.]
    x7_0 = [2*a_lead, 2*a_lead]

    # initSets = []
    # for i in range(0, 6):
    #     v_lead_0_i = v_lead_0[i]
    #     lb = np.array([x_lead_0[0], v_lead_0_i[0], acc_lead_0[0], x_ego_0[0], v_ego_0[0], acc_ego_0[0], x7_0[0]])
    #     ub = np.array([x_lead_0[1], v_lead_0_i[1], acc_lead_0[1], x_ego_0[1], v_ego_0[1], acc_ego_0[1], x7_0[1]])
    #     S = Star(lb, ub)
    #     mu = 0.5*(S.pred_lb + S.pred_ub)
    #     a = 2.5 # coefficience to adjust the distribution
    #     sig = (mu - S.pred_lb)/a
    #     Sig = np.diag(np.square(sig))
    #     I1 = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
    #     initSets.append(I1)


    lb = np.array([x_lead_0[0], v_lead_0[0], acc_lead_0[0], x_ego_0[0], v_ego_0[0], acc_ego_0[0], x7_0[0]])
    ub = np.array([x_lead_0[1], v_lead_0[1], acc_lead_0[1], x_ego_0[1], v_ego_0[1], acc_ego_0[1], x7_0[1]])
    S = Star(lb, ub)
    mu = 0.5*(S.pred_lb + S.pred_ub)
    a = 2.5 # coefficience to adjust the distribution
    sig = (mu - S.pred_lb)/a
    Sig = np.diag(np.square(sig))
    initSet = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)

    # unsafe constraints
    # safety property: actual distance > alpha * safe distance <=> d = (x1 - x4) > alpha * d_safe = alpha * (1.4 * v_ego + 10)
    # unsafe region: x1 - x4 <= alpha * (1.4 * v_ego + 10)

    alpha = 1.0
    unsafe_mat = np.array([[1.0, 0., 0., -1., -alpha*1.4, 0., 0.]])
    unsafe_vec = np.array([alpha*10.0])

    if spec_ids is None: # return systems with unsafe properties
        
        return sys, initSet, refInputs, unsafe_mat, unsafe_vec

    else: # return system with temporal specifications

        # Temporal Specifications

        assert T is not None, 'error: T should be > 0'
        assert t is not None, 'error: t should be > 0'

        EV0T = _EVENTUALLY_(0,T)
        EV0t = _EVENTUALLY_(0,t)
        AND = _AND_()
        OR = _OR_()
        lb = _LeftBracket_()
        rb = _RightBracket_()
        AW0T = _ALWAYS_(0,T)
        AW0t = _ALWAYS_(0,t)

        # phi1 : eventually_[0, T](x_lead - x_ego <= D_safe = 10 + 1.4 v_ego) : A2x <= b2
        A1 = np.array([1., 0., 0., -1., -1.4, 0., 0.])
        b1 = np.array([10.])
        P1 = AtomicPredicate(A1,b1)
        phi1 = Formula([EV0T, lb, P1, rb])

        #phi1c always_[0, T] (x_lead - x_ego >= D_safe = 10 + 1.4 v_ego): A1x <= b1
        P1c = AtomicPredicate(-A1,-b1)
        phi1c = Formula([AW0T, lb, P1c, rb])

        # phi2 : eventually_[0, T](v_lead <= v_lead(0)_min - 0.1 OR v_ego <= v_ego(0)_min - 0.1): A3 <= b3

        # phi2 IS DIFFERENT FOR DIFFERENT INITIAL CONDITION

        A21 = np.array([0., 1., 0., 0., 0, 0., 0.])
        b21 = np.array([min(v_lead_0) - 0.1])
        P21 = AtomicPredicate(A21,b21)

        A22 = np.array([0., 0., 0., 0., 1., 0., 0.])
        b22 = np.array([min(v_ego_0) - 0.1])
        P22 = AtomicPredicate(A22,b22)

        phi2= Formula([EV0T, lb, P21, OR, P22, rb]) #

        P21c = AtomicPredicate(-A21, -b21)
        P22c = AtomicPredicate(-A22, -b22)

        # complement properties
        phi2c = Formula([AW0T, lb, P21c, AND, P22c, rb])

        # phi3 : eventually_[0, T](v_lead <= v_lead(0)_min - 0.1  AND eventually_[0, 10](v_ego <= v_ego(0)_min - 0.1))

        # phi3 is different for different initial condition


        A31 = np.array([0., 1., 0., 0., 0, 0., 0.])
        b31 = np.array([min(v_lead_0) - 0.1])
        P31 = AtomicPredicate(A31,b31)

        A32 = np.array([0., 0., 0., 0., 1., 0., 0.])
        b32 = np.array([min(v_ego_0) - 0.1])
        P32 = AtomicPredicate(A32,b32)

        phi3 = Formula([EV0T, lb, P31, AND, lb, EV0t, P32, rb, rb])


        # phi4 : always_[0, T](x_lead - x_ego <= D_safe -> eventually_[0,10](x_lead - x_ego >= D_safe))
        # equivalent to : always_[0, T](x_lead - x_ego >= D_safe OR eventually_[0,t](x_lead - x_ego >= D_safe))
        # P(phi4) = 1 - P(phi4')
        # P(always(A or B)) = 1 - P(eventually (not A AND not B))
        # not B = not eventually C = always not C

        # phi4' = eventually_[0,T](x_lead-x_ego <= D_safe AND always_[0,t](x_lead - x_ego <= D_safe))

        A41 = np.array([1., 0., 0., -1., -1.4, 0., 0.])
        b41 = np.array([10.])
        P41 = AtomicPredicate(A41,b41)
        P42 = AtomicPredicate(-A41,-b41)

        phi4 = Formula([AW0T, lb, P41, OR, lb, EV0t, P42 , rb, rb])

        phi4c = Formula([EV0T, lb, P41, AND, lb, AW0t, P41, rb, rb])


        # phi5 : always_[0, T](x_lead - x_ego <= D_safe -> eventually_[0,10](x_lead - x_ego >= D*_safe))
        # equivalent to : always_[0, T](x_lead - x_ego >= D_safe OR eventually_[0,t](x_lead - x_ego >= D*_safe))
        # P(phi5) = 1 - P(phi5')
        # P(always(A or B)) = 1 - P(eventually (not A AND not B))
        # not B = not eventually C = always not C


        # ICCPS: 
        # phi5: G_[0,50](x_lead - x_ego < D_safe -> F_[0, 3](x_lead - x_eg > D*_safe))

        # phi5': eventually_[0,T](x_lead-x_ego <= D_safe AND always_[0,t](x_lead - x_ego <= D*_safe))

        A51 = np.array([1., 0., 0., -1., -1.4, 0., 0.])
        b51 = np.array([10.])
        P51 = AtomicPredicate(A51,b51)

        A52 = np.array([1., 0., 0., -1., -1.4, 0., 0.])
        b52 = np.array([12.])
        P52 = AtomicPredicate(-A52,-b52)

        phi5 = Formula([AW0T, lb, P51, OR, lb, EV0t, P52 , rb, rb])

        phi5c = Formula([EV0T, lb, P51, AND, lb, AW0t, P51, rb, rb])

        phi = [phi1, phi1c, phi2, phi2c, phi3, phi4, phi4c, phi5, phi5c]

        assert isinstance(spec_ids, list), 'Error: spec_ids should be a list'
        id_max = max(spec_ids)
        id_min = min(spec_ids)

        if id_min < 0 or id_max > 8:
            raise RuntimeError('Invalid spec_ids, id should be between 0 and 8')

        phi_v = []
        for id in spec_ids:
            phi_v.append(phi[id])
        
        return sys, phi_v, initSet, refInputs
    

def load_AEBS_model():
    """load avanced emergency braking system
    
    This case study is from this paper: 

    Tran et al. "Safety Verification of Learning-enabled Cyber-Physical Systems with Reinforcement Control", EMSOFT 2019
    
    """

    # load transformer network
    # load reinforment controller network
    # load plant dynamics A, B, C
    # load initial conditions

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
    FC1 = FullyConnectedLayer([control_W[0, 0], control_b[0, 0].reshape(control_b[0, 0].shape[1], )])
    FC2 = FullyConnectedLayer([control_W[0, 1], control_b[0, 1].reshape(control_b[0, 1].shape[1], )])
    FC3 = FullyConnectedLayer([control_W[0, 2], control_b[0, 2].reshape(control_b[0, 2].shape[1], )])
    RL1 = ReLULayer()
    RL2 = ReLULayer()
    SL1 = SatLinLayer()
    CLayers = [FC1, RL1, FC2, RL2, FC3, SL1]
    controller = NeuralNetwork(CLayers, net_type='controller')

    # transformer
    TFC1 = FullyConnectedLayer([transform_W[0, 0], transform_b[0, 0].reshape(transform_b[0, 0].shape[1], )])
    TFC2 = FullyConnectedLayer([transform_W[0, 1], transform_b[0, 1].reshape(transform_b[0, 1].shape[1], )])
    TFC3 = FullyConnectedLayer([transform_W[0, 2], transform_b[0, 2].reshape(transform_b[0, 2].shape[1], )])
    TRL1 = ReLULayer()
    TRL2 = ReLULayer()
    
    TLayers = [TFC1, TRL1, TFC2, TRL2, TFC3]
    transformer = NeuralNetwork(TLayers, net_type='transformer')


    # normalization
    norm_mat = np.array([[1/250., 0., 0.], [0., 3.6/120., 0.], [0.,  0., 1/20.]])

    # control signal scale
    scale_mat = np.array([[-15.0*120/3.6, 15.0*120/3.6]])

    # plant matrices

    A = np.array([[1., -1/15., 0], [0., 1., 0.], [0., 0., 0.]])
    B = np.array([[0.], [1/15], [1.]])
    C = np.eye(3)
    
    plant = DLODE(A, B, C)

    # initial conditions

    d_lb = [97., 90., 48., 5.0]
    d_ub = [97.5, 90.5, 48.5, 5.2]
    v_lb = [25.2, 27., 30.2, 1.0]
    v_ub = [25.5, 27.2, 30.4, 1.2]
    
    initSets = []

    a = 2.5 # coefficience to adjust the distribution
    
    for i in range(0, len(d_lb)):
        lb = np.array([d_lb[i], v_lb[i], 0.0])
        ub = np.array([d_ub[i], v_ub[i], 0.0])
        S = Star(lb, ub)
        mu = 0.5*(S.pred_lb + S.pred_ub)
        sig = (mu - S.pred_lb)/a
        Sig = np.diag(np.square(sig))
        I = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
        initSets.append(I)


    return controller, transformer, norm_mat, scale_mat, plant, initSets
    
    
def load_AEBS_temporal_specs():
    'temporal specification for AEBS model'

    T = 50
    #t = 10

    A1 = np.array([1.0, 0., 0.])
    b1 = np.array([2.5])
    P1 = AtomicPredicate(A1, b1)
    
    A2 = np.array([0., -1., 0])
    b2 = np.array([-0.2])
    P2 = AtomicPredicate(A2, b2)

    EV0T = _EVENTUALLY_(0,T)
    #EV0t = _EVENTUALLY_(0,t)
    AND = _AND_()
    OR = _OR_()
    lb = _LeftBracket_()
    rb = _RightBracket_()
    AW0T = _ALWAYS_(0,T)
    #AW0t = _ALWAYS_(0,t)

    # phi1 : eventually_[0,T](d <= 2.5 AND v_ego >= 0.2) : safety scenario

    phi1 = Formula([EV0T, lb, P1, AND, P2, rb])

    # phi2: always_[0. T](v_ego <= 0.2 -> d <= 2.5)      : prevent early stoping scenario

    phi2 = Formula([AW0T, lb, P2, OR, P1, rb])

    specs = [phi1]

    return specs


def load_scherlock_acc():
    #modelNN_controllerNN

    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/Scherlock_ACC_Trapezius/networks.mat'

    mat_file = loadmat(cur_path)
    controller_nn = mat_file['controller_nn']
    model_nn = mat_file['Model_nn']

    Wc = controller_nn[0][0][0].ravel()
    bc = controller_nn[0][0][1].ravel()
    
    n_weight = len(Wc)

    controller_layers = []
    for i in range(n_weight):
        controller_layers.append(FullyConnectedLayer([Wc[i], bc[i].ravel()]))
        controller_layers.append(ReLULayer())
            
    controller_net = NeuralNetwork(controller_layers, net_type='controller_scherlock_acc')

    Wm = model_nn[0][0][0].ravel()
    bm = model_nn[0][0][1].ravel()
    
    n_weight = len(Wm)

    model_layers = []
    for i in range(n_weight):
        model_layers.append(FullyConnectedLayer([Wm[i], bm[i].ravel()]))
        model_layers.append(ReLULayer())
            
    model_net = NeuralNetwork(controller_layers, net_type='model_scherlock_acc')

    return model_net, controller_net


def load_sherlock_acc_trapezius():
    # load_Trapezius_network

    net_name = 'trapezius'
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + f'/data/nets/Scherlock_ACC_Trapezius/{net_name}.mat'
    mat_file = loadmat(cur_path)
    Net = mat_file['Net']
    W = Net[0][0][0].ravel()
    b = Net[0][0][1].ravel()
    act_fun = Net[0][0][2].ravel() # containts activation list of each neurons
    
    n_weight = len(W)
    n_act_fun = len(act_fun)
    
    
    layers = []
    for i in range(n_weight):
        layers.append(FullyConnectedLayer([W[i].toarray(), b[i].toarray().ravel()]))
        if i < n_act_fun:
            layers.append(MixedActivationLayer(act_fun[i]))
            
    return NeuralNetwork(layers, net_type=net_name)
    

def load_building_model():
    """Load LODE building model"""

    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/lodes/build.mat' 
    mat_contents = loadmat(cur_path)
    A = mat_contents['A']
    A = convert_to_numpy(A)
    B = mat_contents['B']
    B = convert_to_numpy(B)
    C = mat_contents['C']
    C = convert_to_numpy(C)

    plant = LODE(A, B,C)
    return plant


def load_iss_model():
    """Load LODE International State Space Model"""
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/lodes/iss.mat' 
    mat_contents = loadmat(cur_path)
    A = mat_contents['A']
    A = convert_to_numpy(A)
    B = mat_contents['B']
    B = convert_to_numpy(B)
    C = mat_contents['C']
    C = convert_to_numpy(C)
    plant = LODE(A, B, C)
    return plant


def load_helicopter_model():
    """Load LODE helicopter model"""
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/lodes/heli28.mat' 
    mat_contents = loadmat(cur_path)
    A = mat_contents['A']
    A = convert_to_numpy(A)
    plant = LODE(A)
    return plant


def load_MNA5_model():
    """Load LODE MNA5 model"""
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/lodes/MNA_5.mat' 
    mat_contents = loadmat(cur_path)
    A = mat_contents['A']
    B = mat_contents['B']
    A = convert_to_numpy(A)
    B = convert_to_numpy(B)
    plant = LODE(A,B)
    return plant


def load_MNA1_model():
    """Load LODE MNA1 model"""
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/lodes/MNA_1.mat' 
    mat_contents = loadmat(cur_path)
    A = mat_contents['A']
    B = mat_contents['B']
    A = convert_to_numpy(A)
    B = convert_to_numpy(B)
    plant = LODE(A, B)
    return plant


def load_mcs_model():
    """Load LODE MCS model"""
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/lodes/mcs.mat' 
    mat_contents = loadmat(cur_path)
    A = mat_contents['A']
    B = mat_contents['B']
    C = mat_contents['C']
    A = convert_to_numpy(A)
    B = convert_to_numpy(B)
    C = convert_to_numpy(C)
    plant = LODE(A, B,C)
    return plant


def load_heat_model():
    """Load LODE HEAT model"""
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/lodes/heat.mat' 
    mat_contents = loadmat(cur_path)
    A = mat_contents['A']
    B = mat_contents['B']
    A = convert_to_numpy(A)
    B = convert_to_numpy(B)
    C = mat_contents['C']
    C = convert_to_numpy(C)
    plant = LODE(A, B, C)
    return plant

def load_beam_model():
    """Load LODE BEAM model"""
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/lodes/beam.mat' 
    mat_contents = loadmat(cur_path)
    A = mat_contents['A']
    B = mat_contents['B']
    A = convert_to_numpy(A)
    B = convert_to_numpy(B)
    C = mat_contents['C']
    C = convert_to_numpy(C)

    plant = LODE(A, B, C)
    return plant


def load_pde_model():
    """Load LODE PDE model"""
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/lodes/pde.mat' 
    mat_contents = loadmat(cur_path)
    A = mat_contents['A']
    B = mat_contents['B']
    A = convert_to_numpy(A)
    B = convert_to_numpy(B)
    C = mat_contents['C']
    C = convert_to_numpy(C)
    plant = LODE(A, B, C)
    return plant


def load_fom_model():
    """Load LODE FOM model"""
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/lodes/fom.mat' 
    mat_contents = loadmat(cur_path)
    A = mat_contents['A']
    B = mat_contents['B']
    A = convert_to_numpy(A)
    B = convert_to_numpy(B)
    C = mat_contents['C']
    C = convert_to_numpy(C)
    plant = LODE(A, B, C)
    return plant


def load_GRU_network(net_dir, net_name='GRU', dtype='float32'):
    model = onnx.load(net_dir)
    model_initializer = model.graph.initializer
    Weights = []
    for init in model_initializer:
        Weights.append(onnx.numpy_helper.to_array(init))

    GRU_W = Weights[0][0, :, :]
    GRU_R = Weights[1][0, :, :]
    GRU_b = Weights[2][0, :].reshape(-1)
    L1 = GRULayer(layer=[[GRU_W, GRU_R, GRU_b]], output_mode='one', module='default', dtype=dtype)
    FC_W = Weights[4].T
    FC_b = Weights[5]
    L2 = FullyConnectedLayer(layer=[FC_W, FC_b], dtype=dtype)

    net_name = 'GRU'
    return NeuralNetwork(layers=[L1, L2], net_type=net_name)


def load_LSTM_network(net_dir, net_name='LSTM', dtype='float32'):
    model = onnx.load(net_dir)
    model_initializer = model.graph.initializer
    Weights = []
    for init in model_initializer:
        Weights.append(onnx.numpy_helper.to_array(init))

    LSTM_W = Weights[0][0, :, :]
    LSTM_R = Weights[1][0, :, :]
    LSTM_b = Weights[2][0, :].reshape(-1)
    L1 = LSTMLayer(layer=[[LSTM_W, LSTM_R, LSTM_b]], output_mode='one', module='default', dtype=dtype)
    FC_W = Weights[5].T
    FC_b = Weights[6]
    L2 = FullyConnectedLayer(layer=[FC_W, FC_b], dtype=dtype)

    net_name = 'LSTM'
    return NeuralNetwork(layers=[L1, L2], net_type=net_name)


def load_sigmoidal_networks(data_type='mnist', net_size='small', func='tanh', opt=False, delta=0.98):

    data_type = data_type.upper()
    assert data_type in ['MNIST', 'CIFAR10'], "Unsupported data type ('data_type'). Only 'mnist' and 'cifar10' datasets are supported."
    assert net_size in ['small', 'med', 'big'], "Unsupported network size ('net_size'). Only 'small', 'med', 'big' network sizes are supported."
    assert func in ['tanh', 'sigmoid'], "Unsupported layer function type ('func'). Only 'tanh' and 'sigmoid' layers are supported."
    
    # import neural network from ONNX
    net_name = data_type + '_FNN' + net_size + '_' + func
    net_dir = 'StarV/util/SigmoidalNetworks/' + data_type + '/nets/FNN' + net_size + '/' + net_name + '.onnx'

    Weights = []
    model = onnx.load(net_dir)
    model_initializer = model.graph.initializer
    for init in model_initializer:
        Weights.append(onnx.numpy_helper.to_array(init).T)

    num_layers = len(model_initializer) // 2
    layers = []
    for i in range(num_layers):
        L1 = FullyConnectedLayer(layer=[Weights[2*i], Weights[2*i+1]])
        if func == 'tanh':
            L2 = TanSigLayer(opt=opt, delta=delta)
        else:
            L2 = LogSigLayer(opt=opt, delta=delta)
        layers.append(L1)
        layers.append(L2)

    net = NeuralNetwork(layers=layers, net_type=net_name)

    # import flattened image dataset from CSV
    data_name = data_type + '_FNN' + net_size + '_' + func + '.csv'
    data_dir = 'StarV/util/SigmoidalNetworks/' + data_type + '/data/' + data_name

    with open(data_dir, 'r') as x:
        read_csv = list(csv.reader(x, delimiter=","))

    read_csv = np.array(read_csv, dtype=np.float32)
    data = read_csv[:, 1:] / 255.0
    label = read_csv[:, 0]

    return net, data, label


def load_convnet(net_dir, net_type, dtype='float32'):

    assert net_type in ['Small', 'Medium', 'Large'], \
    f"There are 3 types of ConvNet networks: /'Small/', /'Medium/', and /'Large/'"

     # loading DNNs into StarV network
    if net_type == 'Small':
        network = load_CAV2020_MNIST_Small_ConvNet(net_dir=net_dir, dtype=dtype)
    elif net_type == 'Medium':
        network = load_CAV2020_MNIST_Medium_ConvNet(net_dir=net_dir, dtype=dtype)
    elif net_type == 'Large':
        network = load_CAV2020_MNIST_Large_ConvNet(net_dir=net_dir, dtype=dtype)
    else:
        raise Exception('Unknown network type for ConvNet')
    return network


def load_CAV2020_MNIST_Small_ConvNet(net_dir, dtype='float32'):
    b = np.array([-22.511615753173828125]).astype(dtype)
    layers = [FullyConnectedLayer(layer=[None, b], dtype=dtype)]

    return load_neural_network_file(net_dir, layer=layers, net_type='smallConvNetMNIST_CAV2020', 
                                    dtype=dtype, channel_last=True, in_shape=None, sparse=False, show=False)

def load_CAV2020_MNIST_Medium_ConvNet(net_dir, dtype='float32'):
    b = np.array([-22.4925937652587890625]).astype(dtype)
    layers = [FullyConnectedLayer(layer=[None, b], dtype=dtype)]

    return load_neural_network_file(net_dir, layer=layers, net_type='mediumConvNetMNIST_CAV2020', 
                                    dtype=dtype, channel_last=True, in_shape=None, sparse=False, show=False)

def load_CAV2020_MNIST_Large_ConvNet(net_dir, dtype='float32'):
    mean_dir = '/'.join(net_dir.split('/')[:-1]) + "/Large_ConvNet_input_layer_mean.mat"
    mat_file = loadmat(mean_dir)
    b = mat_file['mean'].astype(dtype)
    layers = [FullyConnectedLayer(layer=[None, -b], dtype=dtype)]

    return load_neural_network_file(net_dir, layer=layers, net_type='largeConvNetMNIST_CAV2020', 
                                    dtype=dtype, channel_last=True, in_shape=None, sparse=False, show=False)

def load_neural_network_file(file_path, layer=None, net_type=None, dtype='float64', channel_last=True, in_shape=None, sparse=False, show=False):
    
    assert isinstance(file_path, str), 'error: file_path should be a string'

    file_path = file_path

    if file_path.endswith('.onnx'):
        if show: print('loading onnx module')
        model = onnx.load(file_path)
        model = onnx2pytorch.ConvertModel(model)
        # model = convert(model)
        model.eval()

    elif file_path.endswith('.pt') or file_path.endswith('.pth'):
        if show: print('loading pytorch module')
        model = torch.load(file_path)
        model.eval()
    
    else:
        raise Exception('error: unsupported file format {} (supports .onnx, .pt, .pth)'.format(os.path.splitext(file_path)[1]))
    
    return load_neural_network(model, layer=layer, net_type=net_type, dtype=dtype, channel_last=channel_last, in_shape=in_shape, sparse=sparse, show=show)


def load_neural_network(model, layer=None, net_type=None, dtype='float64', channel_last=True, in_shape=None, sparse=False, show=False):
    if sparse is True and in_shape is not None:
        assert len(in_shape) == 3, \
        f"To unroll weight matrix, the input shape (in_shape) must be provided in a  3-tuple containing (H, W, C). Given in_shape = {in_shape}"
        
        if channel_last is False:
            c, h, w = in_shape
            in_shape = (h, w, c)
    
    if isinstance(model, torch.nn.Module):
        if show: print('converting to StarV module')

        if layer is None:
            layers = []
            cnt = 0
        else: 
            layers = copy.deepcopy(layer)
            cnt = len(layers)
            if show:
                for i, layer in layers:
                    print(f"Pre-given layer {i}: {layer}")

        var = None
        for idx, layer in enumerate(model.modules(), cnt):
            if not isinstance(layer, model.__class__):
                DONE = True
                if show:
                    print(f"Parsing layer {idx}: {layer}")

                if type(layer).__name__ == 'Constant':
                    # remove batch by squeeze
                    var = layer.constant.numpy().squeeze(0).astype(dtype)
                    if var.ndim == 3 and not channel_last:
                        var = var.transpose([1, 2, 0])

                elif type(layer).__name__ in ['sub', 'Sub']:
                    if var is None:
                        print(f"{layer} layer is neglected in the analysis because 'Constant' variable is not previously provided")
                        DONE = False
                    else:
                        layers.append(FullyConnectedLayer(layer=[None, -var], dtype=dtype))
                        var = None

                elif type(layer).__name__ in ['add', 'Add']:
                    if var is None:
                        print(f"{layer} layer is neglected in the analysis because 'Constant' variable is not previously provided")
                        DONE = False
                    else:
                        layers.append(FullyConnectedLayer(layer=[None, var], dtype=dtype))
                        var = None

                elif type(layer).__name__ in ['div', 'Div']:
                    if var is None:
                        print(f"{layer} layer is neglected in the analysis because 'Constant' variable is not previously provided")
                        DONE = False
                    else:
                        layers.append(FullyConnectedLayer(layer=[1/var, None], dtype=dtype))
                        var = None

                elif isinstance(layer, torch.nn.Linear):
                    layers.append(FullyConnectedLayer(layer=layer, dtype=dtype))

                elif isinstance(layer, torch.nn.LSTM):
                    layers.append(LSTMLayer(layer=layer, output_mode='many'))

                elif isinstance(layer, torch.nn.GRU):
                    layers.append(GRULayer(layer=layer, output_mode='many'))

                elif isinstance(layer, torch.nn.Sigmoid):
                    layers.append(LogSigLayer())

                elif isinstance(layer, torch.nn.Tanh):
                    layers.append(TanSigLayer())

                elif isinstance(layer, torch.nn.ReLU):
                    layers.append(ReLULayer())

                elif isinstance(layer, torch.nn.Conv2d):
                    layer_ = Conv2DLayer(layer=layer, dtype=dtype, sparse=sparse, in_shape=in_shape)
                    layers.append(layer_)
                    if sparse:
                        in_shape = layer_.out_shape

                elif isinstance(layer, torch.nn.AvgPool2d):
                    layers.append(AvgPool2DLayer(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding, dtype=dtype))

                elif isinstance(layer, torch.nn.AdaptiveAvgPool2d):
                    layers.append(FlattenLayer(channel_last))
                    print(f"{layer} layer is considered as FalttenLayer in the analysis")

                elif isinstance(layer, torch.nn.BatchNorm2d):
                    layers.append(BatchNorm2DLayer(layer=layer, dtype=dtype))

                elif type(layer).__name__ == 'BatchNormUnsafe':
                    gamma = layer.weight.detach().numpy().copy()
                    beta = layer.bias.detach().numpy().copy()
                    num_features = layer.num_features
                    eps = np.array(layer.eps)
                    var = layer.running_var.numpy()
                    mean = layer.running_mean.numpy()
                    layers.append(BatchNorm2DLayer(layer=[gamma, beta, mean, var], num_features = num_features, eps = eps, dtype=dtype))
                    
                elif isinstance(layer, torch.nn.MaxPool2d):
                    layer_ = MaxPool2DLayer(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding, in_shape=in_shape, dtype=dtype)
                    layers.append(layer_)
                    if sparse:
                        in_shape = layer_.out_shape

                elif isinstance(layer, torch.nn.Flatten) or type(layer).__name__ == 'Flatten':
                    layers.append(FlattenLayer(channel_last))

                elif isinstance(layer, torch.nn.Dropout):
                    print(f"{layer} layer is neglected in the analysis")
                    DONE = False

                elif isinstance(layer, torch.nn.Softmax):
                    print(f"{layer} layer is neglected in the analysis")
                    DONE = False

                elif isinstance(layer, torch.nn.modules.container.Sequential):
                    print(f"torch.nn.modules.container.Sequential layer is neglected in the analysis")
                    DONE = False

                else:
                    print(f"{layer} layer is unsupported and neglected in the analysis \n {type(layer)} type \n {type(layer).__name__} name")

                    DONE = False
                    # raise Exception('error: unsupported neural network layer {}'.format(type(layer)))
                
                if show and DONE:
                    print(f"Parsing layer {idx}: {layer} is done successfully")
                
                prev_layer = layer

        return NeuralNetwork(layers, net_type=net_type)
    
    elif isinstance(model, onnx.onnx_ml_pb2.ModelProtolProto):
        pytorch_model = onnx2pytorch.ConvertModel(model)
        return load_neural_network(pytorch_model, net_type, show)
    
    else:
        raise Exception('error: unsupported neural network module {}'.format(type(model)))
    

def find_node_with_input(graph, input_name):
    'find the unique onnx node with the given input, can return None'

    rv = None

    for n in graph.node:
        for i in n.input:
            if i == input_name:
                assert rv is None, f"multiple onnx nodes accept network input {input_name}"
                rv = n

    return rv


def find_node_with_input(graph, input_name):
    'find the unique onnx node with the given input, can return None'

    rv = None

    for n in graph.node:
        for i in n.input:
            if i == input_name:
                assert rv is None, f"multiple onnx nodes accept network input {input_name}"
                rv = n

    return rv

def load_onnx_network(filename, net_type=None, channel_last=True, num_pixel_classes=None, dtype='float64', show=False):
    model = onnx.load(filename)
    onnx.checker.check_model(model)

    graph = model.graph

    #print(graph)

    # find the node with input "input"
    all_input_names = sum([[str(i) for i in n.input] for n in graph.node], [])

    #print(f"all input names: {all_input_names}")

    all_initializer_names = [i.name for i in graph.initializer]
    all_output_names = sum([[str(o) for o in n.output] for n in graph.node], [])

    # the input to the network is the one not in all_inputs_list and not in all_outputs_list
    network_input = None
    
    for i in all_input_names:
        if i not in all_initializer_names and i not in all_output_names:
            assert network_input is None, f"multiple onnx network inputs {network_input} and {i}"        
            network_input = i

    assert network_input, "did not find onnx network input"

    assert len(graph.output) == 1, "onnx network defined multiple outputs"
    network_output = graph.output[0].name

    #print(f"input: '{network_input}', output: '{network_output}'")
    
    #assert network_input == graph.input[0].name, \
    #    f"network_input ({network_input}) != graph.input[0].name ({graph.input[0].name})"
    ##########

    # map names -> structs
    # input_map = {i.name: i for i in graph.input}
    init_map = {i.name: i for i in graph.initializer}

    # i = input_map[network_input]

    # find the node which takes the input (probably node 0)
    cur_node = find_node_with_input(graph, network_input)
    cur_input_name = network_input
    # ok! now proceed recusively
    layers = []

    # data types
    onnx_type_float = 1
    onnx_type_int = 2

    cnt = 0
    while cur_node is not None:
        assert cur_node.input[0] == cur_input_name, \
            f"cur_node.input[0] ({cur_node.input[0]}) should be previous output ({cur_input_name}) in " + \
            f"node:\n{cur_node.name}"
        
        op = cur_node.op_type
        layer = None
        neglect = False

#         if layers:
#             prev_shape = layers[-1].get_output_shape()
#         else:
#             s_node = graph.input[0].type.tensor_type.shape
#             prev_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in s_node.dim)

        if show:
            print(f"Parsing layer {cnt}: {op}")
        
        if op in ['Add', 'Sub']:
            assert len(cur_node.input) == 2
            init = init_map[cur_node.input[1]]
            assert init.data_type == onnx_type_float
            
            var = np.frombuffer(init.raw_data, dtype='<f4') # little endian float32
            if op == 'Sub':
                var = -var
            
            layer = FullyConnectedLayer(layer=[None, var], dtype=dtype)
        
        elif op == 'Flatten':
            layer = FlattenLayer(channel_last)
            
        elif op == 'MatMul':
            assert len(cur_node.input) == 2
            weight_init = init_map[cur_node.input[1]]
            assert weight_init.data_type == onnx_type_float

            shape = weight_init.dims[::-1]
            weight = np.frombuffer(weight_init.raw_data, dtype='<f4') # little endian float32
            weight = weight.reshape(shape, order='F').transpose([1, 0])

            layer = FullyConnectedLayer(layer=[weight, None], dtype=dtype)
            
        
#         elif op == 'Gemm':
#             assert len(cur_node.input) == 3
            
#             weight_init = init_map[cur_node.input[1]]
#             bias_init = init_map[cur_node.input[2]]

#             # weight
#             assert weight_init.data_type == onnx_type_float
#             b = np.frombuffer(weight_init.raw_data, dtype='<f4') # little endian float32
#             shape = tuple(d for d in reversed(weight_init.dims)) # note dims reversed, acasxu has 5, 50 but want 5 cols
#             weight_mat = nn_unflatten(b, shape, order='F')

#             # bias
#             assert bias_init.data_type == onnx_type_float
#             b = np.frombuffer(bias_init.raw_data, dtype='<f4') # little endian float32
#             shape = tuple(d for d in reversed(bias_init.dims)) # note dims reversed, acasxu has 5, 50 but want 5 cols
#             bias_vec = nn_unflatten(b, shape, order='F')

#             for a in cur_node.attribute:
#                 assert a.name in ['alpha', 'beta', 'transB'], "general Gemm node unsupported"

#                 if a.name in ['alpha', 'beta']:
#                     assert a.f == 1.0
#                     assert a.type == onnx_type_float
#                 elif a.name == 'transB':
#                     assert a.type == onnx_type_int
#                     assert a.i == 1
#                     weight_mat = weight_mat.transpose().copy()

#             layer = FullyConnectedLayer([weight_mat, bias_vec], dtype=dtype)

        elif op == 'Conv':
            assert len(cur_node.input) == 3
            weight_init = init_map[cur_node.input[1]]
            bias_init = init_map[cur_node.input[2]]

            assert weight_init.data_type == onnx_type_float
            shape = weight_init.dims[::-1]
            weight = np.frombuffer(weight_init.raw_data, dtype='<f4') # little endian float32
            weight = weight.reshape(shape, order='F').transpose([1, 0, 2, 3])

            if not channel_last:
                weight.transpose([2, 3, 1, 0])

            assert bias_init.data_type == onnx_type_float
            shape = bias_init.dims[::-1]
            bias = np.frombuffer(bias_init.raw_data, dtype='<f4') # little endian float32
            bias = bias.reshape(shape, order='F')
            
            dilation = np.array(cur_node.attribute[1].ints, dtype='b')
            padding = np.array(cur_node.attribute[3].ints, dtype='b')
            padding = np.array([padding[0], padding[2], padding[1], padding[3]])
            stride = np.array(cur_node.attribute[4].ints, dtype='b')

            layer = Conv2DLayer([weight, bias], stride, padding, dilation, dtype=dtype)
        
        elif op == 'BatchNormalization':
            assert len(cur_node.input) == 5
            gamma_init = init_map[cur_node.input[1]]
            beta_init = init_map[cur_node.input[2]]
            mean_init = init_map[cur_node.input[3]]
            var_init = init_map[cur_node.input[4]]
            
            assert gamma_init.data_type == onnx_type_float
            shape = gamma_init.dims[::-1]
            gamma = np.frombuffer(gamma_init.raw_data, dtype='<f4').reshape(shape) # little endian float32
            
            assert beta_init.data_type == onnx_type_float
            shape = beta_init.dims[::-1]
            beta = np.frombuffer(beta_init.raw_data, dtype='<f4').reshape(shape) # little endian float32
           
            assert mean_init.data_type == onnx_type_float
            shape = mean_init.dims[::-1]
            mean = np.frombuffer(mean_init.raw_data, dtype='<f4').reshape(shape) # little endian float32

            assert var_init.data_type == onnx_type_float
            shape = var_init.dims[::-1]
            var = np.frombuffer(var_init.raw_data, dtype='<f4').reshape(shape) # little endian float32
            
            eps = np.array(cur_node.attribute[0].f, dtype='<f4')
            layer = BatchNorm2DLayer(layer=[gamma, beta, mean, var], num_features = shape[0], eps = eps, dtype=dtype)
        
        elif op == 'AveragePool':
            assert len(cur_node.input) == 1
            kernel_size = np.array(cur_node.attribute[0].ints, dtype='b')
            padding = np.array(cur_node.attribute[1].ints, dtype='b')
            padding = np.array([padding[0], padding[2], padding[1], padding[3]])
            stride = np.array(cur_node.attribute[2].ints, dtype='b')
            # count_include_pad = np.array(cur_node.attribute[3].ints, dtype='b')
            layer = AvgPool2DLayer(kernel_size=kernel_size, stride=stride, padding=padding, dtype=dtype)

        elif op == 'MaxPool':
            assert len(cur_node.input) == 1
            kernel_size = np.array(cur_node.attribute[0].ints, dtype='b')
            padding = np.array(cur_node.attribute[1].ints, dtype='b')
            padding = np.array([padding[0], padding[2], padding[1], padding[3]])
            stride = np.array(cur_node.attribute[2].ints, dtype='b')
            layer = MaxPool2DLayer(kernel_size=kernel_size, stride=stride, padding=padding, dtype=dtype)
                        
        elif op == 'ConvTranspose':
            assert len(cur_node.input) == 3
            weight_init = init_map[cur_node.input[1]]
            bias_init = init_map[cur_node.input[2]]

            assert weight_init.data_type == onnx_type_float
            shape = weight_init.dims[::-1]
            weight = np.frombuffer(weight_init.raw_data, dtype='<f4') # little endian float32
            weight = weight.reshape(shape, order='F').transpose([1, 0, 2, 3])

            if not channel_last:
                weight.transpose([2, 3, 1, 0])

            assert bias_init.data_type == onnx_type_float
            shape = bias_init.dims[::-1]
            bias = np.frombuffer(bias_init.raw_data, dtype='<f4') # little endian float32
            bias = bias.reshape(shape, order='F')
            
            for attr in cur_node.attribute:
                name = attr.name
                if name == 'dilations':
                    dilation = np.array(attr.ints, dtype='b')
                elif name == 'pads':
                    padding = np.array(attr.ints, dtype='b')
                    padding = np.array([padding[0], padding[2], padding[1], padding[3]])
                elif name == 'strides':
                    stride = np.array(attr.ints, dtype='b')
                elif name == 'output_padding ':
                    output_padding = np.array(attr.ints, dtype='b')

            layer = ConvTranspose2DLayer([weight, bias], stride, padding, dilation, dtype=dtype)
            
        elif op == 'Relu':
            layer = ReLULayer()
            
        elif op == 'Transpose':
            neglect = True
        
        elif op == 'Softmax':
            neglect = True
            
        else:
            assert False, f"unsupported onnx op_type {op} in node {cur_node.name}"

        if neglect:
            if show:
                print(f"onnx op_type {op} is neglected in the analysis")
        else:
            assert layer is not None
            layers.append(layer)

        assert len(cur_node.output) == 1, f"multiple output at onnx node {cur_node.name}"
        cur_input_name = cur_node.output[0]

        #print(f"{cur_node.name} -> {cur_input_name}")
        cur_node = find_node_with_input(graph, cur_input_name)

        cnt += 1
        

    assert cur_input_name == network_output, \
        f"output witout node {cur_input_name} is not network output {network_output}"

    if num_pixel_classes != None:
        assert isinstance(num_pixel_classes, int), f"num_pixel_classes should be integer, but received {type(num_pixel_classes)}"
        assert num_pixel_classes > 0, f"num_pixel_classes should be a positive integer"
        
        layers.append(PixelClassificationLayer(num_pixel_classes))

    return NeuralNetwork(layers, net_type=net_type)