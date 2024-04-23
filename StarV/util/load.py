"""
load module, to load existing networks for testing/evaluation
Dung Tran, 9/12/2022
"""
import os
from scipy.io import loadmat
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.net.network import NeuralNetwork
from StarV.nncs.nncs import NNCS
from StarV.plant.lode import LODE, DLODE
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_, _OR_
from StarV.spec.dProbStarTL import DynamicFormula
import numpy as np
import torch
import math

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
        L1 = fullyConnectedLayer(Wi, bi)
        L2 = ReLULayer()
        layers.append(L1)
        layers.append(L2)


    bi = b[n-1,0]
    bi = bi.reshape(bi.shape[0],)
    L1 = fullyConnectedLayer(W[0,n-1], bi)
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


def load_AEBS():
    """load avanced emergency braking system
    
    This case study is from this paper: 

    Tran et al. "Safety Verification of Learning-enabled Cyber-Physical Systems with Reinforcement Control", EMSOFT 2019
    
    """

    # load transformer network
    # load reinforment controller network
    # load plant dynamics A, B, C
    # load initial conditions

    

    
    

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
