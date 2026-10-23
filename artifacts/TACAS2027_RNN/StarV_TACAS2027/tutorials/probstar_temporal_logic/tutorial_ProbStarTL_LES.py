"""
ProbStar Temporal Logic  Tutorial

This tutorial demonstrates how to verify the LES using ProbStarTL in StarV.
"""

import copy
import os
import numpy as np
import StarV
from scipy.io import loadmat
from matplotlib import pyplot as plt
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.net.network import NeuralNetwork
from StarV.nncs.nncs import NNCS
from StarV.plant.lode import LODE, DLODE
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.nncs.nncs import VerifyPRM_NNCS, verifyBFS_DLNNCS
from StarV.util.plot import plot_probstar_reachset
from StarV.set.star import Star
from StarV.nncs.nncs import VerifyPRM_NNCS, verifyBFS_DLNNCS, ReachPRM_NNCS, reachBFS_DLNNCS, reachDFS_DLNNCS, verify_temporal_specs_DLNNCS, verify_temporal_specs_DLNNCS_for_full_analysis
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_,_OR_
from StarV.spec.dProbStarTL import DynamicFormula

def probstarTL_nncs_acc_construct():
    """
    Constructing a NNCS for ACC system
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: ACC system Construction =========================================')

    # Load the neural network controller
    starv_root_path = os.path.dirname(StarV.__file__)
    net_path = starv_root_path + '/util/data/nets/ACC/controller_5_20.mat'
    mat_contents = loadmat(net_path)
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
    
    net = NeuralNetwork(layers, 'controller_5_20')

    # Load the plant dynamics
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

    sys = NNCS(net, dplant, type='DLNNCS')

    print(sys)

    print('==========================================================================================')
    print('============================ Done: ACC system Construction ===============================\n\n')


def probstarTL_nncs_acc_initial_states():
    """
    Constructing initial states for ACC system
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: ACC system Initial States =======================================')

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

    print('==========================================================================================')
    print('============================ Done: ACC system Initial States =============================\n\n')


def probstarTL_nncs_acc_temporal_specs():
    """
    Creating temporal specifications
    """

    print('==========================================================================================')
    print('=============== EXAMPLE: ACC system Temporal Specifications ==============================')

    # load temporal and logic operator
    T = 10
    EV0T = _EVENTUALLY_(0,T)
    lb = _LeftBracket_()
    rb = _RightBracket_()
    AW0T = _ALWAYS_(0,T)

    # create atomic predicates
    A1 = np.array([1., 0., 0., -1., -1.4, 0., 0.])
    b1 = np.array([10.])
    P1 = AtomicPredicate(A1,b1)

    A2 = np.array([-1., 0., 0., 1., 1.4, 0., 0.])
    b2= np.array([-10.])
    P1c = AtomicPredicate(A2,b2)

    # convert atomic predicate to temporal specification
    # phi1 : eventually_[0, T](x_lead - x_ego <= D_safe = 10 + 1.4 v_ego) : A1x <= b1
    phi1= Formula([EV0T, lb, P1, rb])
    print(phi1)
    # phi1c : always_[0, T] (x_lead - x_ego >= D_safe = 10 + 1.4 v_ego) : A2x <= b2
    phi1c = Formula([AW0T, lb, P1c, rb])
    print(phi1c)

    specs = [phi1, phi1c]


    print('==========================================================================================')
    print('============================ Done: ACC system Temporal Specifications ====================\n\n')


def probstarTL_nncs_acc_verifying():
    """
    Verifying the ACC system using ProbSatrTL
    """

    print('==========================================================================================')
    print('=============== EXAMPLE: ACC system Verification =========================================')

    # Load the neural network controller
    starv_root_path = os.path.dirname(StarV.__file__)
    net_path = starv_root_path + '/util/data/nets/ACC/controller_5_20.mat'
    mat_contents = loadmat(net_path)
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
    
    net = NeuralNetwork(layers, 'controller_5_20')

    # Load the plant dynamics
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

    sys = NNCS(net, dplant, type='DLNNCS')


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

    # load temporal and logic operator
    T = 10
    EV0T = _EVENTUALLY_(0,T)
    lb = _LeftBracket_()
    rb = _RightBracket_()
    AW0T = _ALWAYS_(0,T)

    # create atomic predicates
    A1 = np.array([1., 0., 0., -1., -1.4, 0., 0.])
    b1 = np.array([10.])
    P1 = AtomicPredicate(A1,b1)

    A2 = np.array([-1., 0., 0., 1., 1.4, 0., 0.])
    b2= np.array([-10.])
    P1c = AtomicPredicate(A2,b2)

    # convert atomic predicate to temporal specification
    # phi1 : eventually_[0, T](x_lead - x_ego <= D_safe = 10 + 1.4 v_ego) : A1x <= b1
    phi1= Formula([EV0T, lb, P1, rb])
    print(phi1)
    # phi1c : always_[0, T] (x_lead - x_ego >= D_safe = 10 + 1.4 v_ego) : A2x <= b2
    phi1c = Formula([AW0T, lb, P1c, rb])
    print(phi1c)

    specs = [phi1, phi1c]

    # verify the system
    
    # reference inputs
    refInputs = np.array([30., 1.4])

    # --- For exact qualitative and quantitative verification ---
    pf = 0.0
    # --- For over-approximate qualitative and quantitative verification ---
    # pf = 0.001

    initSet_id=5
    numSteps= T
    numCores=4

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.temporalSpecs = copy.deepcopy(specs)

    traces, p_SAT_MAX, p_SAT_MIN, reachTime, checkingTime, verifyTime = verify_temporal_specs_DLNNCS(sys, verifyPRM)

    print('Number of traces = {}'.format(len(traces)))
    print('p_SAT_MAX = {}'.format(p_SAT_MAX))
    print('p_SAT_MIN = {}'.format(p_SAT_MIN))
    print('reachTime = {}'.format(reachTime))
    print('checkingTime = {}'.format(checkingTime))
    print('verifyTime = {}'.format(verifyTime))

    print('==========================================================================================')
    print('============================ Done: ACC system Verification ===============================\n\n')




if __name__ == '__main__':
    """
    Main function to run the ProbStarTL of LES tutorials
    """
    probstarTL_nncs_acc_construct()
    probstarTL_nncs_acc_initial_states()
    probstarTL_nncs_acc_temporal_specs()
    probstarTL_nncs_acc_verifying()