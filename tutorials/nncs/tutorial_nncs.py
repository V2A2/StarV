"""
Neural Network Control System (NNCS) Tutorial

This tutorial demonstrates how to verify the Neural Network Control System (NNCS) class in StarV.
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

def nncs_acc_construct():
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


def nncs_acc_initial_states():
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


def nncs_acc_unsafe_property():
    """
    Specifying the unsafe property for ACC system
    """

    print('==========================================================================================')
    print('=============== EXAMPLE: ACC system Unsafe Property ======================================')

    # unsafe constraints
    # safety property: actual distance > alpha * safe distance <=> d = (x1 - x4) > alpha * d_safe = alpha * (1.4 * v_ego + 10)
    # unsafe region: x1 - x4 <= alpha * (1.4 * v_ego + 10)

    alpha = 1.0
    unsafe_mat = np.array([[1.0, 0., 0., -1., -alpha*1.4, 0., 0.]])
    unsafe_vec = np.array([alpha*10.0])

    print('Unsafe matrix: ', unsafe_mat)
    print('Unsafe vector: ', unsafe_vec)

    print('==========================================================================================')
    print('============================ Done: ACC system Unsafe Property ============================\n\n')


def nncs_acc_verifying():
    """
    Verifying the ACC system
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: ACC system Verification =========================================')

    # Load the neural network controller
    mat_contents = loadmat('controller_5_20')
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


    # unsafe constraints
    # safety property: actual distance > alpha * safe distance <=> d = (x1 - x4) > alpha * d_safe = alpha * (1.4 * v_ego + 10)
    # unsafe region: x1 - x4 <= alpha * (1.4 * v_ego + 10)

    alpha = 1.0
    unsafe_mat = np.array([[1.0, 0., 0., -1., -alpha*1.4, 0., 0.]])
    unsafe_vec = np.array([alpha*10.0])


    # verify the system
    
    # reference inputs
    refInputs = np.array([30., 1.4])

    # --- For exact qualitative and quantitative verification ---
    pf = 0.0
    # --- For over-approximate qualitative and quantitative verification ---
    # pf = 0.001

    initSet_id=5
    numSteps=30
    numCores=4

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]

    print('Verifying the ACC system for {} timesteps using exact reachability...'.format(numSteps))
    res = verifyBFS_DLNNCS(sys, verifyPRM)
    RX = res.RX
    Ce = res.CeIn
    Co = res.CeOut

    n = len(RX)
    CE = []
    CO = []
    for i in range(0,n):
        if len(Ce[i]) > 0:
            CE.append(Ce[i])  # to plot counterexample set
            CO.append(Co[i])

    # plot reachable set  (d_actual - d_safe) vs. (v_ego)
    dir_mat1 = np.array([[0., 0., 0., 0., 1., 0., 0.],
                        [1., 0., 0., -1., -1.4, 0., 0.]])
    dir_vec1 = np.array([0., -10.])

    # plot reachable set  d_rel vs. d_safe
    dir_mat2 = np.array([[1., 0., 0., -1., 0., 0., 0.],
                        [0., 0., 0., 0., 1.4, 0., 0.]])
    dir_vec2 = np.array([0., 10.])

    # plot counter input set
    dir_mat3 = np.array([[0., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 1., 0., 0.]])
    dir_vec3 = np.array([0., 0.])


    print('Plot reachable set...')
    fig1 = plt.figure()
    plot_probstar_reachset(RX, dir_mat=dir_mat2, dir_vec=dir_vec2, show_prob=False, label=('$d_{r}$','$d_{safe}$'), show=True)

    fig2 = plt.figure()
    plot_probstar_reachset(RX, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=False, label=('$v_{ego}$','$d_r - d_{safe}$'), show=True)

    print('Plot counter output set...')
    fig3 = plt.figure()
    plot_probstar_reachset(CO, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=False, label=('$v_{ego}$','$d_r - d_{safe}$'), show=True)

    print('Plot counter init set ...')
    fig4 = plt.figure()
    plot_probstar_reachset(CE, dir_mat=dir_mat3, dir_vec=dir_vec3, show_prob=False, label=('$v_{lead}[0]$','$v_{ego}[0]$'), show=True)


    print('==========================================================================================')
    print('============================ Done: ACC system Verification ===============================\n\n')




if __name__ == '__main__':
    """
    Main function to run the NNCS tutorials
    """

    nncs_acc_construct()
    nncs_acc_initial_states()
    nncs_acc_unsafe_property()
    nncs_acc_verifying()