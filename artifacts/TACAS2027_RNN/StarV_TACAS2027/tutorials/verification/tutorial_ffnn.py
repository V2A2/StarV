"""
Feedforward Neural Network Ttutorial

This tutorial demonstrates how to construct a feedforward neural network and perform reachability analysis on it.

"""

import copy
import os
import torch
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import StarV
from scipy.io import loadmat
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.LeakyReLULayer import LeakyReLULayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.SatLinsLayer import SatLinsLayer
from StarV.net.network import NeuralNetwork
from StarV.net.network import reachExactBFS, reachApproxBFS
from StarV.verifier.verifier import checkSafetyStar, checkSafetyProbStar
from StarV.verifier.verifier import qualiVerifyExactBFS, qualiVerifyApproxBFS, quantiVerifyBFS
from StarV.util.plot import plot_star, plot_probstar, plot_Mesh3D_Star, plot_3D_Star
from StarV.util.plot import plot_probstar_distribution, plot_probstar_contour


def ffnn_construct_manually():
    """
    Manually construct a feedforward neural network
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: FFNN Construction (Manually) ====================================')

    # Construct a feedforward neural network
    layers = []
    W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b1 = np.array([0.5, 1.0, -0.5])
    layer1 = [W1, b1]
    L1 = FullyConnectedLayer(layer1)
    L2 = ReLULayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    layer2 = [W2, b2]
    L3 = FullyConnectedLayer(layer2)
    
    layers.append(L1)
    layers.append(L2)
    layers.append(L3)

    F = NeuralNetwork(layers, 'ffnn_tiny_network')
    print(F)

    print('=============== DONE: FFNN Construction (Manually) =======================================')
    print('==========================================================================================\n\n')


def ffnn_construct_from_mat_file():
    """
    Construct a feedforward neural network from a .mat file
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: FFNN Construction (From .mat file) ==============================')

    # Construct a feedforward neural network from a .mat file
    starv_root_path = os.path.dirname(StarV.__file__)
    net_path = starv_root_path + '/util/data/nets/ACASXU/ACASXU_run2a_1_1_batch_2000.mat'
    mat_contents = loadmat(net_path)
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
    F = NeuralNetwork(layers, net_type='ffnn_ACASXU_1_1')
    print(F)

    print('=============== DONE: FFNN Construction (From .mat file) =================================')
    print('==========================================================================================\n\n')


def ffnn_evaluate_input_vector():
    """
    Evaluate a feedforward neural network with a given input vector
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Evaluate Input Vector on FFNN ================================')
    
    # Construct a feedforward neural network
    layers = []
    W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b1 = np.array([0.5, 1.0, -0.5])
    layer1 = [W1, b1]
    L1 = FullyConnectedLayer(layer1)
    L2 = ReLULayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    layer2 = [W2, b2]
    L3 = FullyConnectedLayer(layer2)
    
    layers.append(L1)
    layers.append(L2)
    layers.append(L3)

    F = NeuralNetwork(layers, 'ffnn_tiny_network')

    x = np.array([-1.0, 2.0])
    y = F.evaluate(x)

    print("Input vector:", x)
    print("Output after FFNN:", y)
    print('=============== DONE: Evaluate Input Vector on FFNN ==================================')
    print('==========================================================================================\n\n')    

def ffnn_quali_reachability_star():
    """
    Perform qualitative reachability analysis on a feedforward neural network using Star sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Qualitative Reachability Analysis on FFNN using Star ============')
    # Construct a feedforward neural network
    layers = []
    W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b1 = np.array([0.5, 1.0, -0.5])
    layer1 = [W1, b1]
    L1 = FullyConnectedLayer(layer1)
    L2 = ReLULayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    layer2 = [W2, b2]
    L3 = FullyConnectedLayer(layer2)
    
    layers = [L1, L2, L3]
    F = NeuralNetwork(layers, 'ffnn_tiny_network')

    # Construct input set
    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])
    S = Star(lb, ub)
    plot_star(S)

    # Exact Reachability analysis
    # ----- Parallelized version -----
    numCores = 2
    pool = multiprocessing.Pool(numCores)
    # ----- Single-core version -----
    # pool = None

    print('Exact Reachability analysis:')
    I = [S]
    R_exact = reachExactBFS(F, I, lp_solver='gurobi', pool=pool, show=True)
    print('Number of Exact Output Reachable sets: ', len(R_exact))
    plot_star(R_exact)

    # Over-approximation reachability analysis
    print('Over-approximation Reachability analysis:')
    I = S
    R_approx = reachApproxBFS(F, I, lp_solver='gurobi', show=True)
    print('Number of Over-approximate Output Reachable sets: ', len(R_approx))
    plot_star(R_approx)

    print('=============== DONE: Qualitative Reachability Analysis on FFNN using Star ===============')
    print('==========================================================================================\n\n')



def ffnn_manual_quali_verification_star():
    """
    Perform qualitative verification on a feedforward neural network using Star sets manually
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Manual Qualitative Verification on FFNN using Star ==============')
    # Construct a feedforward neural network
    layers = []
    W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b1 = np.array([0.5, 1.0, -0.5])
    layer1 = [W1, b1]
    L1 = FullyConnectedLayer(layer1)
    L2 = ReLULayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    layer2 = [W2, b2]
    L3 = FullyConnectedLayer(layer2)
    
    layers = [L1, L2, L3]
    F = NeuralNetwork(layers, 'ffnn_tiny_network')

    # Construct input set
    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])
    S = Star(lb, ub)
    plot_star(S)

    # Exact Reachability analysis
    # ----- Parallelized version -----
    numCores = 2
    pool = multiprocessing.Pool(numCores)
    # ----- Single-core version -----
    # pool = None

    I = [S]
    R_exact = reachExactBFS(F, I, lp_solver='gurobi', pool=pool, show=True)
    print('Number of Exact Reachable sets: ', len(R_exact))
    plot_star(R_exact)

    # Over-approximation reachability analysis
    I = S
    R_approx = reachApproxBFS(F, I, lp_solver='gurobi', show=True)
    print('Number of Over-approximate Reachable sets: ', len(R_approx))
    plot_star(R_approx)

    # Specifying a property of an FFNN
    # y1 >= 4
    unsafe_mat = np.array([[-1.0, 0]])
    unsafe_vec = np.array([-4.0])

    # Verification for R_exact
    U_exact = [] # unsafe output set
    Res_exact = [] # verification result
    for R_exact_i in R_exact:
        U1 = checkSafetyStar(unsafe_mat, unsafe_vec, R_exact_i)
        if isinstance(U1, Star):
            U_exact.append(U1)
            Res_exact.append('SAT')
        else:
            Res_exact.append('UNSAT')

    print('Length of unsafe sets: ', len(U_exact))
    print('Verification result: ', Res_exact)

    # Verification for R_approx
    U_approx = checkSafetyStar(unsafe_mat, unsafe_vec, R_approx)
    if isinstance(U_approx, Star):
        Res_approx = 'SAT'
    else:
        Res_approx = 'UNKOWN'
    print('Verification result: ', Res_approx)


    print('=============== DONE: Manual Qualitative Verification on FFNN using Star =================')
    print('==========================================================================================\n\n')


def ffnn_auto_quali_verification_star():
    """
    Perform qualitative verification on a feedforward neural network using Star sets automatically
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Automatic Qualitative Verification on FFNN using Star ===========')
    # Construct a feedforward neural network
    layers = []
    W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b1 = np.array([0.5, 1.0, -0.5])
    layer1 = [W1, b1]
    L1 = FullyConnectedLayer(layer1)
    L2 = ReLULayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    layer2 = [W2, b2]
    L3 = FullyConnectedLayer(layer2)
    
    layers = [L1, L2, L3]
    F = NeuralNetwork(layers, 'ffnn_tiny_network')

    # Construct input set
    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])
    S = Star(lb, ub)
    plot_star(S)

    # Specifying a property of an FFNN
    # y1 <= -2
    unsafe_mat = np.array([[1.0, 0]])
    unsafe_vec = np.array([-2.0])

    # Exact Verification
    numCores = 2
    I = [S]
    R_exact, U_exact, C_exact, Res_exact = qualiVerifyExactBFS(F, I, unsafe_mat, unsafe_vec, numCores=numCores)
    print('Number of Exact Reachable sets: ', len(R_exact))
    print('Number of Exact Unsafe sets: ', len(U_exact))
    print('Number of Counter Input sets: ', len(C_exact))
    print('Verification result: ', Res_exact)
    plot_star(R_exact)
    plot_star(U_exact)
    plot_star(C_exact)


    # Over-approximate Verification
    I = S
    R_approx, U_approx, Res_approx = qualiVerifyApproxBFS(F, I, unsafe_mat, unsafe_vec)
    print('Number of Over-approximate Reachable sets: ', len(R_approx))
    print('Number of Over-approximate Unsafe sets: ', len(U_approx))
    print('Verification result: ', Res_approx)
    plot_star(R_approx)
    plot_star(U_approx)


    print('=============== DONE: Automatic Qualitative Verification on FFNN using Star ==============')
    print('==========================================================================================\n\n')



def ffnn_quanti_reachability_probstar():
    """
    Perform quantitative reachability analysis on a feedforward neural network using ProbStar sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Quantitative Reachability Analysis on FFNN using ProbStar =======')
    # Construct a feedforward neural network
    layers = []
    W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b1 = np.array([0.5, 1.0, -0.5])
    layer1 = [W1, b1]
    L1 = FullyConnectedLayer(layer1)
    L2 = ReLULayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    layer2 = [W2, b2]
    L3 = FullyConnectedLayer(layer2)
    
    layers = [L1, L2, L3]
    F = NeuralNetwork(layers, 'ffnn_tiny_network')

    # Construct input set
    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])
    S = Star(lb, ub)
    mu = 0.5*(S.pred_lb + S.pred_ub)
    a = 2.5 # coefficience to adjust the distribution
    sig = (mu - S.pred_lb)/a
    Sig = np.diag(np.square(sig))
    P = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
    plot_probstar(P)

    # ----- Parallelized version -----
    numCores = 2
    pool = multiprocessing.Pool(numCores)
    # ----- Single-core version -----
    # pool = None

    # Exact Reachability analysis
    I = [P]
    R_exact = reachExactBFS(F, I, lp_solver='gurobi', pool=pool, show=False)
    print('Number of Exact Reachable sets: ', len(R_exact))
    plot_probstar(R_exact)

    # Over-approximation reachability analysis
    I = [P]
    R_approx, p_ignored = reachApproxBFS(F, I, p_filter=0.1, lp_solver='gurobi', pool=pool, show=True)
    print('Number of Over-approximate Reachable sets: ', len(R_approx))
    plot_probstar(R_approx)

    print('=============== DONE: Quantitative Reachability Analysis on FFNN using ProbStar ==========')
    print('==========================================================================================\n\n')


def ffnn_auto_quanti_verification_probstar():
    """
    Perform quantitative verification on a feedforward neural network using ProbStar sets automatically
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Automatic Quantitative Verification on FFNN using ProbStar =======')
    
    # Construct a feedforward neural network
    layers = []
    W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b1 = np.array([0.5, 1.0, -0.5])
    layer1 = [W1, b1]
    L1 = FullyConnectedLayer(layer1)
    L2 = ReLULayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    layer2 = [W2, b2]
    L3 = FullyConnectedLayer(layer2)
    
    layers = [L1, L2, L3]
    F = NeuralNetwork(layers, 'ffnn_tiny_network')

    # Construct input set
    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])
    S = Star(lb, ub)
    mu = 0.5*(S.pred_lb + S.pred_ub)
    a = 2.5 # coefficience to adjust the distribution
    sig = (mu - S.pred_lb)/a
    Sig = np.diag(np.square(sig))
    P = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
    plot_probstar(P)

    # Specifying a property of an FFNN
    # y1 <= -2
    unsafe_mat = np.array([[1.0, 0]])
    unsafe_vec = np.array([-2.0])

    # number of cores
    numCores = 2

    I = [P]

    # Exact Verification
    R_exact, U_exact, C_exact, _, _, _, _ = quantiVerifyBFS(F, I, unsafe_mat, unsafe_vec, p_filter=0.0, numCores=numCores, show=False)
    print('Number of Exact Reachable sets: ', len(R_exact))
    print('Number of Exact Unsafe sets: ', len(U_exact))
    print('Number of Counter Input sets: ', len(C_exact))
    plot_probstar(R_exact)
    plot_probstar(U_exact)
    plot_probstar(C_exact)

    # Over-approximate Verification
    R_approx, U_approx, C_approx, _, _, _, _ = quantiVerifyBFS(F, I, unsafe_mat, unsafe_vec, p_filter=0.1, numCores=numCores, show=True)
    print('Number of Over-approximate Reachable sets: ', len(R_approx))
    print('Number of Over-approximate Unsafe sets: ', len(U_approx))
    print('Number of Counter Input sets: ', len(C_approx))
    plot_probstar(R_approx)
    plot_probstar(U_approx)
    plot_probstar(C_approx)

    print('=============== DONE: Automatic Quantitative Verification on FFNN using ProbStar ==========')
    print('==========================================================================================\n\n')

if __name__ == '__main__':
    """
    Main function to run the FFNN tutorials
    """
    ffnn_construct_manually()
    ffnn_construct_from_mat_file()

    ffnn_evaluate_input_vector()
    
    ffnn_quali_reachability_star()
    ffnn_manual_quali_verification_star()
    ffnn_auto_quali_verification_star()
    
    ffnn_quanti_reachability_probstar()
    ffnn_auto_quanti_verification_probstar()

