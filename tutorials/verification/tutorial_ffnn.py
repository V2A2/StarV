"""
Feedforward Neural Network Ttutorial

This tutorial demonstrates how to construct a feedforward neural network and perform reachability analysis on it.

"""

import copy
import torch
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.LeakyReLULayer import LeakyReLULayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.SatLinsLayer import SatLinsLayer
from StarV.net.network import NeuralNetwork
from StarV.net.network import reachExactBFS, reachApproxBFS
from StarV.util.plot import plot_star, plot_probstar, plot_Mesh3D_Star, plot_3D_Star, plot_probstar_distribution, plot_probstar_contour


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
    output = F.evaluate(x)

    print("Input vector:", x)
    print("Output after FFNN:", output)
    print('=============== DONE: Evaluate Input Vector on FFNN ==================================')
    print('==========================================================================================\n\n')    

def ffnn_reachability_exact_star():
    """
    Perform exact reachability analysis on a feedforward neural network using Star sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on FFNN using Star ==================')
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
    inputs = [S]

    # Reachability analysis
    outputs = reachExactBFS(F, inputs, lp_solver='gurobi', show=True)
    plot_star(outputs)

    # Specifying a property of an FFNN
    unsafe_mat = np.array([[1.0, 0]])
    unsafe_vec = np.array([-2.0])

    

    print('=============== DONE: Exact Reachability Analysis on FFNN using Star =====================')
    print('==========================================================================================\n\n')



if __name__ == '__main__':
    """
    Main function to run the FFNN tutorials
    """
    ffnn_construct_manually()

    ffnn_evaluate_input_vector()
    ffnn_reachability_exact_star()



'''
Steps:
Construct a FFNN
Construct the input set
FFNN reachability
safety sepecifictaion
verification
plot
'''