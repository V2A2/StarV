"""
Fully Connected Layer Tutorial

This tutorial demonstrates how to construct a fully connected layer and perform reachability analysis on it.

1. Fully connected layer construction methods
    - Using specified weight matrix and bias vector
    - Using random weight matrix and bias vector

2. Reachability analysis
"""

import copy
import torch
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.util.plot import plot_star, plot_probstar, plot_Mesh3D_Star, plot_3D_Star, plot_probstar_distribution, plot_probstar_contour



def fullyconnected_construct_with_weight_and_bias():
    """
    Construct a fully connected layer with specified weight and bias
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Fully Connected Layer Construction with Weight and Bias =========')
    W = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b = np.array([0.5, 1.0, -0.5])
    layer = [W, b]
    L_fc = FullyConnectedLayer(layer)
    L_fc.info()

    print(L_fc)

    print('=============== DONE: Fully Connected Layer Construction with Weight and Bias ============')
    print('==========================================================================================\n\n')


def fullyconnected_construct_with_random_weight_and_bias():
    """
    Construct a fully connected layer with random weight and bias
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Random Fully Connected Layer Construction =======================')
    L_fc = FullyConnectedLayer.rand(2, 3)
    L_fc.info()

    print('=============== DONE: Random Fully Connected Layer Construction ==========================')
    print('==========================================================================================\n\n')


def fullyconnected_construct_with_torch_layer():
    """
    Construct a fully connected layer using a torch layer
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Fully Connected Layer Construction using Pytorch Layer ==========')
    
    layer = torch.nn.Linear(2, 3)
    L_fc = FullyConnectedLayer(layer)
    L_fc.info()
    
    print('=============== DONE: Fully Connected Layer Construction using Pytorch Layer =============')
    print('==========================================================================================\n\n')


def fullyconnected_evaluate_input_vector():
    """
    Evaluate an input vector on fully connected layer
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Evaluate Input Vector on Fully Connected Layer ==================')

    x = np.array([1.0, 2.0])

    W = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b = np.array([0.5, 1.0, -0.5])
    layer = [W, b]
    L_fc = FullyConnectedLayer(layer)
    L_fc.info()

    y = L_fc.evaluate(x)

    print('Input vector:', x)
    print('Output vector:', y)

    print('=============== DONE: Evaluate Input Vector on Fully Connected Layer =====================')
    print('==========================================================================================\n\n')


def fullyconnected_reachability_star():
    """
    Conduct reachability analysis on fully connected layer using Star set
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Reachability Analysis on Fully Connected Layer using Star =======')
    W = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b = np.array([0.5, 1.0, -0.5])
    layer = [W, b]
    L_fc = FullyConnectedLayer(layer)
    L_fc.info()


    # Construct input set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    S = Star(lb, ub)
    I = [S]
    print('\nInput (num of sets = {}):'.format(len(I)))
    for I_i in I:
        print(I_i)
    plot_star(I)

    # Reachability analysis
    R = L_fc.reach(I)
    print('\nOutput (num of sets = {}):'.format(len(R)))
    for R_i in R:
        print(R_i)
    plot_3D_Star(R, qhull_option='QJ')

    print('=============== DONE: Reachability Analysis on Fully Connected Layer using Star ==========')
    print('==========================================================================================\n\n')


def fullyconnected_reachability_parallel_star():
    """
    Conduct reachability analysis on fully connected layer with parallel computing using Star set
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Reachability Analysis on Fully Connected Layer with Parallel Computing using Star =======')
    W = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b = np.array([0.5, 1.0, -0.5])
    layer = [W, b]
    L_fc = FullyConnectedLayer(layer)
    L_fc.info()

    # Construct input set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    S1 = Star(lb, ub)
    S2 = Star(lb+2, ub+2)

    I = [S1, S2]
    print('\nInput (num of sets = {}):'.format(len(I)))
    for I_i in I:
        print(I_i)
    plot_star(I)

    # Reachability analysis
    numCores = 2
    pool = multiprocessing.Pool(numCores)
    R = L_fc.reach(I, pool=pool)
    print('\nOutput (num of sets = {}):'.format(len(R)))
    for R_i in R:
        print(R_i)
    plot_3D_Star(R, qhull_option='QJ')

    print('=============== DONE: Reachability Analysis on Fully Connected Layer with Parallel Computing using Star ==========')
    print('==========================================================================================\n\n')


def fullyconnected_reachability_probstar():
    """
    Conduct reachability analysis on fully connected layer using ProbStar set
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Reachability Analysis on Fully Connected Layer using ProbStar =======')
    W = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b = np.array([0.5, 1.0, -0.5])
    layer = [W, b]
    L_fc = FullyConnectedLayer(layer)
    L_fc.info()

    # Construct input set
    dim = 2
    mu = np.zeros(dim)
    Sig = np.eye(dim)
    pred_lb = -np.ones(dim)
    pred_ub = np.ones(dim)
    P = ProbStar(mu, Sig, pred_lb, pred_ub) 

    I = [P]
    print('\nInput (num of sets = {}):'.format(len(I)))
    for I_i in I:
        print(I_i)
    # plot_probstar(I)

    # Reachability analysis
    R = L_fc.reach(I)
    print('\nOutput (num of sets = {}):'.format(len(R)))
    for R_i in R:
        print(R_i)
    # plot_3D_Star(R, qhull_option='QJ')

    print('=============== DONE: Reachability Analysis on Fully Connected Layer using ProbStar ==========')
    print('==========================================================================================\n\n')


def fullyconnected_reachability_parallel_probstar():
    """
    Conduct reachability analysis on fully connected layer with parallel computing using ProbStar set
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Reachability Analysis on Fully Connected Layer with Parallel Computing using ProbStar =======')
    W = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b = np.array([0.5, 1.0, -0.5])
    layer = [W, b]
    L_fc = FullyConnectedLayer(layer)
    L_fc.info()

    # Construct input set
    dim = 2
    mu = np.zeros(dim)
    Sig = np.eye(dim)
    pred_lb = -np.ones(dim)
    pred_ub = np.ones(dim)
    P1 = ProbStar(mu, Sig, pred_lb, pred_ub)
    P2 = ProbStar(mu, Sig * 0.5, pred_lb * 2, pred_ub * 2)

    I = [P1, P2]
    print('\nInput (num of sets = {}):'.format(len(I)))
    for I_i in I:
        print(I_i)
    # plot_probstar(I)

    # Reachability analysis
    numCores = 2
    pool = multiprocessing.Pool(numCores)
    R = L_fc.reach(I, pool=pool)
    print('\nOutput (num of sets = {}):'.format(len(R)))
    for R_i in R:
        print(R_i)
        # plot_3D_Star(output, qhull_option='QJ')

    print('=============== DONE: Reachability Analysis on Fully Connected Layer with Parallel Computing using ProbStar ==========')
    print('==========================================================================================\n\n')


if __name__ == "__main__":
    """
    Main function to run the fully connected layer tutorials.
    """
    fullyconnected_construct_with_weight_and_bias()
    fullyconnected_construct_with_random_weight_and_bias()
    fullyconnected_construct_with_torch_layer()
    fullyconnected_evaluate_input_vector()

    fullyconnected_reachability_star()
    fullyconnected_reachability_parallel_star()

    fullyconnected_reachability_probstar()
    fullyconnected_reachability_parallel_probstar()