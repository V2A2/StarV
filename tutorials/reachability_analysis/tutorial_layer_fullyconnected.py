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
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.util.plot import plot_star, plot_Mesh3D_Star, plot_3D_Star

def fullyconnected_construct_with_weight_and_bias():
    """
    Construct a fully connected layer with specified weight and bias
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Fully Connected Layer Construction with Weight and Bias =========')
    W = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b = np.array([0.5, 1.0, -0.5])
    layer = [W, b]
    L = FullyConnectedLayer(layer)
    print('Fully connected layer with weight and bias:')
    print('Weight matrix: \n', L.W)
    print('Bias vector:', L.b)
    print('Input dimension:', L.in_dim)
    print('Output dimension:', L.out_dim)

    print('=============== DONE: Fully Connected Layer Construction with Weight and Bias ============')
    print('==========================================================================================\n\n')


def fullyconnected_construct_with_random_weight_and_bias():
    """
    Construct a fully connected layer with random weight and bias
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Random Fully Connected Layer Construction =======================')
    L = FullyConnectedLayer.rand(2, 3)
    print('Fully connected layer with weight and bias:')
    print('Weight matrix: \n', L.W)
    print('Bias vector:', L.b)
    print('Input dimension:', L.in_dim)
    print('Output dimension:', L.out_dim)

    print('=============== DONE: Random Fully Connected Layer Construction ==========================')
    print('==========================================================================================\n\n')


def fullyconnected_construct_with_torch_layer():
    """
    Construct a fully connected layer using a torch layer
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Fully Connected Layer Construction using Pytorch Layer ==========')
    
    layer = torch.nn.Linear(2, 3)
    L = FullyConnectedLayer(layer)
    print('Fully connected layer with weight and bias:')
    print('Weight matrix: \n', L.W)
    print('Bias vector:', L.b)
    print('Input dimension:', L.in_dim)
    print('Output dimension:', L.out_dim)
    
    print('=============== DONE: Fully Connected Layer Construction using Pytorch Layer =============')
    print('==========================================================================================\n\n')


def fullyconnected_evaluate_input_vector():
    """
    Evaluate an input vector on fully connected layer
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Evaluate Input Vector on Fully Connected Layer ==================')

    x = np.array([1.0, 2.0])
    print('Input vector:', x)

    W = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b = np.array([0.5, 1.0, -0.5])
    layer = [W, b]
    L = FullyConnectedLayer(layer)
    print('Fully connected layer with weight and bias:')
    print('Weight matrix: \n', L.W)
    print('Bias vector:', L.b)
    print('Input dimension:', L.in_dim)
    print('Output dimension:', L.out_dim)

    y = L.evaluate(x)
    print('Output vector:', y)

    print('=============== DONE: Evaluate Input Vector on Fully Connected Layer =====================')
    print('==========================================================================================\n\n')


def fullyconnected_reachability_single_star():
    """
    Conduct reachability analysis on fully connected layer using Star set
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Reachability Analysis on Fully Connected Layer using Star =======')
    W = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b = np.array([0.5, 1.0, -0.5])
    L = FullyConnectedLayer([W, b])

    print('Fully connected layer with weight and bias:')
    print('Weight matrix:', L.W)
    print('Bias vector:', L.b)
    print('Input dimension:', L.in_dim)
    print('Output dimension:', L.out_dim)

    S = Star.rand(2)
    plot_star(S)

    # Construct input set
    lb = np.array([-1.0, -1.0, -1.0])
    ub = np.array([1.0, 1.0, 1.0])
    S = Star(lb, ub)

    S = Star.rand_polytope(3, 10)

    # 3D Plot option 1
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    plot_3D_Star(S, ax=ax)

    # 3D Plot option 2
    plot_Mesh3D_Star(S)

    # Reachability analysis
    # R = L.reach(S)
    # print('Reachable set:')
    # print(R)

    print('=============== DONE: Reachability Analysis on Fully Connected Layer using Star ==========')
    print('==========================================================================================\n\n')



if __name__ == "__main__":
    """
    Main function to run the fully connected layer tutorials.
    """
    fullyconnected_construct_with_weight_and_bias()
    fullyconnected_construct_with_random_weight_and_bias()
    fullyconnected_construct_with_torch_layer()
    fullyconnected_evaluate_input_vector()

    fullyconnected_reachability_single_star()

    """
    fullyconnected_construct_with_weight_and_bias()
    fullyconnected_construct_with_random_weight_and_bias()
    fullyconnected_reachability_single_star()
    fullyconnected_reachability_multiple_stars()
    fullyconnected_reachability_multiple_stars_parallel()

    fullyconnected_reachability_single_probstar()
    fullyconnected_reachability_multiple_probstars()
    fullyconnected_reachability_multiple_probstars_parallel()
    """