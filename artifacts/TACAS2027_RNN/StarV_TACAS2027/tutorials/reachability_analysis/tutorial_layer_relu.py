"""
ReLU Layer Tutorial

This tutorial demonstrates how to construct a ReLU layer and perform reachability analysis on it.

1. Fully connected layer construction

2. Reachability analysis
"""

import copy
import torch
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.ReLULayer import ReLULayer
from StarV.util.plot import plot_star, plot_probstar, plot_Mesh3D_Star, plot_3D_Star, plot_probstar_distribution, plot_probstar_contour


def relu_construct():
    """
    Construct a ReLU layer
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: ReLU Layer Construction =========================================')
    # Construct a ReLU layer
    L_r = ReLULayer()
    L_r.info()
    
    print('=============== DONE: ReLU Layer Construction =============================================')
    print('==========================================================================================\n\n')


def relu_evaluate_input_vector():
    """
    Evaluate a ReLU layer with a given input vector
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Evaluate Input Vector on ReLU Layer =========================')
    # Construct a ReLU layer
    L_r = ReLULayer()
    x = np.array([-1.0, 2.0, -3.0])
    output = L_r.evaluate(x)
    
    print("Input vector:", x)
    print("Output after ReLU layer:", output)
    
    print('=============== DONE: Evaluate Input Vector on ReLU Layer =============================')
    print('==========================================================================================\n\n')

def relu_reachability_exact_star():
    """
    Perform exact reachability analysis on a ReLU layer using Star sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on ReLU Layer using Star =================')
    # Construct a ReLU layer
    L_r = ReLULayer()

    # Construct input set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    S = Star(lb, ub)
    inputs = [S]
    print('\nInput sets (num of sets = {}):'.format(len(inputs)))
    for input in inputs:
        print(input)
    plot_star(inputs)

    # Reachability analysis
    outputs = L_r.reach(inputs, method='exact')
    print('\nOutput sets (num of sets = {}):'.format(len(outputs)))
    for output in outputs:
        print(output)
    plot_star(outputs)

    print('=============== DONE: Exact Reachability Analysis on ReLU Layer using Star ====================')
    print('==========================================================================================\n\n')
    
def relu_reachability_exact_parallel_star():
    """
    Perform exact reachability analysis on a ReLU layer with parallel computing using Star sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on ReLU Layer with Parallel Computing using Star =================')
    # Construct a ReLU layer
    L_r = ReLULayer()

    # Construct input set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    S1 = Star(lb, ub)
    S2 = Star(lb+2, ub+2)
    
    inputs = [S1, S2]
    print('\nInput sets (num of sets = {}):'.format(len(inputs)))
    for input in inputs:
        print(input)
    plot_star(inputs)

    # Reachability analysis
    pool = multiprocessing.Pool(2)
    outputs = L_r.reach(inputs, method='exact', pool=pool)
    print('\nOutput sets (num of sets = {}):'.format(len(outputs)))
    for output in outputs:
        print(output)
    plot_star(outputs)

    print('=============== DONE: Exact Reachability Analysis on ReLU Layer with Parallel Computing using Star ====================')
    print('==========================================================================================\n\n')

def relu_reachability_approx_star():
    """
    Perform approximate reachability analysis on a ReLU layer using Star sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Approx Reachability Analysis on ReLU Layer using Star =================')
    # Construct a ReLU layer
    L_r = ReLULayer()

    # Construct input set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    input = Star(lb, ub)
    print('\nInput set:')
    print(input)
    plot_star(input)

    # Reachability analysis
    output = L_r.reach(input, method='approx')
    print('\nOutput set:')
    print(output)
    plot_star(output)

    print('=============== DONE: Approx Reachability Analysis on ReLU Layer using Star ====================')
    print('==========================================================================================\n\n')
    
def relu_reachability_relaxed_star():
    """
    Perform relaxed reachability analysis on a ReLU layer using Star sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Relaxed Reachability Analysis on ReLU Layer using Star =================')
    # Construct a ReLU layer
    L_r = ReLULayer()

    # Construct input set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    input = Star(lb, ub)
    print('\nInput set:')
    print(input)
    plot_star(input)

    # Reachability analysis
    output = L_r.reach(input, method='approx', RF=0.4)
    print('\nOutput set:')
    print(output)
    plot_star(output)

    print('=============== DONE: Relaxed Reachability Analysis on ReLU Layer using Star ====================')
    print('==========================================================================================\n\n')

def relu_reachability_exact_probstar():
    """
    Perform exact reachability analysis on a ReLU layer using ProbStar sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on ReLU Layer using ProbStar =================')
    # Construct a ReLU layer
    L_r = ReLULayer()

    # Construct input set
    dim = 2
    mu = np.zeros(dim)
    Sig = np.eye(dim)
    pred_lb = -np.ones(dim)
    pred_ub = np.ones(dim)
    P = ProbStar(mu, Sig, pred_lb, pred_ub) 

    inputs = [P]
    print('\nInput sets (num of sets = {}):'.format(len(inputs)))
    for input in inputs:
        print(input)
    plot_probstar(inputs)

    # Reachability analysis
    outputs = L_r.reach(inputs, method='exact')
    print('\nOutput sets (num of sets = {}):'.format(len(outputs)))
    for output in outputs:
        print(output)
    plot_probstar(outputs)

    print('=============== DONE: Exact Reachability Analysis on ReLU Layer using ProbStar ====================')
    print('==========================================================================================\n\n')

def relu_reachability_exact_parallel_probstar():
    """
    Perform exact reachability analysis on a ReLU layer with parallel computing using ProbStar sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on ReLU Layer with Parallel Computing using ProbStar =================')
    # Construct a ReLU layer
    L_r = ReLULayer()

    # Construct input set
    dim = 2
    mu = np.zeros(dim)
    Sig = np.eye(dim)
    pred_lb = -np.ones(dim)
    pred_ub = np.ones(dim)
    P1 = ProbStar(mu, Sig, pred_lb, pred_ub)
    P2 = ProbStar(mu, Sig * 0.5, pred_lb + 2, pred_ub + 2)

    inputs = [P1, P2]
    print('\nInput sets (num of sets = {}):'.format(len(inputs)))
    for input in inputs:
        print(input)
    plot_probstar(inputs)

    # Reachability analysis
    pool = multiprocessing.Pool(2)
    outputs = L_r.reach(inputs, method='exact', pool=pool)
    print('\nOutput sets (num of sets = {}):'.format(len(outputs)))
    for output in outputs:
        print(output)
    plot_probstar(outputs)
    
    print('=============== DONE: Exact Reachability Analysis on ReLU Layer with Parallel Computing using ProbStar ====================')
    print('==========================================================================================\n\n')


if __name__ == "__main__":
    """
    Main function to run the ReLU layer tutorials
    """
    relu_construct()
    relu_evaluate_input_vector()
    relu_reachability_exact_star()
    relu_reachability_exact_parallel_star()
    relu_reachability_approx_star()
    relu_reachability_relaxed_star()
    relu_reachability_exact_probstar()
    relu_reachability_exact_parallel_probstar()
