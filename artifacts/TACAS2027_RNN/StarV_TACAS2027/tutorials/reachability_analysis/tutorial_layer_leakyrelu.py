"""
LeakyReLU Layer Tutorial

This tutorial demonstrates how to construct a LeakyReLU layer and perform reachability analysis on it.

1. LeakyReLU layer construction methods

2. Reachability analysis
"""

import copy
import torch
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.LeakyReLULayer import LeakyReLULayer
from StarV.util.plot import plot_star, plot_probstar, plot_Mesh3D_Star, plot_3D_Star, plot_probstar_distribution, plot_probstar_contour

def leakyrelu_construct():
    """
    Construct a LeakyReLU layer
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: LeakyReLU Layer Construction =========================================')
    # Construct a LeakyReLU layer
    L_lr = LeakyReLULayer()
    L_lr.info()
    
    print('=============== DONE: LeakyReLU Layer Construction =============================================')
    print('==========================================================================================\n\n')

def leakyrelu_evaluate_input_vector():
    """
    Evaluate a LeakyReLU layer with a given input vector
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Evaluate Input Vector on LeakyReLU Layer =========================')
    # Construct a LeakyReLU layer
    L_lr = LeakyReLULayer()
    x = np.array([-1.0, 2.0, -3.0])
    output = L_lr.evaluate(x, gamma=0.1)
    
    print("Input vector:", x)
    print("Output after LeakyReLU layer:", output)
    
    print('=============== DONE: Evaluate Input Vector on LeakyReLU Layer =============================')
    print('==========================================================================================\n\n')

def leakyrelu_reachability_exact_star():
    """
    Perform exact reachability analysis on a LeakyReLU layer using Star sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on LeakyReLU Layer using Star =================')
    # Construct a LeakyReLU layer
    L_lr = LeakyReLULayer()
    
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
    outputs = L_lr.reach(inputs, method='exact')
    print('\nOutput sets (num of sets = {}):'.format(len(outputs)))
    for output in outputs:
        print(output)
    plot_star(outputs)
    
    print('=============== DONE: Exact Reachability Analysis on LeakyReLU Layer using Star =================')
    print('==========================================================================================\n\n')

def leakyrelu_reachability_exact_parallel_star():
    """
    Perform exact reachability analysis on a LeakyReLU layer using Star sets with parallel computation
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on LeakyReLU Layer using Star (Parallel) =================')
    # Construct a LeakyReLU layer
    L_lr = LeakyReLULayer()
    
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
    outputs = L_lr.reach(inputs, method='exact', pool=pool)
    print('\nOutput sets (num of sets = {}):'.format(len(outputs)))
    for output in outputs:
        print(output)
    plot_star(outputs)
    
    print('=============== DONE: Exact Reachability Analysis on LeakyReLU Layer using Star (Parallel) =================')
    print('==========================================================================================\n\n')

def leakyrelu_reachability_exact_probstar():
    """
    Perform exact reachability analysis on a LeakyReLU layer using ProbStar sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on LeakyReLU Layer using ProbStar =================')
    # Construct a LeakyReLU layer
    L_lr = LeakyReLULayer()
    
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
    outputs = L_lr.reach(inputs, method='exact')
    print('\nOutput sets (num of sets = {}):'.format(len(outputs)))
    for output in outputs:
        print(output)
    plot_probstar(outputs)
    
    print('=============== DONE: Exact Reachability Analysis on LeakyReLU Layer using ProbStar =================')
    print('==========================================================================================\n\n')

def leakyrelu_reachability_exact_parallel_probstar():
    """
    Perform exact reachability analysis on a LeakyReLU layer using ProbStar sets with parallel computation
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on LeakyReLU Layer using ProbStar (Parallel) =================')
    # Construct a LeakyReLU layer
    L_lr = LeakyReLULayer()
    
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
    outputs = L_lr.reach(inputs, method='exact', pool=pool)
    print('\nOutput sets (num of sets = {}):'.format(len(outputs)))
    for output in outputs:
        print(output)
    plot_probstar(outputs)
    
    print('=============== DONE: Exact Reachability Analysis on LeakyReLU Layer using ProbStar (Parallel) =================')
    print('==========================================================================================\n\n')


if __name__ == "__main__":
    """
    Main function to run the LeakyReLU layer tutorials
    """
    leakyrelu_construct()
    leakyrelu_evaluate_input_vector()
    # leakyrelu_reachability_exact_star()
    # leakyrelu_reachability_exact_parallel_star()
    leakyrelu_reachability_exact_probstar()
    # leakyrelu_reachability_exact_parallel_probstar()