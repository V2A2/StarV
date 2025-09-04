"""
SatLins Layer Tutorial

This tutorial demonstrates how to construct a SatLins layer and perform reachability analysis on it.

1. SatLins layer construction methods

2. Reachability analysis
"""

import copy
import torch
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.SatLinsLayer import SatLinsLayer
from StarV.util.plot import plot_star, plot_probstar, plot_Mesh3D_Star, plot_3D_Star, plot_probstar_distribution, plot_probstar_contour

def satlins_construct():
    """
    Construct a SatLins layer
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: SatLins Layer Construction =========================================')
    # Construct a SatLins layer
    L_sts = SatLinsLayer()
    L_sts.info()
    
    print('=============== DONE: SatLins Layer Construction =============================================')
    print('==========================================================================================\n\n')

def satlins_evaluate_input_vector():
    """
    Evaluate a SatLins layer with a given input vector
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Evaluate Input Vector on SatLins Layer =========================')
    # Construct a SatLins layer
    L_sts = SatLinsLayer()
    x = np.array([-1.5, 2.0, -0.5, 0.5])
    output = L_sts.evaluate(x)
    
    print("Input vector:", x)
    print("Output after SatLins layer:", output)
    
    print('=============== DONE: Evaluate Input Vector on SatLins Layer =============================')
    print('==========================================================================================\n\n')

def satlins_reachability_exact_star():
    """
    Perform exact reachability analysis on a SatLins layer using Star sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on SatLins Layer using Star =================')
    # Construct a SatLins layer
    L_sts = SatLinsLayer()
    
    # Construct input set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    S = Star(lb-1, ub+1)
    inputs = [S]
    print('\nInput sets (num of sets = {}):'.format(len(inputs)))
    for input in inputs:
        print(input)
    plot_star(inputs)

    # Reachability analysis
    outputs = L_sts.reach(inputs, method='exact')
    print('\nOutput sets (num of sets = {}):'.format(len(outputs)))
    for output in outputs:
        print(output)
    plot_star(outputs)
    
    print('=============== DONE: Exact Reachability Analysis on SatLins Layer using Star =================')
    print('==========================================================================================\n\n')

def satlins_reachability_exact_parallel_star():
    """
    Perform exact reachability analysis on a SatLins layer using Star sets with parallel computation
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on SatLins Layer using Star (Parallel) =================')
    # Construct a SatLins layer
    L_sts = SatLinsLayer()
    
    # Construct input set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    S1 = Star(lb, ub)
    S2 = Star(lb+1, ub+1)
    
    inputs = [S1, S2]
    print('\nInput sets (num of sets = {}):'.format(len(inputs)))
    for input in inputs:
        print(input)
    plot_star(inputs)

    # Reachability analysis
    numCores = 2
    pool = multiprocessing.Pool(numCores)
    outputs = L_sts.reach(inputs, method='exact', pool=pool)
    print('\nOutput sets (num of sets = {}):'.format(len(outputs)))
    for output in outputs:
        print(output)
    plot_star(outputs)
    
    print('=============== DONE: Exact Reachability Analysis on SatLins Layer using Star (Parallel) =================')
    print('==========================================================================================\n\n')

def satlins_reachability_exact_probstar():
    """
    Perform exact reachability analysis on a SatLins layer using ProbStar sets
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on SatLins Layer using ProbStar =================')
    # Construct a SatLins layer
    L_sts = SatLinsLayer()
    
    # Construct input set
    dim = 2
    mu = np.zeros(dim)
    Sig = np.eye(dim)
    pred_lb = -np.ones(dim)
    pred_ub = np.ones(dim)
    P = ProbStar(mu, Sig, pred_lb-1, pred_ub+1) 

    inputs = [P]
    print('\nInput sets (num of sets = {}):'.format(len(inputs)))
    for input in inputs:
        print(input)
    plot_probstar(inputs)

    # Reachability analysis
    outputs = L_sts.reach(inputs, method='exact')
    print('\nOutput sets (num of sets = {}):'.format(len(outputs)))
    for output in outputs:
        print(output)
    plot_probstar(outputs)
    
    print('=============== DONE: Exact Reachability Analysis on SatLins Layer using ProbStar =================')
    print('==========================================================================================\n\n')

def satlins_reachability_exact_parallel_probstar():
    """
    Perform exact reachability analysis on a SatLins layer using ProbStar sets with parallel computation
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Exact Reachability Analysis on SatLins Layer using ProbStar (Parallel) =================')
    # Construct a SatLins layer
    L_sts = SatLinsLayer()
    
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
    numCores = 2
    pool = multiprocessing.Pool(numCores)
    outputs = L_sts.reach(inputs, method='exact', pool=pool)
    print('\nOutput sets (num of sets = {}):'.format(len(outputs)))
    for output in outputs:
        print(output)
    plot_probstar(outputs)
    
    print('=============== DONE: Exact Reachability Analysis on SatLins Layer using ProbStar (Parallel) =================')
    print('==========================================================================================\n\n')


if __name__ == "__main__":
    """
    Main function to run the SatLins layer tutorials
    """
    satlins_construct()
    satlins_evaluate_input_vector()
    satlins_reachability_exact_star()
    # satlins_reachability_exact_parallel_star()
    satlins_reachability_exact_probstar()
    # satlins_reachability_exact_parallel_probstar()