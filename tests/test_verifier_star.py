"""
Test verifier for star reachability
Author: Yuntao Li
Date: 3/22/2024
"""

from StarV.net.network import NeuralNetwork
from StarV.verifier.verifier_star import reachExactBFS, checkSafetyStar
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.set.star import Star
import numpy as np
import multiprocessing
import time
from StarV.util.plot import plot_star
from matplotlib import pyplot as plt
from StarV.set.star import Star
from tabulate import tabulate
import os
from StarV.util.print_util import print_util

class Test(object):
    """
    Testing module net class and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_reachExactBFS(self):

        self.n_tests = self.n_tests + 2
        W = np.eye(2)
        b = np.zeros(2,)
        L1 = fullyConnectedLayer(W, b)
        L2 = ReLULayer()
        layers = []
        layers.append(L1)
        layers.append(L2)
        net = NeuralNetwork(layers, 'ffnn')

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        In = Star(lb, ub)

        print('Test reachExactBFS method ...')

        inputSet = []
        inputSet.append(In)
        
        try:
            print('Test without parallel computing...')
            S = reachExactBFS(net=net, inputSet=inputSet)
            print('Number of input sets: {}'.format(len(inputSet)))
            print('Number of output sets: {}'.format(len(S)))
            # for i in range(0, len(S)):
            #     S[i].__str__()
        except Exception:
            print('Test Fail!\n')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!\n')

        try:
            print('Test with parallel computing...')
            pool = multiprocessing.Pool(2)
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            print('Number of input sets: {}'.format(len(inputSet)))
            print('Number of output sets: {}'.format(len(S)))
            pool.close()
            # for i in range(0, len(S)):
            #     S[i].__str__()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_checkSafetyStar(self):

        self.n_tests = self.n_tests + 1
        print('Test intersectWithUnsafeRegion method...')
        
        
        try:
            lb = -np.random.rand(3,)
            ub = np.random.rand(3,)
            V = np.random.rand(3, 4)
            C = np.random.rand(3, 3)
            d = np.random.rand(3,)
            S = Star(V, C, d, lb, ub)
            unsafe_mat = np.random.rand(2, 3,)
            unsafe_vec = np.random.rand(2,)
            P = checkSafetyStar(unsafe_mat, unsafe_vec, S)
            S.__str__()
            if isinstance(P, Star):
                print('\nUnsafe Set')
                P.__str__()
            else:
                print('\nSafe')
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


if __name__ == "__main__":

    test_verifier = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    print_util('h1')
    test_verifier.test_reachExactBFS()
    print_util('h1')
    test_verifier.test_checkSafetyStar()
    print_util('h1')

    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing verifier: fails: {}, successfull: {}, \
    total tests: {}'.format(test_verifier.n_fails,
                            test_verifier.n_tests - test_verifier.n_fails,
                            test_verifier.n_tests))
