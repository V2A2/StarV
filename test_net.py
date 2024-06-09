"""
Test net module
Author: Dung Tran
Date: 9/10/2022
"""

from StarV.net.network import NeuralNetwork, rand_ffnn, reachExactBFS, reachApproxBFS
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.set.probstar import ProbStar
import numpy as np
import multiprocessing

class Test(object):
    """
    Testing module net class and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_rand_ffnn(self):

        self.n_tests = self.n_tests + 1
        arch = [2, 3, 3, 2]
        actvs = ['relu', 'relu', 'relu']
        print('Test rand_ffnn method ...')
        net1 = rand_ffnn(arch, actvs) 
        try:
            net1 = rand_ffnn(arch, actvs)
            net1.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_evaluate(self):

        self.n_tests = self.n_tests + 1
        arch = [2, 3, 3, 2]
        actvs = ['relu', 'relu', 'relu']
        print('Test rand_ffnn method ...')
        input_vec = np.random.rand(2,)
        net1 = rand_ffnn(arch, actvs)
        y = net1.evaluate(input_vec)
        try:
            net1 = rand_ffnn(arch, actvs)
            net1.info()
            y = net1.evaluate(input_vec)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

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
        mu = np.zeros(2,)
        Sig = np.eye(2)

        In = ProbStar(mu, Sig, lb, ub)

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
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

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

    def test_reachApproxBFS(self):

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
        mu = np.zeros(2,)
        Sig = np.eye(2)

        In = ProbStar(mu, Sig, lb, ub)

        print('Test reachApproxBFS method ...')

        inputSet = []
        inputSet.append(In)
        p_filter = 0.1165

        try:
            print('Test without parallel computing...')
            S, p_ignored = reachApproxBFS(net=net, inputSet=inputSet, p_filter=p_filter)
            print('Number of input sets: {}'.format(len(inputSet)))
            print('Number of output sets: {}'.format(len(S)))
            print('Total probability of ignored subsets: {}'.format(p_ignored))
            # for i in range(0, len(S)):
            #     S[i].__str__()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

        try:
            print('Test with parallel computing...')
            pool = multiprocessing.Pool(2)
            S, p_ignored = reachApproxBFS(net=net, inputSet=inputSet, p_filter=p_filter, pool=pool)
            print('Number of input sets: {}'.format(len(inputSet)))
            print('Number of output sets: {}'.format(len(S)))
            print('Total probability of ignored subsets: {}'.format(p_ignored))
            pool.close()
            # for i in range(0, len(S)):
            #     S[i].__str__()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

            

if __name__ == "__main__":

    test_net = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_net.test_rand_ffnn()
    test_net.test_evaluate()
    test_net.test_reachExactBFS()
    test_net.test_reachApproxBFS()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_net.n_fails,
                            test_net.n_tests - test_net.n_fails,
                            test_net.n_tests))
