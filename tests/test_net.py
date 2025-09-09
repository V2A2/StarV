"""
Test net module
Author: Dung Tran
Date: 9/10/2022
"""

from StarV.net.network import NeuralNetwork, rand_ffnn, reachExactBFS, reachApproxBFS
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.LeakyReLULayer import LeakyReLULayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.SatLinsLayer import SatLinsLayer
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.util.plot import plot_probstar
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
        L1 = FullyConnectedLayer([W, b])
        # L2 = ReLULayer()
        # L2 = LeakyReLULayer()
        # L2 = SatLinLayer()
        L2 = SatLinsLayer()
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
            for output_set in S:
                output_set.__str__()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

        # try:
        #     print('Test with parallel computing...')
        #     pool = multiprocessing.Pool(2)
        #     S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
        #     print('Number of input sets: {}'.format(len(inputSet)))
        #     print('Number of output sets: {}'.format(len(S)))
        #     pool.close()
        #     # for i in range(0, len(S)):
        #     #     S[i].__str__()
        # except Exception:
        #     print('Test Fail!')
        #     self.n_fails = self.n_fails + 1
        # else:
        #     print('Test Successfull!')

    def test_reachExactBFS2(self):

        self.n_tests = self.n_tests + 2
        # W = np.eye(2)
        # b = np.zeros(2,)
        # L1 = FullyConnectedLayer([W, b])
        # # L2 = ReLULayer()
        # # L2 = LeakyReLULayer()
        # L2 = SatLinLayer()
        # # L2 = SatLinsLayer()
        # layers = []
        # layers.append(L1)
        # layers.append(L2)

        layers = []
        W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
        b1 = np.array([0.5, 1.0, -0.5])
        L1 = FullyConnectedLayer([W1, b1])
        L2 = SatLinLayer()
        W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
        b2 = np.array([-0.2, -1.0])
        L3 = FullyConnectedLayer([W2, b2])
        
        layers.append(L1)
        layers.append(L2)
        layers.append(L3)
        net = NeuralNetwork(layers, 'ffnn')

        lb = np.array([-2.0, -1.0])
        ub = np.array([2.0, 1.0])
        S = Star(lb, ub)
        mu = 0.5*(S.pred_lb + S.pred_ub)
        a = 2.5 # coefficience to adjust the distribution
        sig = (mu - S.pred_lb)/a
        print('Mean of predicate variables: mu = {}'.format(mu))
        print('Standard deviation of predicate variables: sig = {}'.format(sig))
        Sig = np.diag(np.square(sig))
        print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
        # Sig = 1e-2*np.eye(S.nVars)
        In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
        # plot_probstar(In)

        print('Test reachExactBFS method ...')

        inputSet = []
        inputSet.append(In)

        try:
            print('Test without parallel computing...')
            S = reachExactBFS(net=net, inputSet=inputSet)
            # plot_probstar(S)
            print('Number of input sets: {}'.format(len(inputSet)))
            print('Number of output sets: {}'.format(len(S)))
            for output_set in S:
                output_set.__str__()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

        # try:
        #     print('Test with parallel computing...')
        #     pool = multiprocessing.Pool(2)
        #     S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
        #     print('Number of input sets: {}'.format(len(inputSet)))
        #     print('Number of output sets: {}'.format(len(S)))
        #     pool.close()
        #     # for i in range(0, len(S)):
        #     #     S[i].__str__()
        # except Exception:
        #     print('Test Fail!')
        #     self.n_fails = self.n_fails + 1
        # else:
        #     print('Test Successfull!')


    def test_reachApproxBFS(self):

        self.n_tests = self.n_tests + 2
        W = np.eye(2)
        b = np.zeros(2,)
        L1 = FullyConnectedLayer([W, b])
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
            for output_set in S:
                output_set.__str__()
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
            for output_set in S:
                output_set.__str__()
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
    test_net.test_reachExactBFS2()

    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_net.n_fails,
                            test_net.n_tests - test_net.n_fails,
                            test_net.n_tests))
