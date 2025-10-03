"""
Test verifier module for piecewise linear networks
Author: Yuntao Li
Date: 2/8/2024
"""

from StarV.net.network import NeuralNetwork
from StarV.verifier.verifier import reachExactBFS, checkSafetyStar, checkSafetyProbStar, quantiVerifyExactBFS, quantiVerifyBFS, quantiVerifyMC, quantiVerifyProbStarTL
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.LeakyReLULayer import LeakyReLULayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.SatLinsLayer import SatLinsLayer
from StarV.set.probstar import ProbStar
import numpy as np
import multiprocessing
from StarV.util.load_piecewise import load_2017_IEEE_TNNLS_ReLU, load_ACASXU_ReLU, load_tiny_network_ReLU
from StarV.util.load_piecewise import load_2017_IEEE_TNNLS_LeakyReLU, load_ACASXU_LeakyReLU, load_tiny_network_LeakyReLU
from StarV.util.load_piecewise import load_2017_IEEE_TNNLS_SatLin, load_ACASXU_SatLin, load_tiny_network_SatLin
from StarV.util.load_piecewise import load_2017_IEEE_TNNLS_SatLins, load_ACASXU_SatLins, load_tiny_network_SatLins
import time
from StarV.util.plot import plot_probstar_using_Polytope
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


    def test_checkSafetyProbStar(self):

        print_util('h2')
        print('Test intersectWithUnsafeRegion method...')

        self.n_tests = self.n_tests + 1
        
        try:
            S1 = ProbStar.rand(3)
            S1.__str__()
            unsafe_mat1 = np.random.rand(2, 3,)
            unsafe_vec1 = np.random.rand(2,)
            print(unsafe_mat1, unsafe_vec1)
            P1 = checkSafetyProbStar(unsafe_mat1, unsafe_vec1, S1)
            if isinstance(P1, ProbStar):
                print('\nUnsafe Set')
                P1.__str__()
            else:
                print('\nSafe')
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h3')

        try:
            S2 = ProbStar.rand(2)
            S2.__str__()
            # plot_probstar_using_Polytope(S2)
            unsafe_mat2 = np.random.rand(2, 2,)
            unsafe_vec2 = np.random.rand(2,)
            print(unsafe_mat2, unsafe_vec2)
            [P2, prob] = checkSafetyProbStar(unsafe_mat2, unsafe_vec2, S2)

            if isinstance(P2, ProbStar):
                print('\nUnsafe Set')
                P2.__str__()
                # plot_probstar_using_Polytope(P2)
            else:
                print('\nSafe')
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')


    def test_reachExactBFS_ReLU(self):

        print_util('h2')
        print('Test reachExactBFS_ReLU method ...')

        self.n_tests = self.n_tests + 2
        W = np.eye(2)
        b = np.zeros(2,)
        L1 = FullyConnectedLayer(W, b)
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

        print_util('h3')

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
        print_util('h2')


    def test_reachExactBFS_LeakyReLU(self):
        
        print_util('h2')
        print('Test reachExactBFS_LeakyReLU method ...')

        self.n_tests = self.n_tests + 2
        W = np.eye(2)
        b = np.zeros(2,)
        L1 = FullyConnectedLayer(W, b)
        L2 = LeakyReLULayer()
        layers = []
        layers.append(L1)
        layers.append(L2)
        net = NeuralNetwork(layers, 'ffnn')

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        mu = np.zeros(2,)
        Sig = np.eye(2)

        In = ProbStar(mu, Sig, lb, ub)
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
        print_util('h3')

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
        print_util('h2')


    def test_reachExactBFS_SatLin(self):

        print_util('h2')
        print('Test reachExactBFS_SatLin method ...')
        self.n_tests = self.n_tests + 2
        W = np.eye(2)
        b = np.zeros(2,)
        L1 = FullyConnectedLayer(W, b)
        L2 = SatLinLayer()
        layers = []
        layers.append(L1)
        layers.append(L2)
        net = NeuralNetwork(layers, 'ffnn')

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        mu = np.zeros(2,)
        Sig = np.eye(2)

        In = ProbStar(mu, Sig, lb, ub)
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
        print_util('h3')

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
        print_util('h2')
   

    def test_reachExactBFS_SatLins(self):

        print_util('h2')
        print('Test reachExactBFS_SatLins method ...')
        self.n_tests = self.n_tests + 2
        W = np.eye(2)
        b = np.zeros(2,)
        L1 = FullyConnectedLayer(W, b)
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
        print_util('h3')

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
        print_util('h2')

    
    def test_reach_2017_IEEE_TNNLS_ReLU(self):
        """reachability analysis for 2017 IEEE TNNLS ReLU network"""
        print_util('h2')
        print('Test exact reachability of 2017 IEEE TNNLS ReLU network...')

        self.n_tests = self.n_tests + 1
        
        try:
            lb = np.array([-1.0, -1.0, -1.0])
            ub = np.array([1.0, 1.0, 1.0])
            mu = np.zeros(3,)
            Sig = np.eye(3)
            In = ProbStar(mu, Sig, lb, ub)
            inputSet = []
            inputSet.append(In)
            net = load_2017_IEEE_TNNLS_ReLU()
            net.info()
            pool = multiprocessing.Pool(8)
            start = time.time()
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            pool.close()
            print('Number of output sets: {}'.format(len(S)))
            end = time.time()
            print('Reachability time = {}'.format(end - start))
            print(S[0].__str__())
            # plot_probstar_using_Polytope(S[0])
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')

    
    def test_reach_2017_IEEE_TNNLS_LeakyReLU(self):
        """reachability analysis for 2017 IEEE TNNLS LeakyReLU network"""
        print_util('h2')
        print('Test exact reachability of 2017 IEEE TNNLS LeakyReLU network...')

        self.n_tests = self.n_tests + 1
        
        try:
            lb = np.array([-1.0, -1.0, -1.0])
            ub = np.array([1.0, 1.0, 1.0])
            mu = np.zeros(3,)
            Sig = np.eye(3)
            In = ProbStar(mu, Sig, lb, ub)
            inputSet = []
            inputSet.append(In)
            net = load_2017_IEEE_TNNLS_LeakyReLU()
            net.info()
            pool = multiprocessing.Pool(8)
            start = time.time()
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            pool.close()
            print('Number of output sets: {}'.format(len(S)))
            end = time.time()
            print('Reachability time = {}'.format(end - start))
            print(S[0].__str__())
            # plot_probstar_using_Polytope(S[0])
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')


    def test_reach_2017_IEEE_TNNLS_SatLin(self):
        """reachability analysis for 2017 IEEE TNNLS SatLin network"""
        print_util('h2')
        print('Test exact reachability of 2017 IEEE TNNLS SatLin network...')

        self.n_tests = self.n_tests + 1
        
        try:
            lb = np.array([-1.0, -1.0, -1.0])
            ub = np.array([1.0, 1.0, 1.0])
            mu = np.zeros(3,)
            Sig = np.eye(3)
            In = ProbStar(mu, Sig, lb, ub)
            inputSet = []
            inputSet.append(In)
            net = load_2017_IEEE_TNNLS_SatLin()
            net.info()
            pool = multiprocessing.Pool(8)
            start = time.time()
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            pool.close()
            print('Number of output sets: {}'.format(len(S)))
            end = time.time()
            print('Reachability time = {}'.format(end - start))
            print(S[0].__str__())
            # plot_probstar_using_Polytope(S[0])
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')


    def test_reach_2017_IEEE_TNNLS_SatLins(self):
        """reachability analysis for 2017 IEEE TNNLS SatLins network"""
        print_util('h2')
        print('Test exact reachability of 2017 IEEE TNNLS SatLins network...')

        self.n_tests = self.n_tests + 1
        
        try:
            lb = np.array([-1.0, -1.0, -1.0])
            ub = np.array([1.0, 1.0, 1.0])
            mu = np.zeros(3,)
            Sig = np.eye(3)
            In = ProbStar(mu, Sig, lb, ub)
            inputSet = []
            inputSet.append(In)
            net = load_2017_IEEE_TNNLS_SatLins()
            net.info()
            pool = multiprocessing.Pool(8)
            start = time.time()
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            pool.close()
            print('Number of output sets: {}'.format(len(S)))
            end = time.time()
            print('Reachability time = {}'.format(end - start))
            print(S[0].__str__())
            # plot_probstar_using_Polytope(S[0])
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')


    def test_reach_ACASXU_ReLU(self, x, y, spec_id):
        """Reachability analysis of ACASXU ReLU network"""

        print_util('h2')
        print('Test probabilistic reachability of ACASXU N_{}_{} ReLU network under specification {}...'.format(x, y, spec_id))

        self.n_tests = self.n_tests + 1

        try:
            net, lb, ub, _, _ = load_ACASXU_ReLU(x, y, spec_id)
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            Sig = 0.1*np.eye(S.nVars)
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            pool = multiprocessing.Pool(8)
            start = time.time()
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            pool.close()
            print('Number of output sets: {}'.format(len(S)))
            end = time.time()
            print('Reachability time = {}'.format(end - start))
        
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')


    def test_reach_ACASXU_LeakyReLU(self, x, y, spec_id):
        """Reachability analysis of ACASXU LeakyReLU network"""

        print_util('h2')
        print('Test probabilistic reachability of ACASXU N_{}_{} LeakyReLU network under specification {}...'.format(x, y, spec_id))

        self.n_tests = self.n_tests + 1

        try:
            net, lb, ub, _, _ = load_ACASXU_LeakyReLU(x, y, spec_id)
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            Sig = 0.1*np.eye(S.nVars)
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            pool = multiprocessing.Pool(8)
            start = time.time()
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            pool.close()
            print('Number of output sets: {}'.format(len(S)))
            end = time.time()
            print('Reachability time = {}'.format(end - start))
        
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')


    def test_reach_ACASXU_SatLin(self, x, y, spec_id):
        """Reachability analysis of ACASXU SatLin network"""

        print_util('h2')
        print('Test probabilistic reachability of ACASXU N_{}_{} SatLin network under specification {}...'.format(x, y, spec_id))

        self.n_tests = self.n_tests + 1

        try:
            net, lb, ub, _, _ = load_ACASXU_SatLin(x, y, spec_id)
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            Sig = 0.1*np.eye(S.nVars)
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            pool = multiprocessing.Pool(8)
            start = time.time()
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            pool.close()
            print('Number of output sets: {}'.format(len(S)))
            end = time.time()
            print('Reachability time = {}'.format(end - start))
        
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')


    def test_reach_ACASXU_SatLins(self, x, y, spec_id):
        """Reachability analysis of ACASXU SatLins network"""

        print_util('h2')
        print('Test probabilistic reachability of ACASXU N_{}_{} SatLins network under specification {}...'.format(x, y, spec_id))

        self.n_tests = self.n_tests + 1

        try:
            net, lb, ub, _, _ = load_ACASXU_SatLins(x, y, spec_id)
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            Sig = 0.1*np.eye(S.nVars)
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            pool = multiprocessing.Pool(8)
            start = time.time()
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            pool.close()
            print('Number of output sets: {}'.format(len(S)))
            end = time.time()
            print('Reachability time = {}'.format(end - start))
        
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')


if __name__ == "__main__":

    test_verifier = Test()
    print_util('h1')
    # test_verifier.test_checkSafetyProbStar()

    # test_verifier.test_reachExactBFS_ReLU()
    # test_verifier.test_reachExactBFS_LeakyReLU()
    # test_verifier.test_reachExactBFS_SatLin()
    # test_verifier.test_reachExactBFS_SatLins()
    
    # test_verifier.test_reach_2017_IEEE_TNNLS_ReLU()
    # test_verifier.test_reach_2017_IEEE_TNNLS_LeakyReLU()
    # test_verifier.test_reach_2017_IEEE_TNNLS_SatLin()
    # test_verifier.test_reach_2017_IEEE_TNNLS_SatLins()

    # test_verifier.test_reach_ACASXU_ReLU(2, 1, 1)
    # test_verifier.test_reach_ACASXU_LeakyReLU(2, 1, 1)
    # test_verifier.test_reach_ACASXU_SatLin(2, 1, 1)
    # test_verifier.test_reach_ACASXU_SatLins(2, 1, 1)

    print_util('h1')
    print('Testing verifier: fails: {}, successfull: {}, \
    total tests: {}'.format(test_verifier.n_fails,
                            test_verifier.n_tests - test_verifier.n_fails,
                            test_verifier.n_tests))