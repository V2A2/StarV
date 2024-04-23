"""
Test LeakyReLU Class
Author: Yuntao Li
Date: 1/18/2024
"""

from StarV.fun.leakyrelu import LeakyReLU
from StarV.set.probstar import ProbStar
import numpy as np
import multiprocessing
# import ipyparallel as ipp
# import subprocess
from StarV.util.plot import plot_probstar


class Test(object):
    """
    Testing LeakyReLU class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0


    def test_evaluate(self):
        self.n_tests = self.n_tests + 1
        print('\nTesting evaluate method...')
        # x = np.array([[-1], [2]])
        x = np.array([[-1, 2], [-3, -4]])
        gamma = 0.2

        try:
            y = LeakyReLU.evaluate(x, gamma)
            print('\nInput: ')
            print(x)
            print('\nOutput: ')
            print(y)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_stepReach(self):

        self.n_tests = self.n_tests + 1
        print('\nTesting stepReach method...')
        mu = np.array([0.0, 0.0])
        Sig = np.eye(2)
        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        # inputSet = ProbStar(mu, Sig, pred_lb, pred_ub)
        inputSet = ProbStar.rand(2)
        plot_probstar(inputSet)
        
        gamma = 0.1

        try:
            S = LeakyReLU.stepReach(inputSet, 0, gamma)
            plot_probstar(S)
            print('\nInput Set:')
            inputSet.__str__()
            print('\nOutput Set 1:')
            S[0].__str__()
            print('\nOutput Set 2:')
            S[1].__str__()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_stepReachMultiInputs(self):

        self.n_tests = self.n_tests + 1
        print('\nTesting stepReachMultiInputs method...')
        mu = np.array([0.0, 0.0])
        Sig = np.eye(2)
        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        inputSet = ProbStar(mu, Sig, pred_lb, pred_ub)
        In = []
        In.append(inputSet)
        In.append(inputSet)

        gamma = 0.1

        try:
            S = LeakyReLU.stepReachMultipleInputs(In, 1, gamma)
            print('\nNumber of input set = {}'.format(len(In)))
            print('\nNumber of output set = {}'.format(len(S)))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_reachExactSingleInput(self):

        self.n_tests = self.n_tests + 1
        print('\nTesting reachExactSingleInput method...')
        mu = np.array([0.0, 0.0])
        Sig = np.eye(2)
        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        inputSet = ProbStar(mu, Sig, pred_lb, pred_ub)

        gamma = 0.1
        S = LeakyReLU.reachExactSingleInput(inputSet, gamma, 'gurobi')
        try:
            S = LeakyReLU.reachExactSingleInput(inputSet, gamma, 'gurobi')
            print('\nNumber of output set = {}'.format(len(S)))
        except Exception:
            print('\nTest Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_reachExactMultiInputs(self):

        self.n_tests = self.n_tests + 2
        print('\nTesting reachExactMultiInputs method...')
        mu = np.array([0.0, 0.0])
        Sig = np.eye(2)
        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        # inputSet = ProbStar(mu, Sig, pred_lb, pred_ub)
        inputSet = ProbStar.rand(10)
        In = []
        In.append(inputSet)
        In.append(inputSet)
        

        gamma = 0.1

        try:
            print('\n1) using default....')
            S = LeakyReLU.reachExactMultiInputs(In, gamma, 'gurobi')
            print('\nNumber of input sets = {}'.format(len(In)))
            print('\nNumber of output sets = {}'.format(len(S)))
        except Exception:
            print('\nTest Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

        try:
            print('\n1) using multiprocessing package....')
            pool = multiprocessing.Pool(8)
            S = LeakyReLU.reachExactMultiInputs(In, gamma, 'gurobi', pool)
            print('\nNumber of input sets = {}'.format(len(In)))
            print('\nNumber of output sets = {}'.format(len(S)))
            pool.close()
        except Exception:
            print('\nTest Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        
        try:
            print('\n2) using ipyparallel package....')
            # print('\nstart ipcluster...')
            # subprocess.Popen(["ipcluster", "start", "-n=2"])
            # clientIDs = ipp.Client()
            # pool = clientIDs[:]
            # S = LeakyReLU.reachExactMultiInputs(In, 'gurobi', pool)
            # print('\nNumber of input sets = {}'.format(len(In)))
            # print('\nNumber of output sets = {}'.format(len(S)))
            # subprocess.Popen(["ipcluster", "stop"])
        except Exception:
            print('\nTest Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Under testing!')


if __name__ == "__main__":

    test_LeakyReLU = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_LeakyReLU.test_evaluate()
    test_LeakyReLU.test_stepReach()
    test_LeakyReLU.test_stepReachMultiInputs()
    test_LeakyReLU.test_reachExactSingleInput()
    test_LeakyReLU.test_reachExactMultiInputs()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_LeakyReLU.n_fails,
                            test_LeakyReLU.n_tests - test_LeakyReLU.n_fails,
                            test_LeakyReLU.n_tests))
