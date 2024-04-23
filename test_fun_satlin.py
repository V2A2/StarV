"""
Test SatLin Class
Author: Yuntao Li
Date: 1/18/2024
"""

from StarV.fun.satlin import SatLin
from StarV.set.probstar import ProbStar
import numpy as np
import multiprocessing
from StarV.util.plot import plot_probstar, plot_probstar_using_Polytope
# import ipyparallel as ipp
# import subprocess


class Test(object):
    """
    Testing SatLin class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0


    def test_evaluate(self):
        self.n_tests = self.n_tests + 1
        print('\nTesting evaluate method...')
        # x = np.array([[-1], [2]])
        # x = np.array([[-1, 2], [-3, -4]])
        x= np.array([[-1], [0.5], [2]])

        try:
            y = SatLin.evaluate(x)
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
        V = np.array([[0.5,1.5,0],[0,0,2]])
        C = np.array([])
        d = np.array([])
        mu = np.array([0.0, 0.0])
        Sig = np.eye(2)
        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        inputSet = ProbStar(V, C, d, mu, Sig, pred_lb, pred_ub)
        plot_probstar_using_Polytope(inputSet)
        # plot_probstar(inputSet)
        # inputSet.plot()

        try:
            S = SatLin.stepReach(inputSet, 0)
            print('\nInput Set:') 
            inputSet.__str__()
            print('\nOutput Set 1:')
            S[0].__str__()
            print('\nOutput Set 2:')
            S[1].__str__()
            print('\nOutput Set 3:')
            S[2].__str__()
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

        try:
            S = SatLin.stepReachMultiInputs(In, 1)
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
        S = SatLin.reachExactSingleInput(inputSet, 'gurobi')
        try:
            S = SatLin.reachExactSingleInput(inputSet, 'gurobi')
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
        inputSet = ProbStar(mu, Sig, pred_lb, pred_ub)
        In = []
        In.append(inputSet)
        In.append(inputSet)

        try:
            print('\n1) using default....')
            S = SatLin.reachExactMultiInputs(In, 'gurobi')
            print('\nNumber of input sets = {}'.format(len(In)))
            print('\nNumber of output sets = {}'.format(len(S)))
        except Exception:
            print('\nTest Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

        try:
            print('\n2) using multiprocessing package....')
            pool = multiprocessing.Pool(2)
            S = SatLin.reachExactMultiInputs(In, 'gurobi', pool)
            print('\nNumber of input sets = {}'.format(len(In)))
            print('\nNumber of output sets = {}'.format(len(S)))
            pool.close()
        except Exception:
            print('\nTest Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
                
        try:
            print('\n3) using ipyparallel package....')
            # print('\nstart ipcluster...')
            # subprocess.Popen(["ipcluster", "start", "-n=2"])
            # clientIDs = ipp.Client()
            # pool = clientIDs[:]
            # S = SatLin.reachExactMultiInputs(In, 'gurobi', pool)
            # print('\nNumber of input sets = {}'.format(len(In)))
            # print('\nNumber of output sets = {}'.format(len(S)))
            # subprocess.Popen(["ipcluster", "stop"])
        except Exception:
            print('\nTest Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Under testing!')


if __name__ == "__main__":

    test_SatLin = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_SatLin.test_evaluate()
    test_SatLin.test_stepReach()
    test_SatLin.test_stepReachMultiInputs()
    test_SatLin.test_reachExactSingleInput()
    test_SatLin.test_reachExactMultiInputs()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_SatLin.n_fails,
                            test_SatLin.n_tests - test_SatLin.n_fails,
                            test_SatLin.n_tests))
