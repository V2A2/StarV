"""
Test PosLin Class
Author: Dung Tran
Date: 8/30/2022
"""

from StarV.fun.poslin import PosLin
from StarV.set.probstar import ProbStar
import numpy as np
import multiprocessing
# import ipyparallel as ipp
# import subprocess


class Test(object):
    """
    Testing PosLin class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_stepReach(self):

        self.n_tests = self.n_tests + 1
        print('\nTesting stepReach method...')
        mu = np.array([0.0, 0.0])
        Sig = np.eye(2)
        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        inputSet = ProbStar(mu, Sig, pred_lb, pred_ub)

        try:
            S = PosLin.stepReach(inputSet, 1)
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

        try:
            S = PosLin.stepReachMultiInputs(In, 1)
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
        S = PosLin.reachExactSingleInput(inputSet, 'gurobi')
        try:
            S = PosLin.reachExactSingleInput(inputSet, 'gurobi')
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
            print('\n1) using multiprocessing package....')
            pool = multiprocessing.Pool(2)
            S = PosLin.reachExactMultiInputs(In, 'gurobi', pool)
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
            # S = PosLin.reachExactMultiInputs(In, 'gurobi', pool)
            # print('\nNumber of input sets = {}'.format(len(In)))
            # print('\nNumber of output sets = {}'.format(len(S)))
            # subprocess.Popen(["ipcluster", "stop"])
        except Exception:
            print('\nTest Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Under testing!')


if __name__ == "__main__":

    test_PosLin = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_PosLin.test_stepReach()
    test_PosLin.test_stepReachMultiInputs()
    test_PosLin.test_reachExactSingleInput()
    test_PosLin.test_reachExactMultiInputs()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_PosLin.n_fails,
                            test_PosLin.n_tests - test_PosLin.n_fails,
                            test_PosLin.n_tests))
