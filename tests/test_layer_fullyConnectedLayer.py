"""
Test fullyConnectedLayer Class
Author: Dung Tran
Date: 9/9/2022
"""

import numpy as np
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.set.probstar import ProbStar
import multiprocessing

class Test(object):
    """
    Testing fullyConnectedLayer class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_constructor(self):

        self.n_tests = self.n_tests + 1
        W = np.random.rand(2, 3)
        b = np.random.rand(2)
        print('Test fullyConnectedLayer Constructor')

        try:
            fullyConnectedLayer(W, b)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_rand(self):
        self.n_tests = self.n_tests + 1
        print('Test fullyConnectedLayer random method')

        try:
            fullyConnectedLayer.rand(2, 3)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_reachExactSingleInput(self):
        self.n_tests = self.n_tests + 1
        print('Test fullyConnectedLayer reachExactSingleInput method')

        try:
            L = fullyConnectedLayer.rand(2, 3)
            In = ProbStar.rand(2)
            print('Input Set:')
            In.__str__()
            S = L.reachExactSingleInput(In)
            print('Output Set:')
            S.__str__()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_reach(self):
        self.n_tests = self.n_tests + 1
        print('Test fullyConnectedLayer reach method')
        print('Without Parallel Computing')
        
        try:
            L = fullyConnectedLayer.rand(2, 3)
            I1 = ProbStar.rand(2)
            In = []
            In.append(I1)
            In.append(I1)
            print('Number of input sets: {}'.format(len(In)))
            S = L.reach(In)
            print('Number of output Set: {}'.format(len(S)))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

        print('Using Parallel Computing')

        pool = multiprocessing.Pool(2)
        L = fullyConnectedLayer.rand(2, 3)
        I1 = ProbStar.rand(2)
        In = []
        In.append(I1)
        In.append(I1)
        print('Number of input sets: {}'.format(len(In)))
        S = L.reach(inputSet=In, pool=pool)
        print('Number of output Set: {}'.format(len(S)))
        pool.close()
                
        # try:
        #     pool = multiprocessing.Pool(2)
        #     L = fullyConnectedLayer.rand(2, 3)
        #     I1 = ProbStar.rand(2)
        #     In = []
        #     In.append(I1)
        #     In.append(I1)
        #     print('Number of input sets: {}'.format(len(In)))
        #     S = L.reach(inputSet=In, pool=pool)
        #     print('Number of output Set: {}'.format(len(S)))
        #     pool.close()
        # except Exception:
        #     print('Test Fail!')
        #     self.n_fails = self.n_fails + 1
        # else:
        #     print('Test Successfull!')


            
if __name__ == "__main__":
    test_fullyConnectedLayer = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_fullyConnectedLayer.test_constructor()
    test_fullyConnectedLayer.test_rand()
    test_fullyConnectedLayer.test_reachExactSingleInput()
    test_fullyConnectedLayer.test_reach()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_fullyConnectedLayer.n_fails,
                            test_fullyConnectedLayer.n_tests - test_fullyConnectedLayer.n_fails,
                            test_fullyConnectedLayer.n_tests))

    
