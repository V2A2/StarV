"""
Test FullyConnectedLayer Class
Author: Dung Tran
Date: 9/9/2022
"""

import numpy as np
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.set.probstar import ProbStar
import multiprocessing

class Test(object):
    """
    Testing FullyConnectedLayer class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_constructor(self):

        self.n_tests = self.n_tests + 1
        W = np.random.rand(2, 3)
        b = np.random.rand(2)
        print('Test fullFullyConnectedLayeryConnectedLayer Constructor')

        try:
            FullyConnectedLayer([W, b])
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_rand(self):
        self.n_tests = self.n_tests + 1
        print('Test FullyConnectedLayer random method')

        try:
            FullyConnectedLayer.rand(2, 3)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_reachExactSingleInput(self):
        self.n_tests = self.n_tests + 1
        print('Test FullyConnectedLayer reachExactSingleInput method')

        try:
            L = FullyConnectedLayer.rand(2, 3)
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
        print('Test FullyConnectedLayer reach method')
        print('Without Parallel Computing')
        
        try:
            L = FullyConnectedLayer.rand(2, 3)
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
        L = FullyConnectedLayer.rand(2, 3)
        I1 = ProbStar.rand(2)
        In = []
        In.append(I1)
        In.append(I1)
        print('Number of input sets: {}'.format(len(In)))
        S = L.reach(inputSet=In,pool=pool)
        print('Number of output Set: {}'.format(len(S)))
        pool.close()
                
        # try:
        #     pool = multiprocessing.Pool(2)
        #     L = FullyConnectedLayer.rand(2, 3)
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
    test_FullyConnectedLayer = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_FullyConnectedLayer.test_constructor()
    test_FullyConnectedLayer.test_rand()
    test_FullyConnectedLayer.test_reachExactSingleInput()
    test_FullyConnectedLayer.test_reach()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing FullyConnectedLayer Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_FullyConnectedLayer.n_fails,
                            test_FullyConnectedLayer.n_tests - test_FullyConnectedLayer.n_fails,
                            test_FullyConnectedLayer.n_tests))

    
