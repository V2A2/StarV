"""
Test SparseStar methods
Last update: 01/12/2025
Author: Anonymous
"""


import copy
import numpy as np
import scipy.sparse as sp
from StarV.set.sparsestar import SparseStar


class Test(object):
    """
       Testing SparseStar class methods
    """
    
    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0
        
    def test_constructor(self):
        
        self.n_tests = self.n_tests + 1

        dim = 2

        c = np.zeros([2, 1])
        v = np.eye(dim)
        A = np.hstack([c, v])

        C = np.array([
            [-0.22647,  0.06832, -1,  0],
            [ 0.39032,  0.03921,  0, -1],
            [ 0.22647, -0.06832,  1,  0],
            [-0.39032, -0.03921,  0,  1]
        ])
        C = sp.csc_array(C)
        d = np.array([0.3890, 0.1199, 0.5516, 0.1285])
        pred_lb = np.array([-1, -1, -0.68382, -0.54943])
        pred_ub = np.array([ 1,  1,  0.84647,  0.55803])
        pred_depth = np.array([1, 1, 0, 0])
        
        
        print('Testing SparseStar Constructor...')
        try:
            SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)
        except Exception:
            print("Fail in constructing SparseStar object with \
            len(args)= {}".format(6))
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_affine_mapping(self):

        self.n_tests = self.n_tests + 1

        dim = 2

        lb = -np.ones(dim)  # lower bounds: x1 >= -1, x2 >= -1
        ub = np.ones(dim)   # upper bounds: x1 <= 1, x2 <= 1
        S = SparseStar(lb, ub)
        
        A = np.array([[1.0, 0.5], [-0.5, 1.0]])
        b = -np.ones(dim)
        
        print('Testing SparseStar affineMap()...')
        try:
            S.affineMap(A, b)
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")
    
    def test_minkowski_sum(self):

        self.n_tests = self.n_tests + 1

        dim = 2

        lb = -np.ones(dim)  # lower bounds: x1 >= -1, x2 >= -1
        ub = np.ones(dim)   # upper bounds: x1 <= 1, x2 <= 1
        S1 = SparseStar(lb, ub)
        
        W = np.array([[0.5, 0.5],
                    [0.5, -0.5]])
        b = np.array([0, 0])
        S2 = S1.affineMap(W, b)
        
        print('Testing SparseStar minKowskiSum()...')
        try:
            S1.minKowskiSum(S2)
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")
            
    def test_depthReduction(self):
        
        self.n_tests = self.n_tests + 1

        dim = 2
        
        c = np.zeros([2, 1])
        v = np.eye(dim)
        A = np.hstack([c, v])

        C = np.array([
            [-0.22647,  0.06832, -1,  0],
            [ 0.39032,  0.03921,  0, -1],
            [ 0.22647, -0.06832,  1,  0],
            [-0.39032, -0.03921,  0,  1],
            [-0.62899,  0.18975, -1,  0],
            [ 0.52719,  0.05296,  0, -1],
            [ 0.71908, -0.21693,  1,  0],
            [-0.52883, -0.05312,  0,  1]])
        C = sp.csc_array(C)
        d = np.array([0.3890, 0.1199, 0.5516, 0.1285, -0.02773, 0.02212, 0.25219, 0.03252])
        pred_lb = np.array([-1, -1, -0.68382, -0.54943])
        pred_ub = np.array([ 1,  1,  0.84647,  0.55803])
        pred_depth = np.array([1, 1, 0, 0])
        S = SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)
        
        print('Testing SparseStar depthReduction()...')
        try:
            S.depthReduction(DR=1)
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")
            
    def test_getRanges(self):

        self.n_tests = self.n_tests + 1

        dim = 2

        c = np.zeros([2, 1])
        v = np.eye(dim)
        A = np.hstack([c, v])

        C = np.array([
            [-0.22647,  0.06832, -1,  0],
            [ 0.39032,  0.03921,  0, -1],
            [ 0.22647, -0.06832,  1,  0],
            [-0.39032, -0.03921,  0,  1]
        ])
        C = sp.csc_array(C)
        d = np.array([0.3890, 0.1199, 0.5516, 0.1285])
        pred_lb = np.array([-1, -1, -0.68382, -0.54943])
        pred_ub = np.array([ 1,  1,  0.84647,  0.55803])
        pred_depth = np.array([1, 1, 0, 0])
        S = SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)
        
        print('Testing SparseStar getRanges...')
        try:
            l, u = S.getRanges()
            print('State bounds computed with getRanges():')
            print('lower bounds:\n', l)
            print('upper bounds:\n', u)
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")
            
    def test_estimateRanges(self):

        self.n_tests = self.n_tests + 1

        dim = 2

        c = np.zeros([2, 1])
        v = np.eye(dim)
        A = np.hstack([c, v])

        C = np.array([
            [-0.22647,  0.06832, -1,  0],
            [ 0.39032,  0.03921,  0, -1],
            [ 0.22647, -0.06832,  1,  0],
            [-0.39032, -0.03921,  0,  1]
        ])
        C = sp.csc_array(C)
        d = np.array([0.3890, 0.1199, 0.5516, 0.1285])
        pred_lb = np.array([-1, -1, -0.68382, -0.54943])
        pred_ub = np.array([ 1,  1,  0.84647,  0.55803])
        pred_depth = np.array([1, 1, 0, 0])
        S = SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)
        
        print('Testing SparseStar estimateRanges...')
        try:
            l, u = S.estimateRanges()
            print('State bounds computed with estimateRanges():')
            print('lower bounds:\n', l)
            print('upper bounds:\n', u)
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")
        

if __name__ == "__main__":

    test_sparsestar = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_sparsestar.test_constructor()
    test_sparsestar.test_affine_mapping()
    test_sparsestar.test_minkowski_sum()
    test_sparsestar.test_depthReduction()
    test_sparsestar.test_getRanges()
    test_sparsestar.test_estimateRanges()
    
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing SparseStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_sparsestar.n_fails,
                            test_sparsestar.n_tests - test_sparsestar.n_fails,
                            test_sparsestar.n_tests))