"""
Test SparseImageStar methods
Last update: 01/12/2025
Author: Anonymous
"""

import numpy as np
import scipy.sparse as sp
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR




class Test(object):
    """
       Testing SparseImageStar2DCOO and SparseImageStar2DCSR class methods
    """
    
    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0
        
    def test_csr_constructor(self):

        self.n_tests = self.n_tests + 1

        c = np.array([[2, 3], [8, 0]])
        a = np.zeros([2, 2])
        a[0, 0] = 2
        a[1, 0] = 2
        a[0, 1] = 2
        lb = c - a
        ub = c + a
        print('Testing SparseImageStar2DCSR Constructor...')
        try:
            SparseImageStar2DCSR(lb, ub)
        except Exception:
            print("Fail in constructing SparseImageStar2DCSR object with \
            len(args)= {}".format(2))
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")
        
    def test_coo_constructor(self):

        self.n_tests = self.n_tests + 1

        c = np.array([[2, 3], [8, 0]])
        a = np.zeros([2, 2])
        a[0, 0] = 2
        a[1, 0] = 2
        a[0, 1] = 2
        lb = c - a
        ub = c + a
        print('Testing SparseImageStar2DCOO Constructor...')
        try:
            SparseImageStar2DCOO(lb, ub)
        except Exception:
            print("Fail in constructing SparseImageStar2DCOO object with \
            len(args)= {}".format(2))
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")
        
    def test_csr_getRanges(self):

        self.n_tests = self.n_tests + 1

        c = np.array([[2, 3], [8, 0]])
        a = np.zeros([2, 2])
        a[0, 0] = 2
        a[1, 0] = 2
        a[0, 1] = 2
        lb = c - a
        ub = c + a
        SIM_csr = SparseImageStar2DCSR(lb, ub)
        print('Testing SparseImageStar2DCSR getRanges...')
        try:
            l, u = SIM_csr.getRanges()
            
            H, W, C = SIM_csr.shape
            print('Actual state bounds of SIM_csr:')
            print('lower bounds:\n', lb.reshape(H, W))
            print('upper bounds:\n', ub.reshape(H, W))
            print()
            
            print('State bounds computed with getRanges() via LP solver:')
            print('lower bounds:\n', l.reshape(H, W))
            print('upper bounds:\n', u.reshape(H, W))
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")
            
    def test_coo_getRanges(self):

        self.n_tests = self.n_tests + 1

        c = np.array([[2, 3], [8, 0]])
        a = np.zeros([2, 2])
        a[0, 0] = 2
        a[1, 0] = 2
        a[0, 1] = 2
        lb = c - a
        ub = c + a
        SIM_coo = SparseImageStar2DCOO(lb, ub)
        print('Testing SparseImageStar2DCOO getRanges...')
        try:
            l, u = SIM_coo.getRanges()
            
            H, W, C = SIM_coo.shape
            print('Actual state bounds of SIM_coo:')
            print('lower bounds:\n', lb.reshape(H, W))
            print('upper bounds:\n', ub.reshape(H, W))
            print()
            
            print('State bounds computed with getRanges() via LP solver:')
            print('lower bounds:\n', l.reshape(H, W))
            print('upper bounds:\n', u.reshape(H, W))
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")
               
    def test_csr_estimateRanges(self):

        self.n_tests = self.n_tests + 1

        c = np.array([[2, 3], [8, 0]])
        a = np.zeros([2, 2])
        a[0, 0] = 2
        a[1, 0] = 2
        a[0, 1] = 2
        lb = c - a
        ub = c + a
        SIM_csr = SparseImageStar2DCSR(lb, ub)
        print('Testing SparseImageStar2DCSR estimateRanges...')
        try:
            l, u = SIM_csr.estimateRanges()
            
            H, W, C = SIM_csr.shape
            print('Actual state bounds of SIM_csr:')
            print('lower bounds:\n', lb.reshape(H, W))
            print('upper bounds:\n', ub.reshape(H, W))
            print()
            
            print('State bounds computed with estimateRanges() via LP solver:')
            print('lower bounds:\n', l.reshape(H, W))
            print('upper bounds:\n', u.reshape(H, W))
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")
            
    def test_coo_estimateRanges(self):

        self.n_tests = self.n_tests + 1

        c = np.array([[2, 3], [8, 0]])
        a = np.zeros([2, 2])
        a[0, 0] = 2
        a[1, 0] = 2
        a[0, 1] = 2
        lb = c - a
        ub = c + a
        SIM_coo = SparseImageStar2DCOO(lb, ub)
        print('Testing SparseImageStar2DCOO estimateRanges...')
        try:
            l, u = SIM_coo.estimateRanges()
            
            H, W, C = SIM_coo.shape
            print('Actual state bounds of SIM_coo:')
            print('lower bounds:\n', lb.reshape(H, W))
            print('upper bounds:\n', ub.reshape(H, W))
            print()
            
            print('State bounds computed with estimateRanges() via LP solver:')
            print('lower bounds:\n', l.reshape(H, W))
            print('upper bounds:\n', u.reshape(H, W))
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")
    
        
if __name__ == "__main__":

    test_sparseimagestar = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_sparseimagestar.test_csr_constructor()
    test_sparseimagestar.test_coo_constructor()
    test_sparseimagestar.test_csr_getRanges()
    test_sparseimagestar.test_coo_getRanges()
    test_sparseimagestar.test_csr_estimateRanges()
    test_sparseimagestar.test_coo_estimateRanges()
    
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing SparseImageStar Classes: fails: {}, successfull: {}, \
    total tests: {}'.format(test_sparseimagestar.n_fails,
                            test_sparseimagestar.n_tests - test_sparseimagestar.n_fails,
                            test_sparseimagestar.n_tests))
