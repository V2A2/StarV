"""
Sparse Star Class
Sung Woo Choi, 05/23/2023

"""

# !/usr/bin/python3
import copy
import torch
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.linalg import block_diag
import polytope as pc
import glpk
import gurobipy as gp
from gurobipy import GRB
from StarV.set.star import Star

# https://stackabuse.com/courses/graphs-in-python-theory-and-implementation/lessons/representing-graphs-in-code/

class SpStarT(object):
    """
        SpStar Tensor Class (sparse star with constraint matrix in weyr matrix formation) for reachability
        author: Sung Woo Choi
        date: 06/06/2023
        Representation of a SpStar
        ===========================================================================================================================
        SpStar set defined by

        ===========================================================================================================================
    """

    def __init__(self, *args):
        """
            Key Attributes:
            A = [] % independent basis matrix
            pU = [] % a list of block primary upper constraint matrices
            pu = [] % a list of block primary upper constraint vectors
            pL = [] % a list of block primary lower constraint matrices
            pl = [] % a list of block primary lower constraint vectors
            sU = [] % a list of block secondary upper constraint matrices
            su = [] % a list of block secondary upper constraint vectors
            sL = [] % a list of block secondary lower constraint matrices
            sl = [] % a list of block secondary lower constraint vectors
            T = [] % an upper traiangular bidirectional adjacent matrix for block constraint matrices
            D = [] % dimension of block constraints matrices; a list of arrays, which contain dimension (shape) of block constraint matrices

            dim = 0 % dimension of the sparse star set; in other words, number of independent predicate variables
            nPVars = 0 % number of total predicate varibales
            nDvars = 0 % number of dependent predicate variables

            pred_lb = [] % lower bound vector of predicate variables
            pred_ub = [] % upper bound vector of predicate variables
            


            A: [c, IV]
            U: [pU, pu, sU, su]
            L: [pL, pl, sL, sl]
            T: T
            D: D
        """

        len_ = len(args)
        if len_ == 9:
            [A, U, L, T, D, P] = copy.deepcopy(args)

            assert isinstance(A, torch.Tensor), 'independent basis matrix should be a 2D torch tensor'
            assert isinstance(T, sp.coo_matrix), 'adjacent tree should be a 2D scipy sparse matrix in coo format'
            assert isinstance(D, np.ndarray), 'dimension of block constraints matrices should 2D numpy array'

            assert isinstance(pred_lb, torch.Tensor), 'lower bound predicate vector should be a 1D torch tensor'
            assert isinstance(pred_ub, torch.Tensor), 'upper bound predicate vector should be a 1D torch tensor'



            assert len(A.shape) == 2, 'independent basis matrix should be a 2D torch tensor'


            if len(U) > 0:
                assert isinstance(pL, list), 'error: \
                    pL should be a list of block primary lower constraint matrices and vector in a 2D numpy array'
                assert isinstance(pU, list), 'error: \
                    pU should be a list of block primary lower constraint matrices and vector in a 2D numpy array'
                assert isinstance(sL, list), 'error: \
                    sL should be a list of block primary lower constraint matrices and vector in a 2D numpy array'
                assert isinstance(sU, list), 'error: \
                    sU should be a list of block primary lower constraint matrices and vector in a 2D numpy array'
                assert isinstance(pL[0], np.ndarray) and len(pL[0].shape) == 2, 'error: \
                    block primary lower constraint matrices in a 2D numpy array'
                assert isinstance(pU[0], np.ndarray) and len(pU[0].shape) == 2, 'error: \
                    block primary upper constraint matrices in a 2D numpy array'
                assert isinstance(sL[0], np.ndarray) and len(sL[0].shape) == 2, 'error: \
                    block secondary lower constraint matrices in a 2D numpy array'
                assert isinstance(sU[0], np.ndarray) and len(sU[0].shape) == 2, 'error: \
                    block secondary upper constraint matrices in a 2D numpy array'
                assert len(pL) == len(pU) and len(sL) == len(sU) and len(pL) == len(sL), 'error: \
                    number of block constraint matrices should be equivalent in each CpL, Cpu, CsL, CsU'
                assert T.shape[0] == T.shape[1] and T.shape[0] == len(pL), 'error: \
                    number of block constraint matrices should match to number of rows of adjacent matrix'
                assert len(Cstr) == len(pL), 'error: \
                    length of Cstr should match to number of block constraints matrices'         

            assert len(pred_lb.shape) == 1, 'error: \
                lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, 'error: \
                upper bound vector should be a 1D numpy array'
            assert pred_ub.shape[0] == pred_lb.shape[0], 'error: \
                inconsistent number of predicate variables between predicate lower- and upper-boud vectors'

            self.A = A
            self.pL = pL
            self.pU = pU
            self.sL = sL
            self.sU = sU
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.dim = self.A.shape[0]
            self.T = T
            self.Cstr = Cstr
            if len(pL) > 0:
                str_ = np.array(self.Cstr)
                self.nVars = str_[:, 1].sum()
                self.nZVars = self.nVars + 1 - self.A.shape[1]
            else:
                self.nVars = self.A.shape[1] - 1
                self.nZVars = 0 

        elif len_ == 8:
            [A, pL, pU, sL, sU, pred_lb, pred_ub, T] = copy.deepcopy(args)

            assert isinstance(A, np.ndarray), 'error: \
                basis matrix should be a 2D numpy array'
            assert isinstance(pred_lb, np.ndarray), 'error: \
                lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray), 'error: \
                upper bound vector should be a 1D numpy array'
            assert len(A.shape) == 2, 'error: \
                basis matrix should be a 2D numpy array'

            if len(pL) > 0:
                assert isinstance(pL, list), 'error: \
                    pL should be a list of block primary lower constraint matrices in a 2D numpy array'
                assert isinstance(pU, list), 'error: \
                    pU should be a list of block primary lower constraint matrices in a 2D numpy array'
                assert isinstance(sL, list), 'error: \
                    sL should be a list of block primary lower constraint matrices in a 2D numpy array'
                assert isinstance(sU, list), 'error: \
                    sU should be a list of block primary lower constraint matrices in a 2D numpy array'
                assert isinstance(pL[0], np.ndarray) and len(pL[0].shape) == 2, 'error: \
                    block primary lower constraint matrices in a 2D numpy array'
                assert isinstance(pU[0], np.ndarray) and len(pU[0].shape) == 2, 'error: \
                    block primary upper constraint matrices in a 2D numpy array'
                assert isinstance(sL[0], np.ndarray) and len(sL[0].shape) == 2, 'error: \
                    block secondary lower constraint matrices in a 2D numpy array'
                assert isinstance(sU[0], np.ndarray) and len(sU[0].shape) == 2, 'error: \
                    block secondary upper constraint matrices in a 2D numpy array'
                assert len(pL) == len(pU) and len(sL) == len(sU) and len(pL) == len(sL), 'error: \
                    number of block constraint matrices should be equivalent in each CpL, Cpu, CsL, CsU'
                assert T.shape[0] == T.shape[1] and T.shape[0] == len(pL), 'error: \
                    number of block constraint matrices should match to number of rows of adjacent matrix'                

            assert len(pred_lb.shape) == 1, 'error: \
                lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, 'error: \
                upper bound vector should be a 1D numpy array'
            assert pred_ub.shape[0] == pred_lb.shape[0], 'error: \
                inconsistent number of predicate variables between predicate lower- and upper-boud vectors'

            self.A = A
            self.pL = pL
            self.pU = pU
            self.sL = sL
            self.sU = sU
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.dim = self.A.shape[0]
            self.T = T
            self.Cstr = []
            for i in range(len(pL)):
                self.Cstr = self.Cstr.append((pL[i].shape[0], pL[i].shape[0]-1))
            if len(pL) > 0:
                str_ = np.array(self.Cstr)
                self.nVars = str_[:, 1].sum()
                self.nZVars = self.nVars + 1 - self.A.shape[1]
            else:
                self.nVars = self.A.shape[1] - 1
                self.nZVars = 0 

        elif len_ == 2:
            [lb, ub] = copy.deepcopy(args)

            assert isinstance(lb, np.ndarray), 'error: \
                lower bound vector should be a 1D numpy array'
            assert isinstance(ub, np.ndarray), 'error: \
                upper bound vector should be a 1D numpy array'
            assert len(lb.shape) == 1, 'error: \
                lower bound vector should be a 1D numpy array'
            assert len(ub.shape) == 1, 'error: \
                upper bound vector should be a 1D numpy array'

            assert lb.shape[0] == ub.shape[0], 'error: \
                inconsistency between predicate lower bound and upper bound'
            if np.any(ub < lb):
                raise RuntimeError(
                    "The upper bounds must not be less than the lower bounds for all dimensions")

            self.dim = lb.shape[0]
            nv = int(sum(ub > lb))
            self.A = np.zeros((self.dim, nv+1))
            j = 1
            for i in range(self.dim):
                if ub[i] > lb[i]:
                    self.A[i, j] = 1
                    j += 1

            self.pL = []
            self.pU = []
            self.sL = []
            self.sU = []
            self.pred_lb = lb
            self.pred_ub = ub
            self.pred_depth = np.zeros(self.dim)
            self.nVars = self.dim
            self.nZVars = self.dim + 1 - self.A.shape[1]
            self.T = np.array([])
            self.Cstr = []

        else:
            raise Exception(
                'error: invalid number of input arguments (should be 0, 1, 2, 6)'
            )

   