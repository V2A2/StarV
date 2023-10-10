"""
Sparse Star Class
Sung Woo Choi, 05/23/2023

"""

# !/usr/bin/python3
import copy
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

class SpStar(object):
    """
        SpStar Class (sparse star with constraint matrix in weyr matrix formation) for reachability
        author: Sung Woo Choi
        date: 05/23/2023
        Representation of a SpStar
        ===========================================================================================================================
        SpStar set defined by

        ===========================================================================================================================
    """

    def __init__(self, *args):
        """
            Key Attributes:
            A = [] % independent basis matrix
            pL = [] % a list of block primary lower constraint matrices and vector 
            pU = [] % a list of block primary upper constraint matrices and vector 
            sL = [] % a list of block secondary lower constraint matrices and vector 
            sU = [] % a list of block secondary upper constraint matrices and vector 
            Cstr = [] % shape of block constraints matrices; a list of arrays, which contain shape of block constraint matrices
            dim = 0 % dimension of the sparse star set; in other words, number of independent predicate variables
            nVars = 0 % number of predicate varibales
            nZvars = 0 % number of dependent predicate variables
            pred_lb = [] % lower bound vector of predicate variables
            pred_ub = [] % upper bound vector of predicate variables
            T = [] % an  upper traiangular adjacent matrix for block constraint matrices
        """

        len_ = len(args)
        if len_ == 9:
            [A, pL, pU, sL, sU, pred_lb, pred_ub, T, Cstr] = copy.deepcopy(args)

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

    def __str__(self, toDense=None):
        print('SparseStar Set:')
        print('A: \n{}'.format(self.A))
        # if toFull:
            
        #     print('C_{}: \n{}'.format(self.C.getformat(), self.C.todense()))
        # else:
        #     print('C: {}'.format(self.C))
        if toDense is None:
            print('pL: {}'.format(self.pL))
            print('pU: {}'.format(self.pU))
            print('sL: {}'.format(self.sL))
            print('sU: {}'.format(self.sU))
        elif toDense is True:
            print('pL: {}'.format(self.C().todense()))
        elif toDense is False:
            print('pL: {}'.format(self.C()))

        print('d: {}'.format(self.d))
        print('pred_lb: {}'.format(self.pred_lb))
        print('pred_ub: {}'.format(self.pred_ub))
        print('dim: {}'.format(self.dim))
        print('nVars: {}'.format(self.nVars))
        print('nZVars: {}'.format(self.nZVars))
        print('T: {}'.format(self.T))
        print('Cstr: {}'.format(self.Cstr))

        return '\n'

    def __repr__(self):
        print('SparseStar Set:')
        print('A: {}'.format(self.A.shape))
        print('C: {}'.format(self.C.shape))
        print('d: {}'.format(self.d.shape))
        print('pred_lb: {}'.format(self.pred_lb.shape))
        print('pred_ub: {}'.format(self.pred_ub.shape))
        print('dim: {}'.format(self.dim))
        print('nVars: {}'.format(self.nVars))
        print('nZVars: {}'.format(self.nZVars))
        print('T: {}'.format(self.T))
        print('Cstr: {}'.format(self.Cstr))
        return '\n'

    def c(self, index=None):
        """Gets center vector of SparseStar"""
        if index is None:
            return copy.deepcopy(self.A[:, 0].reshape(-1, 1))
        else:
            return copy.deepcopy(self.A[index, 0].reshape(-1, 1))

    def X(self, row=None):
        """Gets basis matrix of predicate variables corresponding to the current dimension"""
        mA = self.A.shape[1]
        if row is None:
            return copy.deepcopy(self.A[:, 1:mA])
        else:
            return copy.deepcopy(self.A[row, 1:mA])

    def V(self, row=None):
        """Gets basis matrix"""
        mA = self.A.shape[1]
        if row is None:
            return copy.deepcopy(np.column_stack([np.zeros((self.dim, self.nZVars)), self.X()]))
        else:
            if isinstance(row, int) or isinstance(row, np.integer):
                return copy.deepcopy(np.hstack([np.zeros(self.nZVars), self.X(row)]))
            else:
                return copy.deepcopy(np.column_stack([np.zeros((len(row), self.nZVars)), self.X(row)]))
            
    def C(self):
        """Constructs constraint matrix of sparse star from lists of block lower constraint matrices"""
        if len(self.CpL) > 0 and len(self.CpU) > 0:
            str_ = np.array(self.Cstr)
            CL = sp.csc_matrix((str_[:,0].sum(), str_[:,1].sum()))
            CU = copy.deepcopy(CL)
            dL = []
            dU = []

            n = len(self.CpL)
            r = 0
            c = 0
            for i in range(n):
                r_ = str_[i, 0]
                c_ = str_[i, 1]
                CL[r:r_, c:c_] = self.pL[i][:,1:c_]
                CU[r:r_, c:c_] = self.pU[i][:, :c_]
                dL = np.append(dL, self.pL[i][:,0])
                dU = np.append(dU, self.pU[i][:,0])
                r = r_
                c = c_
            print('CL: ', CL)
            print('CU: ', CU)
            print('dL: ', dL)
            print('dU: ', dU)

        # if len(self.CsL) > 0 and len(self.CsU) > 0:
        #     pass

    def d(self):
        pass

    def P(self):
        pass
        # predicate constraints

    def affineMap(self, W=None, b=None):
        """Affine mapping of a sparse star: S = W*self + b"""

        if W is None and b is None:
            return copy.deepcopy(self)

        if W is not None:
            assert isinstance(W, np.ndarray), 'error: \
                the mapping matrix should be a 2D numpy array'
            assert W.shape[1] == self.dim, 'error: \
                inconsistency between mapping matrix and SparseStar dimension'

            A = np.matmul(W, self.A)

        if b is not None:
            assert isinstance(b, np.ndarray), 'error: \
                the offset vector should be a 1D numpy array'
            assert len(b.shape) == 1, 'error: \
                offset vector should be a 1D numpy array'

            if W is not None:
                assert W.shape[0] == b.shape[0], 'error: \
                    inconsistency between mapping matrix and offset'
            else:
                assert b.shape[0] == self.dim, 'error: \
                    inconsistency between offset vector and SparseStar dimension'
                A = copy.deepcopy(self.A)

            A[:, 0] += b

        return SpStar(A, self.pL, self.pU, self.sL, self.sU, self.pred_lb, self.pred_ub, self.T, self.Cstr)
 
    
    def estimateRange(self, index):
        """Estimates the minimum and maximum values of a state x[index]"""

        mA = self.A.shape[1]
        n = self.nVars
        p = n-mA+1

        l = self.pred_lb[p:n]
        u = self.pred_ub[p:n]

        pos_f = np.maximum(self.X(index), 0.0)
        neg_f = np.minimum(self.X(index), 0.0)

        xmin = self.c(index).flatten() + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        xmax = self.c(index).flatten() + np.matmul(neg_f, l) + np.matmul(pos_f, u)
        return xmin, xmax

    def estimateRanges(self):
        """Estimates the lower and upper bounds of x"""

        mA = self.A.shape[1]
        n = self.nVars
        p = n-mA+1

        l = self.pred_lb[p:n]
        u = self.pred_ub[p:n]

        pos_f = np.maximum(self.X(), 0.0)
        neg_f = np.minimum(self.X(), 0.0)

        xmin = self.c().flatten() + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        xmax = self.c().flatten() + np.matmul(neg_f, l) + np.matmul(pos_f, u)
        return xmin, xmax