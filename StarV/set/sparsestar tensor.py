"""
Sparse Star Class
Sung Woo Choi, 04/03/2023

"""

# !/usr/bin/python3
import torch
import copy
# import glpk
import gurobipy as gp
from gurobipy import GRB
# import numpy as np
# import scipy.sparse as sp

# from scipy.optimize import linprog
# from scipy.linalg import block_diag
# import polytope as pc


class SparseStar(object):
    """
        SparseStar Class for reachability
        author: Sung Woo Choi
        date: 04/03/2023
        Representation of a SparseStar
        ===========================================================================================================================
        SparseStar set defined by

        ===========================================================================================================================
    """

    def __init__(self, *args):
        """
            Key Attributes:
            A = []; % non-zero basis matrix
            C = []; % constraint matrix
            d = []; % constraint vector
            dim = 0; % dimension of the sparse star set
            nVars = 0; % number of predicate variables
            nZVars = 0; % number of non-basis predicate varaibles
            pred_lb = []; % lower bound of predicate variables
            pred_ub = []; % upper bound of predicate variables
        """

        len_ = len(args)
        if len_ == 6:
            [A, C, d, pred_lb, pred_ub, pred_depth] = copy.deepcopy(args)

            assert isinstance(A, torch.Tensor), 'error: \
                non-zero basis matrix should be a 2D torch tensor'
            assert isinstance(pred_lb, torch.Tensor), 'error: \
                lower bound vector should be a 1D torch tensor'
            assert isinstance(pred_ub, torch.Tensor), 'error: \
                upper bound vector should be a 1D torch tensor'
            assert len(A.shape) == 2, 'error: \
                basis matrix should be a 2D torch tensor'

            if len(C) > 0:
                assert len(C.shape) == 2, 'error: \
                    constraint matrix should be a 2D torch tensor'
                assert len(d.shape) == 1, 'error: \
                    constraint vector should be a 1D torch tensor'
                assert C.shape[0] == d.shape[0], 'error: \
                    inconsistency between constraint matrix and constraint vector'
                assert C.shape[1] == pred_lb.shape[0], 'error: \
                    inconsistent number of predicatve variables between constratint matrix and predicate bound vectors'

            assert len(pred_lb.shape) == 1, 'error: \
                lower bound vector should be a 1D torch tensor'
            assert len(pred_ub.shape) == 1, 'error: \
                upper bound vector should be a 1D torch tensor'
            assert pred_ub.shape[0] == pred_lb.shape[0], 'error: \
                inconsistent number of predicate variables between predicate lower- and upper-boud vectors'
            assert pred_lb.shape[0] == pred_depth.shape[0], 'error: \
                inconsistent number of predicate variables between predicate bounds and predicate depth'

            self.A = A
            self.C = C
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.pred_depth = pred_depth
            self.dim = self.A.shape[0]
            if len(C) > 0:
                self.nVars = self.C.shape[1]
                self.nZVars = self.C.shape[1] + 1 - self.A.shape[1]
            else:
                self.nVars = self.dim
                self.nZVars = self.dim + 1 - self.A.shape[1]

        # elif len_ == 2:
        #     [lb, ub] = copy.deepcopy(args)

        #     assert isinstance(lb, torch.Tensor), 'error: \
        #         lower bound vector should be a 1D torch tensor'
        #     assert isinstance(ub, torch.Tensor), 'error: \
        #         upper bound vector should be a 1D torch tensor'
        #     assert len(lb.shape) == 1, 'error: \
        #         lower bound vector should be a 1D torch tensor'
        #     assert len(ub.shape) == 1, 'error: \
        #         upper bound vector should be a 1D torch tensor'

        #     assert lb.shape[0] == ub.shape[0], 'error: \
        #         inconsistency between predicate lower bound and upper bound'
        #     if torch.any(ub < lb):
        #         raise RuntimeError("The upper bounds must not be less than the lower bounds for all dimensions")

        #     self.dim = lb.shape[0]
        #     nv = int(sum(ub > lb))
        #     c = 0.5 * (lb + ub)
        #     if self.dim == nv:
        #         v = torch.diag(0.5 * (ub - lb))
        #     else:
        #         v = torch.zeros(self.dim, nv)
        #         j = 0
        #         for i in range(self.dim):
        #             if ub[i] > lb[i]:
        #                 v[i, j] = 0.5 * (ub[i] - lb[i])
        #                 j += 1
        #     self.A = torch.column_stack([c, v])

        #     # if dim > 3:
        #     #     self.A = torch.column_stack([c, torch.diag(v)]).to_sparse_csr()
        #     # else:
        #     #     self.A = torch.column_stack([c, torch.diag(v)])

        #     self.C = torch.tensor([])
        #     self.d = torch.tensor([])
        #     self.pred_lb = -torch.ones(self.dim, 1)
        #     self.pred_ub = torch.ones(self.dim, 1)
        #     self.pred_depth = torch.zeros(self.dim)
        #     self.dim = self.dim
        #     self.nVars = self.dim
        elif len_ == 2:
            [lb, ub] = copy.deepcopy(args)

            assert isinstance(lb, torch.Tensor), 'error: \
                lower bound vector should be a 1D torch tensor'
            assert isinstance(ub, torch.Tensor), 'error: \
                upper bound vector should be a 1D torch tensor'
            assert len(lb.shape) == 1, 'error: \
                lower bound vector should be a 1D torch tensor'
            assert len(ub.shape) == 1, 'error: \
                upper bound vector should be a 1D torch tensor'

            assert lb.shape[0] == ub.shape[0], 'error: \
                inconsistency between predicate lower bound and upper bound'
            if torch.any(ub < lb):
                raise RuntimeError(
                    "The upper bounds must not be less than the lower bounds for all dimensions")

            self.dim = lb.shape[0]
            nv = int(sum(ub > lb))
            if self.dim == nv:
                self.A = torch.eye(self.dim)
            else:
                self.A = torch.zeros(self.dim, nv+1)
                j = 1
                for i in range(self.dim):
                    if ub[i] > lb[i]:
                        self.A[i, j] = 1
                        j += 1

            self.C = torch.tensor([])
            self.d = torch.tensor([])
            self.pred_lb = lb
            self.pred_ub = ub
            self.pred_depth = torch.zeros(self.dim)
            self.nVars = self.dim
            self.nZVars = self.dim + 1 - self.A.shape[1]

        elif len_ == 0:
            self.A = torch.tensor([])
            self.C = torch.tensor([])
            self.d = torch.tensor([])
            self.pred_lb = torch.tensor([])
            self.pred_ub = torch.tensor([])
            self.pred_depth = torch.tensor([])
            self.dim = 0
            self.nVars = 0
            self.nZVars = 0

        else:
            raise Exception(
                'error: invalid number of input arguments (should be 0, 2, 6)')

    def __str__(self):
        print('SparseStar Set:')
        print('A: {}'.format(self.A))
        print('C: {}'.format(self.C))
        print('d: {}'.format(self.d))
        print('pred_lb: {}'.format(self.pred_lb))
        print('pred_ub: {}'.format(self.pred_ub))
        print('pred_depth: {}'.format(self.pred_depth))
        print('dim: {}'.format(self.dim))
        print('nVars: {}'.format(self.nVars))
        print('nZVars: {}'.format(self.nZVars))
        return '\n'

    def __repr__(self):
        print('SparseStar Set:')
        print('A: {}'.format(self.A.shape))
        print('C: {}'.format(self.C.shape))
        print('d: {}'.format(self.d.shape))
        print('pred_lb: {}'.format(self.pred_lb.shape))
        print('pred_ub: {}'.format(self.pred_ub.shape))
        print('pred_depth: {}'.format(self.pred_depth.shape))
        print('dim: {}'.format(self.dim))
        print('nVars: {}'.format(self.nVars))
        print('nZVars: {}'.format(self.nZVars))
        return '\n'

    def c(self, index=None):
        if index is None:
            return self.A[:, 0]
        else:
            return self.A[index, 0]

    def X(self, index=None):
        mA = self.A.shape[1]
        if index is None:
            return self.A[:, 1:mA]
        else:
            return self.A[index, 1:mA]

    def V(self, index=None):
        # mA = self.A.shape[1]
        # nnz = len(torch.nonzero(self.A))
        # if nnz > 0.5 * self.dim * self.nVars:
        #     return torch.column_stack([c, torch.diag(v)]).to_sparse_csr()
        # else:
        #     return torch.column_stack([c, torch.diag(v)])

        mA = self.A.shape[1]
        if index is None:
            return torch.column_stack([self.c(), torch.zeros(self.dim, self.nVars+1-mA), self.X()])
        else:
            if isinstance(index, int):
                return torch.column_stack([self.c(index), torch.zeros(1, self.nVars+1-mA), self.X(index)])
            else:
                return torch.column_stack([self.c(index), torch.zeros(len(index), self.nVars+1-mA), self.X(index)])

    def translation(self, v=None):
        """Translation of a sparse star: S = self + v"""
        if v is None:
            return copy.deepcopy(self)

        assert len(v.shape) == 1, 'error: \
            the translation vector should be a 1D torch tensor'
        assert v.shape[0] == self.dim, 'error: \
            inconsistency between translation vector and SparseStar dimension'

        A = copy.deepcopy(self.A)
        A[:, 0] += v
        return SparseStar(A, self.C, self.d, self.pred_lb, self.pred_ub, self.pred_depth)

    def affineMap(self, W=None, b=None):
        """Affine mapping of a sparse star: S = W*self + b"""

        if W is None and b is None:
            return copy.deepcopy(self)

        if W is not None:
            assert isinstance(W, torch.Tensor), 'error: \
                the mapping matrix should be a 2D torch tensor'
            assert W.shape[1] == self.dim, 'error: \
                inconsistency between mapping matrix and SparseStar dimension'

            A = torch.matmul(W, self.A)

        if b is not None:
            assert isinstance(b, torch.Tensor), 'error: \
                the offset vector should be a 1D torch tensor'
            assert len(b.shape) == 1, 'error: \
                offset vector should be a 1D torch tensor'

            if W is not None:
                assert W.shape[0] == b.shape[0], 'error: \
                    inconsistency between mapping matrix and offset'
            else:
                assert b.shape[0] == self.dim, 'error: \
                    inconsistency between offset vector and SparseStar dimension'
                A = copy.deepcopy(self.A)

            A[:, 0] += b

        return SparseStar(A, self.C, self.d, self.pred_lb, self.pred_ub, self.pred_depth)

    def getMin(self, index, lp_solver='gurobi'):
        """Get the minimum value of state x[index] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
        """
        assert index >= 0 and index <= self.dim-1, 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        f = self.X(index)
        if (f == 0).all():
            xmin = self.c(index, 0)
        else:
            if lp_solver == 'gurobi':  # gurobi is the preferred LP solver

                min_ = gp.Model()
                min_.Params.LogToConsole = 0
                min_.Params.OptimalityTol = 1e-9
                if self.pred_lb.size and self.pred_ub.size:
                    x = min_.addMVar(shape=self.nVars, lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = min_.addMVar(shape=self.nVars)
                print(x)
                # min_.setObjective(f @ x, GRB.MINIMIZE)
                min_.setObjective(torch.matmul(f, x), GRB.MINIMIZE)
                if len(self.C) > 0:
                    C = self.C
                    d = self.d
                else:
                    C = torch.zeros(1, self.nVars).to_sparse()
                    d = 0
                min_.addConstr(C @ x <= d)
                min_.optimize()

                if min_.status == 2:
                    xmin = min_.objVal + self.c(index)
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = %d' % (min_.status))

            elif lp_solver == 'linprog':
                pass
            elif lp_solver == 'glpk':
                pass
            else:
                raise Exception(
                    'error: unknown lp solver, should be gurobi or linprog or glpk')
        return xmin

    def getMax(self, index, lp_solver='gurobi'):
        pass

    def getMins(self, map, lp_solver='gurobi'):
        pass

    def getMaxs(self, map, lp_solver='gurobi'):
        pass

    def isEmptySet(self, lp_solver='gurobi'):
        """Check if a SparseStar is an empty set"""
        res = False
        try:
            self.getMin(0, lp_solver)
        except Exception:
            res = True

        return res

    @ staticmethod
    def rand(dim):
        """Randomly generate a SparStar"""

        assert dim > 0, 'error: invalid dimension'
        lb = -torch.rand(dim)
        ub = torch.rand(dim)
        return SparseStar(lb, ub)
