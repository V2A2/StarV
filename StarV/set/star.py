"""
Probabilistics Star Class
Dung Tran, 9/13/2022

"""

# !/usr/bin/python3
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog
from scipy.linalg import block_diag
#import glpk
import polytope as pc


import copy


class Star(object):
    """
        Star Class for reachability
        author: Dung Tran
        date: 9/13/2022
        Representation of a ProbStar
        ==========================================================================
        Star set defined by
        x = c + a[1]*v[1] + a[2]*v[2] + ... + a[n]*v[n]
            = V * b,
        where V = [c v[1] v[2] ... v[n]],
                b = [1 a[1] a[2] ... a[n]]^T,
                C*a <= d, constraints on a[i],
                a~N(mu,sigma) a normal distribution
        ==========================================================================
    """

    def __init__(self, *args):
        """
           Key Attributes:
           V = []; % basis matrix
           C = []; % constraint matrix
           d = []; % constraint vector
           dim = 0; % dimension of the probabilistic star set
           nVars = []; number of predicate variables
           pred_lb = []; % lower bound of predicate variables
           pred_ub = []; % upper bound of predicate variables
        """
        if len(args) == 5:
            [V, C, d, pred_lb, pred_ub] = copy.deepcopy(args)
            assert isinstance(V, np.ndarray), 'error: \
            basis matrix should be a 2D numpy array'
            assert isinstance(pred_lb, np.ndarray), 'error: \
            lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray), 'error: \
            upper bound vector should be a 1D numpy array'
            assert len(V.shape) == 2, 'error: \
            basis matrix should be a 2D numpy array'
            if len(C) != 0:
                assert len(C.shape) == 2, 'error: \
                constraint matrix should be a 2D numpy array'
                assert len(d.shape) == 1, 'error: \
                constraint vector should be a 1D numpy array'
                assert V.shape[1] == C.shape[1] + 1, 'error: \
                Inconsistency between basic matrix and constraint matrix'
                assert C.shape[0] == d.shape[0], 'error: \
                Inconsistency between constraint matrix and constraint vector'
                assert C.shape[1] == pred_lb.shape[0] and \
                    C.shape[1] == pred_ub.shape[0], 'error: \
                    Inconsistency between number of predicate variables and \
                    predicate lower- or upper-bound vectors'
                
            assert len(pred_lb.shape) == 1, 'error: \
            lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, 'error: \
            upper bound vector should be a 1D numpy array'
            
            self.V = V
            self.C = C
            self.d = d
            self.dim = V.shape[0]
            self.nVars = V.shape[1] - 1
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub

        elif len(args) == 2:  # the most common use
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
                raise RuntimeError("Upper bound (ub) must be greater or equal the lower bound (lb) for all D dimensions!")
            
            self.dim = lb.shape[0]
            nVars = int(sum(ub[i] > lb[i] for i in range(0, self.dim)))
            V = np.zeros((self.dim, nVars+1))
            pred_lb = np.zeros(nVars,)
            pred_ub = np.zeros(nVars,)
            j = 0
            for i in range(0, self.dim):
                if ub[i] > lb[i]:
                    pred_lb[j] = lb[i]
                    pred_ub[j] = ub[i]
                    V[i, j+1] = 1.
                    j = j + 1
            
            self.V = V
            self.C = np.array([])
            self.d = np.array([])
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.nVars = nVars        
            
        elif len(args) == 0:  # create an empty ProStar
            self.dim = 0
            self.nVars = 0
            self.V = np.array([])
            self.C = np.array([])
            self.d = np.array([])
            self.pred_lb = np.array([])
            self.pred_ub = np.array([])
        else:
            raise Exception('error: \
            Invalid number of input arguments (should be 2 or 5)')

    def __str__(self):
        print('Star Set:')
        print('V: {}'.format(self.V))
        print('Predicate Constraints:')
        print('C: {}'.format(self.C))
        print('d: {}'.format(self.d))
        print('dim: {}'.format(self.dim))
        print('nVars: {}'.format(self.nVars))
        print('pred_lb: {}'.format(self.pred_lb))
        print('pred_ub: {}'.format(self.pred_ub))
        return '\n'

    def getMinimizedConstraints(self):
        """minimize constraints of a probstar"""

        if len(self.C) == 0:
            Cmin = self.C
            dmin = self.d
        else:

            P = pc.Polytope(self.C, self.d)
            pc.reduce(P)
            Cmin = P.A
            dmin = P.b

        return Cmin, dmin
            
      
    def estimateRange(self, index):
        """Quickly estimate minimum value of a state x[index]"""

        assert index >= 0 and index <= self.dim-1, 'error: invalid index'

        v = self.V[index, 1:self.nVars+1]
        c = self.V[index, 0]
        v1 = copy.deepcopy(v)
        v2 = copy.deepcopy(v)
        c1 = copy.deepcopy(c)
        v1[v1 > 0] = 0  # negative part
        v2[v1 < 0] = 0  # positive part
        v1 = v1.reshape(1, self.nVars)
        v2 = v2.reshape(1, self.nVars)
        c1 = c1.reshape((1,))
        min_val = c1 + np.matmul(v1, self.pred_ub) + \
            np.matmul(v2, self.pred_lb)
        max_val = c1 + np.matmul(v1, self.pred_lb) + \
            np.matmul(v2, self.pred_ub)

        return min_val, max_val

    def estimateRanges(self):
        """Quickly estimate lower bound and upper bound of x"""

        v = self.V[:, 1:self.nVars+1]
        c = self.V[:, 0]
        v1 = copy.deepcopy(v)
        v2 = copy.deepcopy(v)
        c1 = copy.deepcopy(c)
        v1[v1 > 0] = 0  # negative part
        v2[v1 < 0] = 0  # positive part
        
        lb = c1 + np.matmul(v1, self.pred_ub) + \
            np.matmul(v2, self.pred_lb)
        ub = c1 + np.matmul(v1, self.pred_lb) + \
            np.matmul(v2, self.pred_ub)
        
        return lb, ub


    def getMin(self, index, lp_solver='gurobi'):
        """get exact minimum value of state x[index] by solving LP
           lp_solver = 'gurobi' or 'linprog' or 'glpk'
        """

        assert index >= 0 and index <= self.dim-1, 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        f = self.V[index, 1:self.nVars + 1]
        if (f == 0).all():
            xmin = self.V[index, 0]
        else:
            if lp_solver == 'gurobi':  # using gurobi is the preferred choice

                min_ = gp.Model()
                min_.Params.LogToConsole = 0
                min_.Params.OptimalityTol = 1e-9
                if self.pred_lb.size and self.pred_ub.size:
                    x = min_.addMVar(shape=self.nVars, lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = min_.addMVar(shape=self.nVars)
                min_.setObjective(f @ x, GRB.MINIMIZE)
                if len(self.C) == 0:
                    C = sp.csr_matrix(np.zeros((1, self.nVars)))
                    d = 0
                else:
                    C = sp.csr_matrix(self.C)
                    d = self.d
                min_.addConstr(C @ x <= d)
                min_.optimize()

                if min_.status == 2:
                    xmin = min_.objVal + self.V[index, 0]
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = %d' % (min_.status))

            elif lp_solver == 'linprog':

                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
                if len(self.C) == 0:
                    A = np.zeros((1, self.nVars))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.nVars, 1))
                ub = ub.reshape((self.nVars, 1))
                res = linprog(f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))

                if res.status == 0:
                    xmin = res.fun + self.V[index, 0]
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                #  https://pyglpk.readthedocs.io/en/latest/examples.html
                #  https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False

                if len(self.C) == 0:
                    A = np.zeros((1, self.nVars))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.nVars, 1))
                ub = ub.reshape((self.nVars, 1))

                lp = glpk.LPX()  # create the empty problem instance
                lp.obj.maximize = False
                lp.rows.add(A.shape[0])  # append rows to this instance
                for r in lp.rows:
                    r.name = chr(ord('p') + r.index)  # name rows if we want
                    lp.rows[r.index].bounds = None, b[r.index]

                lp.cols.add(self.nVars)
                for c in lp.cols:
                    c.name = 'x%d' % c.index
                    c.bounds = lb[c.index], ub[c.index]

                lp.obj[:] = f.tolist()
                B = A.reshape(A.shape[0]*A.shape[1],)
                lp.matrix = B.tolist()
                # lp.interior()
                lp.simplex()
                # default choice, interior may have a big floating point error

                if lp.status != 'opt':
                    raise Exception('error: cannot find an optimal solution, \
                    lp.status = {}'.format(lp.status))
                else:
                    xmin = lp.obj.value + self.V[index, 0]
            else:
                raise Exception('error: \
                unknown lp solver, should be gurobi or linprog or glpk')
        return xmin

    def getMax(self, index, lp_solver='gurobi'):
        """get exact maximum value of state x[index] by solving LP
           lp_solver = 'gurobi' or 'linprog' or 'glpk'
        """

        assert index >= 0 and index <= self.dim-1, 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        f = self.V[index, 1:self.nVars + 1]
        if (f == 0).all():
            xmax = self.V[index, 0]
        else:
            if lp_solver == 'gurobi':  # using gurobi is the preferred choice

                max_ = gp.Model()
                max_.Params.LogToConsole = 0
                max_.Params.OptimalityTol = 1e-9
                if self.pred_lb.size and self.pred_ub.size:
                    x = max_.addMVar(shape=self.nVars,
                                     lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = max_.addMVar(shape=self.nVars)
                max_.setObjective(f @ x, GRB.MAXIMIZE)
                if len(self.C) == 0:
                    C = sp.csr_matrix(np.zeros((1, self.nVars)))
                    d = 0
                else:
                    C = sp.csr_matrix(self.C)
                    d = self.d
                max_.addConstr(C @ x <= d)
                max_.optimize()

                if max_.status == 2:
                    xmax = max_.objVal + self.V[index, 0]
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = %d' % (max_.status))
            elif lp_solver == 'linprog':
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
                if len(self.C) == 0:
                    A = np.zeros((1, self.nVars))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.nVars, 1))
                ub = ub.reshape((self.nVars, 1))
                res = linprog(-f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))
                if res.status == 0:
                    xmax = -res.fun + self.V[index, 0]
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                # https://pyglpk.readthedocs.io/en/latest/examples.html
                # https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False  # turn off messages/display

                if len(self.C) == 0:
                    A = np.zeros((1, self.nVars))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.nVars, 1))
                ub = ub.reshape((self.nVars, 1))

                lp = glpk.LPX()  # create the empty problem instance
                lp.obj.maximize = True
                lp.rows.add(A.shape[0])  # append rows to this instance
                for r in lp.rows:
                    r.name = chr(ord('p') + r.index)  # name rows if we want
                    lp.rows[r.index].bounds = None, b[r.index]

                lp.cols.add(self.nVars)
                for c in lp.cols:
                    c.name = 'x%d' % c.index
                    c.bounds = lb[c.index], ub[c.index]

                lp.obj[:] = f.tolist()
                B = A.reshape(A.shape[0]*A.shape[1],)
                lp.matrix = B.tolist()

                # lp.interior()
                # default choice, interior may have a big floating point error
                lp.simplex()

                if lp.status != 'opt':
                    raise Exception('error: cannot find an optimal solution, \
                    lp.status = {}'.format(lp.status))
                else:
                    xmax = lp.obj.value + self.V[index, 0]
            else:
                raise Exception('error: \
                unknown lp solver, should be gurobi or linprog or glpk')
        return xmax

    def getRanges(self, lp_solver='gurobi'):
        """get lower bound and upper bound by solving LP"""

        l = np.zeros(self.dim)
        u = np.zeros(self.dim)
        for i in range(0, self.dim):
            l[i] = self.getMin(i, lp_solver)
            u[i] = self.getMax(i, lp_solver)

        return l, u

        
    def affineMap(self, A=None, b=None):
        """Affine mapping of a star: S = A*self + b"""

        if A is not None:
            assert isinstance(A, np.ndarray), 'error: \
        mapping matrix should be an 2D numpy array'
            assert A.shape[1] == self.dim, 'error: \
        inconsistency between mapping matrix and ProbStar dimension'

        if b is not None:
            assert isinstance(b, np.ndarray), 'error: \
        offset vector should be an 1D numpy array'
            if A is not None:
                assert A.shape[0] == b.shape[0], 'error: \
        inconsistency between mapping matrix and offset vector'
            assert len(b.shape) == 1, 'error: \
        offset vector should be a 1D numpy array '


        if A is None and b is None:
            new_set = copy.deepcopy(self)

        if A is None and b is not None:
            V = copy.deepcopy(self.V)
            V[:, 0] = V[:, 0] + b
            new_set = Star(V, self.C, self.d, self.pred_lb, self.pred_ub)

        if A is not None and b is None:
            V = np.matmul(A, self.V)
            new_set = Star(V, self.C, self.d, self.pred_lb, self.pred_ub)

        if A is not None and b is not None:
            V = np.matmul(A, self.V)
            V[:, 0] = V[:, 0] + b

            new_set = Star(V, self.C, self.d, self.pred_lb, self.pred_ub)
        return new_set

    def minKowskiSum(self, Y):
        """MinKowskiSum of two stars"""

        assert isinstance(Y, Star), 'error: input is not a probstar'
        assert self.dim == Y.dim, 'error: inconsistent dimension between the input and the self object'

        V1 = copy.deepcopy(self.V)
        V2 = copy.deepcopy(Y.V)
        V1[:, 0] = V1[:, 0] + V2[:, 0]
        V3 = np.delete(V2, 0, 1)
        V = np.hstack((V1, V3))
        pred_lb = np.concatenate((self.pred_lb, Y.pred_lb))
        pred_ub = np.concatenate((self.pred_ub, Y.pred_ub))  
        C = block_diag(self.C, Y.C)
        d = np.concatenate((self.d, Y.d))
        if len(d) == 0:
            C = []
            d = []
        
        R = Star(V, C, d, pred_lb, pred_ub)

        return R
        

    def isEmptySet(self, lp_solver='gurobi'):
        """Check if a probstar is an empty set"""

        res = False
        try:
            self.getMin(0, lp_solver)
        except Exception:
            res = True

        return res

    @classmethod
    def updatePredicateRanges(cls, newC, newd, pred_lb, pred_ub):
        """update estimated ranges for predicate variables \
        when one new constraint is added"""

        assert isinstance(newC, np.ndarray) and \
            len(newC.shape) == 1, 'error: \
            new constraint matrix should be 1D numpy array'
        assert isinstance(newd, np.ndarray) and \
            len(newd.shape) == 1 and newd.shape[0] == 1, 'error: \
            new constraint vector should be 1D numpy array'
        assert isinstance(pred_lb, np.ndarray) and \
            len(pred_lb.shape) == 1, 'error: \
            lower bound vector should be 1D numpy array'
        assert isinstance(pred_ub, np.ndarray) and \
            len(pred_ub.shape) == 1, 'error: \
            upper bound vector should be 1D numpy array'
        assert pred_lb.shape[0] == pred_ub.shape[0], 'error: \
        inconsistency between the lower bound and upper bound vectors'
        assert newC.shape[0] == pred_lb.shape[0], 'error: \
        inconsistency between the lower bound vector and the constraint matrix'

        new_pred_lb = copy.deepcopy(pred_lb)
        new_pred_ub = copy.deepcopy(pred_ub)

        # estimate new bounds for predicate variables
        for i in range(newC.shape[0]):
            x = newC[i]
            if x > 0:
                v1 = copy.deepcopy(newC)
                d1 = copy.deepcopy(newd)
                v1 = v1/x
                d1 = d1/x
                v1 = np.delete(v1, i)
                v2 = -v1
                v21 = copy.deepcopy(v2)
                v22 = copy.deepcopy(v2)
                v21[v21 < 0] = 0
                v22[v22 > 0] = 0
                v21 = v21.reshape(1, newC.shape[0] - 1)
                v22 = v22.reshape(1, newC.shape[0] - 1)
                lb = copy.deepcopy(pred_lb)
                ub = copy.deepcopy(pred_ub)
                lb = np.delete(lb, i)
                ub = np.delete(ub, i)

                xmax = d1 + np.matmul(v21, ub) + np.matmul(v22, lb)
                new_pred_ub[i] = min(xmax, pred_ub[i])  # update upper bound
            if x < 0:
                v1 = copy.deepcopy(newC)
                d1 = copy.deepcopy(newd)
                v1 = v1/x
                d1 = d1/x
                v1 = np.delete(v1, i)
                v2 = -v1
                v21 = copy.deepcopy(v2)
                v22 = copy.deepcopy(v2)
                v21[v21 < 0] = 0
                v22[v22 > 0] = 0
                v21 = v21.reshape(1, newC.shape[0] - 1)
                v22 = v22.reshape(1, newC.shape[0] - 1)
                lb = copy.deepcopy(pred_lb)
                ub = copy.deepcopy(pred_ub)
                lb = np.delete(lb, i)
                ub = np.delete(ub, i)
                xmin = d1 + np.matmul(v21, lb) + np.matmul(v22, ub)
                new_pred_lb[i] = max(xmin, pred_lb[i])  # update lower bound

        return new_pred_lb, new_pred_ub

    def addConstraint(self, C, d):
        """ Add a single constraint to a ProbStar, self & Cx <= d"""

        assert isinstance(C, np.ndarray) and len(C.shape) == 1, 'error: \
        constraint matrix should be 1D numpy array'
        assert isinstance(d, np.ndarray) and len(d.shape) == 1, 'error: \
        constraint vector should be a 1D numpy array'
        assert C.shape[0] == self.dim, 'error: \
        inconsistency between the constraint matrix and the probstar dimension'

        v = np.matmul(C, self.V)
        newC = v[1:self.nVars+1]
        newd = d - v[0]

        if len(self.C) != 0:
            self.C = np.vstack((newC, self.C))
            self.d = np.concatenate([newd, self.d])
            
        else:
            self.C = newC.reshape(1, self.nVars)
            self.d = newd
        new_pred_lb, new_pred_ub = Star.updatePredicateRanges(newC, newd,
                                                                  self.pred_lb,
                                                                  self.pred_ub)
        self.pred_lb = new_pred_lb
        self.pred_ub = new_pred_ub

        return self

    def addMultipleConstraints(self, C, d):
        """ Add multiple constraint to a Star, self & Cx <= d"""

        assert isinstance(C, np.ndarray), 'error: constraint matrix should be a numpy array'
        assert isinstance(d, np.ndarray), 'error: constraint vector should be a numpy array'
        assert C.shape[0] == d.shape[0], 'error: inconsistency between \
        constraint matrix and constraint vector'
        print(len(d.shape))
        assert len(d.shape) == 1, 'error: constraint vector should be a 1D numpy array'
        
        if C.shape[0] == 1:
            self.addConstraint(C, d)
        else:
            for i in range(0, C.shape[0]):
                self.addConstraint(C[i, :], np.array([d[i]]))

        return self

    def resetRow(self, index):
        """Reset a row with index"""

        if index < 0 or index > self.dim - 1:
            raise Exception('error: invalid index, \
            should be between {} and {}'.format(0, self.dim - 1))
        V = self.V
        V[index, :] = 0.0
        S = Star(V, self.C, self.d, self.pred_lb, self.pred_ub)

        return S

    @staticmethod
    def rand(dim):
        """ Randomly generate a Star """

        assert dim > 0, 'error: invalid dimension'
        lb = -np.random.rand(dim,)
        ub = np.random.rand(dim,)
        
        return Star(lb, ub)
