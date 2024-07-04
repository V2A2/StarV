"""
Probabilistics Star Class
Dung Tran, 8/10/2022

Update: 8/13/2023

"""

# !/usr/bin/python3
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog
from scipy.stats import mvn
from scipy.linalg import block_diag
import glpk
import polytope as pc
from StarV.util.minimax_tilting_sampler import TruncatedMVN


import copy


class ProbStar(object):
    """
        Probabilistic Star Class for quatitative reachability
        author: Dung Tran
        date: 8/9/2022
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
           mu = []; % mean of the multivariate normal distribution
           Sig = []; % covariance (positive semidefinite matrix)
           nVars = []; number of predicate variables
           prob = []; % probability of the probabilistic star
           predicate_lb = []; % lower bound of predicate variables
           predicate_ub = []; % upper bound of predicate variables
        """
        if len(args) == 7:
            [V, C, d, mu, Sig, pred_lb, pred_ub] = copy.deepcopy(args)
            assert isinstance(V, np.ndarray), 'error: \
            basis matrix should be a 2D numpy array'
            assert isinstance(mu, np.ndarray), 'error: \
            median vector should be a 1D numpy array'
            assert isinstance(Sig, np.ndarray), 'error: \
            covariance matrix should be a 2D numpy array'
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
                assert C.shape[1] == mu.shape[0], 'error: Inconsistency \
                between the number of predicate variables and median vector'
                assert C.shape[1] == Sig.shape[1] and \
                    C.shape[1] == Sig.shape[0], 'error: Inconsistency between \
                    the number of predicate variables and covariance matrix'
            assert len(mu.shape) == 1, 'error: \
            median vector should be a 1D numpy array'
            assert len(Sig.shape) == 2, 'error: \
            covariance matrix should be a 2D numpy array'
            assert len(pred_lb.shape) == 1, 'error: \
            lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, 'error: \
            upper bound vector should be a 1D numpy array'
            assert np.all(np.linalg.eigvals(Sig) > 0), 'error: \
            covariance matrix should be positive definite'

            self.V = V
            self.C = C
            self.d = d
            self.dim = V.shape[0]
            self.nVars = V.shape[1] - 1
            self.mu = mu
            self.Sig = Sig
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub

        elif len(args) == 4:  # the most common use
            [mu, Sig, pred_lb, pred_ub] = copy.deepcopy(args)

            assert isinstance(mu, np.ndarray), 'error: \
            median vector should be a 1D numpy array'
            assert isinstance(Sig, np.ndarray), 'error: \
            covariance matrix should be a 2D numpy array'
            assert isinstance(pred_lb, np.ndarray), 'error: \
            lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray), 'error: \
            upper bound vector should be a 1D numpy array'
            assert len(mu.shape) == 1, 'error: \
            median vector should be a 1D numpy array'
            assert len(Sig.shape) == 2, 'error: \
            covariance matrix should be a 2D numpy array'
            assert len(pred_lb.shape) == 1, 'error: \
            lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, 'error: \
            upper bound vector should be a 1D numpy array'

            assert pred_lb.shape[0] == pred_ub.shape[0], 'error: \
            inconsistency between predicate lower bound and upper bound'
            assert pred_lb.shape[0] == mu.shape[0], 'error: \
            inconsitency between predicate lower bound and median vector'
            assert mu.shape[0] == Sig.shape[0] and \
                mu.shape[0] == Sig.shape[1], 'error: \
                inconsistency between median vector and covariance matrix'
            assert np.all(np.linalg.eigvals(Sig) > 0), 'error: \
            covariance matrix should be positive definite'

            self.dim = pred_lb.shape[0]
            self.nVars = pred_lb.shape[0]
            self.mu = mu
            self.Sig = Sig
            self.V = np.hstack((np.zeros((self.dim, 1)), np.eye(self.dim)))
            self.C = np.array([])
            self.d = np.array([])
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub

        elif len(args) == 0:  # create an empty ProStar
            self.dim = 0
            self.nVars = 0
            self.mu = 0.0
            self.Sig = np.array([])
            self.V = np.array([])
            self.C = np.array([])
            self.d = np.array([])
            self.pred_lb = np.array([])
            self.pred_ub = np.array([])
        else:
            raise Exception('error: \
            Invalid number of input arguments (should be 4 or 7)')

    def __str__(self):
        print('ProbStar Set:')
        print('V: {}'.format(self.V))
        print('Predicate Constraints:')
        print('C: {}'.format(self.C))
        print('d: {}'.format(self.d))
        print('dim: {}'.format(self.dim))
        print('nVars: {}'.format(self.nVars))
        print('pred_lb: {}'.format(self.pred_lb))
        print('pred_ub: {}'.format(self.pred_ub))
        print('mu: {}'.format(self.mu))
        print('Sig: {}'.format(self.Sig))
        return '\n'

    def printConstraints(self):
        'Print constraints of probstar'
        P = pc.Polytope(self.C, self.d)
        print(P)

    def estimateProbability(self):
        """estimate probability of a probstar
           using Genz method, Botev method 
           may be a better option
        """

        if len(self.C) == 0:
            prob, _ = mvn.mvnun(self.pred_lb, self.pred_ub, self.mu, self.Sig)
            
        else:
            # C, d = self.getMinimizedConstraints()
            C1 = np.vstack((np.eye(self.nVars), -np.eye(self.nVars)))
            d1 = np.concatenate([self.pred_ub, -self.pred_lb])
            C = np.vstack((self.C, C1))
            d = np.concatenate([self.d, d1])       
            A = np.matmul(np.matmul(C, self.Sig), np.transpose(C)) # A = C*Sig*C'

            if np.all(np.linalg.eigvals(A) > 0):  # No need to introduce auxilary normal variables
                # Check Truncated Normal Matlab Toolbox of Botev to understand this conversion
                new_lb = np.NINF*np.ones(len(d),)  # lb = l - A*mu
                new_ub = d - np.matmul(C, self.mu)  # ub = u - A*mu 
                new_mu = np.zeros(len(d),)          # new_mu = 0
                new_Sig = np.matmul(np.matmul(C, self.Sig), np.transpose(C)) # new_Sig = A*Sig*A'
                prob, _ = mvn.mvnun(new_lb, new_ub, new_mu, new_Sig)

            else:  # Need to introduce auxilary normal variables
                # step 1: SVD decomposition
                # [U, Q, L] = SVD(C), C = U*Q*L'
                # decompose Q = [Q_(r x r); 0_(m-r x r)]
                # U'*U = L'*L = I_r
                U, Q, L = np.linalg.svd(C)
                Q1 = np.diag(Q)
                r = Q1.shape[0]
                L1 = L[0:r, :]
                Q1 = np.matmul(Q1, L1)

                # linear transformation a_r' = Q1*a_r of original normal variables
                mu1 = np.matmul(Q1, self.mu)
                Sig1 = np.matmul(np.matmul(Q1, self.Sig), np.transpose(Q1))
                m = U.shape[0] - len(Q)  # number of auxilary variables
                mu2 = np.zeros(m,)  # auxilary normal variables mean
                Sig2 = (1e-10)*np.eye(m)  # auxilary normal variables variance

                new_mu = np.concatenate([mu1, mu2])
                new_Sig = block_diag(Sig1, Sig2)

                new_lb = np.NINF*np.ones(len(d),)
                new_ub = d - np.matmul(U, new_mu)
                new_Sig = np.matmul(np.matmul(U, new_Sig), np.transpose(U))

                prob, _ = mvn.mvnun(new_lb, new_ub, np.zeros(len(d),), new_Sig)

        return prob

    def getMinimizedConstraints(self):
        """minimize constraints of a probstar"""

        if len(self.C) == 0:
            Cmin = self.C
            dmin = self.d
        else:
            C1 = np.vstack((np.eye(self.nVars), -np.eye(self.nVars)))
            d1 = np.concatenate([self.pred_ub, -self.pred_lb])
            C = np.vstack((self.C, C1))
            d = np.concatenate([self.d, d1])       
            P = pc.Polytope(C, d)
            P1 = pc.reduce(P)
            Cmin = P1.A
            dmin = P1.b

        return Cmin, dmin

    def minimizeConstraints(self):

        if len(self.C) == 0:
            return self
        else:
            C1 = np.vstack((np.eye(self.nVars), -np.eye(self.nVars)))
            d1 = np.concatenate([self.pred_ub, -self.pred_lb])
            C = np.vstack((self.C, C1))
            d = np.concatenate([self.d, d1])       
            P = pc.Polytope(C, d)
            P1 = pc.reduce(P)
            Cmin = P1.A
            dmin = P1.b
            self.C = Cmin
            self.d = dmin

            return self
        
            
      
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
                min_.Params.OptimalityTol = 1e-6
                if self.pred_lb.size and self.pred_ub.size:
                    x = min_.addMVar(shape=self.nVars,
                                     lb=self.pred_lb, ub=self.pred_ub)
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
                max_.Params.OptimalityTol = 1e-6
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
        """Affine mapping of a probstar: S = A*self + b"""

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
            new_set = ProbStar(V, self.C, self.d, self.mu, self.Sig, self.pred_lb, self.pred_ub)

        if A is not None and b is None:
            V = np.matmul(A, self.V)
            new_set = ProbStar(V, self.C, self.d, self.mu, self.Sig, self.pred_lb, self.pred_ub)

        if A is not None and b is not None:
            V = np.matmul(A, self.V)
            V[:, 0] = V[:, 0] + b

            new_set = ProbStar(V, self.C, self.d, self.mu,
                               self.Sig, self.pred_lb, self.pred_ub)
        return new_set

    def minKowskiSum(self, Y):
        """MinKowskiSum of two probstars"""

        assert isinstance(Y, ProbStar), 'error: input is not a probstar'
        assert self.dim == Y.dim, 'error: inconsistent dimension between the input and the self object'

        V1 = copy.deepcopy(self.V)
        V2 = copy.deepcopy(Y.V)
        V1[:, 0] = V1[:, 0] + V2[:, 0]
        V3 = np.delete(V2, 0, 1)
        V = np.hstack((V1, V3))
        pred_lb = np.concatenate((self.pred_lb, Y.pred_lb))
        pred_ub = np.concatenate((self.pred_ub, Y.pred_ub))
        mu = np.concatenate((self.mu, Y.mu))
        Sig = block_diag(self.Sig, Y.Sig)
        C = block_diag(self.C, Y.C)
        d = np.concatenate((self.d, Y.d))
        if len(d) == 0:
            C = []
            d = []
        R = ProbStar(V, C, d, mu, Sig, pred_lb, pred_ub)

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
        new_pred_lb, new_pred_ub = ProbStar.updatePredicateRanges(newC, newd,
                                                                  self.pred_lb,
                                                                  self.pred_ub)
        self.pred_lb = new_pred_lb
        self.pred_ub = new_pred_ub

        return self

    def addConstraintWithoutUpdateBounds(self, C, d):

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

        return self

    def addMultipleConstraints(self, C, d):
        """ Add multiple constraint to a ProbStar, self & Cx <= d"""

        assert isinstance(C, np.ndarray), 'error: constraint matrix should be a numpy array'
        assert isinstance(d, np.ndarray), 'error: constraint vector should be a numpy array'
        assert C.shape[0] == d.shape[0], 'error: inconsistency between \
        constraint matrix and constraint vector'
        assert len(d.shape) == 1, 'error: constraint vector should be a 1D numpy array'
        
     
        for i in range(0, C.shape[0]):
            self.addConstraint(C[i, :], np.array([d[i]]))

        return self

    def addMultipleConstraintsWithoutUpdateBounds(self, C, d):
        """ Add multiple constraint to a ProbStar, self & Cx <= d"""

        assert isinstance(C, np.ndarray), 'error: constraint matrix should be a numpy array'
        assert isinstance(d, np.ndarray), 'error: constraint vector should be a numpy array'
        assert C.shape[0] == d.shape[0], 'error: inconsistency between \
        constraint matrix and constraint vector'
        assert len(d.shape) == 1, 'error: constraint vector should be a 1D numpy array'


        for i in range(0, C.shape[0]):
            self.addConstraintWithoutUpdateBounds(C[i, :], np.array([d[i]]))

        return self

    def resetRow(self, index):
        """Reset a row with index"""

        if index < 0 or index > self.dim - 1:
            raise Exception('error: invalid index, \
            should be between {} and {}'.format(0, self.dim - 1))
        V = self.V
        V[index, :] = 0.0
        S = ProbStar(V, self.C, self.d, self.mu, self.Sig,
                     self.pred_lb, self.pred_ub)

        return S

    def resetRowWithFactor(self, index, factor):
        """Reset a row with index and factor
        Author: Yuntao, Date: 1/30/2024
        """

        if index < 0 or index > self.dim - 1:
            raise Exception('error: invalid index, \
            should be between {} and {}'.format(0, self.dim - 1))
        V = self.V
        V[index, :] *= factor
        S = ProbStar(V, self.C, self.d, self.mu, self.Sig,
                     self.pred_lb, self.pred_ub)

        return S
    
    def resetRowWithUpdatedCenter(self, index, new_c):
        """Reset a row with index, and with new center
        Author: Yuntao, Date: 1/30/2024
        """

        if index < 0 or index > self.dim - 1:
            raise Exception('error: invalid index, \
            should be between {} and {}'.format(0, self.dim - 1))
        V = self.V
        V[index, :] = 0.0
        V[index, 0] = new_c
        S = ProbStar(V, self.C, self.d, self.mu, self.Sig,
                     self.pred_lb, self.pred_ub)

        return S


    def concatenate_with_vector(self, v=[]):
        """
           concatenate a probstar with a vector
           Dung Tran: 11/19/2023
        """

        if len(v) != 0:
            assert isinstance(v, np.ndarray), 'error: input should be a 1-d array'
            assert len(v.shape) == 1, 'error: input should be a 1-d array'

            n = v.shape[0]
            v1 = v.reshape(n,1)
            V1 = np.zeros((n, self.nVars))
            V1 = np.hstack((v1, V1))
            newV = np.vstack((V1, self.V))
            S = ProbStar(newV, self.C, self.d, self.mu, self.Sig, self.pred_lb, self.pred_ub)
            return S
        else:
            return self

    @staticmethod
    def rand(*args):
        """ Randomly generate a ProbStar """

        # Update: 8/13/2023 by Dung Tran

        if len(args) == 1:
            dim = args[0]
            nVars = dim
        elif len(args) == 2:
            dim = args[0]
            nVars = args[1]

        elif len(args) == 4:
            dim = args[0]
            nVars = args[1]
            pred_lb = args[2]
            pred_ub = args[3]

            assert isinstance(pred_lb, np.ndarray), 'predicate_lb should be a 1-d numpy array'
            assert isinstance(pred_ub, np.ndarray), 'predicate_ub should be a 1-d numpy array'

            assert pred_lb.shape[0] == pred_ub.shape[0], 'inconsistency between predicate_lb and predicate_ub'
            assert pred_lb.shape[0] == nVars, 'inconsistency between the length of predicate_lb and number of predicate variables'
        else:
            raise RuntimeError('invalid number of arguments, should be 1 or 2')
            
        V = np.random.rand(dim, nVars + 1)
        if len(args) != 4:    
            pred_lb = -np.random.rand(nVars,)
            pred_ub = np.random.rand(nVars,)
            
        mu = 0.5*(pred_lb + pred_ub)
        a = 3.0
        sig = (mu - pred_lb)/a
        Sig = np.diag(np.square(sig))

        S = ProbStar(V, [], [], mu, Sig, pred_lb, pred_ub)

        return S

    def sampling(self, N):
        """
        Monte Carlo sampling of a probstar, equivalent to MC sampling of multivariate truncated normal distribution

        Ref: 1) https://stackoverflow.com/questions/20115917/truncated-multivariate-normal-in-scipy
             2) code here: https://github.com/V2A2/truncated-mvn-sampler
        """

        tmvn = TruncatedMVN(self.mu, self.Sig, self.pred_lb, self.pred_ub)
        samples = tmvn.sample(N)
        V = self.V[:, 1:self.nVars+1]
        center = self.V[:,0]
        center = center.reshape(self.dim, 1)
        samples = np.matmul(V, samples) + center
        samples = np.unique(samples, axis=1)

        return samples
