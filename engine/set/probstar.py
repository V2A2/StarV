"""
Probabilistics Star Class
Dung Tran, 8/10/2022

"""

# !/usr/bin/python3
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog
import glpk


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
            assert isinstance(V, np.ndarray), 'error: basis matrix should be a 2D numpy array'
            assert isinstance(C, np.ndarray), 'error: constraint matrix should be a 2D numpy array'
            assert isinstance(d, np.ndarray), 'error: constraint vector should be a 1D numpy array'
            assert isinstance(mu, np.ndarray), 'error: median vector should be a 1D numpy array'
            assert isinstance(Sig, np.ndarray), 'error: covariance matrix should be a 2D numpy array'
            assert isinstance(pred_lb, np.ndarray), 'error: lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray), 'error: upper bound vector should be a 1D numpy array'
            
            assert len(V.shape) == 2, 'error: basis matrix should be a 2D numpy array'
            if len(C) != 0:
                assert len(C.shape) == 2, 'error: constraint matrix should be a 2D numpy array'
                assert len(d.shape) == 1, 'error: constraint vector should be a 1D numpy array'
                assert V.shape[1] == C.shape[1] + 1, 'error: Inconsistency between basic matrix and constraint matrix'
                assert C.shape[0] == d.shape[0], 'error: Inconsistency between constraint matrix and constraint vector'
                assert d.shape[0] == pred_lb.shape[0] and d.shape[0] == pred_ub.shape[0], 'error: Inconsistency between number of predicate variables and predicate lower- or upper-bound vectors'
                assert C.shape[1] == mu.shape[0], 'error: Inconsistency between the number of predicate variables and median vector'
                assert C.shape[1] == Sig.shape[1] and C.shape[1] == Sig.shape[0], 'error: Inconsistency between the number of predicate variables and covariance matrix'
            assert len(mu.shape) == 1, 'error: median vector should be a 1D numpy array'
            assert len(Sig.shape) == 2, 'error: covariance matrix should be a 2D numpy array'
            assert len(pred_lb.shape) == 1, 'error: lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, 'error: upper bound vector should be a 1D numpy array'
 
            assert np.all(np.linalg.eigvals(Sig) > 0), 'error: covariance matrix should be positive definite'
            
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

            assert isinstance(mu, np.ndarray), 'error: median vector should be a 1D numpy array'
            assert isinstance(Sig, np.ndarray), 'error: covariance matrix should be a 2D numpy array'
            assert isinstance(pred_lb, np.ndarray), 'error: lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray), 'error: upper bound vector should be a 1D numpy array'
        
            assert len(mu.shape) == 1, 'error: median vector should be a 1D numpy array'
            assert len(Sig.shape) == 2, 'error: covariance matrix should be a 2D numpy array'
            assert len(pred_lb.shape) == 1, 'error: lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, 'error: upper bound vector should be a 1D numpy array'

            assert pred_lb.shape[0] == pred_ub.shape[0], 'error: inconsistency between predicate lower bound and upper bound'
            assert pred_lb.shape[0] == mu.shape[0], 'error: inconsitency between predicate lower bound and median vector'
            assert mu.shape[0] == Sig.shape[0] and mu.shape[0] == Sig.shape[1], 'error: inconsistency between median vector and covariance matrix'
            assert np.all(np.linalg.eigvals(Sig) > 0), 'error: covariance matrix should be positive definite'
            
            self.dim = pred_lb.shape[0]
            self.nVars = pred_lb.shape[0]
            self.mu = mu
            self.Sig = Sig
            self.V = np.hstack((np.zeros((self.dim, 1)), np.eye(self.dim)))
            self.C = np.array([])
            self.d = np.array([])
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub  
        else:
            raise Exception('error: Invalid number of input arguments (should be 4 or 7)')

    def __str__(self): 
        print('ProbStar Set:')
        print('V: {}'.format(self.V))
        print('C: {}'.format(self.C))
        print('d: {}'.format(self.d))
        print('dim: {}'.format(self.dim))
        print('nVars: {}'.format(self.nVars))
        print('pred_lb: {}'.format(self.pred_lb))
        print('pred_ub: {}'.format(self.pred_ub))
        print('mu: {}'.format(self.mu))
        print('Sig: {}'.format(self.Sig))
        return '\n'

    def estimateRange(self, index):
        """Quickly estimate minimum value of a state x[index]"""

        assert index >= 0 and index <= self.dim-1, 'error: invalid index'

        v = self.V[index, 1:self.dim+1]
        c = self.V[index, 0]
        v1 = copy.deepcopy(v)
        v2 = copy.deepcopy(v)
        c1 = copy.deepcopy(c)
        v1[v1 > 0] = 0 # negative part
        v2[v1 < 0] = 0 # positive part
        v1 = v1.reshape(1, self.nVars)
        v2 = v2.reshape(1, self.nVars)
        c1 = c1.reshape((1,))
        min_val = c1 + np.matmul(v1, self.pred_ub) + np.matmul(v2, self.pred_lb)
        max_val = c1 + np.matmul(v1, self.pred_lb) + np.matmul(v2, self.pred_ub)

        return min_val, max_val

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
            if lp_solver == 'gurobi': # using gurobi is the preferred choice

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
                    raise Exception('error: cannot find an optimal solution, exitflag = %d' % (min_.status))
                
            elif lp_solver == 'linprog':
                
                # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html 

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
                    raise Exception('error: cannot find an optimal solution, exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                # reference: https://pyglpk.readthedocs.io/en/latest/examples.html
                #          : https://pyglpk.readthedocs.io/en/latest/

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

                lp = glpk.LPX() # create the empty problem instance
                lp.obj.maximize = False
                lp.rows.add(A.shape[0]) # append rows to this instance
                for r in lp.rows:
                    r.name = chr(ord('p') + r.index) # name rows if we want
                    lp.rows[r.index].bounds = None, b[r.index]

                lp.cols.add(self.nVars)
                for c in lp.cols:
                    c.name = 'x%d'% c.index
                    c.bounds = lb[c.index], ub[c.index]

                lp.obj[:] = f.tolist()
                B = A.reshape(A.shape[0]*A.shape[1],)
                lp.matrix = B.tolist()

                #lp.interior() 
                lp.simplex() # default choice, interior may have a big floating point error

                if lp.status != 'opt':
                    raise Exception('error: cannot find an optimal solution, lp.status = {}'.format(lp.status))
                else:
                    xmin = lp.obj.value + self.V[index, 0]
                
                
            else:
                raise Exception('error: unknown lp solver, should be gurobi or linprog or glpk')
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
            if lp_solver == 'gurobi': # using gurobi is the preferred choice

                max_ = gp.Model()
                max_.Params.LogToConsole = 0
                max_.Params.OptimalityTol = 1e-9
                if self.pred_lb.size and self.pred_ub.size:
                    x = max_.addMVar(shape=self.nVars, lb=self.pred_lb, ub=self.pred_ub)
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
                    raise Exception('error: cannot find an optimal solution, exitflag = %d' % (max_.status))
                
            elif lp_solver == 'linprog':
                
                # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html 

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
                    raise Exception('error: cannot find an optimal solution, exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                # reference: https://pyglpk.readthedocs.io/en/latest/examples.html
                #          : https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False # turn off messages/display
                
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

                lp = glpk.LPX() # create the empty problem instance
                lp.obj.maximize = True
                lp.rows.add(A.shape[0]) # append rows to this instance
                for r in lp.rows:
                    r.name = chr(ord('p') + r.index) # name rows if we want
                    lp.rows[r.index].bounds = None, b[r.index]

                lp.cols.add(self.nVars)
                for c in lp.cols:
                    c.name = 'x%d'% c.index
                    c.bounds = lb[c.index], ub[c.index]

                lp.obj[:] = f.tolist()
                B = A.reshape(A.shape[0]*A.shape[1],)
                lp.matrix = B.tolist()

                #lp.interior() 
                lp.simplex() # default choice, interior may have a big floating point error

                if lp.status != 'opt':
                    raise Exception('error: cannot find an optimal solution, lp.status = {}'.format(lp.status))
                else:
                    xmax = lp.obj.value + self.V[index, 0]
                
                
            else:
                raise Exception('error: unknown lp solver, should be gurobi or linprog or glpk')
        return xmax

    def affineMap(self, A, b):
        """Affine mapping of a probstar: S = A*self + b"""

        assert isinstance(A, np.ndarray), 'error: mapping matrix should be an 2D numpy array'
        assert isinstance(b, np.ndarray), 'error: offset vector should be an 1D numpy array'

        assert A.shape[1] == self.dim, 'error: inconsistency between mapping matrix and ProbStar dimension'
        assert A.shape[0] == b.shape[0], 'error: inconsistency between mapping matrix and offset vector'
        assert len(b.shape) == 1, 'error: offset vector should be a 1D numpy array '

        V = np.matmul(A, self.V)
        V[:,0] = V[:, 0] + b

        new_set = ProbStar(V, self.C, self.d, self.mu, self.Sig, self.pred_lb, self.pred_ub)

        return new_set

    def isEmptySet(self, lp_solver='gurobi'):
        """Check if a probstar is an empty set"""

        res = False
        try:
            self.getMin(0,lp_solver)
        except:
            res = True

        return res
            
            
    @classmethod
    def updatePredicateRanges(cls, newC, newd, pred_lb, pred_ub):
        """update estimated ranges for predicate variables when one new constraint is added"""

        assert isinstance(newC, np.ndarray) and len(newC.shape) == 1, 'error: new constraint matrix should be 1D numpy array'
        assert isinstance(newd, np.ndarray) and len(newd.shape) == 1 and newd.shape[0] == 1, 'error: new constraint vector should be 1D numpy array'
        assert isinstance(pred_lb, np.ndarray) and len(pred_lb.shape) == 1, 'error: lower bound vector should be 1D numpy array'
        assert isinstance(pred_ub, np.ndarray) and len(pred_ub.shape) == 1, 'error: upper bound vector should be 1D numpy array'
        assert pred_lb.shape[0] == pred_ub.shape[0], 'error: inconsistency between the lower bound and upper bound vectors'
        assert newC.shape[0] == pred_lb.shape[0], 'error: inconsistency between the lower bound vector and the constraint matrix'

        
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

        assert isinstance(C, np.ndarray) and len(C.shape) == 1, 'error: constraint matrix should be 1D numpy array'
        assert isinstance(d, np.ndarray) and len(d.shape) == 1, 'error: constraint vector should be a 1D numpy array'
        assert C.shape[0] == self.dim, 'error: inconsistency between the constraint matrix and the probstar dimension'

        v = np.matmul(C, self.V)
        newC = v[1:self.nVars+1]
        newd = d - v[0]

        if len(self.C) != 0:
            self.C = np.vstack((newC, self.C))
            self.d = np.vstack((newd, self.d))
        else:
            self.C = newC
            self.d = newd

        new_pred_lb, new_pred_ub = ProbStar.updatePredicateRanges(newC, newd, self.pred_lb, self.pred_ub)
        self.pred_lb = new_pred_lb
        self.pred_ub = new_pred_ub

        return self

    def addMultipleConstraints(self, C, d):
        """ Add multiple constraint to a ProbStar, self & Cx <= d"""
        pass


class Test(object):
    """
       Testing ProbStar class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_constructor(self):

        self.n_tests = self.n_tests + 1

        # len(agrs) = 4
        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        try:
            print('Testing ProbStar Constructor...')
            S = ProbStar(mu, Sig, pred_lb, pred_ub)
        except:
            print("Fail in constructing probstar object with len(args)= {}".format(4))
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_str(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        
        try:
            print('\nTesting __str__ method...')
            print(S.__str__())
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_estimateRange(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        
        try:
            print('\nTesting estimateMin method...')
            min_val, max_val = S.estimateRange(0)
            print('MinValue = {}, true_val = {}'.format(min_val,pred_lb[0]))
            print('MaxValue = {}, true_val = {}'.format(max_val, pred_ub[0]))
            assert min_val == pred_lb[0] and max_val == pred_ub[0], 'error: wrong results'
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_getMin(self):

        self.n_tests = self.n_tests + 3

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        min_val = S.getMin(0,'gurobi')

        try:
            print('\nTesting getMin method using gurobi...')
            min_val = S.getMin(0,'gurobi')
            print('MinValue = {}, true_val = {}'.format(min_val,pred_lb[0]))
            assert min_val == pred_lb[0], 'error: wrong results'
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        
        try:
            print('\nTesting getMin method using glpk...')
            min_val = S.getMin(0,'glpk')
            print('MinValue = {}, true_val = {}'.format(min_val,pred_lb[0]))
            assert min_val == pred_lb[0], 'error: wrong results'
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        try:
            print('\nTesting getMin method using linprog...')
            min_val = S.getMin(0,'linprog')
            print('MinValue = {}, true_val = {}'.format(min_val,pred_lb[0]))
            assert min_val == pred_lb[0], 'error: wrong results'
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_glpk(self):
        """
           test glpk
           example from here: https://pyglpk.readthedocs.io/en/latest/examples.html

        """

        self.n_tests = self.n_tests + 1
        
        lp = glpk.LPX()
        lp.name = 'test_glpk'
        lp.obj.maximize = True
        lp.rows.add(3)
        for r in lp.rows:
            r.name = chr(ord('p') + r.index)
        lp.rows[0].bounds = None, 100.0
        lp.rows[1].bounds = None, 600.0
        lp.rows[2].bounds = None, 300.0
        lp.cols.add(3)
        for c in lp.cols:
            c.name = 'x%d' % c.index
            c.bounds = 0.0, None

        f = np.array([10.0, 6.0, 4.0])
        lp.obj[:] = f.tolist()
        #lp.obj[:] = [10.0, 6.0, 4.0]
        A = np.array([[1.0, 1.0, 1.0], [10.0, 4.0, 5.0], [2.0, 2.0, 6.0]])
        B = A.reshape(9,)
        a = B.tolist()
        lp.matrix = a 
        #lp.matrix = [1.0, 1.0, 1.0,
        #             10.0, 4.0, 5.0,
        #             2.0, 2.0, 6.0]

        try:
            print('\nTest glpk...')
            lp.simplex()
            print('Z = {}'.format(lp.obj.value))
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_getMax(self):

        self.n_tests = self.n_tests + 3

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        
        try:
            print('\nTesting getMax method using gurobi...')
            max_val = S.getMax(0,'gurobi')
            print('MaxValue = {}, true_val = {}'.format(max_val,pred_ub[0]))
            assert max_val == pred_ub[0], 'error: wrong results'
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        
        try:
            print('\nTesting getMax method using glpk...')
            max_val = S.getMax(0,'glpk')
            print('MaxValue = {}, true_val = {}'.format(max_val,pred_ub[0]))
            assert max_val == pred_ub[0], 'error: wrong results'
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        try:
            print('\nTesting getMax method using linprog...')
            max_val = S.getMax(0,'linprog')
            print('MaxValue = {}, true_val = {}'.format(max_val,pred_ub[0]))
            assert max_val == pred_ub[0], 'error: wrong results'
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_affineMap(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)

        A = np.random.rand(2,3)
        b = np.random.rand(2,)
        
        try:
            print('\nTesting affine mapping method...')
            S1 = S.affineMap(A,b)
            print('original probstar:')
            print(S.__str__())
            print('new probstar:')
            print(S1.__str__())
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_isEmptySet(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)

        try:
            print('\nTesting isEmptySet method...')
            res = S.isEmptySet('gurobi')
            print('res: {}'.format(res))
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_updatePredicateRanges(self):

        self.n_tests = self.n_tests + 2

        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        newC = np.array([-0.25, 1.0])
        newd = np.array([0.25])

        try:
            print('\nTesting updatePredicateRanges method 1...')
            new_pred_lb, new_pred_ub = ProbStar.updatePredicateRanges(newC, newd, pred_lb, pred_ub)
            print('new_pred_lb: {}, new_pred_ub: {}'.format(new_pred_lb, new_pred_ub))
            assert new_pred_ub[1] == 0.5, 'error: wrong results'
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")


        newC = np.array([0.25, -1.0])
        newd = np.array([-0.25])

        try:
            print('\nTesting updatePredicateRanges method 2...')
            new_pred_lb, new_pred_ub = ProbStar.updatePredicateRanges(newC, newd, pred_lb, pred_ub)
            print('new_pred_lb: {}, new_pred_ub: {}'.format(new_pred_lb, new_pred_ub))
            assert new_pred_lb[1] == 0.0, 'error: wrong results'
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_addConstraint(self):
        
        self.n_tests = self.n_tests + 1

        mu = np.random.rand(2,)
        Sig = np.eye(2)
        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        C = np.array([-0.25, 1.0])
        d = np.array([0.25])

        S.addConstraint(C, d)

        try:
            print('\nTesting addConstraint method...')
            print('Before adding new constraint')
            S.__str__()
            S.addConstraint(C, d)
            print('After adding new constraint')
            S.__str__()
        except:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")


       

        
if __name__ == "__main__":
    
    test_probstar = Test()
    
    print('\n======================================================================================================================\n')
    test_probstar.test_constructor()
    test_probstar.test_str()
    test_probstar.test_estimateRange()
    test_probstar.test_glpk()
    test_probstar.test_getMin()
    test_probstar.test_getMax()
    test_probstar.test_affineMap()
    test_probstar.test_isEmptySet()
    test_probstar.test_updatePredicateRanges()
    test_probstar.test_addConstraint()
    print('\n======================================================================================================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, total tests: {}'.format(test_probstar.n_fails, test_probstar.n_tests - test_probstar.n_fails, test_probstar.n_tests))

        
