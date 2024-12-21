"""
Sparse Star Class
Sung Woo Choi, 04/03/2023

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

    def __init__(self, *args, copy_=True):
        """
            Key Attributes:
            A = []; % independent basis matrix 
            C = []; % constraint matrix
            d = []; % constraint vector
            dim = 0; % dimension of the sparse star set
            nVars = 0; % number of predicate variables
            nZVars = 0; % number of non-basis (dependent) predicate varaibles
            pred_lb = []; % lower bound of predicate variables
            pred_ub = []; % upper bound of predicate variables
            pred_depth = []; % depth of predicate varaibles
        """

        len_ = len(args)
        if len_ == 6:
            [A, C, d, pred_lb, pred_ub, pred_depth] = args

            if copy_ is True:
				A = A.copy()
				C = C.copy()
                d = d.copy()
				pred_lb = pred_lb.copy()
				pred_ub = pred_ub.copy()
            
            assert isinstance(A, np.ndarray), \
            'error: basis matrix should be a 2D numpy array'
            assert isinstance(pred_lb, np.ndarray), \
            'error: lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray), \
            'error: upper bound vector should be a 1D numpy array'
            assert len(A.shape) == 2,  \
            'error: basis matrix should be a 2D numpy array'

            if len(d) > 0:
                assert isinstance(C, sp.csc_matrix), \
                'error: constraint matrix should be a 2D scipy sparse csc matrix'
                assert isinstance(d, np.ndarray), \
                'error: constraint vector should be a 1D numpy array'
                assert len(C.shape) == 2, \
                'error: constraint matrix should be a 2D numpy array'
                assert len(d.shape) == 1, \
                'error: constraint vector should be a 1D numpy array'
                assert C.shape[0] == d.shape[0], \
                'error: inconsistency between constraint matrix and constraint vector'
                assert C.shape[1] == pred_lb.shape[0], \
                'error: inconsistent number of predicatve variables between constratint matrix and predicate bound vectors'

            assert len(pred_lb.shape) == 1, \
            'error: lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, \
            'error: upper bound vector should be a 1D numpy array'
            assert pred_ub.shape[0] == pred_lb.shape[0], \
            'error: inconsistent number of predicate variables between predicate lower- and upper-boud vectors'
            assert pred_lb.shape[0] == pred_depth.shape[0], \
            'error: inconsistent number of predicate variables between predicate bounds and predicate depth'

            self.A = A
            self.C = C
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.pred_depth = pred_depth
            self.dim = self.A.shape[0]

            self.nVars = self.C.shape[1] #number of predicate variables
            self.nZVars = self.C.shape[1] + 1 - self.A.shape[1] # number of dependent predicate variables
            self.nIVars = self.A.shape[1] - 1 #number of independent variables



            # if len(d) > 0:
            # self.nVars = self.C.shape[1]
            # self.nZVars = self.C.shape[1] + 1 - self.A.shape[1]
            # else:
            #     self.nVars = self.A.shape[1] - 1
            #     self.nZVars = 0

        # elif len_ == 2:
        #     [lb, ub] = copy.deepcopy(args)

        #     assert isinstance(lb, np.ndarray), 'error: ' + \
        #     'lower bound vector should be a 1D numpy array'
        #     assert isinstance(ub, np.ndarray), 'error: ' + \
        #     'upper bound vector should be a 1D numpy array'
        #     assert len(lb.shape) == 1, 'error: ' + \
        #     'lower bound vector should be a 1D numpy array'
        #     assert len(ub.shape) == 1, 'error: ' + \
        #     'upper bound vector should be a 1D numpy array'

        #     assert lb.shape[0] == ub.shape[0], 'error: ' + \
        #     'inconsistency between predicate lower bound and upper bound'
        #     if np.any(ub < lb):
        #         raise RuntimeError(
        #             "The upper bounds must not be less than the lower bounds for all dimensions")

        #     self.dim = lb.shape[0]
        #     nv = int(sum(ub > lb))
        #     c = 0.5 * (lb + ub)
        #     if self.dim == nv:
        #         v = np.diag(0.5 * (ub - lb))
        #     else:
        #         v = np.zeros((self.dim, nv))
        #         j = 0
        #         for i in range(self.dim):
        #             if ub[i] > lb[i]:
        #                 v[i, j] = 0.5 * (ub[i] - lb[i])
        #                 j += 1

        #     self.A = np.column_stack([c, v])
        #     self.C = sp.csc_matrix((0, self.dim))
        #     self.d = np.empty([0])
        #     self.pred_lb = -np.ones(self.dim)
        #     self.pred_ub = np.ones(self.dim)
        #     self.pred_depth = np.zeros(self.dim)
        #     self.dim = self.dim
        #     self.nVars = self.dim
        #     self.nZVars = self.dim + 1 - self.A.shape[1]

        elif len_ == 2:
            [lb, ub] = args
				
            if copy_ is True:
                lb = lb.copy()
				ub = ub.copy()          

            assert isinstance(lb, np.ndarray), \
            'error: lower bound vector should be a 1D numpy array'
            assert isinstance(ub, np.ndarray), \
            'error: upper bound vector should be a 1D numpy array'
            assert len(lb.shape) == 1, \
            'error: lower bound vector should be a 1D numpy array'
            assert len(ub.shape) == 1, \
            'error: upper bound vector should be a 1D numpy array'

            assert lb.shape[0] == ub.shape[0], \
            'error: inconsistency between predicate lower bound and upper bound'
            if np.any(ub < lb):
                raise Exception(
                    'error: the upper bounds must not be less than the lower bounds for all dimensions')

            self.dim = lb.shape[0]
            # nv = int(sum(ub >= lb)) # if using predicate bounds
            indx = np.argwhere(ub > lb).flatten()
            nv = len(indx)
            # nv = int(sum(ub > lb))
            self.A = np.zeros((self.dim, nv+1))
            j = 1
            for i in range(self.dim):
                if ub[i] == lb[i]:
                    self.A[i, 0] = lb[i]
                # if ub[i] >= lb[i]:
                else:
                    self.A[i, j] = 1
                    j += 1
            
            self.C = sp.csc_matrix((0, nv))
            self.d = np.empty([0])
            self.pred_lb = lb[indx] #lb
            self.pred_ub = ub[indx] #ub
            self.pred_depth = np.zeros(nv)
            self.nVars = nv #self.dim
            self.nZVars = nv + 1 - self.A.shape[1] #self.dim + 1 - self.A.shape[1]
            self.nIVars = nv #self.dim

        elif len_ == 1:
            [P] = copy.deepcopy(args)

            assert isinstance(P, pc.Polytope), \
            'error: input set is not a polytope Polytope'

            c = np.zeros([P.dim, 1])
            I = np.eye(P.dim)

            self.A = np.hstack([c, I])
            self.C = sp.csc_matrix(P.A)
            self.d = P.b
            self.dim = P.dim
            self.nVars = P.dim
            self.nZVars = 0
            self.nIVars = P.dim
            self.pred_lb = np.empty([0])
            self.pred_ub = np.empty([0])
            self.pred_depth = np.zeros(self.dim)
            self.pred_lb, self.pred_ub = self.getRanges()

        elif len_ == 0:
            self.A = np.empty([0, 0])
            self.C = sp.csc_matrix((0, 0))
            self.d = np.empty([0])
            self.pred_lb = np.empty([0])
            self.pred_ub = np.empty([0])
            self.pred_depth = np.empty([0])
            self.dim = 0
            self.nVars = 0
            self.nZVars = 0
            self.nIVars = 0

        else:
            raise Exception(
                'error: invalid number of input arguments (should be 0, 1, 2, 6)')

    def __str__(self, toDense=True):
        print('SparseStar Set:')
        print('A: \n{}'.format(self.A))
        if toDense:
            print('C_{}: \n{}'.format(self.C.getformat(), self.C.todense()))
        else:
            print('C: {}'.format(self.C))
        print('d: {}'.format(self.d))
        print('pred_lb: {}'.format(self.pred_lb))
        print('pred_ub: {}'.format(self.pred_ub))
        print('pred_depth: {}'.format(self.pred_depth))
        print('dim: {}'.format(self.dim))
        print('nVars: {}'.format(self.nVars))
        print('nZVars: {}'.format(self.nZVars))
        print('nIVars: {}'.format(self.nIVars))
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
        print('nIVars: {}'.format(self.nIVars))
        return '\n'
    
    def __len__(self):
        return 1

    def c(self, index=None):
        """Get center column vector of SparseStar"""
        if index is None:
            return copy.deepcopy(self.A[:, 0].reshape(-1, 1))
        else:
            return copy.deepcopy(self.A[index, 0].reshape(-1, 1))

    def X(self, row=None):
        """Get basis matrix of dependent predicate variables"""
        mA = self.A.shape[1]
        if row is None:
            return copy.deepcopy(self.A[:, 1:mA])
        else:
            return copy.deepcopy(self.A[row, 1:mA])

    def V(self, row=None):
        """Get basis matrix"""
        mA = self.A.shape[1]
        if row is None:
            return copy.deepcopy(np.column_stack([np.zeros((self.dim, self.nZVars)), self.X()]))
        else:
            if isinstance(row, int) or isinstance(row, np.integer):
                return copy.deepcopy(np.hstack([np.zeros(self.nZVars), self.X(row)]))
            else:
                return copy.deepcopy(np.column_stack([np.zeros((len(row), self.nZVars)), self.X(row)]))

    def translation(self, v=None):
        """Translation of a sparse star: S = self + v"""
        if v is None:
            return copy.deepcopy(self)

        if isinstance(v, np.ndarray):
            assert isinstance(v, np.ndarray) and v.ndim == 1, \
            'error: the translation vector should be a 1D numpy array or an integer'
            assert v.shape[0] == self.dim, \
            'error: inconsistency between translation vector and SparseStar dimension'

        elif isinstance(v, int) or isinstance(v, float):
            pass

        else:
            raise Exception('the translation vector, v, should a 1D numpy array, an integer, or a float')

        A = copy.deepcopy(self.A)
        A[:, 0] += v
        return SparseStar(A, self.C, self.d, self.pred_lb, self.pred_ub, self.pred_depth)

    def affineMap(self, W=None, b=None):
        """Affine mapping of a sparse star: S = W*self + b"""

        if W is None and b is None:
            return copy.deepcopy(self)
        
        A = copy.deepcopy(self.A)

        if W is not None:
            assert isinstance(W, np.ndarray), 'error: ' + \
            'the mapping matrix should be a 2D numpy array'
            assert W.shape[1] == self.dim, 'error: ' + \
            'inconsistency between mapping matrix and SparseStar dimension'

            A = np.matmul(W, A)

        if b is not None:
            assert isinstance(b, np.ndarray), 'error: ' + \
            'the offset vector should be a 1D numpy array'
            assert len(b.shape) == 1, 'error: ' + \
            'offset vector should be a 1D numpy array'

            if W is not None:
                assert W.shape[0] == b.shape[0], 'error: ' + \
                'inconsistency between mapping matrix and offset'
            else:
                assert b.shape[0] == self.dim, 'error: ' + \
                'inconsistency between offset vector and SparseStar dimension'

            A[:, 0] += b

        return SparseStar(A, self.C, self.d, self.pred_lb, self.pred_ub, self.pred_depth)

    def getMin(self, index, lp_solver='gurobi'):
        """Get the minimum value of state x[index] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
        """
        assert index >= 0 and index <= self.dim-1, 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        f = self.V(index)
        if (f == 0).all():
            xmin = self.c(index)
        else:
            if lp_solver == 'gurobi':  # gurobi is the preferred LP solver

                min_ = gp.Model()
                min_.Params.LogToConsole = 0
                min_.Params.OptimalityTol = 1e-9
                if self.pred_lb.size and self.pred_ub.size:
                    x = min_.addMVar(shape=self.nVars,
                                     lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = min_.addMVar(shape=self.nVars)
                min_.setObjective(f @ x, GRB.MINIMIZE)
                if len(self.d) > 0:
                    C = self.C
                    d = self.d
                else:
                    C = sp.csr_matrix(np.zeros((1, self.nVars)))
                    d = 0
                min_.addConstr(C @ x <= d)
                min_.optimize()

                if min_.status == 2:
                    xmin = min_.objVal + self.c(index)
                else:
                    raise Exception('error: cannot find an optimal solution, ' + \
                        'exitflag = %d' % (min_.status))

            elif lp_solver == 'linprog':

                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
                if len(self.d) == 0:
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
                    xmin = res.fun + self.c(index)
                else:
                    raise Exception('error: cannot find an optimal solution, ' + \
                        'exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                #  https://pyglpk.readthedocs.io/en/latest/examples.html
                #  https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False

                if len(self.d) == 0:
                    A = np.zeros((1, self.nVars))
                    b = np.zeros(1)
                else:
                    A = self.C.toarray()
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
                    raise Exception('error: cannot find an optimal solution, ' + \
                        'lp.status = {}'.format(lp.status))
                else:
                    xmin = lp.obj.value + self.c(index)

            else:
                raise Exception(
                    'error: unknown lp solver, should be gurobi or linprog or glpk')
        return xmin

    def getMax(self, index, lp_solver='gurobi'):
        """Get the minimum value of state x[index] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
        """

        assert index >= 0 and index <= self.dim-1, 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        f = self.V(index)
        if (f == 0).all():
            xmax = self.c(index)
        else:
            if lp_solver == 'gurobi':  # gurobi is the preferred LP solver

                max_ = gp.Model()
                max_.Params.LogToConsole = 0
                max_.Params.OptimalityTol = 1e-9
                if self.pred_lb.size and self.pred_ub.size:
                    x = max_.addMVar(shape=self.nVars,
                                     lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = max_.addMVar(shape=self.nVars)
                max_.setObjective(f @ x, GRB.MAXIMIZE)
                if len(self.d) > 0:
                    C = self.C
                    d = self.d
                else:
                    C = sp.csr_matrix(np.zeros((1, self.nVars)))
                    d = 0
                max_.addConstr(C @ x <= d)
                max_.optimize()

                if max_.status == 2:
                    xmax = max_.objVal + self.c(index)
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = %d' % (max_.status))

            elif lp_solver == 'linprog':
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
                if len(self.d) == 0:
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
                    xmax = -res.fun + self.c(index)
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                # https://pyglpk.readthedocs.io/en/latest/examples.html
                # https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False  # turn off messages/display

                if len(self.d) == 0:
                    A = np.zeros((1, self.nVars))
                    b = np.zeros(1)
                else:
                    A = self.C.toarray()
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
                    xmax = lp.obj.value + self.c(index)
            else:
                raise Exception('error: \
                unknown lp solver, should be gurobi or linprog or glpk')
        return xmax

    def getMins(self, map, lp_solver='gurobi'):
        n = len(map)
        xmin = np.zeros(n)
        for i in range(n):
            xmin[i] = self.getMin(index=map[i], lp_solver=lp_solver)
        return xmin

    def getMaxs(self, map, lp_solver='gurobi'):
        n = len(map)
        xmax = np.zeros(n)
        for i in range(n):
            xmax[i] = self.getMax(index=map[i], lp_solver=lp_solver)
        return xmax

    def estimateRange(self, index):
        """Estimate the minimum and maximum values of a state x[index]"""

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
        """Estimate the lower and upper bounds of x"""

        mA = self.A.shape[1]
        n = self.nVars
        p = n - self.nIVars

        l = self.pred_lb[p:n]
        u = self.pred_ub[p:n]

        pos_f = np.maximum(self.X(), 0.0)
        neg_f = np.minimum(self.X(), 0.0)

        xmin = self.c().flatten() + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        xmax = self.c().flatten() + np.matmul(neg_f, l) + np.matmul(pos_f, u)
        return xmin, xmax

    def getRange(self, index, lp_solver='gurobi'):
        """Get the lower and upper bounds of x[index]"""

        if lp_solver == 'estimate':
            return self.estimateRange(index)
        else:
            l = self.getMin(index, lp_solver=lp_solver)
            u = self.getMax(index, lp_solver=lp_solver)
            return l, u

    def getRanges(self, lp_solver='gurobi', RF=0.0, layer=None, delta=0.98):
        """Get the lower and upper bound vectors of the state
            Args:
                lp_solver: linear programming solver. e.g.: 'gurobi', 'estimate', 'linprog'
                RF: relaxation factor \in [0.0, 1.0]
        """
        
        if RF == 1.0:
            return self.estimateRanges()

        elif RF == 0.0:
            if lp_solver == 'estimate':
                return self.estimateRanges()
            else:
                l = self.getMins(np.arange(self.dim), lp_solver=lp_solver)
                u = self.getMaxs(np.arange(self.dim), lp_solver=lp_solver)
                return l, u

        else:
            assert RF > 0.0 and RF <= 1.0, \
            'error: relaxation factor should be greater than 0.0 but less than or equal to 1.0'
            l, u = self.estimateRanges()
            n1 = round((1 - RF) * self.dim)
            if layer in ['logsig', 'tansig']:
                midx = np.argsort((u - l))[::-1]
                midb = np.argwhere((l[midx] >= -delta) & (u[midx] <= delta))
                
                n2 = n1
                check = midb.flatten().shape[0]
                if n2 > check:
                    n2 = check

                mid = midx[midb[0:n2]]
                l1 = self.getMins(mid)
                u1 = self.getMaxs(mid)
                l[mid] = l1
                u[mid] = u1
            else:
                midx = np.argsort((u - l))[::-1]
                mid = midx[0:n1]
                l1 = self.getMins(mid)
                u1 = self.getMaxs(mid)
                l[mid] = l1
                u[mid] = u1
            return l, u

    def predReduction(self, p_map):
        """Remove selected predicate variables in p_map"""

        assert (p_map >= 0).all() and (p_map < self.nVars).all(), 'error: ' + \
        'invalid predicate indexes, should be between {} and {}'.format(0, self.nVars)

        C = copy.deepcopy(self.C)
        d = copy.deepcopy(self.d)
        pred_lb = copy.deepcopy(self.pred_lb)
        pred_ub = copy.deepcopy(self.pred_ub)
        pred_depth = copy.deepcopy(self.pred_depth)

        nC = C[:, p_map].nonzero()[0]
        q = np.unique(nC)

        pm = np.setdiff1d(np.arange(C.shape[1]), p_map)
        pn = np.setdiff1d(np.arange(C.shape[0]), q)

        # remove linear constraints of predicate variables in p_map
        C = C[pn, :]
        d = d[pn]
        # remove predicate variables in p_map
        C = C[:, pm]
        pred_lb = pred_lb[pm]
        pred_ub = pred_ub[pm]
        pred_depth = pred_depth[pm]

        return SparseStar(self.A, C, d, pred_lb, pred_ub, pred_depth)
    
    # def predReduction2(self, p_map):
    #     """Remove predicate variables that are not listed in p_map"""

    #     assert (p_map >= 0).all() and (p_map < self.nVars).all(), 'error: ' + \
    #     'invalid predicate indexes, should be between {} and {}'.format(0, self.nVars)

    #     C = copy.deepcopy(self.C)
    #     d = copy.deepcopy(self.d)
    #     pred_lb = copy.deepcopy(self.pred_lb)
    #     pred_ub = copy.deepcopy(self.pred_ub)
    #     pred_depth = copy.deepcopy(self.pred_depth)

    #     nC = C[:, p_map].nonzero()[0] # get linear constraints of predicate variables in p_map
    #     q = np.unique(nC)

    #     # keep linear constraints of predicate variables in p_map
    #     C = C[q, :]
    #     d = d[q]
    #     # keep predicate variables in p_map
    #     C = C[:, p_map]
    #     pred_lb = pred_lb[p_map]
    #     pred_ub = pred_ub[p_map]
    #     pred_depth = pred_depth[p_map]

    #     return SparseStar(self.A, C, d, pred_lb, pred_ub, pred_depth)

    def depthReduction(self, DR):
        """
        Reduce predicate variables based on the depth of predicate varibles
        Args:
            DR: (depth reduction) maximum allowed depth of predicate variables
        """

        assert DR >= 0, 'error: maximum allowed predicate variables depth should be non-negative'

        max_depth = self.pred_depth.max()
        if DR > max_depth:
            return copy.deepcopy(self)

        p_map = np.where(self.pred_depth >= DR)[0]
        return self.predReduction(p_map)
    
    # def depthReduction2(self, DR):
    #     """
    #     Reduce predicate variables based on the depth of predicate varibles
    #     Args:
    #         DR: (depth reduction) maximum allowed depth of predicate variables
    #     """

    #     assert DR >= 0, 'error: maximum allowed predicate variables depth should be non-negative'

    #     max_depth = self.pred_depth.max()
    #     if DR > max_depth:
    #         return copy.deepcopy(self)
    
    #     p_map = np.where(self.pred_depth < DR)[0]
    #     return self.predReduction2(p_map)


    def resetRows(self, map):
        """Reset a row with a map of indexes"""

        A = self.A
        A[map, :] = 0.0
        return SparseStar(A, self.C, self.d, self.pred_lb, self.pred_ub, self.pred_depth)

    def isEmptySet(self, lp_solver='gurobi'):
        """Check if a SparseStar is an empty set"""
        res = False
        try:
            self.getMin(0, lp_solver)
        except Exception:
            res = True
        return res  

    def minKowskiSum(self, S):
        """Minkowski Sum of two sparse stars"""

        assert isinstance(S, SparseStar), 'error: input is not a SparseStar'
        assert self.dim == S.dim, 'error: inconsistent dimension between the input and the self object'

        X = np.hstack((self.X(), S.X()))
        c = self.c() + S.c()
        A = np.hstack((c, X))

        OC1 = self.C[:, 0:self.nZVars]
        OC2 = self.C[:, self.nZVars:self.nVars]

        SC1 = S.C[:, 0:S.nZVars]
        SC2 = S.C[:, S.nZVars:S.nVars]

        C1 = sp.block_diag((OC1, SC1))
        C2 = sp.block_diag((OC2, SC2))

        if C1.nnz > 0:
            C = sp.hstack((C1, C2)).tocsc()
        else:
            C = C2.tocsc()
        
        d = np.concatenate((self.d, S.d))
        
        pred_lb = np.hstack((self.pred_lb[0:self.nZVars],          S.pred_lb[0:S.nZVars],
                             self.pred_lb[self.nZVars:self.nVars], S.pred_lb[S.nZVars:S.nVars]))
        pred_ub = np.hstack((self.pred_ub[0:self.nZVars],          S.pred_ub[0:S.nZVars],
                             self.pred_ub[self.nZVars:self.nVars], S.pred_ub[S.nZVars:S.nVars]))
        pred_depth = np.hstack((self.pred_depth[0:self.nZVars], S.pred_depth[0:S.nZVars],
                                self.pred_depth[self.nZVars:self.nVars], S.pred_depth[S.nZVars:S.nVars]))
        
        return SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)
    
    # def Product(self, S):
    #     """Product of two sparse stars """

    #     assert isinstance(S, SparseStar), 'error: input is not a SparseStar'
    #     assert self.dim == S.dim, 'error: inconsistent dimension between the input and the self object'

    #     X = np.hstack((self.X(), S.X()))
    #     c = self.c() * S.c()
    #     A = np.hstack((c, X))

    #     OC1 = self.C[:, 0:self.nZVars]
    #     OC2 = self.C[:, self.nZVars:self.nVars]

    #     SC1 = S.C[:, 0:S.nZVars]
    #     SC2 = S.C[:, S.nZVars:S.nVars]

    #     C1 = sp.block_diag((OC1, SC1))
    #     C2 = sp.block_diag((OC2, SC2))

    #     if C1.nnz > 0:
    #         C = sp.hstack((C1, C2)).tocsc()
    #     else:
    #         C = C2.tocsc()
    #     d = np.concatenate((self.d, S.d))

    #     pred_lb = np.hstack((self.pred_lb[0:self.nZVars], S.pred_lb[0:S.nZVars],
    #                         self.pred_lb[self.nZVars:self.nVars], S.pred_lb[S.nZVars:S.nVars]))
    #     pred_ub = np.hstack((self.pred_ub[0:self.nZVars], S.pred_ub[0:S.nZVars],
    #                         self.pred_ub[self.nZVars:self.nVars], S.pred_ub[S.nZVars:S.nVars]))
    #     pred_depth = np.hstack((self.pred_depth[0:self.nZVars], S.pred_depth[0:S.nZVars],
    #                             self.pred_depth[self.nZVars:self.nVars], S.pred_depth[S.nZVars:S.nVars]))
        
    #     return SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)

    def concatenate(self, S):
        """Concatenate two sparse star sets """

        assert isinstance(S, SparseStar), 'error: input is not a SparseStar'

        c = np.concatenate((self.c(), S.c()))
        X = block_diag(self.X(), S.X())
        A = np.hstack((c, X))

        OC1 = self.C[:, 0:self.nZVars]
        OC2 = self.C[:, self.nZVars:self.nVars]

        SC1 = S.C[:, 0:S.nZVars]
        SC2 = S.C[:, S.nZVars:S.nVars]

        C1 = sp.block_diag((OC1, SC1))
        C2 = sp.block_diag((OC2, SC2))

        if C1.nnz > 0:
            C = sp.hstack((C1, C2)).tocsc()
        else:
            C = C2.tocsc()
        d = np.concatenate((self.d, S.d))

        pred_lb = np.hstack((self.pred_lb[0:self.nZVars], S.pred_lb[0:S.nZVars],
                            self.pred_lb[self.nZVars:self.nVars], S.pred_lb[S.nZVars:S.nVars]))
        pred_ub = np.hstack((self.pred_ub[0:self.nZVars], S.pred_ub[0:S.nZVars],
                            self.pred_ub[self.nZVars:self.nVars], S.pred_ub[S.nZVars:S.nVars]))
        pred_depth = np.hstack((self.pred_depth[0:self.nZVars], S.pred_depth[0:S.nZVars],
                                self.pred_depth[self.nZVars:self.nVars], S.pred_depth[S.nZVars:S.nVars]))
        return SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)

    def sample(self, N):
        """Sample number of points in the feasible SparseStar set"""

        assert N >= 1, 'error: invalid number of samples'

        [lb, ub] = self.getRanges(lp_solver='gurobi', RF=0.0)

        V1 = np.array([])
        for i in range(self.dim):
            X = (ub[i] - lb[i]) * np.random.rand(2*N, 1) + lb[i]
            V1 = np.hstack([V1, X]) if V1.size else X

        V = np.array([])
        for i in range(2*N):
            v1 = V1[i, :]
            if self.contains(v1):
                V = np.vstack([V, v1]) if V.size else V1

        V = V.T
        if V.shape[1] >= N:
            V = V[:, 0:N]
        return V

    def toStar(self):
        """Converts sparse star set into star set"""
        if len(self.d) > 0:
            return Star(np.column_stack((self.c(), self.V())), self.C.todense(), self.d, self.pred_lb, self.pred_ub)
        else:
            return Star(np.column_stack((self.c(), self.V())), np.array([]), self.d, self.pred_lb, self.pred_ub)

    def contains(self, s):
        """ Check if a Star set contains a point.
            s : a star point (1D numpy array)

            return :
                1 -> a star set contains a point, s 
                0 -> a star set does not contain a point, s
                else -> error code from Gurobi LP solver
        """
        assert len(
            s.shape) == 1, 'error: invalid point. It should be 1D numpy array'
        assert s.shape[0] == self.dim, 'error: Dimension mismatch'

        f = np.zeros(self.nVars)
        m = gp.Model()
        # prevent optimization information
        m.Params.LogToConsole = 0
        m.Params.OptimalityTol = 1e-9
        if self.pred_lb.size and self.pred_ub.size:
            x = m.addMVar(shape=self.nVars, lb=self.pred_lb, ub=self.pred_ub)
        else:
            x = m.addMVar(shape=self.nVars)
        m.setObjective(f @ x, GRB.MINIMIZE)
        if len(self.d) > 0:
            C = self.C
            d = self.d
        else:
            C = sp.csr_matrix(np.zeros((1, self.nVars)))
            d = 0
        m.addConstr(C @ x <= d)
        Ae = sp.csr_matrix(self.V())
        be = s - self.c()
        m.addConstr(Ae @ x == be)
        m.optimize()

        if m.status == 2:
            return True
        elif m.status == 3:
            return False
        else:
            raise Exception('error: exitflat = %d' % (m.status))
        
    def get_max_point_cadidates(self):
        """ Quickly estimate max-point candidates """

        lb, ub = self.getRanges('estimate')
        max_id = np.argmax(lb)
        a = (ub > lb[max_id])
        if sum(a) == 1:
            return [max_id]
        else:
            return np.where(a)[0]
        
    def is_p1_larger_than_p2(self, p1_indx, p2_indx):
        """
            Check if an index is larger than the other

            Arg:
                @p1_indx: an index of point 1
                @p2_indx: an index of point 2

            return:
                @bool = 1 if there exists the case that p1 >= p2
                        2 if there is no case that p1 >= p2; p1 < p2
        """

        assert p1_indx >= 0 and p1_indx < self.dim, 'error: invalid index for point 1'
        assert p2_indx >= 0 and p2_indx < self.dim, 'error: invalid index for point 2'

        d1 = self.c(p1_indx) - self.c(p2_indx)
        C1 = self.X(p2_indx) - self.X(p1_indx)
        Z1 = sp.csc_matrix((1, self.nZVars))
        C1 = sp.hstack((Z1, C1))

        d = np.hstack((self.d, d1.flatten()))
        C = sp.vstack((self.C, C1)).tocsc()
        S = SparseStar(self.A, C, d, self.pred_lb, self.pred_ub, self.pred_depth)
        
        if S.isEmptySet():
            return False
        else:
            return True
        # return not S.isEmptySet()

    @staticmethod
    def inf_attack(data, epsilon=0.01, data_type='default', dtype='float64'):
        """Generate a SparseStar set by infinity norm attack on input dataset"""

        assert isinstance(data, np.ndarray), \
        'error: the data should be a 1D numpy array'
        assert len(data.shape) == 1, \
        'error: the data should be a 1D numpy array'

        lb = data - epsilon
        ub = data + epsilon

        data = data.astype(dtype)

        if data_type == 'image':
            lb[lb < 0] = 0
            ub[ub > 1] = 1

        return SparseStar(lb, ub)
    
    @staticmethod
    def inf_attack_sequence(data, seq_index=0, percent=0.5, attack_type='SFSI', dtype='float64'):
        """Generate a Sparse set by infinity norm attack on sequential input dataset"""
        # data: input data ins shape of [input_size, sequence_size]
        # seq_index: an index of sequence to attack in the input data
        # percent: percentage to apply noise attack
        # attack_type: type of noise attack: 'SFSI', 'SFAI', 'MFSI', 'MFAI'

        lb = copy.deepcopy(data)
        ub = copy.deepcopy(data)

        if attack_type == 'SFSI':
            mu = abs(np.mean(data[seq_index, :]))*percent
            lb[seq_index, -1] -= mu
            ub[seq_index, -1] += mu

        elif attack_type == 'SFAI':
            mu = abs(np.mean(data[seq_index, :]))*percent
            lb[seq_index, :] -= mu
            ub[seq_index, :] += mu

        elif attack_type == 'MFSI':
            mu = abs(np.mean(data, axis=1))*percent
            lb[:, -1] -= mu
            ub[:, -1] += mu

        elif attack_type == 'MFAI':
            mu = abs(np.mean(data, axis=1))[:, None]*percent
            lb -= mu
            ub += mu

        #just for testing
        elif attack_type == 'MFSF':
            mu = abs(np.mean(data, axis=1))*percent
            lb[:, 0] -= mu
            ub[:, 0] += mu

        else:
            raise Exception('Unknown noise attack type for audio data input')

        seq = lb.shape[1]
        sets = []
        for i in range(seq):
            sets.append(SparseStar(lb[:, i], ub[:, i]))
        return sets

    @staticmethod
    def rand(dim, N):
        """Generate a random SparseStar set"""

        assert dim > 0, 'error: invalid dimension'
        assert N > dim, 'error: number constraints should be greater than dimension'

        A = np.random.rand(N, dim)
        # compute the convex hull
        P = pc.qhull(A)
        return SparseStar(P)

    @staticmethod
    def rand_bounds(dim):
        """Generate a random SparStar by random bounds"""

        assert dim > 0, 'error: invalid dimension'
        lb = -np.random.rand(dim)
        ub = np.random.rand(dim)
        return SparseStar(lb, ub)
