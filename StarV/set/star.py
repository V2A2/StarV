"""
Updated Star Class
Author: Yuntao Li
Date: 1/20/2024
"""
import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple, Union
import polytope as pc
import copy
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog
import glpk
from scipy.linalg import block_diag

class Star:
    """
    Star Class for reachability analysis.

    Represents a Star set defined by:
    x = c + a[1]*v[1] + a[2]*v[2] + ... + a[n]*v[n] = V * b,
    where:
    - V = [c v[1] v[2] ... v[n]]
    - b = [1 a[1] a[2] ... a[n]]^T
    - C*a <= d (constraints on a[i])

    Attributes:
        V (np.ndarray): Basis matrix [dim x (nVars + 1)]
        C (np.ndarray): Constraint matrix [m x nVars]
        d (np.ndarray): Constraint vector [m]
        dim (int): Dimension of the star set
        nVars (int): Number of predicate variables
        pred_lb (np.ndarray): Lower bound of predicate variables [nVars]
        pred_ub (np.ndarray): Upper bound of predicate variables [nVars]
    """

    # def __init__(self,
    #              V: Optional[np.ndarray] = None,
    #              C: Optional[np.ndarray] = None,
    #              d: Optional[np.ndarray] = None,
    #              pred_lb: Optional[np.ndarray] = None,
    #              pred_ub: Optional[np.ndarray] = None):
    #     """
    #     Initialize a Star object.

    #     Args:
    #         *args: Variable length argument list.
    #             - If 5 arguments: V, C, d, pred_lb, pred_ub
    #             - If 2 arguments: lb, ub (lower and upper bounds)
    #             - If 0 arguments: Create an empty Star
    #     """
    #     if all(arg is not None for arg in [V, C, d, pred_lb, pred_ub]):
    #         self._init_from_matrices(*args)
    #     elif pred_lb is not None and pred_ub is not None:
    #         self._init_from_bounds(pred_lb, pred_ub)
    #     elif all(arg is None for arg in [V, C, d, pred_lb, pred_ub]):
    #         self._init_empty()
    #     else:
    #         raise ValueError("Invalid number of input arguments (should be 0, 2, or 5)")

    def __init__(self, *args):
        """
        Initialize a Star object.

        Args:
            *args: Variable length argument list.
                - If 5 arguments: V, C, d, pred_lb, pred_ub
                - If 2 arguments: lb, ub (lower and upper bounds)
                - If 0 arguments: Create an empty Star
        """
        if len(args) == 5:
            self._init_from_matrices(*args)
        elif len(args) == 2:
            self._init_from_bounds(*args)
        elif len(args) == 0:
            self._init_empty()
        else:
            raise ValueError("Invalid number of input arguments (should be 0, 2, or 5)")

    def _init_from_matrices(self, V: np.ndarray, C: np.ndarray, d: np.ndarray, 
                            pred_lb: np.ndarray, pred_ub: np.ndarray):
        """Initialize Star from matrices and vectors."""
        self._validate_matrices(V, C, d, pred_lb, pred_ub)
        self.V = V
        self.C = C
        self.d = d
        self.dim = V.shape[0]
        self.nVars = V.shape[1] - 1
        self.pred_lb = pred_lb
        self.pred_ub = pred_ub

    def _init_from_bounds(self, lb: np.ndarray, ub: np.ndarray):
        """Initialize Star from lower and upper bounds."""
        self._validate_bounds(lb, ub)
        self.dim = lb.shape[0]
        self.nVars = np.sum(ub > lb)
        
        center = 0.5 * (lb + ub)
        vec = 0.5 * (ub - lb)
        gens = np.diag(vec)
        gens = gens[:, ~np.all(gens == 0, axis=0)]

        self.V = np.hstack((center.reshape(-1, 1), gens))
        self.C = np.array([])
        self.d = np.array([])
        self.pred_lb = -np.ones(self.nVars)
        self.pred_ub = np.ones(self.nVars)

    def _init_empty(self):
        """Initialize an empty Star."""
        self.dim = 0
        self.nVars = 0
        self.V = np.array([])
        self.C = np.array([])
        self.d = np.array([])
        self.pred_lb = np.array([])
        self.pred_ub = np.array([])

    @staticmethod
    def _validate_matrices(V: np.ndarray, C: np.ndarray, d: np.ndarray, 
                           pred_lb: np.ndarray, pred_ub: np.ndarray):
        """Validate input matrices and vectors."""
        assert isinstance(V, np.ndarray) and V.ndim == 2, "Basis matrix should be a 2D numpy array"
        assert isinstance(pred_lb, np.ndarray) and pred_lb.ndim == 1, "Lower bound vector should be a 1D numpy array"
        assert isinstance(pred_ub, np.ndarray) and pred_ub.ndim == 1, "Upper bound vector should be a 1D numpy array"

        if len(C) > 0:
            assert C.ndim == 2, "Constraint matrix should be a 2D numpy array"
            assert d.ndim == 1, "Constraint vector should be a 1D numpy array"
            assert V.shape[1] == C.shape[1] + 1, "Inconsistency between basis matrix and constraint matrix"
            assert C.shape[0] == d.shape[0], "Inconsistency between constraint matrix and constraint vector"
            assert C.shape[1] == pred_lb.shape[0] == pred_ub.shape[0], "Inconsistency between number of predicate variables and predicate bound vectors"

    @staticmethod
    def _validate_bounds(lb: np.ndarray, ub: np.ndarray):
        """Validate lower and upper bounds."""
        assert isinstance(lb, np.ndarray) and lb.ndim == 1, "Lower bound vector should be a 1D numpy array"
        assert isinstance(ub, np.ndarray) and ub.ndim == 1, "Upper bound vector should be a 1D numpy array"
        assert lb.shape == ub.shape, "Inconsistency between predicate lower bound and upper bound"
        if np.any(ub < lb):
            raise ValueError("Upper bound must be greater than or equal to the lower bound for all dimensions")

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
        print('\n')
        return '\n'

    def __repr__(self):
        print('Star Set:')
        print('V: {}'.format(self.V.shape))
        print('Predicate Constraints:')
        print('C: {}'.format(self.C.shape))
        print('d: {}'.format(self.d.shape))
        print('dim: {}'.format(self.dim))
        print('nVars: {}'.format(self.nVars))
        print('pred_lb: {}'.format(self.pred_lb.shape))
        print('pred_ub: {}'.format(self.pred_ub.shape))
        print('\n')
        return '\n'
    
    def clone(self):
        """
        Create a deep copy of the Star object by copying each attribute individually.

        Returns:
            Star: A new Star object that is a deep copy of the current object.
        """
        new_star = Star.__new__(Star)  # Create a new ProbStar instance without calling __init__
        
        # Copy each attribute
        new_star.V = self.V.copy()
        new_star.C = self.C.copy() if self.C.size > 0 else np.array([])
        new_star.d = self.d.copy() if self.d.size > 0 else np.array([])
        new_star.pred_lb = self.pred_lb.copy()
        new_star.pred_ub = self.pred_ub.copy()
        new_star.dim = self.dim
        new_star.nVars = self.nVars

        return new_star

    def getMinimizedConstraints(self):
        """
        Minimize constraints of a Star set.

        Returns:
            tuple: (Cmin, dmin) - Minimized constraint matrix and vector
        """
        if len(self.C) == 0:
            return self.C, self.d
        
        P = pc.Polytope(self.C, self.d)
        pc.reduce(P)
        return P.A, P.b

    def estimateRange(self, index):
        """
        Quickly estimate the range of a state x[index].

        Args:
            index (int): Index of the state to estimate

        Returns:
            tuple: (min_val, max_val) - Estimated minimum and maximum values
        """
        if not 0 <= index < self.dim:
            raise ValueError(f"Invalid index: {index}. Must be between 0 and {self.dim-1}")
        
        v = self.V[index, 1:self.nVars+1]
        c = self.V[index, 0]
        
        v_neg = np.where(v < 0, v, 0)  # negative part
        v_pos = np.where(v > 0, v, 0)  # positive part
        
        min_val = c + np.dot(v_neg, self.pred_ub) + np.dot(v_pos, self.pred_lb)
        max_val = c + np.dot(v_neg, self.pred_lb) + np.dot(v_pos, self.pred_ub)
        
        return min_val, max_val

    def estimateRanges(self):
        """
        Quickly estimate lower and upper bounds of all states.

        Returns:
            tuple: (lb, ub) - Estimated lower and upper bounds for all states
        """
        v = self.V[:, 1:self.nVars+1]
        c = self.V[:, 0]
        
        v_neg = np.where(v < 0, v, 0)  # negative part
        v_pos = np.where(v > 0, v, 0)  # positive part
        
        lb = c + np.dot(v_neg, self.pred_ub) + np.dot(v_pos, self.pred_lb)
        ub = c + np.dot(v_neg, self.pred_lb) + np.dot(v_pos, self.pred_ub)
        
        return lb, ub

    def _solve_lp(self, f, minimize=True, lp_solver='gurobi'):
        """
        Solve a linear programming problem.

        Args:
            f (objective) (np.ndarray): Objective function coefficients.
            minimize (bool): True for minimization, False for maximization.
            lp_solver (str): The LP solver to use ('gurobi', 'linprog', or 'glpk').

        Returns:
            float: The optimal value of the objective function.

        Raises:
            ValueError: If an unknown LP solver is specified.
        """
        if lp_solver == 'gurobi':
            model = gp.Model()
            model.Params.LogToConsole = 0
            model.Params.OptimalityTol = 1e-9
            if self.pred_lb.size and self.pred_ub.size:
                x = model.addMVar(shape=self.nVars, lb=self.pred_lb, ub=self.pred_ub)
            else:
                x = model.addMVar(shape=self.nVars)
            model.setObjective(f @ x, GRB.MINIMIZE if minimize else GRB.MAXIMIZE)
            if len(self.C) == 0:
                C = sp.csr_matrix(np.zeros((1, self.nVars)))
                d = 0
            else:
                C = sp.csr_matrix(self.C)
                d = self.d
            model.addConstr(C @ x <= d)
            model.optimize()

            if model.status == 2:
                return model.objVal
            else:
                raise Exception(f'error: cannot find an optimal solution, exitflag = {model.status}')

        elif lp_solver == 'linprog':
            if len(self.C) == 0:
                A = np.zeros((1, self.nVars))
                b = np.zeros(1)
            else:
                A = self.C
                b = self.d

            lb = self.pred_lb.reshape((self.nVars, 1))
            ub = self.pred_ub.reshape((self.nVars, 1))
            res = linprog(f if minimize else -f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))

            if res.status == 0:
                return res.fun if minimize else -res.fun
            else:
                raise Exception(f'error: cannot find an optimal solution, exitflag = {res.status}')

        elif lp_solver == 'glpk':
            glpk.env.term_on = False

            if len(self.C) == 0:
                A = np.zeros((1, self.nVars))
                b = np.zeros(1)
            else:
                A = self.C
                b = self.d

            lb = self.pred_lb.reshape((self.nVars, 1))
            ub = self.pred_ub.reshape((self.nVars, 1))

            lp = glpk.LPX()
            lp.obj.maximize = not minimize
            lp.rows.add(A.shape[0])
            for r in lp.rows:
                r.name = chr(ord('p') + r.index)
                lp.rows[r.index].bounds = None, b[r.index]

            lp.cols.add(self.nVars)
            for c in lp.cols:
                c.name = f'x{c.index}'
                c.bounds = lb[c.index], ub[c.index]

            lp.obj[:] = f.tolist()
            B = A.reshape(A.shape[0]*A.shape[1],)
            lp.matrix = B.tolist()

            lp.simplex()

            if lp.status != 'opt':
                raise Exception(f'error: cannot find an optimal solution, lp.status = {lp.status}')
            else:
                return lp.obj.value

        else:
            raise Exception('error: unknown lp solver, should be gurobi or linprog or glpk')

    def getMin(self, index, lp_solver='gurobi'):
        """
        Get exact minimum value of state x[index] by solving LP.

        Args:
            index (int): Index of the state variable.
            lp_solver (str): The LP solver to use ('gurobi', 'linprog', or 'glpk').

        Returns:
            float: The minimum value of the state variable.

        Raises:
            ValueError: If the index is invalid.

        Mathematical formulation:
        Minimize f^T * x
        subject to:
            C * x <= d
            pred_lb <= x <= pred_ub
        where f is the basis vector for the given index.
        """
        if not 0 <= index < self.dim:
            raise ValueError(f"Invalid index: {index}. Must be between 0 and {self.dim-1}")

        f = self.V[index, 1:self.nVars + 1]
        if np.allclose(f, 0):
            return self.V[index, 0]
        
        xmin = self._solve_lp(f, minimize=True, lp_solver=lp_solver)
        return xmin + self.V[index, 0]

    def getMax(self, index, lp_solver='gurobi'):
        """
        Get exact maximum value of state x[index] by solving LP.

        Args:
            index (int): Index of the state variable.
            lp_solver (str): The LP solver to use ('gurobi', 'linprog', or 'glpk').

        Returns:
            float: The maximum value of the state variable.

        Raises:
            ValueError: If the index is invalid.

        Mathematical formulation:
        Maximize f^T * x
        subject to:
            C * x <= d
            pred_lb <= x <= pred_ub
        where f is the basis vector for the given index.
        """
        if not 0 <= index < self.dim:
            raise ValueError(f"Invalid index: {index}. Must be between 0 and {self.dim-1}")

        f = self.V[index, 1:self.nVars + 1]
        if np.allclose(f, 0):
            return self.V[index, 0]
        
        xmax = self._solve_lp(f, minimize=False, lp_solver=lp_solver)
        return xmax + self.V[index, 0]

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

    def getRanges(self, lp_solver='gurobi'):
        """
        Get lower and upper bounds for all state variables by solving LPs.

        Args:
            lp_solver (str): The LP solver to use ('gurobi', 'linprog', or 'glpk').

        Returns:
            tuple: (l, u) - Lower and upper bounds for all state variables.

        Mathematical formulation:
        For each dimension i:
            l[i] = min(f_i^T * x) + c_i
            u[i] = max(f_i^T * x) + c_i
        subject to:
            C * x <= d
            pred_lb <= x <= pred_ub
        where f_i is the i-th row of V[:, 1:] and c_i is V[i, 0].
        """
        l = np.zeros(self.dim)
        u = np.zeros(self.dim)
        for i in range(self.dim):
            l[i] = self.getMin(i, lp_solver)
            u[i] = self.getMax(i, lp_solver)
        return l, u

    def affineMap(self, A=None, b=None):
        """
        Affine mapping of a star: S = A*self + b

        Args:
            A (np.ndarray): Mapping matrix (optional)
            b (np.ndarray): Offset vector (optional)

        Returns:
            Star: New Star set after affine mapping
        """
        if A is not None:
            if not isinstance(A, np.ndarray) or A.ndim != 2:
                raise ValueError("Mapping matrix should be a 2D numpy array")
            if A.shape[1] != self.dim:
                raise ValueError("Inconsistency between mapping matrix and Star dimension")

        if b is not None:
            if not isinstance(b, np.ndarray) or b.ndim != 1:
                raise ValueError("Offset vector should be a 1D numpy array")
            if A is not None and A.shape[0] != b.shape[0]:
                raise ValueError("Inconsistency between mapping matrix and offset vector")

        if A is None and b is None:
            return self.clone()

        V = self.V.copy()
        if A is not None:
            V = A @ V
        if b is not None:
            V[:, 0] += b
        return Star(V, self.C, self.d, self.pred_lb, self.pred_ub)

    def minKowskiSum(self, Y):
        """
        MinKowskiSum of two stars

        Args:
            Y (Star): Another Star set

        Returns:
            Star: Resulting Star set after Minkowski sum
        """
        if not isinstance(Y, Star):
            raise ValueError("Input is not a Star")
        if self.dim != Y.dim:
            raise ValueError("Inconsistent dimension between the input and the self object")

        V = np.hstack((self.V.copy(), Y.V[:, 1:]))
        V[:, 0] += Y.V[:, 0]
        
        pred_lb = np.concatenate((self.pred_lb, Y.pred_lb))
        pred_ub = np.concatenate((self.pred_ub, Y.pred_ub))

        if len(self.d) == 0 and len(Y.d) == 0:
            C = []
            d = []
        else:
            C = block_diag(self.C, Y.C)
            d = np.concatenate((self.d, Y.d))

        return Star(V, C, d, pred_lb, pred_ub)

    def isEmptySet(self):
        """
        Checks if Star set is an empty set.
        A Star set is empty if and only if the predicate P(a) is infeasible.

        Returns:
            bool: True if the set is empty, False otherwise

        Raises:
            Exception: If the LP solver encounters an unexpected status
        """
        if not (self.V.size and self.C.size and self.d.size):
            return True

        try:
            # We use a zero objective function as we only care about feasibility
            self._solve_lp(np.zeros(self.nVars), minimize=True, lp_solver='gurobi')
            return False  # If we can solve the LP, the set is not empty
        except Exception as e:
            if "exitflag = 3" in str(e):  # Gurobi status 3 means infeasible
                return True  # The set is empty
            else:
                # Re-raise the exception if it's not about infeasibility
                raise Exception(f'Unexpected error in LP solving: {str(e)}')

    def _validate_index(self, index):
        """Validate the index for row operations."""
        if not 0 <= index < self.dim:
            raise ValueError(f"Invalid index: {index}. Should be between 0 and {self.dim - 1}")

    def resetRow(self, index):
        """
        Reset a row with index.

        Args:
            index (int): The index of the row to reset.

        Returns:
            Star: A new Star object with the specified row reset.
        """
        self._validate_index(index)
        self.V[index, :] = 0.0
        return self

    def resetRows(self, map):
        """Reset a row with a map of indexes"""

        for index in map:
            self._validate_index(index)

        self.V[map, :] = 0.0
        return self

    def resetRowWithFactor(self, index, factor):
        """
        Reset a row with index and factor.

        Args:
            index (int): The index of the row to reset.
            factor (float): The factor to multiply the row by.

        Returns:
            Star: A new Star object with the specified row modified.
        """
        self._validate_index(index)
        self.V[index, :] *= factor
        return self

    def resetRowWithUpdatedCenter(self, index, new_c):
        """
        Reset a row with index, and with new center.

        Args:
            index (int): The index of the row to reset.
            new_c (float): The new center value for the row.

        Returns:
            Star: A new Star object with the specified row reset and new center.
        """
        self._validate_index(index)
        self.V[index, :] = 0.0
        self.V[index, 0] = new_c
        return self

    @staticmethod
    def rand(dim):
        """
        Randomly generate a Star.

        Args:
            dim (int): The dimension of the Star to generate.

        Returns:
            Star: A randomly generated Star object.

        Raises:
            ValueError: If the dimension is not positive.
        """
        if dim <= 0:
            raise ValueError("Dimension must be positive")
        lb = -np.random.rand(dim)
        ub = np.random.rand(dim)
        return Star(lb, ub)

    def toPolytope(self):
        """
        Converts the Star to a Polytope.

        Returns:
            pc.Polytope: The converted Polytope object.
        """
        C = self.C
        d = self.d

        if self.pred_lb.size and self.pred_ub.size:
            I = np.eye(self.nVars)
            C_bounds = np.vstack([I, -I])
            d_bounds = np.hstack([self.pred_ub, -self.pred_lb])
            
            C = np.vstack([C, C_bounds]) if C.size else C_bounds
            d = np.hstack([d, d_bounds]) if d.size else d_bounds

        c = self.V[:, 0]
        V = self.V[:, 1:]

        X, _, _, _ = np.linalg.lstsq(V.T, C.T, rcond=None)
        new_C = X.T
        new_d = d - new_C @ c

        return pc.Polytope(new_C, new_d)

    
    @staticmethod
    def updatePredicateRanges(newC, newd, pred_lb, pred_ub):
        """
        Update estimated ranges for predicate variables when one new constraint is added.

        Args:
            newC (np.ndarray): New constraint matrix (1D array)
            newd (np.ndarray): New constraint vector (1D array with one element)
            pred_lb (np.ndarray): Lower bound vector for predicates (1D array)
            pred_ub (np.ndarray): Upper bound vector for predicates (1D array)

        Returns:
            tuple: Updated lower and upper bound vectors for predicates
        """
        # Input validation
        if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [newC, newd, pred_lb, pred_ub]):
            raise ValueError("All inputs must be 1D numpy arrays")
        if newd.shape[0] != 1:
            raise ValueError("New constraint vector should have exactly one element")
        if not (newC.shape[0] == pred_lb.shape[0] == pred_ub.shape[0]):
            raise ValueError("Inconsistent dimensions between inputs")

        new_pred_lb = pred_lb.copy()
        new_pred_ub = pred_ub.copy()

        for i, x in enumerate(newC):
            if x != 0:
                v1 = newC / x
                d1 = newd[0] / x
                v2 = -np.delete(v1, i)
                v21 = np.maximum(v2, 0)
                v22 = np.minimum(v2, 0)
                lb = np.delete(pred_lb, i)
                ub = np.delete(pred_ub, i)

                if x > 0:
                    xmax = d1 + v21 @ ub + v22 @ lb
                    new_pred_ub[i] = min(xmax, pred_ub[i])
                else:  # x < 0
                    xmin = d1 + v21 @ lb + v22 @ ub
                    new_pred_lb[i] = max(xmin, pred_lb[i])

        return new_pred_lb, new_pred_ub

    def addConstraint(self, C, d):
        """
        Add a single constraint to a Star: self & Cx <= d

        Args:
            C (np.ndarray): Constraint matrix (1D array)
            d (np.ndarray): Constraint vector (1D array with one element)

        Returns:
            Star: The Star object with the new constraint added
        """
        if not (isinstance(C, np.ndarray) and C.ndim == 1):
            raise ValueError("Constraint matrix should be a 1D numpy array")
        if not (isinstance(d, np.ndarray) and d.ndim == 1 and d.shape[0] == 1):
            raise ValueError("Constraint vector should be a 1D numpy array with one element")
        if C.shape[0] != self.dim:
            raise ValueError("Inconsistency between constraint matrix and Star dimension")

        v = C @ self.V
        newC = v[1:self.nVars+1]
        newd = d - v[0]

        if self.C.size:
            self.C = np.vstack((newC, self.C))
            self.d = np.concatenate([newd, self.d])
        else:
            self.C = newC.reshape(1, self.nVars)
            self.d = newd

        self.pred_lb, self.pred_ub = self.updatePredicateRanges(newC, newd, self.pred_lb, self.pred_ub)

        return self

    def addMultipleConstraints(self, C, d):
        """
        Add multiple constraints to a Star: self & Cx <= d

        Args:
            C (np.ndarray): Constraint matrix (2D array)
            d (np.ndarray): Constraint vector (1D array)

        Returns:
            Star: The Star object with the new constraints added
        """
        if not isinstance(C, np.ndarray) or C.ndim != 2:
            raise ValueError("Constraint matrix should be a 2D numpy array")
        if not isinstance(d, np.ndarray) or d.ndim != 1:
            raise ValueError("Constraint vector should be a 1D numpy array")
        if C.shape[0] != d.shape[0]:
            raise ValueError("Inconsistency between constraint matrix and constraint vector")
        if C.shape[1] != self.dim:
            raise ValueError("Inconsistency between constraint matrix and Star dimension")

        for i in range(C.shape[0]):
            self.addConstraint(C[i, :], d[i:i+1])

        return self

    # TODO: Fixing Sample, containts function
    # def sample(self, N):
    #     """Sample number of points in the feasible Star set"""

    #     assert N >= 1, 'error: invalid number of samples'

    #     [lb, ub] = self.getRanges(lp_solver='gurobi')

    #     V1 = np.array([])
    #     for i in range(self.dim):
    #         X = (ub[i] - lb[i]) * np.random.rand(2*N, 1) + lb[i]
    #         V1 = np.hstack([V1, X]) if V1.size else X

    #     print("All points: ", V1)
    #     V = np.array([])

    #     for i in range(2*N):
    #         v1 = V1[i, :]
    #         print("each point: ", v1)
    #         if self.contains(v1):
    #             V = np.vstack([V, v1]) if V.size else V1
    #     print("All points after contain: ", V)

    #     V = V.T
    #     print(V.shape)
    #     if V.shape[1] >= N:
    #         V = V[:, 0:N]
    #     return V

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
        m.Params.OptimalityTol = 1e-6
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
        Ae = sp.csr_matrix(self.V[:, 1:])
        be = s - self.V[:, 0, None]
        m.addConstr(Ae @ x == be)
        m.optimize()

        if m.status == 2:
            return True
        elif m.status == 3:
            return False
        else:
            raise Exception('error: exitflat = %d' % (m.status))


    def sample(self, N):
        """
        Sample N points in the feasible Star set.

        Args:
            N (int): Number of samples to generate.

        Returns:
            np.ndarray: Matrix of sampled points (dim x N).
        """
        if N < 1:
            raise ValueError("Number of samples must be at least 1")

        lb, ub = self.getRanges(lp_solver='gurobi')
        
        # Generate 2N samples initially
        V1 = np.random.uniform(lb[:, np.newaxis], ub[:, np.newaxis], (self.dim, 2*N))
        
        # Filter valid samples
        V = V1[:, [self.contains(v) for v in V1.T]]
        
        # Return N samples (or all if less than N are valid)
        return V[:, :N]

    # def contains(self, s):
    #     """
    #     Check if the Star set contains a point.

    #     Args:
    #         s (np.ndarray): A point to check (1D array).

    #     Returns:
    #         bool: True if the point is contained, False otherwise.

    #     Raises:
    #         ValueError: If the input is invalid or dimensions mismatch.
    #     """
    #     if not isinstance(s, np.ndarray) or s.ndim != 1:
    #         raise ValueError("Input point should be a 1D numpy array")
    #     if s.shape[0] != self.dim:
    #         raise ValueError(f"Dimension mismatch: expected {self.dim}, got {s.shape[0]}")

    #     # Set up the LP problem
    #     f = np.zeros(self.nVars)
    #     Ae = self.V[:, 1:]
    #     be = s - self.V[:, 0]

    #     try:
    #         self._solve_lp(f, minimize=True, lp_solver='gurobi', Ae=Ae, be=be)
    #         return True
    #     except Exception as e:
    #         if "exitflag = 3" in str(e):  # Infeasible solution
    #             return False
    #         raise  # Re-raise other exceptions

    def intersect(self, S):
        """
        Compute the intersection of this Star set with another Star set.
            Intersection of two star sets
            x1 = c1 + V1 a1 in S1 (self) with P(a1) := C1 a1 <= d1
            x2 = c2 + V2 a2 in S2 (S)    with P(a2) := C2 a2 <= d2

            x = x1 \cap x2
              = c1 + V1 a1 + 0 a2        with P'(a) = P'([a1, a2])
              = c2 + V2 a2 + 0 a1        with P'(a) = P'([a1, a2]),
            where
            P'(a) = P1(a1) \wedge P2(a2) \wedge P_eq([a1, a2])
            P_eq([a1, a2]) := c1 + V1 a1 = c2 + V2 a2
                           := c1 - c2 + V1 a1 - V2 a2 = 0
            C_eq = [V1 - V2]
            d_eq = [c1 - c2]

        Args:
            S (Star): Another Star set to intersect with.

        Returns:
            Star or None: The intersection Star set, or None if the intersection is empty.
        """
        if self.dim != S.dim:
            raise ValueError(f"Dimension mismatch: self.dim = {self.dim}, S.dim = {S.dim}")

        # Compute P_eq
        d_eq = self.V[:, 0] - S.V[:, 0]
        C_eq = np.hstack([self.V[:, 1:], -S.V[:, 1:]])

        # New basis matrix
        new_V = np.hstack([self.V, np.zeros((self.dim, S.dim))])

        # New constraints
        C1 = block_diag(self.C, S.C)
        C2 = np.vstack([C_eq, -C_eq])
        new_C = np.vstack([C1, C2])

        d1 = np.hstack([self.d, S.d])
        d2 = np.hstack([-d_eq, d_eq])
        new_d = np.hstack([d1, d2])

        new_pred_lb = np.hstack([self.pred_lb, S.pred_lb])
        new_pred_ub = np.hstack([self.pred_ub, S.pred_ub])

        new_S = Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)
        return new_S if not new_S.isEmptySet() else []

    def intersectHalfSpace(self, H, g):
        """
        Intersect the Star set with a half-space H(x) := Hx <= g.

        Args:
            H (np.ndarray): Half-space matrix.
            g (np.ndarray): Half-space vector.

        Returns:
            Star: A new Star set with additional constraints.
        """
        if not isinstance(H, np.ndarray) or H.ndim != 2:
            raise ValueError("Half-space matrix H must be a 2D numpy array")
        if not isinstance(g, np.ndarray) or g.ndim != 1:
            raise ValueError("Half-space vector g must be a 1D numpy array")
        if H.shape[0] != g.shape[0]:
            raise ValueError("Inconsistent dimensions between H and g")
        if H.shape[1] != self.dim:
            raise ValueError("Inconsistent dimensions between H and Star set")

        C1 = H @ self.V[:, 1:]
        d1 = g - H @ self.V[:, 0]

        if len(self.d) > 0 and len(d1) > 0:
            new_C = np.vstack([self.C, C1])
            new_d = np.hstack([self.d, d1])
        elif len(self.d) > 0:
            new_C, new_d = self.C, self.d
        elif len(d1) > 0:
            new_C, new_d = C1, d1
        else:
            new_C, new_d = np.array([]), np.array([])

        return Star(self.V, new_C, new_d, self.pred_lb, self.pred_ub)