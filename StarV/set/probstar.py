"""
Updated ProbStar Class
Author: Yuntao Li
Date: 1/20/2024
"""
import numpy as np
from scipy.linalg import block_diag, svd
from typing import Optional, Union, Tuple
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog
# from scipy.stats import mvn # comment out because mvnun is not available in scipy 1.7.3
from scipy.stats import multivariate_normal # added because mvn is not available in scipy 1.7.3
import glpk
import polytope as pc
from StarV.util.minimax_tilting_sampler import TruncatedMVN

class ProbStar:
    """
    Probabilistic Star Class for quantitative reachability analysis.

    This class represents a Probabilistic Star set defined by:
    x = c + a[1]*v[1] + a[2]*v[2] + ... + a[n]*v[n] = V * b,
    where:
    - V = [c v[1] v[2] ... v[n]]
    - b = [1 a[1] a[2] ... a[n]]^T
    - C*a <= d (constraints on a[i])
    - a ~ N(mu, Sigma) (a multivariate normal distribution)

    Attributes:
        V (np.ndarray): Basis matrix [c v[1] v[2] ... v[n]]
        C (np.ndarray): Constraint matrix
        d (np.ndarray): Constraint vector
        dim (int): Dimension of the probabilistic star set
        mu (np.ndarray): Mean of the multivariate normal distribution
        Sig (np.ndarray): Covariance matrix (positive semidefinite)
        nVars (int): Number of predicate variables
        pred_lb (np.ndarray): Lower bound of predicate variables
        pred_ub (np.ndarray): Upper bound of predicate variables
    """

    # def __init__(self,
    #              V: Optional[np.ndarray] = None,
    #              C: Optional[np.ndarray] = None,
    #              d: Optional[np.ndarray] = None,
    #              mu: Optional[np.ndarray] = None,
    #              Sig: Optional[np.ndarray] = None,
    #              pred_lb: Optional[np.ndarray] = None,
    #              pred_ub: Optional[np.ndarray] = None):
    #     """
    #     Initialize a ProbStar object.

    #     Args:
    #         V (np.ndarray, optional): Basis matrix
    #         C (np.ndarray, optional): Constraint matrix
    #         d (np.ndarray, optional): Constraint vector
    #         mu (np.ndarray): Mean of the multivariate normal distribution
    #         Sig (np.ndarray): Covariance matrix
    #         pred_lb (np.ndarray): Lower bound of predicate variables
    #         pred_ub (np.ndarray): Upper bound of predicate variables

    #     Raises:
    #         ValueError: If the input arguments are invalid or inconsistent
    #     """
    #     if mu is not None and Sig is not None and pred_lb is not None and pred_ub is not None:
    #         self._init_from_bounds_and_dist(mu, Sig, pred_lb, pred_ub)
    #     elif all(arg is not None for arg in [V, C, d, mu, Sig, pred_lb, pred_ub]):
    #         self._init_from_matrices(V, C, d, mu, Sig, pred_lb, pred_ub)
    #     elif all(arg is None for arg in [V, C, d, mu, Sig, pred_lb, pred_ub]):
    #         self._init_empty()
    #     else:
    #         raise ValueError("Invalid number of input arguments (should be 0, 4, or 7)")

    def __init__(self, *args):
        """
        Initialize a ProbStar object.

        Args:
            V (np.ndarray, optional): Basis matrix
            C (np.ndarray, optional): Constraint matrix
            d (np.ndarray, optional): Constraint vector
            mu (np.ndarray): Mean of the multivariate normal distribution
            Sig (np.ndarray): Covariance matrix
            pred_lb (np.ndarray): Lower bound of predicate variables
            pred_ub (np.ndarray): Upper bound of predicate variables

        Raises:
            ValueError: If the input arguments are invalid or inconsistent
        """
        if len(args) == 7:
            self._init_from_matrices(*args)
        elif len(args) == 4:
            self._init_from_bounds_and_dist(*args)
        elif len(args) == 0:
            self._init_empty()
        else:
            raise ValueError("Invalid number of input arguments (should be 0, 4, or 7)")

    def _init_from_bounds_and_dist(self, mu: np.ndarray, Sig: np.ndarray,
                                      pred_lb: np.ndarray, pred_ub: np.ndarray):
        """Initialize ProbStar from distribution parameters."""
        self._validate_bounds_and_dist(mu, Sig, pred_lb, pred_ub)
        
        self.dim = len(mu)
        self.nVars = self.dim
        self.mu = mu
        self.Sig = Sig
        self.V = np.hstack((np.zeros((self.dim, 1)), np.eye(self.dim)))
        self.C = np.array([])
        self.d = np.array([])
        self.pred_lb = pred_lb
        self.pred_ub = pred_ub

    def _init_from_matrices(self, V: np.ndarray, C: np.ndarray, d: np.ndarray,
                                   mu: np.ndarray, Sig: np.ndarray,
                                   pred_lb: np.ndarray, pred_ub: np.ndarray):
        """Initialize ProbStar from full specification."""
        self._validate_matrices(V, C, d, mu, Sig, pred_lb, pred_ub)
        
        self.V = V
        self.C = C
        self.d = d
        self.dim = V.shape[0]
        self.nVars = V.shape[1] - 1
        self.mu = mu
        self.Sig = Sig
        self.pred_lb = pred_lb
        self.pred_ub = pred_ub

    def _init_empty(self):
        """Initialize an empty ProbStar."""
        self.dim = 0
        self.nVars = 0
        self.mu = np.array([])
        self.Sig = np.array([])
        self.V = np.array([])
        self.C = np.array([])
        self.d = np.array([])
        self.pred_lb = np.array([])
        self.pred_ub = np.array([])

    @staticmethod
    def _validate_bounds_and_dist(mu: np.ndarray, Sig: np.ndarray,
                                  pred_lb: np.ndarray, pred_ub: np.ndarray):
        """Validate inputs for distribution-based initialization."""
        assert isinstance(mu, np.ndarray) and mu.ndim == 1, "Mean vector should be a 1D numpy array"
        assert isinstance(Sig, np.ndarray) and Sig.ndim == 2, "Covariance matrix should be a 2D numpy array"
        assert isinstance(pred_lb, np.ndarray) and pred_lb.ndim == 1, "Lower bound vector should be a 1D numpy array"
        assert isinstance(pred_ub, np.ndarray) and pred_ub.ndim == 1, "Upper bound vector should be a 1D numpy array"
        
        assert len(mu) == len(pred_lb) == len(pred_ub) == Sig.shape[0] == Sig.shape[1], "Inconsistent array shapes"
        
        assert np.all(np.linalg.eigvals(Sig) > 0), "Covariance matrix must be positive definite"
        assert np.all(pred_ub >= pred_lb), "Upper bound must be greater than or equal to the lower bound for all dimensions"

    @staticmethod
    def _validate_matrices(V: np.ndarray, C: np.ndarray, d: np.ndarray,
                           mu: np.ndarray, Sig: np.ndarray,
                           pred_lb: np.ndarray, pred_ub: np.ndarray):
        """Validate inputs for full specification initialization."""
        assert isinstance(V, np.ndarray) and V.ndim == 2, "Basis matrix should be a 2D numpy array"
        assert isinstance(mu, np.ndarray) and mu.ndim == 1, "Mean vector should be a 1D numpy array"
        assert isinstance(Sig, np.ndarray) and Sig.ndim == 2, "Covariance matrix should be a 2D numpy array"
        assert isinstance(pred_lb, np.ndarray) and pred_lb.ndim == 1, "Lower bound vector should be a 1D numpy array"
        assert isinstance(pred_ub, np.ndarray) and pred_ub.ndim == 1, "Upper bound vector should be a 1D numpy array"
        
        if len(C) > 0:
            assert C.ndim == 2, "Constraint matrix should be a 2D numpy array"
            assert d.ndim == 1, "Constraint vector should be a 1D numpy array"
            assert V.shape[1] == C.shape[1] + 1, "Inconsistency between basis matrix and constraint matrix"
            assert C.shape[0] == d.shape[0], "Inconsistency between constraint matrix and constraint vector"
            assert C.shape[1] == pred_lb.shape[0] == pred_ub.shape[0], "Inconsistency between number of predicate variables and predicate bound vectors"
        
        assert V.shape[1] == len(mu) + 1 == len(pred_lb) + 1 == len(pred_ub) + 1, "Inconsistent array shapes"
        assert Sig.shape[0] == Sig.shape[1] == len(mu), "Inconsistent covariance matrix shape"
        assert np.all(np.linalg.eigvals(Sig) > 0), "Covariance matrix must be positive definite"
        assert np.all(pred_ub >= pred_lb), "Upper bound must be greater than or equal to the lower bound for all dimensions"

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

    def __repr__(self):
        print('ProbStar Set:')
        print('V: {}'.format(self.V.shape))
        print('Predicate Constraints:')
        print('C: {}'.format(self.C.shape))
        print('d: {}'.format(self.d.shape))
        print('dim: {}'.format(self.dim))
        print('nVars: {}'.format(self.nVars))
        print('pred_lb: {}'.format(self.pred_lb.shape))
        print('pred_ub: {}'.format(self.pred_ub.shape))
        print('mu: {}'.format(self.mu.shape))
        print('Sig: {}'.format(self.Sig.shape))
        print('')
        return '\n'
    
    def clone(self):
        """
        Create a deep copy of the ProbStar object by copying each attribute individually.

        Returns:
            ProbStar: A new ProbStar object that is a deep copy of the current object.
        """
        new_probstar = ProbStar.__new__(ProbStar)  # Create a new ProbStar instance without calling __init__
        
        # Copy each attribute
        new_probstar.V = self.V.copy()
        new_probstar.C = self.C.copy() if self.C.size > 0 else np.array([])
        new_probstar.d = self.d.copy() if self.d.size > 0 else np.array([])
        new_probstar.mu = self.mu.copy()
        new_probstar.Sig = self.Sig.copy()
        new_probstar.pred_lb = self.pred_lb.copy()
        new_probstar.pred_ub = self.pred_ub.copy()
        new_probstar.dim = self.dim
        new_probstar.nVars = self.nVars

        return new_probstar
    
    def printConstraints(self):
        'Print constraints of probstar'
        P = pc.Polytope(self.C, self.d)
        print(P)

    def calculate_mvn_probability(self, lb: np.ndarray, ub: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        """
        Calculate the probability of a multivariate normal distribution falling within specified bounds.

        This method uses the difference of cumulative distribution functions (CDFs) to compute the probability.

        Args:
            lb (np.ndarray): Lower bounds for each dimension.
            ub (np.ndarray): Upper bounds for each dimension.
            mean (np.ndarray): Mean vector of the multivariate normal distribution.
            cov (np.ndarray): Covariance matrix of the distribution.

        Returns:
            float: Probability of the distribution falling within the specified bounds.
        """
        mvn_dist = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
        return mvn_dist.cdf(ub) - mvn_dist.cdf(lb)

    def estimateProbability(self) -> float:
        """
        Estimate the probability of a ProbStar using the Genz method.

        This method handles two cases:
        1. When there are no additional constraints (C is empty).
        2. When there are additional constraints, which may require introducing auxiliary normal variables.

        Returns:
            float: Estimated probability of the ProbStar.
        """
        if len(self.C) == 0:
            return self.calculate_mvn_probability(self.pred_lb, self.pred_ub, self.mu, self.Sig)

        # Combine constraints
        C, d = self._combine_constraints()

        # Calculate A = C * Sig * C'
        A = np.linalg.multi_dot([C, self.Sig, C.T])

        if np.all(np.linalg.eigvals(A) > 0):
            return self._estimate_probability_without_auxiliary(C, d, A)
        else:
            return self._estimate_probability_with_auxiliary(C, d)

    def _combine_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine original constraints with predicate bounds.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Combined constraint matrix C and vector d.
        """
        C1 = np.vstack((np.eye(self.nVars), -np.eye(self.nVars)))
        d1 = np.concatenate([self.pred_ub, -self.pred_lb])
        C = np.vstack((self.C, C1))
        d = np.concatenate([self.d, d1])
        return C, d

    def _estimate_probability_without_auxiliary(self, C: np.ndarray, d: np.ndarray, A: np.ndarray) -> float:
        """
        Estimate probability without introducing auxiliary normal variables.

        This method is used when A = C * Sig * C' is positive definite.

        Args:
            C (np.ndarray): Combined constraint matrix.
            d (np.ndarray): Combined constraint vector.
            A (np.ndarray): A = C * Sig * C'.

        Returns:
            float: Estimated probability.
        """
        new_lb = np.full(len(d), -np.inf) # lb = l - A*mu
        new_ub = d - C @ self.mu # ub = u - A*mu 
        new_mu = np.zeros(len(d)) # new_mu = 0
        new_Sig = A
        return self.calculate_mvn_probability(new_lb, new_ub, new_mu, new_Sig)

    def _estimate_probability_with_auxiliary(self, C: np.ndarray, d: np.ndarray) -> float:
        """
        Estimate probability by introducing auxiliary normal variables.

        This method is used when A = C * Sig * C' is not positive definite.
        It performs SVD decomposition and introduces auxiliary variables.

        Args:
            C (np.ndarray): Combined constraint matrix.
            d (np.ndarray): Combined constraint vector.

        Returns:
            float: Estimated probability.
        """

        # step 1: SVD decomposition
        # [U, Q, L] = SVD(C), C = U*Q*L'
        # decompose Q = [Q_(r x r); 0_(m-r x r)]
        # U'*U = L'*L = I_r
        U, Q, Vt = svd(C)
        r = np.sum(Q > 1e-10)  # Numerical rank
        Q1 = np.diag(Q[:r])
        L1 = Vt[:r, :]
        Q1 = Q1 @ L1

        # Transform original normal variables
        mu1 = Q1 @ self.mu
        Sig1 = Q1 @ self.Sig @ Q1.T

        # Introduce auxiliary normal variables
        m = U.shape[0] - r
        mu2 = np.zeros(m)
        Sig2 = 1e-10 * np.eye(m)

        new_mu = np.concatenate([mu1, mu2])
        new_Sig = block_diag(Sig1, Sig2)
        new_lb = np.full(len(d), -np.inf)
        new_ub = d - U @ new_mu
        new_Sig = U @ new_Sig @ U.T

        return self.calculate_mvn_probability(new_lb, new_ub, np.zeros(len(d)), new_Sig)

    def get_minimized_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the minimized constraints of the ProbStar.

        This method combines the existing constraints with the predicate bounds
        and uses the polytope library to reduce the resulting polytope. The
        reduction process eliminates redundant constraints, potentially
        simplifying the representation of the ProbStar.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - Cmin: The minimized constraint matrix
                - dmin: The minimized constraint vector
        """
        if len(self.C) == 0:
            return self.C, self.d

        C, d = self._combine_constraints()
        P = pc.Polytope(C, d)
        P_reduced = pc.reduce(P)
        
        return P_reduced.A, P_reduced.b

    def minimize_constraints(self) -> 'ProbStar':
        """
        Minimize the constraints of the ProbStar in-place.

        This method updates the ProbStar's constraints (C and d) with their
        minimized versions. If there are no existing constraints, the method
        returns the ProbStar unchanged.

        Returns:
            ProbStar: The ProbStar instance with minimized constraints
        """
        if len(self.C) == 0:
            return self

        self.C, self.d = self.get_minimized_constraints()
        return self

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
        return ProbStar(V, self.C, self.d, self.mu, self.Sig, self.pred_lb, self.pred_ub)

    def minKowskiSum(self, Y):
        """
        MinKowskiSum of two stars

        Args:
            Y (Star): Another Star set

        Returns:
            Star: Resulting Star set after Minkowski sum
        """
        if not isinstance(Y, ProbStar):
            raise ValueError("Input is not a Prob Star")
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

        return ProbStar(V, C, d, self.mu, self.Sig, pred_lb, pred_ub)

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

    def addConstraintWithoutUpdateBounds(self, C, d):

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

        return self

    def addMultipleConstraintsWithoutUpdateBounds(self, C, d):
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
            self.addConstraintWithoutUpdateBounds(C[i, :], d[i:i+1])

        return self
    
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
            ProbStar: A new ProbStar object with the specified row reset.
        """
        self._validate_index(index)
        self.V[index, :] = 0.0
        return self

    def resetRows(self, map):
        """Reset a row with a map of indexes"""

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
            ProbStar: A new ProbStar object with the specified row modified.
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
            ProbStar: A new ProbStar object with the specified row reset and new center.
        """
        self._validate_index(index)
        self.V[index, :] = 0.0
        self.V[index, 0] = new_c
        return self

    def concatenate_with_vector(self, v: Optional[np.ndarray] = None):
        """
        Concatenate the ProbStar with a vector.

        This method creates a new ProbStar by vertically stacking the input vector
        with the current ProbStar's basis matrix (V).

        Args:
            v (np.ndarray, optional): A 1-D numpy array to concatenate with the ProbStar.

        Returns:
            ProbStar: A new ProbStar instance with the concatenated basis matrix.

        Raises:
            ValueError: If the input is not a 1-D numpy array.
        """
        if v is None or len(v) == 0:
            return self

        if not isinstance(v, np.ndarray) or v.ndim != 1:
            raise ValueError("Input should be a 1-D numpy array")

        n = len(v)
        v1 = v.reshape(n, 1)
        V1 = np.hstack((v1, np.zeros((n, self.nVars))))
        new_V = np.vstack((V1, self.V))

        return ProbStar(new_V, self.C, self.d, self.mu, self.Sig, self.pred_lb, self.pred_ub)

    @staticmethod
    def rand(*args):
        """
        Randomly generate a ProbStar.

        Args:
            dim (int): Dimension of the ProbStar.
            nVars (int, optional): Number of variables. If not provided, it's set to dim.
            pred_lb (np.ndarray, optional): Lower bounds for predicate variables.
            pred_ub (np.ndarray, optional): Upper bounds for predicate variables.

        Returns:
            ProbStar: A randomly generated ProbStar instance.

        Raises:
            ValueError: If the inputs are inconsistent or invalid.
        """
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
            raise RuntimeError('invalid number of arguments, should be 1 or 2 or 4')
            
        V = np.random.rand(dim, nVars + 1)
        if len(args) != 4:    
            pred_lb = -np.random.rand(nVars,)
            pred_ub = np.random.rand(nVars,)
            
        mu = 0.5 * (pred_lb + pred_ub)
        a = 3.0
        sig = (mu - pred_lb) / a
        Sig = np.diag(np.square(sig))

        return ProbStar(V, np.array([]), np.array([]), mu, Sig, pred_lb, pred_ub)

    def sampling(self, N: int) -> np.ndarray:
        """
        Perform Monte Carlo sampling of the ProbStar.

        This method samples from a multivariate truncated normal distribution
        defined by the ProbStar's parameters, then transforms the samples using
        the ProbStar's basis matrix.

        Args:
            N (int): Number of samples to generate.

        Returns:
            np.ndarray: Array of unique samples, shape (self.dim, n_unique_samples).
        """
        tmvn = TruncatedMVN(self.mu, self.Sig, self.pred_lb, self.pred_ub)
        samples = tmvn.sample(N)
        
        V = self.V[:, 1:self.nVars+1]
        center = self.V[:, 0].reshape(self.dim, 1)
        
        samples = V @ samples + center
        return np.unique(samples, axis=1)

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

        return ProbStar(self.V, new_C, new_d, self.mu, self.Sig, self.pred_lb, self.pred_ub)