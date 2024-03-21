"""
Star set implemented by Bryan
Algorithm described in paper "Star-based reachability analysis of Deep Neural Networks"
"""

import glpk
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
from scipy.optimize import linprog
from cvxopt import matrix, solvers


class Star(object):
    """
    Star set implemented by Bryan

    Attributes:
    - V: matrix of centers and unit vectors
    - C: matrix of constraints
    - d: vector of constraints
    - pred_lb: lower bound of the predicate
    - pred_ub: upper bound of the predicate
    - nVars: number of variables

    Methods:
    - __init__(): constructor
    - __str__(): print the star set
    - affineMap(): affine map of the star set
    - getMin(): get minimum value of the star set in the i-th dimension

    """

    def __init__(self, *args) -> None:
        if len(args) == 2:
            [lb, ub] = args
            self.dim = lb.shape[0]
            V = np.eye(self.dim)
            zeros_column = np.zeros((self.dim, 1))
            V = np.hstack((zeros_column, V))
            pred_lb = np.reshape(lb, (-1, 1))
            pred_ub = np.reshape(ub, (-1, 1))

            self.V = V
            self.C = np.array([])
            self.d = np.array([])
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.nVars = V.shape[1] - 1

        elif len(args) == 5:
            [V, C, d, pred_lb, pred_ub] = args
            self.V = V
            self.C = C
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.nVars = V.shape[1] - 1

        elif len(args) == 0:
            self.V = np.array([])
            self.C = np.array([])
            self.d = np.array([])
            self.pred_lb = np.array([])
            self.pred_ub = np.array([])
            self.nVars = 0

        else:
            raise ValueError("Invalid number of arguments")

    def __str__(self):
        return (
            "Star object:\n"
            "V = {}\n"
            "C = {}\n"
            "d = {}\n"
            "nVars = {}\n"
            "pred_lb = {}\n"
            "pred_ub = {}".format(
                self.V, self.C, self.d, self.nVars, self.pred_lb, self.pred_ub
            )
        )

    def affineMap(self, W, b):
        """
        Affine map of the star set
        """
        V = W @ self.V
        V[:, 0] += b

        new_V = V
        new_C = self.C
        new_d = self.d
        new_pred_lb = self.pred_lb
        new_pred_ub = self.pred_ub

        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)

    def getMin(self, i, lp_solver="gurobi"):
        """
        Get minimum value of the star set in the i-th dimension
        """
        if lp_solver == "linprog":
            lb = self.pred_lb
            lb = lb[:, np.newaxis]
            ub = self.pred_ub
            ub = ub[:, np.newaxis]
            opt = linprog(
                self.V[i, 1 : self.nVars + 1],
                A_ub=self.C,
                b_ub=self.d,
                bounds=np.hstack((lb, ub)),
            )
            return opt.fun + self.V[i, 0]

        elif lp_solver == "gurobi":
            # Define the coefficients of the objective function
            c = self.V[i, 1 : self.nVars + 1]

            # Create a new model
            m = gp.Model()

            # Add variables to the model
            x = m.addMVar(shape=self.nVars, lb=self.pred_lb, ub=self.pred_ub)

            # Set the objective function
            m.setObjective(c @ x, GRB.MINIMIZE)

            # Add constraints to the model
            if len(self.C) > 0 and len(self.d) > 0:
                m.addConstr(self.C @ x <= self.d)

            # Solve the model
            m.optimize()

            # Check if the optimization was successful
            if m.status == GRB.OPTIMAL:
                return m.objVal + self.V[i, 0]
            else:
                raise Exception("LP solver error: optimization was not successful")

        elif lp_solver == "glpk":
            # Define the coefficients of the objective function
            c = self.V[i, 1 : self.nVars + 1]

            # Create a new LP problem
            lp = glpk.LPX()

            # objective direction is minimize
            lp.obj.maximize = False

            # constraints of the LP problem
            lp.matrix[:] = self.C
            for j in range(self.nVars):
                lp.rows.add(1)
                lp.rows[j].bounds = None, self.d[j]

            # bounds of the predicates
            for j in range(self.nVars):
                lp.cols.add(1)
                lp.cols[j].bounds = self.pred_lb[j], self.pred_ub[j]

            # objective coefficients
            lp.obj[:] = c.tolist()

            # Solve the LP problem
            lp.simplex()

            if lp.status == "opt":
                return lp.obj.value + self.V[i, 0]
            else:
                raise Exception("LP solver error: optimization was not successful")

    def getMax(self, i, lp_solver="gurobi"):
        """
        Get maximum value of the star set in the i-th dimension
        """
        if lp_solver == "linprog":
            lb = self.pred_lb.reshape(-1, 1)
            ub = self.pred_ub.reshape(-1, 1)
            f = -1 * self.V[i, 1 : self.nVars + 1]
            opt = linprog(
                -1 * self.V[i, 1 : self.nVars + 1],
                A_ub=self.C,
                b_ub=self.d,
                bounds=np.hstack((lb, ub)),
            )
            return -opt.fun + self.V[i, 0]

        elif lp_solver == "gurobi":
            # Define the coefficients of the objective function
            c = self.V[i, 1 : self.nVars + 1]

            # Create a new model
            m = gp.Model()

            # Add variables to the model
            x = m.addMVar(shape=self.nVars, lb=self.pred_lb, ub=self.pred_ub)

            # Set the objective function to maximize
            m.setObjective(c @ x, GRB.MAXIMIZE)
            C = sp.csr_matrix(self.C)
            # Add constraints to the model
            if len(self.C) > 0 and len(self.d) > 0:
                # m.addConstr(self.C @ x <= self.d)
                m.addConstr(sp.csr_matrix(self.C) @ x <= self.d)

            # Solve the model
            m.optimize()

            # Check if the optimization was successful
            if m.status == GRB.OPTIMAL:
                return m.objVal + self.V[i, 0]
            else:
                raise Exception("LP solver error: optimization was not successful")

        elif lp_solver == "glpk":
            # Define the coefficients of the objective function
            c = self.V[i, 1 : self.nVars + 1]

            # Create a new LP problem
            lp = glpk.LPX()

            # objective direction is maximize
            lp.obj.maximize = True

            # constraints of the LP problem
            lp.matrix[:] = self.C
            for j in range(self.nVars):
                lp.rows.add(1)
                lp.rows[j].bounds = None, self.d[j]

            # Add bounds to the predicates
            for j in range(self.nVars):
                lp.cols.add(1)
                lp.cols[j].bounds = self.pred_lb[j], self.pred_ub[j]

            # objective coefficients
            lp.obj[:] = c.tolist()

            # Solve the LP problem
            lp.simplex()

            if lp.status == "opt":
                return lp.obj.value + self.V[i, 0]
            else:
                raise Exception("LP solver error: optimization was not successful")

    def getRanges(self, lp_solver="gurobi") -> list:
        """get lower bound and upper bound of neurons by solving LP"""

        l = [self.getMin(i, lp_solver) for i in range(self.nVars)]
        u = [self.getMax(i, lp_solver) for i in range(self.nVars)]

        return l, u
