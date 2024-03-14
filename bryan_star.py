"""
Star set implemented by Bryan
Algorithm described in paper "Star-based reachability analysis of Deep Neural Networks"
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
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
            opt = linprog(
                self.V[i, 1:],
                A_ub=self.C,
                b_ub=self.d,
                bounds=(self.pred_lb, self.pred_ub),
            )
            return opt.fun

        elif lp_solver == "gurobi":
            # Define the coefficients of the objective function
            c = self.V[i, 1 : self.nVars + 1]

            # Create a new model
            m = gp.Model()

            # Add variables to the model
            x = m.addMVar(shape=self.nVars)

            # Set the objective function
            m.setObjective(c @ x, GRB.MINIMIZE)

            # Add constraints to the model
            if len(self.C) > 0 and len(self.d) > 0:
                m.addConstr(self.C @ x <= self.d)

            # Solve the model
            m.optimize()

            # Check if the optimization was successful
            if m.status == GRB.OPTIMAL:
                return m.objVal
            else:
                raise Exception("LP solver error: optimization was not successful")

        elif lp_solver == "glpk":
            c = matrix(self.V[i, 1 : self.nVars + 1])
            G = matrix(self.C)
            h = matrix(self.d)
            sol = solvers.lp(c, G, h, solver="glpk")
            return sol["primal objective"]
