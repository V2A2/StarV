"""
Probabilistics Star
Bryan Duong 6/11/2024

"""

import numpy as np


class ProbStar:
    def __init__(
        self, V=None, C=None, d=None, mu=None, Sig=None, pred_lb=None, pred_ub=None
    ):
        """
        Initialize a ProbStar instance.
        """
        if all(arg is not None for arg in [V, C, d, mu, Sig, pred_lb, pred_ub]):
            # Full initialization
            self.V = V
            self.C = C
            self.d = d
            self.dim = V.shape[0]
            self.nVars = V.shape[1] - 1
            self.mu = mu
            self.Sig = Sig
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub

        elif all(arg is not None for arg in [mu, Sig, pred_lb, pred_ub]) and all(
            arg is None for arg in [V, C, d]
        ):
            # Partial initialization
            self.dim = pred_lb.shape[0]
            self.nVars = pred_lb.shape[0]
            self.mu = mu
            self.Sig = Sig
            self.V = np.hstack((np.zeros((self.dim, 1)), np.eye(self.dim)))
            self.C = np.array([])
            self.d = np.array([])
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub

        elif all(arg is None for arg in [V, C, d, mu, Sig, pred_lb, pred_ub]):
            # Empty initialization
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
            raise Exception("Invalid number of input arguments (should be 4 or 7)")
