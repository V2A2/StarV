"""
Probabilistics Star
Bryan Duong 6/20/2024

"""

import numpy as np


class ProbStar(object):
    def __init__(
        self, V=None, C=None, d=None, mu=None, Sig=None, pred_lb=None, pred_ub=None
    ):
        """
        Initialize a ProbStar instance.
        """
        if V is not None:
            self.V = V
            self.C = C
            self.d = d
            self.mu = mu
            self.Sig = Sig
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.dim = V.shape[0]
            self.nVars = V.shape[1] - 1
        elif mu is not None:
            self.V = np.hstack(
                (np.zeros((pred_lb.shape[0], 1)), np.eye(pred_lb.shape[0]))
            )
            self.C = np.array([])
            self.d = np.array([])
            self.mu = mu
            self.Sig = Sig
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.dim = pred_lb.shape[0]
            self.nVars = pred_lb.shape[0]
        else:
            self.V = np.array([])
            self.C = np.array([])
            self.d = np.array([])
            self.mu = 0.0
            self.Sig = np.array([])
            self.pred_lb = np.array([])
            self.pred_ub = np.array([])
            self.dim = 0
            self.nVars = 0
