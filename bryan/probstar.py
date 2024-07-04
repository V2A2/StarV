"""
Probabilistics Star
Bryan Duong 7/4/2024

"""

import numpy as np
from StarV.util.minimax_tilting_sampler import TruncatedMVN


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

    def rand(*args):
        """Randomly generate a ProbStar"""

        if len(args) not in [1, 2, 4]:
            raise RuntimeError("Invalid number of arguments, should be 1, 2 or 4")

        dim = args[0]
        nVars = args[1] if len(args) >= 2 else dim

        if len(args) == 4:
            pred_lb, pred_ub = args[2], args[3]
            assert isinstance(pred_lb, np.ndarray) and isinstance(
                pred_lb, np.ndarray
            ), "predicate_lb and predicate_ub should be 1-d numpy arrays"
            assert (
                pred_lb.shape[0] == pred_ub.shape[1] == nVars
            ), "Inconsistency between predicate_lb, predicate_ub and number of predicate variables"

        else:
            pred_lb = -np.random.rand(
                nVars,
            )
            pred_ub = -np.random.rand(
                nVars,
            )

        V = np.random.rand(dim, nVars + 1)
        mu = 0.5 * (pred_lb + pred_ub)
        a = 3.0
        sig = (mu - pred_lb) / a
        Sig = np.diag(np.square(sig))

        S = ProbStar(V, [], [], mu, Sig, pred_lb, pred_ub)
        return S

    def affineMap(self, A=None, b=None):
        """Affine mapping of a probstar: S = A*self + b"""

        V = self.V if A is None else np.matmul(A, self.V)

        if b is not None:
            V = V.copy()
            V[:, 0] += b

        return ProbStar(
            V, self.C, self.d, self.mu, self.Sig, self.pred_lb, self.pred_ub
        )

    def sampling(self, n_samples):
        """Sample n_samples from the ProbStar"""

        tmvn = TruncatedMVN(self.mu, self.Sig, self.pred_lb, self.pred_ub)
        samples = tmvn.sample(n_samples)
        V = self.V[:, 1 : self.nVars + 1]
        center = self.V[:, 0]
        center = center.reshape(self.dim, 1)
        samples = np.matmul(V, samples) + center
        samples = np.unique(samples, axis=1)

        return samples

    def estimateRange(self, index):
        """Estimate the range of a state x[index]"""

        assert 0 <= index <= self.dim - 1, "Invalid index"

        v = self.V[index, 1 : self.nVars + 1]
        c = self.V[index, 0]

        v1 = np.where(v > 0, 0, v)
        v2 = np.where(v < 0, 0, v)

        min_val = c + v1 @ self.pred_ub + v2 @ self.pred_lb
        max_val = c + v1 @ self.pred_lb + v2 @ self.pred_ub

        return min_val, max_val

    def estimateRanges(self):
        """Quickly estimate bounds of x"""
        v = self.V[:, 1 : self.nVars + 1]
        c = self.V[:, 0]

        v1 = np.where(v > 0, 0, v)
        v2 = np.where(v < 0, 0, v)

        lb = c + v1 @ self.pred_ub + v2 @ self.pred_lb
        ub = c + v1 @ self.pred_lb + v2 @ self.pred_ub

        return lb, ub
