#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
##                                                                     ##
##   See CONTRIBUTORS for full author contacts and affiliations.       ##
##   This program is licensed under the BSD 3-Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
PredicatePool: keeps many Star sets on ONE shared, growing predicate vector.

Why this exists
---------------
In a sequence model, many intermediate quantities (x_t, z_t, Delta_t, B_t, C_t, h_t ...)
are all functions of the SAME input uncertainty. If every binary operation
(sum, Hadamard product) treated its operands as independent Star sets, it would
concatenate their predicate vectors: the correlation between operands is lost
(looser bounds) and the same input predicates get duplicated over and over.

Instead, every Star handled by the pool is written over a single predicate vector

        a = [a_input, a_aux_1, a_aux_2, ...],   C a <= d,  lb <= a <= ub

which only ever grows by APPENDING auxiliary variables (from relaxations).
Because the pool is append-only, any Star created earlier can be "lifted" to the
current pool by padding its generator matrix with zero columns.

  * affine maps and sums are exact (no new variables)
  * unary relaxations (SiLU, Softplus, ExpDecay) append variables + constraints
  * Hadamard products use shared-predicate McCormick (appends variables)
"""

import numpy as np
from scipy.linalg import block_diag
from StarV.set.star import Star
from StarV.fun.identityXidentity import IdentityXIdentity


class PredicatePool(object):

    def __init__(self, C, d, pred_lb, pred_ub, lp_solver='gurobi'):
        self.pred_lb = np.asarray(pred_lb, dtype=np.float64).reshape(-1)
        self.pred_ub = np.asarray(pred_ub, dtype=np.float64).reshape(-1)
        self.nVars = self.pred_lb.shape[0]
        self.d = np.asarray(d, dtype=np.float64).reshape(-1)
        self.C = np.asarray(C, dtype=np.float64).reshape(-1, self.nVars) if len(self.d) > 0 \
            else np.empty((0, self.nVars))
        self.lp_solver = lp_solver

    # ------------------------------------------------------------------ build
    @classmethod
    def from_stars(cls, stars, lp_solver='gurobi'):
        """
        Build a pool from a list of mutually INDEPENDENT Star sets (e.g. one input set
        per time step). Returns (pool, lifted_stars), where lifted_stars[i] equals
        stars[i] but expressed over the joint predicate vector.
        """
        Cs, ds, lbs, ubs, offsets = [], [], [], [], []
        off = 0
        for S in stars:
            assert isinstance(S, Star), 'error: PredicatePool only supports Star sets'
            offsets.append(off)
            Cs.append(S.C if len(S.d) > 0 else np.empty((0, S.nVars)))
            ds.append(np.asarray(S.d, dtype=np.float64).reshape(-1))
            lbs.append(S.pred_lb)
            ubs.append(S.pred_ub)
            off += S.nVars

        n = off
        C = block_diag(*Cs) if sum(len(x) for x in ds) > 0 else np.empty((0, n))
        C = C.reshape(-1, n)
        pool = cls(C, np.concatenate(ds) if ds else np.empty(0),
                   np.concatenate(lbs) if lbs else np.empty(0),
                   np.concatenate(ubs) if ubs else np.empty(0), lp_solver=lp_solver)

        lifted = []
        for S, o in zip(stars, offsets):
            V = np.zeros((S.dim, n + 1))
            V[:, 0] = S.V[:, 0]
            V[:, 1 + o:1 + o + S.nVars] = S.V[:, 1:]
            lifted.append(pool._make(V))
        return pool, lifted

    # ------------------------------------------------------------ utilities
    def _make(self, V):
        return Star(V, self.C, self.d, self.pred_lb, self.pred_ub, copy_=False)

    def lift(self, S):
        """Express S (created on an earlier/equal version of this pool) on the current pool."""
        assert S.nVars <= self.nVars, \
            'error: Star has more predicates than the pool (was it created from another pool?)'
        V = np.zeros((S.dim, self.nVars + 1))
        V[:, :S.nVars + 1] = S.V
        return self._make(V)

    def absorb(self, S):
        """Adopt the (extended) predicate vector of S, which must extend the current pool."""
        assert S.nVars >= self.nVars, 'error: Star does not extend the pool'
        self.nVars = S.nVars
        self.pred_lb = np.asarray(S.pred_lb, dtype=np.float64)
        self.pred_ub = np.asarray(S.pred_ub, dtype=np.float64)
        self.d = np.asarray(S.d, dtype=np.float64).reshape(-1)
        self.C = np.asarray(S.C, dtype=np.float64).reshape(-1, self.nVars) if len(self.d) > 0 \
            else np.empty((0, self.nVars))
        return S

    # ------------------------------------------------------- exact operations
    def affine(self, S, W=None, b=None):
        """Exact affine map W S + b (no new predicates)."""
        S = self.lift(S)
        V = S.V if W is None else W @ S.V
        V = np.array(V, dtype=np.float64)
        if b is not None:
            V[:, 0] += np.asarray(b, dtype=np.float64).reshape(-1)
        return self._make(V)

    def add(self, *stars):
        """Exact sum of Stars that live on this pool (keeps correlations; NOT a Minkowski sum)."""
        V = None
        for S in stars:
            L = self.lift(S)
            V = L.V.copy() if V is None else V + L.V
        return self._make(V)

    def stack(self, stars):
        """Concatenate several Stars (same pool) into one higher-dimensional Star."""
        return self._make(np.vstack([self.lift(S).V for S in stars]))

    # ---------------------------------------------- relaxations (extend pool)
    def unary(self, reach_fn, S, *args, **kwargs):
        """Apply a unary relaxation (e.g. SiLU.reach) and extend the pool with its new variables."""
        kwargs.setdefault('lp_solver', self.lp_solver)
        out = reach_fn(self.lift(S), *args, **kwargs)
        return self.absorb(out)

    def mul(self, X, H):
        """Hadamard product X * H with shared-predicate McCormick (extends the pool)."""
        out = IdentityXIdentity.reachApprox_star_shared(self.lift(X), self.lift(H),
                                                        lp_solver=self.lp_solver)
        return self.absorb(out)

    def ranges(self, S):
        return self.lift(S).getRanges(lp_solver=self.lp_solver)
