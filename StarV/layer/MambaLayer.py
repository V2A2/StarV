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
MambaLayer: Star-set reachability for a Mamba (S6 selective state space) block.

Forward pass (identical to mamba_ssm `Mamba` with the reference selective scan):

    xz      = W_in u_t (+ b_in)                    split -> x_t, z_t   (d_inner each)
    x'_t    = sum_k diag(w_k) x_{t-K+1+k} + b_conv  causal depthwise conv1d
    s_t     = SiLU(x'_t)
    [dt, B_t, C_t] = W_x s_t
    Delta_t = softplus(W_dt dt + b_dt)              (d_inner)
    A       = -exp(A_log)                           (d_inner x N), A < 0
    Abar_t  = exp(Delta_t[d] * A[d, n])             (d_inner x N)
    h_t     = Abar_t * h_{t-1} + (Delta_t[d] * s_t[d]) * B_t[n]
    y_t[d]  = sum_n C_t[n] h_t[d, n] + D[d] s_t[d]
    out_t   = W_out (y_t * SiLU(z_t)) (+ b_out)

Reachability: every quantity is a Star on one shared PredicatePool.
    affine / conv / sums             -> exact
    SiLU, softplus, exp(Delta*A)     -> convex relaxations (StarV.fun.silu / softplus / exp_decay)
    products                          -> shared-predicate McCormick (IdentityXIdentity)
The time loop carries the hidden-state Star H_t from step t to step t+1.
"""

import numpy as np
from StarV.set.star import Star
from StarV.set.predicatepool import PredicatePool
from StarV.fun.silu import SiLU
from StarV.fun.softplus import Softplus
from StarV.fun.exp_decay import ExpDecay


class MambaLayer(object):

    def __init__(self, W_in, conv_w, conv_b, W_x, W_dt, b_dt, A_log, D, W_out,
                 b_in=None, b_out=None, lp_solver='gurobi'):
        f = lambda a: None if a is None else np.asarray(a, dtype=np.float64)
        self.W_in, self.b_in = f(W_in), f(b_in)
        self.conv_w = f(conv_w).reshape(f(conv_w).shape[0], -1)     # (d_inner, d_conv)
        self.conv_b = f(conv_b) if conv_b is not None else np.zeros(self.conv_w.shape[0])
        self.W_x, self.W_dt, self.b_dt = f(W_x), f(W_dt), f(b_dt)
        self.A_log, self.D = f(A_log), f(D)
        self.W_out, self.b_out = f(W_out), f(b_out)

        self.d_model = self.W_in.shape[1]
        self.d_inner = self.W_in.shape[0] // 2
        self.d_conv = self.conv_w.shape[1]
        self.d_state = self.A_log.shape[1]
        self.dt_rank = self.W_dt.shape[1]
        assert self.W_x.shape == (self.dt_rank + 2 * self.d_state, self.d_inner), 'error: x_proj shape'
        assert self.W_out.shape == (self.d_model, self.d_inner), 'error: out_proj shape'

        self.in_dim = self.d_model
        self.out_dim = self.d_model
        self.lp_solver = lp_solver

        di, N, R = self.d_inner, self.d_state, self.dt_rank
        self.A = -np.exp(self.A_log)                                  # (d_inner, N), strictly < 0
        self.Rep = np.kron(np.eye(di), np.ones((N, 1)))              # (di*N, di): v[d] -> row d*N+n
        self.Tile = np.kron(np.ones((di, 1)), np.eye(N))             # (di*N, N):  v[n] -> row d*N+n
        self.SumN = self.Rep.T                                        # (di, di*N): sum over n
        self.W_dtx = self.W_dt @ self.W_x[:R]                         # s_t -> dt pre-activation
        self.W_B = self.W_x[R:R + N]
        self.W_C = self.W_x[R + N:]

    @classmethod
    def from_torch(cls, m, lp_solver='gurobi'):
        """Build from a torch module using mamba_ssm parameter names
        (in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj)."""
        t = lambda p: None if p is None else p.detach().cpu().double().numpy()
        return cls(W_in=t(m.in_proj.weight), b_in=t(m.in_proj.bias),
                   conv_w=t(m.conv1d.weight), conv_b=t(m.conv1d.bias),
                   W_x=t(m.x_proj.weight), W_dt=t(m.dt_proj.weight), b_dt=t(m.dt_proj.bias),
                   A_log=t(m.A_log), D=t(m.D),
                   W_out=t(m.out_proj.weight), b_out=t(m.out_proj.bias), lp_solver=lp_solver)

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, u, return_hidden=False):
        """Concrete forward pass. u: (L, d_model). Returns (L, d_model) [and hidden states (L, d_inner*N)]."""
        u = np.asarray(u, dtype=np.float64)
        L = u.shape[0]
        di, K = self.d_inner, self.d_conv
        xz = u @ self.W_in.T + (0 if self.b_in is None else self.b_in)
        x, z = xz[:, :di], xz[:, di:]

        h = np.zeros((di, self.d_state))
        outs, hs = [], []
        for t in range(L):
            xc = self.conv_b.copy()
            for k in range(K):
                tau = t - (K - 1) + k
                if tau >= 0:
                    xc += self.conv_w[:, k] * x[tau]
            s = SiLU.f(xc)
            delta = Softplus.f(self.W_dtx @ s + self.b_dt)
            Bt, Ct = self.W_B @ s, self.W_C @ s
            h = np.exp(delta[:, None] * self.A) * h + (delta * s)[:, None] * Bt[None, :]
            y = h @ Ct + self.D * s
            g = y * SiLU.f(z[t])
            outs.append(self.W_out @ g + (0 if self.b_out is None else self.b_out))
            hs.append(h.reshape(-1).copy())
        if return_hidden:
            return np.array(outs), np.array(hs)
        return np.array(outs)

    # --------------------------------------------------------------------- reach
    def reach(self, input_sets, H0=None, lp_solver=None, show=False):
        """
        Over-approximate reachable sets of the Mamba block.

        Args:
            input_sets: list of L Star sets (dim d_model), one per time step (independent).
            H0: optional Star (dim d_inner*d_state) for the initial hidden state; None = zero state.
        Returns:
            Y: list of L output Stars, H: list of L hidden-state Stars, pool: the PredicatePool.
            All returned Stars share the final pool, so they can be stacked to reason jointly
            over the whole trajectory (pool.stack(Y)).
        """
        lp_solver = self.lp_solver if lp_solver is None else lp_solver
        assert isinstance(input_sets, list) and len(input_sets) > 0, 'error: input_sets must be a non-empty list'
        L = len(input_sets)
        di, K = self.d_inner, self.d_conv

        stars = list(input_sets) + ([H0] if H0 is not None else [])
        pool, lifted = PredicatePool.from_stars(stars, lp_solver=lp_solver)
        U = lifted[:L]
        H_prev = lifted[L] if H0 is not None else None

        # exact: input projection for every time step
        X = [pool.affine(Ut, self.W_in[:di], None if self.b_in is None else self.b_in[:di]) for Ut in U]
        Z = [pool.affine(Ut, self.W_in[di:], None if self.b_in is None else self.b_in[di:]) for Ut in U]

        Y, H = [], []
        for t in range(L):
            if show:
                print(f'[MambaLayer] t = {t}, predicates = {pool.nVars}, constraints = {len(pool.d)}')

            # exact: causal depthwise conv1d (sum over the window, shared predicates)
            terms = [pool.affine(X[t - (K - 1) + k], np.diag(self.conv_w[:, k]))
                     for k in range(K) if t - (K - 1) + k >= 0]
            Xc = pool.affine(pool.add(*terms), b=self.conv_b)

            S = pool.unary(SiLU.reach, Xc)                                  # s_t
            Delta = pool.unary(Softplus.reach,
                               pool.affine(S, self.W_dtx, self.b_dt))      # Delta_t
            Abar = pool.unary(ExpDecay.reach,
                              pool.affine(Delta, self.Rep), self.A.reshape(-1))  # exp(Delta*A)

            P = pool.mul(Delta, S)                                          # Delta_t * s_t
            BU = pool.mul(pool.affine(P, self.Rep),
                          pool.affine(S, self.Tile @ self.W_B))             # (Delta s)[d] * B_t[n]

            if H_prev is None:          # zero initial state: Abar * 0 = 0 exactly
                Ht = pool.lift(BU)
            else:
                Ht = pool.add(pool.mul(Abar, H_prev), BU)

            CH = pool.mul(pool.affine(S, self.Tile @ self.W_C), Ht)         # C_t[n] * h_t[d,n]
            Yssm = pool.add(pool.affine(CH, self.SumN), pool.affine(S, np.diag(self.D)))

            G = pool.unary(SiLU.reach, Z[t])                                # SiLU(z_t)
            Out = pool.affine(pool.mul(Yssm, G), self.W_out, self.b_out)

            Y.append(Out)
            H.append(Ht)
            H_prev = Ht

        Y = [pool.lift(S) for S in Y]
        H = [pool.lift(S) for S in H]
        return Y, H, pool
