#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
##                                                                     ##
##   Primary contacts: Hoang Dung Tran <dungtran@ufl.edu> (UF)         ##
##                     Sung Woo Choi <sungwoo.choi@ufl.edu> (UF)       ##
##                     Yuntao Li <yli17@ufl.edu> (UF)                  ##
##                     Qing Liu <qliu1@ufl.edu> (UF)                   ##
##                                                                     ##
##   See CONTRIBUTORS for full author contacts and affiliations.       ##
##   This program is licensed under the BSD 3‑Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
LayerNorm Layer Class for Vision Transformers (ViTs)
Implements exact mean-centering projection followed by sound reciprocal
standard-deviation linear relaxation and Gaussian moment propagation.
Supports Star, ProbStar, ImageStar, SparseImageStar2DCSR/COO, and
ProbSparseImageStar2DCSR/COO.
"""

import copy
import numpy as np
import scipy.sparse as sp

from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
from StarV.set.probsparseimagestar2dcoo import ProbSparseImageStar2DCOO
from StarV.set.probsparse_base import propagate_general_elementwise_moments


class LayerNormLayer(object):
    """
    Layer Normalization over the last dimension (feature / channel dimension D).
    Given input of total dimension N = num_tokens * normalized_dim:
      For each token t in 0..num_tokens-1 and feature d in 0..D-1:
        z_{t,d} = x_{t,d} - mean_d(x_{t,:})
        y_{t,d} = gamma_d * z_{t,d} / sqrt(mean_d(z_{t,:}^2) + eps) + beta_d
    """

    def __init__(self, normalized_dim, gamma=None, beta=None, eps=1e-5, tol=1e-7):
        if isinstance(normalized_dim, (tuple, list)):
            self.normalized_dim = int(np.prod(normalized_dim))
        else:
            self.normalized_dim = int(normalized_dim)
        self.eps = float(eps)
        self.tol = float(tol)
        if gamma is None:
            self.gamma = np.ones(self.normalized_dim, dtype=np.float64)
        else:
            self.gamma = np.asarray(gamma, dtype=np.float64).ravel()
        if beta is None:
            self.beta = np.zeros(self.normalized_dim, dtype=np.float64)
        else:
            self.beta = np.asarray(beta, dtype=np.float64).ravel()
        assert len(self.gamma) == self.normalized_dim
        assert len(self.beta) == self.normalized_dim
        self.in_dim = self.normalized_dim
        self.out_dim = self.normalized_dim

    def __str__(self):
        print(f"Layer type: {self.__class__.__name__} (D={self.normalized_dim}, eps={self.eps})")
        return '\n'

    def info(self):
        print(self)

    def evaluate(self, x):
        x_arr = np.asarray(x, dtype=np.float64)
        orig_shape = x_arr.shape
        D = self.normalized_dim
        # Support (N, num_samples) batch evaluation as well as single sample (H, W, C) or (N,)
        if x_arr.ndim == 2 and x_arr.shape[0] % D == 0 and x_arr.shape[1] != D:
            # Columns are samples: shape (num_tokens * D, num_samples)
            B = x_arr.shape[1]
            T = x_arr.shape[0] // D
            xr = x_arr.reshape(T, D, B)
            mean = np.mean(xr, axis=1, keepdims=True)
            z = xr - mean
            var = np.mean(np.square(z), axis=1, keepdims=True)
            inv_std = 1.0 / np.sqrt(var + self.eps)
            y = self.gamma[None, :, None] * (z * inv_std) + self.beta[None, :, None]
            return y.reshape(orig_shape)
        elif x_arr.ndim == 4 and x_arr.shape[2] == D:
            # Image batch (H, W, C, B)
            mean = np.mean(x_arr, axis=2, keepdims=True)
            z = x_arr - mean
            var = np.mean(np.square(z), axis=2, keepdims=True)
            inv_std = 1.0 / np.sqrt(var + self.eps)
            return self.gamma[None, None, :, None] * (z * inv_std) + self.beta[None, None, :, None]
        else:
            flat = x_arr.reshape(-1, D)
            mean = np.mean(flat, axis=1, keepdims=True)
            z = flat - mean
            var = np.mean(np.square(z), axis=1, keepdims=True)
            inv_std = 1.0 / np.sqrt(var + self.eps)
            y = self.gamma[None, :] * (z * inv_std) + self.beta[None, :]
            return y.reshape(orig_shape)

    def _build_centering_block_diag(self, total_dim, dtype=np.float64):
        D = self.normalized_dim
        assert total_dim % D == 0, f"Total dim {total_dim} is not divisible by normalized_dim {D}"
        T = total_dim // D
        P_block = np.eye(D, dtype=dtype) - (1.0 / float(D)) * np.ones((D, D), dtype=dtype)
        if T == 1:
            return P_block
        return sp.kron(sp.eye(T, dtype=dtype, format='csr'), sp.csr_array(P_block), format='csr')

    def reachSingleInput(self, In, lp_solver='estimate', RF=0.0, show=False):
        is_prob = isinstance(In, (ProbStar, ProbSparseImageStar2DCSR, ProbSparseImageStar2DCOO))
        D = self.normalized_dim

        if isinstance(In, ImageStar):
            S = In.toStar(copy_=False)
            S_out = self.reachSingleInput(S, lp_solver=lp_solver, RF=RF, show=show)
            new_V = S_out.V.reshape(In.height, In.width, In.num_channel, S_out.nVars + 1)
            return ImageStar(new_V, S_out.C, S_out.d, S_out.pred_lb, S_out.pred_ub)

        if isinstance(In, (Star, ProbStar)):
            N = In.dim
            n = In.nVars
            dtype = In.V.dtype
            assert N % D == 0, f"Dimension {N} not divisible by normalized_dim {D}"
            T = N // D

            # Step 1: Exact linear mean-centering z = P x
            V_reshaped = In.V.reshape(T, D, n + 1)
            V_mean = np.mean(V_reshaped, axis=1, keepdims=True)
            V_z = (V_reshaped - V_mean).reshape(N, n + 1)
            c_z = V_z[:, 0]
            Vz_mat = V_z[:, 1:n + 1]

            # Step 2: Sound bounds [l_z, u_z] on centered coordinates z
            Vz_pos = np.maximum(Vz_mat, 0.0)
            Vz_neg = np.minimum(Vz_mat, 0.0)
            l_z = c_z + Vz_pos @ In.pred_lb + Vz_neg @ In.pred_ub
            u_z = c_z + Vz_pos @ In.pred_ub + Vz_neg @ In.pred_lb

        elif isinstance(In, (SparseImageStar2DCSR, SparseImageStar2DCOO)):
            c_orig, V_orig = In.get_c_V() if hasattr(In, 'get_c_V') else (In.c, In.V)
            N = In.dim
            n = In.num_pred
            dtype = V_orig.dtype
            assert N % D == 0, f"Dimension {N} not divisible by normalized_dim {D}"
            T = N // D

            P_op = self._build_centering_block_diag(N, dtype=dtype)
            c_z = (P_op @ c_orig) if sp.issparse(P_op) else (P_op @ c_orig)
            if sp.issparse(P_op):
                Vz_mat = (P_op @ V_orig.tocsr()).tocsr()
            else:
                Vz_mat = sp.csr_array(P_op @ V_orig.toarray())

            Vz_pos = Vz_mat.maximum(0.0)
            Vz_neg = Vz_mat.minimum(0.0)
            l_z = c_z + Vz_pos @ In.pred_lb + Vz_neg @ In.pred_ub
            u_z = c_z + Vz_pos @ In.pred_ub + Vz_neg @ In.pred_lb
        else:
            raise Exception(f"Unsupported input set type in LayerNormLayer: {type(In)}")

        # Step 3: Token-wise variance & reciprocal standard deviation bounds
        l_z_tok = l_z.reshape(T, D)
        u_z_tok = u_z.reshape(T, D)
        cross_zero = (l_z_tok <= 0.0) & (u_z_tok >= 0.0)
        q_lb = np.where(cross_zero, 0.0, np.minimum(np.square(l_z_tok), np.square(u_z_tok)))
        q_ub = np.maximum(np.square(l_z_tok), np.square(u_z_tok))

        var_min = np.mean(q_lb, axis=1) + self.eps  # (T,)
        var_max = np.mean(q_ub, axis=1) + self.eps  # (T,)
        r_min = (1.0 / np.sqrt(var_max))[:, None]   # (T, 1)
        r_max = (1.0 / np.sqrt(var_min))[:, None]   # (T, 1)

        r_mid = 0.5 * (r_min + r_max)               # (T, 1)
        r_rad = 0.5 * (r_max - r_min)               # (T, 1)

        gamma_tok = self.gamma[None, :]             # (1, D)
        beta_tok = self.beta[None, :]               # (1, D)

        # Effective linear slope wrt centered z_{t,d}
        a_eff = np.broadcast_to(gamma_tok * r_mid, (T, D)).reshape(N)
        b_center = np.broadcast_to(beta_tok, (T, D)).reshape(N)
        max_abs_z = np.maximum(np.abs(l_z_tok), np.abs(u_z_tok))
        rad_y = (np.abs(gamma_tok) * r_rad * max_abs_z).reshape(N)

        b_lb_all = b_center - rad_y
        b_ub_all = b_center + rad_y

        # Sound output range [y_min, y_max]
        cands = np.stack([
            (gamma_tok * r_min * l_z_tok + beta_tok).reshape(N),
            (gamma_tok * r_min * u_z_tok + beta_tok).reshape(N),
            (gamma_tok * r_max * l_z_tok + beta_tok).reshape(N),
            (gamma_tok * r_max * u_z_tok + beta_tok).reshape(N),
        ], axis=1)
        y_min_all = np.min(cands, axis=1)
        y_max_all = np.max(cands, axis=1)

        amb_idx = np.where(rad_y > self.tol)[0]
        det_idx = np.where(rad_y <= self.tol)[0]
        m = len(amb_idx)

        if isinstance(In, (Star, ProbStar)):
            V1 = np.zeros((N, n + 1), dtype=dtype)
            if len(det_idx) > 0:
                V1[det_idx, 0] = a_eff[det_idx] * c_z[det_idx] + b_center[det_idx]
                V1[det_idx, 1:n + 1] = a_eff[det_idx, None] * Vz_mat[det_idx, :]

            if m == 0:
                if isinstance(In, ProbStar):
                    return In.with_updated_state(V=V1)
                return Star(V1, In.C, In.d, In.pred_lb, In.pred_ub)

            V2 = np.zeros((N, m), dtype=dtype)
            V2[amb_idx, np.arange(m)] = 1.0
            new_V = np.hstack([V1, V2])

            V_amb = Vz_mat[amb_idx, :]
            c_amb = c_z[amb_idx]
            a_l = a_eff[amb_idx]
            a_u = a_eff[amb_idx]
            b_l = b_lb_all[amb_idx]
            b_u = b_ub_all[amb_idx]
            y_lb = y_min_all[amb_idx]
            y_ub = y_max_all[amb_idx]

            E = np.eye(m, dtype=dtype)
            C1 = np.hstack([a_l[:, None] * V_amb, -E])
            d1 = -a_l * c_amb - b_l
            C2 = np.hstack([-a_u[:, None] * V_amb, E])
            d2 = a_u * c_amb + b_u

            if len(In.C) == 0:
                new_C = np.vstack([C1, C2])
                new_d = np.hstack([d1, d2])
            else:
                C0 = np.hstack([In.C, np.zeros((In.C.shape[0], m), dtype=dtype)])
                new_C = np.vstack([C0, C1, C2])
                new_d = np.hstack([In.d, d1, d2])

            new_lb = np.hstack([In.pred_lb, y_lb])
            new_ub = np.hstack([In.pred_ub, y_ub])

            if isinstance(In, ProbStar):
                mu_new, Sig_new, env_rec = propagate_general_elementwise_moments(
                    V_pre=V_amb, c_pre=c_amb,
                    a_lb=a_l, b_lb=b_l, a_ub=a_u, b_ub=b_u,
                    pred_lb_new=y_lb, pred_ub_new=y_ub,
                    mu=In.mu, Sig=In.Sig,
                    cov_mode=In.cov_mode, dtype=dtype
                )
                return In.with_updated_state(
                    V=new_V, C=new_C, d=new_d, pred_lb=new_lb, pred_ub=new_ub,
                    mu=mu_new, Sig=Sig_new, new_env_record=env_rec
                )
            return Star(new_V, new_C, new_d, new_lb, new_ub)

        else:
            # SparseImageStar2DCSR or SparseImageStar2DCOO
            amb_idx32 = amb_idx.astype(np.int32)
            new_c = np.zeros(N, dtype=dtype)
            if len(det_idx) > 0:
                new_c[det_idx] = a_eff[det_idx] * c_z[det_idx] + b_center[det_idx]

            if m == N:
                row_new = np.arange(N, dtype=np.int32)
                col_new = np.arange(N, dtype=np.int32) + n
                data_new = np.ones(N, dtype=dtype)
                new_V = sp.csr_array((data_new, (row_new, col_new)), shape=(N, n + N))
                if not isinstance(In, SparseImageStar2DCSR):
                    new_V = new_V.tocoo()
            else:
                scale = np.zeros(N, dtype=dtype)
                if len(det_idx) > 0:
                    scale[det_idx] = a_eff[det_idx]
                V_rem = Vz_mat.multiply(scale[:, None]).tocoo()
                V_rem.data = np.hstack([V_rem.data, np.ones(m, dtype=dtype)])
                V_rem.row = np.hstack([V_rem.row.astype(np.int32), amb_idx32])
                V_rem.col = np.hstack([V_rem.col.astype(np.int32), np.arange(m, dtype=np.int32) + n])
                V_rem._shape = (N, n + m)
                new_V = V_rem.tocsr() if isinstance(In, SparseImageStar2DCSR) else V_rem

            if m == 0:
                if is_prob:
                    return In.with_updated_state(c=new_c, V=new_V, shape=In.shape)
                cls = SparseImageStar2DCSR if isinstance(In, SparseImageStar2DCSR) else SparseImageStar2DCOO
                return cls(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub, In.shape, copy_=False)

            V_amb = Vz_mat[amb_idx32, :]
            c_amb = c_z[amb_idx32]
            a_l = a_eff[amb_idx32]
            a_u = a_eff[amb_idx32]
            b_l = b_lb_all[amb_idx32]
            b_u = b_ub_all[amb_idx32]
            y_lb = y_min_all[amb_idx32]
            y_ub = y_max_all[amb_idx32]

            V_l = V_amb.multiply(a_l[:, None]).tocoo()
            V_u = V_amb.multiply(-a_u[:, None]).tocoo()
            eye_data = np.ones(m, dtype=dtype)
            eye_col = np.arange(m, dtype=np.int32) + n
            eye_row = np.arange(m, dtype=np.int32)

            d1 = -a_l * c_amb - b_l
            d2 = a_u * c_amb + b_u

            data = np.hstack([V_l.data, -eye_data, V_u.data, eye_data])
            row = np.hstack([V_l.row, eye_row, V_u.row + m, eye_row + m])
            col = np.hstack([V_l.col, eye_col, V_u.col, eye_col])
            C = sp.csr_array((data, (row, col)), shape=(2 * m, n + m), copy=False)

            if In.C.nnz > 0:
                data_c = np.hstack([In.C.data, C.data])
                indices_c = np.hstack([In.C.indices, C.indices])
                indptr_c = np.hstack([In.C.indptr, C.indptr[1:] + In.C.nnz])
                new_C = sp.csr_array(
                    (data_c, indices_c, indptr_c),
                    shape=(In.C.shape[0] + C.shape[0], C.shape[1]),
                    copy=False
                )
                new_d = np.hstack([In.d, d1, d2])
            else:
                new_C = C
                new_d = np.hstack([d1, d2])

            new_lb = np.hstack([In.pred_lb, y_lb])
            new_ub = np.hstack([In.pred_ub, y_ub])

            if is_prob:
                mu_new, Sig_new, env_rec = propagate_general_elementwise_moments(
                    V_pre=V_amb, c_pre=c_amb,
                    a_lb=a_l, b_lb=b_l, a_ub=a_u, b_ub=b_u,
                    pred_lb_new=y_lb, pred_ub_new=y_ub,
                    mu=In.mu, Sig=In.Sig,
                    cov_mode=In.cov_mode, dtype=dtype
                )
                return In.with_updated_state(
                    c=new_c, V=new_V, C=new_C, d=new_d,
                    pred_lb=new_lb, pred_ub=new_ub, shape=In.shape,
                    mu=mu_new, Sig=Sig_new, new_env_record=env_rec
                )
            cls = SparseImageStar2DCSR if isinstance(In, SparseImageStar2DCSR) else SparseImageStar2DCOO
            return cls(new_c, new_V, new_C, new_d, new_lb, new_ub, In.shape, copy_=False)

    def reach(self, inputSet, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False, **kwargs):
        if isinstance(inputSet, list):
            return [self.reachSingleInput(s, lp_solver=lp_solver, RF=RF, show=show) for s in inputSet]
        return self.reachSingleInput(inputSet, lp_solver=lp_solver, RF=RF, show=show)
