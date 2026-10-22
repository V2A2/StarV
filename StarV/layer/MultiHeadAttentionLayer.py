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
Multi-Head Self-Attention (MHSA) Layer Class for Vision Transformers (ViTs)
Implements:
  1. Bilinear McCormick / midpoint-radius bounds for scaled dot-product Q K^T / sqrt(d_k)
  2. Exact componentwise monotone Softmax interval enclosure [l_A, u_A]
  3. Linear value-path propagation sum_j a0_{i,j} V_{j,:} + sound bilinear remainder
  4. Moment propagation and linear envelope DAG recording for ProbStar and
     ProbSparseImageStar2DCSR / ProbSparseImageStar2DCOO.
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


def _softmax_last_axis(s):
    s_max = np.max(s, axis=-1, keepdims=True)
    e = np.exp(np.clip(s - s_max, -60.0, 60.0))
    return e / np.sum(e, axis=-1, keepdims=True)


class MultiHeadAttentionLayer(object):
    """
    Multi-Head Self-Attention Layer for Vision Transformers.

    Input flattened layout: (num_tokens * embed_dim,) representing (T, D).
    Weights:
      W_Q, W_K, W_V, W_O: each (embed_dim, embed_dim) where y_proj = x @ W^T + b
      b_Q, b_K, b_V, b_O: each (embed_dim,)
    """

    def __init__(
        self,
        embed_dim,
        num_heads=1,
        W_Q=None, b_Q=None,
        W_K=None, b_K=None,
        W_V=None, b_V=None,
        W_O=None, b_O=None,
        num_tokens=None,
        tol=1e-7
    ):
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / np.sqrt(float(self.head_dim))
        self.num_tokens = None if num_tokens is None else int(num_tokens)
        self.tol = float(tol)

        D = self.embed_dim
        self.W_Q = np.eye(D, dtype=np.float64) if W_Q is None else np.asarray(W_Q, dtype=np.float64)
        self.W_K = np.eye(D, dtype=np.float64) if W_K is None else np.asarray(W_K, dtype=np.float64)
        self.W_V = np.eye(D, dtype=np.float64) if W_V is None else np.asarray(W_V, dtype=np.float64)
        self.W_O = np.eye(D, dtype=np.float64) if W_O is None else np.asarray(W_O, dtype=np.float64)

        self.b_Q = np.zeros(D, dtype=np.float64) if b_Q is None else np.asarray(b_Q, dtype=np.float64).ravel()
        self.b_K = np.zeros(D, dtype=np.float64) if b_K is None else np.asarray(b_K, dtype=np.float64).ravel()
        self.b_V = np.zeros(D, dtype=np.float64) if b_V is None else np.asarray(b_V, dtype=np.float64).ravel()
        self.b_O = np.zeros(D, dtype=np.float64) if b_O is None else np.asarray(b_O, dtype=np.float64).ravel()

        self.in_dim = (self.num_tokens * D) if self.num_tokens is not None else D
        self.out_dim = self.in_dim

    @staticmethod
    def rand(embed_dim, num_heads=2, num_tokens=4, weight_scale=0.25, seed=None):
        rng = np.random.default_rng(seed)
        D = int(embed_dim)
        W_Q = rng.normal(0.0, weight_scale / np.sqrt(D), size=(D, D))
        W_K = rng.normal(0.0, weight_scale / np.sqrt(D), size=(D, D))
        W_V = rng.normal(0.0, weight_scale / np.sqrt(D), size=(D, D))
        W_O = rng.normal(0.0, weight_scale / np.sqrt(D), size=(D, D))
        b_Q = rng.normal(0.0, 0.02, size=(D,))
        b_K = rng.normal(0.0, 0.02, size=(D,))
        b_V = rng.normal(0.0, 0.02, size=(D,))
        b_O = rng.normal(0.0, 0.02, size=(D,))
        return MultiHeadAttentionLayer(
            embed_dim=D, num_heads=num_heads,
            W_Q=W_Q, b_Q=b_Q, W_K=W_K, b_K=b_K,
            W_V=W_V, b_V=b_V, W_O=W_O, b_O=b_O,
            num_tokens=num_tokens
        )

    def __str__(self):
        print(
            f"Layer type: {self.__class__.__name__} "
            f"(embed_dim={self.embed_dim}, num_heads={self.num_heads}, num_tokens={self.num_tokens})"
        )
        return '\n'

    def info(self):
        print(self)

    def _eval_single(self, x_tok):
        """Evaluate MHSA on a single token matrix of shape (T, D)."""
        T, D = x_tok.shape
        H = self.num_heads
        dk = self.head_dim

        Q = (x_tok @ self.W_Q.T + self.b_Q[None, :]).reshape(T, H, dk).transpose(1, 0, 2)  # (H, T, dk)
        K = (x_tok @ self.W_K.T + self.b_K[None, :]).reshape(T, H, dk).transpose(1, 0, 2)  # (H, T, dk)
        V = (x_tok @ self.W_V.T + self.b_V[None, :]).reshape(T, H, dk).transpose(1, 0, 2)  # (H, T, dk)

        scores = (Q @ K.transpose(0, 2, 1)) * self.scale  # (H, T, T)
        attn = _softmax_last_axis(scores)                 # (H, T, T)
        context = (attn @ V).transpose(1, 0, 2).reshape(T, D)  # (T, D)
        out = context @ self.W_O.T + self.b_O[None, :]    # (T, D)
        return out

    def evaluate(self, x):
        x_arr = np.asarray(x, dtype=np.float64)
        orig_shape = x_arr.shape
        D = self.embed_dim
        if x_arr.ndim == 2 and x_arr.shape[0] % D == 0 and (self.num_tokens is not None and x_arr.shape[0] == self.num_tokens * D):
            # Batch of flattened vectors (T*D, B)
            B = x_arr.shape[1]
            T = x_arr.shape[0] // D
            out = np.empty_like(x_arr)
            for b in range(B):
                out[:, b] = self._eval_single(x_arr[:, b].reshape(T, D)).ravel()
            return out
        elif x_arr.ndim == 4:
            # Image batch (H_img, W_img, C, B) where H_img*W_img*C == T*D
            B = x_arr.shape[3]
            total = int(np.prod(x_arr.shape[:3]))
            T = total // D
            out = np.empty_like(x_arr)
            for b in range(B):
                out[:, :, :, b] = self._eval_single(x_arr[:, :, :, b].reshape(T, D)).reshape(x_arr.shape[:3])
            return out
        else:
            flat = x_arr.reshape(-1, D)
            return self._eval_single(flat).reshape(orig_shape)

    def _compute_qkv_bounds(self, c_x, V_x, pred_lb, pred_ub, T):
        """
        Compute interval bounds [l_Q, u_Q], [l_K, u_K], [l_V, u_V] of shape (H, T, dk).
        """
        D = self.embed_dim
        H = self.num_heads
        dk = self.head_dim

        # Token-wise linear map: for X of shape (T, D), Q = X @ W_Q^T + b_Q
        # In flattened (T*D,) coordinates, this is block-diagonal kron(I_T, W_Q).
        I_T = sp.eye(T, dtype=np.float64, format='csr')
        M_Q = sp.kron(I_T, sp.csr_array(self.W_Q), format='csr')
        M_K = sp.kron(I_T, sp.csr_array(self.W_K), format='csr')
        M_V = sp.kron(I_T, sp.csr_array(self.W_V), format='csr')

        c_Q = (M_Q @ c_x) + np.tile(self.b_Q, T)
        c_K = (M_K @ c_x) + np.tile(self.b_K, T)
        c_V = (M_V @ c_x) + np.tile(self.b_V, T)

        V_x_csr = sp.csr_array(V_x) if not sp.issparse(V_x) else V_x.tocsr()
        VQ = (M_Q @ V_x_csr).tocsr()
        VK = (M_K @ V_x_csr).tocsr()
        VV = (M_V @ V_x_csr).tocsr()

        def _bounds(c_vec, V_mat):
            vp = V_mat.maximum(0.0)
            vn = V_mat.minimum(0.0)
            lb = c_vec + vp @ pred_lb + vn @ pred_ub
            ub = c_vec + vp @ pred_ub + vn @ pred_lb
            return lb.reshape(T, H, dk).transpose(1, 0, 2), ub.reshape(T, H, dk).transpose(1, 0, 2)

        l_Q, u_Q = _bounds(c_Q, VQ)
        l_K, u_K = _bounds(c_K, VK)
        l_V, u_V = _bounds(c_V, VV)
        return (l_Q, u_Q), (l_K, u_K), (l_V, u_V), c_Q, VQ, c_K, VK, c_V, VV

    def reachSingleInput(self, In, lp_solver='estimate', RF=0.0, show=False):
        is_prob = isinstance(In, (ProbStar, ProbSparseImageStar2DCSR, ProbSparseImageStar2DCOO))
        D = self.embed_dim
        H = self.num_heads
        dk = self.head_dim

        if isinstance(In, ImageStar):
            S = In.toStar(copy_=False)
            S_out = self.reachSingleInput(S, lp_solver=lp_solver, RF=RF, show=show)
            new_V = S_out.V.reshape(In.height, In.width, In.num_channel, S_out.nVars + 1)
            return ImageStar(new_V, S_out.C, S_out.d, S_out.pred_lb, S_out.pred_ub)

        if isinstance(In, (Star, ProbStar)):
            N = In.dim
            n = In.nVars
            dtype = In.V.dtype
            c_x = In.V[:, 0].copy()
            V_x = In.V[:, 1:n + 1].copy()
        elif isinstance(In, (SparseImageStar2DCSR, SparseImageStar2DCOO)):
            c_x, V_x = In.get_c_V() if hasattr(In, 'get_c_V') else (In.c, In.V)
            c_x = c_x.copy()
            N = In.dim
            n = In.num_pred
            dtype = V_x.dtype
        else:
            raise Exception(f"Unsupported input set type in MultiHeadAttentionLayer: {type(In)}")

        assert N % D == 0, f"Input dimension {N} is not divisible by embed_dim {D}"
        T = N // D

        (l_Q, u_Q), (l_K, u_K), (l_V, u_V), c_Q, VQ, c_K, VK, c_V, VV = self._compute_qkv_bounds(
            c_x, V_x, In.pred_lb, In.pred_ub, T
        )

        # Step 1: Sound interval + affine-generator bounds [l_S, u_S] on S_{h,i,j} = Q_{h,i,:} . K_{h,j,:} / sqrt(dk)
        q_l = l_Q[:, :, None, :]  # (H, T, 1, dk)
        q_u = u_Q[:, :, None, :]  # (H, T, 1, dk)
        k_l = l_K[:, None, :, :]  # (H, 1, T, dk)
        k_u = u_K[:, None, :, :]  # (H, 1, T, dk)

        p1 = q_l * k_l
        p2 = q_l * k_u
        p3 = q_u * k_l
        p4 = q_u * k_u
        prod_lb = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        prod_ub = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))

        l_S_box = np.sum(prod_lb, axis=-1) * self.scale  # (H, T, T)
        u_S_box = np.sum(prod_ub, axis=-1) * self.scale  # (H, T, T)

        # Bilinear Taylor-McCormick decomposition around Q0 = 0.5*(l_Q+u_Q), K0 = 0.5*(l_K+u_K):
        # q(a).k(a) = q0.k(a) + k0.q(a) - q0.k0 + (q(a)-q0).(k(a)-k0)
        Q0 = 0.5 * (l_Q + u_Q)                 # (H, T, dk)
        dQ = 0.5 * np.maximum(u_Q - l_Q, 0.0)  # (H, T, dk)
        K0 = 0.5 * (l_K + u_K)                 # (H, T, dk)
        dK = 0.5 * np.maximum(u_K - l_K, 0.0)  # (H, T, dk)

        VQ_dense = VQ.toarray().reshape(T, H, dk, n).transpose(1, 0, 2, 3)  # (H, T, dk, n)
        VK_dense = VK.toarray().reshape(T, H, dk, n).transpose(1, 0, 2, 3)  # (H, T, dk, n)
        cQ_arr = c_Q.reshape(T, H, dk).transpose(1, 0, 2)                   # (H, T, dk)
        cK_arr = c_K.reshape(T, H, dk).transpose(1, 0, 2)                   # (H, T, dk)

        # c_S[h,i,j] and V_S[h,i,j,:] for the linearized score L_{h,i,j}(alpha)
        c_S = (
            np.einsum('hid,hjd->hij', Q0, cK_arr)
            + np.einsum('hjd,hid->hij', K0, cQ_arr)
            - np.einsum('hid,hjd->hij', Q0, K0)
        ) * self.scale  # (H, T, T)
        V_S = (
            np.einsum('hid,hjdn->hijn', Q0, VK_dense)
            + np.einsum('hjd,hidn->hijn', K0, VQ_dense)
        ) * self.scale  # (H, T, T, n)
        rem_S = np.einsum('hid,hjd->hij', dQ, dK) * self.scale  # (H, T, T)

        VS_pos = np.maximum(V_S, 0.0)
        VS_neg = np.minimum(V_S, 0.0)
        l_S_aff = c_S + np.einsum('hijn,n->hij', VS_pos, In.pred_lb) + np.einsum('hijn,n->hij', VS_neg, In.pred_ub) - rem_S
        u_S_aff = c_S + np.einsum('hijn,n->hij', VS_pos, In.pred_ub) + np.einsum('hijn,n->hij', VS_neg, In.pred_lb) + rem_S

        l_S = np.maximum(l_S_box, l_S_aff)
        u_S = np.minimum(u_S_box, u_S_aff)

        # Step 2: Exact monotone componentwise Softmax bounds [l_A, u_A] with difference-generator cancellation
        l_A = np.zeros((H, T, T), dtype=np.float64)
        u_A = np.zeros((H, T, T), dtype=np.float64)
        for h in range(H):
            for i in range(T):
                for j in range(T):
                    # Difference S[h,i,m] - S[h,i,j] via both interval and affine difference generators
                    d_c = c_S[h, i, :] - c_S[h, i, j]          # (T,)
                    d_V = V_S[h, i, :, :] - V_S[h, i, j, None, :]  # (T, n)
                    d_rem = rem_S[h, i, :] + rem_S[h, i, j]    # (T,)
                    d_rem[j] = 0.0
                    dV_pos = np.maximum(d_V, 0.0)
                    dV_neg = np.minimum(d_V, 0.0)
                    diff_u_aff = d_c + dV_pos @ In.pred_ub + dV_neg @ In.pred_lb + d_rem
                    diff_l_aff = d_c + dV_pos @ In.pred_lb + dV_neg @ In.pred_ub - d_rem

                    diff_u = np.minimum(u_S[h, i, :] - l_S[h, i, j], diff_u_aff)
                    diff_u[j] = 0.0
                    l_A[h, i, j] = 1.0 / np.sum(np.exp(np.clip(diff_u, -60.0, 60.0)))

                    diff_l = np.maximum(l_S[h, i, :] - u_S[h, i, j], diff_l_aff)
                    diff_l[j] = 0.0
                    u_A[h, i, j] = 1.0 / np.sum(np.exp(np.clip(diff_l, -60.0, 60.0)))

        # Simplex-consistent reference attention A0_{h,i,j} in [l_A, u_A] with sum_j A0_{h,i,j} = 1
        span_A = np.maximum(u_A - l_A, 0.0)
        sum_lA = np.sum(l_A, axis=-1, keepdims=True)
        sum_spanA = np.sum(span_A, axis=-1, keepdims=True)
        theta_A = np.where(sum_spanA > 1e-15, np.clip((1.0 - sum_lA) / np.where(sum_spanA > 1e-15, sum_spanA, 1.0), 0.0, 1.0), 0.5)
        A0 = l_A + theta_A * span_A
        dA = np.maximum(A0 - l_A, u_A - A0)  # (H, T, T)

        def _box_simplex_extrema(l_vec, u_vec, v_vec):
            """Exact min and max of sum_j a_j v_j s.t. l_j <= a_j <= u_j and sum_j a_j = 1."""
            rem = max(0.0, 1.0 - float(np.sum(l_vec)))
            cap = np.maximum(u_vec - l_vec, 0.0)
            base = float(np.dot(l_vec, v_vec))
            order_min = np.argsort(v_vec)
            rem_min = rem
            val_min = base
            for idx in order_min:
                take = min(rem_min, float(cap[idx]))
                val_min += take * float(v_vec[idx])
                rem_min -= take
                if rem_min <= 1e-15:
                    break
            order_max = order_min[::-1]
            rem_max = rem
            val_max = base
            for idx in order_max:
                take = min(rem_max, float(cap[idx]))
                val_max += take * float(v_vec[idx])
                rem_max -= take
                if rem_max <= 1e-15:
                    break
            return val_min, val_max

        # Step 3: Project V_h into output space U_h = V_h @ W_{O,h}^T of shape (T, D) per head
        # so box-simplex water-filling happens directly in output coordinates without |W_O| triangle loss
        W_lin = np.zeros((T * D, T * D), dtype=np.float64)
        rad_Y_mat = np.zeros((T, D), dtype=np.float64)
        shift_Y_mat = np.zeros((T, D), dtype=np.float64)
        VV_dense = VV.toarray().reshape(T, H, dk, n)
        cV_arr = c_V.reshape(T, H, dk)

        for h in range(H):
            W_O_h = self.W_O[:, h * dk:(h + 1) * dk]  # (D, dk)
            cU_h = cV_arr[:, h, :] @ W_O_h.T          # (T, D)
            VU_h = np.einsum('od,tdn->ton', W_O_h, VV_dense[:, h, :, :])  # (T, D, n)
            VUp = np.maximum(VU_h, 0.0)
            VUn = np.minimum(VU_h, 0.0)
            lU_h = cU_h + np.einsum('ton,n->to', VUp, In.pred_lb) + np.einsum('ton,n->to', VUn, In.pred_ub)
            uU_h = cU_h + np.einsum('ton,n->to', VUp, In.pred_ub) + np.einsum('ton,n->to', VUn, In.pred_lb)
            U0_h = 0.5 * (lU_h + uU_h)                # (T, D)
            dU_h = 0.5 * np.maximum(uU_h - lU_h, 0.0) # (T, D)

            quad_rem_h = dA[h] @ dU_h                 # (T, D) O(eps^2) remainder in output space
            for i in range(T):
                for d_out in range(D):
                    u_col = U0_h[:, d_out]
                    umin, umax = _box_simplex_extrema(l_A[h, i, :], u_A[h, i, :], u_col)
                    u_ref = float(np.dot(A0[h, i, :], u_col))
                    d_min = umin - u_ref
                    d_max = umax - u_ref
                    shift_Y_mat[i, d_out] += 0.5 * (d_min + d_max)
                    rad_Y_mat[i, d_out] += 0.5 * max(0.0, d_max - d_min) + quad_rem_h[i, d_out]

            for i in range(T):
                row_slice = slice(i * D, (i + 1) * D)
                for j in range(T):
                    col_slice = slice(j * D + h * dk, j * D + (h + 1) * dk)
                    W_lin[row_slice, col_slice] += A0[h, i, j] * W_O_h

        rad_Y = rad_Y_mat.reshape(N)
        shift_Y = shift_Y_mat.reshape(N)

        # Center affine output tilde_Y(alpha) = c_Y0 + V_Y0 * alpha
        b_O_tiled = np.tile(self.b_O, T)
        c_Y0 = W_lin @ c_V + b_O_tiled + shift_Y
        V_Y0_csr = (sp.csr_array(W_lin) @ VV).tocsr()

        # Bounds on tilde_Y(alpha)
        vp_y = V_Y0_csr.maximum(0.0)
        vn_y = V_Y0_csr.minimum(0.0)
        l_Y0 = c_Y0 + vp_y @ In.pred_lb + vn_y @ In.pred_ub
        u_Y0 = c_Y0 + vp_y @ In.pred_ub + vn_y @ In.pred_lb

        y_min_all = l_Y0 - rad_Y
        y_max_all = u_Y0 + rad_Y

        amb_idx = np.where(rad_Y > self.tol)[0]
        det_idx = np.where(rad_Y <= self.tol)[0]
        m = len(amb_idx)

        if isinstance(In, (Star, ProbStar)):
            V_Y0_dense = V_Y0_csr.toarray()
            V1 = np.zeros((N, n + 1), dtype=dtype)
            if len(det_idx) > 0:
                V1[det_idx, 0] = c_Y0[det_idx]
                V1[det_idx, 1:n + 1] = V_Y0_dense[det_idx, :]

            if m == 0:
                if isinstance(In, ProbStar):
                    return In.with_updated_state(V=V1)
                return Star(V1, In.C, In.d, In.pred_lb, In.pred_ub)

            V2 = np.zeros((N, m), dtype=dtype)
            V2[amb_idx, np.arange(m)] = 1.0
            new_V = np.hstack([V1, V2])

            V_amb = V_Y0_dense[amb_idx, :]
            c_amb = c_Y0[amb_idx]
            a_l = np.ones(m, dtype=dtype)
            a_u = np.ones(m, dtype=dtype)
            b_l = -rad_Y[amb_idx]
            b_u = rad_Y[amb_idx]
            y_lb = y_min_all[amb_idx]
            y_ub = y_max_all[amb_idx]

            E = np.eye(m, dtype=dtype)
            C1 = np.hstack([V_amb, -E])
            d1 = -c_amb - b_l
            C2 = np.hstack([-V_amb, E])
            d2 = c_amb + b_u

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
            amb_idx32 = amb_idx.astype(np.int32)
            new_c = np.zeros(N, dtype=dtype)
            if len(det_idx) > 0:
                new_c[det_idx] = c_Y0[det_idx]

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
                    scale[det_idx] = 1.0
                V_rem = V_Y0_csr.multiply(scale[:, None]).tocoo()
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

            V_amb = V_Y0_csr[amb_idx32, :]
            c_amb = c_Y0[amb_idx32]
            a_l = np.ones(m, dtype=dtype)
            a_u = np.ones(m, dtype=dtype)
            b_l = -rad_Y[amb_idx32]
            b_u = rad_Y[amb_idx32]
            y_lb = y_min_all[amb_idx32]
            y_ub = y_max_all[amb_idx32]

            V_l = V_amb.tocoo()
            V_u = V_amb.multiply(-1.0).tocoo()
            eye_data = np.ones(m, dtype=dtype)
            eye_col = np.arange(m, dtype=np.int32) + n
            eye_row = np.arange(m, dtype=np.int32)

            d1 = -c_amb - b_l
            d2 = c_amb + b_u

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
