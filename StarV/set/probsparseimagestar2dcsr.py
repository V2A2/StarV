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
##   This program is licensed under the BSD 3-Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
Probabilistic Sparse Image Star 2D CSR Class (ProbSparseImageStar2DCSR)
Combines memory-efficient SparseImageStar2DCSR representation with
extended-predicate Gaussian moment propagation and dual-projected
soundness guarantees for large vision models.
"""

import copy
import torch
import numpy as np
import scipy.sparse as sp
from scipy.special import ndtri, ndtr

from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.set.probsparse_base import (
    compute_box_input_probability,
    compute_truncated_gaussian_moments_1d,
    compute_preactivation_moments,
    propagate_relu_moments,
    propagate_leakyrelu_moments,
    propagate_general_elementwise_moments,
    propagate_maxpool_moments,
    backsubstitute_linear_bounds,
    verify_prob_spec,
    mvn_cdf_lowdim,
    _MIN_VAR,
    _MIN_STD,
)


def _init_prob_params_from_bounds(
    c, V, pred_lb, pred_ub, shape,
    mu=None, Sig=None, sig_factor=3.0, cov_mode='diagonal'
):
    """Initialize predicate-space (mu, Sig) from normalized predicate bounds or pixel-space inputs."""
    nv = len(pred_lb)
    dtype = pred_lb.dtype if hasattr(pred_lb, 'dtype') else np.float64
    if nv == 0:
        empty_1d = np.empty(0, dtype=dtype)
        empty_sig = empty_1d if cov_mode == 'diagonal' else np.empty((0, 0), dtype=dtype)
        return empty_1d, empty_sig

    if mu is None:
        mu_pred = 0.5 * (pred_lb + pred_ub)
    else:
        mu_arr = np.asarray(mu, dtype=dtype).reshape(-1)
        if len(mu_arr) == nv:
            mu_pred = mu_arr.copy()
        elif c is not None and len(mu_arr) == len(c):
            # Map from pixel space to predicate space using active generator columns
            V_coo = V.tocoo() if sp.issparse(V) else sp.coo_array(V[:, 1:])
            mu_pred = 0.5 * (pred_lb + pred_ub)
            if V_coo.nnz == nv:
                order = np.argsort(V_coo.col)
                rows = V_coo.row[order]
                scales = np.where(np.abs(V_coo.data[order]) > 1e-15, V_coo.data[order], 1.0)
                mu_pred = (mu_arr[rows] - c[rows]) / scales
        else:
            raise ValueError(f"Incompatible mu shape {mu_arr.shape} for num_pred={nv}")

    if Sig is None:
        half_span = np.maximum(0.5 * (pred_ub - pred_lb), _MIN_STD)
        std_vec = half_span / float(sig_factor)
        var_vec = np.maximum(np.square(std_vec), _MIN_VAR).astype(dtype)
        Sig_pred = var_vec if cov_mode == 'diagonal' else np.diag(var_vec)
    else:
        Sig_arr = np.asarray(Sig, dtype=dtype)
        if Sig_arr.ndim == 0:
            var_vec = np.full(nv, max(float(Sig_arr), _MIN_VAR), dtype=dtype)
            Sig_pred = var_vec if cov_mode == 'diagonal' else np.diag(var_vec)
        elif Sig_arr.ndim == 1:
            if len(Sig_arr) == nv:
                var_vec = np.maximum(Sig_arr, _MIN_VAR)
            elif c is not None and len(Sig_arr) == len(c):
                V_coo = V.tocoo() if sp.issparse(V) else sp.coo_array(V[:, 1:])
                order = np.argsort(V_coo.col)
                rows = V_coo.row[order]
                scales = np.where(np.abs(V_coo.data[order]) > 1e-15, V_coo.data[order], 1.0)
                var_vec = np.maximum(Sig_arr[rows] / np.square(scales), _MIN_VAR)
            else:
                raise ValueError(f"Incompatible 1D Sig shape {Sig_arr.shape} for num_pred={nv}")
            Sig_pred = var_vec if cov_mode == 'diagonal' else np.diag(var_vec)
        elif Sig_arr.ndim == 2:
            assert Sig_arr.shape == (nv, nv), f"Expected Sig shape {(nv, nv)}, got {Sig_arr.shape}"
            Sig_pred = np.diag(Sig_arr).copy() if cov_mode == 'diagonal' else Sig_arr.copy()
        else:
            raise ValueError(f"Invalid Sig dimension: {Sig_arr.ndim}")

    return mu_pred, Sig_pred


class ProbSparseImageStar2DCSR(SparseImageStar2DCSR):
    """
    Probabilistic Sparse Image Star in 2D CSR format.
    Subclasses SparseImageStar2DCSR so all existing StarV vision layers
    automatically recognize it as a CSR sparse image star while preserving
    and propagating probabilistic predicate distributions and sound relaxation envelopes.
    """

    def __init__(
        self,
        *args,
        mu=None,
        Sig=None,
        sig_factor=3.0,
        cov_mode='diagonal',
        mu0=None,
        Sig0=None,
        pred_lb0=None,
        pred_ub0=None,
        nVars0=None,
        env_dag=None,
        p_input=None,
        copy_=True
    ):
        len_ = len(args)
        self.cov_mode = cov_mode

        if len_ == 9:
            # (c, V, C, d, mu, Sig, pred_lb, pred_ub, shape)
            c, V, C, d, mu_arg, Sig_arg, pred_lb, pred_ub, shape = args
            super().__init__(c, V, C, d, pred_lb, pred_ub, shape, copy_=copy_)
            mu = mu_arg
            Sig = Sig_arg
        elif len_ == 8:
            # (V, C, d, mu, Sig, pred_lb, pred_ub, shape) for dense V
            V, C, d, mu_arg, Sig_arg, pred_lb, pred_ub, shape = args
            super().__init__(V, C, d, pred_lb, pred_ub, shape, copy_=copy_)
            mu = mu_arg
            Sig = Sig_arg
        elif len_ == 4:
            # (lb, ub, mu, Sig)
            lb, ub, mu_arg, Sig_arg = args
            super().__init__(lb, ub, copy_=copy_)
            mu = mu_arg
            Sig = Sig_arg
        elif len_ in (0, 2, 6, 7):
            super().__init__(*args, copy_=copy_)
        else:
            raise ValueError(
                f"Invalid number of positional arguments ({len_}) for ProbSparseImageStar2DCSR; "
                "expected 0, 2, 4, 6, 7, 8, or 9."
            )

        if len_ == 0:
            self.mu = np.empty(0)
            self.Sig = np.empty(0)
            self.mu0 = np.empty(0)
            self.Sig0 = np.empty(0)
            self.pred_lb0 = np.empty(0)
            self.pred_ub0 = np.empty(0)
            self.nVars0 = 0
            self.env_dag = []
            self._p_input = 0.0
            return

        if mu is None or Sig is None:
            c_ref, V_ref = self.get_c_V()
            mu_init, Sig_init = _init_prob_params_from_bounds(
                c_ref, V_ref, self.pred_lb, self.pred_ub, self.shape,
                mu=mu, Sig=Sig, sig_factor=sig_factor, cov_mode=cov_mode
            )
            self.mu = mu_init
            self.Sig = Sig_init
        else:
            self.mu = np.array(mu, copy=copy_).reshape(-1)
            Sig_arr = np.array(Sig, copy=copy_)
            if Sig_arr.ndim == 2 and cov_mode == 'diagonal':
                self.Sig = np.diag(Sig_arr).copy()
            elif Sig_arr.ndim == 1 and cov_mode == 'full':
                self.Sig = np.diag(Sig_arr)
            else:
                self.Sig = Sig_arr
            if self.Sig.ndim == 1:
                self.cov_mode = 'diagonal'
            else:
                self.cov_mode = 'full'

        if nVars0 is None:
            self.nVars0 = int(self.num_pred)
            self.mu0 = self.mu.copy()
            self.Sig0 = self.Sig.copy()
            self.pred_lb0 = self.pred_lb.copy()
            self.pred_ub0 = self.pred_ub.copy()
        else:
            self.nVars0 = int(nVars0)
            self.mu0 = self.mu[:self.nVars0].copy() if mu0 is None else np.asarray(mu0)
            if Sig0 is None:
                self.Sig0 = (
                    self.Sig[:self.nVars0].copy()
                    if self.Sig.ndim == 1
                    else self.Sig[:self.nVars0, :self.nVars0].copy()
                )
            else:
                self.Sig0 = np.asarray(Sig0)
            self.pred_lb0 = self.pred_lb[:self.nVars0].copy() if pred_lb0 is None else np.asarray(pred_lb0)
            self.pred_ub0 = self.pred_ub[:self.nVars0].copy() if pred_ub0 is None else np.asarray(pred_ub0)

        self.env_dag = list(env_dag) if env_dag is not None else []
        self._p_input = p_input

    @property
    def dim(self):
        return self.V.shape[0]

    @property
    def nVars(self):
        return self.num_pred

    def get_c_V(self):
        """Return (c_1d, V_2d) regardless of whether self.V is sparse CSR or dense [c, X]."""
        if isinstance(self.V, np.ndarray):
            if self.c is None:
                return self.V[:, 0], self.V[:, 1:]
            return self.c, self.V
        return self.c, self.V

    def has_constraints(self):
        if sp.issparse(self.C):
            return self.C.nnz > 0 and len(self.d) > 0
        return len(self.C) > 0 and len(self.d) > 0

    def get_input_probability(self):
        if self._p_input is None:
            self._p_input = compute_box_input_probability(
                self.mu0, self.Sig0, self.pred_lb0, self.pred_ub0
            )
        return self._p_input

    def with_updated_state(
        self, c=None, V=None, C=None, d=None, pred_lb=None, pred_ub=None, shape=None,
        mu=None, Sig=None, new_env_record=None, env_dag=None, copy_=False
    ):
        """Create a child ProbSparseImageStar2DCSR preserving input distribution and DAG."""
        c_use = self.c if (c is None and not isinstance(V, np.ndarray)) else c
        V_use = self.V if V is None else V
        C_use = self.C if C is None else C
        d_use = self.d if d is None else d
        lb_use = self.pred_lb if pred_lb is None else pred_lb
        ub_use = self.pred_ub if pred_ub is None else pred_ub
        shape_use = self.shape if shape is None else shape
        next_mu = self.mu if mu is None else mu
        next_Sig = self.Sig if Sig is None else Sig
        next_dag = list(self.env_dag) if env_dag is None else list(env_dag)
        if new_env_record is not None:
            next_dag.append(new_env_record)

        if c_use is None and isinstance(V_use, np.ndarray):
            c_use = V_use[:, 0].copy()
            V_use = sp.csr_array(V_use[:, 1:])
            V_use.indices = V_use.indices.astype(np.int32)
            V_use.indptr = V_use.indptr.astype(np.int32)
        elif isinstance(V_use, np.ndarray):
            V_use = sp.csr_array(V_use)
            V_use.indices = V_use.indices.astype(np.int32)
            V_use.indptr = V_use.indptr.astype(np.int32)

        return ProbSparseImageStar2DCSR(
            c_use, V_use, C_use, d_use, lb_use, ub_use, shape_use,
            mu=next_mu, Sig=next_Sig, cov_mode=self.cov_mode,
            mu0=self.mu0, Sig0=self.Sig0,
            pred_lb0=self.pred_lb0, pred_ub0=self.pred_ub0,
            nVars0=self.nVars0, env_dag=next_dag,
            p_input=self._p_input, copy_=copy_
        )

    def nbytes_prob(self):
        total = self.mu.nbytes + self.Sig.nbytes
        for rec in self.env_dag:
            total += rec.nbytes()
        return total

    def nbytes(self, include_prob=True):
        base = super().nbytes()
        if include_prob:
            return base + self.nbytes_prob()
        return base

    def resetRow(self, index):
        S = super().resetRow(index)
        return self.with_updated_state(S.c, S.V, S.C, S.d, S.pred_lb, S.pred_ub, S.shape, copy_=False)

    def resetRows(self, map):
        if len(map) == 0:
            return self
        S = super().resetRows(map)
        return self.with_updated_state(S.c, S.V, S.C, S.d, S.pred_lb, S.pred_ub, S.shape, copy_=False)

    def affineMap(self, W=None, b=None):
        if W is None and b is None:
            return self
        S = super().affineMap(W, b)
        return self.with_updated_state(S.c, S.V, S.C, S.d, S.pred_lb, S.pred_ub, S.shape, copy_=False)

    def flatten_affineMap(self, W=None, b=None):
        if W is None and b is None:
            return copy.deepcopy(self)
        S = super().flatten_affineMap(W, b)
        return self.with_updated_state(S.c, S.V, S.C, S.d, S.pred_lb, S.pred_ub, S.shape, copy_=False)

    def flatten_affineMap_sparse(self, W=None, b=None):
        if W is None and b is None:
            return copy.deepcopy(self)
        S = super().flatten_affineMap_sparse(W, b)
        return self.with_updated_state(S.c, S.V, S.C, S.d, S.pred_lb, S.pred_ub, S.shape, copy_=False)

    def residualAdd(self, other):
        """
        Pointwise addition Y = self + other in a SHARED predicate space (for ResNets,
        U-Nets, and Vision Transformers where `self` and `other` originate from the
        same input ProbSparseImageStar and `self` may have accumulated additional
        relaxation predicates along the main branch).
        """
        assert isinstance(other, SparseImageStar2DCSR), (
            f"residualAdd expects a SparseImageStar2DCSR/ProbSparseImageStar2DCSR, got {type(other)}"
        )
        c1, V1 = self.get_c_V()
        if isinstance(other, ProbSparseImageStar2DCSR):
            c2, V2 = other.get_c_V()
        else:
            c2 = other.V[:, 0] if other.c is None else other.c
            V2 = other.V[:, 1:] if other.c is None else other.V

        assert c1.shape == c2.shape, f"Dimension mismatch in residualAdd: {c1.shape} vs {c2.shape}"

        # Identify which object has the larger (more evolved) predicate space
        if self.num_pred >= other.num_pred:
            dom = self
            c_new = c1 + c2
            n_diff = self.num_pred - other.num_pred
            if sp.issparse(V1) and sp.issparse(V2):
                if n_diff > 0:
                    V2_pad = sp.hstack([V2, sp.csr_array((V2.shape[0], n_diff), dtype=V2.dtype)], format='csr')
                else:
                    V2_pad = V2
                V_new = (V1 + V2_pad).tocsr()
                V_new.indices = V_new.indices.astype(np.int32)
                V_new.indptr = V_new.indptr.astype(np.int32)
                return dom.with_updated_state(
                    c_new, V_new, dom.C, dom.d, dom.pred_lb, dom.pred_ub, dom.shape, copy_=False
                )
            else:
                V1_d = V1.toarray() if sp.issparse(V1) else V1.copy()
                V2_d = V2.toarray() if sp.issparse(V2) else V2
                V1_d[:, :other.num_pred] += V2_d
                full_V = np.hstack([c_new[:, None], V1_d])
                return dom.with_updated_state(
                    None, full_V, dom.C, dom.d, dom.pred_lb, dom.pred_ub, dom.shape, copy_=False
                )
        else:
            return other.residualAdd(self)

    def getOutputMoments(self):
        """Return output pixel/logit mean vector mu_y and marginal variance vector var_y."""
        c_out, V_out = self.get_c_V()
        mu_y, var_y, _ = compute_preactivation_moments(
            V_out, c_out, self.mu, self.Sig, cov_mode=self.cov_mode
        )
        return mu_y, var_y

    def getSoundInputBounds(self, U=None, u_spec=None):
        """
        Return sound affine lower and upper bounds in terms of the original input
        predicates alpha_0 in [pred_lb0, pred_ub0]:
            d_lb + W_lb @ alpha_0 <= U @ y - u_spec <= d_ub + W_ub @ alpha_0
        """
        c_out, V_out = self.get_c_V()
        d_lb, W_lb, d_ub, W_ub = backsubstitute_linear_bounds(
            c_out, V_out, self.env_dag, self.nVars0, U=U, u_spec=u_spec
        )
        return W_lb, d_lb, W_ub, d_ub

    def estimateProbability(self):
        """Return probability mass of the input distribution over the truncated domain."""
        return self.get_input_probability()

    def verifySpec(self, unsafe_mat, unsafe_vec, lp_solver='gurobi', use_lp_tightening=True, use_lp=None):
        """
        Verify unsafe specification { y | unsafe_mat @ y <= unsafe_vec } using the
        Two-Stage LP + Dual/Moment Projection algorithm.
        """
        if use_lp is not None:
            use_lp_tightening = use_lp
        return verify_prob_spec(
            self, unsafe_mat, unsafe_vec,
            lp_solver=lp_solver, use_lp_tightening=use_lp_tightening
        )

    def certifyLabel(self, label, lp_solver='gurobi', use_lp_tightening=True, use_lp=None):
        """
        Certify classification robustness for ground-truth class `label`.
        Checks whether any adversarial class j != label can have y_j >= y_label
        (i.e. y_label - y_j <= 0).

        Returns dict with:
            - 'rb': 1 (ROBUST / UNSAT), 0 (UNROBUST / all points misclassified),
                    2 (UNKNOWN qualitatively, quantified probabilistically)
            - 'p_unsafe_min', 'p_unsafe_max', 'p_unsafe_hat': joint violation probabilities
            - 'p_safe_min', 'p_safe_max', 'p_safe_hat': conditional robustness probabilities
              in [0, 1] (where 1.0 = 100% robust over input distribution)
            - 'class_details': per-competitor-class verification results
        """
        if use_lp is not None:
            use_lp_tightening = use_lp
        c_out, V_out = self.get_c_V()
        num_classes = len(c_out)
        target_label = int(np.asarray(label).item())

        if hasattr(self, 'get_max_point_cadidates'):
            cand_indices = self.get_max_point_cadidates()
        else:
            lb_est, ub_est = self.estimateRanges()
            cand_indices = np.where(ub_est >= np.max(lb_est))[0]
        if len(cand_indices) == 1 and int(cand_indices[0]) == target_label:
            p_in = self.get_input_probability()
            return {
                'rb': 1,
                'status': 'ROBUST',
                'p_min': p_in, 'p_max': p_in, 'p_hat': p_in,
                'cond_p_min': 1.0, 'cond_p_max': 1.0, 'cond_p_hat': 1.0,
                'p_unsafe_min': 0.0, 'p_unsafe_max': 0.0, 'p_unsafe_hat': 0.0,
                'cond_p_unsafe_min': 0.0, 'cond_p_unsafe_max': 0.0, 'cond_p_unsafe_hat': 0.0,
                'p_safe_min': 1.0, 'p_safe_max': 1.0, 'p_safe_hat': 1.0,
                'p_input': p_in,
                'class_details': {},
            }

        competitors = [int(c) for c in cand_indices if int(c) != target_label]
        if len(competitors) == 0:
            competitors = [j for j in range(num_classes) if j != target_label]

        class_details = {}
        max_cond_p_min = 0.0
        sum_cond_p_max = 0.0
        max_cond_p_hat = 0.0
        p_in = self.get_input_probability()
        any_ambiguous = False
        any_sat_all = False

        for j in competitors:
            U_j = np.zeros((1, num_classes), dtype=np.float64)
            U_j[0, target_label] = 1.0
            U_j[0, j] = -1.0
            u_j = np.zeros(1, dtype=np.float64)
            res_j = self.verifySpec(U_j, u_j, lp_solver=lp_solver, use_lp_tightening=use_lp_tightening)
            class_details[j] = res_j
            if res_j['status'] == 'SAT_ALL':
                any_sat_all = True
            elif res_j['status'] == 'AMBIGUOUS':
                any_ambiguous = True
            max_cond_p_min = max(max_cond_p_min, res_j['cond_p_min'])
            sum_cond_p_max += res_j['cond_p_max']
            max_cond_p_hat = max(max_cond_p_hat, res_j['cond_p_hat'])

        if any_sat_all:
            rb = 0
            status_str = 'VIOLATED'
            cond_unsafe_min = 1.0
            cond_unsafe_max = 1.0
            cond_unsafe_hat = 1.0
        elif not any_ambiguous:
            rb = 1
            status_str = 'ROBUST'
            cond_unsafe_min = 0.0
            cond_unsafe_max = 0.0
            cond_unsafe_hat = 0.0
        else:
            rb = 2
            status_str = 'UNKNOWN'
            # Union of competitor violations: P(U_j E_j) >= max_j P(E_j) and <= min(1, sum_j P(E_j))
            cond_unsafe_min = float(np.clip(max_cond_p_min, 0.0, 1.0))
            cond_unsafe_max = float(np.clip(sum_cond_p_max, cond_unsafe_min, 1.0))
            # Also evaluate joint multi-competitor Moment-Matched Gaussian if >1 active competitor
            active_comps = [j for j, r in class_details.items() if r['status'] != 'UNSAT']
            if len(active_comps) == 1:
                cond_unsafe_hat = class_details[active_comps[0]]['cond_p_hat']
            else:
                U_all = np.zeros((len(active_comps), num_classes), dtype=np.float64)
                for row_k, j in enumerate(active_comps):
                    U_all[row_k, target_label] = 1.0
                    U_all[row_k, j] = -1.0
                d_all = U_all @ c_out
                W_all = np.asarray(U_all @ V_out, dtype=np.float64)
                mu_all, var_all, V_Sig_all = compute_preactivation_moments(
                    W_all, d_all, self.mu, self.Sig, cov_mode=self.cov_mode
                )
                Sig_all = (
                    V_Sig_all @ W_all.T
                    if V_Sig_all is not None
                    else (W_all * self.Sig[None, :]) @ W_all.T
                )
                # P(safe) = P(all y_label - y_j >= 0) = P(0 <= z_all < inf)
                p_safe_ext = mvn_cdf_lowdim(
                    np.zeros(len(active_comps)),
                    np.full(len(active_comps), np.inf),
                    mu_all, Sig_all
                )
                cond_unsafe_hat = float(np.clip(1.0 - p_safe_ext, cond_unsafe_min, cond_unsafe_max))

        cond_safe_min = float(np.clip(1.0 - cond_unsafe_max, 0.0, 1.0))
        cond_safe_max = float(np.clip(1.0 - cond_unsafe_min, cond_safe_min, 1.0))
        cond_safe_hat = float(np.clip(1.0 - cond_unsafe_hat, cond_safe_min, cond_safe_max))

        return {
            'rb': rb,
            'status': status_str,
            'p_min': p_in * cond_safe_min,
            'p_max': p_in * cond_safe_max,
            'p_hat': p_in * cond_safe_hat,
            'cond_p_min': cond_safe_min,
            'cond_p_max': cond_safe_max,
            'cond_p_hat': cond_safe_hat,
            'p_unsafe_min': p_in * cond_unsafe_min,
            'p_unsafe_max': p_in * cond_unsafe_max,
            'p_unsafe_hat': p_in * cond_unsafe_hat,
            'cond_p_unsafe_min': cond_unsafe_min,
            'cond_p_unsafe_max': cond_unsafe_max,
            'cond_p_unsafe_hat': cond_unsafe_hat,
            'p_safe_min': cond_safe_min,
            'p_safe_max': cond_safe_max,
            'p_safe_hat': cond_safe_hat,
            'p_input': p_in,
            'class_details': class_details,
        }

    def certifySegmentationPixels(self, clean_labels, num_classes=None, lp_solver='estimate', use_lp=False):
        """
        Vectorized pixel-level qualitative + quantitative robustness certification
        for semantic segmentation output sets of shape (H, W, num_classes).
        """
        assert len(self.shape) == 3, f"Expected 3D shape (H, W, C), got {self.shape}"
        H, W, C = self.shape
        if num_classes is None:
            num_classes = C
        assert C == num_classes, f"Channel count {C} != num_classes {num_classes}"
        clean_labels = np.asarray(clean_labels, dtype=np.int64).reshape(H, W)

        lb, ub = self.estimateRanges()
        lb_3d = lb.reshape(H, W, C)
        ub_3d = ub.reshape(H, W, C)

        mu_y, var_y = self.getOutputMoments()
        mu_3d = mu_y.reshape(H, W, C)
        var_3d = var_y.reshape(H, W, C)

        # Back-substituted input linear bounds for sound pixel probability intervals
        W_lb0, d_lb0, W_ub0, d_ub0 = self.getSoundInputBounds()
        var0_vec = self.Sig0 if self.Sig0.ndim == 1 else np.diag(self.Sig0)
        mu0_trunc, var0_trunc = compute_truncated_gaussian_moments_1d(
            self.mu0, var0_vec, self.pred_lb0, self.pred_ub0
        )

        pix_rb = np.ones((H, W), dtype=np.int32)
        p_safe_min = np.ones((H, W), dtype=np.float64)
        p_safe_max = np.ones((H, W), dtype=np.float64)
        p_safe_hat = np.ones((H, W), dtype=np.float64)

        for h in range(H):
            for w in range(W):
                y_true = int(clean_labels[h, w])
                lb_true = lb_3d[h, w, y_true]
                ub_true = ub_3d[h, w, y_true]
                others = [c_idx for c_idx in range(C) if c_idx != y_true]
                max_ub_other = np.max(ub_3d[h, w, others])
                max_lb_other = np.max(lb_3d[h, w, others])

                if lb_true > max_ub_other:
                    pix_rb[h, w] = 1
                    p_safe_min[h, w] = 1.0
                    p_safe_max[h, w] = 1.0
                    p_safe_hat[h, w] = 1.0
                elif ub_true < max_lb_other:
                    pix_rb[h, w] = 0
                    p_safe_min[h, w] = 0.0
                    p_safe_max[h, w] = 0.0
                    p_safe_hat[h, w] = 0.0
                else:
                    pix_rb[h, w] = 2
                    # Ambiguous pixel: compute sound [p_safe_min, p_safe_max] and moment-matched p_safe_hat
                    active_others = [c_idx for c_idx in others if ub_3d[h, w, c_idx] >= lb_true]
                    idx_true = (h * W + w) * C + y_true
                    probs_hat = []
                    probs_uns_ub = []
                    probs_uns_lb = []
                    for c_idx in active_others:
                        idx_c = (h * W + w) * C + c_idx
                        diff_mu = mu_3d[h, w, y_true] - mu_3d[h, w, c_idx]
                        diff_std = np.sqrt(max(var_3d[h, w, y_true] + var_3d[h, w, c_idx], _MIN_VAR))
                        diff_lb = lb_true - ub_3d[h, w, c_idx]
                        diff_ub = ub_true - lb_3d[h, w, c_idx]
                        tot = max(float(ndtr((diff_ub - diff_mu) / diff_std) - ndtr((diff_lb - diff_mu) / diff_std)), 1e-12)
                        uns = float(ndtr((min(0.0, diff_ub) - diff_mu) / diff_std) - ndtr((diff_lb - diff_mu) / diff_std))
                        probs_hat.append(float(np.clip(uns / tot, 0.0, 1.0)))

                        # Sound lower bound on margin z = y_true - y_c >= d_z_lb + w_z_lb @ alpha_0
                        w_z_lb = W_lb0[idx_true, :] - W_ub0[idx_c, :]
                        d_z_lb = d_lb0[idx_true] - d_ub0[idx_c]
                        mu_z_lb = float(d_z_lb + w_z_lb @ mu0_trunc)
                        std_z_lb = float(np.sqrt(max(np.sum(np.square(w_z_lb) * var0_trunc), _MIN_VAR)))
                        probs_uns_ub.append(float(np.clip(ndtr((0.0 - mu_z_lb) / std_z_lb), 0.0, 1.0)))

                        # Sound upper bound on margin z = y_true - y_c <= d_z_ub + w_z_ub @ alpha_0
                        w_z_ub = W_ub0[idx_true, :] - W_lb0[idx_c, :]
                        d_z_ub = d_ub0[idx_true] - d_lb0[idx_c]
                        mu_z_ub = float(d_z_ub + w_z_ub @ mu0_trunc)
                        std_z_ub = float(np.sqrt(max(np.sum(np.square(w_z_ub) * var0_trunc), _MIN_VAR)))
                        probs_uns_lb.append(float(np.clip(ndtr((0.0 - mu_z_ub) / std_z_ub), 0.0, 1.0)))

                    p_uns_min = max(probs_uns_lb) if probs_uns_lb else 0.0
                    p_uns_max = max(p_uns_min, min(1.0, sum(probs_uns_ub))) if probs_uns_ub else 1.0
                    p_uns_hat = max(probs_hat) if probs_hat else 0.0
                    p_safe_min[h, w] = float(np.clip(1.0 - p_uns_max, 0.0, 1.0))
                    p_safe_max[h, w] = float(np.clip(1.0 - p_uns_min, p_safe_min[h, w], 1.0))
                    p_safe_hat[h, w] = float(np.clip(1.0 - p_uns_hat, p_safe_min[h, w], p_safe_max[h, w]))

        p_in = self.get_input_probability()
        return {
            'pix_rb': pix_rb,
            'p_input': p_in,
            'p_safe_min': p_safe_min,
            'p_safe_max': p_safe_max,
            'p_safe_hat': p_safe_hat,
            'p_min_map': p_in * p_safe_min,
            'p_max_map': p_in * p_safe_max,
            'p_hat_map': p_in * p_safe_hat,
            'cond_p_min_map': p_safe_min,
            'cond_p_max_map': p_safe_max,
            'cond_p_hat_map': p_safe_hat,
        }

    def sampling_alpha0(self, N, seed=None):
        """
        Draw N exact samples from the truncated input predicate distribution
        alpha_0 | alpha_0 in [pred_lb0, pred_ub0] where alpha_0 ~ N(mu0, Sig0).
        Uses vectorized inverse-CDF sampling in O(N * nVars0) time when Sig0 is diagonal.
        """
        rng = np.random.default_rng(seed)
        if self.nVars0 == 0:
            return np.empty((0, N), dtype=np.float64)

        if self.Sig0.ndim == 1 or np.max(np.abs(self.Sig0 - np.diag(np.diag(self.Sig0)))) < 1e-12:
            var0 = self.Sig0 if self.Sig0.ndim == 1 else np.diag(self.Sig0)
            std0 = np.maximum(np.sqrt(np.maximum(var0, _MIN_VAR)), _MIN_STD)
            xi_l = (self.pred_lb0 - self.mu0) / std0
            xi_u = (self.pred_ub0 - self.mu0) / std0
            cdf_l = ndtr(xi_l)[:, None]
            cdf_u = ndtr(xi_u)[:, None]
            u_rand = rng.uniform(0.0, 1.0, size=(self.nVars0, N))
            p_vals = np.clip(cdf_l + u_rand * (cdf_u - cdf_l), 1e-15, 1.0 - 1e-15)
            alpha0_samples = self.mu0[:, None] + std0[:, None] * ndtri(p_vals)
            return np.clip(alpha0_samples, self.pred_lb0[:, None], self.pred_ub0[:, None])
        else:
            from StarV.util.minimax_tilting_sampler import TruncatedMVN
            tmvn = TruncatedMVN(self.mu0, self.Sig0, self.pred_lb0, self.pred_ub0)
            return tmvn.sample(N)

    def sampling(self, N, seed=None, reshape_hwc=True, reshape_4d=None):
        """
        Draw N input image samples x^(s) = c^(0) + V^(0) @ alpha_0^(s) from the input
        ProbSparseImageStar (valid when called on an input set with nVars == nVars0).
        """
        if reshape_4d is not None:
            reshape_hwc = reshape_4d
        alpha0 = self.sampling_alpha0(N, seed=seed)
        c_cur, V_cur = self.get_c_V()
        V0 = V_cur[:, :self.nVars0]
        samples = c_cur[:, None] + (V0 @ alpha0)
        if reshape_hwc and len(self.shape) == 3:
            H, W, C = self.shape
            return samples.reshape(H, W, C, N)
        return samples

    @staticmethod
    def inf_prob_attack(data, epsilon=0.01, sig_factor=3.0, data_type='image', dtype='float64', cov_mode='diagonal'):
        """Generate a ProbSparseImageStar2DCSR from an L_infinity Gaussian perturbation."""
        if isinstance(data, torch.Tensor):
            assert data.ndim == 3, "Expected 3D torch tensor (C, H, W)"
            data = data.permute(1, 2, 0).detach().cpu().numpy()
        assert isinstance(data, np.ndarray) and data.ndim == 3, "Expected 3D numpy array (H, W, C)"
        np_dtype = np.float64 if dtype == 'float64' else np.float32
        data = data.astype(np_dtype)
        lb = data - epsilon
        ub = data + epsilon
        if data_type == 'image':
            lb = np.clip(lb, 0.0, 1.0)
            ub = np.clip(ub, 0.0, 1.0)
        return ProbSparseImageStar2DCSR(lb, ub, sig_factor=sig_factor, cov_mode=cov_mode)
