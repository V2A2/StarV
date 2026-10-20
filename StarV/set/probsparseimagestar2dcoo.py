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
Probabilistic Sparse Image Star 2D COO Class (ProbSparseImageStar2DCOO)
Combines memory-efficient SparseImageStar2DCOO representation with
extended-predicate Gaussian moment propagation and dual-projected
soundness guarantees for large vision models.
"""

import copy
import torch
import numpy as np
import scipy.sparse as sp
from scipy.special import ndtri, ndtr

from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.probsparseimagestar2dcsr import _init_prob_params_from_bounds
from StarV.set.probsparse_base import (
    compute_box_input_probability,
    compute_preactivation_moments,
    backsubstitute_linear_bounds,
    verify_prob_spec,
    mvn_cdf_lowdim,
    _MIN_VAR,
    _MIN_STD,
)


class ProbSparseImageStar2DCOO(SparseImageStar2DCOO):
    """
    Probabilistic Sparse Image Star in 2D COO format.
    Subclasses SparseImageStar2DCOO so all existing StarV vision layers
    automatically recognize it as a COO sparse image star while preserving
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
            c, V, C, d, mu_arg, Sig_arg, pred_lb, pred_ub, shape = args
            super().__init__(c, V, C, d, pred_lb, pred_ub, shape, copy_=copy_)
            mu = mu_arg
            Sig = Sig_arg
        elif len_ == 8:
            V, C, d, mu_arg, Sig_arg, pred_lb, pred_ub, shape = args
            super().__init__(V, C, d, pred_lb, pred_ub, shape, copy_=copy_)
            mu = mu_arg
            Sig = Sig_arg
        elif len_ == 4:
            lb, ub, mu_arg, Sig_arg = args
            super().__init__(lb, ub, copy_=copy_)
            mu = mu_arg
            Sig = Sig_arg
        elif len_ in (0, 2, 6, 7):
            super().__init__(*args, copy_=copy_)
        else:
            raise ValueError(
                f"Invalid number of positional arguments ({len_}) for ProbSparseImageStar2DCOO; "
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
            V_use = sp.coo_array(V_use[:, 1:])
            V_use.row = V_use.row.astype(np.int32)
            V_use.col = V_use.col.astype(np.int32)
        elif isinstance(V_use, np.ndarray):
            V_use = sp.coo_array(V_use)
            V_use.row = V_use.row.astype(np.int32)
            V_use.col = V_use.col.astype(np.int32)

        return ProbSparseImageStar2DCOO(
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
        assert isinstance(other, SparseImageStar2DCOO), (
            f"residualAdd expects a SparseImageStar2DCOO/ProbSparseImageStar2DCOO, got {type(other)}"
        )
        c1, V1 = self.get_c_V()
        if isinstance(other, ProbSparseImageStar2DCOO):
            c2, V2 = other.get_c_V()
        else:
            c2 = other.V[:, 0] if other.c is None else other.c
            V2 = other.V[:, 1:] if other.c is None else other.V

        assert c1.shape == c2.shape, f"Dimension mismatch in residualAdd: {c1.shape} vs {c2.shape}"

        if self.num_pred >= other.num_pred:
            dom = self
            c_new = c1 + c2
            n_diff = self.num_pred - other.num_pred
            if sp.issparse(V1) and sp.issparse(V2):
                if n_diff > 0:
                    V2_pad = sp.hstack([V2, sp.coo_array((V2.shape[0], n_diff), dtype=V2.dtype)], format='coo')
                else:
                    V2_pad = V2
                V_new = (V1 + V2_pad).tocoo()
                V_new.row = V_new.row.astype(np.int32)
                V_new.col = V_new.col.astype(np.int32)
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
        c_out, V_out = self.get_c_V()
        mu_y, var_y, _ = compute_preactivation_moments(
            V_out, c_out, self.mu, self.Sig, cov_mode=self.cov_mode
        )
        return mu_y, var_y

    def getSoundInputBounds(self, U=None, u_spec=None):
        c_out, V_out = self.get_c_V()
        d_lb, W_lb, d_ub, W_ub = backsubstitute_linear_bounds(
            c_out, V_out, self.env_dag, self.nVars0, U=U, u_spec=u_spec
        )
        return W_lb, d_lb, W_ub, d_ub

    def estimateProbability(self):
        return self.get_input_probability()

    def verifySpec(self, unsafe_mat, unsafe_vec, lp_solver='gurobi', use_lp_tightening=True, use_lp=None):
        if use_lp is not None:
            use_lp_tightening = use_lp
        return verify_prob_spec(
            self, unsafe_mat, unsafe_vec,
            lp_solver=lp_solver, use_lp_tightening=use_lp_tightening
        )

    def certifyLabel(self, label, lp_solver='gurobi', use_lp_tightening=True, use_lp=None):
        # Delegate to shared logic via ProbSparseImageStar2DCSR.certifyLabel
        from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
        if use_lp is not None:
            use_lp_tightening = use_lp
        return ProbSparseImageStar2DCSR.certifyLabel(
            self, label, lp_solver=lp_solver, use_lp_tightening=use_lp_tightening
        )

    def certifySegmentationPixels(self, clean_labels, num_classes=None, lp_solver='estimate', use_lp=False):
        from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
        return ProbSparseImageStar2DCSR.certifySegmentationPixels(
            self, clean_labels, num_classes=num_classes, lp_solver=lp_solver, use_lp=use_lp
        )

    def sampling_alpha0(self, N, seed=None):
        from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
        return ProbSparseImageStar2DCSR.sampling_alpha0(self, N, seed=seed)

    def sampling(self, N, seed=None, reshape_hwc=True, reshape_4d=None):
        from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
        return ProbSparseImageStar2DCSR.sampling(
            self, N, seed=seed, reshape_hwc=reshape_hwc, reshape_4d=reshape_4d
        )

    @staticmethod
    def inf_prob_attack(data, epsilon=0.01, sig_factor=3.0, data_type='image', dtype='float64', cov_mode='diagonal'):
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
        return ProbSparseImageStar2DCOO(lb, ub, sig_factor=sig_factor, cov_mode=cov_mode)
