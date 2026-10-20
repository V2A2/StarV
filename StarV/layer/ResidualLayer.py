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
Residual / Skip-Connection Layer Class (ResNet, U-Net, ViT Residual Blocks)
Supports both qualitative (Star, ImageStar, SparseImageStar2DCSR/COO) and
quantitative (ProbStar, ProbSparseImageStar2DCSR/COO) reachability in a
shared predicate space without double-counting ancestor uncertainty.
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


def _shared_predicate_add(S1, S2):
    """
    Add two Star/ImageStar/SparseImageStar/ProbStar/ProbSparseImageStar sets
    that share a common ancestor predicate prefix.
    Unlike Minkowski sum (which treats predicates as independent), this adds
    basis columns corresponding to shared predicate variables.
    """
    if hasattr(S1, 'residualAdd'):
        return S1.residualAdd(S2)
    if hasattr(S2, 'residualAdd'):
        return S2.residualAdd(S1)

    if isinstance(S1, Star) and isinstance(S2, Star):
        assert S1.dim == S2.dim, f"Dimension mismatch: {S1.dim} vs {S2.dim}"
        if S1.nVars >= S2.nVars:
            dom, sub = S1, S2
        else:
            dom, sub = S2, S1
        V_new = dom.V.copy()
        V_new[:, :sub.nVars + 1] += sub.V
        return Star(V_new, dom.C, dom.d, dom.pred_lb, dom.pred_ub)

    if isinstance(S1, ImageStar) and isinstance(S2, ImageStar):
        if S1.num_pred >= S2.num_pred:
            dom, sub = S1, S2
        else:
            dom, sub = S2, S1
        V_new = dom.V.copy()
        V_new[:, :, :, :sub.num_pred + 1] += sub.V
        return ImageStar(V_new, dom.C, dom.d, dom.pred_lb, dom.pred_ub)

    if isinstance(S1, SparseImageStar2DCSR) and isinstance(S2, SparseImageStar2DCSR):
        if S1.num_pred >= S2.num_pred:
            dom, sub = S1, S2
        else:
            dom, sub = S2, S1
        c1 = dom.V[:, 0] if dom.c is None else dom.c
        V1 = dom.V[:, 1:] if dom.c is None else dom.V
        c2 = sub.V[:, 0] if sub.c is None else sub.c
        V2 = sub.V[:, 1:] if sub.c is None else sub.V
        c_new = c1 + c2
        n_diff = dom.num_pred - sub.num_pred
        if sp.issparse(V1) and sp.issparse(V2):
            if n_diff > 0:
                V2_pad = sp.hstack([V2, sp.csr_array((V2.shape[0], n_diff), dtype=V2.dtype)], format='csr')
            else:
                V2_pad = V2
            V_new = (V1 + V2_pad).tocsr()
            V_new.indices = V_new.indices.astype(np.int32)
            V_new.indptr = V_new.indptr.astype(np.int32)
            return SparseImageStar2DCSR(c_new, V_new, dom.C, dom.d, dom.pred_lb, dom.pred_ub, dom.shape)
        else:
            V1_d = V1.toarray() if sp.issparse(V1) else V1.copy()
            V2_d = V2.toarray() if sp.issparse(V2) else V2
            V1_d[:, :sub.num_pred] += V2_d
            V_full = np.hstack([c_new[:, None], V1_d])
            return SparseImageStar2DCSR(V_full, dom.C, dom.d, dom.pred_lb, dom.pred_ub, dom.shape)

    if isinstance(S1, SparseImageStar2DCOO) and isinstance(S2, SparseImageStar2DCOO):
        if S1.num_pred >= S2.num_pred:
            dom, sub = S1, S2
        else:
            dom, sub = S2, S1
        c1 = dom.V[:, 0] if dom.c is None else dom.c
        V1 = dom.V[:, 1:] if dom.c is None else dom.V
        c2 = sub.V[:, 0] if sub.c is None else sub.c
        V2 = sub.V[:, 1:] if sub.c is None else sub.V
        c_new = c1 + c2
        n_diff = dom.num_pred - sub.num_pred
        if sp.issparse(V1) and sp.issparse(V2):
            if n_diff > 0:
                V2_pad = sp.hstack([V2, sp.coo_array((V2.shape[0], n_diff), dtype=V2.dtype)], format='coo')
            else:
                V2_pad = V2
            V_new = (V1 + V2_pad).tocoo()
            V_new.row = V_new.row.astype(np.int32)
            V_new.col = V_new.col.astype(np.int32)
            return SparseImageStar2DCOO(c_new, V_new, dom.C, dom.d, dom.pred_lb, dom.pred_ub, dom.shape)
        else:
            V1_d = V1.toarray() if sp.issparse(V1) else V1.copy()
            V2_d = V2.toarray() if sp.issparse(V2) else V2
            V1_d[:, :sub.num_pred] += V2_d
            V_full = np.hstack([c_new[:, None], V1_d])
            return SparseImageStar2DCOO(V_full, dom.C, dom.d, dom.pred_lb, dom.pred_ub, dom.shape)

    raise Exception(f"Unsupported set types for residual addition: {type(S1)} and {type(S2)}")


class ResidualLayer(object):
    """
    Residual / Skip-Connection Layer:
      y = post_activation( main_branch(x) + shortcut_branch(x) )
    where shortcut_branch defaults to identity (x).
    """

    def __init__(self, main_branch, shortcut_branch=None, post_activation=None, in_dim=None, out_dim=None):
        if isinstance(main_branch, list):
            self.main_layers = main_branch
        elif hasattr(main_branch, 'layers'):
            self.main_layers = list(main_branch.layers)
        else:
            self.main_layers = [main_branch]

        if shortcut_branch is None:
            self.shortcut_layers = []
        elif isinstance(shortcut_branch, list):
            self.shortcut_layers = shortcut_branch
        elif hasattr(shortcut_branch, 'layers'):
            self.shortcut_layers = list(shortcut_branch.layers)
        else:
            self.shortcut_layers = [shortcut_branch]

        self.post_activation = post_activation
        self.in_dim = in_dim if in_dim is not None else getattr(self.main_layers[0], 'in_dim', None)
        self.out_dim = out_dim
        if self.out_dim is None:
            for lyr in reversed(self.main_layers):
                if hasattr(lyr, 'out_dim') and lyr.out_dim is not None:
                    self.out_dim = lyr.out_dim
                    break
            if self.out_dim is None:
                self.out_dim = self.in_dim

    def __str__(self):
        main_names = [l.__class__.__name__ for l in self.main_layers]
        short_names = [l.__class__.__name__ for l in self.shortcut_layers] if self.shortcut_layers else ['Identity']
        post_name = self.post_activation.__class__.__name__ if self.post_activation is not None else 'None'
        print(f"ResidualLayer(main={main_names}, shortcut={short_names}, post={post_name})")
        return '\n'

    def info(self):
        print(self)

    def evaluate(self, x):
        orig_ndim = x.ndim
        y_main = x.copy()
        for lyr in self.main_layers:
            y_main = lyr.evaluate(y_main)
        y_short = x.copy()
        for lyr in self.shortcut_layers:
            y_short = lyr.evaluate(y_short)
        if y_main.ndim == 4 and y_short.ndim == 3 and y_main.shape[3] == 1:
            y_short = y_short.reshape(y_main.shape)
        elif y_short.ndim == 4 and y_main.ndim == 3 and y_short.shape[3] == 1:
            y_main = y_main.reshape(y_short.shape)
        y = y_main + y_short
        if self.post_activation is not None:
            y = self.post_activation.evaluate(y)
        if orig_ndim == 3 and y.ndim == 4 and y.shape[3] == 1:
            y = y[:, :, :, 0]
        return y

    def reachSingleInput(self, In, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False, **kwargs):
        S_main = copy.deepcopy(In)
        for lyr in self.main_layers:
            S_main = lyr.reach(S_main, method=method, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show, **kwargs)
            if isinstance(S_main, list) and len(S_main) == 1:
                S_main = S_main[0]

        if len(self.shortcut_layers) == 0:
            S_short = In
        else:
            S_short = copy.deepcopy(In)
            for lyr in self.shortcut_layers:
                S_short = lyr.reach(S_short, method=method, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show, **kwargs)
                if isinstance(S_short, list) and len(S_short) == 1:
                    S_short = S_short[0]

        S_out = _shared_predicate_add(S_main, S_short)
        if self.post_activation is not None:
            S_out = self.post_activation.reach(S_out, method=method, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show, **kwargs)
            if isinstance(S_out, list) and len(S_out) == 1:
                S_out = S_out[0]
        return S_out

    def reach(self, inputSet, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False, **kwargs):
        if isinstance(inputSet, list):
            return [
                self.reachSingleInput(s, method=method, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show, **kwargs)
                for s in inputSet
            ]
        return self.reachSingleInput(inputSet, method=method, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show, **kwargs)
