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
Flatten Layer Class
Sung Woo Choi, 12/18/2023
"""

import copy
import numpy as np
import scipy.sparse as sp

from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar import SparseImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR

class FlattenLayer(object):
    """FlattenLayer class
        Author: Sung Woo Choi
        Date: 12/18/2023
    """

    def __init__(self, channel_last=True):
        self.channel_last = channel_last

    def evaluate(self, x):
        """ flattens x """
        # shape = x.shape
        if x.ndim == 3:
            if not self.channel_last:
                return x.transpose(2, 0, 1).reshape(-1)
                # return x.reshape(shape[0]*shape[1], shape[2]).reshape(-1, order='F')

        elif x.ndim == 4:
            if self.channel_last:
                return x.reshape(np.prod(x.shape[:3]), x.shape[3])
            else:
                return x.transpose([2, 0, 1, 3]).reshape(np.prod(x.shape[:3]), x.shape[3])
            
        return x.reshape(-1)

    def reachSingleInput(self, In):
        if isinstance(In, ImageStar):
            shape = In.V.shape
            if self.channel_last:
                V = In.V.reshape(1, 1, In.num_pixel, In.num_pred+1)
            else:
                # V = In.V.reshape(shape[:2].prod(), shape[2], shape[3]).reshape(1, 1, shape[:3].prod(), shape[3], order='F')
                V = In.V.transpose([2, 0, 1, 3]).reshape(1, 1, In.num_pixel, In.num_pred+1)
            return ImageStar(V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        elif isinstance(In, SparseImageStar2DCOO):
            if len(In.shape) <= 1:
                return In
            
            out_shape = (In.V.shape[0], )
            
            if self.channel_last:
                if isinstance(In.V, sp.coo_array) or isinstance(In.V, sp.coo_matrix):
                    return SparseImageStar2DCOO(In.c, In.V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)
                else:
                    return In
            else:
                # fron                                 H W C N
                # print(self.V.transpose([3,2,0,1])) # N C H W
                # H W C to
                # C H W
                # H W C N to
                # C H W N
                if isinstance(In.V, sp.coo_array) or isinstance(In.V, sp.coo_matrix):
                    c = In.c.reshape(In.shape).transpose([2, 0, 1])
                    shape = In.shape + (In.num_pred, )
                    V = In.V.todense().reshape(shape).transpose([2, 0, 1, 3])
                    V = np.concatenate([c[:, :, :, None], V], axis=3).reshape(In.V.shape[0], In.num_pred + 1)
                    return SparseImageStar2DCOO(V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)
                else:
                    return In
            # V = np.hstack([In.c[:, None], In.V.toarray()])
            # return SparseImageStar2DCOO(V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)

        elif isinstance(In, SparseImageStar2DCSR):
            if len(In.shape) <= 1:
                return In
            
            out_shape = (In.V.shape[0], )
            if self.channel_last:
                if isinstance(In.V, sp.csr_array) or isinstance(In.V, sp.csr_matrix):
                    return SparseImageStar2DCSR(In.c, In.V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)
                else:
                    return In
            else:
                if isinstance(In.V, sp.csr_array) or isinstance(In.V, sp.csr_matrix):
                    c = In.c.reshape(In.shape).transpose([2, 0, 1]) # transpose from (H, W, C) to (C, H, W)
                    shape = In.shape + (In.num_pred, )
                    V = In.V.todense().reshape(shape).transpose([2, 0, 1, 3])
                    V = np.concatenate([c[:, :, :, None], V], axis=3).reshape(In.V.shape[0], In.num_pred + 1)
                    return SparseImageStar2DCSR(V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)
                else:
                    return In

            # V = np.hstack([In.c[:, None], In.V.toarray()])
            # return SparseImageStar2DCSR(V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)
        
        raise Exception('received unknown input set, {}'.format(In.__class__.__name__))

    def reach(self, in_sets, method=None, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """ main flatten layer method 
            Args:
                @in_sets: a list of input sets or a single input set

            Return:
                @S: flattened reachable set to Star or SparseStar
        """

        if isinstance(in_sets, list):
            F = []
            for i in range(1, len(in_sets[i])):
                F.append(self.reachSingleInput(in_sets[i]))
            return F

        else:
            return self.reachSingleInput(in_sets)
            
        
