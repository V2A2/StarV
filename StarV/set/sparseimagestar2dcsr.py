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
Sparse Image Star 2D Class
Sung Woo Choi, 03/17/2024

"""

# !/usr/bin/python3
import copy
import contextlib
import io
import os
import sys
import torch
import numpy as np
import scipy.sparse as sp
import warnings
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog
from scipy.linalg import block_diag
# from scipy.ndimage import shift
# import numba
import glpk
import polytope as pc
from StarV.set.predicate_layout import PredLayout
from StarV.util.lp_solver import solve_index_lp as util_solve_index_lp

from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO

GUROBI_OPT_TOL = 1e-6

class SparseImageStar2DCSR(object):
    """
        Sparse Image Star for reachability
        author: Sung Woo Choi
        date: 03/17/2024
        Representation of a SparseImageStar
        ======================= np.zeros(self.height, self.height, self.num_channel)
        H W C N
        N:batch_size, H:input_img_height, W:input_img_width, C:no.of.channels 
        https://pytorch.org/blog/accelerating-pytorch-vision-models-with-channels-last-on-cpu/
        ==========================================================================  
    """

    def __init__(self, *args, copy_=True):
        """
            Key Attributes:
            c = [] @ 1D numpy array 
            V = [] @ 2D scipy csr matrix
            C = [] @ 2D scipy csr matrix
            d = [] @ 1D numpy array

            num_pred = 0 # number of predicate variables
            pred_lb = [] # lower bound of predicate variables
            pred_ub = [] # upper bound of predicate variables

            height = 0 # height of the image
            width = 0 # width of the image
            num_channel = 0 # number of channels of the image
            num_pred = 0 # number of predicate variables
            num_pixel = 0 # number of pixels in image
        """

    
        len_ = len(args)
        
        if len_ == 8:
            
            [c, V, C, d, pred_lb, pred_ub, pred_layout, shape] = copy.deepcopy(args) if copy_ is True else args

            # if len(shape) == 2:
            #     assert isinstance(V, sp.csr_array) or isinstance(V, sp.csr_matrix) or \
            #     isinstance(V, np.ndarray), \
            #     'error: generator image should be a numpy ndarray or scipy csr array or matrix'
            #     assert shape == V.shape[0], \
            #     'error: inconsistency between shape and shape of basis vector'
            # else:
            #     assert isinstance(V, sp.csr_array) or isinstance(V, sp.csr_matrix), \
            #     'error: generator image should be a scipy csr array or matrix'
            #     assert np.array(shape).prod() == V.shape[0], \
            #     'error: inconsistency between shape and shape of basis vector'

            assert isinstance(V, sp.csr_array) or isinstance(V, sp.csr_matrix), \
            'error: generator image should be a scipy csr array or matrix'
            assert np.array(shape).prod() == V.shape[0], \
            f'error: inconsistency between shape and shape of basis vector; shape={np.array(shape).prod()}, V={V.shape}'
                
            assert isinstance(c, np.ndarray) and c.ndim == 1, \
            'error: anchor image should be a 1D numpy array'
            assert c.shape[0] == V.shape[0], \
            f'error: inconsistency between anchor image and generator image; c.shape[0]={c.shape[0]}, V.shape[0]={V.shape[0]}'
            assert isinstance(pred_lb, np.ndarray) and pred_lb.ndim == 1, \
            'error: lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray) and pred_ub.ndim == 1, \
            'error: upper bound vector should be a 1D numpy array'
            assert pred_ub.shape == pred_lb.shape, \
            'error: inconsistent number of predicate variables between predicate lower- and upper-boud vectors'
            if pred_layout is not None:
                assert isinstance(pred_layout, PredLayout), \
                'error: pred_layout should be an instance of PredLayout or None'
                assert pred_layout.n_total() == pred_lb.shape[0] == pred_ub.shape[0], 'error: ' +\
                'Inconsistency between predicate layout and predicate lower- or upper-bound vectors'
            assert (isinstance(shape, np.ndarray) or isinstance(shape, list) or isinstance(shape, tuple))and len(shape) >= 1 and len(shape) <= 3, \
            'error: shape should be a numpy array, list, or tuple containing shape of generator image but received {}'.format(shape)

            if len(d) > 0:
                assert isinstance(C, sp.csr_array) or isinstance(C, sp.csr_matrix), \
                'error: linear constraints matrix should be a 2D scipy sparse csr array or matrix'
                assert isinstance(d, np.ndarray) and d.ndim == 1, \
                'error: linear constraints vector should be a 1D numpy array'
                assert C.shape[0] == d.shape[0], \
                'error: inconsistency between constraint matrix and constraint vector'
                assert C.shape[1] == pred_lb.shape[0], \
                'error: inconsistent number of predicate variables between constraint matrix and predicate bound vectors'
                assert C.shape[1] == V.shape[1], \
                'error: inconsistent number of predicate variables between constraint matrix and generato image'
            
            self.c = c
            self.V = V
            self.C = C
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.pred_layout = pred_layout
            self.shape = shape # height, width, num_channel
            self.num_pred = V.shape[1]

        elif len_ == 7:
            
            [c, V, C, d, pred_lb, pred_ub, shape] = copy.deepcopy(args) if copy_ is True else args

            # if len(shape) == 2:
            #     assert isinstance(V, sp.csr_array) or isinstance(V, sp.csr_matrix) or \
            #     isinstance(V, np.ndarray), \
            #     'error: generator image should be a numpy ndarray or scipy csr array or matrix'
            #     assert shape == V.shape[0], \
            #     'error: inconsistency between shape and shape of basis vector'
            # else:
            #     assert isinstance(V, sp.csr_array) or isinstance(V, sp.csr_matrix), \
            #     'error: generator image should be a scipy csr array or matrix'
            #     assert np.array(shape).prod() == V.shape[0], \
            #     'error: inconsistency between shape and shape of basis vector'

            assert isinstance(V, sp.csr_array) or isinstance(V, sp.csr_matrix), \
            'error: generator image should be a scipy csr array or matrix'
            assert np.array(shape).prod() == V.shape[0], \
            f'error: inconsistency between shape and shape of basis vector; shape={np.array(shape).prod()}, V={V.shape}'
                
            assert isinstance(c, np.ndarray) and c.ndim == 1, \
            'error: anchor image should be a 1D numpy array'
            assert c.shape[0] == V.shape[0], \
            f'error: inconsistency between anchor image and generator image; c.shape[0]={c.shape[0]}, V.shape[0]={V.shape[0]}'
            assert isinstance(pred_lb, np.ndarray) and pred_lb.ndim == 1, \
            'error: lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray) and pred_ub.ndim == 1, \
            'error: upper bound vector should be a 1D numpy array'
            assert pred_ub.shape == pred_lb.shape, \
            'error: inconsistent number of predicate variables between predicate lower- and upper-boud vectors'
            assert (isinstance(shape, np.ndarray) or isinstance(shape, list) or isinstance(shape, tuple))and len(shape) >= 1 and len(shape) <= 3, \
            'error: shape should be a numpy array, list, or tuple containing shape of generator image but received {}'.format(shape)

            if len(d) > 0:
                assert isinstance(C, sp.csr_array) or isinstance(C, sp.csr_matrix), \
                'error: linear constraints matrix should be a 2D scipy sparse csr array or matrix'
                assert isinstance(d, np.ndarray) and d.ndim == 1, \
                'error: linear constraints vector should be a 1D numpy array'
                assert C.shape[0] == d.shape[0], \
                'error: inconsistency between constraint matrix and constraint vector'
                assert C.shape[1] == pred_lb.shape[0], \
                'error: inconsistent number of predicate variables between constraint matrix and predicate bound vectors'
                assert C.shape[1] == V.shape[1], \
                'error: inconsistent number of predicate variables between constraint matrix and generato image'
            
            self.c = c
            self.V = V
            self.C = C
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.shape = shape # height, width, num_channel
            self.num_pred = V.shape[1]

        elif len_ == 6:

            [V, C, d, pred_lb, pred_ub, shape] = copy.deepcopy(args) if copy_ is True else args

            assert isinstance(V, np.ndarray), \
            'error: basis matrix should be a numpy array'
            if len(shape) == 1:
                assert shape[0] == V.shape[0], \
                'error: inconsistency between shape and shape of basis image, shape={}, V.shape[0]={}'.format(shape, V.shape[0])
            else:
                assert shape == V.shape[:-1], \
                'error: inconsistency between shape and shape of basis image, shape={}, V.shape[0]={}'.format(shape, V.shape[:-1])
            
            assert isinstance(pred_lb, np.ndarray) and pred_lb.ndim == 1, \
            'error: lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray) and pred_ub.ndim == 1, \
            'error: upper bound vector should be a 1D numpy array'
            assert pred_ub.shape == pred_lb.shape, \
            'error: inconsistent number of predicate variables between predicate lower- and upper-boud vectors'
            assert (isinstance(shape, np.ndarray) or isinstance(shape, list) or isinstance(shape, tuple))and len(shape) >= 1 and len(shape) <= 3, \
            'error: shape should be a numpy array, list, or tuple containing shape of generator image but received {}'.format(shape)

            if len(d) > 0:
                assert isinstance(C, sp.csr_array) or isinstance(C, sp.csr_matrix), \
                'error: linear constraints matrix should be a 2D scipy sparse csr array or matrix'
                assert isinstance(d, np.ndarray) and d.ndim == 1, \
                'error: linear constraints vector should be a 1D numpy array'
                assert C.shape[0] == d.shape[0], \
                'error: inconsistency between constraint matrix and constraint vector'
                assert C.shape[1] == pred_lb.shape[0], \
                'error: inconsistent number of predicate variables between constraint matrix and predicate bound vectors'
                assert C.shape[1] == V.shape[-1] - 1, \
                'error: inconsistent number of predicate variables between constraint matrix and generato image'
            
            self.c = None
            self.V = V
            self.C = C
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.shape = shape
            self.num_pred = self.V.shape[-1] - 1

        elif len_ == 2:

            [lb, ub] = copy.deepcopy(args) if copy_ is True else args

            assert isinstance(lb, np.ndarray), \
            'error: lower bound image should be a numpy array'
            assert isinstance(ub, np.ndarray), \
            'error: upper bound image should be a numpy array'
            assert lb.shape == ub.shape, \
            'error: inconsistency between lower bound image and upper bound image'
            assert lb.ndim > 1 and lb.ndim <= 3, \
            'error: lower and upper bound vectors should be a 2D or 3D numpy array'

            if (ub < lb).any():
                raise Exception(
                    'error: the upper bounds must not be less than the lower bounds for all dimensions')
            
            if lb.ndim == 2:
                lb = lb[:, :, None]
                ub = ub[:, :, None]

            self.shape = lb.shape
            dtype = lb.dtype

            lb = lb.reshape(-1)
            ub = ub.reshape(-1)
            dim = lb.shape[0]

            gtr = ub > lb
            nv = gtr.sum()
            data = 0.5 * (ub[gtr] - lb[gtr])
            indices = np.arange(nv, dtype=np.int32)
            if nv == dim:
                indptr = np.arange(dim+1, dtype=np.int32)
            else:
                indptr = np.zeros(dim+1, dtype=np.int32)
                for i in range(dim):
                    if gtr[i]:
                        indptr[i+1] = indptr[i] + 1
                    else:
                        indptr[i+1] = indptr[i]
            
            self.c = 0.5 * (lb + ub)
            self.V = sp.csr_array(
                        (data, indices, indptr), shape=(lb.shape[0], nv)
                    )
            self.C = sp.csr_array((0, 0), dtype=dtype)
            self.d = np.empty([0], dtype=dtype)

            self.pred_lb = -np.ones(nv, dtype=dtype)
            self.pred_ub = np.ones(nv, dtype=dtype)
            self.num_pred = nv

        # elif len_ == 2:
        #     [lb, ub] = args

        #     if copy is True:
        #         lb = lb.copy()
        #         ub = ub.copy()
            
        #     assert isinstance(lb, np.ndarray), \
        #     'error: lower bound image should be a numpy array'
        #     assert isinstance(ub, np.ndarray), \
        #     'error: upper bound image should be a numpy array'
        #     assert lb.shape == ub.shape, \
        #     'error: inconsistency between lower bound image and upper bound image'
        #     assert lb.ndim > 1 and lb.ndim <= 3, \
        #     'error: lower and upper bound vectors should be a 2D or 3D numpy array'

        #     if (ub < lb).any():
        #         raise Exception(
        #             'error: the upper bounds must not be less than the lower bounds for all dimensions')
            
        #     if lb.ndim == 2:
        #         lb = lb[:, :, None]
        #         ub = ub[:, :, None]

        #     self.shape = lb.shape
        #     dtype = lb.dtype

        #     lb = lb.reshape(-1)
        #     ub = ub.reshape(-1)
            
        #     gtr = ub > lb
        #     nv = gtr.sum()
        #     indices = np.where(gtr)[0].astype(np.int32)
        #     indptr = np.arange(nv+1, dtype=np.int32)
        #     data = np.ones(nv, dtype=dtype)

        #     # self.c = np.zeros(lb.shape[0], dtype=dtype)
        #     self.c = (ub == lb) * lb
        #     self.V = sp.csr_array(
        #             (data, indices, indptr), shape=(lb.shape[0], nv)
        #         )
        #     # # avoid creating csr_array by
        #     # #      sp.csr_array(
        #     # #             (data, indices, indptr), shape=(lb.shape[0], nv)
        #     # #         )
        #     # # because it creates indices and indtpr with np.int64 instead of np.int32
        #     # self.V = sp.csr_array((lb.shape[0], nv))
        #     # self.V.data = np.ones(nv, dtype=dtype)
        #     # self.V.indices = np.where(ub > lb)[0].astype(np.int32)
        #     # self.V.indptr = np.arange(nv+1, dtype=np.int32)

        #     self.C = sp.csr_array((0, 0), dtype=dtype)
        #     self.d = np.empty([0], dtype=dtype)

        #     self.pred_lb = lb
        #     self.pred_ub = ub
        #     self.num_pred = nv
        #     # self.num_pixel = lb.shape[0]
        
        elif len_ == 0: 
            self.c = np.empty([0, 0, 0])
            self.V = sp.csr_array((0, 0))
            self.C = sp.csr_array((0, 0))
            self.d = np.empty([0])

            self.pred_lb = np.empty([0])
            self.pred_ub = np.empty([0])
            
            self.shape = [0, 0, 0]
            self.num_pred = 0
            # self.num_pixel = 0

        else:
            raise Exception(
                'error: invalid number of input arguments (should be 0, 1, 2, 6)')
        

    def __str__(self, to_dense=False, to_coo=False):
        print('SparseImageStar2DCSR Set:')
        if self.c is None:
            print(f'V_{self.V.getformat()}: {self.V}')
        else:
            print(f'c: {self.c}')
            if to_dense:
                print(f'V_{self.V.getformat()}: \n{self.V.todense()}')
            elif to_coo:
                print(f'V_{self.V.getformat()}: \n{self.V}')
            else:
                print(f'V_{self.V.getformat()}:\n   data: {self.V.data}\nindices: {self.V.indices}\n indptr: {self.V.indptr}')
        if to_dense:
            print(f'C_{self.C.getformat()}: \n{self.C.todense()}')
        elif to_coo:
            print(f'C_{self.C.getformat()}: \n{self.C}')
        else:
            print(f'C_{self.C.getformat()}:\n   data: {self.C.data}\nindices: {self.C.indices}\n indptr: {self.C.indptr}')
        print(f'd: {self.d}')
        print(f'pred_lb: {self.pred_lb}')
        print(f'pred_ub: {self.pred_ub}')
        print(f'shape: {self.shape}')
        print(f'num_pred: {self.num_pred}')
        print(f'density: {self.density()}')
        if not isinstance(self.V, np.ndarray):
            print(f'nnz: {self.V.nnz}')
        print()
        return ''
    
    def __repr__(self):
        print('SparseImageStar2DCSR Set:')
        if self.c is None:
            print('V: {}, {}'.format(self.V.shape, self.V.dtype))
        else:
            print('c: {}, {}'.format(self.c.shape, self.c.dtype))
            print('V_{}: {}, data: {}, indices: {}, indptr: {}'.format(self.V.getformat(), self.V.shape, self.V.data.dtype, self.V.indices.dtype, self.V.indptr.dtype))

        print('C_{}: {}, {}'.format(self.C.getformat(), self.C.shape, self.C.dtype))
        print('d: {}, {}'.format(self.d.shape, self.d.dtype))
        print('pred_lb: {}, {}'.format(self.pred_lb.shape, self.pred_lb.dtype))
        print('pred_ub: {}, {}'.format(self.pred_ub.shape, self.pred_ub.dtype))
        print('shape: {}'.format(self.shape))
        print('num_pred: {}'.format(self.num_pred))
        print('density: {}'.format(self.density()))
        if not isinstance(self.V, np.ndarray):
            print('nnz: {}'.format(self.V.nnz))
        print('')
        return ''
    
    def __len__(self):
        return 1
    
    def clone(self):
        return copy.deepcopy(self)

    def nbytes_generator(self):
        if isinstance(self.V, np.ndarray):
            return self.V.nbytes
        else:
            return self.V.data.nbytes + self.V.indices.nbytes + self.V.indptr.nbytes + self.c.nbytes
    
    def nbytes_constraints(self):
        return self.C.data.nbytes + self.C.indices.nbytes + self.C.indptr.nbytes + self.d.nbytes
    
    def nbytes(self):
        # V and c
        nbt = self.nbytes_generator()
        # C and d
        nbt += self.nbytes_constraints()
        # pred_lb and pred_ub
        nbt += self.pred_lb.nbytes + self.pred_ub.nbytes
        return nbt
    
    def density(self):
        if isinstance(self.V, np.ndarray):
            return 1.0
        else:
            if self.V.nnz == 0:
                return 0.0  
            return self.V.nnz / (self.V.shape[0] * self.V.shape[1])
    
    def resetRow(self, index):
        '''Reset a row with index'''
        
        if isinstance(self.V, np.ndarray):
            V = copy.deepcopy(self.V)
            V[index, :] = 0
            return SparseImageStar2DCSR(V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
        else:
            c = copy.deepcopy(self.c)
            c[index] = 0
            V = self.resetRow_V(index)
            return SparseImageStar2DCSR(c, V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
    
    def resetRows(self, map):
        '''Reset a row with map of indexes'''
        if isinstance(self.V, np.ndarray):
            V = copy.deepcopy(self.V)
            V[map, :] = 0
            return SparseImageStar2DCSR(V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
        else:
            c = copy.deepcopy(self.c)
            c[map] = 0
            # V = self.resetRows_V2(map) #
            # V = self.resetRows_V3(map)
            V = self.reset_rows_csr(map)
            # V = self.reset_rows_csr_zero_out(map)
            return SparseImageStar2DCSR(c, V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
    
    def resetRow_V(self, index):
        V = copy.deepcopy(self.V)        
        n = V.indptr[index+1] - V.indptr[index]

        if n > 0:
            V.data[V.indptr[index]:-n] = V.data[V.indptr[index+1]:]
            V.data = V.data[:-n]
            V.indices[V.indptr[index]:-n] = V.indices[V.indptr[index+1]:]
            V.indices = V.indices[:-n]
        # V.indptr[index:-1] = V.indptr[index+1:]
        V.indptr[index+1:] -= n
        # V.indptr = V.indptr[:-1]
        # V._shape = (V._shape[0]-1, V._shape[1])
        return V
    
    def resetRows_V_orig(self, map):
        V = copy.deepcopy(self.V)
        
        n = (V.indptr[1:] - V.indptr[:-1]).astype(np.uint32)
        b = np.repeat(n[map][:, None], V.shape[0]+1, axis=1)
        for i, m in enumerate(map):
            b[i, :] = shift(b[i, :], m+1)
        new_indptr = V.indptr - b.sum(axis=0)    

        mask = np.ones(V.shape[0], dtype=bool)
        mask[map] = False
        V = V[mask]
        V.indptr = new_indptr
        V._shape = self.V.shape
        return V

    def resetRows_V(self, map):
        V = copy.deepcopy(self.V)
        
        n = (V.indptr[1:] - V.indptr[:-1]).astype(np.int32)
        new_indptr = copy.deepcopy(V.indptr)
        for e in map: 
            new_indptr[e+1:] -= n[e]

        mask = np.ones(V.shape[0], dtype=bool)
        mask[map] = False
        V = V[mask]

        new_V = sp.csr_array(self.V.shape)
        new_V.data = V.data
        new_V.indices = V.indices.astype(np.int32)
        new_V.indptr = new_indptr.astype(np.int32)
        # return sp.csr_array((V.data, V.indices, new_indptr), shape=self.V.shape)
        return new_V
    
    def resetRows_V2(self, map):
        a = np.ones(self.V.shape[0], dtype=bool)
        a[map] = False
        return self.V.multiply(a[:, None]).tocsr(copy=False)
    
    def resetRows_V3(self, map):
        mask = np.ones(self.V.shape[0], dtype=bool)
        mask[map] = False
        return self.V[mask, :]
    
    def reset_rows_csr_zero_out(self, map, eliminate_zeros=False):
        """
        Zero out given rows in a CSR matrix V without changing its shape.
        """
        indptr = self.V.indptr
        data = self.V.data.copy()

        for r in map:
            start, end = indptr[r], indptr[r + 1]
            data[start:end] = 0

        V = sp.csr_array((data, self.V.indices.copy(), self.V.indptr.copy()), shape=self.V.shape)
        if eliminate_zeros:
            V.eliminate_zeros()
        return V
    
    def reset_rows_csr(self, map):
        n_rows, n_cols = self.V.shape
        indptr, indices, data = self.V.indptr, self.V.indices, self.V.data
        nnz = indptr[-1]

        mask = np.ones(nnz, dtype=bool)

        # reset each row
        for r in map:
            start, end = indptr[r], indptr[r + 1]
            mask[start:end] = False
        
        new_data = data[mask]
        new_indices = indices[mask]

        # Build new indptr
        row_nnz = indptr[1:] - indptr[:-1]
        row_keep_mask = np.ones(n_rows, dtype=bool)
        row_keep_mask[map] = False
        row_nnz_keep = row_nnz * row_keep_mask

        new_indptr = np.zeros(n_rows + 1, dtype=indptr.dtype)
        new_indptr[1:] = np.cumsum(row_nnz_keep)

        return sp.csr_array((new_data, new_indices, new_indptr), shape=self.V.shape)

    # def resetRows_V(self, map):
    #     V = copy.deepcopy(self.V)
    #     map = list(map)
    #     mask = np.ones(V.shape[0], dtype=bool)
    #     mask[map] = False
    #     V = V[mask]
    #     V._shape = self.V.shape
    #     return V

    def affineMap(self, W=None, b=None):
        if W is None and b is None:
            return self
        
        # elif isinstance(self.V, sp.csr_array) or isinstance(self.V, sp.csr_matrix):
        elif len(self.shape) > 1:
            c = self.c.copy()
            if W is not None:
                assert W.ndim == len(self.shape), f"inconsistent number of array dimensions between W and shape of SparseImageStar; len(shape)={len(self.shape)}, W.ndim={W.ndim}"
                
                Wr = W.reshape(-1)
                if np.prod(W.shape) == 1:
                    # scalar multiplication
                    scale = Wr.item()
                    c = c * scale
                    V = self.V * scale
                else:
                    # element-wise multiplication
                    c = c.reshape(self.shape) * W
                    c = c.reshape(-1)

                    # self.V (csr) * W
                    # element-wise multiplication of a csr matrix with a dense array.
                    # csr_matrix in (h*w*c, m) format, Wr in (h*w*c) format 
                    V = self.V.copy()
                    T = self.V.tocoo(copy=False)
                    V.data = V.data * Wr[T.row]
            else:
                V = self.V
            
            if b is not None:
                c = c.reshape(self.shape)
                if b.ndim == len(self.shape):
                    c += b
                elif b.ndim > 1:
                    c += np.expand_dims(b, axis=tuple(np.arange(c.ndim - b.ndim)+b.ndim))
                else:
                    c += b
                c = c.reshape(-1)

            return SparseImageStar2DCSR(c, V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
        
        else:
            return self.flatten_affineMap(W, b)
    
    def flatten_affineMap(self, W=None, b=None):

        # assert len(self.shape) == 1, 'error: SparseImageStar is not flattened. It has shape of {}'.format(self.shape)
        # # assert isinstance(self.V, np.ndarray), 'error: basis and anchor images of SparseImageStar is not np.ndarray'
        
        if W is None and b is None:
            return copy.deepcopy(self)
        
        if isinstance(self.V, np.ndarray):
            dense = True
        else:
            dense = False

        V = copy.deepcopy(self.V)
        shape_prod = np.prod(self.shape)

        if W is not None:
            assert isinstance(W, np.ndarray), 'error: ' + \
            'the mapping matrix should be a 2D numpy array'
            assert W.shape[1] == shape_prod, 'error: ' + \
            'inconsistency between mapping matrix and SparseImageStar dimension, W.shape[1]={} and shape_prod={}'.format(W.shape[1], shape_prod)

            V = W @ V

        if b is not None:
            assert isinstance(b, np.ndarray), 'error: ' + \
            'the offset vector should be a 1D numpy array'
            assert len(b.shape) == 1, 'error: ' + \
            'offset vector should be a 1D numpy array'

            if W is not None:
                assert W.shape[0] == b.shape[0], 'error: ' + \
                'inconsistency between mapping matrix and offset'
            else:
                assert b.shape[0] == shape_prod, 'error: ' + \
                'inconsistency between offset vector and SparseStar dimension'

            if dense:
                V[:, 0] += b
            
            else:
                if W is not None:
                    c = W @ self.c + b
                else:
                    c = self.c + b
                basis = V.toarray() if sp.issparse(V) else V
                V = np.hstack([c[:, None], basis])
        
        else:
            if not dense:
                basis = V.toarray() if sp.issparse(V) else V
                V = np.hstack([self.c[:, None], basis])

        out_shape = (V.shape[0], )
        return SparseImageStar2DCSR(V, self.C, self.d, self.pred_lb, self.pred_ub, out_shape)
        
    def flatten_affineMap_sparse(self, W=None, b=None):

        if W is None and b is None:
            return copy.deepcopy(self)

        if W is not None:
            assert isinstance(W, np.ndarray), 'error: ' + \
            'the mapping matrix should be a 2D numpy array'
            assert W.shape[1] == self.V.shape[0], 'error: ' + \
            'inconsistency between mapping matrix and SparseImageStar dimension'

            V = W @ self.V.toarray()
            c = np.matmul(W, self.c)
        else:
            V = self.V.toarray()
            c = self.c.copy()

        if b is not None:
            assert isinstance(b, np.ndarray), 'error: ' + \
            'the offset vector should be a 1D numpy array'
            assert len(b.shape) == 1, 'error: ' + \
            'offset vector should be a 1D numpy array'

            if W is not None:
                assert W.shape[0] == b.shape[0], 'error: ' + \
                'inconsistency between mapping matrix and offset'
            else:
                assert b.shape[0] == self.V.shape[0], 'error: ' + \
                'inconsistency between offset vector and SparseStar dimension'

            c += b

        V = sp.csr_array(V)
        V.indices = V.indices.astype(np.int32)
        V.indptr = V.indptr.astype(np.int32) 
        return SparseImageStar2DCSR(c, V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
    
    def estimateRange(self, index):
        """Quickly estimate minimum value of a state x[index]"""

        assert index >= 0 and index < self.V.shape[0], 'error: invalid index'
    
        l = self.pred_lb
        u = self.pred_ub

        if isinstance(self.V, np.ndarray):
            X = self.V[index, 1:]
            pos_f = np.maximum(X, 0)
            neg_f = np.minimum(X, 0)

            xmin = self.V[index, 0] + pos_f @ l + neg_f @ u
            xmax = self.V[index, 0] + neg_f @ l + pos_f @ u
        else:
            X = self.V[index, :]
            pos_f = X.maximum(0.0)
            neg_f = X.minimum(0.0)

            xmin = self.c[index] + pos_f @ l + neg_f @ u
            xmax = self.c[index] + neg_f @ l + pos_f @ u

        return xmin, xmax

    def estimateRanges(self):
        """Estimate the lower and upper bounds of x"""
        l = self.pred_lb
        u = self.pred_ub

        if isinstance(self.V, np.ndarray):
            X = self.V[:, 1:]
            pos_f = np.maximum(X, 0)
            neg_f = np.minimum(X, 0)

            xmin = self.V[:, 0] + pos_f @ l + neg_f @ u
            xmax = self.V[:, 0] + neg_f @ l + pos_f @ u

        else:
            pos_f = self.V.maximum(0)
            neg_f = self.V.minimum(0)

            xmin = self.c + pos_f @ l + neg_f @ u
            xmax = self.c + neg_f @ l + pos_f @ u
        
        return xmin, xmax

    def get_binary_predicate_indices(self):
        # Return flat indices of binary predicate variables (ReLU big-M "a/z" blocks).
        pred_layout = getattr(self, 'pred_layout', None)
        if pred_layout is None:
            return np.empty(0, dtype=np.int32)

        a_blocks = getattr(pred_layout, 'a_blocks', None)
        if not a_blocks:
            return np.empty(0, dtype=np.int32)

        idx_parts = [
            np.arange(start, start + block_size, dtype=np.int32)
            for start, block_size in a_blocks
            if block_size > 0
        ]
        if len(idx_parts) == 0:
            return np.empty(0, dtype=np.int32)

        idx = np.concatenate(idx_parts)
        if ((idx < 0) | (idx >= self.num_pred)).any():
            raise ValueError(
                'error: pred_layout contains binary indices outside valid range [0, {})'.format(self.num_pred)
            )
        return np.unique(idx)
    
    def get_objective_and_center(self, index):
        assert index >= 0 and index < self.V.shape[0], 'error: invalid index'

        if isinstance(self.V, np.ndarray):
            f = self.V[index, 1:]
            center = self.V[index, 0]
            if (f == 0).all():
                return None, center
            return np.asarray(f).reshape(-1), center

        f = self.V[[index]]
        center = self.c[index]
        if f.nnz == 0:
            return None, center
        return np.asarray(f.toarray()).reshape(-1), center

    def get_lp_ub(self):
        # Return A, b for the linear constraint A @ x <= b. If there is no constraint, return empty A and b.
        if len(self.d) == 0:
            return sp.csr_array((1, self.num_pred)), np.zeros(1)
        return self.C, self.d

    def add_gurobi_pred_vars(self, model):
        binary_idx = self.get_binary_predicate_indices()
        if binary_idx.size > 0:
            vtype = [GRB.CONTINUOUS] * self.num_pred
            for idx in binary_idx.tolist():
                vtype[idx] = GRB.BINARY
        else:
            vtype = GRB.CONTINUOUS

        if self.pred_lb.size and self.pred_ub.size:
            return model.addMVar(shape=self.num_pred, lb=self.pred_lb, ub=self.pred_ub, vtype=vtype)
        return model.addMVar(shape=self.num_pred, vtype=vtype)

    def solve_index_lp(self, index, sense='min', lp_solver='gurobi', solver_opts=None, model_pack=None, show=False):
        return util_solve_index_lp(self, index=index, sense=sense, lp_solver=lp_solver, 
            solver_opts=solver_opts, model_pack=model_pack, show=show)

    def getMin(self, index, lp_solver='gurobi', solver_opts=None, gurobi_model_pack=None, show=False):
        """get exact minimum value of state x[index] by solving LP
           lp_solver = 'gurobi', 'cupdlp', 'cupdlp-gurobi', 'scipy-milp', 'linprog', or 'glpk'
        """
        assert index >= 0 and index < self.V.shape[0], 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'
        return self.solve_index_lp(index=index, sense='min', lp_solver=lp_solver,
            solver_opts=solver_opts, model_pack=gurobi_model_pack, show=show)
    
    def getMax(self, index, lp_solver='gurobi', solver_opts=None, gurobi_model_pack=None, show=False):
        """get exact maximum value of state x[index] by solving LP
           lp_solver = 'gurobi', 'cupdlp', 'cupdlp-gurobi', 'scipy-milp', 'linprog', or 'glpk'
        """
        assert index >= 0 and index < self.V.shape[0], 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'
        return self.solve_index_lp(index=index, sense='max', lp_solver=lp_solver,
            solver_opts=solver_opts, model_pack=gurobi_model_pack, show=show)

    def getMins(self, map, lp_solver='gurobi', solver_opts=None, show=False):
        n = len(map)
        xmin = np.zeros(n, dtype=self.V.dtype)
        for i in range(n):
            xmin[i] = self.getMin(map[i], lp_solver=lp_solver, solver_opts=solver_opts, show=show)
        return xmin

    def getMaxs(self, map, lp_solver='gurobi', solver_opts=None, show=False):
        n = len(map)
        xmax = np.zeros(n, dtype=self.V.dtype)
        for i in range(n):
            xmax[i] = self.getMax(map[i], lp_solver=lp_solver, solver_opts=solver_opts, show=show)
        return xmax

    def getMins_all(self, lp_solver='gurobi', solver_opts=None, show=False):
        n = self.V.shape[0]
        xmin = np.zeros(n, dtype=self.V.dtype)
        for i in range(n):
            xmin[i] = self.getMin(i, lp_solver=lp_solver, solver_opts=solver_opts, show=show)
        return xmin

    def getMaxs_all(self, lp_solver='gurobi', solver_opts=None, show=False):
        n = self.V.shape[0]
        xmax = np.zeros(n, dtype=self.V.dtype)
        for i in range(n):
            xmax[i] = self.getMax(i, lp_solver=lp_solver, solver_opts=solver_opts, show=show)
        return xmax

    def getRange(self, index, lp_solver='gurobi', solver_opts=None, show=False):
        """Get the lower and upper bounds of x[index]."""
        if lp_solver == 'estimate':
            return self.estimateRange(index)
        l = self.getMin(index, lp_solver=lp_solver, solver_opts=solver_opts, show=show)
        u = self.getMax(index, lp_solver=lp_solver, solver_opts=solver_opts, show=show)
        return l, u

    def getRanges(self, lp_solver='gurobi', RF=0.0, layer=None, delta=0.98, solver_opts=None, show=False):
        """Get the lower and upper bound vectors of the state
            Args:
                lp_solver: one of 'gurobi', 'cupdlp', 'cupdlp-gurobi', 'scipy-milp', 'estimate', 'linprog', 'glpk'
        """

        if lp_solver == 'estimate':
            l, u = self.estimateRanges()
        else:
            l = self.getMins_all(lp_solver=lp_solver, solver_opts=solver_opts, show=show)
            u = self.getMaxs_all(lp_solver=lp_solver, solver_opts=solver_opts, show=show)
        return l, u
    
    def isEmptySet(self, lp_solver='gurobi'):
        """Check if a SparseStar is an empty set"""
        res = False
        try:
            self.getMin(0, lp_solver)
        except Exception:
            res = True
        return res
    
    def geNumAttackedPixels(self):
        """Esimate the number of attacked pixels"""
        V = self.V.toarray().reshape(self.shape + (self.num_pred, )) != 0
        return np.any(V, axis=-1).sum()

    @staticmethod
    def block_diag_csr(A, B, A_shape=None, B_shape=None):
        """ Create a block diagonal sparse matrix from two sparse matrices A and B """
        assert sp.issparse(A) and sp.issparse(B), 'error: A and B should be sparse matrices'
        assert A.format == 'csr' and B.format == 'csr', 'error: A and B should be in CSR format'

        new_data = np.hstack([A.data, B.data])
        new_indices = np.hstack([A.indices, B.indices+ A_shape[1]])
        new_indptr = np.hstack([A.indptr, B.indptr[1:] + A.nnz])
        return sp.csr_array((new_data, new_indices, new_indptr), shape=(A_shape[0]+B_shape[0], A_shape[1]+B_shape[1]))
    
    @staticmethod
    def hstack_csr(A, B, A_shape=None, B_shape=None):
        """ Create a horizontal stack sparse matrix from two sparse matrices A and B """
        assert sp.issparse(A) and sp.issparse(B), 'error: A and B should be sparse matrices'
        assert A.format == 'csr' and B.format == 'csr', 'error: A and B should be in CSR format'
        assert A_shape[0] == B_shape[0], 'error: number of rows of A and B do not match'

        if A.nnz == 0 and B.nnz == 0:
            return sp.csr_array((0, A_shape[1]+B_shape[1]))
        elif A.nnz == 0:
            new_indices = B.indices + A_shape[1]
            return sp.csr_array((B.data, new_indices, B.indptr), shape=(A_shape[0], A_shape[1]+B_shape[1]))
        elif B.nnz == 0:
            return sp.csr_array((A.data, A.indices, A.indptr), shape=(A_shape[0], A_shape[1]+B_shape[1]))

        row_cnt = np.diff(A.indptr) + np.diff(B.indptr)
        indptr = np.zeros(A_shape[0]+1, dtype=A.indptr.dtype)
        indptr[1:] = np.cumsum(row_cnt)
        nnz = indptr[-1]

        dtype = np.result_type(A.data.dtype, B.data.dtype)
        data = np.empty(nnz, dtype=dtype)
        indices = np.empty(nnz, dtype=A.indices.dtype)

        for i in range(A_shape[0]):
            a_b, a_e = A.indptr[i], A.indptr[i+1]     # beginning and end of row i in A
            b_b, b_e = B.indptr[i], B.indptr[i+1]     # beginning and end of row i in B
            da, db = a_e - a_b, b_e - b_b

            s = indptr[i] # starting index of row i in new matrix
            # if there is data in A i-th row, copy data and indices from A
            if da:
                data[s:s + da] = A.data[a_b:a_e]
                indices[s:s + da] = A.indices[a_b:a_e]

            # if there is data in B i-th row, copy data and indices from B
            if db:
                data[s + da:indptr[i+1]] = B.data[b_b:b_e]
                indices[s + da:indptr[i+1]] = B.indices[b_b:b_e] + A_shape[1]

        return sp.csr_array((data, indices, indptr), shape=(A_shape[0], A_shape[1]+B_shape[1]))
    
    @staticmethod
    def vstack_csr(A, B, A_shape=None, B_shape=None):
        """ Create a vertical stack sparse matrix from two sparse matrices A and B """
        assert sp.issparse(A) and sp.issparse(B), 'error: A and B should be sparse matrices'
        assert A.format == 'csr' and B.format == 'csr', 'error: A and B should be in CSR format'

        if A_shape is None:
            A_shape = A.shape
        if B_shape is None:
            B_shape = B.shape
        assert A_shape[1] == B_shape[1], 'error: number of columns of A and B do not match'

        if A.nnz == 0 and B.nnz == 0:
            return sp.csr_array((0, A_shape[1]))
        if A.nnz == 0:
            return sp.csr_array((B.data, B.indices, B.indptr), shape=(A_shape[0]+B_shape[0], A_shape[1]), dtype=B.dtype)
        if B.nnz == 0:
            return sp.csr_array((A.data, A.indices, A.indptr), shape=(A_shape[0]+B_shape[0], A_shape[1]), dtype=A.dtype)

        indptr = np.hstack([A.indptr, B.indptr[1:] + A.nnz])
        data = np.hstack([A.data, B.data])
        indices = np.hstack([A.indices, B.indices])
        return sp.csr_array((data, indices, indptr), shape=(A_shape[0] + B_shape[0], A_shape[1]))

    def get_max_point_cadidates(self):
        """ Quickly estimate max-point candidates """

        lb, ub = self.getRanges('estimate')
        max_id = np.argmax(lb)
        a = (ub >= lb[max_id])
        if sum(a) == 1:
            return [max_id]
        else:
            return np.where(a)[0]
        
    def concatenate(self, X, axis=0):
        """ Concatenate two SparseImageStar2DCSR sets """
        assert isinstance(X, SparseImageStar2DCSR), 'error: X should be a SparseImageStar2DCSR instance'
        # assert self.shape == X.shape, 'error: shapes of the two sets do not match'
        
        # shapes must match on all axes except the concatenation axis
        # shape_self = list(self.shape)
        # shape_X = list(X.shape)  
        # del shape_self[axis]
        # del shape_X[axis]
        # assert shape_self == shape_X, \
        #     f'error: shapes {self.shape} and {X.shape} do not match on non-concatenation axes (axis={axis})'

        if self.C.nnz > 0 and X.C.nnz > 0:
            new_C = SparseImageStar2DCSR.block_diag_csr(A=self.C, B=X.C, A_shape=self.C.shape, B_shape=X.C.shape)  

        elif self.C.nnz > 0 and X.C.nnz == 0:
            new_C = self.C.copy()
            new_C._shape = (self.C.shape[0], self.num_pred + X.num_pred)

        elif X.C.nnz > 0 and self.C.nnz == 0:
            new_C = X.C.copy()
            new_C._shape = (X.C.shape[0], self.num_pred + X.num_pred)

        else:
            new_C = sp.csr_array((0, self.num_pred + X.num_pred))
            
        new_d = np.concatenate([self.d, X.d])

        new_pred_lb = np.concatenate([self.pred_lb, X.pred_lb])
        new_pred_ub = np.concatenate([self.pred_ub, X.pred_ub])

        # V is in dense format
        if self.c is None:
            sc, xc = self.V[:, :, :, 0], X.V[:, :, :, 0]
            c = np.concatenate([sc, xc], axis=axis)[:, :, :, None]
            A = SparseImageStar2DCSR.block_diag_axis_dense(self.V[:, :, :, 1:], X.V[:, :, :, 1:], axis=axis)
            new_V = np.concatenate((c, A), axis=3)
            new_shape = new_V.shape[:3]
            return SparseImageStar2DCSR(new_V, new_C, new_d, new_pred_lb, new_pred_ub, new_shape, copy_=False)
        
        # V is in sparse format
        else:
            new_c = np.concatenate([self.c.reshape(self.shape), X.c.reshape(X.shape)], axis=axis).reshape(-1)
            new_V, new_shape = SparseImageStar2DCSR.block_diag_axis_csr(A=self.V, B=X.V, shapeA=self.shape, shapeB=X.shape, axis=axis)
            return SparseImageStar2DCSR(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub, new_shape, copy_=False)

    def block_diag_axis_csr(A, B, shapeA, shapeB, axis=0):
        """
        Block-diagonal combination of A and B in 4D (flattened to 2D CSR),
        along the specified axis (0, 1, or 2).

        A, B : csr_array
            A.shape = (hA * wA * cA, m1)
            B.shape = (hB * wB * cB, m2)

        shapeA : (hA, wA, cA)
        shapeB : (hB, wB, cB)

        axis ∈ {0, 1, 2}:
        - axis = 0: block diag in (0, 3)
            A: (hA, w,  c, m1),  B: (hB, w,  c, m2)
            out: (hA + hB, w, c, m1 + m2)

        - axis = 1: block diag in (1, 3)
            A: (h,  wA, c, m1),  B: (h,  wB, c, m2)
            out: (h, wA + wB, c, m1 + m2)

        - axis = 2: block diag in (2, 3)
            A: (h, w,  cA, m1), B: (h, w,  cB, m2)
            out: (h, w, cA + cB, m1 + m2)


        Returned matrix:
            csr_array with shape (h_out * w_out * c_out, m1 + m2)
            using C-like (row-major) flattening of (h, w, c).
        """
        assert axis in (0, 1, 2)
        # Accept both csr_matrix and csr_array
        assert sp.issparse(A) and sp.issparse(B), "A and B must be sparse matrices"
        assert A.format == 'csr' and B.format == 'csr', "A and B must be in CSR format"

        # Ensure CSR (for both csr_matrix and csr_array this is cheap / no-op if already CSR)
        hA, wA, cA = shapeA
        hB, wB, cB = shapeB
        m1 = A.shape[1]
        m2 = B.shape[1]

        # Dtype handling
        dtype_data = np.result_type(A.data.dtype, B.data.dtype)
        dtype_ind = np.result_type(A.indices.dtype, B.indices.dtype)
        dtype_ptr = np.result_type(A.indptr.dtype, B.indptr.dtype)

        if axis == 0:
            # A: (hA, w, c, m1), B: (hB, w, c, m2)
            assert wA == wB and cA == cB, "w and c must match for axis=0"
            nrowsA = hA * wA * cA
            nrowsB = hB * wB * cB
            assert A.shape[0] == nrowsA, 'error: unexpected number of rows in A'
            assert B.shape[0] == nrowsB, 'error: unexpected number of rows in B'

            out_shape = (hA + hB, wA, cA)
            nrows_out = nrowsA + nrowsB
            nnzA = A.nnz
            nnzB = B.nnz

            data_out    = np.zeros(nnzA + nnzB, dtype=dtype_data)
            indices_out = np.zeros(nnzA + nnzB, dtype=dtype_ind)
            indptr_out  = np.zeros(nrows_out + 1, dtype=dtype_ptr)

            # A part (rows 0..nrowsA-1, cols 0..m1-1)
            data_out[:nnzA] = A.data.astype(dtype_data, copy=False)
            indices_out[:nnzA] = A.indices.astype(dtype_ind, copy=False)
            indptr_out[:nrowsA+1] = A.indptr.astype(dtype_ptr, copy=False)

            # B part (rows nrowsA.., cols shifted by +m1)
            data_out[nnzA:]    = B.data.astype(dtype_data, copy=False)
            indices_out[nnzA:] = B.indices.astype(dtype_ind, copy=False) + m1
            indptr_out[nrowsA:] = B.indptr.astype(dtype_ptr, copy=False) + nnzA

            return sp.csr_array((data_out, indices_out, indptr_out), shape=(nrows_out, m1 + m2)), out_shape

        if axis == 1:
            # A: (h, wA, c, m1), B: (h, wB, c, m2)
            assert hA == hB and cA == cB, "h and c must match for axis=1"
            H = hA
            W1, W2 = wA, wB
            C = cA
            nrowsA = H * W1 * C
            nrowsB = H * W2 * C
            assert A.shape[0] == nrowsA, 'error: unexpected number of rows in A'
            assert B.shape[0] == nrowsB, 'error: unexpected number of rows in B'

            out_shape = (H, W1 + W2, C)
            nrows_out = H * (W1 + W2) * C
            nnz = A.nnz + B.nnz

            data_out    = np.zeros(nnz, dtype=dtype_data)
            indices_out = np.zeros(nnz, dtype=dtype_ind)
            indptr_out  = np.zeros(nrows_out + 1, dtype=dtype_ptr)

            pos = 0
            row_out = 0
            indptr_out[0] = 0

            # Output order: for each h:
            #   - all (w=0..W1-1, c=0..C-1) from A
            #   - all (w=0..W2-1, c=0..C-1) from B
            for h in range(H):
                # A width block
                for w in range(W1):
                    baseA = (h * W1 + w) * C
                    for c in range(C):
                        rA = baseA + c
                        start = A.indptr[rA]
                        end   = A.indptr[rA + 1]
                        length = end - start
                        if length:
                            data_out[pos:pos+length]    = A.data[start:end]
                            indices_out[pos:pos+length] = A.indices[start:end]
                            pos += length
                        row_out += 1
                        indptr_out[row_out] = pos
                # B width block
                for w in range(W2):
                    baseB = (h * W2 + w) * C
                    for c in range(C):
                        rB = baseB + c
                        start = B.indptr[rB]
                        end   = B.indptr[rB + 1]
                        length = end - start
                        if length:
                            data_out[pos:pos+length]    = B.data[start:end]
                            indices_out[pos:pos+length] = B.indices[start:end] + m1
                            pos += length
                        row_out += 1
                        indptr_out[row_out] = pos

            assert row_out == nrows_out
            assert pos == nnz

            return sp.csr_array((data_out, indices_out, indptr_out), shape=(nrows_out, m1 + m2)), out_shape

        else:
            # A: (h, w, cA, m1), B: (h, w, cB, m2)
            # axis == 2
            assert hA == hB and wA == wB, "h and w must match for axis=2"
            H = hA
            W = wA
            C1, C2 = cA, cB
            nrowsA = H * W * C1
            nrowsB = H * W * C2
            assert A.shape[0] == nrowsA, 'error: unexpected number of rows in A'
            assert B.shape[0] == nrowsB, 'error: unexpected number of rows in B'

            out_shape = (H, W, C1 + C2)
            nrows_out = H * W * (C1 + C2)
            nnz =  A.nnz + B.nnz

            data_out    = np.zeros(nnz, dtype=dtype_data)
            indices_out = np.zeros(nnz, dtype=dtype_ind)
            indptr_out  = np.zeros(nrows_out + 1, dtype=dtype_ptr)

            pos = 0
            row_out = 0
            indptr_out[0] = 0

            # For each spatial (h,w) (flattened as hw), take all c from A then all c from B
            for hw in range(H * W):
                baseA = hw * C1
                for c in range(C1):
                    rA = baseA + c
                    start = A.indptr[rA]
                    end   = A.indptr[rA + 1]
                    length = end - start
                    if length:
                        data_out[pos:pos+length]    = A.data[start:end]
                        indices_out[pos:pos+length] = A.indices[start:end]
                        pos += length
                    row_out += 1
                    indptr_out[row_out] = pos

                baseB = hw * C2
                for c in range(C2):
                    rB = baseB + c
                    start = B.indptr[rB]
                    end   = B.indptr[rB + 1]
                    length = end - start
                    if length:
                        data_out[pos:pos+length]    = B.data[start:end]
                        indices_out[pos:pos+length] = B.indices[start:end] + m1
                        pos += length
                    row_out += 1
                    indptr_out[row_out] = pos

            assert row_out == nrows_out
            assert pos == nnz

            return sp.csr_array((data_out, indices_out, indptr_out), shape=(nrows_out, m1 + m2)), out_shape
    
    @staticmethod
    def block_diag_axis_dense(A, B, axis=0):
        """
        Block diagonal concatenation along a specified axis
        A, B: input 4D numpy arrays with shape (h, w, c, num_pred)
        axis in {0, 1, 2}:
            - axis 0: vertical concatenation:       (h, :, :, m); block diag in (0, 3)
            - axis 1: horizontal concatenation:     (:, w, :, m); block diag in (1, 3)
            - axis 2: channel-wise concatenation:   (:, :, c, m); block diag in (2, 3)
        return:
            new_V: the concatenated 4D numpy array
        """
        assert A.ndim == 4 and B.ndim == 4
        assert axis in (0, 1, 2)

        h1, w1, c1, m1 = A.shape
        h2, w2, c2, m2 = B.shape

        if axis == 0:
            # A: (h1, w, c, m1), B: (h2, w, c, m2)
            assert w1 == w2 and c1 == c2, "w and c must match for axis=0"

            out = np.zeros((h1 + h2, w1, c1, m1 + m2),
                        dtype=np.result_type(A, B))
            out[:h1, :, :, :m1] = A
            out[h1:, :, :, m1:] = B

        elif axis == 1:
            # A: (h, w1, c, m1), B: (h, w2, c, m2)
            assert h1 == h2 and c1 == c2, "h and c must match for axis=1"

            out = np.zeros((h1, w1 + w2, c1, m1 + m2),
                        dtype=np.result_type(A, B))
            out[:, :w1, :, :m1] = A
            out[:, w1:, :, m1:] = B

        else:  # axis == 2
            # A: (h, w, c1, m1), B: (h, w, c2, m2)
            assert h1 == h2 and w1 == w2, "h and w must match for axis=2"

            out = np.zeros((h1, w1, c1 + c2, m1 + m2),
                        dtype=np.result_type(A, B))
            out[:, :, :c1, :m1] = A
            out[:, :, c1:,  m1:] = B

        return out
    
    # @staticmethod
    # def block_diag_concateante3d_dense(A, B, axis=0):
    #     """
    #         Block diagonal concatenate two dense arrays that represents 3D array (tensor) along the specified axis
        
    #         Args:
    #             @A: first dense array
    #             @B: second dense array
    #             @axis: axis along which to concatenate

    #         Return:
    #             @C: concatenated dense array
    #     """
    #     assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), 'error: A and B should be numpy arrays'
    #     assert len(A.shape) == 4 and len(B.shape) == 4, 'error: A and B are not 4D arrays'
    #     assert axis >=0 and axis < 3, 'error: axis should be in [0, 2]'
        
    #     out_shape = list(A.shape)   # in (h, w, ch, m) shape
    #     out_shape[axis] = out_shape[axis] + B.shape[axis]

    #     new_V = np.zeros((out_shape[0], out_shape[1], out_shape[2], A.num_pred + 1), dtype=A.V.dtype)
        
    #     if axis == 0:
    #         new_V[0:A.height, :, :, :] = A.V
    #         new_V[A.height:out_shape[0], :, :, :] = B.V
    #     elif axis == 1:
    #         new_V[:, 0:A.width, :, :] = A.V
    #         new_V[:, A.width:out_shape[1], :, :] = B.V
    #     elif axis == 2:
    #         new_V[:, :, 0:A.num_channel, :] = A.V
    #         new_V[:, :, A.num_channel:out_shape[2], :] = B.V
    #     else:
    #         raise Exception('error: invalid axis to concatenate')
        
    #     return new_V

    # @staticmethod
    # def block_diag_concatenate3d_csr(A, A_shape, B, B_shape, axis=0):
    #     """
    #         Concatenate two csr arrays that represents 3D array (tensor) along the specified axis
        
    #         Args:
    #             @A: first csr array
    #             @A_shape: shape of first csr array
    #             @B: second csr array
    #             @B_shape: shape of second csr array
    #             @axis: axis along which to concatenate

    #         Return:
    #             @C: concatenated csr array
    #             @C_shape: shape of concatenated csr array
    #     """
    #     assert sp.issparse(A) and sp.issparse(B), 'error: A and B should be sparse matrices'
    #     assert A.format == 'csr' and B.format == 'csr', 'error: A and B should be in CSR format'
    #     # assert A.shape == B.shape, 'error: shapes of A and B do not match'
    #     # assert A_shape == B.shape, 'error: 4D shapes of A and B do not match'
    #     assert axis >=0 and axis < 3, 'error: axis should be in [0, 2]'
    #     assert len(A_shape) == 3 and len(B_shape) == 3, 'error: A_shape and B_shape are not 3D shape'
    #     # assert np.prod(A_shape) == A.shape[0]*A.shape[1], 'error: inconsistency between (A_shape and A.shape) and (B_shape and B.shape)'
        
    #     ah, aw, ac = A_shape
    #     bh, bw, bc = B_shape
    #     m = A.shape[1] + B.shape[1]

    #     if axis == 0:
    #         assert aw == bw and ac == bc, 'error: shapes of A and B do not match for concatenation along axis 0'
    #         new_C = SparseImageStar2DCSR.block_diag_csr(A=A, B=B, A_shape=A.shape, B_shape=B.shape)
    #         new_shape = (ah + bh, aw, ac)

    #     elif axis == 1:
    #         assert ah == bh and ac == bc, 'error: shapes of A and B do not match for concatenation along axis 1'
            
    #         rcA, rcB = np.diff(A.indptr), np.diff(B.indptr) 

    #         A_blk = aw * ac # number of rows in A block
    #         B_blk = bw * ac # number of rows in B block

    #         out_blk = (aw + bw) * ac # number of rows in output block

    #         # build new_indptr by composing row counts block-wise
    #         row_counts = np.zeros(ah * out_blk, dtype=np.int64) 

    #         for r in range(ah): 
    #             dA = r * out_blk 
    #             sA = r * A_blk 
    #             row_counts[dA : dA + A_blk] = rcA[sA : sA + A_blk] 

    #             dB = dA + A_blk 
    #             sB = r * B_blk 
    #             row_counts[dB : dB + B_blk] = rcB[sB : sB + B_blk] 

    #         new_indptr = np.zeros(row_counts.size + 1, dtype=A.indptr.dtype) 
    #         np.cumsum(row_counts, out=new_indptr[1:]) 

    #         nnz_out = int(new_indptr[-1]) 
    #         new_data = np.zeros(nnz_out, dtype=np.result_type(A.data.dtype, B.data.dtype)) 
    #         new_indices = np.zeros(nnz_out, dtype=A.indices.dtype) 

    #         # copy contiguous blocks
    #         for r in range(ah): 
    #             # A block 
    #             sA_rows = r * A_blk 
    #             eA_rows = sA_rows + A_blk 

    #             sA_data = A.indptr[sA_rows] 
    #             eA_data = A.indptr[eA_rows] 

    #             dA_rows = r * out_blk 

    #             new_data[new_indptr[dA_rows] : new_indptr[dA_rows + A_blk]] = A.data[sA_data:eA_data] 
    #             new_indices[new_indptr[dA_rows] : new_indptr[dA_rows + A_blk]] = A.indices[sA_data:eA_data] 
    
    #             # B block 
    #             sB_rows = r * B_blk 
    #             eB_rows = sB_rows + B_blk 

    #             sB_data = B.indptr[sB_rows] 
    #             eB_data = B.indptr[eB_rows] 

    #             dB_rows = dA_rows + A_blk 

    #             new_data[new_indptr[dB_rows] : new_indptr[dB_rows + B_blk]] = B.data[sB_data:eB_data] 
    #             new_indices[new_indptr[dB_rows] : new_indptr[dB_rows + B_blk]] = B.indices[sB_data:eB_data] 
    
    #         new_shape = (ah, aw + bw, ac)
    #         flatten_shape = np.prod(new_shape)
    #         new_C = sp.csr_array((new_data, new_indices, new_indptr), shape=(flatten_shape, m)) 

    #     elif axis == 2:
    #         assert ah == bh and aw == bw, 'error: shapes of A and B do not match for concatenation along axis 2'
            
    #         rcA, rcB = np.diff(A.indptr), np.diff(B.indptr) 

    #         rows_out = ah * aw * (ac + bc) 
    #         row_counts = np.empty(rows_out, dtype=np.int64) 


    #         # per (r,w), append A's ac rows then B's bc rows 
    #         for r in range(ah): 
    #             for w in range(aw): 

    #                 # source row ranges 
    #                 a_row0 = (r*aw + w) * ac 
    #                 b_row0 = (r*aw + w) * bc 

    #                 # destination row base 
    #                 d_row0 = (r*aw + w) * (ac + bc) 

    #                 # copy row counts
    #                 row_counts[d_row0 : d_row0 + ac] = rcA[a_row0 : a_row0 + ac] 
    #                 row_counts[d_row0 + ac : d_row0 + ac + bc] = rcB[b_row0 : b_row0 + bc] 

    #         new_indptr = np.zeros(rows_out + 1, dtype=A.indptr.dtype) 
    #         np.cumsum(row_counts, out=new_indptr[1:]) 

    #         nnz_out = int(new_indptr[-1]) 
    #         new_data = np.zeros(nnz_out, dtype=np.result_type(A.data.dtype, B.data.dtype)) 
    #         new_indices = np.zeros(nnz_out, dtype=np.int32)

    #         # copy contiguous blocks per (r,w) 
    #         for r in range(ah): 
    #             for w in range(aw): 
    #                 # A block 
    #                 a_r0 = (r*aw + w) * ac 
    #                 a_r1 = a_r0 + ac 

    #                 a_d0 = A.indptr[a_r0]
    #                 a_d1 = A.indptr[a_r1]
    #                 d_r0 = (r*aw + w) * (ac + bc) 

    #                 new_data[new_indptr[d_r0] : new_indptr[d_r0 + ac]] = A.data[a_d0:a_d1] 
    #                 new_indices[new_indptr[d_r0] : new_indptr[d_r0 + ac]] = A.indices[a_d0:a_d1] 

    #                 # B block 
    #                 b_r0 = (r*aw + w) * bc 
    #                 b_r1 = b_r0 + bc 

    #                 b_d0 = B.indptr[b_r0]; b_d1 = B.indptr[b_r1] 
    #                 d_r1 = d_r0 + ac 

    #                 new_data[new_indptr[d_r1] : new_indptr[d_r1 + bc]] = B.data[b_d0:b_d1] 
    #                 new_indices[new_indptr[d_r1] : new_indptr[d_r1 + bc]] = B.indices[b_d0:b_d1] 

    #         new_shape = (ah, aw, ac + bc)
    #         flatten_shape = np.prod(new_shape)
    #         new_C = sp.csr_array((new_data, new_indices, new_indptr), shape=(flatten_shape, m)) 

    #     return new_C, new_shape


    def minKowskiSum(self, X):
        """ Compute the Minkowski sum of two SparseImageStar2DCSR sets """
        assert isinstance(X, SparseImageStar2DCSR), 'error: X should be a SparseImageStar2DCSR instance'
        assert self.shape == X.shape, 'error: shapes of the two sets do not match'

        if self.C.nnz > 0 and X.C.nnz > 0:
            new_C = SparseImageStar2DCSR.block_diag_csr(A=self.C, B=X.C, A_shape=self.C.shape, B_shape=X.C.shape)  

        elif self.C.nnz > 0 and X.C.nnz == 0:
            new_C = self.C.copy()
            new_C._shape = (self.C.shape[0], self.num_pred + X.num_pred)

        elif X.C.nnz > 0 and self.C.nnz == 0:
            new_C = X.C.copy()
            new_C._shape = (X.C.shape[0], self.num_pred + X.num_pred)

        else:
            new_C = sp.csr_array((0, self.num_pred + X.num_pred))
            
        new_d = np.concatenate([self.d, X.d])

        new_pred_lb = np.concatenate([self.pred_lb, X.pred_lb])
        new_pred_ub = np.concatenate([self.pred_ub, X.pred_ub])

        # V is in dense format
        if self.c is None:
            c = self.V[:, 0] + X.V[:, 0]
            A = block_diag(self.V[:, 1:], X.V[:, 1:])
            new_V = np.hstack([c[:, None], A])
            return SparseImageStar2DCSR(new_V, new_C, new_d, new_pred_lb, new_pred_ub, self.shape, copy_=False) 
        
        # V is in sparse format
        else:
            new_c = self.c + X.c
            new_V = SparseImageStar2DCSR.hstack_csr(A=self.V, B=X.V, A_shape=self.V.shape, B_shape=X.V.shape)
            return SparseImageStar2DCSR(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub, self.shape, copy_=False)


    def is_p1_larger_than_p2(self, p1_indx, p2_indx, lp_solver='gurobi'):
        """
            Check if an index is larger than the other

            Arg:
                @p1_indx: an index of point 1
                @p2_indx: an index of point 2

            return:
                @bool = 1 if there exists the case that p1 >= p2
                        2 if there is no case that p1 >= p2; p1 < p2
        """

        assert p1_indx >= 0 and p1_indx < self.V.shape[0], 'error: invalid index for point 1'
        assert p2_indx >= 0 and p2_indx < self.V.shape[0], 'error: invalid index for point 2'

        if isinstance(self.V, np.ndarray):
            d1 = self.V[p1_indx, 0] - self.V[p2_indx, 0]
            C1 = self.V[p2_indx, 1:] - self.V[p1_indx, 1:]

            if self.C.nnz > 0:
                C1 = sp.csr_array(C1[None, :])
                data = np.hstack([self.C.data, C1.data])
                indices = np.hstack([self.C.indices, C1.indices])
                indptr = np.hstack([self.C.indptr, C1.indptr[1:]+self.C.nnz])
                new_C = sp.csr_array((data, indices, indptr), shape=(self.C.shape[0]+C1.shape[0], C1.shape[1]), copy=False)
                new_d = np.hstack([self.d, d1])

            else:
                new_d = np.array([d1])
                new_C = sp.csr_array(C1[None, :])

            SIM = SparseImageStar2DCSR(self.V, new_C, new_d, self.pred_lb, self.pred_ub, self.shape, copy_=False)

        else:
            d1 = self.c[p1_indx] - self.c[p2_indx]
            C1 = self.V[[p2_indx]] - self.V[[p1_indx]]

            if self.C.nnz > 0:
                C1 = sp.csr_array(C1)
                data = np.hstack([self.C.data, C1.data])
                indices = np.hstack([self.C.indices, C1.indices])
                indptr = np.hstack([self.C.indptr, C1.indptr[1:]+self.C.nnz])
                new_C = sp.csr_array((data, indices, indptr), shape=(self.C.shape[0]+C1.shape[0], C1.shape[1]), copy=False)
                new_d = np.hstack([self.d, d1])

            else:
                new_d = np.array([d1])
                new_C = sp.csr_array(C1)
 
            SIM = SparseImageStar2DCSR(self.c, self.V, new_C, new_d, self.pred_lb, self.pred_ub, self.shape, copy_=False)

        if SIM.isEmptySet(lp_solver=lp_solver):
            return False
        else:
            return True
        
    @staticmethod
    def inf_attack(data, epsilon=0.01, data_type=None, dtype = 'float64'):
        """Generate a SparseImageStar set by infinity norm attack on input dataset"""

        if isinstance(data, np.ndarray):
            assert data.ndim == 3, \
            'error: data should be a 3D numpy array in [height, width, channel] shape'
        
        elif isinstance(data, torch.Tensor):
            assert data.ndim == 3, \
            'error: data should be a 3D torch tensor in [channel, height, width] shape'

            data = data.permute(1, 2, 0).numpy()

        else:
            raise Exception('the data should be a 3D numpy array or 3D torch tensor')

        data = data.astype(dtype)

        lb = data - epsilon
        ub = data + epsilon

        if data_type == 'image':
            lb[lb < 0] = 0
            ub[ub > 1] = 1

        return SparseImageStar2DCSR(lb, ub)
    
    @staticmethod
    def csr_one_element_per_row(rows, cols, n_rows, n_cols, dtype=np.float64):
        """
        Build CSR with exactly one nonzero per listed row (rows unique),
        using vectorized indptr.
        Assumes rows are unique. If not unique, fix upstream.
        
        Args:
        -  rows, cols: 1D arrays of equal length, with rows[i], cols[i] giving the position of the nonzero in row i.
        -  n_rows, n_cols: shape of the output matrix
        -  dtype: data type of the output matrix
        
        Returns:
        - csr_array of shape (n_rows, n_cols) with 1.0 at (rows[i], cols[i]) and 0.0 elsewhere.
        """
        # sort by row to satisfy CSR row order
        p = np.argsort(rows)
        rows = rows[p]
        cols = cols[p]

        counts = np.zeros(n_rows, dtype=np.int32)
        counts[rows] = 1
        indptr = np.zeros(n_rows + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])

        data = np.ones(rows.size, dtype=dtype)
        indices = cols.astype(np.int32, copy=False)
        return sp.csr_array((data, indices, indptr), shape=(n_rows, n_cols), dtype=dtype)
    
    @staticmethod
    def csr_eye(m, dtype=np.float64):
        """Identity matrix in CSR format; m is the number of rows and columns"""
        data = np.ones(m, dtype=dtype)
        indices = np.arange(m, dtype=np.int32)
        indptr = np.arange(m + 1, dtype=np.int32)
        return sp.csr_array((data, indices, indptr), shape=(m, m), dtype=dtype)
    
    @staticmethod
    def csr_diag(v, dtype=np.float64):
        """Diagonal matrix with elements from v on the diagonal, in CSR format
            v can be 1D array of length m, or a scalar (in which case it is repeated m times on the diagonal)
        """
        m = v.size
        data = np.array(v, dtype=dtype)
        indices = np.arange(len(v), dtype=np.int32)
        indptr = np.arange(len(v) + 1, dtype=np.int32)
        return sp.csr_array((data, indices, indptr), shape=(m, m), dtype=dtype)

    @staticmethod
    def csr_extend_ncols(C, new_ncols, dtype=np.float64):
        """
        Extend a CSR matrix C to have new_ncols columns by adding zero columns at the end
        Args:
            - C: input csr_array of shape (n_rows, old_ncols)
            - new_ncols: desired number of columns in the output matrix (>= old_ncols)
            - dtype: data type of the output matrix
        Returns:
            - csr_array of shape (n_rows, new_ncols) with the same data as C in the first old_ncols columns, and zeros in the new columns.
        """
        if C is None or C.nnz == 0:
            return sp.csr_array((0, new_ncols), dtype=dtype)
        return sp.csr_array((C.data, C.indices, C.indptr), shape=(C.shape[0], new_ncols), dtype=dtype)

    
# @numba.njit
def shift(arr, num):
    if num >= 0:
        return np.concatenate((np.zeros(num), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.zeros(-num)))
    
# @numba.njit
# def shift(arr, num, fill_value=np.nan):
#     if num >= 0:
#         return np.concatenate((np.full(num, fill_value), arr[:-num]))
#     else:
#         return np.concatenate((arr[-num:], np.full(-num, fill_value)))