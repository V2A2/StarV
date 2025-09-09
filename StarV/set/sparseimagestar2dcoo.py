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
Sung Woo Choi, 12/28/2023

"""

# !/usr/bin/python3
import copy
import torch
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog
from scipy.linalg import block_diag
# import numba
import glpk
import polytope as pc

GUROBI_OPT_TOL = 1e-6

class SparseImageStar2DCOO(object):
    """
        Sparse Image Star for reachability
        author: Sung Woo Choi
        date: 12/28/2023
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
            V = [] @ 2D scipy coo matrix
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

        if len_ == 7:
            
            [c, V, C, d, pred_lb, pred_ub, shape] = copy.deepcopy(args) if copy_ is True else args
                
            # if len(shape) == 2:
            #     assert isinstance(V, sp.coo_array) or isinstance(V, sp.coo_matrix) or \
            #     isinstance(V, np.ndarray), \
            #     'error: generator image should be a numpy ndarray or scipy coo array or matrix'
            #     assert shape == V.shape[0], \
            #     'error: inconsistency between shape and shape of basis vector'
            # else:
            #     assert isinstance(V, sp.coo_array) or isinstance(V, sp.coo_matrix), \
            #     'error: generator image should be a scipy csr array or matrix'
            #     assert np.array(shape).prod() == V.shape[0], \
            #     'error: inconsistency between shape and shape of basis vector'

            assert isinstance(V, sp.coo_array) or isinstance(V, sp.coo_matrix), \
            'error: generator image should be a scipy coo array or matrix but received {}'.format(type(V))
            assert np.array(shape).prod() == V.shape[0], \
            'error: inconsistency between shape and shape of basis vector'

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
            assert (isinstance(shape, np.ndarray) or isinstance(shape, list) or isinstance(shape, tuple)) and len(shape) >= 1 and len(shape) <= 3, \
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

            assert isinstance(V, np.ndarray), 'error: basis matrix should be a numpy array'
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
            row = np.where(gtr)[0].astype(np.int32)
            col = np.arange(nv, dtype=np.int32)
            
            
            self.c = 0.5 * (lb + ub)
            self.V = sp.coo_array(
                        (data, (row, col)), shape=(lb.shape[0], nv)
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
        #     row = np.where(gtr)[0].astype(np.int32)
        #     col = np.arange(nv, dtype=np.int32)
        #     data = np.ones(nv, dtype=dtype)

        #     # self.c = np.zeros(lb.shape[0], dtype=dtype)
        #     self.c = (ub == lb) * lb
        #     self.V = sp.coo_array(
        #                 (data, (row, col)), shape=(lb.shape[0], nv)
        #             )
        #     self.C = sp.csr_array((0, 0), dtype=dtype)
        #     self.d = np.empty([0], dtype=dtype)

        #     self.pred_lb = lb
        #     self.pred_ub = ub
        #     self.num_pred = nv
        #     # self.num_pixel = lb.shape[0]
        
        elif len_ == 0: 
            self.c = np.empty([0, 0, 0])
            self.V = sp.coo_array((0, 0))
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
        print('SparseImageStar2DCOO Set:')
        if self.c is None:
            print(f'V_{self.V.getformat()}: {self.V}')
        else:
            print(f'c: {self.c}')
            if to_dense:
                print(f'V_{self.V.getformat()}: \n{self.V.todense()}')
            elif to_coo:
                print(f'V_{self.V.getformat()}:\ndata: {self.V.data}\n row: {self.V.row}\n col: {self.V.col}')
            else:
                print(f'V_{self.V.getformat()}: \n{self.V}')
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
        print('SparseImageStar2DCOO Set:')
        if self.c is None:
            print('V: {}, {}'.format(self.V.shape, self.V.dtype))
        else:
            print('c: {}, {}'.format(self.c.shape, self.c.dtype))
            print('V_{}: {}, data: {}, row: {}, col: {}'.format(self.V.getformat(), self.V.shape, self.V.data.dtype, self.V.row.dtype, self.V.col.dtype))

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
            return self.V.data.nbytes + self.V.row.nbytes + self.V.col.nbytes + self.c.nbytes
    
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
            return self.V.nnz / (self.V.shape[0] * self.V.shape[1])
    
    # @staticmethod
    # def index_2hwc(shape, index):
    #     # V is in [height, width, channel] order
    #     nh, nw, nc = shape

    #     index = copy.deepcopy(index)
    #     num = nw * nh
    #     c = index // num
    #     index -= c * num
    #     w = index // nh
    #     h = index % nh
    #     return h, w, c
    
    @staticmethod
    def index_2hwc(shape, index):
        # V is in [height, width, channel] order
        nh, nw, nc = shape

        index = copy.deepcopy(index)
        num = nw * nc
        h = index // num
        index -= h * num
        w = index // nc
        c = index % nc
        return h, w, c
    
    @staticmethod
    def hwc_2index(shape, h, w, c):
        nh, nw, nc = shape
        return h * nw * nc + w * nc + c
        # return c * nw * nh + w * nh + h
    
    def resetRow_orig(self, index):
        '''Reset a row with index'''

        V = copy.deepcopy(self.V)
        row = V.row
        col = V.col
        data = V.data

        remove_indx = np.argwhere(row == index)[:,  0].reshape(-1)

        row = np.delete(row, remove_indx)
        col = np.delete(col, remove_indx)
        data = np.delete(data, remove_indx)

        new_V = sp.coo_array(
            (data, (row, col)), shape=(self.V.shape[0], self.V.shape[1])
        )
        new_c = copy.deepcopy(self.c)
        new_c[index] = 0

        return SparseImageStar2DCOO(new_c, new_V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
    
    def resetRow(self, index):
        '''Reset a row with index'''
        if isinstance(self.V, np.ndarray):
            V = copy.deepcopy(self.V)
            V[index, :] = 0
            return SparseImageStar2DCOO(V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
        else:
            c = copy.deepcopy(self.c)
            c[index] = 0
            V = self.resetRow_V(index)
            return SparseImageStar2DCOO(c, V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
    
    def resetRow_V(self, index):
        V = self.V.tocsr()
        n = V.indptr[index+1] - V.indptr[index]

        if n > 0:
            V.data[V.indptr[index]:-n] = V.data[V.indptr[index+1]:]
            V.data = V.data[:-n]
            V.indices[V.indptr[index]:-n] = V.indices[V.indptr[index+1]:]
            V.indices = V.indices[:-n]
        V.indptr[index+1:] -= n
        return V.tocoo(copy=False)

    def resetRows_orig(self, map):
        '''Reset a row with index'''

        V = copy.deepcopy(self.V)
        row = V.row
        col = V.col
        data = V.data

        remove_indx = np.argwhere(row[:, None] == map[None, :])[:,  0].reshape(-1)

        row = np.delete(row, remove_indx)
        col = np.delete(col, remove_indx)
        data = np.delete(data, remove_indx)

        new_V = sp.coo_array(
            (data, (row, col)), shape=(V.shape[0], V.shape[1])
        )
        new_c = copy.deepcopy(self.c)
        new_c[map] = 0

        return SparseImageStar2DCOO(new_c, new_V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
    
    def resetRows(self, map):
        '''Reset a row with index'''
        if isinstance(self.V, np.ndarray):
            V = copy.deepcopy(self.V)
            V[map, :] = 0
            return SparseImageStar2DCOO(V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
        else:
            c = copy.deepcopy(self.c)
            c[map] = 0
            V = self.resetRows_V2(map)
            return SparseImageStar2DCOO(c, V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
    
    def resetRows_V_orig(self, map):
        '''Reset a row with index'''

        V = copy.deepcopy(self.V)
        row = V.row
        col = V.col
        data = V.data

        remove_indx = np.argwhere(row[:, None] == map[None, :])[:,  0].reshape(-1)

        row = np.delete(row, remove_indx)
        col = np.delete(col, remove_indx)
        data = np.delete(data, remove_indx)
        
        return sp.coo_array(
            (data, (row, col)), shape=(V.shape[0], V.shape[1])
        )
    
    # def resetRows_V_memory_issue(self, map):
    #     '''Reset a row with index'''

    #     V = self.V.tocsr()

    #     n = V.indptr[1:] - V.indptr[:-1]
    #     b = np.repeat(n[map][:, None], V.shape[0]+1, axis=1)
    #     for i, m in enumerate(map):
    #         b[i, :] = shift(b[i, :], m+1)
    #     new_indptr = V.indptr - b.sum(axis=0)    

    #     mask = np.ones(V.shape[0], dtype=bool)
    #     mask[map] = False
    #     V = V[mask]
    #     V.indptr = new_indptr
    #     V._shape = self.V.shape
    #     return V.tocoo(copy=False)
    
    def resetRows_V(self, map):
        
        V = self.V.tocsr()
        
        n = (V.indptr[1:] - V.indptr[:-1]).astype(np.uint32)
        new_indptr = copy.deepcopy(V.indptr)
        for e in map: 
            new_indptr[e+1:] -= n[e]

        mask = np.ones(V.shape[0], dtype=bool)
        mask[map] = False
        V = V[mask]
        return sp.csr_array((V.data, V.indices, new_indptr), shape=self.V.shape).tocoo(copy=False)
    
    def resetRows_V2(self, map):
        a = np.ones(self.V.shape[0], dtype=bool)
        a[map] = False
        return self.V.multiply(a[:, None])
    
    def getRows(self, map):
        n = len(map)

        row = self.V.row
        col = self.V.col
        data = self.V.data

        rows = np.empty(0, dtype=np.int32)
        cols = np.empty(0, dtype=np.int32)
        datas = np.empty(0, dtype=self.V.dtype)

        for i in range(n):
            indx = np.argwhere(row == map[i]).reshape(-1)
            rows = np.hstack([rows, np.ones(len(indx), dtype=np.int32)*i])
            cols = np.hstack([cols, col[indx]])
            datas = np.hstack([datas, data[indx]])

        return sp.coo_array(
            (datas, (rows, cols)), shape=(len(map), self.V.shape[1])
        )
    
    # def getRows(self, map):
    #     n = len(map)
    #     rows = sp.coo_array((n, self.V.shape[1]))
    #     for i in range(n):
    #         mat = self.V._getrow(i)
    #         mat.row += i
    #         rows += mat
    #     return rows

    def affineMap(self, W=None, b=None):
        if W is None and b is None:
            return self
        
        # elif isinstance(self.V, sp.coo_array) or isinstance(self.V, sp.coo_matrix):
        elif len(self.shape) > 1:
            c = self.c.copy()
            if W is not None:
                assert W.ndim == len(self.shape), f"inconsistent number of array dimensions between W and shape of SparseImageStar; len(shape)={len(self.shape)}, W.ndim={W.ndim}"
                
                Wr = W.reshape(-1)
                if np.prod(W.shape) == 1:
                    c = c * Wr
                    V = (self.V * Wr).tocoo() # returns csr format
                else:
                    c = c.reshape(self.shape) * W
                    c = c.reshape(-1)

                    # self.V (csr) * W
                    T = self.V.tocoo(copy=False)
                    row_ch = T.row % self.shape[2]
                    V = copy.deepcopy(self.V)
                    V.data = Wr[row_ch] * V.data
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
            
            return SparseImageStar2DCOO(c, V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)

            # if W is None:
            #     c = self.c.copy()

            #     if b.ndim == len(self.shape):
            #         c += b
            #     elif b.ndim > 1:
            #         c += np.expand_dims(b, axis=tuple(np.arange(c.ndim - b.ndim)+b.ndim)).reshape(-1)
            #         return SparseImageStar2DCOO(c, self.V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
            #     else:
            #         c += b
            #     return SparseImageStar2DCOO(c, self.V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
            
            # elif b is None:
            #     assert W.ndim == len(self.shape), f"inconsistent number of array dimensions between W and shape of SparseImageStar; len(shape)={len(self.shape)}, W.ndim={W.ndim}"

            #     Wr = W.reshape(-1)
            #     if np.prod(W.shape) == 1:
            #         c = self.c * Wr
            #         V = (self.V * Wr).tocoo() # returns csr format
            #     else:
            #         c = self.c.reshape(self.shape) * W
            #         c = c.reshape(-1)

            #         # self.V (coo) * W
            #         V = copy.deepcopy(self.V)
            #         row_ch = V.row % self.shape[2]
            #         V.data = Wr[row_ch] * V.data
                    
            #     return SparseImageStar2DCOO(c, V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)

            # else:
            #     assert W.ndim == len(self.shape), f"inconsistent number of array dimensions between W and shape of SparseImageStar; len(shape)={len(self.shape)}, W.ndim={W.ndim}"

            #     Wr = W.reshape(-1)
            #     if np.prod(W.shape) == 1:
            #         c = self.c * Wr
            #         V = (self.V * Wr).tocoo() # returns csr format
            #     else:
            #         c = self.c.reshape(self.shape) * W 
            #         c = c.reshape(-1)

            #         # self.V (coo) * W
            #         V = copy.deepcopy(self.V)
            #         row_ch = V.row % self.shape[2]
            #         V.data = Wr[row_ch] * V.data

            #     if b.ndim == len(self.shape):
            #         c += b
            #     elif b.ndim > 1:
            #         c += np.expand_dims(b, axis=tuple(np.arange(c.ndim - b.ndim)+b.ndim)).reshape(-1)
            #     else:
            #         c += b
            #     return SparseImageStar2DCOO(c, V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
        
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

        V = self.V.copy()
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
                c = W @ self.c + b
                V = np.hstack([c[:, None], V])
        
        else:
            if not dense:
                V = np.hstack([self.c, V])
        
        out_shape = (V.shape[0], )
        return SparseImageStar2DCOO(V, self.C, self.d, self.pred_lb, self.pred_ub, out_shape)
    
    
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

        if b is not None:
            assert isinstance(b, np.ndarray), 'error: ' + \
            'the offset vector should be a 1D numpy array'
            assert len(b.shape) == 1, 'error: ' + \
            'offset vector should be a 1D numpy array'

            if W is not None:
                assert W.shape[0] == b.shape[0], 'error: ' + \
                'inconsistency between mapping matrlen(self.Cx and offset'
            else:
                assert b.shape[0] == self.V.shape[0], 'error: ' + \
                'inconsistency between offset vector and SparseStar dimension'

            c += b

        V = sp.coo_array(V)
        return SparseImageStar2DCOO(c, V, self.C, self.d, self.pred_lb, self.pred_ub, self.shape)
    
    def estimateRange(self, index):
        """Quickly estimate minimum value of a state x[index]"""

        assert index >= 0 and index < self.V.shape[0], 'error: invalid index'
    
        l = self.pred_lb
        u = self.pred_ub

        # V = self.V.getRow(index).toarray()
        # pos_f = np.maximum(V, 0.0)
        # neg_f = np.minimum(V, 0.0)

        if isinstance(self.V, np.ndarray):
            X = self.V[index, 1:]
            pos_f = np.maximum(X, 0)
            neg_f = np.minimum(X, 0)

            xmin = self.V[index, 0] + pos_f @ l + neg_f @ u
            xmax = self.V[index, 0] + neg_f @ l + pos_f @ u
        else:
            X = self.V.getRow(index)
            pos_f = X.maximum(0.0)
            neg_f = X.minimum(0.0)

            xmin = self.c[index] + pos_f @ l + neg_f @ u
            xmax = self.c[index] + neg_f @ l + pos_f @ u

        # xmin = self.c[index] + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        # xmax = self.c[index] + np.matmul(neg_f, l) + np.matmul(pos_f, u)

        
        return xmin, xmax

    def estimateRanges(self):
        """Estimate the lower and upper bounds of x"""
        l = self.pred_lb
        u = self.pred_ub

        # if (self.density > 0.7):
        #     V = self.V.toarray()
        #     pos_f = np.maximum(V, 0)
        #     neg_f = np.minimum(V, 0)

        #     xmin = self.c + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        #     xmax = self.c + np.matmul(neg_f, l) + np.matmul(pos_f, u)
        # else:
        #     pos_f = self.V.maximum(0)
        #     neg_f = self.V.minimum(0)

        #     xmin = self.c + pos_f @ l + neg_f @ u
        #     xmax = self.c + neg_f @ l + pos_f @ u

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
    
    def getMin(self, index, lp_solver='gurobi'):
        """get exact minimum value of state x[index] by solving LP
           lp_solver = 'gurobi' or 'linprog' or 'glpk'
        """

        assert index >= 0 and index < self.V.shape[0], 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        if isinstance(self.V, np.ndarray):
            f = self.V[index, 1:]
            center = self.V[index, 0]
            if (f == 0).all():
                return center
        else:
            f = self.V._getrow(index)
            center = self.c[index]

            if f.nnz == 0:
                return center

            f = f.toarray()

        if lp_solver == 'gurobi':  # using gurobi is the preferred choice

            min_ = gp.Model()
            min_.Params.LogToConsole = 0
            min_.Params.OptimalityTol = GUROBI_OPT_TOL
            if self.pred_lb.size and self.pred_ub.size:
                x = min_.addMVar(shape=self.num_pred, lb=self.pred_lb, ub=self.pred_ub)
            else:
                x = min_.addMVar(shape=self.num_pred)
            min_.setObjective(f @ x, GRB.MINIMIZE)
            if len(self.d) == 0:
                C = sp.csr_array(np.zeros((1, self.num_pred)))
                d = 0
            else:
                C = self.C
                d = self.d
            min_.addConstr(C @ x <= d)
            min_.optimize()

            if min_.status == 2:
                xmin = min_.objVal + center
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (min_.status))

        elif lp_solver == 'linprog':
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

            if len(self.d) == 0:
                A = np.zeros((1, self.num_pred))
                b = np.zeros(1)
            else:
                A = self.C
                b = self.d

            lb = self.pred_lb
            ub = self.pred_ub
            lb = lb.reshape((self.num_pred, 1))
            ub = ub.reshape((self.num_pred, 1))
            res = linprog(f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))

            if res.status == 0:
                xmin = res.fun + center
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = {}'.format(res.status))

        elif lp_solver == 'glpk':
            #  https://pyglpk.readthedocs.io/en/latest/examples.html
            #  https://pyglpk.readthedocs.io/en/latest/

            glpk.env.term_on = False

            if len(self.d) == 0:
                A = np.zeros((1, self.num_pred))
                b = np.zeros(1)
            else:
                A = self.C
                b = self.d

            lb = self.pred_lb
            ub = self.pred_ub
            lb = lb.reshape((self.num_pred, 1))
            ub = ub.reshape((self.num_pred, 1))

            lp = glpk.LPX()  # create the empty problem instance
            lp.obj.maximize = False
            lp.rows.add(A.shape[0])  # append rows to this instance
            for r in lp.rows:
                r.name = chr(ord('p') + r.index)  # name rows if we want
                lp.rows[r.index].bounds = None, b[r.index]

            lp.cols.add(self.num_pred)
            for c in lp.cols:
                c.name = 'x%d' % c.index
                c.bounds = lb[c.index], ub[c.index]

            lp.obj[:] = f.tolist()
            B = A.reshape(A.shape[0]*A.shape[1],)
            lp.matrix = B.tolist()
            # lp.interior()
            lp.simplex()
            # default choice, interior may have a big floating point error

            if lp.status != 'opt':
                raise Exception('error: cannot find an optimal solution, \
                lp.status = {}'.format(lp.status))
            else:
                xmin = lp.obj.value + center
        else:
            raise Exception('error: unknown lp solver, should be gurobi or linprog or glpk')
        
        return xmin

    def getMax(self, index, lp_solver='gurobi'):
        """get exact maximum value of state x[index] by solving LP
           lp_solver = 'gurobi' or 'linprog' or 'glpk'
        """

        assert index >= 0 and index < self.V.shape[0], 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        if isinstance(self.V, np.ndarray):
            f = self.V[index, 1:]
            center = self.V[index, 0]
            if (f == 0).all():
                return center
        else:
            f = self.V._getrow(index)
            center = self.c[index]

            if f.nnz == 0:
                return center

            f = f.toarray()

        if lp_solver == 'gurobi':  # using gurobi is the preferred choice

            max_ = gp.Model()
            max_.Params.LogToConsole = 0
            max_.Params.OptimalityTol = GUROBI_OPT_TOL
            if self.pred_lb.size and self.pred_ub.size:
                x = max_.addMVar(shape=self.num_pred, lb=self.pred_lb, ub=self.pred_ub)
            else:
                x = max_.addMVar(shape=self.num_pred)
            max_.setObjective(f @ x, GRB.MAXIMIZE)
            if len(self.d) == 0:
                C = sp.csr_array(np.zeros((1, self.num_pred)))
                d = 0
            else:
                C = self.C
                d = self.d
            max_.addConstr(C @ x <= d)
            max_.optimize()

            if max_.status == 2:
                xmax = max_.objVal + center
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (max_.status))
            
        elif lp_solver == 'linprog':
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

            if len(self.d) == 0:
                A = np.zeros((1, self.num_pred))
                b = np.zeros(1)
            else:
                A = self.C
                b = self.d

            lb = self.pred_lb
            ub = self.pred_ub
            lb = lb.reshape((self.num_pred, 1))
            ub = ub.reshape((self.num_pred, 1))
            res = linprog(-f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))
            if res.status == 0:
                xmax = -res.fun + center
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = {}'.format(res.status))

        elif lp_solver == 'glpk':
            # https://pyglpk.readthedocs.io/en/latest/examples.html
            # https://pyglpk.readthedocs.io/en/latest/

            glpk.env.term_on = False  # turn off messages/display

            if len(self.d) == 0:
                A = np.zeros((1, self.num_pred))
                b = np.zeros(1)
            else:
                A = self.C
                b = self.d

            lb = self.pred_lb
            ub = self.pred_ub
            lb = lb.reshape((self.num_pred, 1))
            ub = ub.reshape((self.num_pred, 1))

            lp = glpk.LPX()  # create the empty problem instance
            lp.obj.maximize = True
            lp.rows.add(A.shape[0])  # append rows to this instance
            for r in lp.rows:
                r.name = chr(ord('p') + r.index)  # name rows if we want
                lp.rows[r.index].bounds = None, b[r.index]

            lp.cols.add(self.num_pred)
            for c in lp.cols:
                c.name = 'x%d' % c.index
                c.bounds = lb[c.index], ub[c.index]

            lp.obj[:] = f.tolist()
            B = A.reshape(A.shape[0]*A.shape[1],)
            lp.matrix = B.tolist()

            # lp.interior()
            # default choice, interior may have a big floating point error
            lp.simplex()

            if lp.status != 'opt':
                raise Exception('error: cannot find an optimal solution, lp.status = {}'.format(lp.status))
            else:
                xmax = lp.obj.value + center
        else:
            raise Exception('error: unknown lp solver, should be gurobi or linprog or glpk')
        
        return xmax
    
    def getMins(self, map, lp_solver='gurobi'):
        n = len(map)
        xmin = np.zeros(n, dtype=self.V.dtype)
        for i in range(n):
            xmin[i] = self.getMin(map[i], lp_solver)
        return xmin

    def getMaxs(self, map, lp_solver='gurobi'):
        n = len(map)
        xmax = np.zeros(n, dtype=self.V.dtype)
        for i in range(n):
            xmax[i] = self.getMax(map[i], lp_solver)
        return xmax

    def getMins_all(self, lp_solver='gurobi'):
        n = self.V.shape[0]
        xmin = np.zeros(n, dtype=self.V.dtype)
        for i in range(n):
            xmin[i] = self.getMin(i, lp_solver=lp_solver)
        return xmin

    def getMaxs_all(self, lp_solver='gurobi'):
        n = self.V.shape[0]
        xmax = np.zeros(n, dtype=self.V.dtype)
        for i in range(n):
            xmax[i] = self.getMax(i, lp_solver=lp_solver)
        return xmax

    def getRange(self, index, lp_solver='gurobi'):
        """Get the lower and upper bounds of x[index]"""

        if lp_solver == 'estimate':
            return self.estimateRange(index)
        else:
            l = self.getMin(index, lp_solver)
            u = self.getMax(index, lp_solver)
            return l, u    
        
    def getRanges(self, lp_solver='gurobi', RF=0.0, layer=None, delta=0.98):
        """Get the lower and upper bound vectors of the state
            Args:
                lp_solver: linear programming solver. e.g.: 'gurobi', 'estimate', 'linprog'
        """

        if lp_solver == 'estimate':
            l, u = self.estimateRanges()
        else:
            # if isinstance(self.V, sp.sparray):
            #     if self.V.nnz == 0:
            #         return self.estimateRanges()
            # else:
            #     if self.density

            l = self.getMins_all()
            u = self.getMaxs_all()
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
        return np.max(V, axis=3).sum()

    def get_max_point_cadidates(self):
        """ Quickly estimate max-point candidates """

        lb, ub = self.getRanges('estimate')
        max_id = np.argmax(lb)
        a = (ub >= lb[max_id])
        if sum(a) == 1:
            return [max_id]
        else:
            return np.where(a)[0]
    
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

            SIM = SparseImageStar2DCOO(self.V, new_C, new_d, self.pred_lb, self.pred_ub, self.shape, copy_=False)

        else:
            d1 = self.c[p1_indx] - self.c[p2_indx]
            C1 = self.V._getrow(p2_indx) - self.V._getrow(p1_indx)

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

            SIM = SparseImageStar2DCOO(self.c, self.V, new_C, new_d, self.pred_lb, self.pred_ub, self.shape, copy_=False)

        if SIM.isEmptySet(lp_solver=lp_solver):
            return False
        else:
            return True
    
    @staticmethod
    def inf_attack(data, epsilon=0.01, data_type='default', dtype = 'float64'):
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

        if dtype =='float64':
            data = data.astype(np.float64)
        else:
            data = data.astype(np.float32)

        lb = data - epsilon
        ub = data + epsilon

        if data_type == 'image':
            lb[lb < 0] = 0
            ub[ub > 1] = 1

        return SparseImageStar2DCOO(lb, ub)
    
# @numba.njit
# def shift(arr, num):
#     if num >= 0:
#         return np.concatenate((np.zeros(num), arr[:-num]))
#     else:
#         return np.concatenate((arr[-num:], np.zeros(-num)))