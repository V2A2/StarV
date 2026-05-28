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
Image Star Class
Author: Sung Woo Choi
Created: 08/10/2023

"""
import copy
import torch
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.linalg import block_diag
import polytope as pc
import glpk
import gurobipy as gp
from gurobipy import GRB
from StarV.util.lp_solver import solve_index_lp as util_solve_index_lp
from StarV.util.lp_solver import solve_lp as util_solve_lp
from StarV.set.star import Star
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.set.predicate_layout import PredLayout

GUROBI_OPT_TOL = 1e-6

class ImageStar(object):
    """
        ImageStar Class for reachability
        date: 08/10/2023
        Representation of a ImageStar
        ===========================================================================================================================
        ImageStar set defined by

        Channel First Format
        H W C N

        ===========================================================================================================================
    """

    def __init__(self, *args, copy_=True):
        """
            Key Attributes:
            V = [] #
            c = V[:, :, :, 0] : anchor image
            X = V[:, :, :, 1:] : generator image
            C = [] # linear constraints matrix of the predicate variables
            d = [] # linear constraints vector of the predicate variables
            layout = None # predicate layout

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
        
        if len_ == 6:
            [V, C, d, pred_lb, pred_ub, layout] = copy.deepcopy(args) if copy_ is True else args
                
            assert isinstance(V, np.ndarray), \
            'error: basis matrix should be a numpy array'
            assert isinstance(pred_lb, np.ndarray), \
            'error: lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray), \
            'error: upper bound vector should be a 1D numpy array'

            if d.size > 0:
                assert isinstance(C, np.ndarray), \
                'error: a linear constraint matrix should be a numpy array'
                assert isinstance(d, np.ndarray), \
                'error: a linear constraint vector should be a 1D numpy array'
                assert d.ndim == 1, \
                'error: a linear constraint vector should be a 1D numpy array'
                assert C.shape[0] == d.shape[0], \
                'error: inconsistency between lienar constraints matrix and linear constraints vector'
                assert C.shape[1] == pred_lb.shape[0], \
                'error: inconsistent number of predicatve variables between linear constratints matrix and predicate bound vectors'

            assert len(pred_lb.shape) == 1, \
            'error: lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, \
            'error: upper bound vector should be a 1D numpy array'
            assert pred_ub.shape[0] == pred_lb.shape[0], \
            'error: inconsistent number of predicate variables between predicate lower- and upper-boud vectors'
                
            if V.ndim == 1:
                V = V[None, None, :, None]

            elif V.ndim == 2:
                V = V[None, None, :, :]
            
            elif V.ndim == 3:
                V = V[:, :, :, None]

            elif V.ndim > 4:
                raise Exception(f"error: invalid dimension of basis matrix, V.shape = {V.shape}")
            
            if isinstance(layout, PredLayout):
                pass
            elif layout is None or layout == []:
                layout = PredLayout(n_base = pred_lb.shape[0])
            else:
                raise ValueError('error: layout should be an instance of PredLayout class or None or empty list')
            
            self.height, self.width, self.num_channel = V.shape[:3]
            self.num_pixel = self.height * self.width * self.num_channel
            self.num_pred = pred_lb.shape[0]

            self.V = V
            self.C = C
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.pred_layout = layout
            
        elif len_ == 5:

            [V, C, d, pred_lb, pred_ub] = copy.deepcopy(args) if copy_ is True else args
                
            assert isinstance(V, np.ndarray), \
            'error: basis matrix should be a numpy array'
            assert isinstance(pred_lb, np.ndarray), \
            'error: lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray), \
            'error: upper bound vector should be a 1D numpy array'

            if d.size > 0:
                assert isinstance(C, np.ndarray), \
                'error: a linear constraint matrix should be a numpy array'
                assert isinstance(d, np.ndarray), \
                'error: a linear constraint vector should be a 1D numpy array'
                assert d.ndim == 1, \
                'error: a linear constraint vector should be a 1D numpy array'
                assert C.shape[0] == d.shape[0], \
                'error: inconsistency between lienar constraints matrix and linear constraints vector'
                assert C.shape[1] == pred_lb.shape[0], \
                'error: inconsistent number of predicatve variables between linear constratints matrix and predicate bound vectors'

            assert len(pred_lb.shape) == 1, \
            'error: lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, \
            'error: upper bound vector should be a 1D numpy array'
            assert pred_ub.shape[0] == pred_lb.shape[0], \
            'error: inconsistent number of predicate variables between predicate lower- and upper-boud vectors'
            
            if V.ndim == 1:
                V = V[None, None, :, None]

            elif V.ndim == 2:
                V = V[None, None, :, :]
            
            elif V.ndim == 3:
                V = V[:, :, :, None]

            elif V.ndim > 4:
                raise Exception(f"error: invalid dimension of basis matrix, V.shape = {V.shape}")
            
            self.height, self.width, self.num_channel = V.shape[:3]
            self.num_pixel = self.height * self.width * self.num_channel
            self.num_pred = C.shape[1] if d.size > 0 else V.shape[-1] - 1

            self.V = V
            self.C = C
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
        
        elif len_ == 2:

            [lb, ub] = copy.deepcopy(args) if copy_ is True else args

            assert isinstance(lb, np.ndarray), \
            'error: lower bound vector should be a numpy array'
            assert isinstance(ub, np.ndarray), \
            'error: upper bound vector should be a numpy array'

            assert lb.shape == ub.shape, \
            'error: inconsistency between lower bound image and upper bound image'

            if (ub < lb).any():
                raise Exception(
                    'error: the upper bounds must not be less than the lower bounds for all dimensions')

            img_shape = lb.shape
            img_dim = len(img_shape)
            dtype = lb.dtype

            lb = lb.ravel()
            ub = ub.ravel()
            dim = lb.shape[0]
            
            gtr = ub > lb
            nv = gtr.sum()

            c = 0.5 * (lb + ub)
            if dim == nv:
                v = np.diag(0.5 * (ub - lb))
            else:
                v = np.zeros((dim, nv), dtype=dtype)
                j = 0
                for i in range(dim):
                    if gtr[i] > 0:
                        v[i, j] = 0.5 * (ub[i] - lb[i])
                        j += 1
            V = np.hstack([c[:, None], v])

            self.num_pred = nv

            if img_dim == 3:
                img_shape = img_shape + (self.num_pred+1, )

            elif img_dim == 2:
                img_shape = img_shape + (1, self.num_pred+1)

            elif img_dim == 1:
                img_shape = img_shape + (1, 1, self.num_pred+1)

            self.V = V.reshape(img_shape)
            self.C = np.empty([0, 0], dtype=dtype)
            self.d = np.empty([0], dtype=dtype)
            self.pred_lb = -np.ones(nv, dtype=dtype)
            self.pred_ub = np.ones(nv, dtype=dtype)
            self.height, self.width, self.num_channel = img_shape[0:3]
            self.num_pixel = self.height * self.width * self.num_channel

        # elif len_ == 2:
        #     [lb, ub] = args
			
		# 	if copy_ is True:
		# 		lb = lb.copy()
		# 		ub = ub.copy()

        #     assert isinstance(lb, np.ndarray), \
        #     'error: lower bound vector should be a numpy array'
        #     assert isinstance(ub, np.ndarray), \
        #     'error: upper bound vector should be a numpy array'
        #     assert lb.shape == ub.shape, \
        #     'error: inconsistency between lower bound image and upper bound image'
        #     assert lb.ndim <= 3, \
        #     'error: lower and upper bound vectors should be less than 4D tensor'

        #     if (ub < lb).any():
        #         raise Exception(
        #             'error: the upper bounds must not be less than the lower bounds for all dimensions')
            
        #     img_shape = lb.shape
        #     img_dim = lb.ndim
        #     dtype = lb.dtype

        #     lb = lb.reshape(-1)
        #     ub = ub.reshape(-1)
        #     dim = lb.shape[0]
        #     nv = int(sum(ub > lb))

        #     V = np.zeros((dim, nv+1), dtype=dtype)
        #     j = 1
        #     for i in range(dim):
        #         if ub[i] > lb[i]:
        #             V[i, j] = 1
        #             j += 1

        #     self.num_pred = nv

        #     if img_dim == 3:
        #         img_shape = img_shape + (self.num_pred+1, )

        #     elif img_dim == 2:
        #         img_shape = img_shape + (1, self.num_pred+1)

        #     elif img_dim == 1:
        #         img_shape = img_shape + (1, 1, self.num_pred+1)

        #     self.V = V.reshape(img_shape)
        #     self.C = np.empty([0, 0], dtype=dtype)
        #     self.d = np.empty([0], dtype=dtype)
        #     self.pred_lb = lb
        #     self.pred_ub = ub
        #     self.height, self.width, self.num_channel = img_shape[0:3]
        #     self.num_pixel = self.height * self.width * self.num_channel

        elif len_ == 0:
            self.V = np.ndarray([0, 0])
            self.C = np.empty([0, 0])
            self.d = np.empty([0])
            self.pred_lb = np.empty([0])
            self.pred_ub = np.empty([0])
            self.height = 0
            self.width = 0
            self.num_channel = 0
            self.num_pixel = 0
            self.num_pred = 0

        else:
            raise Exception(
                'error: invalid number of input arguments (should be 0, 2, 5, or 6)')
    
    def __str__(self, channel_first=False):
        if channel_first:
            print('ImageStar Set (channel first):')
            print('V: \n')
            print(self.V.transpose([3,2,0,1])) # N C H W
        else:
            print('ImageStar Set:')
            print('V: \n{}'.format(self.V))
        print('C: \n{}'.format(self.C))
        print('d: {}'.format(self.d))
        print('pred_lb: {}'.format(self.pred_lb))
        print('pred_ub: {}'.format(self.pred_ub))

        print('height: {}'.format(self.height))
        print('width: {}'.format(self.width))
        print('num_channel: {}'.format(self.num_channel))
        print('num_pred: {}'.format(self.num_pred))
        return ''

    def __repr__(self):
        print('ImageStar Set:')
        print('V: {}, {}'.format(self.V.shape, self.V.dtype))
        print('C: {}, {}'.format(self.C.shape, self.C.dtype))
        print('d: {}, {}'.format(self.d.shape, self.d.dtype))
        print('pred_lb: {}, {}'.format(self.pred_lb.shape, self.pred_lb.dtype))
        print('pred_ub: {}, {}'.format(self.pred_ub.shape, self.pred_ub.dtype))

        print('height: {}'.format(self.height))
        print('width: {}'.format(self.width))
        print('num_channel: {}'.format(self.num_channel))
        print('num_pred: {}'.format(self.num_pred))
        print('')
        return ''
    
    def __len__(self):
        return 1
    
    def nbytes_generator(self):
        return self.V.nbytes
    
    def nbytes_constraints(self):
        return self.C.nbytes + self.d.nbytes
    
    def nbytes(self):
        # V and c (generator image and anchor image)
        nbt = self.nbytes_generator()
        # C and d
        nbt += self.nbytes_constraints()
        # pred_lb and pred_ub
        nbt += self.pred_lb.nbytes + self.pred_ub.nbytes
        return nbt

    def c(self, h=None, w=None, c=None):
        """Get anchor image of ImageStar"""
        
        if self.V.ndim == 4:
            if h is None and w is None and c is None:
                return self.V[:, :, :, 0].copy()
            return self.V[h, w, c, 0].copy()
        
        elif self.V.ndim == 2:
            if h is None:
                return self.V[:, 0].copy()
            return self.V[h, 0].copy()
    
    def X(self, h=None, w=None, c=None, n=None):
        """Get generator images of ImageStar"""

        if self.V.ndim == 4:
            
            if h != None and w != None and c != None:
                return self.V[h, w, c, 1:].copy()
            
            if h != None and w != None:
                return self.V[h, w, :, 1:].copy()
            
            return self.V[:, :, :, 1:].copy()
        
        elif self.V.ndim == 2:
            if h is None:
                return self.V[:, 1:]
            return self.V[h, 1:].copy()
        
        else:
            raise Exception('Basis image dimension issue')    
    
    # @property
    # def shape(self):
    #     return self.height, self.width, self.num_channel
    
    def shape(self):
        return self.height, self.width, self.num_channel     
        
    def clone(self):
        return copy.deepcopy(self)
                
    def index_to3D(self, index):
        # V is in [height, width, channel] order

        index = copy.deepcopy(index)
        num = self.width * self.num_channel
        h = index // num
        index -= h * num
        w = index // self.num_channel
        c = index % self.num_channel
        return h, w, c
    
    def resetRow(self, index):
        h_indx, w_indx, c_indx = self.index_to3D(index)
        new_V = copy.deepcopy(self.V)
        new_V[h_indx, w_indx, c_indx, :] = 0
        return ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    def resetRows(self, map):
        ndim = self.V.ndim
        if ndim == 4:
            new_V = self.V.reshape(self.num_pixel, self.num_pred + 1).copy()
        elif ndim == 2:
            new_V = self.V.copy()
        else:
            raise Exception('Invalid basis image dimension')
        
        new_V[map, :] = 0
        
        if ndim == 4:
            new_V = new_V.reshape(self.V.shape)

        return ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)


        # h_map, w_map, c_map = self.index_to3D(map)
        # new_V = copy.deepcopy(self.V)
        # for i in range(len(map)):
        #     new_V[h_map[i], w_map[i], c_map[i], :] = 0
        # return ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    def resetRow_hwc(self, h_indx, w_indx, c_indx):
        new_V = copy.deepcopy(self.V)
        new_V[h_indx, w_indx, c_indx, :] = 0
        return ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    def resetRows_hwc(self, h_map, w_map, c_map):
        assert len(h_map) == len(w_map) == len(c_map), \
        'error: inconsistent lengths of h_map, w_map, and c_map'
        new_V = copy.deepcopy(self.V)
        for i in range(len(h_map)):
            new_V[h_map[i], w_map[i], c_map[i], :] = 0
        return ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    def affineMap(self, W=None, b=None):            

        if W is None and b is None:
            return self
        
        elif self.V.shape[0] == 1 and self.V.shape[1] == 1:
            return self.flatten_affineMap(W, b)
        
        if W is not None:
            if W.shape[1] == self.V.shape[0]:
                w1, w2 = W.shape
                h, w, c, m = self.V.shape
                V = self.V.reshape(h, -1)
                V = np.matmul(W, V).reshape(w1, w, c, m)
            else:
                assert W.ndim == self.V.ndim-1, 'error: ' +\
                f"inconsistent number of array dimensions between W and shape of Image; len(shape)={self.V.ndim-1}, W.ndim={W.ndim}"
                V = self.V * W[:, :, :, None]
        else:
            V = self.V.copy()

        if b is not None:
            if b.ndim == self.V.ndim-1:
                V[:, :, :, 0] += b
            elif b.ndim > 1:
                V[:, :, :, 0] += np.expand_dims(b, axis=tuple(np.arange(V.ndim - 1 - b.ndim)+b.ndim))
            else:
                V[:, :, :, 0] += b
        
        return ImageStar(V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    def flatten_affineMap(self, W=None, b=None):
        if W is None and b is None:
            return copy.deepcopy(self)
        
        assert self.V.shape[0] == 1 and self.V.shape[1] == 1, 'error: ImageStar is not flattened to operate affine mapping, V.shape = {}'.format(self.V.shape)

        V = copy.deepcopy(self.V) #.reshape(self.num_pixel, self.num_pred+1)

        if W is not None:
            assert isinstance(W, np.ndarray), 'error: ' + \
            'the mapping matrix should be a 2D numpy array'
            assert W.shape[1] == self.num_pixel, 'error: ' + \
            'inconsistency between mapping matrix and SparseImageStar dimension'

            V = np.matmul(W, V)

        if b is not None:
            assert isinstance(b, np.ndarray), 'error: ' + \
            'the offset vector should be a 1D numpy array'
            assert len(b.shape) == 1, 'error: ' + \
            'offset vector should be a 1D numpy array'

            if W is not None:
                assert W.shape[0] == b.shape[0], 'error: ' + \
                'inconsistency between mapping matrix and offset'
            else:
                assert b.shape[0] == self.num_pixel, 'error: ' + \
                'inconsistency between offset vector and SparseStar dimension'

            V[:, :, :, 0] += b
        
        return ImageStar(V, self.C, self.d, self.pred_lb, self.pred_ub)

    def getMin(self, *args, solver_opts=None, gurobi_model_pack=None, show=False):
        """Get the minimum value of state x[index] or x[h_indx, w_indx, c_indx] by solving LP
            @lp_solver = 'gurobi', 'linprog', or 'glpk'
            @h_indx: veritcial index
            @w_indx: horizontal index
            @c_indx: channel index
            @index: flattened index
        """
        len_ = len(args)

        if len_ == 4:
            [h_indx, w_indx, c_indx, lp_solver] = args
            # index = None
            return self.getMin_hwc(
                h_indx, w_indx, c_indx, lp_solver,
                solver_opts=solver_opts, gurobi_model_pack=gurobi_model_pack, show=show
            )

        elif len_ == 3:
            [h_indx, w_indx, c_indx] = args
            lp_solver = 'gurobi'
            # index = None
            return self.getMin_hwc(
                h_indx, w_indx, c_indx, lp_solver,
                solver_opts=solver_opts, gurobi_model_pack=gurobi_model_pack, show=show
            )

        elif len_ == 2:
            [index, lp_solver] = args
            return self.getMin_index(
                index, lp_solver,
                solver_opts=solver_opts, gurobi_model_pack=gurobi_model_pack, show=show
            )

        elif len_ == 1:
            [index] = args
            lp_solver = 'gurobi'
            return self.getMin_index(
                index, lp_solver,
                solver_opts=solver_opts, gurobi_model_pack=gurobi_model_pack, show=show
            )

        else:
            raise Exception(
                'error: invalid number of input arguments (should be between 1 and 4)')
    
    def getMax(self, *args, solver_opts=None, gurobi_model_pack=None, show=False):
        """Get the maximum value of state x[index] or x[h_indx, w_indx, c_indx] by solving LP
            @lp_solver = 'gurobi', 'linprog', or 'glpk'
            @h_indx: veritcial index
            @w_indx: horizontal index
            @c_indx: channel index
            @index: flattened index
        """
        len_ = len(args)

        if len_ == 4:
            [h_indx, w_indx, c_indx, lp_solver] = args
            # index = None
            return self.getMax_hwc(
                h_indx, w_indx, c_indx, lp_solver,
                solver_opts=solver_opts, gurobi_model_pack=gurobi_model_pack, show=show
            )

        elif len_ == 3:
            [h_indx, w_indx, c_indx] = args
            lp_solver = 'gurobi'
            # index = None
            return self.getMax_hwc(
                h_indx, w_indx, c_indx, lp_solver,
                solver_opts=solver_opts, gurobi_model_pack=gurobi_model_pack, show=show
            )

        elif len_ == 2:
            [index, lp_solver] = args
            return self.getMax_index(
                index, lp_solver,
                solver_opts=solver_opts, gurobi_model_pack=gurobi_model_pack, show=show
            )
        
        elif len_ == 1:
            [index] = args
            lp_solver = 'gurobi'
            return self.getMax_index(
                index, lp_solver,
                solver_opts=solver_opts, gurobi_model_pack=gurobi_model_pack, show=show
            )
        
        else:
            raise Exception(
                'error: invalid number of input arguments (should be between 1 and 4)')
    
    def get_binary_predicate_indices(self):
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
        assert index >= 0 and index < self.num_pixel, 'error: invalid index'
        V = self.V.reshape(self.num_pixel, self.num_pred + 1)
        f = V[index, 1:]
        center = V[index, 0]
        if (f == 0).all():
            return None, center
        return np.asarray(f).reshape(-1), center
    
    def get_lp_ub(self):
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
    
    def getMin_index(self, index, lp_solver='gurobi', V=None, solver_opts=None, gurobi_model_pack=None, show=False):
        """Get the minimum value of state x[index] by solving LP
            lp_solver = 'gurobi', 'cupdlp', 'cupdlp-gurobi', 'scipy-milp', 'linprog', or 'glpk'
        """

        assert index >= 0 and index < self.num_pixel, 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        if V is None:
            # Internal ImageStar basis is 4D: (h, w, ch, m); flatten to (num_pixel, m)
            V = self.V.reshape(self.num_pixel, self.num_pred + 1)
        elif isinstance(V, np.ndarray) and V.ndim == 4:
            V = V.reshape(self.num_pixel, self.num_pred + 1)
        
        f = V[index, 1:]
        center = V[index, 0]
        if (f == 0).all():
            return center
        A, b = self.get_lp_ub()
        return util_solve_lp(
            f=f, A_ub=A, b_ub=b, lb=self.pred_lb, ub=self.pred_ub, lp_solver=lp_solver,
            sense='min', center=center, binary_idx=self.get_binary_predicate_indices(),
            solver_opts=solver_opts, model_pack=gurobi_model_pack, show=show,
        )
    
    def getMax_index(self, index, lp_solver='gurobi', V=None, solver_opts=None, gurobi_model_pack=None, show=False):
        """Get the maximum value of state x[index] by solving LP
            lp_solver = 'gurobi', 'cupdlp', 'cupdlp-gurobi', 'scipy-milp', 'linprog', or 'glpk'
        """

        assert index >= 0 and index < self.num_pixel, 'error: invalid index'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'
        if V is None:
            V = self.V.reshape(self.num_pixel, self.num_pred + 1)
        elif isinstance(V, np.ndarray) and V.ndim == 4:
            V = V.reshape(self.num_pixel, self.num_pred + 1)
        
        f = V[index, 1:]
        center = V[index, 0]
        if (f == 0).all():
            return center
        A, b = self.get_lp_ub()
        return util_solve_lp(
            f=f, A_ub=A, b_ub=b, lb=self.pred_lb, ub=self.pred_ub, lp_solver=lp_solver,
            sense='max', center=center, binary_idx=self.get_binary_predicate_indices(),
            solver_opts=solver_opts, model_pack=gurobi_model_pack, show=show,
        )
    
    def getMin_hwc(self, h_indx, w_indx, c_indx, lp_solver='gurobi', solver_opts=None, gurobi_model_pack=None, show=False):
        """Get the minimum value of state x[index] or x[h_indx, w_indx, c_indx] by solving LP
            lp_solver = 'gurobi', 'cupdlp', 'cupdlp-gurobi', 'scipy-milp', 'linprog', or 'glpk'
            h_indx: veritcial index
            w_indx: horizontal index
            c_indx: channel index
        """

        assert h_indx >= 0 and h_indx < self.height, \
        'error: invalid vertical index'
        assert w_indx >= 0 and w_indx < self.width, \
        'error: invalid horizontal index'
        assert c_indx >= 0 and c_indx < self.num_channel, \
        'error: invalid channel index'
        index = (h_indx * self.width + w_indx) * self.num_channel + c_indx
        return self.getMin_index(index, lp_solver=lp_solver, solver_opts=solver_opts,
            gurobi_model_pack=gurobi_model_pack, show=show)


    def getMax_hwc(self, h_indx, w_indx, c_indx, lp_solver='gurobi', solver_opts=None, gurobi_model_pack=None, show=False):
        """Get the maximum value of state x[h_indx, w_indx, c_indx] by solving LP
            lp_solver = 'gurobi', 'cupdlp', 'cupdlp-gurobi', 'scipy-milp', 'linprog', or 'glpk'
            h_indx: veritcial index
            w_indx: horizontal index
            c_indx: channel index
        """

        assert h_indx >= 0 and h_indx < self.height, \
        'error: invalid vertical index'
        assert w_indx >= 0 and w_indx < self.width, \
        'error: invalid horizontal index'
        assert c_indx >= 0 and c_indx < self.num_channel, \
        'error: invalid channel index'
        index = (h_indx * self.width + w_indx) * self.num_channel + c_indx
        return self.getMax_index(index, lp_solver=lp_solver, solver_opts=solver_opts,
            gurobi_model_pack=gurobi_model_pack, show=show)

    def getMins_all(self, lp_solver='gurobi', solver_opts=None, show=False):
        xmin = np.zeros([self.height, self.width, self.num_channel], dtype=self.V.dtype)
        for h_ in range(self.height):
            for w_ in range(self.width):
                for c_ in range(self.num_channel):
                    xmin[h_, w_, c_] = self.getMin_hwc(
                        h_, w_, c_, lp_solver, solver_opts=solver_opts, show=show
                    )
        return xmin

    def getMaxs_all(self, lp_solver='gurobi', solver_opts=None, show=False):
        xmax = np.zeros([self.height, self.width, self.num_channel], dtype=self.V.dtype)
        for h_ in range(self.height):
            for w_ in range(self.width):
                for c_ in range(self.num_channel):
                    xmax[h_, w_, c_] = self.getMax_hwc(
                        h_, w_, c_, lp_solver, solver_opts=solver_opts, show=show
                    )
        return xmax

    def getMins(self, *args, solver_opts=None, show=False):
        """Get the maximum values of state x corresponding map indexes
        """
        len_ = len(args)

        if len_ == 4:
            [h_map, w_map, c_map, lp_solver] = args
            map = None

        elif len_ == 3:
            [h_map, w_map, c_map] = args
            lp_solver = 'gurobi'
            map = None

        elif len_ == 2:
            [map, lp_solver] = args

        elif len_ == 1:
            [map] = args
            lp_solver = 'gurobi'

        else:
            raise Exception(
                'error: invalid number of input arguments (should be between 1 and 4)')

        n = len(map) if map is not None else len(h_map)
        xmin = np.zeros(n, dtype=self.V.dtype)

        if map is not None:
            V = self.V.reshape(self.num_pixel, self.num_pred+1)
            # h_map, w_map, c_map = self.index_to3D(map)
            for i in range(n):
                xmin[i] = self.getMin_index(map[i], lp_solver, V=V, solver_opts=solver_opts, show=show)

        else:
            for i in range(n):
                xmin[i] = self.getMin_hwc(h_map[i], w_map[i], c_map[i], lp_solver, solver_opts=solver_opts, show=show)

        return xmin

    def getMaxs(self, *args, solver_opts=None, show=False):
        """Get the maximum values of state x corresponding map indexes
        """
        len_ = len(args)

        if len_ == 4:
            [h_map, w_map, c_map, lp_solver] = args
            map = None

        elif len_ == 3:
            [h_map, w_map, c_map] = args
            lp_solver = 'gurobi'
            map = None

        elif len_ == 2:
            [map, lp_solver] = args

        elif len_ == 1:
            [map] = args
            lp_solver = 'gurobi'

        else:
            raise Exception(
                'error: invalid number of input arguments (should be between 1 and 4)')

        n = len(map) if map is not None else len(h_map)
        xmin = np.zeros(n, dtype=self.V.dtype)

        if map is not None:
            V = self.V.reshape(self.num_pixel, self.num_pred+1)
            # h_map, w_map, c_map = self.index_to3D(map)
            for i in range(n):
                xmin[i] = self.getMax_index(map[i], lp_solver, V=V, solver_opts=solver_opts, show=show)

        else:
            for i in range(n):
                xmin[i] = self.getMax_hwc(h_map[i], w_map[i], c_map[i], lp_solver, solver_opts=solver_opts, show=show)

        return xmin
    
    def estimateRange(self, h_indx, w_indx, c_indx):
        """Estimate the minimum and maximum values of a state x[index]"""

        assert h_indx >= 0 and h_indx < self.height, \
        'error: invalid vertical index'
        assert w_indx >= 0 and w_indx < self.width, \
        'error: invalid horizontal index'
        assert c_indx >= 0 and c_indx < self.num_channel, \
        'error: invalid channel index'
        
        l = self.pred_lb
        u = self.pred_ub

        X = self.X(h_indx, w_indx, c_indx)
        pos_f = np.maximum(X, 0.0)
        neg_f = np.minimum(X, 0.0)

        xmin = self.c(h_indx, w_indx, c_indx) + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        xmax = self.c(h_indx, w_indx, c_indx) + np.matmul(neg_f, l) + np.matmul(pos_f, u)
        return xmin, xmax

    def estimateRanges(self):
        """Estimate the lower and upper bounds of x"""

        l = self.pred_lb
        u = self.pred_ub

        X = self.X().reshape(self.num_pixel, self.num_pred)
        pos_f = np.maximum(X, 0.0)
        neg_f = np.minimum(X, 0.0)

        xmin = self.c().reshape(-1) + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        xmax = self.c().reshape(-1) + np.matmul(neg_f, l) + np.matmul(pos_f, u)
        return xmin, xmax

    def getRange(self, h_indx, w_indx, c_indx, lp_solver='gurobi', solver_opts=None, show=False):
        """Get the lower and upper bounds of x[index]"""

        if lp_solver == 'estimate':
            return self.estimateRange(h_indx, w_indx, c_indx)
        else:
            l = self.getMin(h_indx, w_indx, c_indx, lp_solver, solver_opts=solver_opts, show=show)
            u = self.getMax(h_indx, w_indx, c_indx, lp_solver, solver_opts=solver_opts, show=show)
            return l, u    
        

    def getRanges(self, lp_solver='gurobi', RF=0.0, layer=None, delta=0.98, solver_opts=None, show=False):
        """Get the lower and upper bound vectors of the state
            Args:
                lp_solver: linear programming solver. e.g.: 'gurobi', 'estimate', 'linprog'
        """

        if lp_solver == 'estimate':
            l, u = self.estimateRanges()
        else:
            l = self.getMins_all(lp_solver=lp_solver, solver_opts=solver_opts, show=show)
            u = self.getMaxs_all(lp_solver=lp_solver, solver_opts=solver_opts, show=show)
        return l, u

    # def getRanges(self, lp_solver='gurobi', RF=0.0, layer=None, delta=0.98):
    #     """Get the lower and upper bound vectors of the state
    #         Args:
    #             lp_solver: linear programming solver. e.g.: 'gurobi', 'estimate', 'linprog'
    #             RF: relaxation factor \in [0.0, 1.0]
    #     """
        
    #     shape = self.height, self.width, self.num_channel

    #     if RF == 1.0:
    #         l, u = self.estimateRanges()
        
    #     elif RF == 0.0:
    #         if lp_solver == 'estimate':
    #             l, u = self.estimateRanges()
    #         else:
    #             l = self.getMins()
    #             u = self.getMaxs()
            
    #     else:
    #         assert RF > 0.0 and RF <= 1.0, \
    #         'error: relaxation factor should be greater than 0.0 but less than or equal to 1.0'
    #         l, u = self.estimateRanges()
            
    #         if layer in ['logsig', 'tansig']:
    #             n1 = round(1 - RF) * self.num_pixel

    #             midx = np.argsort((u - l))[::-1]
    #             midb = np.argwhere((l[midx] >= -delta) & (u[midx] <= delta))
                
    #             n2 = n1
    #             check = midb.flatten().shape[0]
    #             if n2 > check:
    #                 n2 = check

    #             mid = midx[midb[0:n2]]
    #             l1 = self.getMins(mid)
    #             u1 = self.getMaxs(mid)
    #             l[mid] = l1
    #             u[mid] = u1

    #         elif  layer in ['poslin', 'relu']:
    #             map = np.argwhere((l < 0) & (u > 0))

    #             n1 = round(1 - RF) * len(map)

                
    #             area = 0.5*abs(u[map]*l[map])
    #             midx = np.argsort(area)[::-1]

    #             mid = midx[0:n1]
    #             l1 = self.getMins(mid)
    #             u1 = self.getMaxs(mid)
    #             l[mid] = l1
    #             u[mid] = u1

    #         else:
    #             n1 = round(1 - RF) * self.num_pixel

    #             midx = np.argsort((u - l))[::-1]
    #             mid = midx[0:n1]
    #             l1 = self.getMins(mid)
    #             u1 = self.getMaxs(mid)
    #             l[mid] = l1
    #             u[mid] = u1
      
    #     return l.reshape(shape), u.reshape(shape)

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
        V = self.V[:, :, :, 1:] != 0
        return np.any(V, axis=-1).sum()
    
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

        assert p1_indx >= 0 and p1_indx < self.num_pixel, 'error: invalid index for point 1'
        assert p2_indx >= 0 and p2_indx < self.num_pixel, 'error: invalid index for point 2'

        V = self.V.reshape(self.num_pixel, self.num_pred+1)

        d1 = V[p1_indx, 0] - V[p2_indx, 0]
        C1 = V[p2_indx, 1:] - V[p1_indx, 1:]

        if len(self.d) > 0:
            new_d = np.hstack([self.d, d1])
            new_C = np.vstack([self.C, C1])
        else:
            new_d = np.array([d1])
            new_C = C1[None, :]

        # S = Star(V, new_C, new_d, self.pred_lb, self.pred_ub, copy_=False)

        V = V[None, None, :, :]
        S = ImageStar(V, new_C, new_d, self.pred_lb, self.pred_ub, copy_=False)
        
        if S.isEmptySet(lp_solver=lp_solver):
            return False
        else:
            return True
     
    def toStar(self, copy_=True):
        """Convert ImageStarTensor class to Star class"""
        V = self.V.reshape(self.num_pixel, self.num_pred+1)
        return Star(V, self.C, self.d, self.pred_lb, self.pred_ub, copy_=True)

    def to_SIM(self, format='csr'):
        assert format in ['csr', 'coo'], f"format should be either 'csr' or 'coo', but received {format}"
        shape = self.V.shape[:3]
        num_pred = self.V.shape[3] - 1
        c = self.V[:, :, :, 0].ravel()
        C = sp.csr_array(self.C)
        if format == 'csr':
            V = sp.csr_array(self.V[:, :, :, 1:].reshape(-1, num_pred))
            return SparseImageStar2DCSR(c, V, C, self.d, self.pred_lb, self.pred_ub, shape)
        else:
            V = sp.coo_array(self.V[:, :, :, 1:].reshape(-1, num_pred))
            return SparseImageStar2DCOO(c, V, C, self.d, self.pred_lb, self.pred_ub, shape)
        
    def concatenate(self, X, axis=0):
        """Concatenate two imagestar sets"""
        assert isinstance(X, ImageStar), \
        'error: the input X should be an ImageStar set'
        assert self.shape() == X.shape(), \
        f'error: the two ImageStar sets should have the same shape to concatenate; shapes are current:{self.shape} and X:{X.shape}'

        sc, xc = self.V[:, :, :, 0], X.V[:, :, :, 0]
        c = np.concatenate([sc, xc], axis=axis)[:, :, :, None]
        A = ImageStar.block_diag_axis(self.V[:, :, :, 1:], X.V[:, :, :, 1:], axis=axis)
        new_V = np.concatenate((c, A), axis=3)

        SC = self.C if len(self.C) > 0 else np.empty((0, self.num_pred))    
        Sd = self.d if len(self.d) > 0 else np.empty((0,))
        XC = X.C if len(X.C) > 0 else np.empty((0, X.num_pred))
        Xd = X.d if len(X.d) > 0 else np.empty((0,))

        new_C = block_diag(SC, XC)
        new_d = np.concatenate((Sd, Xd))

        new_pred_lb = np.concatenate((self.pred_lb, X.pred_lb))
        new_pred_ub = np.concatenate((self.pred_ub, X.pred_ub))
        return ImageStar(new_V, new_C, new_d, new_pred_lb, new_pred_ub, copy_=False)
    
    def block_diag_axis(A, B, axis=0):
        """
        Block diagonal concatenation along a specified axis
        A, B: input 4D numpy arrays with shape (h, w, c, num_pred)
        axis \in {0, 1, 2}:
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
            out[:h1, :, :, :m1] = A      # top-left block
            out[h1:, :, :, m1:] = B      # bottom-right block

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

    def minKowskiSum(self, X):
        """Minkowski sum of two ImageStar sets"""
        shape = self.shape()
        S = self.toStar(copy_=False)
        X = X.toStar(copy_=False)
        R = S.minKowskiSum(X)
        return R.toImageStar(shape)
    
    @staticmethod
    def rand(height, width, channel):
        """"Randomly generate a ImageStar"""
        dim = height*width*channel
        lb = -np.random.rand(dim).reshape(height, width, channel)
        ub = np.random.rand(dim).reshape(height, width, channel)
        return ImageStar(lb, ub)

    @staticmethod
    def isMax(maxMap, ori_image, center, others, lp_solver='gurobi'):
        """
        Check if a pixel value is the maximum value compared with others
        This is the core step for exactly performing maxpooling operation on an ImageStar set
       
        Args:
            @maxMap: the current maxMap ImageStar
            @ori_image: the original ImageStar to compute the maxMap
            @center: is the center pixel position we want to check
                    center = [index]
            @others: is the other pixel position we want to compare with the cetner one
                    others = [index0, index1]
        """
        assert maxMap.num_pred == ori_image.num_pred, \
        'error: Inconsistency between number of predicates in the currrent maxMap and the original image'
        n = len(others)
        
        # the center may be the max point with some extra constraints on the predicate variables
        new_C = np.zeros([n, maxMap.num_pred], dtype=maxMap.V.dtype)
        new_d = np.zeros(n, dtype=maxMap.V.dtype)

        V = ori_image.V.reshape(ori_image.num_pixel, ori_image.num_pred+1)

        for i in range(n):
            new_C[i, :] = V[others[i], 1:] - V[center, 1:]
            new_d[i] = V[center, 0] - V[others[i], 0]
        
        if len(maxMap.d) > 0:
            C1 = np.vstack([maxMap.C, new_C])
            d1 = np.hstack([maxMap.d, new_d])
        else:
            C1 = new_C
            d1 = new_d

        # # remove redundant constraints
        # E = np.hstack([C1, d1[:, None]])
        # E = np.unique(E, axis=0)

        # C1 = E[:, :-1]
        # d1 = E[:, -1:].reshape(-1)
        # ###

        S = Star(V, C1, d1, ori_image.pred_lb, ori_image.pred_ub, copy_=False)
        
        if S.isEmptySet(lp_solver=lp_solver):
            return None, None
        else:
            return C1, d1


    @staticmethod
    def inf_attack(data, epsilon=0.01, data_type='default', dtype = 'float64'):
        """Generate a ImageStar set by infinity norm attack on input dataset"""

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
            lb[lb < 0] = 0.0
            ub[ub > 1] = 1.0

        return ImageStar(lb, ub)
    
    @staticmethod
    def rand_bounds(in_height, in_width, in_channel):
        """Generate a random SparStar by random bounds"""

        lb = -np.random.rand(in_height, in_width, in_channel)
        ub = np.random.rand(in_height, in_width, in_channel)
        return ImageStar(lb, ub)
    
    @staticmethod
    def rand_polytope(h, w, ch, N, dtype='float64'):
        """ Generate a random Star with constraints"""

        assert h > 0 and w > 0 and ch > 0, 'error: invalid dimension'
        assert N > h * w * ch, 'error: number constraints should be greater than dimension'

        dim = h * w * ch
        A = np.random.rand(N, dim)

        # compute the convex hull
        P = pc.qhull(A)

        c = np.zeros([P.dim, 1])
        I = np.eye(P.dim)

        V = np.hstack([c, I]).reshape(h, w, ch, dim + 1)
        pred_lb, pred_ub = P.bounding_box
        return ImageStar(V, P.A, P.b, pred_lb.reshape(-1), pred_ub.reshape(-1))
