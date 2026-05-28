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
IMCrop Layer Class
Sung Woo Choi, 11/13/2025 
"""

import copy
import numpy as np
import scipy.sparse as sp

from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR


class PadLayer(object):
    """PadLayer class (similiar to numpy.pad)
    """

    def __init__(self, left, right, top, bottom, constant_values=0):
        assert top >= 0, "top must be a non-negative integer; but got {}".format(top)
        assert left >= 0, "left must be a non-negative integer; but got {}".format(left)
        assert bottom >= 0, "bottom must be a non-negative integer; but got {}".format(bottom)
        assert right >= 0, "right must be a non-negative integer; but got {}".format(right)
        assert isinstance(constant_values, (int, float)), "constant_values must be an integer or float."
        self.pad_width = np.array([top, bottom, left, right])
        self.constant_values = constant_values

    def evaluate(self, x):
        """ pads x in [height, width, channels] shape """
        top, bottom, left, right = self.pad_width
        if x.ndim == 4:
            x = np.pad(x, ((top, bottom), (left, right), (0, 0), (0, 0)), mode='constant', constant_values=self.constant_values)
        elif x.ndim == 3:
            x = np.pad(x, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=self.constant_values)
        else:
            raise ValueError("Input x must be a 3D or 4D array.")
        return x

    def pad_coo(self, input, shape, tocsc=False):
        if len(self.pad_width) == 4:
            pad = np.array(self.pad_width)
        elif len(self.pad_width) == 2:
            pad = np.array([self.pad_width[0], self.pad_width[0], self.pad_width[1], self.pad_width[1]])
        elif len(self.pad_width) == 1:
            pad = np.ones(4)*self.pad_width[0]

        """Adding padding to coo"""
        row = input.row + (input.row // (shape[1]*shape[2])) * (pad[2]+pad[3])* shape[2]
        row += shape[2]*((shape[1]+pad[2]+pad[3])*pad[0]+pad[2])

        mo = shape[0] + pad[0] + pad[1]
        no = shape[1] + pad[2] + pad[3]
		
        if tocsc is True:
            output = sp.csc_array((input.data, (row, input.col)), shape = (mo*no*shape[2], input.shape[1]))
        else:
            output = sp.coo_array((input.data, (row, input.col)), shape = (mo*no*shape[2], input.shape[1]))
        return output, mo, no
    
    def pad_csr(self, input, shape):
        if len(self.pad_width) == 4:
            pad = np.array(self.pad_width)
        elif len(self.pad_width) == 2:
            pad = np.array([self.pad_width[0], self.pad_width[0], self.pad_width[1], self.pad_width[1]])
        elif len(self.pad_width) == 1:
            pad = np.ones(4)*self.pad_width[0]

        t = shape[1]+pad[2]+pad[3]
        bp = shape[2]*(t*pad[0]+pad[2])
        ep = shape[2]*(t*pad[1]+pad[3])+1
        
        ptr = input.indptr
        dtype = ptr.dtype
        indptr = [np.zeros(bp, dtype=dtype)]
        
        k = shape[1]*shape[2]
        r = np.ones((pad[2]+pad[3])*shape[2], dtype=dtype)
        for i in range(shape[0]):
            b = i*k
            e = (i+1)*k
            indptr.append(ptr[b:e])
            if i < shape[0]-1:
                indptr.append(ptr[e]*r)

        indptr.append(ptr[-1]*np.ones(ep, dtype=dtype))
        indptr = np.concatenate(indptr, dtype=dtype)
        
        mo = shape[0] + pad[0] + pad[1]
        no = shape[1] + pad[2] + pad[3]
        
        output = sp.csr_array((input.data, input.indices, indptr), shape = (mo*no*shape[2], input.shape[1]))

        return output, mo, no
    
    def reachSingleInput(self, In):
        if isinstance(In, ImageStar):
            V = self.evaluate(In.V)
            return ImageStar(V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        elif isinstance(In, SparseImageStar2DCOO):
            c = self.evaluate(In.c.reshape(In.shape)).ravel()
            V, mo, no = self.pad_coo(In.V, In.shape)
            return SparseImageStar2DCOO(c, V, In.C, In.d, In.pred_lb, In.pred_ub, (mo, no, In.shape[2]))
        
        elif isinstance(In, SparseImageStar2DCSR):
            c = self.evaluate(In.c.reshape(In.shape)).ravel()
            V, mo, no = self.pad_csr(In.V, In.shape)
            return SparseImageStar2DCSR(c, V, In.C, In.d, In.pred_lb, In.pred_ub, (mo, no, In.shape[2]))

        else:
            raise TypeError("Unsupported input type for PadLayer reachability analysis.")   
        
    def reach(self, in_sets, method=None, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """ reachability analysis of Pad layer """

        if isinstance(in_sets, list):
            out_sets = []
            for In in in_sets:
                out_sets.append(self.reachSingleInput(In))
            return out_sets
        else:
            return self.reachSingleInput(in_sets)
        