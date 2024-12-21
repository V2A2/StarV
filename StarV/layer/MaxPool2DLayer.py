"""
Max Pooling 2D Layer Class
Sung Woo Choi, 03/28/2024
"""

# !/usr/bin/python3
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR

import time
import copy
import torch
import numpy as np
import scipy.sparse as sp
import multiprocessing
import torch.nn.functional as F
import multiprocessing
import ipyparallel

from timeit import default_timer as timer

class MaxPool2DLayer(object):
    """ 
    MaxPool2DLayer Class
    Author: Sung Woo Choi
    Date: 03/28/2024

    """

    def __init__(
            self,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            in_shape = None,
            module = 'default',
            dtype = 'float64',
        ):

        assert module in ['default', 'pytorch'], \
        'error: MaxPool2DLayer supports moudles: \'default\', which use numpy kernels, and \'pytorch\''
        self.module = module

        if dtype == 'float32' or dtype == np.float32:
            self.numpy_dtype = np.float32
            self.torch_dtype = torch.float32
        else:
            self.numpy_dtype = np.float64
            self.torch_dtype = torch.float64

        if self.module == 'default':
            # check stride, padding, and dilation

            assert isinstance(kernel_size, tuple) or isinstance(kernel_size, list) or \
                   isinstance(kernel_size, int) or isinstance(stride, np.ndarray), \
            f'error: kernel_size should be a tuple, list, numpy ndarray, or int but received {type(kernel_size)}'
            assert isinstance(stride, tuple) or isinstance(stride, list) or \
                   isinstance(stride, int) or isinstance(stride, np.ndarray), \
            f'error: stride should be a tuple, list, numpy ndarray, or int but received {type(stride)}'
            assert isinstance(padding, tuple) or isinstance(padding, list) or \
                   isinstance(padding, int) or isinstance(stride, np.ndarray), \
            f'error: padding should be a tuple, list, numpy ndarray, or in tbut received {type(padding)}'

            if isinstance(kernel_size, int):
                assert kernel_size >= 0, 'error: kernel size should non-negative integer'
                self.kernel_size = np.ones(2, dtype=np.uint16)*kernel_size[0]
            else:
                if len(kernel_size) == 1:
                    assert kernel_size[0] >= 0, 'error: kernel size should non-negative integer'
                    self.kernel_size = np.ones(2, dtype=np.uint16)*kernel_size[0]
                elif len(kernel_size) == 2:
                    assert kernel_size[0] >= 0 and kernel_size[1] >= 0, 'error: kernel size should non-negative integers'              
                    self.kernel_size = np.array(kernel_size).astype(np.uint16)
                else:
                    raise Exception('error: incorrect kernel size')
 
            if isinstance(padding, int):
                assert padding >= 0, 'error: padding should non-negative integers'
                self.padding = np.ones(2, dtype=np.int16)*padding
            else:
                padding = np.array(padding)
                assert (padding >= 0).any(), 'error: padding should non-negative integers'

                if len(padding) == 1:
                    self.padding = np.ones(2, dtype=np.int16)*padding[0]
                else:
                    if len(padding) == 4:
                        if padding[0] == padding[1] and padding[2] == padding[3]:
                            padding = np.array([padding[0], padding[2]])
                    self.padding = np.array(padding)
            assert (self.padding <= self.kernel_size // 2).any(), 'error: padding should be at most half of kernel size'
            
            if isinstance(stride, int):
                assert stride > 0, 'error: stride should positive integer'
                self.stride = np.ones(2, dtype=np.uint16)*stride
            else:
                if len(stride) == 1:
                    assert stride[0] > 0, 'error: stride should positive integer'
                    self.stride = np.ones(2, dtype=np.uint16)*stride[0]
                elif len(stride) == 2:
                    assert stride[0] > 0 and stride[1] > 0, 'error: stride should positive integers'
                    self.stride = np.array(stride).astype(np.uint16)
                else:
                    raise Exception('error: incorrect padding')

        elif self.module == 'pytorch':

            self.layer = torch.nn.MaxPool2d(
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
            )

        else:
            raise Exception('Unknown layer module')

        if in_shape is not None:

            assert len(in_shape) == 3, \
            f"The input shape (in_shape) should be a  3-tuple containing (H, W, C). Given in_shape = {in_shape}"
            
            m, n, c = in_shape
            mo, no = MaxPool2DLayer.get_output_size(m, n, self.kernel_size, stride, padding)
            self.in_shape = in_shape
            self.out_shape = (mo, no, c)

    def info(self):
        print('Max Pooling 2D Layer')
        print('module: {}'.format(self.module))
        print('kernel size: {}'.format(self.kernel_size))
        print('stride: {}'.format(self.stride))
        print('padding: {}'.format(self.padding))
        if self.in_shape is not None:
            print('in_shape: ', self.in_shape)
            print('out_shape: ', self.out_shape)
        return '\n'
    
    def pad_coo(input, shape, padding, tocsr=False):
        row = input.row + (input.row // (shape[1]*shape[2])) * 2 * padding[1] * shape[2]
        row += shape[2]*((shape[1]+2*padding[1])*padding[0]+padding[1])

        mo = shape[0]+2*padding[0]
        no = shape[1]+2*padding[1]
        if tocsr is True:
            output = sp.csr_array((input.data, (row, input.col)), shape = (mo*no*shape[2], input.shape[1]))
        else:
            output = sp.coo_array((input.data, (row, input.col)), shape = (mo*no*shape[2], input.shape[1]))
        return output, mo, no

    def pad_csr(input, shape, padding, tocoo=False):
        if tocoo is True:
            return MaxPool2DLayer.pad_coo(input.tocoo(False), shape, padding)
        else:
            return MaxPool2DLayer.pad_coo(input.tocoo(False), shape, padding, tocsr=True)
        
    def get_output_size(in_height, in_width, kernel_size, stride, padding):
        h, w = in_height, in_width
        H, W = kernel_size
        ho = np.floor(
            ((h + 2*padding[0] - H) // stride[0]) + 1
        ).astype(int)
        wo = np.floor(
            ((w + 2*padding[1] - W) // stride[1]) + 1
        ).astype(int)

        assert ho > 0 and wo > 0, 'error: the shape of resulting output should be positive'
        return ho, wo
    
    def evaluate(self, input):
        """
            For module == 'default' set up:
                @input: (H, W, C, N); H: height, W: width, C: input channel, N: batch or number of predicates

            For module == 'pytorch' set up:
                @input: (N, C, H, W); N: batch or number of predicates, C: input channel, H: height, W: width 
        """

        if self.module == 'pytorch':
            return self.maxpool2d_pytorch(input)
        else:
            return self.maxpool2d(input)
        
    def maxpool2d_pytorch(self, input, bias=True):
        """
            Args:
               @input: dataset in pytorch with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: convolved dataset
        """
        
        assert isinstance(self.layer, torch.nn.MaxPool2d), '\'layer\' should be torch.nn.AvgPool2d for \'pytorch\' module'

        in_dim = input.ndim
        if in_dim == 4:
            H, W, C, N = input.shape
        elif in_dim == 3:
            H, W, C = input.shape
            N = 1
        else:
            raise Exception('input should be either 2D, 3D, or 4D tensor')

        input = copy.deepcopy(input).reshape(H, W, C, N)
        # change input shape from (H, W, C, N) to (N, C, H, W)
        input = input.transpose([3, 2, 0, 1])
        input = torch.from_numpy(input).type(self.torch_dtype)
        output = self.layer(input).detach().numpy()
        # change input shape to H, W, C, N
        output.transpose([2, 3, 1, 0])
        
        # if in_dim == 3:
        #     output = output.reshape(H, W, C) 

        return output

    def maxpool2d_basic(self, input):
        """ 
            Max pooling 2D

            Args:
               @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: convolved dataset
        
        """
        stride = self.stride
        padding = self.padding
        dtype = input.dtype

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'

        in_dim = input.ndim
        assert in_dim >= 2 and in_dim <= 4, \
        'error: input should be 2D, 3D, or 4D numpy ndarray'

        input = copy.deepcopy(input).astype(self.numpy_dtype)
        if in_dim == 2:
            input = input[:, :, None, None]
        elif in_dim == 3:
            input = input[:, :, :, None]
        
        h, w, c, n = input.shape
        H, W = self.kernel_size
        ho, wo = MaxPool2DLayer.get_output_size(h, w, self.kernel_size, stride, padding)
        pad_input = MaxPool2DLayer.add_zero_padding(input, padding).astype(np.float64)

        output = np.zeros((ho, wo, c, n), dtype=dtype)
        for z in range(n):
        
            for k in range(c):
                working_input = pad_input[:, :, k, z]
                out_ch = np.zeros((ho, wo), dtype=dtype)

                for i in range(ho):
                    i_stride = i*stride[0]

                    for j in range(wo):
                        j_stride = j*stride[1]
                        out_ch[i, j] = np.max(working_input[i_stride : i_stride+H, 
                                                            j_stride : j_stride+W])

                output[:, :, k, z] = out_ch
        output = output / (H*W)

        # if in_dim == 2:
        #     output = output.reshape(ho, wo)
        # elif in_dim == 3:
        #     output = output.reshape(ho, wo, c)
        return output


    def maxpool2d(self, input):
        """ 
            Max pooling 2D

            Args:
               @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: convolved dataset
        
        """
        stride = self.stride
        padding = self.padding
        dtype = input.dtype

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'

        in_dim = input.ndim
        assert in_dim >= 2 and in_dim <= 4, \
        'error: input should be 2D, 3D, or 4D numpy ndarray'

        input = input.astype(self.numpy_dtype)
        if in_dim == 2:
            input = input[:, :, None, None]
        elif in_dim == 3:
            input = input[:, :, :, None]
        
        m, n, c, b = input.shape

        p, q = self.kernel_size
        mo, no = MaxPool2DLayer.get_output_size(m, n, self.kernel_size, stride, padding)

        if padding[0] > 0 or padding[1] > 0:
            m += 2*padding[0]
            n += 2*padding[1]
            XF = np.pad(input, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0), (0,0)), mode='constant').reshape(m*n*c, b)
        else:
            XF = input.reshape(m*n*c, b)

        K = np.zeros([p, q, c], dtype=bool)
        K[:, :, 0] = 1
        Z = np.pad(K, ((0, 0), (0, n-q), (0, 0)), mode='constant').reshape(-1)
        Z_indices = np.where(Z > 0)[0]

        noc = no*c
        ko = mo*noc
        
        i_shift = n * c * stride[0]
        j_shift = stride[1]*c
        output = np.zeros([ko, b], dtype=dtype)

        i_ = 0
        for i in range(mo):
            for j in range(no):
                for k in range(c):
                    ind = i*i_shift + j*j_shift + k + Z_indices
                    output[i_, :] = XF[ind, :].max(axis=0) #XF in [m*n*c, b]
                    i_ += 1

        # if in_dim == 2:
        #     output = output.reshape(mo, no)
        # elif in_dim == 3:
        #     output = output.reshape(mo, no, c)
        # else:
        #     output = output.reshape(mo, no, c, b)
        output = output.reshape(mo, no, c, b)
        return output


    def get_max_pixels(LB, UB):
        max_lb_indx = LB.argmax()
        diff = UB - LB[max_lb_indx]
        max_pixels = np.where(diff > 0)[0]
        if len(max_pixels) == 0:
            return max_pixels, diff, max_lb_indx

        max_pixels = np.delete(max_pixels, np.where(max_pixels == max_lb_indx)[0])
        return max_pixels, diff, max_lb_indx
    
    def get_max_multi_pixels(LB, UB):
        max_lb_indx = LB.argmax(axis=1)
        diff = UB - LB[max_lb_indx]

        max_pixels = sp.lil_array(diff.clip(0, None))
        data = max_pixels.data
        rows = max_pixels.rows

        for i, row in enumerate(rows):
            if len(row) > 0:
                indx = np.where(row == max_lb_indx[i])[0]
                rows[i] = np.delete(row, indx)
                data[i] = np.delete(data[i], indx)
        return max_pixels, diff, max_lb_indx

    def get_localMax_index(I, LB, UB, ind, lp_solver):
        
        max_pixels, diff, max_lb_indx = MaxPool2DLayer.get_max_pixels(LB, UB)

        if len(max_pixels) == 0:
            max_indx = np.array([ind[max_lb_indx]])
        else:
            candidates = np.where(diff >= 0)[0]
            new_points = ind[candidates]

            LB = LB[candidates]
            UB = UB[candidates]

            # update lower and uper bounds with LP solver
            for i_, p_ in enumerate(new_points):
                LB[i_] = I.getMin(p_, lp_solver)
                UB[i_] = I.getMax(p_, lp_solver)

            max_pixels, diff, max_lb_indx = MaxPool2DLayer.get_max_pixels(LB, UB)

            if len(max_pixels) == 0:
                max_indx = np.array([new_points[max_lb_indx]])

            else:
                candidates = np.where(diff >= 0)[0]          
                candidates = np.delete(candidates, np.where(candidates == max_lb_indx)[0])
                new_points2 = new_points[candidates]

                m = len(candidates)
                max_id = new_points[max_lb_indx]
                max_indx = [max_id]

                if m > 0:
                    for i_ in range(m):
                        p1 = new_points2[i_]
                        if I.is_p1_larger_than_p2(p1, max_id, lp_solver=lp_solver):
                            max_indx.append(p1)

                max_indx = np.array(max_indx)
                max_indx = new_points[candidates]
        return max_indx

    def get_localMax_multi_index(I, LB, UB, ind, lp_solver='gurobi'):
        
        max_pixels, diff, max_lb_indx = MaxPool2DLayer.get_max_multi_pixels(LB, UB)

        if max_pixels.nnz == 0:
            max_indx = ind[max_lb_indx]

        else:
            candidates = np.where(diff >= 0)
            new_points = ind[candidates]

            LB = LB[candidates]
            UB = UB[candidates]

            # update lower and uper bounds with LP solver
            for i_, p_ in enumerate(new_points):
                LB[i_] = I.getMin(p_, lp_solver)
                UB[i_] = I.getMax(p_, lp_solver)

            max_pixels, diff, max_lb_indx = MaxPool2DLayer.get_max_multi_pixels(LB, UB)

            if len(max_pixels) == 0:
                max_indx = np.array([new_points[max_lb_indx]])

            else:
                candidates = np.where(diff >= 0)[0]          
                candidates = np.delete(candidates, np.where(candidates == max_lb_indx)[0])
                new_points2 = new_points[candidates]

                m = len(candidates)
                max_id = new_points[max_lb_indx]
                max_indx = [max_id]

                if m > 0:
                    for i_ in range(m):
                        p1 = new_points2[i_]
                        if I.is_p1_larger_than_p2(p1, max_id, lp_solver=lp_solver):
                            max_indx.append(p1)

                max_indx = np.array(max_indx)
                max_indx = new_points[candidates]
        return max_indx

    def stepSplit(self, in_image, ori_image, pos, split_index, lp_solver='gurobi'):
        
        assert isinstance(in_image, ImageStar), 'error: input maxMap is not an ImageStar'
        assert isinstance(ori_image, ImageStar), 'error: reference image is not an ImageStar'

        images = []
        n = len(split_index)
        for i in range(n):
            center = split_index[i]
            others = np.delete(split_index, i)
            C, d = ImageStar.isMax(in_image, ori_image, center, others, lp_solver)

            if C is not None and d is not None:
                # V = in_image.V.reshape(in_image.num_pixel, in_image.num_pred+1)
                V = copy.deepcopy(in_image.V.reshape(in_image.num_pixel, in_image.num_pred+1))
                V_ori = ori_image.V.reshape(ori_image.num_pixel, ori_image.num_pred+1)
                V[pos, :] = V_ori[pos, :]
                V = V.reshape(in_image.height, in_image.width, in_image.num_channel, in_image.num_pred+1)
                IM = ImageStar(V, C, d, in_image.pred_lb, in_image.pred_ub)
                images.append(IM)
        return images

    def stepSplitMultiInputs(self, in_images, ori_image, pos, split_index, lp_solver='gurobi'):
        images = []
        n = len(in_images)
        for i in range(n):
            images.extend(self.stepSplit(in_images[i], ori_image, pos, split_index, lp_solver=lp_solver))
        return images

    
    def reach_approx_imagestar_test(self, I, lp_solver='gurobi', show=False):

        assert isinstance(I, ImageStar), 'error: input is not an ImageStar'

        stride = self.stride
        padding = self.padding
        dtype = I.V.dtype

        m, n, c, b = I.height, I.width, I.num_channel, I.num_pred+1
        p, q = self.kernel_size

        mo, no = MaxPool2DLayer.get_output_size(m, n, self.kernel_size, stride, padding)

        V = I.V
        state_lb, state_ub = I.getRanges(lp_solver = 'estimate')
        
        # apply padding to ImageStar
        if padding[0] > 0 or padding[1] > 0:
            state_lb = state_lb.reshape(m, n, c)
            state_ub = state_ub.reshape(m, n, c)
            state_lb = np.pad(state_lb, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant').reshape(-1)
            state_ub = np.pad(state_ub, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant').reshape(-1)

            m += 2*padding[0]
            n += 2*padding[1]
            V = np.pad(V, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0), (0,0)), mode='constant').reshape(m*n*c, b)
        else:
            V = I.V.reshape(m*n*c, b)

        # get indexies of feature map
        K = np.zeros([p, q, c], dtype=bool)
        K[:, :, 0] = True
        Z = np.pad(K, ((0, 0), (0, n-q), (0, 0)), mode='constant').reshape(-1)
        Z_indices = np.where(Z > 0)[0]

        noc = no*c
        ko = mo*noc

        i_shift = n * c * stride[0]
        j_shift = stride[1]*c

        # compute max_index and  when applying maxpooling operation
        # compute new number of predicate
        Vmax = np.zeros([ko, b], dtype=dtype)
        new_pred_index_list = []
        max_index_list = []
        C2_list, d2_list = [], []

  
        ind = np.arange(ko, dtype=np.int32)[:, np.newaxis]
        ind = ind//noc*i_shift + (ind//c)%no*j_shift + ind%c + Z_indices

        # get local maximum points in the feature maps
        # max_indexes = MaxPool2DLayer.get_localMax_multi_index(I, state_lb[ind], state_ub[ind], ind, lp_solver)





        for i in range(mo):
            for j in range(no):
                for k in range(c):
                    ind = i*i_shift + j*j_shift + k + Z_indices

                    # get local maximum points in the feature map
                    max_index = MaxPool2DLayer.get_localMax_index(I, state_lb[ind], state_ub[ind], ind, lp_solver)

                    if len(max_index) == 1:
                        Vmax[i_, :] = V[max_index[0], :]

                    else:
                        max_index_list.append(max_index)
                        # max_index_list.append(ind)
                        new_pred_index_list.append(i_)
                        if show:
                            print(f"The local image has {len(max_index)} max candidates")
                    i_ += 1

        nPred = len(new_pred_index_list)
        tPred = I.num_pred + nPred
        if show:
            print(f"{nPred} new predicate variables are introduced")
        
        if nPred > 0:
            new_V = np.zeros([ko, nPred], dtype=dtype)
            lb = np.zeros(nPred, dtype=dtype)
            ub = np.zeros(nPred, dtype=dtype)
            for i in range(nPred):
                new_V[new_pred_index_list[i], i] = 1
            
                max_index = max_index_list[i]
                N = len(max_index)
                V1 = V[max_index, :]
                E = np.zeros([N, nPred], dtype=dtype)
                E[:, i] = -1
                
                C2_list.append(np.hstack([V1[:, 1:], E]))
                d2_list.append(-V1[:, 0])

                lb[i] = state_lb[max_index].min()
                ub[i] = state_ub[max_index].max()

            new_V = np.hstack([Vmax, new_V]).reshape(mo, no, c, tPred+1)
            
            # update constraint matrix C and contraint vector d
            # case 1: y <= ub
            C1 = np.hstack([np.zeros([nPred, I.num_pred], dtype=dtype), np.eye(nPred, dtype=dtype)])
            d1 = ub

            # case 2: y >= x
            C2 = np.vstack(C2_list)
            d2 = np.hstack(d2_list)

            if len(I.d) == 0: # for Star set as initial C and d might be []
                new_C = np.vstack([C1, C2])
                new_d = np.hstack([d1, d2])

            else:
                m = I.C.shape[0]
                C0 = np.hstack([I.C, np.zeros([m, nPred], dtype=dtype)])
                d0 = I.d
                new_C = np.vstack([C0, C1, C2])
                new_d = np.hstack([d0, d1, d2])

            new_pred_lb = np.hstack([I.pred_lb, lb])
            new_pred_ub = np.hstack([I.pred_ub, ub])
            return ImageStar(new_V, new_C, new_d, new_pred_lb, new_pred_ub)

        # no new predicate variables are introduced
        new_V = Vmax.reshape(mo, no, c, tPred+1)
        return ImageStar(new_V, I.C, I.d, I.pred_lb, I.pred_ub)
    
    def reach_approx_imagestar(self, I, lp_solver='gurobi', show=False):

        assert isinstance(I, ImageStar), 'error: input is not an ImageStar'

        stride = self.stride
        padding = self.padding
        dtype = I.V.dtype

        m, n, c, b = I.height, I.width, I.num_channel, I.num_pred+1
        p, q = self.kernel_size

        mo, no = MaxPool2DLayer.get_output_size(m, n, self.kernel_size, stride, padding)

        V = I.V
        state_lb, state_ub = I.getRanges(lp_solver = 'estimate')
        
        # apply padding to ImageStar
        if padding[0] > 0 or padding[1] > 0:
            state_lb = state_lb.reshape(m, n, c)
            state_ub = state_ub.reshape(m, n, c)
            state_lb = np.pad(state_lb, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant').reshape(-1)
            state_ub = np.pad(state_ub, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant').reshape(-1)

            m += 2*padding[0]
            n += 2*padding[1]
            V = np.pad(V, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0), (0,0)), mode='constant').reshape(m*n*c, b)
        else:
            V = I.V.reshape(m*n*c, b)

        # get indexies of feature map
        K = np.zeros([p, q, c], dtype=bool)
        K[:, :, 0] = True
        Z = np.pad(K, ((0, 0), (0, n-q), (0, 0)), mode='constant').reshape(-1)
        Z_indices = np.where(Z > 0)[0]

        ko = mo*no*c

        i_shift = n * c * stride[0]
        j_shift = stride[1]*c

        # compute max_index and  when applying maxpooling operation
        # compute new number of predicate
        Vmax = np.zeros([ko, b], dtype=dtype)
        new_pred_index_list = []
        max_index_list = []
        C2_list, d2_list = [], []
        i_ = 0

        for i in range(mo):
            for j in range(no):
                for k in range(c):
                    ind = i*i_shift + j*j_shift + k + Z_indices

                    # get local maximum points in the feature map
                    max_index = MaxPool2DLayer.get_localMax_index(I, state_lb[ind], state_ub[ind], ind, lp_solver)

                    if len(max_index) == 1:
                        Vmax[i_, :] = V[max_index[0], :]

                    else:
                        max_index_list.append(max_index)
                        # max_index_list.append(ind)
                        new_pred_index_list.append(i_)
                        if show:
                            print(f"The local image has {len(max_index)} max candidates")
                    i_ += 1

        nPred = len(new_pred_index_list)
        tPred = I.num_pred + nPred
        if show:
            print(f"{nPred} new predicate variables are introduced")
        
        if nPred > 0:
            new_V = np.zeros([ko, nPred], dtype=dtype)
            lb = np.zeros(nPred, dtype=dtype)
            ub = np.zeros(nPred, dtype=dtype)
            for i in range(nPred):
                new_V[new_pred_index_list[i], i] = 1
            
                max_index = max_index_list[i]
                N = len(max_index)
                V1 = V[max_index, :]
                E = np.zeros([N, nPred], dtype=dtype)
                E[:, i] = -1
                
                C2_list.append(np.hstack([V1[:, 1:], E]))
                d2_list.append(-V1[:, 0])

                lb[i] = state_lb[max_index].min()
                ub[i] = state_ub[max_index].max()

            new_V = np.hstack([Vmax, new_V]).reshape(mo, no, c, tPred+1)
            
            # update constraint matrix C and contraint vector d
            # case 1: y <= ub
            C1 = np.hstack([np.zeros([nPred, I.num_pred], dtype=dtype), np.eye(nPred, dtype=dtype)])
            d1 = ub

            # case 2: y >= x
            C2 = np.vstack(C2_list)
            d2 = np.hstack(d2_list)

            if len(I.d) == 0: # for Star set as initial C and d might be []
                new_C = np.vstack([C1, C2])
                new_d = np.hstack([d1, d2])

            else:
                m = I.C.shape[0]
                C0 = np.hstack([I.C, np.zeros([m, nPred], dtype=dtype)])
                d0 = I.d
                new_C = np.vstack([C0, C1, C2])
                new_d = np.hstack([d0, d1, d2])

            new_pred_lb = np.hstack([I.pred_lb, lb])
            new_pred_ub = np.hstack([I.pred_ub, ub])
            return ImageStar(new_V, new_C, new_d, new_pred_lb, new_pred_ub)

        # no new predicate variables are introduced
        new_V = Vmax.reshape(mo, no, c, tPred+1)
        return ImageStar(new_V, I.C, I.d, I.pred_lb, I.pred_ub)
   
    def reach_approx_fmaxpool2d_coo(self, I, lp_solver='gurobi', show=False):
        """
            Convolution 2D for sparse images

            Args:
                @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches
                @input: SparseImage
            Return: 
               @R: convolved dataset
        """

        assert isinstance(I, SparseImageStar2DCOO), 'error: input is not an SparseImageStar COO intead received {}'.format(I.__class__.__name__)

        stride = self.stride
        padding = self.padding
        dtype = I.V.dtype

        b = I.V.shape[1]
        m, n, c = I.shape
        p, q = self.kernel_size

        mo, no = MaxPool2DLayer.get_output_size(m, n, self.kernel_size, stride, padding)

        V = I.V
        center = I.c
        state_lb, state_ub = I.getRanges(lp_solver = 'estimate')

        # apply padding to ImageStar
        if padding[0] > 0 or padding[1] > 0:
            center = center.reshape(m, n, c)
            state_lb = state_lb.reshape(m, n, c)
            state_ub = state_ub.reshape(m, n, c)
            center = np.pad(center, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant').reshape(-1)
            state_lb = np.pad(state_lb, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant').reshape(-1)
            state_ub = np.pad(state_ub, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant').reshape(-1)
            V, m, n = MaxPool2DLayer.pad_coo(V, I.shape, padding, tocsr=True)
        else:
            V = V.tocsr(copy=False)

        ind_dtype = V.indptr.dtype

        # get indexies of feature map
        K = np.zeros([p, q, c], dtype=bool)
        K[:, :, 0] = True
        Z = np.pad(K, ((0, 0), (0, n-q), (0, 0)), mode='constant').reshape(-1)
        Z_indices = np.where(Z > 0)[0]

        ko = mo*no*c

        i_shift = n * c * stride[0]
        j_shift = stride[1]*c

        # compute max_index and  when applying maxpooling operation
        # compute new number of predicate
        new_c = np.zeros(ko, dtype=dtype)
        new_pred_index_list = []
        max_index_list = []
        max_data = []
        max_row = []
        max_col = []
        i_ = 0
        for i in range(mo):
            for j in range(no):
                for k in range(c):
                    ind = i*i_shift + j*j_shift + k + Z_indices

                    # get local maximum points in the feature map
                    max_index = MaxPool2DLayer.get_localMax_index(I, state_lb[ind], state_ub[ind], ind, lp_solver)
                    
                    if len(max_index) == 1:
                        new_c[i_] = center[max_index[0]]
                        ind_s = V.indptr[max_index[0]]
                        ind_e = V.indptr[max_index[0]+1]
                        max_data.append(V.data[ind_s : ind_e])
                        max_row.append(np.ones(ind_e - ind_s, dtype=ind_dtype) * i_)
                        max_col.append(V.indices[ind_s : ind_e])

                    else:
                        max_index_list.append(max_index)
                        # max_index_list.append(ind)
                        new_pred_index_list.append(i_)
                    
                    i_ += 1
        
        nPred = len(new_pred_index_list)
        tPred = I.num_pred + nPred
        if show:
            print(f"{nPred} new predicate variables are introduced")

        # construct an over-approximate SparseImageStarCOO reachable set

        if nPred > 0:
            # add new predicate varaibels
            max_data.append(np.ones(nPred, dtype=dtype))
            max_row.append(new_pred_index_list)
            max_col.append(np.arange(nPred, dtype=ind_dtype) + I.num_pred)
            new_V = sp.coo_array((np.hstack(max_data), (np.hstack(max_row), np.hstack(max_col))), shape=(ko, tPred))

            # update constraint matrix C and contraint vector d
            cand_data = []
            cand_row = []
            cand_col = []
            row_track = nPred
            lb = np.zeros(nPred, dtype=dtype)
            ub = np.zeros(nPred, dtype=dtype)
            for i in range(nPred):
                max_index = max_index_list[i]
                N = len(max_index)

                # create C2_list and d2_list
                V1 = V[max_index].tocoo(copy=False)
                cand_data.extend(V1.data)
                cand_row.extend(V1.row + row_track)
                cand_col.extend(V1.col)

                cand_data.extend(-np.ones(N, dtype=dtype))
                cand_row.extend(np.arange(N, dtype=ind_dtype) + row_track)
                cand_col.extend(np.ones(N, dtype=ind_dtype)*i + I.num_pred)
                row_track += N

                lb[i] = state_lb[max_index].min()
                ub[i] = state_ub[max_index].max()

            eye_data = np.ones(nPred, dtype=dtype)
            eye_col = np.arange(nPred, dtype=ind_dtype) + I.num_pred
            eye_row = np.arange(nPred, dtype=ind_dtype)

            # case 1: y <= ub
            # case 2: y >= x
            d1 = ub
            d2 = -center[np.hstack(max_index_list)]

            data = np.hstack([eye_data, cand_data])
            row = np.hstack([eye_row, cand_row])
            col = np.hstack([eye_col, cand_col])
            C = sp.csr_array((data, (row, col)), shape=(row_track, tPred))

            if I.C.nnz > 0:
                data = np.hstack([I.C.data, C.data])
                indices = np.hstack([I.C.indices, C.indices])
                indptr = np.hstack([I.C.indptr, C.indptr[1:]+I.C.nnz])
                new_C = sp.csr_array((data, indices, indptr), shape=(I.C.shape[0]+C.shape[0], C.shape[1]))
                new_d = np.hstack([I.d, d1, d2])
            else:
                new_C = C
                new_d = np.hstack([d1, d2])

            new_pred_lb = np.hstack([I.pred_lb, lb])
            new_pred_ub = np.hstack([I.pred_ub, ub])

            out_shape = (mo, no, c)
            return SparseImageStar2DCOO(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape)

        new_V = sp.coo_array((np.hstack(max_data), (np.hstack(max_row).astype(np.int32), np.hstack(max_col))), shape=(ko, tPred))
        out_shape = (mo, no, c)
        return SparseImageStar2DCOO(new_c, new_V, I.C, I.d, I.pred_lb, I.pred_ub, out_shape)
    

    def reach_approx_fmaxpool2d_csr(self, I, lp_solver='gurobi', show=False):
        """
            Convolution 2D for sparse images

            Args:
                @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches
                @input: SparseImage
            Return: 
               @R: convolved dataset
        """

        assert isinstance(I, SparseImageStar2DCSR), 'error: input is not an ImageStar'

        stride = self.stride
        padding = self.padding
        dtype = I.V.dtype

        b = I.V.shape[1]
        m, n, c = I.shape
        p, q = self.kernel_size

        mo, no = MaxPool2DLayer.get_output_size(m, n, self.kernel_size, stride, padding)

        V = I.V
        center = I.c
        state_lb, state_ub = I.getRanges(lp_solver = 'estimate')

        # apply padding to ImageStar
        if padding[0] > 0 or padding[1] > 0:
            center = center.reshape(m, n, c)
            state_lb = state_lb.reshape(m, n, c)
            state_ub = state_ub.reshape(m, n, c)
            center = np.pad(center, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant').reshape(-1)
            state_lb = np.pad(state_lb, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant').reshape(-1)
            state_ub = np.pad(state_ub, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant').reshape(-1)
            V, m, n = MaxPool2DLayer.pad_csr(V, I.shape, padding)

        ind_dtype = V.indptr.dtype

        # get indexies of feature map
        K = np.zeros([p, q, c], dtype=bool)
        K[:, :, 0] = True
        Z = np.pad(K, ((0, 0), (0, n-q), (0, 0)), mode='constant').reshape(-1)
        Z_indices = np.where(Z > 0)[0]

        noc = no*c
        ko = mo*noc

        i_shift = n * c * stride[0]
        j_shift = stride[1]*c

        # compute max_index and  when applying maxpooling operation
        # compute new number of predicate
        new_c = np.zeros(ko, dtype=dtype)
        new_pred_index_list = []
        max_index_list = []
        max_data = []
        max_row = []
        max_col = []
        i_ = 0
        for i in range(mo):
            for j in range(no):
                for k in range(c):
                    ind = i*i_shift + j*j_shift + k + Z_indices
                    # get local maximum points in the feature map
                    max_index = MaxPool2DLayer.get_localMax_index(I, state_lb[ind], state_ub[ind], ind, lp_solver)
                    if len(max_index) == 1:
                        new_c[i_] = center[max_index[0]]
                        ind_s = V.indptr[max_index[0]]
                        ind_e = V.indptr[max_index[0]+1]
                        max_data.append(V.data[ind_s : ind_e])
                        max_row.append(np.ones(ind_e - ind_s, dtype=ind_dtype) * i_)
                        max_col.append(V.indices[ind_s : ind_e])
                    else:
                        max_index_list.append(max_index)
                        # max_index_list.append(ind)
                        new_pred_index_list.append(i_)

                    i_ += 1
        nPred = len(new_pred_index_list)
        tPred = I.num_pred + nPred
        if show:
            print(f"{nPred} new predicate variables are introduced")

        # construct an over-approximate SparseImageStarCSR reachable set
        if nPred > 0:
            # add new predicate varaibels
            max_data.append(np.ones(nPred, dtype=dtype))
            max_row.append(new_pred_index_list)
            max_col.append(np.arange(nPred, dtype=ind_dtype) + I.num_pred)
            new_V = sp.csr_array((np.hstack(max_data), (np.hstack(max_row), np.hstack(max_col))), shape=(ko, tPred))

            # update constraint matrix C and contraint vector d
            cand_data = []
            cand_row = []
            cand_col = []
            row_track = nPred
            lb = np.zeros(nPred, dtype=dtype)
            ub = np.zeros(nPred, dtype=dtype)
            for i in range(nPred):
                max_index = max_index_list[i]
                N = len(max_index)

                # create C2_list and d2_list
                V1 = V[max_index].tocoo(copy=False)
                cand_data.extend(V1.data)
                cand_row.extend(V1.row + row_track)
                cand_col.extend(V1.col)

                cand_data.extend(-np.ones(N, dtype=dtype))
                cand_row.extend(np.arange(N, dtype=ind_dtype) + row_track)
                cand_col.extend(np.ones(N, dtype=ind_dtype)*i + I.num_pred)
                row_track += N

                lb[i] = state_lb[max_index].min()
                ub[i] = state_ub[max_index].max()

            eye_data = np.ones(nPred, dtype=dtype)
            eye_col = np.arange(nPred, dtype=ind_dtype) + I.num_pred
            eye_row = np.arange(nPred, dtype=ind_dtype)

            # case 1: y <= ub
            # case 2: y >= x
            d1 = ub
            d2 = -center[np.hstack(max_index_list)]

            data = np.hstack([eye_data, cand_data])
            row = np.hstack([eye_row, cand_row])
            col = np.hstack([eye_col, cand_col])
            C = sp.csr_array((data, (row, col)), shape=(row_track, tPred))
            # C.indices = C.indices.astype(ind_dtype)
            # C.indptr = C.indptr.astype(ind_dtype)

            if I.C.nnz > 0:
                data = np.hstack([I.C.data, C.data])
                indices = np.hstack([I.C.indices, C.indices])
                indptr = np.hstack([I.C.indptr, C.indptr[1:]+I.C.nnz])
                new_C = sp.csr_array((data, indices, indptr), shape=(I.C.shape[0]+C.shape[0], C.shape[1]))
                # new_C = sp.csr_array((I.C.shape[0]+C.shape[0], C.shape[1]))
                # data = np.hstack([I.C.data, C.data])
                # new_C.indices = np.hstack([I.C.indices, C.indices])
                # new_C.indptr = np.hstack([I.C.indptr, C.indptr[1:]+I.C.nnz])
                new_d = np.hstack([I.d, d1, d2])
            else:
                new_C = C
                new_d = np.hstack([d1, d2])

            new_pred_lb = np.hstack([I.pred_lb, lb])
            new_pred_ub = np.hstack([I.pred_ub, ub])

            out_shape = (mo, no, c)

            return SparseImageStar2DCSR(new_c, new_V, new_C, new_d, new_pred_lb, new_pred_ub, out_shape)

        new_V = sp.csr_array((np.hstack(max_data), (np.hstack(max_row), np.hstack(max_col))), shape=(ko, tPred))
        out_shape = (mo, no, c)
        return SparseImageStar2DCSR(new_c, new_V, I.C, I.d, I.pred_lb, I.pred_ub, out_shape)

    def reachApproxSingleInput(self, In, lp_solver='gurobi', show=False):
        if isinstance(In, ImageStar):
            if self.module == 'pytorch':
                raise Exception(
                        'BatchNorm2DLayer does not support \'pyotrch\' moudle for SparseImageStar2DCSR set'
                    )
            elif self.module == 'default':
                return self.reach_approx_imagestar(In, lp_solver, show)
        
        elif isinstance(In, SparseImageStar2DCOO):
            if self.module == 'pytorch':
                raise Exception(
                    'BatchNorm2DLayer does not support \'pyotrch\' moudle for SparseImageStar2DCOO set'
                )
            
            elif self.module == 'default':
                return self.reach_approx_fmaxpool2d_coo(In, lp_solver, show)
    
        elif isinstance(In, SparseImageStar2DCSR):
            if self.module == 'pytorch':
                raise Exception(
                    'BatchNorm2DLayer does not support \'pyotrch\' moudle for SparseImageStar2DCSR set'
                )
            
            elif self.module == 'default':
                return self.reach_approx_fmaxpool2d_csr(In, lp_solver, show)
        
        else:
            raise Exception('error: MaxPool2DLayer support ImageStar and SparseImageStar')

    def reachExactSingleInput(self, In, lp_solver, show=False):
        if isinstance(In, ImageStar):
            return self.reach_exact_imagestar(In, lp_solver, show)
    
        # elif isinstance(In, SparseImageStar2DCSR):
            # if self.module == 'pytorch':
            #     raise Exception(
            #         'BatchNorm2DLayer does not support \'pyotrch\' moudle for SparseImageStar2DCSR set'
            #     )
            
            # elif self.module == 'default':
            #     new_c = self.avgpool2d(In.c)
            #     new_V = self.favgpool2d_csr2(In.V)
                
            # return SparseImageStar2DCSR(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        else:
            raise Exception('error: MaxPool2DLayer support ImageStar and SparseImageStar')
        
    def reachExactMultiInputs(self, In, lp_solver='gurobi', pool=None, show=False):
        """
        Exact reachability with multiple inputs
        Works with bread-first-search verification
        """

        assert isinstance(In, list), 'error: input sets should be in a list'
        S = []
        if pool is None:
            for i in range(len(In)):
                S.extend(self.reachExactSingleInput(In[i], lp_solver, show))
        elif isinstance(pool, multiprocessing.pool.Pool):
            S1 = []
            S1 = S1 + pool.map(self.reachExactSingleInput, zip(In, [lp_solver]*len(In)))
            for i in range(len(S1)):
                S.extend(S1[i])
        elif isinstance(pool, ipyparallel.client.view.DirectView):
            # S1 = pool.map(self.reachExactSingleInput, zip(In, [lp_solver]*len(In)))
            # print('S1 = {}'.format(S1))
            raise Exception('error: ipyparallel option is under testing...')
        else:
            raise Exception('error: unknown/unsupport pool type')    
        return S
    

    def reach(self, In, method=None, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """main reachability method
            Args:
                @I: a list of input set

        
        """
        
        if method == 'exact':
            return self.reachExactMultiInputs(In, lp_solver, pool, show)
        elif method == 'approx':
            return self.reachApproxSingleInput(In, lp_solver, show)
        else:
            raise Exception('error: unknown reachability method')