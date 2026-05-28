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
Transposed Convolutional 2D Layer Class
Sung Woo Choi, 07/27/2024
"""

import os
# Keep kernels single-threaded to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import math
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
import time
import copy
import torch
import psutil
import numpy as np
import scipy.sparse as sp
import multiprocessing
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR

class ConvTranspose2DLayer(object):
    """ ConvTranspose2DLayer Class
        
        properties:

        methods:
    
    """
    def __init__(
        self,
        layer, # [kernel_weight, kernel_bias] or torch.nn.ConvTranspose2d
        stride = 1, # e.g.: stride = (2, 2) or [3, 3] or 1
        padding = 0, # e.g.: padding = (0, 0) or [1, 1] or 2
        dilation = 1, # e.g.: dilation = (0, 0) or [1, 1] or 2
        output_padding = 0, # e.g.: output_padding dilation = (0, 0) or [1, 1] or 2
        module = 'default', # 'default' or 'pytorch'
        dtype = 'float64', # 'float64' or 'float32'
        ):

        """
            For default StarV set up, ConvTranspose2dLayer constructor receieves:
                @layer: [kernel_weight, kernel_bias] [in numpy]
                    - kernel_weight: (H, W, Co, Ci); Co: output channel, Ci: input channel, H: height, W: width
                    - kernel_bias: None or (Co); Co: output channel
                
                @module: "default"

                Args in Conv2DLayer:
                    - layer: None 
                    - weight: (H, W, Co, Ci); H: height, W: width, Ci: input channel, Co: output channel [in numpy]
                    - bias: None or (Co); Co: output channel; [in numpy]
                    - stride
                    - padding
                    - dilation
                    - sparse: if True, unrolls weight matrix
                    - in_shape: (H, W, C)
                    - numpy_dtype [in numpy]
                    - torch_dtype [in pytorch]


            For pytorch set up, Conv2DLayer constructor receieves:
                @layer: torch.nn.ConvTranspose2D [in pytorch]
                    - layer.weight: (Ci, Co, H, W); Co: output channel, Ci: input channel, H: height, W: width 
                    - layer.bias: (Co); Co: output channel
                @module: "pytorch"

                Args in Conv2DLayer:
                    - layer: torch.nn.Conv2DTranspose [in pytorch]
                    - weight: None
                    - bias: None or (Co); Co: output channel [in pytorch]
                    - stride
                    - padding
                    - dilation
                    - numpy_dtype [in numpy]
                    - torch_dtype [in pytorch]


        """

        assert module in ['default', 'pytorch'], \
        'error: ConvTranspose2d supports moudles: \'default\', which use numpy kernels, and \'pytorch\''
        self.module = module

        if dtype == 'float32' or dtype == np.float32:
            self.numpy_dtype = np.float32
            self.torch_dtype = torch.float32
        else:
            self.numpy_dtype = np.float64
            self.torch_dtype = torch.float64

        # input 'layer' is list containing [kernel_weight, kernel_bias]
        if isinstance(layer, list):
            assert len(layer) == 2, \
            'error: \'layer\' should be a list containing kernel weight and bias'

            kernel_weight, kernel_bias = copy.deepcopy(layer)
            assert isinstance(kernel_weight, np.ndarray), \
            'error: kernel weight should be a 2D, 3D, or 4D numpy array'

            if kernel_weight.ndim == 2:
                kernel_weight = kernel_weight[:, :, None, None]
            elif kernel_weight.ndim == 3:
                kernel_weight = kernel_weight[:, :, :, None]
            elif kernel_weight.ndim == 4:
                pass
            else:
                raise Exception('error: kernel weight should be a 2D, 3D, or 4D numpy array')

            # kernel weight in shape (kernel_height, kernel_width, ch_out, ch_in)
            self.in_channel = kernel_weight.shape[-1]
            self.out_channel = kernel_weight.shape[-2]

            if kernel_bias is not None:
                assert isinstance(kernel_bias, np.ndarray) and kernel_bias.ndim == 1, \
                'error: kernel bias should be 1D numpy array' 
                assert kernel_bias.shape[0] == kernel_weight.shape[-2], \
                'error: output channel inconsistency between kernel weight and bias'

            if self.module == 'default':
                # check stride, padding, and dilation

                assert isinstance(stride, tuple) or isinstance(stride, list) or \
                       isinstance(stride, int) or isinstance(stride, np.ndarray), \
                f'error: stride should be a tuple, list, numpy ndarray, or int but received {type(stride)}'
                assert isinstance(padding, tuple) or isinstance(padding, list) or \
                       isinstance(padding, int) or isinstance(padding, np.ndarray), \
                f'error: padding should be a tuple, list, numpy ndarray, or int but received {type(padding)}'
                assert isinstance(dilation, tuple) or isinstance(dilation, list) or \
                       isinstance(dilation, int) or isinstance(dilation, np.ndarray), \
                f'error: dilation should be a tuple, list, numpy ndarray, or int but received {type(dilation)}'
                
                if isinstance(padding, int):
                    assert padding >= 0, 'error: padding should non-negative integers'
                    self.padding = np.ones(4, dtype=np.int16)*padding
                else:
                    padding = np.array(padding)
                    assert (padding >= 0).all(), 'error: padding should non-negative integers'

                    if len(padding) == 1:
                        self.padding = np.ones(4, dtype=np.int16)*padding[0]
                    else:
                        if len(padding) == 2:
                            self.padding = np.array([padding[0], padding[0], padding[1], padding[1]]).astype(np.int16)
                        elif len(padding) == 4:
                            self.padding = np.array(padding).astype(np.int16)
                        else:
                            raise Exception('error: padding should contain 1, 2, 4 elements')
                if isinstance(stride, int):
                    assert stride > 0, 'error: stride should positive integer'
                    self.stride = np.ones(2, dtype=np.int16)*stride
                else:
                    if len(stride) == 1:
                        assert stride[0] > 0, 'error: stride should positive integer'
                        self.stride = np.ones(2, dtype=np.int16)*stride[0]
                    elif len(stride) == 2:
                        assert stride[0] > 0 and stride[1] > 0, 'error: stride should positive integer'
                        self.stride = np.array(stride)
                    else:
                        raise Exception('error: incorrect stride')
                    
                if isinstance(dilation, int):
                    assert dilation > 0, 'error: dilation should positive integer'
                    self.dilation = np.ones(2, dtype=np.int16)*dilation
                else:
                    if len(dilation) == 1:
                        assert dilation[0] > 0, 'error: dilation should positive integer'
                        self.dilation = np.ones(2, dtype=np.int16)*dilation[0]
                    elif len(dilation) == 2:
                        assert dilation[0] > 0 and dilation[1] > 0, 'error: dilation should positive integer'
                        self.dilation = np.array(dilation)
                    else:
                        raise Exception('error: incorrect dilation')

                if isinstance(output_padding, int):
                    self.output_padding = np.ones(2, dtype=np.int16)*output_padding
                else:
                    if len(output_padding) == 1:
                        assert output_padding[0] >= 0, 'error: output_padding should non-negative integer'
                        self.output_padding = np.ones(2, dtype=np.int16)*output_padding[0]
                    elif len(output_padding) == 2:
                        assert output_padding[0] >= 0 and output_padding[1] >= 0, 'error: output_padding should non-negative integers'
                        self.output_padding = np.array(output_padding).astype(np.int16)
                    else:
                        raise Exception('error: incorrect output_padding')

                self.weight = kernel_weight.astype(self.numpy_dtype)            
                if kernel_bias is not None:
                    self.bias = kernel_bias.astype(self.numpy_dtype)
                else:
                    self.bias = None

            # converting kernel weight and bias from numpy to torch.nn.ConvTranspose2D
            elif self.module == 'pytorch':

                self.layer = torch.nn.ConvTranspose2d(
                    in_channels = self.in_channel,
                    out_channels = self.out_channel,
                    kernel_size = kernel_weight.shape[:2], #kernel_weight.shape[2:3],
                    stride = stride,
                    padding = padding,
                    output_padding = output_padding,
                    dilation = dilation,
                    bias = False, # self.layer.bias is false as it is stored in self.bias, because bias must not be added to generators
                )
                # change weight in (H, W, Ci, Co) to (Co, Ci, H, W)
                kernel_weight = kernel_weight.transpose([3, 2, 0, 1])
                # self.layer.weight = torch.nn.Parameter(torch.from_numpy(kernel_weight).type(self.torch_dtype))
                self.layer.weight.data = torch.from_numpy(kernel_weight).type(self.torch_dtype)
                if kernel_bias is not None:
                    self.bias = torch.from_numpy(kernel_bias).type(self.torch_dtype)
                else:
                    self.bias = None
                    
                self.stride = self.layer.stride
                self.padding = self.layer.padding
                self.dilation = self.layer.dilation
                self.output_padding = self.layer.output_padding


        # input 'layer' is torch.nn.ConvTranspose2d layer
        elif isinstance(layer, torch.nn.ConvTranspose2d):

            # kernel weight in shape (ch_in, ch_out, kernel_height, kernel_width)
            self.in_channel = layer.weight.shape[0]
            self.out_channel = layer.weight.shape[1]
            
            self.stride = np.array(layer.stride)
            padding = np.array(layer.padding)
            if len(padding) == 2:
                padding = np.array([padding[0], padding[0], padding[1], padding[1]])
            self.padding = padding
            self.dilation = np.array(layer.dilation)
            self.output_padding = np.array(layer.output_padding)
        
            # converting weight and bias in pytorch to numpy 
            if self.module == 'default':
                # self.weight = layer.weight.detach().numpy().astype(self.numpy_dtype).copy()
                self.weight = layer.weight.data.numpy().astype(self.numpy_dtype).copy()
                # change weight in (Ci, Co, H, W) to (H, W, Co, Ci) 
                self.weight = self.weight.transpose([2, 3, 1, 0])
                if layer.bias is None:
                    self.bias = None
                else:
                    self.bias = layer.bias.data.numpy().astype(self.numpy_dtype).copy()

            elif self.module == 'pytorch':
                self.layer = copy.deepcopy(layer)
                self.layer.weight = self.layer.weight.type(self.torch_dtype)
                if self.layer.bias is None:
                    self.bias = None
                else:
                    self.bias = self.layer.bias.type(self.torch_dtype)
                    self.layer.bias = None
                                    
        else:
            raise Exception('Unknown layer module')
        
    def __str__(self):
        print('Transposed Convolutional 2D Layer')
        print('module: {}'.format(self.module))
        print('in_channel: {}'.format(self.in_channel))
        print('out_channel: {}'.format(self.out_channel))
        print('stride: {}'.format(self.stride))
        print('padding: {}'.format(self.padding))
        print('output_padding: {}'.format(self.output_padding))
        print('dilation: {}'.format(self.dilation))

        if self.module == 'pytorch':
            print('weight: {}'.format(self.layer.weight.shape))            
        else:
            print('weight: {}, {}'.format(self.weight.shape, self.weight.dtype))

        if self.bias is not None:
            print('bias: {}, {}'.format(self.bias.shape, self.bias.dtype))
        else:
            print('bias: {}'.format(self.bias))
        return ''

    def apply_padding(input, padding, output_padding):

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'

        in_dim = input.ndim
        assert in_dim <= 4, \
        f'error: number of input array dimensions should be less than 5, but input.ndim = {in_dim}'

        if (padding == 0).all() and (output_padding == 0).all():
            return input
        
        h, w = input.shape[:2]
        
        output = input
        if in_dim == 2:
            output = output[:, :, None, None]
        elif in_dim == 3:
            output = output[:, :, None]
            
        # remove padding
        pad_t = padding[0] 
        pad_b = padding[1] - output_padding[0]
        pad_l = padding[2]
        pad_r = padding[3] - output_padding[1]

        output = output[pad_t:h-pad_b, pad_l:w-pad_r, :, :]

        # add zero padding
        if pad_b < 0 and pad_r < 0:
            output = np.pad(output, ((0, -pad_b), (0, -pad_r), (0, 0), (0,0)), mode='constant')
        elif pad_b < 0:
            output = np.pad(output, ((0, -pad_b), (0,     0), (0, 0), (0,0)), mode='constant')
        elif pad_r < 0:
            output = np.pad(output, ((0,     0), (0, -pad_r), (0, 0), (0,0)), mode='constant')

        # if in_dim == 2:
        #     output = input[:, :, 0, 0]
        # elif in_dim == 3:
        #     output = input[:, :, :, 0]
        return output
    
    def apply_padding_sparse(output, m, n, ci, padding, output_padding, tocsr=False):
        assert output.format == 'csr', \
        f"error: output should be in 'csr' format but received {output.format} format"
        assert isinstance(padding, np.ndarray), \
        f"error: padding should numpy ndarray but received {type(padding)}"

        # applying padding
        pad = (padding > 0).any()
        out_pad = output_padding[0] > 1 or output_padding[1] > 1
        if  pad or out_pad:
            pad_t = padding[0] 
            pad_b = padding[1] - output_padding[0]
            pad_l = padding[2]
            pad_r = padding[3] - output_padding[1]

            # apply padding
            if pad:
                indx = np.arange(m*n*ci).reshape([m, n, ci])[pad_t:m-pad_b, pad_l:n-pad_r, :]
                m, n = indx.shape[:2]
                output = output[indx.reshape(-1), :]

            # add zero padding
            if pad_b < 0 and pad_r < 0:
                return ConvTranspose2DLayer.pad_coo(output, shape=(m, n, ci), padding=(0, -pad_b, 0, -pad_r), tocsr=tocsr)
            elif pad_b < 0:
                return ConvTranspose2DLayer.pad_coo(output, shape=(m, n, ci), padding=(0, -pad_b, 0, 0), tocsr=tocsr)
            elif pad_r < 0:
                return ConvTranspose2DLayer.pad_coo(output, shape=(m, n, ci), padding=(0, 0, 0, -pad_r), tocsr=tocsr)
        
        if tocsr:
            output = output.tocsr(copy=False)
        else:
            output = output.tocoo(copy=False)
        return output, m, n
    

    def pad_coo(input, shape, padding, tocsr=False):
        if len(padding) == 4:
            pad = np.array(padding)
        elif len(padding) == 2:
            pad = np.array([padding[0], padding[0], padding[1], padding[1]])
        elif len(padding) == 1:
            pad = np.ones(4)*padding[0]

        if input.format != 'coo':
            input = input.tocoo()
            

        """Adding padding to coo"""
        row = input.row + (input.row // (shape[1]*shape[2])) * (pad[2]+pad[3])* shape[2]
        row += shape[2]*((shape[1]+pad[2]+pad[3])*pad[0]+pad[2])
        
        mo = shape[0] + pad[0] + pad[1]
        no = shape[1] + pad[2] + pad[3]
        
        if tocsr is True:
            output = sp.csr_array((input.data, (row, input.col)), shape = (mo*no*shape[2], input.shape[1]))
        else:
            output = sp.coo_array((input.data, (row, input.col)), shape = (mo*no*shape[2], input.shape[1]))
        return output, mo, no

    def pad_csr_via_coo(input, shape, padding, tocsc=False):
        """Adding padding to csr"""
        output, mo, no = ConvTranspose2DLayer.pad_coo(input, shape, padding, tocsc=False)
        return output.tocsr(False), mo, no
    
    def pad_csr(input, shape, padding):
        if len(padding) == 4:
            pad = np.array(padding)
        elif len(padding) == 2:
            pad = np.array([padding[0], padding[0], padding[1], padding[1]])
        elif len(padding) == 1:
            pad = np.ones(4)*padding[0]
        
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

    def get_output_size(self, input, with_output_padding=True):

        padding = self.padding
        if len(padding) == 4:
            pad = padding
        elif len(padding) == 2:
            pad = np.array([padding[0], padding[0], padding[1], padding[1]])
        elif len(padding) == 1:
            pad = np.ones(4)*padding[0]
            
        h, w, c, n = input.shape
        H, W = self.weight.shape[:2]

        if with_output_padding:
            ho = (h - 1)*self.stride[0] - pad[0] - pad[1] + self.dilation[0]*(H - 1) + self.output_padding[0] + 1
            wo = (w - 1)*self.stride[1] - pad[2] - pad[3] + self.dilation[1]*(W - 1) + self.output_padding[1] + 1
        else:
            ho = (h - 1)*self.stride[0] - pad[0] - pad[1] + self.dilation[0]*(H - 1) + 1
            wo = (w - 1)*self.stride[1] - pad[2] - pad[3] + self.dilation[1]*(W - 1) + 1

        assert ho > 0 and wo > 0, 'error: the shape of resulting output should be positive'
        return ho, wo
    
    def get_output_size_sparse(self, in_height, in_width):
        padding = self.padding
        if len(padding) == 4:
            pad = padding
        elif len(padding) == 2:
            pad = np.array([padding[0], padding[0], padding[1], padding[1]])
        elif len(padding) == 1:
            pad = np.ones(4)*padding[0]
            
        h, w = in_height, in_width
        H, W = self.weight.shape[:2]

        ho = (h - 1)*self.stride[0] - pad[0] - pad[1] + self.dilation[0]*(H - 1) + self.output_padding[0] + 1
        wo = (w - 1)*self.stride[1] - pad[2] - pad[3] + self.dilation[1]*(W - 1) + self.output_padding[1] + 1

        assert ho > 0 and wo > 0, 'error: the shape of resulting output should be positive'
        return ho, wo
    
    def evaluate(self, input):
        """
            For module == 'default' set up:
                @input: (H, W, C, N); H: height, W: width, C: input channel, N: batch or number of predicates

            For module == 'pytorch' set up:
                @input: (N, C, H, W); N: batch or number of predicates, C: input channel, H: height, W: width 
        """

        return self.convtrans2d_pytorch(input, bias=True)

        # if self.module == 'pytorch':
        #     return self.convtrans2d_pytorch(input, bias=True)
        
        # else:
        #     return self.convtrans2d_basic(input, bias=True)
        

    def convtrans2d_pytorch(self, input, bias=True):
        """
            Args:
               @input: dataset in pytorch with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: transpose convolved dataset
        """
        
        if isinstance(self, ConvTranspose2DLayer):

            if len(self.padding) == 4:
                padding = np.array([self.padding[1], self.padding[3]])
                if self.padding[0] != self.padding[1] or self.padding[2] != self.padding[3]:
                    warnings.warn(f'ConvTranspose2DLayer has a 4-tuple padding, {self.padding}: [t, b, l, r], but torch.nn.ConvTranspose2d does not accept it; passing padding={self.padding}: [h, w]')
            else: padding = self.padding

            # convert StarV ConvTranspose2d to torch.nn.ConvTranspose2d            
            layer = torch.nn.ConvTranspose2d(
                    in_channels = self.in_channel,
                    out_channels = self.out_channel,
                    kernel_size = self.weight.shape[:2],
                    stride = self.stride,
                    padding = padding,
                    output_padding = self.output_padding,
                    dilation = self.dilation,
                    bias = False, # self.layer.bias is false as it is stored in self.bias, because bias must not be added to generators
                )

            # change weight in (H, W, Co, Ci) to (Ci, Co, H, W)
            layer.weight.data = torch.from_numpy(self.weight.transpose([3, 2, 0, 1]))
            layer.bias = torch.nn.Parameter(torch.from_numpy(self.bias)) if bias==True else None
        else:
            assert isinstance(self.layer, torch.nn.ConvTranspose2d), \
            '\'layer\' should be torch.nn.ConvTranspose2d or StarV.layer.ConvTranspose2DLayer.ConvTranspose2DLayer' 

            layer = self.layer
            # layer.bias = torch.nn.Parameter(torch.from_numpy(self.bias)) if bias==True else None
            layer.bias.data = torch.from_numpy(self.bias) if bias==True else None

        # set the layer in evaluation mode
        layer.eval()

        assert isinstance(input, np.ndarray), 'error: input should be numpy ndarray'

        in_dim = input.ndim
        if in_dim == 4:
            H, W, C, N = input.shape
        elif in_dim == 3:
            H, W, C = input.shape
            N = 1
        else:
            raise Exception('input should be either 2D, 3D, or 4D numpy ndarray')
        
        # input = copy.deepcopy(input).reshape(H, W, C, N)
        input = input.reshape(H, W, C, N)
        # change input shape from (H, W, C, N) to (N, C, H, W)
        input = input.transpose([3, 2, 0, 1])
        input = torch.from_numpy(input)
        
        output = layer(input).detach().numpy()
        # change input shape to H, W, C, N
        output = output.transpose([2, 3, 1, 0])
        return output
    
    def convtrans2d_basic(self, input, bias=True):
        """ 
            Basic, transposed convolution 2D

            Args:
            @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
            @R: transpose convolved dataset

        """
        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding
        dilation = self.dilation
        weight = self.weight
        in_dim = input.ndim
        dtype = input.dtype

        assert isinstance(input, np.ndarray), 'error: input should be numpy ndarray'
        assert in_dim >= 2 and in_dim <= 4, 'error: input should be 2D, 3D, or 4D numpy ndarray'

        if in_dim == 2:
            input = input[:, :, None, None]
        elif in_dim == 3:
            input = input[:, :, :, None]

        h, w, c, n = input.shape
        H, W, Co, Ci = weight.shape
        
        dh = dilation[0]*(H - 1) + 1
        dw = dilation[1]*(W - 1) + 1

        ho = (h - 1)*self.stride[0] + dh
        wo = (w - 1)*self.stride[1] + dw

        output = np.zeros((ho, wo, Co, n), dtype=dtype)
                
        for ci_ in range(Ci):
            for h_ in range(h):
                h_stride = h_ * stride[0]
                for w_ in range(w):
                    w_stride = w_ * stride[1]
                    output[h_stride : h_stride + dh : dilation[0], 
                        w_stride : w_stride + dw : dilation[1], 
                        :, :] += weight[:, :, :, ci_, None] * input[h_, w_, ci_, :]
            
        output = ConvTranspose2DLayer.apply_padding(output, padding, output_padding)

        if bias is True:
            if isinstance(self.bias, np.ndarray):
                output += self.bias[None, None, :, None]
                
        return output


    def convtrans2d(self, input, bias=True):
        """ 
            Basic, transposed convolution 2D

            Args:
            @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
            @R: transpose convolved dataset

            #has error in unrolled weight size

        """
        in_dim = input.ndim
        dtype = input.dtype

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'
        assert in_dim >= 2 and in_dim <= 4, \
        'error: input should be 2D, 3D, or 4D numpy ndarray'

        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'

        stride = self.stride
        dilation = self.dilation
        weight = self.weight
        opad = self.output_padding
        pad = self.padding.copy()
        pad[1] -= opad[0]
        pad[3] -= opad[1]

        h, w, _, m = input.shape
        p, q, co, ci = weight.shape
        ho, wo = self.get_output_size_sparse(h, w)
        wh, ww = ho*wo*co, h*w*ci

        lr_pad = pad[2] == 0 and pad[3] == 0

        X = input.reshape(ww, m)

        a = co*((w-1)*stride[1]+ dilation[1]*(q - 1) + 1)
        W = []
        for j in range(p):
            W1 = sp.coo_array(weight[j, :, :, :].reshape(-1, ci))
            W1._shape = (a, ci)
            
            data_, row_, col_ = [], [], []
            for i in range(w):
                row = W1.row.copy()
                
                if dilation[0] > 1:
                    r = W1.row // (co * q)
                    row += r*(dilation[0]-1)*co*q
                if dilation[1] > 1:
                    r = W1.row // co % q
                    row += r*(dilation[1]-1)*co

                data_.append(W1.data)
                col_.append(W1.col + i*ci)
                row_.append(row + i*stride[1]*co)

            # applying padding[2] and padding[3]
            if lr_pad:
                W2 = sp.coo_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(a, ci*(w)))
            else:
                W2 = sp.csr_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(a, ci*(w)))
                a1 = np.zeros([a], dtype=bool)
                if pad[3] > 0:
                    a1[pad[2]*co:-pad[3]*co] = True
                else: a1[pad[2]*co:] = True
                W2 = W2[a1].tocoo()
            W.append(W2)

        a -= (pad[2]+pad[3])*co

        data_, col_, row_ = [], [], []
        for k in range(h):
            for j in range(p):
                t = k*stride[0] + dilation[0]*j
                if t < pad[0]: continue # padding[0]
                if t - pad[1] >= ho + opad[0]: continue # padding[1]

                col = W[j].col + k*w*ci
                row = W[j].row + k*stride[0]*a + j*dilation[0]*a - (pad[0])*a
                data_.append(W[j].data)
                row_.append(row)
                col_.append(col)
        WF = sp.coo_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(wh, ww))
        output = WF @ X
        output = output.reshape(ho, wo, co, m)
        if bias is True:
            if isinstance(self.bias, np.ndarray):
                output += self.bias[None, None, :, None]
        return output
    
    def fconvtrans2d_dense(self, input, bias=True):
        """
            Flattened Convolution 2D for dense images 
            the variable 'input' is considered as the output of the convolution layer

            Args:
                @input: scipy sparse csr matrix with shape of H*W*C, N, where H: height, W: width, C: input channel, N: number of batches
            Return: 
                @R: convolved dataset in csr matrix

            last checked: 09/09/2025 by Sung Woo Choi
        """

        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'

        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding
        dilation = self.dilation
        weight = self.weight

        assert isinstance(input, np.ndarray) or isinstance(input, np.ndarray), \
        'error: input should be a numpy array'
        
        mo, no, _, b = input.shape
        p, q, ci, co = weight.shape

        m = (mo - 1)*self.stride[0] + dilation[0]*(p - 1) + 1
        n = (no - 1)*self.stride[1] + dilation[1]*(q - 1) + 1

        i_shift = n * stride[0] * ci
        j_shift = stride[1] * ci

        ko = mo*no

        Z = np.pad(weight, ((0, m-p), (0, n-q), (0, 0), (0,0)), mode='constant') # in [m, n, ci, co] shape
        Z_ = sp.csr_array(Z.reshape(np.prod(Z.shape[:3]), co).T, copy=False)
        nnz = Z_.indptr[1:] - Z_.indptr[:-1]
        Z_ind = Z_.indices.copy()

        if dilation[0] > 1:
            ind = Z_.indices // (ci * n)
            Z_ind += ind*(dilation[0]-1)*ci*n
    
        if dilation[1] > 1:
            ind = Z_.indices // ci % n
            Z_ind += ind*(dilation[1]-1)*ci

        data = np.repeat(Z_.data[None, :], ko, axis=0).reshape(-1)

        indices = np.arange(ko, dtype=np.int32)[:, None]
        indices = ((indices//no)*i_shift + (indices%no)*j_shift + Z_ind).reshape(-1)

        indptr = np.hstack([Z_.indptr, ((np.arange((ko-1)*co, dtype=np.int32) + 1 + Z_.shape[0]).reshape(ko-1, co) * nnz).reshape(-1)])

        TZ = sp.csc_array((data, indices, indptr), shape=(Z_.shape[1], ko*co), copy=False) # in [m*n*ci, mo*no*co]
        input = input.reshape(ko*co, b) # in [mo*no*co, b] shape
        output = TZ @ input # in [m*n*ci, b] shape
        output = output.reshape(m, n, ci, b)
        output = ConvTranspose2DLayer.apply_padding(output, padding, output_padding)

        if bias is True:
            if isinstance(self.bias, np.ndarray):
                output += self.bias[None, None, :, None]
        return output
    
    def fconvtrans2d_coo2(self, input, shape):

        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'

        stride = self.stride
        dilation = self.dilation
        weight = self.weight
        opad = self.output_padding
        pad = self.padding.copy()
        pad[1] -= opad[0]
        pad[3] -= opad[1]

        assert isinstance(input, sp.coo_array) or isinstance(input, sp.coo_matrix), \
        'error: input should be a scipy sparse coo array or matrix'

        h, w, _ = shape
        p, q, co, ci = weight.shape
        ho, wo = self.get_output_size_sparse(h, w)
        wh, ww = ho*wo*co, h*w*ci

        lr_pad = pad[2] == 0 and pad[3] == 0

        a = co*((w-1)*stride[1]+ dilation[1]*(q - 1) + 1)
        W = []
        for j in range(p):
            data_, row_, col_ = [], [], []
            for i in range(w):
                W1 = sp.coo_array(weight[j, :, :, :].reshape(-1, ci))
                W1._shape = (a, ci)
                row = W1.row.copy()

                if dilation[0] > 1:
                    r = W1.row // (co * q)
                    row += r*(dilation[0]-1)*co*q
                if dilation[1] > 1:
                    r = W1.row // co % q
                    row += r*(dilation[1]-1)*co

                data_.append(W1.data)
                col_.append(W1.col + i*ci)
                row_.append(row + i*stride[1]*co)

            # applying padding[2] and padding[3]
            if lr_pad:
                W2 = sp.coo_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(a, ci*(w)))
            else:
                W2 = sp.csr_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(a, ci*(w)))
                a1 = np.zeros([a], dtype=bool)
                if pad[3] > 0:
                    a1[pad[2]*co:-pad[3]*co] = True
                else: a1[pad[2]*co:] = True
                W2 = W2[a1].tocoo()
            W.append(W2)

        a -= (pad[2]+pad[3])*co

        data_, col_, row_ = [], [], []
        for k in range(h):
            for j in range(p):
                t = k*stride[0] + dilation[0]*j
                if t < pad[0]: continue # padding[0]
                if t - pad[1] >= ho + opad[0]: continue # padding[1]

                col = W[j].col + k*w*ci
                row = W[j].row + k*stride[0]*a + j*dilation[0]*a - (pad[0])*a
                data_.append(W[j].data)
                row_.append(row)
                col_.append(col)
        WF = sp.coo_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(wh, ww))
        output = (WF @ input).tocoo(copy=False)

        out_shape = (ho, wo, co)
        return output, out_shape
    

    def fconvtrans2d_coo(self, input, shape):
        """
            Flattened Convolution 2D for sparse 2D images 
            This method does not support bias vector

            Args:
                @input: scipy sparse coo matrix with shape of H*W*C, N, where H: height, W: width, C: input channel, N: number of batches
            Return: 
                @R: convolved dataset in coo matrix
        """

        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'

        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding
        dilation = self.dilation
        weight = self.weight

        assert isinstance(input, sp.coo_array) or isinstance(input, sp.coo_matrix), \
        'error: input should be a scipy sparse coo array or matrix'

        mo, no, _ = shape
        p, q, ci, co = weight.shape

        m = (mo - 1)*self.stride[0] + dilation[0]*(p - 1) + 1 # - 2*self.padding[0]
        n = (no - 1)*self.stride[1] + dilation[1]*(q - 1) + 1 # - 2*self.padding[1]
        
        i_shift = n * stride[0] * ci
        j_shift = stride[1] * ci

        ko = mo*no

        Z = np.pad(weight, ((0, m-p), (0, n-q), (0, 0), (0,0)), mode='constant')
        Z_ = sp.csr_array(Z.reshape(np.prod(Z.shape[:3]), co).T, copy=False)
        nnz = Z_.indptr[1:] - Z_.indptr[:-1]
        Z_ind = Z_.indices.copy()

        if dilation[0] > 1:
            q_ind = Z_.indices // ci % q #col
            Z_ind += q_ind*(dilation[0]-1)*ci

        if dilation[1] > 1:
            p_ind = Z_.indices // (ci*q) #row
            Z_ind += p_ind*(dilation[1]-1)*ci*q
        
        data = np.repeat(Z_.data[None, :], ko, axis=0).reshape(-1)
        
        indices = np.arange(ko, dtype=np.int32)[:, None]
        indices = ((indices//no)*i_shift + (indices%no)*j_shift + Z_ind).reshape(-1)
        
        indptr = np.hstack([Z_.indptr, ((np.arange((ko-1)*co, dtype=np.int32) + 1 + Z_.shape[0]).reshape(ko-1, co) * nnz).reshape(-1)] )
        
        TZ = sp.csc_array((data, indices, indptr), shape=(Z_.shape[1], ko*co), copy=False)
        output = (TZ @ input).tocsr()
        
        output, m, n = ConvTranspose2DLayer.apply_padding_sparse(output, m, n, ci, padding, output_padding)
        out_shape = (m, n, ci)
        return output, out_shape
    
    @staticmethod
    def _build_Wj(j, weight, a, ci, w, co, q, stride, dilation, pad, no_lr_pad):
        W1 = sp.coo_array(weight[j, :, :, :].reshape(-1, ci))
        W1._shape = (a, ci)

        data_, row_, col_ = [], [], []
        for i in range(w):
            row = W1.row.copy()
            if dilation[0] > 1:
                r = W1.row // (co * q)
                row += r * (dilation[1] - 1) * co * q
            if dilation[1] > 1:
                r = W1.row // co % q
                row += r * (dilation[1] - 1) * co
            data_.append(W1.data)
            col_.append(W1.col + i * ci)
            row_.append(row + i * stride[1] * co)

        # applying padding[2] and padding[3]
        if no_lr_pad:
            W2 = sp.coo_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(a, ci * w))
        else:
            W2 = sp.csr_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(a, ci * w))
            a1 = np.zeros(a, dtype=bool)
            if pad[3] > 0:
                a1[pad[2] * co : -pad[3] * co] = True
            else:
                a1[pad[2] * co :] = True
            W2 = W2[a1].tocoo()
        return j, W2
    
#################################################
    @staticmethod
    def _available_mem_bytes():
        try:
            return psutil.virtual_memory().available
        except Exception:
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            return int(line.split()[1]) * 1024
            except Exception:
                pass
        return 32 * 1024**3  # fallback ~32GB

    @staticmethod
    def _estimate_task_bytes(weight_slice, a, ci, w, no_lr_pad, dtype):
        # Peak memory building one W[j]: COO build then optional CSR slice
        W1 = sp.coo_array(weight_slice.reshape(-1, ci))
        nnz_total = W1.nnz * w
        data_sz = np.dtype(dtype).itemsize
        idx64, idx32 = 8, 4
        W1_peak = nnz_total * (data_sz + 2 * idx32)  # W1 coo format: data + row + col
        W2_peak = W1_peak if no_lr_pad else nnz_total * (data_sz + idx32) + (a + 1) * idx32 # for single W2 coo/csr format
        return W2_peak + W1_peak

    @staticmethod
    def _choose_nthreads_by_memory(weight, a, ci, w, p, no_lr_pad, headroom=0.9):
        avail = ConvTranspose2DLayer._available_mem_bytes()
        # print('avail: ', avail)
        sample_js = [j for j in {0, p // 2, p - 1} if 0 <= j < p] or list(range(p))
        # print('sample_js: ', sample_js)
        est_bytes = 0
        for j in sample_js:
            est_bytes = max(est_bytes, ConvTranspose2DLayer._estimate_task_bytes(weight[j, :, :, :], a, ci, w, no_lr_pad, weight.dtype))
        if est_bytes <= 0:
            est_bytes = 20 * 1024**3  # ~20GB/task fallback
        mem_threads = max(1, int((avail * headroom) // est_bytes))
        cpu_threads = max(1, os.cpu_count() or 1)
        return max(1, min(p, mem_threads, cpu_threads))

    @staticmethod
    def _choose_k_threads_by_memory(W, a, ww, p, h, headroom=0.9):
        # Estimate worst-case assembled COO footprint per k (sum of W[j] nnz)
        try:
            avail = psutil.virtual_memory().available
        except Exception:
            try:
                with open("/proc/meminfo") as f:
                    avail = next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:"))
            except Exception:
                avail = 32 * 1024**3

        total_nnz = 0
        for j in range(p):
            Wj = W[j]
            if not sp.isspmatrix_coo(Wj):
                try:
                    Wj = Wj.tocoo(copy=False)
                except TypeError:
                    Wj = Wj.tocoo()
            total_nnz += Wj.nnz

        # bytes per nnz in assembled COO (data + int64 row + int64 col)
        if p > 0 and W[0].nnz > 0 and sp.isspmatrix_coo(W[0]):
            data_itemsize = W[0].data.dtype.itemsize
        else:
            data_itemsize = 8  # assume float64
        bytes_per_nnz = data_itemsize + 16
        safety = 1.2
        peak_bytes = int(total_nnz * bytes_per_nnz * safety)

        mem_threads = max(1, int((avail * headroom) // max(1, peak_bytes)))
        cpu_threads = max(1, os.cpu_count() or 1)
        # print('avail memory: ', avail)
        # print('peak_bytes: ', peak_bytes)
        # print('mem_threads: ', mem_threads)
        # print('cpu_threads: ', cpu_threads)
        return max(1, min(mem_threads, cpu_threads, h))

    @staticmethod
    def fconvtrans2d_block_weight__(stride, dilation, weight, pad, w, co, ci, p, q, threads="auto"):
        no_lr_pad = (pad[2] == 0 and pad[3] == 0)
        a = co * ((w - 1) * stride[1] + dilation[1] * (q - 1) + 1)

        if threads is None or threads == 0:
            W = []
            for j in range(p):
                _, W2 = ConvTranspose2DLayer._build_Wj(j, weight, a, ci, w, co, q, stride, dilation, pad, no_lr_pad)
                W.append(W2)
            a_out = a if no_lr_pad else a - (pad[2] + pad[3]) * co
            return W, a_out

        if threads == "auto":
            nthreads = ConvTranspose2DLayer._choose_nthreads_by_memory(weight, a, ci, w, p, no_lr_pad)
        else:
            nthreads = max(1, min(int(threads), os.cpu_count() or 1, p))

        W = [None] * p
        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            futures = {
                ex.submit(ConvTranspose2DLayer._build_Wj, j, weight, a, ci, w, co, q, stride, dilation, pad, no_lr_pad): j
                for j in range(p)
            }
            for f in as_completed(futures):
                j_out, W2 = f.result()
                W[j_out] = W2

        a_out = a if no_lr_pad else a - (pad[2] + pad[3]) * co
        return W, a_out

    @staticmethod
    def fconvtrans2d_csr_row_slice__(start, end, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p, return_coo=False):
        # Memory-aware row-slice
        if return_coo:
            out = sp.coo_array((wh, input.shape[1]), dtype=input.dtype) 
        else:
            out = sp.csr_array((wh, input.shape[1]), dtype=input.dtype) 

        # Precompute COO and nnz per j
        W_coo = []
        nnz_per_j = []
        for j in range(p):
            Wj = W[j]
            if not sp.isspmatrix_coo(Wj):
                try:
                    Wj = Wj.tocoo(copy=False)
                except TypeError:
                    Wj = Wj.tocoo()
            W_coo.append(Wj)
            nnz_per_j.append(Wj.nnz)

        for k in range(start, end):
            valid_j = []
            for j in range(p):
                t = k * stride[0] + dilation[0] * j
                if t < pad[0] or (t - pad[1]) >= ho + opad[0]:
                    continue
                valid_j.append(j)
            if not valid_j:
                continue

            batch = []

            def flush_batch(batch_js):
                nonlocal out  # allow augmented assignment on outer 'out'
                if not batch_js:
                    return
                data_, row_, col_ = [], [], []
                total_new = 0
                for jb in batch_js:
                    Wb = W_coo[jb]
                    if Wb.nnz == 0:
                        continue
                    data_.append(Wb.data)
                    col_.append(Wb.col + k * w * ci)
                    row_.append(Wb.row + jb * dilation[0] * a)
                    total_new += Wb.nnz

                if total_new == 0:
                    return
                
                data_ = np.hstack(data_)
                row_ = np.hstack(row_)
                col_ = np.hstack(col_)

                WF = sp.coo_array((data_, (row_, col_)), shape=(p * a, ww))
                part = (WF @ input).tocoo()
                part._shape = (wh, input.shape[1])
                part.row += (k * stride[0] - pad[0]) * a
                out += part

                del data_, row_, col_, WF
                gc.collect()

            # Single batch flush (you can add memory-based batching if needed)
            batch = valid_j
            flush_batch(batch)

        return out

    def fconvtrans2d_csr4(self, input, shape, threads=None):

        assert self.module == 'default', "error: conv2d_sparse() supports 'default' module"
        
        stride = self.stride
        dilation = self.dilation
        weight = self.weight
        opad = self.output_padding
        pad = self.padding.copy()

        pad[1] -= opad[0]
        pad[3] -= opad[1]
        
        assert isinstance(input, (sp.csr_array, sp.csr_matrix)), \
        "error: input should be a scipy sparse csr array or matrix"

        h, w, _ = shape
        p, q, co, ci = weight.shape

        ho, wo = self.get_output_size_sparse(h, w)
        
        wh, ww = ho * wo * co, h * w * ci

        # Build W blocks with memory-aware threading and return as COO
        W, a = ConvTranspose2DLayer.fconvtrans2d_block_weight__(
            stride, dilation, weight, pad, w, co, ci, p, q, threads="auto"
        )

        output = sp.csr_array((wh, input.shape[1]), dtype=input.dtype)

        if threads is None or threads == 0:
            for k in range(h):
                part = ConvTranspose2DLayer.fconvtrans2d_csr_row_slice__(
                    k, k + 1, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p)
                if part.nnz:
                    output += part
        else:
            # Memory-aware selection of k-loop threads
            if threads == "auto":
                nthreads = ConvTranspose2DLayer._choose_k_threads_by_memory(W, a, ww, p, h)
            else:
                nCPUs = os.cpu_count() or 1
                nthreads = max(1, min(int(threads), nCPUs, h))

            nchunks = nthreads
            chunk_size = math.ceil(h / nchunks)
            ranges = [(s, min(s + chunk_size, h)) for s in range(0, h, chunk_size)]

            with ThreadPoolExecutor(max_workers=nthreads) as ex:
                futures = [
                    ex.submit(ConvTranspose2DLayer.fconvtrans2d_csr_row_slice__,
                              s, e, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p)
                    for (s, e) in ranges
                ]
                merged = 0
                for f in as_completed(futures):
                    part = f.result()
                    if part.nnz:
                        output += part
                    del part
                    merged += 1
                    if merged % 2 == 0:
                        gc.collect()

        out_shape = (ho, wo, co)
        return output, out_shape

    def fconvtrans2d_coo4(self, input, shape, threads=None):

        assert self.module == 'default', "error: conv2d_sparse() supports 'default' module"
        
        stride = self.stride
        dilation = self.dilation
        weight = self.weight
        opad = self.output_padding
        pad = self.padding.copy()
        pad[1] -= opad[0]
        pad[3] -= opad[1]
        
        assert isinstance(input, (sp.coo_array, sp.coo_matrix)), \
        "error: input should be a scipy sparse csr array or matrix"

        h, w, _ = shape
        p, q, co, ci = weight.shape

        ho, wo = self.get_output_size_sparse(h, w)
        
        wh, ww = ho * wo * co, h * w * ci

        # Build W blocks with memory-aware threading and return as COO
        W, a = ConvTranspose2DLayer.fconvtrans2d_block_weight__(
            stride, dilation, weight, pad, w, co, ci, p, q, threads="auto"
        )

        output = sp.csr_array((wh, input.shape[1]), dtype=input.dtype)

        if threads is None or threads == 0:
            for k in range(h):
                part = ConvTranspose2DLayer.fconvtrans2d_csr_row_slice__(
                    k, k + 1, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p)
                if part.nnz:
                    output += part
                
        else:
            # Memory-aware selection of k-loop threads
            if threads == "auto":
                nthreads = ConvTranspose2DLayer._choose_k_threads_by_memory(W, a, ww, p, h)
            else:
                nCPUs = os.cpu_count() or 1
                nthreads = max(1, min(int(threads), nCPUs, h))

            nchunks = nthreads
            chunk_size = math.ceil(h / nchunks)
            ranges = [(s, min(s + chunk_size, h)) for s in range(0, h, chunk_size)]

            with ThreadPoolExecutor(max_workers=nthreads) as ex:
                futures = [
                    ex.submit(ConvTranspose2DLayer.fconvtrans2d_csr_row_slice__,
                              s, e, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p)
                    for (s, e) in ranges
                ]
                merged = 0
                for f in as_completed(futures):
                    part = f.result()
                    if part.nnz:
                        output += part
                    del part
                    merged += 1
                    if merged % 2 == 0:
                        gc.collect()

        # coo + coo -> csr (no reason to keep format in coo; convert the format to coo at the last step)
        out_shape = (ho, wo, co)
        return output.tocoo(), out_shape

######################################################
    
    def fconvtrans2d_csr3(self, input, shape, threads=None):

        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'

        stride = self.stride
        dilation = self.dilation
        weight = self.weight
        opad = self.output_padding
        pad = self.padding.copy()
        pad[1] -= opad[0]
        pad[3] -= opad[1]

        assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
        'error: input should be a scipy sparse csr array or matrix'

        h, w, _ = shape
        p, q, co, ci = weight.shape

        ho, wo = self.get_output_size_sparse(h, w)

        wh, ww = ho*wo*co, h*w*ci

        # W_coo, a = ConvTranspose2DLayer.fconvtrans2d_block_weight(stride, dilation, weight, pad, w, co, ci, p, q, threads=4)
        W, a = ConvTranspose2DLayer.fconvtrans2d_block_weight(stride, dilation, weight, pad, w, co, ci, p, q)

        output = sp.csr_array((wh, input.shape[1]), dtype=input.dtype)       
        if threads is None:
            for k in range(h):
                output += ConvTranspose2DLayer.fconvtrans2d_csr_row_slice(k, k+1, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p)

            out_shape = (ho, wo, co)
            return output, out_shape
        
        
        assert isinstance(threads, int) and threads > 0, \
        'error: threads should be a positive integer or None'

        nCPUs = os.cpu_count() or 1
        nthreads = min(4, nCPUs, threads, h)  # at most 4 threads, not more than CPUs or h
        print('nthreads: ', nthreads)

        # One chunk per thread to minimize intermediate results
        nchunks = nthreads
        chunk_size = math.ceil(h / nchunks)
        ranges = [(s, min(s + chunk_size, h)) for s in range(0, h, chunk_size)]

        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            futures = [
                ex.submit(ConvTranspose2DLayer.fconvtrans2d_csr_row_slice, s, e, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p)
                for (s, e) in ranges
            ]
            merged = 0
            for f in as_completed(futures):
                part = f.result()
                if part.nnz:  # skip empty
                    output += part
                del part
                merged += 1
                if merged % 2 == 0:  # occasional GC to keep memory in check
                    gc.collect()
                
        out_shape = (ho, wo, co)
        return output, out_shape
    
    def fconvtrans2d_coo3(self, input, shape, threads=None):

        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'

        stride = self.stride
        dilation = self.dilation
        weight = self.weight
        opad = self.output_padding
        pad = self.padding.copy()
        pad[1] -= opad[0]
        pad[3] -= opad[1]

        assert isinstance(input, sp.coo_array) or isinstance(input, sp.coo_matrix), \
        'error: input should be a scipy sparse coo array or matrix'

        h, w, _ = shape
        p, q, co, ci = weight.shape

        ho, wo = self.get_output_size_sparse(h, w)

        wh, ww = ho*wo*co, h*w*ci

        # W_coo, a = ConvTranspose2DLayer.fconvtrans2d_block_weight(stride, dilation, weight, pad, w, co, ci, p, q, threads=4)
        W, a = ConvTranspose2DLayer.fconvtrans2d_block_weight(stride, dilation, weight, pad, w, co, ci, p, q)

        output = sp.coo_array((wh, input.shape[1]), dtype=input.dtype)       
        if threads is None:
            for k in range(h):
                output += ConvTranspose2DLayer.fconvtrans2d_csr_row_slice(k, k+1, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p)

            out_shape = (ho, wo, co)
            if output.format == 'csr':
                output = output.tocoo()
            return output, out_shape
        
        
        assert isinstance(threads, int) and threads > 0, \
        'error: threads should be a positive integer or None'

        nCPUs = os.cpu_count() or 1
        nthreads = min(4, nCPUs, threads, h)  # at most 4 threads, not more than CPUs or h
        print('nthreads: ', nthreads)

        # One chunk per thread to minimize intermediate results
        nchunks = nthreads
        chunk_size = math.ceil(h / nchunks)
        ranges = [(s, min(s + chunk_size, h)) for s in range(0, h, chunk_size)]

        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            futures = [
                ex.submit(ConvTranspose2DLayer.fconvtrans2d_csr_row_slice, s, e, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p)
                for (s, e) in ranges
            ]
            merged = 0
            for f in as_completed(futures):
                part = f.result()
                if part.nnz:  # skip empty
                    output += part
                del part
                merged += 1
                if merged % 2 == 0:  # occasional GC to keep memory in check
                    gc.collect()
                
        out_shape = (ho, wo, co)
        if output.format == 'csr':
            output = output.tocoo(copy=False)
        return output, out_shape

    @staticmethod
    def fconvtrans2d_block_weight(stride, dilation, weight, pad, w, co, ci, p, q, threads=None):
        # ConvTranspose2DLayer.fconvtrans2d_block_weight(stride, dilation, weight, pad, w, co, ci, p, q)
        
        no_lr_pad = pad[2] == 0 and pad[3] == 0

        a = co*((w-1)*stride[1]+ dilation[1]*(q - 1) + 1)

        if threads is not None:
            nthreads = min(os.cpu_count()*2 or 1, p)
            W = [None] * p
            with ThreadPoolExecutor(max_workers=nthreads) as ex:
                futures = {
                    ex.submit(ConvTranspose2DLayer._build_Wj, j, weight, a, ci, w, co, q, stride, dilation, pad, no_lr_pad): j
                    for j in range(p)
                }
                for f in as_completed(futures):
                    j, W2 = f.result()
                    W[j] = W2

            a -= (pad[2]+pad[3])*co
            return W, a
        
        W = []
        for j in range(p):
            W1 = sp.coo_array(weight[j, :, :, :].reshape(-1, ci), copy=False)   # in [q*co, ci] shape; small column block in A_i
            W1._shape = (a, ci)                                     # making W1 := A_i, in [a, ci] shape

            data_, row_, col_ = [], [], []
            for i in range(w):
                row = W1.row.copy()

                if dilation[0] > 1:
                    r = W1.row // (co * q)
                    row += r*(dilation[0]-1)*co*q
                if dilation[1] > 1:
                    r = W1.row // co % q
                    row += r*(dilation[1]-1)*co

                data_.append(W1.data)
                col_.append(W1.col + i*ci)
                row_.append(row + i*stride[1]*co)
            
            # applying padding[2] and padding[3]
            if no_lr_pad:
                W2 = sp.coo_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(a, ci*(w)), copy=False)    # block A_i
            else:
                W2 = sp.csr_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(a, ci*(w)), copy=False)    # block A_i
                a1 = np.zeros([a], dtype=bool)                   # a1 is to remove rows corresponding to padding[2] and padding[3]
                if pad[3] > 0:
                    a1[pad[2]*co:-pad[3]*co] = True
                else: a1[pad[2]*co:] = True
                W2 = W2[a1].tocoo()
            W.append(W2)                                # collects all A_i's: W = [A_0, A_1, ..., A_(p-1)]

        a -= (pad[2]+pad[3])*co

        return W, a
    
    
    # @staticmethod
    # def fconvtrans2d_csr_row_slice(start, end, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p):
    #     # for k in range(h):
    #     #         output = ConvTranspose2DLayer.fconvtrans2d_csr_row_slice(0, p, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p)

    #     if wh < W[0].nnz*p:
    #         add_csr_format=True
    #     else:
    #         add_csr_format=False

    #     output = sp.csr_array((wh, input.shape[1]), dtype=input.dtype)
    #     for k in range(start, end):
    #         data_, col_, row_ = [], [], []
    #         for j in range(p):
    #             t = k*stride[0] + dilation[0]*j
    #             if t < pad[0] or (t - pad[1]) >= ho + opad[0]:
    #                 continue
    #             col = W[j].col + k*w*ci
    #             if add_csr_format:
    #                 row = W[j].row + k*stride[0]*a + j*dilation[0]*a - pad[0]*a
    #             else:
    #                 row = W[j].row + j*dilation[0]*a
    #             data_.append(W[j].data)
    #             row_.append(row)
    #             col_.append(col)

    #         if add_csr_format:
    #             WF = sp.coo_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(wh, ww))
    #             output += WF @ input
    #         else:
    #             WF = sp.coo_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(p*a, ww))
    #             out_coo = (WF @ input).tocoo()
    #             out_coo._shape = (wh, input.shape[1])
    #             out_coo.row += (k*stride[0] - pad[0])*a 
    #             output += out_coo

    #     return output

    @staticmethod
    def fconvtrans2d_csr_row_slice(start, end, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p):
        # for k in range(h):
        #         output = ConvTranspose2DLayer.fconvtrans2d_csr_row_slice(0, p, W, input, stride, dilation, pad, ho, opad, w, ci, a, wh, ww, p)

        output = sp.csr_array((wh, input.shape[1]), dtype=input.dtype)
        for k in range(start, end):
            data_, col_, row_ = [], [], []
            for j in range(p):
                t = k*stride[0] + dilation[0]*j
                if t < pad[0] or (t - pad[1]) >= ho + opad[0]:
                    continue
                col = W[j].col + k*w*ci
                row = W[j].row + j*dilation[0]*a
                data_.append(W[j].data)
                row_.append(row)
                col_.append(col)

            WF = sp.coo_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(p*a, ww))
            out_coo = (WF @ input).tocoo()
            out_coo._shape = (wh, input.shape[1])
            out_coo.row += (k*stride[0] - pad[0])*a 
            output += out_coo

        return output

    def fconvtrans2d_csr2(self, input, shape):

        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'

        stride = self.stride
        dilation = self.dilation
        weight = self.weight
        opad = self.output_padding
        pad = self.padding.copy()
        pad[1] -= opad[0]
        pad[3] -= opad[1]

        assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
        'error: input should be a scipy sparse csr array or matrix'

        h, w, _ = shape
        p, q, co, ci = weight.shape
        ho, wo = self.get_output_size_sparse(h, w)
        wh, ww = ho*wo*co, h*w*ci

        lr_pad = pad[2] == 0 and pad[3] == 0

        a = co*((w-1)*stride[1]+ dilation[1]*(q - 1) + 1)
        W = []
        for j in range(p):
            W1 = sp.coo_array(weight[j, :, :, :].reshape(-1, ci))
            W1._shape = (a, ci)
            
            data_, row_, col_ = [], [], []
            for i in range(w):
                row = W1.row.copy()

                if dilation[0] > 1:
                    r = W1.row // (co * q)
                    row += r*(dilation[0]-1)*co*q
                if dilation[1] > 1:
                    r = W1.row // co % q
                    row += r*(dilation[1]-1)*co

                data_.append(W1.data)
                col_.append(W1.col + i*ci)
                row_.append(row + i*stride[1]*co)

            # applying padding[2] and padding[3]
            if lr_pad:
                W2 = sp.coo_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(a, ci*(w)))
            else:
                W2 = sp.csr_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(a, ci*(w)))
                a1 = np.zeros([a], dtype=bool)
                if pad[3] > 0:
                    a1[pad[2]*co:-pad[3]*co] = True
                else: a1[pad[2]*co:] = True
                W2 = W2[a1].tocoo()
            W.append(W2)

        a -= (pad[2]+pad[3])*co

        output = sp.csr_array((wh, input.shape[1]), dtype=input.dtype)
        for k in range(h):
            data_, col_, row_ = [], [], []
            for j in range(p):
                t = k*stride[0] + dilation[0]*j
                if t < pad[0]: continue # padding[0]
                if t - pad[1] >= ho + opad[0]: continue # padding[1]

                col = W[j].col + k*w*ci
                row = W[j].row + j*dilation[0]*a #+ k*stride[0]*a - (pad[0])*a 
                data_.append(W[j].data)
                row_.append(row)
                col_.append(col)
            WF = sp.coo_array((np.hstack(data_), (np.hstack(row_), np.hstack(col_))), shape=(p*a, ww))
            out_coo = (WF @ input).tocoo()
            out_coo._shape = (wh, input.shape[1])
            out_coo.row += (k*stride[0] - pad[0])*a 
            output += out_coo

        out_shape = (ho, wo, co)
        return output, out_shape

    def fconvtrans2d_csr(self, input, shape):
        """
            Flattened Convolution 2D for sparse 2D images 
            This method does not support bias vector

            Args:
                @input: scipy sparse csr matrix with shape of H*W*C, N, where H: height, W: width, C: input channel, N: number of batches
            Return: 
                @R: convolved dataset in csr matrix
        """

        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'

        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding
        dilation = self.dilation
        weight = self.weight

        assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
        'error: input should be a scipy sparse csr array or matrix'
        
        mo, no, _ = shape
        p, q, ci, co = weight.shape

        m = (mo - 1)*self.stride[0] + dilation[0]*(p - 1) + 1 # - 2*self.padding[0]
        n = (no - 1)*self.stride[1] + dilation[1]*(q - 1) + 1 # - 2*self.padding[1]

        i_shift = n * stride[0] * ci
        j_shift = stride[1] * ci

        ko = mo*no

        Z = np.pad(weight, ((0, m-p), (0, n-q), (0, 0), (0,0)), mode='constant')
        Z_ = sp.csr_array(Z.reshape(np.prod(Z.shape[:3]), co).T, copy=False)
        nnz = Z_.indptr[1:] - Z_.indptr[:-1]
        Z_ind = Z_.indices.copy()

        if dilation[0] > 1:
            q_ind = Z_.indices // ci % q #col
            Z_ind += q_ind*(dilation[0]-1)*ci

        if dilation[1] > 1:
            p_ind = Z_.indices // (ci*q) #row
            Z_ind += p_ind*(dilation[1]-1)*ci*q

        data = np.repeat(Z_.data[None, :], ko, axis=0).reshape(-1)

        indices = np.arange(ko, dtype=np.int32)[:, None]
        indices = ((indices//no)*i_shift + (indices%no)*j_shift + Z_ind).reshape(-1)

        indptr = np.hstack([Z_.indptr, ((np.arange((ko-1)*co, dtype=np.int32) + 1 + Z_.shape[0]).reshape(ko-1, co) * nnz).reshape(-1)])

        TZ = sp.csc_array((data, indices, indptr), shape=(Z_.shape[1], ko*co), copy=False)
        output = (TZ @ input).tocsr()

        output, m, n = ConvTranspose2DLayer.apply_padding_sparse(output, m, n, ci, padding, output_padding, tocsr=True)
        out_shape = (m, n, ci)
        return output, out_shape


    def fconvtrans2d_coo_co_loop(self, input, shape):
        """
            Flattened Convolution 2D for sparse 2D images 
            This method does not support bias vector

            Args:
                @input: scipy sparse coo matrix with shape of H*W*C, N, where H: height, W: width, C: input channel, N: number of batches
            Return: 
                @R: convolved dataset in coo matrix
        """
        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'

        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding
        dilation = self.dilation
        weight = self.weight
        dtype = input.dtype

        assert isinstance(input, sp.coo_array) or isinstance(input, sp.coo_matrix), \
        'error: input should be a scipy sparse coo array or matrix'
        
        b = input.shape[1]
        mo, no, _ = shape
        p, q, ci, co = weight.shape

        mi = (mo - 1)*self.stride[0] + dilation[0]*(p - 1) + 1
        ni = (no - 1)*self.stride[1] + dilation[1]*(q - 1) + 1
        m, n = mi, ni

        K = np.pad(weight, ((0, 0), (0, ni-q), (0, 0), (0,0)), mode='constant') 

        pad = padding[0] > 0 or padding[1] > 1
        out_pad = output_padding[0] > 1 or output_padding[1] > 1
        if  pad or out_pad:
            pad_t = padding[0] 
            pad_b = padding[0] - output_padding[0]
            pad_l = padding[1]
            pad_r = padding[1] - output_padding[1]

            if pad:
                indx = np.arange(mi*ni*ci).reshape([mi, ni, ci])[pad_t:mi-pad_b, pad_l:ni-pad_r, :]
                m, n = indx.shape[:2]
                indx = indx.reshape(-1)

        i_shift = ni * stride[0] * ci
        j_shift = stride[1] * ci

        ki = mi*ni*ci
        output = sp.coo_array((m*n*ci, b), dtype=dtype)

        for o in range(mo*no*co):
            
            X = input.getrow(o) 
            if not X.nnz:
                continue

            K_ = sp.csr_array(K[:, :, :, o%co].reshape(1, -1), copy=False)
            K_ind = K_.indices.copy()

            if dilation[0] > 1:
                q_ind = K_.indices // ci % p #col
                K_ind += q_ind*(dilation[0]-1)*ci

            if dilation[1] > 1:
                p_ind = K_.indices // (ci*q) #row
                K_ind += p_ind*(dilation[1]-1)*ci*q       
                
            indices = (o//co//no)*i_shift + (o//co%no)*j_shift + K_ind

            TK = sp.csc_array((K_.data, indices, K_.indptr), shape=(ki, 1), copy=False)
            if pad:
                TK = TK[indx]
            output += X.multiply(TK)
            # if pad:
            #     output += X.multiply(TK)[indx] # csr * csr -> csr
            # else:
            #     output += X.multiply(TK)

        # add zero padding
        if out_pad:
            if pad_b < 0 and pad_r < 0:
                output, m, n =  ConvTranspose2DLayer.pad_coo(output, shape=(m, n, ci), padding=(0, -pad_b, 0, -pad_r))
            elif pad_b < 0:
                output, m, n =  ConvTranspose2DLayer.pad_coo(output, shape=(m, n, ci), padding=(0, -pad_b, 0, 0))
            elif pad_r < 0:
                output, m, n =  ConvTranspose2DLayer.pad_coo(output, shape=(m, n, ci), padding=(0, 0, 0, -pad_r))
        out_shape = (m, n, ci)
        return output, out_shape
        

    def fconvtrans2d_coo_co_loop2(self, input, shape):
        """
            Flattened Convolution 2D for sparse 2D images 
            This method does not support bias vector
            Faster than fconv2d_coo_co_loop

            Args:
                @input: scipy sparse coo matrix with shape of H*W*C, N, where H: height, W: width, C: input channel, N: number of batches
            Return: 
                @R: convolved dataset in coo matrix
        """
        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'

        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding
        dilation = self.dilation
        weight = self.weight
        dtype = input.dtype

        assert isinstance(input, sp.coo_array) or isinstance(input, sp.coo_matrix), \
        'error: input should be a scipy sparse coo array or matrix'
        
        b = input.shape[1]
        mo, no, _ = shape
        p, q, ci, co = weight.shape

        m = (mo - 1)*self.stride[0] + dilation[0]*(p - 1) + 1
        n = (no - 1)*self.stride[1] + dilation[1]*(q - 1) + 1
        
        K = np.pad(weight, ((0, 0), (0, n-q), (0, 0), (0,0)), mode='constant') 

        i_shift = n * stride[0] * ci
        j_shift = stride[1] * ci

        k = m*n*ci
        output = sp.csr_array((k, b), dtype=dtype)

        for o in range(mo*no*co):
            
            X = input.getrow(o) 
            if not X.nnz:
                continue
            
            K_ = sp.csr_array(K[:, :, :, o%co].reshape(1, -1), copy=False)
            K_ind = K_.indices.copy()

            if dilation[0] > 1:
                q_ind = K_.indices // ci % p #col
                K_ind += q_ind*(dilation[0]-1)*ci

            if dilation[1] > 1:
                p_ind = K_.indices // (ci*q) #row
                K_ind += p_ind*(dilation[1]-1)*ci*q       
                
            indices = (o//co//no)*i_shift + (o//co%no)*j_shift + K_ind

            TK = sp.csc_array((K_.data, indices, K_.indptr), shape=(k, 1), copy=False)
            output += TK.multiply(X)

        output, m, n = ConvTranspose2DLayer.apply_padding_sparse(output, m, n, ci, padding, output_padding, tocsr=True)
        out_shape = (m, n, ci)
        return output, out_shape
      
    def fconvtrans2d_csr_co_loop(self, input, shape):
        """
            Flattened Convolution 2D for sparse 2D images 
            This method does not support bias vector

            Args:
                @input: scipy sparse coo matrix with shape of H*W*C, N, where H: height, W: width, C: input channel, N: number of batches
            Return: 
                @R: convolved dataset in coo matrix
        """

        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'

        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding
        dilation = self.dilation
        weight = self.weight
        dtype = input.dtype
        
        assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
        'error: input should be a scipy sparse csr array or matrix'

        b = input.shape[1]
        mo, no, _ = shape
        p, q, ci, co = weight.shape

        mi = (mo - 1)*self.stride[0] + dilation[0]*(p - 1) + 1
        ni = (no - 1)*self.stride[1] + dilation[1]*(q - 1) + 1
        m, n = mi, ni

        K = np.pad(weight, ((0, 0), (0, ni-q), (0, 0), (0,0)), mode='constant') 

        pad = padding[0] > 0 or padding[1] > 1
        out_pad = output_padding[0] > 1 or output_padding[1] > 1
        if  pad or out_pad:
            pad_t = padding[0] 
            pad_b = padding[0] - output_padding[0]
            pad_l = padding[1]
            pad_r = padding[1] - output_padding[1]

            if pad:
                indx = np.arange(mi*ni*ci).reshape([mi, ni, ci])[pad_t:mi-pad_b, pad_l:ni-pad_r, :]
                m, n = indx.shape[:2]
                indx = indx.reshape(-1)
            
        i_shift = ni * stride[0] * ci
        j_shift = stride[1] * ci

        ki = mi*ni*ci
        output = sp.coo_array((m*n*ci, b), dtype=dtype)
        
        for o in range(mo*no*co):
           
            if input.indptr[o+1] - input.indptr[o] <= 0: # nnz[o] <= 0
                continue
            
            X = input.getrow(o) # returns csr
            K_ = sp.csr_array(K[:, :, :, o%co].reshape(1, -1), copy=False)
            K_ind = K_.indices.copy()

            if dilation[0] > 1:
                q_ind = K_.indices // ci % p #col
                K_ind += q_ind*(dilation[0]-1)*ci

            if dilation[1] > 1:
                p_ind = K_.indices // (ci*q) #row
                K_ind += p_ind*(dilation[1]-1)*ci*q       

            indices = (o//co//no)*i_shift + (o//co%no)*j_shift + K_ind

            TK = sp.csc_array((K_.data, indices, K_.indptr), shape=(ki, 1), copy=False)
            if pad:
                TK = TK[indx]
            output += TK.multiply(X)
            # if pad:
            #     output += X.multiply(TK)[indx] # csr*csc -> csc
            # else:
            #     output += X.multiply(TK)

        # add zero padding
        if out_pad:
            if pad_b < 0 and pad_r < 0:
                output, m, n =  ConvTranspose2DLayer.pad_coo(output, shape=(m, n, ci), padding=(0, -pad_b, 0, -pad_r), tocsr=True)
            elif pad_b < 0:
                output, m, n =  ConvTranspose2DLayer.pad_coo(output, shape=(m, n, ci), padding=(0, -pad_b, 0, 0), tocsr=True)
            elif pad_r < 0:
                output, m, n =  ConvTranspose2DLayer.pad_coo(output, shape=(m, n, ci), padding=(0, 0, 0, -pad_r), tocsr=True)
        out_shape = (m, n, ci)
        return output, out_shape

    def fconvtrans2d_csr_co_loop2(self, input, shape):
        """
            Flattened Convolution 2D for sparse 2D images 
            This method does not support bias vector
            Faster than fconv2d_csr_co_loop

            Args:
                @input: scipy sparse coo matrix with shape of H*W*C, N, where H: height, W: width, C: input channel, N: number of batches
            Return: 
                @R: convolved dataset in coo matrix
        """

        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'

        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding
        dilation = self.dilation
        weight = self.weight
        dtype = input.dtype
        
        assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
        'error: input should be a scipy sparse csr array or matrix'

        b = input.shape[1]
        mo, no, _ = shape
        p, q, ci, co = weight.shape

        m = (mo - 1)*self.stride[0] + dilation[0]*(p - 1) + 1
        n = (no - 1)*self.stride[1] + dilation[1]*(q - 1) + 1

        K = np.pad(weight, ((0, 0), (0, n-q), (0, 0), (0,0)), mode='constant') 

        i_shift = n * stride[0] * ci
        j_shift = stride[1] * ci

        k = m*n*ci
        output = sp.csr_array((k, b), dtype=dtype)

        for o in range(mo*no*co):
                
            if input.indptr[o+1] - input.indptr[o] <= 0:  # nnz[o] <= 0
                continue

            X = input.getrow(o) # returns csr
            K_ = sp.csr_array(K[:, :, :, o%co].reshape(1, -1), copy=False)
            K_ind = K_.indices.copy()

            if dilation[0] > 1:
                q_ind = K_.indices // ci % p #col
                K_ind += q_ind*(dilation[0]-1)*ci

            if dilation[1] > 1:
                p_ind = K_.indices // (ci*q) #row
                K_ind += p_ind*(dilation[1]-1)*ci*q       

            indices = (o//co//no)*i_shift + (o//co%no)*j_shift + K_ind

            TK = sp.csc_array((K_.data, indices, K_.indptr), shape=(k, 1), copy=False)
            output += TK.multiply(X) # csc * csr -> csc

        output, m, n = ConvTranspose2DLayer.apply_padding_sparse(output, m, n, ci, padding, output_padding, tocsr=True)
        out_shape = (m, n, ci)
        return output, out_shape
    
    def reachSingleInput(self, In):
        if isinstance(In, ImageStar):

            assert In.V.ndim == 4, 'error: for ConvTranspose2D, basis matrix should be in 4D numpy ndarray'

            if self.module == 'pytorch':
                new_V = self.convtrans2d_pytorch(In.V, bias=False)

            elif self.module == 'default':
                # new_V = self.convtrans2d(In.V, bias=False)
                # new_V = self.convtrans2d_basic(In.V, bias=False)
                new_V = self.convtrans2d_pytorch(In.V, bias=False)

            if self.bias is not None:
                new_V[:, :, :, 0] += self.bias

            return ImageStar(new_V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        elif isinstance(In, SparseImageStar2DCOO):
            if self.module == 'pytorch':
                raise Exception(
                    'Conv2DLayer does not support \'pyotrch\' moudle for SparseImageStar set'
                )
            
            elif self.module == 'default':
                if In.c is None:
                    new_V = self.convtrans2d_pytorch(In.V, bias=False)
                    if self.bias is not None:
                        new_V[:, :, :, 0] += self.bias
                    out_shape = new_V.shape[:3]
                    return SparseImageStar2DCSR(new_V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)
                
                # new_c = self.convtrans2d(In.c.reshape(In.shape), bias=True).reshape(-1)
                new_c = self.convtrans2d_pytorch(In.c.reshape(In.shape), bias=True).reshape(-1)

                new_V, out_shape = self.fconvtrans2d_coo3(In.V, In.shape)
                return SparseImageStar2DCOO(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)
            
            raise Exception(f'error: ConvTranspose2DLayer unsupported module: {self.module}')
        
        elif isinstance(In, SparseImageStar2DCSR):
            if self.module == 'pytorch':
                raise Exception(
                    'Conv2DLayer does not support \'pyotrch\' moudle for SparseImageStar set'
                )
            
            elif self.module == 'default':
                if In.c is None:
                    new_V = self.convtrans2d_pytorch(In.V, bias=False)
                    if self.bias is not None:
                        new_V[:, :, :, 0] += self.bias
                    out_shape = new_V.shape[:3]
                    return SparseImageStar2DCSR(new_V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)
                
                new_c = self.convtrans2d_pytorch(In.c.reshape(In.shape), bias=True).reshape(-1)
                # new_V, out_shape = self.fconvtrans2d_csr4(In.V, In.shape, threads='auto')
                new_V, out_shape = self.fconvtrans2d_csr3(In.V, In.shape)
                # new_V, out_shape = self.fconvtrans2d_csr2(In.V, In.shape)
                return SparseImageStar2DCSR(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)

            raise Exception(f'error: ConvTranspose2DLayer unsupported module: {self.module}')
        
        else:
            raise Exception('error: Conv2DLayer supports ImageStar and SparseImageStar')
    
    def reach(self, inputSet, method=None, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """
            main reachability method
            Args:
                @inputSet: a list of input sets (ImageStar, SparseImageStar)
                @pool: parallel pool: None
                @RF: relaxation factor
                @DR: depth reduction; maximum depth allowed for predicate variables

            Return: 
               @R: a list of reachable set
            Unused inputs: method, lp_solver, RF (relaxation factor), DR (depth reduction)
        """

        if isinstance(inputSet, list):
            S = []

            if pool is None:
                for i in range(0, len(inputSet)):
                    S.append(self.reachSingleInput(inputSet[i]))
            elif isinstance(pool, multiprocessing.pool.Pool):
                S = S + pool.map(self.reachSingleInput, inputSet)
            else:
                raise Exception('error: unknown/unsupport pool type')
                
            return S
        
        else:
            return self.reachSingleInput(inputSet)
