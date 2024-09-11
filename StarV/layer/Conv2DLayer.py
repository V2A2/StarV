"""
Convolutional 2D Layer Class
Sung Woo Choi, 08/11/2023
"""

import time
import copy
import torch
import numpy as np
import scipy.sparse as sp
import multiprocessing
import torch.nn.functional as F
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.set.sparseimagestar import *

from timeit import default_timer as timer

class Conv2DLayer(object):
    """ Conv2DLayer Class
    
        properties:

        methods:

        https://eremo2002.tistory.com/123
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        
        https://publish.illinois.edu/mohammedsheikh/2018/03/29/mathematics-of-convolutions/
        
        https://github.com/slvrfn/vectorized_convolution/blob/master/convolution.py
        
        https://laurentperrinet.github.io/sciblog/posts/2017-09-20-the-fastest-2d-convolution-in-the-world.html
        https://dsp.stackexchange.com/questions/43953/looking-for-fastest-2d-convolution-in-python-on-a-cpu
        https://www.johnaparker.com/blog/fft_2d_performance
        https://stackoverflow.com/questions/6363154/what-is-the-difference-between-numpy-fft-and-scipy-fftpack
        https://www.reddit.com/r/learnpython/comments/m5ig1t/numpy_2d_convolution_is_too_slow_with_a_forbased/?rdt=33555
        https://stackoverflow.com/questions/48097941/strided-convolution-of-2d-in-numpy
        https://github.com/detkov/Convolution-From-Scratch
    
    """

    def __init__(
            self,
            layer, # [kernel_weight, kernel_bias] or torch.nn.Conv2D
            stride = 1, # e.g.: stride = (2, 2) or [3, 3] or 1
            padding = 0, # e.g.: padding = (0, 0) or [1, 1] or 2
            dilation = 1, # e.g.: dilation = (0, 0) or [1, 1] or 2
            sparse = False,
            in_shape = None, # shape of input data in [H, W, C]
            module = 'default', # 'default' or 'pytorch'
            dtype = 'float64', # 'float64' or 'float32'
        ):

        """
            For default StarV set up, Conv2DLayer constructor receieves:
                @layer: [kernel_weight, kernel_bias] [in numpy]
                    - kernel_weight: (Co, Ci, H, W); Co: output channel, Ci: input channel, H: height, W: width
                    - kernel_bias: None or (Co); Co: output channel
                
                @module: "default"

                Args in Conv2DLayer:
                    - layer: None 
                    - weight: (H, W, Ci, Co); H: height, W: width, Ci: input channel, Co: output channel [in numpy]
                    - bias: None or (Co); Co: output channel; [in numpy]
                    - stride
                    - padding
                    - dilation
                    - sparse: if True, unrolls weight matrix
                    - in_shape: (H, W, C)
                    - numpy_dtype [in numpy]
                    - torch_dtype [in pytorch]


            For pytorch set up, Conv2DLayer constructor receieves:
                @layer: torch.nn.Conv2D [in pytorch]
                    - layer.weight: (Co, Ci, H, W); Co: output channel, Ci: input channel, H: height, W: width 
                    - layer.bias: (Co); Co: output channel
                @module: "pytorch"

                Args in Conv2DLayer:
                    - layer: torch.nn.Conv2D [in pytorch]
                    - weight: None
                    - bias: None or (Co); Co: output channel [in pytorch]
                    - stride
                    - padding
                    - dilation
                    - numpy_dtype [in numpy]
                    - torch_dtype [in pytorch]


        """
        
        assert module in ['default', 'pytorch'], \
        'error: Conv2DLayer supports moudles: \'default\', which use numpy kernels, and \'pytorch\''
        self.module = module
        self.sparse = sparse

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

            # kernel weight in shape (kernel_height, kernel_width, ch_in, ch_out)
            self.in_channel = kernel_weight.shape[1]
            self.out_channel = kernel_weight.shape[0]

            if kernel_bias is not None:
                assert isinstance(kernel_bias, np.ndarray) and kernel_bias.ndim == 1, \
                'error: kernel bias should be 1D numpy array' 
                assert kernel_bias.shape[0] == kernel_weight.shape[3], \
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
                f'error: dilation should be a tuple, list, numpy ndarray, or int but received {type(padding)}'
            
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
                        assert dilation[0] > 0 and dilation[1], 'error: dilation should positive integer'
                        self.dilation = np.array(dilation)
                    else:
                        raise Exception('error: incorrect dilation')

                self.weight = kernel_weight.astype(self.numpy_dtype)            
                if kernel_bias is not None:
                    self.bias = kernel_bias.astype(self.numpy_dtype)
                else:
                    self.bias = None

            # converting kernel weight and bias from numpy to torch.nn.Conv2D
            elif self.module == 'pytorch':

                self.layer = torch.nn.Conv2d(
                    in_channels = self.in_channel,
                    out_channels = self.out_channel,
                    kernel_size = kernel_weight.shape[2:3],
                    stride = stride,
                    padding = padding,
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


        # input 'layer' is torch.nn.Conv2d layer
        elif isinstance(layer, torch.nn.Conv2d):

            # kernel weight in shape (ch_out, ch_in, kernel_height, kernel_width)
            self.in_channel = layer.weight.shape[0]
            self.out_channel = layer.weight.shape[1]
            
            self.stride = layer.stride
            self.padding = layer.padding
            self.dilation = layer.dilation
        
            # converting weight and bias in pytorch to numpy 
            if self.module == 'default':
                # self.weight = layer.weight.detach().numpy().astype(self.numpy_dtype).copy()
                self.weight = layer.weight.data.numpy().astype(self.numpy_dtype).copy()
                # change weight in (Co, Ci, H, W) to (H, W, Ci, Co) 
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

        if sparse:
            assert in_shape is not None , \
            f"To unroll weight matrix, the input shape (in_shape) must be provided in a  3-tuple containing (H, W, C). Given in_shape = {in_shape}"
            assert len(in_shape) == 3, \
            f"To unroll weight matrix, the input shape (in_shape) must be provided in a  3-tuple containing (H, W, C). Given in_shape = {in_shape}"

            assert self.module == 'default', \
            f"Sparse Conv2D (unrolling weight matrix) suppors only \'default\' module. Given module = {self.module}"
            
            m, n, c = copy.deepcopy(in_shape)

            p, q, ci, co = self.weight.shape
            
            mo, no = self.get_output_size_sparse(in_height=in_shape[0], in_width=in_shape[1])
            m += 2*self.padding[0]
            n += 2*self.padding[1]
            
            i_shift = n * self.stride[0] * c
            j_shift = self.stride[1] * c
            
            ko = mo*no

            #weight has [H, W, Ci, Co] shape
            #after madding [H + (m-p), W + (n-q), Ci, Co]
            #after flattening [(p + (m-p)) * (q + (n-q)) * Ci, Co]

            Z = np.pad(self.weight, ((0, m-p), (0, n-q), (0, 0), (0,0)), mode='constant')
            
            # CSR implmentation of kernel weight
            Z_ = sp.csr_array(Z.reshape(np.prod(Z.shape[:3]), co).T, copy=False)
            nnz = Z_.indptr[1:] - Z_.indptr[:-1]
            Z_ind = Z_.indices.copy()

            if dilation[0] > 1:
                q_ind = Z_.indices // c % p #col
                Z_ind += q_ind*(dilation[0]-1)*c

            if dilation[1] > 1:
                p_ind = Z_.indices // (c*q) #row
                Z_ind += p_ind*(dilation[1]-1)*c*q
        
            data = np.repeat(Z_.data[None, :], ko, axis=0).reshape(-1)

            indices = np.arange(ko, dtype=np.int32)[:, np.newaxis]
            indices = ((indices//no)*i_shift + (indices%no)*j_shift + Z_ind).reshape(-1)
        
            indptr = np.hstack([Z_.indptr, ((np.arange((ko-1)*co, dtype=np.int32) + 1 + Z_.shape[0]).reshape(ko-1, co) * nnz).reshape(-1)] )
            
            self.weight = sp.csr_array((data, indices, indptr), shape=(ko*co, Z_.shape[1]), copy=False)
            

            """
            # COO implementation of kernel weight
            Z_ = sp.coo_array(Z.reshape(Z.shape[0]*Z.shape[1]*Z.shape[2], co).T, copy=False)

            new_data = np.repeat(Z_.data[None, :], ko, axis=0).reshape(-1)

            new_col = np.arange(ko, dtype=np.int32)[:, np.newaxis]
            new_col = (new_col//no)*i_shift + (new_col%no)*j_shift
            new_col = (new_col + Z_.col).reshape(-1)

            new_row =
            """

            self.in_shape = in_shape
            self.out_shape = (mo, no, co)
            self.kernel_size = (p, q)
        else:
            self.in_shape = in_shape
            self.out_shape = in_shape

        #####
        self.in_dim = self.in_channel
        self.out_dim = self.out_channel

    def __str__(self):
        print('Convolutional 2D Layer')
        print('module: {}'.format(self.module))
        print('in_channel: {}'.format(self.in_channel))
        print('out_channel: {}'.format(self.out_channel))
        print('stride: {}'.format(self.stride))
        print('padding: {}'.format(self.padding))
        print('dilation: {}'.format(self.dilation))
        print('sparse: {}'.format(self.sparse))
        if self.sparse:
            print('in_shape: ', self.in_shape)
            print('out_shape: ', self.out_shape)

        if self.module == 'pytorch':
            print('weight: {}'.format(self.layer.weight.shape))            
        else:
            print('weight: {}, {}'.format(self.weight.shape, self.weight.dtype))

        if self.bias is not None:
            print('bias: {}, {}'.format(self.bias.shape, self.bias.dtype))
        else:
            print('bias: {}'.format(self.bias))
        return ''

    def pad_coo(input, shape, padding, tocsc=False):
        if len(padding) == 4:
            pad = np.array(padding)
        elif len(padding) == 2:
            pad = np.array([padding[0], padding[0], padding[1], padding[1]])
        elif len(padding) == 1:
            pad = np.ones(4)*padding[0]

        """Adding padding to coo"""
        row = input.row + (input.row // (shape[1]*shape[2])) * (padding[2]+padding[3])* shape[2]
        row += shape[2]*((shape[1]+padding[2]+padding[3])*padding[0]+padding[2])

        mo = shape[0] + padding[0] + padding[1]
        no = shape[1] + padding[2] + padding[3]
        print('mo, no: ', mo, no)
        if tocsc is True:
            output = sp.csc_array((input.data, (row, input.col)), shape = (mo*no*shape[2], input.shape[1]))
        else:
            output = sp.coo_array((input.data, (row, input.col)), shape = (mo*no*shape[2], input.shape[1]))
        return output, mo, no

    def pad_csr(input, shape, padding, tocsc=False):
        output, mo, no = Conv2DLayer.pad_coo(input.tocoo(False), shape, padding, tocsc=False)
        return output.tocsr(False), mo, no

    def add_zero_padding(input, padding):

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'

        if padding[0] == 0 and padding[1] == 0:
            return input
        
        in_dim = input.ndim
        if in_dim == 4:
            return np.pad(input, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0), (0,0)), mode='constant')
        elif in_dim == 3:
            return np.pad(input, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant')
        elif in_dim == 2:
            return np.pad(input, ((padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
        else:
            raise Exception(
                'Invalid number of input dimensions; it should be between 2D and 4D'
            )
    
    def add_zero_padding_old(input, padding):

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'

        if padding[0] == 0 and padding[1] == 0:
            return input
        
        in_dim = input.ndim
        if in_dim == 4:
            h, w, c, n = input.shape
            out = np.zeros(
                (h + 2*padding[0], w + 2*padding[1], c, n)
            )
            out[padding[0]:h+padding[0], padding[1]:w+padding[1], :, :] = input

        elif in_dim == 3:
            h, w, c = input.shape
            out = np.zeros(
                (h + 2*padding[0], w + 2*padding[1], c)
            )
            out[padding[0]:h+padding[0], padding[1]:w+padding[1], :] = input

        elif in_dim == 2:
            h, w = input.shape
            out = np.zeros(
                (h + 2*padding[0], w + 2*padding[1])
            )
            out[padding[0]:h+padding[0], padding[1]:w+padding[1]] = input

        else:
            raise Exception(
                'Invalid number of input dimensions; it should be between 2D and 4D'
            )

        return out
    
    def get_output_size(self, input):
        h, w, c, n = input.shape
        H, W = self.weight.shape[:2]

        ho = ((h + 2*self.padding[0] - H - (H - 1) * (self.dilation[0] - 1)) // self.stride[0]) + 1
        wo = ((h + 2*self.padding[1] - H - (H - 1) * (self.dilation[1] - 1)) // self.stride[1]) + 1
        
        assert ho > 0 and wo > 0, 'error: the shape of resulting output should be positive'
        return ho, wo
    
    def get_output_size_sparse(self, in_height, in_width):
        h, w = in_height, in_width
        H, W = self.weight.shape[:2]

        ho = ((h + 2*self.padding[0] - H - (H - 1) * (self.dilation[0] - 1)) // self.stride[0]) + 1
        wo = ((h + 2*self.padding[1] - H - (H - 1) * (self.dilation[1] - 1)) // self.stride[1]) + 1

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
            return self.conv2d_pytorch(input, bias=True)
        
        else:
            return self.conv2d(input, bias=True)
        
    def conv2d_pytorch(self, input, bias=True):
        """
            Args:
               @input: dataset in pytorch with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: convolved dataset
        """
        
        assert isinstance(self.layer, torch.nn.Conv2d), '\'layer\' should be torch.nn.Conv2d for \'pytorch\' module'

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

        conv2d_layer = self.layer
        if bias == True:
            conv2d_layer.bias = torch.nn.Parameter(self.bias)
        else:
            conv2d_layer.bias = None

        output = conv2d_layer(input).detach().numpy()
        # change input shape to H, W, C, Noutput += self.bias[None, None, :, None]
        # if in_dim == 3:
        #     output = output.reshape(H, W, C) 

        return output


    def conv2d_basic(self, input, bias=True):
        """ 
            Basic, tranditaional convolution 2D

            Args:
               @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: convolved dataset
        
        """
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        in_dim = input.ndim
        dtype = input.dtype
        
        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'
        assert in_dim >= 2 and in_dim <= 4, \
        'error: input should be 2D, 3D, or 4D numpy ndarray'

        input = copy.deepcopy(input)
        if in_dim == 2:
            input = input[:, :, None, None]
        elif in_dim == 3:
            input = input[:, :, :, None]
            
        h, w, c, n = input.shape
        H, W, C, F = self.weight.shape
  
        ho, wo = self.get_output_size(input)
        
        pad_input = Conv2DLayer.add_zero_padding(input, padding)
        pn, pm = pad_input.shape[:2]
        assert pn >= H and pm >= W, 'error: kernel shape should not be bigger than that of input'
        
        output = np.zeros((ho, wo, F, n), dtype=dtype)

        b = H // 2, W // 2
        center_h0 = b[0] * dilation[0]
        center_w0 = b[1] * dilation[1] 
        
        odd = H % 2, W % 2
        
        for z in range(n):
            working_input = pad_input[:, :, :, z]
            
            for k in range(F):
                out_ch = output[:, :, k, z]

                for i in range(ho):
                    center_h = center_h0 + stride[0] * i
                    indices_h = [center_h + l * dilation[0] for l in range(-b[0], b[0] + odd[0])]

                    for j in range(wo):
                        center_w = center_w0 + stride[1] * j
                        indices_w = [center_w + l * dilation[1] for l in range(-b[1], b[1] + odd[1])]

                        feature_map = working_input[indices_h, :, :][: , indices_w, :]
                        out_ch[i, j] = np.sum(feature_map * self.weight[:, :, :, k])
        
        if bias is True:
            if isinstance(self.bias, np.ndarray):
                output += self.bias[None, None, :, None]
        
        # if in_dim == 2:
        #     output = output.reshape(ho, wo)
        # elif in_dim == 3:
        #     output = output.reshape(ho, wo, F)
        return output
        
    
    def conv2d_vec(self, input, bias=True):
        """
            Vectorized convolution 2D

            Args:
               @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: convolved dataset
        """

        assert self.module == 'default', 'error: conv2d_vec support \'default\' module'
        
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        in_dim = input.ndim
        dtype = input.dtype
        
        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'
        assert in_dim >= 2 and in_dim <= 4, \
        'error: input should be 2D, 3D, or 4D numpy ndarray'

        input = input.copy()

        if in_dim == 2:
            input = input[:, :, None, None]
        elif in_dim == 3:
            input = input[:, :, :, None]

        h, w, c, n = input.shape
        H, W, C, F  = self.weight.shape

        ho, wo = self.get_output_size(input)
        pad_input = Conv2DLayer.add_zero_padding(input, padding)        
        pn, pm = pad_input.shape[:2]
        assert pn >= H and pm >= W, 'error: kernel shape should not be bigger than that of input'
        
        output = np.zeros((ho, wo, F, n), dtype=dtype)

        b = H // 2, W // 2
        center_h0 = b[0] * dilation[0]
        center_w0 = b[1] * dilation[1]
        
        odd = H % 2, W % 2
        
        for z in range(n):
            working_input = pad_input[:, :, :, z]
            row_feature_map = np.zeros((ho*wo, H*W*c), dtype=dtype)
        
            for i in range(ho):
                center_h = center_h0 + stride[0] * i
                indices_h = [center_h + l * dilation[0] for l in range(-b[0], b[0] + odd[0])]

                for j in range(wo):
                    center_w = center_w0 + stride[1] * j
                    indices_w = [center_w + l * dilation[1] for l in range(-b[1], b[1] + odd[1])]

                    feature_map = working_input[indices_h, :, :][: , indices_w, :]
                    # convert feature_map to row vector
                    row_feature_map[i*wo+j, :] = feature_map.reshape(-1)

            # convert weight matrix into row vector
            # weight = copy.deepcopy(self.weight)
            # col_weight = np.reshape(weight, (-1, F))
            col_weight = self.weight.reshape(-1, F)
            out_n = np.matmul(row_feature_map, col_weight)            
            output[:, :, :, z] = out_n.reshape(ho, wo, F)
        
        if bias is True:
            if isinstance(self.bias, np.ndarray):
                output += self.bias[None, None, :, None]
        
        # if in_dim == 3 or in_dim == 2:
        #     output = output.reshape(ho, wo, F)
        return output

    def conv2d(self, input, bias=True):
        """
            Vectorized convolution 2D

            Args:
               @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: convolved dataset
        """

        assert self.module == 'default', 'error: conv2d_vec support \'default\' module'
        
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        weight = self.weight
        in_dim = input.ndim
        dtype = input.dtype

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'
        assert in_dim >= 2 and in_dim <= 4, \
        'error: input should be 2D, 3D, or 4D numpy ndarray'

        if in_dim == 2:
            input = input[:, :, None, None]
        elif in_dim == 3:
            input = input[:, :, :, None]

        m, n, c, b = input.shape
    
        if padding[0] > 0 or padding[1] > 0:
            m += 2*padding[0]
            n += 2*padding[1]
            XF = np.pad(input, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0), (0,0)), mode='constant').reshape(m*n*c, b)
        else:
            XF = input.reshape(m*n*c, b)

        if self.sparse:
            mo, no, co = self.out_shape
            
            output = weight @ XF # in shape [mo*no*co, b]
            output = output.reshape(mo, no, co, b)
            if bias is True:
                if isinstance(self.bias, np.ndarray):
                    output += self.bias[None, None, :, None]

            if in_dim == 3:
                output = np.squeeze(output, axis=3)

        else:
            p, q, ci, co  = weight.shape
            mo, no = self.get_output_size(input)
        
            Z = np.pad(np.ones([p, q, c], dtype=bool), ((0, m-p), (0, n-q), (0, 0)), mode='constant').reshape(-1)
            Z_ind = np.where(Z > 0)[0]
            Z_indices = Z_ind.copy()

            if dilation[0] > 1:
                q_ind = Z_ind // c % p #col
                Z_indices += q_ind*(dilation[0]-1)*c

            if dilation[1] > 1:
                p_ind = Z_ind // (c*q) #row
                Z_indices += p_ind*(dilation[1]-1)*c*q

            i_shift = n*stride[0]*c
            j_shift = stride[1]*c

            z = p*q*c
            ko = mo*no
            X = np.zeros([b, ko, z], dtype=dtype)
            for i in range(ko):
                ind = (i//no)*i_shift + (i%no)*j_shift + Z_indices
                X[:, i, :] = XF[ind, :].T

            # ind = np.arange(ko, dtype=np.int32)[:, None]
            # ind = (ind//no)*i_shift + (ind%no)*j_shift + Z_indices #ind in [ko, p*q*c]
            # X = XF[ind, :].transpose([2, 0, 1]) # in shape of [ko, p*q*c, b]
            # XF in [m*n*c, b]
            # XF.T in [b, m*n*c]
            # K in [p*q*c, co]

            K = weight.reshape(z, co)
            output = X @ K #in shape [b, ko, co]
            if bias is True:
                if isinstance(self.bias, np.ndarray):
                    output += self.bias[None, None, :]
            output = output.transpose([1, 2, 0]).reshape(mo, no, co, b)
        return output
   
    def add_zero_padding_sparse(input, padding):
        assert isinstance(input, SparseImage), \
        'error: input should be 2D sparse tensor in coo format'

        for i in range(input.num_images):
            for k in range(input.num_channel):
                x = input.images[i].channel[k].image
                if x is not None:
                    row = x.row + 2*padding[0] 
                    col = x.col + 2*padding[0] 
        
        x = copy.deepcopy(input)
        indices = x.indices()
        indices[0] += padding[0]
        indices[1] += padding[1]
        
        h, w = x.shape
        h += 2*padding[0]
        w += 2*padding[1]
        
        return torch.sparse_coo_tensor(indices, x.values(), size=(h,w)).coalesce()

    def conv2d_sparse(self, input, bias=True):
        
        """
            Convolution 2D for sparse images (Dilation is not supported)

            Args:
                @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches
                @input: SparseImage
            Return: 
               @R: convolved dataset
        """

        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'
        
        stride = self.stride
        padding = self.padding
        weight = self.weight
        
        assert isinstance(input, SparseImage), \
        'error: input should be a SparseImage'

        input = copy.deepcopy(input) #.astype(self.numpy_dtype)
        h, w, c, n = input.size()
        H, W, C, F = weight.shape
        ho, wo = self.get_output_size_sparse(h, w)

        h += 2*padding[0]
        w += 2*padding[1]

        b = H // 2, W // 2
        odd = H % 2, W % 2

        r_rep = np.zeros(h, np.ushort)
        c_rep = np.zeros(w, np.ushort)

        for i_ in range(ho):
            center_h = b[0] + stride[0] * i_
            indices_h = [center_h + l_ for l_ in range(-b[0], b[0] + odd[0])]
            r_rep[indices_h] += 1
        indx_r_rep = np.argwhere(r_rep).reshape(-1)

        for j_ in range(wo):
            center_w = b[1] + stride[1] * j_
            indices_w = [center_w + l_ for l_ in range(-b[1], b[1] + odd[1])]
            c_rep[indices_w] += 1
        indx_c_rep = np.argwhere(c_rep).reshape(-1)

        z_r_rep = (r_rep == 0).any()
        z_c_rep = (c_rep == 0).any()

        sp_im = []
        for n_ in range(n): # number of predicates
            nnz = [[] for _ in range(F)]
            val = [[] for _ in range(F)]
            for c_ in range(c): # number of channels
                im = input.images[n_].channel[c_]
                if im is None:
                    continue
                
                # apply padding to input
                row = im.row + padding[0]
                col = im.col + padding[1]
                data = im.data

                # if z_r_rep:
                #     indx_i = np.argwhere( (row == indx_r_rep.reshape(-1, 1)).any() ).reshape(-1)
                #     row = row[indx_i]
                #     col = col[indx_i]
                #     data = data[indx_i]

                # if z_c_rep:
                #     indx_j = np.argwhere( (col == indx_c_rep.reshape(-1, 1)).any() ).reshape(-1)
                #     row = row[indx_j]
                #     col = col[indx_j]
                #     data = data[indx_j]

                # if z_r_rep:
                #     row = row[indx_r_rep]
                #     col = col[indx_r_rep]
                #     data = data[indx_r_rep]

                # if z_c_rep:
                #     row = row[indx_c_rep]
                #     col = col[indx_c_rep]
                #     data = data[indx_c_rep]

                if z_r_rep:
                    indx_ = np.argwhere( (row[None, :] == indx_r_rep[:, None]) )[:, 1]
                    row = row[indx_]
                    col = col[indx_]
                    data = data[indx_]

                if z_c_rep:
                    indx_ = np.argwhere( (col[None, :] == indx_c_rep[:, None]) )[:, 1]
                    row = row[indx_]
                    col = col[indx_]
                    data = data[indx_]

                rl = np.minimum(ho, np.maximum(0, np.ceil((row - H + 1) / stride[0]))).astype(np.uint16)
                ru = np.minimum(ho, row // stride[0] + 1).astype(np.uint16)

                cl = np.minimum(wo, np.maximum(0, np.ceil((col - W + 1) / stride[1]))).astype(np.uint16)
                cu = np.minimum(wo, col // stride[1] + 1).astype(np.uint16)

                for f_ in range(F):
                    for o_ in range(len(row)):
                        for row_ in range(rl[o_], ru[o_]):
                            for col_ in range(cl[o_], cu[o_]):
                                nnz[f_].append([row_, col_])
                                val[f_].append(
                                    weight[
                                        (row[o_] - row_*stride[0]) % H,
                                        (col[o_] - col_*stride[1]) % W,
                                        c_, f_
                                    ] * data[o_]
                                )

            im3d = SparseImage3D(ho, wo, F, n_)
            for f_ in range(F):
                nnz_ = np.array(nnz[f_])
                val_ = np.array(val[f_])

                #inverse_indices
                U, inv_indices = np.unique(nnz_, axis=0, return_inverse=True)
                N = U.shape[0]

                new_v = []
                for q in range(N):
                    new_v.append(val_[inv_indices == q].sum())
                new_v = np.array(new_v)

                if bias is True:
                    if isinstance(self.bias, np.ndarray):
                        new_v += self.bias[f_] 

                new_row = U[:, 0].reshape(-1)
                new_col = U[:, 1].reshape(-1)

                im2d = sp.coo_array((new_v, (new_row, new_col)), shape=(ho, wo))
                im3d.append(im2d, f_, n_)
            sp_im.append(im3d)
        
        return SparseImage(sp_im, n)
    

    def conv2d_sparse2d_as4d(self, input, shape, bias=True):
        
        """
            Convolution 2D for sparse 2D images (Dilation is not supported)

            Args:
                @input: dataset in numpy with shape of H*W*C, N, where H: height, W: width, C: input channel, N: number of batches
                @input: scipy sparse coo matrix
            Return: 
               @R: convolved dataset
        """

        assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'
        
        stride = self.stride
        padding = self.padding
        weight = self.weight
        
        assert isinstance(input, sp.coo_array) or isinstance(input, sp.coo_matrix), \
        'error: input should be a scipy sparse coo array or matrix but received {}'.format(type(input))

        input = copy.deepcopy(input) #.astype(self.numpy_dtype)
        h, w, c = shape
        n = input.shape[1]
        H, W, Ci, Co = weight.shape
        ho, wo = self.get_output_size_sparse(h, w)
        out_shape = np.array([ho, wo, Co])

        indx, pred, data = input.row, input.col, input.data
        row, col, channel = SparseImageStar2DCOO.index_2hwc(shape, indx)
        dtype = data.dtype

        del indx

        # apply padding to input
        h += 2*padding[0]
        w += 2*padding[1]
        row += padding[0]
        col += padding[1]

        b = H // 2, W // 2
        odd = H % 2, W % 2

        r_rep = np.zeros(h, np.ushort)
        c_rep = np.zeros(w, np.ushort)

        for i_ in range(ho):
            center_h = b[0] + stride[0] * i_
            indices_h = [center_h + l_ for l_ in range(-b[0], b[0] + odd[0])]
            r_rep[indices_h] += 1
        indx_r_rep = np.argwhere(r_rep).reshape(-1)

        for j_ in range(wo):
            center_w = b[1] + stride[1] * j_
            indices_w = [center_w + l_ for l_ in range(-b[1], b[1] + odd[1])]
            c_rep[indices_w] += 1
        indx_c_rep = np.argwhere(c_rep).reshape(-1)

        if (r_rep == 0).any():
            indx_ = np.argwhere( (row[None, :] == indx_r_rep[:, None]) )[:, 1]
            row = row[indx_]
            col = col[indx_]
            data = data[indx_]

        if (c_rep == 0).any():
            indx_ = np.argwhere( (col[None, :] == indx_c_rep[:, None]) )[:, 1]
            row = row[indx_]
            col = col[indx_]
            data = data[indx_]

        rl = np.minimum(ho, np.maximum(0, np.ceil((row - H + 1) / stride[0]))).astype(np.uint16)
        ru = np.minimum(ho, row // stride[0] + 1).astype(np.uint16)

        cl = np.minimum(wo, np.maximum(0, np.ceil((col - W + 1) / stride[1]))).astype(np.uint16)
        cu = np.minimum(wo, col // stride[1] + 1).astype(np.uint16)

        n_elem = ((Co) * ((ru - rl) * (cu - cl)).sum()).astype(int)

        nnz = np.zeros([n_elem, 2], dtype=np.int64)
        val = np.zeros([n_elem], dtype=dtype)
        n_ = 0
        for f_ in range(Co):
            for o_ in range(len(row)):
                for row_ in range(rl[o_], ru[o_]):
                    for col_ in range(cl[o_], cu[o_]):
                        indx_ = SparseImageStar2DCOO.hwc_2index(out_shape, row_, col_, f_)
                        nnz[n_, :] = np.array([indx_, pred[o_]])
                        val[n_] = weight[
                                (row[o_] - row_*stride[0]) % H,
                                (col[o_] - col_*stride[1]) % W,
                                channel[o_], f_
                            ] * data[o_]
                        n_ += 1

        #inverse_indices
        U, inv_indices = np.unique(nnz, axis=0, return_inverse=True)

        conv_v = []
        for q in range(U.shape[0]):
            conv_v.append(sum(val[inv_indices == q]))
        conv_v = np.array(conv_v)

        return sp.coo_array(
                (conv_v, (U[:, 0], U[:, 1])), shape=(out_shape.prod(), n)
            ), out_shape
    

    # def conv2d_sparse2d2(self, input, shape, bias=True):
        
    #     """
    #         Convolution 2D for sparse 2D images 

    #         Args:
    #             @input: dataset in numpy with shape of H*W*C, N, where H: height, W: width, C: input channel, N: number of batches
    #             @input: scipy sparse coo matrix
    #         Return: 
    #            @R: convolved dataset
    #     """

    #     assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'
        
    #     stride = self.stride
    #     padding = self.padding
    #     weight = self.weight
        
    #     assert isinstance(input, sp.coo_array), \
    #     'error: input should be a scipy sparse coo array'

    #     input = copy.deepcopy(input) #.astype(self.numpy_dtype)
    #     h, w, c = shape
    #     n = input.shape[1]
    #     H, W, Ci, Co = weight.shape
    #     ho, wo = Conv2DLayer.get_output_size_sparse(h, w, weight, stride, padding)
    #     out_shape = np.array([ho, wo, Co])

    #     indx, pred, data = input.row, input.col, input.data
    #     row, col, channel = SparseImageStar2D.index_2hwc(shape, indx)
    #     dtype = data.dtype

    #     del indx

    #     # apply padding to input
    #     h += 2*padding[0]
    #     w += 2*padding[1]
    #     row += padding[0]
    #     col += padding[1]

    #     b = H // 2, W // 2
    #     odd = H % 2, W % 2

    #     r_rep = np.zeros(h, np.ushort)
    #     c_rep = np.zeros(w, np.ushort)

    #     for i_ in range(ho):
    #         center_h = b[0] + stride[0] * i_
    #         indices_h = [center_h + l_ for l_ in range(-b[0], b[0] + odd[0])]
    #         r_rep[indices_h] += 1
    #     indx_r_rep = np.argwhere(r_rep).reshape(-1)

    #     for j_ in range(wo):
    #         center_w = b[1] + stride[1] * j_
    #         indices_w = [center_w + l_ for l_ in range(-b[1], b[1] + odd[1])]
    #         c_rep[indices_w] += 1
    #     indx_c_rep = np.argwhere(c_rep).reshape(-1)

    #     if (r_rep == 0).any():
    #         indx_ = np.argwhere( (row[None, :] == indx_r_rep[:, None]) )[:, 1]
    #         row = row[indx_]
    #         col = col[indx_]
    #         data = data[indx_]

    #     if (c_rep == 0).any():
    #         indx_ = np.argwhere( (col[None, :] == indx_c_rep[:, None]) )[:, 1]
    #         row = row[indx_]
    #         col = col[indx_]
    #         data = data[indx_]

    #     rl = np.minimum(ho, np.maximum(0, np.ceil((row - H + 1) / stride[0]))).astype(np.uint16)
    #     ru = np.minimum(ho, row // stride[0] + 1).astype(np.uint16)

    #     cl = np.minimum(wo, np.maximum(0, np.ceil((col - W + 1) / stride[1]))).astype(np.uint16)
    #     cu = np.minimum(wo, col // stride[1] + 1).astype(np.uint16)

    #     n_elem = ((Co) * ((ru - rl) * (cu - cl)).sum()).astype(int)

    #     nnz = np.zeros([n_elem, 4], dtype=dtype)
    #     val = np.zeros([n_elem], dtype=dtype)
    #     n_ = 0
    #     for f_ in range(Co):
    #         for o_ in range(len(row)):
    #             for row_ in range(rl[o_], ru[o_]):
    #                 for col_ in range(cl[o_], cu[o_]):
    #                     nnz[n_, :] = np.array([row_, col_, f_, pred[o_]])
    #                     val[n_] = weight[
    #                             (row[o_] - row_*stride[0]) % H,
    #                             (col[o_] - col_*stride[1]) % W,
    #                             channel[o_], f_
    #                         ] * data[o_]
    #                     n_ += 1
        
    #     indx = SparseImageStar2D.hwc_2index(out_shape, nnz[:, 0], nnz[:, 1], nnz[:, 2])
    #     nnz2 = np.column_stack([indx, nnz[:,3]])

    #     #inverse_indices
    #     U, inv_indices = np.unique(nnz2, axis=0, return_inverse=True)

    #     conv_v = []
    #     for q in range(U.shape[0]):
    #         conv_v.append(sum(val[inv_indices == q]))
    #     conv_v = np.array(conv_v)

    #     return sp.coo_array(
    #             (conv_v, (U[:, 0], U[:, 1])), shape=(out_shape.prod(), n)
    #         )

    def fconv2d_coo(self, input, shape):
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
        dilation = self.dilation
        weight = self.weight
        
        assert isinstance(input, sp.coo_array) or isinstance(input, sp.coo_matrix), \
        'error: input should be a scipy sparse coo array or matrix'

        m, n, c = shape
        
        if padding[0] > 0 or padding[1] > 0:
            XF, m, n = Conv2DLayer.pad_coo(input, shape, padding)
            c = shape[2]
        else:
            XF = input

        if self.sparse:
            O = (self.weight @ XF).tocoo(copy=False) # csr @ dense -> csr
            out_shape = self.out_shape

        else:
            p, q, ci, co = weight.shape
            mo, no = self.get_output_size_sparse(in_height=shape[0], in_width=shape[1])

            i_shift = n * stride[0] * c
            j_shift = stride[1] * c
            
            ko = mo*no

            Z = np.pad(weight, ((0, m-p), (0, n-q), (0, 0), (0,0)), mode='constant')
            Z_ = sp.csr_array(Z.reshape(np.prod(Z.shape[:3]), co).T, copy=False)
            nnz = Z_.indptr[1:] - Z_.indptr[:-1]
            Z_ind = Z_.indices.copy()

            if dilation[0] > 1:
                q_ind = Z_.indices // c % p #col
                Z_ind += q_ind*(dilation[0]-1)*c

            if dilation[1] > 1:
                p_ind = Z_.indices // (c*q) #row
                Z_ind += p_ind*(dilation[1]-1)*c*q
        
            data = np.repeat(Z_.data[None, :], ko, axis=0).reshape(-1)

            indices = np.arange(ko, dtype=np.int32)[:, np.newaxis]
            indices = ((indices//no)*i_shift + (indices%no)*j_shift + Z_ind).reshape(-1)
        
            indptr = np.hstack([Z_.indptr, ((np.arange((ko-1)*co, dtype=np.int32) + 1 + Z_.shape[0]).reshape(ko-1, co) * nnz).reshape(-1)] )

            TZ = sp.csr_array((data, indices, indptr), shape=(ko*co, Z_.shape[1]), copy=False)
            O = (TZ @ XF).tocoo(copy=False)

            out_shape = (mo, no, co)

        return O, out_shape
    
    def fconv2d_coo_co_loop(self, input, shape):
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
        dilation = self.dilation
        weight = self.weight
        
        assert isinstance(input, sp.coo_array) or isinstance(input, sp.coo_matrix), \
        'error: input should be a scipy sparse coo array or matrix'

        m, n, c = shape
        
        if padding[0] > 0 or padding[1] > 0:
            XF, m, n = Conv2DLayer.pad_coo(input, shape, padding)
            c = shape[2]
        else:
            XF = input

        if self.sparse:
            O = (self.weight @ XF).tocoo(copy=False) # csr @ coo -> csr
            out_shape = self.out_shape

        else:
            b = input.shape[1]
            p, q, ci, co = weight.shape
            mo, no = self.get_output_size_sparse(in_height=shape[0], in_width=shape[1])

            K = np.pad(weight, ((0, 0), (0, n-q), (0, 0), (0,0)), mode='constant') 

            i_shift = n*stride[0]*c
            j_shift = stride[1]*c
            
            ko = mo*no
            
            val_list = []
            row_list = []
            col_list = []

            for o in range(co):                
                K_ = sp.csr_array(K[:, :, :, o].reshape(1, -1), copy=False)
                K_ind = K_.indices.copy()

                if dilation[0] > 1:
                    q_ind = K_.indices // c % p #col
                    K_ind += q_ind*(dilation[0]-1)*c

                if dilation[1] > 1:
                    p_ind = K_.indices // (c*q) #row
                    K_ind += p_ind*(dilation[1]-1)*c*q                

                data = np.repeat(K_.data[None, :], ko, axis=0).reshape(-1)
                
                indices = np.arange(ko, dtype=np.int32)[:, np.newaxis]
                indices = ((indices//no)*i_shift + (indices%no)*j_shift + K_ind).reshape(-1)

                indptr = np.hstack([K_.indptr, (np.arange(ko-1, dtype=np.int32)+2)*K_.nnz])        

                TK = sp.csr_array((data, indices, indptr), shape=(ko, XF.shape[0]), copy=False)

                P = (TK @ XF).tocoo(copy=False)

                val_list.append(P.data)
                row_list.append((P.row*co + o).astype(np.int32))
                col_list.append(P.col.astype(np.int32))
            
            O = sp.coo_array((np.hstack(val_list), (np.hstack(row_list), np.hstack(col_list))), shape=(ko*co, b), copy=False)
            out_shape = (mo, no, co)

        return O, out_shape
    

    def fconv2d_coo_co_loop2(self, input, shape):
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
        dilation = self.dilation
        weight = self.weight
        
        assert isinstance(input, sp.coo_array) or isinstance(input, sp.coo_matrix), \
        'error: input should be a scipy sparse coo array or matrix'

        dtype = input.dtype
        b = input.shape[1]
        m, n, c = shape
        p, q, ci, co = weight.shape
        mo, no = self.get_output_size_sparse(in_height=shape[0], in_width=shape[1])

        if padding[0] > 0 or padding[1] > 0:
            XF, m, n = Conv2DLayer.pad_coo(input, shape, padding)
            c = shape[2]
        else:
            XF = input
            
        K = np.pad(weight, ((0, 0), (0, n-q), (0, 0), (0,0)), mode='constant') 
                
        i_shift = n*stride[0]*c
        j_shift = stride[1]*c
        
        ko = mo*no
        to = ko*co
        
        data_list = np.array([None for _ in range(to)])
        indices_list = np.array([None for _ in range(to)])
        nnz = np.zeros(to, dtype=np.int32)
        for o in range(co):
            K_ = sp.csr_array(K[:, :, :, o].reshape(1, -1), copy=False)
            K_ind = K_.indices.copy()

            if dilation[0] > 1:
                q_ind = K_.indices // c % p #col
                K_ind += q_ind*(dilation[0]-1)*c

            if dilation[1] > 1:
                p_ind = K_.indices // (c*q) #row
                K_ind += p_ind*(dilation[1]-1)*c*q
            
            data = np.repeat(K_.data[None, :], ko, axis=0).reshape(-1)
            
            indices = np.arange(ko, dtype=np.int32)[:, np.newaxis]
            indices = ((indices//no)*i_shift + (indices%no)*j_shift + K_ind).reshape(-1)

            indptr = np.hstack([K_.indptr, (np.arange(ko-1, dtype=np.int32)+2)*K_.nnz])        

            TK = sp.csr_array((data, indices, indptr), shape=(ko, XF.shape[0]), copy=False)

            P = TK @ XF
 
            for i in range(ko):
                indx = o + i*co
                i_ = P.indptr[i]
                i_1 = P.indptr[i+1]
                data_list[indx] = P.data[i_:i_1]
                indices_list[indx] = P.indices[i_:i_1]
                nnz[indx] = i_1 - i_

        data_list = np.hstack(data_list)
        indices_list = np.hstack(indices_list)

        sum = nnz.sum()
        if sum < pow(2, 31):
            indptr = np.zeros(to + 1, dtype=np.int32)
        else:
            indptr = np.zeros(to + 1, dtype=np.int64)
            indices_list = indices_list.astype(np.int64)

        for i in range(to):
            indptr[i+1] = indptr[i] + nnz[i]

        O = sp.csr_array((data_list, indices_list, indptr), shape=(to, b)).tocoo()
        out_shape = (mo, no, co)
        return O, out_shape
    

    # def fconv2d_csr_test(self, input, shape):
    #     """
    #         Flattened Convolution 2D for sparse 2D images 
    #         This method does not support bias vector

    #         Args:
    #             @input: scipy sparse csr matrix with shape of H*W*C, N, where H: height, W: width, C: input channel, N: number of batches
    #         Return: 
    #             @R: convolved dataset in csr matrix
    #     """

    #     assert self.module == 'default', 'error: conv2d_sparse() supports \'default\' module'
        
    #     stride = self.stride
    #     padding = self.padding
    #     weight = self.weight
        
    #     assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
    #     'error: input should be a scipy sparse csr array or matrix'

    #     m, n, c = shape

    #     if padding[0] > 0 or padding[1] > 0:
    #         XF, m, n = Conv2DLayer.pad_csr(input, shape, padding)
    #     else:
    #         XF = input
        
    #     if self.sparse:
    #         TZ = self.weight
    #         out_shape = self.out_shape

    #     else:
    #         p, q, ci, co = weight.shape
    #         mo, no = Conv2DLayer.get_output_size_sparse(in_height=shape[0], in_width=shape[1], weight=weight, stride=stride, padding=padding)


    #         # TZ @ XF
    #         # XF is csr
    #         # TZ is csr

    #         Z = np.pad(weight, ((0, m-p), (0, n-q), (0, 0), (0,0)), mode='constant')
    #         Z_ = sp.csr_array(Z.reshape(Z.shape[0]*Z.shape[1]*Z.shape[2], co).T, copy=False)
    #         nnz = Z_.indptr[1:] - Z_.indptr[:-1]

    #         W = Z_.data
            

    #         XF_T = sp.csc_array((XF.data, XF.indices, XF.indtpr), shape=XF.shape)



    #         i_shift = n * stride[0] * c
    #         j_shift = stride[1] * c
            
    #         ko = mo*no

    #         Z = np.pad(weight, ((0, m-p), (0, n-q), (0, 0), (0,0)), mode='constant')
    #         Z_ = sp.csr_array(Z.reshape(Z.shape[0]*Z.shape[1]*Z.shape[2], co).T, copy=False)
    #         nnz = Z_.indptr[1:] - Z_.indptr[:-1]
        
    #         new_data = np.repeat(Z_.data[None, :], ko, axis=0).reshape(-1)

    #         ind = np.arange(ko, dtype=np.int32)[:, np.newaxis]
    #         ind = (ind//no)*i_shift + (ind%no)*j_shift
    #         new_indices = ind + Z_.indices
        
    #         new_indptr = np.hstack([Z_.indptr, ((np.arange((ko-1)*co, dtype=np.int32) + 1 + Z_.shape[0]).reshape(ko-1, co) * nnz).reshape(-1)] )

    #         TZ = sp.csr_array((new_data, new_indices.reshape(-1), new_indptr), shape=(ko*co, Z_.shape[1]), copy=False)

    #         out_shape = (mo, no, co)
        
    #     O = TZ @ XF
    #     return O, out_shape
    


    
    def fconv2d_csr(self, input, shape):
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
        dilation = self.dilation
        weight = self.weight
        
        assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
        'error: input should be a scipy sparse csr array or matrix'

        m, n, c = shape

        if padding[0] > 0 or padding[1] > 0:
            XF, m, n = Conv2DLayer.pad_csr(input, shape, padding)
        else:
            XF = input
        
        if self.sparse:
            TZ = self.weight
            out_shape = self.out_shape

        else:
            p, q, ci, co = weight.shape
            mo, no = self.get_output_size_sparse(in_height=shape[0], in_width=shape[1])

            i_shift = n * stride[0] * c
            j_shift = stride[1] * c
            
            ko = mo*no

            Z = np.pad(weight, ((0, m-p), (0, n-q), (0, 0), (0,0)), mode='constant')
            Z_ = sp.csr_array(Z.reshape(np.prod(Z.shape[:3]), co).T, copy=False)
            nnz = Z_.indptr[1:] - Z_.indptr[:-1]
            Z_ind = Z_.indices.copy()

            if dilation[0] > 1:
                q_ind = Z_.indices // c % p #col
                Z_ind += q_ind*(dilation[0]-1)*c

            if dilation[1] > 1:
                p_ind = Z_.indices // (c*q) #row
                Z_ind += p_ind*(dilation[1]-1)*c*q
        
            data = np.repeat(Z_.data[None, :], ko, axis=0).reshape(-1)

            indices = np.arange(ko, dtype=np.int32)[:, np.newaxis]
            indices = ((indices//no)*i_shift + (indices%no)*j_shift + Z_ind).reshape(-1)
        
            indptr = np.hstack([Z_.indptr, ((np.arange((ko-1)*co, dtype=np.int32) + 1 + Z_.shape[0]).reshape(ko-1, co) * nnz).reshape(-1)])

            TZ = sp.csr_array((data, indices, indptr), shape=(ko*co, Z_.shape[1]), copy=False)

            out_shape = (mo, no, co)
        
        O = TZ @ XF
        return O, out_shape

    def fconv2d_csr_co_loop(self, input, shape):
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
        dilation = self.dilation
        weight = self.weight
        
        assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
        'error: input should be a scipy sparse csr array or matrix'

        dtype = input.dtype
        b = input.shape[1]
        m, n, c = shape
        p, q, ci, co = weight.shape
        mo, no = self.get_output_size_sparse(in_height=shape[0], in_width=shape[1])

        if padding[0] > 0 or padding[1] > 0:
            XF, m, n = Conv2DLayer.pad_csr(input, shape, padding)
            c = shape[2]
        else:
            XF = input
            
        K = np.pad(weight, ((0, 0), (0, n-q), (0, 0), (0,0)), mode='constant') 
                
        i_shift = n*stride[0]*c
        j_shift = stride[1]*c
        
        ko = mo*no
        to = ko*co
        
        val_list = []
        row_list = []
        col_list = []

        # data_list = np.array([None for _ in range(to)])
        # indices_list = np.array([None for _ in range(to)])
        # nnz = np.zeros(to, dtype=np.int32)
        # indptr = np.zeros(to + 1, dtype=np.int32)
        for o in range(co):
            K_ = sp.csr_array(K[:, :, :, o].reshape(1, -1), copy=False)
            K_ind = K_.indices.copy()

            if dilation[0] > 1:
                q_ind = K_.indices // c % p #col
                K_ind += q_ind*(dilation[0]-1)*c

            if dilation[1] > 1:
                p_ind = K_.indices // (c*q) #row
                K_ind += p_ind*(dilation[1]-1)*c*q

            data = np.repeat(K_.data[None, :], ko, axis=0).reshape(-1)
            
            indices = np.arange(ko, dtype=np.int32)[:, np.newaxis]
            indices = ((indices//no)*i_shift + (indices%no)*j_shift + K_ind).reshape(-1)

            indptr = np.hstack([K_.indptr, (np.arange(ko-1, dtype=np.int32)+2)*K_.nnz])        

            TK = sp.csr_array((data, indices, indptr), shape=(ko, XF.shape[0]), copy=False)

            P = (TK @ XF).tocoo(copy=False)

            val_list.append(P.data)
            row_list.append((P.row*co + o).astype(np.int32))
            col_list.append(P.col.astype(np.int32))


        # O = sp.csr_array((np.hstack(data_list), np.hstack(indices_list), indptr), shape=(to, b))
        O = sp.csr_array((np.hstack(val_list), (np.hstack(row_list), np.hstack(col_list))), shape=(to, b), copy=False)
        # O = sp.csr_array((to, b))
        # O.data = np.hstack(val_list).astype(dtype)
        # O.indices = np.hstack(ind_list).astype(np.int32)
        # O.indptr = indptr.astype(np.int32)
        out_shape = (mo, no, co)
        return O, out_shape
    
    def fconv2d_csr_co_loop2(self, input, shape):
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
        dilation = self.dilation
        weight = self.weight
        
        assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
        'error: input should be a scipy sparse csr array or matrix'

        dtype = input.dtype
        b = input.shape[1]
        m, n, c = shape
        p, q, ci, co = weight.shape
        mo, no = self.get_output_size_sparse(in_height=shape[0], in_width=shape[1])

        if padding[0] > 0 or padding[1] > 0:
            XF, m, n = Conv2DLayer.pad_csr(input, shape, padding)
            c = shape[2]
        else:
            XF = input
            
        K = np.pad(weight, ((0, 0), (0, n-q), (0, 0), (0,0)), mode='constant') 
                
        i_shift = n*stride[0]*c
        j_shift = stride[1]*c
        
        ko = mo*no
        to = ko*co

        data = np.array([None for _ in range(to)])
        indices = np.array([None for _ in range(to)])
        nnz = np.zeros(to, dtype=np.int32)
        for o in range(co):
            K_ = sp.csr_array(K[:, :, :, o].reshape(1, -1), copy=False)
            K_ind = K_.indices.copy()

            if dilation[0] > 1:
                q_ind = K_.indices // c % p #col
                K_ind += q_ind*(dilation[0]-1)*c

            if dilation[1] > 1:
                p_ind = K_.indices // (c*q) #row
                K_ind += p_ind*(dilation[1]-1)*c*q
            
            data = np.repeat(K_.data[None, :], ko, axis=0).reshape(-1)
            
            indices = np.arange(ko, dtype=np.int32)[:, np.newaxis]
            indices = ((indices//no)*i_shift + (indices%no)*j_shift + K_ind).reshape(-1)

            indptr = np.hstack([K_.indptr, (np.arange(ko-1, dtype=np.int32)+2)*K_.nnz])        

            TK = sp.csr_array((data, indices, indptr), shape=(ko, XF.shape[0]), copy=False)

            P = TK @ XF

            for i in range(ko):
                indx = o + i*co
                i_ = P.indptr[i]
                i_1 = P.indptr[i+1]
                data[indx] = P.data[i_:i_1]
                indices[indx] = P.indices[i_:i_1]
                nnz[indx] = i_1 - i_

        data = np.hstack(data)
        indices = np.hstack(indices)

        sum = nnz.sum()
        if sum < pow(2, 31):
            indptr = np.zeros(to + 1, dtype=np.int32)
        else:
            indptr = np.zeros(to + 1, dtype=np.int64)
            indices = indices.astype(np.int64)

        for i in range(to):
            indptr[i+1] = indptr[i] + nnz[i]

        O = sp.csr_array((data, indices, indptr), shape=(to, b), copy=False)
        out_shape = (mo, no, co)
        return O, out_shape

    def reachSingleInput(self, In):
        if isinstance(In, ImageStar):

            assert In.V.ndim == 4, 'error: for Conv2D, basis matrix should be in 4D numpy ndarray'

            if self.module == 'pytorch':
                new_V = self.conv2d_pytorch(In.V, bias=False)

            elif self.module == 'default':
                new_V = self.conv2d(In.V, bias=False)

            if self.bias is not None:
                new_V[:, :, :, 0] += self.bias

            return ImageStar(new_V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        elif isinstance(In, SparseImageStar):
            if self.module == 'pytorch':
                raise Exception(
                    'Conv2DLayer does not support \'pyotrch\' moudle for SparseImageStar set'
                )
            
            elif self.module == 'default':
                new_c = self.conv2d(In.c, bias=True)
                new_V = self.conv2d_sparse(In.V, bias=False)
                
            return SparseImageStar(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        elif isinstance(In, SparseImageStar2DCOO):
            if self.module == 'pytorch':
                raise Exception(
                    'Conv2DLayer does not support \'pyotrch\' moudle for SparseImageStar set'
                )
            
            elif self.module == 'default':
                new_c = self.conv2d(In.c.reshape(In.shape), bias=True).reshape(-1)
                if self.sparse:
                    new_V, out_shape = self.fconv2d_coo(In.V, In.shape)
                else:
                    new_V, out_shape = self.fconv2d_coo_co_loop(In.V, In.shape)

            return SparseImageStar2DCOO(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)
        
        elif isinstance(In, SparseImageStar2DCSR):
            if self.module == 'pytorch':
                raise Exception(
                    'Conv2DLayer does not support \'pyotrch\' moudle for SparseImageStar set'
                )
            
            elif self.module == 'default':
                new_c = self.conv2d(In.c.reshape(In.shape), bias=True).reshape(-1)
                if self.sparse:
                    new_V, out_shape = self.fconv2d_csr(In.V, In.shape)
                else:
                    new_V, out_shape = self.fconv2d_csr_co_loop(In.V, In.shape)

            return SparseImageStar2DCSR(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)
        
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















































    
    

    

    def get_convolved_output_size_sparse(in_height, in_width, in_channel, in_batch, weight, stride, padding, dilation):
        h, w, c, n = in_height, in_width, in_channel, in_batch
        H, W, C, F = weight.shape
        ho = np.floor(
            ((h + 2*padding[0] - H - (H - 1) * (dilation[0] - 1)) // stride[0]) + 1
        ).astype(int)
        wo = np.floor(
            ((w + 2*padding[1] - W - (W - 1) * (dilation[1] - 1)) // stride[1]) + 1
        ).astype(int)
        
        assert ho > 0 and wo > 0, 'error: the shape of resulting output should be positive'
        return ho, wo
    
    
    
    
    
    def add_zero_padding_sparse(input, H, W, C, N, padding):

        assert isinstance(input, sp.coo_matrix), \
        'error: input should be scipy sparse coo matrix'

        if padding[0] == 0 and padding[1] == 0:
            return input

        input = input.toarray().reshape(H, W, C, N)
        
        in_dim = input.ndim
        if in_dim == 4:
            h, w, c, n = input.shape
            out = np.zeros(
                (h + 2*padding[0], w + 2*padding[1], c, n)
            )
            out[padding[0]:h+padding[0], padding[1]:w+padding[1], :, :] = input

        elif in_dim == 3:
            h, w, c = input.shape
            out = np.zeros(
                (h + 2*padding[0], w + 2*padding[1], c)
            )
            out[padding[0]:h+padding[0], padding[1]:w+padding[1], :] = input

        elif in_dim == 2:
            h, w = input.shape
            out = np.zeros(
                (h + 2*padding[0], w + 2*padding[1])
            )
            out[padding[0]:h+padding[0], padding[1]:w+padding[1]] = input

        else:
            raise Exception(
                'Invalid number of input dimensions; it should be between 2D and 4D'
            )

        out = sp.coo_array(out.reshape(-1, N))

        return out

























    def conv2d_basic_channel_first(self, input):
        """ 
        
        """
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        dtype = input.dtype
        
        if len(input.shape) == 2:
            input = input.reshape(1, 1, input.shape[0], input.shape[1])
        elif len(input.shape) == 3:
            input = input.reshape(1, input.shape[0], input.shape[1], input.shape[2])
            
        if len(self.weight.shape) == 2:
            self.weight = self.weight.reshape(1, self.weight.shape[0], self.weight.shape[1])
            
        input = input.astype(np.float64)
            
        n, c, h, w = input.shape
        F, C, H, W = self.weight.shape
  
        ho, wo = Conv2DLayer.get_convolved_output_size(input=input, weight=self.weight, \
                                                       stride=stride, padding=padding, dilation=dilation)
        
        pad_input = Conv2DLayer.add_zero_padding_channel_first(input, padding)
        pn, pm = pad_input.shape[2:]
        assert pn >= H and pm >= W, 'error: kernel shape should not be bigger than that of input'
     
        out = np.zeros((n, F, ho, wo), dtype=dtype)
        
        b = H // 2, W // 2
        center_h0 = b[0] * dilation[0]
        center_w0 = b[1] * dilation[1] 
        
        odd = H % 2, W % 2
        
        for z in range(n):
            working_input = pad_input[z, :, :, :]
            
            for k in range(F):
                out_ch = np.zeros((ho, wo), dtype=dtype)

                for i in range(ho):
                    center_h = center_h0 + stride[0] * i
                    indices_h = [center_h + l * dilation[0] for l in range(-b[0], b[0] + odd[0])]

                    for j in range(wo):
                        center_w = center_w0 + stride[1] * j
                        indices_w = [center_w + l * dilation[1] for l in range(-b[1], b[1] + odd[1])]

                        feature_map = working_input[:, indices_h, :][:, : , indices_w]
                        out_ch[i, j] = np.sum(feature_map * self.weight[k, :, :, :]).astype(np.float64)

                if self.bias is not None:
                    out_ch += self.bias[k]

                out[z, k, :, :] = out_ch
        
        return out
        
    def conv2d_vec_channel_first(self, input, bias=True):
        # vectorized convolution 2d

        # assert self.module == 'default', 'error: conv2d_vec support'
        assert isinstance(input, np.ndarray), \
        'error: input should be numpy array'
        
        # N C H W
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        dtype = input.dtype
        
        if len(input.shape) == 2:
            input = input.reshape(1, 1, input.shape[0], input.shape[1])
        elif len(input.shape) == 3:
            input = input.reshape(1, input.shape[0], input.shape[1], input.shape[2])
            
        if len(self.weight.shape) == 2:
            self.weight = self.weight.reshape(1, self.weight.shape[0], self.weight.shape[1])
            
        input = input.astype(self.numpy_dtype)

        n, c, h, w = input.shape
        F, C, H, W = self.weight.shape

        ho = np.floor(
            ((h + 2*padding[0] - H - (H - 1) * (dilation[0] - 1)) // stride[0]) + 1
        ).astype(int)
        wo = np.floor(
            ((w + 2*padding[1] - W - (W - 1) * (dilation[1] - 1)) // stride[1]) + 1
        ).astype(int)

        
        assert ho > 0 and wo > 0, 'error: the shape of resulting output should be positive'
            
        pn, pm = input.shape[2:]
        pad_input = Conv2DLayer.add_zero_padding_channel_first(input, padding)        
        
        assert pn >= H and pm >= W, 'error: kernel shape should not be bigger than that of input'
        
        # 2d input to column vectors or so called windows 
        out = np.zeros((n, F, ho, wo), dtype=dtype)

        b = H // 2, W // 2
        center_h0 = b[0] * dilation[0]
        center_w0 = b[1] * dilation[1]
        
        odd = H % 2, W % 2
        
        for z in range(n):
            working_input = pad_input[z, :, :, :]
            
            col_input = np.zeros((c*H*W, ho*wo), dtype=dtype)
        
            for i in range(ho):
                center_h = center_h0 + stride[0] * i
                indices_h = [center_h + l * dilation[0] for l in range(-b[0], b[0] + odd[0])]

                for j in range(wo):
                    center_w = center_w0 + stride[1] * j
                    indices_w = [center_w + l * dilation[1] for l in range(-b[1], b[1] + odd[1])]

                    feature_map = working_input[:, indices_h, :][:, : , indices_w]
                    # convert feature_map to column vector
                    col_input[:, i*wo+j] = feature_map.reshape(-1)

            # convert weight matrix into row vector
            weight_row = np.reshape(self.weight, (F, -1))
            
            out_n = np.matmul(weight_row, col_input)
            
            if bias is True:
                if self.bias is not None:
                    out_n += self.bias[:, np.newaxis]
            
            # row vector to image 
            out[z, :, :, :] = out_n.reshape(F, ho, wo)
            
        return out
    

    # add padding to 2D input
    def add_zero_padding_channel_first(input, padding):
        if padding[0] == 0 and padding[1] == 0:
            return input
        
        in_dim = input.ndim
        if in_dim == 4:
            f, c, h, w = input.shape
            out = np.zeros(
                (f, c, h + 2*padding[0], w + 2*padding[1])
            )
            out[:, :, padding[0]:h+padding[0], padding[1]:w+padding[1]] = input

        elif in_dim == 3:
            c, h, w = input.shape
            out = np.zeros(
                (c, h + 2*padding[0], w + 2*padding[1])
            )
            out[:, padding[0]:h+padding[0], padding[1]:w+padding[1]] = input

        elif in_dim == 2:
            h, w = input.shape
            out = np.zeros(
                (h + 2*padding[0], w + 2*padding[1])
            )
            out[padding[0]:h+padding[0], padding[1]:w+padding[1]] = input

        else:
            raise Exception(
                'Invalid number of input dimensions; it should be between 2D and 4D'
            )

        return out