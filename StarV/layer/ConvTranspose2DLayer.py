"""
Transposed Convolutional 2D Layer Class
Sung Woo Choi, 07/27/2024
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

from timeit import default_timer as timer

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
                    - kernel_weight: (Co, Ci, H, W); Co: output channel, Ci: input channel, H: height, W: width
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

            # kernel weight in shape (kernel_height, kernel_width, ch_in, ch_out)
            self.in_channel = kernel_weight.shape[1]
            self.out_channel = kernel_weight.shape[0]

            if kernel_bias is not None:
                assert isinstance(kernel_bias, np.ndarray) and kernel_bias.ndim == 1, \
                'error: kernel bias should be 1D numpy array' 
                assert kernel_bias.shape[0] == kernel_weight.shape[2], \
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
                    self.padding = np.ones(2, dtype=np.int16)*padding
                else:
                    padding = np.array(padding)
                    assert (padding >= 0).any(), 'error: padding should non-negative integers'

                    if len(padding) == 1:
                        self.padding = np.ones(2, dtype=np.int16)*padding[0]
                    else:
                        if len(padding) == 2:
                            padding = np.array([padding[0], padding[0], padding[1], padding[1]])
                        elif len(padding) == 4:
                            self.padding = padding
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
                        assert dilation[0] > 0 and dilation[1], 'error: dilation should positive integer'
                        self.dilation = np.array(dilation)
                    else:
                        raise Exception('error: incorrect dilation')

                if output_padding == None:
                    self.output_padding = np.zeros(2)
                elif isinstance(output_padding, int):
                    self.output_padding = np.ones(2, dtype=np.int16)*output_padding
                else:
                    if len(output_padding) == 1:
                        assert output_padding[0] > 0, 'error: output_padding should positive integer'
                        self.output_padding = np.ones(2, dtype=np.int16)*output_padding[0]
                    elif len(output_padding) == 2:
                        assert output_padding[0] > 0 and output_padding[1] > 0, 'error: output_padding should positive integers'
                        self.output_padding = np.array(output_padding)
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
                    kernel_size = kernel_weight.shape[2:3],
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

            # kernel weight in shape (ch_out, ch_in, kernel_height, kernel_width)
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

    def pad_csr(input, shape, padding, tocsc=False):
        """Adding padding to csr"""
        output, mo, no = ConvTranspose2DLayer.pad_coo(input, shape, padding, tocsc=False)
        return output.tocsr(False), mo, no

    def get_output_size(self, input):
        h, w, c, n = input.shape
        H, W = self.weight.shape[:2]

        ho = (h - 1)*self.stride[0] - 2*self.padding[0] + self.dilation[0]*(H - 1) + self.output_padding[0] + 1
        wo = (w - 1)*self.stride[1] - 2*self.padding[1] + self.dilation[1]*(W - 1) + self.output_padding[1] + 1

        assert ho > 0 and wo > 0, 'error: the shape of resulting output should be positive'
        return ho, wo
    
    def get_output_size_sparse(self, in_height, in_width):
        h, w = in_height, in_width
        H, W = self.weight.shape[:2]

        ho = (h - 1)*self.stride[0] - 2*self.padding[0] + self.dilation[0]*(H - 1) + self.output_padding[0] + 1
        wo = (w - 1)*self.stride[1] - 2*self.padding[1] + self.dilation[1]*(W - 1) + self.output_padding[1] + 1

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
            return self.convtrans2d_pytorch(input, bias=True)
        
        else:
            return self.convtrans2d(input, bias=True)
        

    def convtrans2d_pytorch(self, input, bias=True):
        """
            Args:
               @input: dataset in pytorch with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: transpose convolved dataset
        """
        
        assert isinstance(self.layer, torch.nn.ConvTranspose2d), '\'layer\' should be torch.nn.ConvTranspose2d for \'pytorch\' module'

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

        layer = self.layer
        if bias == True:
            layer.bias = torch.nn.Parameter(self.bias)
        else:
            layer.bias = None

        output = layer(input).detach().numpy()
        # change input shape to H, W, C, N
        output.transpose([2, 3, 1, 0])
        
        # if in_dim == 3:
        #     output = output.reshape(H, W, C) 

        return output
    
    def convtrans2d(self, input, bias=True):
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

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'
        assert in_dim >= 2 and in_dim <= 4, \
        'error: input should be 2D, 3D, or 4D numpy ndarray'

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

        # if in_dim == 3 or in_dim == 2:
        #     output = output.squeeze(axis = 3)
        return output
    
    def convtrans2d_test(self, input, bias=True):
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
        output_padding = self.output_padding
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

        mo, no, _, b = input.shape
        p, q, ci, co = weight.shape

        mi = (mo - 1)*self.stride[0] + dilation[0]*(p - 1) + 1
        ni = (no - 1)*self.stride[1] + dilation[1]*(q - 1) + 1
        m, n = mi, ni

        K = np.pad(weight, ((0, 0), (0, ni-q), (0, 0), (0,0)), mode='constant') 

        i_shift = ni * stride[0] * ci
        j_shift = stride[1] * ci

        S = np.zeros([mo*no*co, p*q*ci])

        print('weight.shape: ', weight.shape)
        prev_co_ = None
        for o in range(mo*no*co):
            
            K_ = sp.csr_array(K[:, :, :, o%co].reshape(1, -1), copy=False)
            K_ind = K_.indices.copy()

            if dilation[0] > 1:
                q_ind = K_.indices // ci % p #col
                K_ind += q_ind*(dilation[0]-1)*ci

            if dilation[1] > 1:
                p_ind = K_.indices // (ci*q) #row
                K_ind += p_ind*(dilation[1]-1)*ci*q       

            indices = (o//co//no)*i_shift + (o//co%no)*j_shift + K_ind

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

        K = weight.reshape(z, co)
        output = X @ K #in shape [b, ko, co]
        if bias is True:
            if isinstance(self.bias, np.ndarray):
                output += self.bias[None, None, :]
        output = output.transpose([1, 2, 0]).reshape(mo, no, co, b)
        return output
   
    

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
            q_ind = Z_.indices // ci % p #col
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
            q_ind = Z_.indices // ci % p #col
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
                new_V = self.conv2d_pytorch(In.V, bias=False)

            elif self.module == 'default':
                new_V = self.convtrans2d(In.V, bias=False)

            if self.bias is not None:
                new_V[:, :, :, 0] += self.bias

            return ImageStar(new_V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        elif isinstance(In, SparseImageStar2DCOO):
            if self.module == 'pytorch':
                raise Exception(
                    'Conv2DLayer does not support \'pyotrch\' moudle for SparseImageStar set'
                )
            
            elif self.module == 'default':
                new_c = self.convtrans2d(In.c.reshape(In.shape), bias=True).reshape(-1)
                new_V, out_shape = self.fconvtrans2d_coo_co_loop2(In.V, In.shape)
                # new_V, out_shape = self.fconvtrans2d_coo(In.V, In.shape)

            return SparseImageStar2DCOO(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub, out_shape)
        
        elif isinstance(In, SparseImageStar2DCSR):
            if self.module == 'pytorch':
                raise Exception(
                    'Conv2DLayer does not support \'pyotrch\' moudle for SparseImageStar set'
                )
            
            elif self.module == 'default':
                new_c = self.convtrans2d(In.c.reshape(In.shape), bias=True).reshape(-1)
                new_V, out_shape = self.fconvtrans2d_csr_co_loop2(In.V, In.shape)
                # new_V, out_shape = self.fconvtrans2d_csr(In.V, In.shape)

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
