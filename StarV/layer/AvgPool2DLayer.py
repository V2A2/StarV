"""
Average Pooling 2D Layer Class
Sung Woo Choi, 11/16/2023
"""

import time
import copy
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

class AvgPool2DLayer(object):
    """ AvgPool2DLayer Class
    
        properties:

        methods:
        

    """

    def __init__(
            self,
            kernel_size = 1,
            stride = 1, # e.g.: stride = (2, 2) or [3, 3] or 1
            padding = 0, # e.g.: padding = (0, 0) or [1, 1] or 2
            module = 'default', # 'default' or 'pytorch'
            dtype = 'float64', # 'float64' or 'float32'
        ):
        
        assert module in ['default', 'pytorch'], \
        'error: Conv2DLayer supports moudles: \'default\', which use numpy kernels, and \'pytorch\''
        self.module = module

        if dtype == 'float32':
            self.numpy_dtype = np.float32
            self.torch_dtype = torch.float32
        else:
            self.numpy_dtype = np.float64
            self.torch_dtype = torch.float64

        if self.module == 'default':
            # check stride, padding, and dilation

            assert isinstance(kernel_size, tuple) or isinstance(kernel_size, list) or isinstance(kernel_size, int), \
            'error: kernel_size should be a tuple, list, or int'
            assert isinstance(stride, tuple) or isinstance(stride, list) or isinstance(stride, int), \
            'error: stride should be a tuple, list, or int'
            assert isinstance(padding, tuple) or isinstance(padding, list) or isinstance(padding, int), \
            'error: padding should be a tuple, list, or int'

            if isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
                if len(kernel_size) == 1:
                    assert kernel_size[0] >= 0, 'error: kernel size should non-negative integer'
                    self.kernel_size = np.ones(2, dtype=np.uint16)*kernel_size[0]
                elif len(kernel_size) == 2:
                    assert kernel_size[0] >= 0 and kernel_size[1] >= 0, 'error: kernel size should non-negative integers'              
                    self.kernel_size = np.array(kernel_size).astype(np.uint16)
                else:
                    raise Exception('error: incorrect kernel size')
            else:
                assert kernel_size >= 0, 'error: kernel_size should non-negative integer'
                self.kernel_size = np.ones(2, dtype=np.uint16)*kernel_size
 
            if isinstance(padding, tuple) or isinstance(padding, list):
                if len(padding) == 1:
                    assert padding[0] >= 0, 'error: padding should non-negative integer'
                    self.padding = np.ones(2, dtype=np.uint16)*padding[0]
                elif len(padding) == 2:
                    assert padding[0] >= 0 and padding[1] >= 0, 'error: padding should non-negative integers'              
                    self.padding = np.array(padding).astype(np.uint16)
                else:
                    raise Exception('error: incorrect padding')
            else:
                assert padding >= 0, 'error: padding should non-negative integer'
                self.padding = np.ones(2, dtype=np.uint16)*padding
            assert (self.padding <= self.kernel_size // 2).any(), 'error: padding should be at most half of kernel size'
                
            if isinstance(stride, tuple) or isinstance(stride, list):
                if len(stride) == 1:
                    assert stride[0] > 0, 'error: stride should positive integer'
                    self.stride = np.ones(2, dtype=np.uint16)*stride[0]
                elif len(stride) == 2:
                    assert stride[0] > 0 and stride[1] > 0, 'error: stride should positive integers'
                    self.stride = np.array(stride).astype(np.uint16)
                else:
                    raise Exception('error: incorrect padding')
            else:
                assert stride > 0, 'error: stride should positive integer'
                self.stride = np.ones(2, dtype=np.uint16)*stride

        elif self.module == 'pytorch':

            self.layer = torch.nn.AvgPool2d(
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
            )


    def info(self):
        print('Average Pooling 2D Layer')
        print('module: {}'.format(self.module))
        print('kernel size: {}'.format(self.kernel_size))
        print('stride: {}'.format(self.stride))
        print('padding: {}'.format(self.padding))
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
            return AvgPool2DLayer.pad_coo(input.tocoo(False), shape, padding)
        else:
            return AvgPool2DLayer.pad_coo(input.tocoo(False), shape, padding, tocsr=True)
        
    # def pad_csr(input, shape, padding, tocsc=False, tocoo=False):
    #     output, mo, no = AvgPool2DLayer.pad_coo(input.tocoo(False), shape, padding, tocsc=tocsc)
    #     if tocsc is True or tocoo is True:
    #         return output, mo, no
    #     else:
    #         return output.tocsr(False), mo, no
        
    def add_zero_padding(input, padding):

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'

        if padding[0] == 0 and padding[1] == 0:
            return input
        
        in_dim = input.ndim
        if in_dim == 4:
            h, w, c, n = input.shape
            out = np.zeros(
                (h + 2*padding[0], w + 2*padding[1], c, n), dtype=input.dtype
            )
            out[padding[0]:h+padding[0], padding[1]:w+padding[1], :, :] = input

        elif in_dim == 3:
            h, w, c = input.shape
            out = np.zeros(
                (h + 2*padding[0], w + 2*padding[1], c), dtype=input.dtype
            )
            out[padding[0]:h+padding[0], padding[1]:w+padding[1], :] = input

        elif in_dim == 2:
            h, w = input.shape
            out = np.zeros(
                (h + 2*padding[0], w + 2*padding[1]), dtype=input.dtype
            )
            out[padding[0]:h+padding[0], padding[1]:w+padding[1]] = input

        else:
            raise Exception(
                'Invalid number of input dimensions; it should be between 2D and 4D'
            )

        return out
    
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

    def get_output_size_sparse(in_height, in_width, kernel_size, stride, padding):
        h, w = in_height, in_width
        H, W = kernel_size

        ho = (h + 2*padding[0] - H) // stride[0] + 1
        wo = (w + 2*padding[1] - W) // stride[1] + 1
        
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
            return self.avgpool2d_pytorch(input)
        else:
            return self.avgpool2d(input)
        
        
    def avgpool2d_pytorch(self, input, bias=True):
        """
            Args:
               @input: dataset in pytorch with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: convolved dataset
        """
        
        assert isinstance(self.layer, torch.nn.AvgPool2d), '\'layer\' should be torch.nn.AvgPool2d for \'pytorch\' module'

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
        
        if in_dim == 3:
            output = output.reshape(H, W, C) 

        return output


    def avgpool2d_basic(self, input):
        """ 
            Average pooling 2D

            Args:
               @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: convolved dataset
        
        """
        stride = self.stride
        padding = self.padding

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'

        dtype = input.dtype
        in_dim = input.ndim
        assert in_dim >= 2 and in_dim <= 4, \
        'error: input should be 2D, 3D, or 4D numpy ndarray'

        input = copy.deepcopy(input)
        if in_dim == 2:
            input = input[:, :, None, None]
        elif in_dim == 3:
            input = input[:, :, :, None]
        
        h, w, c, n = input.shape
        H, W = self.kernel_size
        ho, wo = AvgPool2DLayer.get_output_size(h, w, self.kernel_size, stride, padding)
        # pad_input = AvgPool2DLayer.add_zero_padding(input, padding)
        pad_input = np.pad(input, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0), (0,0)), mode='constant')

        output = np.zeros((ho, wo, c, n), dtype=dtype)
        for z in range(n):
        
            for k in range(c):
                working_input = pad_input[:, :, k, z]
                out_ch = np.zeros((ho, wo), dtype=dtype)

                for i in range(ho):
                    i_stride = i*stride[0]

                    for j in range(wo):
                        j_stride = j*stride[1]
                        out_ch[i, j] = np.sum(working_input[i_stride : i_stride+H, 
                                                            j_stride : j_stride+W])

                output[:, :, k, z] = out_ch
        output = output / (H*W)

        if in_dim == 2:
            output = output.reshape(ho, wo)
        elif in_dim == 3:
            output = output.reshape(ho, wo, c)
        return output
    
    def avgpool2d(self, input):
        """ 
            Average pooling 2D

            Args:
               @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: convolved dataset
        
        """
        stride = self.stride
        padding = self.padding

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'

        dtype = input.dtype
        in_dim = input.ndim
        assert in_dim >= 2 and in_dim <= 4, \
        'error: input should be 2D, 3D, or 4D numpy ndarray'

        if in_dim == 2:
            input = input[:, :, None, None]
        elif in_dim == 3:
            input = input[:, :, :, None]
        
        m, n, c, b = input.shape
        p, q = self.kernel_size
        mo, no = AvgPool2DLayer.get_output_size(m, n, self.kernel_size, stride, padding)

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
                    output[i_, :] = XF[ind, :].sum(axis=0) #XF in [m*n*c, b]
                    i_ += 1
        output = output / (p*q)

        # i_shift = n * stride[0] * c
        # j_shift = stride[1]

        # noc = no*c
        # ko = mo*noc
        # output = np.zeros([ko, b])
        # for i in range(ko):
        #     ind = (i//noc)*i_shift + (i%noc)*j_shift + Z_indices
        #     output[i, :] = XF[ind, :].sum(axis=0) #XF in [m*n*c, b]
        # output = output / (p*q)

        ## slower
        # output = np.zeros([ko, p*q, b])
        # for i in range(ko):s
        #     ind = (i//noc)*i_shift + (i%noc)*j_shift + Z_indices
        #     output[i, :, :] = XF[ind, :] #XF in [m*n*c, b]
        # output = output.sum(axis=1) / (p*q)
            
        if in_dim == 2:
            output = output.reshape(mo, no)
        elif in_dim == 3:
            output = output.reshape(mo, no, c)
        else:
            output = output.reshape(mo, no, c, b)
        return output
    
    def avgpool2d_sparse(self, input):
        """
            Convolution 2D for sparse images

            Args:
                @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches
                @input: SparseImage
            Return: 
               @R: convolved dataset
        """

        assert self.module == 'default', 'error: avgpool2d_sparse() support \'default\' module'
        
        stride = self.stride
        padding = self.padding
        dtype = input.dtype
        
        assert isinstance(input, SparseImage), \
        'error: input should be a SparseImage'

        input = copy.deepcopy(input)
        h, w, c, n = input.size()
        H, W = self.kernel_size
        ho, wo = AvgPool2DLayer.get_output_size_sparse(h, w, self.kernel_size, stride, padding)

        h += 2*padding[0]
        w += 2*padding[1]

        div = H*W
        sp_im = []
        for n_ in range(n): # number of predicates
            
            im3d = SparseImage3D(ho, wo, c, n_)
            for c_ in range(c): # number of channels
                nnz, val = [], []
                im = input.images[n_].channel[c_]
                if im is None:
                    continue
                
                # apply padding to input
                row = im.row + padding[0]
                col = im.col + padding[1]
                data = im.data

                rl = np.minimum(ho, np.maximum(0, np.ceil((row - H + 1) / stride[0]))).astype(np.uint16)
                ru = np.minimum(ho, row // stride[0] + 1).astype(np.uint16)

                cl = np.minimum(wo, np.maximum(0, np.ceil((col - W + 1) / stride[1]))).astype(np.uint16)
                cu = np.minimum(wo, col // stride[1] + 1).astype(np.uint16)

                for o_ in range(len(row)):
                    for row_ in range(rl[o_], ru[o_]):
                        for col_ in range(cl[o_], cu[o_]):
                            nnz.append([row_, col_])
                            val.append(data[o_])

                nnz = np.array(nnz)
                val = np.array(val, dtype=dtype)

                #inverse_indices
                U, inv_indices = np.unique(nnz, axis=0, return_inverse=True)
                N = U.shape[0]

                new_v = []
                for q in range(N):
                    new_v.append(val[inv_indices == q].sum() / div)
                new_v = np.array(new_v)

                new_row = U[:, 0].reshape(-1)
                new_col = U[:, 1].reshape(-1)

                im2d = sp.coo_array((new_v, (new_row, new_col)), shape=(ho, wo))
                im3d.append(im2d, c_, n_)
            sp_im.append(im3d)        
        return SparseImage(sp_im)
    

    def favgpool2d_coo(self, input, shape):
        """
            Convolution 2D for sparse images

            Args:
                @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches
                @input: SparseImage
            Return: 
               @R: convolved dataset
        """

        stride = self.stride
        padding = self.padding

        assert isinstance(input, sp.coo_array) or isinstance(input, sp.coo_matrix), \
        'error: input should be a scipy sparse coo array or matrix'

        b = input.shape[1]
        m, n, c = shape
        p, q = self.kernel_size
        mo, no = AvgPool2DLayer.get_output_size(m, n, self.kernel_size, stride, padding)

        if padding[0] > 0 or padding[1] > 0:
            XF, m, n = AvgPool2DLayer.pad_coo(input, shape, padding, tocsr=True)
        else:
            XF = input.tocsr()
        
        K = np.zeros([p, q, c], dtype=bool)
        K[:, :, 0] = 1
        Z = np.pad(K, ((0, 0), (0, n-q), (0, 0)), mode='constant').reshape(-1)
        Z_indices = np.where(Z > 0)[0]

        i_shift = n*stride[0]*c
        j_shift = stride[1]*c
        

        data = []
        row = []
        col = []
        noc = no*c
        ko = mo*noc

        i_ = 0
        for i in range(mo):
            for j in range(no):
                for k in range(c):
                    ind = i * i_shift + j*j_shift + k + Z_indices
                    O = sp.coo_array(XF[ind, :].sum(axis=0))
                    data.append(O.data)
                    row.append(O.row+i_)
                    col.append(O.col)
                    i_ += 1

        output = sp.coo_array((np.hstack(data), (np.hstack(row), np.hstack(col))), shape=(ko, b))
        output.data = output.data / (p*q)
        
        # for i in range(ko):
        #     ind = (i//noc)*i_shift + (i%noc)*j_shift - i%2 + Z_indices
        #     O = sp.coo_array(XF[ind, :].sum(axis=0))
        #     data.append(O.data)
        #     row.append(O.row+i)
        #     col.append(O.col)
        # output = sp.coo_array((np.hstack(data), (np.hstack(row), np.hstack(col))), shape=(ko, b))
        # output.data = output.data / (p*q)

        out_shape = (mo, no, c)
        return output, out_shape
    
    def favgpool2d_coo2(self, input, shape):
        """
            Convolution 2D for sparse images

            Args:
                @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches
                @input: SparseImage
            Return: 
               @R: convolved dataset
        """

        stride = self.stride
        padding = self.padding
        dtype = input.dtype

        assert isinstance(input, sp.coo_array) or isinstance(input, sp.coo_matrix), \
        'error: input should be a scipy sparse coo array or matrix'

        b = input.shape[1]
        m, n, c = shape
        p, q = self.kernel_size
        mo, no = AvgPool2DLayer.get_output_size(m, n, self.kernel_size, stride, padding)

        if padding[0] > 0 or padding[1] > 0:
            XF, m, n = AvgPool2DLayer.pad_coo(input, shape, padding)
        else:
            XF = input
        
        K = np.zeros([p, q, c], dtype=bool)
        K[:, :, 0] = 1
        Z = np.pad(K, ((0, 0), (0, n-q), (0, 0)), mode='constant').reshape(-1)
        Z_indices = np.where(Z > 0)[0]

        z = p*q
        noc = no*c
        ko = mo*noc
        
        i_shift = (np.arange(ko, dtype=np.int32)[:, None]//noc) * n * c * stride[0] 
        j_shift = ((np.arange(c, dtype=np.int32) + np.arange(no, dtype=np.int32)[:, None] * stride[1] * c).reshape(-1) * (np.ones(mo, dtype=np.int32)[:, None])).reshape(-1, 1)
        
        data = np.ones(z*ko, dtype=dtype)
        indices = (i_shift + j_shift + Z_indices).reshape(-1)
        indptr = np.arange(ko+1, dtype=np.int32)*z


        # i_shift = n*stride[0]*c
        # j_shift = stride[1]

        # z = p*q
        # noc = no*c
        # ko = mo*noc

        # data = np.ones(z*ko)
        # indptr = np.arange(ko+1)*z

        # indices = np.arange(ko)[:, np.newaxis] 
        # indices = (indices//noc)*i_shift + (indices%noc)*j_shift - indices%2 + Z_indices
            
        TK = sp.csr_array((data, indices, indptr), shape=(ko, m*n*c))
        O = (TK @ XF).tocoo()
        O.data = O.data / z
        out_shape = (mo, no, c)
        return O, out_shape
    
    def favgpool2d_csr(self, input, shape):
        """
            Convolution 2D for sparse images

            Args:
                @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches
                @input: SparseImage
            Return: 
               @R: convolved dataset
        """

        stride = self.stride
        padding = self.padding

        assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
        'error: input should be a scipy sparse csr array or matrix'

        b = input.shape[1]
        m, n, c = shape
        p, q = self.kernel_size
        mo, no = AvgPool2DLayer.get_output_size(m, n, self.kernel_size, stride, padding)

        if padding[0] > 0 or padding[1] > 0:
            XF, m, n = AvgPool2DLayer.pad_csr(input, shape, padding)
        else:
            XF = input
        
        K = np.zeros([p, q, c], dtype=bool)
        K[:, :, 0] = 1
        Z = np.pad(K, ((0, 0), (0, n-q), (0, 0)), mode='constant').reshape(-1)
        Z_indices = np.where(Z > 0)[0]

        i_shift = n*stride[0]*c
        j_shift = stride[1]*c

        noc = no*c
        ko = mo*noc
        data = []
        indices = []
        indptr = np.zeros(ko+1, dtype=np.int32)
        i_ = 0
        for i in range(mo):
            for j in range(no):
                for k in range(c):
                    ind = i*i_shift + j*j_shift + k + Z_indices
                    O = sp.csr_array(XF[ind, :].sum(axis=0))
                    data.append(O.data)
                    indices.append(O.indices)
                    indptr[i_+1] = O.nnz + indptr[i_] 
                    i_ += 1    
        output = sp.csr_array((np.hstack(data), np.hstack(indices), indptr), shape=(ko, b))
        output.data = output.data / (p*q)

        # for i in range(ko):
        #     ind = (i//noc)*i_shift + (i%noc)*j_shift - i%2 + Z_indices
        #     O = sp.csr_array(XF[ind, :].sum(axis=0))
        #     data.append(O.data)
        #     indices.append(O.indices)
        #     indptr[i+1] = O.nnz + indptr[i] 
        # output = sp.csr_array((np.hstack(data), np.hstack(indices), indptr), shape=(ko, b))
        # output.data = output.data / (p*q)
        out_shape = (mo, no, c)
        return output, out_shape
    
    def favgpool2d_csr2(self, input, shape):
        """
            Convolution 2D for sparse images

            Args:
                @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches
                @input: SparseImage
            Return: 
               @R: convolved dataset
        """

        stride = self.stride
        padding = self.padding
        dtype = input.dtype

        assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
        'error: input should be a scipy sparse csr array or matrix'

        b = input.shape[1]
        m, n, c = shape
        p, q = self.kernel_size
        mo, no = AvgPool2DLayer.get_output_size(m, n, self.kernel_size, stride, padding)

        if padding[0] > 0 or padding[1] > 0:
            XF, m, n = AvgPool2DLayer.pad_csr(input, shape, padding)
        else:
            XF = input
        
        K = np.zeros([p, q, c], dtype=bool)
        K[:, :, 0] = 1
        Z = np.pad(K, ((0, 0), (0, n-q), (0, 0)), mode='constant').reshape(-1)
        Z_indices = np.where(Z > 0)[0]

        z = p*q
        noc = no*c
        ko = mo*noc
        
        i_shift = (np.arange(ko, dtype=np.int32)[:, None]//noc) * n * c * stride[0] 
        j_shift = ((np.arange(c, dtype=np.int32) + np.arange(no, dtype=np.int32)[:, None] * stride[1] * c).reshape(-1) * (np.ones(mo, dtype=np.int32)[:, None])).reshape(-1, 1)
        
        data = np.ones(z*ko, dtype=dtype)
        indices = (i_shift + j_shift + Z_indices).reshape(-1)
        indptr = np.arange(ko+1, dtype=np.int32)*z

        # i_shift = n*stride[0]*c
        # j_shift = stride[1]

        # z = p*q
        # noc = no*c
        # ko = mo*noc

        # data = np.ones(z*ko)
        # indices = []
        # indptr = np.arange(ko+1)*z

        # indices = np.arange(ko)[:, np.newaxis] 
        # indices = (indices//noc)*i_shift + (indices%noc)*j_shift - indices%2 + Z_indices
            
        TK = sp.csr_array((data, indices, indptr), shape=(ko, m*n*c))
        O = (TK @ XF)
        O.data = O.data / z
        out_shape = (mo, no, c)
        return O, out_shape
    
    # def favgpool2d_csr3(self, input, shape):
    #     """
    #         Convolution 2D for sparse images

    #         Args:
    #             @input: dataset in numpy with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches
    #             @input: SparseImage
    #         Return: 
    #            @R: convolved dataset
    #     """

    #     stride = self.stride
    #     padding = self.padding

    #     assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
    #     'error: input should be a scipy sparse csr array or matrix'

    #     b = input.shape[1]
    #     m, n, c = shape
    #     p, q = self.kernel_size
    #     mo, no = AvgPool2DLayer.get_output_size(m, n, self.kernel_size, stride, padding)

    #     if padding[0] > 0 or padding[1] > 0:
    #         XF, m, n = AvgPool2DLayer.pad_csr(input, shape, padding, tocoo=True)
    #     else:
    #         XF = input
        
    #     K = np.zeros([p, q, c], dtype=bool)
    #     K[:, :, 0] = 1
    #     Z = np.pad(K, ((0, 0), (0, n-q), (0, 0)), mode='constant').reshape(-1)
    #     Z_indices = np.argwhere(Z > 0).reshape(-1)

    #     i_shift = n*stride[0]*c
    #     j_shift = stride[1]

    #     z = p*q
    #     noc = no*c
    #     ko = mo*noc

    #     data = np.ones(z*ko)
    #     indices = []
    #     indptr = np.arange(ko+1)*z

    #     # for i in range(ko):
    #     #     indices.append((i//noc)*i_shift + (i%noc)*j_shift + Z_indices)

    #     indices = np.arange(ko)[:, np.newaxis]
    #     indices = (indices//noc)*i_shift + (indices%noc)*j_shift - indices%2 + Z_indices
            
    #     TK = sp.csr_array((data, np.hstack(indices), indptr), shape=(ko, m*n*c))
    #     O = (TK @ XF)
    #     O.data = O.data / z
    #     out_shape = (mo, no, c)
    #     return O, out_shape


    


    def reachExactSingleInput(self, In):
        if isinstance(In, ImageStar):
            if self.module == 'pytorch':
                new_V = self.avgpool2d_pytorch(In.V)

            elif self.module == 'default':
                new_V = self.avgpool2d(In.V)

            return ImageStar(new_V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        elif isinstance(In, SparseImageStar):
            if self.module == 'pytorch':
                raise Exception(
                    'AvgPool2DLayer does not support \'pyotrch\' moudle for SparseImageStar set'
                )
            
            elif self.module == 'default':
                new_c = self.avgpool2d(In.c)
                new_V = self.avgpool2d_sparse(In.V)
                
            return SparseImageStar(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        elif isinstance(In, SparseImageStar2DCOO):
            if self.module == 'pytorch':
                raise Exception(
                    'BatchNorm2DLayer does not support \'pyotrch\' moudle for SparseImageStar2DCOO set'
                )
            
            elif self.module == 'default':
                new_c = self.avgpool2d(In.c)
                new_V = self.favgpool2d_coo2(In.V)
                
            return SparseImageStar2DCOO(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        elif isinstance(In, SparseImageStar2DCSR):
            if self.module == 'pytorch':
                raise Exception(
                    'BatchNorm2DLayer does not support \'pyotrch\' moudle for SparseImageStar2DCSR set'
                )
            
            elif self.module == 'default':
                new_c = self.avgpool2d(In.c)
                new_V = self.favgpool2d_csr2(In.V)
                
            return SparseImageStar2DCSR(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        else:
            raise Exception('error: AvgPool2DLayer support ImageStar and SparseImageStar')



    def reach(self, inputSet, method=None, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """
            Args:
        
        """

        pool = None
        
        if isinstance(inputSet, list):
            S = []
            if pool is None:
                for i in range(0, len(inputSet)):
                    S.append(self.reachExactSingleInput(inputSet[i]))
            elif isinstance(pool, multiprocessing.pool.Pool):
                S = S + pool.map(self.reachExactSingleInput, inputSet)
            else:
                raise Exception('error: unknown/unsupport pool type')         

        else:
            S = self.reachExactSingleInput(inputSet)
                
        return S