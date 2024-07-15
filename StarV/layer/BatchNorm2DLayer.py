"""
Batch Normalization 2D Layer Class
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

class BatchNorm2DLayer(object):
    """ BatchNorm2DLayer Class
    
        properties:

            learnable parameters:
                gamma
                beta



        methods:
        

    """

    def __init__(
            self,
            layer, #[gamma weight, beta weight, mean, var] or torch.nn.BatchNorm2D
            num_features = None, 
            eps = 1e-05,
            module = 'default',
            dtype = 'float64',
        ):

        assert module in ['default', 'pytorch'], \
        'error: BatchNorm2DLayer supports moudles: \'default\', which use numpy kernels, and \'pytorch\''
        self.module = module

        if dtype == 'float32':
            self.numpy_dtype = np.float32
            self.torch_dtype = torch.float32
        else:
            self.numpy_dtype = np.float64
            self.torch_dtype = torch.float64

        # input 'layer' is list containing [gamma weight, beta weight, mean, var]
        if isinstance(layer, list):
            assert len(layer) == 4, \
            'error: \'layer\' should be a list containing gamma and beta weights, mean, and variance'

            gamma, beta, mean, var = copy.deepcopy(layer)
            assert isinstance(gamma, np.ndarray), 'error: gamma should be a 1D numpy array or None'
            assert isinstance(beta, np.ndarray), 'error: beta should be a 1D numpy array'
            assert isinstance(mean, np.ndarray), 'error: mean should be a 1D numpy array'
            assert isinstance(var, np.ndarray), 'error: variance should be a 1D numpy array'
            assert gamma.ndim == beta.ndim == 1, 'error: gamma and bias weights should be 1D numpy array'
            assert gamma.shape[0] == beta.shape[0] == num_features, 'error: inconsistency between gamma and bias weights and number of features'
            assert mean.ndim == var.ndim == 1, 'error: mean and variance should be 1D numpy array'
            assert mean.shape[0] == var.shape[0] == num_features, 'error: inconsistency between mean, variance, and number of features'

            if self.module == 'default':
                self.gamma = gamma.astype(self.numpy_dtype)
                self.beta = beta.astype(self.numpy_dtype)
                self.num_features = num_features
                self.eps = eps.astype(self.numpy_dtype)
                self.mean = mean.astype(self.numpy_dtype)
                self.var = var.astype(self.numpy_dtype)

            elif self.module == 'pytorch':
                self.layer = torch.nn.BatchNorm2d(
                    num_features = num_features,
                    eps = eps,
                    affine = True,    
                )
                # self.layer.weight = torch.nn.Parameter(torch.from_numpy(gamma).type(self.torch_dtype))
                # self.layer.bias = torch.nn.Parameter(torch.from_numpy(beta).type(self.torch_dtype))
                self.layer.weight.data = torch.from_numpy(gamma).type(self.torch_dtype)
                self.layer.bias.data = torch.from_numpy(beta).type(self.torch_dtype)
                self.layer.running_mean = torch.from_numpy(mean).type(self.torch_dtype)
                self.layer.running_var = torch.from_numpy(var).type(self.torch_dtype)
                
                
        # input 'layer' is torch.nn.BatchNorm2d layer
        elif isinstance(layer, torch.nn.BatchNorm2d):
            
            # converting weight and bias in pytorch to numpy 
            if self.module == 'default':
                self.gamma = layer.weight.detach().numpy().copy().astype(self.numpy_dtype)
                self.beta = layer.bias.detach().numpy().copy().astype(self.numpy_dtype)
                self.num_features = layer.num_features
                self.eps = layer.eps
                self.mean = layer.running_mean.numpy().astype(self.numpy_dtype)
                self.var = layer.running_var.numpy().astype(self.numpy_dtype)

            elif self.module == 'pytorch':
                self.layer = copy.deepcopy(layer)
                self.layer.weight = self.layer.weight.type(self.torch_dtype)
                self.layer.bias = self.layer.bias.type(self.torch_dtype)
                self.layer.running_mean = self.layer.running_mean.type(self.torch_dtype)
                self.layer.running_var = self.layer.running_var.type(self.torch_dtype)


        elif layer == None:

            assert num_features is None, 'error: num_features is not provided'
            
            if self.module == 'default':
                self.gamma = None
                self.beta = None
                self.num_features = num_features
                self.eps = eps
                self.mean = mean
                self.var = var

            elif self.module == 'pytorch':
                self.layer = torch.nn.BatchNorm2d(
                    num_features = num_features,
                    eps = eps,
                    affine = False,
                )
                self.layer.running_mean = torch.from_numpy(mean)
                self.layer.running_var = torch.from_numpy(var)

        else:
            raise Exception('Unknown layer module')
        
    def info(self):
        print(f"Batch Normalization 2D Layer")
        print(f"module: {self.module}")
        print(f"gamma: {self.gamma}")
        print(f"beta: {self.beta}")
        print(f"num_features: {self.num_features}")
        print(f"epsilon: {self.eps}")
        print(f"mean: {self.mean}")
        print(f"variance: {self.var}")
        return '\n'

    def evaluate(self, input):
        """
            For module == 'default' set up:
                @input: (H, W, C, N); H: height, W: width, C: input channel, N: batch or number of predicates

            For module == 'pytorch' set up:
                @input: (N, C, H, W); N: batch or number of predicates, C: input channel, H: height, W: width 
        """

        if self.module == 'pytorch':
            return self.batchnorm2d_pytorch(input)
        else:
            return self.batchnorm2d(input)

    def batchnorm2d_pytorch(self, input):
        """
            Args:
               @input: dataset in pytorch with shape of H, W, C, N, where H: height, W: width, C: input channel, N: number of batches

            Return: 
               @R: batch normalized dataset
        """

        assert isinstance(self.layer, torch.nn.BatchNorm2d), '\'layer\' should be torch.nn.BatchNorm2d for \'pytorch\' module'

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
        # set the layer in evaluation mode
        self.layer.eval()
        output = self.layer(input).detach().numpy()
        # change input shape to H, W, C, N
        output.transpose([2, 3, 1, 0])
        
        if in_dim == 3:
            output = output.reshape(H, W, C) 

        return output
    
    def batchnorm2d(self, input, bias=True):
        
        in_dim = input.ndim

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'
        assert in_dim >= 2 and in_dim <= 4, \
        'error: input should be 2D, 3D, or 4D numpy ndarray'

        input = copy.deepcopy(input)
        if in_dim == 2:
            input = input[:, :, None, None]
        elif in_dim == 3:
            input = input[:, :, :, None]
    
        h, w, c = input.shape[:3]
        F = self.num_features

        assert c == F, 'error: inconsistency between number of input channel and number of features'

        eps = self.eps
        var = self.var[None, None, :, None]
        gamma = self.gamma[None, None, :, None]

        if bias:
            mean = self.mean[None, None, :, None]
            beta = self.beta[None, None, :, None]
            output = gamma * (input - mean) / np.sqrt(var + eps) + beta

        else:
            output = gamma * input / np.sqrt(var + eps)

        if in_dim == 2:
            output = output.reshape(h, w)
        elif in_dim == 3:
            output = output.reshape(h, w, F)
        return output
    
    def batchnorm2d_basis_matrix(self, input, bias=True):
        
        in_dim = input.ndim

        assert isinstance(input, np.ndarray), \
        'error: input should be numpy ndarray'
        assert in_dim >= 2 and in_dim <= 4, \
        'error: input should be 2D, 3D, or 4D numpy ndarray'

        input = copy.deepcopy(input)
        if in_dim == 2:
            input = input[:, :, None, None]
        elif in_dim == 3:
            input = input[:, :, :, None]
    
        h, w, c = input.shape[:3]
        F = self.num_features

        assert c == F, 'error: inconsistency between number of input channel and number of features'

        eps = self.eps
        var = self.var[None, None, :, None]
        gamma = self.gamma[None, None, :, None]

        A = gamma / np.sqrt(var + eps)
        output = A*input
        
        if bias:
            mean = self.mean[None, None, :, None]
            beta = self.beta[None, None, :, None]
            b = -mean*A + beta 
            output[:, :, :, 0] += b[:, :, :, 0]

        # output = gamma * (input - mean) / np.sqrt(var + eps) + beta


        if in_dim == 2:
            output = output.reshape(h, w)
        elif in_dim == 3:
            output = output.reshape(h, w, F)
        return output

    def batchnorm2d_sparse(self, input):

        assert isinstance(input, SparseImage), \
        'error: input should be a SparseImage'

        eps = self.eps
        mean = self.mean
        var = self.var
        gamma = self.gamma
        beta = self.beta

        input = copy.deepcopy(input) #.astype(self.numpy_dtype)
        h, w, c, n = input.size()

        sp_im = []
        for n_ in range(n): # number of predicates
            
            im3d = SparseImage3D(h, w, c, n_)
            for c_ in range(c): # number of channels
                nnz, val = [], []
                im = input.images[n_].channel[c_]
                if im is None:
                    continue
                im = gamma[c_] * (im - mean[c_]) / np.sqrt(var[c_] + eps) + beta[c_]
                im3d.append(im, c_, n_)
            sp_im.append(im3d)        
        return SparseImage(sp_im)
    
    def fbatchnorm2d_coo(self, input, shape, bias=False):
        
        assert isinstance(input, sp.coo_array) or isinstance(input, sp.coo_matrix), \
        'error: input should be a scipy coo or csr array or matrix'

        eps = self.eps
        mean = self.mean
        var = self.var
        gamma = self.gamma
        c = shape[2]
        
        output = copy.deepcopy(input)
        row_ch = output.row % c
        if bias:
            mean = self.mean
            beta = self.beta
            output.data = gamma[row_ch] * (output.data - mean[row_ch]) / np.sqrt(var[row_ch] + eps) + beta[row_ch]
        else:
            output.data = gamma[row_ch] * output.data / np.sqrt(var[row_ch] + eps)
        return output
    
    def fbatchnorm2d_csr(self, input, shape, bias=False):
        
        assert isinstance(input, sp.csr_array) or isinstance(input, sp.csr_matrix), \
        'error: input should be a scipy coo or csr array or matrix'

        eps = self.eps
        
        var = self.var
        gamma = self.gamma
        beta = self.beta
        c = shape[2]
        
        T = input.tocoo(copy=False)
        output = copy.deepcopy(input)
        row_ch = T.row % c
        if bias:
            mean = self.mean
            beta = self.beta
            output.data = gamma[row_ch] * (output.data - mean[row_ch]) / np.sqrt(var[row_ch] + eps) + beta[row_ch]
        else:
            output.data = gamma[row_ch] * output.data / np.sqrt(var[row_ch] + eps)
        return output


    def reachExactSingleInput(self, In):
        if isinstance(In, ImageStar):
            if self.module == 'pytorch':
                new_V = self.batchnorm2d_pytorch(In.V)
                print('need to fix')

            elif self.module == 'default':
                new_V = self.batchnorm2d_basis_matrix(In.V)

            return ImageStar(new_V, In.C, In.d, In.pred_lb, In.pred_ub)

        elif isinstance(In, SparseImageStar):
            if self.module == 'pytorch':
                raise Exception(
                    'BatchNorm2DLayer does not support \'pyotrch\' moudle for SparseImageStar set'
                )
            
            elif self.module == 'default':
                new_c = self.batchnorm2d(In.c)
                new_V = self.batchnorm2d_sparse(In.V)
                
            return SparseImageStar(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        elif isinstance(In, SparseImageStar2DCOO):
            if self.module == 'pytorch':
                raise Exception(
                    'BatchNorm2DLayer does not support \'pyotrch\' moudle for SparseImageStar2DCOO set'
                )
            
            elif self.module == 'default':
                new_c = self.batchnorm2d(In.c.reshape(In.shape), bias=True).reshape(-1)
                new_V = self.fbatchnorm2d_coo(In.V, In.shape, bias=False)
                
            return SparseImageStar2DCOO(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub, In.shape)
        
        elif isinstance(In, SparseImageStar2DCSR):
            if self.module == 'pytorch':
                raise Exception(
                    'BatchNorm2DLayer does not support \'pyotrch\' moudle for SparseImageStar2DCSR set'
                )
            
            elif self.module == 'default':
                new_c = self.batchnorm2d(In.c.reshape(In.shape), bias=True).reshape(-1)
                new_V = self.fbatchnorm2d_csr(In.V, In.shape, bias=False)
                
            return SparseImageStar2DCSR(new_c, new_V, In.C, In.d, In.pred_lb, In.pred_ub, In.shape)
        
        else:
            raise Exception('error: Conv2DLayer support ImageStar only')
        

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
                    S.append(self.reachExactSingleInput(inputSet[i]))
            elif isinstance(pool, multiprocessing.pool.Pool):
                S = S + pool.map(self.reachExactSingleInput, inputSet)
            else:
                raise Exception('error: unknown/unsupport pool type')
                
            return S
        
        else:
            return self.reachExactSingleInput(inputSet)