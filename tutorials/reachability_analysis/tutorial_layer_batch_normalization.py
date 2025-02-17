import torch
import numpy as np
import scipy.sparse as sp
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.layer.BatchNorm2DLayer import BatchNorm2DLayer

def batchnorm2d_construct_with_parameters():
    """
    Construct a convolution layer with specified weight and bias
    """
    print('==========================================================================================')
    print('============ EXAMPLE: Batch Normalization Layer Construction with Parameters =============')
    
    in_channels = 3
    gamma = np.array([0.2, 0.7, 0.6])
    beta = np.array([0.01, 0.2, 0.4])
    mean = np.zeros(in_channels)
    var = np.ones(in_channels)
    layer = [gamma, beta, mean, var]
    L_bn = BatchNorm2DLayer(layer, eps=2e-05)
    print(L_bn)
    
    print('============ DONE: Batch Normalization Layer Construction with Parameters ================')
    print('==========================================================================================\n\n')

def batchnorm2d_construct_with_torch_layer():
    """
    Construct a batch normalization layer using a torch layer
    """
    print('==========================================================================================')
    print('=========== EXAMPLE: Batch Normalization Layer Construction using Pytorch Layer ==========')
    
    in_channels = 3
    layer = torch.nn.BatchNorm2d(in_channels)
    L_bn = BatchNorm2DLayer(layer)
    print(L_bn)
    
    print('=========== DONE: Batch Normalization Layer Construction using Pytorch Layer =============')
    print('==========================================================================================\n\n')

def batchnorm2d_reachability_imagestar():
    """
    Conduct reachability analysis on batch normalization layer using ImageStar set
    """
    print('==========================================================================================')
    print('======== EXAMPLE: Reachability Analysis on Batch Normalization Layer using ImageStar =====')
    h, w = 2, 2
    in_channels = 2
    shape = (h, w, in_channels)
    
    gamma = np.array([0.2, 0.7])
    beta = np.array([0.01, 0.4])
    mean = np.zeros(in_channels)
    var = np.ones(in_channels)
    layer = [gamma, beta, mean, var]
    L_bn = BatchNorm2DLayer(layer, eps=2e-05)
    print(L_bn)

    lb = -np.arange(np.prod(shape))/10
    ub = np.arange(np.prod(shape))[::-1]/5
    
    lb = lb.reshape(h, w, in_channels)
    ub = ub.reshape(h, w, in_channels)
    IM = ImageStar(lb, ub)
    print('Input ImageStar set:\n', IM)
    
    R = L_bn.reach(IM)
    print('Output ImageStar set:\n', R)
    
    print('========= DONE: Reachability Analysis on Batch Normalization Layer using ImageStar =======')
    print('==========================================================================================\n\n')

def batchnorm2d_reachability_sparseimagestar():
    """
    Conduct reachability analysis on batch normalization layer using ImageStar set
    """
    print('==========================================================================================')
    print('==== EXAMPLE: Reachability Analysis on Batch Normalization Layer using SparseImageStar ===')
    
    h, w = 2, 2
    in_channels = 2
    shape = (h, w, in_channels)
    
    gamma = np.array([0.2, 0.7])
    beta = np.array([0.01, 0.4])
    mean = np.zeros(in_channels)
    var = np.ones(in_channels)
    layer = [gamma, beta, mean, var]
    L_bn = BatchNorm2DLayer(layer, eps=2e-05)
    print(L_bn)
    
    lb = -np.arange(np.prod(shape))/10
    ub = np.arange(np.prod(shape))[::-1]/5
    
    lb = lb.reshape(h, w, in_channels)
    ub = ub.reshape(h, w, in_channels)
    
    SIM_coo = SparseImageStar2DCOO(lb, ub)
    print('Input SparseImageStar COO set:\n', SIM_coo)
    
    R_coo = L_bn.reach(SIM_coo)
    print('Output SparseImageStar COO set:\n', R_coo)
    
    SIM_csr = SparseImageStar2DCSR(lb, ub)
    print('Input SparseImageStar CSR set:\n', SIM_csr)
    
    R_csr = L_bn.reach(SIM_csr)
    print('Output SparseImageStar CSR set:\n', R_csr)
    
    print('====== DONE: Reachability Analysis on Batch Normalization Layer using SparseImageStar ====')
    print('==========================================================================================\n\n')

if __name__ == "__main__":
    batchnorm2d_construct_with_parameters()
    batchnorm2d_construct_with_torch_layer()
    batchnorm2d_reachability_imagestar()
    batchnorm2d_reachability_sparseimagestar()