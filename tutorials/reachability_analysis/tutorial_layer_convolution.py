import torch
import numpy as np
import scipy.sparse as sp
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.layer.Conv2DLayer import Conv2DLayer

def conv2d_construct_with_weight_and_bias():
    """
    Construct a convolution layer with specified weight and bias
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Convolution Layer Construction with Weight and Bias =============')
    W = np.array([[1, 1], [-1, 0]])
    b = np.array([-1])
    
    layer = [W, b]
    stride = [2, 2]
    padding = [0, 0, 0, 0]
    dilation = [1, 1]
    L_conv = Conv2DLayer(layer, stride, padding, dilation)
    print(L_conv)
    
    print('=============== DONE: Convolution Layer Construction with Weight and Bias ================')
    print('==========================================================================================\n\n')

def conv2d_construct_with_torch_layer():
    """
    Construct a convolution layer using a torch layer
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Convolution Layer Construction using Pytorch Layer ==============')
    
    layer = torch.nn.Conv2d(2, 3, (3, 3), stride=(2, 2), padding=(2, 2))
    L_conv = Conv2DLayer(layer)
    print(L_conv)
    
    print('=============== DONE: Convolution Layer Construction using Pytorch Layer =================')
    print('==========================================================================================\n\n')


def conv2d_construct_random_layer():
    """
    Construct a convolution layer with random weight and bias
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Random Convolution Layer Construction ===========================')
    L_conv = Conv2DLayer.rand(3, 3, 4, 5)
    L_conv.info()

    print('=============== DONE: Random Convolution Layer Construction ==============================')
    print('==========================================================================================\n\n')
    
def conv2d_reachability_imagestar():
    """
    Conduct reachability analysis on convolution layer using ImageStar set
    """
    print('==========================================================================================')
    print('============== EXAMPLE: Reachability Analysis on Convolution Layer using ImageStar =======')
    W = np.array([[1, 1], [-1, 0]])
    b = np.array([-1])
    
    layer = [W, b]
    stride = [2, 2]
    padding = [0, 0, 0, 0]
    dilation = [1, 1]
    L_conv = Conv2DLayer(layer, stride, padding, dilation)
    print(L_conv)
    
    # Construct input set
    h, w, ci, co = 4, 4, 1, 1
    c = np.array([[0, 4, 1, 2], [2, 3, 2, 3], [1, 3, 1, 2], [2, 1, 3, 2]]).reshape(h, w, ci, co)
    v = np.zeros([h, w, ci, co])
    v[0, 1, 0, 0] = 1
    
    V = np.concatenate([c, v], axis=3)
    C = np.array([[1], [-1]])
    d = 2*np.ones([2])
    pred_lb = -2*np.ones(1)
    pred_ub = 2*np.ones(1)
    
    IM = ImageStar(V, C, d, pred_lb, pred_ub)
    print('Input ImageStar set:\n', IM)
    
    R = L_conv.reach(IM)
    print('Output ImageStar set:\n', R)
    
    print('=============== DONE: Reachability Analysis on Convolution Layer using ImageStar =========')
    print('==========================================================================================\n\n')
    
def conv2d_reachability_sparseimagestar():
    """
    Conduct reachability analysis on convolution layer using ImageStar set
    """
    print('==========================================================================================')
    print('==========EXAMPLE: Reachability Analysis on Convolution Layer using SparseImageStar ======')
    W = np.array([[1, 1], [-1, 0]])
    b = np.array([-1])
    
    layer = [W, b]
    stride = [2, 2]
    padding = [0, 0, 0, 0]
    dilation = [1, 1]
    L_conv = Conv2DLayer(layer, stride, padding, dilation)
    print(L_conv)
    
    # Construct input set
    h, w, ci, co = 4, 4, 1, 1
    shape = (h, w, ci)
    
    c = np.array([[0, 4, 1, 2], [2, 3, 2, 3], [1, 3, 1, 2], [2, 1, 3, 2]]).ravel()
    V = np.zeros([h, w, ci, co])
    V[0, 1, 0, 0] = 1
    V = V.reshape(-1, 1)
    V_csr = sp.csr_array(V)
    V_coo = sp.coo_array(V)
    
    C = np.array([[1], [-1]])
    C = sp.csr_array(C)
    d = 2*np.ones([2])
    pred_lb = -2*np.ones(1)
    pred_ub = 2*np.ones(1)
    
    SIM_coo = SparseImageStar2DCOO(c, V_coo, C, d, pred_lb, pred_ub, shape)
    print('Input SparseImageStar COO set:\n', SIM_coo)
    
    R_coo = L_conv.reach(SIM_coo)
    print('Output SparseImageStar COO set:\n', R_coo)
    
    SIM_csr = SparseImageStar2DCSR(c, V_csr, C, d, pred_lb, pred_ub, shape)
    print('Input SparseImageStar CSR set:\n', SIM_csr)
    
    R_csr = L_conv.reach(SIM_csr)
    print('Output SparseImageStar CSR set:\n', R_csr)
    
    print('============ DONE: Reachability Analysis on Convolution Layer using SparseImageStar ======')
    print('==========================================================================================\n\n')
    

if __name__ == "__main__":
    conv2d_construct_with_weight_and_bias()
    conv2d_construct_with_torch_layer()
    conv2d_construct_random_layer()
    conv2d_reachability_imagestar()
    conv2d_reachability_sparseimagestar()