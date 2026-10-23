import torch
import numpy as np
import scipy.sparse as sp
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.layer.MaxPool2DLayer import MaxPool2DLayer

def maxpool2d_construct():
    """
    Construct a max pooling layer
    """
    print('==========================================================================================')
    print('======================= EXAMPLE: Max Pooling Layer Construction ==========================')
    kernel_size = [2, 2]
    stride = [2, 2]
    padding = [0, 0, 0, 0]
    L_max = MaxPool2DLayer(kernel_size, stride, padding)
    print(L_max)
    
    print('======================== DONE: Max Pooling Layer Construction  ===========================')
    print('==========================================================================================\n\n')
    
    
def maxpool2d_reachability_imagestar():
    """ 
    Conduct reachability analysis on max pooling layer using ImageStar set
    """
    print('==========================================================================================')
    print('========= EXAMPLE: Reachability Analysis on Max Pooling Layer using ImageStar ============')

    kernel_size = [2, 2]
    stride = [2, 2]
    padding = [0, 0, 0, 0]
    L_max = MaxPool2DLayer(kernel_size, stride, padding)

    # Construct input set
    h, w, ci, m = 4, 4, 1, 1
    c = np.array([[2, 3, 4, 1], [8, 0, 1, 3], [2, 5, 7, 0], [7, 9, 6, 1]]).reshape(h, w, ci, m)
    v = np.zeros([h, w, ci, m])
    v[2, 2, 0, 0] = 1

    V = np.concatenate([c, v], axis=3)
    C = np.array([[1], [-1]])
    d = 2*np.ones([2])
    pred_lb = -2*np.ones(1)
    pred_ub = 2*np.ones(1)

    IM = ImageStar(V, C, d, pred_lb, pred_ub)
    print('Input ImageStar set:\n', IM)

    R = L_max.reach(IM, 'approx')
    print('Output ImageStar set:\n', R)

    print('=========== DONE: Reachability Analysis on Max Pooling Layer using ImageStar =============')
    print('==========================================================================================\n\n')


def maxpool2d_reachability_sparseimagestar():
    """ 
    Conduct reachability analysis on max pooling layer using ImageStar set
    """
    print('==========================================================================================')
    print('========= EXAMPLE: Reachability Analysis on Max Pooling Layer using ImageStar ============')
    
    kernel_size = [2, 2]
    stride = [2, 2]
    padding = [0, 0, 0, 0]
    L_max = MaxPool2DLayer(kernel_size, stride, padding)
    
    # Construct input set
    h, w, ci, m = 4, 4, 1, 1
    shape = (h, w, ci)
    
    c = np.array([[2, 3, 4, 1], [8, 0, 1, 3], [2, 5, 7, 0], [7, 9, 5, 1]]).ravel()
    V = np.zeros([h, w, ci, m])
    V[2, 2, 0, 0] = 1
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
    
    R_coo = L_max.reach(SIM_coo, 'approx')
    print('Output SparseImageStar COO set:\n', R_coo)
    
    SIM_csr = SparseImageStar2DCSR(c, V_csr, C, d, pred_lb, pred_ub, shape)
    print('Input SparseImageStar CSR set:\n', SIM_csr)
    
    R_csr = L_max.reach(SIM_csr, 'approx')
    print('Output SparseImageStar CSR set:\n', R_csr)
    
    lb, ub = R_csr.getRanges()
    print(f'Ranges: \nlb:\n{lb.reshape(2, 2)}\nub:\n{ub.reshape(2, 2)}')
    
    print('=========== DONE: Reachability Analysis on Max Pooling Layer using ImageStar =============')
    print('==========================================================================================\n\n')

if __name__ == "__main__":
    maxpool2d_construct()
    maxpool2d_reachability_imagestar()
    maxpool2d_reachability_sparseimagestar()