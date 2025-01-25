# Tutorial: Sparse Image Star (SIM)
# Formats:
#     - coordiante format (SIM_coo)
#     - compressed row format (SIM_csr)

# SIM set = <c, V, P> is defied as
# x = c + v[:,0]a[1] + v[:,1]a[2] + ... + v[:,m]a[m]
#   = c + Va,
# where V = [v[0], v[1], ..., v[m]],
#       a = [a[0], a[1], ..., a[m]].
# P(a) \triangleq C a <= d \wedge pred_lb <= a <= pred_ub.


import numpy as np
import scipy.sparse as sp
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR

def sim_coo_construct_via_bounds():
    print('======================================================================')
    print('========== EXAMPLE: construct SIM_coo set via state bounds ===========')
    c = np.array([[2, 3], [8, 0]])
    a = np.zeros([2, 2])
    a[0, 0] = 2
    a[1, 0] = 2
    a[0, 1] = 2
    lb = c - a
    ub = c + a
    SIM_coo = SparseImageStar2DCOO(lb, ub)
    print(SIM_coo)
    repr(SIM_coo)
    print('=============== DONE: construct SIM_coo via state bounds =============')
    print('======================================================================\n\n')
    
    
def sim_csr_construct_via_bounds():
    print('======================================================================')
    print('========== EXAMPLE: construct SIM_csr set via state bounds ===========')
    c = np.array([[2, 3], [8, 0]])
    a = np.zeros([2, 2])
    a[0, 0] = 2
    a[1, 0] = 2
    a[0, 1] = 2
    lb = c - a
    ub = c + a
    SIM_csr = SparseImageStar2DCSR(lb, ub)
    print(SIM_csr)
    repr(SIM_csr)
    print('=============== DONE: construct SIM_csr via state bounds =============')
    print('======================================================================\n\n')
    
    
def sim_coo_construct():
    print('======================================================================')
    print('=================== EXAMPLE: construct SIM_coo set ===================')
    """
    Construct a 2x2x1 image with bounded disturbance in [-2, 2] on 
    the pixel positions (0, 0), (1, 0), and (0, 1)
    """
    shape = (2, 2, 1)
    # numpred of predicates = number of attacked pixels
    num_pred = 3
    
    # center image
    c = np.array([2, 3, 8, 0])
    # generator image
    V = np.zeros(shape + (3,))
    V[0, 0, 0, 0] = 1
    V[1, 0, 0, 1] = 1
    V[0, 1, 0, 2] = 1
    V_coo = sp.coo_array(V.reshape(-1, 3))
    
    # predicate constraints
    E = np.eye(num_pred)
    C = np.vstack([E, -E])
    C = sp.csr_array(C)
    d = 2*np.ones(2*num_pred)
    
    # predicate bounds
    pred_lb = -2*np.ones(num_pred)
    pred_ub = 2*np.ones(num_pred)
    SIM_coo = SparseImageStar2DCOO(c, V_coo, C, d, pred_lb, pred_ub, shape)
    print(SIM_coo)
    repr(SIM_coo)
    print('===================== DONE: construct SIM_coo set ====================')
    print('======================================================================\n\n')
    
def sim_csr_construct():
    print('======================================================================')
    print('=================== EXAMPLE: construct SIM_csr set ===================')
    """
    Construct a 2x2x1 image with bounded disturbance in [-2, 2] on 
    the pixel positions (0, 0), (1, 0), and (0, 1)
    """
    shape = (2, 2, 1)
    # numpred of predicates = number of attacked pixels
    num_pred = 3
    
    # center image
    c = np.array([2, 3, 8, 0])
    # generator image
    V = np.zeros(shape + (3,))
    V[0, 0, 0, 0] = 1
    V[1, 0, 0, 1] = 1
    V[0, 1, 0, 2] = 1
    V_csr = sp.csr_array(V.reshape(-1, 3))
    
    # predicate constraints
    E = np.eye(num_pred)
    C = np.vstack([E, -E])
    C = sp.csr_array(C)
    d = 2*np.ones(2*num_pred)
    
    # predicate bounds
    pred_lb = -2*np.ones(num_pred)
    pred_ub = 2*np.ones(num_pred)
    SIM_csr = SparseImageStar2DCSR(c, V_csr, C, d, pred_lb, pred_ub, shape)
    print(SIM_csr)
    repr(SIM_csr)
    print('===================== DONE: construct SIM_csr set ====================')
    print('======================================================================\n\n')
    
    
def sim_coo_affineMap():
    print('======================================================================')
    print('=============== EXAMPLE: affine mapping of SIM_coo set ===============')
    # Create random Image Star
    shape = (2, 2, 1)
    dim = np.prod(shape)
    data = np.arange(dim).reshape(shape)
    eps = 0.1
    lb = np.clip(data - eps, 0, 1)
    ub = np.clip(data + eps, 0, 1)
    SIM_coo = SparseImageStar2DCOO(lb, ub)
    print('original SIM_coo: \n')
    print(SIM_coo)
    print()
    
    # Apply affine mapping operation
    W = 2*np.random.rand(shape[0], shape[1], shape[2]).reshape(shape) - 1
    b = 2*np.random.rand(shape[0], shape[1], shape[2]).reshape(shape) - 1
    R = SIM_coo.affineMap(W, b)
    
    print(f'affine mapping matrix: \n{W}')
    print(f'affine mappint bias: \n{b}\n')
    
    print('affine mapped SIM_coo:')
    print(R)

    print('================ DONE:  affine mapping of SIM_csr set ================')
    print('======================================================================\n\n')

def sim_csr_affineMap():
    print('======================================================================')
    print('=============== EXAMPLE: affine mapping of SIM_csr set ===============')
    # Create random Image Star
    shape = (2, 2, 1)
    data = np.random.rand(shape[0], shape[1], shape[2]).reshape(shape)
    eps = 0.1
    lb = np.clip(data - eps, 0, 1)
    ub = np.clip(data + eps, 0, 1)
    SIM_csr = SparseImageStar2DCSR(lb, ub)
    print('original SIM_csr: \n')
    print(SIM_csr)
    print()
    
    # Apply affine mapping operation
    W = 2*np.random.rand(shape[0], shape[1], shape[2]).reshape(shape) - 1
    b = 2*np.random.rand(shape[0], shape[1], shape[2]).reshape(shape) - 1
    R = SIM_csr.affineMap(W, b)
    
    print(f'affine mapping matrix: \n{W}')
    print(f'affine mappint bias: \n{b}\n')
    
    print('affine mapped SIM_csr:')
    print(R)

    print('================= DONE:  affine mapping of SIM_csr set ===============')
    print('======================================================================\n\n')
    

def sim_coo_getRanges():
    print('======================================================================')
    print('============= EXAMPLE: getting state ranges of SIM_coo set ===========')
    
    c = np.array([[2, 3], [8, 0]])
    a = np.zeros([2, 2])
    a[0, 0] = 2
    a[1, 0] = 2
    a[0, 1] = 2
    lb = c - a
    ub = c + a
    
    SIM_coo = SparseImageStar2DCOO(lb, ub)
    H, W, C = SIM_coo.shape
    print('Actual state bounds of SIM_coo:')
    print('lower bounds:\n', lb.reshape(H, W))
    print('upper bounds:\n', ub.reshape(H, W))
    print()
    
    l, u = SIM_coo.getRanges()
    print('State bounds computed with getRanges() via LP solver:')
    print('lower bounds:\n', l.reshape(H, W))
    print('upper bounds:\n', u.reshape(H, W))
    print('============= DONE:  getting state ranges of SIM_coo set =============')
    print('======================================================================\n\n')

def sim_csr_getRanges():
    print('======================================================================')
    print('============= EXAMPLE: getting state ranges of SIM_csr set ===========')
    c = np.array([[2, 3], [8, 0]])
    a = np.zeros([2, 2])
    a[0, 0] = 2
    a[1, 0] = 2
    a[0, 1] = 2
    lb = c - a
    ub = c + a
    
    SIM_csr = SparseImageStar2DCSR(lb, ub)
    H, W, C = SIM_csr.shape
    print('Actual state bounds of SIM_csr:')
    print('lower bounds:\n', lb.reshape(H, W))
    print('upper bounds:\n', ub.reshape(H, W))
    print()
    
    l, u = SIM_csr.getRanges()
    print('State bounds computed with getRanges() via LP solver:')
    print('lower bounds:\n', l.reshape(H, W))
    print('upper bounds:\n', u.reshape(H, W))
    print('============= DONE:  getting state ranges of SIM_csr set =============')
    print('======================================================================\n\n')

def sim_coo_estimateRanges():
    print('======================================================================')
    print('============= EXAMPLE: getting state ranges of SIM_coo set ===========')
    c = np.array([[2, 3], [8, 0]])
    a = np.zeros([2, 2])
    a[0, 0] = 2
    a[1, 0] = 2
    a[0, 1] = 2
    lb = c - a
    ub = c + a
    
    SIM_coo = SparseImageStar2DCOO(lb, ub)
    H, W, C = SIM_coo.shape
    print('Actual state bounds of SIM_coo:')
    print('lower bounds:\n', lb.reshape(H, W))
    print('upper bounds:\n', ub.reshape(H, W))
    print()
    
    l, u = SIM_coo.estimateRanges()
    print('State bounds computed with estimateRanges():')
    print('lower bounds:\n', l.reshape(H, W))
    print('upper bounds:\n', u.reshape(H, W))
    print()
    
    l, u = SIM_coo.getRanges('estimate')
    print('State bounds computed with getRanges(\'estimate\')')
    print('lower bounds:\n', l.reshape(H, W))
    print('upper bounds:\n', u.reshape(H, W))
    print()
    print('============= DONE:  getting state ranges of SIM_coo set =============')
    print('======================================================================\n\n')


def sim_csr_estimateRanges():
    print('======================================================================')
    print('============= EXAMPLE: getting state ranges of SIM_csr set ===========')
    c = np.array([[2, 3], [8, 0]])
    a = np.zeros([2, 2])
    a[0, 0] = 2
    a[1, 0] = 2
    a[0, 1] = 2
    lb = c - a
    ub = c + a
    
    SIM_csr = SparseImageStar2DCSR(lb, ub)
    H, W, C = SIM_csr.shape
    print('Actual state bounds of SIM_csr:')
    print('lower bounds:\n', lb.reshape(H, W))
    print('upper bounds:\n', ub.reshape(H, W))
    print()
    
    l, u = SIM_csr.estimateRanges()
    print('State bounds computed with estimateRanges():')
    print('lower bounds:\n', l.reshape(H, W))
    print('upper bounds:\n', u.reshape(H, W))
    print()
    
    l, u = SIM_csr.getRanges('estimate')
    print('State bounds computed with getRanges(\'estimate\')')
    print('lower bounds:\n', l.reshape(H, W))
    print('upper bounds:\n', u.reshape(H, W))
    print()
    print('============= DONE:  getting state ranges of SIM_csr set =============')
    print('======================================================================\n\n')

if __name__ == "__main__":
    sim_coo_construct_via_bounds()
    sim_csr_construct_via_bounds()
    sim_coo_construct()
    sim_csr_construct()
    sim_coo_affineMap()
    sim_csr_affineMap()
    sim_coo_getRanges()
    sim_csr_getRanges()
    sim_coo_estimateRanges()
    sim_csr_estimateRanges()