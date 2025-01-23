
# Tutorial: Image Star (ImageStar)

# ImageStar set = <c, V, P> is defied as
# x = c + v[:,:,:,0]*a[0] + ... v[:,:,:,m-1]*a[m-1]
#   = V * b,
# where V = [c, v[:,:,:,0], ..., v[:,:,:,m-1]],
#       b = [1, a[0], ..., a[m-1]]^T,
# P(a) \triangleq C a <= d \wedge pred_lb <= a <= pred_ub.

import numpy as np
from StarV.set.imagestar import ImageStar

def imagestar_construct_using_state_bounds():
    print('======================================================================')
    print('======== EXAMPLE: construct image star set using state bounds ========')
    """
    Construct a 2x2x1 image with bounded disturbance in [-2, 2] on 
    the pixel positions (0, 0), (1, 0), and (0, 1)
    """
    c = np.array([[2, 3], [8, 0]])
    a = np.zeros([2, 2])
    a[0, 0] = 2
    a[1, 0] = 2
    a[0, 1] = 2
    lb = c - a
    ub = c + a
    IM = ImageStar(lb, ub)
    print(IM)
    repr(IM)  
    print('========== DONE: construct image star set using state bounds =========')
    print('======================================================================\n\n')
    
def imagestar_construct():
    print('======================================================================')
    print('================== EXAMPLE: construct image star set =================')
    """
    Construct a 4x4x1 image with bounded disturbance in [-2, 2] on 
    the pixel positions (1, 2, 1)
    """
    c = np.array([[0, 4, 1, 2], [2, 3, 2, 3], [1, 3, 1, 2], [2, 1, 3, 2]])[:, :, None, None] # shape in [4, 4, 1, 1]
    V = np.zeros([4, 4, 1, 1])
    V[0, 1, 0, 0] = 1
    V = np.concatenate([c, V], axis=3)

    # predicate constraints
    C = np.zeros([2, 1])
    d = np.zeros(2)
    # 1 a <= 2
    C[0, 0] = 1
    d[0] = 2
    # -1 a <= 2
    C[1, 0] = -1
    d[1] = 2

    # predicate bounds
    # -2 <= a <= 2
    pred_lb = -2*np.ones(1)
    pred_ub = 2*np.ones(1)

    IM = ImageStar(V, C, d, pred_lb, pred_ub)
    print(IM)
    repr(IM)    
    print('==================== DONE: construct image star set ==================')
    print('======================================================================\n\n')

def imagestar_affineMap():
    print('======================================================================')
    print('============== EXAMPLE: affine mapping of image star set =============')
    # Create random Image Star
    shape = (2, 2, 1)
    dim = np.prod(shape)
    data = np.arange(dim).reshape(shape) / dim
    eps = 0.1
    lb = np.clip(data - eps, 0, 1)
    ub = np.clip(data + eps, 0, 1)
    IM = ImageStar(lb, ub)
    print('original ImageStar: \n')
    print(IM)
    print()
    
    # Apply affine mapping operation
    W = np.array([[[-0.3463], [-0.5628]], [[ 0.5825], [ 0.6715]]])
    b = np.array([[[ 0.3383], [-0.2682]], [[-0.5748], [-0.3584]]])
    
    # Affine mapped ImageStar
    R = IM.affineMap(W, b)
    
    print(f'affine mapping matrix: \n{W}')
    print(f'affine mapping bias: \n{b}\n')
    
    print('affine mapped ImageStar:')
    print(R)

    print('=============== DONE:  affine mapping of image star set ==============')
    print('======================================================================\n\n')
    
def imagestar_getRanges():
    print('======================================================================')
    print('=========== EXAMPLE: getting state ranges of image star set ==========')
    
    c = np.array([[2, 3], [8, 0]])
    a = np.zeros([2, 2])
    a[0, 0] = 2
    a[1, 0] = 2
    a[0, 1] = 2
    lb = c - a
    ub = c + a
    
    IM = ImageStar(lb, ub)
    H, W, C = IM.shape
    print('Actual state bounds of ImageStar:')
    print('lower bounds:\n', lb.reshape(H, W))
    print('upper bounds:\n', ub.reshape(H, W))
    print()
    
    l, u = IM.getRanges()
    print('State bounds computed with getRanges() via LP solver:')
    print('lower bounds:\n', l.reshape(H, W))
    print('upper bounds:\n', u.reshape(H, W))
    print()
    
    print('============ DONE:  getting state ranges of image star set ===========')
    print('======================================================================\n\n')
    

def imagestar_estimateRanges():
    print('======================================================================')
    print('=========== EXAMPLE: getting state ranges of image star set ==========')
    
    c = np.array([[2, 3], [8, 0]])
    a = np.zeros([2, 2])
    a[0, 0] = 2
    a[1, 0] = 2
    a[0, 1] = 2
    lb = c - a
    ub = c + a
    
    IM = ImageStar(lb, ub)
    H, W, C = IM.shape
    print('Actual state bounds of ImageStar:')
    print('lower bounds:\n', lb.reshape(H, W))
    print('upper bounds:\n', ub.reshape(H, W))
    print()
    
    l, u = IM.estimateRanges()
    print('State bounds computed with estimateRanges():')
    print('lower bounds:\n', l.reshape(H, W))
    print('upper bounds:\n', u.reshape(H, W))
    print()
    
    l, u = IM.getRanges('estimate')
    print('State bounds computed with getRanges(''estimate'')')
    print('lower bounds:\n', l.reshape(H, W))
    print('upper bounds:\n', u.reshape(H, W))
    print()
    
    print('============ DONE:  getting state ranges of image star set ===========')
    print('======================================================================\n\n')
    
if __name__ == "__main__":
    imagestar_construct_using_state_bounds()
    imagestar_construct()
    imagestar_affineMap()
    imagestar_getRanges()
    imagestar_estimateRanges()