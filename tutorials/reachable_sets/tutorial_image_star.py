"""
Tutorial: Image Star (ImageStar)

ImageStar set = <c, V, P> is defied as
x = c + v[:,:,:,0]*a[0] + ... v[:,:,:,m-1]*a[m-1]
  = V * b,
where V = [c, v[:,:,:,0], ..., v[:,:,:,m-1]],
      b = [1, a[0], ..., a[m-1]]^T,
P(a) \triangleq C a <= d \wedge pred_lb <= a <= pred_ub.
"""

import numpy as np
from StarV.set.imagestar import ImageStar

def imagestar_construct_via_bounds():
    print('======================================================================')
    print('========= EXAMPLE: construct image star set via state bounds =========')
    # state image shape
    shape = (2, 2, 3)
    dim = np.prod(shape)
    lb = -np.random.rand(dim).reshape(shape)
    ub = np.random.rand(dim).reshape(shape)
    IM = ImageStar(lb, ub)
    print(IM)
    repr(IM)
    print('=========== DONE: construct image star set via state bounds ==========')
    print('======================================================================\n\n')
    
def imagestar_construct():
    print('======================================================================')
    print('================== EXAMPLE: construct image star set =================')
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
    print('==================== DONE: construct image star set ==================')
    print('======================================================================\n\n')

def imagestar_affineMap():
    print('======================================================================')
    print('============== EXAMPLE: affine mapping of image star set =============')
    # Create random Image Star
    shape = (2, 2, 1)
    data = np.random.rand(shape[0], shape[1], shape[2]).reshape(shape)
    eps = 0.1
    lb = np.clip(data - eps, 0, 1)
    ub = np.clip(data + eps, 0, 1)
    IM = ImageStar(lb, ub)
    print('original ImageStar: \n')
    print(IM)
    print()
    
    # Apply affine mapping operation
    W = 2*np.random.rand(shape[0], shape[1], shape[2]).reshape(shape) - 1
    b = 2*np.random.rand(shape[0], shape[1], shape[2]).reshape(shape) - 1
    R = IM.affineMap(W, b)
    
    print(f'affine mapping matrix: \n{W}')
    print(f'affine mappint bias: \n{b}\n')
    
    print('affine mapped ImageStar:')
    print(R)

    print('=============== DONE:  affine mapping of image star set ==============')
    print('======================================================================\n\n')
    
def imagestar_getRanges():
    print('======================================================================')
    print('=========== EXAMPLE: getting state ranges of image star set ==========')
    
    H, W, C = 2, 2, 3
    IM = ImageStar.rand_bounds(H, W, C)
    print('random ImageStar: \n')
    print(IM)
    
    l, u = IM.getRanges()
    print('getRanges with LP solver')
    print('lower bounds:\n', l.reshape(H, W, C))
    print('upper bounds:\n', u.reshape(H, W, C))
    print()
    
    l, u = IM.getRanges('estimate')
    print('getRanges with estimation')
    print('lower bounds:\n', l.reshape(H, W, C))
    print('upper bounds:\n', u.reshape(H, W, C))
    print()
    
    print('============ DONE:  getting state ranges of image star set ===========')
    print('======================================================================\n\n')
    
if __name__ == "__main__":
    imagestar_construct_via_bounds()
    imagestar_construct()
    imagestar_affineMap()
    imagestar_getRanges()