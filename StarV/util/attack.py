"""
	Apply adversarial attacks for verifying networks
	
	Sung Woo Choi, 10/20/2024
"""
import numpy as np

def brightening_attack(image, delta=0.05, d=240, dtype=np.float32):
    # d is for threshold for brightnening attack
    shape = image.shape
    n = np.prod(shape)
    
    flatten_img = image.reshape(-1)

    lb = flatten_img.copy().astype(dtype)
    ub = flatten_img.copy().astype(dtype)
    for i in range(n):
        if lb[i] >= d:
            lb[i] = 0
            ub[i] *= delta
            
    lb = lb.reshape(shape)
    ub = ub.reshape(shape)
    return lb, ub