#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
##                                                                     ##
##   Primary contacts: Hoang Dung Tran <dungtran@ufl.edu> (UF)         ##
##                     Sung Woo Choi <sungwoo.choi@ufl.edu> (UF)       ##
##                     Yuntao Li <yli17@ufl.edu> (UF)                  ##
##                     Qing Liu <qliu1@ufl.edu> (UF)                   ##
##                                                                     ##
##   See CONTRIBUTORS for full author contacts and affiliations.       ##
##   This program is licensed under the BSD 3‑Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
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