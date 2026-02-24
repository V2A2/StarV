"""
Script test for high-dimensional sine-on-index reachability.
Author: Zhuoyang Zhou
Date: 02/23/2026
"""
# import os 
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# pkg_root = os.path.abspath(os.path.join(current_dir, ".."))
# if pkg_root not in sys.path:
#     sys.path.insert(0, pkg_root)

import numpy as np
from StarV.set.star import Star
from StarV.operators.Sine import SinLayer
from StarV.util.plot import plot_star


if __name__ == '__main__':
    # Build a 2D input box for [x, theta]
    lb = np.array([-1.0, -0.7])
    ub = np.array([1.0, 0.7])
    I = Star(lb, ub)

    # New high-dimensional mode: append y = sin(theta) with idx=1
    layer = SinLayer()
    R_idx = layer.reach(I, idx=1, method='approx', lp_solver='gurobi', RF=0.0)

    print('=== High-dimensional sin(idx) ===')
    print('dim   =', R_idx.dim)
    print('nVars =', R_idx.nVars)
    l_idx, u_idx = R_idx.getRanges(lp_solver='gurobi')
    print('range lower =', l_idx)
    print('range upper =', u_idx)

    # Plot projection [theta, sin(theta)] where y is the appended last dimension
    dir_mat = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    plot_star(R_idx, dir_mat=dir_mat, label=(r'$\\theta$', r'$\\sin(\\theta)$'), show=True)
    

    # Optional old behavior check on 1D Star
    # I1 = Star(np.array([-0.7]), np.array([0.7]))
    # R1 = layer.reach(I1, method='approx', lp_solver='gurobi', RF=0.0)
    # l1, u1 = R1.getRanges(lp_solver='gurobi')
    # print('=== Old 1D behavior ===')
    # print('dim   =', R1.dim)
    # print('nVars =', R1.nVars)
    # print('range lower =', l1)
    # print('range upper =', u1)