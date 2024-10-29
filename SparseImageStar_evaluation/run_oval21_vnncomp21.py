"""
Verify oval21 (VNNCOMP2021) against infinity norm
Author: Sung Woo Choi
Date: 06/24/2024
"""

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle
import re

import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.verifier.certifier import certifyRobustness
from StarV.util.load import *
from StarV.util.vnnlib import *


def verify_oval21_network(net_type='base', dtype='float64'):

    print('=================================================================================')
    print(f"Verification of oval21 Network against Infinity Norm Attack")
    print('=================================================================================\n')

    assert net_type in ['base', 'deep', 'wide'], \
    f"network type for oval21 should be /'base/', /'deep'/, or /'wide'/ but received {net_type}"

    folder_dir = f"./SparseImageStar_evaluation/vnncomp2021/oval21/"
    net_dir = f"{folder_dir}/onnx/cifar_{net_type}_kw.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())


    shape = (3, 32, 32)

    vnnlib_dir = f"{folder_dir}/vnnlib"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.startswith(f"cifar_{net_type}_kw") and f.endswith('.vnnlib')]

    N = len(vnnlib_files)
    rbIM = np.zeros(N)
    vtIM = np.zeros(N)
    rbCSR = np.zeros(N)
    vtCSR = np.zeros(N)
    rbCOO = np.zeros(N)
    vtCOO = np.zeros(N)

    rb_table = []
    vt_table = []
    
    print(f"Verifying cifar_{net_type}_kw with ImageStar")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0]) #check label is correct

        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        # bounds = np.array(box, dtype=inp_dtype)
        bounds = np.array(box, dtype=dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0])
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0])

        IM = ImageStar(lb, ub)
        rbIM[i], vtIM[i], _, _ = certifyRobustness(net=starvNet, inputs=IM, labels=label,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=False)
    rb_table.append((rbIM == 1).sum())
    vt_table.append((vtIM.sum() / N))
    del IM

    print(f"\nVerifying cifar_{net_type}_kw with SparseImageStar in CSR format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])


        vnnlib_rv = read_vnnlib_simple(vnnlib_file, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        # bounds = np.array(box, dtype=inp_dtype)
        bounds = np.array(box, dtype=dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0])
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0])

        CSR = SparseImageStar2DCSR(lb, ub)
        rbCSR[i], vtCSR[i], _, _ = certifyRobustness(net=starvNet, inputs=CSR, labels=label,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=False)
    rb_table.append((rbCSR == 1).sum())
    vt_table.append((vtCSR.sum() / N))
    del CSR

    print(f"\nVerifying cifar_{net_type}_kw with SparseImageStar in COO format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])


        vnnlib_rv = read_vnnlib_simple(vnnlib_file, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        # bounds = np.array(box, dtype=inp_dtype)
        bounds = np.array(box, dtype=dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0])
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0])
    
        COO = SparseImageStar2DCOO(lb, ub)
        rbCOO[i], vtCOO[i], _, _ = certifyRobustness(net=starvNet, inputs=COO, labels=label,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=False)
    rb_table.append((rbCOO ==1).sum())
    vt_table.append((vtCOO.sum() / N))
    del COO

    # save verification results
    path = f"./SparseImageStar_evaluation/results"
    if not os.path.exists(path):
        os.makedirs(path)

    headers = [f"", f"ImageStar", f"SIM_CSR", f"SIM_COO"]

    # Robustness Resluts
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate(rb_table, headers=headers))
    print()

    Tlatex = tabulate(rb_table, headers=headers, tablefmt='latex')
    with open(path+f"cifar_{net_type}_kw_vnncomp21_results_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate(vt_table, headers=headers))
    print()

    Tlatex = tabulate(vt_table, headers=headers, tablefmt='latex')
    with open(path+f"cifar_{net_type}_kw_vnncomp21_results_vt.tex", "w") as f:
        print(Tlatex, file=f)

    save_file = path + f"/cifar_{net_type}_kw_vnncomp21_results.pkl"
    pickle.dump([rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, rb_table, vt_table], open(save_file, "wb"))

    print('=====================================================')
    print('DONE!')
    print('=====================================================')

if __name__ == "__main__":
    verify_oval21_network(net_type='base', dtype='float64')
    verify_oval21_network(net_type='deep', dtype='float64')
    verify_oval21_network(net_type='wide', dtype='float64')