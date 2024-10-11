"""
Verify vgg16 (VNNCOMP2023) against infinity norm
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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text)]

def verify_vgg16_network(dtype='float64'):

    print('=================================================================================')
    print(f"Verification of VGG16 Network against Infinity Norm Attack")
    print('=================================================================================\n')

    folder_dir = f"./SparseImageStar_evaluation/vnncomp2023/vggnet16"
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())


    shape = (3, 224, 224)
    
    # VNNLIB_FILE = 'vnncomp2023_benchmarks/benchmarks/vggnet16/vnnlib/spec0_pretzel.vnnlib'
    vnnlib_dir = f"{folder_dir}/vnnlib"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)

    N = len(vnnlib_files)
    rbIM = np.zeros(N)
    vtIM = np.zeros(N)
    rbCSR = np.zeros(N)
    vtCSR = np.zeros(N)
    rbCOO = np.zeros(N)
    vtCOO = np.zeros(N)

    rb_table = []
    vt_table = []
 
    print(f"\n\nVerifying vggnet16 with ImageStar")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0])
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0])

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        if num_attack_pixel > 50:
            print(f"Skipping {vnnlib_file} to avoid RAM issue")
            rbIM[i] = np.nan
            vtIM[i] = np.nan
        else:
            IM = ImageStar(lb, ub)
            rbIM[i], vtIM[i], _, _ = certifyRobustness(net=starvNet, inputs=IM, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
            
            if rbIM[i] == 1:
                print(f"ROBUSTNESS RESULT: ROBUST")
            elif rbIM[1] == 2:
                print(f"ROBUSTNESS RESULT: UNKNOWN")
            elif rbIM[i] == 0:
                print(f"ROBUSTNESS RESULT: UNROBUST")

            print(f"VERIFICATION TIME: {vtIM[i]}")

    rb_table.append((rbIM == 1).sum())
    vt_table.append((vtIM.sum() / N))
    del IM

    print(f"\n\nVerifying vggnet16 with SparseImageStar in CSR format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])


        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        if num_attack_pixel > 150:
            print(f"Skipping {vnnlib_file} to avoid RAM issue")
            rbCSR[i] = np.nan
            vtCSR[i] = np.nan
        else:
            CSR = SparseImageStar2DCSR(lb, ub)
            rbCSR[i], vtCSR[i], _, _ = certifyRobustness(net=starvNet, inputs=CSR, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
        
            if rbCSR[i] == 1:
                print(f"ROBUSTNESS RESULT: ROBUST")
            elif rbCSR[1] == 2:
                print(f"ROBUSTNESS RESULT: UNKNOWN")
            elif rbCSR[i] == 0:
                print(f"ROBUSTNESS RESULT: UNROBUST")

            print(f"VERIFICATION TIME: {vtCSR[i]}")

    rb_table.append((rbCSR == 1).sum())
    vt_table.append((vtCSR.sum() / N))
    del CSR

    print(f"\n\nVerifying vggnet16 with SparseImageStar in COO format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])


        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        if num_attack_pixel > 150:
            print(f"Skipping {vnnlib_file} to avoid RAM issue")
            rbCOO[i] = np.nan
            vtCOO[i] = np.nan
        else:
            COO = SparseImageStar2DCOO(lb, ub)
            rbCOO[i], vtCOO[i], _, _ = certifyRobustness(net=starvNet, inputs=COO, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
            
            if rbCOO[i] == 1:
                print(f"ROBUSTNESS RESULT: ROBUST")
            elif rbCOO[1] == 2:
                print(f"ROBUSTNESS RESULT: UNKNOWN")
            elif rbCOO[i] == 0:
                print(f"ROBUSTNESS RESULT: UNROBUST")

            print(f"VERIFICATION TIME: {vtCOO[i]}")

    rb_table.append((rbCOO == 1).sum())
    vt_table.append((vtCOO.sum() / N))
    del COO

    # save verification results
    path = f"./SparseImageStar_evaluation/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_results.pkl"
    pickle.dump([rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, rb_table, vt_table], open(save_file, "wb"))

    headers = [f"ImageStar", f"SIM_CSR", f"SIM_COO"]

    # Robustness Resluts
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate([rb_table], headers=headers))
    print()

    Tlatex = tabulate([rb_table], headers=headers, tablefmt='latex')
    with open(path+f"vggnet16_vnncomp23_results_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate([vt_table], headers=headers))
    print()

    Tlatex = tabulate([vt_table], headers=headers, tablefmt='latex')
    with open(path+f"vggnet16_vnncomp23_results_vt.tex", "w") as f:
        print(Tlatex, file=f)

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def verify_vgg16_converted_network(dtype='float64'):

    print('=================================================================================')
    print(f"Verification of VGG16 Converted Network against Infinity Norm Attack")
    print('=================================================================================\n')

    folder_dir = f"./SparseImageStar_evaluation/vnncomp2023/vggnet16"
    net_dir = f"{folder_dir}/onnx/vgg16-7_converted.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())


    shape = (3, 224, 224)
    
    # VNNLIB_FILE = 'vnncomp2023_benchmarks/benchmarks/vggnet16/vnnlib/spec0_pretzel.vnnlib'
    vnnlib_dir = f"{folder_dir}/vnnlib"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)

    N = len(vnnlib_files)
    rbIM = np.zeros(N)
    vtIM = np.zeros(N)

    print(f"\n\nVerifying vggnet16 with ImageStar")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0])
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0])

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        if num_attack_pixel > 50:
            print(f"Skipping {vnnlib_file} to avoid RAM issue")
            rbIM[i] = np.nan
            vtIM[i] = np.nan
        else:
            IM = ImageStar(lb, ub)
            rbIM[i], vtIM[i], _, _ = certifyRobustness(net=starvNet, inputs=IM, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
            
            if rbIM[i] == 1:
                print(f"ROBUSTNESS RESULT: ROBUST")
            elif rbIM[1] == 2:
                print(f"ROBUSTNESS RESULT: UNKNOWN")
            elif rbIM[i] == 0:
                print(f"ROBUSTNESS RESULT: UNROBUST")

            print(f"VERIFICATION TIME: {vtIM[i]}")
    del IM

    # save verification results
    path = f"./SparseImageStar_evaluation/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_converted_results.pkl"
    pickle.dump([rbIM, vtIM], open(save_file, "wb"))

    headers = [f"ImageStar"]

    # Robustness Resluts
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), rbIM], headers=headers))
    print()

    Tlatex = tabulate([np.arange(N), rbIM], headers=headers, tablefmt='latex')
    with open(path+f"vggnet16_vnncomp23_converted_results_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), vtIM], headers=headers))
    print()

    Tlatex = tabulate([np.arange(N), vtIM], headers=headers, tablefmt='latex')
    with open(path+f"vggnet16_vnncomp23_converted_results_vt.tex", "w") as f:
        print(Tlatex, file=f)

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def plot_table_vgg16_network():
    folder_dir = 'SparseImageStar_evaluation/results/'
    file_dir = folder_dir + 'vggnet16_vnncomp23_results.pkl'
    with open(file_dir, 'rb') as f:
        rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, _, _ = pickle.load(f)
    file_dir = folder_dir + 'vggnet16_vnncomp23_converted_results.pkl'
    with open(file_dir, 'rb') as f:
        rbIMc, vtIMc = pickle.load(f)


    folder_dir = f"./SparseImageStar_evaluation/vnncomp2023/vggnet16"
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)
    vnnlib_dir = f"{folder_dir}/vnnlib"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)
    shape = (3, 224, 224)

    num_attack_pixel = []
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype='float32')
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0])
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0])
        num_attack_pixel.append(int((lb != ub).sum())) 


    N = 15
    vt_NNENUM = [3.5, 3.4, 9.3, 4.8, 18.1, 35.7, 6.5, 18.3, 133.85, 10.6, 40.9, 57.6, 'T/O', 236.52, 746.60]

    headers = ['Specs', '$e$', 'Result', 'IM', 'SIM_csr', 'SIM_coo', 'IM', 'NNENUM']
    result = 'UNSAT'
    
    data = []
    for i in range(N):
        vt_im = 'O/M' if np.isnan(vtIM[i]) else f"{vtIM[i]:0.1f}"
        vt_imc = 'O/M' if np.isnan(vtIMc[i]) else f"{vtIMc[i]:0.1f}"
        data.append([i, result, num_attack_pixel[i], vt_im, f"{vtCSR[i]:0.1f}", f"{vtCOO[i]:0.1f}", vt_imc, vt_NNENUM[i]])
    print(tabulate(data, headers=headers))

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(folder_dir+f"vggnet16_vnncomp23_results_full_table.tex", "w") as f:
        print(Tlatex, file=f)


if __name__ == "__main__":
    # verify_vgg16_network(dtype='float64')
    # verify_vgg16_converted_network(dtype='float64')
    plot_table_vgg16_network()