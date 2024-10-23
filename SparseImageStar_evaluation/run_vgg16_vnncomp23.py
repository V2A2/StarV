"""
Verify vgg16 (VNNCOMP2023) against infinity norm
Author: Sung Woo Choi
Date: 06/24/2024
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import torchvision.transforms as transforms
import torchvision
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
            elif rbIM[i] == 2:
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
            elif rbCSR[i] == 2:
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
            elif rbCOO[i] == 2:
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


def verify_vgg16_network_get_num_pred(dtype='float64'):

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
    rbCSR = np.zeros(N)
    vtCSR = np.zeros(N)
    numPred = np.zeros(N)
 
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
            rbCSR[i], vtCSR[i], _, Y = certifyRobustness(net=starvNet, inputs=CSR, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
            numPred[i] = Y.num_pred
        
            if rbCSR[i] == 1:
                print(f"ROBUSTNESS RESULT: ROBUST")
            elif rbCSR[i] == 2:
                print(f"ROBUSTNESS RESULT: UNKNOWN")
            elif rbCSR[i] == 0:
                print(f"ROBUSTNESS RESULT: UNROBUST")

            print(f"VERIFICATION TIME: {vtCSR[i]}")
            print(f"NUM_PRED: {numPred[i]}")
    del CSR

    # save verification results
    path = f"./SparseImageStar_evaluation/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_csr_num_pred__results.pkl"
    pickle.dump([rbCSR, vtCSR, numPred], open(save_file, "wb"))

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
            elif rbIM[i] == 2:
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


def verify_vgg16_converted_network_relaxation(dtype='float64'):

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
    M = 4
    RF = (np.arange(M)+1)*0.25
    rbIM = np.zeros([M, N])
    vtIM = np.zeros([M, N])

    print(f"\n\nVerifying vggnet16 with ImageStar")
    for j in range(M):
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
                rbIM[j, i] = np.nan
                vtIM[j, i] = np.nan
            else:
                IM = ImageStar(lb, ub)
                rbIM[j, i], vtIM[j, i], _, _ = certifyRobustness(net=starvNet, inputs=IM, labels=label,
                    veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                    RF=RF[j], DR=0, return_output=False, show=False)
                
                if rbIM[j, i] == 1:
                    print(f"ROBUSTNESS RESULT: ROBUST")
                elif rbIM[j, i] == 2:
                    print(f"ROBUSTNESS RESULT: UNKNOWN")
                elif rbIM[j, i] == 0:
                    print(f"ROBUSTNESS RESULT: UNROBUST")

                print(f"VERIFICATION TIME: {vtIM[j, i]}")
    del IM

    # save verification results
    path = f"./SparseImageStar_evaluation/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_converted_relaxation_results.pkl"
    pickle.dump([RF, rbIM, vtIM], open(save_file, "wb"))

    headers = [f"ImageStar"]

    # Robustness Resluts
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print()
    data = [np.arange(N)]
    for j in range(M):
        data.append(rbIM[j,:])
    print(tabulate(data, headers=headers))

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(path+f"vggnet16_vnncomp23_converted_results_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print()
    data = [np.arange(N)]
    for j in range(M):
        data.append(vtIM[j,:])
    print(tabulate(data, headers=headers))

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(path+f"vggnet16_vnncomp23_converted_relaxation_results.tex", "w") as f:
        print(Tlatex, file=f)

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def verify_vgg16_network_spec_cn(dtype='float64'):

    print('=================================================================================')
    print(f"Verification of VGG16 Network against Infinity Norm Attack Spec_cn")
    print('=================================================================================\n')

    folder_dir = f"./SparseImageStar_evaluation/vnncomp2023/vggnet16"
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())


    shape = (3, 224, 224)
    
    # VNNLIB_FILE = 'vnncomp2023_benchmarks/benchmarks/vggnet16/vnnlib/spec_cn/spec_c0_corn_atk200.vnnlib'
    vnnlib_dir = f"{folder_dir}/vnnlib/spec_cn"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)

    # save verification results
    path = f"./SparseImageStar_evaluation/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_spec_cn_results.pkl"

    N = len(vnnlib_files)
    rbCSR = np.zeros(N)
    vtCSR = np.zeros(N)
    rbCOO = np.zeros(N)
    vtCOO = np.zeros(N)
    numPred = np.zeros(N)

    show = True
 
    print(f"\n\nVerifying vggnet16 with SparseImageStar in CSR format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        print(f"\n Loading a VNNLIB file")
        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        CSR = SparseImageStar2DCSR(lb, ub)
        del lb, ub, bounds

        rbCSR[i], vtCSR[i], _, Y = certifyRobustness(net=starvNet, inputs=CSR, labels=label,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=show)
        numPred[i] = Y.num_pred
    
        if rbCSR[i] == 1:
            print(f"ROBUSTNESS RESULT: ROBUST")
        elif rbCSR[i] == 2:
            print(f"ROBUSTNESS RESULT: UNKNOWN")
        elif rbCSR[i] == 0:
            print(f"ROBUSTNESS RESULT: UNROBUST")

        print(f"VERIFICATION TIME: {vtCSR[i]}")
        print(f"NUM_PRED: {numPred[i]}")
        pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))
    del CSR, Y

    print(f"\n\nVerifying vggnet16 with SparseImageStar in COO format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        print(f"\n Loading a VNNLIB file")
        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        COO = SparseImageStar2DCOO(lb, ub)
        del lb, ub, bounds

        rbCOO[i], vtCOO[i], _, _ = certifyRobustness(net=starvNet, inputs=COO, labels=label,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=show)
        
        if rbCOO[i] == 1:
            print(f"ROBUSTNESS RESULT: ROBUST")
        elif rbCOO[i] == 2:
            print(f"ROBUSTNESS RESULT: UNKNOWN")
        elif rbCOO[i] == 0:
            print(f"ROBUSTNESS RESULT: UNROBUST")

        print(f"VERIFICATION TIME: {vtCOO[i]}")
        pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))

    pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))

    headers = [f"SIM_csr, SIM_coo"]

    # Robustness Resluts
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), rbCSR, rbCOO], headers=headers))
    print()

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), vtCSR, vtCOO], headers=headers))
    print()

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def verify_vgg16_network_spec_cn_relaxation(dtype='float64'):

    print('=================================================================================')
    print(f"Verification of VGG16 Network against Infinity Norm Attack Spec_cn")
    print('=================================================================================\n')

    folder_dir = f"./SparseImageStar_evaluation/vnncomp2023/vggnet16"
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())


    shape = (3, 224, 224)
    
    # VNNLIB_FILE = 'vnncomp2023_benchmarks/benchmarks/vggnet16/vnnlib/spec_cn/spec_c0_corn_atk200.vnnlib'
    vnnlib_dir = f"{folder_dir}/vnnlib/spec_cn"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)

    # save verification results
    path = f"./SparseImageStar_evaluation/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_spec_cn_relaxation_results.pkl"

    N = len(vnnlib_files)
    M = 4
    RF = (np.arange(M)+1)*0.25
    rbCSR = np.zeros([M, N])
    vtCSR = np.zeros([M, N])
    rbCOO = np.zeros([M, N])
    vtCOO = np.zeros([M, N])
    numPred = np.zeros([M, N])

    show = True

    for j in range(M):

        print(f"\n\nVerifying vggnet16 with SparseImageStar in CSR format")
        for i, vnnlib_file in enumerate(vnnlib_files):
            vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

            with open(vnnlib_file_dir) as f:
                first_line = f.readline().strip('\n')
            label = int(re.findall(r'\b\d+\b', first_line)[0])

            print(f"\n Loading a VNNLIB file")
            vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

            box, spec_list = vnnlib_rv[0]
            bounds = np.array(box, dtype=inp_dtype)
            # transpose from [C, H, W] to [H, W, C]
            lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
            ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

            num_attack_pixel = (lb != ub).sum()
            print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

            CSR = SparseImageStar2DCSR(lb, ub)
            del lb, ub, bounds

            rbCSR[i], vtCSR[i], _, Y = certifyRobustness(net=starvNet, inputs=CSR, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=RF[j], DR=0, return_output=False, show=show)
            numPred[i] = Y.num_pred
        
            if rbCSR[i] == 1:
                print(f"ROBUSTNESS RESULT: ROBUST")
            elif rbCSR[i] == 2:
                print(f"ROBUSTNESS RESULT: UNKNOWN")
            elif rbCSR[i] == 0:
                print(f"ROBUSTNESS RESULT: UNROBUST")

            print(f"VERIFICATION TIME: {vtCSR[i]}")
            print(f"NUM_PRED: {numPred[i]}")
            pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))
        del CSR, Y

        print(f"\n\nVerifying vggnet16 with SparseImageStar in COO format")
        for i, vnnlib_file in enumerate(vnnlib_files):
            vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

            with open(vnnlib_file_dir) as f:
                first_line = f.readline().strip('\n')
            label = int(re.findall(r'\b\d+\b', first_line)[0])

            print(f"\n Loading a VNNLIB file")
            vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

            box, spec_list = vnnlib_rv[0]
            bounds = np.array(box, dtype=inp_dtype)
            # transpose from [C, H, W] to [H, W, C]
            lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
            ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

            num_attack_pixel = (lb != ub).sum()
            print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

            COO = SparseImageStar2DCOO(lb, ub)
            del lb, ub, bounds

            rbCOO[i], vtCOO[i], _, _ = certifyRobustness(net=starvNet, inputs=COO, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=RF[j], DR=0, return_output=False, show=show)
            
            if rbCOO[i] == 1:
                print(f"ROBUSTNESS RESULT: ROBUST")
            elif rbCOO[i] == 2:
                print(f"ROBUSTNESS RESULT: UNKNOWN")
            elif rbCOO[i] == 0:
                print(f"ROBUSTNESS RESULT: UNROBUST")

            print(f"VERIFICATION TIME: {vtCOO[i]}")
            pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))

    pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))

    headers = [f"SIM_csr, SIM_coo"]

    # Robustness Resluts
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), rbCSR, rbCOO], headers=headers))
    print()

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), vtCSR, vtCOO], headers=headers))
    print()

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, .224, 0.225)

class VGG16_crop() :
    def __init__(self, resize) :
        self.crop = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize)
        ])
        
    def __call__(self, img) :
        return self.crop(img)
    
class VGG16_normalizer() :
    def __init__(self, mean, std) :
        self.normalize = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        
    def __call__(self, img) :
        return self.normalize(img)
    
def verify_vgg16_network_spec_cn_direct(dtype='float64'):

    print('=================================================================================')
    print(f"Verification of VGG16 Network against Infinity Norm Attack Spec_cn")
    print('=================================================================================\n')

    folder_dir = f"./SparseImageStar_evaluation/vnncomp2023/vggnet16"
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())


    shape = (3, 224, 224)
    
    crop = VGG16_crop(resize)
    normalizer = VGG16_normalizer(mean, std)

    epsilon = 0.01/(255) 

    image_dir = folder_dir + f"/vnnlib/spec_cn/ILSVRC2012_val_00011122.JPEG"
    image = torchvision.io.read_image(path = image_dir) / 255.0
    image = crop(image)

    data = normalizer(image).permute(1, 2, 0).numpy().copy()
    y = starvNet.evaluate(data)
    label = np.array([y.argmax()])[0]
    m = np.prod(shape)

    print(f"image_file_path: {image_dir}")
    print(f"label: {label}")
    print(f"epsilon: {epsilon}")

    index_pixel2attack_list = np.load('vgg16_attack_pixel_lists_200_300_400_500_1000_2000_3000.npy', allow_pickle=True)

    # save verification results
    path = f"./SparseImageStar_evaluation/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_spec_cn_results.pkl"

    N = len(index_pixel2attack_list)
    rbCSR = np.zeros(N)
    vtCSR = np.zeros(N)
    rbCOO = np.zeros(N)
    vtCOO = np.zeros(N)
    numPred = np.zeros(N)

    show = True
 
    print(f"\n\nVerifying vggnet16 with SparseImageStar in CSR format")
    for i, attack_pixel_index in enumerate(index_pixel2attack_list):
        
        image_lb = image.clone().reshape(-1)
        image_ub = image.clone().reshape(-1)
        image_lb[attack_pixel_index] = (image_lb[attack_pixel_index] - epsilon).clamp(0, 1)
        image_ub[attack_pixel_index] = (image_ub[attack_pixel_index] + epsilon).clamp(0, 1)
        image_lb = image_lb.reshape(shape)
        image_ub = image_ub.reshape(shape)

        lb = normalizer(image_lb).permute(1, 2, 0).numpy().ravel()
        ub = normalizer(image_ub).permute(1, 2, 0).numpy().ravel()

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying image with {num_attack_pixel}, {len(attack_pixel_index)} attacked pixels")

        CSR = SparseImageStar2DCSR(lb, ub)
        del lb, ub, bounds

        rbCSR[i], vtCSR[i], _, Y = certifyRobustness(net=starvNet, inputs=CSR, labels=label,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=show)
        numPred[i] = Y.num_pred
    
        if rbCSR[i] == 1:
            print(f"ROBUSTNESS RESULT: ROBUST")
        elif rbCSR[i] == 2:
            print(f"ROBUSTNESS RESULT: UNKNOWN")
        elif rbCSR[i] == 0:
            print(f"ROBUSTNESS RESULT: UNROBUST")

        print(f"VERIFICATION TIME: {vtCSR[i]}")
        print(f"NUM_PRED: {numPred[i]}")
        pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))
    del CSR, Y

    print(f"\n\nVerifying vggnet16 with SparseImageStar in COO format")
    for i, attack_pixel_index in enumerate(index_pixel2attack_list):
        
        image_lb = image.clone().reshape(-1)
        image_ub = image.clone().reshape(-1)
        image_lb[attack_pixel_index] = (image_lb[attack_pixel_index] - epsilon).clamp(0, 1)
        image_ub[attack_pixel_index] = (image_ub[attack_pixel_index] + epsilon).clamp(0, 1)
        image_lb = image_lb.reshape(shape)
        image_ub = image_ub.reshape(shape)

        lb = normalizer(image_lb).permute(1, 2, 0).numpy().ravel()
        ub = normalizer(image_ub).permute(1, 2, 0).numpy().ravel()

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying image with {num_attack_pixel}, {len(attack_pixel_index)} attacked pixels")

        COO = SparseImageStar2DCOO(lb, ub)
        del lb, ub, bounds

        rbCOO[i], vtCOO[i], _, _ = certifyRobustness(net=starvNet, inputs=COO, labels=label,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=show)
        
        if rbCOO[i] == 1:
            print(f"ROBUSTNESS RESULT: ROBUST")
        elif rbCOO[i] == 2:
            print(f"ROBUSTNESS RESULT: UNKNOWN")
        elif rbCOO[i] == 0:
            print(f"ROBUSTNESS RESULT: UNROBUST")

        print(f"VERIFICATION TIME: {vtCOO[i]}")
        pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))

    pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))

    headers = [f"SIM_csr, SIM_coo"]

    # Robustness Resluts
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), rbCSR, rbCOO], headers=headers))
    print()

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), vtCSR, vtCOO], headers=headers))
    print()

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
    file_dir = folder_dir + 'vggnet16_vnncomp23_csr_num_pred__results.pkl'
    with open(file_dir, 'rb') as f:
        _, _, num_pred = pickle.load(f)

    mat_file = scipy.io.loadmat(f"{folder_dir}nnv_vggnet16_results.mat")
    rbNNV = mat_file['rb_im'].ravel()
    vtNNV = mat_file['vt_im'].ravel()

    mat_file = scipy.io.loadmat(f"{folder_dir}nnv_vggnet16_converted_results.mat")
    rbNNVc = mat_file['rb_im'].ravel()
    vtNNVc = mat_file['vt_im'].ravel()

    file_dir = folder_dir + 'vggnet16_vnncomp23_spec_cn_results.pkl'
    with open(file_dir, 'rb') as f:
        numPred_cn, rbCSR_cn, vtCSR_cn, rbCOO_cn, vtCOO_cn = pickle.load(f)

    f_dir = f"./SparseImageStar_evaluation/vnncomp2023/vggnet16"
    net_dir = f"{f_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)
    vnnlib_dir = f"{f_dir}/vnnlib"
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
    vt_NNENUM = [3.5, 3.4, 9.3, 4.8, 18.1, 35.7, 6.5, 18.3, 133.8, 10.6, 40.9, 57.6, 'T/O', 236.5, 746.6]
    vt_DP = 'O/M'

    headers = ['Specs', 'm', 'e', 'Result', 'IM', 'SIM_csr', 'SIM_coo', 'DP', 'NNV', 'IM', 'NNV', 'NNENUM']

    result = 'UNSAT'
    
    data = []
    for i in range(N):
        vt_im = 'O/M' if np.isnan(vtIM[i]) else f"{vtIM[i]:0.1f}"
        vt_imc = 'O/M' if np.isnan(vtIMc[i]) else f"{vtIMc[i]:0.1f}"
        vt_nnv = 'O/M' if vtNNV[i] < 0 else f"{vtNNV[i]:0.1f}"
        vt_nnvc = 'O/M' if vtNNVc[i] < 0 else f"{vtNNVc[i]:0.1f}"
        nPred = 'NA' if np.isnan(vtCSR[i]) else f"{num_pred[i]}"
        data.append([i, nPred, num_attack_pixel[i], result,  vt_im, f"{vtCSR[i]:0.1f}", f"{vtCOO[i]:0.1f}", vt_DP, vt_nnv, vt_imc, vt_nnvc, vt_NNENUM[i]])

    num_attack_pixel_cn = [200, 300, 400, 500, 1000, 2000, 3000]
    N_cn = len(numPred_cn)
    vt_NNENUM_cn = [744.02, 1060.96, 1354.75, 1781.26, 'T/O', 'T/O', 'O/M']
    for i in range(N_cn):
        vt_im = 'O/M' 
        vt_imc = 'O/M'
        vt_nnv = 'O/M'
        vt_nnvc = 'O/M'
        nPred = 'NA' if np.isnan(vtCSR_cn[i]) else f"{numPred_cn[i]}"
        data.append([f"c_{i}", nPred, num_attack_pixel_cn[i], result,  vt_im, f"{vtCSR_cn[i]:0.1f}", f"{vtCOO_cn[i]:0.1f}", vt_DP, vt_nnv, vt_imc, vt_nnvc, vt_NNENUM_cn[i]])

    print(tabulate(data, headers=headers))

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(folder_dir+f"vggnet16_vnncomp23_results_full_table.tex", "w") as f:
        print(Tlatex, file=f)

def plot_table_vgg16_network_with_relaxation():
    folder_dir = 'SparseImageStar_evaluation/results/'
    file_dir = folder_dir + 'vggnet16_vnncomp23_results.pkl'
    with open(file_dir, 'rb') as f:
        rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, _, _ = pickle.load(f)
    file_dir = folder_dir + 'vggnet16_vnncomp23_converted_results.pkl'
    with open(file_dir, 'rb') as f:
        rbIMc, vtIMc = pickle.load(f)
    file_dir = folder_dir + 'vggnet16_vnncomp23_converted_relaxation_results.pkl'
    with open(file_dir, 'rb') as f:
        RF, rbIMc_RF, vtIMc_RF = pickle.load(f)


    f_dir = f"./SparseImageStar_evaluation/vnncomp2023/vggnet16"
    net_dir = f"{f_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)
    vnnlib_dir = f"{f_dir}/vnnlib"
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

    headers = ['Specs', '$e$', 'Result', 'IM', 'SIM_csr', 'SIM_coo', 'IM']
    for rf_ in RF:
        headers.append(f'{rf_:0.2f}')
    headers.append('NNENUM')
                
    result = 'UNSAT'
    
    data = []
    for i in range(N):
        vt_im = 'O/M' if np.isnan(vtIM[i]) else f"{vtIM[i]:0.1f}"
        vt_imc = 'O/M' if np.isnan(vtIMc[i]) else f"{vtIMc[i]:0.1f}"
        data_ = [i, num_attack_pixel[i], result,  vt_im, f"{vtCSR[i]:0.1f}", f"{vtCOO[i]:0.1f}", vt_imc]
        for j in range(len(RF)):
            vt_imc_rf = 'O/M' if np.isnan(vtIMc_RF[j, i]) else f"{vtIMc_RF[j, i]:0.1f}"
            data_.append(vt_imc_rf)
        data_.append(vt_NNENUM[i])
        data.append(data_)
    print(tabulate(data, headers=headers))

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(folder_dir+f"vggnet16_vnncomp23_results_full_table.tex", "w") as f:
        print(Tlatex, file=f)


if __name__ == "__main__":
    # verify_vgg16_network(dtype='float64')
    # verify_vgg16_converted_network(dtype='float64')
    # verify_vgg16_converted_network_relaxation(dtype='float64')
    # plot_table_vgg16_network_with_relaxation()
    # verify_vgg16_network_get_num_pred(dtype='float64')
    # verify_vgg16_network_spec_cn()
    # verify_vgg16_network_spec_cn_direct()
    verify_vgg16_network_spec_cn_relaxation()
    # plot_table_vgg16_network()