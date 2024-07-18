"""
Verify MNIST ConvNet (CAV2020) against brightening attack 
Author: Sung Woo Choi
Date: 06/24/2024
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle

import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.verifier.certifier import certifyRobustness
from StarV.util.load import *

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

def load_convnet(net_dir, net_type, dtype='float32'):

    assert net_type in ['Small', 'Medium', 'Large'], \
    f"There are 3 types of ConvNet networks: /'Small/', /'Medium/', and /'Large/'"

     # loading DNNs into StarV network
    if net_type == 'Small':
        network = load_CAV2020_MNIST_Small_ConvNet(net_dir=net_dir, dtype=dtype)
    elif net_type == 'Medium':
        network = load_CAV2020_MNIST_Medium_ConvNet(net_dir=net_dir, dtype=dtype)
    elif net_type == 'Large':
        network = load_CAV2020_MNIST_Large_ConvNet(net_dir=net_dir, dtype=dtype)
    else:
        raise Exception('Unknown network type for ConvNet')
    return network

def verify_convnet_network(net_type='Small', dtype='float32'):
    
    print('=================================================================================')
    print(f"Verification of CAV2020 {net_type} ConvNet Network against Brightnening Attack")
    print('=================================================================================\n')

    folder_dir = f"./SparseImageStar_evaluation/CAV2020_ImageStar/MNIST_NETS/{net_type}"
    net_dir = f"{folder_dir}/{net_type}_ConvNet.onnx"
    starvNet = load_convnet(net_dir, net_type, dtype=dtype)
    print()
    print(starvNet.info())

    mat_file = scipy.io.loadmat(f"{folder_dir}/images.mat")
    data = mat_file['IM_data'].astype(dtype)
    labels = mat_file['IM_labels'] - 1

    delta = [0.005, 0.01, 0.015]
    d = [250, 245, 240]
    N = 100 # number of test images used for robustness verification
    M = len(delta)
    P = len(d)

    IML = []
    CSRL = []
    COOL = []
    labelL = []

    for i, d_ in enumerate(d):
        IM_j = [] 
        CSR_j = [] 
        COO_j = [] 
        label_j = []
        
        for j, delta_ in enumerate(delta):

            count = 0
            IM_k = [] 
            CSR_k = [] 
            COO_k = [] 
            label_k = []
            for k in range(2000):
                lb, ub = brightening_attack(data[:, :, k], delta=delta_, d=d_, dtype=dtype)
                IM_k.append(ImageStar(lb, ub))
                CSR_k.append(SparseImageStar2DCSR(lb, ub))
                COO_k.append(SparseImageStar2DCOO(lb, ub))
                label_k.append(labels[k])
                count += 1
                    
                if count == N:
                    break
                
            IM_j.append(IM_k)
            CSR_j.append(CSR_k)
            COO_j.append(COO_k)
            label_j.append(label_k)
            
        IML.append(IM_j)
        CSRL.append(CSR_j)
        COOL.append(COO_j)
        labelL.append(label_j)

    rbIM = np.zeros([P, M, N])
    vtIM = np.zeros([P, M, N])
    rbCSR = np.zeros([P, M, N])
    vtCSR = np.zeros([P, M, N])
    rbCOO = np.zeros([P, M, N])
    vtCOO = np.zeros([P, M, N])

    print(f"Verifying {net_type} ConvNet with ImageStar")
    rbIM_table = []
    vtIM_table = []
    for i in range(P):
        rb_delta_table = ['ImageStar']
        vt_delta_table = ['ImageStar']
        for j in range(M):
            IMs = IML[i][j]
            IDs = labelL[i][j]
            print(f"Verifying netowrk with d = {d[i]}, delta = {delta[j]}")

            rbIM[i, j, :], vtIM[i, j, :], _, _ = certifyRobustness(net=starvNet, inputs=IMs, labels=IDs,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
            rb_delta_table.append((rbIM[i, j, :]==1).sum())
            vt_delta_table.append((vtIM[i, j, :].sum() / N))
        rbIM_table.append(rb_delta_table)
        vtIM_table.append(vt_delta_table)

    print(f"\nVerifying {net_type} ConvNet with Sparse Image Star in CSR format")
    rbCSR_table = []
    vtCSR_table = []
    for i in range(P):
        rb_delta_table = ['SIM_CSR']
        vt_delta_table = ['SIM_CSR']
        for j in range(M):
            CSRs = CSRL[i][j]
            IDs = labelL[i][j]
            print(f"Verifying netowrk with d = {d[i]}, delta = {delta[j]}")

            rbCSR[i, j, :], vtCSR[i, j, :], _, _ = certifyRobustness(net=starvNet, inputs=CSRs, labels=IDs,
                    veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                    RF=0.0, DR=0, return_output=True, show=False)
            rb_delta_table.append((rbCSR[i, j, :]==1).sum())
            vt_delta_table.append((vtCSR[i, j, :].sum() / N))
        rbCSR_table.append(rb_delta_table)
        vtCSR_table.append(vt_delta_table)
            
    print(f"\nVerifying {net_type} ConvNet with Sparse Image Star in COO format")
    rbCOO_table = []
    vtCOO_table = []
    for i in range(P):
        rb_delta_table = ['SIM_COO']
        vt_delta_table = ['SIM_COO']
        for j in range(M):
            COOs = COOL[i][j]
            IDs = labelL[i][j]
            print(f"Verifying netowrk with d = {d[i]}, delta = {delta[j]}")

            rbCOO[i, j, :], vtCOO[i, j, :], _, _ = certifyRobustness(net=starvNet, inputs=COOs, labels=IDs,
                    veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                    RF=0.0, DR=0, return_output=True, show=False)
            rb_delta_table.append((rbCOO[i, j, :]==1).sum())
            vt_delta_table.append((vtCOO[i, j, :].sum() / N))
        rbCOO_table.append(rb_delta_table)
        vtCOO_table.append(vt_delta_table)

    # save verification results
    path = f"./SparseImageStar_evaluation/results"
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(P):
        headers = [f"d={d[i]}", f"delta={delta[0]}", f"delta={delta[1]}", f"delta={delta[2]}"]
        # Robustness Resluts
        data = [rbIM_table[i], rbCSR_table[i], rbCOO_table[i]]
        print('-----------------------------------------------------')
        print('Robustness')
        print('-----------------------------------------------------')
        print(tabulate(data, headers=headers))
        print()

        Tlatex = tabulate(data, headers=headers, tablefmt='latex')
        with open(path+f"/{net_type}ConvNet_brightAttack_Table_d{d[i]}_rb.tex", "w") as f:
            print(Tlatex, file=f)

        # Verification Time Results
        data = [vtIM_table[i], vtCSR_table[i], vtCOO_table[i]]
        print('-----------------------------------------------------')
        print('Verification Time')
        print('-----------------------------------------------------')
        print(tabulate(data, headers=headers))
        print()

        Tlatex = tabulate(data, headers=headers, tablefmt='latex')
        with open(path+f"/{net_type}ConvNet_brightAttack_Table_d{d[i]}_vt.tex", "w") as f:
            print(Tlatex, file=f)
        
    save_file = path + f"/{net_type}ConvNet_brightAttack_results.pkl"
    pickle.dump([rbIM, vtIM, rbIM_table, vtIM_table, rbCSR, vtCSR, rbCSR_table, vtCSR_table, rbCOO, vtCOO, rbCOO_table, vtCOO_table], open(save_file, "wb"))

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


if __name__ == "__main__":
    verify_convnet_network(net_type='Small', dtype='float64')
    verify_convnet_network(net_type='Medium', dtype='float64')
    verify_convnet_network(net_type='Large', dtype='float64')
