
"""
Verify MNIST ConvNet (CAV2020) against infinity norm attack 
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

def infinity_norm_attack(image, epsilon, dtype=np.float32):
    lb = image.copy().astype(dtype)
    ub = image.copy().astype(dtype)
    lb = (lb - epsilon).clip(0, 255)
    ub = (ub + epsilon).clip(0, 255)
    return lb, ub

def load_convnet(net_dir, net_type, dtype='float32'):

    assert net_type in ['Small', 'Medium', 'Large'], \
    f"There are 3 types of ConvNet networks: /'Small/', /'Medium/', and /'Large/'"

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
    print(f"Verification of CAV2020 {net_type} ConvNet Network against Infinity Norm Attack")
    print('=================================================================================\n')

    folder_dir = f"./SparseImageStar_evaluation/CAV2020_ImageStar/MNIST_NETS/{net_type}"
    net_dir = f"{folder_dir}/{net_type}_ConvNet.onnx"
    starvNet = load_convnet(net_dir, net_type, dtype=dtype)
    print()
    print(starvNet.info())

    mat_file = scipy.io.loadmat(f"{folder_dir}/images.mat")
    data = mat_file['IM_data'].astype(dtype)
    labels = mat_file['IM_labels'] - 1

    epsilon = [1, 2, 5]
    N = 100 # number of test images used for robustness verification
    E = len(epsilon)

    IML = []
    CSRL = []
    COOL = []
    labelL = []
        
    for j, eps_ in enumerate(epsilon):

        count = 0
        IM_k = [] 
        CSR_k = [] 
        COO_k = [] 
        label_k = []

        for k in range(2000):
            lb, ub = infinity_norm_attack(data[:, :, k], epsilon=eps_, dtype=dtype)
            IM_k.append(ImageStar(lb, ub))
            CSR_k.append(SparseImageStar2DCSR(lb, ub))
            COO_k.append(SparseImageStar2DCOO(lb, ub))
            label_k.append(labels[k])
            count += 1
                
            if count == N:
                break
            
        IML.append(IM_k)
        CSRL.append(CSR_k)
        COOL.append(COO_k)
        labelL.append(label_k)

    rbIM = np.zeros([E, N])
    vtIM = np.zeros([E, N])
    rbCSR = np.zeros([E, N])
    vtCSR = np.zeros([E, N])
    rbCOO = np.zeros([E, N])
    vtCOO = np.zeros([E, N])

    print(f"Verifying {net_type} ConvNet with ImageStar")
    rbIM_table = ['ImageStar']
    vtIM_table = ['ImageStar']
    for i in range(E):
        IMs = IML[i]
        IDs = labelL[i]
        print(f"Verifying netowrk with epsilon = {epsilon[j]}")

        rbIM[i, :], vtIM[i, :], _, _ = certifyRobustness(net=starvNet, inputs=IMs, labels=IDs,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=False)
        rbIM_table.append((rbIM[i, :]==1).sum())
        vtIM_table.append((vtIM[i, :].sum() / N))

    print(f"\nVerifying {net_type} ConvNet with Sparse Image Star in CSR format")
    rbCSR_table = ['SIM_CSR']
    vtCSR_table = ['SIM_CSR']
    for i in range(E):
        CSRs = CSRL[i]
        IDs = labelL[i]
        print(f"Verifying netowrk with epsilon = {epsilon[j]}")

        rbCSR[i, :], vtCSR[i, :], _, _ = certifyRobustness(net=starvNet, inputs=CSRs, labels=IDs,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=True, show=False)
        rbCSR_table.append((rbCSR[i, :]==1).sum())
        vtCSR_table.append((vtCSR[i, :].sum() / N))
            
    print(f"\nVerifying {net_type} ConvNet with Sparse Image Star in COO format")
    rbCOO_table = ['SIM_COO']
    vtCOO_table = ['SIM_COO']
    for i in range(E):
        COOs = COOL[i]
        IDs = labelL[i]
        print(f"Verifying netowrk with epsilon = {epsilon[j]}")

        rbCOO[i, :], vtCOO[i, :], _, _ = certifyRobustness(net=starvNet, inputs=COOs, labels=IDs,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=True, show=False)
        rbCOO_table.append((rbCOO[i, :]==1).sum())
        vtCOO_table.append((vtCOO[i, :].sum() / N))

    # save verification results
    path = f"./SparseImageStar_evaluation/results"
    if not os.path.exists(path):
        os.makedirs(path)

    headers = [f"", f"eps={epsilon[0]}", f"eps={epsilon[1]}", f"eps={epsilon[2]}"]

    # Robustness Resluts
    data = [rbIM_table, rbCSR_table, rbCOO_table]
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate(data, headers=headers))
    print()

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(path+f"/{net_type}ConvNet_infNormAttack_Table_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    data = [vtIM_table, vtCSR_table, vtCOO_table]
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate(data, headers=headers))
    print()

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(path+f"/{net_type}ConvNet_infNormAttack_Table_vt.tex", "w") as f:
        print(Tlatex, file=f)
        
    save_file = path + f"/{net_type}ConvNet_infNormAttack_results.pkl"
    pickle.dump([rbIM, vtIM, rbIM_table, vtIM_table, rbCSR, vtCSR, rbCSR_table, vtCSR_table, rbCOO, vtCOO, rbCOO_table, vtCOO_table], open(save_file, "wb"))

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


if __name__ == "__main__":
    verify_convnet_network(net_type='Small', dtype='float32')
    verify_convnet_network(net_type='Medium', dtype='float32')
    verify_convnet_network(net_type='Large', dtype='float32')