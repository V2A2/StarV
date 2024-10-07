"""
Verify vgg16 (VNNCOMP2023) against infinity norm
Author: Sung Woo Choi
Date: 06/24/2024
"""

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import torchvision.transforms as transforms
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

def load_eran_network(net_dir, net_type, data_type, trained_type, dtype='float64'):

    assert net_type in ['Small', 'Big'], \
    f"network type for ERAN convolutional networks should 'Small', or 'Big' but received {net_type}"
    assert data_type in ['MNIST', 'CIFAR10'], \
    f"data type for ERAN convolutional networks should 'MNIST', or 'CIFAR10' but received {data_type}"
    if net_type == 'Big':
        assert trained_type == 'DiffAI', \
        f"only DiffAI is availiable for convBigRELU network"
    else:
        assert trained_type in ['DiffAI', 'PGDK', 'Point'], \
        f"trained type for ERAN convolutional networks should be 'DiffAI', 'PGDK', 'Point' but received {trained_type}"

    # loading DNNs into StarV network
    network = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    return network

def load_eran_dataset(data_type, folder_dir, dtype='float64'):
    data_dir = f"{folder_dir}/{data_type.lower()}_test.csv" 
    tests = np.genfromtxt(data_dir, delimiter=',', dtype=dtype)
    N = len(tests)
    return tests, N

def normalize(image, data_type):
    img = image.copy()
    if data_type == 'CIFAR10':
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img -= mean
        img /= std
    return img

def verify_eran_network(net_type='Small', data_type='MNIST', trained_type='DiffAI', dtype='float64', show=False):

    print('==================================================================================================')
    print(f"Verification of ERAN DEEPPOLY_{data_type} conv{net_type}ReLU Network against Infinity Norm Attack")
    print('==================================================================================================\n')

    folder_dir = f"./SparseImageStar_evaluation/ERAN/DEEPPOLY_{data_type}"
    net_dir = f"{folder_dir}/conv{net_type}RELU__{trained_type}.onnx"
    starvNet = load_eran_network(net_dir, net_type, data_type, trained_type, dtype=dtype)
    print()
    print(starvNet.info())

    dataset, N = load_eran_dataset(data_type, folder_dir)
    
    if data_type == 'CIFAR10':
        if net_type == 'Small':
            epsilon = [0.002, 0.004, 0.006, 0.008, 0.01, 0.012]
        else:
            epsilon = [0.006, 0.008]

        shape = (32, 32, 3)

    else:
        # MNIST dataset
        if net_type == 'Small':
            epsilon = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
        else:
            epsilon = [0.1, 0.2, 0.3]

        shape = (28, 28)

    E = len(epsilon)
    
    rbIM = np.zeros([E, N])
    vtIM = np.zeros([E, N])
    rbCSR = np.zeros([E, N])
    vtCSR = np.zeros([E, N])
    rbCOO = np.zeros([E, N])
    vtCOO = np.zeros([E, N])

    rbIM_table = ['ImageStar']
    vtIM_table = ['ImageStar']
    rbCSR_table = ['SIM_CSR']
    vtCSR_table = ['SIM_CSR']
    rbCOO_table = ['SIM_COO']
    vtCOO_table = ['SIM_COO']

    print(f"Verifying conv{net_type}RELU__{trained_type} {data_type} with ImageStar")
    for i, eps_ in enumerate(epsilon):
        print(f"Verifying netowrk with epsilon = {eps_}")
        for j, data in enumerate(dataset):
            if show: print(f"Working on image {j}")
            img = data[1:].reshape(shape) / 255
            label = int(data[0]) 
            # infinity norm attack
            lb = normalize((img - eps_).clip(0, 1), data_type)
            ub = normalize((img + eps_).clip(0, 1), data_type)
            
            IM = ImageStar(lb, ub)
            rbIM[i, j], vtIM[i, j], _, _ = certifyRobustness(net=starvNet, inputs=IM, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None,
                RF=0.0, DR=0, return_output=False, show=False)
        rbIM_table.append((rbIM[i, :]==1).sum())
        vtIM_table.append((vtIM[i, :].sum() / N))
    del IM

    print(f"\nVerifying {net_type} conv{net_type}RELU__{trained_type} {data_type} with SparseImageStar in CSR format")
    for i, eps_ in enumerate(epsilon):
        print(f"Verifying netowrk with epsilon = {eps_}")
        for j, data in enumerate(dataset):
            if show: print(f"Working on image {j}")
            img = data[1:].reshape(shape) / 255
            label = int(data[0]) 
            # infinity norm attack
            lb = normalize((img - eps_).clip(0, 1), data_type)
            ub = normalize((img + eps_).clip(0, 1), data_type)

            CSR = SparseImageStar2DCSR(lb, ub)
            rbCSR[i, j], vtCSR[i, j], _, _ = certifyRobustness(net=starvNet, inputs=CSR, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
        rbCSR_table.append((rbCSR[i, :]==1).sum())
        vtCSR_table.append((vtCSR[i, :].sum() / N))
    del CSR

    print(f"\nVerifying {net_type} conv{net_type}RELU__{trained_type} {data_type} with SparseImageStar in COO format")
    for i, eps_ in enumerate(epsilon):
        print(f"Verifying netowrk with epsilon = {eps_}")
        for j, data in enumerate(dataset):
            if show: print(f"Working on image {j}")
            img = data[1:].reshape(shape) / 255
            label = int(data[0]) 
            # infinity norm attack
            lb = normalize((img - eps_).clip(0, 1), data_type)
            ub = normalize((img + eps_).clip(0, 1), data_type)

            COO = SparseImageStar2DCOO(lb, ub)
            rbCOO[i, j], vtCOO[i, j], _, _ = certifyRobustness(net=starvNet, inputs=COO, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
        rbCOO_table.append((rbCOO[i, :]==1).sum())
        vtCOO_table.append((vtCOO[i, :].sum() / N))
    del COO

    # save verification results
    path = f"./SparseImageStar_evaluation/results"
    if not os.path.exists(path):
        os.makedirs(path)

    headers = [""]
    for eps_ in epsilon:
        headers.append(f"eps={eps_}")
    # Robustness Resluts
    data = [rbIM_table, rbCSR_table, rbCOO_table]
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate(data, headers=headers))
    print()

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(path+f"/conv{net_type}RELU__{trained_type}_{data_type}_eran_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    data = [vtIM_table, vtCSR_table, vtCOO_table]
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate(data, headers=headers))
    print()

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(path+f"/conv{net_type}RELU__{trained_type}_{data_type}_eran_vt.tex", "w") as f:
        print(Tlatex, file=f)
        
    save_file = path + f"/conv{net_type}RELU__{trained_type}_{data_type}_eran_results.pkl"
    pickle.dump([rbIM, vtIM, rbIM_table, vtIM_table, rbCSR, vtCSR, rbCSR_table, vtCSR_table, rbCOO, vtCOO, rbCOO_table, vtCOO_table], open(save_file, "wb"))

    print('=====================================================')
    print('DONE!')
    print('=====================================================')

if __name__ == "__main__":
    show = False
    verify_eran_network(net_type='Small', data_type='MNIST', trained_type='DiffAI', dtype='float32', show=show)
    verify_eran_network(net_type='Small', data_type='MNIST', trained_type='PGDK', dtype='float32', show=show)
    verify_eran_network(net_type='Small', data_type='MNIST', trained_type='Point', dtype='float32', show=show)

    verify_eran_network(net_type='Big', data_type='MNIST', trained_type='DiffAI', dtype='float32', show=show)

    verify_eran_network(net_type='Small', data_type='CIFAR10', trained_type='DiffAI', dtype='float32', show=show)
    verify_eran_network(net_type='Small', data_type='CIFAR10', trained_type='PGDK', dtype='float32', show=show)
    verify_eran_network(net_type='Small', data_type='CIFAR10', trained_type='Point', dtype='float32', show=show)

    verify_eran_network(net_type='Big', data_type='CIFAR10', trained_type='DiffAI', dtype='float32', show=show)