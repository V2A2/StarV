"""
Verify oval21 (VNNCOMP2021) against infinity norm (different epsilons used compared to VNNCOMP2021)
Author: Sung Woo Choi
Date: 06/24/2024
"""

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle

import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.verifier.certifier import certifyRobustness
from StarV.util.load import *


def load_oval21(net_dir, net_type, dtype='float32'):

    assert net_type in ['base', 'deep', 'wide'], \
    f"network type for oval21 should be /'base/', /'deep'/, or /'wide'/ but received {net_type}"

    # loading DNNs into StarV network
    network = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    return network

def verify_oval21_network(net_type='base', dtype='float32'):
    
    print('=================================================================================')
    print(f"Verification of {net_type} OVAL21 Network against Infinity Norm Attack")
    print('=================================================================================\n')

    folder_dir = f"./SparseImageStar_evaluation/vnncomp2021/oval21/onnx"
    net_dir = f"{folder_dir}/cifar_{net_type}_kw.onnx"
    starvNet = load_oval21(net_dir, net_type, dtype=dtype)
    print()
    print(starvNet.info())
    
    # loading cifar10 dataset
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    cifar_test = datasets.CIFAR10('./cifardata/', train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor()]))
    
    N = 100 # number of images
    E = 2 # number of epsilons
    
    # select pseudo-random images
    rand_ind = np.random.RandomState(1000).randint(1, len(cifar_test), size=N)
    
    if net_type == 'base':
        epsilon = (np.arange(E)+1)/255
    # elif net_type == 'deep':
    #     epsilon = (np.arange(E)+1)*0.75/255
    else:
        epsilon = (np.arange(E)+1)*0.5/255

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

    print(f"\nVerifying {net_type} oval21 with SparseImageStar in CSR format")
    for i, eps_ in enumerate(epsilon):
        print(f"Verifying netowrk with epsilon = {eps_}")
        for j in range(N):
            img, label = cifar_test[rand_ind[i]]
            # infinity norm attack and convert image to channel last (i.e. [H, W, C])
            lb = normalizer((img - eps_).clamp(0, 1)).permute(1, 2, 0).numpy().copy()
            ub = normalizer((img + eps_).clamp(0, 1)).permute(1, 2, 0).numpy().copy()

            CSR = SparseImageStar2DCSR(lb, ub)
            rbCSR[i, j], vtCSR[i, j], _, _ = certifyRobustness(net=starvNet, inputs=CSR, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
        rbCSR_table.append((rbCSR[i, :]==1).sum())
        vtCSR_table.append((vtCSR[i, :].sum() / N))
    del CSR
    
    print(f"Verifying {net_type} oval21 with ImageStar")
    for i, eps_ in enumerate(epsilon):
        print(f"Verifying netowrk with epsilon = {eps_}")
        for j in range(N):
            img, label = cifar_test[rand_ind[i]]
            # infinity norm attack and convert image to channel last (i.e. [H, W, C])
            lb = normalizer((img - eps_).clamp(0, 1)).permute(1, 2, 0).numpy().copy()
            ub = normalizer((img + eps_).clamp(0, 1)).permute(1, 2, 0).numpy().copy()
            
            IM = ImageStar(lb, ub)
            rbIM[i, j], vtIM[i, j], _, _ = certifyRobustness(net=starvNet, inputs=IM, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
        rbIM_table.append((rbIM[i, :]==1).sum())
        vtIM_table.append((vtIM[i, :].sum() / N))
    del IM


    print(f"\nVerifying {net_type} oval21 with SparseImageStar in COO format")
    for i, eps_ in enumerate(epsilon):
        print(f"Verifying netowrk with epsilon = {eps_}")
        for j in range(N):
            img, label = cifar_test[rand_ind[i]]
            # infinity norm attack and convert image to channel last (i.e. [H, W, C])
            lb = normalizer((img - eps_).clamp(0, 1)).permute(1, 2, 0).numpy().copy()
            ub = normalizer((img + eps_).clamp(0, 1)).permute(1, 2, 0).numpy().copy()
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
    print(tabulate(data, headers=headers))
    print()

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(path+f"/{net_type}_oval21_infNormAttack_Table_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    data = [vtIM_table, vtCSR_table, vtCOO_table]
    print(tabulate(data, headers=headers))
    print()

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(path+f"/{net_type}_oval21_infNormAttack_Table_vt.tex", "w") as f:
        print(Tlatex, file=f)
        
    save_file = path + f"/{net_type}_oval21_infNormAttack_results.pkl"
    pickle.dump([rbIM, vtIM, rbIM_table, vtIM_table, rbCSR, vtCSR, rbCSR_table, vtCSR_table, rbCOO, vtCOO, rbCOO_table, vtCOO_table], open(save_file, "wb"))

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def verify_oval21_network_for_saving(net_type='base', dtype='float32'):
    
    print('=================================================================================')
    print(f"Verification of {net_type} OVAL21 Network against Infinity Norm Attack")
    print('=================================================================================\n')

    folder_dir = f"./SparseImageStar_evaluation/vnncomp2021/oval21/onnx"
    net_dir = f"{folder_dir}/cifar_{net_type}_kw.onnx"
    starvNet = load_oval21(net_dir, net_type, dtype=dtype)
    print()
    print(starvNet.info())
    
    # loading cifar10 dataset
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    cifar_test = datasets.CIFAR10('./cifardata/', train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor()]))
    
    N = 100 # number of images
    E = 5 # number of epsilons
    
    # select pseudo-random images
    rand_ind = np.random.RandomState(1000).randint(1, len(cifar_test), size=N)
    
    if net_type == 'base':
        epsilon = (np.arange(E)+1)*1/255
    # elif net_type == 'deep':
    #     epsilon = (np.arange(E)+1)*0.75/255
    else:
        epsilon = (np.arange(E)+1)*0.5/255

    IML = []
    CSRL = []
    COOL = []
    labelL = []

    for i, eps_ in enumerate(epsilon):
        
        IM_j = []
        CSR_j = []
        COO_j = []
        label_j = []

        for j in range(N):
            img, label = cifar_test[rand_ind[i]]
            # infinity norm attack and convert image to channel last (i.e. [H, W, C])
            lb = normalizer((img - eps_).clamp(0, 1)).permute(1, 2, 0).numpy().copy()
            ub = normalizer((img + eps_).clamp(0, 1)).permute(1, 2, 0).numpy().copy()
            
            IM_j.append(ImageStar(lb, ub))
            CSR_j.append(SparseImageStar2DCSR(lb, ub))
            COO_j.append(SparseImageStar2DCOO(lb, ub))
            label_j.append(label)

        IML.append(IM_j)
        CSRL.append(CSR_j)
        COOL.append(COO_j)
        labelL.append(label_j)

    rbIM = np.zeros([E, N])
    vtIM = np.zeros([E, N])
    rbCSR = np.zeros([E, N])
    vtCSR = np.zeros([E, N])
    rbCOO = np.zeros([E, N])
    vtCOO = np.zeros([E, N])

    print(f"Verifying {net_type} oval21 with ImageStar")
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

    print(f"\nVerifying {net_type} oval21 with Sparse Image Star in CSR format")
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
            
    print(f"\nVerifying {net_type} oval21 with Sparse Image Star in COO format")
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

    headers = [f"", f"delta={epsilon[0]}", f"delta={epsilon[1]}", f"delta={epsilon[2]}"]
    # Robustness Resluts
    data = [rbIM_table, rbCSR_table, rbCOO_table]
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate(data, headers=headers))
    print()

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(path+f"/cifar_{net_type}_kw_oval21_infNormAttack_Table_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    data = [vtIM_table, vtCSR_table, vtCOO_table]
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate(data, headers=headers))
    print()

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(path+f"/cifar_{net_type}_kw_oval21_infNormAttack_Table_vt.tex", "w") as f:
        print(Tlatex, file=f)
        
    save_file = path + f"/cifar_{net_type}_kw_oval21_infNormAttack_results.pkl"
    pickle.dump([rbIM, vtIM, rbIM_table, vtIM_table, rbCSR, vtCSR, rbCSR_table, vtCSR_table, rbCOO, vtCOO, rbCOO_table, vtCOO_table], open(save_file, "wb"))

    print('=====================================================')
    print('DONE!')
    print('=====================================================')

if __name__ == "__main__":
    verify_oval21_network(net_type='base', dtype='float32')
    # verify_oval21_network(net_type='deep', dtype='float32')
    # verify_oval21_network(net_type='wide', dtype='float32')