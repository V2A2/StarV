import time
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import scipy

import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from StarV.util.load import *
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])

if __name__ == "__main__":

    print('=================================================================================')
    print(f"Memory Usage of OVAL21")
    print('=================================================================================\n')

    data_path = 'StarV/util/data/nets/'
    cifar_test = datasets.CIFAR10(data_path+'cifardata/', train=False, download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))

    shape = (3, 32, 32)
    dtype = 'float64'

    net_dir = data_path + 'oval21/onnx/cifar_deep_kw.onnx'
    starv_net = load_neural_network_file(net_dir, net_type='oval21', dtype=dtype, channel_last=False)
    print(starv_net.info())

    mat_file = scipy.io.loadmat(data_path+"oval21/nnv/nnv_oval21_memory_usage.mat")
    nnv_nb = mat_file['memory_usage'].ravel()
    nnv_time = mat_file['reach_time'].ravel()

    eps = 0.005
    img_num = 876 + 1
    img_py, label = cifar_test[img_num]

    img = img_py.type(torch.float64)
    lb = normalizer((img - eps).clamp(0, 1)).permute(1, 2, 0).numpy().astype(dtype)
    ub = normalizer((img + eps).clamp(0, 1)).permute(1, 2, 0).numpy().astype(dtype)

    IM = ImageStar(lb, ub)
    COO = SparseImageStar2DCOO(lb, ub)
    CSR = SparseImageStar2DCSR(lb, ub)

    IM_time = []; COO_time = []; CSR_time = []; 
    IM_nb = [IM.nbytes()]; COO_nb = [COO.nbytes()]; CSR_nb = [CSR.nbytes()]
    density = [CSR.density()]

    print('working on ImageStar')
    for i in range(starv_net.n_layers):
        start = time.perf_counter()
        IM = starv_net.layers[i].reach(IM, method='approx', show=False)
        IM_time.append(time.perf_counter() - start)
        IM_nb.append(IM.nbytes())

    print('working on SparseImageStar CSR')
    for i in range(starv_net.n_layers):
        start = time.perf_counter()
        CSR = starv_net.layers[i].reach(CSR, method='approx', show=False)
        CSR_time.append(time.perf_counter() - start)
        CSR_nb.append(CSR.nbytes())
        density.append(CSR.density())
        
    print('working on SparseImageStar COO')
    for i in range(starv_net.n_layers):
        start = time.perf_counter()
        COO = starv_net.layers[i].reach(COO, method='approx', show=False)
        COO_time.append(time.perf_counter() - start)
        COO_nb.append(COO.nbytes())

    # save verification results
    path = f"artifacts/HSCC2025_SparseImageStar/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/memory_usage_oval21_results.pkl"
    pickle.dump([IM_nb, CSR_nb, COO_nb, COO_time, IM_time, CSR_time, COO_time, density], open(save_file, "wb"))

    x = np.arange(len(IM_time))
    x_ticks_labels = []
    for i in range(starv_net.n_layers):
        if starv_net.layers[i].__class__.__name__ == 'Conv2DLayer':
            l_name = '$L_c$'
        elif starv_net.layers[i].__class__.__name__ == 'ReLULayer':
            l_name = '$L_r$'
        elif starv_net.layers[i].__class__.__name__ == 'FlattenLayer':
            l_name = '$L_{{flat}}$'
        elif starv_net.layers[i].__class__.__name__ == 'FullyConnectedLayer':
            l_name = '$L_f$'
        x_ticks_labels.append(f"{l_name}_{i}")

    plt.rcParams["figure.figsize"] = [8.50, 5.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1,1)
    plt.title("Computation Time", fontsize=14)
    plt.plot(x, IM_time, color="red", linewidth=2)
    plt.plot(x, COO_time, color='black', linewidth=2)
    plt.plot(x, CSR_time, color="magenta", linewidth=2)
    plt.plot(x, nnv_time[1:], color='blue', linewidth=2)
    plt.xlabel("Layers", fontsize=12)
    plt.ylabel("Computation Time (sec)", fontsize=12)

    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks_labels, rotation=80, fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    # set legend
    ax.legend(['ImageStar', 'SIM COO', 'SIM CSR', 'NNV'], fontsize=12)

    plt.savefig(f'{path}/Figure_5b_computation_time_oval21.png')
    # plt.show()
    plt.close()


    
    x = np.arange(len(IM_nb))
    x_ticks_labels.insert(0, 'Input')

    plt.rcParams["figure.figsize"] = [8.50, 5.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1,1)
    plt.title("Memory Usage", fontsize=14)
    plt.plot(x, IM_nb, color="red", linewidth=2)
    plt.plot(x, COO_nb, color='black', linewidth=2)
    plt.plot(x, CSR_nb, color="magenta", linewidth=2)
    plt.plot(x, nnv_nb, color='blue', linewidth=2)
    plt.xlabel("Layers", fontsize=12)
    plt.ylabel("Bytes", fontsize=12)

    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks_labels, rotation=80, fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    # set legend
    ax.legend(['ImageStar', 'SIM COO', 'SIM CSR', 'NNV'], loc='center right', fontsize=12)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.plot(x, density, color="green", linewidth=2)
    ax2.legend(['density'], fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)

    plt.savefig(f'{path}/Figure_5a_memory_usage_oval21.png')
    # plt.show()
    plt.close()