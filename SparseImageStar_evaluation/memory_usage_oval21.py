import time
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
    cifar_test = datasets.CIFAR10('SparseImageStar_evaluation/cifardata/', train=False, download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))

    shape = (3, 32, 32)

    net_dir = 'SparseImageStar_evaluation/vnncomp2021/oval21/onnx/cifar_deep_kw.onnx'
    starv_net = load_neural_network_file(net_dir, net_type='oval21', dtype='float64', channel_last=False)
    print(starv_net.info())

    eps = 0.005
    img_num = 876 + 1
    img_py, label = cifar_test[img_num]

    img = img_py.type(torch.float64)
    lb = normalizer((img - eps).clamp(0, 1)).permute(1, 2, 0).numpy()
    ub = normalizer((img + eps).clamp(0, 1)).permute(1, 2, 0).numpy()

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


    x = np.arange(len(IM_time))
    x_ticks_labels = []
    for i in range(starv_net.n_layers):
        x_ticks_labels.append(f"{starv_net.layers[i].__class__.__name__}_{i}")

    fig, ax = plt.subplots(1,1) 
    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks_labels, rotation=80, fontsize=10)
    # set legend
    ax.legend(['ImageStar', 'SparseCOO', 'SparseCSR'])

    plt.title("Computation time (sec)")
    plt.rcParams["figure.figsize"] = [7.50, 5.50]
    plt.rcParams["figure.autolayout"] = True
    plt.plot(x, IM_time, color="red")
    plt.plot(x, COO_time, color='black')
    plt.plot(x, CSR_time, color="magenta")
    plt.savefig('SparseImageStar_evaluation//results/memory_usage_oval21_computation_time_differences.png')
    # plt.show()
    plt.close()


    
    x = np.arange(len(IM_nb))
    x_ticks_labels = ['Input']
    for i in range(starv_net.n_layers):
        x_ticks_labels.append(f"{starv_net.layers[i].__class__.__name__}_{i}")

    fig, ax = plt.subplots(1,1) 
    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks_labels, rotation=80, fontsize=10)
    # set legend
    ax.legend(['ImageStar', 'SIM COO', 'SIM CSR'])

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.plot(x, density, color="green")
    ax2.legend(['density'])
    ax2.set_ylabel(r"Density")

    plt.title("Memory Usage")
    plt.rcParams["figure.figsize"] = [7.50, 5.50]
    plt.rcParams["figure.autolayout"] = True
    plt.plot(x, IM_nb, color="red")
    plt.plot(x, COO_nb, color='black')
    plt.plot(x, CSR_nb, color="magenta")
    plt.xlabel("Layers")
    plt.ylabel("Bytes")
    
    plt.savefig('SparseImageStar_evaluation//results/memory_usage_oval21_memory_usage_differences.png')
    # plt.show()
    plt.close()