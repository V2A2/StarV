import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle
import re
import time

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

def memory_usage_vgg16(spec):
    dtype = 'float64'

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

    vnnlib_file = vnnlib_files[spec]
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

    IM = ImageStar(lb, ub)
    COO = SparseImageStar2DCOO(lb, ub)
    CSR = SparseImageStar2DCSR(lb, ub)
    
    IM_time = []; COO_time = []; CSR_time = []; 
    IM_nb = [IM.nbytes()]; COO_nb = [COO.nbytes()]; CSR_nb = [CSR.nbytes()]
    IM_shape = [IM.V.shape]; COO_shape = [COO.shape + (COO.num_pred, )]; CSR_shape = [CSR.shape + (CSR.num_pred, )]
    density = [CSR.density()]
    
    for i in range(starvNet.n_layers):
        start = time.perf_counter()
        IM = starvNet.layers[i].reach(IM, method='approx', show=False)
        IM_time.append(time.perf_counter() - start)
        IM_nb.append(IM.nbytes())
    del IM
    
    for i in range(starvNet.n_layers):
        start = time.perf_counter()
        CSR = starvNet.layers[i].reach(CSR, method='approx', show=False)
        CSR_time.append(time.perf_counter() - start)
        CSR_nb.append(CSR.nbytes())
        density.append(CSR.density())
    del CSR
        
    for i in range(starvNet.n_layers):
        start = time.perf_counter()
        COO = starvNet.layers[i].reach(COO, method='approx', show=False)
        COO_time.append(time.perf_counter() - start)
        COO_nb.append(COO.nbytes())
    del COO

    x = np.arange(len(CSR_time))
    x_ticks_labels = []
    for i in range(starvNet.n_layers):
        if starvNet.layers[i].__class__.__name__ == 'Conv2DLayer':
            l_name = '$L_c$'
        elif starvNet.layers[i].__class__.__name__ == 'ReLULayer':
            l_name = '$L_r$'
        elif starvNet.layers[i].__class__.__name__ == 'FlattenLayer':
            l_name = '$L_{{flat}}$'
        elif starvNet.layers[i].__class__.__name__ == 'FullyConnectedLayer':
            l_name = '$L_f$'
        elif starvNet.layers[i].__class__.__name__ == 'MaxPool2DLayer':
            l_name = '$L_m$'
        elif starvNet.layers[i].__class__.__name__ == 'BatchNorm2DLayer':
            l_name = '$L_b$'
        else:
            raise Exception('Unknown layer')
        x_ticks_labels.append(f"{l_name}_{i}")

    plt.rcParams["figure.figsize"] = [8.50, 5.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1,1) 
    plt.title("Computation Time")
    plt.plot(x, IM_time, color='red')
    plt.plot(x, COO_time, color='black')
    plt.plot(x, CSR_time, color="magenta")
    plt.xlabel("Layers")
    plt.ylabel("Computation Time (sec)")

    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks_labels, rotation=80, fontsize=10)
    # set legend
    ax.legend(['ImageStar', 'SparseCOO', 'SparseCSR'])

    plt.savefig('SparseImageStar_evaluation//results/memory_usage_vgg16_computation_time_differences.png')
    # plt.show()
    plt.close()



    x = np.arange(len(IM_nb))
    x_ticks_labels.insert(0, 'Input')

    plt.rcParams["figure.figsize"] = [8.50, 5.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1,1) 
    plt.title("Memory Usage")
    plt.plot(x, IM_nb, color="red")
    plt.plot(x, COO_nb, color='black')
    plt.plot(x, CSR_nb, color="magenta")
    plt.xlabel("Layers")
    plt.ylabel("Bytes")

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

    plt.savefig('SparseImageStar_evaluation//results/memory_usage_vgg16_memory_usage_differences.png')
    # plt.show()
    plt.close()

if __name__ == "__main__":
    memory_usage_vgg16(spec=11)
    