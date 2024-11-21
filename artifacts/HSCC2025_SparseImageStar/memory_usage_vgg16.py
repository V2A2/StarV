import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle
import scipy
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

    print('=================================================================================')
    print(f"Memory Usage of VGG16")
    print('=================================================================================\n')

    dtype = 'float64'
    
    folder_dir = 'StarV/util/data/nets/vggnet16'
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())

    mat_file = scipy.io.loadmat(f"{folder_dir}/nnv/nnv_vgg16_memory_usage.mat")
    nnv_nb = mat_file['memory_usage'].ravel()
    nnv_time = mat_file['reach_time'].ravel()

    shape = (3, 224, 224)

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
    nPred = [CSR.num_pred]
    density = [CSR.density()]
    
    for i in range(starvNet.n_layers):
        start = time.perf_counter()
        IM = starvNet.layers[i].reach(IM, method='approx', show=False)
        IM_time.append(time.perf_counter() - start)
        IM_nb.append(IM.nbytes())
        IM_shape.append(IM.V.shape)
    del IM
    
    for i in range(starvNet.n_layers):
        start = time.perf_counter()
        CSR = starvNet.layers[i].reach(CSR, method='approx', show=False)
        CSR_time.append(time.perf_counter() - start)
        CSR_nb.append(CSR.nbytes())
        nPred.append(CSR.num_pred)
        CSR_shape.append(CSR.shape)
        density.append(CSR.density())
    del CSR
        
    for i in range(starvNet.n_layers):
        start = time.perf_counter()
        COO = starvNet.layers[i].reach(COO, method='approx', show=False)
        COO_time.append(time.perf_counter() - start)
        COO_nb.append(COO.nbytes())
        COO_shape.append(COO.shape)
    del COO


    # save verification results
    path = f"artifacts/HSCC2025_SparseImageStar/results"
    if not os.path.exists(path):
        os.makedirs(path)

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
    plt.title("Computation Time", fontsize=14)
    plt.plot(x, IM_time, color='red', linewidth=2)
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

    plt.savefig(f'{path}/Figure_4b_computation_time_vgg16_spec_{spec}.png')
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
    ax2.legend(['density'], loc='upper center', fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)
    
    plt.savefig(f'{path}/Figure_4a_memory_usage_vgg16_spec_{spec}.png')
    # plt.show()
    plt.close()

    save_file = path + f"/memory_usage_vgg16_results_spec_{spec}.pkl"
    pickle.dump([IM_time, COO_time, CSR_time, IM_nb, COO_nb, CSR_nb, \
                 IM_shape, COO_shape, CSR_nb, nPred, density], open(save_file, "wb"))
    

def memory_usage_vgg16_spec_cn(spec):
    dtype = 'float64'

    folder_dir = 'StarV/util/data/nets/vggnet16'
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())

    shape = (3, 224, 224)

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

    COO = SparseImageStar2DCOO(lb, ub)
    CSR = SparseImageStar2DCSR(lb, ub)
    
    COO_time = []; CSR_time = []; 
    COO_nb = [COO.nbytes()]; CSR_nb = [CSR.nbytes()]
    COO_shape = [COO.shape + (COO.num_pred, )]; CSR_shape = [CSR.shape + (CSR.num_pred, )]
    nPred = [CSR.num_pred]
    density = [CSR.density()]
    
    for i in range(starvNet.n_layers):
        start = time.perf_counter()
        CSR = starvNet.layers[i].reach(CSR, method='approx', show=False)
        CSR_time.append(time.perf_counter() - start)
        CSR_nb.append(CSR.nbytes())
        nPred.append(CSR.num_pred)
        CSR_shape.append(CSR.shape)
        density.append(CSR.density())
    del CSR
        
    for i in range(starvNet.n_layers):
        start = time.perf_counter()
        COO = starvNet.layers[i].reach(COO, method='approx', show=False)
        COO_time.append(time.perf_counter() - start)
        COO_nb.append(COO.nbytes())
        COO_shape.append(COO.shape)
    del COO


    # save verification results
    path = f"artifacts/HSCC2025_SparseImageStar/results"
    if not os.path.exists(path):
        os.makedirs(path)

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
    plt.title("Computation Time", fontsize=14)
    plt.plot(x, COO_time, color='black', linewidth=2)
    plt.plot(x, CSR_time, color="magenta", linewidth=2)
    plt.xlabel("Layers", fontsize=12)
    plt.ylabel("Computation Time (sec)", fontsize=12)

    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks_labels, rotation=80, fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    # set legend
    ax.legend(['SIM COO', 'SIM CSR'], fontsize=12)

    plt.savefig(f'{path}/Figure_6b_computation_time_vgg16_spec_{spec}.png')
    # plt.show()
    plt.close()


    x = np.arange(len(CSR_nb))
    x_ticks_labels.insert(0, 'Input')

    plt.rcParams["figure.figsize"] = [8.50, 5.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1,1) 
    plt.title("Memory Usage", fontsize=14)
    plt.plot(x, COO_nb, color='black', linewidth=2)
    plt.plot(x, CSR_nb, color="magenta", linewidth=2)
    plt.xlabel("Layers", fontsize=12)
    plt.ylabel("Bytes", fontsize=12)

    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks_labels, rotation=80, fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    # set legend
    ax.legend(['SIM COO', 'SIM CSR'], loc='center right', fontsize=12)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.plot(x, density, color="green", linewidth=2)
    ax2.legend(['density'], fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)

    plt.savefig(f'{path}/Figure_6a_memory_usage_vgg16_spec_c{spec}.png')
    # plt.show()
    plt.close()

    save_file = path + f"/memory_usage_vgg16_spec_c{spec}_results.pkl"
    pickle.dump([COO_time, CSR_time, COO_nb, CSR_nb, \
                 COO_shape, CSR_nb, nPred, density], open(save_file, "wb"))


if __name__ == "__main__":
    memory_usage_vgg16(spec=11)
    memory_usage_vgg16_spec_cn(spec=4)
    