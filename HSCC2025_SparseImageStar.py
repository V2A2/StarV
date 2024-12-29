"""
HSCC2025: Memory-Efficient Verification for
          Deep Convolutional Neural Networks using SparseImageStar
Evaluation

Author: Anomynous
Date: 12/25/2024
"""

import time
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import scipy

from tabulate import tabulate
from StarV.util.load import *
from StarV.util.vnnlib import *
from StarV.util.attack import brightening_attack
from StarV.verifier.certifier import certifyRobustness
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR


# normalizer for oval21
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text)]

def memory_usage_oval21():
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

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


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

    print('=====================================================')
    print('DONE!')
    print('=====================================================')



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

    vnnlib_dir = f"{folder_dir}/vnnlib/spec_cn"
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

    print('=====================================================')
    print('DONE!')
    print('=====================================================')



def verify_convnet_network(net_type='Small', dtype='float32'):

    print('=================================================================================')
    print(f"Verification of CAV2020 {net_type} ConvNet Network against Brightnening Attack")
    print('=================================================================================\n')

    folder_dir = f"StarV/util/data/nets/CAV2020_MNIST_ConvNet"
    net_dir = f"{folder_dir}/onnx/{net_type}_ConvNet.onnx"
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
    path = f"artifacts/HSCC2025_SparseImageStar/results"
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


def plot_table_covnet_network(net_type):
    assert net_type in ['Small', 'Medium', 'Large'], \
    f"There are 3 types of ConvNet networks: /'Small/', /'Medium/', and /'Large/'"

    mat_path = f"StarV/util/data/nets/CAV2020_MNIST_ConvNet/nnv/"
    mat_file = scipy.io.loadmat(mat_path + f"NNV_{net_type}_ConvNet_Results_brightAttack.mat")
    rbNNV = mat_file['r_star']
    vtNNV = mat_file['VT_star']

    delta = [0.005, 0.01, 0.015]
    d = [250, 245, 240]
    N = 100

    dir = f"artifacts/HSCC2025_SparseImageStar/results/"
    if net_type == 'Small':
        sdir = dir + f'Table_1__{net_type}_MNIST_ConvNet_brightAttack'
    elif net_type == 'Medium':
        sdir = dir + f'Table_2__{net_type}_MNIST_ConvNet_brightAttack'
    else:
        sdir = dir + f'Table_3__{net_type}_MNIST_ConvNet_brightAttack'
    dir += f'{net_type}ConvNet_brightAttack'

    result_dir = dir + '_results.pkl'
    save_dir = sdir + '.tex'

    with open(result_dir, 'rb') as f:
        [rbIM, vtIM, rbIM_table, vtIM_table, rbCSR, vtCSR, rbCSR_table, vtCSR_table, rbCOO, vtCOO, rbCOO_table, vtCOO_table] = pickle.load(f)

    file = open(save_dir, "w")
    L = [
    f"\\begin{{table}}[!]\n",
    f"\\scriptsize\n",
    f"\\centering\n",
    f"\\begin{{tabular}}{{c||c:c:c||c:c:c}}\n",
    f"      & \\multicolumn{{3}}{{c||}}{{Robustness results ($\%$)}} & \multicolumn{{3}}{{c}}{{Verification time (sec)}} \\\\\n",
    f"\\hline\n \n"
    ]
    file.writelines(L)

    for i in range(len(d)):
        file.write(f"\\hline\n")
        line = f"$d = {d[i]}$"
        for j in delta:
            line += f" & $\\delta = {j}$"
        for j in delta:
            line += f" & $\\delta = {j}$"
        line += f" \\\\\n \\hline\n"
        file.write(line)
        file.write(f'IM & {rbIM_table[i][1]} & {rbIM_table[i][2]} &  {rbIM_table[i][3]} & {vtIM_table[i][1] :.3f} & {vtIM_table[i][2] :.3f} &  {vtIM_table[i][3] :.3f} \\\\ \\hline\n')
        file.write(f'SIM\\_csr & {rbCSR_table[i][1]} & {rbCSR_table[i][2]} &  {rbCSR_table[i][3]} & {vtCSR_table[i][1] :.3f} & {vtCSR_table[i][2] :.3f} &  {vtCSR_table[i][3] :.3f} \\\\ \\hline\n')
        file.write(f'SIM\\_coo & {rbCOO_table[i][1]} & {rbCOO_table[i][2]} &  {rbCOO_table[i][3]} & {vtCOO_table[i][1] :.3f} & {vtCOO_table[i][2] :.3f}  &  {vtCOO_table[i][3] :.3f} \\\\ \\hline\n')
        file.write(f'NNV & {int(rbNNV[i, 0]*100)} & {int(rbNNV[i, 1]*100)} & {int(rbNNV[i, 2]*100)} & {vtNNV[i, 0]/100 :.3f} & {vtNNV[i, 1]/100 :.3f} & {vtNNV[i, 2]/100 :.3f} \\\\ \\hline\n')

        file.write(f"\\hline\n\n")

    L = [
    f"\\end{{tabular}}\n",
    f"\\caption{{Verification results of the {net_type} MNIST CNN (CAV2020).}}\n",
    f"\\label{{tab:CAV2020_convNet_{net_type}}}\n",
    f"\\end{{table}}",
    ]
    file.writelines(L)
    file.close()

    print('=====================================================')
    print('DONE!')
    print('=====================================================')



def verify_vgg16_network(dtype='float64'):

    print('=================================================================================')
    print(f"Verification of VGG16 Network against Infinity Norm Attack")
    print('=================================================================================\n')

    folder_dir = f"StarV/util/data/nets/vggnet16"
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
    numPred = np.zeros(N)

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

        if num_attack_pixel >= 148406:
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

    rb_table.append((rbCSR == 1).sum())
    vt_table.append((vtCSR.sum() / N))
    del CSR, Y

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

        if num_attack_pixel >= 148406:
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
    path = f"artifacts/HSCC2025_SparseImageStar/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_results.pkl"
    pickle.dump([rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, numPred], open(save_file, "wb"))

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

    folder_dir = f"StarV/util/data/nets/vggnet16"
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
    path = f"artifacts/HSCC2025_SparseImageStar/results"
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


def verify_vgg16_network_spec_cn(dtype='float64'):

    print('=================================================================================')
    print(f"Verification of VGG16 Network against Infinity Norm Attack Spec_cn")
    print('=================================================================================\n')

    folder_dir = f"StarV/util/data/nets/vggnet16"
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())


    shape = (3, 224, 224)

    vnnlib_dir = f"{folder_dir}/vnnlib/spec_cn"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)

    # save verification results
    path = f"artifacts/HSCC2025_SparseImageStar/results"
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


def plot_table_vgg16_network():
    folder_dir = f"artifacts/HSCC2025_SparseImageStar/results"
    file_dir = folder_dir + 'vggnet16_vnncomp23_results.pkl'
    with open(file_dir, 'rb') as f:
        rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, num_pred = pickle.load(f)
    file_dir = folder_dir + 'vggnet16_vnncomp23_converted_results.pkl'
    with open(file_dir, 'rb') as f:
        rbIMc, vtIMc = pickle.load(f)

    nnv_dir = 'StarV/util/data/nets/vggnet16/nnv'
    mat_file = scipy.io.loadmat(f"{nnv_dir}/nnv_vggnet16_results.mat")
    rbNNV = mat_file['rb_im'].ravel()
    vtNNV = mat_file['vt_im'].ravel()

    mat_file = scipy.io.loadmat(f"{nnv_dir}/nnv_vggnet16_converted_results.mat")
    rbNNVc = mat_file['rb_im'].ravel()
    vtNNVc = mat_file['vt_im'].ravel()

    file_dir = folder_dir + 'vggnet16_vnncomp23_spec_cn_results.pkl'
    with open(file_dir, 'rb') as f:
        numPred_cn, rbCSR_cn, vtCSR_cn, rbCOO_cn, vtCOO_cn = pickle.load(f)

    f_dir = f"StarV/util/data/nets/vggnet16"
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
    vt_marabou = 'T/O'
    vt_bcrown = [7.355725526809692, 8.868661165237427, 8.908552885055542, 9.075981855392456, 8.986030578613281, 8.999144315719604, 8.916476249694824, 9.294207572937012, 10.620023727416992, 9.017800092697144, 9.108751058578491, 9.2491958141326, 594.9671733379364, 17.784186124801636, 34.14556264877319]
    vt_bcrown = np.array(vt_bcrown, dtype='float64')
    vt_abcrown = [302.8435814380646, 243.49199199676514, 174.6395332813263, 622.3142883777618, 430.933091878891, 622.221896648407, 664.8663415908813, 709.2889895439148, 708.833279132843, 893.600474357605, 897.9993720054626, 860.9506402015686, 945.6725194454193, 1077.005056142807, 1191.9225597381592]

    headers = ['Specs', 'e', 'Result', 'm', 'IM', 'SIM_csr', 'SIM_coo', 'NNV', 'DeepPoly',  'Marabou', 'IM', 'NNV', 'NNENUM', 'ab-CROWN', 'b-CROWN']

    result = 'UNSAT'

    data = []
    for i in range(N):
        vt_im = 'O/M' if np.isnan(vtIM[i]) else f"{vtIM[i]:0.1f}"
        vt_imc = 'O/M' if np.isnan(vtIMc[i]) else f"{vtIMc[i]:0.1f}"
        vt_nnv = 'O/M' if vtNNV[i] < 0 else f"{vtNNV[i]:0.1f}"
        vt_nnvc = 'O/M' if vtNNVc[i] < 0 else f"{vtNNVc[i]:0.1f}"

        nPred = 'NA' if np.isnan(vtCSR[i]) else f"{num_pred[i]}"
        data.append([i, num_attack_pixel[i], result, nPred,  vt_im, f"{vtCSR[i]:0.1f}", f"{vtCOO[i]:0.1f}", vt_nnv, vt_DP, vt_marabou, vt_imc, vt_nnvc, vt_NNENUM[i], f"{vt_abcrown[i]:0.1f}", f"{vt_bcrown[i]:0.1f}"])

    num_attack_pixel_cn = [200, 300, 400, 500, 1000, 2000, 3000]
    N_cn = len(numPred_cn)
    vt_NNENUM_cn = [744.02, 1060.96, 1354.75, 1781.26, 'T/O', 'T/O', 'O/M']
    vt_bcrown_cn = [26782.327130317688, 37052.68477010727, 'T/O', 'T/O', 'T/O', 'T/O', 'T/O']
    vt_abcrown_cn = 'T/O'
    for i in range(N_cn):
        vt_im = 'O/M'
        vt_imc = 'O/M'
        vt_nnv = 'O/M'
        vt_nnvc = 'O/M'
        vt_bcrown_cd = vt_bcrown_cn[i] if vt_bcrown_cn[i] == 'T/O' else f"{np.array(vt_bcrown_cn[i], dtype='float64'):0.1f}"

        nPred = 'NA' if np.isnan(vtCSR_cn[i]) else f"{numPred_cn[i]}"
        data.append([f"c_{i}", num_attack_pixel_cn[i], result, nPred,  vt_im, f"{vtCSR_cn[i]:0.1f}", f"{vtCOO_cn[i]:0.1f}", vt_nnv, vt_DP, vt_marabou, vt_imc, vt_nnvc, vt_NNENUM_cn[i], vt_abcrown_cn, vt_bcrown_cd])

    print(tabulate(data, headers=headers))

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(folder_dir+f"Table_4_vggnet16_vnncomp23_results.tex", "w") as f:
        print(Tlatex, file=f)

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


if __name__ == "__main__":

    # Table 1: Verification results of the Small MNIST CNN (CAV2020)
    verify_convnet_network(net_type='Small', dtype='float64')
    plot_table_covnet_network(net_type = 'Small')                       # Table 1

    # Table 2: Verification results of the Medium MNIST CNN (CAV2020)
    verify_convnet_network(net_type='Medium', dtype='float64')
    plot_table_covnet_network(net_type = 'Medium')                      # Table 2

    # Table 3: Verification results of the Large MNIST CNN (CAV2020)
    verify_convnet_network(net_type='Large', dtype='float64')
    plot_table_covnet_network(net_type = 'Large')                       # Table 3

    # Table 4: Verification results of VGG16 in seconds (vnncomp2023)
    verify_vgg16_network(dtype='float64')
    verify_vgg16_converted_network(dtype='float64')
    verify_vgg16_network_spec_cn()
    plot_table_vgg16_network()                                          # Table 4

    # Figure 4: Memory usage and computation time comparison between ImageStar and
    # SparseImageStar (SIM) in verifying the vggnet16 network (vnncomp2023) with spec 11 image
    memory_usage_vgg16(spec=11)                                         # Figure 4

    # Figure 5: Memory usage and computation time comparison between ImageStar and
    # SparseImageStar (SIM) in verifying the oval21 network with 𝑙∞ norm attack on all pixels.
    memory_usage_oval21()                                               # Figure 5

    # Figure 6: Memory usage and computation time comparison between ImageStar and
    # SparseImageStar (SIM) in verifying the vggnet16 network (vnncomp2023) with spec c4 image
    memory_usage_vgg16_spec_cn(spec=4)                                  # Figure 6
