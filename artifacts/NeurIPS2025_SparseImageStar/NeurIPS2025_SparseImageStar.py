"""
NeurIPS 2025: Memory-Efficient Verification for Deep Convolutional Neural Networks using SparseImageStar
Evaluation

Author: Anomynous
Date: 02/16/2025
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

artifact = 'NeurIPS2025_SparseImageStar'
RESULT_dir = f'artifacts/{artifact}'
STORE_dir = f'artifacts/{artifact}/results'

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

    with open(data_path + 'oval21/nnenum/oval21_memory_usage_nnenum_lp_star.pkl', 'rb') as f:
        nnenum_lpstar = pickle.load(f)
    nb = np.array(nnenum_lpstar[1]['nbytes'])
    vt = np.array(nnenum_lpstar[2]['vt'])
    nnenum_lpstar_nb = np.zeros(13)
    nnenum_lpstar_nb[:9] = nb[:9]
    nnenum_lpstar_nb[9] = nb[8] #nnenum skips flatten layer
    nnenum_lpstar_nb[10:] = nb[9:]
    nnenum_lpstar_time = np.zeros(12)
    nnenum_lpstar_time[:8] = vt[:8]
    nnenum_lpstar_time[9:] = vt[8:]

    with open(data_path + 'oval21/nnenum/oval21_memory_usage_nnenum_zono.pkl', 'rb') as f:
        nnenum_zono = pickle.load(f)
    nb = np.array(nnenum_zono[1]['nbytes'])
    vt = np.array(nnenum_zono[2]['vt'])
    nnenum_zono_nb = np.zeros(13)
    nnenum_zono_nb[:9] = nb[:9]
    nnenum_zono_nb[9] = nb[8] #nnenum skips flatten layer
    nnenum_zono_nb[10:] = nb[9:]
    nnenum_zono_time = np.zeros(12)
    nnenum_zono_time[:8] = vt[:8]
    nnenum_zono_time[9:] = vt[8:]

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
    path = STORE_dir
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
    plt.plot(x, nnenum_zono_time, color='cyan', linewidth=2)
    plt.plot(x, nnenum_lpstar_time, color='yellow', linewidth=2)
    plt.xlabel("Layers", fontsize=12)
    plt.ylabel("Computation Time (sec)", fontsize=12)

    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks_labels, rotation=80, fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    # set legend
    ax.legend(['ImageStar', 'SIM COO', 'SIM CSR', 'NNV', 'NNENUM Zono', 'NNENUM LP'], fontsize=12)

    plt.savefig(f'{RESULT_dir}/Figure_5b_computation_time_oval21.png')
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
    plt.plot(x, nnenum_zono_nb, color='cyan', linewidth=2)
    plt.plot(x, nnenum_lpstar_nb, color='yellow', linewidth=2)
    plt.xlabel("Layers", fontsize=12)
    plt.ylabel("Bytes", fontsize=12)

    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks_labels, rotation=80, fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    # set legend
    ax.legend(['ImageStar', 'SIM COO', 'SIM CSR', 'NNV', 'NNENUM Zono', 'NNENUM LP'], loc='center right', fontsize=12)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.plot(x, density, color="green", linewidth=2)
    ax2.legend(['density'], fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)

    plt.savefig(f'{RESULT_dir}/Figure_5a_memory_usage_oval21.png')
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

    plt.savefig(f'{RESULT_dir}/computation_time_vgg16_spec_{spec}.png')
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

    plt.savefig(f'{RESULT_dir}/Figure_4_memory_usage_vgg16_spec_{spec}.png')
    # plt.show()
    plt.close()

    save_file = STORE_dir + f"/memory_usage_vgg16_results_spec_{spec}.pkl"
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

    COO_time = []; CSR_time = []
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

    plt.savefig(f'{RESULT_dir}/Figure_7b_computation_time_vgg16_spec_{spec}.png')
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

    plt.savefig(f'{RESULT_dir}/Figure_7a_memory_usage_vgg16_spec_c{spec}.png')
    # plt.show()
    plt.close()

    save_file = STORE_dir + f"/memory_usage_vgg16_spec_c{spec}_results.pkl"
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

    mat_file = scipy.io.loadmat(f"{folder_dir}/nnv/{net_type}_images.mat")
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
    path = STORE_dir
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


def plot_table_covnet_network_all():
    net_types = ['Small', 'Medium', 'Large']
    save_dir = f'{RESULT_dir}/Table_1__Verification_results_of_the_MNIST_CNN.tex'
    
    rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO = [], [], [], [], [], []
    for net_type in net_types:
        result_dir = STORE_dir + f'/{net_type}ConvNet_brightAttack_results.pkl'
        
        with open(result_dir, 'rb') as f:
            [_, _, rbim, vtim, _, _, rbcsr, vtcsr, _, _, rbcoo, vtcoo] = pickle.load(f)
        rbIM.append(rbim)
        vtIM.append(vtim)
        rbCSR.append(rbcsr)
        vtCSR.append(vtcsr)
        rbCOO.append(rbcoo)
        vtCOO.append(vtcoo)
    
    rbNNV, vtNNV = [], []
    mat_path = f"StarV/util/data/nets/CAV2020_MNIST_ConvNet/nnv/"
    for net_type in net_types:
        mat_file = scipy.io.loadmat(mat_path + f"NNV_{net_type}_ConvNet_Results_brightAttack.mat")
        rbNNV.append(mat_file['r_star'])
        vtNNV.append(mat_file['VT_star'])        

    delta = [0.005, 0.01, 0.015]
    d = [250, 245, 240]
    N = 100
    
    file = open(save_dir, "w")
    L = [
        r"\begin{table}[]" + '\n',
        r"\centering" + '\n',
        r"\resizebox{\columnwidth}{!}{" + '\n',
        r"\footnotesize" + '\n',
        r"\begin{tabular}{r||c||c:c:c||c:c:c||c:c:c ||c:c:c||c:c:c||c:c:c}" + '\n',
        r"      & & \multicolumn{9}{c||}{Robustness results ($\%$)}  & \multicolumn{9}{c}{Verification time (sec)} \\" + '\n',
        r"\hline" + '\n',
        r"      & & \multicolumn{3}{c||}{Small} & \multicolumn{3}{c||}{Medium} & \multicolumn{3}{c||}{Large} & " +
        r"\multicolumn{3}{c||}{Small} & \multicolumn{3}{c||}{Medium} & \multicolumn{3}{c}{Large} \\"  + '\n',
        r"      & & $\delta = 0.005$ & $\delta = 0.01$ & $\delta = 0.015$ & $\delta = 0.005$ & $\delta = 0.01$ & $\delta = 0.015$ & " +
        r"$\delta = 0.005$ & $\delta = 0.01$ & $\delta = 0.015$ & $\delta = 0.005$ & $\delta = 0.01$ & $\delta = 0.015$ & " +
        r"$\delta = 0.005$ & $\delta = 0.01$ & $\delta = 0.015$ & $\delta = 0.005$ & $\delta = 0.01$ & $\delta = 0.015$ \\" + '\n',
        r"\hline" + '\n',
    ]
    file.writelines(L)
    
    for i in range(len(d)):
        file.write(r"\hline" + '\n')
        line = r"\multirow{4}{*}{\rotatebox{90}{$d=" + f"{d[i]}" + r"$}}" + '\n'
        file.write(line)
        
        line = f'& IM'
        for j in range(len(net_types)):
            line += f' & {rbIM[j][i][1]} & {rbIM[j][i][2]} &  {rbIM[j][i][3]}'
        for j in range(len(net_types)):
            line += f' & {vtIM[j][i][1] :.3f} & {vtIM[j][i][2] :.3f} &  {vtIM[j][i][3] :.3f}'
        file.write(line + ' \\\\\n')
        
        line = f'& SIM\\_csr'
        for j in range(len(net_types)):
            line += f' & {rbCSR[j][i][1]} & {rbCSR[j][i][2]} &  {rbCSR[j][i][3]}'
        for j in range(len(net_types)):
            line += f' & {vtCSR[j][i][1] :.3f} & {vtCSR[j][i][2] :.3f} &  {vtCSR[j][i][3] :.3f}'
        file.write(line + ' \\\\\n')
        
        line = f'& SIM\\_coo'
        for j in range(len(net_types)):
            line += f' & {rbCOO[j][i][1]} & {rbCOO[j][i][2]} &  {rbCOO[j][i][3]}'
        for j in range(len(net_types)):
            line += f' & {vtCOO[j][i][1] :.3f} & {vtCOO[j][i][2] :.3f} &  {vtCOO[j][i][3] :.3f}'
        file.write(line + ' \\\\\n')
        
        line = f'& NNV'
        for j in range(len(net_types)):
            line += f' & {int(rbNNV[j][i, 0]*100)} & {int(rbNNV[j][i, 1]*100)} &  {int(rbNNV[j][i, 2]*100)}'
        for j in range(len(net_types)):
            line += f' & {vtNNV[j][i, 0]/100 :.3f} & {vtNNV[j][i, 1]/100 :.3f} &  {vtNNV[j][i, 2]/100 :.3f}'
        file.write(line + ' \\\\\n')
        
        file.write(r"\hline" + '\n')
            
    L = [
        r"\end{tabular}" + '\n',
        r"}" + '\n',
        r"\caption{Verification results of the MNIST CNN \cite{tran2020cav}.}" + '\n',
        r"\label{tab:CAV2020_mnist_convnet} + '\n'"
        r"\end{table} + '\n'"
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
    path = STORE_dir
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
    with open(path+f"/vggnet16_vnncomp23_results_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate([vt_table], headers=headers))
    print()

    Tlatex = tabulate([vt_table], headers=headers, tablefmt='latex')
    with open(path+f"/vggnet16_vnncomp23_results_vt.tex", "w") as f:
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
    path = STORE_dir
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
    with open(path+f"/vggnet16_vnncomp23_converted_results_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), vtIM], headers=headers))
    print()

    Tlatex = tabulate([np.arange(N), vtIM], headers=headers, tablefmt='latex')
    with open(path+f"/vggnet16_vnncomp23_converted_results_vt.tex", "w") as f:
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
    path = STORE_dir
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


def verify_vgg16_network_spec_cn_random(dtype='float64'):

    print('=================================================================================')
    print(f"Verification of VGG16 Network against Infinity Norm Attack Spec_cn new (random images)")
    print('=================================================================================\n')

    folder_dir = f"StarV/util/data/nets/vggnet16"
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())


    shape = (3, 224, 224)

    vnnlib_dir = f"{folder_dir}/vnnlib/spec_cn_random"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)

    ########
    vnnlib_files = [vnnlib_files[5]]

    # save verification results
    path = f"artifacts/{artifact}_SparseImageStar/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_spec_cn_random_results.pkl"

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
    print('=================================================================================')
    print('Table 2: Verification results of VGG16 in seconds (vnncomp2023)')
    print('=================================================================================\n')
    folder_dir = STORE_dir
    file_dir = folder_dir + '/vggnet16_vnncomp23_results.pkl'
    with open(file_dir, 'rb') as f:
        rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, num_pred = pickle.load(f)
    file_dir = folder_dir + '/vggnet16_vnncomp23_converted_results.pkl'
    with open(file_dir, 'rb') as f:
        rbIMc, vtIMc = pickle.load(f)

    nnv_dir = 'StarV/util/data/nets/vggnet16/nnv'
    mat_file = scipy.io.loadmat(f"{nnv_dir}/nnv_vggnet16_results.mat")
    rbNNV = mat_file['rb_im'].ravel()
    vtNNV = mat_file['vt_im'].ravel()

    mat_file = scipy.io.loadmat(f"{nnv_dir}/nnv_vggnet16_converted_results.mat")
    rbNNVc = mat_file['rb_im'].ravel()
    vtNNVc = mat_file['vt_im'].ravel()

    file_dir = folder_dir + '/vggnet16_vnncomp23_spec_cn_results.pkl'
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

    headers = ['Specs', 'k', 'Result', 'm', 'IM', 'SIM_csr', 'SIM_coo', 'NNV', 'DeepPoly',  'Marabou', 'IM', 'NNV', 'NNENUM', 'ab-CROWN', 'b-CROWN']

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
    with open(RESULT_dir+f"/Table_2_vggnet16_vnncomp23_results.tex", "w") as f:
        print(Tlatex, file=f)

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def plot_table_vgg16_network_random_images():
    print('=================================================================================')
    print('Table: Verification results of VGG16 with specifications generated with random images (results in seconds)')
    print('=================================================================================\n')
    folder_dir = STORE_dir
    file_dir = folder_dir + '/vggnet16_vnncomp23_spec_cn_random_results.pkl'
    with open(file_dir, 'rb') as f:
        numPred, rbCSR, vtCSR, rbCOO, vtCOO = pickle.load(f)

    # headers = ['Specs', 'k', 'Result', 'm', 'SIM_csr', 'SIM_coo', 'NNENUM', 'ab-CROWN', 'b-CROWN']
    headers = ['Specs', 'k', 'Result', 'm', 'SIM_csr', 'SIM_coo', 'NNENUM']
    result = 'UNSAT'

    num_attack_pixel = [200, 200, 400, 600, 1000, 2000, 4000, 6000]
    N = len(num_attack_pixel)

    vt_bcrown = 'N/A' #'T/O'
    vt_abcrown = 'N/A' #'T/O'

    vt_NNENUM = [849.65, 787.31, 1155.52, 1948.67, 'T/O', 'O/M', 'O/M', 'O/M'] #'T/O', 'O/M'

    data = []
    for i in range(N):
        nPred = 'NA' if np.isnan(vtCSR[i]) else f"{numPred[i]}"
        # data.append([f"cr_{i}", num_attack_pixel[i], result, nPred, f"{vtCSR[i]:0.1f}", f"{vtCOO[i]:0.1f}", vt_NNENUM[i], vt_abcrown, vt_bcrown])
        data.append([f"cr_{i}", num_attack_pixel[i], result, nPred, f"{vtCSR[i]:0.1f}", f"{vtCOO[i]:0.1f}", vt_NNENUM[i]])

    print(tabulate(data, headers=headers))

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(folder_dir+f"Table_XXX_vggnet16_vnncomp23_random_image_results_crn.tex", "w") as f:
        print(Tlatex, file=f)


def plot_table_vgg16_network_full_specification():
    print('=================================================================================')
    print('Table: Verification results of VGG16 with specifications from vnncomp2023, but modified to contain full output specifications (results in seconds)')
    print('=================================================================================\n')
    folder_dir = STORE_dir
    file_dir = folder_dir + '/vggnet16_vnncomp23_results.pkl'
    with open(file_dir, 'rb') as f:
        rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, num_pred = pickle.load(f)

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
    

    headers = ['Specs', 'k', 'Result', 'm', 'IM', 'SIM_csr', 'SIM_coo', 'NNENUM', 'ab-CROWN', 'b-CROWN']
    result = 'UNSAT'

    N = 15

    vt_NNENUM = [3.5, 3.4, 9.3, 4.8, 18.1, 35.7, 6.5, 18.3, 133.8, 10.6, 40.9, 57.6, 'T/O', 236.5, 746.6]
    vt_bcrown = [1221.888, 1417.046, 2619.361, 1432.293, 20225.768]
    vt_abcrown = ['T/O', 'T/O', 'T/O', 'T/O', 'T/O']

    attack_pixel_idxs = [0, 5, 8, 11, 14]

    data = []
    j = 0
    for i in attack_pixel_idxs:
        vt_im = 'O/M' if np.isnan(vtIM[i]) else f"{vtIM[i]:0.1f}"
        nPred = 'NA' if np.isnan(vtCSR[i]) else f"{num_pred[i]}"
        data.append([f"{i}", num_attack_pixel[i], result, nPred, vt_im, f"{vtCSR[i]:0.1f}", f"{vtCOO[i]:0.1f}", vt_NNENUM[i], vt_abcrown[j], vt_bcrown[j]])
        j += 1

    print(tabulate(data, headers=headers))

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(folder_dir+f"Table_XXX_vggnet16_vnncomp23_random_image_results_crn.tex", "w") as f:
        print(Tlatex, file=f)


def worst_case_vgg16():
    print('=================================================================================')
    print(f"Worst Case Memory Usage of VGG16")
    print('=================================================================================\n')

    dtype = 'float64'

    folder_dir = 'StarV/util/data/nets/vggnet16'
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())

    a = np.random.rand(224, 224, 3, 1)

    n_pred = np.prod(a.shape[:2]) #0
    data_nb = [np.prod(a.shape)]
    dim = 3 
    for i, layer in enumerate(starvNet.layers):
        print(f'layer {i}: {type(layer)}')
        in_shape = np.array(a.shape)
        print('in_shape: ', in_shape)
        if len(in_shape) == 1:
            in_shape = np.append(in_shape, 1) 
        in_shape[dim] += n_pred
        print('input shape: ', in_shape)
        
        if isinstance(layer, FlattenLayer):
            dim = 1

        a = layer.evaluate(a)
        out_shape = np.array(a.shape)
        if isinstance(layer, MaxPool2DLayer) or isinstance(layer, ReLULayer):
            print('adding predicate: ', a.shape[:3])
            n_pred += np.prod(a.shape[:3])
            
        print('len(out_shape): ', len(out_shape))
        if len(out_shape) == 1:
            out_shape = np.append(out_shape, 1) 
            dim = 1

        print('out_shape: ', out_shape)
        out_shape[dim] += n_pred
        data_nb.append(np.prod(out_shape))
        print('output shape: ', out_shape)
        print('data_nb: ', data_nb[i])
        print('\n\n')

    data_nb = np.array(data_nb)
    if dtype == 'float64':
        data_nb *= 8
    elif dtype == 'float32':
        data_nb *= 4

    
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

    x = np.arange(starvNet.n_layers + 1)
    x_ticks_labels.insert(0, 'Input')

    plt.rcParams["figure.figsize"] = [8.50, 5.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1,1)
    plt.title("Memory Usage", fontsize=14)
    plt.plot(x, data_nb, color='red', linewidth=2)
    plt.xlabel("Layers", fontsize=12)
    plt.ylabel("Bytes", fontsize=12)

    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks_labels, rotation=80, fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    # set legend
    # ax.legend(['SIM COO', 'SIM CSR'], loc='center right', fontsize=12)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    plt.savefig(RESULT_dir + f'/Figure_1__worst_case_memory_usage_vgg16.png')
    plt.show()
    plt.close()


def verify_vnncomp2021_verivital(net_type = 'maxpool', dtype = 'float32'):

    folder_dir = f"StarV/util/data/nets/vnncomp2021/verivital"
    net_dir = f"{folder_dir}/Convnet_{net_type}.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    # starvNet = load_onnx_network(net_dir, channel_last=True, dtype=dtype, show=False)
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())

    shape = (28, 28, 1)

    vnnlib_dir = f"{folder_dir}/specs/{net_type}_specs"
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

    print(f"\n\nVerifying vnncomp2021 verivital with ImageStar")
    for i, vnnlib_file in enumerate(vnnlib_files):

        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"
        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        lb = bounds[:, 0].reshape(shape).astype(dtype)
        ub = bounds[:, 1].reshape(shape).astype(dtype)
        print('lb: ', lb.dtype)

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

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

    print(f"\n\nVerifying vnncomp2021 verivital with SparseImageStar in CSR format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])


        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).astype(dtype)
        ub = bounds[:, 1].reshape(shape).astype(dtype)

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

    print(f"\n\nVerifying vnncomp2021 verivital with SparseImageStar in COO format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])


        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).astype(dtype)
        ub = bounds[:, 1].reshape(shape).astype(dtype)

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
    path = f"artifacts/{artifact}_SparseImageStar/results/"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"verivital_vnncomp21_results_{dtype}.pkl"
    pickle.dump([rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, numPred], open(save_file, "wb"))

    headers = [f"ImageStar", f"SIM_CSR", f"SIM_COO"]

    # Robustness Resluts
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate([rb_table], headers=headers))
    print()

    Tlatex = tabulate([rb_table], headers=headers, tablefmt='latex')
    with open(path+f"verivital_vnncomp21_results_{dtype}_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate([vt_table], headers=headers))
    print()

    Tlatex = tabulate([vt_table], headers=headers, tablefmt='latex')
    with open(path+f"verivital_vnncomp21_results_{dtype}_vt.tex", "w") as f:
        print(Tlatex, file=f)

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def plot_table_vnncomp2021_verivital(dtype='float32'):
    print('=================================================================================')
    print('Table: Verification results of Verivital network from vnncomp2021')
    print('=================================================================================\n')
    folder_dir = STORE_dir
    file_dir = folder_dir + f"/verivital_vnncomp21_results_{dtype}.pkl"
    with open(file_dir, 'rb') as f:
        rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, numPred = pickle.load(f)

    headers = ['Property', 'Result', 'IM', 'SIM_csr', 'SIM_coo', 'ab-CROWN']
    
    N = 20
    vt_abcrown = [0.568, 0.548, 0.546, 0.564, 0.549, 0.546, 0.553, 0.584, 0.564, 0.551, 0.550, 0.562, 0.564, 0.546, 0.945, 0.584, 0.548, 0.550, 0.552, 0.559]

    data = []
    for i in range(N):
        result = 'UNSAT' if rbIM[i] == 1 else f"'SAT/UNK'"
        data.append([f"{i}", result, f"{vtIM[i]:0.3f}", f"{vtCSR[i]:0.3f}", f"{vtCOO[i]:0.3f}", f"{vt_abcrown[i]:0.3f}"])

    print(tabulate(data, headers=headers))

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(folder_dir+f"Table_XXX_vnncomp2021_verivital.tex", "w") as f:
        print(Tlatex, file=f)
		

def verify_mnist_cnn():
    verify_convnet_network(net_type='Small', dtype='float64')
    verify_convnet_network(net_type='Medium', dtype='float64')
    verify_convnet_network(net_type='Large', dtype='float64')
    plot_table_covnet_network_all()
    
def verify_vgg16():
    verify_vgg16_network(dtype='float64')
    verify_vgg16_converted_network(dtype='float64')
    verify_vgg16_network_spec_cn()
    plot_table_vgg16_network() 

if __name__ == "__main__":
    # Figure 1: The worse-case memory usage of generator and center images in ImageStar
    worst_case_vgg16() 

    # Table 1: Verification results of the MNIST CNN (CAV2020)
    verify_mnist_cnn()

    # Table 2: Verification results of VGG16 in seconds (vnncomp2023)
    verify_vgg16()

    # Figure 4: Memory usage comparison in verifying the VGG16 (vnncomp2023) with spec 11 image
    memory_usage_vgg16(spec=11)

    # Figure 5: Memory usage and computation time comparison in verifying the oval21 network with 𝑙∞ norm attack on all pixels.
    memory_usage_oval21()

    # Appendix:
    # Figure 7: Memory usage and computation time comparison between SparseImageStar CSR and COO in verifying the VGG16 (vnncomp2023) with spec c4 image
    memory_usage_vgg16_spec_cn(spec=4)

    # Verification results of VGG16 in seconds with specifications generated with random images
    verify_vgg16_network_spec_cn_random()
    plot_table_vgg16_network_random_images()

    # Verification results of VGG16 with specifications from vnncomp2023, but modified to contain a full output specification (results in seconds)
    plot_table_vgg16_network_full_specification()

    # Verification results of Verivital network from vnncomp2021
    verify_vnncomp2021_verivital(net_type='maxpool', dtype='float32')
    plot_table_vnncomp2021_verivital(dtype='float32')

    # Verification results of the MNIST CNN (CAV2020) [add ab-crown]

