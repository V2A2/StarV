"""
Load rnn network and input data from matlab (nvv)
Author: Bryan Duong
Date: 12/09/2024
"""

# Add the parent directory to sys.path

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)

import scipy.io
import numpy as np
from StarV.set.probstar import ProbStar
from StarV.net.network import NeuralNetwork
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.RecurrentLayer import RecurrentLayer
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer

nnv_path = "/home/bryan/Desktop/nnv/examples/artifact/HSCC2023/"


def load_raw_data(M=25):
    path_input = nnv_path + "/N_2_0/points.mat"
    input_data = scipy.io.loadmat(path_input)
    Xn = input_data["pickle_data"]
    return Xn[:M]


def load_data_points(index=0, M=25):
    path_input = nnv_path + "/N_2_0/points.mat"
    input_data = scipy.io.loadmat(path_input)
    Xn = input_data["pickle_data"]
    eps = 0.01

    lb, ub = zip(*[(x - eps, x + eps) for x in Xn[:25]])
    lb = list(lb)
    ub = list(ub)
    lb = np.array(lb)
    ub = np.array(ub)
    mu = (lb + ub) / 2
    sigma = (ub - mu) / 2.5
    Sig = [np.diag(sig**2) for sig in sigma]

    Sn = [ProbStar(mu_, Sig_, lb_, ub_) for mu_, Sig_, lb_, ub_ in zip(mu, Sig, lb, ub)]

    In = [Sn[index] for _ in range(M)]

    for I in In:
        I.C = np.zeros((1, I.V.shape[1] - 1))
        I.d = np.zeros((1,))

    return In


def load_rnn_matlab(rnn_path, o=2):

    Whh = scipy.io.loadmat(rnn_path)["recurrent_kernel"]
    bh = scipy.io.loadmat(rnn_path)["bias"]
    Whx = scipy.io.loadmat(rnn_path)["kernel"]

    if o == 2:
        Woh = np.eye(2)
        bo = np.zeros((2,))
    elif o == 4:
        Woh = np.eye(4)
        bo = np.zeros((4,))
    elif o == 8:
        Woh = np.eye(8)
        bo = np.zeros((8,))

    return Whh, bh.flatten(), Whx, Woh, bo


def load_fc_layers_matlab(dense_path):

    # Load the weights from matlab
    W = scipy.io.loadmat(dense_path)["W"]
    b = scipy.io.loadmat(dense_path)["b"]

    layers = []
    L0 = fullyConnectedLayer(W[0, 0], b[0, 0].flatten())
    L0_act = ReLULayer()
    L1 = fullyConnectedLayer(W[0, 1], b[0, 1].flatten())
    L1_act = ReLULayer()
    L2 = fullyConnectedLayer(W[0, 2], b[0, 2].flatten())
    L2_act = ReLULayer()
    L3 = fullyConnectedLayer(W[0, 3], b[0, 3].flatten())
    L3_act = ReLULayer()
    L4 = fullyConnectedLayer(W[0, 4], b[0, 4].flatten())
    L4_act = ReLULayer()
    L5 = fullyConnectedLayer(W[0, 5], b[0, 5].flatten())

    layers.append(L0)
    layers.append(L0_act)
    layers.append(L1)
    layers.append(L1_act)
    layers.append(L2)
    layers.append(L2_act)
    layers.append(L3)
    layers.append(L3_act)
    layers.append(L4)
    layers.append(L4_act)
    layers.append(L5)

    return layers


def load_rnn(x, y):
    print("Load RNN N_{}_{}".format(x, y))

    if x == 2 and y == 0:
        return load_N_2_0()
    elif x == 2 and y == 2:
        return load_N_2_2()
    elif x == 4 and y == 0:
        return load_N_4_0()
    elif x == 4 and y == 2:
        return load_N_4_2()
    elif x == 4 and y == 4:
        return load_N_4_4()
    elif x == 8 and y == 0:
        return load_N_8_0()
    else:
        raise Exception("Error: net not found")


def load_N_2_0():
    print("Load N_2_0")

    rnn_path = nnv_path + "N_2_0/py_simple_rnn.mat"
    dense_path = nnv_path + "N_2_0/py_dense.mat"
    Whh, bh, Whx, Woh, bo = load_rnn_matlab(rnn_path, o=2)
    FC_layers = load_fc_layers_matlab(dense_path)

    RNN = RecurrentLayer(Whh, bh, Whx, Woh, bo, fh="ReLU", fo="purelin")

    layers = []
    layers.append(RNN)
    layers.extend(FC_layers)

    net = NeuralNetwork(layers, net_type="rnn_N_2_0")

    return net


def load_N_2_2():

    print("Load N_2_2")

    rnn_1_path = nnv_path + "N_2_2/py_simple_rnn_6.mat"
    rnn_2_path = nnv_path + "N_2_2/py_simple_rnn_7.mat"
    dense_path = nnv_path + "N_2_2/py_dense.mat"

    Whh_1, bh_1, Whx_1, Woh_1, bo_1 = load_rnn_matlab(rnn_1_path, o=2)
    Whh_2, bh_2, Whx_2, Woh_2, bo_2 = load_rnn_matlab(rnn_2_path, o=2)
    FC_layers = load_fc_layers_matlab(dense_path)

    RNN1 = RecurrentLayer(Whh_1, bh_1, Whx_1, Woh_1, bo_1, fh="ReLU", fo="purelin")
    RNN2 = RecurrentLayer(Whh_2, bh_2, Whx_2, Woh_2, bo_2, fh="ReLU", fo="purelin")

    layers = []
    layers.append(RNN1)
    layers.append(RNN2)
    layers.extend(FC_layers)

    net = NeuralNetwork(layers, net_type="rnn_N_2_2")

    return net


def load_N_4_0():

    print("Load N_4_0")

    rnn_path = nnv_path + "N_4_0/py_simple_rnn_1.mat"
    dense_path = nnv_path + "N_4_0/py_dense.mat"

    Whh, bh, Whx, Woh, bo = load_rnn_matlab(rnn_path, o=4)
    FC_layers = load_fc_layers_matlab(dense_path)

    RNN = RecurrentLayer(Whh, bh, Whx, Woh, bo, fh="ReLU", fo="purelin")

    layers = []
    layers.append(RNN)
    layers.extend(FC_layers)

    net = NeuralNetwork(layers, net_type="rnn_N_4_0")

    return net


def load_N_4_2():

    print("Load N_4_2")

    rnn_1_path = nnv_path + "N_4_2/py_simple_rnn_4.mat"
    rnn_2_path = nnv_path + "N_4_2/py_simple_rnn_5.mat"
    dense_path = nnv_path + "N_4_2/py_dense.mat"

    Whh_1, bh_1, Whx_1, Woh_1, bo_1 = load_rnn_matlab(rnn_1_path, o=4)
    Whh_2, bh_2, Whx_2, Woh_2, bo_2 = load_rnn_matlab(rnn_2_path, o=2)
    FC_layers = load_fc_layers_matlab(dense_path)

    RNN1 = RecurrentLayer(Whh_1, bh_1, Whx_1, Woh_1, bo_1, fh="ReLU", fo="purelin")
    RNN2 = RecurrentLayer(Whh_2, bh_2, Whx_2, Woh_2, bo_2, fh="ReLU", fo="purelin")

    layers = []
    layers.append(RNN1)
    layers.append(RNN2)
    layers.extend(FC_layers)

    net = NeuralNetwork(layers, net_type="rnn_N_4_2")

    return net


def load_N_4_4():

    print("Load N_4_4")

    rnn_1_path = nnv_path + "N_4_4/py_simple_rnn_8.mat"
    rnn_2_path = nnv_path + "N_4_4/py_simple_rnn_9.mat"
    dense_path = nnv_path + "N_4_4/py_dense.mat"

    Whh_1, bh_1, Whx_1, Woh_1, bo_1 = load_rnn_matlab(rnn_1_path, o=4)
    Whh_2, bh_2, Whx_2, Woh_2, bo_2 = load_rnn_matlab(rnn_2_path, o=4)
    FC_layers = load_fc_layers_matlab(dense_path)

    RNN1 = RecurrentLayer(Whh_1, bh_1, Whx_1, Woh_1, bo_1, fh="ReLU", fo="purelin")
    RNN2 = RecurrentLayer(Whh_2, bh_2, Whx_2, Woh_2, bo_2, fh="ReLU", fo="purelin")

    layers = []
    layers.append(RNN1)
    layers.append(RNN2)
    layers.extend(FC_layers)

    net = NeuralNetwork(layers, net_type="rnn_N_4_4")

    return net


def load_N_8_0():

    print("Load N_8_0")

    rnn_path = nnv_path + "N_8_0/py_simple_rnn_3.mat"
    dense_path = nnv_path + "N_8_0/py_dense.mat"

    Whh, bh, Whx, Woh, bo = load_rnn_matlab(rnn_path, o=8)
    FC_layers = load_fc_layers_matlab(dense_path)

    RNN = RecurrentLayer(Whh, bh, Whx, Woh, bo, fh="ReLU", fo="purelin")

    layers = []
    layers.append(RNN)
    layers.extend(FC_layers)

    net = NeuralNetwork(layers, net_type="rnn_N_8_0")

    return net
