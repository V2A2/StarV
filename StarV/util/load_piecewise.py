"""
load module (ReLU, LeakyReLU, SatLin, SatLins networks) for testing/evaluation, 
including tiny network, 2017 IEEE TNNLS paper, ACASXU networks, HCAS networks, etc.
Yuntao Li, 2/6/2024
"""

import os
from scipy.io import loadmat
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.LeakyReLULayer import LeakyReLULayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.SatLinsLayer import SatLinsLayer
from StarV.net.network import NeuralNetwork
from StarV.plant.lode import LODE
import numpy as np
import torch
import math


def load_tiny_network_ReLU():
    """Load a tiny 2-inputs 2-output network as a running example"""

    layers = []
    W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b1 = np.array([0.5, 1.0, -0.5])
    L1 = fullyConnectedLayer(W1, b1)
    L2 = ReLULayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    L3 = fullyConnectedLayer(W2,b2)
    
    layers.append(L1)
    layers.append(L2)
    layers.append(L3)

    net = NeuralNetwork(layers, 'ffnn_tiny_network')

    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])

    unsafe_mat = np.array([[1.0, 0]])
    unsafe_vec = np.array([-2.0])

    return net, lb, ub, unsafe_mat, unsafe_vec


def load_2017_IEEE_TNNLS_ReLU():
    """Load network from the IEEE TNNLS 2017 paper
       refs: https://arxiv.org/abs/1712.08163
    """
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/2017_IEEE_TNNLS.mat'
    mat_contents = loadmat(cur_path)
    W = mat_contents['W']
    b = mat_contents['b']
    layers = []
    for i in range(0, b.shape[1]-1):
        Wi = W[0, i]
        bi = b[0, i]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = ReLULayer()
        layers.append(L2)

    Wi = W[0, b.shape[1]-1]
    bi = b[0, b.shape[1]-1]
    bi = bi.reshape(bi.shape[0],)
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)
    net = NeuralNetwork(layers, net_type='ffnn_2017_IEEE_TNNLS')

    return net


def load_ACASXU_ReLU(x, y, spec_id, actv='relu'):
    """Load ACASXU networks
       Args:
           @ network id (x,y)
           @ specification_id: 1, 2, 3, or 4

       Return
           @net: network
           @lb: normalized lower bound of inputs
           @ub: normalized upper bound of inputs
           @unsafe_mat: unsafe matrix, i.e., unsafe region of the outputs
           @unsafe_vec: unsafe vector: 
           ***unsafe region: (unsafe_mat * y <= unsafe_vec)
    """

    # StarV/util/data/nets/ACASXU_ReLU/ReLU_ACASXU_run2a_1_1_batch_2000.mat
    net_name = 'ReLU_ACASXU_run2a_{}_{}_batch_2000'.format(x,y)
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/ACASXU_Benchmarks/ACASXU_ReLU/' + net_name
    mat_contents = loadmat(cur_path)
    W = mat_contents['W']
    b = mat_contents['b']
    means_for_scaling = mat_contents['means_for_scaling']
    means_for_scaling = means_for_scaling.reshape(6,)
    range_for_scaling = mat_contents['range_for_scaling']
    range_for_scaling = range_for_scaling.reshape(6,)
    
    layers = []
    for i in range(0, b.shape[1]-1):
        Wi = W[0, i]
        bi = b[0, i]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = ReLULayer()
        layers.append(L2)

    Wi = W[0, b.shape[1]-1]
    bi = b[0, b.shape[1]-1]
    bi = bi.reshape(bi.shape[0],)
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)
    net = NeuralNetwork(layers, net_type='ffnn_ACASXU_{}_{}'.format(x,y))

    # Get input constraints and specs (unsafe constraints on the outputs)
    # paper: https://arxiv.org/pdf/1702.01135.pdf
    
    if spec_id == 1 or spec_id == 2:

        # Input Constraints:
        #      55947.69 <= i1(\rho) <= 60760
        #
        # Input Constraints
        # 55947.69 <= i1(\rho) <= 60760,
        # -3.14 <= i2 (\theta) <= 3.14,
        # -3.14 <= i3 (\shi) <= -3.14
        #  1145 <= i4 (\v_own) <= 1200, 
        #  0 <= i5 (\v_in) <= 60
        lb = np.array([55947.69, -3.14, -3.14, 1145, 0])
        ub = np.array([60760, 3.14, 3.14, 1200, 60])

        # Output constraints (specifications)
        if spec_id == 1:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # verify safety: COC <= 1500 or x1 <= 1500 after normalization
        
            # safe region before normalization
            # x1' <= (1500 - 7.5189)/373.9499 = 3.9911, 373.9499 is from range_for_scaling[5]
            unsafe_mat = np.array([-1, 0, 0, 0, 0])
            unsafe_vec = np.array([-3.9911])  # unsafe region x1' > 3.9911
        if spec_id == 2:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # safety property: COC is not the maximal score
            # unsafe region: COC is the maximal score: x1 >= x2; x1 >= x3; x1 >= x4, x1 >= x5
            unsafe_mat = np.array([[-1.0, 1.0, 0., 0., 0.], [-1., 0., 1., 0., 0.], [-1., 0., 0., 1., 0.], [-1., 0., 0., 0., 1.,]])
            unsafe_vec = np.array([0., 0., 0., 0.])
        

    elif spec_id == 3:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # 3.1 <= i3 (\shi) <= 3.14
        # 980 <= i4 (\v_own) <= 1200, 
        # 960 <= i5 (\v_in) <= 1200
        # ****NOTE There was a slight mismatch of the ranges of
        # this i5 input for the conference paper, FM2019 "Star-based Reachability of DNNs"
        lb = np.array([1500, -0.06, 3.1, 980, 960])
        ub = np.array([1800, 0.06, 3.14, 1200, 1200])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 4:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # (\shi) = 0
        # 1000 <= i4 (\v_own) <= 1200, 
        # 700 <= i5 (\v_in) <= 800
        lb = np.array([1500, -0.06, 0, 1000, 700])
        ub = np.array([1800, 0.06, 0, 1200, 800])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    else:
        raise Exception('Invalide Specification ID')

    # Normalize input
    for i in range(0, 5):
        lb[i] = (lb[i] - means_for_scaling[i])/range_for_scaling[i]
        ub[i] = (ub[i] - means_for_scaling[i])/range_for_scaling[i]

   
    return net, lb, ub, unsafe_mat, unsafe_vec


def load_HCAS_ReLU(prev_acv, tau, spec_id):
    """Load HCAS networks
       Args:
           @ network id (prev_acv, tau)
           @ specification_id: 2, 3, or 4

       Return
           @net: network
           @lb: normalized lower bound of inputs
           @ub: normalized upper bound of inputs
           @unsafe_mat: unsafe matrix, i.e., unsafe region of the outputs
           @unsafe_vec: unsafe vector: 
           ***unsafe region: (unsafe_mat * y <= unsafe_vec)
    """

    # /home/yuntao/Documents/Verification/StarV_Project/StarV/StarV/util/data/nets/HCAS_ReLU/HCAS_rect_v6_pra0_tau00_25HU.nnet
    net_name = 'HCAS_rect_v6_pra{}_tau{}_25HU.nnet'.format(prev_acv, tau)
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/HCAS_Benchmarks/HCAS_ReLU/' + net_name
    
    # load the network
    with open(cur_path) as f:
        line = f.readline()
        cnt = 1
        while line[0:2] == "//":
            line=f.readline() 
            cnt+= 1
        #numLayers does't include the input layer!
        numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
        line=f.readline()

        #input layer size, layer1size, layer2size...
        layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

        line=f.readline()
        symmetric = int(line.strip().split(",")[0])

        line = f.readline()
        inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

        weights=[]
        biases = []
        for layernum in range(numLayers):

            previousLayerSize = layerSizes[layernum]
            currentLayerSize = layerSizes[layernum+1]
            weights.append([])
            biases.append([])
            weights[layernum] = np.zeros((currentLayerSize,previousLayerSize))
            for i in range(currentLayerSize):
                line=f.readline()
                aux = [float(x) for x in line.strip().split(",")[:-1]]
                for j in range(previousLayerSize):
                    weights[layernum][i,j] = aux[j]
            #biases
            biases[layernum] = np.zeros(currentLayerSize)
            for i in range(currentLayerSize):
                line=f.readline()
                x = float(line.strip().split(",")[0])
                biases[layernum][i] = x

    layers = []
    for i in range(0, numLayers-1):
        Wi = weights[i]
        bi = biases[i]
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = ReLULayer()
        layers.append(L2)

    Wi = weights[numLayers-1]
    bi = biases[numLayers-1]
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)

    net = NeuralNetwork(layers, net_type='ffnn_HCAS_{}_{}'.format(prev_acv, tau))

    # Get input constraints and specs (unsafe constraints on the outputs)
    # paper: https://arxiv.org/pdf/1702.01135.pdf
    
    if spec_id == 2:

        # Input Constraints:
        #      55947.69 <= i1(\rho) <= 60760
        #
        # Input Constraints
        # 55947.69 <= i1(\rho) <= 60760,
        # -3.14 <= i2 (\theta) <= 3.14,
        # -3.14 <= i3 (\shi) <= -3.14
        #  1145 <= i4 (\v_own) <= 1200, 
        #  0 <= i5 (\v_in) <= 60
        lb = np.array([55947.69, -3.14, -3.14, 1145, 0])
        ub = np.array([60760, 3.14, 3.14, 1200, 60])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the maximal score
        # unsafe region: COC is the maximal score: x1 >= x2; x1 >= x3; x1 >= x4, x1 >= x5
        unsafe_mat = np.array([[-1.0, 1.0, 0., 0., 0.], [-1., 0., 1., 0., 0.], [-1., 0., 0., 1., 0.], [-1., 0., 0., 0., 1.,]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 3:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # 3.1 <= i3 (\shi) <= 3.14
        # 980 <= i4 (\v_own) <= 1200, 
        # 960 <= i5 (\v_in) <= 1200
        # ****NOTE There was a slight mismatch of the ranges of
        # this i5 input for the conference paper, FM2019 "Star-based Reachability of DNNs"
        lb = np.array([1500, -0.06, 3.1, 980, 960])
        ub = np.array([1800, 0.06, 3.14, 1200, 1200])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 4:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # (\shi) = 0
        # 1000 <= i4 (\v_own) <= 1200, 
        # 700 <= i5 (\v_in) <= 800
        lb = np.array([1500, -0.06, 0, 1000, 700])
        ub = np.array([1800, 0.06, 0, 1200, 800])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    else:
        raise Exception('Invalide Specification ID')
    
        # Normalize input
    for i in range(0, 5):
        lb[i] = (lb[i] - inputMeans[i])/inputRanges[i]
        ub[i] = (ub[i] - inputMeans[i])/inputRanges[i]

   
    return net, lb, ub, unsafe_mat, unsafe_vec



def load_tiny_network_LeakyReLU():
    """Load a tiny 2-inputs 2-output network as a running example"""

    layers = []
    W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b1 = np.array([0.5, 1.0, -0.5])
    L1 = fullyConnectedLayer(W1, b1)
    L2 = LeakyReLULayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    L3 = fullyConnectedLayer(W2,b2)
    
    layers.append(L1)
    layers.append(L2)
    layers.append(L3)

    net = NeuralNetwork(layers, 'ffnn_tiny_network')

    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])

    unsafe_mat = np.array([[1.0, 0]])
    unsafe_vec = np.array([-2.0])

    return net, lb, ub, unsafe_mat, unsafe_vec


def load_2017_IEEE_TNNLS_LeakyReLU():
    """Load network from the IEEE TNNLS 2017 paper
       refs: https://arxiv.org/abs/1712.08163
    """
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/2017_IEEE_TNNLS.mat'
    mat_contents = loadmat(cur_path)
    W = mat_contents['W']
    b = mat_contents['b']
    layers = []
    for i in range(0, b.shape[1]-1):
        Wi = W[0, i]
        bi = b[0, i]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = LeakyReLULayer()
        layers.append(L2)

    Wi = W[0, b.shape[1]-1]
    bi = b[0, b.shape[1]-1]
    bi = bi.reshape(bi.shape[0],)
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)
    net = NeuralNetwork(layers, net_type='ffnn_2017_IEEE_TNNLS')

    return net


def load_ACASXU_LeakyReLU(x, y, spec_id, actv='relu'):
    """Load ACASXU networks
       Args:
           @ network id (x,y)
           @ specification_id: 1, 2, 3, or 4

       Return
           @net: network
           @lb: normalized lower bound of inputs
           @ub: normalized upper bound of inputs
           @unsafe_mat: unsafe matrix, i.e., unsafe region of the outputs
           @unsafe_vec: unsafe vector: 
           ***unsafe region: (unsafe_mat * y <= unsafe_vec)
    """

    # StarV/util/data/nets/ACASXU_LeakyReLU/LeakyReLU_ACASXU_run2a_1_1_batch_2000.mat
    net_name = 'LeakyReLU_ACASXU_run2a_{}_{}_batch_2000'.format(x,y)
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/ACASXU_Benchmarks/ACASXU_LeakyReLU/' + net_name
    mat_contents = loadmat(cur_path)
    W = mat_contents['W']
    b = mat_contents['b']
    means_for_scaling = mat_contents['means_for_scaling']
    means_for_scaling = means_for_scaling.reshape(6,)
    range_for_scaling = mat_contents['range_for_scaling']
    range_for_scaling = range_for_scaling.reshape(6,)
    
    layers = []
    for i in range(0, b.shape[1]-1):
        Wi = W[0, i]
        bi = b[0, i]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = LeakyReLULayer()
        layers.append(L2)

    Wi = W[0, b.shape[1]-1]
    bi = b[0, b.shape[1]-1]
    bi = bi.reshape(bi.shape[0],)
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)
    net = NeuralNetwork(layers, net_type='ffnn_ACASXU_{}_{}'.format(x,y))

    # Get input constraints and specs (unsafe constraints on the outputs)
    # paper: https://arxiv.org/pdf/1702.01135.pdf
    
    if spec_id == 1 or spec_id == 2:

        # Input Constraints:
        #      55947.69 <= i1(\rho) <= 60760
        #
        # Input Constraints
        # 55947.69 <= i1(\rho) <= 60760,
        # -3.14 <= i2 (\theta) <= 3.14,
        # -3.14 <= i3 (\shi) <= -3.14
        #  1145 <= i4 (\v_own) <= 1200, 
        #  0 <= i5 (\v_in) <= 60
        lb = np.array([55947.69, -3.14, -3.14, 1145, 0])
        ub = np.array([60760, 3.14, 3.14, 1200, 60])

        # Output constraints (specifications)
        if spec_id == 1:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # verify safety: COC <= 1500 or x1 <= 1500 after normalization
        
            # safe region before normalization
            # x1' <= (1500 - 7.5189)/373.9499 = 3.9911, 373.9499 is from range_for_scaling[5]
            unsafe_mat = np.array([-1, 0, 0, 0, 0])
            unsafe_vec = np.array([-3.9911])  # unsafe region x1' > 3.9911
        if spec_id == 2:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # safety property: COC is not the maximal score
            # unsafe region: COC is the maximal score: x1 >= x2; x1 >= x3; x1 >= x4, x1 >= x5
            unsafe_mat = np.array([[-1.0, 1.0, 0., 0., 0.], [-1., 0., 1., 0., 0.], [-1., 0., 0., 1., 0.], [-1., 0., 0., 0., 1.,]])
            unsafe_vec = np.array([0., 0., 0., 0.])
        

    elif spec_id == 3:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # 3.1 <= i3 (\shi) <= 3.14
        # 980 <= i4 (\v_own) <= 1200, 
        # 960 <= i5 (\v_in) <= 1200
        # ****NOTE There was a slight mismatch of the ranges of
        # this i5 input for the conference paper, FM2019 "Star-based Reachability of DNNs"
        lb = np.array([1500, -0.06, 3.1, 980, 960])
        ub = np.array([1800, 0.06, 3.14, 1200, 1200])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 4:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # (\shi) = 0
        # 1000 <= i4 (\v_own) <= 1200, 
        # 700 <= i5 (\v_in) <= 800
        lb = np.array([1500, -0.06, 0, 1000, 700])
        ub = np.array([1800, 0.06, 0, 1200, 800])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    else:
        raise Exception('Invalide Specification ID')

    # Normalize input
    for i in range(0, 5):
        lb[i] = (lb[i] - means_for_scaling[i])/range_for_scaling[i]
        ub[i] = (ub[i] - means_for_scaling[i])/range_for_scaling[i]

   
    return net, lb, ub, unsafe_mat, unsafe_vec


def load_HCAS_LeakyReLU(prev_acv, tau, spec_id):
    """Load HCAS networks
       Args:
           @ network id (prev_acv, tau)
           @ specification_id: 2, 3, or 4

       Return
           @net: network
           @lb: normalized lower bound of inputs
           @ub: normalized upper bound of inputs
           @unsafe_mat: unsafe matrix, i.e., unsafe region of the outputs
           @unsafe_vec: unsafe vector: 
           ***unsafe region: (unsafe_mat * y <= unsafe_vec)
    """

    # /home/yuntao/Documents/Verification/StarV_Project/StarV/StarV/util/data/nets/HCAS_ReLU/HCAS_rect_v6_pra0_tau00_25HU.nnet
    net_name = 'HCAS_rect_v6_pra{}_tau{}_25HU.nnet'.format(prev_acv, tau)
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/HCAS_Benchmarks/HCAS_LeakyReLU/' + net_name
    
    # load the network
    with open(cur_path) as f:
        line = f.readline()
        cnt = 1
        while line[0:2] == "//":
            line=f.readline() 
            cnt+= 1
        #numLayers does't include the input layer!
        numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
        line=f.readline()

        #input layer size, layer1size, layer2size...
        layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

        line=f.readline()
        symmetric = int(line.strip().split(",")[0])

        line = f.readline()
        inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

        weights=[]
        biases = []
        for layernum in range(numLayers):

            previousLayerSize = layerSizes[layernum]
            currentLayerSize = layerSizes[layernum+1]
            weights.append([])
            biases.append([])
            weights[layernum] = np.zeros((currentLayerSize,previousLayerSize))
            for i in range(currentLayerSize):
                line=f.readline()
                aux = [float(x) for x in line.strip().split(",")[:-1]]
                for j in range(previousLayerSize):
                    weights[layernum][i,j] = aux[j]
            #biases
            biases[layernum] = np.zeros(currentLayerSize)
            for i in range(currentLayerSize):
                line=f.readline()
                x = float(line.strip().split(",")[0])
                biases[layernum][i] = x

    layers = []
    for i in range(0, numLayers-1):
        Wi = weights[i]
        bi = biases[i]
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = LeakyReLULayer()
        layers.append(L2)

    Wi = weights[numLayers-1]
    bi = biases[numLayers-1]
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)

    net = NeuralNetwork(layers, net_type='ffnn_HCAS_{}_{}'.format(prev_acv, tau))

    # Get input constraints and specs (unsafe constraints on the outputs)
    # paper: https://arxiv.org/pdf/1702.01135.pdf
    
    if spec_id == 2:

        # Input Constraints:
        #      55947.69 <= i1(\rho) <= 60760
        #
        # Input Constraints
        # 55947.69 <= i1(\rho) <= 60760,
        # -3.14 <= i2 (\theta) <= 3.14,
        # -3.14 <= i3 (\shi) <= -3.14
        #  1145 <= i4 (\v_own) <= 1200, 
        #  0 <= i5 (\v_in) <= 60
        lb = np.array([55947.69, -3.14, -3.14, 1145, 0])
        ub = np.array([60760, 3.14, 3.14, 1200, 60])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the maximal score
        # unsafe region: COC is the maximal score: x1 >= x2; x1 >= x3; x1 >= x4, x1 >= x5
        unsafe_mat = np.array([[-1.0, 1.0, 0., 0., 0.], [-1., 0., 1., 0., 0.], [-1., 0., 0., 1., 0.], [-1., 0., 0., 0., 1.,]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 3:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # 3.1 <= i3 (\shi) <= 3.14
        # 980 <= i4 (\v_own) <= 1200, 
        # 960 <= i5 (\v_in) <= 1200
        # ****NOTE There was a slight mismatch of the ranges of
        # this i5 input for the conference paper, FM2019 "Star-based Reachability of DNNs"
        lb = np.array([1500, -0.06, 3.1, 980, 960])
        ub = np.array([1800, 0.06, 3.14, 1200, 1200])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 4:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # (\shi) = 0
        # 1000 <= i4 (\v_own) <= 1200, 
        # 700 <= i5 (\v_in) <= 800
        lb = np.array([1500, -0.06, 0, 1000, 700])
        ub = np.array([1800, 0.06, 0, 1200, 800])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    else:
        raise Exception('Invalide Specification ID')
    
        # Normalize input
    for i in range(0, 5):
        lb[i] = (lb[i] - inputMeans[i])/inputRanges[i]
        ub[i] = (ub[i] - inputMeans[i])/inputRanges[i]

   
    return net, lb, ub, unsafe_mat, unsafe_vec



def load_tiny_network_SatLin():
    """Load a tiny 2-inputs 2-output network as a running example"""

    layers = []
    W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b1 = np.array([0.5, 1.0, -0.5])
    L1 = fullyConnectedLayer(W1, b1)
    L2 = SatLinLayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    L3 = fullyConnectedLayer(W2,b2)
    
    layers.append(L1)
    layers.append(L2)
    layers.append(L3)

    net = NeuralNetwork(layers, 'ffnn_tiny_network')

    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])

    unsafe_mat = np.array([[1.0, 0]])
    unsafe_vec = np.array([-2.0])

    return net, lb, ub, unsafe_mat, unsafe_vec


def load_2017_IEEE_TNNLS_SatLin():
    """Load network from the IEEE TNNLS 2017 paper
       refs: https://arxiv.org/abs/1712.08163
    """
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/2017_IEEE_TNNLS.mat'
    mat_contents = loadmat(cur_path)
    W = mat_contents['W']
    b = mat_contents['b']
    layers = []
    # for i in range(0, b.shape[1]-1):
    for i in range(0, b.shape[1]-5):
        Wi = W[0, i]
        bi = b[0, i]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = SatLinLayer()
        layers.append(L2)

    Wi = W[0, b.shape[1]-1]
    bi = b[0, b.shape[1]-1]
    bi = bi.reshape(bi.shape[0],)
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)
    net = NeuralNetwork(layers, net_type='ffnn_2017_IEEE_TNNLS')

    return net


def load_ACASXU_SatLin(x, y, spec_id):
    """Load ACASXU networks
       Args:
           @ network id (x,y)
           @ specification_id: 1, 2, 3, or 4

       Return
           @net: network
           @lb: normalized lower bound of inputs
           @ub: normalized upper bound of inputs
           @unsafe_mat: unsafe matrix, i.e., unsafe region of the outputs
           @unsafe_vec: unsafe vector: 
           ***unsafe region: (unsafe_mat * y <= unsafe_vec)
    """

    # StarV/util/data/nets/ACASXU_SatLin/SatLin_ACASXU_run2a_1_1_batch_2000.mat
    net_name = 'SatLin_ACASXU_run2a_{}_{}_batch_2000'.format(x,y)
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/ACASXU_Benchmarks/ACASXU_SatLin/' + net_name
    mat_contents = loadmat(cur_path)
    W = mat_contents['W']
    b = mat_contents['b']
    means_for_scaling = mat_contents['means_for_scaling']
    means_for_scaling = means_for_scaling.reshape(6,)
    range_for_scaling = mat_contents['range_for_scaling']
    range_for_scaling = range_for_scaling.reshape(6,)
    
    layers = []
    for i in range(0, b.shape[1]-1):
    # for i in range(0, b.shape[1]-5):
        Wi = W[0, i]
        bi = b[0, i]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = SatLinLayer()
        layers.append(L2)

    Wi = W[0, b.shape[1]-1]
    bi = b[0, b.shape[1]-1]
    bi = bi.reshape(bi.shape[0],)
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)
    net = NeuralNetwork(layers, net_type='ffnn_ACASXU_{}_{}'.format(x,y))

    # Get input constraints and specs (unsafe constraints on the outputs)
    # paper: https://arxiv.org/pdf/1702.01135.pdf
    
    if spec_id == 1 or spec_id == 2:

        # Input Constraints:
        #      55947.69 <= i1(\rho) <= 60760
        #
        # Input Constraints
        # 55947.69 <= i1(\rho) <= 60760,
        # -3.14 <= i2 (\theta) <= 3.14,
        # -3.14 <= i3 (\shi) <= -3.14
        #  1145 <= i4 (\v_own) <= 1200, 
        #  0 <= i5 (\v_in) <= 60
        lb = np.array([55947.69, -3.14, -3.14, 1145, 0])
        ub = np.array([60760, 3.14, 3.14, 1200, 60])

        # Output constraints (specifications)
        if spec_id == 1:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # verify safety: COC <= 1500 or x1 <= 1500 after normalization
        
            # safe region before normalization
            # x1' <= (1500 - 7.5189)/373.9499 = 3.9911, 373.9499 is from range_for_scaling[5]
            unsafe_mat = np.array([-1, 0, 0, 0, 0])
            unsafe_vec = np.array([-3.9911])  # unsafe region x1' > 3.9911
        if spec_id == 2:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # safety property: COC is not the maximal score
            # unsafe region: COC is the maximal score: x1 >= x2; x1 >= x3; x1 >= x4, x1 >= x5
            unsafe_mat = np.array([[-1.0, 1.0, 0., 0., 0.], [-1., 0., 1., 0., 0.], [-1., 0., 0., 1., 0.], [-1., 0., 0., 0., 1.,]])
            unsafe_vec = np.array([0., 0., 0., 0.])
        

    elif spec_id == 3:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # 3.1 <= i3 (\shi) <= 3.14
        # 980 <= i4 (\v_own) <= 1200, 
        # 960 <= i5 (\v_in) <= 1200
        # ****NOTE There was a slight mismatch of the ranges of
        # this i5 input for the conference paper, FM2019 "Star-based Reachability of DNNs"
        lb = np.array([1500, -0.06, 3.1, 980, 960])
        ub = np.array([1800, 0.06, 3.14, 1200, 1200])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 4:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # (\shi) = 0
        # 1000 <= i4 (\v_own) <= 1200, 
        # 700 <= i5 (\v_in) <= 800
        lb = np.array([1500, -0.06, 0, 1000, 700])
        ub = np.array([1800, 0.06, 0, 1200, 800])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    else:
        raise Exception('Invalide Specification ID')

    # Normalize input
    for i in range(0, 5):
        lb[i] = (lb[i] - means_for_scaling[i])/range_for_scaling[i]
        ub[i] = (ub[i] - means_for_scaling[i])/range_for_scaling[i]

   
    return net, lb, ub, unsafe_mat, unsafe_vec



def load_HCAS_SatLin(prev_acv, tau, spec_id):
    """Load HCAS networks
       Args:
           @ network id (prev_acv, tau)
           @ specification_id: 2, 3, or 4

       Return
           @net: network
           @lb: normalized lower bound of inputs
           @ub: normalized upper bound of inputs
           @unsafe_mat: unsafe matrix, i.e., unsafe region of the outputs
           @unsafe_vec: unsafe vector: 
           ***unsafe region: (unsafe_mat * y <= unsafe_vec)
    """

    # /home/yuntao/Documents/Verification/StarV_Project/StarV/StarV/util/data/nets/HCAS_ReLU/HCAS_rect_v6_pra0_tau00_25HU.nnet
    # net_name = 'HCAS_rect_v6_pra{}_tau{}_25HU.nnet'.format(prev_acv, tau)
    # cur_path = os.path.dirname(__file__)
    # cur_path = cur_path + '/data/nets/HCAS_SatLin/' + net_name

    net_name = 'HCAS_rect_v6_pra{}_tau{}_10HU.nnet'.format(prev_acv, tau)
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/HCAS_Benchmarks/HCAS_SatLin_small/' + net_name
    
    # load the network
    with open(cur_path) as f:
        line = f.readline()
        cnt = 1
        while line[0:2] == "//":
            line=f.readline() 
            cnt+= 1
        #numLayers does't include the input layer!
        numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
        line=f.readline()

        #input layer size, layer1size, layer2size...
        layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

        line=f.readline()
        symmetric = int(line.strip().split(",")[0])

        line = f.readline()
        inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

        weights=[]
        biases = []
        for layernum in range(numLayers):

            previousLayerSize = layerSizes[layernum]
            currentLayerSize = layerSizes[layernum+1]
            weights.append([])
            biases.append([])
            weights[layernum] = np.zeros((currentLayerSize,previousLayerSize))
            for i in range(currentLayerSize):
                line=f.readline()
                aux = [float(x) for x in line.strip().split(",")[:-1]]
                for j in range(previousLayerSize):
                    weights[layernum][i,j] = aux[j]
            #biases
            biases[layernum] = np.zeros(currentLayerSize)
            for i in range(currentLayerSize):
                line=f.readline()
                x = float(line.strip().split(",")[0])
                biases[layernum][i] = x

    layers = []
    for i in range(0, numLayers-1):
        Wi = weights[i]
        bi = biases[i]
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = SatLinLayer()
        layers.append(L2)

    Wi = weights[numLayers-1]
    bi = biases[numLayers-1]
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)

    net = NeuralNetwork(layers, net_type='ffnn_HCAS_{}_{}'.format(prev_acv, tau))

    # Get input constraints and specs (unsafe constraints on the outputs)
    # paper: https://arxiv.org/pdf/1702.01135.pdf
    
    if spec_id == 2:

        # Input Constraints:
        #      55947.69 <= i1(\rho) <= 60760
        #
        # Input Constraints
        # 55947.69 <= i1(\rho) <= 60760,
        # -3.14 <= i2 (\theta) <= 3.14,
        # -3.14 <= i3 (\shi) <= -3.14
        #  1145 <= i4 (\v_own) <= 1200, 
        #  0 <= i5 (\v_in) <= 60
        lb = np.array([55947.69, -3.14, -3.14, 1145, 0])
        ub = np.array([60760, 3.14, 3.14, 1200, 60])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the maximal score
        # unsafe region: COC is the maximal score: x1 >= x2; x1 >= x3; x1 >= x4, x1 >= x5
        unsafe_mat = np.array([[-1.0, 1.0, 0., 0., 0.], [-1., 0., 1., 0., 0.], [-1., 0., 0., 1., 0.], [-1., 0., 0., 0., 1.,]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 3:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # 3.1 <= i3 (\shi) <= 3.14
        # 980 <= i4 (\v_own) <= 1200, 
        # 960 <= i5 (\v_in) <= 1200
        # ****NOTE There was a slight mismatch of the ranges of
        # this i5 input for the conference paper, FM2019 "Star-based Reachability of DNNs"
        lb = np.array([1500, -0.06, 3.1, 980, 960])
        ub = np.array([1800, 0.06, 3.14, 1200, 1200])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 4:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # (\shi) = 0
        # 1000 <= i4 (\v_own) <= 1200, 
        # 700 <= i5 (\v_in) <= 800
        lb = np.array([1500, -0.06, 0, 1000, 700])
        ub = np.array([1800, 0.06, 0, 1200, 800])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    else:
        raise Exception('Invalide Specification ID')
    
        # Normalize input
    for i in range(0, 5):
        lb[i] = (lb[i] - inputMeans[i])/inputRanges[i]
        ub[i] = (ub[i] - inputMeans[i])/inputRanges[i]

   
    return net, lb, ub, unsafe_mat, unsafe_vec



def load_tiny_network_SatLins():
    """Load a tiny 2-inputs 2-output network as a running example"""

    layers = []
    W1 = np.array([[1.0, -2.0], [-1., 0.5], [1., 1.5]])
    b1 = np.array([0.5, 1.0, -0.5])
    L1 = fullyConnectedLayer(W1, b1)
    L2 = SatLinsLayer()
    W2 = np.array([[-1.0, -1.0, 1.0], [2.0, 1.0, -0.5]])
    b2 = np.array([-0.2, -1.0])
    L3 = fullyConnectedLayer(W2,b2)
    
    layers.append(L1)
    layers.append(L2)
    layers.append(L3)

    net = NeuralNetwork(layers, 'ffnn_tiny_network')

    lb = np.array([-2.0, -1.0])
    ub = np.array([2.0, 1.0])

    unsafe_mat = np.array([[1.0, 0]])
    unsafe_vec = np.array([-2.0])

    return net, lb, ub, unsafe_mat, unsafe_vec


def load_2017_IEEE_TNNLS_SatLins():
    """Load network from the IEEE TNNLS 2017 paper
       refs: https://arxiv.org/abs/1712.08163
    """
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/2017_IEEE_TNNLS.mat'
    mat_contents = loadmat(cur_path)
    W = mat_contents['W']
    b = mat_contents['b']
    layers = []
    # for i in range(0, b.shape[1]-1):
    for i in range(0, b.shape[1]-5):
        Wi = W[0, i]
        bi = b[0, i]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = SatLinsLayer()
        layers.append(L2)

    Wi = W[0, b.shape[1]-1]
    bi = b[0, b.shape[1]-1]
    bi = bi.reshape(bi.shape[0],)
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)
    net = NeuralNetwork(layers, net_type='ffnn_2017_IEEE_TNNLS')

    return net


def load_ACASXU_SatLins(x, y, spec_id):
    """Load ACASXU networks
       Args:
           @ network id (x,y)
           @ specification_id: 1, 2, 3, or 4

       Return
           @net: network
           @lb: normalized lower bound of inputs
           @ub: normalized upper bound of inputs
           @unsafe_mat: unsafe matrix, i.e., unsafe region of the outputs
           @unsafe_vec: unsafe vector: 
           ***unsafe region: (unsafe_mat * y <= unsafe_vec)
    """

    # StarV/util/data/nets/ACASXU_SatLins/SatLins_ACASXU_run2a_1_1_batch_2000.mat
    net_name = 'SatLins_ACASXU_run2a_{}_{}_batch_2000'.format(x,y)
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/ACASXU_Benchmarks/ACASXU_SatLins/' + net_name
    mat_contents = loadmat(cur_path)
    W = mat_contents['W']
    b = mat_contents['b']
    means_for_scaling = mat_contents['means_for_scaling']
    means_for_scaling = means_for_scaling.reshape(6,)
    range_for_scaling = mat_contents['range_for_scaling']
    range_for_scaling = range_for_scaling.reshape(6,)
    
    layers = []
    # for i in range(0, b.shape[1]-1):
    for i in range(0, b.shape[1]-5):
        Wi = W[0, i]
        bi = b[0, i]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = SatLinsLayer()
        layers.append(L2)

    Wi = W[0, b.shape[1]-1]
    bi = b[0, b.shape[1]-1]
    bi = bi.reshape(bi.shape[0],)
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)
    net = NeuralNetwork(layers, net_type='ffnn_ACASXU_{}_{}'.format(x,y))

    # Get input constraints and specs (unsafe constraints on the outputs)
    # paper: https://arxiv.org/pdf/1702.01135.pdf
    
    if spec_id == 1 or spec_id == 2:

        # Input Constraints:
        #      55947.69 <= i1(\rho) <= 60760
        #
        # Input Constraints
        # 55947.69 <= i1(\rho) <= 60760,
        # -3.14 <= i2 (\theta) <= 3.14,
        # -3.14 <= i3 (\shi) <= -3.14
        #  1145 <= i4 (\v_own) <= 1200, 
        #  0 <= i5 (\v_in) <= 60
        lb = np.array([55947.69, -3.14, -3.14, 1145, 0])
        ub = np.array([60760, 3.14, 3.14, 1200, 60])

        # Output constraints (specifications)
        if spec_id == 1:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # verify safety: COC <= 1500 or x1 <= 1500 after normalization
        
            # safe region before normalization
            # x1' <= (1500 - 7.5189)/373.9499 = 3.9911, 373.9499 is from range_for_scaling[5]
            unsafe_mat = np.array([-1, 0, 0, 0, 0])
            unsafe_vec = np.array([-3.9911])  # unsafe region x1' > 3.9911
        if spec_id == 2:
            # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
            # safety property: COC is not the maximal score
            # unsafe region: COC is the maximal score: x1 >= x2; x1 >= x3; x1 >= x4, x1 >= x5
            unsafe_mat = np.array([[-1.0, 1.0, 0., 0., 0.], [-1., 0., 1., 0., 0.], [-1., 0., 0., 1., 0.], [-1., 0., 0., 0., 1.,]])
            unsafe_vec = np.array([0., 0., 0., 0.])
        

    elif spec_id == 3:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # 3.1 <= i3 (\shi) <= 3.14
        # 980 <= i4 (\v_own) <= 1200, 
        # 960 <= i5 (\v_in) <= 1200
        # ****NOTE There was a slight mismatch of the ranges of
        # this i5 input for the conference paper, FM2019 "Star-based Reachability of DNNs"
        lb = np.array([1500, -0.06, 3.1, 980, 960])
        ub = np.array([1800, 0.06, 3.14, 1200, 1200])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 4:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # (\shi) = 0
        # 1000 <= i4 (\v_own) <= 1200, 
        # 700 <= i5 (\v_in) <= 800
        lb = np.array([1500, -0.06, 0, 1000, 700])
        ub = np.array([1800, 0.06, 0, 1200, 800])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    else:
        raise Exception('Invalide Specification ID')

    # Normalize input
    for i in range(0, 5):
        lb[i] = (lb[i] - means_for_scaling[i])/range_for_scaling[i]
        ub[i] = (ub[i] - means_for_scaling[i])/range_for_scaling[i]

   
    return net, lb, ub, unsafe_mat, unsafe_vec


def load_HCAS_SatLins(prev_acv, tau, spec_id):
    """Load HCAS networks
       Args:
           @ network id (prev_acv, tau)
           @ specification_id: 2, 3, or 4

       Return
           @net: network
           @lb: normalized lower bound of inputs
           @ub: normalized upper bound of inputs
           @unsafe_mat: unsafe matrix, i.e., unsafe region of the outputs
           @unsafe_vec: unsafe vector: 
           ***unsafe region: (unsafe_mat * y <= unsafe_vec)
    """

    # /home/yuntao/Documents/Verification/StarV_Project/StarV/StarV/util/data/nets/HCAS_ReLU/HCAS_rect_v6_pra0_tau00_25HU.nnet
    # net_name = 'HCAS_rect_v6_pra{}_tau{}_25HU.nnet'.format(prev_acv, tau)
    # cur_path = os.path.dirname(__file__)
    # cur_path = cur_path + '/data/nets/HCAS_SatLins/' + net_name

    net_name = 'HCAS_rect_v6_pra{}_tau{}_10HU.nnet'.format(prev_acv, tau)
    cur_path = os.path.dirname(__file__)
    cur_path = cur_path + '/data/nets/HCAS_Benchmarks/HCAS_SatLins_small/' + net_name
    
    # load the network
    with open(cur_path) as f:
        line = f.readline()
        cnt = 1
        while line[0:2] == "//":
            line=f.readline() 
            cnt+= 1
        #numLayers does't include the input layer!
        numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
        line=f.readline()

        #input layer size, layer1size, layer2size...
        layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

        line=f.readline()
        symmetric = int(line.strip().split(",")[0])

        line = f.readline()
        inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

        weights=[]
        biases = []
        for layernum in range(numLayers):

            previousLayerSize = layerSizes[layernum]
            currentLayerSize = layerSizes[layernum+1]
            weights.append([])
            biases.append([])
            weights[layernum] = np.zeros((currentLayerSize,previousLayerSize))
            for i in range(currentLayerSize):
                line=f.readline()
                aux = [float(x) for x in line.strip().split(",")[:-1]]
                for j in range(previousLayerSize):
                    weights[layernum][i,j] = aux[j]
            #biases
            biases[layernum] = np.zeros(currentLayerSize)
            for i in range(currentLayerSize):
                line=f.readline()
                x = float(line.strip().split(",")[0])
                biases[layernum][i] = x

    layers = []
    for i in range(0, numLayers-1):
        Wi = weights[i]
        bi = biases[i]
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        L2 = SatLinsLayer()
        layers.append(L2)

    Wi = weights[numLayers-1]
    bi = biases[numLayers-1]
    L1 = fullyConnectedLayer(Wi, bi)
    layers.append(L1)

    net = NeuralNetwork(layers, net_type='ffnn_HCAS_{}_{}'.format(prev_acv, tau))

    # Get input constraints and specs (unsafe constraints on the outputs)
    # paper: https://arxiv.org/pdf/1702.01135.pdf
    
    if spec_id == 2:

        # Input Constraints:
        #      55947.69 <= i1(\rho) <= 60760
        #
        # Input Constraints
        # 55947.69 <= i1(\rho) <= 60760,
        # -3.14 <= i2 (\theta) <= 3.14,
        # -3.14 <= i3 (\shi) <= -3.14
        #  1145 <= i4 (\v_own) <= 1200, 
        #  0 <= i5 (\v_in) <= 60
        lb = np.array([55947.69, -3.14, -3.14, 1145, 0])
        ub = np.array([60760, 3.14, 3.14, 1200, 60])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the maximal score
        # unsafe region: COC is the maximal score: x1 >= x2; x1 >= x3; x1 >= x4, x1 >= x5
        unsafe_mat = np.array([[-1.0, 1.0, 0., 0., 0.], [-1., 0., 1., 0., 0.], [-1., 0., 0., 1., 0.], [-1., 0., 0., 0., 1.,]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 3:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # 3.1 <= i3 (\shi) <= 3.14
        # 980 <= i4 (\v_own) <= 1200, 
        # 960 <= i5 (\v_in) <= 1200
        # ****NOTE There was a slight mismatch of the ranges of
        # this i5 input for the conference paper, FM2019 "Star-based Reachability of DNNs"
        lb = np.array([1500, -0.06, 3.1, 980, 960])
        ub = np.array([1800, 0.06, 3.14, 1200, 1200])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    elif spec_id == 4:
        # Input Constraints
        # 1500 <= i1(\rho) <= 1800,
        # -0.06 <= i2 (\theta) <= 0.06,
        # (\shi) = 0
        # 1000 <= i4 (\v_own) <= 1200, 
        # 700 <= i5 (\v_in) <= 800
        lb = np.array([1500, -0.06, 0, 1000, 700])
        ub = np.array([1800, 0.06, 0, 1200, 800])

        # Output constraints (specifications)
        # output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        # safety property: COC is not the minimal score
        # unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1 <= x5
        unsafe_mat = np.array([[1., -1., 0., 0., 0.], [1., 0., -1., 0., 0.], [1., 0., 0., -1., 0.], [1., 0., 0., 0., -1.]])
        unsafe_vec = np.array([0., 0., 0., 0.])
        
    else:
        raise Exception('Invalide Specification ID')
    
        # Normalize input
    for i in range(0, 5):
        lb[i] = (lb[i] - inputMeans[i])/inputRanges[i]
        ub[i] = (ub[i] - inputMeans[i])/inputRanges[i]

   
    return net, lb, ub, unsafe_mat, unsafe_vec