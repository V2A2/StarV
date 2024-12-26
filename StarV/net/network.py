"""
  Generic Network Class
  
  Dung Tran, 9/10/2022
  Update: 12/20/2024 (Sung Woo Choi, merging)
"""

import copy
import torch
import numpy as np
from typing import List, Union, Tuple, Optional
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.FlattenLayer import FlattenLayer
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.Conv2DLayer import Conv2DLayer
from StarV.layer.ConvTranspose2DLayer import ConvTranspose2DLayer
from StarV.layer.AvgPool2DLayer import AvgPool2DLayer
from StarV.layer.MaxPool2DLayer import MaxPool2DLayer
from StarV.layer.BatchNorm2DLayer import BatchNorm2DLayer
from StarV.layer.LogSigLayer import LogSigLayer
from StarV.layer.TanSigLayer import TanSigLayer
from StarV.layer.PixelClassificationLayer import PixelClassificationLayer
from StarV.layer.RecurrentLayer import RecurrentLayer

from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import copy
import multiprocessing
import itertools
from collections import Counter

class NeuralNetwork(object):
    """Generic serial Neural Network class

    It can be:
     * feedforward
     * concolutional
     * semantic segmentation
     * recurrent (may be)
     * binary

    Properties:
        @type: network type
        @layers: a list of layers
        @n_layers: number of layers
        @in_dim: input dimension
        @out_dim: output dimension

    Methods:
        @rand: randomly  generate a network
    """

    def __init__(self, layers, net_type=None):

        assert isinstance(layers, list), 'error: layers should be a list'
        self.type = net_type
        self.layers = layers
        self.n_layers = len(layers)
        self.in_dim = layers[0].in_dim
        for i in range(len(layers) - 1, -1, -1):
            if hasattr(layers[i], 'out_dim'):
                self.out_dim = layers[i].out_dim
                break

    def __str__(self):
        """print information of the network"""

        print('\n=============NETWORK===============')
        print('Network type: {}'.format(self.type))
        print('Input Dimension: {}'.format(self.in_dim))
        print('Output Dimension: {}'.format(self.out_dim))
        print('Number of Layers: {}'.format(self.n_layers))
        print('Layer types:')
        for i in range(0, self.n_layers):
            str_ = 'Layer {}: {}'.format(i, type(self.layers[i]))
            layer_ = self.layers[i]
            if isinstance(layer_, FullyConnectedLayer) or isinstance(layer_, fullyConnectedLayer):
                if layer_.W is not None:
                    str_ += ' ({}, {}, dtype={})'.format(layer_.out_dim, layer_.in_dim, layer_.W.dtype)
                else:
                    str_ += ' ({}, {}, dtype={})'.format(layer_.out_dim, layer_.in_dim, layer_.b.dtype)

            elif isinstance(layer_, LogSigLayer) or isinstance(layer_, TanSigLayer):
                str_ += ' (opt = {}, delta = {})'.format(layer_.opt, layer_.delta)
            elif isinstance(layer_, Conv2DLayer):
                if layer_.sparse:
                    str_ += ' ({}, {}, kernel_size = {}, stride = {}, padding = {}, dtype={})'.format(layer_.in_shape[2], layer_.out_shape[2], layer_.kernel_size, layer_.stride, layer_.padding, layer_.weight.dtype)
                else:
                    str_ += ' ({}, {}, kernel_size = {}, stride = {}, padding = {}, dtype={})'.format(layer_.weight.shape[2], layer_.weight.shape[3], layer_.weight.shape[:2], layer_.stride, layer_.padding, layer_.weight.dtype)
            elif isinstance(layer_, ConvTranspose2DLayer):
                str_ += ' ({}, {}, kernel_size = {}, stride = {}, padding = {}, output_padding={}, dtype={})'.format(
                    layer_.weight.shape[2], layer_.weight.shape[3], layer_.weight.shape[:2], layer_.stride, layer_.padding, layer_.output_padding,  layer_.weight.dtype)
            elif isinstance(layer_, AvgPool2DLayer):
                str_ += ' (kernel_size = {}, stride = {}, padding = {})'.format(layer_.kernel_size, layer_.stride, layer_.padding)
            elif isinstance(layer_, MaxPool2DLayer):
                str_ += ' (kernel_size = {}, stride = {}, padding = {})'.format(layer_.kernel_size, layer_.stride, layer_.padding)
            elif isinstance(layer_, BatchNorm2DLayer):
                str_ += ' ({}, eps={}, dtype={})'.format(layer_.num_features, layer_.eps, layer_.gamma.dtype)
            elif isinstance(layer_, FlattenLayer):
                str_ += ' (channel_last={})'.format(layer_.channel_last)
            print(str_)
        return ''

    def info(self):
        print(self)

    def evaluate(self, input_vec, show=False):
        'evaluate a network on a specific input vector'

        assert isinstance(input_vec, np.ndarray), 'error: input vector is not a numpy array'
        # assert len(input_vec.shape) == 1, 'error: input vector should be a 1-d numpy array'

        y = input_vec.copy()
        for i in range(self.n_layers):
            if show: print(f"evaluating {i} layer: {self.layers[i].__class__.__name__}"); print(f"input shape: {y.shape}")
            y = self.layers[i].evaluate(y)
            if show: print(f"output shape: {y.shape}")
        return y

def rand_ffnn(arch, actvs):
    """randomly generate feedforward neural network
    Args:
        @arch: network architecture list of layer's neurons ex. [2 3 2]
        @actvs: list of activation functions
    """

    assert isinstance(arch, list), 'error: network architecture should be in a list object'
    assert isinstance(actvs, list), 'error: activation functions should be in a list object'
    assert len(arch) >= 2, 'error: network should have at least one layer'
    assert len(arch) == len(actvs) + 1, 'error: inconsistent between the network architecture and activation list'

    for i in range(0, len(arch)):
        if arch[i] <= 0:
            raise Exception('error: invalid number of neural at {}^th layer'.format(i+1))

    for i in range(0, len(actvs)):
        if actvs[i] != 'poslin' and actvs[i] != 'relu' and actvs[i] != None:
            raise Exception('error: {} is an unsupported/unknown activation function'.format(actvs[i]))

    layers = []
    for i in range(0, len(actvs)):
        W = np.random.rand(arch[i+1], arch[i])
        b = np.random.rand(arch[i+1])
        layers.append(fullyConnectedLayer(W, b))
        if actvs[i] == 'poslin' or actvs[i] == 'relu':
            layers.append(ReLULayer())

    return NeuralNetwork(layers, 'ffnn')

def filterProbStar(*args):
    """Filtering out some probstars"""

    if isinstance(args[0], tuple):
        args1 = args[0]
    else:
        args1 = args
    p_filter = args1[0]
    S = args1[1]
    assert isinstance(S, ProbStar), 'error: input is not a probstar'
    prob = S.estimateProbability()
    if prob >= p_filter:
        P = S
        p_ignored = 0.0
    else:
        P = []
        p_ignored = prob

    return P, p_ignored
    

def reachExactBFS(net, inputSet, lp_solver='gurobi', pool=None, show=True):
    """Compute Reachable Set layer-by-layer"""

    assert isinstance(net, NeuralNetwork), 'error: first input should be a NeuralNetwork object'
    assert isinstance(inputSet, list), 'error: second input should be a list of Star/ProbStar set'

    S = copy.deepcopy(inputSet)
    for i in range(0, net.n_layers):
        if show:
            print('Computing layer {} {} reachable set...'.format(i, net.layers[i].__class__.__name__))
        S = net.layers[i].reach(S, method='exact', lp_solver=lp_solver, pool=pool)
        if show:
            print('Number of stars/probstars: {}'.format(len(S)))

    return S

def reachApproxBFS(net, inputSet, p_filter, lp_solver='gurobi', pool=None, show=True):
    """Compute Approximate Reachable Set layer-by-layer"""

    assert isinstance(net, NeuralNetwork), 'error: first input should be a NeuralNetwork object'
    assert isinstance(inputSet, list), 'error: second input should be a list of Star/ProbStar set'

    # compute and filter reachable sets
    I = copy.deepcopy(inputSet)
    p_ignored = 0.0
    for i in range(0, net.n_layers):
        if show:
            print('================ Layer {} ================='.format(i))
            print('Computing layer {} reachable set...'.format(i))
        S = net.layers[i].reach(I, method='exact', lp_solver=lp_solver, pool=pool)
        if show:
            print('Number of probstars: {}'.format(len(S)))
            print('Filtering probstars whose probabilities < {}...'.format(p_filter))
        P = []
        if pool is None:
            for S1 in S:
                P1, prob1 = filterProbStar(p_filter, S1)
                if isinstance(P1, ProbStar):
                    P.append(P1)
                p_ignored = p_ignored + prob1  # update the total probability of ignored sets
        else:
            S1 = pool.map(filterProbStar, zip([p_filter]*len(S), S))
            for S2 in S1:
                if isinstance(S2[0], ProbStar):
                    P.append(S2[0])
                p_ignored = p_ignored + S2[1]
        I = P            
        if show:
            print('Number of ignored probstars: {}'.format(len(S) - len(I)))
            print('Number of remaining probstars: {}'.format(len(I)))

        if len(I) == 0:
            break


    return I, p_ignored


def reach_exact_bfs_star_relu(net: NeuralNetwork, inputSet: List[Star], method = 'exact',
                  lp_solver: str = 'gurobi', pool: Optional[multiprocessing.Pool] = None, 
                  show: bool = True) -> List[Union[Star, ProbStar]]:
    """Compute Reachable Set layer-by-layer"""
    S = [star.clone() for star in inputSet]
    for i, layer in enumerate(net.layers):
        if show:
            print(f'Computing layer {i} reachable set...')
        S = layer.reach(S, method, lp_solver=lp_solver, pool=pool)
        if show:
            print(f'Number of stars: {len(S)}\n')
    return S