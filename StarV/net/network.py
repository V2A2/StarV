#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
##                                                                     ##
##   Primary contacts: Hoang Dung Tran <dungtran@ufl.edu> (UF)         ##
##                     Sung Woo Choi <sungwoo.choi@ufl.edu> (UF)       ##
##                     Yuntao Li <yli17@ufl.edu> (UF)                  ##
##                     Qing Liu <qliu1@ufl.edu> (UF)                   ##
##                                                                     ##
##   See CONTRIBUTORS for full author contacts and affiliations.       ##
##   This program is licensed under the BSD 3‑Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
  Generic Network Class
  
  Dung Tran, 9/10/2022
  Update: 12/20/2024 (Sung Woo Choi, merging)
"""

import time
import copy
import torch
import numpy as np
from typing import List, Union, Tuple, Optional
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.FlattenLayer import FlattenLayer
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
from StarV.layer.PadLayer import PadLayer
from StarV.layer.CropLayer import CropLayer
from StarV.layer.ConcatenateLayer import ConcatenateLayer

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

    def __init__(self, layers, net_type=None, UNet=None):

        assert isinstance(layers, list), 'error: layers should be a list'
        self.type = net_type
        self.layers = layers
        self.n_layers = len(layers)
        self.in_dim = layers[0].in_dim
        for i in range(len(layers) - 1, -1, -1):
            if hasattr(layers[i], 'out_dim'):
                self.out_dim = layers[i].out_dim
                break
        #UNet = [down, up];
        # down = [d0, ..., dn] is a list containing layer number from which x needs to be stored for concatenation
        # up   = [u0, ..., un] is a list containing layer number from which the stored x based on down need to be concatenated to
        if UNet is not None:
            down, up = UNet
            assert len(down) == len(up), f'length of up and down lists must be equivalent but down: {len(down)} while up: {len(up)}'
            self.unet_up = up
            self.unet_down = down
        else:
            self.unet_up = None
            self.unet_down = None

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
            if isinstance(layer_, FullyConnectedLayer):
                if layer_.W is not None:
                    str_ += ' ({}, {}, dtype={})'.format(layer_.out_dim, layer_.in_dim, layer_.W.dtype)
                else:
                    str_ += ' ({}, {}, dtype={})'.format(layer_.out_dim, layer_.in_dim, layer_.b.dtype)

            elif isinstance(layer_, LogSigLayer) or isinstance(layer_, TanSigLayer):
                str_ += ' (opt = {}, delta = {})'.format(layer_.opt, layer_.delta)
            elif isinstance(layer_, Conv2DLayer):
                if layer_.sparse:
                    str_ += ' ({}, {}, kernel_size = {}, stride = {}, padding = {}, dilation = {}, dtype={})'.format(
                        layer_.in_shape[2], layer_.out_shape[2], layer_.kernel_size, layer_.stride, layer_.padding, layer_.dilation, layer_.weight.dtype)
                else:
                    str_ += ' ({}, {}, kernel_size = {}, stride = {}, padding = {}, dilation = {}, dtype={})'.format(
                        layer_.weight.shape[2], layer_.weight.shape[3], layer_.weight.shape[:2], layer_.stride, layer_.padding, layer_.dilation, layer_.weight.dtype)
            elif isinstance(layer_, ConvTranspose2DLayer):
                str_ += ' ({}, {}, kernel_size = {}, stride = {}, padding = {}, output_padding={}, dilation = {}, dtype={})'.format(
                    layer_.weight.shape[3], layer_.weight.shape[2], layer_.weight.shape[:2], layer_.stride, layer_.padding, layer_.output_padding, layer_.dilation, layer_.weight.dtype)
            elif isinstance(layer_, AvgPool2DLayer):
                str_ += ' (kernel_size = {}, stride = {}, padding = {})'.format(layer_.kernel_size, layer_.stride, layer_.padding)
            elif isinstance(layer_, MaxPool2DLayer):
                str_ += ' (kernel_size = {}, stride = {}, padding = {})'.format(layer_.kernel_size, layer_.stride, layer_.padding)
            elif isinstance(layer_, BatchNorm2DLayer):
                str_ += ' ({}, eps={}, dtype={})'.format(layer_.num_features, layer_.eps, layer_.gamma.dtype)
            elif isinstance(layer_, FlattenLayer):
                str_ += ' (channel_last={})'.format(layer_.channel_last)
            elif isinstance(layer_, PadLayer):
                str_ += ' (padding={})'.format(layer_.pad_width)
            elif isinstance(layer_, CropLayer):
                str_ += ' (top={}, left={}, height={}, width={})'.format(layer_.top, layer_.left, layer_.height, layer_.width)
            elif isinstance(layer_, PixelClassificationLayer):
                str_ += ' (num_pix_classes={})'.format(layer_.classes)
                if layer_.threshold is not None:
                    str_ += f', threshold={layer_.threshold}'
            elif isinstance(layer_, ConcatenateLayer):
                str_ += ' (axis={})'.format(layer_.axis)
            print(str_)
        if self.unet_down is not None and self.unet_up is not None:
            print(f'down (unet): {self.unet_down}')
            print(f'up (unet): {self.unet_up}')
        return ''

    def info(self):
        print(self)

    def evaluate(self, input_vec, show=False):
        'evaluate a network on a specific input vector'

        assert isinstance(input_vec, np.ndarray), 'error: input vector is not a numpy array'
        # assert len(input_vec.shape) == 1, 'error: input vector should be a 1-d numpy array'

        if self.unet_down is not None and self.unet_up is not None:
            if show: print('unet_down: {}, unet_up: {}'.format(self.unet_down, self.unet_up))
            unet_up = self.unet_up[::-1]  #reverse the up list for easier pop operation
            y = input_vec.copy()
            stored_x = dict()
            if show: evaluate_start = time.perf_counter()
            for i in range(self.n_layers):
                if show: 
                    print(f"evaluating {i} layer: {self.layers[i].__class__.__name__}")
                    print(f"input shape: {y.shape}")
                    start = time.perf_counter()
                if i in self.unet_down:
                    i_ = len(stored_x)
                    stored_x[unet_up[i_]] = y.copy()
                    if show: print(f'Storing x at layer {i} for UNet concatenation')
                if i in self.unet_up:
                    if show: print(f'Concatenating stored x at layer {i} for UNet concatenation')
                    x = stored_x[i]
                    y = self.layers[i].evaluate(y, x)
                else:
                    y = self.layers[i].evaluate(y)
                if show: 
                    print(f"output shape, dtype: {y.shape}, {y.dtype}")
                    print(f'computation time: {time.perf_counter()-start} seconds')
            
            if show: print(f'total evaluation time: {time.perf_counter()-evaluate_start} seconds')
            return y
        
        y = input_vec.copy()
        if show: evaluate_start = time.perf_counter()
        for i in range(self.n_layers):
            if show: 
                print(f"evaluating {i} layer: {self.layers[i].__class__.__name__}")
                print(f"input shape: {y.shape}")
                start = time.perf_counter()
            y = self.layers[i].evaluate(y)
            if show: 
                print(f"output shape, dtype: {y.shape}, {y.dtype}")
                print(f'computation time: {time.perf_counter()-start} seconds')
        if show: print(f'total evaluation time: {time.perf_counter()-evaluate_start} seconds')
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
        layers.append(FullyConnectedLayer([W, b]))
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

def reachExactDFS(net, inputSet, f=None, label=None, lp_solver='gurobi', pool=None, show=True):
    """Compute Reachable Set using DFS method
    args:
        @net: NeuralNetwork object
        @inputSet: a list of input set (Star/ProbStar)
        @f: a function to verify robustness; f(Ri) returns:
            rb = 0: unknown
            rb = 1: robust
            rb = 2: not robust
        @lp_solver: lp solver: 'gurobi' (default), 'glpk', or 'linprog'
        @pool: parallel pool: None or multiprocessing.pool.Pool
        @show: boolean variable to show progress
    return:
        @Y: a list of all final reachable sets
        @rb: robustness verification result
        @vt: verification time

    @author: Sung Woo Choi
    """

    assert isinstance(net, NeuralNetwork), 'error: first input should be a NeuralNetwork object'
    assert isinstance(inputSet, list), 'error: second input should be a list of Star/ProbStar set'

    if f is not None:
        assert label is not None, 'error: label must be provided for robustness verification'

    # contains all remaining reachable set at all k steps, 
    # remains = [RM1, RM2, ..., RMk], 
    #     RMj = [I1, I2, ..., Im]
    remains = [copy.deepcopy(inputSet)]

    rb = [] if f is not None else None # robustness verification result
    Y = [] # a list of all final reachable sets (should be equivalent to BFS method, but order may be different)
    start = time.perf_counter()
    while True:
        # get the current k step
        k = len(remains) - 1
        if k >= 0:
            # get the remaining reachable sets at current k step
            RMj = remains[k]
        else:
            # no remaining reachable sets at current k step
            break
        
        if len(RMj) > 0:
            # pop out one input set at current k step
            Ii = RMj.pop(0)
            
            # in the worset case m = 2^n reachable sets can be generated from one input set
            # Rm = [R1, R2, ..., Rm]
            if show:
                print(f'Computing reachable set at layer {k} {net.layers[k].__class__.__name__}...')

            Rm = net.layers[k].reach([Ii], method='exact', lp_solver=lp_solver, pool=pool, show=show)

            # process reachable sets at k+1 step
            if k == net.n_layers - 1:
                # final reachable sets
                # therefor no need to store reachable sets at k+1 step; simply add them to output list Y
                Y.extend(Rm)

                # for robustness verification
                rb = f(Rm, label) if f is not None else None
                if rb == 2 or rb == 0:
                    # unsafe or unknown, stop the process; unkonwn should not happen in exact analysis
                    vt = time.perf_counter() - start
                    return Y, rb, vt

            # add reachable sets at k+1 step
            elif len(remains) == k + 1:
                # first time to add reachable sets at k+1 step
                remains.append(Rm)

            # k + 1 step already exists
            else:
                # not the first time to add reachable sets at k+1 step
                # append reachable sets at k+1 step
                remains[k + 1].extend(Rm)

        else:
            # all reachable sets at current k step have been processed
            # pop out remains at k step
            remains.pop(k)

    vt = time.perf_counter() - start
    return Y, rb, vt
    

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

def reachApproxBFS(net, inputSet, p_filter=0.0, lp_solver='gurobi', pool=None, show=True):
    """Compute Approximate Reachable Set layer-by-layer"""

    assert isinstance(net, NeuralNetwork), 'error: first input should be a NeuralNetwork object'
    assert isinstance(inputSet, list) or isinstance(inputSet, Star), 'error: second input should be a list of Star/ProbStar set or just a Star set'

    # compute and filter reachable sets
    I = copy.deepcopy(inputSet)

    if isinstance(inputSet, list):
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
    
    if isinstance(I, Star):
        for i in range(0, net.n_layers):
            if show:
                print('Computing layer {} {} reachable set...'.format(i, net.layers[i].__class__.__name__))
                # print('Computing layer {} reachable set...'.format(i))
            S = net.layers[i].reach(I, method='approx', lp_solver=lp_solver, pool=pool)
            I = S
            if show:
                print('Number of stars: {}'.format(len(S)))
        return S

