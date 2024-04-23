"""
  Generic Network Class
  
  Dung Tran, 9/10/2022
"""

import numpy as np
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
import copy
import multiprocessing
from StarV.util.print_util import print_util

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

    def info(self):
        """print information of the network"""

        print('\n=============NETWORK===============')
        print('Network type: {}'.format(self.type))
        print('Input Dimension: {}'.format(self.in_dim))
        print('Output Dimension: {}'.format(self.out_dim))
        print('Number of Layers: {}'.format(self.n_layers))
        print('Layer types:')
        for i in range(0, self.n_layers):
            print('Layer {}: {}'.format(i, type(self.layers[i])))

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

def reachExactBFS(net, inputSet, lp_solver='gurobi', pool=None, show=True):
    """Compute Reachable Set layer-by-layer"""

    assert isinstance(net, NeuralNetwork), 'error: first input should be a NeuralNetwork object'
    assert isinstance(inputSet, list), 'error: second input should be a list of Star/ProbStar set'

    S = copy.deepcopy(inputSet)
    for i in range(0, net.n_layers):
        if show:
            print('Computing layer {} reachable set...'.format(i))
        S = net.layers[i].reach(S, method='exact', lp_solver=lp_solver, pool=pool)
        for j in range(0, len(S)):
            print_util("h3")
            print('Layer {} Reachable set #{}'.format(i, j))
            S[j].__str__()
            lb_e, ub_e = S[j].estimateRanges()
            print('Estimated Range: ', lb_e, ub_e)
            lb_lp, ub_lp = S[j].getRanges()
            print('Exact Range: ', lb_lp, ub_lp)
            print_util("h3")
        if show:
            print('Number of stars/probstars: {}'.format(len(S)))

    for i in range(0, len(S)):
        print_util("h3")
        print('Final Reachable set #', i)
        S[i].__str__()
        lb_e, ub_e = S[i].estimateRanges()
        print('Estimated Range: ', lb_e, ub_e)
        lb_lp, ub_lp = S[i].getRanges()
        print('Exact Range: ', lb_lp, ub_lp)
        print_util("h3")

    return S

