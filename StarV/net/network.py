"""
Generic Network Class
Author: Yuntao Li
Date: 1/20/2024
"""

import numpy as np
from typing import List, Union, Tuple, Optional
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import copy
import multiprocessing

class NeuralNetwork:
    """
    Generic serial Neural Network class

    It can be:
     * feedforward
     * convolutional
     * semantic segmentation
     * recurrent (maybe)
     * binary

    Attributes:
        type (str): Network type
        layers (List): List of layers
        n_layers (int): Number of layers
        in_dim (int): Input dimension
        out_dim (int): Output dimension
    """

    def __init__(self, layers: List, net_type: Optional[str] = None):
        if not isinstance(layers, list):
            raise ValueError('Layers should be a list')
        self.type = net_type
        self.layers = layers
        self.n_layers = len(layers)
        self.in_dim = layers[0].in_dim
        self.out_dim = next((layer.out_dim for layer in reversed(layers) if hasattr(layer, 'out_dim')), None)

    def info(self):
        """Print information about the network"""
        print('\n=============NETWORK===============')
        print(f'Network type: {self.type}')
        print(f'Input Dimension: {self.in_dim}')
        print(f'Output Dimension: {self.out_dim}')
        print(f'Number of Layers: {self.n_layers}')
        print('Layer types:')
        for i, layer in enumerate(self.layers):
            print(f'Layer {i}: {type(layer)}')

    def evaluate(self, input_vec: np.ndarray) -> np.ndarray:
        """Evaluate the network on a specific input vector"""
        if not isinstance(input_vec, np.ndarray) or input_vec.ndim != 1:
            raise ValueError('Input vector should be a 1-d numpy array')

        y = input_vec
        for layer in self.layers:
            y = layer.evaluate(y)
        return y

def rand_ffnn(arch: List[int], actvs: List[str]) -> NeuralNetwork:
    """
    Randomly generate feedforward neural network

    Args:
        arch (List[int]): Network architecture list of layer's neurons ex. [2, 3, 2]
        actvs (List[str]): List of activation functions

    Returns:
        NeuralNetwork: Randomly generated feedforward neural network
    """
    if not isinstance(arch, list) or not isinstance(actvs, list):
        raise ValueError('Network architecture and activation functions should be lists')
    if len(arch) < 2 or len(arch) != len(actvs) + 1:
        raise ValueError('Invalid network architecture or activation list')
    if any(neurons <= 0 for neurons in arch):
        raise ValueError('Invalid number of neurons in a layer')
    if any(actv not in ['poslin', 'relu', None] for actv in actvs):
        raise ValueError('Unsupported activation function')

    layers = []
    for i, actv in enumerate(actvs):
        W = np.random.rand(arch[i+1], arch[i])
        b = np.random.rand(arch[i+1])
        layers.append(fullyConnectedLayer(W, b))
        if actv in ['poslin', 'relu']:
            layers.append(ReLULayer())

    return NeuralNetwork(layers, 'ffnn')

def filterProbStar(args: Tuple[float, ProbStar]) -> Tuple[Union[ProbStar, List], float]:
    """
    Filter out some ProbStars based on probability threshold

    Args:
        args (Tuple[float, ProbStar]): Probability threshold and ProbStar

    Returns:
        Tuple[Union[ProbStar, List], float]: Filtered ProbStar (or empty list) and ignored probability
    """
    p_filter, S = args
    if not isinstance(S, ProbStar):
        raise ValueError('Input is not a ProbStar')
    prob = S.estimateProbability()
    return (S, 0.0) if prob >= p_filter else ([], prob)

def reachExactBFS(net: NeuralNetwork, inputSet: List[Union[Star, ProbStar]], 
                  lp_solver: str = 'gurobi', pool: Optional[multiprocessing.Pool] = None, 
                  show: bool = True) -> List[Union[Star, ProbStar]]:
    """Compute Reachable Set layer-by-layer"""
    S = [probstar.clone() for probstar in inputSet]
    for i, layer in enumerate(net.layers):
        if show:
            print(f'Computing layer {i} reachable set...')
        S = layer.reach(S, method='exact', lp_solver=lp_solver, pool=pool)
        if show:
            print(f'Number of stars/probstars: {len(S)}')
    return S

def reachApproxBFS(net: NeuralNetwork, inputSet: List[ProbStar], p_filter: float,
                   lp_solver: str = 'gurobi', pool: Optional[multiprocessing.Pool] = None, 
                   show: bool = True) -> Tuple[List[ProbStar], float]:
    """Compute Approximate Reachable Set layer-by-layer"""
    I = [probstar.clone() for probstar in inputSet]
    p_ignored = 0.0
    for i, layer in enumerate(net.layers):
        if show:
            print(f'================ Layer {i} =================')
            print(f'Computing layer {i} reachable set...')
        S = layer.reach(I, method='exact', lp_solver=lp_solver, pool=pool)
        if show:
            print(f'Number of probstars: {len(S)}')
            print(f'Filtering probstars whose probabilities < {p_filter}...')
        
        if pool is None:
            filtered = [filterProbStar((p_filter, s)) for s in S]
        else:
            filtered = pool.map(filterProbStar, zip([p_filter]*len(S), S))
        
        I = [f[0] for f in filtered if isinstance(f[0], ProbStar)]
        p_ignored += sum(f[1] for f in filtered)
        
        if show:
            print(f'Number of ignored probstars: {len(S) - len(I)}')
            print(f'Number of remaining probstars: {len(I)}')
        
        if not I:
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