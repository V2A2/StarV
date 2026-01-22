
"""
ProbStar Reachability Analysis for RNNs
Qing Liu, 01/20/2026
"""
from scipy.io import loadmat
import os
import mat73
import numpy as np
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.net.network import NeuralNetwork
from StarV.util.load_rnn import load_simple_rnn, get_Star_set,get_ProbStar_set

class RNN_ProbStar_reachability:

    def construct_input_probstar(self):
        