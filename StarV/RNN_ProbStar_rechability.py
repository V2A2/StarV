
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
from StarV.util.load_rnn import load_CMAPSS_data

class RNN_ProbStar_reachability:

    def construct_input_probstar(self,engine_id, time_step):

        train_processed,test_processed,y_test = load_CMAPSS_data()
        # select one engine unit data for reachability analysis
        engine_data = test_processed.loc[test_processed['unit_number'] == engine_id]
        print("engine_data shape:",engine_data.shape)
        print("engine_data samples:",engine_data.head(5))       
        # select one time step data for reachability analysis
        # engine_data = engine_data.reset_index(drop=True)
        # input_data = engine_data[:time_step].values[:, 2:] # remove unit_number and time_cycles columns
        input_data = engine_data[engine_data["time_cycles"] <= 50].values[:, 2:]
        print("input_data shape:",input_data.shape)
        print("input_data:",input_data)

if __name__ == "__main__":
    RNN_reach = RNN_ProbStar_reachability()
    RNN_reach.construct_input_probstar(engine_id=1, time_step=50)
        