
"""
ProbStar Reachability Analysis for RNNs
Qing Liu, 01/20/2026
"""
from scipy.io import loadmat
import os
import time
import mat73
import numpy as np
import matplotlib.pyplot as plt
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.RecurrentLayer import RecurrentLayer
from StarV.net.network import NeuralNetwork
from StarV.verifier.verifier import checkSafetyProbStar, reachExactBFS,reachApproxBFS
from StarV.util.plot import plot_probstar_signal,plot_probstar
from StarV.util.load_rnn import load_trained_CMAPSS_data,get_ProbStar_set_RNN,load_trained_params
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_,_OR_


def construct_input_probstar(engine_id, time_step,shifts):

    train_processed,test_processed,y_test= load_trained_CMAPSS_data()
    # select one engine unit data for reachability analysis
    engine_data = train_processed.loc[train_processed['unit_number'] == engine_id]
    print(f"engine {engine_id} data shape:{engine_data.shape}")
    print(f"engine {engine_id} data samples:{engine_data.head(10)}")       
    if engine_data["time_cycles"].max() < time_step:
        print(f"Engine {engine_id} has only {engine_data['time_cycles'].max()} time cycles, less than the specified time step {time_step}.")
        input_data = engine_data.values[:, 2:]
    else:
        input_data = engine_data.values[shifts-1:shifts+time_step-1, 2:]
        print("input_data shape:",input_data.shape)
        print("input_data:",input_data)

    # add standard gaussian noise to the input data for sertain feature, pressures, speed, temperature sensors
    all_noises = []
    noise_mean = 0.0
    pressure_noise_std = 0.005
    speed_noise_std = 0.0025
    temperature_noise_std = 0.0075
    temperature_noise = np.round(np.random.normal(noise_mean, temperature_noise_std),decimals=4)
    pressure_noise = np.round(np.random.normal(noise_mean, pressure_noise_std),decimals=4)
    speed_noise = np.round(np.random.normal(noise_mean, speed_noise_std),decimals=4)
    print(f"temperature_noise:{temperature_noise}, pressure_noise:{pressure_noise}, speed_noise:{speed_noise}")
    all_noises.append(temperature_noise)
    all_noises.append(pressure_noise)
    all_noises.append(speed_noise)


    feature_idx = []
    temperature_sensor_indices = [2,3,4]
    pressure_sensor_indices = [5,6]
    speed_sensor_indices = [7,8]

    feature_idx.append(temperature_sensor_indices)
    feature_idx.append(pressure_sensor_indices,)
    feature_idx.append(speed_sensor_indices)

    X = get_ProbStar_set_RNN(input_data,noises = all_noises,feature_idx= feature_idx)

    return X


def reachability_with_RNN(X, relu_method="exact", lp_solver="gurobi", RF=0.0, DR=0, p_filter=0.001, show=True):

    if relu_method not in ["exact", "approx", "relax", "basic"]:
        raise Exception(f"error: unknown relu_method: {relu_method}")

    Whx,bhx,Whh,bhh,Woh,boh,fc_w,fc_b = load_trained_params()

    # create NN
    L1 = RecurrentLayer(Whx,Whh,bhx,Woh,boh,bhh)
    
    mat =[]
    for i in range(len(fc_w)):
        W_fc = fc_w[i]
        b_fc = np.array(fc_b[i])
        paramas =[W_fc,b_fc]
        mat.append(paramas)

    L2 = FullyConnectedLayer(mat[0])
    L3 = ReLULayer()
    L4 = FullyConnectedLayer(mat[1])
    layers = [L1,L2,L3,L4]
    net = NeuralNetwork(layers=layers)


    RS = X
    RS,p_ignored = reachApproxBFS(net, RS, p_filter=0.001, lp_solver='gurobi', pool=None, show=True)
    final_layer_output = RS
    map_mat = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])

    # map to two dimenison for plotting
    All_map_sets=[]
    for i, S1 in enumerate(final_layer_output):
        print(f"\n Step {i}: number of sets = {len(S1)}")
        print(f"len of output set at each time step:{len(S1)}")
        print(f"type of output set at each time step:{type(S1)}")
        if len(S1) > 1:
            map_sets=[]
            for j, S2 in enumerate(S1):
                p = S2.estimateProbability()
                Map_set = S2.affineMap(map_mat)
                print(f" Set {i}{j}: \n nVars:{S2.nVars},\nC:{S2.C}{S2.C.shape}, \ndims:{S2.dim},\n V:{S2.V}, \nd:{S2.d}, \n probability = {p}")
                map_sets.append(Map_set)

            All_map_sets.append(map_sets)
        else:
            p = S1.estimateProbability()
            Map_set = S1.affineMap(map_mat)
            print(f" Set {i}{j}: \n nVars:{S1.nVars},\nC:\nC:{Map_set.C}{Map_set.C.shape}, \ndims:{Map_set.dim},\n V:{Map_set.V}, \nd:{Map_set.d}, \n probability = {p}")
            All_map_sets.append(Map_set)
 
if __name__ == "__main__":
    np.random.seed(25)
    for i in range(1,2):
        print(f"======================== Process the first 20 cycles of {i}th engine ========================")
        X = construct_input_probstar(engine_id=i, time_step=20,shifts=5)
        reachability_with_RNN(X, relu_method="exact", lp_solver="gurobi", RF=0.0, p_filter=0.001, show=True)
        
