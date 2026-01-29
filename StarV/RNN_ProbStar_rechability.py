
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
from StarV.layer.RecurrentLayer import RecurrentLayer
from StarV.net.network import NeuralNetwork
from StarV.util.load_rnn import load_trained_CMAPSS_data,get_ProbStar_set_RNN,load_trained_params


def construct_input_probstar(engine_id, time_step):

    train_processed,test_processed,y_test= load_trained_CMAPSS_data()
    # select one engine unit data for reachability analysis
    engine_data = train_processed.loc[train_processed['unit_number'] == engine_id]
    print("engine_data shape:",engine_data.shape)
    print("engine_data samples:",engine_data.head(5))       
    # select one time step data for reachability analysis
    # engine_data = engine_data.reset_index(drop=True)
    # input_data = engine_data[:time_step].values[:, 2:] # remove unit_number and time_cycles columns
    if engine_data["time_cycles"].max() < time_step:
        print(f"Engine {engine_id} has only {engine_data['time_cycles'].max()} time cycles, less than the specified time step {time_step}.")
        input_data = engine_data.values[:, 2:]
    else:
        input_data = engine_data[engine_data["time_cycles"] <= time_step].values[:, 2:]
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
    # print("number of ProbStar set constructed for engine id {}: {}".format(engine_id, len(X)))

    return X

def reachability_with_RNN(X):

    Whx,bhx,Whh,bhh,Woh,boh,fc_w,fc_b = load_trained_params()

    # create NN
    L1 = RecurrentLayer(Whx,Whh,bhx,Woh,boh,bhh)
    
    mat =[]
    for i in range(len(fc_w)):
        W_fc = fc_w[i]
        # print(f"w_fc{i}:",W_fc)
        # print(f"w_fc{i} shape:",W_fc.shape)
        b_fc = np.array(fc_b[i])
        # print("b_fc:",b_fc)
        # print(f"b_fc{i} shape:",b_fc.shape)
        paramas =[W_fc,b_fc]
        mat.append(paramas)
        # print(f"mat{i}:{mat[i]}")

    L2 = FullyConnectedLayer(mat[0])
    L3 = ReLULayer()
    L4 = FullyConnectedLayer(mat[1])
    L5 = ReLULayer()
    L6 = FullyConnectedLayer(mat[2])
    # L7 = ReLULayer()


    layers = [L1,L2,L3,L4,L5,L6]
    net = NeuralNetwork(layers=layers)


    Numlayers = len(layers)
    Layer_RS = []
    RS = X
    for j in range(0,Numlayers):
        print(f"=========processing layer{j+1}=============")
        # layers[j].info()
        RS1 = net.layers[j].reach(RS, method = "exact", lp_solver='gurobi', pool=None, RF=0.0, DR=0)
        # print("\n number of Output sets after layer {} : {}".format(j+1,len(RS1)))
        # print("output set types after layer {} : {}".format(j+1,type(RS1)))
        # print("output set[0] types after layer {} : {}{}".format(j+1,type(RS1[0]),RS1[0]))
        # print("num of output set[0]  after layer {} : {}".format(j+1,len(RS1[0])))
        Layer_RS.append(RS1)
        RS = RS1
    final_layer_output = Layer_RS[-1]
    print("len of final output sets:",len(final_layer_output))
    # for i in range(len(X)):
    #     # print("final_output_set:",final_layer_output[i][0])
    #     print("final_output_set_prob:",final_layer_output[i][0].estimateProbability())
   
    for i, step in enumerate(final_layer_output):
        print(f"\n Step {i}: number of sets = {len(step)}")
        for j, S in enumerate(step):
            p = S.estimateProbability()
            print(f" Set {j}:{step[j]}, \n nVars:{step[j].nVars} ,probability = {p}")

if __name__ == "__main__":
    np.random.seed(25)
    X=construct_input_probstar(engine_id=1, time_step=25)
    reachability_with_RNN(X)
        