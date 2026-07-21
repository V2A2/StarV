
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
    # select one time step data for reachability analysis
    # engine_data = engine_data.reset_index(drop=True)
    # input_data = engine_data[:time_step].values[:, 2:] # remove unit_number and time_cycles columns
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
    # speed_noise = np.random.normal(noise_mean, speed_noise_std)
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
    # print("number of ProbStar set constructed for engine id {}: {}".format(engine_id, len(X)))

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

    # p_ignored = 0.0
    # for i, layer in enumerate(layers):
    #     if show:
    #         print('================ Layer {} ({}) ================='.format(i, layer.__class__.__name__))
    #     if isinstance(layer, RecurrentLayer):
    #         RS = layer.reach(RS, method=relu_method, lp_solver=lp_solver, RF=RF, DR=DR)
    #         print(f"type of each out set:{type(RS[0])}")
    #     else:
    #         print(f"len of RS from previous layer:{len(RS)}, type:{type(RS[0])}")
    #         RS = layer.reach(RS, method=relu_method, lp_solver=lp_solver, RF=RF, DR=DR, show=False)

    
    # Numlayers = len(layers)
    # Layer_RS = []
    # for j in range(0,Numlayers):
    #     print(f"=========processing layer{j+1}=============")
    #     layers[j].info()
    #     RS1 = net.layers[j].reach(RS, method=relu_method, lp_solver=lp_solver, pool=None, RF=RF, DR=DR)
    #     # print(f"RS at layer {j}:{RS1[0]}")
    #     for i in range(20):
    #         X = RS1
    #         print("\n number of Output sets after layer {} at each step{}: {}".format(j+1,i,len(X[i])))
    #     # # print("output set types after layer {} : {}".format(j+1,type(RS1)))
    #     # # print("output set[0] types after layer {} : {}{}".format(j+1,type(RS1[0]),RS1[0]))
    #     # print("num of output set[14]  after layer {} : {}".format(j+1,len(RS1[14])))
    #     Layer_RS.append(RS1)
    #     RS = RS1
    # final_layer_output = Layer_RS[-1]

    final_layer_output = RS
    # print(f"number of final layer output set:{len(final_layer_output)}")
    # map_mat = np.array([[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
    map_mat = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])

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
                # plot_probstar(Map_set)
                print(f" Set {i}{j}: \n nVars:{S2.nVars},\nC:{S2.C}{S2.C.shape}, \ndims:{S2.dim},\n V:{S2.V}, \nd:{S2.d}, \n probability = {p}")
                # print(f" Set {i}{j}: \n nVars:{Map_set.nVars},\nC:{Map_set.C.shape}, \ndims:{Map_set.dim},\n V:{Map_set.V}, \nd:{Map_set.d}, \n probability = {p}")
                map_sets.append(Map_set)
                # map_sets.append(S2)

            All_map_sets.append(map_sets)
        else:
            p = S1.estimateProbability()
            Map_set = S1.affineMap(map_mat)
            # plot_probstar(Map_set)
            print(f" Set {i}{j}: \n nVars:{S1.nVars},\nC:\nC:{Map_set.C}{Map_set.C.shape}, \ndims:{Map_set.dim},\n V:{Map_set.V}, \nd:{Map_set.d}, \n probability = {p}")
            # print(f" Set {i}: \n nVars:{Map_set.nVars},\nC:{Map_set.C.shape}, \ndims:{Map_set.dim},\n V:{Map_set.V}\nd:{Map_set.d}, \n probability = {p}")
            All_map_sets.append(Map_set)
    
    # plot map sets
    # print(f"====plot original map reachable sets====")
    # S0 =All_map_sets[0][0]
    # S1 =All_map_sets[0][1]
    # print(f"S0:{S0} \n S1:{S1}")
    # S3 =[]
    # S3.append(S0)
    # S3.append(S1)
    # plot_probstar_signal(S3)

        
    # print(f"\n ====plot concat map reachable sets====")
    # combine_set_1 = S0.Combine(S1)
    # S4 =[]
    # S4.append(combine_set_1)
    # plot_probstar(combine_set_1)
    # plot_probstar(combine_set_2)
    # plot_probstar_signal(S4)

    # plot_probstar(All_map_sets[0])

    
    # ===================================== with CX <= d
    # verify output reachable sets 
    # unsafe_mat = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0]])
    # unsafe_vec = np.array([-5])

    # all_check_sets=[]
    # all_check_prob=[]
    # for i, S1 in enumerate(All_map_sets):
    #     # print(f"type of S1:{type(S1)}")
    #     # print(f" S1:{S1}")
    #     P = []
    #     prob = []
    #     for j,S2 in enumerate(S1):
    #         if len(S1) <= 1:
    #             continue
    #         else:
    #             concat_set = S1[0]
    #             for j in range(1, len(S1)):
    #                 concat_set = concat_set.concatenate(S1[j])   
    #             S2 = concat_set
    #         P1, prob1 = checkSafetyProbStar(unsafe_mat, unsafe_vec, S2)
    #         if isinstance(P1, ProbStar):
    #             print(f"prob1 of S{i}{j}:{prob1}")
    #             P.append(P1)
    #             prob.append(prob1)
    #         else:
    #             print(f"S{i}{j} is an empty set, prob = 0.0")

    #     if len(P) != 0:
    #         all_check_sets.append(P)
    #         all_check_prob.append(prob)


    # create temporal specifications
    # AND = _AND_()
    # OR = _OR_()                                        
    # lb = _LeftBracket_()
    # rb = _RightBracket_()

    # A1 = np.array([-1., 0.])
    # b1 = np.array([-20])
    # P1 = AtomicPredicate(A1,b1)

    # A2 = np.array([0,-1])
    # b2 = np.array([-15])
    # P2 = AtomicPredicate(A2,b2)

    # EVOT =_EVENTUALLY_(0,20)
    # AWOT = _ALWAYS_(11,15)
    # EVOT1 =_EVENTUALLY_(5,15)
    # AWOT1 = _ALWAYS_(0,5)
    

    # specs =[]
    # spec = Formula([EVOT,P1])
    # spec1 = Formula([AWOT,lb,P2,rb])
    # spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P2,rb,rb])
    # specs =[spec,spec1,spec2]

    # checking_time = []
    # data=[]

    # # verification using ProbSatrTL
    # print(f"\n========Start Verification TL===================")
    # for i in range(0,len(specs)):
    #     check_start = time.time()
    #     spec = specs[i]
    #     print('\n==================Specification{}====================: '.format(i))
    #     spec.print()
    #     DNF_spec = spec.getDynamicFormula()
    #     print(f"====== Dynamic Formula=======\n {DNF_spec.print()}")
    #     Nadnf = DNF_spec.length
    #     print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
    #     _,p_max, p_min,Ncdnf = DNF_spec.evaluate(All_map_sets)
    #     end = time.time()
    #     checking_time = end -check_start 
    #     print("p_min:",p_min)
    #     print("p_max:",p_max) 
    #     print(f"check_TL_spces_time:{checking_time}")
        # verify_time=checking_time + reach_time_duration    

    
if __name__ == "__main__":
    np.random.seed(25)
    for i in range(1,2):
        print(f"======================== Process the first 20 cycles of {i}th engine ========================")
        X = construct_input_probstar(engine_id=i, time_step=20,shifts=5)
        reachability_with_RNN(X, relu_method="exact", lp_solver="gurobi", RF=0.0, p_filter=0.001, show=True)
        
