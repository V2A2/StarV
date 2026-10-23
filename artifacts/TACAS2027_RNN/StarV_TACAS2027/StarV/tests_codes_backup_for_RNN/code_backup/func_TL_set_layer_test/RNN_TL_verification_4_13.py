
"""
ProbStar Reachability Analysis and Verification for RNNs
Two case study: CMAPSS and LIMO
Qing Liu, 01/20/2026
"""
from scipy.io import loadmat
import os
import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.RecurrentLayer import RecurrentLayer
from StarV.net.network import NeuralNetwork
from StarV.set.probstar import ProbStar
from StarV.verifier.verifier import checkSafetyProbStar, reachExactBFS,reachApproxBFS
from StarV.util.plot import plot_probstar_signal,plot_probstar
from StarV.util.load_rnn import load_trained_CMAPSS_data, load_trained_params_CMAPSS,load_trained_params_LIMO,load_LIMO_data,get_input_ProbStar_CMAPSS,get_input_ProbStar_LIMO
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_,_OR_

np.set_printoptions(precision=12, suppress=False)
# P_BRANCH_FILTER = 0.00000000001
# P_STEP_FILTER = 0.0000001

def check_sat_on_branch_for_RNN(*args):
    """evaluate one RNN branch against one temporal spec."""
    if isinstance(args[0], tuple):
        args1 = args[0]
    else:
        args1 = args
    spec = args1[0]      # user-define spec
    branch = args1[1]     # ProbStar signal of one branch
    if not isinstance(branch, list):
        raise RuntimeError('error: each branch signal should be a list')
    
    DNF_spec = spec.getDynamicFormula()
    _, p_max, p_min, p_ig, cdnf_len = DNF_spec.evaluate_for_RNN(branch)
    return p_max, p_min, p_ig, cdnf_len

def construct_CMAPSS_input_probstar(time_step, shifts,engine_id):

    ### load CMAPSS data ###

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
        input_engine_data = engine_data.values[:, 2:]
    else:
        input_engine_data = engine_data.values[shifts-1:shifts+(time_step*2)-1, 2:]  
        print("input_engine_data shape:",input_engine_data.shape)
        print("input_engine_data:",input_engine_data)

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


    X = get_input_ProbStar_CMAPSS(input_engine_data, noises=all_noises, feature_idx=feature_idx)


    return X

def construct_LIMO_input_probstar(time_step, shifts,engine_id=None):

    ### load LIMO data ###

    processed_input_data,processed_target_data = load_LIMO_data()
    
    # select 40 steps data to create 20 window, each window is used to predict next 20 trajectory states
    input_LIMO_data = processed_input_data[shifts-1:shifts+(time_step*2)-1,:]
    input_target_LIMO_data = input_LIMO_data[time_step:,:5]
    # print("input_LIMO_data:",input_LIMO_data)
    # print("input_LIMO_data_shape:",input_LIMO_data.shape)

    # print("input_target_LIMO_data:",input_target_LIMO_data)
    # print("input_target_LIMO_data_shape:",input_target_LIMO_data.shape)

    X = get_input_ProbStar_LIMO(input_data=input_LIMO_data,noise=0.001)

    return X

def TL_verify_CMAPSS(time_step, shifts, engine_id, numCores=None):
    print(f"======================== Start CMAPSS reachability and verification for engine {engine_id} ========================")
    X = construct_CMAPSS_input_probstar(engine_id=engine_id, time_step=time_step, shifts=shifts)

    if numCores is None or numCores < 1:
        numCores = 1

    # Exact branch-based reachability
    Whx, bhx, Whh, bhh, Woh, boh, fc_w, fc_b = load_trained_params_CMAPSS()
    L1 = RecurrentLayer(Whx, Whh, bhx, Woh, boh, bhh)
    mat = []
    for i in range(len(fc_w)):
        mat.append([fc_w[i], np.array(fc_b[i])])
    # The first FC layer is already encoded in RecurrentLayer via (Woh, boh).
    # Remaining post layers from fc_w: ReLU -> FC(mat[0]) -> ReLU -> FC(mat[1]) -> ReLU -> FC(mat[2]).
    L2 = ReLULayer()
    L3 = FullyConnectedLayer(mat[0])
    L4 = ReLULayer()
    L5 = FullyConnectedLayer(mat[1])
    L6 = ReLULayer()
    L7 = FullyConnectedLayer(mat[2])

    v_r = time.time()
    branches, _, p_ignored = L1.reachExactBranches(
        X,
        post_layers=[L2, L3, L4, L5, L6, L7],
        lp_solver="gurobi",
        pool=None,
        p_filter = 1e-10,
        show=True,
    )
    t_r = time.time() - v_r
    print(f"Reachability analysis time: {t_r:.4f} seconds")
    print(f"total branches after reachability:{len(branches)}")

    # TL verification
    AND = _AND_()
    OR = _OR_()
    lb = _LeftBracket_()
    rb = _RightBracket_()

    # Current CMAPSS output dimension is 6:
    # [var_7, var_11, var_12, var_15, var_20, var_21]


    # P1: var_12 <= 0.42
    A1 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    b1 = np.array([0.42])
    P1 = AtomicPredicate(A1, b1)

    # P2: var_21 <= 0.45
    A2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    b2 = np.array([0.45])
    P2 = AtomicPredicate(A2, b2)

    # P3: var_15 >= 0.30  -> -var_15 <= -0.30
    A3 = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    b3 = np.array([-0.30])
    P3 = AtomicPredicate(A3, b3)

    # P4, P5: |var_21 - var_20| <= 0.22
    A4 = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 1.0])  # var_21 - var_20 <= 0.22
    b4 = np.array([0.22])
    P4 = AtomicPredicate(A4, b4)
    A5 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0])  # var_20 - var_21 <= 0.22
    b5 = np.array([0.22])
    P5 = AtomicPredicate(A5, b5)

    # P6, P7: |var_12 - var_7| <= 0.18
    A6 = np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # var_12 - var_7 <= 0.18
    b6 = np.array([0.2])
    P6 = AtomicPredicate(A6, b6)
    A7 = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0])  # var_7 - var_12 <= 0.18
    b7 = np.array([0.2])
    P7 = AtomicPredicate(A7, b7)

    # P8, P9: 0.15 <= var_11 <= 0.70
    A8 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])   # var_11 <= 0.70
    b8 = np.array([0.70])
    P8 = AtomicPredicate(A8, b8)
    A9 = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])  # var_11 >= 0.15
    b9 = np.array([-0.15])
    P9 = AtomicPredicate(A9, b9)

    EVOT = _EVENTUALLY_(0, 10)
    AWOT = _ALWAYS_(0, 15)
    EVOT1 = _EVENTUALLY_(5, 15)
    AWOT1 = _ALWAYS_(0, 5)

    # spec0: eventually both var_12 and var_21 are in low zone
    spec = Formula([EVOT, lb, P1, AND, P2, rb])
    # spec1: always keep var_11 band and (var_20,var_21) coupling
    spec1 = Formula([AWOT, lb, P8, AND, P9, AND, P4, AND, P5, rb])
    # spec2: eventually var_15 is high and then |var_12-var_7| stays small for next steps
    spec2 = Formula([EVOT1, lb, P3, AND, lb, AWOT1, lb, P6, AND, P7, rb, rb, rb])

    specs = [spec, spec1, spec2]

    t_c = 0.0
    p_SAT_MIN = []
    p_SAT_MAX = []
    p_IG_approx = []
    for k, spec in enumerate(specs):
        print(f"\n==================Verify Spec {k}====================")
        p_MIN_spec = 0.0
        p_MAX_spec = 0.0
        spec.print()
        start_time = time.time()
        p_total_max = []
        p_total_min = []
        p_ignored_approx = []

        if numCores > 1:
            pool = multiprocessing.Pool(numCores)   
            print(f"Using multiprocessing for TL checking with numCores={numCores}")
            with multiprocessing.Pool(numCores) as mp_pool:
                results = mp_pool.map(
                    check_sat_on_branch_for_RNN,
                    zip([spec] * len(branches), branches)
                )
            for r in results:
                p_max = r[0]
                p_min = r[1]
                p_ig = r[2]
                p_total_max.append(p_max)
                p_total_min.append(p_min)
                p_ignored_approx.append(p_ig)
        else:
            for i, sig in enumerate(branches):
                if not isinstance(sig, list):
                    raise RuntimeError('error: each branch signal should be a list')
                print(f"=====================Checking branch {i} =============")
                p_max, p_min, p_ig, _ = check_sat_on_branch_for_RNN(spec, sig)
                # print(f"Branch {i} p_max: {p_max}, p_min: {p_min}, p_ig_approx: {p_ig}")
                p_total_max.append(p_max)
                p_total_min.append(p_min)
                p_ignored_approx.append(p_ig)

        p_MAX_spec = sum(p_total_max)
        p_MIN_spec = sum(p_total_min)
        p_IG_approx_spec = sum(p_ignored_approx)
        p_MAX_spec = p_MAX_spec + p_ignored

        p_SAT_MIN.append(p_MIN_spec)
        p_SAT_MAX.append(p_MAX_spec)
        p_IG_approx.append(p_IG_approx_spec)

        t_c_b = time.time() - start_time
        print(f"Checking time for Spec {k}: {t_c_b:.4f} seconds")
        t_c += t_c_b
        print(f"p_total_max_spec: {p_MAX_spec}")
        print(f"p_total_min_spec: {p_MIN_spec}")
        print(f"p_ignored during checking using estimate probability: {p_IG_approx_spec}")

    print(f"p_ignored during reachability using filter: {p_ignored}")
    print(f"Total checking time for all specs: {t_c:.4f} seconds")
    t_v = t_r + t_c
    print(f"Total verification time (reachability + checking): {t_v:.4f} seconds")

    return p_SAT_MAX, p_SAT_MIN, p_IG_approx, p_ignored

def map_branch_signals(branch_signals, map_mat=None, map_vec=None):
    """Apply an affine map to every ProbStar in every branch signal."""
    mapped_branches = []
    for sig in branch_signals:
        sig_map=[]
        for item in sig:
            if isinstance(item, tuple) and len(item) == 2:
                t, S = item
                if S is None:
                    sig_map.append((t, None))
                else:
                    S1 = S.affineMap(map_mat, map_vec)
                    sig_map.append((t, S1))
            elif item is None:
                sig_map.append(None)
            else:
                S1 = item.affineMap(map_mat, map_vec)
                sig_map.append(S1)
        mapped_branches.append(sig_map)
    return mapped_branches

def TL_verify_LIMO(time_step, shifts, numCores=None):

    print(f"======================== Start LIMO reachability and verification ========================")
    X = construct_LIMO_input_probstar(engine_id=engine_id, time_step=time_step, shifts=shifts)

    if numCores is None or numCores < 1:
        numCores = 1


    # Exact branch-based reachability
    Whx, bhx, Whh, bhh, Woh, boh, fc_w, fc_b = load_trained_params_LIMO()
    L1 = RecurrentLayer(Whx, Whh, bhx, Woh, boh, bhh)
    mat = []
    for i in range(len(fc_w)):
        mat.append([fc_w[i], np.array(fc_b[i])])
    L2 = ReLULayer()
    L3 = FullyConnectedLayer(mat[0])
    L4 = ReLULayer()
    L5 = FullyConnectedLayer(mat[1])
    # L5 = ReLULayer()
    # L6 = FullyConnectedLayer(mat[2])
    # L7 = ReLULayer()

    v_r = time.time()
    branches,_,p_ignored = L1.reachExactBranches(
        X,
        post_layers=[L2,L3,L4,L5],
        lp_solver="gurobi",
        pool=None,
        p_filter= 1e-10,
        show=True,
    )
    t_r = time.time() - v_r
    print(f"Reachability analysis time: {t_r:.4f} seconds")
    # # print(f"ignored probability during reachability when constructing branches: {p_ignored}")
    # print(f"total branches after reachability:{len(branches)}")
    # for i in range(len(branches)):
    #     print("Branch {} type: {}".format(i,type(branches[i])))
    #     print(f"Branch {i} number of sets: {len(branches[i])}")
    #     if len(branches[i])>1:
    #         for j in range(len(branches[i])):
    #             print(f"Branch {i} set {j}: {branches[i][j]}")

        
    # TL verification
    # LIMO output state order from load_LIMO_data():
    # [x, y, yaw, v, w]
    # idx: 0  1   2    3  4
    
    AND = _AND_()
    OR = _OR_()
    lb = _LeftBracket_()
    rb = _RightBracket_()

    # P1: y >= -3  
    A1 = np.array([0,-1.,0,0,0])
    b1 = np.array([3])
    P1 = AtomicPredicate(A1, b1)

    # P2: y <= 3
    A2 = np.array([0,1,0,0,0])
    b2 = np.array([3])
    P2 = AtomicPredicate(A2, b2)

    EVOT = _EVENTUALLY_(4, 9)
    AWOT = _ALWAYS_(11, 15)
    EVOT1 = _EVENTUALLY_(5, 15)
    AWOT1 = _ALWAYS_(0, 5)

    # spec: eventually (within [5,10]) satisfy both P1 and P2.
    spec = Formula([EVOT, lb,P1, AND,P2,rb])
    # spec1: always (within [11,15]) satisfy both P1 and P2.
    spec1 = Formula([AWOT, lb, P1,AND,P2, rb])
    # spec2: eventually (within [5,15]) satisfy P1 OR (always within [0,5] satisfy P2).
    spec2 = Formula([EVOT1, lb, P1, OR, lb, AWOT1, P2, rb, rb])

    specs = [spec]

    map_mat = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
    map_vec = None
    # mapped_branch_signals = map_branch_signals(branches,map_mat=map_mat, map_vec=map_vec)

    t_c = 0.0
    p_SAT_MIN = []
    p_SAT_MAX=[]
    p_IG_approx= []
    for k, spec in enumerate(specs):
        print(f"\n==================Verify Spec {k}====================")
        p_MIN_spec = 0.0
        p_MAX_spec = 0.0
        spec.print()
        start_time = time.time()
        p_total_max = []
        p_total_min = []
        p_ignored_approx = []

        if numCores > 1:
            pool = multiprocessing.Pool(numCores)
            print(f"=====================Checking branch {i} =============")
            print(f"Using multiprocessing for TL checking with numCores={numCores}")
            with multiprocessing.Pool(numCores) as pool:
                results = pool.map(
                    check_sat_on_branch_for_RNN,
                    zip([spec]*len(branches), branches)
                )
            for r in results:
                p_max = r[0]
                p_min = r[1]
                p_ig = r[2]
                p_total_max.append(p_max)
                p_total_min.append(p_min)
                p_ignored_approx.append(p_ig)
        else:
            for i, sig in enumerate(branches):
                if not isinstance(sig, list):
                    raise RuntimeError('error: each branch signal should be a list')
                print(f"=====================Checking branch {i} =============")
                p_max, p_min, p_ig, _ = check_sat_on_branch_for_RNN(spec, sig)
                # print(f"Branch {i} p_max: {p_max}, p_min: {p_min}, p_ig_approx: {p_ig}")
                p_total_max.append(p_max)
                p_total_min.append(p_min)
                p_ignored_approx.append(p_ig)

        p_MAX_spec = sum(p_total_max)
        p_MIN_spec = sum(p_total_min)
        p_IG_approx_spec = sum(p_ignored_approx)

        p_MAX_spec = p_MAX_spec + p_ignored

        p_SAT_MIN.append(p_MIN_spec)
        p_SAT_MAX.append(p_MAX_spec)
        p_IG_approx.append(p_IG_approx_spec)

        t_c_b = time.time() - start_time
        print(f"Checking time for Spec {k}: {t_c_b:.4f} seconds")
        t_c += t_c_b
        
        print(f"p_total_max_spec: {p_MAX_spec}")
        print(f"p_total_min_spec: {p_MIN_spec}")
        print(f"p_ignored during checking using estimate probability: {p_IG_approx_spec}")


    print(f"p_ignored during reachability using filter: {p_ignored}")
    print(f"Total checking time for all specs: {t_c:.4f} seconds")
    t_v = t_r + t_c
    print(f"Total verification time (reachability + checking): {t_v:.4f} seconds")

    return p_SAT_MAX, p_SAT_MIN, p_IG_approx, p_ignored
    
if __name__ == "__main__":
    np.random.seed(25)
    time_step = 20
    engine_id =1
    shifts =1
    num_cores = 1
    # TL_verify_CMAPSS(engine_id= engine_id,time_step=time_step,shifts=shifts)
    # construct_LIMO_input_probstar(time_step,shifts)
    # p_SAT_MAX_LIMO, p_SAT_MIN_LIMO, p_IG_approx_LIMO, p_ignored_LIMO =TL_verify_LIMO(time_step, shifts, numCores=num_cores)
    p_SAT_MAX_CMAPSS, p_SAT_MIN_CMAPSS, p_IG_approx_CMAPSS, p_ignored_CMAPSS =TL_verify_CMAPSS(time_step= time_step, engine_id= engine_id, shifts=shifts, numCores=num_cores)
   


# verify phi3
# approximate with p_filyer = 1e-10:
# p_total_max: 0.6459390775840915
# p_total_min: 0.6459390012033069

# exact with p_filter = 0.0:
# p_total_max: 0.6521874936468794
# p_total_min: 0.652187419780777
