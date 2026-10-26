
"""
ProbStar Reachability Analysis and Verification for RNNs
Two case study: CMAPSS and LIMO
Qing Liu, 01/20/2026
"""
import os
import time
import multiprocessing
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.RecurrentLayer import RecurrentLayer
from StarV.net.network import NeuralNetwork
from StarV.set.probstar import ProbStar
from StarV.verifier.verifier import checkSafetyProbStar, reachExactBFS,reachApproxBFS
from StarV.util.plot import plot_probstar_signal,plot_probstar
from StarV.util.load_rnn import load_trained_CMAPSS_data, load_trained_params_CMAPSS,load_trained_params_LIMO,load_LIMO_data,get_input_set_CMAPSS,get_input_set_LIMO
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_,_OR_

np.set_printoptions(precision=12, suppress=False)


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
    rng = np.random.default_rng(25)
    temperature_noise = np.round(rng.normal(noise_mean, temperature_noise_std),decimals=4)
    pressure_noise = np.round(rng.normal(noise_mean, pressure_noise_std),decimals=4)
    speed_noise = np.round(rng.normal(noise_mean, speed_noise_std),decimals=4)
    # speed_noise = np.random.normal(noise_mean, speed_noise_std)
    # print(f"temperature_noise:{temperature_noise}, pressure_noise:{pressure_noise}, speed_noise:{speed_noise}")

    all_noises.append(temperature_noise)
    all_noises.append(0.0)
    all_noises.append(speed_noise)


    feature_idx = []
    temperature_sensor_indices = [2,3,4]
    pressure_sensor_indices = [5,6]
    speed_sensor_indices = [7,8]

    feature_idx.append(temperature_sensor_indices)
    feature_idx.append(pressure_sensor_indices,)
    feature_idx.append(speed_sensor_indices)


    X = get_input_set_CMAPSS(input_engine_data, noises=all_noises, feature_idx=feature_idx,set = "ProbStar")
    print(f"probability of the initial ProbStar set : {X[0].estimateProbability()}")


    return X

def construct_LIMO_input_probstar(time_step, shifts):

    ### load LIMO data ###

    processed_input_data,processed_target_data = load_LIMO_data()
    
    # select 40 steps data to create 20 window, each window is used to predict next 20 trajectory states
    input_LIMO_data = processed_input_data[shifts-1:shifts+(time_step*2)-1,:]
    input_target_LIMO_data = input_LIMO_data[time_step:,:5]
    # print("input_LIMO_data:",input_LIMO_data)
    # print("input_LIMO_data_shape:",input_LIMO_data.shape)

    # print("input_target_LIMO_data:",input_target_LIMO_data)
    # print("input_target_LIMO_data_shape:",input_target_LIMO_data.shape)
    rng = np.random.default_rng(25)
    noise = np.round(rng.normal(0.0, 0.0125),decimals=4)

    X = get_input_set_LIMO(input_data=input_LIMO_data,noise=noise,set = "ProbStar")
    print(f"probability of the initial ProbStar set : {X[0].estimateProbability()}")
    return X

def create_specs_CMAPSS(time_horizon=None):
    """CMAPSS specs for 6-D outputs: [var_7, var_11, var_12, var_15, var_20, var_21]."""

    AND = _AND_()
    OR = _OR_()
    lb = _LeftBracket_()
    rb = _RightBracket_()
    T = time_horizon

    # P1: var_12 <= 0.60
    P1 = AtomicPredicate(np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]), np.array([-0.65]))
    P11= AtomicPredicate(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.array([0.65]))
    # P2: var_21 <= 0.6
    P2 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), np.array([0.6]))
    P21 =  AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]), np.array([-0.6]))
    # P3: var_15 >= 0.30  
    P3 = AtomicPredicate(np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]), np.array([-0.30]))
    P31 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), np.array([0.30]))
    # P4, P5: |var_21 - var_20| <= 0.3
    P4 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, -1.0, 1.0]), np.array([0.3]))
    P5 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0]), np.array([0.3]))
    P41 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, -1.0, 1.0]), np.array([-0.3]))
    P51 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0]), np.array([-0.3]))
    # P6, P7: |var_12 - var_7| <= 0.20
    P6 = AtomicPredicate(np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.array([0.20]))
    P7 = AtomicPredicate(np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0]), np.array([0.20]))
    P61 = AtomicPredicate(np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0]), np.array([-0.20]))
    P71 = AtomicPredicate(np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.array([-0.20]))
   # P8, P9: 0.15 <= var_11 <= 0.45
    P8 = AtomicPredicate(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.45]))
    P9 = AtomicPredicate(np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]), np.array([-0.15]))
    P81 = AtomicPredicate(np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]), np.array([-0.45]))
    P91 = AtomicPredicate(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.15]))

    # Use fixed temporal windows (no bounded-interval clamping).
    EVOT = _EVENTUALLY_(0, T)
    AWOT = _ALWAYS_(0, T)
    EVOT1 = _EVENTUALLY_(0, 5)
    AWOT1 = _ALWAYS_(0, 5)

    phi1 = Formula([EVOT, lb, P1, rb])
    phi11 = Formula([AWOT, lb, P11, rb])

    phi2 = Formula([EVOT, lb, P2, OR, P3, rb])
    phi21 = Formula([AWOT, lb, P21, AND, P31,rb])


    phi3 = Formula([EVOT, lb, P8, AND, lb, AWOT1, P9, rb,rb])
    
    phi31 = Formula([AWOT, lb, P81, OR, lb, EVOT1, P91, rb,rb])
    # verify phi31 is very expensive in getting dynamic formula
    # we transform phi31 to an equivalent form with simpler dynamic formula, which is phi3, and verify phi3 instead of phi31.
    # much cheaper to verify

    # phi3 = Formula([EVOT, lb, P9, AND, lb, AWOT1, P9, rb,rb])
    
    # phi31 = Formula([AWOT, lb, P9, OR, lb, EVOT1, P91, rb,rb])

    
    phi4 = Formula([EVOT, lb, P4, AND,P5, lb, EVOT1, P4, AND, P5, rb, rb])

    specs = [phi1,phi11,phi2,phi21,phi3,phi4]
    names = [rf'$\varphi_{i + 1}$' for i in range(len(specs))]

    return specs, names

def create_specs_LIMO(time_horizon=None):
    """LIMO specs for 5-D outputs: [x, y, yaw, v, w]."""

    AND = _AND_()
    OR = _OR_()
    lb = _LeftBracket_()
    rb = _RightBracket_()
    T = time_horizon
    # Current LIMO output dimension is 5:
    # [x, y, yaw, v, w]

    # P1: x >= 0.55  
    P1 = AtomicPredicate(np.array([-1.0, 0.0, 0.0, 0.0, 0.0]), np.array([-0.55]))
    P11 = AtomicPredicate(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.55]))
    # P3: y >= 0.85
    P3 = AtomicPredicate(np.array([0.0, -1.0, 0.0, 0.0, 0.0]), np.array([-0.85]))
    P31 = AtomicPredicate(np.array([0.0, 1.0, 0.0, 0.0, 0.0]), np.array([0.85]))
    # P4: w >= 0.6
    P4 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, -1.0]), np.array([-0.60]))
    P41 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 1.0]), np.array([0.60]))
    # P5: w <= 0.95
    P5 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 1.0]), np.array([0.95]))
    P51 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, -1.0]), np.array([-0.95]))
    # P7: v >= 0.45
    P7 = AtomicPredicate(np.array([0.0, 0.0, 0.0, -1.0, 0.0]), np.array([-0.45]))
    P71 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 1.0, 0.0]), np.array([0.45]))
    # P8: v <= 0.65
    P8 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 1.0, 0.0]), np.array([0.65]))
    P81 = AtomicPredicate(np.array([0.0, 0.0, 0.0, -1.0, 0.0]), np.array([-0.65]))


    # Use fixed temporal windows (no bounded-interval clamping).
    EVOT = _EVENTUALLY_(0,T)
    AWOT = _ALWAYS_(0, T)
    EVOT1 = _EVENTUALLY_(0,5)
    AWOT1 = _ALWAYS_(0, 5)

    phi1 = Formula([EVOT, lb, P1, rb])

    phi11 = Formula([AWOT, lb, P11, rb])

    phi2 = Formula([EVOT, lb, P1, OR, P3, rb])

    phi21 = Formula([AWOT, lb, P11, AND, P31, rb])

    phi3 = Formula([EVOT, lb, P5, AND, lb, AWOT1, P4, rb,rb])
    
    phi31 = Formula([AWOT, lb, P51, OR, lb, EVOT1, P41, rb,rb])

    
    phi4 = Formula([EVOT, lb, P7, AND,P8, lb, EVOT1, P7, AND, P8, rb, rb])


    specs = [phi1,phi11,phi2,phi21,phi3,phi4]
    names = [rf'$\varphi_{i + 1}$' for i in range(len(specs))]
    return specs, names

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

def evaluate_specs_over_branches(branches, specs, spec_names, p_ignored, numCores=1, verbose=True):
    """Evaluate all specs on branches and return per-spec SAT bounds and checking times."""

    p_SAT_MIN = []
    p_SAT_MAX = []
    p_IG_approx = []
    t_c_per_spec = []
    conserv_list = []
    consist_list=[]

    for k, spec in enumerate(specs):
        if verbose:
            print(f"\n==================Verify Spec {k} ({spec_names[k]})====================")
            spec.print()

        start_time = time.time()
        p_total_max = []
        p_total_min = []
        p_ignored_approx = []

        if numCores > 1 and len(branches) > 0:
            if verbose:
                print(f"Using multiprocessing for TL checking with numCores={numCores}")
            # if k==0:
            #     print(f" \n ProbStat set {[b.V for b in branches[0]]}\n")
            with multiprocessing.Pool(numCores) as mp_pool:
                results = mp_pool.map(
                    check_sat_on_branch_for_RNN,
                    zip([spec] * len(branches), branches)
                )
            for r in results:
                print(f"\n\nResult from one branch: p_max={r[0]}, p_min={r[1]}, p_ig={r[2]}, cdnf_len={r[3]}")
                p_total_max.append(r[0])
                p_total_min.append(r[1])
                p_ignored_approx.append(r[2])
        else:
            for i, sig in enumerate(branches):
                if not isinstance(sig, list):
                    raise RuntimeError('error: each branch signal should be a list')
                if verbose:
                    print(f"=====================Checking branch {i} =============")
                # print(f"branch{i}, {[s.V for s in sig]}")
                p_max, p_min, p_ig, _ = check_sat_on_branch_for_RNN(spec, sig)
                p_total_max.append(p_max)
                p_total_min.append(p_min)
                p_ignored_approx.append(p_ig)

        p_max_spec = sum(p_total_max) + p_ignored
        p_min_spec = sum(p_total_min)
        p_ig_spec = sum(p_ignored_approx)
        t_c_spec = time.time() - start_time

        p_SAT_MAX.append(p_max_spec)
        p_SAT_MIN.append(p_min_spec)
        p_IG_approx.append(p_ig_spec)
        t_c_per_spec.append(t_c_spec)


        if p_max_spec != 0:
            conserv1 = 100*(p_max_spec - p_min_spec)/p_max_spec
            constit1 = 100*(p_ig_spec+ p_ignored)/p_max_spec
        else:
            conserv1 = 0.0
            constit1 = 0.0
        conserv_list.append(conserv1)
        consist_list.append(constit1)

        if verbose:
            print(f"Checking time for Spec {k}: {t_c_spec:.4f} seconds")
            print(f"p_total_max_spec: {p_max_spec}")
            print(f"p_total_min_spec: {p_min_spec}")
            print(f"conservativeness for Spec {k}: {conserv1:.6f}%")
            print(f"constitution for Spec {k}: {constit1:.6f}%")
            print(f"p_ignored during checking using estimate probability: {p_ig_spec}")

    return p_SAT_MAX, p_SAT_MIN, p_IG_approx, t_c_per_spec, conserv_list, consist_list

def TL_verify_CMAPSS(time_step, shifts, engine_id, numCores=None, p_filter=1e-10,
                     specs=None, spec_names=None, verbose=True):
    if verbose:
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

    # The first FC layer is encoded in RecurrentLayer via (Woh, boh).
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
        p_filter=p_filter,
        show=verbose,
    )
    t_r = time.time() - v_r
    if verbose:
        print(f"p_ignored during reachability using filter: {p_ignored}")
        print(f"Reachability analysis time: {t_r:.4f} seconds")
        print(f"total branches after reachability:{len(branches)}")

    if specs is None or spec_names is None:
        specs, spec_names = create_specs_CMAPSS(time_horizon=time_step)

    p_SAT_MAX, p_SAT_MIN, p_IG_approx, t_c_per_spec, conserv_list, consist_list = evaluate_specs_over_branches(
        branches=branches,
        specs=specs,
        spec_names=spec_names,
        p_ignored=p_ignored,
        numCores=numCores,
        verbose=verbose,
    )

    t_c_total = sum(t_c_per_spec)
    if verbose:
        print(f"Total checking time for all specs: {t_c_total:.4f} seconds")
        print(f"Total verification time (reachability + checking): {t_r + t_c_total:.4f} seconds")

    return {
            "case": "CMAPSS",
            "T": time_step,
            "p_filter": p_filter,
            "spec_names": spec_names,
            "rho_max": p_SAT_MAX,
            "rho_min": p_SAT_MIN,
            "t_r": [t_r] * len(spec_names),
            "t_c": t_c_per_spec,
            "t_v": [t_r + tc for tc in t_c_per_spec],
            "p_ig_evaluate": p_IG_approx,
            "p_ignored_reach": p_ignored,
            "n_branches": len(branches),
            "conserv_list": conserv_list,
            "consist_list": consist_list,
        }

def TL_verify_LIMO(time_step, shifts, numCores=None, p_filter=1e-10, verbose=True):

    if verbose:
        print(f"======================== Start LIMO reachability and verification ========================")
    X = construct_LIMO_input_probstar(time_step=time_step, shifts=shifts)

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

    v_r = time.time()
    branches, _, p_ignored_reach = L1.reachExactBranches(
        X,
        post_layers=[L2, L3, L4, L5],
        lp_solver="gurobi",
        pool=None,
        p_filter=p_filter,
        show=verbose,
    )
    t_r = time.time() - v_r
    if verbose:
        print(f"Reachability analysis time: {t_r:.4f} seconds")
        print(f"total branches after reachability:{len(branches)}")
        print(f"p_ignored during reachability using filter: {p_ignored_reach}")


    specs, spec_names = create_specs_LIMO(time_horizon=time_step)

    p_SAT_MAX, p_SAT_MIN, p_IG_approx, t_c_per_spec, conserv_list, consist_list = evaluate_specs_over_branches(
        branches=branches,
        specs=specs,
        spec_names=spec_names,
        p_ignored=p_ignored_reach,
        numCores=numCores,
        verbose=verbose,
    )

    t_c_total = sum(t_c_per_spec)

    if verbose:
        print(f"Total checking time for all specs: {t_c_total:.4f} seconds")
        print(f"Total verification time (reachability + checking): {t_r + t_c_total:.4f} seconds")

    
    return {
        "case": "LIMO",
        "T": time_step,
        "p_filter": p_filter,
        "spec_names": spec_names,
        "rho_max": p_SAT_MAX,
        "rho_min": p_SAT_MIN,
        "t_r": [t_r] * len(spec_names),
        "t_c": t_c_per_spec,
        "t_v": [t_r + tc for tc in t_c_per_spec],
        "p_ig_check": p_IG_approx,
        "p_ignored_reach": p_ignored_reach,
        "n_branches": len(branches),
        "conserv_list": conserv_list,
        "consist_list": consist_list,
    }


def result_to_detail_rows(result):
    """Convert one verification result dict to per-spec rows."""
    rows = []
    for i, spec_name in enumerate(result["spec_names"]):
        row = {
            "case": result["case"],
            "T": int(result["T"]),
            "p_filter": float(result["p_filter"]),
            "spec_name": spec_name,
            "rho_max": float(result["rho_max"][i]),
            "rho_min": float(result["rho_min"][i]),
            "t_r": float(result["t_r"][i]),
            "t_c": float(result["t_c"][i]),
            "t_v": float(result["t_v"][i]),
            "n_branches": int(result.get("n_branches", 0)),
            "p_ignored_reach": float(result.get("p_ignored_reach", 0.0)),
        }

        if "p_ig_check" in result:
            row["p_ig_check"] = float(result["p_ig_check"][i])
        if "p_ig_evaluate" in result:
            row["p_ig_evaluate"] = float(result["p_ig_evaluate"][i])
        if "conserv_list" in result:
            row["conserv_list"] = float(result["conserv_list"][i])
        if "consist_list" in result:
            row["consist_list"] = float(result["consist_list"][i])

        rows.append(row)
    return rows


def result_to_summary_row(result):
    """Convert one verification result dict to one run-level summary row."""
    t_r = float(result["t_r"][0]) if len(result["t_r"]) > 0 else np.nan
    t_c_total = float(np.sum(result["t_c"]))
    t_v_total = float(t_r + t_c_total)
    return {
        "case": result["case"],
        "T": int(result["T"]),
        "p_filter": float(result["p_filter"]),
        "t_r": t_r,
        "t_c": t_c_total,
        "t_v": t_v_total,
        "n_specs": int(len(result["spec_names"])),
        "n_branches": int(result.get("n_branches", 0)),
        "p_ignored_reach": float(result.get("p_ignored_reach", 0.0)),
    }



def _format_pf_tag(pf):
    return str(float(pf)).replace(".", "p")


def _format_pf_label(pf):
    return f"{float(pf):g}"


def _sanitize_spec_tag(spec_name, fallback_idx=None):
    spec_text = str(spec_name)
    idx_match = re.search(r"\d+", spec_text)
    if idx_match is not None:
        return f"spec_{idx_match.group(0)}"

    clean = re.sub(r"[^0-9A-Za-z]+", "_", spec_text).strip("_").lower()
    if clean:
        return clean

    if fallback_idx is not None:
        return f"spec_{fallback_idx + 1}"
    return "spec"


def _safe_metric_value(result, key, idx):
    values = result.get(key)
    if values is None:
        return float(np.nan)
    try:
        return float(values[idx])
    except (TypeError, ValueError, IndexError):
        return float(np.nan)


def _render_table(rows, headers, tablefmt=None):
    if tabulate is not None:
        if tablefmt is None:
            return tabulate(rows, headers=headers)
        return tabulate(rows, headers=headers, tablefmt=tablefmt)

    df = pd.DataFrame(rows, columns=headers)
    if tablefmt == "latex":
        return df.to_latex(index=False)
    return df.to_string(index=False)


def verify_temporal_specs_RNN_case(
    case_name,
    time_steps=None,
    p_filters=None,
    shifts=1,
    numCores=1,
    engine_id=1,
    verbose=True,
    out_dir=None,
    save_results=True,
):

    case_name = case_name.upper()
    if case_name not in ("LIMO", "CMAPSS"):
        raise RuntimeError(f"Unknown case_name: {case_name}")

    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results",
            "TL_verification_benchmark",
            case_name,
        )

    table_dir = os.path.join(out_dir, "table")
    os.makedirs(table_dir, exist_ok=True)

    detail_rows = []
    summary_rows = []
    verification_data_by_pf = {float(pf): [] for pf in p_filters}

    headers = [
        "Spec.",
        "T",
        "p_max",
        "p_min",
        "reachTime",
        "checkTime",
        "verifyTime",
        "p_ig_check",
        "p_ig_eval",
        "conserv(%)",
        "constit(%)",
    ]

    for T in time_steps:
        for pf in p_filters:
            print(f"\n======================= {case_name} verification: T={T}, p_f={pf} ========================")

            if case_name == "LIMO":
                result = TL_verify_LIMO(
                    time_step=T,
                    shifts=shifts,
                    numCores=numCores,
                    p_filter=pf,
                    verbose=verbose,
                )
            else:
                result = TL_verify_CMAPSS(
                    time_step=T,
                    shifts=shifts,
                    engine_id=engine_id,
                    numCores=numCores,
                    p_filter=pf,
                    verbose=verbose,
                )

            detail_rows.extend(result_to_detail_rows(result))
            summary_rows.append(result_to_summary_row(result))

            pf_key = float(pf)
            for i, spec_name in enumerate(result["spec_names"]):
                verification_data_by_pf[pf_key].append(
                    [
                        spec_name,
                        T,
                        float(result["rho_max"][i]),
                        float(result["rho_min"][i]),
                        float(result["t_r"][i]),
                        float(result["t_c"][i]),
                        float(result["t_v"][i]),
                        _safe_metric_value(result, "p_ig_check", i),
                        _safe_metric_value(result, "p_ig_evaluate", i),
                        _safe_metric_value(result, "conserv_list", i),
                        _safe_metric_value(result, "consist_list", i),

                    ]
                )

    if save_results:
        detail_df = pd.DataFrame(detail_rows)
        summary_df = pd.DataFrame(summary_rows)

        detail_csv = os.path.join(out_dir, f"{case_name.lower()}_verification_all_detail.csv")
        summary_csv = os.path.join(out_dir, f"{case_name.lower()}_verification_all_summary.csv")
        detail_df.to_csv(detail_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved detail CSV: {detail_csv}")
        print(f"Saved summary CSV: {summary_csv}")

        table_files = []
        for pf in p_filters:
            pf_key = float(pf)
            rows = verification_data_by_pf[pf_key]
            print(f"\n======================= VERIFICATION RESULTS {case_name} WITH FILTERING pf = {pf_key} ==========================")
            print(_render_table(rows, headers=headers))

            pf_tag = _format_pf_tag(pf_key)
            txt_path = os.path.join(table_dir, f"{case_name}_verification_tab_pf_{pf_tag}.txt")
            tex_path = os.path.join(table_dir, f"{case_name}_verification_tab_pf_{pf_tag}.tex")

            with open(txt_path, "w", encoding="utf-8") as f:
                print(_render_table(rows, headers=headers), file=f)

            with open(tex_path, "w", encoding="utf-8") as f:
                print(_render_table(rows, headers=headers, tablefmt="latex"), file=f)

            table_files.extend([txt_path, tex_path])


    return {
        "detail_csv": detail_csv,
        "summary_csv": summary_csv,
        "table_files": table_files,
        "figure_path": figure_path,
        "per_spec_conserv_constitution_figures": per_spec_conserv_constitution_figures,
        "per_spec_analysis_time_figures": per_spec_analysis_time_figures,
        "detail_df": detail_df,
        "summary_df": summary_df,
    }


    
if __name__ == "__main__":
    np.random.seed(25)
    time_steps = [10,15,20,25]
    p_filters = [0.0,0.005,0.01,0.05,0.1]
    engine_id = 1
    shifts = 1
    num_cores = 4
    verbose = True

    result_out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results_LIMO",
        "TL_verification_benchmark_LIMO",
    )



    verify_temporal_specs_RNN_case(
        case_name="LIMO",
        time_steps=time_steps,
        p_filters=p_filters,
        shifts=shifts,
        engine_id=engine_id,
        numCores=num_cores,
        verbose=verbose,
        out_dir=result_out_dir,
        save_results=True,
    )

# verify phi3
# approximate with p_filyer = 1e-10:
# p_total_max: 0.6459390775840915
# p_total_min: 0.6459390012033069

# exact with p_filter = 0.0:
# p_total_max: 0.6521874936468794
# p_total_min: 0.652187419780777
