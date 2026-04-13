
"""
ProbStar Reachability Analysis and Verification for RNNs
Two case study: CMAPSS and LIMO
Qing Liu, 01/20/2026
"""
from scipy.io import loadmat
import os
import time
import multiprocessing
import pandas as pd
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
        input_engine_data = engine_data.values[shifts-1:shifts+time_step-1, 2:]
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

def create_specs_CMAPSS():
    """CMAPSS specs for 6-D outputs: [var_7, var_11, var_12, var_15, var_20, var_21]."""

    AND = _AND_()
    OR = _OR_()
    lb = _LeftBracket_()
    rb = _RightBracket_()

    # P1: var_12 <= 0.42
    P1 = AtomicPredicate(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.array([0.42]))
    # P2: var_21 <= 0.45
    P2 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), np.array([0.45]))
    # P3: var_15 >= 0.30  
    P3 = AtomicPredicate(np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]), np.array([-0.30]))
    # P4, P5: |var_21 - var_20| <= 0.22
    P4 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, -1.0, 1.0]), np.array([0.22]))
    P5 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0]), np.array([0.22]))
    # P6, P7: |var_12 - var_7| <= 0.20
    P6 = AtomicPredicate(np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.array([0.20]))
    P7 = AtomicPredicate(np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0]), np.array([0.20]))
    # P8, P9: 0.15 <= var_11 <= 0.70
    P8 = AtomicPredicate(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.70]))
    P9 = AtomicPredicate(np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]), np.array([-0.15]))

    # Use fixed temporal windows (no bounded-interval clamping).
    EVOT = _EVENTUALLY_(0, 10)
    AWOT = _ALWAYS_(0, 15)
    EVOT1 = _EVENTUALLY_(5, 15)
    AWOT1 = _ALWAYS_(0, 5)

    # phi1: eventually low var_12 and var_21.
    spec1 = Formula([EVOT, lb, P1, OR, P2, rb])
    # phi2: always keep var_11 band and var_20-var_21 coupling.
    spec2 = Formula([AWOT, lb, P8, AND, P9, AND, P4, AND, P5, rb])
    # phi3: eventually high var_15, then always small |var_12-var_7| in next short horizon.
    spec3 = Formula([EVOT1, lb, P3, AND, lb, AWOT1, lb, P6, AND, P7, rb, rb, rb])

    return [spec1, spec2, spec3], [r'$\varphi_1$', r'$\varphi_2$', r'$\varphi_3$']

def create_specs_LIMO(time_horizon=None):
    """LIMO specs for 5-D outputs: [x, y, yaw, v, w]."""

    AND = _AND_()
    OR = _OR_()
    lb = _LeftBracket_()
    rb = _RightBracket_()
    T = time_horizon
    # Current LIMO output dimension is 5:
    # [x, y, yaw, v, w]

    # P1: x >= 0.60  
    P1 = AtomicPredicate(np.array([-1.0, 0.0, 0.0, 0.0, 0.0]), np.array([-0.60]))
    # P3: y >= 0.8
    P3 = AtomicPredicate(np.array([0.0, -1.0, 0.0, 0.0, 0.0]), np.array([-0.90]))
    # P4: w >= 0.7
    P4 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, -1.0]), np.array([-0.80]))
    # P5: w <= 0.9
    P5 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 1.0]), np.array([0.97]))
    # P7: v >= 0.45 
    P7 = AtomicPredicate(np.array([0.0, 0.0, 0.0, -1.0, 0.0]), np.array([-0.45]))
    # P8: v <= 0.65
    P8 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 1.0, 0.0]), np.array([0.65]))


    # Use fixed temporal windows (no bounded-interval clamping).
    EVOT = _EVENTUALLY_(0,T)
    AWOT = _ALWAYS_(0, T)
    EVOT1 = _EVENTUALLY_(0,5)
    AWOT1 = _ALWAYS_(0, 5)

    phi1 = Formula([EVOT, P1, OR, P3])

    phi11 = Formula([AWOT, P1, AND, P3])

    phi2 = Formula([EVOT, lb, P7, OR, P8, rb])

    phi21 = Formula([AWOT, lb, P7, AND, P8, rb])

    phi3 = Formula([EVOT, lb, P1, AND,P3, lb, EVOT1, P7, AND, P8, rb, rb])

    phi4 = Formula([EVOT, lb, lb,P4, AND, P5,rb, AND, lb, AWOT1,lb, P4, AND, P5, rb, rb,rb])
    
    phi41 = Formula([AWOT, lb,lb, P4, AND, P5, rb, OR, lb, EVOT1,lb, P4, AND, P5, rb, rb,rb])



    specs = [phi1, phi11, phi2, phi21, phi3, phi4, phi41]
    names = [r'$\varphi_1$', r'$\varphi_2$', r'$\varphi_3$', r'$\varphi_4$', r'$\varphi_5$', r'$\varphi_6$']
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
            with multiprocessing.Pool(numCores) as mp_pool:
                results = mp_pool.map(
                    check_sat_on_branch_for_RNN,
                    zip([spec] * len(branches), branches)
                )
            for r in results:
                p_total_max.append(r[0])
                p_total_min.append(r[1])
                p_ignored_approx.append(r[2])
        else:
            for i, sig in enumerate(branches):
                if not isinstance(sig, list):
                    raise RuntimeError('error: each branch signal should be a list')
                if verbose:
                    print(f"=====================Checking branch {i} =============")
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

        if verbose:
            print(f"Checking time for Spec {k}: {t_c_spec:.4f} seconds")
            print(f"p_total_max_spec: {p_max_spec}")
            print(f"p_total_min_spec: {p_min_spec}")
            print(f"p_ignored during checking using estimate probability: {p_ig_spec}")

    return p_SAT_MAX, p_SAT_MIN, p_IG_approx, t_c_per_spec

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
        print(f"Reachability analysis time: {t_r:.4f} seconds")
        print(f"total branches after reachability:{len(branches)}")

    if specs is None or spec_names is None:
        specs, spec_names = create_specs_CMAPSS(time_horizon=time_step)

    p_SAT_MAX, p_SAT_MIN, p_IG_approx, t_c_per_spec = evaluate_specs_over_branches(
        branches=branches,
        specs=specs,
        spec_names=spec_names,
        p_ignored=p_ignored,
        numCores=numCores,
        verbose=verbose,
    )

    t_c_total = sum(t_c_per_spec)
    if verbose:
        print(f"p_ignored during reachability using filter: {p_ignored}")
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

    p_SAT_MAX, p_SAT_MIN, p_IG_approx, t_c_per_spec = evaluate_specs_over_branches(
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
    }


def save_single_case_latex_table(result, out_dir=None):

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "TL_single_run_tables")
    os.makedirs(out_dir, exist_ok=True)

    case_name = result["case"]
    T = result["T"]
    p_filter = result["p_filter"]

    rows = []
    for i, spec_name in enumerate(result["spec_names"]):
        rows.append({
            "Spec.": spec_name,
            "T": T,
            "p_filter": p_filter,
            "rho_max": result["rho_max"][i],
            "rho_min": result["rho_min"][i],
            "t_r": result["t_r"][i],
            "t_c": result["t_c"][i],
            "t_v": result["t_v"][i],
        })

    df = pd.DataFrame(rows)
    stem = f"{case_name.lower()}_T{T}_pf{p_filter}"

    csv_path = os.path.join(out_dir, f"{stem}.csv")
    df.to_csv(csv_path, index=False)

    tex_table_path = os.path.join(out_dir, f"{stem}.tex")

    label = f"tab:{case_name.lower()}_T{T}_pf{str(p_filter).replace('.', 'p')}"
    latex_table = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\label{{{label}}}\n"
        # f"{latex_tabular}\n"
        "\\end{table}\n"
    )
    with open(tex_table_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"Saved single-run {case_name} table files:")
    print(csv_path)
    # print(tex_tabular_path)
    print(tex_table_path)

    return csv_path, tex_table_path


# def _build_case_table(df_case, p_filters, time_steps, spec_names):
    # """Build wide, image-style table with grouped columns by p_filter."""
    # import pandas as pd

    # metrics = ["rho_max", "rho_min", "t_r", "t_c", "t_v"]
    # rows = []
    # for T in time_steps:
    #     for spec in spec_names:
    #         row = {("Spec.", ""): spec, ("T", ""): T}
    #         for pf in p_filters:
    #             subset = df_case[
    #                 (df_case["T"] == T) &
    #                 (df_case["Spec."] == spec) &
    #                 (np.isclose(df_case["p_filter"], pf))
    #             ]
    #             if len(subset) == 0:
    #                 for m in metrics:
    #                     row[(pf_label(pf), m)] = np.nan
    #             else:
    #                 r = subset.iloc[0]
    #                 for m in metrics:
    #                     row[(pf_label(pf), m)] = float(r[m])
    #         rows.append(row)

    # table = pd.DataFrame(rows)
    # ordered_cols = [("Spec.", ""), ("T", "")]
    # for pf in p_filters:
    #     for m in metrics:
    #         ordered_cols.append((pf_label(pf), m))
    # table = table[ordered_cols]
    # return table


def save_verify_results(df_long, p_filters, time_steps, out_dir=None, case_order=("LIMO", "CMAPSS")):
    """
    Build and save one grouped benchmark table for each case study.
    Returns a list of saved file paths.
    """

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "TL_verification_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    saved_files = []
    for case_name in case_order:
        if case_name == "LIMO":
            _, spec_names = create_specs_LIMO(time_horizon=2 * int(time_steps[0]))
        else:
            _, spec_names = create_specs_CMAPSS()
        df_case = df_long[df_long["Case"] == case_name].copy()
        
        table = save_verify_results(
            df_case=df_case,
            p_filters=p_filters,
            time_steps=time_steps,
            spec_names=spec_names,
        )

        # Save wide CSV (flattened headers for easy parsing)
        flat = table.copy()
        flat.columns = [
            c0 if c1 == "" else f"{c0}__{c1}"
            for c0, c1 in flat.columns
        ]
        csv_path = os.path.join(out_dir, f"{case_name.lower()}_benchmark_table.csv")
        flat.to_csv(csv_path, index=False)
        saved_files.append(csv_path)

        # Save LaTeX tabular with grouped headers
        tex_tabular_path = os.path.join(out_dir, f"{case_name.lower()}_benchmark_table_tabular.tex")
        latex_tabular = table.to_latex(
            index=False,
            multicolumn=True,
            multirow=False,
            escape=False,
            float_format=lambda x: f"{x:.6g}",
        )
        with open(tex_tabular_path, "w", encoding="utf-8") as f:
            f.write(latex_tabular)
        saved_files.append(tex_tabular_path)

        # Save full table environment for direct paper inclusion.
        tex_table_path = os.path.join(out_dir, f"{case_name.lower()}_benchmark_table.tex")
        caption = (
            rf"Verification results for {case_name} under different filtering thresholds "
            rf"$p_f \in \{{{', '.join([str(p) for p in p_filters])}\}}$ and horizons "
            rf"$T \in \{{{', '.join([str(t) for t in time_steps])}\}}$."
        )
        label = f"tab:{case_name.lower()}_pf_benchmark"
        latex_table = (
            "\\begin{table*}[t]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"{latex_tabular}\n"
            "\\end{table*}\n"
        )
        with open(tex_table_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        saved_files.append(tex_table_path)

    return saved_files

    
if __name__ == "__main__":
    np.random.seed(25)
    time_step =20
    engine_id =1
    shifts =1
    num_cores = 1
    save_results = True
    p_filter = 0.01


    result_limo = TL_verify_LIMO(
        time_step=time_step,
        shifts=shifts,
        numCores=num_cores,
        p_filter=p_filter,
        verbose=True,
    )
    if save_results:
        save_single_case_latex_table(result_limo)

    # result_cmapss = TL_verify_CMAPSS(
    #     time_step=time_step,
    #     engine_id=engine_id,
    #     shifts=shifts,
    #     numCores=num_cores,
    #     p_filter=p_filter,
    #     return_details=True,
    #     verbose=True,
    # )
    # if save_results:
    #     save_single_case_latex_table(result_cmapss)


# verify phi3
# approximate with p_filyer = 1e-10:
# p_total_max: 0.6459390775840915
# p_total_min: 0.6459390012033069

# exact with p_filter = 0.0:
# p_total_max: 0.6521874936468794
# p_total_min: 0.652187419780777
