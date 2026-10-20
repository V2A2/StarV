"""
StarTL verification for the ATVA2026 RNN case studies.

This script evaluates dStarTL.py on:
    1. CMAPSS engine degradation predication
    2. LIMO trajectory prediction

"""

import csv
import multiprocessing
import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.RecurrentLayer import RecurrentLayer
from StarV.set.star import Star
from StarV.spec.dStarTL import ExpandedFormula, GetSatisfactionFraction_for_RNN
from StarV.spec.dProbStarTL import (
    _ALWAYS_,
    _EVENTUALLY_,
    AtomicPredicate,
    Formula,
    _LeftBracket_,
    _RightBracket_,
    _AND_,
    _OR_,
)
from StarV.util.load_rnn import (
    load_LIMO_data,
    load_trained_CMAPSS_data,
    load_trained_params_CMAPSS,
    load_trained_params_LIMO,
    get_input_set_CMAPSS,
    get_input_set_LIMO,
)


def construct_CMAPSS_input_star(time_step, shifts, engine_id, verbose=True):

    ### load CMAPSS data ###

    train_processed, _, _ = load_trained_CMAPSS_data()
    # select one engine unit data for reachability analysis
    engine_data = train_processed.loc[train_processed['unit_number'] == engine_id]
    if verbose:
        print(f"engine {engine_id} data shape:{engine_data.shape}")
        print(f"engine {engine_id} data samples:{engine_data.head(10)}")
    # select one time step data for reachability analysis
    # input_data = engine_data[:time_step].values[:, 2:] # remove unit_number and time_cycles columns
    if engine_data["time_cycles"].max() < time_step:
        if verbose:
            print(f"Engine {engine_id} has only {engine_data['time_cycles'].max()} time cycles, less than the specified time step {time_step}.")
        input_engine_data = engine_data.values[:, 2:]
    else:
        input_engine_data = engine_data.values[shifts-1:shifts+(time_step*2)-1, 2:]
        if verbose:
            print("input_engine_data shape:", input_engine_data.shape)
            print("input_engine_data:", input_engine_data)

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

    print(f"Create Star set input sequence for RNN")
    X = get_input_set_CMAPSS(
        input_engine_data,
        noises=all_noises,
        feature_idx=feature_idx,
        set="star"
    )
    if verbose:
        print("noise added to each feature:")
        for i in range(len(all_noises)):
            print(f"  Feature {i}: {all_noises[i]}")
        print(f"created CMAPSS Star input sequence with {len(X)} sets")


    return X

def construct_LIMO_input_star(time_step, shifts, verbose=True):

    ### load LIMO data ###

    processed_input_data, _ = load_LIMO_data()
    
    # select 40 steps data to create 20 window, each window is used to predict next 20 trajectory states
    input_LIMO_data = processed_input_data[shifts-1:shifts+(time_step*2)-1,:]
    input_target_LIMO_data = input_LIMO_data[time_step:,:5]
    # print("input_LIMO_data:",input_LIMO_data)
    # print("input_LIMO_data_shape:",input_LIMO_data.shape)

    # print("input_target_LIMO_data:",input_target_LIMO_data)
    # print("input_target_LIMO_data_shape:",input_target_LIMO_data.shape)
    rng = np.random.default_rng(25)
    # noise = np.round(rng.normal(0.0, 0.0125), decimals=4)
    noise = 0.0018
    print(f"Create Star set input sequence for RNN")
    X = get_input_set_LIMO(input_data=input_LIMO_data, noise=noise, set="star")
    if verbose:
        print("noise added to LIMO data:",noise)
        print("input_LIMO_data shape:", input_LIMO_data.shape)
        print("input_target_LIMO_data shape:", input_target_LIMO_data.shape)
        print(f"created LIMO Star input sequence with {len(X)} sets")
    return X


def create_specs_CMAPSS(time_horizon=None):
    """CMAPSS specs for 6-D outputs: [var_7, var_11, var_12, var_15, var_20, var_21]."""
    AND = _AND_()
    OR = _OR_()
    lb = _LeftBracket_()
    rb = _RightBracket_()
    if time_horizon is None or time_horizon < 2:
        raise RuntimeError('time_horizon should contain at least two output sets')
    T = time_horizon - 1

    P1 = AtomicPredicate(np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]), np.array([-0.65]))
    P11 = AtomicPredicate(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.array([0.65]))
    P2 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), np.array([0.6]))
    P21 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]), np.array([-0.6]))
    P3 = AtomicPredicate(np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]), np.array([-0.30]))
    P31 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), np.array([0.30]))
    P4 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, -1.0, 1.0]), np.array([0.3]))
    P5 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0]), np.array([0.3]))
    P8 = AtomicPredicate(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.45]))
    P9 = AtomicPredicate(np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]), np.array([-0.15]))

    EVOT = _EVENTUALLY_(0, T)
    AWOT = _ALWAYS_(0, T)
    EVOT1 = _EVENTUALLY_(0, 5)
    AWOT1 = _ALWAYS_(0, 5)

    phi1 = Formula([EVOT, lb, P1, rb])
    phi11 = Formula([AWOT, lb, P11, rb])
    phi2 = Formula([EVOT, lb, P2, OR, P3, rb])
    phi21 = Formula([AWOT, lb, P21, AND, P31, rb])
    phi3 = Formula([EVOT, lb, P8, AND, lb, AWOT1, P9, rb, rb])
    phi4 = Formula([
        EVOT, lb, P4, AND, P5, AND,
        lb, EVOT1, P4, AND, P5, rb, rb
    ])

    specs = [phi1, phi11, phi2, phi21, phi3, phi4]
    names = [r'$\varphi_{}$'.format(i + 1) for i in range(len(specs))]
    return specs, names


def create_specs_LIMO(time_horizon=None):
    """LIMO specs for 5-D outputs: [x, y, yaw, v, w]."""
    AND = _AND_()
    OR = _OR_()
    lb = _LeftBracket_()
    rb = _RightBracket_()
    if time_horizon is None or time_horizon < 2:
        raise RuntimeError('time_horizon should contain at least two output sets')
    T = time_horizon - 1

    # P1 = AtomicPredicate(np.array([-1.0, 0.0, 0.0, 0.0, 0.0]), np.array([-0.55]))
    # P11 = AtomicPredicate(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.55
    #                                                                      ]))
    x_threshold = 0.59
    P1 = AtomicPredicate(
        np.array([-1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([-x_threshold])
    )
    P11 = AtomicPredicate(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([x_threshold])
    )
    P3 = AtomicPredicate(np.array([0.0, -1.0, 0.0, 0.0, 0.0]), np.array([-0.85]))
    P31 = AtomicPredicate(np.array([0.0, 1.0, 0.0, 0.0, 0.0]), np.array([0.85]))
    # P4 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, -1.0]), np.array([-0.6]))
    P4 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, -1.0]), np.array([-0.65]))
    P5 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 1.0]), np.array([0.95]))
    P7 = AtomicPredicate(np.array([0.0, 0.0, 0.0, -1.0, 0.0]), np.array([-0.45]))
    P8 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 1.0, 0.0]), np.array([0.65]))


    # P1 = AtomicPredicate(np.array([-1.0, 0.0, 0.0, 0.0, 0.0]), np.array([-0.65]))
    # P11 = AtomicPredicate(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.65
    #                                                                      ]))
    # P3 = AtomicPredicate(np.array([0.0, -1.0, 0.0, 0.0, 0.0]), np.array([-0.85]))
    # P31 = AtomicPredicate(np.array([0.0, 1.0, 0.0, 0.0, 0.0]), np.array([0.85]))
    # P4 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, -1.0]), np.array([-0.65]))
    # P5 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 0.0, 1.0]), np.array([0.95]))
    # P7 = AtomicPredicate(np.array([0.0, 0.0, 0.0, -1.0, 0.0]), np.array([-0.45]))
    # P8 = AtomicPredicate(np.array([0.0, 0.0, 0.0, 1.0, 0.0]), np.array([0.65]))


    EVOT = _EVENTUALLY_(0, T)
    AWOT = _ALWAYS_(0, T)
    EVOT1 = _EVENTUALLY_(0, 5)
    AWOT1 = _ALWAYS_(0, 5)

    phi1 = Formula([EVOT, lb, P1, rb])
    phi11 = Formula([AWOT, lb, P11, rb])
    phi2 = Formula([EVOT, lb, P1, OR, P3, rb])
    phi21 = Formula([AWOT, lb, P11, AND, P31, rb])
    phi3 = Formula([EVOT, lb, P5, AND, lb, AWOT1, P4, rb, rb])
    phi4 = Formula([
        EVOT, lb, P7, AND, P8, AND,
        lb, EVOT1, P7, AND, P8, rb, rb
    ])

    specs = [phi1, phi11, phi2, phi21, phi3, phi4]
    names = [r'$\varphi_{}$'.format(i + 1) for i in range(len(specs))]
    return specs, names



def make_rnn_post_layers(fc_w, fc_b, case_name):
    """Create the post-RNN feedforward layers used by each benchmark."""
    mats = [[fc_w[i], np.array(fc_b[i])] for i in range(len(fc_w))]

    if case_name == 'CMAPSS':
        return [
            ReLULayer(),
            FullyConnectedLayer(mats[0]),
            ReLULayer(),
            FullyConnectedLayer(mats[1]),
            ReLULayer(),
            FullyConnectedLayer(mats[2]),
        ]

    if case_name == 'LIMO':
        return [
            ReLULayer(),
            FullyConnectedLayer(mats[0]),
            ReLULayer(),
            FullyConnectedLayer(mats[1]),
        ]

    raise RuntimeError('unknown case name: {}'.format(case_name))


def load_rnn_case_study(case_name, time_step, shifts=1, engine_id=1, verbose=True):
    """Load Star inputs, RNN network, post layers, and temporal specs."""
    case_name = case_name.upper()

    if case_name == 'CMAPSS':
        inputs = construct_CMAPSS_input_star(
            time_step=time_step,
            shifts=shifts,
            engine_id=engine_id,
            verbose=verbose
        )
        Whx, bhx, Whh, bhh, Woh, boh, fc_w, fc_b = load_trained_params_CMAPSS()
        specs, spec_names = create_specs_CMAPSS(time_horizon=time_step)

    elif case_name == 'LIMO':
        inputs = construct_LIMO_input_star(
            time_step=time_step,
            shifts=shifts,
            verbose=verbose
        )
        Whx, bhx, Whh, bhh, Woh, boh, fc_w, fc_b = load_trained_params_LIMO()
        specs, spec_names = create_specs_LIMO(time_horizon=time_step)

    else:
        raise RuntimeError('case_name should be CMAPSS or LIMO')

    recurrent_layer = RecurrentLayer(Whx, Whh, bhx, Woh, boh, bhh)
    post_layers = make_rnn_post_layers(fc_w, fc_b, case_name)
    return inputs, recurrent_layer, post_layers, specs, spec_names


def reach_rnn_star_branches(
        case_name, time_step, shifts=1, engine_id=1,
        lp_solver='gurobi', verbose=True, numCores=1):
    """Run branch-based RNN reachability using Star input sets."""
    inputs, recurrent_layer, post_layers, specs, spec_names = load_rnn_case_study(
        case_name,
        time_step,
        shifts=shifts,
        engine_id=engine_id,
        verbose=verbose
    )

    start = time.time()
    branches, _, p_ignored = recurrent_layer.reachExactBranches(
        inputs,
        post_layers=post_layers,
        lp_solver=lp_solver,
        p_filter=0.0,
        show=verbose,
        numCores=numCores,
    )
    reach_time = time.time() - start

    if p_ignored != 0.0:
        raise RuntimeError('Star dStarTL only support p_ignored = 0.0')

    return branches, specs, spec_names, reach_time


def evaluate_branch_star_trace(
        star_trace, spec, method='sampling', num_samples=2000, seed=0,
        burn_in=None, thinning=1, lp_solver='linprog',
        compute_exact_robustness=True, compute_region_balls=True):
    """Evaluate one Star branch trace against one temporal specification."""
    if not isinstance(star_trace, list):
        raise RuntimeError('star_trace should be a list')
    if not all(isinstance(reachable_set, Star) for reachable_set in star_trace):
        raise RuntimeError('each reachable set in star_trace should be a Star')

    expanded_spec = ExpandedFormula(spec, T=len(star_trace))
    checker = GetSatisfactionFraction_for_RNN(
        star_trace,
        expanded_spec,
        method=method,
        num_samples=num_samples,
        seed=seed,
        burn_in=burn_in,
        thinning=thinning,
        lp_solver=lp_solver,
        compute_exact_robustness=compute_exact_robustness,
        compute_region_balls=compute_region_balls
    )
    return checker.getRNNSatisfactionFraction()


def evaluate_branch_star_trace_worker(args):
    """Multiprocessing worker for one RNN branch/spec check."""
    return evaluate_branch_star_trace(*args)


def evaluate_specs_on_star_branches(
        branches, specs, spec_names, method='sampling', num_samples=2000,
        seed=0, burn_in=None, thinning=1, lp_solver='linprog', verbose=True,
        compute_exact_robustness=True, compute_region_balls=True,
        numCores=1):
    """Evaluate all dStarTL specs over all Star RNN branch traces."""
    rows = []

    for spec_id, spec in enumerate(specs):
        if verbose:
            print('\n================== dStarTL Spec {} ({}) =================='
                  .format(spec_id, spec_names[spec_id]))
            spec.print()

        start = time.time()
        worker_args = [
            (
                star_trace,
                spec,
                method,
                num_samples,
                None if seed is None else seed + branch_id,
                burn_in,
                thinning,
                lp_solver,
                compute_exact_robustness,
                compute_region_balls
            )
            for branch_id, star_trace in enumerate(branches)
        ]
        if numCores > 1 and len(worker_args) > 1:
            print(
                'Checking spec {} on {} RNN branches using {} cores...'
                .format(spec_id, len(worker_args), numCores)
            )
            with multiprocessing.Pool(numCores) as pool:
                branch_results = pool.map(evaluate_branch_star_trace_worker, worker_args)
        else:
            branch_results = []
            for branch_id, args in enumerate(worker_args):
                if verbose:
                    print('Checking branch {} against spec {}...'.format(branch_id, spec_id))
                branch_results.append(evaluate_branch_star_trace_worker(args))

        checking_time = time.time() - start
        if verbose:
            for branch_id, result in enumerate(branch_results):
                print(
                    'Branch {} checked by {} method'.format(
                        branch_id,
                        result.get('method')
                    )
                )

        rho_lbs = np.array([result['rho_lb'] for result in branch_results], dtype=float)
        rho_ubs = np.array([result['rho_ub'] for result in branch_results], dtype=float)
        fractions = np.array(
            [result['satisfying_fraction'] for result in branch_results],
            dtype=float
        )
        mixed_mask = (
            np.isfinite(rho_lbs)
            & np.isfinite(rho_ubs)
            & (rho_lbs < 0.0)
            & (rho_ubs >= 0.0)
        )
        finite_mask = np.isfinite(rho_lbs) & np.isfinite(rho_ubs)
        sat_mask = finite_mask & (rho_lbs >= 0.0)
        violated_mask = finite_mask & (rho_ubs < 0.0)
        invalid_mask = ~(sat_mask | violated_mask | mixed_mask)
        invalid_branch_count = int(np.sum(invalid_mask))
        if invalid_branch_count > 0:
            invalid_ids = np.flatnonzero(invalid_mask)
            raise RuntimeError(
                '{} RNN branches returned invalid robustness intervals for '
                'spec {}. First invalid branch IDs: {}'
                .format(
                    invalid_branch_count,
                    spec_id,
                    invalid_ids[:10].tolist()
                )
            )

        exact_rho_lbs = np.array(
            [
                result.get('exact_rho_lb', np.nan) if mixed_mask[branch_id] else np.nan
                for branch_id, result in enumerate(branch_results)
            ],
            dtype=float
        )
        exact_rho_ubs = np.array(
            [
                result.get('exact_rho_ub', np.nan) if mixed_mask[branch_id] else np.nan
                for branch_id, result in enumerate(branch_results)
            ],
            dtype=float
        )

        exact_rho_lb = (
            float(np.nanmin(exact_rho_lbs))
            if not np.all(np.isnan(exact_rho_lbs)) else np.nan
        )
        exact_rho_ub = (
            float(np.nanmax(exact_rho_ubs))
            if not np.all(np.isnan(exact_rho_ubs)) else np.nan
        )
        mixed_branch_count = int(np.sum(mixed_mask))

        sat_ball_fraction_lb = np.nan
        viol_ball_fraction_lb = np.nan
        if method == 'sampling' and mixed_branch_count > 0:
            sat_ball_fraction_lbs = []
            viol_ball_fraction_lbs = []
            for result in branch_results:
                branch_rho_lb = result['rho_lb']
                branch_rho_ub = result['rho_ub']
                exact_lb = result.get('exact_rho_lb', np.nan)
                exact_ub = result.get('exact_rho_ub', np.nan)

                if np.isfinite(branch_rho_lb) and branch_rho_lb >= 0.0:
                    sat_ball_fraction_lbs.append(1.0)
                    viol_ball_fraction_lbs.append(0.0)
                elif np.isfinite(branch_rho_ub) and branch_rho_ub < 0.0:
                    sat_ball_fraction_lbs.append(0.0)
                    viol_ball_fraction_lbs.append(1.0)
                elif np.isfinite(exact_lb) and exact_lb >= 0.0:
                    sat_ball_fraction_lbs.append(1.0)
                    viol_ball_fraction_lbs.append(0.0)
                elif np.isfinite(exact_ub) and exact_ub < 0.0:
                    sat_ball_fraction_lbs.append(0.0)
                    viol_ball_fraction_lbs.append(1.0)
                else:
                    sat_ball_fraction_lbs.append(
                        result.get('sat_ball_fraction_lb', 0.0)
                    )
                    viol_ball_fraction_lbs.append(
                        result.get('viol_ball_fraction_lb', 0.0)
                    )

            sat_ball_fraction_lb = float(np.mean(sat_ball_fraction_lbs))
            viol_ball_fraction_lb = float(np.mean(viol_ball_fraction_lbs))

        print("robustness interval bounds: lb ={}, ub ={}".format(
            float(np.nanmin(rho_lbs)) if not np.all(np.isnan(rho_lbs)) else np.nan,
            float(np.nanmax(rho_ubs)) if not np.all(np.isnan(rho_ubs)) else np.nan
        ))

        rows.append({
            'spec_id': spec_id,
            'spec_name': spec_names[spec_id],
            'rho_lb': (
                float(np.nanmin(rho_lbs))
                if not np.all(np.isnan(rho_lbs)) else np.nan
            ),
            'rho_ub': (
                float(np.nanmax(rho_ubs))
                if not np.all(np.isnan(rho_ubs)) else np.nan
            ),
            'exact_rho_lb': exact_rho_lb,
            'exact_rho_ub': exact_rho_ub,
            'sat_fraction': (
                float(np.nanmean(fractions))
                if not np.all(np.isnan(fractions)) else np.nan
            ),
            'sat_ball_fraction_lb': sat_ball_fraction_lb,
            'viol_ball_fraction_lb': viol_ball_fraction_lb,
            'num_sat_branches': int(np.sum(sat_mask)),
            'num_violated_branches': int(np.sum(violated_mask)),
            'num_mixed_branches': mixed_branch_count,
            'checking_time': checking_time,
        })

    return rows


def print_result_table(rows):
    """Print exact- or sampling-specific benchmark columns."""
    if len(rows) == 0:
        return

    sampling_mode = rows[0].get('method') == 'sampling'
    if sampling_mode:
        header = (
            '{:<8} {:<5} {:<8} {:<8} {:>9} {:>9} {:>9} {:>9} '
            '{:>12} {:>11} {:>11} {:>7} {:>7} {:>7} {:>10} {:>10} {:>10}'
        ).format(
            'system', 'T', 'spec_id', 'branches', 'rho_lb', 'rho_ub',
            'exact_lb', 'exact_ub', 'sat_fraction', 'sat_ball', 'viol_ball',
            'sat', 'viol', 'mixed', 'reach_t', 'check_t', 'verify_t'
        )
    else:
        header = (
            '{:<8} {:<5} {:<8} {:<8} {:>9} {:>9} {:>9} {:>9} '
            '{:>12} {:>7} {:>7} {:>7} {:>10} {:>10} {:>10}'
        ).format(
            'system', 'T', 'spec_id', 'branches', 'rho_lb', 'rho_ub',
            'exact_lb', 'exact_ub', 'sat_fraction', 'sat', 'viol', 'mixed',
            'reach_t', 'check_t', 'verify_t'
        )

    print('\n================ CMAPSS/LIMO dStarTL Extended Results ================')
    print(header)
    print('-' * len(header))

    for row in rows:
        common_values = (
            row['case'], row['T'], row['spec_id'], row['num_branches'],
            row['rho_lb'], row['rho_ub'], row['exact_rho_lb'],
            row['exact_rho_ub'], row['sat_fraction']
        )
        if sampling_mode:
            print((
                '{:<8} {:<5} {:<8} {:<8} {:>9.4g} {:>9.4g} {:>9.4g} {:>9.4g} '
                '{:>12.6g} {:>11.4g} {:>11.4g} {:>7} {:>7} {:>7} '
                '{:>10.4f} {:>10.4f} {:>10.4f}'
            ).format(
                *common_values,
                row['sat_ball_fraction_lb'], row['viol_ball_fraction_lb'],
                row['num_sat_branches'], row['num_violated_branches'],
                row['num_mixed_branches'], row['reach_time'],
                row['checking_time'], row['verification_time']
            ))
        else:
            print((
                '{:<8} {:<5} {:<8} {:<8} {:>9.4g} {:>9.4g} {:>9.4g} {:>9.4g} '
                '{:>12.6g} {:>7} {:>7} {:>7} {:>10.4f} {:>10.4f} {:>10.4f}'
            ).format(
                *common_values,
                row['num_sat_branches'], row['num_violated_branches'],
                row['num_mixed_branches'], row['reach_time'],
                row['checking_time'], row['verification_time']
            ))



def save_rows_to_csv(rows, out_file):
    """Save result rows to CSV."""
    if len(rows) == 0:
        return None

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print('Saved dStarTL comparison CSV: {}'.format(out_file))
    return out_file


def verify_rnn_case_dStarTL(
        case_name, time_steps, shifts=1, engine_id=1,
        method='sampling', num_samples=2000, seed=0, burn_in=None, thinning=1,
        reach_lp_solver='gurobi', check_lp_solver='linprog', verbose=True,
        compute_exact_robustness=True, compute_region_balls=True,
        numCores=1):
    """Run dStarTL extended verification for one RNN benchmark."""
    case_name = case_name.upper()
    all_rows = []

    for time_step in time_steps:
        print('\n================ {} dStarTL: T={}, p_filter=0.0 ================'
              .format(case_name, time_step))

        branches, specs, spec_names, reach_time = reach_rnn_star_branches(
            case_name,
            time_step,
            shifts=shifts,
            engine_id=engine_id,
            lp_solver=reach_lp_solver,
            verbose=verbose,
            numCores=numCores
        )

        spec_rows = evaluate_specs_on_star_branches(
            branches,
            specs,
            spec_names,
            method=method,
            num_samples=num_samples,
            seed=seed,
            burn_in=burn_in,
            thinning=thinning,
            lp_solver=check_lp_solver,
            verbose=verbose,
            compute_exact_robustness=compute_exact_robustness,
            compute_region_balls=compute_region_balls,
            numCores=numCores
        )

        for row in spec_rows:
            row.update({
                'case': case_name,
                'T': time_step,
                'method': method,
                'num_samples': num_samples,
                'compute_exact_robustness': compute_exact_robustness,
                'compute_region_balls': compute_region_balls,
                'numCores': numCores,
                'num_branches': len(branches),
                'p_filter': 0.0,
                'reach_time': reach_time,
                'verification_time': reach_time + row['checking_time'],
            })
            all_rows.append(row)

    return all_rows


def verify_CMAPSS_LIMO_dStarTL(
        cases=None, time_steps=None, shifts=1, engine_id=1,
        method='sampling', num_samples=2000, seed=0, burn_in=None, thinning=1,
        reach_lp_solver='gurobi', check_lp_solver='linprog', verbose=True,
        save_results=True, compute_exact_robustness=True,
        compute_region_balls=True, numCores=1):
    """Run the dStarTL extended comparison for CMAPSS and LIMO."""
    if cases is None:
        cases = ['CMAPSS', 'LIMO']
    if time_steps is None:
        raise RuntimeError('time_steps must be specified')

    rows = []
    for case_name in cases:
        rows.extend(verify_rnn_case_dStarTL(
            case_name,
            time_steps,
            shifts=shifts,
            engine_id=engine_id,
            method=method,
            num_samples=num_samples,
            seed=seed,
            burn_in=burn_in,
            thinning=thinning,
            reach_lp_solver=reach_lp_solver,
            check_lp_solver=check_lp_solver,
            verbose=verbose,
            compute_exact_robustness=compute_exact_robustness,
            compute_region_balls=compute_region_balls,
            numCores=numCores
        ))

    print_result_table(rows)

    # if save_results:
    #     out_file = os.path.join(
    #         PROJECT_ROOT,
    #         'artifacts',
    #         'ATVA2026_RNN',
    #         'results',
    #         'dStarTL_comparison',
    #         'cmapss_limo_dstartl_results.csv'
    #     )
    #     save_rows_to_csv(rows, out_file)

    return rows


if __name__ == '__main__':
    np.random.seed(25)

    cases = ['LIMO']
    time_steps = [10,15,20,25,30]
    shifts = 1
    engine_id = 1

    method = 'sampling'
    num_samples = 10000
    seed = 0
    burn_in = 10
    thinning = 1
    compute_exact_robustness = True
    compute_region_balls = True
    numCores = 4

    reach_lp_solver = 'gurobi'
    check_lp_solver = 'linprog'
    verbose = True
    save_results = True

    verify_CMAPSS_LIMO_dStarTL(
        cases=cases,
        time_steps=time_steps,
        shifts=shifts,
        engine_id=engine_id,
        method=method,
        num_samples=num_samples,
        seed=seed,
        burn_in=burn_in,
        thinning=thinning,
        reach_lp_solver=reach_lp_solver,
        check_lp_solver=check_lp_solver,
        verbose=verbose,
        save_results=save_results,
        compute_exact_robustness=compute_exact_robustness,
        compute_region_balls=compute_region_balls,
        numCores=numCores
    )
