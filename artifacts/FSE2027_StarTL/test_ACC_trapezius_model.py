"""
ACC trapezius-model comparison ( NeuroSymbolic, ProbStarTL, StarTL).

This script reruns:
1) reachability on the pre-transformed ACC phi3 robustness networks,
2) ProbStarTL verification using the trapezius ACC setup, and
3) StarTL verification using the same ACC trapezius setup with the initial
   ProbStar converted to a Star.

"""

import copy
import csv
import os
import sys
import time

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.net.network import reachExactBFS
from StarV.nncs.nncs import VerifyPRM_NNCS, verify_temporal_specs_DLNNCS
from StarV.util.load import load_acc_trapezius, load_acc_trapezius_model
from StarV.test_ACC_dStarTL import verify_temporal_specs_DLNNCS_ACC


def make_trapezius_transformed_input():
    """Construct the input set used by the pre-transformed ACC robustness network."""
    center = np.array([100, 32.1, 0, 10.5, 30.1, 0])
    epsilon = np.array([10, 0.1, 0, 0.5, 0.1, 0])
    lb = center - epsilon
    ub = center + epsilon

    star_set = Star(lb, ub)
    mu = 0.5 * (star_set.pred_lb + star_set.pred_ub)
    sig = (mu - star_set.pred_lb) / 2.5
    Sig = np.diag(np.square(sig))
    return [ProbStar(
        star_set.V,
        star_set.C,
        star_set.d,
        mu,
        Sig,
        star_set.pred_lb,
        star_set.pred_ub
    )]


def probstar_to_star(probstar):
    """Drop the Gaussian distribution and keep the same predicate polytope."""
    return Star(
        probstar.V,
        probstar.C,
        probstar.d,
        probstar.pred_lb,
        probstar.pred_ub
    )


def run_transformed_network_verification(time_steps, lp_solver='linprog', numCores=1, show=False):
    """Rerun exact reachability on the pre-transformed ACC phi3 robustness networks."""
    rows = []
    input_set = make_trapezius_transformed_input()

    for time_step in time_steps:
        print('\nRunning transformed robustness network: T = {}'.format(time_step), flush=True)
        robustness_net = load_acc_trapezius(t=time_step)

        pool = None
        if numCores > 1:
            import multiprocessing
            pool = multiprocessing.Pool(numCores)

        try:
            start = time.perf_counter()
            output_sets = reachExactBFS(
                robustness_net,
                input_set,
                lp_solver=lp_solver,
                pool=pool,
                show=show
            )
            verify_time = time.perf_counter() - start
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        lower_bounds = []
        upper_bounds = []
        for output_set in output_sets:
            lb, ub = output_set.getRanges(lp_solver=lp_solver)
            lower_bounds.append(lb)
            upper_bounds.append(ub)

        rows.append({
            'method': 'TransformedNN',
            'net': 'prebuilt_phi3_tnn',
            'T': time_step,
            'spec_id': 'phi3',
            'value_max': np.nan,
            'value_min': np.nan,
            'rho_lb': float(np.min(lower_bounds)),
            'rho_ub': float(np.max(upper_bounds)),
            'branches': len(output_sets),
            'reach_time': verify_time,
            'check_time': 0.0,
            'verify_time': verify_time,
        })

    return rows


def run_probstarTL_trapezius(
        nets, time_steps, spec_ids, t=3, plant='linear',
        lp_solver='linprog', numCores=1, pf=0.0):
    """Rerun ProbStarTL on the same trapezius setup as HSCC2025_ProbStarTL."""
    rows = []

    for net in nets:
        for time_step in time_steps:
            print('\nRunning ProbStarTL trapezius: net = {}, T = {}'.format(net, time_step), flush=True)
            ncs, specs, initSet, refInputs = load_acc_trapezius_model(
                net,
                plant,
                spec_ids,
                time_step,
                t
            )

            init_probability = initSet.estimateProbability()

            verifyPRM = VerifyPRM_NNCS()
            verifyPRM.initSet = copy.deepcopy(initSet)
            verifyPRM.refInputs = copy.deepcopy(refInputs)
            verifyPRM.numSteps = time_step
            verifyPRM.pf = pf
            verifyPRM.numCores = numCores
            verifyPRM.lpSolver = lp_solver
            verifyPRM.temporalSpecs = copy.deepcopy(specs)

            traces, p_max, p_min, reachTime, checkingTime, verifyTime = (
                verify_temporal_specs_DLNNCS(ncs, verifyPRM)
            )

            p_max = np.asarray(p_max, dtype=float)
            p_min = np.asarray(p_min, dtype=float)
            pc_max = init_probability - p_max
            pc_min = init_probability - p_min

            for index, spec_id in enumerate(spec_ids):
                rows.append({
                    'method': 'ProbStarTL',
                    'net': net,
                    'T': time_step,
                    'spec_id': spec_id,
                    'value_max': float(pc_max[index]),
                    'value_min': float(pc_min[index]),
                    'rho_lb': np.nan,
                    'rho_ub': np.nan,
                    'branches': len(traces),
                    'reach_time': reachTime,
                    'check_time': checkingTime[index],
                    'verify_time': verifyTime[index],
                })

    return rows


def run_starTL_trapezius(
        nets, time_steps, spec_ids, t=3, plant='linear',
        lp_solver='linprog', numCores=1, method='sampling',
        num_samples=2000, seed=0, burn_in=None, thinning=1,
        volume_seed=0, compute_exact_robustness=True,
        compute_region_balls=False):
    """Rerun StarTL on the same ACC trapezius models used by ProbStarTL."""
    rows = []

    for net in nets:
        for time_step in time_steps:
            print('\nRunning StarTL trapezius: net = {}, T = {}'.format(net, time_step), flush=True)
            ncs, specs, initProbStar, refInputs = load_acc_trapezius_model(
                net,
                plant,
                spec_ids,
                time_step,
                t
            )
            initStar = probstar_to_star(initProbStar)

            verifyPRM = VerifyPRM_NNCS()
            verifyPRM.initSet = copy.deepcopy(initStar)
            verifyPRM.refInputs = copy.deepcopy(refInputs)
            verifyPRM.numSteps = time_step
            verifyPRM.pf = 0.0
            verifyPRM.lpSolver = lp_solver
            verifyPRM.show = False
            verifyPRM.numCores = numCores
            verifyPRM.temporalSpecs = copy.deepcopy(specs)

            analysis = verify_temporal_specs_DLNNCS_ACC(
                ncs,
                verifyPRM,
                num_samples=num_samples,
                seed=seed,
                burn_in=burn_in,
                thinning=thinning,
                method=method,
                volume_seed=volume_seed,
                compute_exact_robustness=compute_exact_robustness,
                compute_region_balls=compute_region_balls
            )

            for index, spec_id in enumerate(spec_ids):
                if method == 'exact':
                    sat_fraction = analysis['exact_sat_fraction'][index]
                else:
                    sat_fraction = analysis['sampling_sat_fraction'][index]

                complement_fraction = (
                    1.0 - sat_fraction
                    if sat_fraction is not None and np.isfinite(sat_fraction)
                    else np.nan
                )
                raw_rho_lb = analysis['rho_lb'][index]
                raw_rho_ub = analysis['rho_ub'][index]
                complement_rho_lb = -raw_rho_ub if np.isfinite(raw_rho_ub) else np.nan
                complement_rho_ub = -raw_rho_lb if np.isfinite(raw_rho_lb) else np.nan

                rows.append({
                    'method': 'StarTL',
                    'net': net,
                    'T': time_step,
                    'spec_id': spec_id,
                    'value_max': float(complement_fraction),
                    'value_min': float(complement_fraction),
                    'rho_lb': complement_rho_lb,
                    'rho_ub': complement_rho_ub,
                    'branches': analysis['num_traces'],
                    'reach_time': analysis['reachTime'],
                    'check_time': analysis['checkingTime'][index],
                    'verify_time': analysis['verifyTime'][index],
                })

    return rows


def print_comparison_table(rows):
    """Print compact comparison table."""
    headers = [
        'method', 'net', 'T', 'spec_id', 'value_max', 'value_min',
        'rho_lb', 'rho_ub', 'branches', 'reach_t', 'check_t', 'verify_t'
    ]
    print('\n================ ACC Trapezius Runtime Comparison ================', flush=True)
    header = (
        '{:<14} {:<18} {:<5} {:<8} {:>11} {:>11} {:>11} {:>11} '
        '{:>9} {:>10} {:>10} {:>10}'
    ).format(*headers)
    print(header, flush=True)
    print('-' * len(header), flush=True)
    for row in rows:
        print(
            '{:<14} {:<18} {:<5} {:<8} {:>11.6g} {:>11.6g} '
            '{:>11.6g} {:>11.6g} {:>9} {:>10.4f} {:>10.4f} {:>10.4f}'
            .format(
                row['method'],
                row['net'],
                row['T'],
                row['spec_id'],
                row['value_max'],
                row['value_min'],
                row['rho_lb'],
                row['rho_ub'],
                row['branches'],
                row['reach_time'],
                row['check_time'],
                row['verify_time'],
            )
            ,
            flush=True
        )


def save_rows_to_csv(rows, out_file):
    """Save comparison rows to CSV."""
    if len(rows) == 0:
        return None
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved ACC trapezius comparison results to {}'.format(out_file), flush=True)
    return out_file


def run_acc_trapezius_comparison(
        nets=None, time_steps=None, spec_ids=None, run_transformed=True,
        run_probstar=True, run_startl=True, numCores=1, lp_solver='linprog',
        star_method='sampling', num_samples=2000, seed=0, save_results=True):
    """Run transformed-network, ProbStarTL, and StarTL ACC trapezius experiments."""
    if nets is None:
        nets = [
            'controller_3_20',
            'controller_5_20',
            'controller_7_20',
            'controller_10_20',
        ]
    if time_steps is None:
        time_steps = [10, 20, 30, 50]
    if spec_ids is None:
        spec_ids = [8]

    rows = []
    if run_transformed:
        rows.extend(run_transformed_network_verification(
            time_steps=time_steps,
            lp_solver=lp_solver,
            numCores=numCores,
            show=False
        ))

    if run_probstar:
        rows.extend(run_probstarTL_trapezius(
            nets=nets,
            time_steps=time_steps,
            spec_ids=spec_ids,
            lp_solver=lp_solver,
            numCores=numCores,
            pf=0.0
        ))

    if run_startl:
        rows.extend(run_starTL_trapezius(
            nets=nets,
            time_steps=time_steps,
            spec_ids=spec_ids,
            lp_solver=lp_solver,
            numCores=numCores,
            method=star_method,
            num_samples=num_samples,
            seed=seed,
            compute_exact_robustness=True,
            compute_region_balls=False
        ))

    print_comparison_table(rows)

    if save_results:
        out_file = os.path.join(
            PROJECT_ROOT,
            'artifacts',
            'HSCC2025_ProbStarTL',
            'ACC',
            'results',
            'acc_trapezius_runtime_comparison.csv'
        )
        save_rows_to_csv(rows, out_file)

    return rows


if __name__ == '__main__':
    print('Starting ACC trapezius comparison runner...', flush=True)
    nets = [
        'controller_3_20',
        # 'controller_5_20',
        # 'controller_7_20',
        # 'controller_10_20',
    ]
    time_steps = [10, 20, 30,50]
    spec_ids = [8]
    numCores = 1
    lp_solver = 'linprog'
    star_method = 'exact'  # choose from: 'exact', 'sampling'/'sample', 'both'
    num_samples = 10000
    seed = 0

    run_acc_trapezius_comparison(
        nets=nets,
        time_steps=time_steps,
        spec_ids=spec_ids,
        run_transformed=True,
        run_probstar=True,
        run_startl=True,
        numCores=numCores,
        lp_solver=lp_solver,
        star_method=star_method,
        num_samples=num_samples,
        seed=seed,
        save_results=True
    )
