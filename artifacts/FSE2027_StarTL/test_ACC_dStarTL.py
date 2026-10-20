"""
ACC StarTL verification example.

This script mirrors the ACC case study in
artifacts/HSCC2025_ProbStarTL/HSCC2025_ProbStarTL.py, but uses:

    1. Star initial sets instead of ProbStar initial sets.
    2. Star reachability through reachDFS_DLNNCS in StarV/nncs/nncs.py.
    3. dStarTL robustness intervals and exact/sampling satisfaction fractions.

"""

import copy
import math
import multiprocessing
import time
from StarV.nncs.nncs import NNCS, ReachPRM_NNCS, VerifyPRM_NNCS, reachDFS_DLNNCS
from StarV.util.load import load_acc_model_dStarTL
from StarV.spec.dStarTL import (
        ExpandedFormula,
        GetSatisfactionFraction,
    )


def clamp_fraction(value):
    """Clamp a computed fraction while preserving None for uncomputed methods."""
    if value is None:
        return None
    return min(1.0, max(0.0, value))


def finite_min(values):
    """Return min finite value, or nan if no finite values exist."""
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    return min(finite_values) if len(finite_values) > 0 else float('nan')


def finite_max(values):
    """Return max finite value, or nan if no finite values exist."""
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    return max(finite_values) if len(finite_values) > 0 else float('nan')


def value_or_default(value, default):
    """Use default when an optional metric is not applicable."""
    return default if value is None else value


def normalize_satisfaction_method(method):
    """Normalize user-facing satisfaction computation method names."""
    if method == 'sample':
        return 'sampling'
    if method not in ('exact', 'sampling', 'both'):
        raise RuntimeError("method should be 'exact', 'sampling', 'sample', or 'both'")
    return method


def compute_trace_branch_volumes(
        traces, volume_checker, volume_seed=None, volume_method=None,
        numCores=1):
    """Compute volume(R[-1]) once for each coherent ReLU trace."""
    if numCores > 1 and len(traces) > 1:
        print('Computing branch volumes using {} cores...'.format(numCores))
        with multiprocessing.Pool(numCores) as pool:
            return pool.map(compute_trace_branch_volume_worker, enumerate(traces))

    branch_volumes = []
    for trace_id, trace in enumerate(traces):
        branch_common_set = trace[-1]
        common_A, common_b = volume_checker.getBasePredicateConstraints(branch_common_set)
        branch_volume = volume_checker.computeBaseVolumeByHalfspace(common_A, common_b)
        branch_volume = 0.0 if branch_volume is None else branch_volume
        print('trace {} R[-1] volume = {}'.format(trace_id, branch_volume))
        branch_volumes.append(branch_volume)
    return branch_volumes


def compute_trace_branch_volume_worker(trace_item):
    """Multiprocessing worker for volume(R[-1]) of one branch."""
    trace_id, trace = trace_item
    volume_checker = object.__new__(GetSatisfactionFraction)
    branch_common_set = trace[-1]
    common_A, common_b = volume_checker.getBasePredicateConstraints(branch_common_set)
    branch_volume = volume_checker.computeBaseVolumeByHalfspace(common_A, common_b)
    return 0.0 if branch_volume is None else branch_volume


def evaluate_trace_StarTL(
        trace, spec, trace_id=0, num_samples=1000, seed=0,
        burn_in=None, thinning=1, lp_solver='linprog', method='sampling',
        branch_volume=None, compute_exact_robustness=False,
        compute_region_balls=False):
    """Evaluate one relu split coherent Star trace with StarTL exact and sampling methods."""
    method = normalize_satisfaction_method(method)

    expanded_spec = ExpandedFormula(spec, T=len(trace))
    exact_result = None
    sampling_result = None
    checker = None
    if method in ('exact', 'both'):
        checker = GetSatisfactionFraction(
            trace,
            expanded_spec,
            method='exact',
            num_samples=num_samples,
            seed=None if seed is None else seed + trace_id,
            burn_in=burn_in,
            thinning=thinning,
            lp_solver=lp_solver,
            compute_exact_robustness=compute_exact_robustness,
            compute_region_balls=False
        )
        exact_result = checker.getSatisfactionFraction()

    if method in ('sampling', 'both'):
        checker = GetSatisfactionFraction(
            trace,
            expanded_spec,
            method='sampling',
            num_samples=num_samples,
            seed=None if seed is None else seed + trace_id,
            burn_in=burn_in,
            thinning=thinning,
            lp_solver=lp_solver,
            compute_exact_robustness=compute_exact_robustness,
            compute_region_balls=False
        )
        sampling_result = checker.getSatisfactionFraction()

    if checker is None:
        raise RuntimeError('no satisfaction checker was created')

    selected_result = sampling_result if sampling_result is not None else exact_result
    if branch_volume is None:
        branch_common_set = trace[-1]
        common_A, common_b = checker.getBasePredicateConstraints(branch_common_set)
        branch_volume = checker.computeBaseVolumeByHalfspace(common_A, common_b)
        branch_volume = 0.0 if branch_volume is None else branch_volume
    exact_satisfying_volume = (
        branch_volume * exact_result['satisfying_fraction']
        if exact_result is not None else None
    )
    sampling_satisfying_volume = (
        branch_volume * sampling_result['satisfying_fraction']
        if sampling_result is not None else None
    )
    return {
        'trace_id': trace_id,
        'num_time_sets': len(trace),
        'rho_lb': selected_result['rho_lb'],
        'rho_ub': selected_result['rho_ub'],
        'exact_rho_lb': selected_result.get('exact_rho_lb', float('nan')),
        'exact_rho_ub': selected_result.get('exact_rho_ub', float('nan')),
        'exact_method': exact_result['method'] if exact_result is not None else None,
        'sampling_method': sampling_result['method'] if sampling_result is not None else None,
        'exact_satisfying_fraction': (
            exact_result['satisfying_fraction'] if exact_result is not None else None
        ),
        'sampling_satisfying_fraction': (
            sampling_result['satisfying_fraction'] if sampling_result is not None else None
        ),
        'branch_volume': branch_volume,
        'exact_satisfying_volume': exact_satisfying_volume,
        'sampling_satisfying_volume': sampling_satisfying_volume,
        'num_samples': sampling_result.get('num_samples', num_samples) if sampling_result is not None else None,
        'n_satisfied': sampling_result.get('n_satisfied', None) if sampling_result is not None else None,
    }


def evaluate_trace_StarTL_worker(args):
    """Multiprocessing worker for one StarTL trace/spec check."""
    return evaluate_trace_StarTL(*args)


def verify_temporal_specs_DLNNCS_ACC(
        ncs, verifyPRM, num_samples=1000, seed=0, burn_in=None,
        thinning=1, method='sampling', max_traces=None, volume_seed=None,
        compute_exact_robustness=False, compute_region_balls=False):
    """Verify StarTL specs on Star traces, similar to NNCS full-analysis API."""
    method = normalize_satisfaction_method(method)
    if not isinstance(ncs, NNCS):
        raise RuntimeError('ncs should be an NNCS object')
    if not isinstance(verifyPRM, VerifyPRM_NNCS):
        raise RuntimeError('verifyPRM should be a VerifyPRM_NNCS object')
    if verifyPRM.initSet is None:
        raise RuntimeError('verifyPRM.initSet is required')
    if verifyPRM.temporalSpecs is None:
        raise RuntimeError('verifyPRM.temporalSpecs is required')

    reachPRM = ReachPRM_NNCS()
    reachPRM.initSet = copy.deepcopy(verifyPRM.initSet)
    reachPRM.refInputs = copy.deepcopy(verifyPRM.refInputs)
    reachPRM.numSteps = copy.deepcopy(verifyPRM.numSteps)
    reachPRM.filterProb = copy.deepcopy(verifyPRM.pf)
    reachPRM.lpSolver = copy.deepcopy(verifyPRM.lpSolver)
    reachPRM.show = copy.deepcopy(verifyPRM.show)
    reachPRM.numCores = copy.deepcopy(verifyPRM.numCores)
    if reachPRM.filterProb != 0.0:
        raise RuntimeError('Star trace reachability does not support probability filtering')

    print('Get Star traces using reachDFS_DLNNCS...')
    start = time.time()
    traces, p_ignored = reachDFS_DLNNCS(ncs, reachPRM)
    capped = False
    if max_traces is not None and len(traces) > max_traces:
        traces = traces[:max_traces]
        capped = True
    reachTime = time.time() - start
    print('Total Star traces: {}'.format(len(traces)))
    if capped:
        print('Trace generation stopped early at max_traces = {}'.format(max_traces))

    specs = verifyPRM.temporalSpecs
    init_volume_checker = object.__new__(GetSatisfactionFraction)
    init_A, init_b = init_volume_checker.getBasePredicateConstraints(verifyPRM.initSet)
    init_volume = init_volume_checker.computeBaseVolumeByHalfspace(init_A, init_b)
    init_volume = 0.0 if init_volume is None else init_volume
    print('initial input set volume = {}'.format(init_volume))
    start = time.time()
    branch_volumes = compute_trace_branch_volumes(
        traces,
        init_volume_checker,
        volume_seed=volume_seed,
        volume_method='halfspace',
        numCores=verifyPRM.numCores
    )
    branchVolumeTime = time.time() - start
    total_trace_volume = sum(branch_volumes)

    checkingTime = []
    verifyTime = []
    rho_lb = []
    rho_ub = []
    exact_rho_lb = []
    exact_rho_ub = []
    trace_volume = []
    exact_sat_volume = []
    sampling_sat_volume = []
    exact_sat_fraction = []
    sampling_sat_fraction = []
    raw_exact_sat_fraction = []
    raw_sampling_sat_fraction = []
    volume_coverage_ratio = []
    exact_trace_union_sat_fraction = []
    sampling_trace_union_sat_fraction = []
    num_sat_traces = []
    num_violated_traces = []
    num_mixed_traces = []
    trace_results = []

    print('Verifying traces against StarTL temporal specifications...')
    for spec_id, spec in enumerate(specs):
        start = time.time()
        worker_args = [
            (
                trace,
                spec,
                trace_id,
                num_samples,
                seed,
                burn_in,
                thinning,
                verifyPRM.lpSolver,
                method,
                branch_volumes[trace_id],
                compute_exact_robustness,
                compute_region_balls
            )
            for trace_id, trace in enumerate(traces)
        ]
        if verifyPRM.numCores > 1 and len(worker_args) > 1:
            print(
                'Checking spec {} on {} traces using {} cores...'
                .format(spec_id, len(worker_args), verifyPRM.numCores)
            )
            with multiprocessing.Pool(verifyPRM.numCores) as pool:
                spec_trace_results = pool.map(evaluate_trace_StarTL_worker, worker_args)
        else:
            spec_trace_results = []
            for trace_id, args in enumerate(worker_args):
                print('Verifying trace {} against spec {}...'.format(trace_id, spec_id))
                trace_result = evaluate_trace_StarTL_worker(args)
                spec_trace_results.append(trace_result)

        checking_time = time.time() - start
        exact_satisfying_volume = (
            sum(item['exact_satisfying_volume'] for item in spec_trace_results)
            if method in ('exact', 'both') else None
        )
        sampling_satisfying_volume = (
            sum(item['sampling_satisfying_volume'] for item in spec_trace_results)
            if method in ('sampling', 'both') else None
        )
        exact_aggregate_fraction = (
            exact_satisfying_volume / init_volume
            if exact_satisfying_volume is not None and init_volume > 0.0 else None
        )
        sampling_aggregate_fraction = (
            sampling_satisfying_volume / init_volume
            if sampling_satisfying_volume is not None and init_volume > 0.0 else None
        )
        exact_fraction = clamp_fraction(exact_aggregate_fraction)
        sampling_fraction = clamp_fraction(sampling_aggregate_fraction)
        coverage_init_ratio = (
            total_trace_volume / init_volume
            if init_volume > 0.0 else None
        )
        exact_union_fraction = (
            exact_satisfying_volume / total_trace_volume
            if exact_satisfying_volume is not None and total_trace_volume > 0.0 else None
        )
        sampling_union_fraction = (
            sampling_satisfying_volume / total_trace_volume
            if sampling_satisfying_volume is not None and total_trace_volume > 0.0 else None
        )
        sat_trace_count = sum(item['rho_lb'] >= 0.0 for item in spec_trace_results)
        violated_trace_count = sum(item['rho_ub'] < 0.0 for item in spec_trace_results)
        mixed_trace_count = sum(
            item['rho_lb'] < 0.0 and item['rho_ub'] >= 0.0
            for item in spec_trace_results
        )
        print(
            'spec {} trace robustness cases: satisfied = {}, violated = {}, mixed = {}'
            .format(spec_id, sat_trace_count, violated_trace_count, mixed_trace_count)
        )

        checkingTime.append(checking_time)
        verifyTime.append(checking_time + reachTime + branchVolumeTime)
        rho_lb.append(min(item['rho_lb'] for item in spec_trace_results))
        rho_ub.append(max(item['rho_ub'] for item in spec_trace_results))
        exact_rho_lb.append(finite_min(item['exact_rho_lb'] for item in spec_trace_results))
        exact_rho_ub.append(finite_max(item['exact_rho_ub'] for item in spec_trace_results))
        trace_volume.append(total_trace_volume)
        exact_sat_volume.append(exact_satisfying_volume)
        sampling_sat_volume.append(sampling_satisfying_volume)
        exact_sat_fraction.append(exact_fraction)
        sampling_sat_fraction.append(sampling_fraction)
        raw_exact_sat_fraction.append(exact_aggregate_fraction)
        raw_sampling_sat_fraction.append(sampling_aggregate_fraction)
        volume_coverage_ratio.append(coverage_init_ratio)
        exact_trace_union_sat_fraction.append(exact_union_fraction)
        sampling_trace_union_sat_fraction.append(sampling_union_fraction)
        num_sat_traces.append(sat_trace_count)
        num_violated_traces.append(violated_trace_count)
        num_mixed_traces.append(mixed_trace_count)
        trace_results.append(spec_trace_results)

    return {
        'traces': traces,
        'branch_volumes': branch_volumes,
        'num_traces': len(traces),
        'trace_generation_capped': capped,
        'p_ignored': p_ignored,
        'init_volume': init_volume,
        'rho_lb': rho_lb,
        'rho_ub': rho_ub,
        'exact_rho_lb': exact_rho_lb,
        'exact_rho_ub': exact_rho_ub,
        'trace_volume': trace_volume,
        'exact_sat_volume': exact_sat_volume,
        'sampling_sat_volume': sampling_sat_volume,
        'exact_sat_fraction': exact_sat_fraction,
        'sampling_sat_fraction': sampling_sat_fraction,
        'raw_exact_sat_fraction': raw_exact_sat_fraction,
        'raw_sampling_sat_fraction': raw_sampling_sat_fraction,
        'volume_coverage_ratio': volume_coverage_ratio,
        'exact_trace_union_sat_fraction': exact_trace_union_sat_fraction,
        'sampling_trace_union_sat_fraction': sampling_trace_union_sat_fraction,
        'num_sat_traces': num_sat_traces,
        'num_violated_traces': num_violated_traces,
        'num_mixed_traces': num_mixed_traces,
        'reachTime': reachTime,
        'branchVolumeTime': branchVolumeTime,
        'volume_seed': volume_seed,
        'checkingTime': checkingTime,
        'verifyTime': verifyTime,
        'trace_results': trace_results,
    }


def verify_acc_StarTL(
        net='controller_5_20', plant='linear', spec_id=6, initSet_id=5,
        numSteps=5, t=2, num_samples=1000, seed=0, burn_in=None,
        thinning=1, lp_solver='linprog', method='sampling', max_traces=None,
        volume_seed=None, numCores=1, compute_exact_robustness=False,
        compute_region_balls=False):
    """Run ACC Star reachability and evaluate StarTL satisfaction."""
    method = normalize_satisfaction_method(method)

    print('Loading ACC controller, plant, reference inputs, and spec...')
    ncs, specs, initSet, refInputs = load_acc_model_dStarTL(
        netname=net,
        plant=plant,
        spec_ids=[spec_id],
        initSet_id=initSet_id,
        T=numSteps,
        t=t
    )
    spec = specs[0]

    print('Constructed Star initial set:')
    print('  dim = {}'.format(initSet.dim))
    print('  nVars = {}'.format(initSet.nVars))

    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSet)
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = 0.0
    verifyPRM.lpSolver = lp_solver
    verifyPRM.show = False
    verifyPRM.numCores = numCores
    verifyPRM.temporalSpecs = [spec]

    analysis = verify_temporal_specs_DLNNCS_ACC(
        ncs,
        verifyPRM,
        num_samples=num_samples,
        seed=seed,
        burn_in=burn_in,
        thinning=thinning,
        method=method,
        max_traces=max_traces,
        volume_seed=volume_seed,
        compute_exact_robustness=compute_exact_robustness,
        compute_region_balls=compute_region_balls
    )
    selected_fraction = (
        analysis['exact_sat_fraction'][0]
        if method == 'exact' else analysis['sampling_sat_fraction'][0]
    )
    if method == 'both':
        selected_fraction = analysis['sampling_sat_fraction'][0]

    result = {
        'net': net,
        'system': 'ACC',
        'spec_id': spec_id,
        'initSet_id': initSet_id,
        'numSteps': numSteps,
        'method': method,
        'volume_seed': volume_seed,
        'num_traces': analysis['num_traces'],
        'num_sampling_samples_per_trace': num_samples,
        'initial_set_volume': analysis['init_volume'],
        'summed_trace_volume': analysis['trace_volume'][0],
        'rho_lb': analysis['rho_lb'][0],
        'rho_ub': analysis['rho_ub'][0],
        'exact_rho_lb': analysis['exact_rho_lb'][0],
        'exact_rho_ub': analysis['exact_rho_ub'][0],
        'num_sat_traces': analysis['num_sat_traces'][0],
        'num_violated_traces': analysis['num_violated_traces'][0],
        'num_mixed_traces': analysis['num_mixed_traces'][0],
        'exact_sat_volume': analysis['exact_sat_volume'][0],
        'exact_satisfaction_fraction': analysis['exact_sat_fraction'][0],
        'sampling_sat_volume': analysis['sampling_sat_volume'][0],
        'sampling_satisfaction_fraction': analysis['sampling_sat_fraction'][0],
        'sat_fraction': selected_fraction,
        'trace_generation_capped': analysis['trace_generation_capped'],
        'reach_time': analysis['reachTime'],
        'branch_volume_time': analysis['branchVolumeTime'],
        'checking_time': analysis['checkingTime'][0],
        'verification_time': analysis['verifyTime'][0],
        'raw_exact_satisfaction_fraction': analysis['raw_exact_sat_fraction'][0],
        'raw_sampling_satisfaction_fraction': analysis['raw_sampling_sat_fraction'][0],
        'volume_coverage_ratio': analysis['volume_coverage_ratio'][0],
        'exact_trace_union_satisfaction_fraction': analysis['exact_trace_union_sat_fraction'][0],
        'sampling_trace_union_satisfaction_fraction': analysis['sampling_trace_union_sat_fraction'][0],
        'trace_results': analysis['trace_results'][0],
        'full_analysis': analysis,
    }

    print('\n================ ACC dStarTL Summary ================')
    print('net = {}'.format(net))
    print('spec_id = {}'.format(spec_id))
    print('initSet_id = {}'.format(initSet_id))
    print('numSteps = {}'.format(numSteps))
    print('method = {}'.format(method))
    print('num_traces = {}'.format(result['num_traces']))
    print('sampling num_samples per trace = {}'.format(num_samples))
    print('initial set volume = {}'.format(result['initial_set_volume']))
    print('summed trace volume = {}'.format(result['summed_trace_volume']))
    print('num sat traces = {}'.format(result['num_sat_traces']))
    print('num violated traces = {}'.format(result['num_violated_traces']))
    print('num mixed traces = {}'.format(result['num_mixed_traces']))
    if method in ('exact', 'both'):
        print('exact sat volume = {}'.format(result['exact_sat_volume']))
        print('exact satisfaction fraction = {}'.format(result['exact_satisfaction_fraction']))
    if method in ('sampling', 'both'):
        print('sampling sat volume = {}'.format(result['sampling_sat_volume']))
        print('sampling satisfaction fraction = {}'.format(result['sampling_satisfaction_fraction']))
    print('checking time = {:.6f} seconds'.format(result['checking_time']))

    return result


def print_acc_StarTL_table(rows, numSteps, method='sampling'):
    """Print one ACC StarTL result table for a fixed reachability horizon."""
    method = normalize_satisfaction_method(method)

    print('\n================ ACC dStarTL Table: numSteps = {} ================'.format(numSteps))
    header = (
        '{:<8} {:<5} {:<8} {:<8} {:>9} {:>9} {:>9} {:>9} {:>12} '
        '{:>7} {:>7} {:>7} {:>10} {:>10} {:>10}'
    ).format(
        'system', 'T', 'spec_id', 'branches', 'rho_lb', 'rho_ub',
        'exact_lb', 'exact_ub', 'sat_fraction',
        'sat', 'viol', 'mixed', 'reach_t', 'check_t', 'verify_t'
    )
    print(header)
    print('-' * len(header))
    for row in rows:
        print(
            '{:<8} {:<5} {:<8} {:<8} {:>9.4g} {:>9.4g} {:>9.4g} {:>9.4g} '
            '{:>12.6g} {:>7} {:>7} {:>7} {:>10.4f} {:>10.4f} {:>10.4f}'
            .format(
                row['system'],
                row['numSteps'],
                row['spec_id'],
                row['num_traces'],
                row['rho_lb'],
                row['rho_ub'],
                row['exact_rho_lb'],
                row['exact_rho_ub'],
                row['sat_fraction'],
                row['num_sat_traces'],
                row['num_violated_traces'],
                row['num_mixed_traces'],
                row['reach_time'],
                row['checking_time'],
                row['verification_time']
            )
        )


def verify_acc_StarTL_table(
        net='controller_5_20', plant='linear', spec_ids=None, initSet_id=5,
        numSteps_list=None, t=5, num_samples=1000, seed=0, burn_in=None,
        thinning=1, lp_solver='linprog', method='sampling', max_traces=None,
        volume_seed=None, numCores=1, compute_exact_robustness=False,
        compute_region_balls=False):
    """Verify all selected ACC StarTL specs and print grouped result tables."""
    method = normalize_satisfaction_method(method)

    all_rows = []
    grouped_rows = {}
    for numSteps in numSteps_list:
        print('\nRunning ACC dStarTL for numSteps = {}...'.format(numSteps))
        ncs, specs, initSet, refInputs = load_acc_model_dStarTL(
            netname=net,
            plant=plant,
            spec_ids=spec_ids,
            initSet_id=initSet_id,
            T=numSteps,
            t=t
        )

        verifyPRM = VerifyPRM_NNCS()
        verifyPRM.initSet = copy.deepcopy(initSet)
        verifyPRM.refInputs = copy.deepcopy(refInputs)
        verifyPRM.numSteps = numSteps
        verifyPRM.pf = 0.0
        verifyPRM.lpSolver = lp_solver
        verifyPRM.show = False
        verifyPRM.numCores = numCores
        verifyPRM.temporalSpecs = specs

        analysis = verify_temporal_specs_DLNNCS_ACC(
            ncs,
            verifyPRM,
            num_samples=num_samples,
            seed=seed,
            burn_in=burn_in,
            thinning=thinning,
            method=method,
            max_traces=max_traces,
            volume_seed=volume_seed,
            compute_exact_robustness=compute_exact_robustness,
            compute_region_balls=compute_region_balls
        )

        rows = []
        for index, spec_id in enumerate(spec_ids):
            selected_fraction = (
                analysis['exact_sat_fraction'][index]
                if method == 'exact' else analysis['sampling_sat_fraction'][index]
            )
            if method == 'both':
                selected_fraction = analysis['sampling_sat_fraction'][index]
            row = {
                'net': net,
                'system': 'ACC',
                'spec_id': spec_id,
                'initSet_id': initSet_id,
                'numSteps': numSteps,
                'method': method,
                'volume_seed': volume_seed,
                'num_traces': analysis['num_traces'],
                'num_sampling_samples_per_trace': num_samples,
                'initial_set_volume': analysis['init_volume'],
                'summed_trace_volume': analysis['trace_volume'][index],
                'rho_lb': analysis['rho_lb'][index],
                'rho_ub': analysis['rho_ub'][index],
                'exact_rho_lb': analysis['exact_rho_lb'][index],
                'exact_rho_ub': analysis['exact_rho_ub'][index],
                'num_sat_traces': analysis['num_sat_traces'][index],
                'num_violated_traces': analysis['num_violated_traces'][index],
                'num_mixed_traces': analysis['num_mixed_traces'][index],
                'exact_sat_volume': analysis['exact_sat_volume'][index],
                'exact_satisfaction_fraction': analysis['exact_sat_fraction'][index],
                'sampling_sat_volume': analysis['sampling_sat_volume'][index],
                'sampling_satisfaction_fraction': analysis['sampling_sat_fraction'][index],
                'sat_fraction': selected_fraction,
                'reach_time': analysis['reachTime'],
                'branch_volume_time': analysis['branchVolumeTime'],
                'checking_time': analysis['checkingTime'][index],
                'verification_time': analysis['verifyTime'][index],
                'trace_generation_capped': analysis['trace_generation_capped'],
                'full_analysis': analysis,
            }
            rows.append(row)
            all_rows.append(row)

        grouped_rows[numSteps] = rows
        print_acc_StarTL_table(rows, numSteps, method)

    return {
        'net': net,
        'initSet_id': initSet_id,
        'method': method,
        'volume_seed': volume_seed,
        'compute_region_balls': compute_region_balls,
        'compute_exact_robustness': compute_exact_robustness,
        'spec_ids': spec_ids,
        'numSteps_list': numSteps_list,
        'num_sampling_samples_per_trace': num_samples,
        'numCores': numCores,
        'rows': all_rows,
        'grouped_rows': grouped_rows,
    }


if __name__ == '__main__':
   
    # Use a string for one controller:
    # net = 'controller_3_20'
    # Or use a list to run multiple controllers:
    net = ['controller_5_20']
    plant = 'linear'
    spec_ids = list(range(6))
    # spec_ids = [5]
    initSet_id = 5
    numSteps_list = [10, 20, 30,40,50]
    t = 5
    num_samples = 10000
    seed = 0
    burn_in = None
    thinning = 1
    lp_solver = 'linprog'
    method = 'exact'  # choose from: 'exact', 'sampling'/'sample', 'both'
    volume_seed = 0
    compute_region_balls = False
    compute_exact_robustness = True
    numCores = 1
    max_traces = None

    if isinstance(net, str):
        nets = [net]
    elif isinstance(net, list):
        nets = net
    else:
        raise RuntimeError('net should be a string or a list of strings')

    results = []
    for net_name in nets:
        print('\n\n================ Running ACC StarTL for {} ================'.format(net_name))
        result = verify_acc_StarTL_table(
            net=net_name,
            plant=plant,
            spec_ids=spec_ids,
            initSet_id=initSet_id,
            numSteps_list=numSteps_list,
            t=t,
            num_samples=num_samples,
            seed=seed,
            burn_in=burn_in,
            thinning=thinning,
            lp_solver=lp_solver,
            method=method,
            volume_seed=volume_seed,
            numCores=numCores,
            compute_exact_robustness=compute_exact_robustness,
            compute_region_balls=compute_region_balls,
            max_traces=max_traces
        )
        results.append(result)
