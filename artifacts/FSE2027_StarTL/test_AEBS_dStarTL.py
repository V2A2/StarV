"""
AEBS StarTL verification example.

"""

import copy
import math
import multiprocessing
import time
from StarV.nncs.nncs import AEBS_NNCS, ReachPRM_NNCS, VerifyPRM_NNCS, reachDFS_DLNNCS
from StarV.util.load import load_AEBS_model_dStarTL, load_AEBS_temporal_specs
from StarV.spec.dStarTL import ExpandedFormula, GetSatisfactionFraction


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
        traces, volume_checker, volume_seed=None, volume_method='halfspace',
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
    """Evaluate one coherent Star trace with StarTL exact and sampling methods."""
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
    if not selected_result.get('trace_feasible', True):
        return {
            'trace_id': trace_id,
            'num_time_sets': len(trace),
            'trace_feasible': False,
            'rho_lb': None,
            'rho_ub': None,
            'exact_rho_lb': float('nan'),
            'exact_rho_ub': float('nan'),
            'exact_method': exact_result['method'] if exact_result is not None else None,
            'sampling_method': sampling_result['method'] if sampling_result is not None else None,
            'exact_satisfying_fraction': 0.0 if exact_result is not None else None,
            'sampling_satisfying_fraction': 0.0 if sampling_result is not None else None,
            'branch_volume': 0.0,
            'exact_satisfying_volume': 0.0 if exact_result is not None else None,
            'sampling_satisfying_volume': 0.0 if sampling_result is not None else None,
            'num_samples': sampling_result.get('num_samples', num_samples) if sampling_result is not None else None,
            'n_satisfied': sampling_result.get('n_satisfied', None) if sampling_result is not None else None,
        }

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
        'trace_feasible': True,
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


def verify_temporal_specs_AEBS_StarTL(
        aebs, verifyPRM, num_samples=1000, seed=0, burn_in=None,
        thinning=1, method='sampling', max_traces=None, volume_seed=0,
        compute_exact_robustness=False, compute_region_balls=False):
    """Verify StarTL specs on AEBS Star traces."""
    method = normalize_satisfaction_method(method)
    if not isinstance(aebs, AEBS_NNCS):
        raise RuntimeError('aebs should be an AEBS_NNCS object')
    if not isinstance(verifyPRM, VerifyPRM_NNCS):
        raise RuntimeError('verifyPRM should be a VerifyPRM_NNCS object')
    if verifyPRM.initSet is None:
        raise RuntimeError('verifyPRM.initSet is required')
    if verifyPRM.temporalSpecs is None:
        raise RuntimeError('verifyPRM.temporalSpecs is required')

    reachPRM = ReachPRM_NNCS()
    reachPRM.initSet = copy.deepcopy(verifyPRM.initSet)
    reachPRM.numSteps = copy.deepcopy(verifyPRM.numSteps)
    reachPRM.filterProb = copy.deepcopy(verifyPRM.pf)
    reachPRM.lpSolver = copy.deepcopy(verifyPRM.lpSolver)
    reachPRM.show = copy.deepcopy(verifyPRM.show)
    reachPRM.numCores = copy.deepcopy(verifyPRM.numCores)
    if reachPRM.filterProb != 0.0:
        raise RuntimeError('StarTL verification supports pf = 0.0 only')

    print('Get AEBS Star traces using reachDFS_DLNNCS...')
    start = time.time()
    traces, p_ignored = reachDFS_DLNNCS(aebs, reachPRM)
    capped = False
    if max_traces is not None and len(traces) > max_traces:
        traces = traces[:max_traces]
        capped = True
    reachTime = time.time() - start
    print('Total Star traces: {}'.format(len(traces)))
    num_generated_traces = len(traces)
    if capped:
        print('Trace generation stopped early at max_traces = {}'.format(max_traces))

    volume_checker = object.__new__(GetSatisfactionFraction)
    init_A, init_b = volume_checker.getBasePredicateConstraints(verifyPRM.initSet)
    init_volume = volume_checker.computeBaseVolumeByHalfspace(init_A, init_b)
    init_volume = 0.0 if init_volume is None else init_volume
    print('initial input set volume = {}'.format(init_volume))

    start = time.time()
    branch_volumes = compute_trace_branch_volumes(
        traces,
        volume_checker,
        volume_seed=volume_seed,
        volume_method='halfspace',
        numCores=verifyPRM.numCores
    )
    branchVolumeTime = time.time() - start
    nonempty_trace_data = [
        (trace, branch_volume)
        for trace, branch_volume in zip(traces, branch_volumes)
        if branch_volume > 0.0
    ]
    num_volume_filtered_traces = len(traces) - len(nonempty_trace_data)
    if len(nonempty_trace_data) != len(traces):
        print(
            'Filtered {} zero-volume/infeasible traces before StarTL checking'
            .format(num_volume_filtered_traces)
        )
    traces = [trace for trace, _ in nonempty_trace_data]
    branch_volumes = [branch_volume for _, branch_volume in nonempty_trace_data]
    if len(traces) == 0:
        raise RuntimeError('all AEBS traces have zero volume after reachability')

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
    num_checked_traces = []
    num_filtered_traces = []
    trace_results = []

    print('Verifying traces against AEBS StarTL temporal specifications...')
    for local_spec_id, spec in enumerate(verifyPRM.temporalSpecs):
        start = time.time()
        spec_trace_results = []
        spec_infeasible_trace_count = 0
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
                .format(local_spec_id, len(worker_args), verifyPRM.numCores)
            )
            with multiprocessing.Pool(verifyPRM.numCores) as pool:
                trace_results_for_spec = pool.map(evaluate_trace_StarTL_worker, worker_args)
        else:
            trace_results_for_spec = []
            for trace_id, args in enumerate(worker_args):
                print('Verifying trace {} against spec {}...'.format(trace_id, local_spec_id))
                trace_results_for_spec.append(evaluate_trace_StarTL_worker(args))

        for trace_id, trace_result in enumerate(trace_results_for_spec):
            if trace_result.get('trace_feasible', True):
                spec_trace_results.append(trace_result)
            else:
                spec_infeasible_trace_count += 1
                print('Skipping infeasible trace {} for spec {}'.format(trace_id, local_spec_id))

        checking_time = time.time() - start
        if len(spec_trace_results) == 0:
            print('spec {} has no feasible traces after dStarTL feasibility filtering'.format(local_spec_id))
            checkingTime.append(checking_time)
            verifyTime.append(checking_time + reachTime + branchVolumeTime)
            rho_lb.append(float('nan'))
            rho_ub.append(float('nan'))
            exact_rho_lb.append(float('nan'))
            exact_rho_ub.append(float('nan'))
            trace_volume.append(0.0)
            exact_sat_volume.append(0.0 if method in ('exact', 'both') else None)
            sampling_sat_volume.append(0.0 if method in ('sampling', 'both') else None)
            exact_sat_fraction.append(0.0 if method in ('exact', 'both') else None)
            sampling_sat_fraction.append(0.0 if method in ('sampling', 'both') else None)
            raw_exact_sat_fraction.append(0.0 if method in ('exact', 'both') else None)
            raw_sampling_sat_fraction.append(0.0 if method in ('sampling', 'both') else None)
            volume_coverage_ratio.append(0.0)
            exact_trace_union_sat_fraction.append(0.0 if method in ('exact', 'both') else None)
            sampling_trace_union_sat_fraction.append(0.0 if method in ('sampling', 'both') else None)
            num_sat_traces.append(0)
            num_violated_traces.append(0)
            num_mixed_traces.append(0)
            num_checked_traces.append(0)
            num_filtered_traces.append(num_volume_filtered_traces + spec_infeasible_trace_count)
            trace_results.append(spec_trace_results)
            continue

        exact_satisfying_volume = (
            sum(item['exact_satisfying_volume'] for item in spec_trace_results)
            if method in ('exact', 'both') else None
        )
        sampling_satisfying_volume = (
            sum(item['sampling_satisfying_volume'] for item in spec_trace_results)
            if method in ('sampling', 'both') else None
        )
        spec_trace_volume = sum(item['branch_volume'] for item in spec_trace_results)
        exact_aggregate_fraction = (
            exact_satisfying_volume / init_volume
            if exact_satisfying_volume is not None and init_volume > 0.0 else None
        )
        sampling_aggregate_fraction = (
            sampling_satisfying_volume / init_volume
            if sampling_satisfying_volume is not None and init_volume > 0.0 else None
        )
        exact_union_fraction = (
            exact_satisfying_volume / spec_trace_volume
            if exact_satisfying_volume is not None and spec_trace_volume > 0.0 else None
        )
        sampling_union_fraction = (
            sampling_satisfying_volume / spec_trace_volume
            if sampling_satisfying_volume is not None and spec_trace_volume > 0.0 else None
        )
        coverage_init_ratio = (
            spec_trace_volume / init_volume
            if init_volume > 0.0 else None
        )
        sat_trace_count = sum(item['rho_lb'] >= 0.0 for item in spec_trace_results)
        violated_trace_count = sum(item['rho_ub'] < 0.0 for item in spec_trace_results)
        mixed_trace_count = sum(
            item['rho_lb'] < 0.0 and item['rho_ub'] >= 0.0
            for item in spec_trace_results
        )
        print(
            'spec {} trace robustness cases: satisfied = {}, violated = {}, mixed = {}'
            .format(local_spec_id, sat_trace_count, violated_trace_count, mixed_trace_count)
        )

        checkingTime.append(checking_time)
        verifyTime.append(checking_time + reachTime + branchVolumeTime)
        rho_lb.append(min(item['rho_lb'] for item in spec_trace_results))
        rho_ub.append(max(item['rho_ub'] for item in spec_trace_results))
        exact_rho_lb.append(finite_min(item['exact_rho_lb'] for item in spec_trace_results))
        exact_rho_ub.append(finite_max(item['exact_rho_ub'] for item in spec_trace_results))
        trace_volume.append(spec_trace_volume)
        exact_sat_volume.append(exact_satisfying_volume)
        sampling_sat_volume.append(sampling_satisfying_volume)
        exact_sat_fraction.append(clamp_fraction(exact_aggregate_fraction))
        sampling_sat_fraction.append(clamp_fraction(sampling_aggregate_fraction))
        raw_exact_sat_fraction.append(exact_aggregate_fraction)
        raw_sampling_sat_fraction.append(sampling_aggregate_fraction)
        volume_coverage_ratio.append(coverage_init_ratio)
        exact_trace_union_sat_fraction.append(exact_union_fraction)
        sampling_trace_union_sat_fraction.append(sampling_union_fraction)
        num_sat_traces.append(sat_trace_count)
        num_violated_traces.append(violated_trace_count)
        num_mixed_traces.append(mixed_trace_count)
        num_checked_traces.append(len(spec_trace_results))
        num_filtered_traces.append(num_volume_filtered_traces + spec_infeasible_trace_count)
        trace_results.append(spec_trace_results)

    return {
        'traces': traces,
        'branch_volumes': branch_volumes,
        'num_generated_traces': num_generated_traces,
        'num_traces': len(traces),
        'num_volume_filtered_traces': num_volume_filtered_traces,
        'num_checked_traces': num_checked_traces,
        'num_filtered_traces': num_filtered_traces,
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
        'checkingTime': checkingTime,
        'verifyTime': verifyTime,
        'trace_results': trace_results,
    }


def print_AEBS_StarTL_table(rows, method='sampling'):
    """Print AEBS dStarTL result table."""
    method = normalize_satisfaction_method(method)
    print('\n================ AEBS dStarTL Table ================')
    header = (
        '{:<8} {:<8} {:<5} {:<8} {:<8} {:>9} {:>9} {:>9} {:>9} {:>12} '
        '{:>7} {:>7} {:>7} {:>10} {:>10} {:>10}'
    ).format(
        'system', 'initSet', 'T', 'spec_id', 'branches', 'rho_lb', 'rho_ub',
        'exact_lb', 'exact_ub', 'sat_fraction',
        'sat', 'viol', 'mixed', 'reach_t', 'check_t', 'verify_t'
    )
    print(header)
    print('-' * len(header))
    for row in rows:
        print(
            '{:<8} {:<8} {:<5} {:<8} {:<8} {:>9.4g} {:>9.4g} {:>9.4g} {:>9.4g} '
            '{:>12.6g} {:>7} {:>7} {:>7} {:>10.4f} {:>10.4f} {:>10.4f}'
            .format(
                row['system'],
                row['initSet_name'],
                row['numSteps'],
                row['spec_id'],
                row['num_generated_traces'],
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


def verify_AEBS_StarTL_table(
        initSet_ids=None, numSteps_list=None, spec_ids=None, num_samples=1000,
        seed=0, burn_in=None, thinning=1, lp_solver='linprog',
        method='sampling', numCores=1, max_traces=None, volume_seed=0,
        compute_exact_robustness=False, compute_region_balls=False):
    """Run AEBS Star reachability and evaluate StarTL specifications."""
    method = normalize_satisfaction_method(method)
    print('Loading AEBS controller, transformer, plant, and Star initial sets...')
    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model_dStarTL()
    aebs = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)
    specs_all = load_AEBS_temporal_specs()

    if initSet_ids is None:
        initSet_ids = list(range(len(initSets)))
    if numSteps_list is None:
        numSteps_list = [10, 20, 40, 50]
    if spec_ids is None:
        spec_ids = list(range(len(specs_all)))

    specs = [copy.deepcopy(specs_all[spec_id]) for spec_id in spec_ids]
    rows = []
    grouped_rows = {}
    for initSet_id in initSet_ids:
        initset_rows = []
        for numSteps in numSteps_list:
            print('\nRunning AEBS dStarTL: X0_{}, numSteps = {}...'.format(initSet_id, numSteps))
            verifyPRM = VerifyPRM_NNCS()
            verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
            verifyPRM.numSteps = numSteps
            verifyPRM.pf = 0.0
            verifyPRM.lpSolver = lp_solver
            verifyPRM.show = False
            verifyPRM.numCores = numCores
            verifyPRM.temporalSpecs = copy.deepcopy(specs)

            analysis = verify_temporal_specs_AEBS_StarTL(
                aebs,
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

            for local_index, spec_id in enumerate(spec_ids):
                selected_fraction = (
                    analysis['exact_sat_fraction'][local_index]
                    if method == 'exact' else analysis['sampling_sat_fraction'][local_index]
                )
                if method == 'both':
                    selected_fraction = analysis['sampling_sat_fraction'][local_index]
                row = {
                    'system': 'AEBS',
                    'initSet_id': initSet_id,
                    'initSet_name': 'X0_{}'.format(initSet_id),
                    'spec_id': spec_id,
                    'numSteps': numSteps,
                    'method': method,
                    'num_generated_traces': analysis['num_generated_traces'],
                    'num_traces': analysis['num_traces'],
                    'num_checked_traces': analysis['num_checked_traces'][local_index],
                    'num_filtered_traces': analysis['num_filtered_traces'][local_index],
                    'num_volume_filtered_traces': analysis['num_volume_filtered_traces'],
                    'num_sampling_samples_per_trace': num_samples,
                    'initial_set_volume': analysis['init_volume'],
                    'summed_trace_volume': analysis['trace_volume'][local_index],
                    'coverage_ratio': analysis['volume_coverage_ratio'][local_index],
                    'rho_lb': analysis['rho_lb'][local_index],
                    'rho_ub': analysis['rho_ub'][local_index],
                    'exact_rho_lb': analysis['exact_rho_lb'][local_index],
                    'exact_rho_ub': analysis['exact_rho_ub'][local_index],
                    'num_sat_traces': analysis['num_sat_traces'][local_index],
                    'num_violated_traces': analysis['num_violated_traces'][local_index],
                    'num_mixed_traces': analysis['num_mixed_traces'][local_index],
                    'exact_sat_volume': analysis['exact_sat_volume'][local_index],
                    'exact_satisfaction_fraction': analysis['exact_sat_fraction'][local_index],
                    'sampling_sat_volume': analysis['sampling_sat_volume'][local_index],
                    'sampling_satisfaction_fraction': analysis['sampling_sat_fraction'][local_index],
                    'sat_fraction': selected_fraction,
                    'reach_time': analysis['reachTime'],
                    'branch_volume_time': analysis['branchVolumeTime'],
                    'checking_time': analysis['checkingTime'][local_index],
                    'verification_time': analysis['verifyTime'][local_index],
                    'trace_generation_capped': analysis['trace_generation_capped'],
                    'full_analysis': analysis,
                }
                rows.append(row)
                initset_rows.append(row)
            grouped_rows[(initSet_id, numSteps)] = [
                row for row in initset_rows if row['numSteps'] == numSteps
            ]
        grouped_rows[initSet_id] = initset_rows
        print_AEBS_StarTL_table(initset_rows, method=method)

    return {
        'system': 'AEBS',
        'method': method,
        'compute_region_balls': compute_region_balls,
        'compute_exact_robustness': compute_exact_robustness,
        'initSet_ids': initSet_ids,
        'spec_ids': spec_ids,
        'numSteps_list': numSteps_list,
        'num_sampling_samples_per_trace': num_samples,
        'rows': rows,
        'grouped_rows': grouped_rows,
    }


if __name__ == '__main__':
    initSet_ids = [0, 1, 2, 3]
    numSteps_list = [10,20,30]
    spec_ids = None
    num_samples = 10000
    seed = 0
    burn_in = None
    thinning = 1
    lp_solver = 'linprog'
    method = 'exact'  # choose from: 'exact', 'sampling'/'sample', 'both'
    numCores = 1
    max_traces = None
    volume_seed = 0
    compute_region_balls = False
    compute_exact_robustness = True

    verify_AEBS_StarTL_table(
        initSet_ids=initSet_ids,
        numSteps_list=numSteps_list,
        spec_ids=spec_ids,
        num_samples=num_samples,
        seed=seed,
        burn_in=burn_in,
        thinning=thinning,
        lp_solver=lp_solver,
        method=method,
        numCores=numCores,
        max_traces=max_traces,
        volume_seed=volume_seed,
        compute_exact_robustness=compute_exact_robustness,
        compute_region_balls=compute_region_balls
    )
