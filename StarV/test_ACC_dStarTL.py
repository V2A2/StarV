"""
ACC StarTL verification example.

This script mirrors the ACC case study in
artifacts/HSCC2025_ProbStarTL/HSCC2025_ProbStarTL.py, but uses:

    1. Star initial sets instead of ProbStar initial sets.
    2. Star reachability through the NNCS stepReach_DLNNCS routine.
    3. dStarTL robustness intervals and sampling-based satisfaction fractions.

"""

import copy
import os
import sys
import time
import numpy as np
from StarV.nncs.nncs import NNCS, ReachPRM_NNCS, VerifyPRM_NNCS, stepReach_DLNNCS
from StarV.set.star import Star
from StarV.util.load import load_acc_model
from StarV.spec.dStarTL import (
        ExpandedFormula,
        GetSatisfactionFraction,
    )


def construct_acc_initial_star(initSet_id=5):
    """Construct the ACC initial set as a Star using the bounds in load_acc_model."""
    x_lead_0 = [90., 92.]
    v_lead_0 = [
        [29., 30.],
        [28., 29.],
        [27., 28.],
        [26., 27.],
        [25., 26.],
        [20., 21.],
    ]
    acc_lead_0 = [0., 0.]
    x_ego_0 = [30., 31.]
    v_ego_0 = [30., 30.5]
    acc_ego_0 = [0., 0.]
    a_lead = -5.0
    x7_0 = [2 * a_lead, 2 * a_lead]

    if initSet_id < 0 or initSet_id >= len(v_lead_0):
        raise RuntimeError('initSet_id should be between 0 and 5')

    lb = np.array([
        x_lead_0[0],
        v_lead_0[initSet_id][0],
        acc_lead_0[0],
        x_ego_0[0],
        v_ego_0[0],
        acc_ego_0[0],
        x7_0[0],
    ])
    ub = np.array([
        x_lead_0[1],
        v_lead_0[initSet_id][1],
        acc_lead_0[1],
        x_ego_0[1],
        v_ego_0[1],
        acc_ego_0[1],
        x7_0[1],
    ])
    return Star(lb, ub)


def reach_star_traces_acc_nncs(ncs, reachPRM, max_traces=None):
    """Generate coherent Star traces using the ACC NNCS stepReach_DLNNCS."""
    if not isinstance(ncs, NNCS):
        raise RuntimeError('ncs should be an NNCS object')
    if not isinstance(reachPRM, ReachPRM_NNCS):
        raise RuntimeError('reachPRM should be a ReachPRM_NNCS object')
    if reachPRM.initSet is None:
        raise RuntimeError('reachPRM.initSet is required')
    if reachPRM.numSteps < 1:
        raise RuntimeError('numSteps should be >= 1')

    remains = [[copy.deepcopy(reachPRM.initSet)]]
    partial_trace = []
    traces = []
    capped = False

    while True:
        depth = len(remains)
        if depth == 0:
            break

        current_level = remains[depth - 1]
        if len(current_level) == 0:
            remains.pop(depth - 1)
            if len(partial_trace) != 0:
                partial_trace.pop(len(partial_trace) - 1)
            continue

        current_set = current_level.pop(0)
        partial_trace.append(current_set)
        next_sets, _ = stepReach_DLNNCS(
            ncs.ncs,
            current_set,
            reachPRM
        )

        if depth == reachPRM.numSteps:
            for next_set in next_sets:
                trace = copy.deepcopy(partial_trace)
                trace.append(next_set)
                traces.append(trace)
                if max_traces is not None and len(traces) >= max_traces:
                    capped = True
                    return traces, capped
            partial_trace.pop(len(partial_trace) - 1)
        else:
            remains.append(next_sets)

    return traces,capped


def evaluate_trace_StarTL(
        trace, spec, trace_id=0, num_samples=1000, seed=0,
        burn_in=None, thinning=1, lp_solver='linprog'):
    """Evaluate one coherent Star trace with StarTL robustness and sampling."""

    expanded_spec = ExpandedFormula(spec, T=len(trace))
    satisfaction_checker = GetSatisfactionFraction(
        trace,
        expanded_spec,
        method='sampling',
        num_samples=num_samples,
        seed=None if seed is None else seed + trace_id,
        burn_in=burn_in,
        thinning=thinning,
        lp_solver=lp_solver
    )

    result = satisfaction_checker.getSatisfactionFraction()
    branch_base_set = trace[-1]
    base_A, base_b = satisfaction_checker.getBasePredicateConstraints(branch_base_set)
    branch_volume = satisfaction_checker.computeBaseVolume(base_A, base_b)
    satisfying_volume = branch_volume * result['satisfying_fraction']

    return {
        'trace_id': trace_id,
        'num_time_sets': len(trace),
        'rho_lb': result['rho_lb'],
        'rho_ub': result['rho_ub'],
        'method': result['method'],
        'satisfying_fraction': result['satisfying_fraction'],
        'branch_volume': branch_volume,
        'satisfying_volume': satisfying_volume,
        'num_samples': result.get('num_samples', num_samples),
        'n_satisfied': result.get('n_satisfied', None),
    }


def verify_StarTL_specs_DLNNCS_ACC():
    pass


if __name__ == '__main__':
    pass

