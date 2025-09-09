#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
##                                                                     ##
##   Primary contacts: Hoang Dung Tran <dungtran@ufl.edu> (UF)         ##
##                     Sung Woo Choi <sungwoo.choi@ufl.edu> (UF)       ##
##                     Yuntao Li <yli17@ufl.edu> (UF)                  ##
##                     Qing Liu <qliu1@ufl.edu> (UF)                   ##
##                                                                     ##
##   See CONTRIBUTORS for full author contacts and affiliations.       ##
##   This program is licensed under the BSD 3‑Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
Fairness Verifier
Yuntao Li
"""

from StarV.net.network import NeuralNetwork, reach_exact_bfs_star_relu
import copy
import multiprocessing
import numpy as np
import polytope as pc
from StarV.util.print_util import print_util
from typing import List, Union, Tuple, Optional
from StarV.set.star import Star
from itertools import repeat
import time
from contextlib import contextmanager
from typing import Dict
from StarV.util import plot

class Timer:
    """Simple timer class that tracks total execution time"""
    def __init__(self, name: str):
        self.name = name
        self.total_time = 0
        self._start_time = None
        
    def __enter__(self):
        self._start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        self.total_time = time.perf_counter() - self._start_time

class TiminGCanager:
    """Manager class for handling multiple timers"""
    def __init__(self):
        self.timers: Dict[str, Timer] = {}
        
    @contextmanager
    def timer(self, name: str):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        with self.timers[name] as t:
            yield t
            
    def get_time(self, name: str) -> float:
        return self.timers[name].total_time if name in self.timers else 0
    
    def get_all_times(self) -> Dict[str, float]:
        return {name: timer.total_time for name, timer in self.timers.items()}


class Verifier(object):
    """
       Verifier Class

       Properties: (Verification Settings)

        @lp_solver: lp solver: 'gurobi' (default), 'glpk', 'linprog'
        @method: verification method: BFS "bread-first-search" or DFS "depth-first-search"
        @n_processes: number of processes used for verification
        @time_out: time_out for a single verification querry (single input)

      Methods:
        @verify: main verification method
        @evaluate: evaluate method on a specific input 

    """

    def __init__(self, lp_solver='gurobi', method='BFS', n_processes=1, time_out=None):
        self.lp_solver = lp_solver
        self.method = method
        self.n_processes = n_processes
        self.time_out = time_out

    def verify(self, net, inputSet):
        """main verification method"""

        assert isinstance(net, NeuralNetwork), 'error: input is not a NeuralNetwork object'
        pass



def fairness_specification() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos_spec_mat = np.array([[-1.0]])
    pos_spec_vec = np.array([0])

    neg_spec_mat = np.array([[1.0]])
    neg_spec_vec = np.array([0])

    return pos_spec_mat, pos_spec_vec, neg_spec_mat, neg_spec_vec


def parse_inputs_strict(model):

    if model == 'AC':
        # AC: age, workclass, education, education-num, marital-status, 
        # occupation, relationship, (race), (sex), capital-gain, 
        # capital-loss, hours-per-week, native-country
        lb_AC = np.array([10, 0, 0, 1, 0, 
                        0, 0, 0, 0, 0, 
                        0, 1, 0])
        ub_AC = np.array([100, 6, 15, 16, 6, 
                        13, 5, 4, 0, 19, 
                        19, 100, 40])
        # ub_AC = np.array([100, 6, 15, 16, 6, 
        #                 13, 5, 4, 1, 19, 
        #                 19, 100, 40])
        # AC_ids = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12]

        return lb_AC, ub_AC

    elif model == 'BM':
        # BM: job, marital, education, default, housing,
        # loan, contact, month, day_of_week, emp.var.rate, 
        # duration, campaign, pdays, previous, poutcome
        # (age)
        lb_BM = np.array([0, 0, 0, 0, 0, 
                        0, 0, 0, 0, -3, 
                        0, 1, 0, 0, 0, 
                        0])
        ub_BM = np.array([10, 2, 6, 1, 1, 
                        1, 1, 11, 6, 1, 
                        5000, 50, 999, 7, 2, 
                        0])
        # ub_BM = np.array([10, 2, 6, 1, 1, 
        #                 1, 1, 11, 6, 1, 
        #                 5000, 50, 999, 7, 2, 
        #                 1])
        # BM_ids = [1, 2, 3, 4, 5, 6, 7, 8]

        return lb_BM, ub_BM


    elif model == 'GC':
        # GC: status, month, credit_history, purpose, credit_amount, savings, employment, investment_as_income_percentage,
        # savings, employment, investment_as_income_percentage, other_debtors, residence_since, 
        # property, (age), installment_plans, housing, number_of_credits, 
        # skill_level, people_liable_for, telephone, foreign_worker, (sex)
        lb_GC = np.array([0, 0, 0, 0, 0, 
                        0, 0, 1, 0, 1, 
                        0, 0, 0, 0, 1, 
                        0, 1, 0, 0, 0])
        ub_GC = np.array([2, 80, 2, 9, 20000, 
                        2, 2, 4, 2, 4, 
                        2, 0, 2, 2, 4, 
                        3, 2, 1, 1, 1])
        # ub_GC = np.array([2, 80, 2, 9, 20000, 
        #                 2, 2, 4, 2, 4, 
        #                 2, 1, 2, 2, 4, 
        #                 3, 2, 1, 1, 1])
        # GC_ids = [1, 2, 3, 4, 5]

        return lb_GC, ub_GC
    

def parse_protected_attr(model, prot_attr_val, inputSet):
    # AC: 8, BM: 15, GC: 19
    new_inputSet = inputSet.clone()
    if model == 'AC':
        # AC: age, workclass, education, education-num, marital-status, 
        # occupation
        prot_attr_index = 8
        new_inputSet.V[prot_attr_index, 0] = prot_attr_val
    elif model == 'BM':
        # BM: age
        prot_attr_index = 15
        new_inputSet.V[prot_attr_index, 0] = prot_attr_val
    elif model == 'GC':
        # GC
        # prot_attr_index = 19
        prot_attr_index = 11
        new_inputSet.V[prot_attr_index, 0] = prot_attr_val
    
    return new_inputSet


def evaluate(*args) -> np.ndarray:
    """Evaluate the network on a set of samples"""
    args1 = args[0] if isinstance(args[0], tuple) else args
    net, samples = args1

    if not isinstance(net, NeuralNetwork):
        raise ValueError('net should be a NeuralNetwork object')

    x = samples
    for layer in net.layers:
        x = layer.evaluate(x)
    return x


def checkSafetyStar(unsafe_mat: np.ndarray, unsafe_vec: np.ndarray, S: Star) -> Union[Star, List]:
    """
    Intersect Star with unsafe region.

    Args:
        unsafe_mat (np.ndarray): Constraint matrix.
        unsafe_vec (np.ndarray): Constraint vector.
        S (Star): Star object to check.

    Returns:
        Union[Star, List]: Intersected Star or empty list if no intersection.

    Raises:
        ValueError: If inputs are not of correct type or shape.
    """
    if not isinstance(unsafe_mat, np.ndarray) or not isinstance(unsafe_vec, np.ndarray):
        raise ValueError('Constraint matrix and vector should be numpy arrays')
    if unsafe_vec.ndim != 1 or unsafe_mat.shape[0] != unsafe_vec.shape[0]:
        raise ValueError('Inconsistency between constraint matrix and vector')

    P = S.clone()
    # v = np.matmul(unsafe_mat, P.V)
    v = unsafe_mat @ P.V
    newC = v[:, 1:P.nVars+1]
    newd = unsafe_vec - v[:, 0]

    if len(P.C) != 0:
        P.C = np.vstack((newC, P.C))
        P.d = np.concatenate([newd, P.d])
    else:
        P.C = newC.reshape(1, P.nVars) if newC.ndim == 1 else newC
        P.d = newd

    return P if not P.isEmptySet() else []


def fairness_verification_exact_strict_both(net, inputSet1, inputSet2, lp_solver='gurobi', numCores=1, show=True):
    """Run fairness verification in both directions efficiently."""
    timing_manager = TiminGCanager()
    
    with timing_manager.timer('total_execution'):
        pool = multiprocessing.Pool(numCores) if numCores > 1 else None
        # Compute reachable sets once
        with timing_manager.timer('reach_S1'):
            S1 = reach_exact_bfs_star_relu(net, [inputSet1], 'exact', lp_solver, pool, show)
        
        with timing_manager.timer('reach_S2'):
            S2 = reach_exact_bfs_star_relu(net, [inputSet2], 'exact', lp_solver, pool, show)
            
        # Run verifications in both directions using the same reachable sets
        pos_results = fairness_verification_exact_strict_single(S1, S2, inputSet1, True, pool)
        neg_results = fairness_verification_exact_strict_single(S1, S2, inputSet1, False, pool)

        if pool is not None:
            pool.close()
        
    timing = {
        'total_time': timing_manager.get_time('total_execution'),
        'reach_S1_time': timing_manager.get_time('reach_S1'),
        'reach_S2_time': timing_manager.get_time('reach_S2'),
        'pos_first': pos_results['timing'],
        'neg_first': neg_results['timing']
    }
    print(timing)
    
    return {
        'pos_first': pos_results['results'],
        'neg_first': neg_results['results'],
        'timing': timing
    }


def fairness_verification_exact_strict_single(S1, S2, inputSet1, pos_first=True, pool=None):
    """Single direction verification using starmap with zip for efficient parallel processing."""
    timing_manager = TiminGCanager()
    
    with timing_manager.timer('total'):
        pos_spec_mat, pos_spec_vec, neg_spec_mat, neg_spec_vec = fairness_specification()
        
        P1, P2 = [], []
        C1 = []
        
        # Select specifications
        spec1_mat = pos_spec_mat if pos_first else neg_spec_mat
        spec1_vec = pos_spec_vec if pos_first else neg_spec_vec
        spec2_mat = neg_spec_mat if pos_first else pos_spec_mat
        spec2_vec = neg_spec_vec if pos_first else pos_spec_vec

        with timing_manager.timer('verification'):
            if pool is None:
                for S in S1:
                    P = checkSafetyStar(spec1_mat, spec1_vec, S)
                    if isinstance(P, Star):
                        P1.append(P)
                for S in S2:
                    P = checkSafetyStar(spec2_mat, spec2_vec, S)
                    if isinstance(P, Star):
                        P2.append(P)

                for p1 in P1:
                    for p2 in P2:
                        c1 = Star(inputSet1.V, np.vstack([p1.C, p2.C]), 
                                np.hstack([p1.d, p2.d]), 
                                inputSet1.pred_lb, inputSet1.pred_ub)
                        if not c1.isEmptySet():
                            C1.append(c1)
            else:
                results_S1 = pool.starmap(checkSafetyStar, 
                    zip(repeat(spec1_mat), repeat(spec1_vec), S1))
                for p in results_S1:
                    if isinstance(p, Star):
                        P1.append(p)

                results_S2 = pool.starmap(checkSafetyStar, 
                    zip(repeat(spec2_mat), repeat(spec2_vec), S2))
                for p in results_S2:
                    if isinstance(p, Star):
                        P2.append(p)

                if P1 and P2:
                    batch_size = min(1000, len(P2))
                    for i in range(0, len(P2), batch_size):
                        batch_P2 = P2[i:i + batch_size]
                        for p2 in batch_P2:
                            results = pool.starmap(check_counter_example_exact,
                                zip(P1, 
                                    repeat(p2, len(P1)), 
                                    repeat(inputSet1, len(P1))))
                            C1.extend([r for r in results if r is not None])    
    return {
        'results': {'S1': S1, 'S2': S2, 'P1': P1, 'P2': P2, 'C1': C1},
        'timing': timing_manager.get_all_times()
    }


# Helper function for counter example checking
def check_counter_example_exact(p1, p2, inputSet):
    """Check for counter examples with explicit arguments."""
    c1 = Star(inputSet.V, np.vstack([p1.C, p2.C]), np.hstack([p1.d, p2.d]), 
              inputSet.pred_lb, inputSet.pred_ub)
    return c1 if not c1.isEmptySet() else None