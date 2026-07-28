'''

Star Temporal Logic Specification Language in discrete-time domain

Author: Qing Liu
Date: 6/4/2026

==================================================================================
DESCRIPTION:
-----------
* Star Set Temporal Logic (StarTL) is adaptive from ProbStarTL, which is a specification language for quantitative monitoring and verification of temporal behaviors in autonomous and learning-enabled cyber-physical systems.

* StarTL has similar syntax as STL, including Boolean and temporal operators

* StarTL is difference from ProbStarTL, which compute satisfaction based on probability.

* StarTL has quantitative semantics that allows answering the quantify a system satisfying a property.

* Given a temporal specification \phi, StarTL can answer:
    whether the reachable set sequence satisfy \phi;
    whether the reachable set sequence violate \phi;
    how robustly the specification is satisfied or violated (robustness interval:[\rho_{\phi}_lb, \rho_{\phi}_ub]); 
    how much (what fraction) of predicate-space reachable sets satisfy (satisfaction fraction \phi (\q_{\phi})).

==================================================================================

StarTL SYNTAX and BOOLEAN SEMANTICS are same as dStarTLobStarTL, which is defined as follows:
------------------

* Atomic Predicate (AP): a single linear constraint of the form: 

             AP: Ax <= b, A in R^{1 x n}, x in R^n, b in R^n


* Operators:

   * logic operators: NOT, AND, OR

   * temporal operators: NEXT (NE), ALWAYS (AW), EVENTUALLY (ET), UNTIL (UT)


* Formulas: p:= T | AP | NOT p | p AND w | p U_[a,b] w

    * Eventually: ET_[a,b] p = T U_[a,b] p

    * Always: AW_[a,b] p = NOT (ET_[a,b] NOT p)

=================================================================================

StarTL BOOLEAN SEMANTICS
----------------------------

Defined on BOUNDED TIME REACHABLE SET X = [X[1], X[2], ..... X[T]]

The satisfaction (|=) of a formula p by a reachable set X at time step 1 <= t <= T


* (X, t) |= AP <=> exist x in X[t], Ax <= b <=> X[t] AND AP is feasible

* (X, t) |= p AND AP <=> X[t] AND p AND AP is feasible (different from STL semantics)

* (X, t) |= NOT p <=> X[t] AND NOT p is feasiable  

* (X, t) |= p U_[a, b] w <=> exist t' in [t + a, t + b] such that (X, t') |= w AND for all t'' in [t, t'), (X, t'') |= p

* Eventually: ET_[a, b] p = T U_[a, b] p

  (X, t) |= ET_[a, b] w <=> exist t' in [t + a, t + b] such that (x, t') |= w

* Always: AW_[a, b] p = NOT (ET_[a, b] NOT p)

  (X, t) |= AW_[a, b] <=> for all t' in [t + a, t + b] such that (X, t') |= w

==================================================================================

'''

import numpy as np
from itertools import combinations
import polytope as pc
from scipy.optimize import linprog
from StarV.set.star import Star
from StarV.spec.dProbStarTL import (
    _UNTIL_,
    AtomicPredicate,
    Formula,
    _AND_,
    _OR_,
    _NOT_,
    _NEXT_,
    _ALWAYS_,
    _EVENTUALLY_,
    _LeftBracket_,
    _RightBracket_,
)



class ExpandedFormula(object):
    """
    Expand a temporal logic formula into a time-indexed expression tree.

    The expanded tree can be evaluated on a reachable set sequence. The
    outermost temporal operator determines how repeated time clauses are joined:
    ALWAYS uses AND, and EVENTUALLY uses OR.

    Example:
        AW_[0, 1] (P1 AND (ET_[0, 1] P2))

    Expanded expression formula:
        (P1[t=0] AND (P2[t=0] OR P2[t=1]))
        AND
        (P1[t=1] AND (P2[t=1] OR P2[t=2]))
    """

    def __init__(self, formula, T=None):
        """
        Args:
            formula: Formula object or formula token list.
            T: reachable sequence length
        """

        if isinstance(formula, Formula):
            self.original_formula = formula # original formula object, useful when convert to DNF
            self.formula_tokens = formula.formula # stores only the raw token list inside the formula, like [EVOT, lb, P1, OR, P2, rb], used to get expanded formula tree (self.expr)

        elif isinstance(formula, list):
            self.original_formula = Formula(formula)
            self.formula_tokens = formula
        else:
            raise RuntimeError('input should be a Formula object or list')
        if T is not None and (not isinstance(T, int) or T < 1):
            raise RuntimeError('T should be a positive integer')

        self.formula = self.formula_tokens
        self.T = T
        self.outer_operator = self.get_outer_operator()
        self.expr = self.getExpandedFormula(self.formula_tokens, 0, len(self.formula_tokens), 0)
        # expr is a nested tuple representation of the temporal logic formula after time expansion. Each nodehas this form:(op, subformulas)

        if self.expr is None:
            raise RuntimeError('expanded formula is empty within the given T')
        self.F = self.expr

    def print(self):
        print(self)

    def __str__(self):
        return self.format_expression(self.expr)

    def print_format(self, expr, parent_op=None):
        return self.format_expression(expr, parent_op)

    def format_expression(self, expr, parent_op=None):
        """Return a readable string for an expanded expression tree."""
        op = expr[0]
        if op == 'AP':
            predicate = expr[1]
            if isinstance(predicate, AtomicPredicate):
                return '{} * x[t={}] <= {}'.format(predicate.A, predicate.t, predicate.b)
            return str(predicate)
        if op == 'NOT':
            sub_expr = expr[1][0]
            sub_text = self.format_expression(sub_expr, parent_op=op)
            if sub_expr[0] != 'AP':
                sub_text = '({})'.format(sub_text)
            return 'NOT {}'.format(sub_text)

        sub_exprs = expr[1]
        separator = ' {} '.format(op)
        is_outermost = parent_op is None and op == self.outer_operator
        if is_outermost:
            separator = '\n{}\n'.format(op)

        formatted_text = []
        for sub_expr in sub_exprs:
            sub_text = self.format_expression(sub_expr, parent_op=op)
            if is_outermost and sub_expr[0] != 'AP':
                if not (sub_text.startswith('(') and sub_text.endswith(')')):
                    sub_text = '({})'.format(sub_text)
            formatted_text.append(sub_text)
        text = separator.join(formatted_text)
        if parent_op == 'AND' and op == 'OR':
            return '({})'.format(text)
        if parent_op == 'OR' and op == 'AND':
            return '({})'.format(text)
        return text

    def get_outer_operator(self):
        if len(self.formula_tokens) < 2:
            return None
        if not isinstance(self.formula_tokens[1], _LeftBracket_):
            return None
        if self.match_right_loop_id(self.formula_tokens, 1) != len(self.formula_tokens) - 1:
            return None
        if isinstance(self.formula_tokens[0], _EVENTUALLY_):
            return 'OR'
        if isinstance(self.formula_tokens[0], _ALWAYS_):
            return 'AND'
        return None


    def getExpandedFormula(self, tokens, start, end, time_offset=0):
        '''
            Read formula tokens from left to right and build a nested expression tree.
            The returned tree preserves brackets and temporal structure.

            Example:
            ('AND', [
            ('AP', P1[t=0]),
            ('OR', [
                ('AP', P2[t=0]),
                ('AP', P2[t=1])
            ])])

            Each tree node is either ('AP', predicate) or
            (op, [sub_expr1, sub_expr2, ...]), where op is 'NOT',
            'AND', or 'OR' and chid_expr is a nested expression tree.
        '''
        expr_terms = []
        bool_ops = []
        token_ids = start
        while token_ids < end:
            token = tokens[token_ids]
            if isinstance(token, AtomicPredicate):
                predicate_time = 0 if token.t is None else token.t
                shifted_time = predicate_time + time_offset
                if self.isvalid_time(shifted_time):
                    expr_terms.append(('AP', token.at_time(shifted_time)))
                token_ids += 1
            elif isinstance(token, _NOT_):
                next_index = token_ids + 1
                if next_index >= end:
                    raise RuntimeError('NOT must be followed by a subformula')
                sub_tokens, token_ids = self.expand_subformula(tokens, next_index, end)
                sub_expr = self.getExpandedFormula(sub_tokens, 0, len(sub_tokens), time_offset)
                if sub_expr is not None:
                    expr_terms.append(('NOT', [sub_expr]))
            elif isinstance(token, _NEXT_):
                next_index = token_ids + 1
                if next_index >= end:
                    raise RuntimeError('NEXT must be followed by a subformula')
                sub_tokens, token_ids = self.expand_subformula(tokens, next_index, end)
                sub_expr = self.getExpandedFormula(
                    sub_tokens, 0, len(sub_tokens), time_offset + 1
                )
                if sub_expr is not None:
                    expr_terms.append(sub_expr)
            elif isinstance(token, _ALWAYS_) or isinstance(token, _EVENTUALLY_):
                next_index = token_ids + 1
                if next_index >= end:
                    raise RuntimeError('temporal operator must be followed by a subformula')
                sub_tokens, token_ids = self.expand_subformula(tokens, next_index, end)

                expanded_terms = [
                    sub_expr
                    for dt in self.get_time_range(token)
                    for sub_expr in [
                        self.getExpandedFormula(
                            sub_tokens, 0, len(sub_tokens), time_offset + dt
                        )
                    ]
                    if sub_expr is not None
                ]
                if len(expanded_terms) == 0:
                    continue
                op = 'AND' if isinstance(token, _ALWAYS_) else 'OR'
                expr_terms.append((op, expanded_terms))
            elif isinstance(token, _UNTIL_):
                if len(expr_terms) == 0:
                    raise RuntimeError('UNTIL must have a left subformula')

                next_index = token_ids + 1
                if next_index >= end:
                    raise RuntimeError('UNTIL must be followed by a right subformula')

                left_expr = expr_terms.pop()
                right_tokens, token_ids = self.expand_subformula(tokens, next_index, end)
                expr_terms.append(
                    self.UNTIL_expand(left_expr, right_tokens, token, time_offset)
                )
            elif isinstance(token, _LeftBracket_):
                right_bracket_index = self.match_right_loop_id(tokens, token_ids)
                sub_tokens = Formula(tokens).getSubFormula(token_ids + 1, right_bracket_index)
                sub_expr = self.getExpandedFormula(sub_tokens, 0, len(sub_tokens), time_offset)
                if sub_expr is not None:
                    expr_terms.append(sub_expr)
                token_ids = right_bracket_index + 1
            elif isinstance(token, _AND_):
                bool_ops.append('AND')
                token_ids += 1
            elif isinstance(token, _OR_):
                bool_ops.append('OR')
                token_ids += 1
            elif isinstance(token, _RightBracket_):
                token_ids += 1
            else:
                raise RuntimeError('unsupported item in formula: {}'.format(type(token)))

        if len(expr_terms) == 0:
            return None
        if len(bool_ops) != len(expr_terms) - 1 and len(set(bool_ops)) > 1:
            raise RuntimeError('invalid subformula segment: operators and terms do not match')
        if len(bool_ops) == 0 or len(expr_terms) == 1:
            return expr_terms[0]

        if len(set(bool_ops)) > 1:
            raise RuntimeError(
                'mixed AND/OR operations must be bracketed, e.g., '
                '(P1 OR P2) AND P3 or P1 OR (P2 AND P3)'
            )

        op = bool_ops[0]
        if op not in ('AND', 'OR'):
            raise RuntimeError('unknown operator {}'.format(op))
        return (op, expr_terms)

    def expand_subformula(self, tokens, start_index, end):
        """Return the single term or bracketed subformula that starts at start_index."""
        if isinstance(tokens[start_index], _LeftBracket_):
            right_bracket_index = self.match_right_loop_id(tokens, start_index)
            if right_bracket_index >= end:
                raise RuntimeError('right bracket is outside the current formula segment')
            return Formula(tokens).getSubFormula(start_index + 1, right_bracket_index), right_bracket_index + 1
        return Formula(tokens).getSubFormula(start_index, start_index + 1), start_index + 1

    def UNTIL_expand(self, left_expr, right_tokens, until_op, time_offset):
        """Expand p1 U_[a,b] p2 into an OR over all possible witness times."""

        assert isinstance(until_op, _UNTIL_), 'error: input should be an UNTIL operator'
        assert isinstance(right_tokens, list), 'error: right_tokens should be a list'
        assert until_op.start_time >= 0, 'error: t_start should be >= 0'

        until_terms = []
        for witness_time in self.get_time_range(until_op):
            left_terms = [
                self.shift_time(left_expr, dt)
                for dt in range(0, witness_time)
            ]
            left_terms = [term for term in left_terms if term is not None]
            right_terms = self.getExpandedFormula(
                right_tokens, 0, len(right_tokens), time_offset + witness_time
            )
            if right_terms is None:
                continue
            until_terms.append(('AND', left_terms + [right_terms]))
        if len(until_terms) == 0:
            return None
        return ('OR', until_terms)

    @staticmethod
    def get_time_range(temporal_operator):
        end_time = getattr(temporal_operator, 'end_time', None)
        if end_time is None or end_time == float('inf'):
            raise RuntimeError('only bounded temporal intervals can be expanded')
        return range(temporal_operator.start_time, end_time + 1)

    def shift_time(self, expr, time_offset):
        """Return a copy of expr with every atomic predicate shifted in time."""
        op = expr[0]
        if op == 'AP':
            predicate = expr[1]
            shifted_time = predicate.t + time_offset
            if not self.isvalid_time(shifted_time):
                return None
            return ('AP', predicate.at_time(shifted_time))

        shifted_subformula = [
            self.shift_time(sub_expr, time_offset)
            for sub_expr in expr[1]
        ]
        shifted_subformula = [
            sub_expr for sub_expr in shifted_subformula
            if sub_expr is not None
        ]
        if len(shifted_subformula) == 0:
            return None
        return (op, shifted_subformula)

    def isvalid_time(self, time_idx):
        """Return False when time_idx is outside the reachable sequence."""
        if time_idx < 0:
            return False
        return self.T is None or time_idx < self.T

    def match_right_loop_id(self, tokens, left_index):
        if left_index >= len(tokens) or not isinstance(tokens[left_index], _LeftBracket_):
            raise RuntimeError('left_index must point to a left bracket')

        lb_idxes, rb_idxes = Formula(tokens).getLoopIds()
        if left_index not in lb_idxes:
            raise RuntimeError('left bracket index not found in formula loop ids')

        bracket_ids = sorted(lb_idxes + rb_idxes)
        left_ids = set(lb_idxes)
        depth = 0
        for bracket_id in bracket_ids:
            if bracket_id < left_index:
                continue
            if bracket_id in left_ids:
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    return bracket_id
        raise RuntimeError('unbalanced brackets')

class GetRobustnessInterval(object):
    """Compute robustness intervals for expanded StarTL formulas."""

    def __init__(self, R, lp_solver='linprog'):
        if not isinstance(R, (list, tuple)):
            raise RuntimeError('R should be a reachable set sequence stored as a list or tuple')
        self.R = R
        self.lp_solver = lp_solver

    def getRobustnessInterval(self, expanded_formula):
        '''
        Compute the robustness interval of a temporal formula given a reachable set sequence.
        \rho_{\phi}_lb is the lower bound of the robustness interval, which is the minimum robustness value within predicate-space range.
        \rho_{\phi}_ub is the upper bound of the robustness interval, which is the maximum robustness value within predicate-space range.

        Args:
            expanded_formula: ExpandedFormula object or expanded expression tuple.

        Returns:
            (rho_lb, rho_ub): robustness interval of the expanded formula.
        '''
        if isinstance(expanded_formula, ExpandedFormula):
            expr = expanded_formula.expr
        elif isinstance(expanded_formula, tuple):
            expr = expanded_formula
        else:
            raise RuntimeError('expanded_formula should be an ExpandedFormula object or expression tuple')

        return self.getExpandedRobustnessInterval(expr)

    def getExpandedRobustnessInterval(self, expr):
        """Recursively compute the robustness interval of an expanded expression."""
        if isinstance(expr, ExpandedFormula):
            expr = expr.expr
        if not isinstance(expr, tuple) or len(expr) != 2:
            raise RuntimeError('invalid expanded expression: {}'.format(expr))

        op, children_or_predicate = expr
        if op == 'AP':
            return self.getAtomicRobustnessInterval(children_or_predicate)

        child_exprs = children_or_predicate
        if not isinstance(child_exprs, list) or len(child_exprs) == 0:
            raise RuntimeError('{} operator should contain a nonempty list of subformulas'.format(op))

        child_intervals = []
        for child_expr in child_exprs:
            child_interval = self.getExpandedRobustnessInterval(child_expr)
            child_intervals.append(child_interval)

        if op == 'NOT':
            if len(child_intervals) != 1:
                raise RuntimeError('NOT operator should have exactly one subformula')
            rho_lb, rho_ub = child_intervals[0]
            return -rho_ub, -rho_lb

        if op == 'AND':
            # rho(phi1 AND ... AND phin) = min rho(phi,...,phin)
            return (
                min(rho_lb for rho_lb, _ in child_intervals),
                min(rho_ub for _, rho_ub in child_intervals)
            )

        if op == 'OR':
            # rho(phi1 OR ... OR phin) = max rho(phi,...,phin)
            return (
                max(rho_lb for rho_lb, _ in child_intervals),
                max(rho_ub for _, rho_ub in child_intervals)
            )

        raise RuntimeError('unsupported expanded formula operator: {}'.format(op))

    def getAtomicRobustnessInterval(self, predicate):
        """Compute min/max of atomic robustness b - A*x[t] over reachable set R[t]."""
        if not isinstance(predicate, AtomicPredicate):
            raise RuntimeError('Missing an AtomicPredicate')

        t = 0 if predicate.t is None else predicate.t
        if t < 0 or t >= len(self.R):
            raise RuntimeError('invalid time t={}'.format(t))

        reachable_sets = self.R[t]
        if isinstance(reachable_sets, tuple):
            reachable_sets = list(reachable_sets)
        if not isinstance(reachable_sets, list):
            reachable_sets = [reachable_sets]
        if len(reachable_sets) == 0:
            raise RuntimeError('reachable set at time {} is empty'.format(t))

        lower_bounds = []
        upper_bounds = []
        robustness_map = -predicate.A.reshape(1, predicate.A.shape[0])
        robustness_offset = predicate.b
        for reachable_set in reachable_sets:
            # Predicate A*x <= b has robustness rho = b - A*x. If R[t] has
            # several Star sets, take the min/max over their union.
            robustness_set = reachable_set.affineMap(robustness_map, robustness_offset)
            lower_bounds.append(robustness_set.getMin(0, self.lp_solver))
            upper_bounds.append(robustness_set.getMax(0, self.lp_solver))

        return min(lower_bounds), max(upper_bounds)

class GetSatisfactionFraction(object):
    def __init__(self, R, expanded_formula, num_samples=10000, lp_solver='linprog', error=1e-10, method=None, seed=None, burn_in=None, thinning=1):
        if not isinstance(R,list):
            raise RuntimeError('R should be a reachable set sequence as a list')
        if not isinstance(num_samples, int) or num_samples < 1:
            raise RuntimeError('num_samples should be a positive integer')
        if not isinstance(thinning, int) or thinning < 1:
            raise RuntimeError('thinning should be a positive integer')
        if method not in (None, 'exact', 'sampling'):
            raise RuntimeError(" Unkown satisfaction fraction compuation method")
        if not isinstance(expanded_formula, ExpandedFormula):
            raise RuntimeError('expanded_formula should be an ExpandedFormula object')

        self.R = R
        self.expanded_formula = expanded_formula
        self.num_samples = num_samples
        self.lp_solver = lp_solver
        self.error = error
        self.method = method
        self.seed = seed
        self.burn_in = burn_in
        self.thinning = thinning

    def getSatisfactionFraction(self):
        """Compute the fraction of the feasible predicate space satisfying a TL formula."""
        expr = self.expanded_formula.expr # exapnded expression tree of the temporal logic formula

        robustness_interval = GetRobustnessInterval(self.R, self.lp_solver)
        rho_lb, rho_ub = robustness_interval.getRobustnessInterval(expr)
        result = {
            'method': None,
            'rho_lb': rho_lb,
            'rho_ub': rho_ub,
            'satisfying_fraction': None,
        }
        print("Robustness reslts: lb ={}, ub ={}".format(rho_lb,rho_ub))

        # Since the atomic predicate is A*x <= b, rho == 0 is satisfying.
        if rho_lb >= 0.0:
            result['method'] = 'robustness'
            result['satisfying_fraction'] = 1.0
        elif rho_ub < 0.0:
            result['method'] = 'robustness'
            result['satisfying_fraction'] = 0.0
        else:
            method = 'exact' if self.method is None else self.method
            if method == 'exact':
                result['method'] = 'exact'
                result['satisfying_fraction'] = self.computeSatFractionExact()
            # elif method in ('exact-DNF', 'dnf'):
            #     # Keep the old DNF-based exact path available for comparison.
            #     result['method'] = 'exact-DNF'
            #     result['satisfying_fraction'] = computeSatFractionDNF(
            #         self.R,
            #         self.expanded_formula
            #     )
            elif method == 'sampling':
                result['method'] = 'sampling'
                sampling_result = self.computeSatFractionSampling()
                result.update(sampling_result)

        return result

    def computeSatFractionExact(self):
        """Compute exact satisfaction fraction from the expanded formula tree.

        The shared base Star is R[-1]. This is for reachable sequences whose time-step Stars share the same
        original predicate variable alpha and have the maximum number of constraints
        """
        base_star = self.R[-1]
        base_A, base_b = self.getBasePredicateConstraints(base_star)

        total_volume = self.computeBaseVolume(base_A, base_b)
        if total_volume == 0.0:
            raise RuntimeError('shared base predicate space has zero volume')

        satisfying_volume = self.evaluateFormulaVolumeExact(
            base_star,
            [self.expanded_formula.expr],
            base_A,
            base_b
        )
        fraction = satisfying_volume / total_volume
        return min(1.0, max(0.0, fraction))

    def evaluateFormulaVolumeExact(self, base_star, exprs, base_A, base_b):
        """Compute volume of region satisfying a conjunction of expression trees."""
        if len(exprs) == 0:
            return self.computeBaseVolume(base_A, base_b)

        expr = exprs[0]
        rest_exprs = exprs[1:]
        op, subformulas = expr

        if op == 'AP':
            C, d = self.getAtomicPredicateConstraints(base_star, subformulas)
            next_A = np.vstack((base_A, C))
            next_b= np.hstack((base_b, d))
            return self.evaluateFormulaVolumeExact(base_star, rest_exprs, next_A, next_b)

        if op == 'NOT':
            if not isinstance(subformulas, list) or len(subformulas) != 1:
                raise RuntimeError('NOT operator should have exactly one subformula')
            total_volume = self.evaluateFormulaVolumeExact(
                base_star, rest_exprs, base_A, base_b
            )
            excluded_volume = self.evaluateFormulaVolumeExact(
                base_star, [subformulas[0]] + rest_exprs, base_A, base_b
            )
            return max(0.0, total_volume - excluded_volume)

        if op == 'AND':
            if not isinstance(subformulas, list) or len(subformulas) == 0:
                raise RuntimeError('AND operator should contain subformulas')
            return self.evaluateFormulaVolumeExact(
                base_star, subformulas + rest_exprs, base_A, base_b
            )

        if op == 'OR':
            if not isinstance(subformulas, list) or len(subformulas) == 0:
                raise RuntimeError('OR operator should contain subformulas')
            union_volume = 0.0
            child_ids = range(len(subformulas))
            for size in range(1, len(subformulas) + 1):
                sign = 1.0 if size % 2 == 1 else -1.0
                for selected_ids in combinations(child_ids, size):
                    selected_exprs = [subformulas[i] for i in selected_ids]
                    union_volume += sign * self.evaluateFormulaVolumeExact(
                        base_star,
                        selected_exprs + rest_exprs,
                        base_A,
                        base_b
                    )
            return max(0.0, union_volume)

        raise RuntimeError('Unknown expanded expression operator {}'.format(op))

    def computeSatFractionSampling(self):
        """Estimate satisfaction fraction by hit-and-run samples in base alpha space."""
        base_star = self.R[-1]
        base_A, base_b = self.getBasePredicateConstraints(base_star)
        samples_base= self.hitAndRunSampling(base_A,base_b, num_samples=self.num_samples,seed=self.seed,burn_in=self.burn_in,thinning=self.thinning,feasibility_tol=self.error) #generates random samples of the Star predicate variables alpha from the base polytope
        satisfied = self.evaluateFormulaVolumeSampling(
            base_star,
            self.expanded_formula.expr,
            samples_base
        )
        n_satisfied = int(np.count_nonzero(satisfied))
        fraction = float(n_satisfied) / float(self.num_samples)
        return {
            'satisfying_fraction': fraction,
            'num_samples': self.num_samples,
            'n_satisfied': n_satisfied
        }

    def evaluateFormulaVolumeSampling(self, base_star, expr, samples):
        """Evaluate an expanded formula tree on predicate-space samples."""
        op, subformulas = expr

        if op == 'AP':
            C, d = self.getAtomicPredicateConstraints(base_star, subformulas)
            return np.all(
                C @ samples.T <= d[:, np.newaxis] + self.error,
                axis=0
            )

        values= []
        for sub_expr in subformulas:
            value = self.evaluateFormulaVolumeSampling(base_star, sub_expr, samples)
            values.append(value)

        if op == 'NOT':
            if len(values) != 1:
                raise RuntimeError('NOT operator should have exactly one subformula')
            return np.logical_not(values[0])
        if op == 'AND':
            return np.logical_and.reduce(values)
        if op == 'OR':
            return np.logical_or.reduce(values)
        raise RuntimeError('Unknown expanded expression operator {}'.format(op))

    def computeBaseVolume(self, A, b):
        """Compute the exact volume of a bounded polytope {x | A*x <= b}."""
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        if A.ndim != 2:
            raise RuntimeError('A should be a 2D numpy array')
        if A.shape[0] != b.shape[0]:
            raise RuntimeError('A and b should have the same number of constraints')

        poly = pc.Polytope(A, b)
        if pc.is_empty(poly):
            return 0.0

        return float(pc.volume(poly))

    def getAtomicPredicateConstraints(self, base_star, atomic_predicate):
        if not isinstance(atomic_predicate, AtomicPredicate):
            raise RuntimeError('Missing an AtomicPredicate')

        time_idx = 0 if atomic_predicate.t is None else atomic_predicate.t

        if time_idx < 0 or time_idx >= len(self.R):
            raise RuntimeError('invalid time t={}'.format(time_idx))

        current_set = self.R[time_idx]
        context = 'reachable set at time {}'.format(time_idx)
        if isinstance(current_set, tuple):
            current_set = list(current_set)
        if isinstance(current_set, list):
            if len(current_set) != 1:
                raise RuntimeError(
                    '{} supports one Star set per time step'.format(context)
                )
            current_set = current_set[0]

        assert isinstance(current_set, Star), '{} should be a Star'.format(context)
        assert current_set.nVars == base_star.nVars, (
            'reachable set at time {} does not use the shared alpha dimension'
            .format(time_idx)
        )

        C = np.matmul(
            atomic_predicate.A.reshape(1, -1),
            current_set.V[:, 1:]
        ).reshape(1, -1)
        d = atomic_predicate.b - np.matmul(
            atomic_predicate.A.reshape(1, -1),
            current_set.V[:, 0]
        )
        d = d.reshape(-1)

        if len(current_set.C) != 0:
            C = np.vstack((C, current_set.C))
            d = np.hstack((d, current_set.d))

        return C, d

    def getBasePredicateConstraints(self, base_star):
        base_A = []
        base_b = []
        if len(base_star.C) != 0:
            base_A.append(np.asarray(base_star.C, dtype=float))
            base_b.append(np.asarray(base_star.d, dtype=float).reshape(-1))
        base_A.append(np.eye(base_star.nVars))
        base_b.append(np.asarray(base_star.pred_ub, dtype=float).reshape(-1))
        base_A.append(-np.eye(base_star.nVars))
        base_b.append(-np.asarray(base_star.pred_lb, dtype=float).reshape(-1))
        base_A = np.vstack(base_A)
        base_b = np.hstack(base_b)
        return base_A, base_b

    def findFeasiblePoint(self, A, b, feasibility_tol=None):
        """Find a feasible point inside the polytope."""
        if feasibility_tol is None:
            feasibility_tol = self.error
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        m, n = A.shape
        row_norm = np.linalg.norm(A, axis=1)
        objective = np.hstack((np.zeros(n), -1.0))
        A_ub = np.hstack((A, row_norm.reshape(m, 1)))
        bounds = [(None, None)] * n + [(0.0, None)]
        res = linprog(objective, A_ub=A_ub, b_ub=b, bounds=bounds, method='highs')
        if res.success and np.all(A @ res.x[:n] <= b + feasibility_tol):
            return res.x[:n]

        res = linprog(
            np.zeros(n),
            A_ub=A,
            b_ub=b,
            bounds=[(None, None)] * n,
            method='highs'
        )
        if res.success and np.all(A @ res.x <= b + feasibility_tol):
            return res.x
        raise RuntimeError('could not find a feasible point in shared predicate space')

    def hitAndRunSampling(self, A, b, num_samples=None, seed=None, burn_in=None, thinning=None,
            feasibility_tol=None):
        """ Uniform hit-and-run sampler for bounded polytopes A*alpha <= b.
            step 1. Find one feasible starting point alpha_0 inside the polytope.
            step 2. Pick a random direction (u) through the current point.
            step 3. Find the interval of lambda values that keeps the point inside the polytope: A alpha_k + lambda A u <= b
            step 4. Sample random point uniformly on that interval and move to the next point.
            step 5. Repeat steps 2-4.
            step 6. Discard the first burn_in steps.
            step 7. Keep every thinning-th sample.
            """
        if num_samples is None:
            num_samples = self.num_samples
        if seed is None:
            seed = self.seed
        if thinning is None:
            thinning = self.thinning
        if feasibility_tol is None:
            feasibility_tol = self.error

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        if A.ndim != 2:
            raise RuntimeError('A should be a 2D numpy array')
        if A.shape[0] != b.shape[0]:
            raise RuntimeError('A and b should have the same number of constraints')

        n_vars = A.shape[1]
        if n_vars == 0:
            return np.zeros((num_samples, 0))

        rng = np.random.default_rng(seed)
        alpha = self.findFeasiblePoint(A, b, feasibility_tol=feasibility_tol)  # step 1: find a feasible starting point
        if burn_in is None:
            burn_in = max(100, 10 * n_vars)
        if burn_in < 0:
            raise RuntimeError('burn_in should be nonnegative')

        samples = []
        total_steps = burn_in + num_samples * thinning
        for step in range(total_steps): # step 5: Repeat
            direction = rng.normal(size=n_vars) # step 2: pick a random direction
            direction_norm = np.linalg.norm(direction)
            while direction_norm == 0.0:
                direction = rng.normal(size=n_vars)
                direction_norm = np.linalg.norm(direction)
            direction = direction / direction_norm

            A_u = A @ direction # step 3: compute the line interval that stays inside A alpha <= b
            right_side = b - A @ alpha
            lambda_lower = -np.inf
            lambda_upper = np.inf

            positive = A_u > feasibility_tol
            negative = A_u < -feasibility_tol
            if np.any(positive):
                lambda_upper = min(lambda_upper, np.min(right_side[positive] / A_u[positive]))
            if np.any(negative):
                lambda_lower = max(lambda_lower, np.max(right_side[negative] / A_u[negative]))
            if not np.isfinite(lambda_lower) or not np.isfinite(lambda_upper):
                raise RuntimeError('hit-and-run requires a bounded predicate polytope')

            if lambda_lower > lambda_upper:
                raise RuntimeError('hit-and-run encountered an empty line interval')

            alpha = alpha + rng.uniform(lambda_lower, lambda_upper) * direction # step 4: sample random point uniformly on that interval and move to the next point
            if step >= burn_in and (step - burn_in) % thinning == 0: # step 6: Discard the first burn_in steps and step 7: Keep every thinning-th sample
                samples.append(alpha.copy())

        return np.asarray(samples[:num_samples])



if __name__ == "__main__":

    EVOT = _EVENTUALLY_(0, 1)
    EV12 = _EVENTUALLY_(1, 2)
    AWOT = _ALWAYS_(0, 1)
    AW03 = _ALWAYS_(0, 3)
    lb  = _LeftBracket_()
    rb  = _RightBracket_()
    AND = _AND_()
    OR  = _OR_()
    UNTIL = _UNTIL_(2, 3)
    P1 = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.05]))
    P2 = AtomicPredicate(np.array([0.0, 1.0]), np.array([0.02]))  
    P3 = AtomicPredicate(np.array([-1.0, 0.0]), np.array([0.01]))

    # EVENTUALLY_[0,1] (P1 OR ALWAYS_[0,1] P2)
    spec1= Formula([EVOT,lb,P1,OR,lb,AWOT,P2,rb,rb])
    spec1.print()

    # EVENTUALLY_[0,1] (P1 AND (P2 UNTIL_[2,3] P3))
    spec2= Formula([EVOT,lb,P1,AND,lb,P2,UNTIL,P3,rb,rb])
    spec2.print()

    # Test ExpandedFormula class
    Ex_F = ExpandedFormula(spec2)
    print("Expanded formula:")
    Ex_F.print()

    # Example 1: Test getRobustnessInterval function
    print("\n============================ Example 1 =============================")
    X0 = Star(np.array([0.0, 0.0]), np.array([0.4, 0.2]))
    X1 = Star(np.array([0.5, -0.1]), np.array([1.0, 0.1]))
    X2 = Star(np.array([1.0, 0.0]), np.array([1.6, 0.3]))
    X3 = Star(np.array([1.5, -0.2]), np.array([1.8, 0.2]))
    R = [X0,X1,X2,X3]
    print("Reachable set sequence:")
    for t, R_t in enumerate(R):
        print("R[{}]: {}".format(t, R_t))

    # Example 2:  Test Always operator Robustness Interval
    # Always_[0,3] (x <= 2.0)
    print("\n============================ Example 2 =============================")
    P2_safe = AtomicPredicate(np.array([1.0, 0.0]), np.array([2.0]))
    P2_spec = Formula([AW03, lb, P2_safe, rb])
    P2_spec.print()
    Ex_P2 = ExpandedFormula(P2_spec)
    print("Expanded ALWAYS formula:")
    Ex_P2.print()
    RI_2 = GetRobustnessInterval(R,lp_solver='linprog')
    rho_lb, rho_ub = RI_2.getRobustnessInterval(Ex_P2)
    print("Always: ==> rho_lb = {}, rho_ub = {}".format(rho_lb, rho_ub))
    # Result is Always: ==> rho_lb = 0.20000000000000007, rho_ub = 0.5000000000000001
    # which means the reachable set sequence satisfies the specification, and the robustness interval is [0.2, 0.5].


    # Example 3: Test nested operator Robustness Interval
    # EVENTUALLY_[0,1] (x <= -0.5 AND ALWAYS_[0,1] (y <= 1.0))
    print("\n============================ Example 3 =============================")
    P3_x = AtomicPredicate(np.array([1.0, 0.0]), np.array([-0.5]))
    P3_y = AtomicPredicate(np.array([0.0, 1.0]), np.array([1.0]))
    P3_spec = Formula([EVOT, lb,P3_x, AND, lb, AWOT, P3_y, rb,rb])
    P3_spec.print()
    Ex_P3 = ExpandedFormula(P3_spec)
    print("Expanded nested formula:")
    Ex_P3.print()
    RI_3 = GetRobustnessInterval(R, lp_solver='linprog')
    rho_lb, rho_ub = RI_3.getRobustnessInterval(Ex_P3)
    print("Nested: ==> rho_lb = {}, rho_ub = {}".format(rho_lb, rho_ub))
    # Result is Nested: ==> rho_lb = -0.8999999999999999, rho_ub = -0.49999999999999994
    # which means the reachable set sequence violates the specification, and the robustness interval is [-0.9, -0.5].

    # Example 4: Test exact satisfaction fraction in a mixed case on R
    # EVENTUALLY_[1,2] (x <= 0.75) checks R[1] and R[2].
    # R[1] has x in [0.5, 1.0], so part of the predicate space satisfies it.
    # R[2] does not add any satisfying region for this threshold.
    print("\n============================ Example 4 =============================")

    P4_mixed = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.75]))
    P4_spec = Formula([EV12, lb, P4_mixed, rb])
    P4_spec.print()
    Ex_P4 = ExpandedFormula(P4_spec, T=len(R))
    print("Expanded mixed formula:")
    Ex_P4.print()

    SAT_4 = GetSatisfactionFraction(R,Ex_P4,method='exact',lp_solver='linprog')
    R_mixed_dnf_result = SAT_4.getSatisfactionFraction()
    print("R mixed satisfaction fraction: {}".format(R_mixed_dnf_result))
    # Result is : {'method': 'exact', 'rho_lb': -0.25, 'rho_ub': 0.25, 'satisfying_fraction': 0.5}

    # Example 5: Test exact satisfaction fraction for a nested mixed specification
    # EVENTUALLY_[0,1] (y <= 0.05 AND EVENTUALLY_[1,2] (x <= 0.75))
    print("\n============================ Example 5 =============================")

    P5_mixed_y = AtomicPredicate(np.array([0.0, 1.0]), np.array([0.05]))
    P5_mixed_x = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.75]))
    P5_spec = Formula([
        EVOT, lb, P5_mixed_y, AND, lb, EV12, lb, P5_mixed_x, rb, rb, rb
    ])
    P5_spec.print()
    Ex_P5 = ExpandedFormula(P5_spec, T=len(R))
    print("Expanded nested mixed formula:")
    Ex_P5.print()

    SAT_5 = GetSatisfactionFraction( R,Ex_P5,method='exact',lp_solver='linprog')
    nested_mixed_dnf_result = SAT_5.getSatisfactionFraction()
    print("Nested mixed satisfaction fraction: {}".format(nested_mixed_dnf_result))
    # Result is:{'method': 'exact', 'rho_lb': -0.25, 'rho_ub': 0.05, 'satisfying_fraction': 0.125}
    '''
    ======================== Exaplanation of Example 5 ============================
    Example 5 uses the nested specification:
    EVENTUALLY_[0,1] (y <= 0.05 AND EVENTUALLY_[1,2] (x <= 0.75))
    After expansion, the exapaned formula this becomes:
    (y[t=0] <= 0.05 AND (x[t=1] <= 0.75 OR x[t=2] <= 0.75))
    OR
    (y[t=1] <= 0.05 AND (x[t=2] <= 0.75 OR x[t=3] <= 0.75))
    
    The reachable set sequence is R = [X0, X1, X2, X3], which is defined above begin from line 786
    For X0, we have:
        x[t=0] in [0.0, 0.4]
        y[t=0] in [0.0, 0.2]
    For X1, we have:
        x[t=1] in [0.5, 1.0]
        y[t=1] in [-0.1, 0.1]
    For X2, we have:
        x[t=2] in [1.0, 1.6]
        y[t=2] in [0.0, 0.3]
    For X3, we have:
        x[t=3] in [1.5, 1.8]
        y[t=3] in [-0.2, 0.2]
    
    Based on exapaned formula, we can see that x[t=2] <= 0.75 and x[t=3] <= 0.75 are infeasible, because the smallest x value at time 2 is 1.0 and the smallest x value at time 3 is 1.5. Both are larger than 0.75.
    Therefore, the second part of the expanded formula does not contribute a satisfying region, and the useful part is:
    y[t=0] <= 0.05 AND x[t=1] <= 0.75
    
    Now we compute this in the shared base predicate space. The shared base predicate variables are alpha1 and alpha2, with:
    alpha1 in [-1, 1]
    alpha2 in [-1, 1]
    So the total shared base predicate-space area is: 2 * 2 = 4
    At time 0, y is represented as: y[t=0] = 0.1 + 0.1 alpha2
    The predicate y[t=0] <= 0.05 gives:0.1 + 0.1 alpha2 <= 0.05
    So alpha2 <= -0.5
    Inside alpha2 in [-1, 1], this keeps the interval: alpha2 in [-1, -0.5]
    The length of this interval is 0.5, which is 1/4 of the full alpha2 range.
    
    At time 1, x is represented as:  x[t=1] = 0.75 + 0.25 alpha1
    The predicate x[t=1] <= 0.75 gives: 0.75 + 0.25 alpha1 <= 0.75
    So alpha1 <= 0
    Inside alpha1 in [-1, 1], this keeps the interval: alpha1 in [-1, 0]
    The length of this interval is 1, which is 1/2 of the full alpha1 range.
    
    Therefore, the satisfying region in predicate space has area: 1 * 0.5 = 0.5
    The total shared base predicate-space area is: 4
    So the satisfaction fraction is: 0.5 / 4 = 0.125
    Equivalently: 1/2 * 1/4 = 1/8 = 0.125
    Therefore, the result of Example 5 is:
    rho_lb = -0.25
    rho_ub = 0.05
    satisfying_fraction = 0.125
'''


    # Example 6: Test exact and sampling satisfaction fraction for a nested mixed specification
    # EVENTUALLY_[0,1] ((y <= 0.05 AND EVENTUALLY_[1,2] (x <= 0.75)) OR x <= 0.10)
    print("\n============================ Example 6 =============================")
    P6_y = AtomicPredicate(np.array([0.0, 1.0]), np.array([0.05]))
    P6_x = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.75]))
    P6_x1 = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.10]))
    P6_spec = Formula([EVOT, lb,lb, P6_y, AND, lb, EV12, lb, P6_x, rb, rb, rb,OR,P6_x1,rb])
    P6_spec.print()
    Ex_P6= ExpandedFormula(P6_spec, T=len(R))
    print("Expanded nested mixed tree formula:")
    Ex_P6.print()

    SAT_6 = GetSatisfactionFraction( R,Ex_P6,method='exact',lp_solver='linprog')

    exact_result = SAT_6.getSatisfactionFraction()
    print("Nested mixed tree exact satisfaction fraction: {}".format(exact_result))
    # Expected result is approximately:
    # {'method': 'exact-tree', 'rho_lb': -0.25, 'rho_ub': 0.1, 'satisfying_fraction': 0.3125}

    SAT_6_sampling = GetSatisfactionFraction( R,Ex_P6,method='sampling',num_samples=20000,seed=1,lp_solver='linprog')
    sampling_result = SAT_6_sampling.getSatisfactionFraction()
    print("Nested mixed tree sampling satisfaction fraction: {}".format(sampling_result))
    # Sampling result should be close to 0.3125.

    # Example 6b: Aggregate volume over coherent ReLU split traces(branches).
    # Each trace starts from the same R0, then follows one coherent split chain:
    # trace 1: R0, r11, r21, r31
    # trace 2: R0, r11, r22, r32
    # trace 3: R0, r12, r23, r33
    # trace 4: R0, r12, r24, r34
    # for _, trace in all_traces:
    #     trace_volume = computeBaseVolume(trace)
    #     branch_satisfying_volume = evaluateFormulaVolumeExact()
    #     branch_fraction =  branch_satisfying_volume / branch_volume
    #     total_branch_volume += branch_volume
    #     total_satisfying_volume += branch_satisfying_volume

    # branch_aggregated_fraction = total_satisfying_volume / base_volume
