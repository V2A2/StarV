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
    how robustly the specification is satisfied or violated ( robustness interval:[\rho_{\phi}_lb, \rho_{\phi}_ub]); 
    how much (what fraction) of predicate-space reachable sets satisfy (satisfaction fraction \phi (\q_{\phi})).

==================================================================================

StarTL SYNTAX and BOOLEAN SEMANTICS are same as dProbStarTL, which is defined as follows:
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
            self.original_formula = formula
            self.formula_tokens = formula.formula
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
            child_text = self.format_expression(sub_expr, parent_op=op)
            if sub_expr[0] != 'AP':
                child_text = '({})'.format(child_text)
            return 'NOT {}'.format(child_text)

        sub_exprs = expr[1]
        separator = ' {} '.format(op)
        is_outermost = parent_op is None and op == self.outer_operator
        if is_outermost:
            separator = '\n{}\n'.format(op)

        formatted_children = []
        for sub_expr in sub_exprs:
            child_text = self.format_expression(sub_expr, parent_op=op)
            if is_outermost and sub_expr[0] != 'AP':
                if not (child_text.startswith('(') and child_text.endswith(')')):
                    child_text = '({})'.format(child_text)
            formatted_children.append(child_text)
        text = separator.join(formatted_children)
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

                expanded_terms = []
                for dt in self.get_time_range(token):
                    sub_expr = self.getExpandedFormula(
                        sub_tokens, 0, len(sub_tokens), time_offset + dt
                    )
                    if sub_expr is not None:
                        expanded_terms.append(sub_expr)
                if len(expanded_terms) == 0:
                    continue
                if isinstance(token, _ALWAYS_):
                    expr_terms.append(('AND', expanded_terms))
                else:
                    expr_terms.append(('OR', expanded_terms))
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

        op_types = set(bool_ops)
        if len(op_types) > 1:
            raise RuntimeError(
                'mixed AND/OR operations must be bracketed, e.g., '
                '(P1 OR P2) AND P3 or P1 OR (P2 AND P3)'
            )

        op = bool_ops[0]
        if op == 'AND':
            return ('AND', expr_terms)
        if op == 'OR':
            return ('OR', expr_terms)
        raise RuntimeError('unknown operator {}'.format(op))

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

        shifted_children = [
            self.shift_time(sub_expr, time_offset)
            for sub_expr in expr[1]
        ]
        shifted_children = [
            sub_expr for sub_expr in shifted_children
            if sub_expr is not None
        ]
        if len(shifted_children) == 0:
            return None
        return (op, shifted_children)

    def isvalid_time(self, time_index):
        """Return False when time_index is outside the reachable sequence."""
        if time_index < 0:
            return False
        return self.T is None or time_index < self.T

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


def getRobustnessInterval(R, expanded_formula, lp_solver='linprog'):
    '''
    Compute the robustness interval of a temporal formula given a reachable set sequence.
    \rho_{\phi}_lb is the lower bound of the robustness interval, which is the minimum robustness value within predicate-space range.
    \rho_{\phi}_ub is the upper bound of the robustness interval, which is the maximum robustness value within predicate-space range.

    Args:
        R: reachable set sequence. R[t] is the Star reachable set at
           time t. R[t] may also be a list of reachable sets due to ReLU.
        expanded_formula: ExpandedFormula object, for example
                 Ex_F = ExpandedFormula(spec).
        lp_solver: LP solver passed to Star getMin/getMax.

    Returns:
        (rho_lb, rho_ub): robustness interval of the expanded formula.
    '''
    if not isinstance(R, (list, tuple)):
        raise RuntimeError('R should be a reachable set sequence stored as a list or tuple')
    if not isinstance(expanded_formula, ExpandedFormula):
        raise RuntimeError('expanded_formula should be an ExpandedFormula object')

    return getExpandedRobustnessInterval(R, expanded_formula.expr, lp_solver)


def getExpandedRobustnessInterval(R, expr, lp_solver='linprog'):
    """Recursively compute the robustness interval of an expanded expression."""
    op, children_or_predicate = expr
    if op == 'AP':
        return getAtomicRobustnessInterval(R, children_or_predicate, lp_solver)

    sub_exprs = children_or_predicate
    if not isinstance(sub_exprs, list) or len(sub_exprs) == 0:
        raise RuntimeError('{} operator should contain a nonempty list of subformulas'.format(op))

    child_intervals = []
    for sub_expr in sub_exprs:
        child_interval = getExpandedRobustnessInterval(R, sub_expr, lp_solver)
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


def getAtomicRobustnessInterval(R, predicate, lp_solver='linprog'):
    """Compute min/max of atomic robustness b - A*x[t] over reachable set R[t]."""
    if not isinstance(predicate, AtomicPredicate):
        raise RuntimeError('Missing an AtomicPredicate')

    t = 0 if predicate.t is None else predicate.t
    if t < 0 or t >= len(R):
        raise RuntimeError('invalid time t={}'.format(t))

    reachable_sets = R[t]
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
        lower_bounds.append(robustness_set.getMin(0, lp_solver))
        upper_bounds.append(robustness_set.getMax(0, lp_solver))

    return min(lower_bounds), max(upper_bounds)


def getSatisfactionFraction(
        R, expanded_formula, num_samples=10000, lp_solver='linprog',
        error=1e-10, method=None):
    '''
    Compute the fraction of the feasible predicate space satisfying a TL formula.

    '''
    if not isinstance(R,list):
        raise RuntimeError('R should be a reachable set sequence as a list')
    if not isinstance(num_samples, int) or num_samples < 1:
        raise RuntimeError('num_samples should be a positive integer')
    if method not in (
            None, 'base-polytope', 'exact', 'expanded',
            'expanded-polytope', 'exact-DNF', 'exact-expanded',
            'sampling'):
        raise RuntimeError(" Unkown satisfaction fraction compuation method")
    if not isinstance(expanded_formula, ExpandedFormula):
        raise RuntimeError('expanded_formula should be an ExpandedFormula object')

    rho_lb, rho_ub = getRobustnessInterval(R, expanded_formula, lp_solver)
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
        if method is None:
            method = 'exact-expanded'
        if method in ('exact', 'base-polytope', 'exact-DNF'):
            # Use DNF to compute SAT fraction.
            result['method'] = 'exact-DNF'
            result['satisfying_fraction'] = computeSatFraction(
                R,
                expanded_formula
            )
        elif method == 'sampling':
            # TODO: implement sampling for high-dimensional predicate spaces.
            result['method'] = 'sampling'

    return result


def computePolyVolume(A, b, tol=1e-9):
    """Compute the exact volume of a bounded polytope {x | A*x <= b}.
    """
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


def computeSatFraction(R, expanded_formula):
    """Compute exact fraction over the initial Star predicate space using DNF.

    The base Star is always R[0]. This is for reachable sequences whose time-step Stars share the same
    original predicate variable alpha
    """
    if not isinstance(R, list):
        raise RuntimeError('R should be a reachable set sequence as a list')

    base_star = R[0]
    if isinstance(base_star, tuple):
        base_star = list(base_star)
    if isinstance(base_star, list):
        if len(base_star) != 1:
            raise RuntimeError(
                'base-polytope satisfaction fraction supports one Star set per time step'
            )
        base_star = base_star[0]

    if not isinstance(base_star, Star):
        raise RuntimeError('reachable set should be a Star')

    if not isinstance(expanded_formula, ExpandedFormula):
        raise RuntimeError('expanded_formula should be an ExpandedFormula object')

    A = []
    B = []
    if len(base_star.C) != 0:
        A.append(base_star.C)
        B.append(base_star.d)
    A.append(np.eye(base_star.nVars))
    B.append(base_star.pred_ub)
    A.append(-np.eye(base_star.nVars))
    B.append(-base_star.pred_lb)
    base_A = np.vstack(A)
    base_b = np.hstack(B)

    total_volume = computePolyVolume(base_A, base_b)
    if total_volume == 0.0:
        raise RuntimeError('base predicate space has zero volume')

    # use DNF for satisfaction computation
    dynamic_formula = expanded_formula.original_formula.getDynamicFormula() # convert formula to DNF
    dnf_clauses = dynamic_formula.F

    print("\n======== Create Polytope of each DNF cluase ========")
    clause_polytopes = createDNFPolytopes(
        R,
        base_star,
        dnf_clauses
    )

    print("\n======== Compute volume of Polytope of each DNF cluase ========")
    satisfying_volume = getVolumeOfPolytopes(clause_polytopes)
    fraction = satisfying_volume / total_volume
    return min(1.0, max(0.0, fraction))


def createDNFPolytopes(R, base_star, dnf_clauses):

    """Realize DNF clauses and build one predicate-space polytope per disjucnt.
    F = ( P1 or P2 or P3 ,..., or Pn), each Pi is a polytope
    """
    if not isinstance(base_star, Star):
        raise RuntimeError('base_star should be a Star')

    T = len(R)
    nVars = base_star.nVars

    A = []
    b = []

    if len(base_star.C) != 0:
        A.append(base_star.C)
        b.append(base_star.d)
    A.append(np.eye(nVars))
    b.append(base_star.pred_ub)
    A.append(-np.eye(nVars))
    b.append(-base_star.pred_lb)
    base_A = np.vstack(A)
    base_b = np.hstack(b)

    # store the realized constraints for each valid DNF clause.
    constraints = []
    for clause in dnf_clauses:
        C = None
        d = None
        referenced_times = set()
        reachable_sets = {}
        for predicate in clause:
            if not isinstance(predicate, AtomicPredicate):
                raise RuntimeError('Missing an AtomicPredicate')

            time_index =  predicate.t
            if time_index < 0 or time_index >= T:
                C = None
                d = None
                break

            reachable_set = R[time_index]

            if not isinstance(reachable_set, Star):
                raise RuntimeError('reachable set should be a Star')

            if reachable_set.nVars != base_star.nVars:
                raise RuntimeError(
                    'reachable set at time {} does not use the base alpha dimension'.format(time_index)
                )

            C1 = np.matmul(
                predicate.A.reshape(1, -1),
                reachable_set.V[:, 1:]
            ).reshape(1, -1)
            d1 = predicate.b - np.matmul(
                predicate.A.reshape(1, -1),
                reachable_set.V[:, 0]
            )
            d1 = d1.reshape(-1)

            if C is None:
                C = C1
                d = d1
            else:
                C = np.vstack((C, C1))
                d = np.concatenate((d, d1))

            referenced_times.add(time_index)
            reachable_sets[time_index] = reachable_set

        if C is not None:
            constraints.append([C, d, referenced_times, reachable_sets])

    clause_polytopes = []
    for C, d, referenced_times, reachable_sets in constraints:
        A = [base_A]
        B = [base_b]
        A.append(C)
        B.append(d)

        for time_index in sorted(referenced_times):
            reachable_set = reachable_sets[time_index]
            if len(reachable_set.C) != 0:
                A.append(reachable_set.C)
                B.append(reachable_set.d)

        # Combines all constraints(base_star + each Pi) into one polytope:
        clause_A = np.vstack(A)
        clause_b = np.hstack(B)
        if computePolyVolume(clause_A, clause_b) > 0.0:
            clause_polytopes.append((clause_A, clause_b))

    return clause_polytopes


def getVolumeOfPolytopes(clause_polytopes):
    """ compute the union volume of all satisfying clause polytopes."""
    if len(clause_polytopes) == 0:
        return 0.0

    VOL = 0.0

    N = range(0,len(clause_polytopes)) # number of polytope needed tobe unioned ( P1 or P2 or P3,...)
    for i in range(0, len(clause_polytopes)):
        volume = 0.0
        comb = combinations(N, i+1)
        for j in comb:
            A = []
            B = []
            for clause_id in j:
                clause_A, clause_b = clause_polytopes[clause_id]
                A.append(clause_A)
                B.append(clause_b)
            print("\n======== Compute Polytope volume of each combination ========")
            vol = computePolyVolume(np.vstack(A), np.hstack(B))
            volume = volume + (-1)**i * vol

        VOL = volume + VOL

    return max(0.0, VOL)


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
    P_safe = AtomicPredicate(np.array([1.0, 0.0]), np.array([2.0]))
    always_spec = Formula([AW03, lb, P_safe, rb])
    always_spec.print()
    Ex_AW = ExpandedFormula(always_spec)
    print("Expanded ALWAYS formula:")
    Ex_AW.print()
    rho_lb, rho_ub = getRobustnessInterval(
        R, Ex_AW, lp_solver='linprog'
    )
    print("Always: ==> rho_lb = {}, rho_ub = {}".format(rho_lb, rho_ub))
    # Result is Always: ==> rho_lb = 0.20000000000000007, rho_ub = 0.5000000000000001
    # which means the reachable set sequence satisfies the specification, and the robustness interval is [0.2, 0.5].


    # Example 3: Test nested operator Robustness Interval
    # EVENTUALLY_[0,1] (x <= -0.5 AND ALWAYS_[0,1] (y <= 1.0))
    P_x = AtomicPredicate(np.array([1.0, 0.0]), np.array([-0.5]))
    P_y = AtomicPredicate(np.array([0.0, 1.0]), np.array([1.0]))
    nested_spec = Formula([EVOT, lb,P_x, AND, lb, AWOT, P_y, rb,rb])
    nested_spec.print()
    Ex_nested = ExpandedFormula(nested_spec)
    print("Expanded nested formula:")
    Ex_nested.print()
    rho_lb, rho_ub = getRobustnessInterval(
        R, Ex_nested, lp_solver='linprog'
    )
    print("Nested: ==> rho_lb = {}, rho_ub = {}".format(rho_lb, rho_ub))
    # Result is Nested: ==> rho_lb = -0.8999999999999999, rho_ub = -0.49999999999999994
    # which means the reachable set sequence violates the specification, and the robustness interval is [-0.9, -0.5].

    # Example 4: Test exact satisfaction fraction in a mixed case on R
    # EVENTUALLY_[1,2] (x <= 0.75) checks R[1] and R[2].
    # R[1] has x in [0.5, 1.0], so part of the predicate space satisfies it.
    # R[2] does not add any satisfying region for this threshold.
    P_mixed = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.75]))
    mixed_spec = Formula([EV12, lb, P_mixed, rb])
    mixed_spec.print()
    Ex_mixed = ExpandedFormula(mixed_spec, T=len(R))
    print("Expanded mixed formula:")
    Ex_mixed.print()

    R_mixed_dnf_result = getSatisfactionFraction(
        R,
        Ex_mixed,
        method='exact-DNF',
        lp_solver='linprog'
    )
    print("R mixed DNF satisfaction fraction: {}".format(R_mixed_dnf_result))
    # Result is : {'method': 'exact-DNF', 'rho_lb': -0.25, 'rho_ub': 0.25, 'satisfying_fraction': 0.5}


    # Example 5: Test exact satisfaction fraction for a nested mixed specification
    # EVENTUALLY_[0,1] (y <= 0.05 AND EVENTUALLY_[1,2] (x <= 0.75))
    P_mixed_y = AtomicPredicate(np.array([0.0, 1.0]), np.array([0.05]))
    P_mixed_x = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.75]))
    nested_mixed_spec = Formula([
        EVOT, lb, P_mixed_y, AND, lb, EV12, lb, P_mixed_x, rb, rb, rb
    ])
    nested_mixed_spec.print()
    Ex_nested_mixed = ExpandedFormula(nested_mixed_spec, T=len(R))
    print("Expanded nested mixed formula:")
    Ex_nested_mixed.print()

    nested_mixed_dnf_result = getSatisfactionFraction(
        R,
        Ex_nested_mixed,
        method='exact-DNF',
        lp_solver='linprog'
    )
    print("Nested mixed DNF satisfaction fraction: {}".format(nested_mixed_dnf_result))
    # Result is:{'method': 'exact-DNF', 'rho_lb': -0.25, 'rho_ub': 0.05, 'satisfying_fraction': 0.125}
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
    
    Now we compute this in the base predicate space. The base predicate variables are alpha1 and alpha2, with:
    alpha1 in [-1, 1]
    alpha2 in [-1, 1]
    So the total base predicate-space area is: 2 * 2 = 4
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
    The total base predicate-space area is: 4
    So the satisfaction fraction is: 0.5 / 4 = 0.125
    Equivalently: 1/2 * 1/4 = 1/8 = 0.125
    Therefore, the result of Example 5 is:
    rho_lb = -0.25
    rho_ub = 0.05
    satisfying_fraction = 0.125
'''