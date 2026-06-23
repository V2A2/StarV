r'''

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

    def __init__(self, formula):

        if isinstance(formula, Formula):
            self.formula_tokens = formula.formula
        elif isinstance(formula, list):
            self.formula_tokens = formula
        else:
            raise RuntimeError('input should be a Formula object or list')

        self.formula = self.formula_tokens
        self.outer_operator = self.get_outer_operator()
        self.expr = self.getExpandedFormula(self.formula_tokens, 0, len(self.formula_tokens), 0)
        self.F = self.expr  # Backward-compatible alias for older examples.

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
            child_expr = expr[1][0]
            child_text = self.format_expression(child_expr, parent_op=op)
            if child_expr[0] != 'AP':
                child_text = '({})'.format(child_text)
            return 'NOT {}'.format(child_text)

        child_exprs = expr[1]
        separator = ' {} '.format(op)
        is_outermost = parent_op is None and op == self.outer_operator
        if is_outermost:
            separator = '\n{}\n'.format(op)

        formatted_children = []
        for child_expr in child_exprs:
            child_text = self.format_expression(child_expr, parent_op=op)
            if is_outermost and child_expr[0] != 'AP':
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
            (op, [child_expr1, child_expr2, ...]), where op is 'NOT',
            'AND', or 'OR' and chid_expr is a nested expression tree.
        '''
        expr_terms = []
        bool_ops = []
        token_index = start
        while token_index < end:
            token = tokens[token_index]
            if isinstance(token, AtomicPredicate):
                predicate_time = 0 if token.t is None else token.t
                expr_terms.append(('AP', token.at_time(predicate_time + time_offset)))
                token_index += 1
            elif isinstance(token, _NOT_):
                next_index = token_index + 1
                if next_index >= end:
                    raise RuntimeError('NOT must be followed by a subformula')

                subformula_tokens, token_index = self.expand_subformula(tokens, next_index, end)
                expr_terms.append((
                    'NOT',
                    [self.getExpandedFormula(subformula_tokens, 0, len(subformula_tokens), time_offset)]
                ))
            elif isinstance(token, _NEXT_):
                next_index = token_index + 1
                if next_index >= end:
                    raise RuntimeError('NEXT must be followed by a subformula')

                subformula_tokens, token_index = self.expand_subformula(tokens, next_index, end)
                expr_terms.append(
                    self.getExpandedFormula(
                        subformula_tokens, 0, len(subformula_tokens), time_offset + 1
                    )
                )
            elif isinstance(token, _ALWAYS_) or isinstance(token, _EVENTUALLY_):
                next_index = token_index + 1
                if next_index >= end:
                    raise RuntimeError('temporal operator must be followed by a subformula')

                subformula_tokens, token_index = self.expand_subformula(tokens, next_index, end)

                expanded_terms = []
                for dt in self.get_time_range(token):
                    expanded_terms.append(
                        self.getExpandedFormula(subformula_tokens, 0, len(subformula_tokens), time_offset + dt)
                    )
                if isinstance(token, _ALWAYS_):
                    expr_terms.append(('AND', expanded_terms))
                else:
                    expr_terms.append(('OR', expanded_terms))
            elif isinstance(token, _UNTIL_):
                if len(expr_terms) == 0:
                    raise RuntimeError('UNTIL must have a left subformula')

                next_index = token_index + 1
                if next_index >= end:
                    raise RuntimeError('UNTIL must be followed by a right subformula')

                left_expr = expr_terms.pop()
                right_tokens, token_index = self.expand_subformula(tokens, next_index, end)
                expr_terms.append(
                    self.UNTIL_expand(left_expr, right_tokens, token, time_offset)
                )
            elif isinstance(token, _LeftBracket_):
                right_bracket_index = self.match_right_loop_id(tokens, token_index)
                subformula_tokens = Formula(tokens).getSubFormula(token_index + 1, right_bracket_index)
                expr_terms.append(self.getExpandedFormula(subformula_tokens, 0, len(subformula_tokens), time_offset))
                token_index = right_bracket_index + 1
            elif isinstance(token, _AND_):
                bool_ops.append('AND')
                token_index += 1
            elif isinstance(token, _OR_):
                bool_ops.append('OR')
                token_index += 1
            elif isinstance(token, _RightBracket_):
                token_index += 1
            else:
                raise RuntimeError('unsupported item in formula: {}'.format(type(token)))

        if len(expr_terms) == 0:
            raise RuntimeError('empty subformula segment')
        if len(bool_ops) != len(expr_terms) - 1:
            raise RuntimeError('invalid subformula segment: operators and terms do not match')
        if len(bool_ops) == 0:
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
            right_terms = self.getExpandedFormula(
                right_tokens, 0, len(right_tokens), time_offset + witness_time
            )
            until_terms.append(('AND', left_terms + [right_terms]))
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
            return ('AP', predicate.at_time(predicate.t + time_offset))
        return (op, [self.shift_time(child_expr, time_offset) for child_expr in expr[1]])

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
    r'''
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

    child_exprs = children_or_predicate
    if not isinstance(child_exprs, list) or len(child_exprs) == 0:
        raise RuntimeError('{} operator should contain a nonempty list of subformulas'.format(op))

    child_intervals = []
    for child_expr in child_exprs:
        child_interval = getExpandedRobustnessInterval(R, child_expr, lp_solver)
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


def computeSatisfactionFraction(
        R, expanded_formula, num_samples=10000, lp_solver='linprog',
        error=1e-10, method=None):
    '''
    Compute the fraction of the feasible predicate space satisfying a TL formula.

    '''
    if not isinstance(R, (list, tuple)):
        raise RuntimeError('R should be a reachable set sequence as a list or tuple')
    if not isinstance(num_samples, int) or num_samples < 1:
        raise RuntimeError('num_samples should be a positive integer')
    if method not in (None, 'polytope', 'sampling'):
        raise RuntimeError("method should be None, 'polytope', or 'sampling'")
    if not isinstance(expanded_formula, ExpandedFormula):
        raise RuntimeError('expanded_formula should be an ExpandedFormula object')

    rho_lb, rho_ub = getRobustnessInterval(R, expanded_formula, lp_solver)
    result = {
        'method': None,
        'rho_lb': rho_lb,
        'rho_ub': rho_ub,
        'satisfying_fraction': None,
    }

    if rho_lb >= 0.0:
        result['method'] = 'robustness'
        result['satisfying_fraction'] = 1.0
    elif rho_ub < 0.0:
        result['method'] = 'robustness'
        result['satisfying_fraction'] = 0.0
    else:
        # TODO: compute the mixed case with exact polytope volume for small
        # fixed predicate spaces, or sampling for larger spaces.
        result['method'] = method

    return result


if __name__ == "__main__":

    EVOT = _EVENTUALLY_(0, 1)
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

    # Test getRobustnessInterval function
    X0 = Star(np.array([0.0, 0.0]), np.array([0.4, 0.2]))
    X1 = Star(np.array([0.5, -0.1]), np.array([1.0, 0.1]))
    X2 = Star(np.array([1.0, 0.0]), np.array([1.6, 0.3]))
    X3 = Star(np.array([1.5, -0.2]), np.array([1.8, 0.2]))
    R = [X0,X1,X2,X3]
    print("Reachable set sequence:")
    for t, R_t in enumerate(R):
        print("R[{}]: {}".format(t, R_t))

    # Test Always operator Robustness Interval
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


    # Test nested operator Robustness Interval
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

