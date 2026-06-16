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
    whether the trajectory satisfy \phi;
    whether the trajectory violate \phi;
    how robustly the specification is satisfied or violated ( robustness interval:[\rho_{\phi}_lb, \rho_{\phi}_ub]); 
    how much (what fraction) of predicate-space trajectories satisfy (satisfaction fraction \phi (\q_{\phi})).

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


# Expand the temporal logic specification formulas

class DynamicFormula(object):

    """
    Store and print the time-expanded expression tree produced by ExtendFormula.

    This class does not expand the formula into ADNF/DNF.  It keeps the nested
    AND/OR structure so the printed formula stays close to the original temporal
    specification.

    """

    def __init__(self, F=None, outer_operator=None):
        if F is None:
            F = []
        self.F = F
        self.length = len(F) if isinstance(F, list) else 1
        self.outer_operator = outer_operator

    def print(self):
        print(self)

    def __str__(self):
        if isinstance(self.F, tuple):
            return self._format_(self.F, multiline_outer=True)
        if self.length == 0:
            return '{}'.format(self.F)
        return '{}'.format(self.F)

    def _format_(self, node, outer_op=None, multiline_outer=False):
        op = node[0]
        if op == 'AP':
            pred = node[1]
            if isinstance(pred, AtomicPredicate):
                return '{} * x[t={}] <= {}'.format(pred.A, pred.t, pred.b)
            return str(pred)
        if op == 'NOT':
            arg_text = self._format_(node[1][0], outer_op=op)
            if node[1][0][0] != 'AP':
                arg_text = '({})'.format(arg_text)
            return 'NOT {}'.format(arg_text)

        args = node[1]
        separator = ' {} '.format(op)
        if multiline_outer and op == self.outer_operator:
            separator = '\n{}\n'.format(op)

        formatted_args = []
        for arg in args:
            arg_text = self._format_(arg, outer_op=op)
            if multiline_outer and op == self.outer_operator and arg[0] != 'AP':
                if not (arg_text.startswith('(') and arg_text.endswith(')')):
                    arg_text = '({})'.format(arg_text)
            formatted_args.append(arg_text)
        text = separator.join(formatted_args)
        if outer_op == 'AND' and op == 'OR':
            return '({})'.format(text)
        if outer_op == 'OR' and op == 'AND':
            return '({})'.format(text)
        return text


class getExpandedFormula(object):
    """
    Convert a temporal Formula into a time-expanded DynamicFormula.

    The outermost temporal operator determines how the time-step clauses are
    joined: ALWAYS uses AND, and EVENTUALLY uses OR.

    Example:
        AW_[0, 1] (P1 AND (ET_[0, 1] P2))

    Expanded expression formula:
        (P1[t=0] AND (P2[t=0] OR P2[t=1]))
        AND
        (P1[t=1] AND (P2[t=1] OR P2[t=2]))
    """

    def __init__(self, formula):

        if isinstance(formula, Formula):
            self.formula = formula.formula
        elif isinstance(formula, list):
            self.formula = formula
        else:
            raise RuntimeError('input should be a Formula object or list')

        self.outer_operator = self.outer_operator()
        expr = self.parse_formula(self.formula, 0, len(self.formula), 0)
        self.expanded_formula = DynamicFormula(expr, self.outer_operator)

    def expand(self):
        'returns the expanded dynamic temporal formula'
        return self.expanded_formula

    def print(self):
        print(self)

    def __str__(self):
        return str(self.expanded_formula)

    def outer_operator(self):
        if len(self.formula) < 2:
            return None
        if not isinstance(self.formula[1], _LeftBracket_):
            return None
        if self.match_right_loop_id(self.formula, 1) != len(self.formula) - 1:
            return None
        if isinstance(self.formula[0], _EVENTUALLY_):
            return 'OR'
        if isinstance(self.formula[0], _ALWAYS_):
            return 'AND'
        return None

    def parse_formula(self, tokens, start, end, time_offset=0):
        '''
            It reads the formula from left to right and builds a nested expression tree that preserves brackets and temporal structure(not in ADNF), for example:
            ('AND', [
            ('AP', P1[t=0]),
            ('OR', [
                ('AP', P2[t=0]),
                ('AP', P2[t=1])
            ])])
        '''
        terms = []
        ops = []
        i = start
        while i < end:
            item = tokens[i]
            if isinstance(item, AtomicPredicate):
                t = 0 if item.t is None else item.t
                terms.append(('AP', item.at_time(t + time_offset)))
                i += 1
            elif isinstance(item, _NOT_):
                next_id = i + 1
                if next_id >= end:
                    raise RuntimeError('NOT must be followed by a subformula')

                sub_tokens, i = self.get_temporal_subformula(tokens, next_id, end)
                terms.append((
                    'NOT',
                    [self.parse_formula(sub_tokens, 0, len(sub_tokens), time_offset)]
                ))
            elif isinstance(item, _NEXT_):
                next_id = i + 1
                if next_id >= end:
                    raise RuntimeError('NEXT must be followed by a subformula')

                sub_tokens, i = self.get_temporal_subformula(tokens, next_id, end)
                terms.append(
                    self.parse_formula(sub_tokens, 0, len(sub_tokens), time_offset + 1)
                )
            elif isinstance(item, _ALWAYS_) or isinstance(item, _EVENTUALLY_):
                next_id = i + 1
                if next_id >= end:
                    raise RuntimeError('temporal operator must be followed by a subformula')

                sub_tokens, i = self.get_temporal_subformula(tokens, next_id, end)

                expanded = []
                for dt in self.get_time_range(item):
                    expanded.append(
                        self.parse_formula(sub_tokens, 0, len(sub_tokens), time_offset + dt)
                    )
                if isinstance(item, _ALWAYS_):
                    terms.append(('AND', expanded))
                else:
                    terms.append(('OR', expanded))
            elif isinstance(item, _UNTIL_):
                if len(terms) == 0:
                    raise RuntimeError('UNTIL must have a left subformula')

                next_id = i + 1
                if next_id >= end:
                    raise RuntimeError('UNTIL must be followed by a right subformula')

                left_expr = terms.pop()
                right_tokens, i = self.get_temporal_subformula(tokens, next_id, end)
                terms.append(
                    self.UNTIL_expand(left_expr, right_tokens, item, time_offset)
                )
            elif isinstance(item, _LeftBracket_):
                rb = self.match_right_loop_id(tokens, i)
                sub_tokens = Formula(tokens).getSubFormula(i + 1, rb)
                terms.append(self.parse_formula(sub_tokens, 0, len(sub_tokens), time_offset))
                i = rb + 1
            elif isinstance(item, _AND_):
                ops.append('AND')
                i += 1
            elif isinstance(item, _OR_):
                ops.append('OR')
                i += 1
            elif isinstance(item, _RightBracket_):
                i += 1
            else:
                raise RuntimeError('unsupported item in formula: {}'.format(type(item)))

        if len(terms) == 0:
            raise RuntimeError('empty formula segment')
        if len(ops) != len(terms) - 1:
            raise RuntimeError('invalid formula segment: operators and terms do not match')
        if len(ops) == 0:
            return terms[0]

        op_types = set(ops)
        if len(op_types) > 1:
            raise RuntimeError(
                'mixed AND/OR operations must be bracketed, e.g., '
                '(P1 OR P2) AND P3 or P1 OR (P2 AND P3)'
            )

        op = ops[0]
        if op == 'AND':
            return ('AND', terms)
        if op == 'OR':
            return ('OR', terms)
        raise RuntimeError('unknown operator {}'.format(op))

    def get_temporal_subformula(self, tokens, start_id, end): #find the formula that comes after a temporal operator. It can be either a single term (e.g., an atomic predicate) or a bracketed subformula.
        if isinstance(tokens[start_id], _LeftBracket_):
            rb = self.match_right_loop_id(tokens, start_id)
            if rb >= end:
                raise RuntimeError('right bracket is outside the current formula segment')
            return Formula(tokens).getSubFormula(start_id + 1, rb), rb + 1
        return Formula(tokens).getSubFormula(start_id, start_id + 1), start_id + 1

    def UNTIL_expand(self, left_expr, right_tokens, until_op, time_offset):
        'expand p1 U_[a,b] p2 into an OR over all possible witness times'

        assert isinstance(until_op, _UNTIL_), 'error: input should be an UNTIL operator'
        assert isinstance(right_tokens, list), 'error: right_tokens should be a list'
        assert until_op.start_time >= 0, 'error: t_start should be >= 0'

        until_terms = []
        for witness_time in self.get_time_range(until_op):
            left_terms = [
                self.shift_time(left_expr, dt)
                for dt in range(0, witness_time)
            ]
            right_terms = self.parse_formula(
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

    def shift_time(self, node, time_offset):
        op = node[0]
        if op == 'AP':
            pred = node[1]
            return ('AP', pred.at_time(pred.t + time_offset))
        return (op, [self.shift_time(arg, time_offset) for arg in node[1]])

    def getLoopIds(self, formula=None):
        if formula is None:
            formula = self.formula
        return Formula(formula).getLoopIds()

    def match_right_loop_id(self, tokens, left_index):
        if left_index >= len(tokens) or not isinstance(tokens[left_index], _LeftBracket_):
            raise RuntimeError('left_index must point to a left bracket')

        lb_idxes, rb_idxes = self.getLoopIds(tokens)
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


class getRobustnessInterval(object):
    r'''
    Compute the robustness interval of a temporal formula given a reachable set sequence.
    \rho_{\phi}_lb is the lower bound of the robustness interval, which is the minimum robustness value of all predicate-space trajectories.
    \rho_{\phi}_ub is the upper bound of the robustness interval, which is the maximum robustness value of all predicate-space trajectories.
    '''


class computeSatisfactionFraction(object):
    r'''
    Compute the satisfaction fraction of a temporal formula.
    if \rho_{\phi}_lb > 0, then the satisfaction fraction is 1; ( All predicate-space trajectories satisfy the specification)
    if \rho_{\phi}_ub < 0, then the satisfaction fraction is 0; ( No predicate-space trajectory satisfies the specification)
    if \rho_{\phi}_lb <= 0 <= \rho_{\phi}_ub, then the satisfaction fraction is in (0, 1) and can be computed by sampling or optimization. ( Some predicate-space trajectories satisfy the specification, and some do not)
    '''


if __name__ == "__main__":

    EVOT = _EVENTUALLY_(0, 1)
    AWOT = _ALWAYS_(0, 1)
    lb  = _LeftBracket_()
    rb  = _RightBracket_()
    AND = _AND_()
    OR  = _OR_()
    UNTIL = _UNTIL_(2, 3)
    P1 = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.05]))
    P2 = AtomicPredicate(np.array([0.0, 1.0]), np.array([0.02]))  

    spec1= Formula([EVOT,lb,P1,OR,lb,AWOT,P2,rb,rb])
    spec1.print()
    
    spec2= Formula([P1,UNTIL,P2])

    Ex_F = getExpandedFormula(spec2)
    print("Expanded formula:")
    Ex_F.print()

    # Dy_F = Ex_F.expand()
    # print("Dynamic formula:")
    # Dy_F.print()
