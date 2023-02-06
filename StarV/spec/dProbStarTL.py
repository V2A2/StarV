"""

Probabilistic Star Temporal Logic Specification Language in discrete-time domain


Author: Dung Tran
Date: 12/2/2022

==================================================================================
DESCRIPTION:
-----------
* This specification language enables quantitative monitoring and verification 
of temporal behaviors of Autonomous CPS and Learning-enabled CPS

* This replies on Probabilistic Star Reachable Set

* Unlike Signal Temporal Logic (STL) or Linear Temporal Logic, ProbStarTL defines
on a reachable set called reachable set signal, not the traditional signal or trace

* ProbStarTL has similar syntax as STL

* ProbStarTL has quantitative semantics that allows answering the probability of 
a system satisfying a property.

* Unlike STL, Quantitative semantics of ProbStarTL is defined based on probability 
==================================================================================

dProbStarTL SYNTAX
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

dProbStarTL BOOLEAN SEMANTICS
----------------------------

Defined on BOUNDED TIME REACHABLE SET X = [X[1], X[2], ..... X[T]]

The satisfaction (|=) of a formula p by a reachable set X at time step 1 <= t <= T


* (X, t) |= AP <=> exist x in X[t], Ax <= b <=> X[t] AND AP is feasible

* (X, t) |= p AND AP <=> X[t] AND p AND AP is feasible (different from STL semantics)

* (X, t) |= NOT p <=> X[t] AND NOT p is feasiable  

* (X, t) |= p U_[a, b] w <=> exist t' in [t + a, t + b] such that (X, t') |= w AND for all t'' in [t, t'], (X, t'') |= p

* Eventually: ET_[a, b] p = T U_[a, b] p

  (X, t) |= ET_[a, b] w <=> exist t' in [t + a, t + b] such that (x, t') |= w

* Always: AW_[a, b] p = NOT (ET_[a, b] NOT p)

  (X, t) |= AW_[a, b] <=> for all t' in [t + a, t + b] such that (X, t') |= w

==================================================================================

dProbStarTL QUANTITATIVE SEMANTICS 

"""

import numpy as np
from StarV.set.probstar import ProbStar
import copy


class AtomicPredicate(object):

    'P:= Ax <= b'

    def __init__(self, A, b):

        assert isinstance(A, np.ndarray), 'error: A should be a numpy array'
        assert isinstance(b, np.ndarray), 'error: b should be a numpy array'
        assert len(b.shape) == 1, 'error: b should be 1D numpy array'
        assert len(A.shape) == 1, 'error: A should be 1D numpy array'
        assert b.shape[0] == 1, 'error: b should be a scalar'

        self.A = A
        self.b = b
        self.type = 'AtomicPredicate'

    def print(self):

        str = ''
        for i in range(0, self.A.shape[0]):
            s = '{}*x[{}]'.format(self.A[i], i)
            if i < self.A.shape[0] -1:
                if self.A[i + 1] >= 0:
                    s = s + '+'
            str = str + s

        str = str + ' <= '
        str = str + '{}'.format(self.b[0])

        return str

    def render(self, probstar_sig):
        'obtain a concrete set of constraints for satisfaction of an atomic predicate on multiple reach set'

        assert isinstance(probstar_sig, list), 'error: input should be a list of probstars'
        C = None
        d = None
        nVarMax = 0
        nVarMaxID = 0
        for i in range(0, len(probstar_sig)):
            R = copy.deepcopy(probstar_sig[i])
            R.addConstraint(self.A, self.b)
            C1 = R.C
            d1 = R.d
            if C is None:
                C = copy.deepcopy(C1)
                d = copy.deepcopy(d1)
                nVarMax = R.nVars
            else:
                n, m = C.shape  # m is the number of predicate variables in the current constraint
                n1, m1 = C1.shape # m1 is the number of predicate variables in the new constraint
                
                if m < m1:  # usually happen
                    dC = np.zeros((n, m1-m))
                    C = np.append(C, dC, axis=1)
                if m1 < m: # not usually happen
                    dC = np.zeros((n, m-m1))
                    C1 = np.append(C1, dC, axis=1)
                    
                C = np.concatenate((C, C1), axis=0)
                d = np.concatenate((d, d1))

                if R.nVars > nVarMax:
                    nVarMax = R.nVars
                    nVarMaxID = i
                

        # combine all constraints into a single set of constraints
        
        _, m = C.shape
        V = np.eye(m,m)
        center = np.zeros((m, 1))
        V = np.append(center, V, axis=1)
        S = ProbStar(V, C, d, probstar_sig[nVarMaxID].mu, probstar_sig[nVarMaxID].Sig,\
                     probstar_sig[nVarMaxID].pred_lb, probstar_sig[nVarMaxID].pred_ub)
        # S is a probstar that contains all points for satisfaction
        
        return S

    @staticmethod
    def rand(nVars):
        'generate random predicate'

        A = np.random.rand(nVars)
        b = np.random.rand(1)
        P = AtomicPredicate(A, b)

        return P    
    
    
class _AND_(object):

    'AND'

    def __init__(self):
        
        self.type = 'BooleanOperator'
        self.operator = 'AND '

    def print(self):

        return self.operator

class _OR_(object):

    def __init__(self):

        self.type = 'BooleanOperator'
        self.operator = 'OR '

    def print(self):

        return self.operator

class _NOT_(object):

    def __init__(self):

        self.type = 'BooleanOperator'
        self.operator = 'NOT '

    def print(self):

        return self.operator

class _IMPLY_(object):

    def __init__(self):

        self.type = 'BooleanOperator'
        self.operator = ' --> '

    def print(self):

        return self.operator

class _ALWAYS_(object):

    def __init__(self, start_time, end_time=None):

        assert start_time >= 0, 'error: invalid start_time'
        self.start_time = start_time
        
        if end_time is not None:
            assert end_time > start_time, 'error: invalid end_time'
            self.end_time = end_time

        if end_time is not None:
            self.operator = 'ALWAYS_[{},{}] '.format(start_time, end_time)
        else:
            self.operator = 'ALWAYS_[{}, inf] '.format(start_time)
        self.type = 'TemporalOperator'

    def print(self):

        return self.operator            
        
            
class _EVENTUALLY_(object):

    def __init__(self, start_time, end_time=None):

        assert start_time >= 0, 'error: invalid start_time'
        self.start_time = start_time
        
        if end_time is not None:
            assert end_time > start_time, 'error: invalid end_time'
            self.end_time = end_time

        else:
            self.end_time = float('inf')
        
        self.operator = 'EVENTUALLY_[{},{}] '.format(start_time, end_time)
        self.type = 'TemporalOperator'

    def print(self):

        return self.operator

class _UNTIL_(object):

    pass

class _LeftBracket_(object):

    def __init__(self):

        self.type = 'LeftBracket'
        self.operator = '('

    def print(self):

        return self.operator

class _RightBracket_(object):

    def __init__(self):

        self.type = 'RightBracket'
        self.operator = ')'

    def print(self):

        return self.operator
        

class Formula(object):
    """
      Specification is made by Predicate & OPERATORS & Brackets
      A list of objects including Predicate, OPERATORS and Brackets
    """
    
    def __init__(self, formula):

        assert isinstance(formula, list), 'error: invalid spec, should be a list'

        nL = 0
        nR = 0

        nANDs = 0
        nORs = 0
        nIMPLYs = 0
        
        for obj in formula:
            
            if not isinstance(obj, AtomicPredicate) and not isinstance(obj, _AND_) and not isinstance(obj, _OR_) \
               and not isinstance(obj, _ALWAYS_) and not isinstance(obj, _NOT_) and not isinstance(obj, _UNTIL_) \
               and not isinstance(obj, _EVENTUALLY_) and not isinstance(obj, _LeftBracket_) and not isinstance(obj, _RightBracket_):

                raise RuntimeError('Invalid Spec, unknown object')

            if isinstance(obj, _LeftBracket_):
                nL = nL + 1
            if isinstance(obj, _RightBracket_):
                nR = nR +1

            if obj.type == 'BooleanOperator':
                if obj.operator == 'AND ':
                    nANDs = nANDs + 1
                elif obj.operator == 'OR ':
                    nORs = nORs + 1
                elif obj.operator == 'IMPLY ':
                    nIMPLYs = nIMPLYs + 1
                else:
                    raise RuntimeError('Unknown boolean operator')
    
        if nL != nR:
            raise RuntimeError('Unbalance number of brackets: nL = {} # nR = {}'.format(nL, nR))

        # formula type

        if isinstance(formula[0], _ALWAYS_):
            if (nORs == 0 and nANDs > 0) or (nORs == 0 and nANDs == 0):
                self.formula_type = 'ConjunctiveAlways'
            elif nORs > 0 and nANDs == 0:
                self.formula_type = 'DisjunctiveAlways'
            elif nORs*nANDs > 0:
                self.formula_type = 'MixingAlways'
            else:
                self.formula_type = 'UnknownAlways'

        elif isinstance(formula[0], _EVENTUALLY_):

            if (nORs == 0 and nANDs > 0) or (nORs == 0 and nANDs == 0):
                self.formula_type = 'ConjunctiveEventually'
            elif nORs > 0 and nANDs == 0:
                self.formula_type = 'DisjunctiveEventually'
            elif nORs*nANDs > 0:
                self.formula_type = 'MixingEventually'
            else:
                self.formula_type = 'UnknownEventually'

        else:
            self.formula_type = 'UnknownType'


        self.formula = formula
        self.length = len(formula)
        

    def print(self):
        'Print the formula'

        str = ''
        for obj in self.formula:
            str = str + '\n' + obj.print()


        print('Formula: ')
        print('Formula type: {}'.format(self.formula_type))
        print('Formula length: {}'.format(self.length))
        print(str)
        
        return str

    def render(self, probstar_signal):
        'render a formula on a probstar_signal, return a concrete probstar with constraints for statisfaction'
    
        assert isinstance(probstar_signal, list), 'error: probstar_signal should be a list of probstars'
        for probstar in probstar_signal:
            assert isinstance(probstar, ProbStar), 'error: probstar_signal should contain ProbStar objects'

        # a temporal formula should be started with a temporal operator (ALWAYS or EVENTUALLY), followed by a bracket
        
        # BASIC FORMULA TYPES:
        # 1) Always formula:
        #        Conjunction: ALWAYS_[t1, t2 ] (AP1 AND AP2 AND ... APn)
        #        Disjunction: ALWAYS_[t1, t2] (AP1 OR AP2)
        #        Conditional: ALWAYS_[t1, t2] (AP1 IMPLY AP2)
        # 
        # 2) Eventually formula:
        #        Conjunction: EVENTUALLY_[t1, t2] (AP1 AND AP2 AND ...APn)
        #        Disjunction: EVENTUALLY_[t1, t2] (AP1 OR AP2 OR ...APn)
        #        Conditional: EVENTUALLY_[t1, t2] (AP1 IMPLY AP2)
        # 
        # COMPOSITE FORMULA:
        #

        S = None
        if self.formula_type == 'ConjunctiveAlways':
            S = renderConjunctiveAlwaysFormula(self, probstar_signal)
        else:
            raise RuntimeError('Not support rendering {} formula yet'.format(self.formula_type))

        return S

def renderConjunctiveAlwaysFormula(f, probstar_signal):
    'rendering conjective always formula on a reachable set signal'

    assert isinstance(f, Formula), 'error: f should be a Formula object'
    assert f.formula_type == 'ConjunctiveAlways', 'error: formula is not a conjunctive always type'

    S = []
    for item in f.formula[1: f.length]:
        if isinstance(item, AtomicPredicate):
            if f.formula[0].end_time is None:
                S1 = item.render(probstar_signal)
            else:
                required_length = f.formula[0].end_time - f.formula[0].start_time + 1
                if len(probstar_signal) < required_length:
                    raise RuntimeError('probstar signal has insufficient length to evaluate the formula')
                else:
                    S1 = item.render(probstar_signal[f.formula[0].start_time:f.formula[0].end_time + 1])

            S.append(S1)

    # combining all stars into a single star
    S = combineProbStars(S)

    return S


def combineProbStars(probstar_sig):
    'combine multiple probstars with the same distribution for the predicates into a single one'

    assert isinstance(probstar_sig, list), 'error: input should be a list of probstars'

    
    C = None
    d = None
    nVarMax = 0
    nVarMaxID = 0
    for i in range(0, len(probstar_sig)):
        R = copy.deepcopy(probstar_sig[i])
        C1 = R.C
        d1 = R.d
        if C is None:
            C = copy.deepcopy(C1)
            d = copy.deepcopy(d1)
            nVarMax = R.nVars
        else:
            n, m = C.shape  # m is the number of predicate variables in the current constraint
            n1, m1 = C1.shape # m1 is the number of predicate variables in the new constraint

            if m < m1:  # usually happen
                dC = np.zeros((n, m1-m))
                C = np.append(C, dC, axis=1)
            if m1 < m: # not usually happen
                dC = np.zeros((n, m-m1))
                C1 = np.append(C1, dC, axis=1)

            C = np.concatenate((C, C1), axis=0)
            d = np.concatenate((d, d1))

            if R.nVars > nVarMax:
                nVarMax = R.nVars
                nVarMaxID = i


    # combine all constraints into a single set of constraints

    _, m = C.shape
    V = np.eye(m,m)
    center = np.zeros((m, 1))
    V = np.append(center, V, axis=1)
    S = ProbStar(V, C, d, probstar_sig[nVarMaxID].mu, probstar_sig[nVarMaxID].Sig,\
                 probstar_sig[nVarMaxID].pred_lb, probstar_sig[nVarMaxID].pred_ub)
    # S is a probstar that contains all points for satisfaction

    return S
