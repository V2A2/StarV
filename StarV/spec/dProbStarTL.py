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

Probabilistic Star Temporal Logic Specification Language in discrete-time domain


Author: Dung Tran
Date: 12/2/2022
Update: 7/4/2024
      - 11/17/2025: add composed predicate (Predicate Class and its methods, update AtomicPredicate methods)

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
from itertools import combinations
import copy
import polytope as pc


class AtomicPredicate(object):

    'P:= Ax <= b'

    def __init__(self, A, b, t=None):

        assert isinstance(A, np.ndarray), 'error: A should be a numpy array'
        assert isinstance(b, np.ndarray), 'error: b should be a numpy array'
        assert len(b.shape) == 1, 'error: b should be 1D numpy array'
        assert len(A.shape) == 1, 'error: A should be 1D numpy array'
        assert b.shape[0] == 1, 'error: b should be a scalar'

        self.A = A
        self.b = b
        self.type = 'AtomicPredicate'
        self.t = t

    def at_time(self, t):

        assert t >= 0, 'error: invalid time step, t should be >= 0'
        
        return AtomicPredicate(self.A, self.b, t)

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

    def print_info(self):

        print('{} * x[t={}] <= {}\n'.format(self.A, self.t, self.b))


    @staticmethod
    def rand(nVars, t=None):
        'generate random predicate'

        A = np.random.rand(nVars)
        b = np.random.rand(nVars )
        P = AtomicPredicate(A, b)
        if t is not None:
            P.at_time(t)

        return P
    

class Predicate(object):
    'Composed predicate object: P: Ax <= b'

    # Dung Tran: date: 11/17/2025

    def __init__(self, A=None, b=None, t=None):

        if A is not None and B is not None:
            assert isinstance(A, np.ndarray), 'error: A should be a numpy array'
            assert len(A.shape) == 2, 'error: A should be 2D numpy array'
            assert isinstance(b, np.ndarray), 'error: b should be a numpy array'
            assert len(b.shape) == 1, 'error: b should be 1D numpy array'
            assert (A.shape[1] == b.shape[0]), 'error: inconsistency between predicate matrix and predicate vector'

        
        self.A = A
        self.b = b
        self.type = 'ComposedPredicate'
        self.t = t

    def at_time(self, t):

        assert t >= 0, 'error: invalid time step, t should be >= 0'
        
        return Predicate(self.A, self.b, t)

    def print_info(self):

        print('{} * x[t={}] <= {}\n'.format(self.A, self.t, self.b))

    @staticmethod
    def rand(nVars, t=None):
        'generate random predicate'

        A = np.random.rand(nVars, nVars)
        b = np.random.rand(nVars)
        P = Predicate(A, b)
        if t is not None:
            P.at_time(t)

        return P

    def compose(self, other):
        'compose with other predicate or an atomic predicate'

        P = []
        if isinstance(other, Predicate):
            if other.A is None:
                P = self
            else:
                if self.A is None:
                    P = other
                else:
                    if self.A.shape[1] != other.A.shape[1]:
                        raise ValueError('inconsistent dimension (number of variables) between two predicates')
                    elif self.t != other.t:
                        raise ValueError('inconsistent time between two predicates')
                    else:
                        newA = np.vstack((self.A, other.A))
                        newb = np.hstack((self.b, other.b))
                        P = Predicate(newA, newb, self.t)
        elif isinstance(other, AtomicPredicate):
            if self.A is None:
                newA = other.A.reshape(1, other.A.shape[0])
                P = Predicate(newA, other.b, other.t)
            else:
                if self.t != other.t:
                    raise ValueError('inconsistent time between two predicates')
                else:
                    A1 = other.A.reshape(1, other.A.shape[0])
                    if A1.shape[1] != self.A.shape[1]:
                        raise ValueError('inconsistent dimension (number of variables) between two predicates')
                    else:
                        newA = np.vstack((self.A, other.A))
                        newb = np.hstack((self.b, other.b))
                        P = Predicate(newA, newb, self.t)

        else:
            raise TypeError('Unknown input type, should be an AtomicPredicate or a Predicate Object')

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


class CDNF(object):
    """
     This is the computable disjuctive normal form of a formula

     Dung Tran: 1/24/2024

     Update: 7/8/2024

    """

    def __init__(self, constraints, base_probstar):

        # constraints should be in a list constraints = [H1 or H2 or ... Hn]  # after realization
        # base_probstar shares common constraints among all probstar in probstar signal
        assert isinstance(constraints, list), 'error: ADNF is not a Formula object'
        assert isinstance(base_probstar, ProbStar), 'error: base_probstar should be a ProbStar object'

        self.constraints = constraints
        self.base_probstar = base_probstar
        self.length = len(constraints)

        
    def print(self):
        'print cdnf formula'

        print('=======CDNF formula information======:')
        print('***Length of CDNF = {}\n'.format(self.length))
        print('***Constraints:')
        for i in range(0, self.length):
            print('=========P{}=========:'.format(i))
            const = self.constraints[i]
            print('C{} = {}'.format(i, const[0]))
            print('d{} = {}'.format(i, const[1]))

        print('\n***Shared base probstar:')
        self.base_probstar.__str__()

    def getTraceProbability(self):
        'get probability of a trace, it is the base probstar probability'

        return self.base_probstar.estimateProbability()

    def estimateProbability(self, combination_ids):
        'estimate probability of a combination of constraints'

        assert isinstance(combination_ids, tuple), 'error: combination_ids should be a tuple'

        for i in range(0, len(combination_ids)):
            id = combination_ids[i]
            const = self.constraints[id]
            if i==0:
                C = const[0]
                d = const[1]
            else:
                C = np.vstack((C, const[0]))
                d = np.concatenate((d, const[1]))

        if len(self.base_probstar.C) != 0:
            C = np.vstack((C, self.base_probstar.C))
            d = np.concatenate((d, self.base_probstar.d))

        if len(C.shape) == 1:
            C = C.reshape(1, self.base_probstar.nVars)

        S = ProbStar(self.base_probstar.V, C, d, self.base_probstar.mu, self.base_probstar.Sig, self.base_probstar.pred_lb, self.base_probstar.pred_ub)

        prob = S.estimateProbability()

        return prob

    def toProbStarSignals(self, probstar_signal):
        'return a set of probstar signals (for plotting)'

        probstar_signals = []
        for i in range(0, self.length):
            const = self.constraints[i]
            C = const[0]
            d = const[1]

            if len(C.shape) == 1:
                C = C.reshape(1, self.base_probstar.nVars)

            if len(self.base_probstar.C) != 0:
                
                newC = np.vstack((C, self.base_probstar.C))
                newd = np.concatenate((d, self.base_probstar.d))

            else:
                newC = C
                newd = d

            probstar_sig_out = []
            for PS in probstar_signal:
                newPS = ProbStar(PS.V, newC, newd, PS.mu, PS.Sig, self.base_probstar.pred_lb, self.base_probstar.pred_ub)

                if not newPS.isEmptySet():
                    probstar_sig_out.append(newPS)

            probstar_signals.append(probstar_sig_out)

        return probstar_signals

    def shorten(self, n):
        'shorten CDNF to new CDNF with length n, i.e., n stars with highest probability'

        # Dung Tran, 7/4/2024

        if n >= self.length:
            shorten_CDNF = self
            remained_CDNF = []
        else:
            
            prob = []
            for i in range(0, self.length):
                   prob.append(self.estimateProbability((i,)))
            a_prob = np.array(prob)
            max_ids = np.argsort(-a_prob)   # indexes of (decending) probability of probstars

            compute_ids = max_ids[0:n]
            remain_ids = max_ids[n:len(max_ids)]


            new_constraints = []
            remained_constraints = []

            for i in range(0, len(compute_ids)):
                new_constraints.append(self.constraints[compute_ids[i]])

            for i in range(0, len(remain_ids)):
                remained_constraints.append(self.constraints[remain_ids[i]])

            shorten_CDNF = CDNF(new_constraints, self.base_probstar)
            remained_CDNF = CDNF(remained_constraints, self.base_probstar)

        return shorten_CDNF, remained_CDNF    

            
    def getSATProbabilityBounds(self):
        'compute the upper/lower bounds probability of satisfaction'

        # upper bound probability = sum of all probability of all probstars
        # lower bound probability =

        SAT = []
        p_base_probstar = self.base_probstar.estimateProbability()
        for i in range(0, self.length):
            SAT.append(self.estimateProbability((i,)))
        p_SAT_MIN = max(SAT)
        p_SAT_MAX = min(sum(SAT), p_base_probstar)

        return p_SAT_MAX, p_SAT_MIN

    def getExactSATProbability(self):
        'compute exact probability of an CDNF'

        SAT_prob = 0
        if self.length > 11:
            raise RuntimeError('Cannot compute the exact probability of an CDNF with length > 11, please shorten the CDNF')
        else:
            N = range(0, self.length)
            for i in range(0, self.length):
                print('i = {}/{}'.format(i, self.length))
                SAT1 = 0.0
                # get combinations
                comb = combinations(N, i+1)   # get all combinations
                for j in list(comb):
                    # compute probability of sub-combincation , i.e., Pj[1] AND Pj[2]
                    prob = (-1)**i * self.estimateProbability(j)
                    SAT1 = SAT1 + prob

                SAT_prob = SAT_prob + SAT1

        return SAT_prob


    def getApproxSATProbability(self, n):
        'compute approximate bound of probability of Satisfaction by split an CDNF into two CDNF of length n and CDNF.length - n'

        p_SAT_MAX = 0
        p_SAT_MIN = 0

        if n < 1 or n > 11:
            raise RuntimeError('Invalid number of probstars for splitting the CDNF, should be 1 <= n <= 11')

        else:
            if n >= self.length:
                print('Compute exact probability of statisfaction...')
                p_SAT_MIN = self.getExactSATProbability()
                p_SAT_MAX = p_SAT_MIN
            else:
                print('Compute approximate probability of statisfaction...')
                shorten_CDNF, remained_CDNF = self.shorten(n)
                p_shorten = shorten_CDNF.getExactSATProbability()
                p_remain_max, p_remain_min = remained_CDNF.getSATProbabilityBounds()

                p_SAT_MIN = p_shorten + p_remain_min
                p_SAT_MAX = p_shorten + p_remain_max

        return p_SAT_MAX, p_SAT_MIN
            

            

class Formula(object):
    """
      Specification is made by Predicate & OPERATORS & Brackets
      A list of objects including Predicate, OPERATORS and Brackets
    
      This is abstract disjunctive normal form (ADNF)
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
        print(self)

    def __str__(self):
        'Print the formula'

        str = ''
        for obj in self.formula:
            str = str + '\n' + obj.print()

        print('Formula: ')
        print('Formula type: {}'.format(self.formula_type))
        print('Formula length: {}'.format(self.length))
        print(str)

        return '\n'

    def getDynamicFormula(self):
        'automatically generate dynamic formula: a abstract disjunctive normal form for verification'

        # This works like a parser
        # Dung Tran: 06/2023



        # an example EV_[0,5](P1 AND EV_[2,4] (P2 AND P3)), P1, P2, P3 are atomic predicates
        # formula F = [op1 lb1 P1 op2 op3 lb2 P2 op4 P3 rb2 rb1]

        # Automatically generate the abstract constraints of this formula

        # algorithm: idea: inner loop to outer loop algorithm, start from inner most loop and expand to outer most loop
        # Inner most loop  (loop 0): (P2 AND P3)
        # Inner loop 1: EV_[2,4] (P2 AND P3)
        # Inner loop 2: P1 AND E_[2,4] (P2 AND P3)
        # Outer loop EV_[0,5] (P1 AND EV_[2,4] (P2 AND P3))

        # We start with an empty dynamic formula F0 = [],
        # We expand the dynamic formula F0 from the right to the left of the fomula

        # Step 0: F0 = []
        # Step 1: F0 = P3 AND P2
        # Step 2: F0 = EV_[2,4] F0 = (P3 AND P2)_t=2 OR (P3 AND P2)_t=3 OR (P3 AND P2)_t=4

        # Step 3: F0 = P1 AND F0
        # Step 4: F0 = EV_[0,5] F0

        lb_idxes, rb_idxes = self.getLoopIds()
        inner_f = self.getInnerMostLoopFormula()
        # expand the formula from inner loop to outer loop

        DF = DynamicFormula(F=[])
        DF.subFormula_expand(inner_f)
        

        if len(lb_idxes) >= 2:
        
            for i in range(len(lb_idxes), 1, -1):
                # search for next outer loop
                start_id = lb_idxes[i-2]
                end_id = lb_idxes[i-1]
                sub_f = self.getSubFormula(start_id+1, end_id)
                DF.subFormula_expand(sub_f)

            if start_id > 0:
                sub_f = self.getSubFormula(0, start_id)
                DF.subFormula_expand(sub_f)

        if len(lb_idxes) == 1:
            sub_f = self.getSubFormula(0, lb_idxes[0])
            DF.subFormula_expand(sub_f)
            
        return DF
        
                     
    def getSubFormula(self, start_id, end_id):
        return self.formula[start_id:end_id]

    def getInnerMostLoopFormula(self):
        'get the inner most loop subformula'

        lb_idxes, rb_idxes = self.getLoopIds()
        nloops = len(lb_idxes)
        if nloops == 0:
            F = self.formula
        else:
            lb_id = lb_idxes[nloops-1]
            rb_id = rb_idxes[0]
            F = self.formula[lb_id +1:rb_id]

        return F
        
    def getLoopIds(self):
        'get all loop ids'

        lb_idxes = []  # indexes of left brackets
        rb_idxes = []  # indexes of right brackets

        # we use left bracket and right brackets' indexes to determine inner loops
        
        for id in range(0,self.length):
            if isinstance(self.formula[id], _LeftBracket_):
                lb_idxes.append(id)
            if isinstance(self.formula[id], _RightBracket_):
                rb_idxes.append(id)

        if len(lb_idxes) != len(rb_idxes):
            raise RuntimeError('error: syntax error, number of left brackets is not equal to number of right brackets')

        return lb_idxes, rb_idxes


def combineProbStars(probstar_sig):
    'combine multiple probstars with the same distribution for the predicates into a single one'

    # Update 8/13/2023 by Dung Tran

    # In a probstar signal S = [S0, S1, ..., St] we may have several cases:
    
    # case 1) S0, S1, S2, ...St have the same number of predicates and their pred_lb, pred_ub are the same
    #    --> This happen when we perform reachability analysis for linear system x[t+1] = Ax[t] + b
    #    --> In this case, we use S0's predicate bounds for the combined probstar

    # case 2) S0, S1, S2, ... St have the same number of predicates, but their pred_lb, pred_ub are different
    #    --> This may hapen when we do exact reachability analysis Neural Network Control System (NNCS)
    #    --> In this case the S0's bound is the largest bound, we will use it for the combined probstar
    
    # case 3) S0, S1, S2 .. St have different number of predicates
    #    --> This may happen when we do approximate analysis for NNCS or RNN
    #    --> In this case, we use the bound for the Si that has the maximum number of predicates
    
  
    

    assert isinstance(probstar_sig, list), 'error: input should be a list of probstars'
        
    if len(probstar_sig) == 1:
        return probstar_sig[0]
    else:
        
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
                    dC = np.zeros((n1, m-m1))
                    C1 = np.append(C1, dC, axis=1)

                C = np.concatenate((C, C1), axis=0)
                d = np.concatenate((d, d1))

                if R.nVars > nVarMax:
                    nVarMax = R.nVars
                    nVarMaxID = i


        # combine all constraints into a single set of constraints, we use the lower bound and upper bound vectors of
        # the probstar that has the maximum number of predicate variables

        _, m = C.shape
        V = np.eye(m,m)
        center = np.zeros((m, 1))
        V = np.append(center, V, axis=1)

        S = ProbStar(V, C, d, probstar_sig[nVarMaxID].mu, probstar_sig[nVarMaxID].Sig,\
                     probstar_sig[nVarMaxID].pred_lb, probstar_sig[nVarMaxID].pred_ub)
        # S is a probstar that contains all points for satisfaction
        # nVarMaxID is the index of the probstar contains all predicate variables

        return S

def combineProbStars_with_indexes(probstar_sig, idxs):
    'combine ProbStars in a probstar signal with specific indexes'


    assert isinstance(idxs, tuple), 'error: indexes should be in a list'
    S0 = []
    for id in idxs:
        S0.append(probstar_sig[id])
    S = combineProbStars(S0)
    
    return S
   
        
class DynamicFormula(object):
    '''
    Dynamic formula F = [[P0] [P1] [P2] [] ... [Pn]] list of n lists

    Dynamic formula F is in disjunctive normal form (DNF): i.e., P0 OR P1 OR P2 ... OR Pn

    [Pi] : [C1 C2 ... Ck], Ci: is a timed atomic predicate (TAP) (a linear constraint)

    Timed atomic predicate: is a predicate at a time step t

    Author: Dung Tran, Date 4/3/2022, last update: 23/6/2023
    '''

    def __init__(self, F=[]):

        assert isinstance(F, list), 'error: input should be a list'
        self.F = F
        self.length = len(F)

    def print(self):
        'print the dynamic formula'

        if self.length == 0:
            print('{}'.format(self.F))

        else:
            if self.length == 1:
                print('Timed-Abstract Dynamic Formula: F = [[P0]], where:')
            elif self.length == 2:
                print('Timed-Abstract Dynamic Formula: F = [[P0] [P1]], where')
            else:
                print('Timed-Abstract Dynamic Formula: F = [[P0] OR ...OR [P{}]]'.format(self.length - 1))
    
            for i in range(0, self.length):
                print('\nP{}:\n'.format(i))
                Pi = self.F[i]
                for j in range(0, len(Pi)):
                    Pi[j].print_info()
                

    def OR_expand(self, P):
        'expand the formula with OR operation'

        assert isinstance(P, AtomicPredicate), 'error: input should be a atomic predicate'

        self.F.append([P])
        self.length = self.length + 1

    def OR_concatenate(self, df):
        'or expand with other dynamics formula'

        # [[P0] [P1]] OR [[P2] [P3]] = [[P0] [P1] [P2] [P3]]

        
        assert isinstance(df, DynamicFormula), 'error: input should be a dynamcicFormula object'

        if df.length >=1:
            self.F = self.F + df.F
            self.length = len(self.F)

    def OR_concatenate_multiple_formulas(self, dfs):

        if len(dfs) > 0:
            for df in dfs:
                self.OR_concatenate(df)

    def AND_concatenate(self, df):
        'AND expansion with other dynamic formula'

        # [[P0] [P1]] AND [[P2] [P3]] = [[P0 P2] [P1 P2] [P0 P3] [P1 P3]]

        assert isinstance(df, DynamicFormula), 'error: input should be a dynamic formula object'

        if len(self.F) == 0:
            self.F = df.F
        else:
            F1 = []
            if df.length >= 1:
                F1 = [x + y for x in self.F for y in df.F]
            self.F = F1
        self.length = len(self.F)

    def AND_concatenate_multiple_formulas(self, dfs):

        if len(dfs) > 0:
            for df in dfs:
                self.AND_concatenate(df)
        
    def EVENTUALLY_expand(self, P, t_start, t_final):
        'only work when self is an empty dynamicformula'

        if self.length > 0:
            raise RuntimeError ('Eventually_expand operation only work with an empty dynamic formula ')

        assert isinstance(P, AtomicPredicate), 'error: input predicate should be an atomic predicate'
        assert t_start >= 0, 'error: t_start should be >= 0'
        assert t_start < t_final, 'error: t_start should be smaller than t_final'

        for t in range(t_start, t_final + 1):
            self.OR_expand(P.at_time(t))

    def EVENTUALLY_selfExpand(self, t_start, t_final):
        'work when self is a none-empty dynamicformula'

        if self.length == 0:
            raise RuntimeError ('Eventually_selfExpand operation only work with a none-empty dynamic formula')

        assert t_start >= 0, 'error: t_start should be >= 0'
        assert t_start < t_final, 'error: t_start should be smaller than t_final'

        # Example: Eventually_[2,3] [[P0] [P1]] = [[P0] [P1]]_t=2 OR [[P0] [P1]]_t=3
        df = copy.deepcopy(self)
        newDF = DynamicFormula(F=[])       
        for t in range(t_start, t_final + 1):
            df1 = copy.deepcopy(df)
            df1.update_time(t)
            newDF.OR_concatenate(df1)

        self.F = newDF.F
        self.length = len(self.F)
        

    def ALWAYS_expand(self, P, t_start, t_final):
        
        if self.length > 0:
            raise RuntimeError ('Always_expand operation only work with an empty dynamic formula ')

        assert isinstance(P, AtomicPredicate), 'error: input predicate should be an atomic predicate'
        assert t_start >= 0, 'error: t_start should be >= 0'
        assert t_start < t_final, 'error: t_start should be smaller than t_final'

        for t in range(t_start, t_final + 1):
            self.AND_expand(P.at_time(t))

    def ALWAYS_selfExpand(self, t_start, t_final):

        'work when self is a none-empty dynamicformula'

        if self.length == 0:
            raise RuntimeError ('Eventually_selfExpand operation only work with a none-empty dynamic formula')

        assert t_start >= 0, 'error: t_start should be >= 0'
        assert t_start < t_final, 'error: t_start should be smaller than t_final'

        # Example: ALWAYS_[2,3] [[P0] [P1]] = [[P0] [P1]]_t=2 AND [[P0] [P1]]_t=3

        df = copy.deepcopy(self)
        newDF = DynamicFormula(F=[])       
        for t in range(t_start, t_final + 1):
            df1 = copy.deepcopy(df)
            df1.update_time(t)
            newDF.AND_concatenate(df1)

        self.F = newDF.F
        self.length = len(self.F)

    def AND_expand(self, P):
        'expand the formula with AND operation'

        # [[P0] [P1]] AND P = [[P0 P] [P1 P]] 
        
        assert isinstance(P, AtomicPredicate), 'error: input should be a atomic predicate'

        if self.length == 0:
            self.F.append([P])
        else:
            for i in range(0, self.length):
                self.F[i].append(P)
        self.length = len(self.F)
    
    def update_time(self, time_offset):
        'update time information of all predicates in a dynamic formula'

        # Predicate P: P.t = P.t + time_offset

        # F = [[P0] [P1]]

        for i in range(0, len(self.F)):
            C = copy.deepcopy(self.F[i])
            for j in range(0, len(C)):
                C[j].t = C[j].t + time_offset          
            self.F[i] = C

    def set_time(self, t):
        'set time information of all predicates in a dynamic formula'

        # Predicate P: P.t = t
        # F = [[P0] [P1]]

        for i in range(0, len(self.F)):
            C = copy.deepcopy(self.F[i])
            for j in range(0, len(C)):
                C[j].t = t
                
            self.F[i] = C

        
    def subFormula_expand(self, sub_formula):
        'generate subdynamic formula from inner subformula'

        assert isinstance(sub_formula, list), 'error: sub_formula should be a list'

        # we go from the right to the left
        # ex: sub_formula = [P2 AND P3]

        i = len(sub_formula)-1
        while i >= 0:
            item = sub_formula[i]
            if isinstance(item, AtomicPredicate):
                self.AND_expand(item.at_time(0))
            elif isinstance(item, _AND_):
                i = i-1
                item = sub_formula[i]
                if isinstance(item, AtomicPredicate):
                    self.AND_expand(item.at_time(0))
                else:
                    raise RuntimeError('Invalid subformula')
            elif isinstance(item, _OR_):
                i = i-1
                item = sub_formula[i]
                if isinstance(item, AtomicPredicate):
                    self.OR_expand(item.at_time(0))
                else:
                    raise RuntimeError('Ivalid subformula')
            elif isinstance(item, _EVENTUALLY_):
                self.EVENTUALLY_selfExpand(item.start_time, item.end_time)
            elif isinstance(item, _ALWAYS_):
                self.ALWAYS_selfExpand(item.start_time, item.end_time)
            else:
                raise RuntimeError('Unknown item')
                
            i = i-1

    def realization(self, probstar_sig):
        'realization of abstract, timed dynamic formula on a probstar signal'     
            
        assert isinstance(probstar_sig, list), 'error: probstar signal should be a list'
        T = len(probstar_sig)
        constraints = []
        base_probstar = copy.deepcopy(probstar_sig[0])
        nVars = base_probstar.nVars
        for P in self.F:
            H = []
            C = None
            d = None
            for Pi in P:
                if Pi.t >= T:
                    C = None
                    d = None
                    break
                else:  
                    if C is None:
                        d = Pi.b - np.matmul(Pi.A, probstar_sig[Pi.t].V[:,0])
                        C = np.matmul(Pi.A, probstar_sig[Pi.t].V[:, 1:nVars+1])
                    else:
                        d1 = Pi.b - np.matmul(Pi.A, probstar_sig[Pi.t].V[:,0])
                        C1 = np.matmul(Pi.A, probstar_sig[Pi.t].V[:, 1:nVars+1])

                        C = np.vstack((C, C1))
                        d = np.concatenate((d, d1))

            if C is not None:
                if len(base_probstar.C) != 0:
                    C1 = np.vstack((C, base_probstar.C))
                    d1 = np.concatenate((d, base_probstar.d))
                else:
                    C1 = C
                    d1 = d

                if len(C1.shape) == 1:
                    C1 = C1.reshape(1,nVars)
               
                S1 = ProbStar(base_probstar.V, C1, d1, base_probstar.mu, \
                              base_probstar.Sig, base_probstar.pred_lb, base_probstar.pred_ub)

                if not S1.isEmptySet():
                    H = [C, d]
                    constraints.append(H)

        cdnf = CDNF(constraints, base_probstar)

        return cdnf


    def evaluate(self, probstar_sig):
        'evaluate the satisfaction of the abtract-timed dyanmic formula on a probstar signal'

        # last update: Dung Tran, 7/4/2024

        print('Realizing Abstract DNF specification on a ProbStar Signal...')
        cdnf = self.realization(probstar_sig)
        print('Length of Computable DNF = {}'.format(cdnf.length))

        p_trace = cdnf.base_probstar.estimateProbability()  # probability of the probstar signal
        SAT = []
        p_SAT_MIN = 0.0
        p_SAT_MAX = 0.0

        if cdnf.length != 0:
            for i in range(0, cdnf.length):
               SAT.append(cdnf.estimateProbability((i,)))
            if cdnf.length > 11:
                print('*****WARNING*****: CDNF (len = {}) is too large for exact verification'.format(cdnf.length))
                print('We ignore this CDNF, return the estimate probability uperbound')
                p_SAT_MIN = max(SAT)
                p_SAT_MAX = max(p_SAT_MIN, p_trace) # this eleminates the numerical issue in estimating probability
            else:
                N = range(0, cdnf.length)
                print('Computing exact probability of satisfaction...')
                for i in range(0, cdnf.length):
                    print('i = {}/{}'.format(i, cdnf.length))
                    SAT1 = 0.0
                    # get combinations
                    comb = combinations(N, i+1)   # get all combinations
                    for j in list(comb):
                        # compute probability of sub-combincation , i.e., Pj[1] AND Pj[2]
                        prob = (-1)**i * cdnf.estimateProbability(j)
                        SAT1 = SAT1 + prob
                    p_SAT_MAX = p_SAT_MAX + SAT1
                p_SAT_MIN = p_SAT_MAX 


        return SAT, p_SAT_MAX, p_SAT_MIN, cdnf.length


    def evaluate_for_full_analysis(self, probstar_sig):
        'evaluate the satisfaction of the abtract-timed dyanmic formula on a probstar signal'
        # last update: Dung Tran, 7/4/2024

        print('Realizing Abstract DNF specification on a ProbStar Signal...')
        cdnf = self.realization(probstar_sig)
        print('Length of Computable DNF = {}'.format(cdnf.length))

        p_trace = cdnf.base_probstar.estimateProbability()  # probability of the probstar signal
        SAT = []
        p_SAT_MIN = 0.0
        p_SAT_MAX = 0.0
        p_ig = 0.0
        cdnf_SAT = []
        cdnf_IG = []
        sat_trace = []
        if cdnf.length != 0:
            for i in range(0, cdnf.length):
               SAT.append(cdnf.estimateProbability((i,)))
               
            if cdnf.length > 11:
                print('*****WARNING*****: CDNF (len = {}) is too large for exact verification'.format(cdnf.length))
                print('We ignore this CDNF, return the estimate probability uperbound')
                p_SAT_MIN = max(SAT)
                p_SAT_MAX = max(p_SAT_MIN, p_trace) # this eleminates the numerical issue in estimating probability
                p_ig = p_trace
                cdnf_IG = cdnf
            else:
                cdnf_SAT = cdnf
                N = range(0, cdnf.length)
                print('Computing exact probability of satisfaction...')
                for i in range(0, cdnf.length):
                    print('i = {}/{}'.format(i, cdnf.length))
                    SAT1 = 0.0
                    # get combinations
                    comb = combinations(N, i+1)   # get all combinations
                    for j in list(comb):
                        # compute probability of sub-combincation , i.e., Pj[1] AND Pj[2]
                        prob = (-1)**i * cdnf.estimateProbability(j)
                        SAT1 = SAT1 + prob
                    p_SAT_MAX = p_SAT_MAX + SAT1
                p_SAT_MIN = p_SAT_MAX 
                sat_trace = [probstar_sig, cdnf]


        return p_SAT_MAX, p_SAT_MIN, p_ig, cdnf_SAT, cdnf_IG, sat_trace
        
        

    
