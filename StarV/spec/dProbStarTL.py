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
from itertools import combinations
import copy


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
    def rand(nVars, t=None):
        'generate random predicate'

        A = np.random.rand(nVars)
        b = np.random.rand(1)
        P = AtomicPredicate(A, b)
        if t is not None:
            P.at_time(t)

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
        T = len(probstar_sig)  # time
        for i in range(0, T):
            S = probstar_sig[i]  # a star set
            if not isinstance(S, ProbStar):
                raise RuntimeError('item {} is not a probstar'.format(i))

        res = []
        for P in self.F:
            resi = []
            i  = 0
            for Pi in P:
                if Pi.t >= T:
                    resi = []
                    break
                else:
                    X = copy.deepcopy(probstar_sig[Pi.t])
                    X.addConstraintWithoutUpdateBounds(Pi.A, Pi.b)
                    resi.append(X)
                
            if len(resi) != 0:
                S1 = combineProbStars(resi)
                res.append(S1)

        return res

    def evaluate(self, probstar_sig, semantics='exact'):
        'evaluate the satisfaction of the abtract-timed dyanmic formula on a probstar signal'

        res = self.realization(probstar_sig)

        SAT = []
        for i in range(0, len(res)):
            SAT.append(res[i].estimateProbability())
        SAT_MIN = max(SAT)

        SAT_EXACT = 0.0
        N = range(0, len(res))
        for i in range(0, len(res)):
            SAT1 = 0.0
            # get combinations
            comb = combinations(N, i+1)   # get all combinations
            for j in list(comb):
                # compute probability of sub-combincation , i.e., Pj[1] AND Pj[2]
                S = combineProbStars_with_indexes(res, j)
                prob = (-1)**i * S.estimateProbability()
                SAT1 = SAT1 + prob
            SAT_EXACT = SAT_EXACT + SAT1
            

        # SAT_EXACT should be always >= SAT_MIN, however, \
        # due to the uncertainties in probability estimation SAT_EXACT may be slightly smaller than SAT_MIN
        SAT_EXACT = max(SAT_MIN, SAT_EXACT) 
            
        return SAT, SAT_MIN, SAT_EXACT
        
        

    
