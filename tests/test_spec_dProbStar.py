"""
Test for dProbStar module
Dung Tran
1/11/2023
"""

import numpy as np
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_
from StarV.spec.dProbStarTL import DynamicFormula, Predicate
from StarV.set.probstar import ProbStar

class Test(object):
    """
    Testing module net class and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_Formula(self):

        self.n_tests = self.n_tests + 1

        try:
            # example spec1: always_[0, 2](3x[0] + 2x[1] <= 2 )
            A = np.array([3., 2.])
            b = np.array([2.])
            P1 = AtomicPredicate(A,b)
            op1 = _ALWAYS_(0, 2)
            lb1 = _LeftBracket_()
            rb1 = _RightBracket_()

            f = [op1, lb1, P1, rb1]

            spec1 = Formula(f)

            spec1.print()

            # spec2: eventually_[0,2] (3x[1] - x[2] <= 1)
            A = np.array([3., -1.])
            b = np.array([1.])

            P1 = AtomicPredicate(A, b)
            op1 = _EVENTUALLY_(0, 2)
            lb1 = _LeftBracket_()
            rb1 = _RightBracket_()

            f = [op1, lb1, P1, rb1]

            spec2 = Formula(f)

            spec2.print()

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

   
    def test_AtomicPredicate_rand(self):

        self.n_tests = self.n_tests + 1

        try:
            P = AtomicPredicate.rand(3)
            print(P.print())
        except Exception:
            self.n_fails = self.n_fails + 1
            print('Test Fail!')
        else:
            print('Test Successfull!')

    def test_Predicate_rand(self):

        self.n_tests = self.n_tests + 1
       
        try:
             P = Predicate.rand(3)
             P.print_info()
        except Exception:
            self.n_fails = self.n_fails + 1
            print('Test Fail!')
        else:
            print('Test Successfull!')

    def test_Predicate_compose(self):

        self.n_tests = self.n_tests + 1

        
        try:
            P1 = Predicate.rand(3)
            A1 = AtomicPredicate.rand(3)
            P3 = Predicate.rand(3)
            P3 = P3.at_time(2)
            P1 = P1.at_time(2)
            A1 = A1.at_time(2)

            A1.print_info()
            P1.print_info()
            P3.print_info()
        
            P2 = P1.compose(A1)
            P2.print_info()

            P4 = P1.compose(P3)
            P4.print_info()
        
        except Exception:
            self.n_fails = self.n_fails + 1
            print('Test Fail!')
        else:
            print('Test Successfull!')

    def test_getDynamicFormula(self):
        'test automatic algorithm to generate dynamic formula for verification'

        self.n_tests = self.n_tests + 1
        
        # EV_[0,5] (P1 AND EV_[1,3] (P2 AND P3))
        P1 = AtomicPredicate.rand(2)
        P2 = AtomicPredicate.rand(2)
        P3 = AtomicPredicate.rand(2)
        
        EV03 = _EVENTUALLY_(0,3)
        EV12 = _EVENTUALLY_(1,2)
        AND = _AND_()
        lb = _LeftBracket_()
        rb = _RightBracket_()
        AW03 = _ALWAYS_(0,3)

        
        #f1 = Formula([AW03, P1])
        #f1.print()
        #F1 = f1.getDynamicFormula()
        #F1.print()

        #f2 = Formula([EV03, P1])
        #f2.print()
        #F2 = f2.getDynamicFormula()
        #F2.print()
        
        f3 = Formula([EV03, lb, P1, AND, EV12, lb, P2, AND, P3, rb, rb])
        f3.print()
        F3 = f3.getDynamicFormula()
        F3.print()

        #f4 = Formula([AW03, lb, P1, AND, EV12, lb, P2, AND, P3, rb, rb])
        #f4.print()
        #F4 = f4.getDynamicFormula()
        #F4.print()

        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])

        R1 = ProbStar().rand(2,2, pred_lb, pred_ub)
        R2 = ProbStar().rand(2,2, pred_lb, pred_ub)
        R3 = ProbStar().rand(2,2, pred_lb, pred_ub)
        R4 = ProbStar().rand(2,2, pred_lb, pred_ub)
        probstar_sig = [R1, R2, R3, R4]
        # print('probstar_sig = {}'.format(probstar_sig))
        print('probstar_sig = [')
        for ps in probstar_sig:
            print('  {}'.format(ps))
        print(']')

        # cdnf = F3.realization2(probstar_sig)
        cdnf = F3.realization(probstar_sig)
        cdnf.print()

        comb_ids = (0, 3)
        print('probability of cdnf({}) = {}'.format(comb_ids, cdnf.estimateProbability(comb_ids)))

        #SAT, SAT_MIN, SAT_EXACT = F3.evaluate(probstar_sig)
        #print('SAT = {}'.format(SAT))
        #print('SAT-MIN = {}'.format(SAT_MIN))
        #print('SAT-EXACT = {}'.format(SAT_EXACT))
        
        
    def test_DynamicFormula_evaluate(self):

        self.n_tests = self.n_tests + 1
        
        # use a small example in the slide to test the accuaracy of the semantics

        mu = np.array([0.0])
        sig = np.array([[1.0]])
        pred_lb = np.array([-np.inf])
        pred_ub = np.array([np.inf])
        
        # Dynamical system
        # x[k+1] = 2*x[k] + 1.0

        X0 = ProbStar(mu, sig, pred_lb, pred_ub)  # initial set
        A = np.array([[2.0]])
        b = np.array([1.0])
        X1 = X0.affineMap(A, b)
        X2 = X1.affineMap(A, b)
        X3 = X2.affineMap(A, b)
        X4 = X3.affineMap(A, b)
        X5 = X4.affineMap(A, b)
        
        # predicate: Eventually_[1,3] x >= 0.5
        P = AtomicPredicate(np.array([-1]), np.array([-0.5]))
        

        # predicates
        # P1: x-0.5 >= 0
        # P2: -x + 0.4 >= 0
        # P3: -x + 0.8 >= 0

        P1 = AtomicPredicate(np.array([-1.0]), np.array([-0.5]))
        P2 = AtomicPredicate(np.array([1.]), np.array([0.4]))
        P3 = AtomicPredicate(np.array([1.]), np.array([0.8]))

        # temporal operators
        AW13 = _ALWAYS_(1,3)
        EV13 = _EVENTUALLY_(1,3)
        EV05 = _EVENTUALLY_(0,5)
        AND = _AND_()
        lb = _LeftBracket_()
        rb = _RightBracket_()

        # formula
        f1 = Formula([EV13, P1])
        f2 = Formula([AW13, P1])
        f3 = Formula([EV05, lb, P1, AND, EV13, lb, P3, rb, rb])
        
        #f1.print()

        # timed-abstract dynamic formula
        F1 = f1.getDynamicFormula()
        F1.print()

        # reachable set
        #X0.__str__()
        #X1.__str__()
        #X2.__str__()
        #X3.__str__()
        
        probstar_sig = [X0, X1, X2, X3, X4, X5]

        res = F1.realization(probstar_sig)
        # print('res_length = {}'.format(len(res)))
        print('res = ')
        res.print()
        # print('res = {}'.format(res))

        SAT, SAT_MIN, SAT_EXACT, _ = F1.evaluate(probstar_sig)
        print('SAT = {}'.format(SAT))
        print('SAT-MIN = {}'.format(SAT_MIN))
        print('SAT-EXACT = {}'.format(SAT_EXACT))
        
        
    def test_DynamicFormula(self):
        'test DynamicFormula object'
        self.n_tests = self.n_tests + 1

        # EV_[0,5] (A AND EV_[1,3] B)
        # EV_[0,5] (A AND EV_[1,3] B AND EV_[2, 4] C)

        # what is the benefit of moving backward if you have bounded time
        # How to optimize the derivation? 

        A = AtomicPredicate.rand(2)
        B = AtomicPredicate.rand(2)

        # we need 6 empty subformula to construct the final one
        F0 = DynamicFormula([]) # t=0
        F1 = DynamicFormula([]) # t=1
        F2 = DynamicFormula([]) # t=2
        F3 = DynamicFormula([]) # t=3
        F4 = DynamicFormula([]) # t=4
        F5 = DynamicFormula([]) # t=5

        F = DynamicFormula([]) # the formula EV_[0,5] (A AND EV_[1,3] B)

        # t = 0
        F0.AND_expand(A.at_time(0))
        F0.print()
        F01 = DynamicFormula([]) # EV_[1,3] B
        F01.EVENTUALLY_expand(B, 1, 3)
        F01.print()
        F0.AND_concatenate(F01)
        F0.print()

        # t = 1
        F1.AND_expand(A.at_time(1))
        F11 = DynamicFormula([])
        F11.EVENTUALLY_expand(B, 2, 4)
        F1.AND_concatenate(F11)
        #F1.print()
        
        # t = 2
        F2.AND_expand(A.at_time(2))
        F21 = DynamicFormula([])
        F21.EVENTUALLY_expand(B, 3, 5)
        F2.AND_concatenate(F21)
        #F2.print()

        # t=3
        F3.AND_expand(A.at_time(3))
        F31 = DynamicFormula([])
        F31.EVENTUALLY_expand(B, 4, 5)
        F3.AND_concatenate(F31)
        #F3.print()

        # t=4
        F4.AND_expand(A.at_time(4))
        F41 = DynamicFormula([])
        F41.AND_expand(B.at_time(5))
        F4.AND_concatenate(F41)
        #F4.print()

        # t = 5, F5 = []

        F.OR_concatenate_multiple_formulas([F0,F1,F2,F3,F4])
        F.print()

        
        #try:
        #    F = DynamicFormula([])  # create an empty formula
        #    P = TimedAtomicPredicate.rand(nVars=2, t_limit=5)
        #    F.AND_expand(P)
        #except Exception:
        #    print('Test Fail')
        #    self.n_fails = self.n_fails + 1
        #else:
        #    print('Test Successfull')
            
        

if __name__ == "__main__":

    test_dProbStar = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    #test_dProbStar.test_Formula()
    #test_dProbStar.test_AtomicPredicate_rand()
    #test_dProbStar.test_Predicate_rand()
    test_dProbStar.test_Predicate_compose()
    #test_dProbStar.test_DynamicFormula()
    #test_dProbStar.test_getDynamicFormula()
    #test_dProbStar.test_DynamicFormula_evaluate()
   
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing dProbStarTL module: fails: {}, successfull: {}, \
    total tests: {}'.format(test_dProbStar.n_fails,
                            test_dProbStar.n_tests - test_dProbStar.n_fails,
                            test_dProbStar.n_tests))
