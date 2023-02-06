"""
Test for dProbStar module
Dung Tran
1/11/2023
"""

import numpy as np
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_
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

    def test_AtomicPredicate_render(self):

        self.n_tests = self.n_tests + 1

        try:
            
            A = np.array([3., 2.])
            b = np.array([2.])
            P = AtomicPredicate(A,b)
            
            R1 = ProbStar().rand(2, 3)
            R2 = ProbStar().rand(2, 4)
            R = [R1, R2]
            S = P.render(R)
            S.__str__()
            S.printConstraints()
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


    def test_Always_render(self):

        self.n_tests = self.n_tests + 1

        # example spec1: always_[0, 2](3x[0] + 2x[1] <= 2 )
        A = np.array([3., 2.])
        b = np.array([2.])
        P1 = AtomicPredicate(A,b)
        op2 = _AND_()
        P2 = AtomicPredicate.rand(2)
        op1 = _ALWAYS_(1, 2)
        lb1 = _LeftBracket_()
        rb1 = _RightBracket_()

        f = Formula([op1, lb1, P1, op2, P2, rb1])
        f.print()
        R1 = ProbStar().rand(2,3)
        R2 = ProbStar().rand(2,4)
        R3 = ProbStar().rand(2,5)
        probstar_sig = [R1, R2, R3]
        print('probstar_sig = {}'.format(probstar_sig))
        # S = op1.render(preds, probstar_sig)
        S = f.render(probstar_sig)

        print('Satisfied ProbStar: ')
        S.__str__()

        try:
            
            # example spec1: always_[0, 2](3x[0] + 2x[1] <= 2 )
            A = np.array([3., 2.])
            b = np.array([2.])
            P1 = AtomicPredicate(A,b)
            op2 = _AND_()
            P2 = AtomicPredicate.rand(2)
            op1 = _ALWAYS_(1, 2)
            lb1 = _LeftBracket_()
            rb1 = _RightBracket_()

            f = Formula([op1, lb1, P1, op2, P2, rb1])
            f.print()
            R1 = ProbStar().rand(2,3)
            R2 = ProbStar().rand(2,4)
            R3 = ProbStar().rand(2,5)
            probstar_sig = [R1, R2, R3]
            print('probstar_sig = {}'.format(probstar_sig))
            # S = op1.render(preds, probstar_sig)
            S = f.render(probstar_sig)

            print('Satisfied ProbStar: ')
            S.__str__()

        except Exception as e:
            print('Test Fail!')
            print(e)
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


if __name__ == "__main__":

    test_dProbStar = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    #test_dProbStar.test_Formula()
    #test_dProbStar.test_AtomicPredicate_render()
    #test_dProbStar.test_AtomicPredicate_rand()
    test_dProbStar.test_Always_render()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing dProbStarTL module: fails: {}, successfull: {}, \
    total tests: {}'.format(test_dProbStar.n_fails,
                            test_dProbStar.n_tests - test_dProbStar.n_fails,
                            test_dProbStar.n_tests))
