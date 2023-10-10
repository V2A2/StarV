"""
Test ProbStar methods
Last update: 11/25/2022
Author: Dung Tran
"""

from StarV.set.probstar import ProbStar
import numpy as np
import glpk
import polytope as pc


class Test(object):
    """
       Testing ProbStar class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_constructor(self):

        self.n_tests = self.n_tests + 1

        # len(agrs) = 4
        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        print('Testing ProbStar Constructor...')
        try:
            ProbStar(mu, Sig, pred_lb, pred_ub)
        except Exception:
            print("Fail in constructing probstar object with \
            len(args)= {}".format(4))
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_str(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        print('\nTesting __str__ method...')

        try:
            print(S.__str__())
        except Exception:
            print("Test Fail :( !")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_estimateRange(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        print('\nTesting estimateMin method...')

        try:
            min_val, max_val = S.estimateRange(0)
            print('MinValue = {}, true_val = {}'.format(min_val, pred_lb[0]))
            print('MaxValue = {}, true_val = {}'.format(max_val, pred_ub[0]))
            assert min_val == pred_lb[0] and \
                max_val == pred_ub[0], 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_estimateRanges(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        print('\nTesting estimateRanges method...')

        try:
            lb, ub = S.estimateRanges()
            print('lb = {}, true_lb = {}'.format(lb, pred_lb))
            print('ub = {}, true_ub = {}'.format(ub, pred_ub))
            
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")


    def test_getMin(self):

        self.n_tests = self.n_tests + 3

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        min_val = S.getMin(0, 'gurobi')
        print('\nTesting getMin method using gurobi...')

        try:
            min_val = S.getMin(0, 'gurobi')
            print('MinValue = {}, true_val = {}'.format(min_val, pred_lb[0]))
            assert min_val == pred_lb[0], 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        print('\nTesting getMin method using glpk...')

        try:
            min_val = S.getMin(0, 'glpk')
            print('MinValue = {}, true_val = {}'.format(min_val, pred_lb[0]))
            assert min_val == pred_lb[0], 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        print('\nTesting getMin method using linprog...')

        try:
            min_val = S.getMin(0, 'linprog')
            print('MinValue = {}, true_val = {}'.format(min_val, pred_lb[0]))
            assert min_val == pred_lb[0], 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_glpk(self):
        """
        test glpk example from here: \
        https://pyglpk.readthedocs.io/en/latest/examples.html

        """

        self.n_tests = self.n_tests + 1
        lp = glpk.LPX()
        lp.name = 'test_glpk'
        lp.obj.maximize = True
        lp.rows.add(3)
        for r in lp.rows:
            r.name = chr(ord('p') + r.index)
        lp.rows[0].bounds = None, 100.0
        lp.rows[1].bounds = None, 600.0
        lp.rows[2].bounds = None, 300.0
        lp.cols.add(3)
        for c in lp.cols:
            c.name = 'x%d' % c.index
            c.bounds = 0.0, None

        f = np.array([10.0, 6.0, 4.0])
        lp.obj[:] = f.tolist()
        # lp.obj[:] = [10.0, 6.0, 4.0]
        A = np.array([[1.0, 1.0, 1.0], [10.0, 4.0, 5.0], [2.0, 2.0, 6.0]])
        B = A.reshape(9,)
        a = B.tolist()
        lp.matrix = a
        # lp.matrix = [1.0, 1.0, 1.0,
        #             10.0, 4.0, 5.0,
        #             2.0, 2.0, 6.0]

        print('\nTest glpk...')

        try:
            lp.simplex()
            print('Z = {}'.format(lp.obj.value))
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_getMax(self):

        self.n_tests = self.n_tests + 3

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        print('\nTesting getMax method using gurobi...')
        try:
            max_val = S.getMax(0, 'gurobi')
            print('MaxValue = {}, true_val = {}'.format(max_val, pred_ub[0]))
            assert max_val == pred_ub[0], 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        print('\nTesting getMax method using glpk...')

        try:
            max_val = S.getMax(0, 'glpk')
            print('MaxValue = {}, true_val = {}'.format(max_val, pred_ub[0]))
            assert max_val == pred_ub[0], 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        print('\nTesting getMax method using linprog...')

        try:
            max_val = S.getMax(0, 'linprog')
            print('MaxValue = {}, true_val = {}'.format(max_val, pred_ub[0]))
            assert max_val == pred_ub[0], 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_affineMap(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)

        A = np.random.rand(2, 3)
        b = np.random.rand(2,)

        print('\nTesting affine mapping method...')

        try:
            S1 = S.affineMap(A, b)
            print('original probstar:')
            print(S.__str__())
            print('new probstar:')
            print(S1.__str__())
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_minKowskiSum(self):

        self.n_tests = self.n_tests + 1
        X = ProbStar.rand(3)
        Y = ProbStar.rand(3)
        Z = X.minKowskiSum(Y)
        try:
            print('\nTesting minKowskiSum method...')
            X.__str__()
            Z = X.minKowskiSum(Y)
            Z.__str__()
            
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_isEmptySet(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = ProbStar(mu, Sig, pred_lb, pred_ub)

        try:
            print('\nTesting isEmptySet method...')
            res = S.isEmptySet('gurobi')
            print('res: {}'.format(res))
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_updatePredicateRanges(self):

        self.n_tests = self.n_tests + 2

        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        newC = np.array([-0.25, 1.0])
        newd = np.array([0.25])

        print('\nTesting updatePredicateRanges method 1...')

        try:
            new_pred_lb, new_pred_ub = ProbStar.updatePredicateRanges(newC,
                                                                      newd,
                                                                      pred_lb,
                                                                      pred_ub)
            print('new_pred_lb: {}, new_pred_ub: {}'.format(new_pred_lb,
                                                            new_pred_ub))
            assert new_pred_ub[1] == 0.5, 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        newC = np.array([0.25, -1.0])
        newd = np.array([-0.25])

        print('\nTesting updatePredicateRanges method 2...')

        try:
            new_pred_lb, new_pred_ub = ProbStar.updatePredicateRanges(newC,
                                                                      newd,
                                                                      pred_lb,
                                                                      pred_ub)
            print('new_pred_lb: {}, new_pred_ub: {}'.format(new_pred_lb,
                                                            new_pred_ub))
            assert new_pred_lb[1] == 0.0, 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_addConstraint(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(2,)
        Sig = np.eye(2)
        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        C = np.array([-0.25, 1.0])
        d = np.array([0.25])

        print('\nTesting addConstraint method...')

        try:
            print('Before adding new constraint')
            S.__str__()
            S.addConstraint(C, d)
            print('After adding new constraint')
            S.__str__()
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_addMultipleConstraints(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(2,)
        Sig = np.eye(2)
        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        C = np.array([[-0.25, 1.0], [0., -1.0]])
        d = np.array([0.25, 0.])

        print('\nTesting addMultipleConstraints method...')
        S.addMultipleConstraints(C, d)

        try:
            print('Before adding new constraints')
            S.__str__()
            S.addMultipleConstraints(C, d)
            print('After adding new constraints')
            S.__str__()
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")


    def test_rand(self):

        self.n_tests = self.n_tests + 1
        print('\nTesting rand method...')
        try:
            S = ProbStar.rand(3)
            S.__str__()
        except Exception:
            print('Test Fails')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_estimateProbability(self):

        self.n_tests = self.n_tests + 1
        print('\nTesting estimateProbability method...')
        
        try:
            S = ProbStar.rand(3)
            prob = S.estimateProbability()
            print('prob = {}'.format(prob))
            V = np.random.rand(5, 4)
            C = np.random.rand(4, 3)
            d = np.random.rand(4,)
            S2 = ProbStar(V, C, d, S.mu, S.Sig, S.pred_lb, S.pred_ub)
            prob2 = S2.estimateProbability()
            print('prob2 = {}'.format(prob2))
            print('isemptySet = {}'.format(S2.isEmptySet()))
            p = pc.Polytope(C, d)
            print('isemptySet = {}'.format(pc.is_empty(p)))
        except Exception:
            print('Test Fails')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_sampling(self):

        self.n_tests = self.n_tests + 1

        try:

            S = ProbStar.rand(2, 3)
            samples = S.sampling(3)
            lb, ub = S.getRanges()
            print('lb = {}'.format(lb))
            print('ub = {}'.format(ub))
            print('Samples = {}'.format(samples))
            lb = lb.reshape(2,1)
            ub = ub.reshape(2,1)
            print('Samples - lb = {}'.format(samples - lb))
            print('Samples - ub = {}'.format(samples - ub))
            

        except Exception:
            print('Test Fails')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


if __name__ == "__main__":

    test_probstar = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_probstar.test_constructor()
    test_probstar.test_str()
    test_probstar.test_estimateRange()
    test_probstar.test_glpk()
    test_probstar.test_getMin()
    test_probstar.test_getMax()
    test_probstar.test_affineMap()
    test_probstar.test_minKowskiSum()
    test_probstar.test_isEmptySet()
    test_probstar.test_updatePredicateRanges()
    test_probstar.test_addConstraint()
    test_probstar.test_rand()
    test_probstar.test_estimateRanges()
    test_probstar.test_estimateProbability()
    test_probstar.test_addMultipleConstraints()
    test_probstar.test_sampling()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_probstar.n_fails,
                            test_probstar.n_tests - test_probstar.n_fails,
                            test_probstar.n_tests))
