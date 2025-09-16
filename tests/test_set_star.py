"""
Test Star methods
Last update: 11/25/2022
Author: Dung Tran
"""

from StarV.set.star import Star
import numpy as np
import glpk
import polytope as pc


class Test(object):
    """
       Testing Star class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_constructor(self):

        self.n_tests = self.n_tests + 1

        # len(agrs) = 2
        # pred_lb = np.random.rand(3,)
        # pred_ub = pred_lb + 0.2
        pred_lb = np.array([-1, -1, -1])
        pred_ub = pred_lb + 0.2
        print('Testing Star Constructor...')
        try:
            Star(pred_lb, pred_ub)
        except Exception:
            print("Fail in constructing Star object with \
            len(args)= {}".format(2))
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_str(self):

        self.n_tests = self.n_tests + 1

        # pred_lb = np.random.rand(3,)
        # pred_ub = pred_lb + 0.2
        pred_lb = np.array([-1, -1, -1])
        pred_ub = pred_lb + 0.2
        S = Star(pred_lb, pred_ub)
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

        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = Star(pred_lb, pred_ub)
        print('\nTesting estimateMin method...')

        try:
            min_val, max_val = S.estimateRange(0)
            print('MinValue = {}, true_val = {}'.format(min_val, pred_lb[0]))
            print('MaxValue = {}, true_val = {}'.format(max_val, pred_ub[0]))

            # Yuntao: Need to compare values with tolerance
            # Default: np.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
            assert np.isclose(min_val, pred_lb[0]) and \
                np.isclose(max_val, pred_ub[0]), 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_estimateRanges(self):

        self.n_tests = self.n_tests + 1

        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = Star(pred_lb, pred_ub)
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

        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = Star(pred_lb, pred_ub)
        # min_val = S.getMin(0, 'gurobi')
        print('\nTesting getMin method using gurobi...')

        try:
            min_val = S.getMin(0, 'gurobi')
            print('MinValue = {}, true_val = {}'.format(min_val, pred_lb[0]))
            assert np.isclose(min_val, pred_lb[0]), 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        print('\nTesting getMin method using glpk...')

        try:
            min_val = S.getMin(0, 'glpk')
            print('MinValue = {}, true_val = {}'.format(min_val, pred_lb[0]))
            assert np.isclose(min_val, pred_lb[0]), 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        print('\nTesting getMin method using linprog...')

        try:
            min_val = S.getMin(0, 'linprog')
            print('MinValue = {}, true_val = {}'.format(min_val, pred_lb[0]))
            assert np.isclose(min_val, pred_lb[0]), 'error: wrong results'
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

        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = Star(pred_lb, pred_ub)
        print('\nTesting getMax method using gurobi...')
        try:
            max_val = S.getMax(0, 'gurobi')
            print('MaxValue = {}, true_val = {}'.format(max_val, pred_ub[0]))
            assert np.isclose(max_val, pred_ub[0]), 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        print('\nTesting getMax method using glpk...')

        try:
            max_val = S.getMax(0, 'glpk')
            print('MaxValue = {}, true_val = {}'.format(max_val, pred_ub[0]))
            assert np.isclose(max_val, pred_ub[0]), 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

        print('\nTesting getMax method using linprog...')

        try:
            max_val = S.getMax(0, 'linprog')
            print('MaxValue = {}, true_val = {}'.format(max_val, pred_ub[0]))
            assert np.isclose(max_val, pred_ub[0]), 'error: wrong results'
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_affineMap(self):

        self.n_tests = self.n_tests + 1

        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = Star(pred_lb, pred_ub)

        A = np.random.rand(2, 3)
        b = np.random.rand(2,)

        print('\nTesting affine mapping method...')

        try:
            S1 = S.affineMap(A, b)
            print('original Star:')
            print(S.__str__())
            print('new Star:')
            print(S1.__str__())
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_minKowskiSum(self):

        self.n_tests = self.n_tests + 1
        X = Star.rand(3)
        Y = Star.rand(3)
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

        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        S = Star(pred_lb, pred_ub)

        try:
            print('\nTesting isEmptySet method...')
            res = S.isEmptySet()
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
            new_pred_lb, new_pred_ub = Star.updatePredicateRanges(newC,
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
            new_pred_lb, new_pred_ub = Star.updatePredicateRanges(newC,
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

        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        S = Star(pred_lb, pred_ub)
        C = np.array([-0.25, 1.0])
        d = np.array([0.25])

        print('\nTesting addConstraint method...')

        try:
            S1 = S.addConstraint(C, d)
            S2 = S.addConstraint(C, d, tighten_bounds=False)
            print('Before adding new constraint')
            S.__str__()
            print('After adding new constraint')
            S1.__str__()
            print('After adding new constraint without tightening bounds')
            S2.__str__()
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_addMultipleConstraints(self):

        self.n_tests = self.n_tests + 1

        pred_lb = np.array([-1.0, -1.0])
        pred_ub = np.array([1.0, 1.0])
        S = Star(pred_lb, pred_ub)
        C = np.array([[-0.25, 1.0], [0., -1.0]])
        d = np.array([0.25, 0.])

        print('\nTesting addMultipleConstraints method...')

        try:
            S1 = S.addMultipleConstraints(C, d)
            S2 = S.addMultipleConstraints(C, d, tighten_bounds=False)
            print('Before adding new constraints')
            S.__str__()
            print('After adding new constraints')
            S1.__str__()
            print('After adding new constraints without tightening bounds')
            S2.__str__()
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")


    def test_rand(self):

        self.n_tests = self.n_tests + 1
        print('\nTesting rand method...')
        try:
            S = Star.rand(3)
            S.__str__()
        except Exception:
            print('Test Fails')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_sampling(self):

        self.n_tests = self.n_tests + 1

        try:

            S = Star.rand(2, 3)
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


    def test_concatenate_with_vector(self):

        self.n_tests = self.n_tests + 1

        try:
            S = Star.rand(2, 3)
            v = np.random.rand(3,)
            S1 = S.concatenate_with_vector(v)
            v2 = []
            S2 = S.concatenate_with_vector(v2)
            print('\nBefore Concatenation: ')
            S.__str__()
            print('\nAfter Concatenation:')
            S1.__str__()
            S2.__str__()

        except Exception:
            print('Test Fails')
            self.n_fails = self.n_fails + 1

        else:
            print('Test Sucessfull!')

    def test_resetRow(self):

        self.n_tests = self.n_tests + 1
        print('\nTesting resetRow method...')

        try:
            S = Star.rand(2)
            S1 = S.resetRow(0)
            print('\nBefore resetRow: ')
            S.__str__()
            print('\nAfter resetRow: ')
            S1.__str__()

        except Exception:
            print('Test Fails')
            self.n_fails = self.n_fails + 1

        else:
            print('Test Sucessfull!')

    def test_resetRows(self):

        self.n_tests = self.n_tests + 1
        print('\nTesting resetRows method...')

        try:
            S = Star.rand(2)
            rows = [0, 1]
            S1 = S.resetRows(rows)
            print('\nBefore resetRows: ')
            S.__str__()
            print('\nAfter resetRows: ')
            S1.__str__()

        except Exception:
            print('Test Fails')
            self.n_fails = self.n_fails + 1

        else:
            print('Test Sucessfull!')

    def test_resetRowWithFactor(self):
        
        self.n_tests = self.n_tests + 1
        print('\nTesting resetRowWithFactor method...')

        try:
            S = Star.rand(2)
            factor = 0.01
            S1 = S.resetRowWithFactor(0, factor)
            print('\nBefore resetRowWithFactor: ')
            S.__str__()
            print('\nAfter resetRowWithFactor: ')
            S1.__str__()

        except Exception:
            print('Test Fails')
            self.n_fails = self.n_fails + 1

        else:
            print('Test Sucessfull!')

    def test_resetRowsWithFactor(self):
        
        self.n_tests = self.n_tests + 1
        print('\nTesting resetRowsWithFactor method...')

        try:
            S = Star.rand(2)
            rows = [0, 1]
            factor = 0.01
            S1 = S.resetRowsWithFactor(rows, factor)
            print('\nBefore resetRowsWithFactor: ')
            S.__str__()
            print('\nAfter resetRowsWithFactor: ')
            S1.__str__()
        except Exception:
            print('Test Fails')
            self.n_fails = self.n_fails + 1

        else:
            print('Test Sucessfull!')

    def test_resetRowWithUpdatedCenter(self):
        
        self.n_tests = self.n_tests + 1
        print('\nTesting resetRowWithUpdatedCenter method...')

        try:
            S = Star.rand(2)
            newCenter = 1.0
            S1 = S.resetRowWithUpdatedCenter(0, newCenter)
            print('\nBefore resetRowWithUpdatedCenter: ')
            S.__str__()
            print('\nAfter resetRowWithUpdatedCenter: ')
            S1.__str__()

        except Exception:
            print('Test Fails')
            self.n_fails = self.n_fails + 1

        else:
            print('Test Sucessfull!')

    def test_resetRowsWithUpdatedCenter(self):
        
        self.n_tests = self.n_tests + 1
        print('\nTesting resetRowsWithUpdatedCenter method...')

        try:
            S = Star.rand(2)
            newCenters = 1.0
            S1 = S.resetRowsWithUpdatedCenter([0, 1], newCenters)
            print('\nBefore resetRowsWithUpdatedCenter: ')
            S.__str__()
            print('\nAfter resetRowsWithUpdatedCenter: ')
            S1.__str__()

        except Exception:
            print('Test Fails')
            self.n_fails = self.n_fails + 1

        else:
            print('Test Sucessfull!')

if __name__ == "__main__":

    test_Star = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    # test_Star.test_constructor()
    # test_Star.test_str()
    # test_Star.test_estimateRange()
    # test_Star.test_estimateRanges()
    # test_Star.test_glpk()
    # test_Star.test_getMin()
    # test_Star.test_getMax()
    # test_Star.test_affineMap()
    # test_Star.test_minKowskiSum()
    # test_Star.test_isEmptySet()
    # test_Star.test_updatePredicateRanges()
    test_Star.test_addConstraint()
    # test_Star.test_rand()
    test_Star.test_addMultipleConstraints()
    # test_Star.test_sampling()
    # test_Star.test_concatenate_with_vector()
    # test_Star.test_resetRow()
    # test_Star.test_resetRows()
    # test_Star.test_resetRowWithFactor()
    # test_Star.test_resetRowsWithFactor()
    # test_Star.test_resetRowWithUpdatedCenter()
    # test_Star.test_resetRowsWithUpdatedCenter()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing Star Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_Star.n_fails,
                            test_Star.n_tests - test_Star.n_fails,
                            test_Star.n_tests))
