"""
Test of dlode module
Author: Dung Tran
Date: 11/24/2022
"""

from StarV.plant.dlode import DLODE
import numpy as np
from StarV.set.star import Star
from StarV.set.probstar import ProbStar

class Test(object):

    def __init__(self):
        
        self.n_fails = 0
        self.n_tests = 0

    def test_DLODE_constructor(self):

        self.n_tests = self.n_tests + 1

        A = np.random.rand(3,3)
        B = np.random.rand(3,2)
        C = np.random.rand(1,3)
        D = np.random.rand(1,2)


        print('Testing DLODE class constructor....')
        
        try:
            plant1 = DLODE(A, B, C, D)
            plant1.info()
            plant2 = DLODE(A)
            plant2.info()
            
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test successfull!')

    def test_DLODE_rand(self):

        self.n_tests = self.n_tests + 1

        print('\nTesting DLODE rand method...')

        try:
            plant1 = DLODE.rand(3, 2, 1)
            plant1.info()
            plant2 = DLODE.rand(2, 0)
            plant2.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successful!')

    def test_DLODE_stepReach(self):

        self.n_tests = self.n_tests + 1

        print('\nTesting DLODE stepReach method...')

        plant = DLODE.rand(2, 2, 1)

        try:
            print('\nInputs and States are ProbStars')
            
            X0 = ProbStar.rand(2)
            X1, Y1 = plant.stepReach(X0)
            Y1.__str__()
            X1.__str__()

            U0 = np.array([1.0, 0.5])
            X2, Y2 = plant.stepReach(X0, U0)

            X2.__str__()
            Y2.__str__()

            U0 = ProbStar.rand(2)
            X3, Y3 = plant.stepReach(X0, U0, subSetPredicate=False)
            X3.__str__()
            Y3.__str__()

            X4, Y4 = plant.stepReach(X0, U0, subSetPredicate=True)
            X4.__str__()
            Y4.__str__()

            X0 = np.array([0.5, 0.5])
            X5, Y5 = plant.stepReach(X0, U0)

            X6, Y6 = plant.stepReach()
            print('\nX6 = {}'.format(X6))
            print('\nY6 = {}'.format(Y6))

            U0 = np.array([1.0, 0.5])
            X7, Y7 = plant.stepReach(U0=U0)
            print('\nX7 = {}'.format(X7))
            print('\nY7 = {}'.format(Y7))

            # ======================= Star set ===================

            print('\n========Inputs and States are Stars========')
            X0 = Star.rand(2)
            X1, Y1 = plant.stepReach(X0)
            Y1.__str__()
            X1.__str__()

            U0 = np.array([1.0, 0.5])
            X2, Y2 = plant.stepReach(X0, U0)

            X2.__str__()
            Y2.__str__()

            U0 = Star.rand(2)
            X3, Y3 = plant.stepReach(X0, U0, subSetPredicate=False)
            X3.__str__()
            Y3.__str__()

            X4, Y4 = plant.stepReach(X0, U0, subSetPredicate=True)
            X4.__str__()
            Y4.__str__()

            X0 = np.array([0.5, 0.5])
            X5, Y5 = plant.stepReach(X0, U0)
            X5.__str__()
            Y5.__str__()

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_DLODE_multiStepReach(self):

        self.n_tests = self.n_tests + 1
        
        try:
            print('\nTesting DLODE multiStepReach method....')
            k = 3
            X0 = ProbStar.rand(2)
            plant = DLODE.rand(2, 2, 1)
            plant.info()
            #U0 = np.random.rand(10, 2)
            U0 = []
            for i in range(0, k):
                U0.append(ProbStar.rand(2))
            X, Y = plant.multiStepReach(X0, U0, k)
            print('X = {}'.format(X))
            print('Y = {}'.format(Y))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

if __name__ == "__main__":

    test_dlode = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_dlode.test_DLODE_constructor()
    test_dlode.test_DLODE_rand()
    test_dlode.test_DLODE_stepReach()
    test_dlode.test_DLODE_multiStepReach()
    
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing DLODE Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_dlode.n_fails,
                            test_dlode.n_tests - test_dlode.n_fails,
                            test_dlode.n_tests))
