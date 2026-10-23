"""
Test of dlode module
Author: Dung Tran
Date: 11/24/2022
"""

from StarV.plant.lode import LODE
import numpy as np
from StarV.set.star import Star
from StarV.set.probstar import ProbStar

class Test(object):

    def __init__(self):
        
        self.n_fails = 0
        self.n_tests = 0

    def test_LODE_constructor(self):

        self.n_tests = self.n_tests + 1

        A = np.random.rand(3,3)
        B = np.random.rand(3,2)
        C = np.random.rand(1,3)
        D = np.random.rand(1,2)


        print('Testing LODE class constructor....')
        
        try:
            plant1 = LODE(A, B, C, D)
            plant1.info()
            plant2 = LODE(A)
            plant2.info()
            
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test successfull!')

    def test_LODE_rand(self):

        self.n_tests = self.n_tests + 1

        print('\nTesting LODE rand method...')

        try:
            plant1 = LODE.rand(3, 2, 1)
            plant1.info()
            plant2 = LODE.rand(2, 0)
            plant2.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successful!')

    def test_LODE_compute_gA_gB(self):

        self.n_tests = self.n_tests + 1

        print('\n Test LODE compute_gA_gB method...')

        plant = LODE.rand(2,2,1)
        plant.info()

        plant.compute_gA_gB(dt=0.1)
        print('\ngA = {}'.format(plant.gA))
        print('\ngB = {}'.format(plant.gB))

    def test_LODE_stepReach(self):

        self.n_tests = self.n_tests + 1

        print('\nTesting LODE stepReach method...')

        plant = LODE.rand(2, 2, 1)

        try:
            print('\nInputs and States are ProbStars')
            
            X0 = ProbStar.rand(2)
            X1 = plant.stepReach(dt=0.1, X0=X0)
            X1.__str__()

            X0 = Star.rand(2)
            X1 = plant.stepReach(dt=0.1, X0=X0)
            X1.__str__()


            U0 = np.array([1.0, 0.5])
            X2 = plant.stepReach(dt=0.1, X0=X0, U=U0)
            X2.__str__()

            U0 = ProbStar.rand(2)
            X3 = plant.stepReach(dt=0.1, X0=X0, U=U0, subSetPredicate=False)
            X3.__str__()
            
            X4 = plant.stepReach(dt=0.1, X0=X0, U=U0, subSetPredicate=True)
            X4.__str__()
            

            X0 = np.array([0.5, 0.5])
            X5 = plant.stepReach(dt=0.1, X0=X0, U=U0)
            X5.__str__()

            X6 = plant.stepReach(dt=0.1)
            print('\nX6 = {}'.format(X6))

            U0 = np.array([1.0, 0.5])
            X7 = plant.stepReach(dt=0.1, U=U0)
            print('\nX7 = {}'.format(X7))


        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_LODE_multiStepReach(self):

        self.n_tests = self.n_tests + 1
                
        try:
            print('\nTesting LODE multiStepReach method....')
            k = 3
            X0 = np.random.rand(2,)
            # X0 = ProbStar.rand(2)
            plant = LODE.rand(2, 2, 1)
            U = ProbStar.rand(2)
            X = plant.multiStepReach(dt=0.1, X0=X0, U=U, k=k)
            print('X = {}'.format(X))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

if __name__ == "__main__":

    test_lode = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    # test_lode.test_LODE_constructor()
    # test_lode.test_LODE_rand()
    # test_lode.test_LODE_compute_gA_gB()
    # test_lode.test_LODE_stepReach()
    test_lode.test_LODE_multiStepReach()
    
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing LODE Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_lode.n_fails,
                            test_lode.n_tests - test_lode.n_fails,
                            test_lode.n_tests))
