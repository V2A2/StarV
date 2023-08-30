"""
Test for nncs module
Dung Tran
8/14/2023
"""

from StarV.net.network import rand_ffnn
from StarV.nncs.nncs import NNCS
from StarV.plant.dlode import DLODE

class Test(object):
    """Testing nncs module"""

    def __init__(self):
        self.n_fails = 0
        self.n_tests = 0

    def test_constructor(self):

        self.n_tests = self.n_tests + 1

        controller = rand_ffnn([1, 2, 1], ['relu', 'relu'])
        plant = DLODE.rand(2,1,1)
        sys = NNCS(controller, plant, type='linear-NNCS')
        sys.info()

        try:
            controller = rand_ffnn([1, 2, 1], ['relu', 'relu'])
            plant = DLODE.rand(2,1,1)
            sys = NNCS(controller, plant, type='linear-NNCS')
            sys.info()

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successful!')


if __name__ == "__main__":
    test = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test.test_constructor()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing NNCS Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test.n_fails,
                            test.n_tests - test.n_fails,
                            test.n_tests))
            
