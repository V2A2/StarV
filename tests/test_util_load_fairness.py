"""
Test util load  fairness models
Yuntao Li, 3/22/2024
"""

from StarV.util import load_fairness
import polytope as pc
from StarV.util.print_util import print_util

class Test(object):
    """
    Testing module net class and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_load_fairness_adult(self, id):

        self.n_tests = self.n_tests + 1
        try:
            print_util('h2')
            net = load_fairness.load_fairness_adult(id)
            net.info()
            print_util('h2')
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_load_fairness_bank(self, id):
            
        self.n_tests = self.n_tests + 1
        try:
            print_util('h2')
            net = load_fairness.load_fairness_bank(id)
            net.info()
            print_util('h2')
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_load_fairness_german(self, id):
            
        self.n_tests = self.n_tests + 1
        try:
            print_util('h2')
            net = load_fairness.load_fairness_german(id)
            net.info()
            print_util('h2')
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

if __name__ == "__main__":
    # Perform any additional operations or tests here
    # print_util('h1')
    # adult_net = load_fairness.load_fairness_adult(1)
    # adult_net.info()
    # print_util('h1')

    # print_util('h1')
    # bank_net = load_fairness.load_fairness_bank(1)
    # bank_net.info()
    # print_util('h1')

    # print_util('h1')
    # german_net = load_fairness.load_fairness_german(1)
    # german_net.info()
    # print_util('h1')

    test_load_fairness = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    adult = [1,2,3,4,5,6,7,8,9,10,11,12]
    bank = [1,2,3,4,5,6,7,8]
    german = [1,2,3,4,5]
    for id in adult:
        test_load_fairness.test_load_fairness_adult(id)

    for id in bank:
        test_load_fairness.test_load_fairness_bank(id)

    for id in german:
        test_load_fairness.test_load_fairness_german(id)
    # test_load_fairness.test_load_fairness_adult(adult)
    # test_load_fairness.test_load_fairness_bank(bank)
    # test_load_fairness.test_load_fairness_german(german)
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing Load module: fails: {}, successfull: {}, \
    total tests: {}'.format(test_load_fairness.n_fails,
                            test_load_fairness.n_tests - test_load_fairness.n_fails,
                            test_load_fairness.n_tests))

    
