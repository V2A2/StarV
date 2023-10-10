"""
Test net module
Author: Dung Tran
Date: 9/10/2022
"""

from StarV.net.network import rand_ffnn

class Test(object):
    """
    Testing module net class and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_rand_ffnn(self):

        self.n_tests = self.n_tests + 1
        arch = [2, 3, 3, 2]
        actvs = ['relu', 'relu', 'relu']
        print('Test rand_ffnn method ...')
        net1 = rand_ffnn(arch, actvs) 
        try:
            net1 = rand_ffnn(arch, actvs)
            net1.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')    
            

if __name__ == "__main__":

    test_net = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_net.test_rand_ffnn()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_net.n_fails,
                            test_net.n_tests - test_net.n_fails,
                            test_net.n_tests))
