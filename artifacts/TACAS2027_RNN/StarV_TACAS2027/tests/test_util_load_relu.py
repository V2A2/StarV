"""
Test util load module for relu network
Yuntao Li, 2/6/2024
"""

from StarV.util import load_relu
import polytope as pc

class Test(object):
    """
    Testing module net class and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0


    def test_tiny_network(self):

        self.n_tests = self.n_tests + 1
        try:
            [net, lb, ub, unsafe_mat, unsafe_vec] = load_relu.load_tiny_network()
            net.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_2017_IEEE_TNNLS(self):

        self.n_tests = self.n_tests + 1
        try:
            net = load_relu.load_2017_IEEE_TNNLS()
            net.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_ACASXU(self, x, y, spec_id):

        self.n_tests = self.n_tests + 1
            
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_relu.load_ACASXU(x,y,spec_id)
            net.info()
            print('input lower bound: {}'.format(lb))
            print('output lower bound: {}'.format(ub))
            print('unsafe region of spec_id = {}: {}'.format(spec_id, pc.Polytope(unsafe_mat, unsafe_vec)))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


if __name__ == "__main__":

    test_load = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_load.test_tiny_network()
    test_load.test_load_2017_IEEE_TNNLS()
    test_load.test_load_ACASXU(x=1,y=2,spec_id=2)
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing Load module: fails: {}, successfull: {}, \
    total tests: {}'.format(test_load.n_fails,
                            test_load.n_tests - test_load.n_fails,
                            test_load.n_tests))
    
