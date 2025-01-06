"""
Test util load module for piecewise networks
Yuntao Li, 2/6/2024
"""

from StarV.util import load_piecewise
import polytope as pc

class Test(object):
    """
    Testing module net class and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0


    def test_tiny_network_relu(self):

        self.n_tests = self.n_tests + 1
        try:
            [net, lb, ub, unsafe_mat, unsafe_vec] = load_piecewise.load_tiny_network_ReLU()
            net.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_2017_IEEE_TNNLS_relu(self):

        self.n_tests = self.n_tests + 1
        try:
            net = load_piecewise.load_2017_IEEE_TNNLS_ReLU()
            net.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_ACASXU_relu(self, x, y, spec_id):

        self.n_tests = self.n_tests + 1
            
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_piecewise.load_ACASXU_ReLU(x,y,spec_id)
            net.info()
            print('input lower bound: {}'.format(lb))
            print('output lower bound: {}'.format(ub))
            print('unsafe region of spec_id = {}: {}'.format(spec_id, pc.Polytope(unsafe_mat, unsafe_vec)))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')



    def test_tiny_network_leakyrelu(self):

        self.n_tests = self.n_tests + 1
        try:
            [net, lb, ub, unsafe_mat, unsafe_vec] = load_piecewise.load_tiny_network_LeakyReLU()
            net.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_2017_IEEE_TNNLS_leakyrelu(self):

        self.n_tests = self.n_tests + 1
        try:
            net = load_piecewise.load_2017_IEEE_TNNLS_LeakyReLU()
            net.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_ACASXU_leakyrelu(self, x, y, spec_id):

        self.n_tests = self.n_tests + 1
            
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_piecewise.load_ACASXU_LeakyReLU(x,y,spec_id)
            net.info()
            print('input lower bound: {}'.format(lb))
            print('output lower bound: {}'.format(ub))
            print('unsafe region of spec_id = {}: {}'.format(spec_id, pc.Polytope(unsafe_mat, unsafe_vec)))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')



    def test_tiny_network_satlin(self):

        self.n_tests = self.n_tests + 1
        try:
            [net, lb, ub, unsafe_mat, unsafe_vec] = load_piecewise.load_tiny_network_SatLin()
            net.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_2017_IEEE_TNNLS_satlin(self):

        self.n_tests = self.n_tests + 1
        try:
            net = load_piecewise.load_2017_IEEE_TNNLS_SatLin()
            net.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_ACASXU_satlin(self, x, y, spec_id):

        self.n_tests = self.n_tests + 1
            
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_piecewise.load_ACASXU_SatLin(x,y,spec_id)
            net.info()
            print('input lower bound: {}'.format(lb))
            print('output lower bound: {}'.format(ub))
            print('unsafe region of spec_id = {}: {}'.format(spec_id, pc.Polytope(unsafe_mat, unsafe_vec)))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')



    def test_tiny_network_satlins(self):

        self.n_tests = self.n_tests + 1
        try:
            [net, lb, ub, unsafe_mat, unsafe_vec] = load_piecewise.load_tiny_network_SatLins()
            net.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_2017_IEEE_TNNLS_satlins(self):

        self.n_tests = self.n_tests + 1
        try:
            net = load_piecewise.load_2017_IEEE_TNNLS_SatLins()
            net.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_ACASXU_satlins(self, x, y, spec_id):

        self.n_tests = self.n_tests + 1
            
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_piecewise.load_ACASXU_SatLins(x,y,spec_id)
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
    test_load.test_tiny_network_relu()
    test_load.test_load_2017_IEEE_TNNLS_relu()
    test_load.test_load_ACASXU_relu(x=1,y=2,spec_id=2)

    test_load.test_tiny_network_leakyrelu()
    test_load.test_load_2017_IEEE_TNNLS_leakyrelu()
    test_load.test_load_ACASXU_leakyrelu(x=1,y=2,spec_id=2)

    test_load.test_tiny_network_satlin()
    test_load.test_load_2017_IEEE_TNNLS_satlin()
    test_load.test_load_ACASXU_satlin(x=1,y=2,spec_id=2)

    test_load.test_tiny_network_satlins()
    test_load.test_load_2017_IEEE_TNNLS_satlins()
    test_load.test_load_ACASXU_satlins(x=1,y=2,spec_id=2)
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing Load module: fails: {}, successfull: {}, \
    total tests: {}'.format(test_load.n_fails,
                            test_load.n_tests - test_load.n_fails,
                            test_load.n_tests))
    
