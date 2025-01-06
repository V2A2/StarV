"""
Test of RecurrentLayer layer
Bryan Duong, 12/09/2024

"""

import h5py
import scipy.io
import numpy as np
from StarV.layer.RecurrentLayer import RecurrentLayer
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
import multiprocessing
from StarV.util import load_rnn


class Test(object):
    """
    Testing RecurrentLayer class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_constructor(self):

        self.n_tests = self.n_tests + 1
        self.Whh = np.random.rand(2, 2)
        self.bh = np.random.rand(2)
        self.Whx = np.random.rand(2, 2)
        self.Who = np.random.rand(2, 2)
        self.bo = np.random.rand(2)
        print("Test RecurrentLayer Constructor")

        try:
            RecurrentLayer(self.Whh, self.bh, self.Whx, self.Who, self.bo)
        except Exception:
            print("Test Fail")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_rand(self):
        self.n_tests = self.n_tests + 1
        print("Test RecurrentLayer random method")

        try:
            RecurrentLayer.rand(2, 2)
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_reach(self):
        self.n_tests = self.n_tests + 1
        print("Test RecurrentLayer exactReach method")

        # test with Starset
        # I1 = Star.rand(2)
        # I2 = Star.rand(2)
        # I3 = Star.rand(2)
        # In = [I1, I2, I3]

        # test with ProbStar
        lb = np.array([-2, -1])
        ub = np.array([2, 1])
        mu = (lb + ub) / 2
        sigma = (ub - mu) / 2.5
        Sig = np.diag(sigma**2)
        In = [ProbStar(mu, Sig, lb, ub) for _ in range(2)]
        for I in In:
            I.C = np.zeros((1, I.V.shape[1] - 1))
            I.d = np.zeros((1,))

        net = RecurrentLayer.rand(2, 2)
        try:
            output_set = net.exactReach(In)
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def load_matlab_data(self):
        """
        Load the weights and input set from matlab
        """

        # Load the weights from matlab
        path_fc = "/StarV/util/data/matlab/HSCC2023/small_RNN/dense_D.mat"
        Woh = scipy.io.loadmat(path_fc)["W"]
        bo = scipy.io.loadmat(path_fc)["b"]

        path_RecurrentLayer = "/StarV/util/data/matlab/HSCC2023/small_RNN/simple_rnn_D.mat"
        Whh = scipy.io.loadmat(path_RecurrentLayer)["recurrent_kernel"]
        bh = scipy.io.loadmat(path_RecurrentLayer)["bias"]
        Whx = scipy.io.loadmat(path_RecurrentLayer)["kernel"]

        # Load the input set from matlab
        path_input = "/StarV/util/data/matlab/HSCC2023/small_RNN/points.mat"
        input_data = scipy.io.loadmat(path_input)
        In = input_data["pickle_data"]
        eps = 0.01
        Sn = [Star(x - eps, x + eps) for x in In]
        In = [Sn[0] for _ in range(3)]

        # NOTE: [minKowsi function] constrainst and basic matrix consistency
        for I in In:
            I.C = np.zeros((1, I.V.shape[1] - 1))
            I.d = np.zeros((1,))

        return Whh, bh.flatten(), Whx, Woh, bo.flatten(), In

    def compare_reach(self):
        """
        Test the exactReach method by comparing with matlab results
        """

        self.n_tests = self.n_tests + 1
        print("Test RecurrentLayer exactReach method by comparing with matlab results")

        Whh, bh, Whx, Woh, bo, In = self.load_matlab_data()

        net = RecurrentLayer(Whh, bh, Whx, Woh, bo)

        reach_set = net.exactReach(In, "glpk")

        for m, step in enumerate(reach_set):
            for n, set in enumerate(step):
                lb, ub = set.getRanges("glpk")
                print("reach_set[{},{}]\n".format(m + 1, n + 1))
                print("lb = {}\n ub = {}\n".format(lb, ub))

    def test_evaluate(self):
        self.n_tests = self.n_tests + 1

        Xn = load_rnn.load_raw_data(25)

        net = load_rnn.load_N_2_0()
        # output = net.evaluate(Xn)
        # print()
        try:
            net.evaluate(Xn[:1])
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1


if __name__ == "__main__":
    t = Test()
    t.test_constructor()
    t.test_rand()
    t.test_reach()
    # t.test_evaluate()
    print("Number of tests: ", t.n_tests)
    print("Number of fails: ", t.n_fails)
