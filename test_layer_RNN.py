"""
Test of RNN layer
Bryan Duong, 5/14/2024
"""

import h5py
import scipy.io
import numpy as np
from StarV.layer.RNN import RNN
from StarV.set.star import Star
import multiprocessing


class Test(object):
    """
    Testing RNN class methods
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
        print("Test RNN Constructor")

        try:
            RNN(self.Whh, self.bh, self.Whx, self.Who, self.bo)
        except Exception:
            print("Test Fail")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_rand(self):
        self.n_tests = self.n_tests + 1
        print("Test RNN random method")

        try:
            RNN.rand(2, 2)
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_reach(self):
        self.n_tests = self.n_tests + 1
        print("Test RNN exactReach method")

        I1 = Star.rand(2)
        I2 = Star.rand(2)
        I3 = Star.rand(2)
        In = [I1, I2, I3]

        net = RNN.rand(2, 2)
        try:
            net.exactReach(In)
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def load_matlab_data(self):
        """
        Load the weights and input set from matlab
        """

        # path_outputSet = "matlab/HSCC2023/small_RNN/outputSet.mat"
        # path_lb = "matlab/HSCC2023/small_RNN/lb.mat"
        # path_ub = "matlab/HSCC2023/small_RNN/ub.mat"

        # outputSet = scipy.io.loadmat(path_outputSet)["outputSet"]
        # lb = scipy.io.loadmat(path_lb)["lb"]

        # Load the weights from matlab
        path_fc = "matlab/HSCC2023/small_RNN/dense_D.mat"
        Woh = scipy.io.loadmat(path_fc)["W"]
        bo = scipy.io.loadmat(path_fc)["b"]

        # with h5py.File(path_fc, "r") as fc:
        #     Woh = fc["W"][:]
        #     bo = fc["b"][:]

        path_rnn = "matlab/HSCC2023/small_RNN/simple_rnn_D.mat"
        Whh = scipy.io.loadmat(path_rnn)["recurrent_kernel"]
        bh = scipy.io.loadmat(path_rnn)["bias"]
        Whx = scipy.io.loadmat(path_rnn)["kernel"]

        # with h5py.File(path_rnn, "r") as rnn:
        #     Whh = rnn["recurrent_kernel"][:]
        #     bh = rnn["bias"][:]
        #     Whx = rnn["kernel"][:]

        # Load the input set from matlab
        path_input = "matlab/HSCC2023/small_RNN/points.mat"
        input_data = scipy.io.loadmat(path_input)
        In = input_data["pickle_data"]
        eps = 0.01
        In = [Star(x - eps, x + eps) for x in In]

        # path_input1 = "matlab/HSCC2023/small_RNN/inputSet1.mat"
        # input_data1 = scipy.io.loadmat(path_input1)
        # I1 = input_data1["pickle_data"]

        # In = [I1]

        return Whh, bh.flatten(), Whx, Woh, bo.flatten(), In

    def compare_reach(self):
        """
        Test the exactReach method by comparing with matlab results
        """

        self.n_tests = self.n_tests + 1
        print("Test RNN exactReach method by comparing with matlab results")

        Whh, bh, Whx, Woh, bo, In = self.load_matlab_data()

        net = RNN(Whh, bh, Whx, Woh, bo)

        reach_set = net.exactReach(In[:10], "glpk")

        for m, step in enumerate(reach_set):
            for n, set in enumerate(step):
                lb, ub = set.getRanges("glpk")
                print("reach_set[{},{}]\n".format(m + 1, n + 1))
                print("lb = {}\n ub = {}\n".format(lb, ub))


# if __name__ == "__main__":
#     t = Test()
#     t.test_constructor()
#     t.test_rand()
#     t.test_reach()
#     print("Number of tests: ", t.n_tests)
#     print("Number of fails: ", t.n_fails)

t = Test()
t.compare_reach()
