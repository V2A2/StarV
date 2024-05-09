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
        # net.exactReach(input_set=In, lp_solver="gurobi")


if __name__ == "__main__":
    t = Test()
    t.test_constructor()
    t.test_rand()
    t.test_reach()
    print("Number of tests: ", t.n_tests)
    print("Number of fails: ", t.n_fails)

# t = Test()
# t.test_reach()
