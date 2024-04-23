"""
Test of plot module
Author: Dung Tran
Date: 9/11/2022
"""

import numpy as np
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
# from StarV.util.plot import probstar2polytope, plot_probstar
from StarV.util.plot import plot_probstar, plot_probstar_using_Polytope, plot_star, plot_star_using_Polytope



class Test(object):
    """
       Testing class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_probstar2polytope(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        print('Testing probstar2Polytope method...')
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        try:
            P = probstar2polytope(S)
            print('Polytope = {}'.format(P))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_plot_probstar(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(2,)
        Sig = np.eye(2)
        pred_lb = np.random.rand(2,)
        pred_ub = pred_lb + 0.2
        print('Testing plot_probstar method...')
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        S1 = S.affineMap(A=np.random.rand(2,2))
        P = []
        P.append(S)
        P.append(S1)
        plot_probstar(P)
        try:
            S = ProbStar(mu, Sig, pred_lb, pred_ub)
            S1 = S.affineMap(A=np.random.rand(2,2))
            P = []
            P.append(S)
            P.append(S1)
            plot_probstar(P)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    
    def test_plot_probstar_using_polyhedron(self):

        self.n_tests = self.n_tests + 1
        
        try:
            V = np.array([[-1.2, 1.0, -0.5],
                        [0.0, -1.0, 0.5]])
            C = np.array([[1.0, 1.5],
                        [1.0, -2.0]])
            d = np.array([0.5, -0.5])

            mu = np.random.rand(2,)
            Sig = np.eye(2)
            
            pred_lb = np.array([-2.0, -0.75])
            pred_ub = np.array([1.5, 1.0])

            S = ProbStar(V, C, d, mu, Sig, pred_lb, pred_ub)
            plot_probstar_using_Polytope(S)

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")



if __name__ == "__main__":

    test_plot = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    # test_plot.test_probstar2polytope()
    # test_plot.test_plot_probstar()
    test_plot.test_plot_probstar_using_polyhedron()
    
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_plot.n_fails,
                            test_plot.n_tests - test_plot.n_fails,
                            test_plot.n_tests))
