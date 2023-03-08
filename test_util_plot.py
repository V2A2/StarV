"""
Test of plot module
Author: Dung Tran
Date: 9/11/2022
"""

import numpy as np
from StarV.set.probstar import ProbStar
#from StarV.util.plot import probstar2polytope, plot_probstar, plot_2D_Star

from StarV.util.plot import plot_probstar, plot_2D_Star, plot_quantstar
from StarV.set.star import Star
from StarV.set.quantstar import QuantizedStar


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
            
    def test_plot_quantstar(self):        
        self.n_tests += 1
        
        star_lb = np.array([-4.5, -3.3])
        star_ub = np.array([4.5, 3.3])
        
        test_star = Star(star_lb, star_ub)

        #plot_2D_Star(test_star)
        
        test_quant_star = QuantizedStar(test_star)
        
        plot_quantstar(test_quant_star)


if __name__ == "__main__":

    test_plot = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    #test_plot.test_probstar2polytope()
    #test_plot.test_plot_probstar()
    test_plot.test_plot_quantstar()
    
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_plot.n_fails,
                            test_plot.n_tests - test_plot.n_fails,
                            test_plot.n_tests))
