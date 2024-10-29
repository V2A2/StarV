"""
Verify Tiny Network
Author: Dung Tran
Date: 8/10/2022
"""


from StarV.verifier.verifier import quantiVerifyBFS
from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load import load_tiny_network
import time
from StarV.util.plot import plot_probstar, plot_probstar_distribution, plot_probstar_reachset_with_unsafeSpec
from matplotlib import pyplot as plt
from StarV.set.star import Star
from tabulate import tabulate

import os


def quantiverify_tiny_network(numCores):
    """verify a tiny network"""

    
    print('=====================================================')
    print('Quantitative Verification of Tiny Network')
    print('=====================================================')
    net, lb, ub, unsafe_mat, unsafe_vec = load_tiny_network()
    net.info()
    S = Star(lb, ub)
    mu = 0.5*(S.pred_lb + S.pred_ub)
    a = 2.5 # coefficience to adjust the distribution
    sig = (mu - S.pred_lb)/a
    print('Mean of predicate variables: mu = {}'.format(mu))
    print('Standard deviation of predicate variables: sig = {}'.format(sig))
    Sig = np.diag(np.square(sig))
    print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
    # Sig = 1e-2*np.eye(S.nVars)
    In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)

    # plot probstar input set
    plot_probstar_distribution(In)
    
    inputSet = []
    inputSet.append(In)
    # plot reachable sets and unsafe reachable sets
    OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                              unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
    dir_mat = np.array([[1., 0.], [0., 1.]])
    unsafe_mat = np.array([[1.0, 0.], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    unsafe_vec = np.array([-2.0, 5.0, 8.2, 2.5])
    plot_probstar_reachset_with_unsafeSpec(OutputSet, unsafe_mat, unsafe_vec, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
    plt.show()
    plot_probstar(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True, show=False)
    plt.show()
    plot_probstar(counterInputSet,dir_mat=dir_mat,dir_vec=None, label= ("$x_1$", "$x_2$"), show_prob=True, show=False)
    plt.show()

    print('=====================================================')
    print('DONE!')
    print('=====================================================')
        

if __name__ == "__main__":

    quantiverify_tiny_network(numCores=4)
