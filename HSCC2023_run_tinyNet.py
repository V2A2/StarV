"""
Test verifier module
Author: Dung Tran
Date: 8/10/2022
"""


from StarV.verifier.verifier import quantiVerifyBFS
from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load import load_tiny_network
import time
from StarV.util.plot import plot_probstar
from matplotlib import pyplot as plt
from StarV.set.star import Star
from tabulate import tabulate

import os


def quantiverify_tiny_network(numCores):
    """verify a tiny network"""

    try:
        print('=====================================================')
        print('Quantitative Verification of Tiny Network')
        print('=====================================================')
        net, lb, ub, unsafe_mat, unsafe_vec = load_tiny_network()
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
        inputSet = []
        inputSet.append(In)
        inputProb = inputSet[0].estimateProbability()

        p_filters = [0.0, 0.1]
        data = []
        for p_filter in p_filters:
            start = time.time()
            OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                     unsafe_vec=unsafe_vec, numCores=numCores, p_filter=p_filter)
            end = time.time()
            verifyTime = end-start
            data.append([p_filter, len(OutputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputProb, verifyTime])

        # print verification results
        print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                  "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

        # save verification results
        path = "artifacts/HSCC2023/tinyNet"
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path+"/tinyNetTable.tex", "w") as f:
            print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                          "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)


        # plot reachable sets and unsafe reachable sets
        OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                                  unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
        dir_mat = np.array([[1., 0.], [0., 1.]])
        plot_probstar(OutputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
        plt.savefig(path+"/OutputSet.png", bbox_inches='tight')  # save figure
        plt.show()
        plot_probstar(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True, show=False)
        plt.savefig(path+"/UnsafeOutputSet.png", bbox_inches='tight')  # save figure
        plt.show()
        plot_probstar(counterInputSet,dir_mat=dir_mat,dir_vec=None, label= ("$x_1$", "$x_2$"), show_prob=True, show=False)
        plt.savefig(path+"/CounterInputSet.png", bbox_inches='tight')  # save figure
        plt.show()

        print('=====================================================')
        print('DONE!')
        print('=====================================================')

    except Exception:
        print('Verification Fail!')
       
    else:
        print('Verification Successfull!')

        

if __name__ == "__main__":

    quantiverify_tiny_network(numCores=4)
