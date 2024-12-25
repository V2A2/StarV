"""
HSCC2023: Quantitative Verification for Neural Networks using ProbStars
Evaluation

Author: Dung Tran
Date: 8/10/2022
"""

import os
import time
import numpy as np

from tabulate import tabulate
from matplotlib import pyplot as plt
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.util.plot import plot_probstar
from StarV.util.load import load_ACASXU, load_tiny_network, load_DRL
from StarV.verifier.verifier import quantiVerifyBFS


def quantiverify_tiny_network(numCores):
    """verify a tiny network"""

    
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
    path = "artifacts/HSCC2023_ProbStar/tinyNet"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/tinyNetTable.tex", "w") as f:
        print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                      "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)


    # plot reachable sets and unsafe reachable sets
    OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                              unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
    dir_mat = np.array([[1., 0.], [0., 1.]])
    plot_probstar(inputSet[0], dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
    plt.savefig(path+"/InputSet.png", bbox_inches='tight')  # save figure
    plt.show()
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


def quantiverify_RocketNet(numCores, net_ids, spec_ids, p_filters):
    """Verify all DRL networks"""

    data = []

    for net_id in net_ids:           
        for spec_id in spec_ids: # running property 2 need more memory
            prob_lbs = []
            prob_ubs = []
            prob_mins = []
            prob_maxs = []
            counterInputSets = []
            verifyTimes = []
            net, lb, ub, unsmat, unsvec = load_DRL(net_id, spec_id)
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            a  = 3.0 # coefficience to adjust the distribution
            sig = (mu - S.pred_lb)/a
            print('Mean of predicate variables: mu = {}'.format(mu))
            print('Standard deviation of predicate variables: sig = {}'.format(sig))
            Sig = np.diag(np.square(sig))
            print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            inputSetProb = inputSet[0].estimateProbability()
            name = 'rocketNet_{}'.format(net_id)
            spec = 'P_{}'.format(spec_id)

            for p_filter in p_filters:
                start = time.time()
                outputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, \
                                                                                                                    inputSet=inputSet, unsafe_mat=unsmat, \
                                                                                                                    unsafe_vec=unsvec, numCores=numCores, p_filter=p_filter)
                end = time.time()
                verifyTime = end-start

                # store data for plotting
                prob_lbs.append(prob_lb)
                prob_ubs.append(prob_ub)
                prob_maxs.append(prob_max)
                prob_mins.append(prob_min)
                counterInputSets.append(len(counterInputSet))
                verifyTimes.append(verifyTime)

                data.append([name, spec, p_filter, len(outputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputSetProb, verifyTime])

            fig, (ax1, ax2, ax3) = plt.subplots(3,1)
            ax1.plot(p_filters, prob_lbs, marker='o', label='prob_lb')
            ax1.plot(p_filters, prob_ubs, marker='*', label='prob_ub')
            ax1.plot(p_filters, prob_mins, marker='x', label='prob-min')
            ax1.plot(p_filters, prob_maxs, marker= 'p',label='prob-max')
            # ax1.set_title('Probability')
            ax1.set_ylabel('$p$', fontsize=13)
            ax1.set_xlabel('$p_f$', fontsize=13)
            ax1.legend(loc='upper left')

            ax2.plot(p_filters, counterInputSets)
            # ax2.set_title('CounterSet', fontsize=13)
            ax2.set_xlabel('$p_f$', fontsize=13)
            ax2.set_ylabel('$N_C$', fontsize=13)

            ax3.plot(p_filters, verifyTimes)
            # ax3.set_title('VT')
            ax3.set_xlabel('$p_f$', fontsize=13)
            ax3.set_ylabel('$VT (s)$', fontsize=13)

            # save verification results
            path = "artifacts/HSCC2023_ProbStar/rocketNet"
            if not os.path.exists(path):
                os.makedirs(path)

            plt.savefig(path+"/rocketNet_{}_spec_{}.png".format(net_id, spec_id), bbox_inches='tight')  # save figure


    print(tabulate(data, headers=["Network_ID", "Property", "p_filter", "OutputSet", "UnsafeOutputSet", "counterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "InputSetProbability", "VerificationTime"]))



    with open(path+"/rocketNetTable.tex", "w") as f:
        print(tabulate(data, headers=["Network_ID", "Property","p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                          "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    return data


def quantiverify_ACASXU_all(x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
    """Verify all ACASXU networks with spec_id"""

    data = []
    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')

    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for p_filter in p_filters:
            net, lb, ub, unsmat, unsvec = load_ACASXU(x[i], y[i], spec_ids[i])
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            a  = 3.0 # coefficience to adjust the distribution
            sig = (mu - S.pred_lb)/a
            print('Mean of predicate variables: mu = {}'.format(mu))
            print('Standard deviation of predicate variables: sig = {}'.format(sig))
            Sig = np.diag(np.square(sig))
            print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            inputSetProb = inputSet[0].estimateProbability()
            netName = '{}-{}'.format(x[i], y[i])
            start = time.time()
            OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsmat, \
                                                                     unsafe_vec=unsvec, numCores=numCores, p_filter=p_filter)
            end = time.time()
            verifyTime = end-start
            data.append([spec_ids[i], netName, p_filter, len(OutputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputSetProb, verifyTime])
    # print verification results
    print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                              "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

    # save verification results
    path = "artifacts/HSCC2023_ProbStar/ACASXU"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/ACASXUTable_full.tex", "w") as f:
        print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                              "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    return data


def run_ACASXu_full():
    x = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1]
    y = [2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 4, 5, 6, 7, 8, 9, 1, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9] 
    s = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4] # property id
    quantiverify_ACASXU_all(x=x, y=y, spec_ids=s, numCores=4, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5])


def run_ACASXu_small():
    x = [1, 2, 2, 3, 3, 3, 4, 4, 5, 1, 1]
    y = [6, 2, 9, 1, 6, 7, 1, 7, 3, 7, 9] 
    s = [2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4] # property id
    quantiverify_ACASXU_all(x=x, y=y, spec_ids=s, numCores=8, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5])


if __name__ == "__main__":

    # Figure 5 & Table 1
    quantiverify_tiny_network(numCores=4)

    # Table 2
    run_ACASXu_small()

    # Table 3
    run_ACASXu_full()

    # Table 5
    quantiverify_RocketNet(numCores=4, net_ids=[0, 1], spec_ids=[1,2], p_filters=[0.0, 1e-8, 1e-5, 1e-3])