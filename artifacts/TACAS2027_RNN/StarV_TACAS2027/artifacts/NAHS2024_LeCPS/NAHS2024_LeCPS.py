"""
NAHS2024: Quantitative Verification of Learning-enabled Systems using ProbStars Reachability
Evaluation

Author: Yuntao Li
Date: 01/08/2025
"""


import numpy as np
import time
import os
import copy
import multiprocessing
import pickle
import pandas as pd
from tabulate import tabulate
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from matplotlib.pyplot import step, show
from StarV.util.print_util import print_util
from StarV.verifier.verifier import quantiVerifyBFS, quantiVerifyMC
from StarV.set.probstar import ProbStar
from StarV.util.load_piecewise import load_tiny_network_ReLU, load_tiny_network_LeakyReLU, load_tiny_network_SatLin, load_tiny_network_SatLins, load_tiny_network_SatLins
from StarV.util.load_piecewise import load_HCAS_ReLU, load_HCAS_LeakyReLU, load_HCAS_SatLin, load_HCAS_SatLins
from StarV.util.load_piecewise import load_ACASXU_ReLU, load_ACASXU_LeakyReLU, load_ACASXU_SatLin, load_ACASXU_SatLins
from StarV.util.load import load_DRL
from StarV.util.load import load_acc_model, load_AEBS_model
from StarV.nncs.nncs import VerifyPRM_NNCS, verifyBFS_DLNNCS, reachBFS_AEBS, ReachPRM_NNCS, AEBS_NNCS
from StarV.util.plot import plot_probstar
from StarV.util.plot import plot_probstar_reachset, plot_probstar_reachset_with_unsafeSpec
from StarV.set.star import Star


def quantiverify_tiny_network_ReLU(numCores):
    """verify a tiny ReLU network"""

    print_util('h2')
    print_util('h3')
    print('Quantitative Verification of Tiny ReLU Network')
    print_util('h3')

    net, lb, ub, unsafe_mat, unsafe_vec = load_tiny_network_ReLU()
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
    path = "artifacts/NAHS2024_LeCPS/tinyNet/ReLU/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/tinyNetTable.tex", "w") as f:
        print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                        "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)


    # plot reachable sets and unsafe reachable sets
    OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                                unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
    
    dir_mat = np.array([[1., 0.], [0., 1.]])
    plot_probstar(inputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
    plt.savefig(path+"/InputSet_ReLU.png", bbox_inches='tight')  # save figure
    plt.show()
    plot_probstar(OutputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
    plt.savefig(path+"/OutputSet_ReLU.png", bbox_inches='tight')  # save figure
    plt.show()
    plot_probstar(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True, show=False)
    plt.savefig(path+"/UnsafeOutputSet_ReLU.png", bbox_inches='tight')  # save figure
    plt.show()
    plot_probstar(counterInputSet,dir_mat=dir_mat,dir_vec=None, label= ("$x_1$", "$x_2$"), show_prob=True, show=False)
    plt.savefig(path+"/CounterInputSet_ReLU.png", bbox_inches='tight')  # save figure
    plt.show()

    print_util('h3')
    print('DONE!')
    print_util('h3')



def quantiverify_tiny_network_LeakyReLU(numCores):
    """verify a tiny LeakyReLU network"""

    print_util('h2')
    print_util('h3')
    print('Quantitative Verification of Tiny LeakyReLU Network')
    print_util('h3')

    net, lb, ub, unsafe_mat, unsafe_vec = load_tiny_network_LeakyReLU()
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
    path = "artifacts/NAHS2024_LeCPS/tinyNet/LeakyReLU/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/tinyNetTable.tex", "w") as f:
        print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                        "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)


    # plot reachable sets and unsafe reachable sets
    OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                                unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
    
    dir_mat = np.array([[1., 0.], [0., 1.]])
    # plot_probstar(inputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
    # plt.savefig(path+"/InputSet_LeakyReLU.png", bbox_inches='tight')  # save figure
    # plt.show()
    plot_probstar(OutputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
    plt.savefig(path+"/OutputSet_LeakyReLU.png", bbox_inches='tight')  # save figure
    plt.show()
    plot_probstar(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True, show=False)
    plt.savefig(path+"/UnsafeOutputSet_LeakyReLU.png", bbox_inches='tight')  # save figure
    plt.show()
    plot_probstar(counterInputSet,dir_mat=dir_mat,dir_vec=None, label= ("$x_1$", "$x_2$"), show_prob=True, show=False)
    plt.savefig(path+"/CounterInputSet_LeakyReLU.png", bbox_inches='tight')  # save figure
    plt.show()

    print_util('h3')
    print('DONE!')
    print_util('h3')



def quantiverify_tiny_network_SatLin(numCores):
    """verify a tiny SatLin network"""

    print_util('h2')
    print_util('h3')
    print('Quantitative Verification of Tiny SatLin Network')
    print_util('h3')

    net, lb, ub, unsafe_mat, unsafe_vec = load_tiny_network_SatLin()
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
    path = "artifacts/NAHS2024_LeCPS/tinyNet/SatLin/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/tinyNetTable.tex", "w") as f:
        print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                        "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)


    # plot reachable sets and unsafe reachable sets
    OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                                unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
    
    dir_mat = np.array([[1., 0.], [0., 1.]])
    # plot_probstar(inputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
    # plt.savefig(path+"/InputSet_SatLin.png", bbox_inches='tight')  # save figure
    # plt.show()
    plot_probstar(OutputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
    plt.savefig(path+"/OutputSet_SatLin.png", bbox_inches='tight')  # save figure
    plt.show()
    plot_probstar(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True, show=False)
    plt.savefig(path+"/UnsafeOutputSet_SatLin.png", bbox_inches='tight')  # save figure
    plt.show()
    plot_probstar(counterInputSet,dir_mat=dir_mat,dir_vec=None, label= ("$x_1$", "$x_2$"), show_prob=True, show=False)
    plt.savefig(path+"/CounterInputSet_SatLin.png", bbox_inches='tight')  # save figure
    plt.show()

    print_util('h3')
    print('DONE!')
    print_util('h3')



def quantiverify_tiny_network_SatLins(numCores):
    """verify a tiny SatLins network"""

    print_util('h2')
    print_util('h3')
    print('Quantitative Verification of Tiny SatLins Network')
    print_util('h3')

    net, lb, ub, unsafe_mat, unsafe_vec = load_tiny_network_SatLins()
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
    path = "artifacts/NAHS2024_LeCPS/tinyNet/SatLins/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/tinyNetTable.tex", "w") as f:
        print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                        "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)


    # plot reachable sets and unsafe reachable sets
    OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                                unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
    
    dir_mat = np.array([[1., 0.], [0., 1.]])
    # plot_probstar(inputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
    # plt.savefig(path+"/InputSet_SatLins.png", bbox_inches='tight')  # save figure
    # plt.show()
    plot_probstar(OutputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
    plt.savefig(path+"/OutputSet_SatLins.png", bbox_inches='tight')  # save figure
    plt.show()
    plot_probstar(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True, show=False)
    plt.savefig(path+"/UnsafeOutputSet_SatLins.png", bbox_inches='tight')  # save figure
    plt.show()
    plot_probstar(counterInputSet,dir_mat=dir_mat,dir_vec=None, label= ("$x_1$", "$x_2$"), show_prob=True, show=False)
    plt.savefig(path+"/CounterInputSet_SatLins.png", bbox_inches='tight')  # save figure
    plt.show()

    print_util('h3')
    print('DONE!')
    print_util('h3')



def quantiverify_HCAS_all_ReLU(prev_acv, tau, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
    """Verify all HCAS ReLU networks with spec_id"""

    print_util('h2')
    data = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')

    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for p_filter in p_filters:
            print_util('h3')
            print('quanti verify of HCAS N_{}_{} ReLU network under specification {}...'.format(prev_acv[i], tau[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_HCAS_ReLU(prev_acv[i], tau[i], spec_ids[i])
            net.info()
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
            print_util('h3')
    # print verification results
    print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

    # save verification results
    path = "artifacts/NAHS2024_LeCPS/HCAS/ReLU"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/HCASTable_all.tex", "w") as f:
        print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    
    print_util('h2')
    return data



def quantiverify_HCAS_all_LeakyReLU(prev_acv, tau, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
    """Verify all HCAS LeakyReLU networks with spec_id"""

    print_util('h2')
    data = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')

    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for p_filter in p_filters:
            print_util('h3')
            print('quanti verify of HCAS N_{}_{} ReLU network under specification {}...'.format(prev_acv[i], tau[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_HCAS_LeakyReLU(prev_acv[i], tau[i], spec_ids[i])
            net.info()
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
            print_util('h3')
    # print verification results
    print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

    # save verification results
    path = "artifacts/NAHS2024_LeCPS/HCAS/LeakyReLU"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/HCASTable_all.tex", "w") as f:
        print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    
    print_util('h2')
    return data



def quantiverify_HCAS_all_SatLin(prev_acv, tau, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
    """Verify all HCAS SatLin networks with spec_id"""

    print_util('h2')
    data = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')

    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for p_filter in p_filters:
            print_util('h3')
            print('quanti verify of HCAS N_{}_{} ReLU network under specification {}...'.format(prev_acv[i], tau[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_HCAS_SatLin(prev_acv[i], tau[i], spec_ids[i])
            net.info()
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
            print_util('h3')
    # print verification results
    print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

    # save verification results
    path = "artifacts/NAHS2024_LeCPS/HCAS/SatLin"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/HCASTable_all_filter.tex", "w") as f:
        print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    
    print_util('h2')
    return data



def quantiverify_HCAS_all_SatLins(prev_acv, tau, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
    """Verify all HCAS SatLins networks with spec_id"""

    print_util('h2')
    data = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')

    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for p_filter in p_filters:
            print_util('h3')
            print('quanti verify of HCAS N_{}_{} ReLU network under specification {}...'.format(prev_acv[i], tau[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_HCAS_SatLins(prev_acv[i], tau[i], spec_ids[i])
            net.info()
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
            print_util('h3')
    # print verification results
    print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

    # save verification results
    path = "artifacts/NAHS2024_LeCPS/HCAS/SatLins"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/HCASTable_all_filter_p3.tex", "w") as f:
        print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    
    print_util('h2')
    return data



def quantiverify_ACASXU_all_ReLU(x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
    """Verify all ACASXU ReLU networks with spec_id"""

    print_util('h2')
    data = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')

    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for p_filter in p_filters:
            print_util('h3')
            print('quanti verify of ACASXU N_{}_{} ReLU network under specification {}...'.format(x[i], y[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_ACASXU_ReLU(x[i], y[i], spec_ids[i])
            net.info()
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
            print_util('h3')
    # print verification results
    print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

    # save verification results
    path = "artifacts/NAHS2024_LeCPS/ACASXU/ReLU"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/ACASXUTable.tex", "w") as f:
        print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    
    print_util('h2')
    return data



def quantiverify_ACASXU_all_ReLU_MC(x, y, spec_ids, unsafe_mat, unsafe_vec, numSamples, nTimes, numCore):
    """Verify all ACASXU ReLU networks with spec_id"""

    print_util('h2')
    data = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')

    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for numSample in numSamples:
            print_util('h3')
            print('quanti verify using Monte Carlo of ACASXU N_{}_{} ReLU network under specification {}...'.format(x[i], y[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_ACASXU_ReLU(x[i], y[i], spec_ids[i])
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            a  = 3.0 # coefficience to adjust the distribution
            sig = (mu - S.pred_lb)/a
            print('Mean of predicate variables: mu = {}'.format(mu))
            print('Standard deviation of predicate variables: sig = {}'.format(sig))
            Sig = np.diag(np.square(sig))
            print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            netName = '{}-{}'.format(x[i], y[i])

            start = time.time()
            unsafe_prob = quantiVerifyMC(net=net, inputSet=In, unsafe_mat=unsmat, unsafe_vec=unsvec, numSamples=numSample, nTimes=nTimes, numCores=numCore)
            end = time.time()

            verifyTime = end-start
            data.append([spec_ids[i], netName, numSample, unsafe_prob, verifyTime])
            print_util('h3')
    # print verification results
    print(tabulate(data, headers=["Prop.", "Net", "N-samples", "UnsafeProb", "VerificationTime"]))

    # save verification results
    path = "artifacts/NAHS2024_LeCPS/ACASXU/ReLU/MonteCarlo"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/ACASXUTableMC_full.tex", "w") as f:
        print(tabulate(data, headers=["Prop.", "Net", "N-samples", "UnsafeProb", "VerificationTime"], tablefmt='latex'), file=f)
    
    print_util('h2')
    return data


def format_net_name(x, y):
    """Format network name as x-y"""
    return f"{x}-{y}"

def quantiverify_ACASXU_ReLU_table_3(x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
    """Verify all ACASXU ReLU networks with spec_id"""
    print_util('h2')
    results = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')
    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for p_filter in p_filters:
            print_util('h3')
            print('quanti verify of ACASXU N_{}_{} ReLU network under specification {}...'.format(x[i], y[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_ACASXU_ReLU(x[i], y[i], spec_ids[i])
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            a = 3.0  # coefficient to adjust the distribution
            sig = (mu - S.pred_lb)/a
            print('Mean of predicate variables: mu = {}'.format(mu))
            print('Standard deviation of predicate variables: sig = {}'.format(sig))
            Sig = np.diag(np.square(sig))
            print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            inputSetProb = inputSet[0].estimateProbability()
            
            start = time.time()
            OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(
                net=net, inputSet=inputSet, unsafe_mat=unsmat,
                unsafe_vec=unsvec, numCores=numCores, p_filter=p_filter
            )
            end = time.time()
            verifyTime = end-start

            # Store results in dictionary format
            result = {
                'Prop': spec_ids[i],
                'Net': format_net_name(x[i], y[i]),
                'p_f': p_filter,
                'O': len(OutputSet),
                'US-O': len(unsafeOutputSet),
                'C': len(counterInputSet),
                'US-Prob-LB': prob_lb,
                'US-Prob-UB': prob_ub,
                'US-Prob-Min': prob_min,
                'US-Prob-Max': prob_max,
                'I-Prob': inputSetProb,
                'VT': verifyTime
            }
            results.append(result)
            print_util('h3')

    # Print verification results
    print(tabulate([[r['Prop'], r['Net'], r['p_f'], r['O'], r['US-O'], r['C'], 
                    r['US-Prob-LB'], r['US-Prob-UB'], r['US-Prob-Min'], r['US-Prob-Max'],
                    r['I-Prob'], r['VT']] for r in results],
                  headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",
                          "UnsafeProb-LB", "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max",
                          "inputSet Probability", "VerificationTime"]))

    # Save results to pickle file
    path = "artifacts/NAHS2024_LeCPS/ACASXU/ReLU"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/NAHS2024_AcasXu_ReLU_ProbStar.pkl', 'wb') as f:
        pickle.dump(results, f)

    print_util('h2')
    return results


def quantiverify_ACASXU_ReLU_MC_table_3(x, y, spec_ids, unsafe_mat, unsafe_vec, numSamples, nTimes, numCore):
    """Verify all ACASXU ReLU networks with spec_id using Monte Carlo"""
    print_util('h2')
    results = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')
    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for numSample in numSamples:
            print_util('h3')
            print('quanti verify using Monte Carlo of ACASXU N_{}_{} ReLU network under specification {}...'.format(x[i], y[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_ACASXU_ReLU(x[i], y[i], spec_ids[i])
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            a = 3.0  # coefficient to adjust the distribution
            sig = (mu - S.pred_lb)/a
            print('Mean of predicate variables: mu = {}'.format(mu))
            print('Standard deviation of predicate variables: sig = {}'.format(sig))
            Sig = np.diag(np.square(sig))
            print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)

            start = time.time()
            unsafe_prob = quantiVerifyMC(net=net, inputSet=In, unsafe_mat=unsmat, 
                                         unsafe_vec=unsvec, numSamples=numSample, nTimes=nTimes, numCores=numCore)
            end = time.time()
            verifyTime = end-start

            # Store results in dictionary format
            result = {
                'Prop': spec_ids[i],
                'Net': format_net_name(x[i], y[i]),
                'p_f': 0,  # Monte Carlo results only correspond to p_f = 0
                'MC_US-Prob': unsafe_prob,
                'MC_VT': verifyTime
            }
            results.append(result)
            print_util('h3')

    # Print verification results
    print(tabulate([[r['Prop'], r['Net'], r['MC_US-Prob'], r['MC_VT']] for r in results],
                  headers=["Prop.", "Net", "UnsafeProb", "VerificationTime"]))

    # Save results to pickle file
    path = "artifacts/NAHS2024_LeCPS/ACASXU/ReLU"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/NAHS2024_AcasXu_ReLU_MC.pkl', 'wb') as f:
        pickle.dump(results, f)

    print_util('h2')
    return results


def qualiverify_ACASXU_ReLU_other_tools_table_3():
    """
    Verify all ACASXU ReLU networks with spec_id using other verification tools
    (NNV, Marabou, NNenum)
    """
    data = [
    {"x": 1, "y": 6, "s": 2, "Result": "violated", "NNV_exact": 13739.970, "Marabou": 166.58, "NNenum": 1.5938},
    {"x": 2, "y": 2, "s": 2, "Result": "violated", "NNV_exact": 21908.227, "Marabou": 9.27, "NNenum": 0.89502},
    {"x": 2, "y": 9, "s": 2, "Result": "violated", "NNV_exact": 74328.776, "Marabou": 31.19, "NNenum": 1.0538},
    {"x": 3, "y": 1, "s": 2, "Result": "violated", "NNV_exact": 5601.779, "Marabou": 3.50, "NNenum": 0.86799},
    {"x": 3, "y": 6, "s": 2, "Result": "violated", "NNV_exact": 74664.104, "Marabou": 36.28, "NNenum": 0.91804},
    {"x": 3, "y": 7, "s": 2, "Result": "violated", "NNV_exact": 23282.763, "Marabou": 105.32, "NNenum": 61.198},
    {"x": 4, "y": 1, "s": 2, "Result": "violated", "NNV_exact": 17789.960, "Marabou": 9.58, "NNenum": 0.87820},
    {"x": 4, "y": 7, "s": 2, "Result": "violated", "NNV_exact": 40696.630, "Marabou": 8.67, "NNenum": 0.90657},
    {"x": 5, "y": 3, "s": 2, "Result": "violated", "NNV_exact": 2740.739, "Marabou": 113.12, "NNenum": 1.9434},
    {"x": 1, "y": 7, "s": 3, "Result": "violated", "NNV_exact": 0.943, "Marabou": 0.25, "NNenum": 0.86683},
    {"x": 1, "y": 9, "s": 4, "Result": "violated", "NNV_exact": 1.176, "Marabou": 0.31, "NNenum": 0.86635}
    ]

    results = []
    for entry in data:
        result = {
            'Prop': entry['s'],
            'Net': format_net_name(entry['x'], entry['y']),
            'p_f': 0,  # Other tools results only correspond to p_f = 0
            'NNV': entry['NNV_exact'],
            'Marabou': entry['Marabou'],
            'NNenum': entry['NNenum']
        }
        results.append(result)
    
    # Save to pickle file
    path = "artifacts/NAHS2024_LeCPS/ACASXU/ReLU"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/NAHS2024_AcasXu_ReLU_other_tools.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results


def generate_table_3_AcasXu_ReLU_quanti_verify_vs_other_tools():
    """
    Generate LaTeX table combining verification results for ACASXU ReLU networks:
    1. Quantitative Verification (ProbStar)
    2. Monte Carlo results
    3. Other verification tools (NNV, Marabou, NNenum)
    """
    def format_number(value):
        """Preserve scientific notation for very small numbers and original format"""
        if pd.isna(value) or value is None:
            return ''
        elif isinstance(value, int):
            return f'{value}'
        elif isinstance(value, float):
            if abs(value) < 1e-5:  # Use scientific notation for very small numbers
                return f'{value:.6e}'
            elif abs(value) >= 1:  # Use fixed notation with reasonable precision
                return f'{value:.6f}'.rstrip('0').rstrip('.')
            else:  # For numbers between 0 and 1
                return f'{value:.6f}'.rstrip('0').rstrip('.')
        return str(value)

    def load_pickle_file(filename):
        """Load data from pickle file with error handling"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            return []
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return []
        
    path = "artifacts/NAHS2024_LeCPS/ACASXU/ReLU"
    # Load all data sources
    probstar_data = load_pickle_file(path + '/NAHS2024_AcasXu_ReLU_ProbStar.pkl')
    mc_data = load_pickle_file(path + '/NAHS2024_AcasXu_ReLU_MC.pkl')
    other_tools_data = load_pickle_file(path + '/NAHS2024_AcasXu_ReLU_other_tools.pkl')

    # Create lookup dictionaries for MC and other tools data
    mc_dict = {(d['Prop'], d['Net']): d for d in mc_data}
    other_dict = {(d['Prop'], d['Net']): d for d in other_tools_data}

    # Sort probstar data by Prop, Net, and p_f
    sorted_data = sorted(probstar_data, key=lambda x: (x['Prop'], x['Net'], x['p_f']))

    # Generate LaTeX table
    table_lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllllllllll||ll||l|l|l}",
        r"\hline",
        r"    \multicolumn{12}{c||}{\textbf{Quantitative Verification}}  & " + 
        r"\multicolumn{2}{c||}{\textbf{Monte Carlo (NS: $10^7$)}} & " +
        r"\multicolumn{3}{c}{\textbf{Qualitative Verification}} \\",
        r"\hline",
        r"\textbf{Prop} & \textbf{Net} & \textbf{$p_f$} & " +
        r"\textbf{$\mathcal{O}$} & \textbf{$\mathcal{US-O}$} & " +
        r"\textbf{$\mathcal{C}$} & \textbf{US-Prob-LB} & \textbf{US-Prob-UB} & " +
        r"\textbf{US-Prob-Min} & \textbf{US-Prob-Max} & \textbf{I-Prob} & " +
        r"\textbf{VT} & \textbf{US-Prob} & \textbf{VT} & \textbf{NNV} & " +
        r"\textbf{Marabou} & \textbf{NNenum}\\",
        r"\hline"
    ]

    # Add data rows
    for entry in sorted_data:
        prop = entry['Prop']
        net = entry['Net']
        p_f = entry['p_f']
        
        # Base row with Quantitative Verification data
        row = [
            str(prop),
            net,
            str(p_f),
            format_number(entry['O']),
            format_number(entry['US-O']),
            format_number(entry['C']),
            format_number(entry['US-Prob-LB']),
            format_number(entry['US-Prob-UB']),
            format_number(entry['US-Prob-Min']),
            format_number(entry['US-Prob-Max']),
            format_number(entry['I-Prob']),
            format_number(entry['VT'])
        ]

        # Add Monte Carlo and Other Tools data only if p_f = 0
        if p_f == 0:
            mc_entry = mc_dict.get((prop, net), {})
            other_entry = other_dict.get((prop, net), {})
            
            row.extend([
                format_number(mc_entry.get('MC_US-Prob', '')),
                format_number(mc_entry.get('MC_VT', '')),
                format_number(other_entry.get('NNV', '')),
                format_number(other_entry.get('Marabou', '')),
                format_number(other_entry.get('NNenum', ''))
            ])
        else:
            # Add empty cells for Monte Carlo and Other Tools columns
            row.extend([''] * 5)

        table_lines.append(' & '.join(row) + r' \\')

    # Add table footer
    table_lines.extend([
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ])

    # Join all lines with newlines and save to file
    table_content = '\n'.join(table_lines)
    with open(path + '/Table_3_AcasXu_ReLU_quanti_verify_vs_other_tools.tex', 'w') as f:
        f.write(table_content)

    print("Table has been generated and saved to 'Table_3_AcasXu_ReLU_quanti_verify_vs_other_tools.tex'")
    return table_content


def quantiverify_ACASXU_all_LeakyReLU(x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
    """Verify all ACASXU LeakyReLU networks with spec_id"""

    print_util('h2')
    data = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')

    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for p_filter in p_filters:
            print_util('h3')
            print('quanti verify of ACASXU N_{}_{} LeakyReLU network under specification {}...'.format(x[i], y[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_ACASXU_LeakyReLU(x[i], y[i], spec_ids[i])
            net.info()
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
            print_util('h3')
    # print verification results
    print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

    # save verification results
    path = "artifacts/NAHS2024_LeCPS/ACASXU/LeakyReLU"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/ACASXUTable.tex", "w") as f:
        print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    
    print_util('h2')
    return data



def quantiverify_ACASXU_all_SatLin(x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
    """Verify all ACASXU SatLin networks with spec_id"""

    print_util('h2')
    data = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')

    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for p_filter in p_filters:
            print_util('h3')
            print('quanti verify of ACASXU N_{}_{} SatLin network under specification {}...'.format(x[i], y[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_ACASXU_SatLin(x[i], y[i], spec_ids[i])
            net.info()
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
            print_util('h3')
    # print verification results
    print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

    # save verification results
    path = "artifacts/NAHS2024_LeCPS/ACASXU/SatLin"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/ACASXUTable.tex", "w") as f:
        print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    
    print_util('h2')
    return data



def quantiverify_ACASXU_all_SatLins(x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
    """Verify all ACASXU SatLins networks with spec_id"""

    print_util('h2')
    data = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')

    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for p_filter in p_filters:
            print_util('h3')
            print('quanti verify of ACASXU N_{}_{} SatLins network under specification {}...'.format(x[i], y[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_ACASXU_SatLins(x[i], y[i], spec_ids[i])
            net.info()
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
            print_util('h3')
    # print verification results
    print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

    # save verification results
    path = "artifacts/NAHS2024_LeCPS/ACASXU/SatLins"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/ACASXUTable.tex", "w") as f:
        print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    
    print_util('h2')
    return data



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
            path = "artifacts/NAHS2024_LeCPS/rocketNet"
            if not os.path.exists(path):
                os.makedirs(path)

            plt.savefig(path+"/rocketNet_{}_spec_{}.png".format(net_id, spec_id), bbox_inches='tight')  # save figure


    print(tabulate(data, headers=["Network_ID", "Property", "p_filter", "OutputSet", "UnsafeOutputSet", "counterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "InputSetProbability", "VerificationTime"]))



    with open(path+"/rocketNetTable.tex", "w") as f:
        print(tabulate(data, headers=["Network_ID", "Property","p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                          "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    return data



def generate_exact_reachset_figs(net_id='5x20', show=False):
    'generate 4 pictures and save in NAHS2024_LeCPS/pics/'

    if net_id == '5x20':
        net = 'controller_5_20'
    elif net_id == '3x20':
        net = 'controller_3_20'
    elif net_id == '7x30':
        net = 'controller_7_20'
    else:
        raise RuntimeError('Invalid net_id')

    plant='linear'
    initSet_id=5
    pf=0.0
    numSteps=30
    numCores=4

    # load NNCS ACC system
    print('Loading the ACC system...')
    ncs, initSets, refInputs, unsafe_mat, unsafe_vec = load_acc_model(net,plant)



    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]

    print('Verifying the ACC system for {} timesteps using exact reachability...'.format(numSteps))
    res = verifyBFS_DLNNCS(ncs, verifyPRM)
    RX = res.RX
    Ce = res.CeIn
    Co = res.CeOut

    n = len(RX)
    CE = []
    CO = []
    for i in range(0,n):
        if len(Ce[i]) > 0:
            CE.append(Ce[i])  # to plot counterexample set
            CO.append(Co[i])

    # plot reachable set  (d_actual - d_safe) vs. (v_ego)
    dir_mat1 = np.array([[0., 0., 0., 0., 1., 0., 0.],
                        [1., 0., 0., -1., -1.4, 0., 0.]])
    dir_vec1 = np.array([0., -10.])

    # plot reachable set  d_rel vs. d_safe
    dir_mat2 = np.array([[1., 0., 0., -1., 0., 0., 0.],
                        [0., 0., 0., 0., 1.4, 0., 0.]])
    dir_vec2 = np.array([0., 10.])

    # plot counter input set
    dir_mat3 = np.array([[0., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 1., 0., 0.]])
    dir_vec3 = np.array([0., 0.])


    path = "artifacts/NAHS2024_LeCPS/ACC/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    print('Plot reachable set...')
    fig1 = plt.figure()
    plot_probstar_reachset(RX, dir_mat=dir_mat2, dir_vec=dir_vec2, show_prob=False, \
                           label=('$d_{r}$','$d_{safe}$'), show=False)
    plt.savefig(path+"/dr_vs_dsafe_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure
    if show: plt.show()

    if net_id=='5x20':
        fig2 = plt.figure()
        plot_probstar_reachset(RX, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=False, \
                               label=('$v_{ego}$','$d_r - d_{safe}$'), show=False)
        plt.savefig(path+"/Figure_10_a__dr_dsafe_vs_vego_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure 6 (a)
        if show: plt.show()

        print('Plot counter output set...')
        fig3 = plt.figure()
        plot_probstar_reachset(CO, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=False, \
                               label=('$v_{ego}$','$d_r - d_{safe}$'), show=False)
        plt.savefig(path+"/Figure_10_b__counterOutputSet_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure 6 (b)
        if show: plt.show()

        print('Plot counter init set ...')
        fig4 = plt.figure()
        plot_probstar_reachset(CE, dir_mat=dir_mat3, dir_vec=dir_vec3, show_prob=False, \
                               label=('$v_{lead}[0]$','$v_{ego}[0]$'), show=False)
        plt.savefig(path+"/Figure_10_c__counterInitSet_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure 6 (c)
        if show: plt.show()

    else:
        fig1 = plt.figure()
        plot_probstar_reachset(RX, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=False, \
                               label=('$v_{ego}$','$d_r - d_{safe}$'), show=False)
        plt.savefig(path+"/dr_dsafe_vs_vego_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure
        if show: plt.show()

    print('Done!')



def generate_exact_Q2_verification_results(net_id='5x20'):
    'generate Q2 verification results'

    if net_id == '5x20':
        net = 'controller_5_20'
    elif net_id == '3x20':
        net = 'controller_3_20'
    elif net_id == '7x30':
        net = 'controller_7_20'
    else:
        raise RuntimeError('Invalid net_id')

    plant='linear'
    initSet_id=5
    pf=0.0
    numSteps=30
    numCores=4

    # load NNCS ACC system
    print('Loading the ACC system...')
    ncs, initSets, refInputs, unsafe_mat, unsafe_vec = load_acc_model(net,plant)



    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]

    print('Verifying the ACC system controlled by Net_3x20 for {} timesteps using exact reachability...'.format(numSteps))
    res = verifyBFS_DLNNCS(ncs, verifyPRM)
    Ql = res.Ql
    Qt = res.Qt
    Qt_max = res.Qt_max

    t = range(0, 1, numSteps)

    path = "artifacts/NAHS2024_LeCPS/ACC/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    print('Plot Q2 verification results...')

    print('Plot exact qualitative results...')
    fig1 = plt.figure()
    xaxis = np.arange(0, len(Ql))
    yaxis = np.array(Ql)
    step(xaxis, yaxis)
    label=('t', 'SAT')
    plt.xlabel(label[0], fontsize=20)
    plt.ylabel(label[1], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if net_id == '5x20':
        plt.savefig(path+"/Figure_9_a__Ql_exact_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure [5 (a) N_{5x20}]
    else:
        plt.savefig(path+"/Ql_exact_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure

    fig2 = plt.figure()

    print('Plot exact quantitative results Qt...')
    yaxis1 = np.array(Qt)
    yaxis2 = np.array(Qt_max)

    plt.plot(xaxis, yaxis1, color='blue', marker='o', label='$Qt$')
    plt.plot(xaxis, yaxis2, color='red', marker='x', label='$Qt_{max}$')
    label=('t', '$Qt, Qt_{max}$')
    plt.xlabel(label[0], fontsize=20)
    plt.ylabel(label[1], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    if net_id == '5x20':
        plt.savefig(path+"/Figure_9_b__Qt_Qt_max_exact_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure [5 (b) N_{5x20}; Appendix N_{3x20}]
    else:
        plt.savefig(path+"/Qt_Qt_max_exact_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure

    print('Done!')



def generate_approx_Q2_verification_results(net_id='5x20', pf=0.001):
    'generate approx Q2 verification results'

    if net_id == '5x20':
        net = 'controller_5_20'
    elif net_id == '3x20':
        net = 'controller_3_20'
    elif net_id == '7x30':
        net = 'controller_7_20'
    else:
        raise RuntimeError('Invalid net_id')

    plant='linear'
    initSet_id=5
    numSteps=30
    numCores=4

    # load NNCS ACC system
    print('Loading the ACC system...')
    ncs, initSets, refInputs, unsafe_mat, unsafe_vec = load_acc_model(net,plant)



    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]

    print('Verifying the ACC system for {} timesteps using approx reachability. with pf = {}..'.format(numSteps, pf))
    res = verifyBFS_DLNNCS(ncs, verifyPRM)
    Ql = res.Ql
    Qt = res.Qt
    Qt_ub = res.Qt_ub
    Qt_max = res.Qt_max

    t = range(0, 1, numSteps)

    path = "artifacts/NAHS2024_LeCPS/ACC/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    print('Plot Q2 verification results...')

    print('Plot approx qualitative results...')
    fig1 = plt.figure()
    xaxis = np.arange(0, len(Ql))
    yaxis = np.array(Ql)
    step(xaxis, yaxis)
    label=('t', 'SAT')
    plt.xlabel(label[0], fontsize=20)
    plt.ylabel(label[1], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(path+"/Ql_approx_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure

    fig2 = plt.figure()

    print('Plot approx quantitative results...')
    yaxis1 = np.array(Qt)
    yaxis2 = np.array(Qt_ub)
    yaxis3 = np.array(Qt_max)

    plt.plot(xaxis, yaxis1, color='blue', marker='o', label='$Qt$')
    plt.plot(xaxis, yaxis2, color='orange', marker='*', label='$Qt_{ub}$')
    plt.plot(xaxis, yaxis3, color='red', marker='x', label='$Qt_{max}$')
    label=('t', '$Qt, Qt_{lb}, Qt_{ub}$')
    plt.xlabel(label[0], fontsize=20)
    plt.ylabel(label[1], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    if net_id == '5x20':
        plt.savefig(path+"/Figure_9_c__Qt_Qtub_Qt_max_approx_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure [5 (c) N_{5x20}; Appendix N_{3x20}]
    else:
        plt.savefig(path+"/Qt_Qtub_Qt_max_approx_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure

    print('Done!')



def generate_VT_Conv_vs_pf_net():
    'generate verification time vesus pf and networks'

    nets=['controller_3_20', 'controller_5_20']
    plant='linear'
    initSet_id=5
    #pf = [0.0, 0.1]
    pf = [0.0, 0.005, 0.01, 0.015]
    numSteps=30
    numCores=4

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()



    m = len(pf)
    n = len(nets)
    VT = np.zeros((m, n))
    NO = np.zeros((m,n))
    VT_improv = np.zeros((m, n))  # improvement in verification time
    Conv = np.zeros((m, n))       # conservativeness of prediction
    Qt_ub_sum = np.zeros((m, n))
    Qt_sum = np.zeros((m, n))
    Qt_exact_sum = np.zeros((n,))
    data = []
    for i in range(0, m):
        for j in range(0, n):
            # load NNCS ACC system
            print('Loading the ACC system with network {}...'.format(nets[j]))
            ncs, initSets, refInputs, unsafe_mat, unsafe_vec = load_acc_model(nets[j],plant)
            verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
            verifyPRM.refInputs = copy.deepcopy(refInputs)
            verifyPRM.numSteps = numSteps
            verifyPRM.pf = pf[i]
            verifyPRM.numCores = numCores
            verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]

            print('Verifying the ACC system with {} for {} \
            timesteps using approx reachability with pf = {}...'.format(nets[j], numSteps, pf[i]))
            start_time = time.time()
            res = verifyBFS_DLNNCS(ncs, verifyPRM)
            end_time = time.time()


            RX = res.RX
            Qt = res.Qt
            Qt_ub = res.Qt_ub
            Qt_ub_sum[i,j] = sum(Qt_ub)
            Qt_sum[i,j] = sum(Qt)
            if i==0:
                Qt_exact_sum[j] = sum(Qt)
            M = len(RX)
            NO[i,j] = len(RX[M-1])
            VT[i, j] = end_time - start_time
            VT_improv[i,j] = (VT[0,j] - VT[i, j])*100/VT[0,j]
            Conv[i,j] = 100*(Qt_ub_sum[i,j] - Qt_exact_sum[j])/Qt_exact_sum[j]
            strg1 = '{}'.format(pf[i])
            strg2 = '{}'.format(nets[j])
            data.append([strg1, strg2, VT[i,j], NO[i,j], VT_improv[i,j], Conv[i,j]])

    print(tabulate(data, headers=["p_filter", "network", "verification time", "number of output sets", "VT_improve (in %)", "Convativeness (in %)"]))

    path = "artifacts/NAHS2024_LeCPS/ACC/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/Table_9__VT_Conv_vs_pf_net.tex", "w") as f:
         print(tabulate(data, headers=["p_filter", "network", "verification time", "number of output sets", "VT_improve", "Conservativeness"], tablefmt='latex'), file=f) # Table1


    print('Done!')



def generate_exact_reachset_figs_AEBS(show=False):

    # load initial conditions of NNCS AEBS system
    print('Loading initial conditions of AEBS system...')

    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
    AEBS = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)


    # stepReach computation for AEBS

    # step 0: initial state of plant
    # step 1: normalizing state
    # step 2: compute brake output of RL controller
    # step 3: compose brake output and normalized speed
    # step 4: compute transformer output
    # step 5: get control output
    # step 6: scale control output
    # step 7: compute reachable set of the plannt with the new control input and initial state

    # step 8: go back to step 1, .... (another stepReach)

    # initial bound on states
    d_lb = [97., 90., 48., 5.0]
    d_ub = [97.5, 90.5, 48.5, 5.2]
    v_lb = [25.2, 27., 30.2, 1.0]
    v_ub = [25.5, 27.2, 30.4, 1.2]



    reachPRM = ReachPRM_NNCS()
    reachPRM.numSteps = 50
    reachPRM.filterProb = 0.0
    reachPRM.numCores = 4


    for i in range(0, len(initSets)):
        reachPRM.initSet = initSets[i]
        X, _ = reachBFS_AEBS(AEBS, reachPRM)

        # plot reachable set  (d) vs. (v_ego)
        dir_mat1 = np.array([[1., 0., 0.],
                            [0., 1., 0]])
        dir_vec1 = np.array([0., 0.])



        path = "artifacts/NAHS2024_LeCPS/AEBS/pics"
        if not os.path.exists(path):
            os.makedirs(path)

        # 0.5 <= d_k <= 2.5 AND 0.2 <= v_k <= v_ub
        unsafe_mat = np.array([[1.0, 0.0], [-1., 0.], [0., 1.], [0., -1.]])
        unsafe_vec = np.array([2.5, -0.5, v_ub[i], -0.2])

        print('Plot reachable set...')
        fig = plt.figure()
        plot_probstar_reachset_with_unsafeSpec(X, unsafe_mat, unsafe_vec, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=False, \
                               label=('$d$','$v_{ego}$'), show=False, color='g')
        plt.savefig(path+"/Figure_11_{}__d_vs_vego_init_{}.png".format(chr(i+97), i), bbox_inches='tight')  # save figure 7
        if show: plt.show()



def generate_AEBS_Q2_verification_results(initSet_id=0, pf=0.005):
    'generate Q2 verification results'


    # load NNCS AEBS system
    print('Loading the AEBS system...')
    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
    AEBS = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)


    numSteps = 50
    numCores = 4
    # 0.5 <= d_k <= 2.5 and v_k >= 0.2
    unsafe_mat = np.array([[1., 0., 0.], [-1., 0., 0], [0., -1., 0.]])
    unsafe_vec = np.array([2.5, -0.5, -0.2])


    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.numSteps = numSteps

    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]


    verifyPRM.pf = 0.0
    res = verifyBFS_DLNNCS(AEBS, verifyPRM)
    Qt_exact = res.Qt

    verifyPRM.pf = pf
    res = verifyBFS_DLNNCS(AEBS, verifyPRM)
    Qt_approx = res.Qt
    Qt_ub = res.Qt_ub
    Qt_max = res.Qt_max

    t = range(0, 1, numSteps)

    path = "artifacts/NAHS2024_LeCPS/AEBS/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    print('Plot Q2 verification results...')


    fig2 = plt.figure()
    xaxis = np.arange(0, len(Qt_exact))
    yaxis1 = np.array(Qt_exact)
    yaxis2 = np.array(Qt_approx)
    yaxis3 = np.array(Qt_ub)
    yaxis4 = np.array(Qt_max)

    plt.plot(xaxis, yaxis1, color='blue', marker='o', label='$Qt_{exact}$')
    plt.plot(xaxis, yaxis2, color='green', marker='>', label='$Qt_{approx}$')
    plt.plot(xaxis, yaxis3, color='orange', marker='*', label='$Qt_{ub}$')
    plt.plot(xaxis, yaxis4, color='red', marker='x', label='$Qt_{max}$')
    label=('t', '$Qt_{exact}, Qt_{approx}, Qt_{ub}, Qt_{max}$')
    plt.xlabel(label[0], fontsize=20)
    plt.ylabel(label[1], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(path+"/Figure_12_{}__Qt_exact_Qt_approx_Qt_ub_Qt_max_approx_initSet_{}.png".format(chr(initSet_id+97), initSet_id), bbox_inches='tight')  # save figure 8

    print('Done!')



def generate_AEBS_VT_Conv_vs_pf_initSets():
    'generate verification time vesus pf and networks'


    # load NNCS AEBS system
    print('Loading the AEBS system...')
    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
    AEBS = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)


    numSteps = 50
    numCores = 4
    #pf = [0.0, 0.2]
    pf = [0.0, 0.0025, 0.005]

    # 0.5 <= d_k <= 2.5 and v_k >= 0.2
    unsafe_mat = np.array([[1., 0., 0.], [-1., 0., 0], [0., -1., 0.]])
    unsafe_vec = np.array([2.5, -0.5, -0.2])

    d_lb = [97., 90., 48., 5.0]
    d_ub = [97.5, 90.5, 48.5, 5.2]
    v_lb = [25.2, 27., 30.2, 1.0]
    v_ub = [25.5, 27.2, 30.4, 1.2]

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()

    m = len(pf)
    n = len(initSets)
    VT = np.zeros((m, n))
    NO = np.zeros((m,n))
    VT_improv = np.zeros((m, n))  # improvement in verification time
    Conv = np.zeros((m, n))       # conservativeness of prediction
    Qt_ub_sum = np.zeros((m, n))
    Qt_sum = np.zeros((m, n))
    Qt_exact_sum = np.zeros((n,))
    data = []
    for i in range(0, m):
        for j in range(0, n):

            verifyPRM.initSet = copy.deepcopy(initSets[j])
            verifyPRM.numSteps = numSteps
            verifyPRM.pf = pf[i]
            verifyPRM.numCores = numCores
            verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]

            print('Verifying the AEBS system with initSet {} for {} \
            timesteps using approx reachability with pf = {}...'.format(j, numSteps, pf[i]))
            start_time = time.time()
            res = verifyBFS_DLNNCS(AEBS, verifyPRM)
            end_time = time.time()

            RX = res.RX
            Qt = res.Qt
            Qt_ub = res.Qt_ub
            Qt_ub_sum[i,j] = sum(Qt_ub)
            Qt_sum[i,j] = sum(Qt)
            if i==0:
                Qt_exact_sum[j] = sum(Qt)
            M = len(RX)
            NO[i,j] = len(RX[M-1])
            VT[i, j] = end_time - start_time
            VT_improv[i,j] = (VT[0,j] - VT[i, j])*100/VT[0,j]
            Conv[i,j] = 100*(Qt_ub_sum[i,j] - Qt_exact_sum[j])/Qt_exact_sum[j]
            strg1 = '{}'.format(pf[i])
            strg2 = '[{},{}][{},{}]'.format(d_lb[j], d_ub[j], v_lb[j], v_ub[j])
            data.append([strg1, strg2, VT[i,j], NO[i,j], VT_improv[i,j], Conv[i,j]])

    print(tabulate(data, headers=["p_filter", "initSet", "verification time", "number of output sets", "VT_improve (in %)", "Conservativeness (in %)"]))

    path = "artifacts/NAHS2024_LeCPS/AEBS/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/Table_10__VT_Conv_vs_pf_initSet.tex", "w") as f:
         print(tabulate(data, headers=["p_filter", "initSet", "verification time", "number of output sets", "VT_improve", "Conservativeness"], tablefmt='latex'), file=f) # Table 2


    print('Done!')



if __name__ == "__main__":
    # verify Tiny networks
    quantiverify_tiny_network_ReLU(numCores=4) # Figure 5 and Table 1: Tiny ReLU network
    quantiverify_tiny_network_LeakyReLU(numCores=4) # Figure 5 and Table 1: Tiny LeakyReLU network
    quantiverify_tiny_network_SatLin(numCores=4) # Figure 5 and Table 1: Tiny SatLin network
    quantiverify_tiny_network_SatLins(numCores=4) # Figure 5 and Table 1: Tiny SatLins network



    # verify HCAS networks
    x = [0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 
     2, 2, 2, 2, 2, 2, 2, 2, 
     3, 3, 3, 3, 3, 3, 3, 3, 
     4, 4, 4, 4, 4, 4, 4, 4,
     0, 0, 0, 0, 0, 0, 0, 0, 
     1, 1, 1, 1, 1, 1, 1, 1, 
     2, 2, 2, 2, 2, 2, 2, 2, 
     3, 3, 3, 3, 3, 3, 3, 3, 
     4, 4, 4, 4, 4, 4, 4, 4,
     0, 0, 0, 0, 0, 0, 0, 0, 
     1, 1, 1, 1, 1, 1, 1, 1, 
     2, 2, 2, 2, 2, 2, 2, 2, 
     3, 3, 3, 3, 3, 3, 3, 3, 
     4, 4, 4, 4, 4, 4, 4, 4]
    y = [00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60,
        00, 5, 10, 15, 20, 30, 40, 60]
    s = [2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4]
    quantiverify_HCAS_all_ReLU(prev_acv=x, tau=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5]) # Table 2 and 13 (Appendix): HCAS ReLU networks
    quantiverify_HCAS_all_LeakyReLU(prev_acv=x, tau=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5]) # Table 2 and 14 (Appendix): HCAS LeakyReLU networks
    quantiverify_HCAS_all_SatLin(prev_acv=x, tau=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5]) # Table 2 and 15 (Appendix): HCAS SatLin networks
    quantiverify_HCAS_all_SatLins(prev_acv=x, tau=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5]) # Table 2 and 16 (Appendix): HCAS SatLins networks



    # verify ACASXU networks
    x = [1, 2, 2, 3, 3, 3, 4, 4, 5, 1, 1]
    y = [6, 2, 9, 1, 6, 7, 1, 7, 3, 7, 9] 
    s = [2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4] # property id
    quantiverify_ACASXU_ReLU_table_3(x=x, y=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5]) # AcasXu ReLU networks, ProbStar
    numSamplesList = [10000000]
    quantiverify_ACASXU_ReLU_MC_table_3(x=x, y=y, spec_ids=s, unsafe_mat=None, unsafe_vec=None, numSamples=numSamplesList, nTimes=10, numCore=16) # AcasXu ReLU networks, Monte Carlo
    qualiverify_ACASXU_ReLU_other_tools_table_3() # AcasXu ReLU networks, other tools
    generate_table_3_AcasXu_ReLU_quanti_verify_vs_other_tools() # Table 3: Combined AcasXu ReLU networks, ProbStar vs MC vs other tools

    quantiverify_ACASXU_all_LeakyReLU(x=x, y=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5]) # Table 4: AcasXu LeakyReLU networks
    quantiverify_ACASXU_all_SatLin(x=x, y=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[1e-5]) # Table 4: AcasXu SatLin networks
    quantiverify_ACASXU_all_SatLins(x=x, y=y, spec_ids=s, numCores=8, unsafe_mat=None, unsafe_vec=None, p_filters=[1e-5]) # Table 4: AcasXu SatLins networks

    x = [1]
    y = [2] 
    s = [2] # property id
    numSamplesList = [1000, 10000, 100000, 1000000, 10000000] # 10^3-10^7
    quantiverify_ACASXU_all_ReLU_MC(x=x, y=y, spec_ids=s, unsafe_mat=None, unsafe_vec=None, numSamples=numSamplesList, nTimes=10, numCore=16) # Table 5: AcasXu ReLU network 1-2 on property 2, Monte Carlo

    x = [1]
    y = [6] 
    s = [2] # property id
    numSamplesList = [1000, 10000, 100000, 1000000, 10000000, 100000000] # 10^3-10^8
    quantiverify_ACASXU_all_ReLU_MC(x=x, y=y, spec_ids=s, unsafe_mat=None, unsafe_vec=None, numSamples=numSamplesList, nTimes=10, numCore=16) # Table 6: AcasXu ReLU network 1-6 on property 2, Monte Carlo

    x = [5]
    y = [3] 
    s = [2] # property id
    numSamplesList = [1000, 10000, 100000, 1000000, 10000000, 100000000] # 10^3-10^8
    quantiverify_ACASXU_all_ReLU_MC(x=x, y=y, spec_ids=s, unsafe_mat=None, unsafe_vec=None, numSamples=numSamplesList, nTimes=10, numCore=16) # Table 7: AcasXu ReLU networks 5-3 on property 2, Monte Carlo


    x = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1]
    y = [2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 4, 5, 6, 7, 8, 9, 1, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9] 
    s = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4] # property id
    quantiverify_ACASXU_all_ReLU(x=x, y=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5]) # Table 17 (Appendix): AcasXu ReLU networks
    quantiverify_ACASXU_all_LeakyReLU(x=x, y=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5]) # Table 18 (Appendix): AcasXu LeakyReLU networks
    quantiverify_ACASXU_all_SatLin(x=x, y=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[1e-5]) # Table 19 (Appendix): AcasXu SatLin networks
    quantiverify_ACASXU_all_SatLins(x=x, y=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[1e-5]) # Table 20 (Appendix): AcasXu SatLins networks



    # verify RocketNet networks
    quantiverify_RocketNet(numCores=16, net_ids=[0, 1], spec_ids=[1,2], p_filters=[0.0, 1e-8, 1e-5, 1e-3]) # Table 8: RocketNet networks



    # verify ACC model
    generate_exact_Q2_verification_results(net_id='3x20') # Figure 15 (Appendix) (a, b)
    generate_exact_Q2_verification_results(net_id='5x20') # Figure 9 (a, b)
    generate_approx_Q2_verification_results(net_id='3x20', pf=0.01) # Figure 15 (Appendix) (c)
    generate_approx_Q2_verification_results(net_id='5x20', pf=0.01) # Figure 9 (c)
    generate_exact_reachset_figs(net_id='3x20') # Figure 16 (Appendix) (a, b)
    generate_exact_reachset_figs(net_id='5x20') # Figure 10 (a, b, c)
    generate_VT_Conv_vs_pf_net() # Table 9



    # verify AEBS model
    generate_exact_reachset_figs_AEBS() # Figure 11
    generate_AEBS_Q2_verification_results(initSet_id=0, pf=0.005) # Figure 12 (a)
    generate_AEBS_Q2_verification_results(initSet_id=1, pf=0.005) # Figure 12 (b)
    generate_AEBS_Q2_verification_results(initSet_id=2, pf=0.005) # Figure 12 (c)
    generate_AEBS_Q2_verification_results(initSet_id=3, pf=0.005) # Figure 12 (d)
    generate_AEBS_VT_Conv_vs_pf_initSets() # Table 10