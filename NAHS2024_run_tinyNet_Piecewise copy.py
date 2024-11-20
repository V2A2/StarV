"""
Verify Tiny Network (ReLU, LeakyReLU, SatLin, SatLins)
Author: Yuntao Li
Date: 2/10/2024
"""


from StarV.verifier.verifier import quantiVerifyBFS
from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load_piecewise import load_tiny_network_ReLU, load_tiny_network_LeakyReLU, load_tiny_network_SatLin, load_tiny_network_SatLins, load_tiny_network_SatLins
import time
from StarV.util.plot import plot_probstar
from matplotlib import pyplot as plt
from StarV.set.star import Star
from tabulate import tabulate
import os
from StarV.util.print_util import print_util




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
    path = "artifacts/NAHS2024/tinyNet/ReLU/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/tinyNetTable.tex", "w") as f:
        print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                        "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)


    # plot reachable sets and unsafe reachable sets
    OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                                unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
    
    for output_set in OutputSet:
        output_set.__str__()
    for unsafe_output_set in unsafeOutputSet:
        unsafe_output_set.__str__()
    for counter_input_set in counterInputSet:
        counter_input_set.__str__()

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
    path = "artifacts/NAHS2024/tinyNet/LeakyReLU/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/tinyNetTable.tex", "w") as f:
        print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                        "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)


    # plot reachable sets and unsafe reachable sets
    OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                                unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
    

    for output_set in OutputSet:
        output_set.__str__()
    for unsafe_output_set in unsafeOutputSet:
        unsafe_output_set.__str__()
    for counter_input_set in counterInputSet:
        counter_input_set.__str__()
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
    path = "artifacts/NAHS2024/tinyNet/SatLin/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/tinyNetTable.tex", "w") as f:
        print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                        "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)


    # plot reachable sets and unsafe reachable sets
    OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                                unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
    
    for output_set in OutputSet:
        output_set.__str__()
    for unsafe_output_set in unsafeOutputSet:
        unsafe_output_set.__str__()
    for counter_input_set in counterInputSet:
        counter_input_set.__str__()
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
    path = "artifacts/NAHS2024/tinyNet/SatLins/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/tinyNetTable.tex", "w") as f:
        print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                        "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)


    # plot reachable sets and unsafe reachable sets
    OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                                unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
    

    for output_set in OutputSet:
        output_set.__str__()
    for unsafe_output_set in unsafeOutputSet:
        unsafe_output_set.__str__()
    for counter_input_set in counterInputSet:
        counter_input_set.__str__()
    print_util('h3')
    print('DONE!')
    print_util('h3')
        

if __name__ == "__main__":

    quantiverify_tiny_network_ReLU(numCores=1)

    quantiverify_tiny_network_LeakyReLU(numCores=1)

    quantiverify_tiny_network_SatLin(numCores=1)

    quantiverify_tiny_network_SatLins(numCores=1)

    quantiverify_tiny_network_ReLU(numCores=4)

    quantiverify_tiny_network_LeakyReLU(numCores=4)

    quantiverify_tiny_network_SatLin(numCores=4)

    quantiverify_tiny_network_SatLins(numCores=4)