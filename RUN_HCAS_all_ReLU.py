"""
Verify HCAS networks ReLU all table
Author: Yuntao Li
Date: 2/10/2024
"""


from StarV.verifier.verifier import quantiVerifyBFS
from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load_relu import load_HCAS_ReLU
import time
from StarV.set.star import Star
from tabulate import tabulate
import os
from StarV.util.print_util import print_util


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
    path = "artifacts/Journal/HCAS/ReLU"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/HCASTable_all.tex", "w") as f:
        print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                            "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
    
    print_util('h2')
    return data


if __name__ == "__main__":

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

    quantiverify_HCAS_all_ReLU(prev_acv=x, tau=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5])

    
    # property id: 2
    # x = [0, 0, 0, 0, 0, 0, 0, 0]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [2, 2, 2, 2, 2, 2, 2, 2] # property id

    # x = [1, 1, 1, 1, 1, 1, 1, 1]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [2, 2, 2, 2, 2, 2, 2, 2] # property id

    # x = [2, 2, 2, 2, 2, 2, 2, 2]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [2, 2, 2, 2, 2, 2, 2, 2] # property id

    # x = [3, 3, 3, 3, 3, 3, 3, 3]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [2, 2, 2, 2, 2, 2, 2, 2] # property id

    # x = [4, 4, 4, 4, 4, 4, 4, 4]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [2, 2, 2, 2, 2, 2, 2, 2] # property id

    # # property id: 3
    # x = [0, 0, 0, 0, 0, 0, 0, 0]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [3, 3, 3, 3, 3, 3, 3, 3] # property id

    # x = [1, 1, 1, 1, 1, 1, 1, 1]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [3, 3, 3, 3, 3, 3, 3, 3] # property id

    # x = [2, 2, 2, 2, 2, 2, 2, 2]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [3, 3, 3, 3, 3, 3, 3, 3] # property id

    # x = [3, 3, 3, 3, 3, 3, 3, 3]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [3, 3, 3, 3, 3, 3, 3, 3] # property id

    # x = [4, 4, 4, 4, 4, 4, 4, 4]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [3, 3, 3, 3, 3, 3, 3, 3] # property id

    # # property id: 2
    # x = [0, 0, 0, 0, 0, 0, 0, 0]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [4, 4, 4, 4, 4, 4, 4, 4] # property id

    # x = [1, 1, 1, 1, 1, 1, 1, 1]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [4, 4, 4, 4, 4, 4, 4, 4] # property id

    # x = [2, 2, 2, 2, 2, 2, 2, 2]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [4, 4, 4, 4, 4, 4, 4, 4] # property id

    # x = [3, 3, 3, 3, 3, 3, 3, 3]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [4, 4, 4, 4, 4, 4, 4, 4] # property id

    # x = [4, 4, 4, 4, 4, 4, 4, 4]
    # y = [00, 5, 10, 15, 20, 30, 40, 60]
    # s = [4, 4, 4, 4, 4, 4, 4, 4] # property id