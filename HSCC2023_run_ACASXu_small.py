"""
Test verifier module
Author: Dung Tran
Date: 8/10/2022
"""


from StarEV.verifier.verifier import quantiVerifyBFS
from StarEV.set.probstar import ProbStar
import numpy as np
from StarEV.util.load import load_ACASXU
import time
from StarEV.set.star import Star
from tabulate import tabulate
import os



def quantiverify_ACASXU_all(self, x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
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
    path = "artifacts/HSCC2023/ACASXU"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/ACASXUTable_small.tex", "w") as f:
        print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                              "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)


    return data

        

if __name__ == "__main__":
   
    x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y = [2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9] 
    s = [2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4] # property id
    quantiverify_ACASXU_all(x=x, y=y, spec_ids=s, numCores=8, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5])

    
