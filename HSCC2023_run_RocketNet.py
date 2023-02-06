"""
Verify RocketNet
Author: Dung Tran
Date: 8/10/2022
"""


from StarV.verifier.verifier import  quantiVerifyBFS
from StarV.set.probstar import ProbStar
import numpy as np

from StarV.util.load import load_DRL
import time
from matplotlib import pyplot as plt
from StarV.set.star import Star
from tabulate import tabulate
import os

      
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
            path = "artifacts/HSCC2023/rocketNet"
            if not os.path.exists(path):
                os.makedirs(path)

            plt.savefig(path+"/rocketNet_{}_spec_{}.png".format(net_id, spec_id), bbox_inches='tight')  # save figure


    print(tabulate(data, headers=["Network_ID", "Property", "p_filter", "OutputSet", "UnsafeOutputSet", "counterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "InputSetProbability", "VerificationTime"]))



    with open(path+"/rocketNetTable.tex", "w") as f:
        print(tabulate(data, headers=["Network_ID", "Property","p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                          "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)


    return data

        

if __name__ == "__main__":

    quantiverify_RocketNet(numCores=8, net_ids=[0, 1], spec_ids=[1,2], p_filters=[0.0, 1e-8, 1e-5, 1e-3])
