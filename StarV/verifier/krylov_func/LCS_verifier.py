
import numpy as np
import time
import copy
import numpy as np
from StarV.set.probstar import ProbStar
from StarV.verifier.verifier import checkSafetyProbStar


def quantiVerifier_LCS(R, inputSet,unsafe_mat, unsafe_vec,time_step):
    
    C =[]
    P = []  # unsafe output set
    prob = []  # probability of unsafe output set
    R_index = [] # list to store the corresponding indices in R

    verify_time = time.time()

  # handle one output constraints
    if len(unsafe_vec) ==1 and isinstance(unsafe_mat,np.ndarray):
        for i in range(1,len(R)):
            P1,prob1 = checkSafetyProbStar(unsafe_mat, unsafe_vec,R[i])
            if prob1 > 0 and not P1.isEmptySet():
                P.append(P1)
                C1 = ProbStar(inputSet.V, P1.C, P1.d, P1.mu, P1.Sig, P1.pred_lb, P1.pred_ub)
                C.append(C1)
                prob.append(prob1)
                R_index.append(i)



    # handle multiple output constraints, satisfy both conditions
    if len(unsafe_vec) > 1 and isinstance(unsafe_mat,np.ndarray):
        for i in range(1,len(R)):
            P1 = R[i].addMultipleConstraintsWithoutUpdateBounds(unsafe_mat,unsafe_vec)
            prob1= P1.estimateProbability()
            if prob1 > 0 and not P1.isEmptySet():
                P.append(P1)
                C1 = ProbStar(inputSet.V, P1.C, P1.d, P1.mu, P1.Sig, P1.pred_lb, P1.pred_ub)
                C.append(C1)
                prob.append(prob1)
                R_index.append(i)

   # handle multiple output constraints, satisfy one of conditions
    if isinstance(unsafe_vec,list):
        R_index = []
        for i in range(1,len(R)):
            for j in range(len(unsafe_mat)):
                P1,prob1= checkSafetyProbStar(unsafe_mat[j], unsafe_vec[j],R[i])
                if prob1 > 0 and not P1.isEmptySet():
                        P.append(P1)
                        prob.append(prob1)
                        if i not in R_index:
                            R_index.append(i)
                            C1 = ProbStar(inputSet.V, P1.C, P1.d, P1.mu, P1.Sig, P1.pred_lb, P1.pred_ub)
                            C.append(C1)


    verify_time = time.time()- verify_time

    if prob:  # check if prob is not empty
        p_min_idx = np.argmin(prob)  
        if  R_index[p_min_idx] == 0 and len(prob)>1:# check if the first R's prob is 0, then skip this
            p_min_idx= np.argmin(prob[1:]) + 1
        R_index_min = R_index[p_min_idx]  # index in the R list
        p_min = prob[p_min_idx]
        smallest_prob_time_step = R_index_min* time_step

        p_max_idx= np.argmax(prob) 
        R_index_max = R_index[p_max_idx] 
        p_max = prob[p_max_idx] 
        largest_prob_time_step = R_index_max* time_step
    else:
        p_min = None
        smallest_prob_time_step = None
        p_max = None
        largest_prob_time_step =None
        print("No unsafe Reacable ProbStar sets found")



    print('Number of output unsafe ProbStar: P = {}'.format(len(P)))
    print('The samllest probility of all unsafe ProbStar: Prob = {}'.format(p_min))
    print('The biggest probility of all unsafe ProbStar: Prob = {}'.format(p_max))
    print('The time of samllest probility of all unsafe ProbStar: t_min = {}'.format(smallest_prob_time_step))
    print('The time of biggest probility of all unsafe ProbStar: t_max = {}'.format(largest_prob_time_step))

        
    return  prob,p_min,smallest_prob_time_step, p_max, largest_prob_time_step,P, C

