"""

"""


from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load import load_mcs_model
import time
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
import os
from StarV.util.plot import  plot_probstar,plot_star,plot_1D_Star,plot_1D_Star_time
from StarV.verifier.krylov_func.simKrylov_with_projection import simReachKrylov as sim3
from StarV.verifier.krylov_func.simKrylov_with_projection import combine_mats,random_two_dims_mapping
from StarV.verifier.krylov_func.LCS_verifier import quantiVerifier_LCS
from tabulate import tabulate
 
 

def run_mcs_model_Krylov(use_arnoldi = None,use_init_space=None):
        
    print('=====================================================')
    print('Quantitative Verification of MCS Model Using Krylov Subspace')
    print('=====================================================')

    plant = load_mcs_model()
    combined_mat = combine_mats(plant.A,plant.B)
    dims = combined_mat.shape[0]

    #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    initial_dim = plant.A.shape[0]
    for dim in range(dims):
        if dim == 1:
            lb =0.002
            ub = 0.0025
        elif dim == 2:
            lb =0.001
            ub = 0.0015    
        elif dim < initial_dim:
            lb = ub = 0 
        elif dim == initial_dim :
            # first input
            lb = 0.16
            ub = 0.3
        elif dim > initial_dim:
            # second input
            lb = 0.2
            ub = 0.4
        else:         
            raise RuntimeError('Unknown dimension: {}'.format(dim))        
        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]


    # create Star for initial state 
    X0 = Star(init_state_lb,init_state_ub)


    mu_X0 = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_X0 = (mu_X0 - X0.pred_lb)/a
    Sig_X0 = np.diag(np.square(sig_X0))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_X0, Sig_X0,X0.pred_lb,X0.pred_ub)


    h = [0.1,0.01,0.001]
    time_bound = 20
    m = 4
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9


    output_space = plant.C
    expand_mat = np.zeros((2,2))
    output_space = np.hstack((output_space,expand_mat))
    initial_space = X0_probstar.V 

    if use_init_space ==True:
        i = output_space.shape[1]

    else:
        i = output_space.shape[0]


    unsafe_mat_list = np.array([[-1,0],[1,0],[0,-1],[0,1]])
    unsafe_vec_list =np.array([-0.3,0.4,-0.4,0.6])

    inputProb = X0_probstar.estimateProbability()
   
    data = []
    for hi in h:
        N = int (time_bound/ hi)
        reach_start_time = time.time()

        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time
        plot_probstar(R,safety_value=0.3)


        p_min,smallest_prob_time_step, p_max, largest_prob_time_step,unsafeOutputSet, counterInputSet= quantiVerifier_LCS(R = R, inputSet=X0_probstar, unsafe_mat=unsafe_mat_list, \
                                                                                unsafe_vec=unsafe_vec_list,time_step=hi)
        verify_time_duration = time.time() - reach_start_time


        data.append([i,hi,len(R), len(unsafeOutputSet), len(counterInputSet),p_min, smallest_prob_time_step,p_max,largest_prob_time_step, inputProb, krylov_time,reach_time_duration,verify_time_duration])
        

    # print verification results
    if use_init_space ==True:
        print(tabulate(data, headers=[ "i","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"]))
    else:
        print(tabulate(data, headers=[ "o","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"]))

    # save verification results
    path = "artifacts/LCS/MCS"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/MCS.tex", "a") as f:
        if use_init_space == True:
            print(tabulate(data, headers=[ "i","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"], tablefmt='latex'), file=f)
        else:
            print(tabulate(data, headers=[ "o","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"], tablefmt='latex'), file=f)


    return R
     



if __name__ == '__main__':


    Xt = run_mcs_model_Krylov(use_arnoldi = True,use_init_space=True)
 