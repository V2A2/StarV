
import math
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


def ha_example(use_arnoldi =None,use_init_space=None):
    A = np.array([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,0,0]])
    h = math.pi/4
    N = int((math.pi)/h)
    m = 2
    target_error = 1e-9
    tolerance = 1e-9
    samples = 51

    #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    dims = A.shape[0]
    for dim in range(dims):
        if dim == 0: # x == -5
            lb = -6
            ub = -5
        elif dim == 1: # y in [0, 1]
            lb = 0
            ub = 1
        elif dim == 2: # t == 0
            lb = 0
            ub = 0
        elif dim == 3: # a == 1
            lb = 1
            ub = 1
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))

        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    X0 = Star(init_state_lb,init_state_ub)
   
   
    # create ProbStar for initial state 
    mu_U = 0.5*(X0.pred_lb + X0.pred_ub) 
    a  = 3
    sig_U = (X0.pred_ub-mu_U )/a
    epsilon = 1e-10
    sig_U = np.maximum(sig_U, epsilon)
    Sig_U = np.diag(np.square(sig_U))

    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)

    output_space = random_two_dims_mapping(X0_probstar,1,2) # C =([1,0,0,0],[0,1,0,0])
    # output_space = np.array([[1,0,0,0]])  # C =([1,0,0,0],[0,1,0,0])
    initial_space = X0_probstar.V # 

    if use_init_space ==True:

        i = output_space.shape[1]
    else:
        i = output_space.shape[0]


    unsafe_mat_list = np.array([[-1,0]])
    unsafe_vec_list =np.array([-4])

    inputProb = X0_probstar.estimateProbability()

    reach_start_time = time.time()
    R, krylov_time = sim3(A,X0_probstar,h, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)

    reach_time_duration = time.time() - reach_start_time
    plot_probstar(R,safety_value=4)
            

    # verify_start_time = time.time()
    p_min,smallest_prob_time_step, p_max, largest_prob_time_step,unsafeOutputSet, counterInputSet= quantiVerifier_LCS(R = R, inputSet=X0_probstar, unsafe_mat=unsafe_mat_list, \
                                                                            unsafe_vec=unsafe_vec_list,time_step=h)
    verify_time_duration = time.time() - reach_start_time
    plot_probstar(unsafeOutputSet,safety_value=4) 


    first_counter = counterInputSet[0]
    V2= output_space @ first_counter.V
    C1 = ProbStar(V2,first_counter.C,first_counter.d, first_counter.mu, first_counter.Sig,first_counter.pred_lb,first_counter.pred_ub)
    plot_probstar(C1)


    data = []

    data.append([i,h,len(R), len(unsafeOutputSet), len(counterInputSet),p_min, smallest_prob_time_step,p_max,largest_prob_time_step, inputProb, krylov_time,reach_time_duration,verify_time_duration])
    

    # print verification results
    if use_init_space ==True:
        print(tabulate(data, headers=[ "i","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"]))
    else:
        print(tabulate(data, headers=[ "o","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"]))

    # save verification results
    path = "artifacts/LCS/Hamonic"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/Hamonic.tex", "a") as f:
        if use_init_space == True:
            print(tabulate(data, headers=[ "i","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"], tablefmt='latex'), file=f)
        else:
            print(tabulate(data, headers=[ "o","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"], tablefmt='latex'), file=f)



    return R
     

if __name__ == '__main__':

    Xt = ha_example(use_arnoldi = True,use_init_space=True) 
  