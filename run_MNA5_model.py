"""

"""


from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load import load_MNA5_model
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
 


def run_MNA5_model_krylov(use_arnoldi=None,use_init_space=None):
       
    print('=====================================================')
    print('Quantitative Verification of MNA5 Model Using Krylov Subspace')
    print('=====================================================')

    plant = load_MNA5_model()
    combined_mat = combine_mats(plant.A,plant.B)
    dims = combined_mat.shape[0]


    #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    initial_dim = plant.A.shape[0]
    for dim in range(dims):
        if dim < 10:
            lb = 0.0002
            ub = 0.00025
        elif dim < initial_dim:
            lb = ub = 0
        elif dim >= initial_dim and dim < initial_dim + 5:
            # first 5 inputs
            lb = ub = 0.1
        elif dim >= initial_dim + 5 and dim < initial_dim + 9:
            # second 4 inputs
            lb = ub = 0.2
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))
            
        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)
    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    # create Star for initial state 
    X0 = Star(init_state_lb,init_state_ub)
  
    # create ProbStar for initial state 
    mu_U = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_U = (mu_U - X0.pred_lb)/a
    epsilon = 1e-10
    sig_U = np.maximum(sig_U, epsilon)
    Sig_U = np.diag(np.square(sig_U))
    # U_probstar = ProbStar(U.V, U.C, U.d,mu_U, Sig_U,U.pred_lb,U.pred_ub)
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)


    h = [0.1]
    time_bound = 20
    m = 10
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9

    output_space = random_two_dims_mapping(X0_probstar,1,2)
    initial_space = X0_probstar.V

    if use_init_space == True:
        i = output_space.shape[1]
    else:
        i = output_space.shape[0]

    unsafe_mat_list = np.array([[-1,0]])
    unsafe_vec_list =np.array([-0.1])

    inputProb = X0_probstar.estimateProbability()
   
    data = []

    for hi in h:
        N = int (time_bound/ hi)
        reach_start_time = time.time()

        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time
        # plot_probstar(R,safety_value=0.1)


        prob,p_min,smallest_prob_time_step, p_max, largest_prob_time_step,unsafeOutputSet, counterInputSet= quantiVerifier_LCS(R = R, inputSet=X0_probstar, unsafe_mat=unsafe_mat_list, \
                                                                                unsafe_vec=unsafe_vec_list,time_step=hi)
        verify_time_duration = time.time() - reach_start_time


        data.append([i,hi,len(R), len(unsafeOutputSet), len(counterInputSet),p_min, smallest_prob_time_step,p_max,largest_prob_time_step, inputProb, krylov_time,reach_time_duration,verify_time_duration])
        

    # print verification results
    if use_init_space ==True:
        print(tabulate(data, headers=[ "i","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"]))
    else:
        print(tabulate(data, headers=[ "o","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"]))

    # save verification results
    path = "artifacts/LCS/MNA5"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/MNA5.tex", "a") as f:
        if use_init_space == True:
            print(tabulate(data, headers=[ "i","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"], tablefmt='latex'), file=f)
        else:
            print(tabulate(data, headers=[ "o","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"], tablefmt='latex'), file=f)




    return R
     
# def run_MNA5_model_krylov_projection(use_arnoldi=None,use_init_space=None):
#         #--------------------krylov method -------------------------

#     plant = load_MNA5_model()
#     combined_mat = combine_mats(plant.A,plant.B)
#     dims = combined_mat.shape[0]


#     #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
#     init_state_bounds_list = []
#     initial_dim = plant.A.shape[0]
#     for dim in range(dims):
#         if dim < 10:
#             lb = 0.0002
#             ub = 0.00025
#         elif dim < initial_dim:
#             lb = ub = 0
#         elif dim >= initial_dim and dim < initial_dim + 5:
#             # first 5 inputs
#             lb = ub = 0.1
#         elif dim >= initial_dim + 5 and dim < initial_dim + 9:
#             # second 4 inputs
#             lb = ub = 0.2
#         else:
#             raise RuntimeError('Unknown dimension: {}'.format(dim))
            
#         init_state_bounds_list.append((lb, ub))

#     init_state_bounds_array = np.array(init_state_bounds_list)
#     init_state_lb = init_state_bounds_array[:, 0]
#     init_state_ub = init_state_bounds_array[:, 1]

#     # create Star for initial state 
#     X0 = Star(init_state_lb,init_state_ub)


#     # create ProbStar for initial state 
#     mu_U = 0.5*(X0.pred_lb + X0.pred_ub)
#     a  = 3
#     sig_U = (mu_U - X0.pred_lb)/a
#     epsilon = 1e-10
#     sig_U = np.maximum(sig_U, epsilon)
#     Sig_U = np.diag(np.square(sig_U))
#     # U_probstar = ProbStar(U.V, U.C, U.d,mu_U, Sig_U,U.pred_lb,U.pred_ub)
#     X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)

    
#     h = [0.1]
#     time_bound = 20
#     # N = int (time_bound/ h)
#     # print("num_steps_N:",N)
#     m = 10
#     target_error = 1e-6
#     samples = 51
#     tolerance = 1e-9


#     output_space = random_two_dims_mapping(X0_probstar,1,2)
#     initial_space = X0_probstar.V

#     if use_init_space == True:
#         i = output_space.shape[1]
#     else:
#         i = output_space.shape[0]

#     unsafe_mat_list = np.array([[-1,0]])
#     unsafe_vec_list =np.array([-0.1])

#     # if use_init_space == True:
#     #     # use init space as init vectors 
#     #     key_dir_mat = random_two_dims_mapping(X0_probstar,1,2)
#     #     output_mat = X0_probstar.V
#     #     i = output_mat.shape[1]
#     # else:
#     #     # use output_mat as init vectors
#     #     key_dir_mat = X0_probstar.V.T 
#     #     output_mat = random_two_dims_mapping(X0_probstar,1,2)
#     #     # output_mat = np.zeros((1,dims))
#     #     output_mat[0,0] =1
#     #     i = output_mat.shape[0]

#     # unsafe_mat = np.array([[-1, 0]])
#     # unsafe_mat_list = [unsafe_mat]
#     # unsafe_vec = np.array([-0.1]) 
#     # unsafe_vec_list = [unsafe_vec]

#     unsafe_mat_list = np.array([[-1,0]])
#     unsafe_vec_list =np.array([-0.1])

#     inputProb = X0_probstar.estimateProbability()
   
#     data = []

#     for hi in h:
#         N = int (time_bound/ hi)
#         reach_start_time = time.time()

#         R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
#         # R = sim3(A,U,h, N, m,samples,tolerance,target_error=target_error,key_dir_mat=key_dir_mat,output_mat=output_mat,use_arnoldi = True,use_init_space=False)
#         # R = sim1(A,X0_star,h, N, m,samples,target_error=target_error)
#         reach_time_duration = time.time() - reach_start_time
#         # plot_probstar(R,prob=None,safety_value=4)


#         prob,p_min,smallest_prob_time_step, p_max, largest_prob_time_step,unsafeOutputSet, counterInputSet= quantiVerifier_LCS(R = R, inputSet=X0_probstar, unsafe_mat=unsafe_mat_list, \
#                                                                                 unsafe_vec=unsafe_vec_list,time_step=hi)
#         verify_time_duration = time.time() - reach_start_time


#         data.append([i,hi,len(R), len(unsafeOutputSet), len(counterInputSet),p_min, smallest_prob_time_step,p_max,largest_prob_time_step, inputProb, krylov_time,reach_time_duration,verify_time_duration])
        

#     # print verification results
#     if use_init_space ==True:
#         print(tabulate(data, headers=[ "i","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"]))
#     else:
#         print(tabulate(data, headers=[ "o","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"]))

#     # save verification results
#     path = "artifacts/LCS/MNA5"
#     if not os.path.exists(path):
#         os.makedirs(path)

#     with open(path+"/MNA5.tex", "a") as f:
#         if use_init_space == True:
#             print(tabulate(data, headers=[ "i","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"], tablefmt='latex'), file=f)
#         else:
#             print(tabulate(data, headers=[ "o","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"], tablefmt='latex'), file=f)




#     return R
     
  



     

if __name__ == '__main__':
    

    Xt  = run_MNA5_model_krylov(use_arnoldi = True,use_init_space=False)
