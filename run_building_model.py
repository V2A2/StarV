"""

"""


from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load_model import load_building_model
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
 



def run_building_model_krylov(use_arnoldi =None,use_init_space=None):
        print('=====================================================')
        print('Quantitative Verification of Building Model Using Krylov Subspace')
        print('=====================================================')
        plant = load_building_model()
        combined_mat = combine_mats(plant.A,plant.B)
        dims = combined_mat.shape[0]
        print("combined_dim:",dims)

        
        #  returns list of initial states for each dimension
        init_state_bounds_list = []
        initial_dim = plant.A.shape[0]

        for dim in range(dims):
            if dim < 10:
                lb = 0.0002
                ub = 0.00025
            elif dim == 24:
                lb = -0.0001
                ub = 0.0001
            elif dim < initial_dim: 
                lb = ub = 0 
            elif dim >= initial_dim:
                lb = 0.8
                ub = 1
            else:
                raise RuntimeError('Unknown dimension: {}'.format(dim))
            
            init_state_bounds_list.append((lb, ub))

        # init_sate_bounds_array=[np.array(list).reshape(48, 1) for list in init_sate_bounds_list]
        init_state_bounds_array = np.array(init_state_bounds_list)

        init_state_lb = init_state_bounds_array[:, 0]
        init_state_ub = init_state_bounds_array[:, 1]


        X0 = Star(init_state_lb,init_state_ub)
      
        # create ProbStar for initial state 
        mu_U = 0.5*(X0.pred_lb + X0.pred_ub)
        a  = 3
        sig_U = (mu_U - X0.pred_lb)/a
        Sig_U = np.diag(np.square(sig_U))
        X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)


        h = [0.1,0.01,0.001]
        time_bound = 20
        m = 4
        target_error = 1e-6
        samples = 51
        tolerance = 1e-9

        output_space = plant.C
        expand_mat = np.zeros((1,1))
        output_space = np.hstack((output_space,expand_mat))
        initial_space = X0_probstar.V  

        if use_init_space ==True:
            i = output_space.shape[1]
        else:
            i = output_space.shape[0]

        unsafe_mat_list = np.array([[-1]])
        unsafe_vec_list =np.array([-0.004])
        inputProb = X0_probstar.estimateProbability()

        data = []
        
        for hi in h:
            N = int (time_bound/ hi)
            reach_start_time = time.time()

            R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
            reach_time_duration = time.time() - reach_start_time
            plot_1D_Star_time(R,20,0.001,safety_value=0.004 )


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
        path = "artifacts/LCS/building"
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path+"/building.tex", "a") as f:
            if use_init_space == True:
                print(tabulate(data, headers=[ "i","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"], tablefmt='latex'), file=f)
            else:
                print(tabulate(data, headers=[ "o","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"], tablefmt='latex'), file=f)


        return R

     

if __name__ == '__main__':


    Xt = run_building_model_krylov(use_arnoldi = True,use_init_space=True) 



