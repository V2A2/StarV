import math
from scipy.io import loadmat
import numpy as np
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
 

def make_heat_dia(samples, diffusity_const, heat_exchange_const):
    '''fast dense matrix construction for heat3d dynamics'''

    samples_sq = samples**2
    dims = samples**3
    step = 1.0 / (samples + 1)

    a = diffusity_const * 1.0 / step**2
    d = -6.0 * a  # Since we have six neighbors in 3D

    # Initialize dense matrix
    matrix = np.zeros((dims, dims))

    for i in range(dims):
        z = i // samples_sq
        y = (i % samples_sq) // samples
        x = i % samples

        if z > 0:
            matrix[i, i - samples_sq] = a  # Interaction with the point below
        if y > 0:
            matrix[i, i - samples] = a     # Interaction with the point in front
        if x > 0:
            matrix[i, i - 1] = a           # Interaction with the point to the left

        matrix[i, i] = d

        if z == 0 or z == samples - 1:
            matrix[i, i] += a  # Boundary adjustment for z-axis
        if y == 0 or y == samples - 1:
            matrix[i, i] += a  # Boundary adjustment for y-axis
        if x == 0:
            matrix[i, i] += a  # Boundary adjustment for x=0
        if x == samples - 1:
            matrix[i, i] += a / (1 + heat_exchange_const * step)  # Boundary adjustment for x=samples-1

        if x < samples - 1:
            matrix[i, i + 1] = a           # Interaction with the point to the right
        if y < samples - 1:
            matrix[i, i + samples] = a     # Interaction with the point behind
        if z < samples - 1:
            matrix[i, i + samples_sq] = a  # Interaction with the point above

    return matrix

def make_init_star(a,samples):
    '''returns a Star '''

    dims = a.shape[0]

    data = []
    inds = []

    assert samples >= 10 and samples % 10 == 0

    for z in range(int(samples / 10 + 1)):
        zoffset = z * samples * samples

        for y in range(int(2 * samples / 10 + 1)):
            yoffset = y * samples

            for x in range(int(4 * samples / 10 + 1)):
                dim = x + yoffset + zoffset

                data.append(1)
                inds.append(dim)

    init_space = np.zeros((dims, 1))
    for i in inds:
        init_space[i, 0] = 1

    init_state_bounds_list = []

    for dim in range(dims):
       if init_space[dim] == 1:
            lb = 0.9
            ub = 1.1
            init_state_bounds_list.append((lb, ub))
       else:
            lb = 0
            ub = 0
            init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    X0 = Star(init_state_lb,init_state_ub)

    return X0


def run_hest_sym(use_arnoldi=None,use_init_space=None):

    print('=====================================================')
    print('Quantitative Verification of Sysmmetric Heat 3D Model Using Krylov Subspace')
    print('=====================================================')
   
    diffusity_const = 0.01
    heat_exchange_const = 0.5
    samples_per_side = 10
    dims =samples_per_side**3



    print ("Making {}x{}x{} ({} dims) 3d Heat Plate ODEs...".format(samples_per_side, samples_per_side, \
                                                                samples_per_side, samples_per_side**3))
    a_matrix = make_heat_dia(samples_per_side, diffusity_const, heat_exchange_const)

    # initial input probstar set        
    X0= make_init_star(a_matrix,samples_per_side)
    
    mu_X0 = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_X0 = (mu_X0 - X0.pred_lb)/a
    Sig_X0 = np.diag(np.square(sig_X0))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_X0, Sig_X0,X0.pred_lb,X0.pred_ub)

    h = [0.1,0.01,0.001]
    time_bound = 20
    m = 8
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9

    center_x = int(math.floor(samples_per_side/2.0))
    center_y = int(math.floor(samples_per_side/2.0))
    center_z = int(math.floor(samples_per_side/2.0))
    center_dim = center_z * samples_per_side * samples_per_side + center_y * samples_per_side + center_x


    output_space = np.zeros((1,dims))
    output_space[0,center_dim] = 1
    initial_space = X0.V # 

    if use_init_space == True:
        i = output_space.shape[1]
    else:
        i = output_space.shape[0]


    unsafe_mat_list = np.array([[-1]])
    unsafe_vec_list =np.array([-0.012])

    inputProb = X0_probstar.estimateProbability()
    data = []
    

    for hi in h:
        N = int (time_bound/ hi)
        reach_start_time = time.time()

        R, krylov_time = sim3(a_matrix,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time
        plot_1D_Star_time(R,time_bound,hi,0.012)

        p_min,smallest_prob_time_step, p_max, largest_prob_time_step,unsafeOutputSet, counterInputSet= quantiVerifier_LCS(R = R, inputSet=X0_probstar, unsafe_mat=unsafe_mat_list, \
                                                                                unsafe_vec=unsafe_vec_list,time_step=hi)
        verify_time_duration = time.time() - reach_start_time
        print("======verification time===========:",verify_time_duration)
        data.append([i,hi,len(R), len(unsafeOutputSet), len(counterInputSet),p_min, smallest_prob_time_step,p_max,largest_prob_time_step, inputProb, krylov_time,reach_time_duration,verify_time_duration])
        

        # print verification results
    if use_init_space ==True:
        print(tabulate(data, headers=[ "i","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"]))
    else:
        print(tabulate(data, headers=[ "o","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"]))

    # save verification results
    path = "artifacts/LCS/Heat"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/Heat.tex", "a") as f:
        if use_init_space == True:
            print(tabulate(data, headers=[ "i","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"], tablefmt='latex'), file=f)
        else:
            print(tabulate(data, headers=[ "o","TimeStep","ReachableSet", "UnsafeReachableSet", "CounterInputSet", "US-prob-Min","US-prob-Min-Timestep","US-prob-Max", "US-prob-Max-Timestep","inputSet Probability", "Krylov-Time","ReachabilityTime","VerificationTime"], tablefmt='latex'), file=f)




        return R

if __name__ == '__main__':


    Xt = run_hest_sym(use_arnoldi=True,use_init_space=False)



