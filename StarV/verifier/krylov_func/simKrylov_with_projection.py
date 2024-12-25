# from scipy.sparse.linalg import norm 
import numpy as np
from scipy.sparse.linalg import eigsh
# np.set_printoptions(precision=10000)
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from numpy.linalg import norm
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
import math
from StarV.util.plot import  plot_probstar,plot_star
import time
from StarV.verifier.krylov_func.krylov_compute_error import compute_posteriori_error
from StarV.verifier.krylov_func.arnoldi_lanzcos import arnoldi,lanczos


def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T)

def combine_mats(a,b):
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape
    new_size = a_rows + b_cols
    combined_mat = np.zeros((new_size, new_size))
    combined_mat[:a_rows, :a_cols] = a
    combined_mat[:b_rows, a_cols:a_cols+b_cols] = b
    combined_mat[a_rows:a_rows+b_rows, :new_size] = 0
    return combined_mat

def simKrylov(A,x0,m,h, N, V, H):
   
    # Initialize the list to store results
    X = []
    
    # Compute expm(H0) which is the initial step 0.
    H0 = 0 * -h * H # the initial step
    expm_H0 = expm(H0)
    e1 = expm_H0[:, 0]  # e{At} * e1
    a = np.dot(V, e1) # V*e{At}*e1
    # Project and store the initial state (step 0)
    X.append(a)
    
    # Compute the first state after initial time step
    H1 = 1 * -h* H
    expm_H1 = expm(H1)
    e1 = np.dot(expm_H1, e1) # e{At} * e1
    a = np.dot(V,e1)  # V*e{At}*e1
    X.append(a)
    
    # Start simulation from step 2 to N
    print("==========Start simKrylov===========")
    for i in range(2, N + 1):
        e1 = np.dot(expm_H1, e1)  # Update state
        X.append(np.dot(V, e1)) 
    
    return X


def sim_with_error(A,x0,h,N,m,samples,tolerance,error_limit=None,projection_mat = None,use_arnoldi = None):
    '''
    Run an arnoldi simulation with a fixed number of iterations and a target max error.
    If error_limit is None, just run the whole simulation

    returns a 2-tuple (a, b) with:
    a: projected simulation at each step, or None if the error limit was exceeded.
    b: the number of arnoldi iterations actually used
    '''
    error = None
    iteration_time = time.time()
    if use_arnoldi == True:
        V, H = arnoldi(A, x0, m, tolerance,compute_error = True,projection_mat = projection_mat)
    else:
        V, H = lanczos(A, x0, m, tolerance,compute_error = True,projection_mat = projection_mat)
    krylov_time_duration = time.time() - iteration_time
    
    H_square = H[:-1, :]
    V_square = V[:, :-1]

    # print("V_square_shape:",V_square.shape)
    # print("H_square_shape:",H_square.shape)

    if H.shape[0] <= m: # handle early ternimation situation, run without error limit
        error = 0
        m = H.shape[0]-1
        print ("The Hm_norm calculate in iteration is ever less than tolerance, iteration terminated early (after {} iterations), approximation is exact without error limit.".format(m))
    elif error_limit is not None:
        error = compute_posteriori_error(A,H_square,h,N,samples,compute_error = True,use_arnoldi=use_arnoldi)
        print("compute_posterio_error is:",error)

    if error_limit is None or error < error_limit:
        if error_limit is not None:
            print ("compute_post_error < error_limit, Kcrylov error {} was below threshold {} with {} iterations".format(error, error_limit, m))
        if error_limit is None:
            print ("error limit is None: (1) ternimate early (2) m >= n, run whole  iterations with {} iterations".format(m,m))
        X = simKrylov(A,x0,m,h,N,V_square,H_square)
    else:
        print("error_limit is : {}".format(error_limit))
        print("error_limit is not None, compute_posteriori_error({}) >= error_limit({}), X return None, need to increase current arnoldi iterations  {}".format(error,error_limit,m))
        X = None

    return X, m, krylov_time_duration

def sim_autotune(krylov_time,A,x0,h,N,m,samples,tolerance,target_error=None,projection_mat = None,use_arnoldi = None):
    '''
    Perform a projected simulation from a given initial vector. This auto-tunes the number
    of krylov iterations based on the desired error.

    returns the projected simulation at each step.
    '''

    n = A.shape[0]
    while True:
        error_limit = target_error if m < n else None

        X, m, krylov_time_duration = sim_with_error(A,x0,h,N,m,samples,tolerance,error_limit = error_limit,projection_mat = projection_mat,use_arnoldi = use_arnoldi)
        print("current iterations is {}".format(m))
        krylov_time += krylov_time_duration

        if X is None:
            print("increase iterations from {} to {} by multiplying 1.5 ".format(m,int(np.ceil(1.5 * m))))
            m = int(np.ceil(1.5 * m))
            if m >= n:
                print("increased interation m >= n, set m = n")
                m = n
         
        else:

            break
   

    return X,krylov_time

def simReachKrylov(A, X0_star, h, N,m,samples,tolerance,target_error=None, initial_space = None, output_space = None, use_arnoldi = None, use_init_space =None):

    """
    Simulate reachable set using the Krylov subspace method.

    Args:
    - A: System matrix.
    - X0: Initial set of states (a probstar set).
    - h: Time step for simulation.
    - N: Number of steps.
    - m: The number of basic vectors for Krylov subspace Km.Vm = [v0, .., vm-1]

    Returns:
    - R: Reachable set after auto-tuning the number of interations for each basic vector in Star set within target error
    """

    if use_init_space == True:
        print("---------------using init sapce as basic vectors, use output space as projection mat ----------------")
        basis_vec = initial_space
        projection_mat= output_space
    else: 
        print("----------------using output sapce as basic vectors, use initial sapce as projection mat --------------") # transpose dynamics property
        basis_vec = output_space.T # using O.T
        projection_mat = initial_space.T # using I.T
        A = A.T # using A.T
   
    k = basis_vec.shape[1]
    Z = []
    krylov_time = 0
    
    for i in range(k):
        print("\n========== Start the {}th basic vector projection in Star.V ===========".format(i+1))

        X,krylov_time = sim_autotune(krylov_time,A, basis_vec[:, i],h,N,m,samples,tolerance,target_error = target_error,projection_mat=projection_mat,use_arnoldi = use_arnoldi)

        X = np.array(X)

        Z.append(X)


    R = []

  
    for i in range(N+1): # include inital step
        V =[]   
    
        for j in range(k):
            V.append(Z[j][i,:])
        V = np.vstack(V)        

        # use transpose property
        if use_init_space == True:
            V= V.T 
      
        S = ProbStar(V, X0_star.C, X0_star.d,X0_star.mu,X0_star.Sig, X0_star.pred_lb, X0_star.pred_ub)
        R.append(S)

    return R,krylov_time

def random_two_dims_mapping(Xt, dim1,dim2):
        dims = Xt.dim
        M = np.zeros((2, dims))
        M[0][dim1-1] = 1
        M[1][dim2-1] = 1
        return M


