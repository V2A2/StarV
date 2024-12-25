import math
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
from scipy.integrate import simps
from scipy.sparse import csr_matrix
import time


def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T)



def compute_v_A(A,use_arnoldi = None):

    tol = 1e-5
    A = csr_matrix(A)
    if use_arnoldi == True:
        A = ( A + A.T) / 2
   
    # 'find the maximum eigenvalue of the A matrix'
    start_time = time.time()

    eig = -eigsh(A, k=1, which='LA', tol=tol, return_eigenvectors=False)[0].real
   
    return eig

def compute_ht(Hm, m, h_list_time_step, samples):

    h_list = []

    Im = np.eye(m)
    e1 = Im[:, 0]

    step_exmp = expm(-h_list_time_step * Hm)
    cur_vec = e1

    h_list.append(cur_vec[m-1])

    for _ in range(1,samples):
        cur_vec = np.dot(step_exmp, cur_vec)
        h_list.append(cur_vec[m-1])
    return h_list

def compute_posteriori_error(A,H,h,N,samples, compute_error = None,use_arnoldi = None):
    '''
    Compute the a posteriori error bound

    from Theorem 3.1 in "Error Bounds for the Krylov Subspace Methods for Computations of Matrix Exponentials",
    by Hao Wang and Qiang Ye

    Err[T] <= H[k+1, k] * integral_{0}^{T} |h(t)| * g(t) dt
    where: h(t) = e_k^T * exp(-t * H_k) * e_1
           g(t) = exp((t - T) * v(A))
           v(A) = lambda_{min} (A + A^T)/2
    '''
    
    max_time = N * h
    m = H.shape[1]

    # make sure Hm is square    
    if H.shape[0] == m:
        Hm = H
    else:
        Hm = H[:-1, :]

    ht_time_step = max_time / (samples -1)

    # Compute ht for each step
    ht = compute_ht(Hm, m, ht_time_step, samples)

    eig = compute_v_A(A,use_arnoldi = use_arnoldi)

    def g(t):
        g_t = math.exp((t - max_time) * eig)

        return g_t

    def h_t(step):
        'h(t) = e_k^T * exp(-t * H_k) * e_1'

        return ht[step]

    def integral(t, step):
        '|h(t)| * g(t)'
        f1 = abs(h_t(step))
        f = f1* g(t)
        return f

    if eig * -max_time > 200: # would overflow exp() in computation of g()
        error = np.inf
        print ("Warning: excessive value of eig * -max_time:{}".format(eig * -max_time))
    else:
        x_list = []
        y_list = []

        for i in range(samples):
            x = i * ht_time_step # sample point to integration
    
            y = integral(x, i) # array to be sampled
            x_list.append(x)
            y_list.append(y)
        
        intergal = simps(y_list, x_list, even='avg')
        error = H[m-1, m-2] * intergal 


    return error

