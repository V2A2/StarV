import numpy as np


def arnoldi(A, x0, m, tolerance,compute_error = None,projection_mat=None):
    """
    Arnoldi iteration to compute orthonormal basis for Krylov subspace.
    Projecting the initial n-dimensional state onto the smaller, m-dimensional Krylov subspace, 
    Computing the matrix exponential using the projected linear transformation Hm ,    

    Args:
    - A: n x n input matrix, not required to be sysmetirc
    - x0: normalized n x 1 initial column vector.
    - m: The number of basic vectors for the (m-dims) Krylov subspace Km, m iterations in arnoldi method, user defined.
    - tolerance :used for handle early termination
    - compute_error : compute porterior error
    - key_dir_mat: projection matrix

    Returns:
    - Vm: n x m orthonormal basis matrix.
    - Hm: m x m upper Hessenberg matrix.
    """

    if compute_error == True:
        # print("==== using -A in arnoldi inter ======:")
        A = -A
   
    if m is None:
        m = n

    n = A.shape[0] # A matrix dims

    x0_norm =  np.linalg.norm(x0,ord=2)
    x0_normlalize = x0 / x0_norm  # normalized (n ,)init vectoer x0 to norm = 1, ensure stability, validation, scalability


    # if V_mat has None type give it to 0
    inds = np.isnan(x0_normlalize)
    x0_normlalize[inds] = 0


    Hm = np.zeros((m+1, m)) # initial Hm : m x m hessenberg matrix
    Vm = np.zeros((n, m+1)) # initial Vm : n x m orthonormal basis vectors
    Vm[:,0] = x0_normlalize # column vector (n,)


    i = 0
    while i < m:
        wi = A @ Vm[:, i] 
        for j in range(i+1):
            Vm_T = np.transpose(Vm[:, j])
            Hm[j, i] = Vm_T @ wi
            wi = wi - Hm[j, i] * Vm[:, j]

        Hm[i+1, i] = np.linalg.norm(wi,ord=2)

        if Hm[i+1, i] >= tolerance:
            Vm[:, i+1] = wi/ Hm[i+1, i]
        elif i > 0 and Hm[i+1, i] < tolerance: 
                #break early
                # print("=====break early in arnoldi ====")
                Vm = Vm[:,:i+2]
                Hm = Hm[:i+2,:i+1]  
                break # jump out the while loop   
        
        i = i + 1

    if projection_mat is None:
        pv_mat = None
    else:
        pv_mat = projection_mat @ Vm
        pv_mat = pv_mat  * x0_norm

    
    return pv_mat, Hm

def lanczos(A, x0, m, tolerance,compute_error = None,projection_mat=None):
    """
    Lanczos iteration to compute orthonormal basis for Krylov subspace when for large dimension system matrix, which is both sparse and symmetric.
    Projecting the initial n-dimensional state onto the smaller, m-dimensional Krylov subspace, 
    Computing the matrix exponential using the projected linear transformation Hm,    

    Args:
    - A: n x n input matrix, A = A^T, required to be sysmetirc.
    - x0: normalized n x 1 initial column vector.
    - m: The number of basic vectors for the (m-dims) Krylov subspace Km, m iterations in arnoldi method, user defined.
    - tolerance :used for handle early termination
    - compute_error : compute porterior error
    - key_dir_mat: projection matrix

    Returns:
    - Vm: n x m orthonormal basis matrix.
    - Hm: m x m upper Hessenberg matrix, is also symmtric and tridiagonal.
    """

    if compute_error == True:
        # print("==== using -A in lanczos iter ======:")
        A = -A
   
    if m is None:
        m = n

    n = A.shape[0] # A matrix dims
    x0_norm =  np.linalg.norm(x0,ord=2)
    x0_normlalize = x0 / x0_norm  # normalized (n ,)init vectoer x0 to norm = 1, ensure stability, validation, scalability

    # if V_mat has None type give it to 0
    inds = np.isnan(x0_normlalize)
    x0_normlalize[inds] = 0

    Hm = np.zeros((m+1, m)) # initial Hm : m x m hessenberg matrix
    pv_mat = np.zeros((n, m+1)) # initial Vm : n x m orthonormal basis vectors
    pv_mat[:,0] = x0_normlalize # column vector (n,)


    i = 0
    prev_v= None
    cur_v = pv_mat[:,0] 
    prev_prev_v = None
    prev_norm_H = None

    if projection_mat is not None:
        key_dirs = projection_mat.shape[0]
        pv_mat = np.zeros((key_dirs,m+1))
        v = projection_mat  @ x0_normlalize # (*,n) @ (n,) 
        v1= v[:]
        pv_mat[:,0] = v1
        

    while i < m:
        prev_prev_v = prev_v # (n,)
        prev_v = cur_v # (n,)
        cur_v = A @ prev_v

        if i > 0 and prev_prev_v is not None:
            dot_val = prev_norm_H 
            Hm[i-1,i] = dot_val
            cur_v = cur_v - prev_prev_v * dot_val

        dot_val = prev_v.T @ cur_v
        Hm[i,i] = dot_val
        cur_v = cur_v - prev_v * dot_val
        prev_norm_H = np.linalg.norm(cur_v,2)
        Hm[i+1,i] = prev_norm_H
        # print("in iteration {} Hm_norm_lanczos = {}".format(i+1, prev_norm_H))

        if prev_norm_H >= tolerance:
            Hm[i+1,i] = prev_norm_H
            cur_v = cur_v/prev_norm_H
            if projection_mat is not None:
                X = projection_mat @ cur_v
                Y = X[:]
                pv_mat[:,i+1] = Y

        elif i > 0 and prev_norm_H < tolerance:
                # print("=====break early in lanczos====")
                pv_mat = pv_mat[:, :i+2]
                Hm = Hm[:i+2,:i+1]
                break     
        i = i + 1

    if projection_mat is None:
        rv_pv = None

    else:

        rv_pv = pv_mat * x0_norm

    return rv_pv,Hm