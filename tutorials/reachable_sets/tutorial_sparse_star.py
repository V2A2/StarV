import copy
import numpy as np
import scipy.sparse as sp
from StarV.set.sparsestar import SparseStar
from StarV.util.plot import plot_star


def sparsestar_construct_using_state_bounds():
    """
    Constructs a SparseStar set using state bounds.
    Creates a box-shaped SparseStar set defined by lower and upper bounds.
    """
    print('======================================================================')
    print('========= EXAMPLE: SparseStar Construction with State Bounds =========')
    dim = 2
    
    # Define bounds
    lb = -np.ones(dim)      # lower bounds: x1 >= -1, x2 >= -1
    ub = np.ones(dim)       # upper bounds: x1 <= 1, x2 <= 1
    
    # Create star set
    S = SparseStar(lb, ub)
    
    print("Created box-shaped SparseStar set:")
    print(S)
    plot_star(S)
    
    print('========= DONE: SparseStar Construction with State Bounds ============')
    print('======================================================================\n\n')


def sparsestar_construct():
    print('======================================================================')
    print('================== EXAMPLE: SparseStar Construction ==================')
    dim = 2

    c = np.zeros([2, 1])
    v = np.eye(dim)
    A = np.hstack([c, v])

    C = np.array([
        [-0.22647,  0.06832, -1,  0],
        [ 0.39032,  0.03921,  0, -1],
        [ 0.22647, -0.06832,  1,  0],
        [-0.39032, -0.03921,  0,  1]
    ])
    C = sp.csc_array(C)
    d = np.array([0.3890, 0.1199, 0.5516, 0.1285])
    pred_lb = np.array([-1, -1, -0.68382, -0.54943])
    pred_ub = np.array([ 1,  1,  0.84647,  0.55803])
    pred_depth = np.array([1, 1, 0, 0])
    S = SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)
    
    print("Created SparseStar set with independent basis matrix and constraints:")
    repr(S)
    print(S)
    plot_star(S)
    
    print('==================== DONE: SparseStar Construction ===================')
    print('======================================================================\n\n')

def sparsestar_affine_mapping():
    print('======================================================================')
    print('=============== EXAMPLE: affine mapping of SparseStar ================')
    
    dim = 2
        
    # original SparseStar
    lb = -np.ones(dim)  # lower bounds: x1 >= -1, x2 >= -1
    ub = np.ones(dim)   # upper bounds: x1 <= 1, x2 <= 1
    S = SparseStar(lb, ub)
    print("\nOriginal SparseStar set:")
    print(S)
    plot_star(S)

    # affine mapped SparseStar
    A = np.array([[1.0, 0.5], [-0.5, 1.0]])
    b = -np.ones(dim)
    S1 = S.affineMap(A, b)

    print("SparseStar set after affine mapping:")
    print(S1)
    plot_star(S1)
    
    print('================= DONE: affine mapping of SparseStar =================')
    print('======================================================================\n\n') 


def sparsestar_minkowski_sum():
    print('======================================================================')
    print('============= EXAMPLE: SparseStar Minkowski Sum ======================')
    dim = 2
    
    # Create first SparseStar set
    lb = -np.ones(dim)
    ub = np.ones(dim)
    S1 = SparseStar(lb, ub)

    print("SparseStar set 1:")
    print(S1)
    plot_star(S1)

    # Create second SparseStar set using affine map of first set
    W = np.array([[0.5, 0.5],
                  [0.5, -0.5]])
    b = np.array([0, 0])
    S2 = S1.affineMap(W, b)

    print("SparseStar set 2:")
    print(S2)
    plot_star(S2)

    # Compute Minkowski sum
    S = S1.minKowskiSum(S2)
    
    print("SparseStar after Minkowski sum:")
    print(S)
    plot_star(S)

    print('============= DONE: SparseStar Minkowski Sum =========================')
    print('======================================================================\n\n')
    
def sparsestar_depthReduction():
    print('======================================================================')
    print('================= EXAMPLE: SparseStar depthReduction =================')
    dim = 2
    
    c = np.zeros([2, 1])
    v = np.eye(dim)
    A = np.hstack([c, v])

    C = np.array([
        [-0.22647,  0.06832, -1,  0],
        [ 0.39032,  0.03921,  0, -1],
        [ 0.22647, -0.06832,  1,  0],
        [-0.39032, -0.03921,  0,  1],
        [-0.62899,  0.18975, -1,  0],
        [ 0.52719,  0.05296,  0, -1],
        [ 0.71908, -0.21693,  1,  0],
        [-0.52883, -0.05312,  0,  1]])
    C = sp.csc_array(C)
    d = np.array([0.3890, 0.1199, 0.5516, 0.1285, -0.02773, 0.02212, 0.25219, 0.03252])
    pred_lb = np.array([-1, -1, -0.68382, -0.54943])
    pred_ub = np.array([ 1,  1,  0.84647,  0.55803])
    pred_depth = np.array([1, 1, 0, 0])
    S = SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)

    print('Original SparseStar:')
    print(S)
    plot_star(S)

    print('Predicate depth reduced SparseStar with DR=1:')
    D = S.depthReduction(DR=1)
    print(D)
    plot_star(D)
    
    print('=================== DONE: SparseStar depthReduction ==================')
    print('======================================================================\n\n')

def sparsestar_getRanges():
    print('======================================================================')
    print('=========== EXAMPLE: getting state ranges of sparse star set =========')
    dim = 2

    c = np.zeros([2, 1])
    v = np.eye(dim)
    A = np.hstack([c, v])

    C = np.array([
        [-0.22647,  0.06832, -1,  0],
        [ 0.39032,  0.03921,  0, -1],
        [ 0.22647, -0.06832,  1,  0],
        [-0.39032, -0.03921,  0,  1]
    ])
    C = sp.csc_array(C)
    d = np.array([0.3890, 0.1199, 0.5516, 0.1285])
    pred_lb = np.array([-1, -1, -0.68382, -0.54943])
    pred_ub = np.array([ 1,  1,  0.84647,  0.55803])
    pred_depth = np.array([1, 1, 0, 0])
    S = SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)
    print('Constructed SparseStar:')
    print(S)
    plot_star(S)
    
    l, u = S.getRanges()
    print('State bounds computed with getRanges() via LP solver:')
    print('lower bounds:\n', l)
    print('upper bounds:\n', u)
    print()
    
    print('============ DONE:  getting state ranges of image star set ===========')
    print('======================================================================\n\n')
    
def sparsestar_estimateRanges():
    print('======================================================================')
    print('=========== EXAMPLE: getting state ranges of sparse star set =========')
    dim = 2

    c = np.zeros([2, 1])
    v = np.eye(dim)
    A = np.hstack([c, v])

    C = np.array([
        [-0.22647,  0.06832, -1,  0],
        [ 0.39032,  0.03921,  0, -1],
        [ 0.22647, -0.06832,  1,  0],
        [-0.39032, -0.03921,  0,  1]
    ])
    C = sp.csc_array(C)
    d = np.array([0.3890, 0.1199, 0.5516, 0.1285])
    pred_lb = np.array([-1, -1, -0.68382, -0.54943])
    pred_ub = np.array([ 1,  1,  0.84647,  0.55803])
    pred_depth = np.array([1, 1, 0, 0])
    S = SparseStar(A, C, d, pred_lb, pred_ub, pred_depth)
    print('Constructed SparseStar:')
    print(S)
    plot_star(S)
    
    l, u = S.estimateRanges()
    print('State bounds computed with estimateRanges():')
    print('lower bounds:\n', l)
    print('upper bounds:\n', u)
    print()
    
    l, u = S.getRanges('estimate')
    print('State bounds computed with getRanges(''estimate'')')
    print('lower bounds:\n', l)
    print('upper bounds:\n', u)
    print()
    
    print('============ DONE:  getting state ranges of sparse star set ==========')
    print('======================================================================\n\n')
    

if __name__ == "__main__":
    sparsestar_construct_using_state_bounds()
    sparsestar_construct()
    sparsestar_affine_mapping()
    sparsestar_minkowski_sum()
    sparsestar_depthReduction()
    sparsestar_getRanges()
    sparsestar_estimateRanges()