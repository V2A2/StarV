import copy
import numpy as np
from StarV.set.probstar import ProbStar
from StarV.util.plot import plot_probstar, plot_probstar_distribution, plot_probstar_contour


def probstar_construct_with_bounded_gaussian_distribution():
    print('======================================================================')
    print('=== EXAMPLE: construct ProbStar with bounded Gaussian distribution ===')
    
    dim = 2
    
    mu = np.zeros(dim)
    Sig = np.diag([0.1, 0.08])
    pred_lb = -np.ones(dim)
    pred_ub = np.ones(dim)
    
    # Create ProbStar set
    P = ProbStar(mu, Sig, pred_lb, pred_ub) 
    
    print("\nCreated ProbStar set with bounded Gaussian distribution:")
    print(P)
    plot_probstar(P)
    plot_probstar_distribution(P)
    plot_probstar_contour(P)
    
    print('===== DONE: construct ProbStar with bounded Gaussian distribution ====')
    print('======================================================================\n\n')
    
    
def probstar_construct():
    print('======================================================================')
    print('===================== EXAMPLE: construct ProbStar ====================')
    
    dim = 2
    
    center = np.zeros([dim, 1])
    basis_matrix = np.array([[0.5, 0.5], [-0.5, 0.5]])
    V = np.concatenate([center, basis_matrix], axis=1)
    
    C = np.concatenate([np.eye(dim), -np.eye(dim)], axis=0)
    d = np.array([2, 2, 1, 1])
    
    mu = np.zeros(dim)
    Sig = np.diag([0.05, 0.1])
    pred_lb = -np.ones(dim)
    pred_ub = 2*np.ones(dim)

    # Create ProbStar set
    P = ProbStar(V, C, d, mu, Sig, pred_lb, pred_ub) 
    
    print("\nCreated ProbStar set:")
    print(P)
    plot_probstar(P)
    plot_probstar_distribution(P)
    plot_probstar_contour(P)
    # plot_probstar_distribution_error(P)
    
    print('====================== DONE: construct ProbStar ======================')
    print('======================================================================\n\n')
    

def probstar_affine_map():
    print('======================================================================')
    print('================ EXAMPLE: affine mapping of ProbStar =================')
    
    dim = 2
        
    # original ProbStar
    mu = np.array([0.5, 1.0])
    Sig = np.diag([0.1, 0.08])
    pred_lb = -np.ones(dim)
    pred_ub = np.ones(dim)

    P = ProbStar(mu, Sig, pred_lb, pred_ub)
    print(P)
    plot_probstar(P)
    plot_probstar_distribution(P)
    plot_probstar_contour(P)

    # affine mapped ProbStar
    A = np.array([[1.0, 0.5], [-0.5, 1.0]])
    b = -np.ones(dim)
    P1 = P.affineMap(A, b)

    print(P1)
    plot_probstar(P1)
    plot_probstar_distribution(P1)
    plot_probstar_contour(P1)
    
    print('================== DONE: affine mapping of ProbStar ==================')
    print('======================================================================\n\n') 
    
    
def probstar_estimate_range():
    dim = 2
    
    center = np.zeros([dim, 1])
    basis_matrix = np.array([[1.0, 0.5], [-0.5, 1.0]])
    V = np.concatenate([center, basis_matrix], axis=1)
    
    C = np.concatenate([np.eye(dim), -np.eye(dim)], axis=0)
    d = np.array([2, 2, 1, 1])
    
    mu = np.zeros(dim)
    Sig = np.diag([0.05, 0.1])
    pred_lb = -np.ones(dim)
    pred_ub = 2*np.ones(dim)

    # Create ProbStar set
    P = ProbStar(V, C, d, mu, Sig, pred_lb, pred_ub) 
    
    # Estimated range for {index} dimension
    index = 0
    l_est_0, u_est_0 = P.estimateRange(index)
    
    print(f'\nEstimated range for {index} dimension:')
    print(f'Estimated {index}th lower state bound (l_est_0) = {l_est_0}')
    print(f'Estimated {index}th upper state bound (u_est_0) = {u_est_0}')
    
    
def probstar_estimate_ranges():
    dim = 2
    
    center = np.zeros([dim, 1])
    basis_matrix = np.array([[1.0, 0.5], [-0.5, 1.0]])
    V = np.concatenate([center, basis_matrix], axis=1)
    
    C = np.concatenate([np.eye(dim), -np.eye(dim)], axis=0)
    d = np.array([2, 2, 1, 1])
    
    mu = np.zeros(dim)
    Sig = np.diag([0.05, 0.1])
    pred_lb = -np.ones(dim)
    pred_ub = 2*np.ones(dim)

    # Create ProbStar set
    P = ProbStar(V, C, d, mu, Sig, pred_lb, pred_ub) 
    
    # Estimate ranges for all dimensions
    l_est, u_est = P.estimateRanges()
    
    print('\nEstimated ranges for all dimensions:')
    print(f'Estimated lower state bounds (l_est) = {l_est}')
    print(f'Estimated upper state bounds (u_est) = {u_est}')
    
    
def probstar_get_ranges():
    dim = 2
    
    center = np.zeros([dim, 1])
    basis_matrix = np.array([[1.0, 0.5], [-0.5, 1.0]])
    V = np.concatenate([center, basis_matrix], axis=1)
    
    C = np.concatenate([np.eye(dim), -np.eye(dim)], axis=0)
    d = np.array([2, 2, 1, 1])
    
    mu = np.zeros(dim)
    Sig = np.diag([0.05, 0.1])
    pred_lb = -np.ones(dim)
    pred_ub = 2*np.ones(dim)

    # Create ProbStar set
    P = ProbStar(V, C, d, mu, Sig, pred_lb, pred_ub) 
    
    # Estimate ranges for all dimensions
    x_mins, x_maxs = P.getRanges()
    
    print('Optimized ranges for all dimensions:')
    print(f'Optimized lower state bounds (x_mins) = {x_mins}')
    print(f'Optimized upper state bounds (x_maxs) = {x_maxs}')
    

def probstar_minkowski_sum():
    dim = 2
    
    mu = np.zeros(dim)
    Sig = np.diag([0.1, 0.08])
    pred_lb = -np.ones(dim)
    pred_ub = np.ones(dim)
    
    # Create first ProbStar set
    P1 = ProbStar(mu, Sig, pred_lb, pred_ub) 
    
    center = np.zeros([dim, 1])
    basis_matrix = np.array([[0.5, 0.5], [-0.5, 0.5]])
    V = np.concatenate([center, basis_matrix], axis=1)
    
    C = []
    d = []
    
    mu = np.zeros(dim)
    Sig = np.diag([0.05, 0.1])
    pred_lb = -np.ones(dim)
    pred_ub = 2*np.ones(dim)

    # Create second ProbStar set
    P2 = ProbStar(V, C, d, mu, Sig, pred_lb, pred_ub)
    
    # Minkowski sum of two ProbStar sets
    P = P1.minKowskiSum(P2)
    
    plot_probstar(P1)
    plot_probstar(P2)
    plot_probstar(P)
    
if __name__ == "__main__":
    probstar_construct_with_bounded_gaussian_distribution()
    probstar_construct()
    probstar_affine_map()
    probstar_estimate_range()
    probstar_estimate_ranges()
    probstar_get_ranges()
    probstar_minkowski_sum()