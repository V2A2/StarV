import copy
import numpy as np
from StarV.set.probstar import ProbStar
from StarV.util.plot import plot_probstar, plot_probstar_distribution, plot_probstar_contour
from StarV.util.print_util import print_util


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
    basis_vector = np.array([[0.5, 0.5], [-0.5, 0.5]])
    V = np.concatenate([center, basis_vector], axis=1)
    
    C = np.concatenate([np.eye(dim), -np.eye(dim)], axis=0)
    d = np.ones(2*dim)
    
    mu = np.zeros(dim)
    Sig = np.diag([0.05, 0.1])
    pred_lb = -1.5*np.ones(dim)
    pred_ub = 2*np.ones(dim)
    
    # Create ProbStar set
    P = ProbStar(V, C, d, mu, Sig, pred_lb, pred_ub) 
    
    print("\nCreated ProbStar set:")
    print(P)
    plot_probstar(P)
    plot_probstar_distribution(P)
    plot_probstar_contour(P)
    
    print('====================== DONE: construct ProbStar ======================')
    print('======================================================================\n\n')
    
if __name__ == "__main__":
    probstar_construct_with_bounded_gaussian_distribution()
    probstar_construct()