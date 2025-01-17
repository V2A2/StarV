"""
Star Set Tutorial

This tutorial demonstrates various operations on Star sets:
1. Star set construction methods:
   - Using V, C, d, pred_lb, pred_ub
   - Using state bounds
   - Random generation

2. Range estimation methods:
   - estimateRange for single dimension
   - estimateRanges for multiple dimensions

3. Min/Max operations:
   - getMin/getMax for single dimension
   - getMins/getMaxs for multiple dimensions
   - getRanges for exact ranges

4. Transformations:
   - Affine map operations

5. Constraint operations:
   - Adding single/multiple constraints
   - Intersecting with half-spaces

6. Set operations:
   - Emptiness checking
   - Minkowski sum
"""

import copy
import numpy as np
from StarV.set.star import Star
from StarV.util.plot import plot_star

def star_construct_with_basis_and_constraints():
    """
    Constructs a star set using basis matrix and constraints (Method 1).
    A star set is defined by:
    x = c + a[1]*v[1] + a[2]*v[2] + ... + a[n]*v[n]
        = V * b, where:
    - V = [c v[1] v[2] ... v[n]],
    - b = [1 a[1] a[2] ... a[n]]^T,
    - C * a <= d (predicate constraints)
    - pred_lb <= a <= pred_ub (predicate bounds)
    """
    print('======================================================================')
    print('=============== EXAMPLE: Star Construction with Basis Matrix =========')
    
    # Define components
    c1 = np.array([[0], [0]])         # center vector
    v1 = np.array([[1], [0]])         # basis vector 1
    v2 = np.array([[0], [1]])         # basis vector 2
    V = np.hstack((c1, v1, v2))       # V = [c1 v1 v2]
    
    pred_lb = np.array([-1, -1])      # -1 <= a1, -1 <= a2
    pred_ub = np.array([1, 1])        # a1 <= 1, a2 <= 1
    C = np.array([[1, 1]])            # a1 + a2 <= 1
    d = np.array([1])
    
    # Create star set
    star = Star(V, C, d, pred_lb, pred_ub)
    
    print("\nCreated star set with basis matrix and constraints:")
    print(star)
    plot_star(star)
    
    print('=============== DONE: Star Construction with Basis Matrix ============')
    print('======================================================================\n\n')


def star_construct_with_bounds():
    """
    Constructs a star set using state bounds (Method 2).
    Creates a box-shaped star set defined by lower and upper bounds.
    """
    print('======================================================================')
    print('=============== EXAMPLE: Star Construction with Bounds ===============')
    
    # Define bounds
    lb = np.array([-1.0, -1.0])       # lower bounds: x1 >= -1, x2 >= -1
    ub = np.array([1.0, 1.0])         # upper bounds: x1 <= 1, x2 <= 1
    
    # Create star set
    star = Star(lb, ub)
    
    print("\nCreated box-shaped star set:")
    print(star)
    plot_star(star)
    
    print('=============== DONE: Star Construction with Bounds ==================')
    print('======================================================================\n\n')


def star_construct_random():
    """
    Constructs a star set using random state bounds (Method 3).
    """
    print('======================================================================')
    print('=============== EXAMPLE: Random Star Construction ====================')
    
    # Create random star set
    star = Star.rand(2)              # 2D random star set
    
    print("\nCreated random star set:")
    print(star)
    plot_star(star)
    
    print('=============== DONE: Random Star Construction =======================')
    print('======================================================================\n\n')


def star_construct_with_random_H_polytope():
    """
    Constructs a star set using random H-representation polytope (Method 4).
    """
    print('======================================================================')
    print('=============== EXAMPLE: Random Star with Constraints ================')
    
    # Create random star set with constraints
    star = Star.rand_polytope(2, 3)  # 2D random star with 3 constraints
    
    print("\nCreated random constrained star set:")
    print(star)
    plot_star(star)
    
    print('=============== DONE: Random Star with Constraints ===================')
    print('======================================================================\n\n')


def star_affine_map():
    """
    Demonstrates affine mapping operations on a Star set.
    """
    print('======================================================================')
    print('=============== EXAMPLE: Affine Map Operation on Star Set ============')

    # Create initial star set
    lb = np.array([-1.0, -1.0])           # lower bounds: x1 >= -1, x2 >= -1
    ub = np.array([1.0, 1.0])             # upper bounds: x1 <= 1, x2 <= 1
    star = Star(lb, ub)

    print('\nOriginal star set:')
    print(star)
    plot_star(star)

    # Define affine transformation
    W = np.array([[0.5, 0.5],
                  [0.5, -0.5]])           # Scaling + rotation
    b = np.array([1, 0])                  # Translation

    print("\nAffine mapping parameters:")
    print(f"Transformation matrix (W):\n{W}")
    print(f"Offset vector (b):\n{b}")

    # Apply affine transformation
    star_mapped = star.affineMap(W, b)

    print('\nAffine mapped star set:')
    print(star_mapped)
    plot_star(star_mapped)

    print('=============== DONE: Affine Map Operation on Star Set ===============')
    print('======================================================================\n\n')


def star_estimate_range():
    """
    Demonstrates range estimation for a single dimension of a Star set.
    """
    print('======================================================================')
    print('=============== EXAMPLE: Star Single Dimension Range Estimation ======')

    # Define bounds
    lb = np.array([-1.0, -1.0])       # lower bounds: x1 >= -1, x2 >= -1
    ub = np.array([1.0, 1.0])         # upper bounds: x1 <= 1, x2 <= 1
    
    # Create star set
    star = Star(lb, ub)

    print("\nCreated star set with bounds:")
    print(f"Lower bounds (lb): {lb}")
    print(f"Upper bounds (ub): {ub}")

    # Estimate range for first dimension
    dim = 0
    est_min, est_max = star.estimateRange(dim)
    
    print(f'\nEstimated range for dimension {dim}:')
    print(f'Estimated min (l_est[{dim}]) = {est_min}')
    print(f'Estimated max (u_est[{dim}]) = {est_max}')

    print('=============== DONE: Star Single Dimension Range Estimation =========')
    print('======================================================================\n\n')


def star_estimate_ranges():
    """
    Demonstrates range estimation for all dimensions of a Star set.
    """
    print('======================================================================')
    print('=============== EXAMPLE: Star Multiple Dimensions Range Estimation ===')

    # Define bounds
    lb = np.array([-1.0, -1.0])       # lower bounds: x1 >= -1, x2 >= -1
    ub = np.array([1.0, 1.0])         # upper bounds: x1 <= 1, x2 <= 1
    
    # Create star set
    star = Star(lb, ub)

    print("\nCreated star set with bounds:")
    print(f"Lower bounds (lb): {lb}")
    print(f"Upper bounds (ub): {ub}")

    # Estimate ranges for all dimensions
    est_mins, est_maxs = star.estimateRanges()
    
    print('\nEstimated ranges for all dimensions:')
    print(f'Estimated mins (l_est) = {est_mins}')
    print(f'Estimated maxs (u_est) = {est_maxs}')

    print('=============== DONE: Star Multiple Dimensions Range Estimation ======')
    print('======================================================================\n\n')


def star_exact_min_max():
    """
    Demonstrates exact range computation for a single dimension of a Star set.
    """
    print('======================================================================')
    print('=============== EXAMPLE: Star Single Dimension Exact Range ===========')

    # Define bounds
    lb = np.array([-1.0, -1.0])       # lower bounds: x1 >= -1, x2 >= -1
    ub = np.array([1.0, 1.0])         # upper bounds: x1 <= 1, x2 <= 1
    
    # Create star set
    star = Star(lb, ub)

    print("\nCreated star set with bounds:")
    print(f"Lower bounds (l): {lb}")
    print(f"Upper bounds (u): {ub}")

    # Compute exact range for first dimension
    dim = 0
    exact_min = star.getMin(dim, 'gurobi')
    exact_max = star.getMax(dim, 'gurobi')
    
    print(f'\nExact range for dimension {dim}:')
    print(f'Minimum (x[{dim}]_min) = {exact_min:.6f}')
    print(f'Maximum (x[{dim}]_max) = {exact_max:.6f}')

    print('=============== DONE: Star Single Dimension Exact Range ==============')
    print('======================================================================\n\n')


def star_exact_mins_maxs():
    """
    Demonstrates exact range computation for selected dimensions of a Star set.
    """
    print('======================================================================')
    print('=============== EXAMPLE: Star Multiple Dimensions Exact Range ========')

    # Define bounds
    lb = np.array([-1.0, -1.0])       # lower bounds: x1 >= -1, x2 >= -1
    ub = np.array([1.0, 1.0])         # upper bounds: x1 <= 1, x2 <= 1
    
    # Create star set
    star = Star(lb, ub)

    print("\nCreated star set with bounds:")
    print(f"Lower bounds (l): {lb}")
    print(f"Upper bounds (u): {ub}")

    # Compute ranges for selected dimensions
    map = [0, 1]  # Compute ranges for first two dimensions
    exact_mins = star.getMins(map, 'gurobi')
    exact_maxs = star.getMaxs(map, 'gurobi')
    
    print('\nExact ranges for selected dimensions:')
    for i, dim in enumerate(map):
        print(f'Dimension {dim}:')
        print(f'Minimum (x[{dim}]_min) = {exact_mins[i]:.6f}')
        print(f'Maximum (x[{dim}]_max) = {exact_maxs[i]:.6f}')

    print('=============== DONE: Star Multiple Dimensions Exact Range ===========')
    print('======================================================================\n\n')


def star_exact_ranges():
    """
    Demonstrates exact range computation for all dimensions of a Star set.
    """
    print('======================================================================')
    print('=============== EXAMPLE: Star All Dimensions Exact Range =============')

    # Define bounds
    lb = np.array([-1.0, -1.0])       # lower bounds: x1 >= -1, x2 >= -1
    ub = np.array([1.0, 1.0])         # upper bounds: x1 <= 1, x2 <= 1
    
    # Create star set
    star = Star(lb, ub)

    print("\nCreated star set with bounds:")
    print(f"Lower bounds (l): {lb}")
    print(f"Upper bounds (u): {ub}")

    # Compute exact ranges for all dimensions
    exact_mins, exact_maxs = star.getRanges('gurobi')
    
    print('\nExact ranges for all dimensions:')
    print(f'Exact mins (x_min) = {exact_mins}')
    print(f'Exact maxs (x_max) = {exact_maxs}')

    print('=============== DONE: Star All Dimensions Exact Range ================')
    print('======================================================================\n\n')


def star_add_single_constraint():
    """
    Demonstrates adding a single constraint to a Star set.
    """
    print('======================================================================')
    print('=============== EXAMPLE: Star Add Single Constraint ==================')

    # Create initial star set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    star = Star(lb, ub)

    print("\nOriginal star set:")
    print(star)
    plot_star(star)

    # Add first constraint: a1 + a2 <= 1
    C1 = np.array([1.0, 1.0])
    d1 = np.array([1])
    star.addConstraint(C1, d1)

    # Add second constraint: -a1 - a2 <= 1
    C2 = np.array([-1.0, -1.0])
    d2 = np.array([1])
    star.addConstraint(C2, d2)

    print("\nStar set after adding two constraints:")
    print(star)
    plot_star(star)

    print('=============== DONE: Star Add Single Constraint =====================')
    print('======================================================================\n\n')


def star_add_multiple_constraints():
    """
    Demonstrates adding multiple constraints at once to a Star set.
    """
    print('======================================================================')
    print('=============== EXAMPLE: Star Add Multiple Constraints ===============')

    # Create initial star set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    star = Star(lb, ub)

    print("\nOriginal star set:")
    print(star)
    plot_star(star)

    # Add multiple constraints at once
    C = np.array([[1.0, 1.0],
                  [-1.0, -1.0]])
    d = np.array([1, 1])
    star.addMultipleConstraints(C, d)

    print("\nStar set after adding multiple constraints:")
    print(star)
    plot_star(star)

    print('=============== DONE: Star Add Multiple Constraints ==================')
    print('======================================================================\n\n')


def star_intersect_halfspace():
    """
    Demonstrates intersecting a Star set with half-spaces.
    """
    print('======================================================================')
    print('=============== EXAMPLE: Star Intersect Half-Space ===================')

    # Create initial star set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    star = Star(lb, ub)

    print("\nOriginal star set:")
    print(star)
    plot_star(star)

    # Intersect with half-spaces
    H = np.array([[1.0, 1.0],
                  [-1.0, -1.0]])
    g = np.array([1, 1])
    star = star.intersectHalfSpace(H, g)

    print("\nStar set after half-space intersection:")
    print(star)
    plot_star(star)

    print('=============== DONE: Star Intersect Half-Space ======================')
    print('======================================================================\n\n')


def star_check_emptiness():
    """
    Demonstrates emptiness checking of a Star set.
    """
    print('======================================================================')
    print('=============== EXAMPLE: Star Emptiness Checking =====================')

    # Create initial star set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    star = Star(lb, ub)

    print('\nOriginal star set:')
    print(star)

    empty = star.isEmptySet()
    print(f"\nIs original star set empty? {empty}")

    # Create empty star set by adding infeasible constraint
    star_constrained = copy.deepcopy(star)
    C = np.array([-1.0, 0.0])    # Constraint: -a1 <= -2
    d = np.array([-2])           # Infeasible with a1 <= 1
    star_constrained.addConstraint(C, d)

    print("\nStar set with infeasible constraint:")
    print(star_constrained)
    empty_constrained = star_constrained.isEmptySet()
    print(f"Is constrained star set empty? {empty_constrained}")

    print('=============== DONE: Star Emptiness Checking ========================')
    print('======================================================================\n\n')


def star_minkowski_sum():
    """
    Demonstrates Minkowski sum of two Star sets.
    """
    print('======================================================================')
    print('=============== EXAMPLE: Star Minkowski Sum ==========================')

    # Create first star set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    star1 = Star(lb, ub)

    print("\nStar set 1:")
    print(star1)
    plot_star(star1)

    # Create second star set using affine map of first set
    W = np.array([[0.5, 0.5],
                  [0.5, -0.5]])
    b = np.array([0, 0])
    star2 = star1.affineMap(W, b)

    print("\nStar set 2:")
    print(star2)
    plot_star(star2)

    # Compute Minkowski sum
    star_sum = star1.minKowskiSum(star2)
    
    print("\nMinkowski sum:")
    print(star_sum)
    plot_star(star_sum)

    print('=============== DONE: Star Minkowski Sum =============================')
    print('======================================================================\n\n')


if __name__ == "__main__":
    """
    Main function to run the star set tutorials.
    """
    star_construct_with_basis_and_constraints()
    star_construct_with_bounds()
    star_construct_random()
    star_construct_with_random_H_polytope()
    star_affine_map()
    star_estimate_range()
    star_estimate_ranges()
    star_exact_min_max()
    star_exact_mins_maxs()
    star_exact_ranges()
    star_add_single_constraint()
    star_add_multiple_constraints()
    star_intersect_halfspace()
    star_check_emptiness()
    star_minkowski_sum()
    
