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
   - affine map operations

5. Constraint operations:
   - adding single/multiple constraints
   - intersecting with half-spaces

6. Set operations:
   - emptiness checking
   - Minkowski sum
"""

from StarV.set.star import Star
import numpy as np
from StarV.util.plot import plot_star
from StarV.util.print_util import print_util
import copy

def star_construction():
    """
    Tutorial demonstrating different methods of Star set construction.

    Mathematical Background:
    ----------------------
    A generalized star set Θ = ⟨c, V, P, pred_lb, pred_ub⟩ is a symbolic representation 
    of a convex set, where:
    - c ∈ ℝⁿ: center vector
    - V = {v₁, ..., vₘ} ∈ ℝⁿˣᵐ: basis matrix
    - P(α) ≡ Cα ≤ d: predicate (linear constraints)
    - pred_lb ≤ α ≤ pred_ub: predicate bounds

    The set can be represented in two equivalent forms:

    1. Θ = {x ∈ ℝⁿ | x = c + Σ(αᵢvᵢ), where Cα ≤ d, pred_lb ≤ α ≤ pred_ub}

    2. Θ = {x | x = V * β}, where:
       - V = [c v₁ v₂ ... vₘ] ∈ ℝⁿˣ⁽ᵐ⁺¹⁾ (basis matrix)
       - β = [1 α₁ α₂ ... αₘ]ᵀ ∈ ℝᵐ⁺¹ (coefficient vector)
       - C * α ≤ d (predicate constraints)
       - pred_lb ≤ α ≤ pred_ub (predicate bounds)

    Construction Methods:
    1. Full specification: Using V, C, d, pred_lb, pred_ub
    2. Box specification: Using state bounds lb, ub (creates a box-shaped star set)
    3. Random generation: Either with state bounds or with constraints
    """
    print_util('h1')
    print("Starting Star Set Construction Tutorial...")

    # Method 1: Construction using V, C, d, pred_lb, pred_ub
    print_util('h2')
    print("""1. Construction using basis matrix and constraints
    Example of 2D star set with:
    - Center vector: c = [0, 0]ᵀ
    - Basis vectors: v₁ = [1, 0]ᵀ, v₂ = [0, 1]ᵀ
    - Constraints: α₁ + α₂ ≤ 1
    - Bounds: -1 ≤ α₁ ≤ 1, -1 ≤ α₂ ≤ 1""")
    
    c1 = np.array([[0], [0]])        # center vector
    v1 = np.array([[1], [0]])         # basis vector 1
    v2 = np.array([[0], [1]])         # basis vector 2
    V = np.hstack((c1, v1, v2))       # V = [c1 v1 v2]
    
    pred_lb = np.array([-1, -1])       # -1 ≤ α₁, -1 ≤ α₂
    pred_ub = np.array([1, 1])        # α₁ ≤ 1, α₂ ≤ 1
    C = np.array([[1, 1]])            # α₁ + α₂ ≤ 1
    d = np.array([1])
    
    star1 = Star(V, C, d, pred_lb, pred_ub)
    print("\nCreated star set with basis matrix and constraints:")
    repr(star1)
    print(star1)
    plot_star(star1)
    print_util('h2')
    
    # Method 2: Construction using bounds only
    print_util('h2')
    print("""2. Construction using state bounds
    Example of 2D box-shaped star set with:
    - State bounds: -1 ≤ x₁ ≤ 1, -1 ≤ x₂ ≤ 1""")
    
    lb = np.array([-1.0, -1.0])           # lower bounds: x₁ ≥ -1, x₂ ≥ -1
    ub = np.array([1.0, 1.0])            # upper bounds: x₁ ≤ 1, x₂ ≤ 1
    
    star2 = Star(lb, ub)
    print("\nCreated box-shaped star set:")
    print(star2)
    plot_star(star2)
    print_util('h2')
    
    # Method 3: Random star set construction
    print_util('h2')
    print("3. Random star set construction with state bounds")
    star3 = Star.rand(2)              # 2D random star set
    print("\nCreated random star set:")
    print(star3)
    plot_star(star3)
    print_util('h2')

    # Method 4: Random star set with constraints
    print_util('h2')
    print("4. Random star set construction with constraints")
    star4 = Star.rand_polytope(2, 3)  # 2D random star with 3 constraints, number of constraints > number of dimensions
    print("\nCreated random constrained star set:")
    print(star4)
    plot_star(star4)
    print_util('h2')

    print("\nTutorial completed!")
    print_util('h1')

def star_affine_map():
    """
    Tutorial demonstrating affine mapping operations on Star sets.

    ----------------------
    Given a star set Θ = ⟨c, V, P, l, u⟩, an affine mapping defined by:
    Θ̄ = {y | y = Wx + b, x ∈ Θ}

    results in a new star set Θ̄ = ⟨c̄, V̄, P̄, l̄, ū⟩ where:
    - c̄ = Wc + b (new center)
    - V̄ = {Wv₁, Wv₂, ..., Wvₘ} (new basis vectors)
    - P̄ ≡ P (predicates unchanged)
    - l̄ ≡ l (lower bounds unchanged)
    - ū ≡ u (upper bounds unchanged)
    """
    print_util('h1')
    print("Starting Star Set Affine Mapping Tutorial...")

    lb = np.array([-1.0, -1.0])           # lower bounds: x₁ ≥ -1, x₂ ≥ -1
    ub = np.array([1.0, 1.0])            # upper bounds: x₁ ≤ 1, x₂ ≤ 1
    star = Star(lb, ub)

    print('\nOriginal star set:')
    print(star)
    plot_star(star)

    print_util('h2')
    print("""1. Affine mapping transformation
    y = Wx + b, where:
    - W: transformation matrix
    - b: offset vector""")

    # Define affine transformation
    W = np.array([[0.5, 0.5],
                    [0.5, -0.5]]) # Scaling + rotation
    b = np.array([1, 0]) # Translation

    print("\nAffine mapping parameters:")
    print(f"Transformation matrix (W):\n{W}")
    print(f"Offset vector (b):\n{b}")

    # Apply affine transformation
    star_mapped = star.affineMap(W, b)

    print('\nAffine mapped star set:')
    print(star_mapped)
    plot_star(star_mapped)
    print_util('h2')

    print("\nTutorial completed!")
    print_util('h1')

def star_estimate_ranges():
    """
    Tutorial demonstrating how to estimate ranges of Star sets.

    ----------------------
    Given a star set Θ = ⟨c, V, P, l, u⟩, where:
    - c is the center vector
    - V is the basis matrix
    - P defines the predicate constraints
    - l, u are the lower and upper bounds of predicate variables

    The range of the state vector x can be estimated using:
    l_est ≤ x = c + Vα ≤ u_est

    where:
    l_est = c + max(0,V)l + min(0,V)u
    u_est = c + max(0,V)u + min(0,V)l

    This method provides a quick over-approximation of the actual ranges
    without solving linear programming optimization problems.
    """
    print_util('h1')
    print("Starting Star Set Range Estimation Tutorial...")

    # Create random star set for demonstration
    lb = -np.random.rand(2)           # Random lower bounds in [-1,0]
    ub = np.random.rand(2)            # Random upper bounds in [0,1]
    star = Star(lb, ub)               # Random 2D star set

    print("\nCreated random star set with bounds:")
    print(f"Lower bounds (lb): {lb}")
    print(f"Upper bounds (ub): {ub}")

    print_util('h2')
    print("""1. Estimate range for single dimension i
    l_est[i] = c[i] + max(0,V[i,:])l + min(0,V[i,:])u
    u_est[i] = c[i] + max(0,V[i,:])u + min(0,V[i,:])l""")
    
    dim = 0  # Estimate range for first dimension
    est_min, est_max = star.estimateRange(dim)
    print(f'\nEstimated range for dimension {dim}:')
    print(f'Estimated min (l_est[{dim}]) = {est_min}')
    print(f'Estimated max (u_est[{dim}]) = {est_max}')
    print_util('h2')

    print_util('h2')
    print("""2. Estimate ranges for all dimensions
    l_est = c + max(0,V)l + min(0,V)u
    u_est = c + max(0,V)u + min(0,V)l""")
    
    est_mins, est_maxs = star.estimateRanges()
    print('\nEstimated ranges for all dimensions:')
    print(f'Estimated mins (l_est) = {est_mins}')
    print(f'Estimated maxs (u_est) = {est_maxs}')
    print_util('h2')

    print("\nTutorial completed!")
    print_util('h1')

def star_get_ranges():
    """
    Tutorial demonstrating how to compute exact ranges of Star sets.

    ----------------------
    Given a star set Θ = ⟨c, V, P, l, u⟩, the exact range of the iᵗʰ state x[i] 
    can be computed by solving linear programming optimization problems:

    x[i]_min = min(c[i] + Σⱼ(vⱼ[i]αⱼ))
    x[i]_max = max(c[i] + Σⱼ(vⱼ[i]αⱼ))
    subject to:
        - P(α) ≡ Cα ≤ d
        - l ≤ α ≤ u

    The problem is solved using linear programming (LP) with available solvers:
    - Gurobi (default): Commercial solver, most efficient
    - GLPK: Open-source alternative
    - linprog: SciPy's built-in solver
    """
    print_util('h1')
    print("Starting Star Set Exact Range Computation Tutorial...")
    np.set_printoptions(precision=6, suppress=True)

    # Create a random star set
    lb = -np.random.rand(2)
    ub = np.random.rand(2)
    star = Star(lb, ub)

    print("\nCreated random star set with bounds:")
    print(f"Lower bounds (l): {lb}")
    print(f"Upper bounds (u): {ub}")

    print_util('h2')
    print("""1. Computing exact min/max for single dimension
    min/max x[i] = c[i] + Σⱼ(vⱼ[i]αⱼ)
    subject to star set constraints""")

    dim = 0  # Compute range for first dimension
    exact_min = star.getMin(dim, 'gurobi')
    exact_max = star.getMax(dim, 'gurobi')
    print(f'\nExact range for dimension {dim}:')
    print(f'Minimum (x[{dim}]_min) = {exact_min:.6f}')
    print(f'Maximum (x[{dim}]_max) = {exact_max:.6f}')
    print_util('h2')

    print_util('h2')
    print("""2. Computing exact min/max for multiple dimensions
    For each dimension i in the selected map:
    min/max x[i] = c[i] + Σⱼ(vⱼ[i]αⱼ)""")

    map = [0, 1]  # Compute ranges for first two dimensions
    exact_mins = star.getMins(map, 'gurobi')
    exact_maxs = star.getMaxs(map, 'gurobi')
    print('\nExact ranges for selected dimensions:')
    for i, dim in enumerate(map):
        print(f'Dimension {dim}:')
        print(f'Minimum (x[{dim}]_min) = {exact_mins[i]:.6f}')
        print(f'Maximum (x[{dim}]_max) = {exact_maxs[i]:.6f}')
    print_util('h2')

    print_util('h2')
    print("""3. Computing exact ranges for all dimensions
    Solves optimization problems for all state variables""")
    exact_mins, exact_maxs = star.getRanges('gurobi')
    print('\nExact ranges for all dimensions:')
    print(f'Exact mins (x_min) = {exact_mins}')
    print(f'Exact maxs (x_max) = {exact_maxs}')
    print_util('h2')

    print("\nTutorial completed!")
    print_util('h1')

def star_constraint_operations():
    """
    Tutorial demonstrating constraint operations on Star sets.

    ----------------------
    Given a star set Θ = ⟨c, V, P, l, u⟩, constraints can be added in three equivalent ways:

    1. Adding single/multiple constraints

    2. Half-space intersection:
        Given half-space ℋ = {x | Hx ≤ g}, 
        Θ̄ = Θ ∩ ℋ = ⟨c̄, V̄, P̄, l̄, ū⟩ where:
        - c̄ = c (center unchanged)
        - V̄ = V (basis vectors unchanged)
        - P̄ = P ∧ P′ where P′(α) ≡ (H×Vₘ)α ≤ g - H×c
        Vₘ = [v₁ v₂ ... vₘ]
        - l̄ = l, ū = u (bounds unchanged)
    """
    print_util('h1')
    print("Starting Star Set Constraint Operations Tutorial...")

    # Create initial star set
    lb = np.array([-1.0, -1.0])           # lower bounds: x₁ ≥ -1, x₂ ≥ -1
    ub = np.array([1.0, 1.0])            # upper bounds: x₁ ≤ 1, x₂ ≤ 1
    star = Star(lb, ub)

    print("\nOriginal star set Θ:")
    print(star)
    plot_star(star)


    print_util('h2')
    print("""1. Adding a single constraint at a time
    Original: P(α) ≡ Cα ≤ d
    New constraint: c′α ≤ d′
    Result: C := [C; c′], d := [d; d′]""")

    star1 = copy.deepcopy(star)
    C1 = np.array([-1.0, 1.0])  # -α₁ + α₂ ≤ 1
    d1 = np.array([1])
    star1.addConstraint(C1, d1)

    C2 = np.array([1.0, -1.0]) # α₁ - α₂ ≤ 1
    d2 = np.array([1]) 
    star1.addConstraint(C2, d2)

    print("\nAfter adding constraint -α₁ + α₂ ≤ 1 and α₁ - α₂ ≤ 1:")
    print(star1)
    plot_star(star1)
    print_util('h2')


    print_util('h2')
    print("""2. Adding multiple constraints at once
    New constraints: C′α ≤ d′
    Result: C := [C; C′], d := [d; d′]""")

    star2 = copy.deepcopy(star)
    C = np.array([[-1.0, 1.0], 
                    [1.0, -1.0]])  # Combined constraints
    d = np.array([1, 1])
    star2.addMultipleConstraints(C, d)
    print("\nAfter adding multiple constraints:")
    print(star2)
    plot_star(star2)
    print_util('h2')


    print_util('h2')
    print("""3. Intersecting with half-space
    Half-space ℋ = {x | Hx ≤ g}
    Results in P′(α) ≡ (H×Vₘ)α ≤ g - H×c""")

    star3 = copy.deepcopy(star)
    H = np.array([[-1.0, 1.0],  # Half-space constraints
                    [1.0, -1.0]])
    g = np.array([1, 1])
    star3 = star3.intersectHalfSpace(H, g)
    print("\nAfter half-space intersection:")
    print(star3)
    plot_star(star3)
    print_util('h2')

    print("\nTutorial completed!")
    print_util('h1')

def star_set_operations():
    """
    Tutorial demonstrating set operations on Star sets.

    ----------------------
    1. Empty Set Checking:
        A star set is empty if its predicate P(α) has no feasible solution.

    2. Minkowski Sum:
        Given two stars Θ₁ = ⟨c₁, V₁, P₁, l₁, u₁⟩ and Θ₂ = ⟨c₂, V₂, P₂, l₂, u₂⟩,
        their Minkowski sum Θ = Θ₁ ⊕ Θ₂ = ⟨c, V, P, l, u⟩ where:
        - c = c₁ + c₂ (sum of centers)
        - V = [V₁ V₂] (concatenated basis matrices)
        - P = P₁ ∧ P₂ (conjunction of predicates)
        - l = [l₁ l₂]ᵀ (combined lower bounds)
        - u = [u₁ u₂]ᵀ (combined upper bounds)
    """
    print_util('h1')
    print("Starting Star Set Operations Tutorial...")

    print_util('h2')
    print("1. Emptiness checking\n")
    # Create non-empty star set
    lb = np.array([-1.0, -1.0])           # lower bounds: x₁ ≥ -1, x₂ ≥ -1
    ub = np.array([1.0, 1.0])            # upper bounds: x₁ ≤ 1, x₂ ≤ 1
    star = Star(lb, ub)
    print('Original star set:')
    print(star)

    empty = star.isEmptySet() # Check if star set is empty
    print(f"Is original star set empty? {empty}")

    # Create empty star set by adding infeasible constraint
    star1 = copy.deepcopy(star)
    C = np.array([-1.0, 0.0])  # Constraint: -α₁ ≤ -2
    d = np.array([-2])         # Infeasible with α₁ ≥ -1
    star1.addConstraint(C, d)
    print("\nStar set with infeasible constraint:")
    print(star1)
    empty1 = star1.isEmptySet()
    print(f"Is constrained star set empty? {empty1}")
    print_util('h2')


    print_util('h2')
    print("""2. Minkowski sum
    Θ = Θ₁ ⊕ Θ₂:
    - Center: c = c₁ + c₂
    - Basis: V = [V₁ V₂]
    - Predicates: P = P₁ ∧ P₂
    - Bounds: l = [l₁ l₂]ᵀ, u = [u₁ u₂]ᵀ""")

    # Create two random star sets
    # star1 = Star.rand(2)
    star1 = Star.rand_polytope(2, 3)
    print("\nStar set 1 (Θ₁):")
    print(star1)
    plot_star(star1)

    # star2 = Star.rand(2)
    star2 = Star.rand_polytope(2, 3)
    print("\nStar set 2 (Θ₂):")
    print(star2)
    plot_star(star2)

    # Compute Minkowski sum
    star_sum = star1.minKowskiSum(star2)
    print("\nMinkowski sum (Θ₁ ⊕ Θ₂):")
    print(star_sum)
    plot_star(star_sum)
    print_util('h2')

    print("\nTutorial completed!")
    print_util('h1')

if __name__ == "__main__":
    """
    Main function to run the star set tutorials.
    """
    star_construction()
    star_affine_map()
    star_estimate_ranges()
    star_get_ranges()
    star_constraint_operations()
    star_set_operations()