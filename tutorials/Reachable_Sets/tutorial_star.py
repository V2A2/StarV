"""
This tutorial demonstrates the two main ways to construct Star sets:
1. Using basis matrix, constraint matrix, and constraint vector, and predicate bounds
2. Using lower and upper predicate bounds

Star set defined by
x = c + a[1]*v[1] + a[2]*v[2] + ... + a[n]*v[n]
    = V * b,
where V = [c v[1] v[2] ... v[n]],
        b = [1 a[1] a[2] ... a[n]]^T,
        C*a <= d, constraints on a[i]
"""

from StarV.set.star import Star
import numpy as np
from StarV.util.plot import plot_star


def tutorial_custom_star():
    """
    Demonstrates creating a star set using basis matrix and constraints, and predicate bounds.
    """
    print("\n=== Example 1: Custom Star Set Creation ===")
    
    try:
        # Define basis vectors
        c1 = np.array([[1], [-1]])         # center vector
        v1 = np.array([[1], [0]])          # basis vector 1
        v2 = np.array([[0], [1]])          # basis vector 2
        
        # Combine into basis matrix V = [c1 v1 v2]
        V = np.hstack((c1, v1, v2))

        # equivalent to 
        # V = np.array([[1, 1, 0], [-1, 0, 1]])
        
        # Predicate constraints: C*[a] <= d
        # -1 <= a1 <= 1
        # 0 <= a2 <= 1
        # a1 + a2 <= 1

        pred_lb = np.array([-1, 0]) # -1 <= a1, 0 <= a2
        pred_ub = np.array([1, 1]) # a1 <= 1, a2 <= 1
        C = np.array([[1, 1]]) # a1 + a2 <= 1
        d = np.array([1])
        
        # Create star set
        star1 = Star(V, C, d, pred_lb, pred_ub)
        
        print("Successfully created star set")
        
    except Exception as e:
        print(f"Error creating star set: {str(e)}")

def tutorial_bounded_star():
    """
    Demonstrates creating a star set using bounds.
    """
    print("\n=== Example 2: Bounded Star Set Creation ===")
    
    try:
        # Define bounds for predicates
        lb = np.array([-2, -1])   # lower bounds: x1 >= -2, x2 >= -1
        ub = np.array([2, 1])     # upper bounds: x1 <= 2,  x2 <= 1
        
        # Create star set from predicate bounds
        star2 = Star(lb, ub)
        
        print("Successfully created star set")
        
    except Exception as e:
        print(f"Error creating star set: {str(e)}")

def main():
    """
    Main function to run the tutorial examples.
    """
    print("Starting Star Construction Tutorial...")
    
    # Run tutorial sections
    tutorial_custom_star()
    tutorial_bounded_star()
    
    print("\nTutorial completed!")

if __name__ == "__main__":
    main()