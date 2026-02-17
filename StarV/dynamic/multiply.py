"""
Multiply operator: z = x * y with McCormick over-approximation
Zhuoyang Zhou, 02/15/2026
"""

import numpy as np
from StarV.set.star import Star


class Multiply(object):
    """
    Multiply Class for reachability analysis
    Implements McCormick envelope sound over-approximation for z = x * y
    Author: Zhuoyang Zhou
    Date: 02/15/2026
    """

    @staticmethod
    def reachApprox_star(I, idx_x, idx_y, lp_solver='gurobi', RF=0.0, split=False, max_splits=0):
        """
        Compute reachable set approximation for z = x * y using McCormick over-approximation.
        
        The product z is appended as the last dimension of the output Star.
        
        Parameters:
        -----------
        I : Star
            Input Star set
        idx_x : int
            Index of first operand (x) in the state
        idx_y : int
            Index of second operand (y) in the state
        lp_solver : str
            Linear programming solver ('gurobi', 'glpk', etc.)
        RF : float
            Relaxation factor for range computation (0 to 1)
        
        Returns:
        --------
        Star
            Output Star set with z = x*y appended as last dimension
        """
        if split:
            raise Exception("error: split is not implemented in dynamic.multiply yet")
        assert isinstance(I, Star), 'error: input set must be a Star set'
        assert isinstance(idx_x, int), 'error: idx_x must be an integer'
        assert isinstance(idx_y, int), 'error: idx_y must be an integer'
        assert 0 <= idx_x < I.dim, f'error: idx_x {idx_x} is out of range [0, {I.dim-1}]'
        assert 0 <= idx_y < I.dim, f'error: idx_y {idx_y} is out of range [0, {I.dim-1}]'
        
        N = I.dim
        
        # Get ranges for the two variables under constraints
        lx = I.getMin(index=idx_x, lp_solver=lp_solver)
        ux = I.getMax(index=idx_x, lp_solver=lp_solver)
        ly = I.getMin(index=idx_y, lp_solver=lp_solver)
        uy = I.getMax(index=idx_y, lp_solver=lp_solver)
        
        # Apply relaxation factor if specified
        if RF > 0.0 and RF <= 1.0:
            wx = (ux - lx) * RF / 2
            wy = (uy - ly) * RF / 2
            lx = lx - wx
            ux = ux + wx
            ly = ly - wy
            uy = uy + wy
        
        # Compute output bounds for z = x * y
        zl = np.min([lx * ly, lx * uy, ux * ly, ux * uy])
        zu = np.max([lx * ly, lx * uy, ux * ly, ux * uy])
        
        # Center and radius for z
        cz = 0.5 * (zl + zu)
        vz = 0.5 * (zu - zl)
        
        # Create new V matrix (extend with one row for z, one column for new predicate beta)
        new_V = np.zeros((N + 1, I.V.shape[1] + 1))
        new_V[0:N, 0:I.V.shape[1]] = I.V
        new_V[N, 0] = cz
        new_V[N, I.nVars + 1] = vz
        
        # Extend existing constraints with zeros for the new predicate variable
        if len(I.C) > 0:
            C0 = np.hstack([I.C, np.zeros((I.C.shape[0], 1))])
            d0 = I.d.copy()
        else:
            C0 = np.empty((0, I.nVars + 1))
            d0 = np.empty((0,))
        
        # Build 4 McCormick constraints
        C_mcc_list = []
        d_mcc_list = []
        
        # Constraint 1: z >= lx*y + ly*x - lx*ly
        # Rearranged: -z + lx*y + ly*x <= lx*ly
        c1 = np.hstack([lx * I.V[idx_y, 1:] + ly * I.V[idx_x, 1:], np.array([-vz])])
        d1 = lx * ly + cz - lx * I.V[idx_y, 0] - ly * I.V[idx_x, 0]
        C_mcc_list.append(c1)
        d_mcc_list.append(d1)
        
        # Constraint 2: z >= ux*y + uy*x - ux*uy
        # Rearranged: -z + ux*y + uy*x <= ux*uy
        c2 = np.hstack([ux * I.V[idx_y, 1:] + uy * I.V[idx_x, 1:], np.array([-vz])])
        d2 = ux * uy + cz - ux * I.V[idx_y, 0] - uy * I.V[idx_x, 0]
        C_mcc_list.append(c2)
        d_mcc_list.append(d2)
        
        # Constraint 3: z <= ux*y + ly*x - ux*ly
        # Rearranged: z - ux*y - ly*x <= -ux*ly
        c3 = np.hstack([-ux * I.V[idx_y, 1:] - ly * I.V[idx_x, 1:], np.array([vz])])
        d3 = -ux * ly - cz + ux * I.V[idx_y, 0] + ly * I.V[idx_x, 0]
        C_mcc_list.append(c3)
        d_mcc_list.append(d3)
        
        # Constraint 4: z <= lx*y + uy*x - lx*uy
        # Rearranged: z - lx*y - uy*x <= -lx*uy
        c4 = np.hstack([-lx * I.V[idx_y, 1:] - uy * I.V[idx_x, 1:], np.array([vz])])
        d4 = -lx * uy - cz + lx * I.V[idx_y, 0] + uy * I.V[idx_x, 0]
        C_mcc_list.append(c4)
        d_mcc_list.append(d4)
        
        C_mcc = np.array(C_mcc_list)
        d_mcc = np.array(d_mcc_list)
        
        # Stack all constraints
        if C_mcc.shape[0] > 0:
            new_C = np.vstack([C0, C_mcc])
            new_d = np.hstack([d0, d_mcc])
        else:
            new_C = C0
            new_d = d0
        
        # Create new predicate bounds
        new_pred_lb = np.hstack([I.pred_lb, -1.0])
        new_pred_ub = np.hstack([I.pred_ub, 1.0])
        
        # Create and return new Star
        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)