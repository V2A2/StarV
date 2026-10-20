"""
Multiply operator
Zhuoyang Zhou, 02/15/2026
"""
from StarV.dynamic.multiply import Multiply
from StarV.set.star import Star

class MultiplyOperater(object):
    """ Multiply class for reachability with McCormick-based approach
        Author: Zhuoyang Zhou
        Date: 02/15/2026
    """
    
    @staticmethod
    def reach(In, idx_x, idx_y, method='approx', lp_solver='gurobi', RF=0.0, split=False, max_splits=0):
        """main reachability method for multiplication operation
            Args:
                @In: an input set (Star or list of Stars)
                @idx_x: index of first operand variable (int)
                @idx_y: index of second operand variable (int)
                @method: method: 'approx' (default, only supported method)
                @lp_solver: lp solver: 'gurobi' (default), 'glpk', or 'linprog'
                @RF: relax-factor from 0 to 1 (0 by default)
                @split: whether to split the result (False by default)
                @max_splits: maximum number of splits (0 by default)

            Return:
                @R: a reachable set (Star or list of Stars)        
        """
        
        # Validate inputs
        assert isinstance(idx_x, int), "idx_x must be an integer"
        assert isinstance(idx_y, int), "idx_y must be an integer"
        
        if method != 'approx':
            raise Exception('error: only approx method is supported for multiply operator')
        
        # Handle list input
        if isinstance(In, list):
            result = []
            for I in In:
                # Validate indices for Star input
                if isinstance(I, Star):
                    assert 0 <= idx_x < I.dim, f"idx_x {idx_x} is out of range [0, {I.dim-1}]"
                    assert 0 <= idx_y < I.dim, f"idx_y {idx_y} is out of range [0, {I.dim-1}]"
                
                # Call dynamic multiply
                R = Multiply.reachApprox_star(I, idx_x, idx_y, lp_solver=lp_solver, 
                                             RF=RF, split=split, max_splits=max_splits)
                
                # Handle split results
                if split and isinstance(R, list):
                    result.extend(R)
                else:
                    result.append(R)
            
            return result
        else:
            # Single Star input
            if isinstance(In, Star):
                assert 0 <= idx_x < In.dim, f"idx_x {idx_x} is out of range [0, {In.dim-1}]"
                assert 0 <= idx_y < In.dim, f"idx_y {idx_y} is out of range [0, {In.dim-1}]"
            
            return Multiply.reachApprox_star(In, idx_x, idx_y, lp_solver=lp_solver, RF=RF, split=split, max_splits=max_splits)