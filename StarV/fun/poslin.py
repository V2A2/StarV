"""
PosLin Class for reachability analysis of neural network layers with PosLin activation function.
Author: Yuntao Li
Date: 1/10/2024
"""
import numpy as np
from typing import List, Union, Tuple
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import multiprocessing
import ipyparallel
import copy


class PosLin:
    """
    PosLin Class for qualitative and quantitative reachability analysis of rectified linear units (ReLU).

    This class implements methods for computing the reachable set of a ReLU activation function,
    which is defined as f(x) = max(0, x). It supports both single-step and multi-step reachability
    analysis for Star and ProbStar sets.
    """

    @staticmethod
    def evaluate(x: np.ndarray) -> np.ndarray:
        """
        Evaluate the ReLU function element-wise on the input array.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Result of applying ReLU to the input array.
        """
        return np.maximum(x, 0)

    @staticmethod
    def stepReach(*args) -> List[Union[Star, ProbStar]]:
        """
        Compute the reachable set for a single step of the ReLU function.

        This method handles the case where the input set may partially overlap
        with the positive orthant, resulting in up to two output sets.

        Args:
            args: Variable length argument list.
                - args[0]: Input set (Star or ProbStar)
                - args[1]: Index of the dimension being processed
                - args[2] (optional): LP solver to use (default: 'gurobi')

        Returns:
            List[Union[Star, ProbStar]]: List of output sets after applying ReLU.

        Raises:
            ValueError: If the number of arguments is invalid or the input set type is unsupported.
        """
        if len(args) == 2:
            I, index = args
            lp_solver = 'gurobi'
        elif len(args) == 3:
            I, index, lp_solver = args
        else:
            raise ValueError("Invalid number of arguments. Expected 2 or 3.")

        if not isinstance(I, (Star, ProbStar)):
            raise ValueError(f"Input must be a Star or ProbStar set, got {type(I)}")

        xmin, xmax = I.estimateRange(index)

        if xmin >= 0:
            return [I]
        elif xmax <= 0:
            return [I.resetRow(index)]

        xmax = I.getMax(index, lp_solver)
        if xmax <= 0:
            return [I.resetRow(index)]

        xmin = I.getMin(index, lp_solver)
        if xmin >= 0:
            return [I]

        # Case where the set intersects both positive and negative orthants
        C = np.zeros(I.dim)
        C[index] = 1.0
        d = np.zeros(1)

        S1 = I.clone()
        S2 = I.clone()
        # S1 = copy.deepcopy(I)
        # S2 = copy.deepcopy(I)

        S1.addConstraint(C, d)  # x <= 0
        S1.resetRow(index)
        S2.addConstraint(-C, d)  # x >= 0

        return [S1, S2]

    @staticmethod
    def stepReachMultiInputs(*args) -> List[Union[Star, ProbStar]]:
        """
        Compute the reachable set for a single step of the ReLU function with multiple inputs.

        Args:
            args: Variable length argument list.
                - args[0]: List of input sets (Star or ProbStar)
                - args[1]: Index of the dimension being processed
                - args[2] (optional): LP solver to use (default: 'gurobi')

        Returns:
            List[Union[Star, ProbStar]]: List of output sets after applying ReLU to all inputs.

        Raises:
            ValueError: If the number of arguments is invalid or the input is not a list.
        """
        if len(args) == 2:
            I, index = args
            lp_solver = 'gurobi'
        elif len(args) == 3:
            I, index, lp_solver = args
        else:
            raise ValueError("Invalid number of arguments. Expected 2 or 3.")

        if not isinstance(I, list):
            raise ValueError(f"Input must be a list, got {type(I)}")

        S = []
        for input_set in I:
            S.extend(PosLin.stepReach(input_set, index, lp_solver))
        return S

    @staticmethod
    def reachExactSingleInput(*args) -> List[Union[Star, ProbStar]]:
        """
        Compute the exact reachable set for a single input set through all dimensions.

        This method applies the ReLU function to each dimension of the input set sequentially.

        Args:
            args: Variable length argument list.
                - args[0]: Input set (Star or ProbStar) or tuple containing the input set
                - args[1] (optional): LP solver to use (default: 'gurobi')

        Returns:
            List[Union[Star, ProbStar]]: List of output sets after applying ReLU to all dimensions.

        Raises:
            ValueError: If the number of arguments is invalid or the input set type is unsupported.
        """
        if isinstance(args[0], tuple):
            args = list(args[0])

        if len(args) == 1:
            In = args[0]
            lp_solver = 'gurobi'
        elif len(args) == 2:
            In, lp_solver = args
        else:
            raise ValueError("Invalid number of arguments. Expected 1 or 2.")

        if not isinstance(In, (Star, ProbStar)):
            raise ValueError(f"Input must be a Star or ProbStar set, got {type(In)}")

        S = [In]
        for i in range(In.dim):
            S = PosLin.stepReachMultiInputs(S, i, lp_solver)

        return S

    @staticmethod
    def reachExactMultiInputs(*args) -> List[Union[Star, ProbStar]]:
        """
        Compute the exact reachable set for multiple input sets.

        This method supports parallel computation if a pool is provided.

        Args:
            args: Variable length argument list.
                - args[0]: List of input sets (Star or ProbStar)
                - args[1] (optional): LP solver to use (default: 'gurobi')
                - args[2] (optional): Pool for parallel computation

        Returns:
            List[Union[Star, ProbStar]]: List of output sets after applying ReLU to all input sets.

        Raises:
            ValueError: If the number of arguments is invalid or the input is not a list.
        """
        if len(args) == 1:
            In = args[0]
            lp_solver = 'gurobi'
            pool = None
        elif len(args) == 2:
            In, lp_solver = args
            pool = None
        elif len(args) == 3:
            In, lp_solver, pool = args
        else:
            raise ValueError("Invalid number of arguments. Expected 1, 2, or 3.")

        if not isinstance(In, list):
            raise ValueError(f"Input must be a list, got {type(In)}")

        if pool is None:
            S = []
            for input_set in In:
                S.extend(PosLin.reachExactSingleInput(input_set, lp_solver))
        # elif isinstance(pool, multiprocessing.pool.Pool):
        #     S = []
        #     results = pool.map(PosLin.reachExactSingleInput, zip(In, [lp_solver] * len(In)))
        #     for result in results:
        #         S.extend(result)
        elif isinstance(pool, multiprocessing.pool.Pool):
            results = pool.starmap(PosLin.reachExactSingleInput, 
                                   [(input_set, lp_solver) for input_set in In])
            S = [item for sublist in results for item in sublist]
        else:
            raise ValueError(f"Unsupported pool type: {type(pool)}")

        return S





    @staticmethod
    def step_reach_star_approx(In: Star, index: int, lp_solver: str = 'gurobi') -> List[Star]:
        """
        Perform step-wise reachability approximation for a specified neuron index.

        This method approximates the reachable set for a single neuron within a Star or ProbStar set.
        The approximation uses linear programming to compute bounds and constructs new constraints
        to represent the approximated behavior of the ReLU function.

        Args:
            In (Union[Star, ProbStar]): The input set for approximation
            index (int): Index of the neuron to approximate
            lp_solver (str, optional): Linear programming solver to use. Defaults to 'gurobi'.

        Returns:
            Star: Approximated output set after applying reachability analysis

        Raises:
            TypeError: If input is not a Star or ProbStar set
            ValueError: If index is out of bounds

        Notes:
            The approximation process involves:
            1. Computing lower and upper bounds for the specified neuron
            2. Handling different cases based on these bounds:
               - If lower bound > 0: return input unchanged
               - If upper bound ≤ 0: zero out the corresponding row
               - Otherwise: construct new constraints for approximation
        """
        if not isinstance(In, (Star)):
            raise TypeError(f'Input must be a Star set, got {type(In).__name__}')
        
        if not 0 <= index < In.dim:
            raise ValueError(f'Index {index} out of bounds for dimension {In.dim}')

        # Get bounds for the neuron
        l = In.getMin(index=index, lp_solver=lp_solver)
        u = In.getMax(index=index, lp_solver=lp_solver)

        # Case 1: If lower bound > 0, neuron is always active
        if l > 0:
            return [In]

        # Case 2: If upper bound ≤ 0, neuron is always inactive
        if u <= 0:
            V = In.V.copy()
            V[index, :] = 0
            return [Star(V, In.C, In.d, In.pred_lb, In.pred_ub)]

        # Case 3: Mixed sign case - needs approximation
        n = In.nVars + 1

        # Construct new basis matrix
        V = np.hstack([In.V, np.zeros((In.dim, 1))])
        V[index] = 0
        V[index, n] = 1

        # Construct new constraints
        C1 = np.zeros((1, n))
        C1[0, n-1] = -1

        C2 = np.hstack([In.V[index, 1:n], -1])

        a = -u / (u - l)  # Slope for approximation
        C3 = np.hstack([a * In.V[index, 1:n], 1])

        # Handle empty constraint case
        if len(In.d) == 0:
            C0 = np.empty((0, n))
            d0 = np.empty(0)
        else:
            C0 = np.hstack([In.C, np.zeros((In.C.shape[0], 1))])
            d0 = In.d

        # Combine all constraints
        C = np.vstack([C0, C1, C2, C3])
        d = np.concatenate([d0, [0, -In.V[index, 0], a * (l - In.V[index, 0])]])
        
        # Update predicate bounds
        pred_lb = np.append(In.pred_lb, 0)
        pred_ub = np.append(In.pred_ub, u)

        return Star(V, C, d, pred_lb, pred_ub)

    @staticmethod
    def reach_star_approx(In: List[Star], lp_solver: str = 'gurobi') -> List[Star]:
        """
        Approximate reachability analysis for a Star set across all neurons.

        This method performs approximate reachability analysis by:
        1. Identifying neurons that are always inactive (upper bound ≤ 0)
        2. Processing neurons with mixed sign behavior (lower bound < 0 and upper bound > 0)
        using step-wise approximation.

        Args:
            In (Union[Star, ProbStar]): The input set for approximation
            lp_solver (str, optional): Linear programming solver to use. Defaults to 'gurobi'.

        Returns:
            Star: Approximated output set after applying reachability analysis

        Raises:
            TypeError: If input is not a Star or ProbStar set
        """
        if not isinstance(In, list):
            raise ValueError(f"Input must be a list, got {type(In)}")

        if not isinstance(In[0], (Star)):
            raise TypeError(f'Input set from list must be a Star set, got {type(In[0]).__name__}')

        # Get ranges for all neurons
        l, u = In[0].estimateRanges()

        # Handle neurons that are always inactive
        neg_ub_map = np.where(u <= 0)[0]
        V = In[0].V.copy()
        V[neg_ub_map, :] = 0
        I = Star(V, In[0].C, In[0].d, In[0].pred_lb, In[0].pred_ub)

        # Process neurons with mixed sign behavior
        mixed_sign_map = np.where((l < 0) & (u > 0))[0]
        for i in mixed_sign_map:
            I = PosLin.step_reach_star_approx(I, index=i, lp_solver=lp_solver)

        return [I]


    @staticmethod
    def multi_step_reach_star_approx(I: Star, 
                                   l_bounds: np.ndarray, 
                                   u_bounds: np.ndarray, 
                                   lp_solver: str = 'gurobi') -> Tuple[Star, np.ndarray, np.ndarray, np.ndarray]:
        """
        Approximate the ReLU function over a given input set using linear programming.

        This method performs a multi-step approximation process:
        1. Identifies neurons with non-positive upper bounds
        2. Processes neurons with mixed-sign behavior
        3. Calculates maximum and minimum values using LP for specific neurons

        Args:
            I (Union[Star, ProbStar]): Input set to be approximated
            l_bounds (np.ndarray): Lower bounds of neurons
            u_bounds (np.ndarray): Upper bounds of neurons
            lp_solver (str, optional): Linear programming solver. Defaults to 'gurobi'.

        Returns:
            Tuple containing:
                - Union[Star, ProbStar]: Modified input set after approximation
                - np.ndarray: Lower bounds of the active set
                - np.ndarray: Upper bounds of the active set
                - np.ndarray: Indices of neurons in the active set

        Raises:
            TypeError: If input is not a Star or ProbStar set
        """
        if not isinstance(I, (Star)):
            raise TypeError(f'Input must be a Star set, got {type(I).__name__}')

        # Find neurons with non-positive upper bounds
        n_p_ub_idxs = np.where(u_bounds <= 0)[0]

        # Find neurons with mixed-sign bounds
        m_sign_idxs = np.where((l_bounds < 0) & (u_bounds > 0))[0]
        
        # Calculate maximums for mixed-sign neurons
        max_vals = I.getMaxs(m_sign_idxs, lp_solver)
        
        # Process neurons with non-positive maximums
        n_p_max_idxs = m_sign_idxs[max_vals <= 0]
        
        # Reset rows for inactive neurons
        reset_idxs = np.concatenate([n_p_ub_idxs, n_p_max_idxs])
        In = I.resetRows(reset_idxs)

        # Process neurons with positive maximums
        p_max_mask = max_vals > 0
        p_max_idxs = m_sign_idxs[p_max_mask]
        xmax_ps = max_vals[p_max_mask]

        # Calculate minimums for potentially active neurons
        min_vals = I.getMins(p_max_idxs, lp_solver)
        
        # Final processing of active neurons
        n_min_mask = min_vals < 0
        final_idxs = p_max_idxs[n_min_mask]
        lb = min_vals[n_min_mask]
        ub = xmax_ps[n_min_mask]

        return In, lb, ub, final_idxs

    @staticmethod
    def add_tri_constraints(I: Star, 
                          idxs: np.ndarray, 
                          l: np.ndarray, 
                          u: np.ndarray) -> Star:
        """
        Add triangular constraints to model activation functions like ReLU.

        This method constructs new constraints and updates the Star set by:
        1. Creating new basis matrix with additional dimensions
        2. Constructing constraint matrices for activation function behavior
        3. Combining existing and new constraints

        Args:
            I (Union[Star, ProbStar]): Input set to be modified
            idxs (np.ndarray): Indices of neurons for constraint application
            l (np.ndarray): Lower bounds of neurons
            u (np.ndarray): Upper bounds of neurons

        Returns:
            Star: Modified Star set with new constraints

        Raises:
            TypeError: If input is not a Star or ProbStar set
        """
        if not isinstance(I, (Star)):
            raise TypeError(f'Input must be a Star set, got {type(I).__name__}')

        N, m = I.dim, len(idxs)
        n = I.nVars

        # Construct new basis matrix
        V1 = I.V.copy()
        V1[idxs] = 0
        V2 = np.zeros((N, m))
        V2[idxs, np.arange(m)] = 1
        new_V = np.hstack([V1, V2])

        # Handle existing constraints
        if I.C.size == 0:
            C0 = np.empty((0, n + m))
            d0 = np.empty(0)
        else:
            C0 = np.hstack([I.C, np.zeros((I.C.shape[0], m))])
            d0 = I.d.copy()

        # Construct new constraint matrices
        C1 = np.hstack([np.zeros((m, n)), -np.eye(m)])
        d1 = np.zeros(m)

        C2 = np.hstack([I.V[idxs, 1:n+1], -V2[idxs]])
        d2 = -I.V[idxs, 0]

        # Compute slope coefficients
        coeff = u / (u - l)
        C3 = np.hstack([-coeff[:, None] * I.V[idxs, 1:n+1], V2[idxs]])
        d3 = coeff * (I.V[idxs, 0] - l)

        # Combine all constraints
        new_C = np.vstack([C0, C1, C2, C3])
        new_d = np.concatenate([d0, d1, d2, d3])

        # Update predicate bounds
        new_pred_lb = np.concatenate([I.pred_lb, np.zeros(m)])
        new_pred_ub = np.concatenate([I.pred_ub, u])

        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)

    @staticmethod
    def reach_star_approx_fast(In: List[Star], lp_solver: str = 'gurobi') -> List[Star]:
        """
        Perform fast step-wise reachability approximation.

        This method combines multi-step approximation with triangular constraint addition
        for efficient reachability analysis.

        Args:
            In (Union[Star, ProbStar]): Input set for approximation
            lp_solver (str, optional): Linear programming solver. Defaults to 'gurobi'.

        Returns:
            Star: Approximated output set

        Raises:
            TypeError: If input is not a Star set
            ValueError: If approximation fails
        """

        if not isinstance(In, list):
            raise ValueError(f"Input must be a list, got {type(In)}")

        if not isinstance(In[0], (Star)):
            raise TypeError(f'Input set from list must be a Star set, got {type(In[0]).__name__}')

        I = In[0].clone()
        l, u = I.estimateRanges()

        # Quick returns for simple cases
        if np.all(l > 0):
            return I
        elif np.all(u < 0):
            return I.resetRows(np.arange(I.dim))

        # Perform multi-step approximation
        new_I, new_l, new_u, idx_map = PosLin.multi_step_reach_star_approx(
            I=I, l_bounds=l, u_bounds=u, lp_solver=lp_solver)

        if not isinstance(new_I, (Star)):
            raise ValueError('Approximate reachability only supports Star types')

        return [PosLin.add_tri_constraints(I=new_I, idxs=idx_map, l=new_l, u=new_u)]


        
