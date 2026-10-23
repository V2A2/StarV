"""
Recurrent Layer Class
Qing Liu, 07/15/2025
"""
from scipy.io import loadmat
import os
import mat73
import numpy as np
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.net.network import NeuralNetwork
from StarV.util.load_rnn import load_simple_rnn, get_Star_set,get_ProbStar_set


class RecurrentLayer(object):
    """ RecurrentLayer class
        properties: 
            Whx: weights_mat for input states to hidden ststes
            Whh: weights mat for hidden states to hiedden states
            bh: bias vector for hidden states
            fh: activation function for hidden nodes
            Woh:  weights mat for hidden states to output states
            bo:  bias vector for output states
            fo: activation function for output nodes
    """
    def __init__(self,Whx, Whh, bhx, Woh, bo,bhh=None):
        assert isinstance(Whh, np.ndarray), " Weights mat for hidden states to hiedden states should be a 2d numpy array"
        assert isinstance(bhx, np.ndarray), "Input to hidden layer bias vector should be a 1d numpy array"
        assert isinstance(Whx, np.ndarray), "Weights_mat for input states should be a 2d numpy array"
        if bhh is not None:
            assert isinstance(bhh, np.ndarray), "Bias between hidden states should be a 1d numpy array"
        assert isinstance(Woh, np.ndarray), "Weight mat for hidden states to output states should be a 2d numpy array"
        assert isinstance(bo, np.ndarray), "Output later bias vector should be a 1d numpy array"

        self.Whx = Whx
        self.Whh = Whh
        self.bhx = bhx
        if bhh is not None:
            self.bhh = bhh
        self.Woh = Woh
        self.bo = bo
        self.in_dim = Whx.shape[1] 
        self.out_dim = Woh.shape[0] 

    @classmethod
    def rand(cls,in_dim, out_dim):
        Whx = np.round(np.random.rand(out_dim, in_dim),2)
        Whh = np.random.rand(out_dim, out_dim)
        bh =np.round(np.random.rand(out_dim),2)
        Woh = np.random.rand(out_dim, out_dim)
        bo = np.random.rand(out_dim)
        bhh = np.random.rand(out_dim)
        return RecurrentLayer( Whx,Whh, bh, Woh, bo,bhh)
    
    def __str__(self):
        print('Layer type: {}'.format(self.__class__.__name__))
        print('Input state to Hiiden state weight matrix: {}'.format(self.Whx))
        print('Hidden state bias vector: {}'.format(self.bhx))
        print('Hidden state to Output state weight matrix: {}'.format(self.Woh))
        print('Output state bias vector: {}'.format(self.bo))
        print('')
        return '\n'

    def info(self):
        print(self)



    def reachExact(self, In, method="exact", lp_solver="gurobi", pool=None, RF=0.0, DR=0):
        """
        Perform exact reachability analysis of an RNN with ReLU activation.

        Args:
            In (list): List of input sets (one per timestep).
            method (str): Reachability method, default "exact".
            lp_solver (str): Linear programming solver, default 'gurobi'.
            pool: Optional multiprocessing pool.
            RF (float): Reserved for future use.
            DR (int): Reserved for future use.

        Returns:
            list: List of reachable output sets at each timestep.
        """
         
        assert isinstance(In,list), 'error: input must be a list'

        print(f"\n~~~~~~~~ Using {method} method for reachability ~~~~~~~~")

        H = []  # Hidden state reachable sets per timestep
        O = []  # Output reachable sets per timestep

        for t, I in enumerate(In):
            print(f"\n----- Processing timestep {t} -----")
            print(f"=========== number of input sets in step {t}:{len(I)}==========")

            if t == 0:
                # First timestep: h0 = ReLU(Whx * x + bhx)
        
                WIn = I.affineMap(self.Whx, self.bhx)
                h_out  = ReLULayer.reach([WIn], method=method)
                hidden_states = h_out

            else:
                # Subsequent timesteps: h_t = ReLU(Whx * x_t + bhx + Whh * h_{t-1})
                hidden_states = []
                prev_hidden = H[t - 1]
                # print("===== first affine for initial input set ========")
                WIn = I.affineMap(self.Whx, self.bhx)
                # print(f" for minsum === \n WIn{t}: V_shape:{WIn.V.shape}, C:{WIn.C},C_shape:{WIn.C.shape}d:{WIn.d}")

                for k, h_prev in enumerate(prev_hidden):
                    # print("==== second affine for h_recurrent====")
                    if self.bhh is not None:
                        h_recurrent = h_prev.affineMap(self.Whh,self.bhh)
                    else:
                        h_recurrent = h_prev.affineMap(self.Whh)
                    # print("\n====== end affine======")

                    # if len(h_recurrent.C) == 0 :
                        # print(f"\n h_recurrent  V_type:{(type(h_recurrent.V))}, V_shape:{h_recurrent.V.shape},\n C_type:{type(h_recurrent.C)},C: {h_recurrent.C}d:{h_recurrent.d},h_pred_lb:{h_recurrent.pred_lb}")
                    # else:
                        # print(f"\n h_recurrent  V_type:{(type(h_recurrent.V))}, V_shape:{h_recurrent.V.shape},\n C_type:{type(h_recurrent.C.shape)},C: {h_recurrent.C}d:{h_recurrent.d},h_pred_lb:{h_recurrent.pred_lb}")

                    summed = h_recurrent.minKowskiSum(WIn)
                    # print(f"summed V_type:{(type(summed.V))}, V_shape:{summed.V.shape},\n C_type:{type(summed.C)},C: {summed.C}d:{summed.d},h_pred_lb:{summed.pred_lb}")
                    # Apply ReLU
                    h_out = ReLULayer.reach([summed], method=method)
                    # print(f"Number of h_out sets after relu in step {t}:{len(h_out)}")
                    hidden_states.extend(h_out)
                # print(f"Number of hidden_states sets after minsum in step {t}:{len(hidden_states)}")

            # Save hidden states
            H.append(hidden_states)

            oi = []
            print(f"number of output sets in step {t} for hidden states:{len(hidden_states)}")
            for h in hidden_states:
                 outputs_t = h.affineMap(self.Woh, self.bo) 
                 oi.append(outputs_t)
            O.append(oi)

        print("\n===== Reachability analysis using exactReach complete =====")
        print(f"Total timesteps: {len(O)}")


        return O




    def reachApprox(self, In, method="approx", lp_solver="gurobi", pool=None, RF=0.0, DR=0):
        """
        Perform approximate reachability analysis of an RNN with ReLU activation.

        Args:
            In (list): List of input sets (one per timestep).
            method (str): Reachability method, default "approx".
            lp_solver (str): Linear programming solver, default 'gurobi'.
            pool: Optional multiprocessing pool.
            RF (float): Reserved for future use.
            DR (int): Reserved for future use.

        Returns:
            list: List of reachable output sets at each timestep.
        """
        print(f"\n~~~~~~~~ Using {method} method for reachability ~~~~~~~~")

        assert isinstance(In,list), 'error: input must be a list'
        # assert isinstance(In[0],Star), 'error: input set is not a Star set'

        H = []  # Hidden state reachable sets
        O = []  # Output reachable sets

        for t, I in enumerate(In):
            print(f"\n----- Processing timestep {t} -----")

            if t == 0:
                WIn= I.affineMap(self.Whx, self.bhx)
                hidden_states = ReLULayer.reach(WIn, method=method, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=False)

            else:
                h_prev = H[t - 1]
                # Remaining timesteps: h_t = ReLU(Whx * x_t + bhx + Whh * h_{t-1})
                WIn = I.affineMap(self.Whx, self.bhx)
                if self.bhh is not None:
                    h_recurrent = h_prev.affineMap(self.Whh,self.bhh)
                else:
                    h_recurrent = h_prev.affineMap(self.Whh)
                h_sum = h_recurrent.minKowskiSum(WIn)
                hidden_states = ReLULayer.reach(
                    h_sum, method=method, lp_solver=lp_solver, pool=pool,
                    RF=RF, DR=DR, show=False
                )

            # Save hidden state
            print(f"number of output sets in step {t} for hidden states:{len(hidden_states)}")
            H.append(hidden_states)

            # Compute output: y_t = Woh * h_t + bo
            o_t = hidden_states.affineMap(self.Woh, self.bo)
            O.append(o_t)

        print("\n===== Approximate reachability analysis with reachApprox complete =====")
        print(f"Total timesteps: {len(O)}")
        print(f"type of each out set:{type(O[0])}")
        return O


    # def reachExactBranches(self, In, post_layers=None, lp_solver="gurobi", pool=None,
    #                        p_filter=None, show=False):
    #     """Exact reachability with branch tracking (for ProbStarTL).

    #     This returns *branch signals* instead of per-time unions:
    #         branch_k = [Y0, Y1, ..., Y(T-1)]

    #     Key assumptions for correctness (Method 1):
    #     - `In` is a list of ProbStars that all share the same global predicate vector
    #       (same nVars, mu, Sig, pred_lb, pred_ub).
    #     - ReLU is handled with the exact split method (no new predicate variables).
    #     - Any additional ReLU layers in `post_layers` will also be tracked as branches,
    #       and their predicate constraints will be propagated forward by restricting
    #       the hidden state with the same predicate constraints.
    #     """

    #     assert isinstance(In, list), 'error: input must be a list'
    #     assert len(In) > 0, 'error: input is empty'
    #     assert all(isinstance(s, ProbStar) for s in In), 'error: input must be a list of ProbStars'

    #     if post_layers is None:
    #         post_layers = []
    #     assert isinstance(post_layers, list), 'error: post_layers must be a list'

    #     # All time steps must live in the same predicate space.
    #     nVars0 = In[0].nVars
    #     for i, s in enumerate(In):
    #         assert s.nVars == nVars0, f'error: In[{i}].nVars={s.nVars} differs from In[0].nVars={nVars0}'

    #     def _apply_post_layers_exact(sets):
    #         """Apply post_layers to a list of ProbStars at one timestep (exact, with splitting)."""
    #         cur = sets
    #         for layer in post_layers:
    #             if isinstance(layer, FullyConnectedLayer):
    #                 nxt = []
    #                 for S in cur:
    #                     nxt.append(S.affineMap(layer.W, layer.b))
    #                 cur = nxt
    #             elif isinstance(layer, ReLULayer):
    #                 # ReLULayer expects a list input and returns a list output in exact mode.
    #                 cur = ReLULayer.reach(cur, method="exact", lp_solver=lp_solver, pool=pool, RF=0.0, DR=0, show=False)
    #             else:
    #                 # fallback: use the layer's reach API (if any)
    #                 if hasattr(layer, "reach"):
    #                     cur = layer.reach(cur, method="exact", lp_solver=lp_solver, pool=pool, RF=0.0, DR=0, show=False)
    #                 else:
    #                     raise Exception(f"error: unsupported post layer type: {type(layer)}")
    #         return cur

    #     def _restrict_hidden_by_output(h, y):
    #         """Propagate branch constraints to hidden state for future time steps."""
    #         if len(y.C) == 0:
    #             return h
    #         return ProbStar(h.V, y.C, y.d, h.mu, h.Sig, y.pred_lb, y.pred_ub)

    #     branches = []  # list of (hidden_state, signal)

    #     for t, I in enumerate(In):
    #         if show:
    #             print(f"[reachExactBranches] timestep {t}: {len(branches) if t > 0 else 0} active branches")

    #         if t == 0:
    #             # h0 = ReLU(Whx*x0 + bhx)
    #             WIn = I.affineMap(self.Whx, self.bhx)
    #             hidden_sets = ReLULayer.reach([WIn], method="exact", lp_solver=lp_solver, pool=pool, show=False)

    #             new_branches = []
    #             for h in hidden_sets:
    #                 o0 = h.affineMap(self.Woh, self.bo)
    #                 outs = _apply_post_layers_exact([o0])
    #                 for y in outs:
    #                     h_next = _restrict_hidden_by_output(h, y)
    #                     if p_filter is not None and h_next.estimateProbability() < p_filter:
    #                         continue
    #                     new_branches.append((h_next, [y]))
    #             branches = new_branches
    #         else:
    #             WIn = I.affineMap(self.Whx, self.bhx)
    #             new_branches = []
    #             for h_prev, sig in branches:
    #                 # h_t = ReLU(Whx*x_t + bhx + Whh*h_{t-1} + bhh)
    #                 if self.bhh is not None:
    #                     h_recurrent = h_prev.affineMap(self.Whh, self.bhh)
    #                 else:
    #                     h_recurrent = h_prev.affineMap(self.Whh)

    #                 # IMPORTANT: `add` is pointwise addition in the same predicate space (global predicates).
    #                 summed = h_recurrent.add(WIn)
    #                 hidden_sets = ReLULayer.reach([summed], method="exact", lp_solver=lp_solver, pool=pool, show=False)
    #                 for h in hidden_sets:
    #                     ot = h.affineMap(self.Woh, self.bo)
    #                     outs = _apply_post_layers_exact([ot])
    #                     for y in outs:
    #                         h_next = _restrict_hidden_by_output(h, y)
    #                         if p_filter is not None and h_next.estimateProbability() < p_filter:
    #                             continue
    #                         new_branches.append((h_next, sig + [y]))
    #             branches = new_branches

    #     return [sig for _, sig in branches]

    

    def reach(self,In, method = "exact", lp_solver='gurobi', pool=None, RF=0.0, DR=0):
        if method is None:
            method = "exact"
        if method == "exact":
            S = self.reachExact(In, method, lp_solver, pool, RF, DR)
            return S
        elif method == "approx":
            return self.reachApprox(In, method, lp_solver, pool, RF, DR)
        else:
            raise Exception(f"error: unknown reachability method: {method}")

