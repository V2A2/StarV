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
                WIn = I.affineMap(self.Whx, self.bhx)

                for k, h_prev in enumerate(prev_hidden):
                    if self.bhh is not None:
                        h_recurrent = h_prev.affineMap(self.Whh,self.bhh)
                    else:
                        h_recurrent = h_prev.affineMap(self.Whh)
                    summed = h_recurrent.minKowskiSum(WIn)
                    # Apply ReLU
                    h_out = ReLULayer.reach([summed], method=method)
                    hidden_states.extend(h_out)

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


    def reachExactBranches(self, In, post_layers=None, lp_solver="gurobi", pool=None,
                           p_filter=None, show=False, post_start_t=None):
        """  Branch-based reachability for RNNs

        Args:
            post_layers:
                List of post layers to apply to each RNN output after post_start_t. Supported layer types: FullyConnectedLayer, ReLULayer. Default: None (no post layers).
            p_filter:
                Branch pruning threshold at/after post_start_t based on output
                set probability. If p_filter = 0, no pruning (exact method); if > 0, prune branches with estimated output probability < p_filter (approximate method).
            post_start_t:
                Start timestep to record outputs into branch traces.
                Default: len(In)//2.
        """

        # Qing Liu, 02/15/2026
        # Update: post-reachability branch filtering with time tags, 04/03/2026

        assert isinstance(In, list), 'error: input must be a list'
        assert len(In) > 0, 'error: input is empty'
        assert all(isinstance(s, ProbStar) for s in In), 'error: input must be a list of ProbStars'

        if post_layers is None:
            post_layers = []
        assert isinstance(post_layers, list), 'error: post_layers must be a list'

        if post_start_t is None:
            post_start_t = len(In) // 2
        else:
            assert isinstance(post_start_t, int), 'error: post_start_t should be an int'
            assert 0 <= post_start_t <= len(In), 'error: invalid post_start_t'

        if p_filter is None:
            p_filter = 0.0
        if p_filter < 0.0:
            raise RuntimeError('error: p_filter should be >= 0')
        
        if show:
            if p_filter == 0.0:
                print(f"Using exact reachability with branch tracing and no pruning (p_filter=0.0)")
            else:
                print(f"Using approximate reachability with branch tracing and pruning threshold p_filter={p_filter}")
        
        p_ignored = 0.0


        # Branch elements: (hidden_state_for_next_step, trace_after_post_start)
        branches = [(None, [])]
        hidden_states_all_steps = []
        hidden_output_all_steps = []

        def propagate_hidden_through_post_layers(h, h_out):
            """Apply post layers to one RNN output and return (h_next, y) pairs."""
            current = [h_out]
            for layer in post_layers:
                if isinstance(layer, FullyConnectedLayer):
                    nxt = []
                    for S in current:
                        S1 = S.affineMap(layer.W, layer.b)
                        nxt.append(S1)
                    current = nxt
                elif isinstance(layer, ReLULayer):
                    current = ReLULayer.reach(
                        current,
                        method="exact",
                        lp_solver=lp_solver,
                        pool=pool,
                        RF=0.0,
                        DR=0,
                        show=False,
                    )
                else:
                    raise Exception(f"error: unsupported post layer type: {type(layer)}")

            h_pairs = []
            for net_out in current:
                # Keep predicate consistency across branch evolution by reusing
                # post-layer constraints on hidden state for next recurrent step.
                if len(net_out.C) == 0:
                    h_next_set = h
                else:
                    h_next_set = ProbStar(h.V, net_out.C, net_out.d, h.mu, h.Sig, h.pred_lb, h.pred_ub)
                h_pairs.append((h_next_set, net_out))
            return h_pairs

        for t, I in enumerate(In):
            new_branches = []
            hidden_sets_step = []
            hidden_output_step = []

            WIn = I.affineMap(self.Whx, self.bhx)

            for i, (h_prev_post, trace) in enumerate(branches):
                if t == 0:
                    hidden_sets = ReLULayer.reach([WIn], method="exact", lp_solver=lp_solver, pool=pool, show=False)
                else:
                    if self.bhh is not None:
                        h_recurrent = h_prev_post.affineMap(self.Whh, self.bhh)
                    else:
                        h_recurrent = h_prev_post.affineMap(self.Whh)
                    summed = h_recurrent.minKowskiSum(WIn)
                    hidden_sets = ReLULayer.reach([summed], method="exact", lp_solver=lp_solver, pool=pool, show=False)

                hidden_sets_step.extend(hidden_sets)

                for h in hidden_sets:
                    h_out = h.affineMap(self.Woh, self.bo)
                    hidden_output_step.append(h_out)

                    if t < post_start_t:
                        new_branches.append((h, trace.copy())) # only save hidden state for next step before post_start_t
                    else:
                        h_pairs = propagate_hidden_through_post_layers(h, h_out)
                        for h_next, y in h_pairs:
                            if p_filter == 0.0 :
                                new_trace = trace.copy()
                                new_trace.append(y)
                                new_branches.append((h_next, new_trace))
                            if  p_filter > 0.0:
                                p_y = y.estimateProbability()
                                print(f"Step {t}: Branch {i} with output set probability {p_y:.12f} evaluated against threshold {p_filter}")
                                if  p_y <= p_filter:
                                    p_ignored += p_y
                                    print(f"Step {t}: Branch {i} with output set probability {p_y:.12f} ignored (threshold {p_filter})")
                                    continue
                                else:
                                    print(f"Step {t}: Branch {i} with output set probability {p_y:.12f} kept (threshold {p_filter})")
                                    new_trace = trace.copy()
                                    new_trace.append(y)
                                    new_branches.append((h_next, new_trace))

            hidden_states_all_steps.append(hidden_sets_step)
            hidden_output_all_steps.append(hidden_output_step)

            branches = new_branches
            if show:
                print(f"number of output sets in step {t} for hidden states(after relu):{len(hidden_sets_step)}")
                print(f"number of branches after step {t}: {len(branches)}")

        branch_signals = []
        for _, sig in branches:
            branch_signals.append(sig)

        return branch_signals, hidden_output_all_steps, p_ignored



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
