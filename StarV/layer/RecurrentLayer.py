"""
Recurrent Layer Class
Qing Liu, 07/15/2025
"""
import multiprocessing
import numpy as np
from scipy.optimize import linprog
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer


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

    def _hidden_affine_map(self, h):
        if self.bhh is None:
            return h.affineMap(self.Whh)
        return h.affineMap(self.Whh, self.bhh)

    def reachExact(self, In, lp_solver="gurobi", pool=None, show=False):
        """Exact RNN reachability for Star or ProbStar input sequences."""
        assert isinstance(In, list), 'error: input must be a list'
        assert all(isinstance(s, (Star, ProbStar)) for s in In), \
            'error: exact RNN reachability supports Star or ProbStar inputs'

        hidden_sets_by_time = []
        outputs_by_time = []

        for t, input_set in enumerate(In):
            WIn = input_set.affineMap(self.Whx, self.bhx)
            if t == 0:
                hidden_sets = ReLULayer.reach(
                    [WIn], method='exact', lp_solver=lp_solver,
                    pool=pool, show=False
                )
            else:
                hidden_sets = []
                for h_prev in hidden_sets_by_time[t - 1]:
                    summed = self._hidden_affine_map(h_prev).minKowskiSum(WIn)
                    hidden_sets.extend(ReLULayer.reach(
                        [summed], method='exact', lp_solver=lp_solver,
                        pool=pool, show=False
                    ))

            hidden_sets_by_time.append(hidden_sets)
            outputs_by_time.append([h.affineMap(self.Woh, self.bo) for h in hidden_sets])

            if show:
                print('RNN exact step {}: {} output sets'.format(t, len(hidden_sets)))

        return outputs_by_time

    def reachExactBranches(self, In, post_layers=None, lp_solver="gurobi", pool=None,
                           p_filter=None, show=False, post_start_t=None,
                           numCores=1):
        """Branch-based exact RNN reachability.
        """
        if numCores is None:
            numCores = 1
        assert isinstance(numCores, int), 'error: numCores should be an int'
        assert numCores >= 1, 'error: numCores should be >= 1'

        if pool is not None or numCores == 1:
            return self.reachExactBranchesWithPool(
                In, post_layers=post_layers, lp_solver=lp_solver,
                pool=pool, p_filter=p_filter, show=show,
                post_start_t=post_start_t
            )

        if show:
            print(f"Using {numCores} cores for RNN ReLU reachability")
        with multiprocessing.Pool(numCores) as reach_pool:
            return self.reachExactBranchesWithPool(
                In, post_layers=post_layers, lp_solver=lp_solver,
                pool=reach_pool, p_filter=p_filter, show=show,
                post_start_t=post_start_t
            )

    def reachExactBranchesWithPool(self, In, post_layers=None, lp_solver="gurobi",
                                    pool=None, p_filter=None, show=False,
                                    post_start_t=None):
        """  Branch-based reachability for RNNs

        Args:
            pool:
                Optional multiprocessing pool used by ReLU exact reachability.
            post_layers:
                List of post layers to apply to each RNN output after post_start_t. Supported layer types: FullyConnectedLayer, ReLULayer. Default: None (no post layers).
            p_filter:
                Branch pruning threshold at/after post_start_t based on output
                set probability. If p_filter = 0, no pruning; if > 0, prune branches with estimated output probability < p_filter.
            post_start_t:
                Start timestep to record outputs into branch traces.
                Default: len(In)//2.
        """
        # Qing Liu, 02/15/2026
        # Update: post-reachability branch filtering with time tags, 04/03/2026

        assert isinstance(In, list), 'error: input must be a list'
        assert len(In) > 0, 'error: input is empty'
        is_star_input = all(isinstance(s, Star) for s in In)
        is_probstar_input = all(isinstance(s, ProbStar) for s in In)
        assert is_star_input or is_probstar_input, \
            'error: input must be a list of Stars or a list of ProbStars'

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
        if is_star_input and p_filter > 0.0:
            raise RuntimeError('Star branch reachability does not support p_filter')
        
        if show:
            if p_filter == 0.0:
                print("Using exact ReLU reachability with branch tracing and no pruning (p_filter=0.0)")
            else:
                print(f"Using exact ReLU reachability with branch tracing and pruning threshold p_filter={p_filter}")
        
        p_ignored = 0.0


        # Branch elements: (hidden_state_for_next_step, trace_after_post_start)
        branches = [(None, [])]
        hidden_states_all_steps = []
        hidden_output_all_steps = []

        def reach_relu_sets(input_sets):
            """Apply exact ReLU reachability and always return a list."""
            if not isinstance(input_sets, list):
                input_sets = [input_sets]

            return ReLULayer.reach(
                input_sets,
                method='exact',
                lp_solver=lp_solver,
                pool=pool,
                show=False,
            )

        def is_feasible_star(star_set):
            """Return False only when the predicate polytope is LP-infeasible."""
            if star_set.nVars == 0:
                return len(star_set.d) == 0 or np.all(star_set.d >= 0.0)

            constraints = star_set.C if len(star_set.C) > 0 else None
            bounds = list(zip(star_set.pred_lb, star_set.pred_ub))
            result = linprog(
                np.zeros(star_set.nVars),
                A_ub=constraints,
                b_ub=star_set.d if constraints is not None else None,
                bounds=bounds,
                method='highs',
            )
            if result.status == 0:
                return True
            if result.status == 2:
                return False
            raise RuntimeError(
                'RNN Star feasibility LP failed with status {}: {}'
                .format(result.status, result.message)
            )

        def lift_hidden_set_to_output_predicates(hidden_set, output_set):
            """Use output predicate constraints on hidden dynamics."""
            target_nvars = output_set.nVars
            if target_nvars < hidden_set.nVars:
                raise RuntimeError(
                    'post-layer output has fewer predicate variables than hidden state'
                )

            if target_nvars == hidden_set.nVars:
                lifted_V = hidden_set.V
            else:
                zero_generators = np.zeros(
                    (hidden_set.dim, target_nvars - hidden_set.nVars),
                    dtype=hidden_set.V.dtype
                )
                lifted_V = np.hstack((hidden_set.V, zero_generators))

            if isinstance(hidden_set, Star):
                return Star(
                    lifted_V,
                    output_set.C,
                    output_set.d,
                    output_set.pred_lb,
                    output_set.pred_ub
                )

            if isinstance(output_set, ProbStar):
                return ProbStar(
                    lifted_V,
                    output_set.C,
                    output_set.d,
                    output_set.mu,
                    output_set.Sig,
                    output_set.pred_lb,
                    output_set.pred_ub
                )

            return ProbStar(
                lifted_V,
                output_set.C,
                output_set.d,
                hidden_set.mu,
                hidden_set.Sig,
                output_set.pred_lb,
                output_set.pred_ub
            )

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
                    current = reach_relu_sets(current)
                else:
                    raise Exception(f"error: unsupported post layer type: {type(layer)}")

            h_pairs = []
            for net_out in current:
                # Keep predicate consistency across branch evolution by reusing
                # post-layer constraints on hidden state for next recurrent step.
                if len(net_out.C) == 0 and net_out.nVars == h.nVars:
                    h_next_set = h
                else:
                    h_next_set = lift_hidden_set_to_output_predicates(h, net_out)
                h_pairs.append((h_next_set, net_out))
            return h_pairs

        for t, I in enumerate(In):
            new_branches = []
            hidden_sets_step = []
            infeasible_pruned_step = 0
            hidden_output_step = []
            p_ignored_step = 0.0
            best_pruned_candidate = None  # (p_y, h_next, new_trace)

            WIn = I.affineMap(self.Whx, self.bhx)

            for i, (h_prev_post, trace) in enumerate(branches):
                if t == 0:
                    hidden_sets = reach_relu_sets([WIn])
                else:
                    if self.bhh is not None:
                        h_recurrent = h_prev_post.affineMap(self.Whh, self.bhh)
                    else:
                        h_recurrent = h_prev_post.affineMap(self.Whh)
                    summed = h_recurrent.minKowskiSum(WIn)
                    hidden_sets = reach_relu_sets([summed])

                hidden_sets_step.extend(hidden_sets)

                for h in hidden_sets:
                    if is_star_input and not is_feasible_star(h):
                        infeasible_pruned_step += 1
                        continue

                    h_out = h.affineMap(self.Woh, self.bo)
                    hidden_output_step.append(h_out)

                    if t < post_start_t:
                        new_branches.append((h, trace.copy())) # only save hidden state for next step before post_start_t
                    else:
                        h_pairs = propagate_hidden_through_post_layers(h, h_out)
                        for h_next, y in h_pairs:
                            if is_star_input and not is_feasible_star(y):
                                infeasible_pruned_step += 1
                                continue

                            if p_filter == 0.0 :
                                new_trace = trace.copy()
                                new_trace.append(y)
                                new_branches.append((h_next, new_trace))
                            if  p_filter > 0.0:
                                p_y = y.estimateProbability()
                                # print(f"Step {t}: Branch {i} with output set probability {p_y:.12f} evaluated against threshold {p_filter}")
                                new_trace = trace.copy()
                                new_trace.append(y)
                                if  p_y <= p_filter:
                                    p_ignored_step += p_y
                                    if best_pruned_candidate is None or p_y > best_pruned_candidate[0]:
                                        best_pruned_candidate = (p_y, h_next, new_trace)
                                    # print(f"Step {t}: Branch {i} with output set probability {p_y:.12f} ignored (threshold {p_filter})")
                                    continue
                                else:
                                    # print(f"Step {t}: Branch {i} with output set probability {p_y:.12f} kept (threshold {p_filter})")
                                    new_branches.append((h_next, new_trace))

            hidden_states_all_steps.append(hidden_sets_step)
            hidden_output_all_steps.append(hidden_output_step)

            # if all candidates are pruned at this step, keep the best pruned one.
            if p_filter > 0.0 and t >= post_start_t and len(new_branches) == 0 and best_pruned_candidate is not None:
                p_best, h_best, trace_best = best_pruned_candidate
                new_branches.append((h_best, trace_best))
                p_ignored_step -= p_best
                if show:
                    print(
                        f"Step {t}: all branches pruned by p_filter={p_filter}; "
                        f"kept best pruned branch with p={p_best:.12g}."
                    )

            if is_star_input and len(new_branches) == 0:
                raise RuntimeError(
                    'all RNN Star branches are infeasible at step {}'.format(t)
                )

            p_ignored += p_ignored_step
            branches = new_branches
            if show:
                # print(f"number of output sets in step {t} for hidden states(after relu):{len(hidden_sets_step)}")
                print(f"number of branches after step {t}: {len(branches)}")
                if infeasible_pruned_step > 0:
                    print(
                        'pruned {} infeasible RNN Star branches at step {}'
                        .format(infeasible_pruned_step, t)
                    )

        branch_signals = []
        for _, sig in branches:
            branch_signals.append(sig)

        return branch_signals, hidden_output_all_steps, p_ignored



    def reach(self, In, method="exact", lp_solver='gurobi', pool=None,
              show=False):
        """Exact RNN reachability."""
        if method is None:
            method = "exact"
        if method == "exact":
            return self.reachExact(
                In, lp_solver=lp_solver, pool=pool, show=show
            )
        else:
            raise Exception("error: RecurrentLayer only supports exact reachability")
