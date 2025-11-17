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
    def __init__(self,Whx, Whh, bh, Woh, bo):
        assert isinstance(Whh, np.ndarray), " Weights mat for hidden states to hiedden states should be a 2d numpy array"
        assert isinstance(bh, np.ndarray), "Hidden layer bias vector should be a 1d numpy array"
        assert isinstance(Whx, np.ndarray), "Weights_mat for input states should be a 2d numpy array"
        assert isinstance(Woh, np.ndarray), "Weight mat for hidden states to output states should be a 2d numpy array"
        assert isinstance(bo, np.ndarray), "Output later bias vector should be a 1d numpy array"

        self.Whx = Whx
        self.Whh = Whh
        self.bhx = bh
        self.Woh = Woh
        self.bo = bo
        self.in_dim = Whx.shape[1] 
        self.out_dim = Woh.shape[0] 

    @classmethod
    def rand(cls,in_dim, out_dim):
        Whx = np.random.rand(out_dim, in_dim)
        Whh = np.random.rand(out_dim, out_dim)
        bh = np.random.rand(out_dim)
        Woh = np.random.rand(out_dim, out_dim)
        bo = np.random.rand(out_dim)
        return RecurrentLayer( Whx,Whh, bh, Woh, bo)
    
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

            if t == 0:
                # First timestep: h0 = ReLU(Whx * x + bhx)
                print(f"===========number of input sets in step {t}:{len(I)}==========")
                hidden_states = []
                if isinstance(I, list):
                    print(f"Input sets is a list ===========number of input sets in step {t}:{len(I)}==========")
                    for i in range(len(I)):
                        WIn = I[i].affineMap(self.Whx, self.bhx)
                        h_out  = ReLULayer.reach([WIn], method=method)
                        hidden_states.extend(h_out )
                        print(f"Input sets is a list Hidden states len in step {t}, input set {i}:{len(hidden_states)}")
                    print(f"Hidden states len in step {t}:{len(hidden_states)}")
                else:
                    WIn = I.affineMap(self.Whx, self.bhx)
                    print("WIn t =0 dim:",WIn.dim)
                    h_out  = ReLULayer.reach([WIn], method=method)
                    hidden_states.extend(h_out)
                    print(f"Hidden states len in step {t}:{len(hidden_states)}")

            else:
                # Subsequent timesteps: h_t = ReLU(Whx * x_t + bhx + Whh * h_{t-1})
                hidden_states = []
                prev_hidden = H[t - 1]
                WIn_list = []
                if isinstance(I, list):
                    print(f"===========number of input sets  from prev output for this layer in step {t}:{len(I)}==========")
                    for i in range(len(I)):
                        WIn = I[i].affineMap(self.Whx, self.bhx)
                        WIn_list.append(WIn)
                    print(f"===========number of WIn_list in step {t}:{len(WIn_list)}==========")
                    # summed = []
                    for h_prev in prev_hidden:
                        h_recurrent = h_prev.affineMap(self.Whh)
                        # summed= []
                        for WIn in WIn_list:
                            h_p= h_recurrent.minKowskiSum(WIn)
                            h_recurrent= h_p
                        h_out = ReLULayer.reach([h_recurrent], method=method)
                        print(f"Number of h_out sets after relu in step {t}:{len(h_out)}")
                        hidden_states.extend(h_out)
                    print(f"Number of hidden_states sets after minsum in step {t}:{len(hidden_states)}")

                else:
                    WIn = I.affineMap(self.Whx, self.bhx)
                    for h_prev in prev_hidden:
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
        assert isinstance(In[0],Star), 'error: input set is not a Star set'

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
                h_weight = h_prev.affineMap(self.Whh)
                h_sum = h_weight.minKowskiSum(WIn)
                hidden_states = ReLULayer.reach(h_sum, method=method)

            # Save hidden state
            H.append(hidden_states)

            # Compute output: y_t = Woh * h_t + bo
            o_t = hidden_states.affineMap(self.Woh, self.bo)
            O.append(o_t)

        print("\n===== Approximate reachability analysis with reachApprox complete =====")
        print(f"Total timesteps: {len(O)}")
        return O


    def reach(self,In, method = None, lp_solver='gurobi', pool=None, RF=0.0, DR=0):
        if method is None:
            method = "exact"
        if method == "exact":
            S = self.reachExact(In, method, lp_solver, pool, RF, DR)
            # print("(Rcuurent layer)Final output set len:",len(S))
            return S
        elif method == "approx":
            return self.reachApprox(In, method, lp_solver, pool, RF, DR)
        else:
            raise Exception(f"error: unknown reachability method: {method}")
