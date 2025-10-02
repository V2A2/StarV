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
from StarV.util.load_rnn import load_simple_rnn, get_reachable_set


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
            RF (float): Reserved for future use (regularization factor).
            DR (int): Reserved for future use (dropout rate).

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
                WIn = I.affineMap(self.Whx, self.bhx)
                hidden_states = ReLULayer.reach([WIn], method=method)

            else:
                # Subsequent timesteps: h_t = ReLU(Whx * x_t + bhx + Whh * h_{t-1})
                hidden_states = []
                prev_hidden = H[t - 1]
                WIn = I.affineMap(self.Whx, self.bhx)

                for h_prev in prev_hidden:
                    h_recurrent = h_prev.affineMap(self.Whh)
                    summed = h_recurrent.minKowskiSum(WIn)

                    # Apply ReLU
                    h_out = ReLULayer.reach([summed], method=method)
                    hidden_states.extend(h_out)

            # Save hidden states
            H.append(hidden_states)

            # Compute outputs: y_t = Woh * h_t + bo
            for h in hidden_states:
                outputs_t = h.affineMap(self.Woh, self.bo) 
                O.append([outputs_t])

            print(f"  Hidden states: {len(hidden_states)} sets")
            print(f"  Outputs: {len(outputs_t)} sets")

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

def test_simple_rnn():

    Whx,Whh,bh,Woh,bo,data_points, W_ff,b_ff =load_simple_rnn()

    # create NN
    L1 = RecurrentLayer(Whx,Whh,bh,Woh,bo)
    mat= []
    for i in range(len(W_ff[0])):
        W_b = [W_ff[0][i],b_ff[0][i].reshape(-1)]
        # print("w_B:",W_b)
        mat.append(W_b)
    print("len(mat):",len(mat))
    L2 = FullyConnectedLayer(mat[0],fo='relu')
    L3 = FullyConnectedLayer(mat[1],fo ='relu')
    L4 = FullyConnectedLayer(mat[2],fo='relu')
    L5 = FullyConnectedLayer(mat[3],fo='relu')
    L6 = FullyConnectedLayer(mat[4],fo='relu')
    L7 = FullyConnectedLayer(mat[5],fo='relu')


    layers = [L1,L2,L3,L4,L5,L6,L7]
    net = NeuralNetwork(layers=layers)

    # create input reachable sets
    x_len = 10
    print("data_points_shape:",data_points.shape)
    print("data_points:",data_points)
    x = data_points[:x_len,:]
    x = x.T
    print("x_length:",len(x))
    print("x(i)_shape:",len(x[1]))
    print("x.T:",x)
    eps = 0.01
    T = [5 ,10 ,15, 20]
    # N = len(T)
    results = []
    for k in range(0,x_len):
        print("\n\n\n\n!!!!!!!!!!!!!!!!! Compute the {}th input seq !!!!!!!!!!!!".format(k+1))
        # for _ in range (len(T)):
        xk = np.array(x[:, k]).reshape(-1,1)
        print("xk:",xk) 
        print("xk_type:",type(xk)) 
        print("xk_shape:",xk.shape) 
        input_points = []  
        col_points = []
        for _ in range(T[0]) :        
            col_points.append(xk) # repeating T times
        # print("col_points:",col_points)
        input_points = np.hstack(col_points) # (40, T)
        print("input_points-len:",len(input_points))
        print("input_points-shape:",input_points.shape)
        S = get_reachable_set(input_points=input_points,eps=eps)
        # print("@@@@@@ input Star set: @@@@@@",S)
        Numlayers = len(layers)
        print("number of layers:",Numlayers)
        Layer_RS = []
        RS = S
        print("@@@@@@ number of input Star set: @@@@@@",len(RS))
        for j in range(0,Numlayers):
            print("\n====== Process the {}th layer of NN ===========".format(j+1))
            layers[j].info()
            RS1 = net.layers[j].reach(RS, method = "approx", lp_solver='gurobi', pool=None, RF=0.0, DR=0)
            print("The {}th layer ouput set len RS1:{}".format(j+1,len(RS1)))
            print("The {}th layer ouput set type RS1:{}".format(j+1,type(RS1)))
            print("The {}th layer ouput set type RS1[i]:{}".format(j+1,type(RS1[0])))
            
            # print("The {}th layer ouput set RS1[0][0]:{}".format(j+1,RS1[0][0].V))
            # for i in range(len(RS1)):  
            #     l = len(RS1[i])
            #     for m in range(l) :             
            #         print("The {}th layer of {}{}th ouput set for {}th input sequences:{}".format(j+1,i+1,m+1,k+1,RS1[i][m]))
            for i in range(len(RS1)):           
                    print("The {}th layer of {}th ouput set for {}th input sequences:{}".format(j+1,i+1,k+1,RS1[i]))
            RS = RS1
            Layer_RS.append(RS1)
        result = RS1
        print("====== NN result len:=========",len(result))
        results.append(result)
    print("====== NN results len:=========",len(results))
    
    return results



if __name__ == '__main__':


    results = test_simple_rnn()


    for i, result in enumerate(results):
        print("====== number of output set of {}th input_seq:{}=====".format(i,len(result)))
        print("length result{} in results:{}".format(i,len(result)))
        for j,r in enumerate(result):
            print("r_type:",type(r))
            print("\n\nr{}{}= {}".format(i,j,r))
    