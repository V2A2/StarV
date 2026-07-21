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
        # assert isinstance(In, list), 'Error: input sets should be a list'
        # assert fh is not None, " Hidden layer activation fucntion should not be None"
        # assert method is not None, " Rechability method should not be None"
        self.Whx = Whx
        self.Whh = Whh
        self.bhx = bh
        self.Woh = Woh
        self.bo = bo
        # self.fo = fo
        # self.In = In
        # self.method = method

        # if fh == "relu":
        #     self.fh = ReLULayer()

        self.in_dim = Whx.shape[1] 
        print("number of input sets:",self.in_dim)
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
        print('Input dimension: {}'.format(self.in_dim))
        print('Output dimension: {}'.format(self.out_dim))
   
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

            oi = []
            print(f"number of output sets in step {t} for hidden states:{len(hidden_states)}")
            for h in hidden_states:
                 outputs_t = h.affineMap(self.Woh, self.bo) 
                 oi.append(outputs_t)
            O.append(oi)

        print("\n===== Reachability analysis using exactReach complete =====")
        print(f"Total timesteps: {len(O)}")
        return O



    # def reachExact(self, In, method="exact", lp_solver="gurobi", pool=None, RF=0.0, DR=0):
    #     """
    #     Perform exact reachability analysis of an RNN with ReLU activation.

    #     Args:
    #         In (list): List of input sets (one per timestep).
    #         method (str): Reachability method, default "exact".
    #         lp_solver (str): Linear programming solver, default 'gurobi'.
    #         pool: Optional multiprocessing pool.
    #         RF (float): Reserved for future use (regularization factor).
    #         DR (int): Reserved for future use (dropout rate).

    #     Returns:
    #         list: List of reachable output sets at each timestep.
    #     """
         
    #     assert isinstance(In,list), 'error: input must be a list'

    #     print(f"\n~~~~~~~~ Using {method} method for reachability ~~~~~~~~")

    #     H = []  # Hidden state reachable sets per timestep
    #     O = []  # Output reachable sets per timestep

    #     for t, I in enumerate(In):
    #         print(f"\n----- Processing timestep {t} -----")
    #         hidden_states = []
    #         if t == 0:
    #             # First timestep: h0 = ReLU(Whx * x + bhx)
    #             # Combine_In = []
    #             for i in range(len(I)):
    #                 if len(I) ==1 :
    #                     WIn = I.affineMap(self.Whx, self.bhx)
    #                     hidden_states = ReLULayer.reach([WIn], method=method)
    #                 else:
    #                     print("=====input set at t0 from previous layer is a list======")
    #                     WIn_i = I[i].affineMap(self.Whx, self.bhx)
    #                     hidden_i = ReLULayer.reach([WIn_i], method=method)
    #                     hidden_states.extend(hidden_i)
    #             # print("number of hidden state after relu in i ==0:",len(h_out))
    #             # hidden_states.extend(h_out)
    #             print("number of hidden state after relu in i ==0:",len(hidden_states))
    #             # print("type:",type(hidden_states))

    #         else:
    #             # Subsequent timesteps: h_t = ReLU(Whx * x_t + bhx + Whh * h_{t-1})
    #             prev_hidden = H[t - 1]
    #             for i in range(len(I)):
    #                 if len(I) ==1 :
    #                     WIn = I.affineMap(self.Whx, self.bhx)
    #                     for h_prev in prev_hidden:
    #                         h_recurrent = h_prev.affineMap(self.Whh)
    #                         summed = h_recurrent.minKowskiSum(WIn)
    #                         # Apply ReLU
    #                         h_out = ReLULayer.reach([summed], method=method)
    #                         hidden_states.extend(h_out)
    #                 else:
    #                     WIn_i = I[i].affineMap(self.Whx, self.bhx)
    #                     for j in range(len(prev_hidden)):
    #                         print("previous time step have multi output hidden sets")
    #                         h_recurrent_j = h_prev[j].affineMap(self.Whh)
    #                         summed_j = h_recurrent_j.minKowskiSum(WIn)
    #                         # Apply ReLU
    #                         h_out_j = ReLULayer.reach([summed_j], method=method)
    #                         hidden_states.extend(h_out_j)

    #         # Save hidden states
    #         H.append(hidden_states)

    #         # Compute outputs: y_t = Woh * h_t + bo
    #         oi = []
    #         print(f"number of output sets in step {t} for hidden states:{len(hidden_states)}")
    #         for h in hidden_states:
    #             outputs_t = h.affineMap(self.Woh, self.bo) 
    #             oi.append(outputs_t)
    #         print(f"oi_len at step {t}:{len(oi)}")
    #         O.append(oi)

    #         print(f"  Hidden states: {len(hidden_states)} sets")
    #         print(f"  Outputs: {len(outputs_t)} sets")

    #     print("\n===== Reachability analysis using exactReach complete =====")
    #     print(f"Total timesteps: {len(O)}")
    #     return O


    # def reachExact(self,In, method = "exact", lp_solver='gurobi', pool=None, RF=0.0, DR=0):
    #     print("~~~~~~~~~~~~~~` Using {} method for reachability ~~~~~~~".format(method))
                
    #     # Weight_In = [] # store a list of mapped input list(contain only one set), T sets
    #     H = []
    #     O = []
    #     # for i in range(0,len(In)):
    #     #     print("--------------------------------Computing weighted input --------------------------")
    #     #     WIn=[]
    #     #     MIn = In[i].affineMap(self.Whx,self.bhx)
    #     #     WIn.append(MIn)
    #     #     # print("======= WIn type:==========",type(WIn))
    #     #     # Weight_In.append(WIn)
    #     #     # print("number of AffineMap input sets:",len(Weight_In))
    #     #     # print("======= Weight_In type:==========",type(Weight_In))
    #     #     # print("======= WIn[i]:==========",Weight_In[i][0])
    #     for i in range(0,len(In)):
    #         #H_hat[i]= In[i].affineMap(self.Whx,self.bh)
    #         if i == 0 :
    #             print("\n--------------------------------If i == 0 --------------------------")
    #             H0 =[]
    #             Weighted_In = In[i].affineMap(self.Whx,self.bhx)
    #             # print("Weight_In[0]_dim",Weight_In[0][0].dim)
    #             H0_out = ReLULayer.reach([Weighted_In], method = method)
    #             # print("======= H0_out type:==========",type(H0_out))
    #             # print("number of ReLU output sets:",len(H0_out))
    #             # print("H0_out[0].V:",H0_out[i].V)
    #             H0.extend(H0_out)
    #             H.append(H0)
    #             # print("======= len(H) if i == 0 : ===========:",len(H))
    #             # m0 = len(H0[i])
    #             # print("======= m0 ===========:",m0)
    #             O1 = []
    #             for h in H0:
    #                 # print("========== h in H0=====:",h)
    #                 HO_out = h.affineMap(self.Woh,self.bo)
    #                 # print("V_shape, if i == 0 :",HO_out.V.shape)
    #                 # O.append(HO_out)
    #                 O1.append(HO_out)
    #                 # print("len(O1):",len(O))
    #             O.append(O1)
    #             # print("======= len(Ooutput) if i == 0 : ===========:",len(O))
    #         else: # i > 1
    #             print("\n--------------------------------If i > 1 --------------------------")
    #             pre_H = H[i-1]
    #             # print("H2:",H2)
    #             H3 = []
    #             for h in pre_H:
    #                 # H4= []
    #                 # print("========== H2_1.C:",H2[0].C)
    #                 H2_weight = h.affineMap(self.Whh)
    #                 # print("========== H2_[j].C:",H2[j].C)
    #                 # print("H2_weight.C:",H2_weight.C)
    #                 W_In = In[i].affineMap(self.Whx,self.bhx)
    #                 H2_sum = H2_weight.minKowskiSum(W_In)
    #                 # print("========== H2_sum:",H2_sum)
    #                 # H4.append(H2_sum)e
    #                 # print("========len(H4):{},type(H4):{}, if i > 1======".format(len(H4),type(H4)))
    #                 H2_out = ReLULayer.reach([H2_sum], method = method)
    #                 # print("H2_out_V_shape, if i > 1:",H2_out[0].V.shape)
    #                 H3.extend(H2_out)
    #             H.append(H3)
    #             # print("========== len(H), if i >1 =======:",len(H))
    #             # print("======= m2 ===========",m2)
    #             O2 = []
    #             for h in H3:
    #                 # print("========== H3[k]:",H3[k])
    #                 HO_out = h.affineMap(self.Woh,self.bo)
    #                 # print("========== HO_out:", HO_out)
    #                 # O.append(HO_out)
    #                 O2.append(HO_out)
    #             O.append(O2)
    #             print("====== len of output set, if i > 1:======", len(O))
    #             # print("====== output set:======", O[i])
    #     print("output sets type:",type(O))
    #     print("output sets type O[i]:",type(O[0]))

    #     return O
    
    # def reachApprox(self,In, method = "approx", lp_solver='gurobi', pool=None, RF=0.0, DR=0):
    #     print("~~~~~~~~~~~~~~` Using {} method for reachability ~~~~~~~".format(method))
              
    #     Weight_In = [] # store a list of mapped input list(contain only one set), T sets
    #     H = []
    #     O = []
    #     for i in range(0,len(In)):
    #         print("--------------------------------Computing weighted input --------------------------")
    #         MIn = In[i].affineMap(self.Whx,self.bhx)
    #         Weight_In.append(MIn)
    #         print("Weight_In[i].nVar:",Weight_In[i].nVars)
    #     # print("======= WIn type:==========",type(Weight_In))
    #     # print("number of AffineMap input sets:",len(Weight_In))
    #     for i in range(0,len(In)):
    #         if i == 0 :
    #             print("\n--------------------------------If i == 0 --------------------------")
    #             print('Weight_In[0]_V_shape',Weight_In[0].V.shape)
    #             H0_out = ReLULayer.reach(Weight_In[i], method = method, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False)
    #             # print("======= H0_out type:==========",type(H0_out))
    #             # print("number of ReLU output sets in i == 0:",len(H0_out))
    #             # print("H0_out[0].V:",H0_out.V)
    #             # print("H0_out[0].V_shape:",H0_out.V.shape)
    #             H.append(H0_out)
    #             # print("======= len(H) if i == 0 : ===========:",len(H))
    #             HO_out = H0_out.affineMap(self.Woh,self.bo)
    #             # print("HO_out_V_shape, if i == 0 :",HO_out.V.shape)
    #             # print("HO_out_V:",HO_out.V)
    #             O.append(HO_out)
    #             # print("len(O1):",len(O))
    #         else: # i > 1
    #             print("\n--------------------------------If i > 1 --------------------------")
    #             pre_H = H[i-1]
    #             # print('pre_H_V_shape:',pre_H.V.shape)
    #             # print("pre_H_V:",pre_H.V)
    #             # print("pre_H.C:",pre_H.C)
    #             H2_weight = pre_H.affineMap(self.Whh)
    #             # print("H2_weight.V:",H2_weight.V)
    #             # print("H2_weight_V_shape:",H2_weight.V.shape)
    #             # print("H2_weight_C:",H2_weight.C)
    #             # print("H2_weight_pred_lb:",H2_weight.pred_lb)
    #             # print("H2_weight_pred_lb_shape:",H2_weight.pred_lb.shape[0])
    #             # print("H2_weight_nVar:",H2_weight.nVars)
    #             # print("Weight_In[i].nVar:",Weight_In[i].nVars)
    #             # print("Weight_In[i]_V_shape:",Weight_In[i].V.shape)
    #             # print("Weight_In[i]_C:",Weight_In[i].C)
    #             # print("Weight_In[i]_pred_lb_shape:",Weight_In[i].pred_lb.shape[0])
    #             H2_sum = H2_weight.minKowskiSum(Weight_In[i])
    #             H2_out = ReLULayer.reach(H2_sum, method = method)
    #             H.append(H2_out)
    #             HO_out = H2_out.affineMap(self.Woh,self.bo)
    #             O.append(HO_out)
    #             # print("========== len(H), if i >1 =======:",len(H))
    #             print("====== len of output set, if i > 1:======", len(O))

    #     return O

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
                # First timestep: h0 = ReLU(Whx * x0 + bhx)
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
            method = "approx"
        if method == "exact":
            return self.reachExact(In, method, lp_solver, pool, RF, DR)
        elif method == "approx" or method == "relax":
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
        col_point = np.array(x[:, k]).reshape(-1,1)
        print("col_point:",col_point) 
        print("col_point_type:",type(col_point)) 
        print("col_point_shape:",col_point.shape) 
        
        # input_points = []  
        # col_points = []
        # for _ in range(T[0]) :        
        #     col_points.append(xk) # repeating T times
        # # print("col_points:",col_points)
        # input_points = np.hstack(col_points) # (40, T)
        # print("input_points-len:",len(input_points))
        # print("input_points-shape:",input_points.shape)
        S = get_ProbStar_set(col_point,eps,T[0])
        print("@@@@@@ input Star set: @@@@@@",S[0])
        Numlayers = len(layers)
        print("number of layers:",Numlayers)
        Layer_RS = []
        RS = S
        print("@@@@@@ number of input Star set: @@@@@@",len(RS))
        for j in range(0,Numlayers):
            print("\n====== Process the {}th layer of NN ===========".format(j+1))
            layers[j].info()
            RS1 = net.layers[j].reach(RS, method = "exact", lp_solver='gurobi', pool=None, RF=0.0, DR=0)
            print("The {}th layer ouput set len RS1:{}".format(j+1,len(RS1)))
            print("The {}th layer ouput set type RS1:{}".format(j+1,type(RS1)))
            print("The {}th layer ouput set type RS1[i]:{}".format(j+1,type(RS1[0])))
            
            print("The {}th layer ouput set RS1[0][0]:{}".format(j+1,RS1[0][0].V))
            for i in range(len(RS1)):  
                l = len(RS1[i])
                for m in range(l) :             
                    print("The {}th layer of {}{}th ouput set for {}th input sequences:{}, prob= {}".format(j+1,i+1,m+1,k+1,RS1[i][m],RS1[i][m].estimateProbability()))
            # for i in range(len(RS1)):           
            #         print("The {}th layer of {}th ouput set for {}th input sequences:{}".format(j+1,i+1,k+1,RS1[i]))
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
    