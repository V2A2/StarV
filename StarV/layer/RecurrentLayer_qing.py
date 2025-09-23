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
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.net.network import NeuralNetwork



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
        print('')
        return '\n'

    def info(self):
        print(self)


    def reachExact(self,In, method = "exact", lp_solver='gurobi', pool=None, RF=0.0, DR=0):
        print("~~~~~~~~~~~~~~` Using {} method for reachability ~~~~~~~".format(method))
                
        Weight_In = [] # store a list of mapped input list(contain only one set), T sets
        H = []
        O = []
        for i in range(0,len(In)):
            print("--------------------------------Computing weighted input --------------------------")
            WIn=[]
            MIn = In[i].affineMap(self.Whx,self.bhx)
            WIn.append(MIn)
            print("======= WIn type:==========",type(WIn))
            Weight_In.append(WIn)
            print("number of AffineMap input sets:",len(Weight_In))
            print("======= Weight_In type:==========",type(Weight_In))
            # print("======= WIn[i]:==========",Weight_In[i][0])
        for i in range(0,len(In)):
            #H_hat[i]= In[i].affineMap(self.Whx,self.bh)
            if i == 0 :
                print("\n--------------------------------If i == 0 --------------------------")
                H0 =[]
                H0_out = ReLULayer.reach(Weight_In[i], method = method)
                print("======= H0_out type:==========",type(H0_out))
                print("number of ReLU output sets:",len(H0_out))
                print("H0_out[0].V:",H0_out[i].V)
                H0.extend(H0_out)
                H.append(H0)
                print("======= len(H) if i == 0 : ===========:",len(H))
                # m0 = len(H0[i])
                # print("======= m0 ===========:",m0)
                # O1 = []
                for h in H0:
                    print("========== h in H0=====:",h)
                    HO_out = h.affineMap(self.Woh,self.bo)
                    print("V_shape, if i == 0 :",HO_out.V.shape)
                    O.append(HO_out)
                    print("len(O1):",len(O))
                # O.append(O1)
                # print("======= len(Ooutput) if i == 0 : ===========:",len(O))
            else: # i > 1
                print("\n--------------------------------If i > 1 --------------------------")
                pre_H = H[i-1]
                # print("H2:",H2)
                H3 = []
                for h in pre_H:
                    H4= []
                    # print("========== H2_1.C:",H2[0].C)
                    H2_weight = h.affineMap(self.Whh)
                    # print("========== H2_[j].C:",H2[j].C)
                    print("H2_weight.C:",H2_weight.C)
                    H2_sum = H2_weight.minKowskiSum(Weight_In[i][0])
                    # print("========== H2_sum:",H2_sum)
                    H4.append(H2_sum)
                    print("========len(H4):{},type(H4):{}, if i > 1======".format(len(H4),type(H4)))
                    H2_out = ReLULayer.reach(H4, method = method)
                    print("H2_out_V_shape, if i > 1:",H2_out[0].V.shape)
                    H3.extend(H2_out)
                H.append(H3)
                print("========== len(H), if i >1 =======:",len(H))
                m2 = len(H3)
                print("======= m2 ===========",m2)
                # O2 = []
                for h in H3:
                    # print("========== H3[k]:",H3[k])
                    HO_out = h.affineMap(self.Woh,self.bo)
                    # print("========== HO_out:", HO_out)
                    O.append(HO_out)
                # O.append(O2)
                # print("====== len of output set, if i > 1:======", len(O))
                # print("====== output set:======", O[i])

        return O
    
    def reachApprox(self,In, method = "approx", lp_solver='gurobi', pool=None, RF=0.0, DR=0):
        print("~~~~~~~~~~~~~~` Using {} method for reachability ~~~~~~~".format(method))
              
        Weight_In = [] # store a list of mapped input list(contain only one set), T sets
        H = []
        O = []
        for i in range(0,len(In)):
            print("--------------------------------Computing weighted input --------------------------")
            MIn = In[i].affineMap(self.Whx,self.bhx)
            Weight_In.append(MIn)
        print("======= WIn type:==========",type(Weight_In))
        print("number of AffineMap input sets:",len(Weight_In))
        for i in range(0,len(In)):
            if i == 0 :
                print("\n--------------------------------If i == 0 --------------------------")
                print("Weight_In[0]",Weight_In[0])
                H0_out = ReLULayer.reach(Weight_In[i], method = method, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False)
                print("======= H0_out type:==========",type(H0_out))
                print("number of ReLU output sets in i == 0:",len(H0_out))
                print("H0_out[0].V:",H0_out.V)
                H.append(H0_out)
                print("======= len(H) if i == 0 : ===========:",len(H))
                HO_out = H0_out.affineMap(self.Woh,self.bo)
                print("V_shape, if i == 0 :",HO_out.V.shape)
                O.append(HO_out)
                print("len(O1):",len(O))
            else: # i > 1
                print("\n--------------------------------If i > 1 --------------------------")
                pre_H = H[i-1]
                H2_weight = pre_H.affineMap(self.Whh)
                print("H2_weight.C:",H2_weight.C)
                H2_sum = H2_weight.minKowskiSum(Weight_In[i])
                H2_out = ReLULayer.reach(H2_sum, method = method)
                H.append(H2_out)
                HO_out = H2_out.affineMap(self.Woh,self.bo)
                O.append(HO_out)
                print("========== len(H), if i >1 =======:",len(H))
                print("====== len of output set, if i > 1:======", len(O))

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


def load_simple_rnn(dtype=float):
    """Load RNN model"""

    cur_path = os.path.dirname(os.path.abspath(__file__))
    # print("current_path:",cur_path)
    # example_path = cur_path + '/simple_rnn.mat' 
    mat_contents = loadmat( cur_path + "/simple_rnn_v7.mat")
    Whx = np.asarray(mat_contents["kernel"], dtype)              # (H x I)
    Whh = np.asarray(mat_contents["recurrent_kernel"], dtype)    # (H x H)
    bh = np.asarray(mat_contents["bias"], dtype).reshape(-1) # (H,)
    print("\n########### hiddem layer bias: ##############\n",bh)

    H,I = Whx.shape
    Woh = np.eye(2, H)   
    print("Woh:",Woh)
    print("Whx:",Whx) 
    print("Whh:",Whh)                   
    bo  = np.zeros(2,)
    print("bo:",bo)

    """Load input data points"""
    data_contents = loadmat( cur_path + '/points.mat')
    data_points = np.asarray(data_contents["pickle_data"], dtype)     

    W_contents = mat73.loadmat( cur_path + "/dense.mat")
    W_ff = np.asarray(W_contents["W"], dtype=object)
    b_ff = np.asarray(W_contents["b"], dtype=object)
    # print("W_ff:",W_ff[0])
    # print("b_ff:",b_ff[0])

    
    return  Whx,Whh,bh,Woh,bo,data_points, W_ff,b_ff

def get_reachable_set(input_points, eps):

    x = input_points
    # print("input_points:",input_points)
    # print("input_points_shape:",input_points.shape)
    n = x.shape[1]
    print("n_length:",n)
    X = []
    for i in range (0,n):
        X.append(Star(x[:,i] - eps, x[:, i] + eps))
        print("number of input star_set:",len(X))
        print("X[i].V:",X[i].V)
        print("X[i].V:",X[i].V.shape)


    return X

def test_simple_rnn():

    Whx,Whh,bh,Woh,bo,data_points, W_ff,b_ff =load_simple_rnn()

    # create NN
    L1 = RecurrentLayer(Whx,Whh,bh,Woh,bo)
    mat= []
    for i in range(len(W_ff)):
        W_b = [W_ff[i],b_ff[i]]
        print("w_B:",W_b)
        mat.append(W_b)
    print("len(mat):",len(mat))
    L2 = FullyConnectedLayer(mat[0])
    L3 = FullyConnectedLayer(mat[1])
    L4 = FullyConnectedLayer(mat[2])
    L5 = FullyConnectedLayer(mat[3])
    L6 = FullyConnectedLayer(mat[4])
    L7 = FullyConnectedLayer(mat[5])


    layers = [L1,L2,L3,L4,L5,L6,L7]
    net = NeuralNetwork(layers=layers)

    # create input reachable sets
    x_len = 5
    x = data_points[:x_len,:]
    x = x.T
    print("x_length:",len(x))
    print("x(i)_shape:",len(x[1]))
    # print("x.T:",x)
    eps = 0.01
    T = [5 ,10 ,15, 20]
    # N = len(T)
    results = []
    for k in range(0,x_len):
        print("!!!!!!!!!!!!!!!!! Compute the {}th input seq !!!!!!!!!!!!".format(k))
        # for _ in range (len(T)):
        xk = np.array(x[:, k]).reshape(-1,1)
        # print("xk:",xk) 
        # print("xk_type:",type(xk)) 
        print("xk_shape:",xk.shape) 
        input_points = []  
        col_points = []
        for _ in range(T[0]) :        
            col_points.append(xk) # repeating T times
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
            print("\n====== Process the {}th layer of NN ===========".format(j))
            layers[j].info()
            RS1 = net.layers[j].reach(RS, method = "approx", lp_solver='gurobi', pool=None, RF=0.0, DR=0)
            print("The {}th layer ouput set len:{}".format(j,len(RS1)))
            print("The {}th layer ouput set type:{}".format(j,type(RS1)))
            for i in range(5):
                print("The {}th layer {}th ouput set for {}th input sequences:{}".format(j,i,RS1[i],k))
            RS = RS1
            Layer_RS.append(RS1)
        result = RS1
        print("====== NN result len:=========",len(result))
        results.append(result)
    print("====== NN results len:=========",len(results))
    
    return results



if __name__ == '__main__':


    results = test_simple_rnn()

    # for i, Y in enumerate(result):
    #     print("====== number of output set of {}th input_seq:{}=====".format(i,len(Y)))
    #     print("length Y{} in result:{}".format(i,len(Y)))
    #     for j,y in enumerate(Y):
    #         print("y_type:",type(y))
    #         print("Y{}{}:{}".format(i,j,y[j]))
    

    # for i, Y in enumerate(results):
    #     print("====== number of output set of {}th input_seq:{}=====".format(i,results[i]))
    #     print("y_type:",type(Y))

    for i, result in enumerate(results):
        print("====== number of output set of {}th input_seq:{}=====".format(i,len(result)))
        print("length result{} in results:{}".format(i,len(result)))
        for j,r in enumerate(result):
            print("r_type:",type(r))
            print("\n\nr{}{}= {}".format(i,j,r))
    