"""
Recurrent Layer Class
Qing Liu, 07/15/2025
"""
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import os
import numpy as np
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.LeakyReLULayer import LeakyReLULayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.SatLinsLayer import SatLinsLayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer

# def as_list(x):
#     """flatten multiple list of output to a list of sets."""
#     return list(x) if isinstance(x, (list, tuple)) else [x]

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
    def __init__(self,Whx, Whh, bh, Woh, bo, In):
        assert isinstance(Whh, np.ndarray), " Weights mat for hidden states to hiedden states should be a 2d numpy array"
        assert isinstance(bh, np.ndarray), "Hidden layer bias vector should be a 1d numpy array"
        assert isinstance(Whx, np.ndarray), "weights_mat for input states should be a 2d numpy array"
        assert isinstance(Woh, np.ndarray), "Weight mat for hidden states to output states should be a 2d numpy array"
        assert isinstance(bo, np.ndarray), "Output later bias vector should be a 1d numpy array"
        assert isinstance(In, list), 'error: input sets should be a list'
        self.Whx = Whx
        self.Whh = Whh
        self.bhx = bh
        self.Woh = Woh
        self.bo = bo
        self.In = In

        self.nIn = len(In) # sequences of input sets at each time step
        print("number of input sets:",self.nIn)
        # self.nH = nh # number of memory units in hidden layer
        #self.T =T # maximum time

    def reach(self,In,method= 'exact', lp_solver='gurobi', pool=None, RF=0.0, DR=0):
                
        Weight_In = [] # store a list of mapped input list(contain only one set), T sets
        H = []
        O = []
        for i in range(0,len(In)):
            print("--------------------------------Computing weighted input --------------------------")
            WIn=[]
            MIn = self.In[i].affineMap(self.Whx,self.bhx)
            WIn.append(MIn)
            print("======= WIn type:==========",type(WIn))
            Weight_In.append(WIn)
            print("number of AffineMap input sets:",len(Weight_In))
            print("======= Weight_In type:==========",type(Weight_In))
            # print("======= WIn[i]:==========",Weight_In[i][0])
        for i in range(0,len(In)):
            #H_hat[i]= self.In[i].affineMap(self.Whx,self.bh)
            if i == 0 :
                print("\n--------------------------------If i == 1 --------------------------")
                H0 =[]
                H0_out = ReLULayer.reach(Weight_In[i], method = method)
                print("H0_out[0].V:",H0_out[i].V)
                H0.extend(H0_out)
                H.append(H0)
                print("======= len(H) if i == 0 : ===========:",len(H))
                # m0 = len(H0[i])
                # print("======= m0 ===========:",m0)
                O1 = []
                for h in H0:
                    print("========== h in H0=====:",h)
                    HO_out = h.affineMap(self.Woh,self.bo)
                    print("V_shape, if i == 1 :",HO_out.V.shape)
                    O1.append(HO_out)
                    print("len(O1):",len(O1))
                O.append(O1)
                print("======= len(Ooutput) if i == 1 : ===========:",len(O))
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
                O2 = []
                for h in H3:
                    # print("========== H3[k]:",H3[k])
                    HO_out = h.affineMap(self.Woh,self.bo)
                    # print("========== HO_out:", HO_out)
                    O2.append(HO_out)
                O.append(O2)
                print("====== len of output set, if i > 1:======", len(O))
                # print("====== output set:======", O[i])

        return O,H

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
    bo  = np.zeros(2,)

    """Create input data points"""
    data_contents = loadmat( cur_path + '/points.mat')
    data_points = np.asarray(data_contents["pickle_data"], dtype)     
    
    return  Whx,Whh,bh,Woh,bo,data_points

def RNN_get_reachable_set(input_points, eps):
    x = input_points
    # print("input_points:",input_points)
    # print("input_points_shape:",input_points.shape)
    n = x.shape[1]
    print("n_length:",n)
    X = []
    for i in range (0,n):
        X.append(Star(x[:,i] - eps, x[:, i] + eps))
        print("number of input star_set:",len(X))
        # print("X[i].V:",X[i].V)

    return X

def test_simple_rnn():
    Whx,Whh,bh,Woh,bo,data_points =load_simple_rnn()
    print("bh_shape:{},bo_shape:{}".format(bh.shape,bo.shape))
    x_len = 5
    x = data_points[:x_len,:]
    x= x.T
    print("x_length:",len(x))
    print("x(i)_shape:",len(x[1]))
    # print("x.T:",x)
    eps = 0.01
    T = [5 ,10 ,15, 20]
    # N = len(T)
    results = []
    for k in range(0,x_len):
        # for i in range (0,N):
            xk = np.array(x[:, k]).reshape(-1,1)
            # print("xk:",xk) 
            print("xk_type:",type(xk)) 
            print("xk_shape:",xk.shape) 
            input_points = []  
            col_points = []
            # for i in len(T):
            for _ in range(T[1]) :        
                col_points.append(xk) # repeating T times
            input_points = np.hstack(col_points) # (40, 5)
            print("input_points-type:",type(input_points))
            print("input_points-shape:",input_points.shape)
            X = RNN_get_reachable_set(input_points=input_points,eps=eps)
            RNN = RecurrentLayer(Whx, Whh, bh, Woh, bo, X)
            Y,H= RNN.reach(X)
            # print("+++ Y_type++++:",type(Y))
            print("+++ Outputset length:{}, Hidden layer set length:{} ++++:".format(len(Y),len(H)))
            results.append(Y)
            results.append(H)
            
    return result



if __name__ == '__main__':
    
    # Whx,Whh,bh,Woh,bo,data_points =load_simple_rnn()
    # print("Whx:{},Whh{}".format(Whx,Whh))
    H = []
    X = []
    x1 = [1,2]
    X.extend(x1)
    x2 = [1,2,3,4]
    X.extend(x2)
    H.append(X)
    print("X:",X)
    print("X1:",X[0])
    print("H:",H)

    result = test_simple_rnn()
    # # print("number of output:",NO)
    # for i, Oi in enumerate(result):
    #         print(f"input_seq = {i}: , |O[i]|={len(Oi):3d}")

    for i, Y in enumerate(result):
        print("====== number of output set of {}th input_seq:{}=====".format(i,len(Y)))
        print("length Y{} in result:{}".format(i,len(Y)))
        for j,y in enumerate(Y):
            print("y_type:",type(y))
            print("Y{}{}:{}".format(i,j,y[j]))
