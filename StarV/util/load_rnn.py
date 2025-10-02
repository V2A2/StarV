from scipy.io import loadmat
import os
import numpy as np
from StarV.set.star import Star
from StarV.set.probstar import ProbStar



def load_simple_rnn(dtype=float):
    """Load RNN model"""

    cur_path = os.path.dirname(os.path.abspath(__file__))
    mat_contents = loadmat( cur_path + "/data/simple_rnn/simple_rnn.mat")
    Whx = np.asarray(mat_contents["kernel"], dtype)              # (H x I)
    Whh = np.asarray(mat_contents["recurrent_kernel"], dtype)    # (H x H)
    bh = np.asarray(mat_contents["bias"], dtype).reshape(-1) # (H,)

    H,I = Whx.shape
    Woh = np.eye(2, H)                    
    bo  = np.zeros(2,)

    """Load input data points"""
    data_contents = loadmat( cur_path + '/data/simple_rnn/points.mat')
    data_points = np.asarray(data_contents["pickle_data"], dtype)     

    W_contents = loadmat( cur_path + "/data/simple_rnn/dense.mat")
    W_ff = np.asarray(W_contents["W"], dtype=object)
    b_ff = np.asarray(W_contents["b"], dtype=object)
    # print("W_ff:",W_ff[0])
    # print("b_ff:",b_ff[0])

    return  Whx,Whh,bh,Woh,bo,data_points, W_ff,b_ff

def get_Star_set(col_point, eps,Ti):

    input_points = []  
    col_points = []
    for _ in range(Ti) :        
        col_points.append(col_point) # repeating Ti times
    input_points = np.hstack(col_points) 
    print("input_points-len:",len(input_points))
    print("input_points-shape:",input_points.shape)
    x = input_points
    n = x.shape[1]
    X = []
    for i in range (0,n):
        S = Star(x[:,i] - eps, x[:, i] + eps)
        S.C = np.zeros([1,S.nVars])  
        S.d = np.zeros([1])
        X.append(S)
   
    return X

def get_ProbStar_set(col_point, eps,Ti):

    input_points = []  
    col_points = []
    for _ in range(Ti) :        
        col_points.append(col_point) # repeating Ti times
    input_points = np.hstack(col_points) 
    print("input_points-len:",len(input_points))
    print("input_points-shape:",input_points.shape)
    x = input_points
    n = x.shape[1]
    X = []
    for i in range (0,n):
        S = Star(x[:,i] - eps, x[:, i] + eps)
        S.C = np.zeros([1,S.nVars])  
        S.d = np.zeros([1])
        mu = 0.5*(S.pred_lb + S.pred_ub) 
        a  = 3
        sig= (S.pred_ub-mu )/a
        epsilon = 1e-10
        sig = np.maximum(sig, epsilon)
        Sig = np.diag(np.square(sig))
        S_probstar = ProbStar(S.V, S.C, S.d,mu, Sig,S.pred_lb,S.pred_ub)
        X.append(S_probstar)
   
    return X
   
