from scipy.io import loadmat
import os
import numpy as np
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
import pandas as pd

def load_CMAPSS_data():
        ''' Load Data '''
        directory = os.path.dirname(os.path.abspath(__file__))
        print("current directory:",directory)
        data_path = directory + "/data/CMAPSS/CMAPSS_processed"
        print("current data path:",data_path)
        rul_path = directory + "/data/CMAPSS/CMAPSSData"

        # col_names = index_names + operational_names + sensor_names

        train_processed = pd.read_csv(data_path + '/train_FD001_processed_4f.csv',sep=',',header=0,index_col=False)
        test_processed = pd.read_csv(data_path + '/test_FD001_processed_4f.csv',sep=',',header=0,index_col=False)
        y_test = pd.read_csv(rul_path + '/RUL_FD001.txt',sep='\s+',header=None,index_col=False,names=['RUL'])

        print("all_train_data_shape:",train_processed.shape)
        print("all_test_data_shape:",test_processed.shape)
        print("all_test_URL_shape:",y_test.shape)
        print("test_data_info:",test_processed.describe())

        train_samples = train_processed.head(10)
        test_samples = test_processed.head(10)
        print("train_data_samples:",train_samples)
        print("test_data_samples:",test_samples)
        # pd.set_option('display.max_column', 30)
        # print("train_data_samples:",train_samples)


        # group by engine unit
        grouped_engine_data = test_processed.groupby("unit_number")
        print("grouped_engine_data.size:",grouped_engine_data.size())
        print("type of grouped_engine_data:",type(grouped_engine_data))
        print("grouped_engine_data groups:",grouped_engine_data.first())   


        ''' Load Weights and Biases '''  
        params_path = directory + "/data/CMAPSS/saved_models/RNN_model_parameters_1.npz"
        params = np.load(params_path)
        for key in params:
            print("Parameter name:", key, " shape:", params[key].shape)
            print("Parameter values:", params[key])
      

        return train_processed,test_processed,y_test


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

if __name__ == "__main__":
    load_CMAPSS_data()
   
