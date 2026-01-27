from scipy.io import loadmat
import os
import numpy as np
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
import pandas as pd


def load_trained_CMAPSS_data():
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
        # print("test_data_info:",test_processed.describe())

        train_samples = train_processed.head(10)
        test_samples = test_processed.head(10)
        # print("train_data_samples:",train_samples)
        # print("test_data_samples:",test_samples)
        # pd.set_option('display.max_column', 30)
        # print("train_data_samples:",train_samples)


        # group by engine unit
        grouped_engine_data = test_processed.groupby("unit_number")
        print("grouped_engine_data.size:",grouped_engine_data.size())
        print("type of grouped_engine_data:",type(grouped_engine_data))
        print("grouped_engine_data groups:",grouped_engine_data.first())   


        return train_processed,test_processed,y_test

def load_trained_params():
        ''' Load Weights and Biases ''' 
        directory = os.path.dirname(os.path.abspath(__file__)) 
        params_path = directory + "/data/CMAPSS/saved_models/RNN_model_parameters_1.npz"
        params = np.load(params_path)
        W_hx = params["rnn.weight_ih_l0"]
        W_hh = params["rnn.weight_hh_l0"]
        b_hx = params["rnn.bias_ih_l0"]
        b_hh = params["rnn.bias_hh_l0"]
        W_oh = params["fc.0.weight"]
        b_oh = params["fc.0.bias"]
        fc_weights = []
        fc_biases = []

        # sort keys to keep layer order
        fc_weight_keys = sorted([k for k in params if k.startswith("fc.") and k.endswith(".weight")])
        fc_bias_keys   = sorted([k for k in params if k.startswith("fc.") and k.endswith(".bias")])
        fc_weight_keys = fc_weight_keys[1:]
        fc_bias_keys   = fc_bias_keys[1:]

        for w_key, b_key in zip(fc_weight_keys, fc_bias_keys):
            fc_weights.append(params[w_key])
            fc_biases.append(params[b_key])
        
        # print("fc_w:",fc_weights)
        # print("type_of_fc_w:",len(fc_weights))
        # print("fc_b:",fc_biases)

        # for key in params:
        #     print("Parameter name:", key, " shape:", params[key].shape)
        #     print("Parameter values:", params[key])

        return W_hx,b_hx,W_hh,b_hh,W_oh,b_oh,fc_weights,fc_biases



def load_simple_rnn(dtype=np.float64):
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

def get_ProbStar_set_RNN(input_data, noises,feature_idx):


    temperature_noise = noises[0]
    pressure_noise = noises[1]
    speed_noise = noises[2]

    temperature_sensor_indices = feature_idx[0]
    pressure_sensor_indices = feature_idx[1]
    speed_sensor_indices = feature_idx[2]

    transposed_input_data = input_data.T

    # returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    for i in range(transposed_input_data.shape[1]):
        single_data_point = transposed_input_data[:, i]
        single_data_points_bounds = []
        # print("single_data_point:",single_data_point)
        dims = single_data_point.shape[0]
        for dim in range(dims):
            if dim in temperature_sensor_indices:
                # print("temp_dim:",dim)
                # print("tempreture noise:",temperature_noise)
                if temperature_noise <0:
                    temperature_noise = -temperature_noise
                lb = single_data_point[dim] - temperature_noise
                ub = single_data_point[dim] + temperature_noise
            elif dim in pressure_sensor_indices:
                # print("pressure_dim:",dim)
                # print("pressure noise:",pressure_noise)
                if pressure_noise <0:
                    pressure_noise = -pressure_noise
                lb = single_data_point[dim] - pressure_noise
                ub = single_data_point[dim] + pressure_noise   
            elif dim in speed_sensor_indices:
                if speed_noise <0:
                    speed_noise = -speed_noise
                lb = single_data_point[dim] - speed_noise
                ub = single_data_point[dim] + speed_noise
            elif dim in range(dims): # add some small noise o oather sensor
                lb = single_data_point[dim] 
                ub = single_data_point[dim] 
            else:  
                raise ValueError("Dimension index out of range")
            single_data_points_bounds.append((lb, ub))
        # print("single_data_points_bounds:",single_data_points_bounds)
        # print("shape of single_data_points_bounds:",len(single_data_points_bounds))
        init_state_bounds_list.append(single_data_points_bounds)
    # print("init_state_bounds_list:",init_state_bounds_list)
    # print("shape of init_state_bounds_list:",len(init_state_bounds_list))

    # create Star for initial state 
    X = []
    np.set_printoptions(precision=12, suppress=False)

    for bounds in init_state_bounds_list:
        init_state_lb = np.array([b[0] for b in bounds])
        # print("init_state_lb:",init_state_lb)
        init_state_ub = np.array([b[1] for b in bounds])

        # mu = 0.5*(init_state_lb+init_state_ub)
        # a = 3
        # sig = (mu-init_state_lb)/a
        # print("sig====:",sig)
        # epsilon = 1e-6
        # sig = np.maximum(sig, epsilon,).astype(np.float64)
        # print("sig_max:",sig)
        # Sig = np.diag(np.square(sig)).astype(np.float64)
        # print("Sig==============:",Sig)
        # X0_probstar = ProbStar(mu, Sig,init_state_lb,init_state_ub)
        # # X0_probstar.C = np.zeros([1,X0_probstar.nVars])  
        # # X0_probstar.d = np.zeros([1])
        # print("initial probstar set:",X0_probstar)
        # X.append(X0_probstar)
        # # print("Input ProbStar set constructed:",X0_probstar)
        # print("probability of the initial ProbStar set:",X0_probstar.estimateProbability())
        
        
        X0 = Star(init_state_lb,init_state_ub)
        # X0.C = np.empty([1,X0.nVars])  
        # X0.d = np.empty([1])
        print("X0:",X0)
        mu = 0.5*(X0.pred_lb + X0.pred_ub) 
        a  = 3
        sig= (mu - X0.pred_lb)/a
        epsilon = 1e-6
        sig = np.maximum(sig, epsilon)
        Sig = np.diag(np.square(sig))
        X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu, Sig,X0.pred_lb,X0.pred_ub)
        print("initial probstar set:",X0_probstar)
        X.append(X0_probstar)
        # print("Input ProbStar set constructed:",X0_probstar)
        print("probability of the initial ProbStar set:",X0_probstar.estimateProbability())

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
    load_trained_CMAPSS_data()
    load_trained_params()
   
