from scipy.io import loadmat
import os
import numpy as np
from scipy.linalg import block_diag
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
import pandas as pd
from StarV.util.plot import plot_probstar,plot_2D_Star


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
        grouped_engine_data = train_processed.groupby("unit_number")
        print("grouped_engine_data.size:",grouped_engine_data.size())
        print("type of all grouped_engine_data :",type(grouped_engine_data))
        print("grouped_engine_data first cycle in each groups:",grouped_engine_data.first())   


        return train_processed,test_processed,y_test

def load_trained_params():
        ''' Load Weights and Biases ''' 
        directory = os.path.dirname(os.path.abspath(__file__)) 
        params_path = directory + "/data/CMAPSS/saved_models/RNN_model_parameters_1_29_win25_h32_f64.npz"
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


    temperature_noise= noises[0]
    pressure_noise = noises[1]
    speed_noise = noises[2]

    print(f"all added noises pct:{noises}")

    temperature_sensor_indices = feature_idx[0]
    pressure_sensor_indices = feature_idx[1]
    speed_sensor_indices = feature_idx[2]

    # print(f"input data shape:{input_data.shape},\n input data head 20:{input_data}")

    # transposed_input_data = input_data.T

    # print(f"transposed input data shape:{transposed_input_data.shape}")

    # returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    for i in range(input_data.shape[0]):
        single_data_point = input_data[i, :]
        single_data_points_bounds = []
        # print("single_data_point:",single_data_point)
        dims = single_data_point.shape[0]
        for dim in range(dims):
            if dim in temperature_sensor_indices:
                # print("temp_dim:",dim)
                # print("tempreture noise:",temperature_noise)
                # if temperature_noise <0:
                #     temperature_noise = -temperature_noise
                sig = temperature_noise * np.abs(single_data_point[dim])
                sig = np.maximum(sig, 1e-6)
                delta = 3 * sig
                lb = single_data_point[dim] - delta
                ub = single_data_point[dim] + delta
            elif dim in pressure_sensor_indices:
                # print("pressure_dim:",dim)
                # print("pressure noise:",pressure_noise)
                # if pressure_noise <0:
                #     pressure_noise = -pressure_noise
                sig = pressure_noise * np.abs(single_data_point[dim])
                sig = np.maximum(sig, 1e-6)
                delta = 3 * sig
                lb = single_data_point[dim] - delta
                ub = single_data_point[dim] + delta 
            elif dim in speed_sensor_indices:
                sig = speed_noise * np.abs(single_data_point[dim])
                sig = np.maximum(sig, 1e-6)
                delta = 3 * sig
                lb = single_data_point[dim] - delta
                ub = single_data_point[dim] + delta
            elif dim in range(dims):
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

    for i,bounds in enumerate(init_state_bounds_list):
        init_state_lb = np.array([b[0] for b in bounds])
        # print("init_state_lb:",init_state_lb)
        init_state_ub = np.array([b[1] for b in bounds])

        # X0 = Star(init_state_lb,init_state_ub)
        # lb_X0 = X0.getRanges()[0]
        # ub_X0 = X0.getRanges()[1]
        # print("==== get ranges LB======:",lb_X0)
        # print("==== get ranges UB ======:",ub_X0)

        # mu = 0.5*(lb_X0+ub_X0)
        # a = 3
        # sig = (mu-lb_X0)/a
        # epsilon = 1e-12
        # sig = np.maximum(sig, epsilon).astype(np.float64)
        # Sig = np.diag(np.square(sig)).astype(np.float64)
        # X0_probstar = ProbStar(mu, Sig,lb_X0,ub_X0)
        # # X0_probstar.C = np.zeros([1,X0_probstar.nVars])  
        # # X0_probstar.d = np.zeros([1])
        # print(f"initial probstar set {i}:{X0_probstar}")
        # print(f"probability of the initial ProbStar set {i}:{X0_probstar.estimateProbability()}")
        # X.append(X0_probstar)
        
        # map_mat = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
        X0 = Star(init_state_lb,init_state_ub)
        # print("X0_d.shape[0]:",X0.d.shape[0])
        X0.C = np.empty([X0.d.shape[0],X0.nVars])  
        X0.d = np.empty([X0.d.shape[0]])
        # print("X0:",X0)
        mu = 0.5*(X0.pred_lb + X0.pred_ub) 
        a  = 3
        sig= (mu - X0.pred_lb)/a
        epsilon = 1e-6
        sig = np.maximum(sig, epsilon)
        Sig = np.diag(np.square(sig))
        X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu, Sig,X0.pred_lb,X0.pred_ub)
        # print(f"initial probstar set {i}:{X0_probstar}")
        # print(f"each initial probsatrset V:{X0_probstar.V}, C:{X0_probstar.C},d:{X0_probstar.d}")
        print(f"probability of the initial ProbStar set {i}:{X0_probstar.estimateProbability()}")
        X.append(X0_probstar)

        # star_set = X0.affineMap(map_mat)
        # # plot_probstar(set)
        # plot_2D_Star(star_set)

        # print("Input ProbStar set constructed:",X0_probstar)

    return X


def get_ProbStar_set_RNN_global(input_data, noises, feature_idx):
    """Option B: build a ProbStar signal with a single global predicate vector.

    We lift the entire input sequence into one predicate vector:
        a = [a0, a1, ..., a(T-1)]
    with block-diagonal covariance (independent blocks).

    Each timestep's input ProbStar uses only its own block in the basis matrix,
    but *shares* the same (mu, Sig, pred_lb, pred_ub) with all other timesteps.

    This is the representation required for exact ProbStarTL evaluation over an
    RNN signal with independent per-step uncertainties.
    """

    # Reuse the per-step construction to keep bounds/noise logic consistent.
    local = get_ProbStar_set_RNN(input_data, noises=noises, feature_idx=feature_idx)
    assert isinstance(local, list) and all(isinstance(s, ProbStar) for s in local), \
        'error: expected get_ProbStar_set_RNN to return a list of ProbStars'

    infos = []
    for s in local:
        infos.append({
            "V": s.V,
            "C": s.C,
            "d": s.d,
            "mu": s.mu,
            "Sig": s.Sig,
            "pred_lb": s.pred_lb,
            "pred_ub": s.pred_ub,
            "nVars": s.nVars,
        })

    nVars_total = int(sum(info["nVars"] for info in infos))
    if nVars_total == 0:
        return local

    mu_global = np.concatenate([info["mu"] for info in infos])
    Sig_global = block_diag(*[info["Sig"] for info in infos])
    pred_lb_global = np.concatenate([info["pred_lb"] for info in infos])
    pred_ub_global = np.concatenate([info["pred_ub"] for info in infos])

    X = []
    offset = 0
    for info in infos:
        V_local = info["V"]
        n_i = int(info["nVars"])

        V_global = np.zeros((V_local.shape[0], 1 + nVars_total), dtype=V_local.dtype)
        V_global[:, 0] = V_local[:, 0]
        if n_i > 0:
            V_global[:, 1 + offset:1 + offset + n_i] = V_local[:, 1:1 + n_i]

        if len(info["C"]) != 0:
            C_i = info["C"]
            C_global = np.zeros((C_i.shape[0], nVars_total), dtype=C_i.dtype)
            C_global[:, offset:offset + n_i] = C_i
            d_global = info["d"]
        else:
            C_global = np.empty((0, nVars_total))
            d_global = np.empty((0,))

        S = ProbStar(V_global, C_global, d_global,
                    mu_global, Sig_global,
                    pred_lb_global, pred_ub_global)
        X.append(S)
        offset += n_i

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
   
