import numpy as np
import multiprocessing
import scipy
import time

from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.MixedActivationLayer import MixedActivationLayer
from StarV.net.network import NeuralNetwork, reachExactBFS

def load_modelNN_controllerNN():
    folder_dir = './artifacts/Scherlock_ACC_Trapezius'
    net_name = 'networks'
    mat_file = scipy.io.loadmat(f"{folder_dir}/{net_name}.mat")
    controller_nn = mat_file['controller_nn']
    model_nn = mat_file['Model_nn']

    Wc = controller_nn[0][0][0].ravel()
    bc = controller_nn[0][0][1].ravel()
    
    n_weight = len(Wc)

    controller_layers = []
    for i in range(n_weight):
        controller_layers.append(fullyConnectedLayer(Wc[i], bc[i].ravel()))
        controller_layers.append(ReLULayer())
            
    controller_net = NeuralNetwork(controller_layers, net_type='controller network')

    Wm = model_nn[0][0][0].ravel()
    bm = model_nn[0][0][1].ravel()
    
    n_weight = len(Wm)

    model_layers = []
    for i in range(n_weight):
        model_layers.append(fullyConnectedLayer(Wm[i], bm[i].ravel()))
        model_layers.append(ReLULayer())
            
    model_net = NeuralNetwork(controller_layers, net_type='model network')

    return model_net, controller_net

def load_Trapezius_network():
    folder_dir = './artifacts/Scherlock_ACC_Trapezius'
    net_name = 'linear_nonliear_trapezius_network__scherlock-acc'
    mat_file = scipy.io.loadmat(f"{folder_dir}/{net_name}.mat")
    Net = mat_file['Net']
    W = Net[0][0][0].ravel()
    b = Net[0][0][1].ravel()
    act_fun = Net[0][0][2].ravel() # containts activation list of each neurons
    
    n_weight = len(W)
    n_act_fun = len(act_fun)
    
    
    layers = []
    for i in range(n_weight):
        layers.append(fullyConnectedLayer(W[i].toarray(), b[i].toarray().ravel()))
        if i < n_act_fun:
            layers.append(MixedActivationLayer(act_fun[i]))
            
    network = NeuralNetwork(layers, net_type=net_name)
    return network

def quantiverify_trapezius_network(lp_solver = 'gurobi', numCores=1, show=True):
    center = np.array([100, 32.1, 0, 10.5, 30.1, 0])
    epsilon = np.array([10, 0.1, 0, 0.5, 0.1, 0])
    lb = center - epsilon
    ub = center + epsilon

    S = Star(lb, ub)
    mu = 0.5*(S.pred_lb + S.pred_ub)
    a = 2.5 # coefficience to adjust the distribution
    sig = (mu - S.pred_lb)/a
    print('Mean of predicate variables: mu = {}'.format(mu))
    print('Standard deviation of predicate variables: sig = {}'.format(sig))
    Sig = np.diag(np.square(sig))
    print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
    InputSet = [ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)]

    starv_net = load_Trapezius_network()
    print(starv_net)

    if numCores > 1:
        pool = multiprocessing.Pool(numCores)
    else:
        pool = None
    
    start = time.perf_counter()
    OutputSet = reachExactBFS(starv_net, InputSet, lp_solver, pool, show)

    k = len(OutputSet)
    lb = np.zeros(k)
    ub = np.zeros(k)
    for i in range(k):
        lb[i], ub[i] = OutputSet[i].getRanges()

    lb = lb.min()
    ub = ub.max()
    interval = [lb, ub]
    vt = time.perf_counter() - start

    print(f'interval: {interval}')
    print(f'verification time: {vt}')


if __name__ == "__main__":
    # quantiverify_trapezius_network(numCores = 1)
    load_modelNN_controllerNN()