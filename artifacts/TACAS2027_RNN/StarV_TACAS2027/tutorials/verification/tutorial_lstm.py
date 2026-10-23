import os
import copy
import csv
import StarV
import numpy as np
from StarV.set.star import Star
from StarV.set.sparsestar import SparseStar
from StarV.util.load import load_LSTM_network
from StarV.verifier.certifier import reachBFS, certifyRobustness_sequence

path = os.path.dirname(StarV.__file__)

def lstm_rechability_sparsestar():

    type='lstm'
    in_shape = 784
    num_seq = 2
    frame_size = in_shape // num_seq
    hidden=15
    
    # load dataset 
    data_dir = f'{path}/util/data/nets/NAHS2024_RNN'
    with open(f'{data_dir}/mnist_test.csv', 'r') as x:
        test_data = list(csv.reader(x, delimiter=","))

    test_data = np.array(test_data)
    # data
    XTest = copy.deepcopy(test_data[:, 1:]).astype('float32').T / 255
    XTest = XTest.reshape([frame_size, num_seq, 10000], order='F')
    data = XTest[:, :, 0]
    
    # load LSTM neural network
    net_name = f'MNIST_{type.upper()}{hidden}net'
    net_dir = f'{data_dir}/{net_name}.onnx'
    net = load_LSTM_network(net_dir, net_name)
    
    epsilon = 0.005
    # Construct input sequential SparseStar sets
    print('Input sequential SparseStar sets:')
    X = []
    for i in range(num_seq):
        S = SparseStar(np.clip(data[:, i] - epsilon, 0, 1), 
                       np.clip(data[:, i] + epsilon, 0, 1))
        X.append(S)
        print(f'sequence {i}: \n')
        repr(S)
        print()

    # Reachability Analysis  
    Y, _ = reachBFS(net=net, inputSet=X, reachMethod='approx', lp_solver='gurobi')
    print('Output SparseStar after reachability analysis:')
    print(Y)
    

def lstm_certification_sparsestar():  
    type='lstm'
    in_shape = 784
    num_seq = 2
    frame_size = in_shape // num_seq
    hidden=15
    
    # load dataset 
    data_dir = f'{path}/util/data/nets/NAHS2024_RNN'
    with open(f'{data_dir}/mnist_test.csv', 'r') as x:
        test_data = list(csv.reader(x, delimiter=","))

    test_data = np.array(test_data)
    # data
    XTest = copy.deepcopy(test_data[:, 1:]).astype('float32').T / 255
    XTest = XTest.reshape([frame_size, num_seq, 10000], order='F')
    data = XTest[:, :, 0]
    
    # load LSTM neural network
    net_name = f'MNIST_{type.upper()}{hidden}net'
    net_dir = f'{data_dir}/{net_name}.onnx'
    net = load_LSTM_network(net_dir, net_name)
    
    Y, rb, vt = certifyRobustness_sequence(net, data, epsilon=0.005, lp_solver='gurobi', DR=0, show=False)
    print('Output SparseStar after reachability analysis:')
    repr(Y)
    rb = 'Robust' if rb[0] else 'Unknown'
    print('Certification result: ', rb)
    print(f'Verification time: {vt} sec')

if __name__ == "__main__":
    lstm_rechability_sparsestar()
    lstm_certification_sparsestar()