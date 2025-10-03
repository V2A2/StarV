"""
Test RecurrentLayer Class
Author: Qing Liu
Date: 9/28/2025
"""

import numpy as np
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.RecurrentLayer import RecurrentLayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.net.network import NeuralNetwork
from StarV.util.load_rnn import load_simple_rnn, get_Star_set,get_ProbStar_set
from StarV.util.plot import plot_2D_Star



class Test(object):
    """
    Testing RecurrenltLayer class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_constructor(self):

        self.n_tests = self.n_tests + 1
        Whx = np.random.rand(2, 3)
        Whh = np.random.rand(3, 3)
        bh = np.random.rand(3)
        Woh = np.random.rand(3, 3)
        bo = np.random.rand(3)
        print('\nTest RecurrentLayer Constructor\n')

        try:
            L = RecurrentLayer(Whx, Whh, bh, Woh, bo)
            # L.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_rand(self):
        self.n_tests = self.n_tests + 1
        print('\nTest RecurrentLayer random method\n')

        try:
            RecurrentLayer.rand(3, 2)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_reachExact(self):
        self.n_tests = self.n_tests + 1
        print('\nTest RecurrentLayer reachExact method\n')

        try:
            L = RecurrentLayer.rand(2, 3)
            # L.info()
            In =[]
            I=Star.rand(2)
            print('Input Set:')
            I.__str__()
            I.C = np.zeros([1,I.nVars])  
            I.d = np.zeros([1])
            In.__str__()
            for i in range(5):
                In.append(I)
            S = L.reachExact(In)
            print('Number of utput Set:'.format(len(S)))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_reachApprox(self):
        self.n_tests = self.n_tests + 1
        print('\nTest RecurrentLayer reachExact method\n')

        try:
            L = RecurrentLayer.rand(2, 3)
            In = []
            I = Star.rand(2)
            print('Input Set:')
            I.__str__()
            I.C = np.zeros([1,I.nVars])  
            I.d = np.zeros([1])
            for i in range(5):
                In.append(I)
            S = L.reachApprox(In)
            print('Number of utput Set:'.format(len(S)))
            # S[i].__str__()

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_reach(self):
        self.n_tests = self.n_tests + 1
        print('\nTest RecurrentLayer reach method\n')
        
        try:
            L = RecurrentLayer.rand(2, 3)
            In = []
            I = Star.rand(2)
            I.C = np.zeros([1,I.nVars])  
            I.d = np.zeros([1])
            for i in range(5):
                In.append(I)
            print('Number of input sets: {}'.format(len(In)))
            S = L.reach(In)
            print('Number of output Set: {}'.format(len(S)))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_simple_rnn(self):

        self.n_tests = self.n_tests + 1
        print('\n Test Simple Recurrent Neural Network\n')

        Whx,Whh,bh,Woh,bo,data_points, W_ff,b_ff = load_simple_rnn()

        # create NN
        L1 = RecurrentLayer(Whx,Whh,bh,Woh,bo)
        mat= []
        for i in range(len(W_ff[0])):
            W_b = [W_ff[0][i],b_ff[0][i].reshape(-1)]
            mat.append(W_b)

        L2 = FullyConnectedLayer(mat[0],fo='relu')
        L3 = FullyConnectedLayer(mat[1],fo ='relu')
        L4 = FullyConnectedLayer(mat[2],fo='relu')
        L5 = FullyConnectedLayer(mat[3],fo='relu')
        L6 = FullyConnectedLayer(mat[4],fo='relu')
        L7 = FullyConnectedLayer(mat[5],fo='relu')


        layers = [L1,L2,L3,L4,L5,L6,L7]
        net = NeuralNetwork(layers=layers)

        num_input_seq= 5
        x = data_points[:num_input_seq,:]
        x = x.T
    
        eps = 0.01
        time_steps = 5 
        results = []
        try :
            for k in range(0,num_input_seq):
                xk = np.array(x[:, k]).reshape(-1,1)
                input_points = []  
                col_points = []
                for _ in range(time_steps) :        
                    col_points.append(xk) # repeating T times
                input_points = np.hstack(col_points) # (40, T)
                S = get_ProbStar_set(input_points=input_points,eps=eps)
                Numlayers = len(layers)
                Layer_RS = []
                RS = S
                for j in range(0,Numlayers):
                    # layers[j].info()
                    RS1 = net.layers[j].reach(RS, method = "exact", lp_solver='gurobi', pool=None, RF=0.0, DR=0)
                    RS = RS1
                    Layer_RS.append(RS1)
                result = RS1
                results.append(result)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

        
        return results

    def test_multiRandomLayers(self):
        self.n_tests = self.n_tests + 1
        print('\nTest multiRecurrentLayer\n')
        
        try:
            L1 = RecurrentLayer.rand(2, 8)
            L2 = FullyConnectedLayer.rand(8,12)
            layers = [L1,L2]
            net = NeuralNetwork(layers=layers)
            Numlayers = len(layers)
            In = []
            I = Star.rand(2)
            I.C = np.zeros([1,I.nVars])  
            I.d = np.zeros([1])
            for i in range(5):
                In.append(I)
            print('Number of input sets: {}'.format(len(In)))
            results = []
            RS = In
            for j in range(0,Numlayers):
                print("\n====== Process the {}th layer of NN ===========".format(j+1))
                layers[j].info()
                RS1 = net.layers[j].reach(RS, method = "exact", lp_solver='gurobi', pool=None, RF=0.0, DR=0)
                RS = RS1
            result = RS1
            results.append(result)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
            
if __name__ == "__main__":

    test_RecurrentLayer = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    # test_RecurrentLayer.test_constructor()
    # test_RecurrentLayer.test_rand()
    test_RecurrentLayer.test_reachExact()
    # test_RecurrentLayer.test_reachApprox()
    # test_RecurrentLayer.test_reach()
    # test_RecurrentLayer.test_simple_rnn()
    # test_RecurrentLayer.test_multiRandomLayers()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing Recurrent Layer Class: fails: {}, successfull: {}, \
    total tests: {}'.format(    test_RecurrentLayer.n_fails,
                                test_RecurrentLayer.n_tests - test_RecurrentLayer.n_fails,
                                test_RecurrentLayer.n_tests))

    

