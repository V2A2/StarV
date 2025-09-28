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
from StarV.util.load_rnn import load_simple_rnn, get_reachable_set


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

        self.Whx = Whx
        self.Whh = Whh
        self.bhx = bh
        self.Woh = Woh
        self.bo = bo
        self.in_dim = Whx.shape[1] 
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
        print('')
        return '\n'

    def info(self):
        print(self)


    def reachExact(self,In, method = "exact", lp_solver='gurobi', pool=None, RF=0.0, DR=0):

        print("~~~~~~~~~~~~~~` Using {} method for reachability ~~~~~~~".format(method))
        assert isinstance(In, list), "Input should be a list of Star/ProbStar sets"

        Weight_In = [] # store a list of mapped input list(contain only one set), T sets
        H = []
        O = []
        for i in range(0,len(In)):
            WIn=[]
            # if type(In[i]) is list: # previous output is a list of list  sets
            #     for I in In[i]:
            #         MIn = I.affineMap(self.Whx,self.bhx)
            #         WIn.append(MIn)
            #     Weight_In.append(WIn)
            # else: 
            MIn = In[i].affineMap(self.Whx,self.bhx)
            WIn.append(MIn)
            Weight_In.append(WIn)
        for i in range(0,len(In)):
            if i == 0 :
                H0 =[]
                H0_out = ReLULayer.reach(Weight_In[i], method = method)
                H0.extend(H0_out)
                H.append(H0)
                O1 = []
                for h in H0:
                    HO_out = h.affineMap(self.Woh,self.bo)
                    O1.append(HO_out)
                O.append(O1)
            else: # i > 1
                pre_H = H[i-1]
                H3 = []
                for h in pre_H:
                    H4= []
                    H2_weight = h.affineMap(self.Whh)
                    H2_sum = H2_weight.minKowskiSum(Weight_In[i][0])
                    H4.append(H2_sum)
                    H2_out = ReLULayer.reach(H4, method = method)
                    H3.extend(H2_out)
                H.append(H3)
                O2 = []
                for h in H3:
                    HO_out = h.affineMap(self.Woh,self.bo)
                    O2.append(HO_out)
                O.append(O2)


        return O
    
    def reachApprox(self,In, method = "approx", lp_solver='gurobi', pool=None, RF=0.0, DR=0):
        print("~~~~~~~~~~~~~~` Using {} method for reachability ~~~~~~~".format(method)) 
        
        assert isinstance(In, list), "Input should be a list of Star/ProbStar sets"

        Weight_In = [] # store a list of mapped input list(contain only one set), T sets
        H = []
        O = []
        for i in range(0,len(In)):
            MIn = In[i].affineMap(self.Whx,self.bhx)
            Weight_In.append(MIn)
        for i in range(0,len(In)):
            if i == 0 :
                H0_out = ReLULayer.reach(Weight_In[i], method = method, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False)
                H.append(H0_out)
                HO_out = H0_out.affineMap(self.Woh,self.bo)
                O.append(HO_out)
            else: # i > 1
                pre_H = H[i-1]
                H2_weight = pre_H.affineMap(self.Whh)
                H2_sum = H2_weight.minKowskiSum(Weight_In[i])
                H2_out = ReLULayer.reach(H2_sum, method = method)
                H.append(H2_out)
                HO_out = H2_out.affineMap(self.Woh,self.bo)
                O.append(HO_out)

        return O

    def reach(self,In, method = None, lp_solver='gurobi', pool=None, RF=0.0, DR=0):
        if method is None:
            method = "exact"
        if method == "exact":
            S = self.reachExact(In, method, lp_solver, pool, RF, DR)
            # print("(Rcuurent layer)Final output set len:",len(S))
            return S
        elif method == "approx":
            return self.reachApprox(In, method, lp_solver, pool, RF, DR)
        else:
            raise Exception(f"error: unknown reachability method: {method}")

