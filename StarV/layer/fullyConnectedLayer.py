import numpy as np
import multiprocessing
from typing import List, Union
from StarV.set.probstar import ProbStar
from StarV.set.star import Star

class fullyConnectedLayer:
    """fullyConnectedLayer class
        properties: 
            @W: weight matrix
            @b: bias vector
        methods:
            @evaluate: evaluate method
            @reach: reach method
            @rand: random generate a fullyConnectedLayer
    """
    def __init__(self, W, b):
        if not isinstance(W, np.ndarray):
            raise ValueError('Weight matrix should be a numpy array')
        if not isinstance(b, np.ndarray):
            raise ValueError('Bias vector should be a numpy array')
        if W.shape[0] != b.shape[0]:
            raise ValueError('Inconsistent dimension between weight matrix and bias vector')
        
        self.W = W
        self.b = b
        self.in_dim = W.shape[1]
        self.out_dim = W.shape[0]

    def evaluate(self, x):
        """Evaluation on an input vector/array x"""
        if not isinstance(x, np.ndarray):
            raise ValueError('Input vector should be a numpy array')
        if x.shape[0] != self.in_dim:
            raise ValueError('Inconsistent dimension between the weight matrix and input vector')
        
        return np.matmul(self.W, x) + self.b.reshape(self.out_dim, 1)

    @staticmethod
    def rand(in_dim, out_dim):
        """Random generate a fullyConnectedLayer"""
        W = np.random.rand(out_dim, in_dim)
        b = np.random.rand(out_dim)
        return fullyConnectedLayer(W, b)

    def reachExactSingleInput(self, In):
        """Compute exact reachable set for a single input"""
        return In.affineMap(self.W, self.b)
        
    def reach(self, inputSet, method=None, lp_solver='gurobi', pool=None, RF=0.0):
        """Main reachability method
           Args:
               @inputSet: a list of input set (Star or ProbStar)
               @method: unused
               @lp_solver: unused
               @pool: parallel pool: None or multiprocessing.pool.Pool
               @RF: unused
            
            Return: 
               @S: a list of reachable sets
        """
        print("\nfullyConnectedLayer reach function\n")        
        if pool is None:
            return [self.reachExactSingleInput(input_set) for input_set in inputSet]
        elif isinstance(pool, multiprocessing.pool.Pool):
            return pool.map(self.reachExactSingleInput, inputSet)
        else:
            raise ValueError('Unknown or unsupported pool type')