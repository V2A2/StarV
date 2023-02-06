"""
Fully Connected Layer Class
Dung Tran, 9/9/2022
"""

import numpy as np
import multiprocessing

class fullyConnectedLayer(object):
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

        assert isinstance(W, np.ndarray), 'error: weight matrix should be a numpy array'
        assert isinstance(b, np.ndarray), 'error: bias vector should be a numpy array'

        assert W.shape[0] == b.shape[0], 'error: inconsistent dimension between weight matrix and bias vector'
        self.W = W
        self.b = b
        self.in_dim = W.shape[1]
        self.out_dim = W.shape[0]

    def evaluate(self, x):
        'evaluation on an input vector/array x'

        assert isinstance(x, np.ndarray), 'error: input vector should be a numpy array'
        assert x.shape[0] == self.in_dim, 'error: inconsistent dimension between the weight matrix and input vector'
        
        b1 = self.b.reshape(self.out_dim, 1)
        y = np.matmul(self.W, x) + b1
        return y

    @staticmethod
    def rand(in_dim, out_dim):
        """ Random generate a fullyConnectedLayer"""

        W = np.random.rand(out_dim, in_dim)
        b = np.random.rand(out_dim)

        return fullyConnectedLayer(W, b)

    def reachExactSingleInput(self, In):
        return In.affineMap(self.W, self.b)
        
    def reach(self, inputSet, method=None, lp_solver='gurobi', pool=None, RF=0.0):
        """main reachability method
           Args:
               @I: a list of input set (Star or ProbStar)
               @pool: parallel pool: None or multiprocessing.pool.Pool
               
            Return: 
               @R: a list of reachable set
            Unused inputs: method, lp_solver, RF (relaxation factor)

        """

        pool = None
        
        S = []
        if pool is None:
            for i in range(0, len(inputSet)):
                S.append(self.reachExactSingleInput(inputSet[i]))
        elif isinstance(pool, multiprocessing.pool.Pool):
            S = S + pool.map(self.reachExactSingleInput, inputSet)
        else:
            raise Exception('error: unknown/unsupport pool type')              
                
        return S
