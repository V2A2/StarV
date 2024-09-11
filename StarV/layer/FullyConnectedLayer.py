"""
Fully Connected Layer Class
Sung Woo Choi, 07/06/2023
"""

import torch
import numpy as np
import multiprocessing

class FullyConnectedLayer(object):
    """ FullyConnectedLayer class

        properties: 
            @W: weight matrix
            @b: bias vector
        methods:
            @evaluate: evaluate method
            @reach: reach method
            @rand: random generate a FullyConnectedLayer
    """

    def __init__(self, layer, dtype='float64'):

        if isinstance(layer, list):
            W, b = layer

        elif isinstance(layer, torch.nn.Linear):
            W = layer.weight.data.numpy()
            b = layer.bias.data.numpy()

        else:
            raise Exception('error: unsupported neural network module')

        # assert (isinstance(W, np.ndarray) and len(W.shape) == 2) or W is None, 'error: weight matrix should be a 2D numpy array'
        # assert (isinstance(b, np.ndarray) and len(b.shape) == 1) or b is None, 'error: bias vector should be a 1D numpy array'

        assert isinstance(W, np.ndarray) or W is None, 'error: weight matrix should be a numpy array'
        assert isinstance(b, np.ndarray) or b is None, 'error: bias vector should be a numpy array'

        if dtype == 'float32' or dtype == np.float32:
            self.torch_dtype = torch.float32
        else:
            self.numpy_dtype = np.float64
        
        if W is not None and b is not None:
            assert W.shape[0] == b.shape[0], 'error: inconsistent dimension between weight matrix and bias vector'
        
        if W is None:
            self.W = W
        else:
            self.W = W.astype(dtype)
        if b is None:
            self.b = b
        else:
            self.b = b.astype(dtype)
        
        if W is not None:
            self.in_dim = W.shape[1]
            self.out_dim = W.shape[0]
        else:
            self.in_dim = b.shape[0]
            self.out_dim = b.shape[0]

    def evaluate(self, x):
        """evaluation on an input vector/array x"""

        assert isinstance(x, np.ndarray), 'error: input vector should be a numpy array'
        assert x.shape[0] == self.in_dim or self.in_dim == 1, 'error: inconsistent dimension between the weight matrix and input vector'
        
        W = self.W
        b = self.b

        if W is not None and b is not None: 
            if x.ndim == 1:
                return np.matmul(W, x) + b
            else:
                return np.matmul(W, x) + b[:, np.newaxis]
        
        elif W is not None:
            if x.ndim == self.W.ndim:
                return W * x 
        
            return np.matmul(W, x)

        elif b is not None:

            if x.ndim > 1:
                return x + np.expand_dims(b, axis=tuple(np.arange(x.ndim-b.ndim)+b.ndim))
            
            return x + b
        
            
        return x
        

    @staticmethod
    def rand(in_dim, out_dim):
        """ Random generate a FullyConnectedLayer"""

        W = np.random.rand(out_dim, in_dim)
        b = np.random.rand(out_dim)

        return FullyConnectedLayer(layer=[W, b], module='default')

    def reachExactSingleInput(self, In):
        return In.affineMap(self.W, self.b)
        
    def reach(self, inputSet, method=None, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """main reachability method
           Args:
               @inputSet: a list of input set (Star, ProbStar, or SparseStar)
               @pool: parallel pool: None or multiprocessing.pool.Pool
               @RF: relaxation factor
               @DR: depth reduction; maximum depth allowed for predicate variables
               
            Return: 
               @R: a list of reachable set
            Unused inputs: method, lp_solver, RF (relaxation factor), DR (depth reduction)

        """

        pool = None
        
        if isinstance(inputSet, list):
            S = []
            if pool is None:
                for i in range(0, len(inputSet)):
                    S.append(self.reachExactSingleInput(inputSet[i]))
            elif isinstance(pool, multiprocessing.pool.Pool):
                S = S + pool.map(self.reachExactSingleInput, inputSet)
            else:
                raise Exception('error: unknown/unsupport pool type')         

        else:
            S = self.reachExactSingleInput(inputSet)
                
        return S
