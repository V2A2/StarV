"""
Zonotope Class
Author: Sung Woo Choi
Created: 06/13/2025

"""
import copy
import torch
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from itertools import product
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp

# from scipy.optimize import linprog
from scipy.linalg import block_diag
# import glpk
# import polytope as pc

from StarV.layer.TanSigLayer import TanSig
from StarV.layer.LogSigLayer import LogSig 
# from StarV.fun.invsqrt import InvSqrt


class Zonotope(object):
    """
        Zonotope Class for reachability
        Representation of a Zonotope (normalized/unormalized)
        ==========================================================================
        Zonotope set defined by
        x = c + a[1]*v[1] + a[2]*v[2] + ... + a[n]*v[n]
            = G * b,
        where G = [c v[1] v[2] ... v[n]],
              b = [1 a[1] a[2] ... a[n]]^T,
        if norm:
            -1 <= a <= u
        else:
            l <= a <= u
        ==========================================================================
    """

    def __init__(self, *args, copy_=True):
        """
            Key Attributes:
            V = []; generator matrix
            c = []; center vector
            G = [c V];
            l = []; lower predicate bound
            u = []; upper predicate bound
            normalized = bool; normalized predicate between [-1, 1]
        """
        if len(args) == 5:
            [G, l, u, shape, norm] = copy.deepcopy(args) if copy_ is True else args

            assert isinstance(G, np.ndarray), 'error: ' +\
            'generator matrix should be a 2D numpy array'
            assert isinstance(l, np.ndarray), 'error: ' +\
            'predicate lower bound vector should be a 1D numpy array'
            assert isinstance(u, np.ndarray), 'error: ' +\
            'predicate upper bound vector should be a 1D numpy array'
            assert len(G.shape) == 2, 'error: ' +\
            'generator matrix should be a 2D numpy array'
            assert np.prod(shape) == G.shape[0], 'error: ' +\
            'inconsistent shape information'
            assert isinstance(norm, bool), 'error: ' +\
            'normalized flag should be a bool parameter'

            self.G = G
            self.l = l
            self.u = u
            self.dim = G.shape[0]
            self.nVars = G.shape[1] - 1
            self.shape = shape
            self.normalized = norm

        elif len(args) == 4:
            [G, l, u, norm] = copy.deepcopy(args) if copy_ is True else args

            assert isinstance(G, np.ndarray), 'error: ' +\
            'generator matrix should be a 2D numpy array'
            assert isinstance(l, np.ndarray), 'error: ' +\
            'predicate lower bound vector should be a 1D numpy array'
            assert isinstance(u, np.ndarray), 'error: ' +\
            'predicate upper bound vector should be a 1D numpy array'
            assert len(G.shape) == 2, 'error: ' +\
            'generator matrix should be a 2D numpy array'
            assert isinstance(norm, bool), 'error: ' +\
            'normalized flga should be a bool parameter'

            self.G = G
            self.l = l
            self.u = u
            self.dim = G.shape[0]
            self.nVars = G.shape[1] - 1
            self.shape = G.shape[0]
            self.normalized = norm

        elif len(args) == 3:
            [G, l, u] = copy.deepcopy(args) if copy_ is True else args

            assert isinstance(G, np.ndarray), 'error: ' +\
            'generator matrix should be a 2D numpy array'
            assert isinstance(l, np.ndarray), 'error: ' +\
            'predicate lower bound vector should be a 1D numpy array'
            assert isinstance(u, np.ndarray), 'error: ' +\
            'predicate upper bound vector should be a 1D numpy array'
            assert len(G.shape) == 2, 'error: ' +\
            'generator matrix should be a 2D numpy array'

            self.G = G
            self.l = l
            self.u = u
            self.dim = G.shape[0]
            self.nVars = G.shape[1] - 1
            self.shape = G.shape[0]
            self.normalized = True if (self.l == 1).all() and (self.u == 1).all() else False
                

        elif len(args) == 2:
            [l, u] = copy.deepcopy(args) if copy_ is True else args

            assert isinstance(l, np.ndarray), 'error: ' +\
            'lower bound vector should be a 1D numpy array'
            assert isinstance(u, np.ndarray), 'error: ' +\
            'upper bound vector should be a 1D numpy array'
            assert l.shape == u.shape, 'error: ' +\
            'inconsistency between lower bound and upper bound vectors'
            
            shape = l.shape
            
            if l.ndim > 1:
                l = l.reshape(-1)
                u = u.reshape(-1)

            if np.any(u < l): 
                raise Exception('error: ' +\
            'the upper bound must not be smaller than the lower bound')

            dtype = l.dtype
            dim = l.shape[0]
            nv = int(sum(u > l))

            # center = 0.5*(l + u)[:, None]
            # vec = 0.5*(u - l)
            # gens = np.diag(vec)
            # gens = gens[:,~np.all(gens == 0, axis=0)]
            # G = np.hstack((center, gens))

            G = np.zeros((dim, nv+1), dtype=dtype)
            j = 1
            for i in range(dim):
                if u[i] > l[i]:
                    G[i, j] = 1.0
                    j += 1

            self.G = G
            self.nVars = nv
            self.l = l
            self.u = u
            # self.l = -np.ones(nv,)
            # self.u = np.ones(nv,)
            self.dim = G.shape[0]
            # self.normalized = True
            self.shape = shape
            self.normalized = False

        elif len(args) == 1:
            [G] = copy.deepcopy(args) if copy_ is True else args

            assert isinstance(G, np.ndarray), 'error: ' +\
            'generator matrix should be a 2D numpy array'
            assert len(G.shape) == 2, 'error: ' +\
            'generator matrix should be a 2D numpy array'

            self.G = G
            self.nVars = G.shape[1] - 1
            self.l = -np.ones(self.nVars,)
            self.u = np.ones(self.nVars,)
            self.dim = G.shape[0]
            self.shape = shape
            self.normalized = True
        
        else:
            raise Exception('error: ' + \
            'invalid number of input arguments (should be 1, 2, or 3)')


    def __str__(self):
        print('Zonotope Set:')
        print(f'normalized: {self.normalized}')
        print(f'nVars: {self.nVars}')
        print(f'G:\n{self.G}')
        if not self.normalized:
            print(f'l: {self.l}')
            print(f'u: {self.u}')
        return ''
    
    def __repr__(self):
        print('Zonotope Set:')
        print(f'normalized: {self.normalized}')
        print(f'nVars: {self.nVars}')
        print(f'G:\n{self.G.shape}')
        if not self.normalized:
            print(f'l: {self.l.shape}')
            print(f'u: {self.u.shape}')
        print('')
        return ''
    
    @property
    def c(self):
        return self.G[:, 0]
    
    @property
    def V(self):
        return self.G[:, 1:]

    def clone(self):
        return copy.deepcopy(self)
    
    def normalize(self):
        if self.normalized:
            return self
        
        G = copy.deepcopy(self.G)
        c = G[:, 0][:, None]
        gen = G[:, 1:]

        mid = 0.5 * (self.l + self.u)
        rad = 0.5 * (self.u - self.l)

        c += np.matmul(gen, mid[:, None])
        gen *= rad[None, :]

        G = np.hstack([c, gen])
        return Zonotope(G, copy_=False)
    
    def isNoramlized(self):
        return np.all(self.l == -1) and np.all(self.u == 1)
    
    def dot(self, A):
        'dot product'
        assert isinstance(A, np.ndarray), 'error: ' + \
        'the input A should be a numpy array'
        if np.prod(A.shape) == 1:
            A = A.ravel()

        G = A*self.G

        if self.normalized:
            return Zonotope(G)
        return Zonotope(G, self.l, self.u, self.normalized)
    
    def add(self, b):
        'addition'

        assert isinstance(b, np.ndarray), 'error: ' + \
            'the offset b should be a numpy array'
        
        G = copy.deepcopy(self.G)
            
        if np.prod(b.shape) == 1:
            b = b.ravel()

        G[:, 0] += b

        if self.normalized:
            return Zonotope(G)
        return Zonotope(G, self.l, self.u, self.normalized)
    
    def affineMap(self, A=None, b=None):
        if A is None and b is None:
            return self
        
        G = copy.deepcopy(self.G)
        
        if A is not None:
            assert isinstance(A, np.ndarray), 'error: ' + \
            'the mapping matrix A should be a 2D numpy array'

            if np.prod(A.shape) == 1:
                G = A.ravel()*G
            else:
                assert A.shape[1] == self.dim, 'error: ' + \
                'inconsistency between the mapping matrix and Zonotope dimension'
                G = np.matmul(A, G)
                
        if b is not None:
            assert isinstance(b, np.ndarray), 'error: ' + \
            'the offset vector b should be a 1D numpy array'
            
            if np.prod(b.shape) == 1:
                b = b.ravel()

            assert len(b.shape) == 1, 'error: ' + \
            f'the offset vector b should be a 1D numpy array, but received {b.shape} shape'
            G[:, 0] += b
        
        if self.normalized:
            return Zonotope(G)
        return Zonotope(G, self.l, self.u, self.normalized)
    
    def getRanges(self):
        """Get lower and upper bound of state x"""

        if self.normalized:
            norm_inf = np.sum(np.abs(self.V), axis=1)
            min = self.c - norm_inf
            max = self.c + norm_inf
            return min, max
        
        pos = np.maximum(self.V, 0.0)
        neg = np.minimum(self.V, 0.0)
        
        min = self.c + np.matmul(neg, self.u) + np.matmul(pos, self.l)
        max = self.c + np.matmul(pos, self.u) + np.matmul(neg, self.l)
        return min, max
    
    def estimateRanges(self):
        return self.getRanges()

    def MinkowskiSum(self, X):
        assert isinstance(X, Zonotope), 'error: ' + \
        'an input set X is not a Zonotope'
        assert self.dim == X.dim, 'error: ' + \
        'inconsistency between the Zonotope and an input Zonotope'
        new_c = self.c + X.c
        new_G = np.hstack([new_c[:, None], self.V, X.V])
        new_l = np.hstack([self.l, X.l])
        new_u = np.hstack([self.u, X.u])
        new_norm = self.normalized & X.normalized
        return Zonotope(new_G, new_l, new_u, new_norm)

    # def toPolynomialStar(self):
    #     from StarV.set.polynomialstar import PolynomialStar
    #     return PolynomialStar(self.c, self.V, None, np.eye(self.dim))
    
    @staticmethod
    def rand(dim):
        """ Randomly generate a Zonotope """

        assert dim > 0, 'error: invalid dimension'
        l = np.random.rand(dim, ) * 2 - 1
        u = np.random.rand(dim, ) + l
        return Zonotope(l, u)

    def poslin(self, method='deepz', norm='un-norm', test=False):
        """
            Args:
                @method: over approximation method option; ['area', 'deepz']
                @norm: the new beta is in 
                    'unit': [0, 1]
                    'norm': [-1, 1]
                    'un-norm': [l_i, u_i]
            over-approximation of poslin (ReLU) activation function,
            poslin(x) = max(0, x)
            references:
                [1] Gagandeep Singh, Timon Gehr, Matthew Mirman, Markus Püschel, and Martin Vechev. 2018. Fast and effective robustness certification. In Proceedings of the 32nd International Conference on Neural Information Processing Systems (NIPS'18). 
        """
        assert norm in ['unit', 'norm', 'un-norm'], f'unknown normalization method, should be \'unit\', \'norm\', or \'un-norm\'; but received {type(norm)}'
        normalize = norm == 'norm' or self.normalized
        # norm |= self.normalized
        # compute state bounds
        dtype = self.G.dtype
        l, u = self.getRanges()

        new_G = copy.deepcopy(self.G)
        
        # for case u < 0: reset to zero
        map = np.argwhere(u <= 0).ravel()
        if len(map) > 0:
            new_G[map, :] = 0.0

        # for case u >= 0 and l <= 0: over-approximate
        map = np.argwhere((u > 0) & (l < 0)).ravel()
        m = len(map)
        if m > 0:
            l_ = l[map]; u_ = u[map]
            w_ = u_ - l_

            # compute the slope of a chord line segment
            if method == 'area':
                a = u_ / w_
                mu = -l_ * a
                if normalize:
                    mu *= 0.5
                    
            elif method == 'deepz':
                a = u_ / w_
                mu = np.maximum(-a*l_, (1.0 - a)*u_)
                if normalize:
                    mu *= 0.5
  
            else:
                raise Exception('unknown poslin overapproximate method')
            
            # y = {0 if u <= 0, x if l >= 0, a*x + mu + b*beta otherwise

            # a*x + mu
            new_G[map, :] *= a[:, None]
            new_G[map, 0] += mu

            # over-approximate with new predicate beta
            # a*x + mu + b*beta
            b = np.zeros([self.dim, m])
            if normalize:
                b[map, :] = np.diag(mu)

                if test:
                    b = np.zeros([self.dim])
                    b[map] = mu

            else:
                if norm == 'unit':
                    b[map, :] = np.diag(-mu)
                    
                elif norm == 'un-norm':
                    new_G[map, 0] += mu * l_ / w_
                    b[map, :] = np.diag(-mu / w_)

            new_G = np.hstack([new_G, b])


        if self.normalized:
            # no need to copy since new_G is not related to current Zonotope in terms of memory address
            return Zonotope(new_G, copy_=False)
        
        if m > 0:
            if normalize:
                l_ = -np.ones(m, dtype=dtype)
                u_ =  np.ones(m, dtype=dtype)
            else:
                if norm == 'unit':
                    l_ = np.zeros(m, dtype=dtype)
                    u_ = np.ones(m, dtype=dtype)

            lb = np.hstack([self.l, l_])
            ub = np.hstack([self.u, u_])
        else:
            lb = self.l
            ub = self.u

        # no need to copy since variables are not related to current Zonotope in terms of memory address
        return Zonotope(new_G, lb, ub, copy_=False)

    def tansig(self, method='tangent', norm='un-norm'):
        """TanSig (TanH) reachability
        
            Args:
                @method: over approximation method option; ['tangent', 'opt']
                @norm: the new beta is in 
                    'unit': [0, 1]
                    'norm': [-1, 1]
                    'un-norm': [l_i, u_i]
            over-approximation of tansig (TanH) activation function,
            poslin(x) = max(0, x)
        
        """
        # TODO: move to StarV.set.fun.tansig.py for Zonotope

        assert norm in ['unit', 'norm', 'un-norm'], f'unknown normalization method, should be \'unit\', \'norm\', or \'un-norm\'; but received {type(norm)}'
        normalize = norm == 'norm' or self.normalized

        dtype = self.G.dtype

        l, u = self.getRanges()
        fl, fu = TanSig.f(l), TanSig.f(u)
        dl, du = TanSig.df(l), TanSig.df(u)

        m = l.shape[0]
        new_G = copy.deepcopy(self.G)
        
        ## y = a*x + mu + b*beta
        # over-approximate by a tangent line
        # obtain the slope a
        if method == 'tangent': # outputs the tightest bounds
            a = np.minimum(dl, du)

        # over-approximate by a optimal line
        elif method == 'opt':
            xou = TanSig.optimal_iter_approx_upper(l, u, iter=5)
            xol = TanSig.optimal_iter_approx_lower(l, u, iter=5)
            dxou = TanSig.df(xou)
            dxol = TanSig.df(xol)
            a = np.minimum(dxou, dxol)

        else:
            raise Exception('unknown tansig over-approximate method for Zonotope')
        
        diff_ = u - l
        sum_ = u + l
        mu = 0.5*(fu + fl - a*sum_)
        b  = 0.5*(fu - fl - a*diff_)

        if normalize:
            # a*x + mu
            new_G *= a[:, None]
            new_G[:, 0] += mu

            # over-approximate with new predicate beta
            # a*x + mu + b*beta
            B = np.diag(b)
            new_G = np.hstack([new_G, B])

            # no need to copy since new_G is not related to current Zonotope in terms of memory address
            if self.normalized:
                return Zonotope(new_G, copy_=False)
            
            lb = np.hstack([self.l, -np.ones(m)])
            ub = np.hstack([self.u,  np.ones(m)])
            
            # no need to copy since variables are not related to current Zonotope in terms of memory address
            return Zonotope(new_G, lb, ub, copy_=False)

        elif norm == 'unit':
            # a*x + mu
            new_G *= a[:, None]
            new_G[:, 0] += mu - b

            B = np.diag(2.0 * b)
            new_G = np.hstack([new_G, B])

            lb = np.hstack([self.l, np.zeros(m)])
            ub = np.hstack([self.u, np.ones(m)])
            
            # no need to copy since variables are not related to current Zonotope in terms of memory address
            return Zonotope(new_G, lb, ub, copy_=False)

        elif norm == 'un-norm':
            # a*x + mu
            new_G *= a[:, None]
            new_G[:, 0] += mu - b * sum_ / diff_

            B = np.diag(2.0*b / diff_)
            new_G = np.hstack([new_G, B])

            lb = np.hstack([self.l, l])
            ub = np.hstack([self.u, u])
            
            # no need to copy since variables are not related to current Zonotope in terms of memory address
            return Zonotope(new_G, lb, ub, copy_=False)

    def logsig(self, method='tangent', norm='un-norm'):
        """LogSig (Sigmoid) reachability
        
            Args:
                @method: over approximation method option; ['tangent', 'opt']
                @norm: the new beta is in 
                    'unit': [0, 1]
                    'norm': [-1, 1]
                    'un-norm': [l_i, u_i]
            over-approximation of logsig (Sigmoid) activation function
        
        """
        # TODO: move to StarV.set.fun.logsig.py for Zonotope
        
        assert norm in ['unit', 'norm', 'un-norm'], f'unknown normalization method, should be \'unit\', \'norm\', or \'un-norm\'; but received {type(norm)}'
        normalize = norm == 'norm' or self.normalized

        dtype = self.G.dtype

        l, u = self.getRanges()
        fl, fu = LogSig.f(l), LogSig.f(u)
        dl, du = LogSig.df(l), LogSig.df(u)

        m = l.shape[0]
        new_G = copy.deepcopy(self.G)
        
        # y = a*x + mu + b*beta
        if method == 'tangent': # outputs the tightest bounds
            a = np.minimum(dl, du)
        
        elif method == 'opt':
            xou = LogSig.optimal_iter_approx_upper(l, u, iter=5)
            xol = LogSig.optimal_iter_approx_lower(l, u, iter=5)
            dxou = LogSig.df(xou)
            dxol = LogSig.df(xol)
            a = np.minimum(dxou, dxol)

        else:
            raise Exception('unknown tansig over-approximate method for Zonotope')
        
        diff_ = u - l
        sum_ = u + l
        mu = 0.5*(fu + fl - a*sum_)
        b  = 0.5*(fu - fl - a*diff_)

        if normalize:
            # a*x + mu
            new_G *= a[:, None]
            new_G[:, 0] += mu

            # over-approximate with new predicate beta
            # a*x + mu + b*beta
            B = np.diag(b)
            new_G = np.hstack([new_G, B])

            # no need to copy since new_G is not related to current Zonotope in terms of memory address
            if self.normalized:
                return Zonotope(new_G, copy_=False)
            
            lb = np.hstack([self.l, -np.ones(m, dtype=dtype)])
            ub = np.hstack([self.u,  np.ones(m, dtype=dtype)])
            
            # no need to copy since variables are not related to current Zonotope in terms of memory address
            return Zonotope(new_G, lb, ub, copy_=False)

        elif norm == 'unit':
            # a*x + mu
            new_G *= a[:, None]
            new_G[:, 0] += mu - b

            B = np.diag(2.0 * b)
            new_G = np.hstack([new_G, B])

            lb = np.hstack([self.l, np.zeros(m, dtype=dtype)])
            ub = np.hstack([self.u, np.ones(m, dtype=dtype)])
            
            # no need to copy since variables are not related to current Zonotope in terms of memory address
            return Zonotope(new_G, lb, ub, copy_=False)

        elif norm == 'un-norm':
            # a*x + mu
            new_G *= a[:, None]
            new_G[:, 0] += mu - b * sum_ / diff_

            B = np.diag(2.0*b / diff_)
            new_G = np.hstack([new_G, B])

            lb = np.hstack([self.l, l])
            ub = np.hstack([self.u, u])
            
            # no need to copy since variables are not related to current Zonotope in terms of memory address
            return Zonotope(new_G, lb, ub, copy_=False)
    
    # def invsqrt(self, method='chord', norm='un-norm'):
    #     """Inverse Square root function of Zonotope, i.e., invsqrt(x), x \in Zonotope"""
        
    #     assert norm in ['unit', 'norm', 'un-norm'], f'unknown normalization method, should be \'unit\', \'norm\', or \'un-norm\'; but received {type(norm)}'
    #     normalize = norm == 'norm' or self.normalized

    #     dtype = self.G.dtype

    #     # compute state bounds
    #     l, u = self.getRanges()
    #     assert np.any(l > 0), 'error: Out of domain, x in (0, inf]'
    #     fl, fu = InvSqrt.f(l), InvSqrt.f(u)

    #     m = l.shape[0]
    #     new_G = copy.deepcopy(self.G)
        
    #     # y = a*x + mu + b*beta
    #     if method == 'chord': # outputs the tightest bounds
    #         a = (fu - fl) / (u - l) # slope
     
    #     b = fl - a*l # intercept
    #     new_G *= a[:, None]
        

    #     x_opt = ((-1.0/(2.0*a))**(2.0/3.0))
    #     # x_opt = (2.0*a)**(-2.0/3.0)
    #     x_opt = np.clip(x_opt, l, u)
    #     delta = (a*x_opt + b) - InvSqrt.f(x_opt)

    #     if normalize:
    #         new_G[:, 0] += b - 0.5*delta
    #         # over-approximate with new predicate beta
    #         # a*x + mu + b*beta
    #         B = np.diag(0.5*delta)
    #         new_G = np.hstack([new_G, B])

    #         # no need to copy since new_G is not related to current Zonotope in terms of memory address
    #         if self.normalized:
    #             return Zonotope(new_G, copy_=False)
            
    #         lb = np.hstack([self.l, -np.ones(m, dtype=dtype)])
    #         ub = np.hstack([self.u,  np.ones(m, dtype=dtype)])
            
    #         # no need to copy since variables are not related to current Zonotope in terms of memory address
    #         return Zonotope(new_G, lb, ub, copy_=False)
        
    #     elif norm == 'unit':
    #         new_G[:, 0] += b
    #         # over-approximate with new predicate beta
    #         # a*x + mu + b*beta
    #         B = np.diag(-delta)
    #         new_G = np.hstack([new_G, B])

    #         lb = np.hstack([self.l, np.zeros(m, dtype=dtype)])
    #         ub = np.hstack([self.u, np.ones(m, dtype=dtype)])
            
    #         # no need to copy since variables are not related to current Zonotope in terms of memory address
    #         return Zonotope(new_G, lb, ub, copy_=False)

    #     elif norm == 'un-norm':
    #         new_G[:, 0] += b
    #         # w_ = u - l
    #         # g = 2.0 * delta / w_
    #         # new_G[:, 0] -= delta * (u + l) / w_

    #         new_G[:, 0] += delta * l / (u - l)

    #         B = np.diag(-delta / (u - l))
    #         new_G = np.hstack([new_G, B])

    #         lb = np.hstack([self.l, l])
    #         ub = np.hstack([self.u, u])
            
    #         # no need to copy since variables are not related to current Zonotope in terms of memory address
    #         return Zonotope(new_G, lb, ub, copy_=False)


    # def invsqrt_(self, method='chord', norm='un-norm'):
    #     """Inverse Square root function of Zonotope, i.e., invsqrt(x), x \in Zonotope"""
        
    #     assert norm in ['unit', 'norm', 'un-norm'], f'unknown normalization method, should be \'unit\', \'norm\', or \'un-norm\'; but received {type(norm)}'
    #     normalize = norm == 'norm' or self.normalized

    #     dtype = self.G.dtype

    #     # compute state bounds
    #     l, u = self.getRanges()
    #     assert np.any(l > 0), 'error: Out of domain, x in (0, inf]'
    #     fl, fu = InvSqrt.f(l), InvSqrt.f(u)

    #     m = l.shape[0]
    #     new_G = copy.deepcopy(self.G)
        
    #     # y = a*x + mu + b*beta
    #     if method == 'chord': # outputs the tightest bounds
    #         a = (fu - fl) / (u - l)
     
    #     # b = fl - a*l
    #     # new_G *= a[:, None]
    #     # new_G[:, 0] += b

    #     # # x* = f'(a)
    #     # # xo = 0.5*(u + l)
    #     # # dyo = InvSqrt.df(xo)
    #     # # delta = -dyo*(b - xo) - InvSqrt.f(xo)
    #     # x_opt = InvSqrt.df(a)
    #     # x_opt = np.clip(x_opt, l, u)
    #     # # mid = 0.5*(u + l)
    #     # # x_opt  = InvSqrt.df(mid)
    #     # delta = (a*x_opt + b) - InvSqrt.f(x_opt)

    #     mid = 0.5*(u - l)
    #     fmid = InvSqrt.f(mid)
    #     diff_ = mid - l
    #     sum_ = mid + l
    #     mu = 0.5*(fmid + fl - a*sum_)
    #     b  = 0.5*(fmid - fl - a*diff_)

    #     if normalize:
    #         # a*x + mu
    #         new_G *= a[:, None]
    #         new_G[:, 0] += mu

    #         # over-approximate with new predicate beta
    #         # a*x + mu + b*beta
    #         B = np.diag(b)
    #         new_G = np.hstack([new_G, B])

    #         # no need to copy since new_G is not related to current Zonotope in terms of memory address
    #         if self.normalized:
    #             return Zonotope(new_G, copy_=False)
            
    #         lb = np.hstack([self.l, -np.ones(m, dtype=dtype)])
    #         ub = np.hstack([self.u,  np.ones(m, dtype=dtype)])
            
    #         # no need to copy since variables are not related to current Zonotope in terms of memory address
    #         return Zonotope(new_G, lb, ub, copy_=False)

    #     elif norm == 'unit':
    #         # a*x + mu
    #         new_G *= a[:, None]
    #         new_G[:, 0] += mu - b

    #         B = np.diag(2.0 * b)
    #         new_G = np.hstack([new_G, B])

    #         lb = np.hstack([self.l, np.zeros(m, dtype=dtype)])
    #         ub = np.hstack([self.u, np.ones(m, dtype=dtype)])
            
    #         # no need to copy since variables are not related to current Zonotope in terms of memory address
    #         return Zonotope(new_G, lb, ub, copy_=False)

    #     elif norm == 'un-norm':
    #         # a*x + mu
    #         new_G *= a[:, None]
    #         new_G[:, 0] += mu - b * sum_ / diff_

    #         B = np.diag(2.0 * b / diff_)
    #         new_G = np.hstack([new_G, B])

    #         lb = np.hstack([self.l, l])
    #         ub = np.hstack([self.u, u])
            
    #         # no need to copy since variables are not related to current Zonotope in terms of memory address
    #         return Zonotope(new_G, lb, ub, copy_=False)
    
    def conv2d(self, W, b, stride, padding, dilation):
        """2D convolution for Zonotope set
        
            Args:
                W: convolutional filter weights in (KH, KW, Ci, Co) shape
                b: convolutional filter bias in (Co,) shape
                stride: convolutional stride
                padding: convolutional padding

            Returns:
                Zonotope after 2D convolution
        """
        assert isinstance(W, np.ndarray), 'error: ' + \
        'the convolutional filter weights W should be a numpy array'
        assert isinstance(b, np.ndarray), 'error: ' + \
        'the convolutional filter bias b should be a numpy array'
        assert len(W.shape) == 4, 'error: ' + \
        'the convolutional filter weights W should be a 4D numpy array'
        assert len(b.shape) == 1, 'error: ' + \
        'the convolutional filter bias b should be a 1D numpy array'
        assert W.shape[3] == b.shape[0], 'error: ' + \
        'inconsistency between the convolutional filter weights W and bias b'
        assert len(self.shape) == 3, 'error: ' + \
        'the Zonotope shape should be in (H, W, C) format'
        assert self.shape[2] == W.shape[2], 'error: ' + \
        'inconsistency between the Zonotope channel shape and input channel shape of convolutional filter weights W'

        m = self.nVars
        G = self.G.reshape(self.shape + (m+1,)).transpose([3, 2, 0, 1])  # (H, W, Ci, m+1) -> (m+1, Ci, H, W)
        
        layer = torch.nn.Conv2d(in_channels=W.shape[2],
                                out_channels=W.shape[3],
                                kernel_size=(W.shape[:2]),
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=True)
        
        # W in (H, W, Ci, Co) shape -> (Co, Ci, H, W) shape for torch Conv2d
        layer.weight.data = torch.from_numpy(W.transpose([3, 2, 0, 1]))
        layer.bias = torch.nn.Parameter(torch.tensor(b, dtype=torch.float32))

        new_G = layer(torch.from_numpy(G)).detach().numpy()  # (m+1, Co, H_out, W_out)
        new_G = new_G.transpose([2, 3, 1, 0]).reshape(-1, m+1)  # (H_out, W_out, Co, m+1) -> (H_out*W_out*Co, m+1)
        return Zonotope(new_G, self.l, self.u)
    
    def convtrans2d(self, W, b, stride, padding, output_padding, dilation):
        """2D transpose convolution for Zonotope set
        
            Args:
                W: convolutional filter weights in (KH, KW, Ci, Co) shape
                b: convolutional filter bias in (Co,) shape
                stride: convolutional stride
                padding: convolutional padding

            Returns:
                Zonotope after 2D convolution
        """
        assert isinstance(W, np.ndarray), 'error: ' + \
        'the convolutional filter weights W should be a numpy array'
        assert isinstance(b, np.ndarray), 'error: ' + \
        'the convolutional filter bias b should be a numpy array'
        assert len(W.shape) == 4, 'error: ' + \
        'the convolutional filter weights W should be a 4D numpy array'
        assert len(b.shape) == 1, 'error: ' + \
        'the convolutional filter bias b should be a 1D numpy array'
        assert W.shape[3] == b.shape[0], 'error: ' + \
        'inconsistency between the convolutional filter weights W and bias b'
        assert self.shape.ndim == 3, 'error: ' + \
        'the Zonotope shape should be in (H, W, C) format'
        assert self.shape[2] == W.shape[2], 'error: ' + \
        'inconsistency between the Zonotope channel shape and input channel shape of convolutional filter weights W'

        m = self.nVars
        G = self.G.reshape(self.shape + (m+1,)).transpose([3, 2, 0, 1])  # (H, W, Ci, m+1) -> (m+1, Ci, H, W)
        
        layer = torch.nn.ConvTranspose2d(in_channels=W.shape[2],
                                out_channels=W.shape[3],
                                kernel_size=(W.shape[:2]),
                                stride=stride,
                                padding=padding,
                                output_padding=output_padding,
                                dilation=dilation,
                                bias=True)
        
        # W in (H, W, Ci, Co) shape -> (Co, Ci, H, W) shape for torch Conv2d
        layer.weight.data = torch.from_numpy(W.transpose([3, 2, 0, 1]))
        layer.bias = torch.nn.Parameter(torch.tensor(b, dtype=torch.float32))

        new_G = layer(torch.from_numpy(G)).detach().numpy()  # (m+1, Co, H_out, W_out)
        new_G = new_G.transpose([2, 3, 1, 0]).reshape(-1, m+1)  # (H_out, W_out, Co, m+1) -> (H_out*W_out*Co, m+1)
        return Zonotope(new_G, self.l, self.u)
    
    def pad(self, padding, constant_values=0):
        """Pad Zonotope set
        
            Args:
                shape: new shape after padding (H_new, W_new, C)
                padding: tuple of pad widths (top, bottom, left, right)
                constant_values: padding constant value

            Returns:
                Zonotope after padding
        """
        m = self.nVars
        G = self.G.reshape(shape + (m+1,))  # (H, W, C, m+1)
        pad_width = ((padding[0], padding[1]), (padding[2], padding[3]), (0, 0), (0, 0))
        new_G = np.pad(G, pad_width, mode='constant', constant_values=constant_values)
        new_G = new_G.reshape(-1, m+1)
        return Zonotope(new_G, self.l, self.u)
    
    def avgpool2d(self, kernel_size, stride, padding):
        """2D average pooling for Zonotope set
        
            Args:
                shape: input shape (H, W, C)
                kernel_size: pooling kernel size
                stride: pooling stride
                padding: pooling padding

            Returns:
                Zonotope after 2D average pooling
        """
        assert len(self.shape) == 3, 'error: ' + \
        'the Zonotope shape should be in (H, W, C) format'

        m = self.nVars
        G = self.G.reshape(self.shape + (m+1,)).transpose([3, 2, 0, 1])  # (H, W, C, m+1) -> (m+1, C, H, W)
        
        layer = torch.nn.AvgPool2d(kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding)

        new_G = layer(torch.from_numpy(G)).detach().numpy()  # (m+1, C, H_out, W_out)
        new_G = new_G.transpose([2, 3, 1, 0]).reshape(-1, m+1)  # (H_out, W_out, C, m+1) -> (H_out*W_out*C, m+1)
        return Zonotope(new_G, self.l, self.u)
    
    def batchnorm2d(self, gamma, beta, mean, var, eps=1e-5):
        """2D batch normalization for Zonotope set
        
            Args:
                shape: input shape (H, W, C)
                gamma: scale parameter in (C,) shape
                beta: shift parameter in (C,) shape
                mean: running mean in (C,) shape
                var: running variance in (C,) shape
                eps: small constant to avoid division by zero

            Returns:
                Zonotope after 2D batch normalization
        """
        assert len(self.shape) == 3, 'error: ' + \
        'the Zonotope shape should be in (H, W, C) format'

        shape = self.shape
        m = self.nVars
        G = self.G.reshape(shape + (m+1,))  # (H, W, C, m+1)
        
        layer = torch.nn.BatchNorm2d(num_features=shape[2],
                                     eps=eps,
                                     momentum=0.1,
                                     affine=True,
                                     track_running_stats=True)
        
        layer.weight.data = torch.from_numpy(gamma)
        layer.bias.data = torch.from_numpy(beta)
        layer.running_mean = torch.from_numpy(mean)
        layer.running_var = torch.from_numpy(var)

        new_G = layer(torch.from_numpy(G).permute(3, 2, 0, 1)).permute(2, 3, 1, 0).detach().numpy()  # (H, W, C, m+1)
        new_G = new_G.reshape(-1, m+1)
        return Zonotope(new_G, self.l, self.u)
            
    def hadamard_product(self, Z, method='approx'):
        """ 
        Element-wise multiplication
        Let [*] denots hadamard product.
        X * Z = (c_x + V_x * alpha_x) [*] (c_z + V_z * alpha_z)
                = (c_x [*] c_z) +                       #center term
                (c_z [*] V_x * alpha_x) +               #linear x term
                (c_x [*] V_z * alpha_z) +               #linear z term
                (V_x * alpha_x [*] V_z * alpha_z)       #cross xz term
        """
        # TODO: test this function; cross xz term right now is [-1, 1] but it can be in [0, 1]; think about unormalized term
        assert isinstance(Z, Zonotope), 'error: ' + \
        'the input Z should be a Zonotope'

        X = self

        # center term
        c = X.c * Z.c

        # linear x term
        g1 = Z.c[:, None] * X.V
        # linear z term
        g2 = X.c[:, None] * Z.V

        #cross xz term
        n = X.dim
        xm, zm = X.nVars, Z.nVars
        cross_g = X.V[:, :, None] * Z.V[:, None, :] # (n, xm, zm); m is number of generators
        cross_g = cross_g.reshape(n, xm*zm)

        if self.normalized:
            new_G = np.hstack([c, g1, g2, cross_g])
            return Zonotope(new_G, copy_=False)
        
        # for case with upper and lower bounds
        # bilinear cross terms X_alpha_i * Z_alpha_j
        # each X.V[:, i] [*] Z.V[:, j] creates 4 vertices (corner of element-wise product of two bounds)
        p1 = np.outer(X.l, Z.l) # (xm, zm)
        p2 = np.outer(X.l, Z.u)
        p3 = np.outer(X.u, Z.l)
        p4 = np.outer(X.u, Z.u)
        cross_l = np.minimum.reduce([p1, p2, p3, p4]).ravel()  # (xm*zm,)
        cross_u = np.maximum.reduce([p1, p2, p3, p4]).ravel()
        new_l = np.hstack([X.l, Z.l, cross_l])
        new_u = np.hstack([X.u, Z.u, cross_u])
        return Zonotope(new_G, new_l, new_u)
    
    def sample(self, N):
        """
        Sample N points in the feasible Zonotope set.

        Args:
            N (int): Number of samples to generate.

        Returns:
            np.ndarray: Matrix of sampled points (dim x N).
        """
        if N < 1:
            raise ValueError("Number of samples must be at least 1")

        lb, ub = self.getRanges()
        
        # Generate 2N samples initially
        V1 = np.random.uniform(lb[:, np.newaxis], ub[:, np.newaxis], (self.dim, 2*N))
        
        # Filter valid samples
        V = V1[:, [self.contains(v) for v in V1.T]]
        
        # Return N samples (or all if less than N are valid)
        return V[:, :N]
    
    def contains(self, v):
        """ Check if a Zontope set contains a point.
            v: a point in 1D numpy array

            return:
                0 -> a Zonotope set does not contain a point v
                1 -> a Zonotope set does contain a point v    
        """
        assert len(v.shape) == 1, 'error: ' +\
        'invalid point. It should be 1D numpy array'
        assert v.shape[0] == self.dim, 'error: Dimension mismatch'

        f = np.zeros(self.nVars)
        m = gp.Model()
        # prevent optimization information
        m.Params.LogToConsole = 0
        m.Params.OptimalityTol = 1e-6
        if not self.normalized:
            x = m.addMVar(shape=self.nVars, lb=self.l, ub=self.u)
        else:
            x = m.addMVar(shape=self.nVars)
        m.setObjective(f @ x, GRB.MINIMIZE)

        Ae = sp.csr_matrix(self.V)
        be = v - self.c
        m.addConstr(Ae @ x == be)
        m.optimize()

        if m.status == 2:
            return True
        elif m.status == 3:
            return False
        else:
            raise Exception('error: exitflat = %d' % (m.status))

    def sample_vertices(self):
        """
        sample veritices by enumerating 2^m vertices
        returns sample of vertices in (2^m, n) shape
        """
        m = self.nVars
        v = []
        
        Z = self.normalize()
        # cartesian product of np.ones(m) and -np.ones(m)
        for signs in product(*[(-1, 1)]*m):
            v.append(Z.c + np.matmul(Z.V, np.array(signs)))
        return np.vstack(v)

    def project(self, dims):
        """
        project onto a subset of coordinate axes
        Args:
            dims: list, tuple, or np.ndarray
        """
        assert self.G.shape[0] >= len(dims), 'error: '+\
        f'projections dimension should be smaller than the dimension of Zonotope'
        dims = np.array(dims)
        G = np.hstack([self.c[dims, None], self.V[dims, :]])

        if self.normalized:
            return Zonotope(G)
        else:
            return Zonotope(G, self.l, self.u)
        
    # def concatenate(self, Z):
    #     """ Concatenate two Zonotope sets along the dimension axis.
    #         Args:
    #             Z: another Zonotope set to be concatenated

    #         Returns:
    #             A new Zonotope set after concatenation
    #     """
    #     assert isinstance(Z, Zonotope), 'error: ' + \
    #     'the input Z should be a Zonotope'
        
    #     c = np.hstack([self.G[:, 0], Z.G[:, 0]])  # new center
    #     G = block_diag([self.G[:, 1:], Z.G[:, 1:]])

    #     if self.normalized and Z.normalized:
    #         return Zonotope(G)
    #     else:
    #         if self.normalized:
    #             l1 = -np.ones(self.nVars, dtype=self.G.dtype)
    #             u1 =  np.ones(self.nVars, dtype=self.G.dtype)
    #         else:
    #             l1 = self.l
    #             u1 = self.u

    #         if Z.normalized:
    #             l2 = -np.ones(Z.nVars, dtype=self.G.dtype)
    #             u2 =  np.ones(Z.nVars, dtype=self.G.dtype)
    #         else:
    #             l2 = Z.l
    #             u2 = Z.u
            
    #         l = np.hstack([l1, l2])
    #         u = np.hstack([u1, u2])

    #         # shape = self.shape ???? what to do?
    #         return Zonotope(G, l, u, self)
        
    def plot(self, dims=(0,1), ax=None, color='C3', alpha=0.4, edgecolor='k', show=False):
        # project the zonotope onto dims dimensions
        Z = self.project(dims)
        Z = Z.normalize()
        
        # compute convex hull
        verts = Z.sample_vertices() # (2^m, n)
        hull = ConvexHull(verts)
        hull_verts = verts[hull.vertices]

        if ax is None:
            fig, ax = plt.subplots()
        ax.fill(hull_verts[:, 0], hull_verts[:, 1], alpha=alpha, edgecolor=edgecolor)
        ax.scatter(*Z.c, color='r', label='center')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel(f'$x_{{{dims[0]}}}$')
        ax.set_ylabel(f'$x_{{{dims[1]}}}$')
        if show:
            plt.show()
        return ax

    def toStar(self):
        from StarV.set.star import Star
        return Star(self.G, [], [], self.l, self.u)











