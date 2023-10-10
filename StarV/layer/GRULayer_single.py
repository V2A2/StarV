"""
GRU (Gate Recurrent Unit) layer class
Sung Woo Choi, 04/11/2023
"""

# !/usr/bin/python3
import copy
import torch
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.linalg import block_diag
from StarV.fun.logsig import LogSig
from StarV.fun.tansig import TanSig
from StarV.fun.identityXidentity import IdentityXIdentity
from StarV.fun.identityXidentityOLD import IdentityXIdentityOLD
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star


class GRULayer(object):
    """ GRULayer class for qualitative reachability
        Author: Sung Woo Choi
        Date: 04/11/2023
    """
    
    def __init__(self, W, R, w, r, model='pytorch', lp_solver='gurobi', pool=None, RF=0.0, DR=0):

        n = int(W.shape[0]/3)

        if model == 'onnx':
            
            # Wz, Wr, Wc = np.split(W, 3, axis=0)
            # Rz, Rr, Rc = np.split(R, 3, axis=0)
            # wz, wr, wc = np.split(w, 3, axis=0)
            # rz, rr, rc = np.split(r, 3, axis=0)

            # update gate:
            #   z[t] = sigmoid(Wz @ x[t] + wz + Rz @ h[t-1] + rz)
            self.Wz = W[0:n, :] # weight matrix from input state x[t] to update gate z[t]
            self.Rz = R[0:n, :] # weight matrix from memory state h[t-1] to update gate z[t]
            self.wz = w[0:n] # bias vector for update gate
            self.rz = r[0:n]

            # reset gate:
            #   r[t] = sigmoid(Wr * x[t] + wz + Rr * h[t-1] + rr)
            self.Wr = W[n:2*n, :] # weight matrix from intput state x[t] to reset gate r[t]
            self.Rr = R[n:2*n, :] # weight matrix from memory state h[t-1] to reset gate r[t]
            self.wr = w[n:2*n] # bias vector for reset gate
            self.rr = r[n:2*n]

            # cadidate current state:
            #   c[t] = tanh(Wc * x[t] + wc + r[t] o (Rc * h[t-1]  + rc)), where o is
            #       Hadamard product
            self.Wc = W[2*n:3*n, :] # weight matrix from input state x[t] to cadidate current state
            self.Rc = R[2*n:3*n, :] # weight matrix from reset gate r[t] and memory state h[t-1] to 
            #    candidate current state
            self.wc = w[2*n:3*n] # bias vector for candidate state
            self.rc = r[2*n:3*n]

        elif model == 'pytorch':

            # reset gate:
            #   r[t] = sigmoid(Wr * x[t] + wz + Rr * h[t-1] + rr)
            self.Wr = W[0:n, :] # weight matrix from intput state x[t] to reset gate r[t]
            self.Rr = R[0:n, :] # weight matrix from memory state h[t-1] to reset gate r[t]
            self.wr = w[0:n] # bias vector for reset gate
            self.rr = r[0:n]

            # update gate:
            #   z[t] = sigmoid(Wz @ x[t] + wz + Rz @ h[t-1] + rz)
            self.Wz = W[n:2*n, :] # weight matrix from input state x[t] to update gate z[t]
            self.Rz = R[n:2*n, :] # weight matrix from memory state h[t-1] to update gate z[t]
            self.wz = w[n:2*n] # bias vector for update gate
            self.rz = r[n:2*n]

            # cadidate current state:
            #   c[t] = tanh(Wc * x[t] + rc + r[t] o (Rc * h[t-1]  + rc)), where o is
            #       Hadamard product
            self.Wc = W[2*n:3*n, :] # weight matrix from input state x[t] to cadidate current state
            self.Rc = R[2*n:3*n, :] # weight matrix from reset gate r[t] and memory state h[t-1] to 
            #    candidate current state
            self.wc = w[2*n:3*n] # bias vector for candidate state
            self.rc = r[2*n:3*n]

        else:
            raise Exception('error: unsupported network module')


        # output / hidden state:
        # h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]
        #      => sigmoid( ... ) o h[t-1] + (1 - sigmoid( ...)) o tanh( ... )
        #      => ZC + ZH

        # self.in_dim = self.Wz.shape[1] # number of input nodes
        # self.nH = self.Wz.shape[0] # number of nodes at output current state h[t]

        self.pool = pool
        self.lp_solver = lp_solver # lp solver option, 'linprog' or 'glpk'
        self.RF = RF # use only for approx-star method
        self.DR = DR # use for predicate reduction based on the depth of predicate tree

        self.in_dim = self.Wz.shape[1]
        self.out_dim = self.Wz.shape[0]


    def __str__(self):
        print('Wz: \n{}'.format(self.Wz))
        print('Rz: \n{}'.format(self.Rz))
        print('wz: \n{}'.format(self.wz))
        print('rz: \n{}'.format(self.rz))

        print('Wr: \n{}'.format(self.Wr))
        print('Rr: \n{}'.format(self.Rr))
        print('wr: \n{}'.format(self.wr))
        print('rr: \n{}'.format(self.rr))

        print('Wc: \n{}'.format(self.Wc))
        print('Rc: \n{}'.format(self.Rc))
        print('wc: \n{}'.format(self.wc))
        print('rc: \n{}'.format(self.rc))
        return '\n'
    
    def __repr__(self):
        print('Wz: {}'.format(self.Wz.shape))
        print('Rz: {}'.format(self.Rz.shape))
        print('wz: {}'.format(self.wz.shape))
        print('rz: {}'.format(self.rz.shape))

        print('Wr: {}'.format(self.Wr.shape))
        print('Rr: {}'.format(self.Rr.shape))
        print('wr: {}'.format(self.wr.shape))
        print('rr: {}'.format(self.rr.shape))

        print('Wc: {}'.format(self.Wc.shape))
        print('Rc: {}'.format(self.Rc.shape))
        print('wc: {}'.format(self.wc.shape))
        print('rc: {}'.format(self.rc.shape))
        return '\n'
    
    def evaluate(self, x, ho=None,reshape=True):

        assert isinstance(x, np.ndarray), 'error: ' + \
        'x should be an numpy array'
        
        if reshape:
            assert len(x.shape) == 3, 'error: ' + \
            'x should be a 3D numpy array (sequence, batch_size, input_size)'
            s, b, n = x.shape
            x = x.reshape(s, n).T
        else:
            assert len(x.shape) == 2, 'error: ' + \
            'x should be a 2D numpy array (input_size, sequence)'
            n, s = x.shape

        assert n == self.in_dim, 'error: ' + \
        'inconsistent dimension between the input vector and the network input'
        assert s >= 0, 'error: invalid input sequence'

        # update gate:
        z = np.zeros((self.out_dim, s))
        # reset gate:
        r = np.zeros((self.out_dim, s))
        # cadidate current state:
        c = np.zeros((self.out_dim, s))
        # output state
        h = np.zeros((self.out_dim, s))

        WZ = np.matmul(self.Wz, x)
        WR = np.matmul(self.Wr, x)
        WC = np.matmul(self.Wc, x)

        for t in range(s):
            if t == 0:
                z[:, t] = LogSig.f(WZ[:, t] + self.wz + self.rz)
                r[:, t] = LogSig.f(WR[:, t] + self.wr + self.rr)
                c[:, t] = TanSig.f(WC[:, t] + self.wc + r[:, t] * self.rc)
                h[:, t] = (1 - z[:, t]) * c[:, t]

            else:
                z[:, t] = LogSig.f(WZ[:, t] + self.wz + np.matmul(self.Rz, h[:, t-1]) + self.rz)
                r[:, t] = LogSig.f(WR[:, t] + self.wr + np.matmul(self.Rr, h[:, t-1]) + self.rr)
                c[:, t] = TanSig.f(WC[:, t] + self.wc + r[:, t] * (np.matmul(self.Rc, h[:, t-1]) + self.rc))
                h[:, t] = z[:, t] * h[:, t-1] + (1 - z[:, t]) * c[:, t]

        if reshape:
            return h.T.reshape(s, b, n)
        return h
    
    def reach(self, I, method='approx', lp_solver=None, pool=None, RF=None, DR=None, show=False):
        n = len(I)

        Wz = self.Wz
        Rz = self.Rz
        wz = self.wz
        rz = self.rz

        Wr = self.Wr
        Rr = self.Rr
        wr = self.wr
        rr = self.rr

        Wc = self.Wc
        Rc = self.Rc
        wc = self.wc
        rc = self.rc

        if DR is None:
            DR = self.DR
        if RF is None:
            RF = self.RF
        if lp_solver is None:
            lp_solver = self.lp_solver
        if pool is None:
            pool = self.pool
        
        WZ = []
        WR = []
        WC = []
        for i in range(n):
            assert isinstance(I[i], SparseStar) or isinstance(I[i], Star), 'error: ' + \
            'Star and SparseStar are only supported for GRULayer reachability'

            WZ.append(I[i].affineMap(Wz, wz+rz))
            WR.append(I[i].affineMap(Wr, wr+rr))
            WC.append(I[i].affineMap(Wc, wc))

        H1 = []
        for t in range(n):
            if show:
                print('\n (GRU) t: %d\n' % t)
            if t == 0:
                """
                    z[t] = sigmoid(Wz * x[t] + wz + rz) => Zt
                    r[t] = sigmoid(Wr * x[t] + wr + rr) => Rt
                    c[t] = tanh(Wc * x[t] + wc + r[t] o rc)
                         => tansig(WC + Rt o rc)
                         = tansig(WC + Rtrc)
                         = tansig(T) = Ct
                    h[t] = (1 - z[t]) o c[t]
                         => (1 - Zt) o Ct
                         = InZt o Ct
                """
                
                Zt = LogSig.reach(WZ[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                Rt = LogSig.reach(WR[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                Rtrc = Rt.affineMap(np.diag(rc))
                T = WC[t].minKowskiSum(Rtrc)
                Ct = TanSig.reach(T, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                
                InZt = Zt.affineMap(-np.eye(Zt.dim), np.ones(Zt.dim))
                H1.append(IdentityXIdentity.reach(InZt, Ct, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR))

            else:
                """
                    z[t] = sigmoid(Wz * x[t] + wz + Rz * h[t-1] + rz)
                         => sigmoid(WZ + RZ) = Zt
                    r[t] = sigmoid(Wr * x[t] + wr + Rr * h[t-1] + rr)
                         => sigmoid(WR + RR) = sigmoid(WRr) = Rt
                    c[t] = tanh(Wc * x[t] + wc + r[t] o (Rc * h[t-1]  + rc))
                         => tanh(WC + Rt o RC)
                         = tanh(WC + IdentityXIdentity(Rt, RC))
                         = tanh(WRc) = Ct1
                    h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]
                         => IdentityXIdentity(Zt, H[t-1]) + IdentityXIdentity(1-Zt, Ct)
                         = ZtHt_1 + ZtCt = H1[t]
                """
                
                Ht_1 = H1[t-1]

                RZ = Ht_1.affineMap(Rz)
                WRz = RZ.minKowskiSum(WZ[t])
                Zt = LogSig.reach(WRz, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                RR = Ht_1.affineMap(Rr)
                WRr = RR.minKowskiSum(WR[t])
                Rt = LogSig.reach(WRr, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                RC = Ht_1.affineMap(Rc, rc)
                RtUC = IdentityXIdentity.reach(Rt, RC, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                WRc = RtUC.minKowskiSum(WC[t])
                Ct1 = TanSig.reach(WRc, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                
                ZtHt_1 = IdentityXIdentity.reach(Zt, Ht_1, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                InZt = Zt.affineMap(-np.eye(Zt.dim), np.ones(Zt.dim))
                ZtCt = IdentityXIdentity.reach(InZt, Ct1, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                H1.append(ZtHt_1.minKowskiSum(ZtCt))
        return H1
    

    def get_weights(self, module='default'):

        if module == 'default':
            return self.Wz, self.Rz, self.wz, self.rz, \
                self.Wr, self.Rr, self.wr, self.rr, \
                self.Wc, self.Rc, self.wc, self.rc
            
        elif module == 'pytorch':
            # In PyTorch,
            # i is input, h is recurrent
            # r, z, n are the reset, update, and new gates, respectively
            # weight_ih_l0: (W_ir|W_iz|W_in)
            weight_ih_l0 = torch.nn.Parameter(torch.from_numpy(np.vstack((self.Wr, self.Wz, self.Wc))))
            # weight_hh_l0: (W_hr|W_hz|W_hn)
            weight_hh_l0 = torch.nn.Parameter(torch.from_numpy(np.vstack((self.Rr, self.Rz, self.Rc))))
            # bias_ih_l0: (b_ir|b_iz|b_in)
            bias_ih_l0 = torch.nn.Parameter(torch.from_numpy(np.hstack((self.wr, self.wz, self.wc))))
            # bias_hh_l0: (b_hr|b_hz|b_hn)
            bias_hh_l0 = torch.nn.Parameter(torch.from_numpy(np.hstack((self.rr, self.rz, self.rc))))
            return weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0

        elif module == 'onnx':
            # In Onnx,
            # W: (Wz | Wr | Wc), parameter weight matrix for update, reset, and hidden gates, respectively
            W = np.vstack((self.Wz, self.Wr, self.Wc))
            # R: (Rz | Rr | Rc), recurrence weight matrix for update, reset, and hidden gates, respectively
            R = np.hstack((self.Rz, self.Rr, self.Rc))
            b = np.hstack((self.wz, self.wr, self.wc, \
                           self.rz, self.rr, self.rc))
            return W, R, b
    
        else:
            raise Exception('error: unsupported network module')
        

    @staticmethod
    def rand(out_dim, in_dim):
        """ Randomly generate a GRULayer
            Args:
                @out_dim: number of hidden units
                @in_dim: number of inputs (sequence, batch_size, input_size)
        """

        assert out_dim > 0 and in_dim > 0, 'error: invalid number of hidden units or inputs'

        W = np.random.rand(out_dim*3, in_dim)
        R = np.random.rand(out_dim*3, out_dim)
        w = np.random.rand(out_dim*3)
        r = np.random.rand(out_dim*3)
        return GRULayer(W=W, R=R, w=w, r=r)