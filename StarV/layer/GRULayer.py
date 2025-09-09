#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
##                                                                     ##
##   Primary contacts: Hoang Dung Tran <dungtran@ufl.edu> (UF)         ##
##                     Sung Woo Choi <sungwoo.choi@ufl.edu> (UF)       ##
##                     Yuntao Li <yli17@ufl.edu> (UF)                  ##
##                     Qing Liu <qliu1@ufl.edu> (UF)                   ##
##                                                                     ##
##   See CONTRIBUTORS for full author contacts and affiliations.       ##
##   This program is licensed under the BSD 3‑Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
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
from StarV.fun.logsigXidentity import LogsigXIdentity
from StarV.fun.omlogsigXtansig import OmLogsigXTansig
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star


class GRULayer(object):
    """ GRULayer class for qualitative reachability
        Author: Sung Woo Choi
        Date: 04/11/2023
    """

    def __init__(self, layer, output_mode='many', module='default', lp_solver='gurobi', pool=None, RF=0.0, DR=0, dtype='float64'):

        if module == 'default':
            '''
                layer = [L_0, L_1, ..., L_n], where
                L_0 = [W, R, b].
                
                W is weight matrices for update, reset, and new gates; z_t, r_t, c_t                 
                R is recurrent weight matrices for update, reset, and new gates; z_t, r_t, c_t 
                b is bias vectors of both input and reccurent
            '''
            assert isinstance(layer, list), 'error: provided layer is not a list'

            num_layers = len(layer)

            # input weight matrix
            self.Wz, self.Wr, self.Wc = [], [], []
            # reccurent weight matrix
            self.Rz, self.Rr, self.Rc = [], [], []
            # input bias vector
            self.wz, self.wr, self.wc = [], [], []
            # recurrent bias vector
            self.rz, self.rr, self.rc = [], [], []

            for i in range(num_layers):
                W, R, b = layer[i]
                Wz, Wr, Wc = np.split(W, 3, axis=0)
                self.Wz.append(Wz)
                self.Wr.append(Wr)
                self.Wc.append(Wc)

                Rz, Rr, Rc = np.split(R, 3, axis=0)
                self.Rz.append(Rz)
                self.Rr.append(Rr)
                self.Rc.append(Rc)

                wz, wr, wc, rz, rr, rc = np.split(b, 6)
                self.wz.append(wz)
                self.wr.append(wr)
                self.wc.append(wc)
                self.rz.append(rz)
                self.rr.append(rr)
                self.rc.append(rc)

            self.in_dim = self.Wz[0].shape[1]
            self.out_dim = self.Wz[num_layers-1].shape[0]
            self.num_layers = len(self.Wz)
            self.bias = True
            self.batch_first = True

        elif module == 'pytorch':
            assert isinstance(layer, torch.nn.GRU), 'error: provided layer is not torch.nn.GRU layer'

            # input weight matrix
            self.Wr, self.Wz , self.Wc = [], [], []
            # recurrent weight matrix
            self.Rr, self.Rz, self.Rc = [], [], []
            # input bias vector
            self.wr, self.wz, self.wc = [], [], []
            # recurrent bias vector
            self.rr, self.rz, self.rc = [], [], []

            for ln in range(layer.num_layers):
                Wr, Wz, Wc = torch.split(getattr(layer, f"weight_ih_l{ln}"), layer.hidden_size, 0)
                self.Wr.append(Wr.detach().numpy().astype(dtype))
                self.Wz.append(Wz.detach().numpy().astype(dtype))
                self.Wc.append(Wc.detach().numpy().astype(dtype))

                Rr, Rz, Rc = torch.split(getattr(layer, f"weight_hh_l{ln}"), layer.hidden_size, 0)
                self.Rr.append(Rr.detach().numpy().astype(dtype))
                self.Rz.append(Rz.detach().numpy().astype(dtype))
                self.Rc.append(Rc.detach().numpy().astype(dtype))

                wr, wz, wc = torch.split(getattr(layer, f"bias_ih_l{ln}"), layer.hidden_size, 0)
                self.wr.append(wr.detach().numpy().astype(dtype))
                self.wz.append(wz.detach().numpy().astype(dtype))
                self.wc.append(wc.detach().numpy().astype(dtype))

                rr, rz, rc = torch.split(getattr(layer, f"bias_hh_l{ln}"), layer.hidden_size, 0)
                self.rr.append(rr.detach().numpy().astype(dtype))
                self.rz.append(rz.detach().numpy().astype(dtype))
                self.rc.append(rc.detach().numpy().astype(dtype))

            self.in_dim = layer.input_size
            self.out_dim = layer.hidden_size
            self.num_layers = layer.num_layers
            self.bias = layer.bias
            self.batch_first = layer.batch_first

        else:
            raise Exception('error: unsupported neural network module')
        
        self.output_mode = output_mode
        self.pool = pool
        self.lp_solver = lp_solver  # lp solver option, 'linprog' or 'glpk'
        self.RF = RF  # use only for approx-star method
        self.DR = DR  # use for predicate reduction based on the depth of predicate tree


    def info(self):
        print('input dimension (input_size): {}'.format(self.in_dim))
        print('output dimension (hidden_size): {}'.format(self.in_dim))
        print('number of layers: {}'.format(self.num_layers))
        print('output mode: {}'.format(self.output_mode))
        print('bias: {}'.format(self.bias))
        print('batch_first: {}'.format(self.batch_first))


    def __str__(self):
        for ln in range(self.num_layers):
            print('(GRU) {} layer'.format(ln))
            print('Wz: \n{}'.format(self.Wz[ln]))
            print('Rz: \n{}'.format(self.Rz[ln]))
            print('wz: \n{}'.format(self.wz[ln]))
            print('rz: \n{}'.format(self.rz[ln]))

            print('Wr: \n{}'.format(self.Wr[ln]))
            print('Rr: \n{}'.format(self.Rr[ln]))
            print('wr: \n{}'.format(self.wr[ln]))
            print('rr: \n{}'.format(self.rr[ln]))

            print('Wc: \n{}'.format(self.Wc[ln]))
            print('Rc: \n{}'.format(self.Rc[ln]))
            print('wc: \n{}'.format(self.wc[ln]))
            print('rc: \n{}'.format(self.rc[ln]))
        return '\n'
    

    def __repr__(self):
        for ln in range(self.num_layers):
            print('(GRU) {} layer'.format(ln))
            print('Wz: {}'.format(self.Wz[ln].shape))
            print('Rz: {}'.format(self.Rz[ln].shape))
            print('wz: {}'.format(self.wz[ln].shape))
            print('rz: {}'.format(self.rz[ln].shape))

            print('Wr: {}'.format(self.Wr[ln].shape))
            print('Rr: {}'.format(self.Rr[ln].shape))
            print('wr: {}'.format(self.wr[ln].shape))
            print('rr: {}'.format(self.rr[ln].shape))

            print('Wc: {}'.format(self.Wc[ln].shape))
            print('Rc: {}'.format(self.Rc[ln].shape))
            print('wc: {}'.format(self.wc[ln].shape))
            print('rc: {}'.format(self.rc[ln].shape))
        return '\n'
    
    
    def evaluate(self, x, h0=None,reshape=True):
        """
            update gate:
            z[t] = sigmoid(Wz @ x[t] + wz + Rz @ h[t-1] + rz)
            reset gate:
            r[t] = sigmoid(Wr * x[t] + wz + Rr * h[t-1] + rr)
            cadidate current state:
            c[t] = tanh(Wc * x[t] + wc + r[t] o (Rc * h[t-1]  + rc)), where o is
                Hadamard product
            output / hidden state:
            h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]
        """
        
        assert isinstance(x, np.ndarray), 'error: ' + \
        'x should be an numpy array'

        if h0 is not None:
            assert isinstance(h0, np.ndarray), 'error: ' + \
            'h0 should be an numpy array'

        x_shape = len(x.shape)
        if x_shape == 3:
            if self.batch_first:
                batch_size, sequence, input_size = x.shape
                ''' transpose to (batch_size, input_size, sequence) '''
                x = np.transpose(x, (0, 2, 1))

            else:
                sequence, batch_size, input_size = x.shape
                ''' transpose to (batch_size, input_size, sequence) '''
                x = np.transpose(x, (1, 2, 0))

        elif x_shape == 2:
            ''' (input_size, sequence) '''
            input_size, sequence = x.shape
            batch_size = 1
            x = x[np.newaxis, :]
        
        else:
            raise Exception('samples, x, should be 2D or 3D numpy array')

        assert input_size == self.in_dim, 'error: ' + \
            'inconsistent dimension between the input vector and the network input'
        assert sequence >= 0, 'error: invalid input sequence'


        for l in range(self.num_layers):
            if l > 0:
                hprev = copy.deepcopy(h)
            
            # hidden state:
            h = np.zeros((batch_size, self.out_dim, sequence))

            bz = self.wz[l] + self.rz[l]
            br = self.wr[l] + self.rr[l]

            for b in range(batch_size):

                if l == 0:
                    xl = x[b, : , :]
                else:
                    xl = hprev[b, : , :]

                WZ = np.matmul(self.Wz[l], xl) + bz[:, np.newaxis]
                WR = np.matmul(self.Wr[l], xl) + br[:, np.newaxis]
                WC = np.matmul(self.Wc[l], xl) + self.wc[l][:, np.newaxis]

                for t in range(sequence):

                    if t == 0:
                        if h0 is None:
                            z = LogSig.f(WZ[:, t])
                            r = LogSig.f(WR[:, t])
                            c = TanSig.f(WC[:, t] + r * self.rc[l])
                            h[b, :, t] = (1 - z) * c

                            continue
                        
                        else:
                            ht_1 = h0[l, b, :]
                
                    else:
                        ht_1 = h[b, :, t-1]

                    z = LogSig.f(WZ[:, t] + np.matmul(self.Rz[l], ht_1))
                    r = LogSig.f(WR[:, t] + np.matmul(self.Rr[l], ht_1))
                    c = TanSig.f(WC[:, t] + r * (np.matmul(self.Rc[l], ht_1) + self.rc[l]))
                    h[b, :, t] = z * ht_1 + (1 - z) * c

        if x_shape == 3:
            if self.batch_first:
                # from (batch_size, input_size, sequence)
                # transpose to (batch_size, sequence, input_size)
                h = np.transpose(h, (0, 2, 1))
                if self.output_mode == 'one':
                    h = h[:, -1, :]
            else:
                # from (batch_size, output_size, sequence)
                # transpose to (sequence, batch_size, output_size)
                h = np.transpose(h, (2, 0, 1))       
                if self.output_mode == 'one':
                    h = h[-1, :, :]

        elif x_shape == 2:
            # from (batch_size, output_size, sequence)
            ''' transpose to (output_size, sequence) '''
            h = h[0, :]
            if self.output_mode == 'one':
                h = h[:, -1]

        return h

    def reachApprox(self, I, H0=None, lp_solver=None, pool=None, RF=None, DR=None, show=False):
        """
            @output_mode: -one: returns the single output reachable set of the last layer
                         -many: returns many output reachable sets of the last layer

            @H0: the initial hidden state for each sets in the input sequence.
        
            ## currently batch is not supported

            z[t] = sigmoid(Wz @ x[t] + wz + Rz @ h[t-1] + rz)
            r[t] = sigmoid(Wr * x[t] + wz + Rr * h[t-1] + rr)
            c[t] = tanh(Wc * x[t] + wc + r[t] o (Rc * h[t-1]  + rc))
            h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]
        """
        
        assert isinstance(I, list), 'error: ' + \
        'the input, I, should be a list of sparse star sets'

        sequence = len(I)

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

        outMode = self.output_mode == 'one'
        num_layers = self.num_layers

        for l in range(num_layers):
            if show:
                    print('(GRU) layer: {}'.format(l))

            WZ = []
            WR = []
            WC = []
            if l == 0:
                Xl = I
            else:
                Xl = H1

            for i in range(sequence):       
                if l == 0:
                    assert isinstance(Xl[i], SparseStar) or isinstance(Xl[i], Star), 'error: ' + \
                    'Star and SparseStar are only supported for GRULayer reachability'

                WZ.append(Xl[i].affineMap(Wz[l], wz[l]+rz[l]))
                WR.append(Xl[i].affineMap(Wr[l], wr[l]+rr[l]))
                WC.append(Xl[i].affineMap(Wc[l], wc[l]))

            H1 = []
            for t in range(sequence):

                if show:
                    print('(GRU) t: {}'.format(t))
                    
                if t == 0:
                    """
                        z[t] = sigmoid(Wz * x[t] + wz + rz) => Zt
                        r[t] = sigmoid(Wr * x[t] + wr + rr) => Rt
                        c[t] = tanh(Wc * x[t] + wc + r[t] o rc)
                            => tansig(WC + Rt o rc)
                            = tansig(WC + Rtrc)
                            = tansig(T) = Ct
                        h[t] = (1 - z[t]) o c[t]
                            => InZt o Ct
                    """

                    if H0 is None:
                        Rt = LogSig.reach(WR[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                        Rtrc = Rt.affineMap(np.diag(rc[l]))
                        Ic = WC[t].minKowskiSum(Rtrc)

                        if outMode and l == num_layers-1: # one output mode
                            H1 = [OmLogsigXTansig.reach(WZ[t], Ic, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)]
                        else: # many output mode
                            H1.append(OmLogsigXTansig.reach(WZ[t], Ic, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR))

                        continue

                    else:
                        Ht_1 = H0

                else:
                    if outMode and l == num_layers-1:
                        Ht_1 = H1[0]
                    else:
                        Ht_1 = H1[t-1]

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
                        => IdentityXIdentity(Zt, H[t-1]) + IdentityXIdentity(InZt, Ct)
                        = ZtHt_1 + ZtCt
                """
                RZ = Ht_1.affineMap(Rz[l])
                Iz = RZ.minKowskiSum(WZ[t])

                RR = Ht_1.affineMap(Rr[l])
                Ir = RR.minKowskiSum(WR[t])

                RC = Ht_1.affineMap(Rc[l], rc[l])
                IrRC = LogsigXIdentity.reach(Ir, RC, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                Ic = IrRC.minKowskiSum(WC[t])

                IzHt_1 = LogsigXIdentity.reach(Iz, Ht_1, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                IzIc = OmLogsigXTansig.reach(Iz, Ic, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                if outMode and l == num_layers-1: # one output mode
                    H1 = [IzHt_1.minKowskiSum(IzIc)]
                else: # many output mode
                    H1.append(IzHt_1.minKowskiSum(IzIc))
        
        return H1
    
    def reachApprox_identity(self, I, H0=None, lp_solver=None, pool=None, RF=None, DR=None, show=False):
        """
            @output_mode: -one: returns the single output reachable set of the last layer
                         -many: returns many output reachable sets of the last layer

            @H0: the initial hidden state for each sets in the input sequence.
        
            ## currently batch is not supported

            z[t] = sigmoid(Wz @ x[t] + wz + Rz @ h[t-1] + rz)
            r[t] = sigmoid(Wr * x[t] + wz + Rr * h[t-1] + rr)
            c[t] = tanh(Wc * x[t] + wc + r[t] o (Rc * h[t-1]  + rc))
            h[t] = z[t] o h[t-1] + (1 - z[t]) o c[t]
        """
        
        assert isinstance(I, list), 'error: ' + \
        'the input, I, should be a list of sparse star sets'

        sequence = len(I)

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

        outMode = self.output_mode == 'one'
        num_layers = self.num_layers

        for l in range(num_layers):
            if show:
                    print('(GRU) layer: {}'.format(l))

            WZ = []
            WR = []
            WC = []
            if l == 0:
                Xl = I
            else:
                Xl = H1

            for i in range(sequence):       
                if l == 0:
                    assert isinstance(Xl[i], SparseStar) or isinstance(Xl[i], Star), 'error: ' + \
                    'Star and SparseStar are only supported for GRULayer reachability'

                WZ.append(Xl[i].affineMap(Wz[l], wz[l]+rz[l]))
                WR.append(Xl[i].affineMap(Wr[l], wr[l]+rr[l]))
                WC.append(Xl[i].affineMap(Wc[l], wc[l]))

            H1 = []
            for t in range(sequence):

                if show:
                    print('(GRU) t: {}'.format(t))
                    
                if t == 0:
                    """
                        z[t] = sigmoid(Wz * x[t] + wz + rz) => Zt
                        r[t] = sigmoid(Wr * x[t] + wr + rr) => Rt
                        c[t] = tanh(Wc * x[t] + wc + r[t] o rc)
                            => tansig(WC + Rt o rc)
                            = tansig(WC + Rtrc)
                            = tansig(T) = Ct
                        h[t] = (1 - z[t]) o c[t]
                            => InZt o Ct
                    """

                    if H0 is None:
                        Zt = LogSig.reach(WZ[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                        Rt = LogSig.reach(WR[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                        Rtrc = Rt.affineMap(np.diag(rc[l]))
                        T = WC[t].minKowskiSum(Rtrc)
                        Ct = TanSig.reach(T, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                        InZt = Zt.affineMap(-np.eye(Zt.dim), np.ones(Zt.dim))
                        if outMode and l == num_layers-1: # one output mode
                            H1 = [IdentityXIdentity.reach(InZt, Ct, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)]
                        else: # many output mode
                            H1.append(IdentityXIdentity.reach(InZt, Ct, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR))
                        
                        continue

                    else:
                        Ht_1 = H0

                else:
                    if outMode and l == num_layers-1:
                        Ht_1 = H1[0]
                    else:
                        Ht_1 = H1[t-1]

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
                        => IdentityXIdentity(Zt, H[t-1]) + IdentityXIdentity(InZt, Ct)
                        = ZtHt_1 + ZtCt
                """

                RZ = Ht_1.affineMap(Rz[l])
                WRz = RZ.minKowskiSum(WZ[t])
                Zt = LogSig.reach(WRz, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                RR = Ht_1.affineMap(Rr[l])
                WRr = RR.minKowskiSum(WR[t])
                Rt = LogSig.reach(WRr, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                RC = Ht_1.affineMap(Rc[l], rc[l])
                RtUC = IdentityXIdentity.reach(Rt, RC, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                WRc = RtUC.minKowskiSum(WC[t])
                Ct1 = TanSig.reach(WRc, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                
                InZt = Zt.affineMap(-np.eye(Zt.dim), np.ones(Zt.dim))
                ZtHt_1 = IdentityXIdentity.reach(Zt, Ht_1, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                ZtCt = IdentityXIdentity.reach(InZt, Ct1, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                if outMode and l == num_layers-1: # one output mode
                    H1 = [ZtHt_1.minKowskiSum(ZtCt)]
                else: # many output mode
                    H1.append(ZtHt_1.minKowskiSum(ZtCt))
        
        return H1
    

    # def get_weights(self, module='default'):

    #     if module == 'default':
    #         return self.Wz, self.Rz, self.wz, self.rz, \
    #             self.Wr, self.Rr, self.wr, self.rr, \
    #             self.Wc, self.Rc, self.wc, self.rc
            
    #     elif module == 'pytorch':
    #         # In PyTorch,
    #         # i is input, h is recurrent
    #         # r, z, n are the reset, update, and new gates, respectively
    #         # weight_ih_l0: (W_ir|W_iz|W_in)
    #         weight_ih_l0 = torch.nn.Parameter(torch.from_numpy(np.vstack((self.Wr, self.Wz, self.Wc))))
    #         # weight_hh_l0: (W_hr|W_hz|W_hn)
    #         weight_hh_l0 = torch.nn.Parameter(torch.from_numpy(np.vstack((self.Rr, self.Rz, self.Rc))))
    #         # bias_ih_l0: (b_ir|b_iz|b_in)
    #         bias_ih_l0 = torch.nn.Parameter(torch.from_numpy(np.hstack((self.wr, self.wz, self.wc))))
    #         # bias_hh_l0: (b_hr|b_hz|b_hn)
    #         bias_hh_l0 = torch.nn.Parameter(torch.from_numpy(np.hstack((self.rr, self.rz, self.rc))))
    #         return weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0

    #     elif module == 'onnx':
    #         # In Onnx,
    #         # W: (Wz | Wr | Wc), parameter weight matrix for update, reset, and hidden gates, respectively
    #         W = np.vstack((self.Wz, self.Wr, self.Wc))
    #         # R: (Rz | Rr | Rc), recurrence weight matrix for update, reset, and hidden gates, respectively
    #         R = np.hstack((self.Rz, self.Rr, self.Rc))
    #         b = np.hstack((self.wz, self.wr, self.wc, \
    #                        self.rz, self.rr, self.rc))
    #         return W, R, b
    
    #     else:
    #         raise Exception('error: unsupported network module')

    def reach_identity(self, In, H0=None, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """main reachability method
            Args:
                @In: an input set (Star, SparseStar, or Probstar)
                @method: reachability method: 'approx' or 'exact'
                @lp_solver: lp solver: 'gurobi' (default), 'glpk', or 'linprog'
                @pool: parallel pool: None or multiprocessing.pool.Pool
                @RF: relax-factor from 0 to 1 (0 by default)
                @DR: depth reduction from 1 to k-Layers (0 by default)
            
            Return:
                @R: a reachable set
        """
        if method == 'exact':
            raise Exception('error: exact method for GRU layer is not supported')
        elif method == 'approx':
            return self.reachApprox_identity(I=In, H0=H0, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
        raise Exception('error: unknown reachability method')
    

    def reach(self, In, H0=None, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """main reachability method
            Args:
                @In: an input set (Star, SparseStar, or Probstar)
                @method: reachability method: 'approx' or 'exact'
                @lp_solver: lp solver: 'gurobi' (default), 'glpk', or 'linprog'
                @pool: parallel pool: None or multiprocessing.pool.Pool
                @RF: relax-factor from 0 to 1 (0 by default)
                @DR: depth reduction from 1 to k-Layers (0 by default)
            
            Return:
                @R: a reachable set
        """
        if method == 'exact':
            raise Exception('error: exact method for GRU layer is not supported')
        elif method == 'approx':
            return self.reachApprox(I=In, H0=H0, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
        raise Exception('error: unknown reachability method')

        
    @staticmethod
    def rand(in_dim, out_dim, num_layers):
        """ Randomly generate a GRULayer
            Args:
                @out_dim: number of hidden units
                @in_dim: number of input units
                @num_layers: number of layers

            layer = [W, R, w, r], where
                W = [Wr, Wz, Wc], a list of input weight matrices
                R = [Rr, Rz, Rc], a list of reccurent weight matrices
                w = [wr, wz, wc], a list of input bias vectors
                r = [rr, rz, rc], a list of recurrent bias vectors

            Wr = [W_ir[0], ... , W_ir[l]]
            ...
            Wc = [W_ic[0], ... , W_ic[l]]

            wr = [b_ir[0], ... , b_ir[l]]
            ...
            wc = [b_ic[0], ... , b_ic[l]]
            
            similiar logic applies to reccurent side
        """

        assert out_dim > 0, 'error: invalid number of hidden units'
        assert in_dim > 0, 'error: invalid number of input units'
        assert num_layers > 0, 'error: invalid number of layers'

        Wr, Wz, Wc = [], [], []
        Rr, Rz, Rc = [], [], []
        wr, wz, wc = [], [], []
        rr, rz, rc = [], [], []

        for l in range(num_layers):
            Wr.append(np.random.rand(out_dim, in_dim))
            Wz.append(np.random.rand(out_dim, in_dim))
            Wc.append(np.random.rand(out_dim, in_dim))
            
            Rr.append(np.random.rand(out_dim, in_dim))
            Rz.append(np.random.rand(out_dim, in_dim))
            Rc.append(np.random.rand(out_dim, in_dim))

            wr.append(np.random.rand(out_dim))
            wz.append(np.random.rand(out_dim))
            wc.append(np.random.rand(out_dim))

            rr.append(np.random.rand(out_dim))
            rz.append(np.random.rand(out_dim))
            rc.append(np.random.rand(out_dim))
        
        W = [Wr, Wz, Wc]
        R = [Rr, Rz, Rc]
        w = [wr, wz, wc]
        r = [rr, rz, rc]
        return GRULayer([W, R, w, r], module='default')