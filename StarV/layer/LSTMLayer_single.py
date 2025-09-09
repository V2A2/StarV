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
LSTM (Long Short Term-Memory) layer class
Sung Woo Choi, 06/12/2023
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
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star


class LSTMLayer_single(object):
    """ LSTMLayer class for qualitative reachability
        Author: Sung Woo Choi
        Date: 06/12/2023
    """

    # def __init__(self, layer, module='pytorch', lp_solver='gurobi', pool=None, RF=0.0, DR=0):

    #     if module == 'pytorch':
    #         assert isinstance(
    #             layer, torch.nn.LSTM), 'error: provided layer is not torch.nn.LSTM layer'

    #         self.in_dim = layer.input_size
    #         self.out_dim = layer.hidden_size
    #         # self.input_size = layer.input_size
    #         # self.hidden_size = layer.hidden_size
    #         self.num_layers = layer.num_layers
    #         self.bias = layer.bias
    #         self.batch_first = layer.batch_first

    #         self.pool = pool
    #         self.lp_solver = lp_solver  # lp solver option, 'linprog' or 'glpk'
    #         self.RF = RF  # use only for approx-star method
    #         self.DR = DR  # use for predicate reduction based on the depth of predicate tree

    #         # input weight matrix
    #         self.Wi = [np.empty(0) for _ in range(self.num_layers)]
    #         self.Wf = [np.empty(0) for _ in range(self.num_layers)]
    #         self.Wg = [np.empty(0) for _ in range(self.num_layers)]
    #         self.Wo = [np.empty(0) for _ in range(self.num_layers)]
    #         # recurrent weight matrix
    #         self.Ri = [np.empty(0) for _ in range(self.num_layers)]
    #         self.Rf = [np.empty(0) for _ in range(self.num_layers)]
    #         self.Rg = [np.empty(0) for _ in range(self.num_layers)]
    #         self.Ro = [np.empty(0) for _ in range(self.num_layers)]
    #         # input bias vector
    #         self.wi = [np.empty(0) for _ in range(self.num_layers)]
    #         self.wf = [np.empty(0) for _ in range(self.num_layers)]
    #         self.wg = [np.empty(0) for _ in range(self.num_layers)]
    #         self.wo = [np.empty(0) for _ in range(self.num_layers)]
    #         # recurrent bias vector
    #         self.ri = [np.empty(0) for _ in range(self.num_layers)]
    #         self.rf = [np.empty(0) for _ in range(self.num_layers)]
    #         self.rg = [np.empty(0) for _ in range(self.num_layers)]
    #         self.ro = [np.empty(0) for _ in range(self.num_layers)]

    #         for ln in range(self.num_layers):
    #             self.Wi[ln], self.Wf[ln], self.Wg[ln], self.Wo[ln] = torch.split(
    #                 getattr(layer, f"weight_ih_l{ln}"), layer.hidden_size, 0)
    #             self.Ri[ln], self.Rf[ln], self.Rg[ln], self.Ro[ln] = torch.split(
    #                 getattr(layer, f"weight_hh_l{ln}"), layer.hidden_size, 0)
    #             self.wi[ln], self.wf[ln], self.wg[ln], self.wo[ln] = torch.split(
    #                 getattr(layer, f"bias_ih_l{ln}"), layer.hidden_size, 0)
    #             self.ri[ln], self.rf[ln], self.rg[ln], self.ro[ln] = torch.split(
    #                 getattr(layer, f"bias_ih_l{ln}"), layer.hidden_size, 0)

    #     else:
    #         raise Exception('error: unsupported neural network module')

    def __init__(self, W, R, w, r, model='pytorch', lp_solver='gurobi', pool=None, RF=0.0, DR=0):

        n = int(W.shape[0]/4)

        if model == 'onnx':
            # input gate:
            #   i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
            #
            # forgate gate:
            #   f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)

            # cell gate:
            #   g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)

            # output gate:
            #   o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)
            pass

        elif model == 'pytorch':
            # input gate:
            #   i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
            self.Wi = W[0:n, :]
            self.Ri = R[0:n, :]
            self.wi = w[0:n]
            self.ri = r[0:n]

            # forgate gate:
            #   f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)
            self.Wf = W[n:2*n, :]
            self.Rf = R[n:2*n, :]
            self.wf = w[n:2*n]
            self.rf = r[n:2*n]

            # cell gate:
            #   g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)
            self.Wg = W[2*n:3*n, :]
            self.Rg = R[2*n:3*n, :]
            self.wg = w[2*n:3*n]
            self.rg = r[2*n:3*n]

            # output gate:
            #   o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)
            self.Wo = W[3*n:4*n, :]
            self.Ro = R[3*n:4*n, :]
            self.wo = w[3*n:4*n]
            self.ro = r[3*n:4*n]

        else:
            raise Exception('error: unsupported network module')

        # cell state:
        #   c[t] = f[t] o c[t-1] + i[t] o g[t], where o is Hadamard product

        # hidden state:
        #   h[t] = o[t] o tanh(c[t])

        self.pool = pool
        self.lp_solver = lp_solver  # lp solver option, 'linprog' or 'glpk'
        self.RF = RF  # use only for approx-star method
        self.DR = DR  # use for predicate reduction based on the depth of predicate tree

        self.in_dim = self.Wi.shape[1]
        self.out_dim = self.Wo.shape[0]

    def __str__(self):
        print('Wi: \n{}'.format(self.Wi))
        print('Ri: \n{}'.format(self.Ri))
        print('wi: \n{}'.format(self.wi))
        print('ri: \n{}'.format(self.ri))

        print('Wf: \n{}'.format(self.Wf))
        print('Rf: \n{}'.format(self.Rf))
        print('wf: \n{}'.format(self.wf))
        print('rf: \n{}'.format(self.rf))

        print('Wg: \n{}'.format(self.Wg))
        print('Rg: \n{}'.format(self.Rg))
        print('wg: \n{}'.format(self.wg))
        print('rg: \n{}'.format(self.rg))

        print('Wo: \n{}'.format(self.Wo))
        print('Ro: \n{}'.format(self.Ro))
        print('wo: \n{}'.format(self.wo))
        print('ro: \n{}'.format(self.ro))
        return '\n'

    def __repr__(self):
        print('Wi: {}'.format(self.Wi.shape))
        print('Ri: {}'.format(self.Ri.shape))
        print('wi: {}'.format(self.wi.shape))
        print('ri: {}'.format(self.ri.shape))

        print('Wf: {}'.format(self.Wf.shape))
        print('Rf: {}'.format(self.Rf.shape))
        print('wf: {}'.format(self.wf.shape))
        print('rf: {}'.format(self.rf.shape))

        print('Wg: {}'.format(self.Wg.shape))
        print('Rg: {}'.format(self.Rg.shape))
        print('wg: {}'.format(self.wg.shape))
        print('rg: {}'.format(self.rg.shape))

        print('Wo: {}'.format(self.Wo.shape))
        print('Ro: {}'.format(self.Ro.shape))
        print('wo: {}'.format(self.wo.shape))
        print('ro: {}'.format(self.ro.shape))
        return '\n'

    def evaluate(self, x, ho=None, reshape=True):

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

        # input gate:
        i = np.zeros((self.out_dim, s))
        # forgate gate:
        f = np.zeros((self.out_dim, s))
        # cell gate:
        g = np.zeros((self.out_dim, s))
        # output gate:
        o = np.zeros((self.out_dim, s))
        # cell state:
        c = np.zeros((self.out_dim, s))
        # hidden state:
        h = np.zeros((self.out_dim, s))

        WI = np.matmul(self.Wi, x)
        WF = np.matmul(self.Wf, x)
        WG = np.matmul(self.Wg, x)
        WO = np.matmul(self.Wo, x)

        for t in range(s):
            if t == 0:
                i[:, t] = LogSig.f(WI[:, t] + self.wi + self.ri)
                f[:, t] = LogSig.f(WF[:, t] + self.wf + self.rf)
                g[:, t] = TanSig.f(WG[:, t] + self.wg + self.rg)
                o[:, t] = LogSig.f(WO[:, t] + self.wo + self.ro)
                c[:, t] = i[:, t] * g[:, t]
                h[:, t] = o[:, t] * TanSig.f(c[:, t])

            else:
                # i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
                i[:, t] = LogSig.f(WI[:, t] + self.wi + np.matmul(self.Ri, h[:, t-1]) + self.ri)
                # f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)
                f[:, t] = LogSig.f(WF[:, t] + self.wf + np.matmul(self.Rf, h[:, t-1]) + self.rf)
                # g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)
                g[:, t] = TanSig.f(WG[:, t] + self.wg + np.matmul(self.Rg, h[:, t-1]) + self.rg)
                # o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)
                o[:, t] = LogSig.f(WO[:, t] + self.wo + np.matmul(self.Ro, h[:, t-1]) + self.ro)
                # c[t] = f[t] o c[t-1] + i[t] o g[t]
                c[:, t] = f[:, t] * c[:, t-1] + i[:, t] * g[:, t]
                h[:, t] = o[:, t] * TanSig.f(c[:, t])

        if reshape:
            return h.T.reshape(s, b, n)
        return h

    def reach(self, I, method='approx', lp_solver=None, pool=None, RF=None, DR=None, show=False):
        # input gate:
        #   i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
        #
        # forgate gate:
        #   f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)

        # cell gate:
        #   g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)

        # output gate:
        #   o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)

        # cell state:
        #   c[t] = f[t] o c[t-1] + i[t] o g[t], where o is Hadamard product

        # hidden state:
        #   h[t] = o[t] o tanh(c[t])

        n = len(I)

        Wi = self.Wi
        Ri = self.Ri
        wi = self.wi
        ri = self.ri

        Wf = self.Wf
        Rf = self.Rf
        wf = self.wf
        rf = self.rf

        Wg = self.Wg
        Rg = self.Rg
        wg = self.wg
        rg = self.rg

        Wo = self.Wo
        Ro = self.Ro
        wo = self.wo
        ro = self.ro

        if DR is None:
            DR = self.DR
        if RF is None:
            RF = self.RF
        if lp_solver is None:
            lp_solver = self.lp_solver
        if pool is None:
            pool = self.pool

        WI = []
        WF = []
        WG = []
        WO = []
        for i in range(n):
            assert isinstance(I[i], SparseStar) or isinstance(I[i], Star), 'error: ' + \
                'Star and SparseStar are only supported for GRULayer reachability'

            WI.append(I[i].affineMap(Wi, wi+ri))
            WF.append(I[i].affineMap(Wf, wf+rf))
            WG.append(I[i].affineMap(Wg, wg+rg))
            WO.append(I[i].affineMap(Wo, wo+ro))

        H1 = []
        C1 = []
        for t in range(n):
            if show:
                print('\n (LSTM) t: %d\n' % t)
            if t == 0:
                """
                    i[t] = sigmoid(Wi @ x[t] + wi + ri)
                    f[t] = sigmoid(Wf @ x[t] + wf + rf)
                    g[t] = tanh(Wg @ x[t] + wg + rg)
                    o[t] = sigmoid(Wo @ x[t] + wo + ro)
                    c[t] = i[t] o g[t], where o is Hadamard product
                    h[t] = o[t] o tanh(c[t])
                """
                It = LogSig.reach(WI[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                Ft = LogSig.reach(WF[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                Gt = TanSig.reach(WG[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                Ot = LogSig.reach(WO[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                C1.append(IdentityXIdentity.reach(It, Gt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR))
                TCt = TanSig.reach(C1[t])
                H1.append(IdentityXIdentity.reach(Ot, TCt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR))

            else:
                """
                    i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
                         => sigmoid(WI + RI_) = It
                    f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)
                         => sigmoid(WF + RF_) = Ft
                    g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)
                         => tanh(WG + RG_) = Gt
                    o[t] = sigmoid(Wo @ x[t] + wo + 4)
                         => sigmoid(WO + RO_) = Ot
                    c[t] = f[t] o c[t-1] + i[t] o g[t], where o is Hadamard product
                         => IdentityXIdentity(Ft, C[t-1]) + IdentityXIdentity(It, Gt)
                         = FCt_1 + IGt = Ct
                    h[t] = o[t] o tanh(c[t])
                         => IdentityXIdentity(Ot, TCt) = H1[t]
                """
                Ht_1 = H1[t-1]

                RI_ = Ht_1.affineMap(Ri)
                WRi = RI_.minKowskiSum(WI[t])
                It = LogSig.reach(WRi, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                RF_ = Ht_1.affineMap(Rf)
                WRf = RF_.minKowskiSum(WF[t])
                Ft = LogSig.reach(WRf, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                RG_ = Ht_1.affineMap(Rg)
                WRg = RG_.minKowskiSum(WG[t])
                Gt = TanSig.reach(WRg, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                RO_ = Ht_1.affineMap(Ro)
                WRo = RO_.minKowskiSum(WO[t])
                Ot = LogSig.reach(WRo, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                FCt_1 = IdentityXIdentity.reach(Ft, C1[t-1], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                IGt = IdentityXIdentity.reach(It, Gt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                C1.append(FCt_1.minKowskiSum(IGt))

                TCt = TanSig.reach(C1[t])
                H1.append(IdentityXIdentity.reach(Ot, TCt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR))
        return H1

    def get_weights(self, module='default'):

        if module == 'default':
            return self.Wi, self.Wf, self.Wg, self.Wo, \
                self.Ri, self.Rf, self.Rg, self.Ro, \
                self.wi, self.wf, self.wg, self.ro, \
                self.ri, self.rf, self.rg, self.ro

        elif module == 'pytorch':
            # In PyTorch,
            # i is input, h is recurrent
            # i, f, g, o are the input, forget, cell, and output gates, respectively
            # weight_ih_l0: (W_ii|W_if|W_ig|W_io)
            weight_ih_l0 = torch.nn.Parameter(torch.from_numpy(
                np.vstack((self.Wi, self.Wf, self.Wg, self.Wo))))
            # weight_hh_l0: (W_hi|W_hf|W_hg|W_ho)
            weight_hh_l0 = torch.nn.Parameter(torch.from_numpy(
                np.vstack((self.Ri, self.Rf, self.Rg, self.Ro))))
            # bias_ih_l0: (b_ii|b_if|b_ig|b_io)
            bias_ih_l0 = torch.nn.Parameter(torch.from_numpy(
                np.hstack((self.wi, self.wf, self.wg, self.wo))))
            # bias_hh_l0: (b_hi|b_hf|b_hg|b_ho)
            bias_hh_l0 = torch.nn.Parameter(torch.from_numpy(
                np.hstack((self.ri, self.rf, self.rg, self.ro))))
            return weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0

        elif module == 'onnx':
            # In Onnx,
            pass

        else:
            raise Exception('error: unsupported network module')

    @staticmethod
    def rand(out_dim, in_dim):
        """ Randomly generate a LSTMLayer
            Args:
                @out_dim: number of hidden units
                @in_dim: number of inputs (sequence, batch_size, input_size)
        """

        assert out_dim > 0 and in_dim > 0, 'error: invalid number of hidden units or inputs'

        W = np.random.rand(out_dim*4, in_dim)
        R = np.random.rand(out_dim*4, out_dim)
        w = np.random.rand(out_dim*4)
        r = np.random.rand(out_dim*4)
        return LSTMLayer_single(W=W, R=R, w=w, r=r)
