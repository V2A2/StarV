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
from StarV.fun.logsigXidentity import LogsigXIdentity
from StarV.fun.logsigXtansig import LogsigXTansig
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star


class LSTMLayer(object):
    """ LSTMLayer class for qualitative reachability
        Author: Sung Woo Choi
        Date: 06/12/2023
    """

    def __init__(self, layer, output_mode='many', module='default', lp_solver='gurobi', pool=None, RF=0.0, DR=0, dtype='float64'):
    
        if module == 'default':
            '''
                layer = [L_0, L_1, ..., L_n], where
                L_0 = [W, R, b].
                
                W is weight matrices for input, output, forget, and cell gates; i_t, f_t, o_t, g_t
                R is recurrent weight matrices for input, output, forget, and cell gates; i_t, f_t, o_t, g_t
                b is bias vectors of both input and reccurent
            '''
            assert isinstance(layer, list), 'error: provided layer is not a list'

            num_layers = len(layer)

            # input weight matrix
            self.Wi, self.Wo, self.Wf, self.Wg = [], [], [], []
            # recurrent weight matrix
            self.Ri, self.Ro, self.Rf, self.Rg = [], [], [], []
            # input bias vector
            self.wi, self.wo, self.wf, self.wg = [], [], [], []
            # recurrent bias vector
            self.ri, self.ro, self.rf, self.rg = [], [], [], []
            
            for i in range(num_layers):
                W, R, b = layer[i]
                Wi, Wo, Wf, Wg = np.split(W, 4, axis=0)
                self.Wi.append(Wi)
                self.Wo.append(Wo)
                self.Wf.append(Wf)
                self.Wg.append(Wg)
                
                Ri, Ro, Rf, Rg = np.split(R, 4, axis=0)
                self.Ri.append(Ri)
                self.Ro.append(Ro)
                self.Rf.append(Rf)
                self.Rg.append(Rg)

                wi, wo, wf, wg, ri, ro, rf, rg = np.split(b, 8)
                self.wi.append(wi)
                self.wo.append(wo)
                self.wf.append(wf)
                self.wg.append(wg)
                self.ri.append(ri)
                self.ro.append(ro)
                self.rf.append(rf)
                self.rg.append(rg)

            self.in_dim = self.Wi[0].shape[1]
            self.out_dim = self.Wi[num_layers-1].shape[0]
            self.num_layers = len(self.Wi)
            self.bias = True
            self.batch_first = True

        elif module == 'pytorch':
            assert isinstance(layer, torch.nn.LSTM), 'error: provided layer is not torch.nn.LSTM layer'

            # input weight matrix
            self.Wi, self.Wf, self.Wg, self.Wo = [], [], [], []
            # recurrent weight matrix
            self.Ri, self.Rf, self.Rg, self.Ro = [], [], [], []
            # input bias vector
            self.wi, self.wf, self.wg, self.wo = [], [], [], []
            # recurrent bias vector
            self.ri, self.rf, self.rg, self.ro = [], [], [], []

            for ln in range(layer.num_layers):
                Wi, Wf, Wg, Wo = torch.split(getattr(layer, f"weight_ih_l{ln}"), layer.hidden_size, 0)
                self.Wi.append(Wi.detach().numpy().astype(dtype))
                self.Wf.append(Wf.detach().numpy().astype(dtype))
                self.Wg.append(Wg.detach().numpy().astype(dtype))
                self.Wo.append(Wo.detach().numpy().astype(dtype))

                Ri, Rf, Rg, Ro = torch.split(getattr(layer, f"weight_hh_l{ln}"), layer.hidden_size, 0)
                self.Ri.append(Ri.detach().numpy().astype(dtype))
                self.Rf.append(Rf.detach().numpy().astype(dtype))
                self.Rg.append(Rg.detach().numpy().astype(dtype))
                self.Ro.append(Ro.detach().numpy().astype(dtype))

                wi, wf, wg, wo = torch.split(getattr(layer, f"bias_ih_l{ln}"), layer.hidden_size, 0)
                self.wi.append(wi.detach().numpy().astype(dtype))
                self.wf.append(wf.detach().numpy().astype(dtype))
                self.wg.append(wg.detach().numpy().astype(dtype))
                self.wo.append(wo.detach().numpy().astype(dtype))

                ri, rf, rg, ro = torch.split(getattr(layer, f"bias_hh_l{ln}"), layer.hidden_size, 0)
                self.ri.append(ri.detach().numpy().astype(dtype))
                self.rf.append(rf.detach().numpy().astype(dtype))
                self.rg.append(rg.detach().numpy().astype(dtype))
                self.ro.append(ro.detach().numpy().astype(dtype))
            
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
        print('LSTM Layer')
        print('input dimension (input_size): {}'.format(self.in_dim))
        print('output dimension (hidden_size): {}'.format(self.in_dim))
        print('number of layers: {}'.format(self.num_layers))
        print('output mode: {}'.format(self.output_mode))
        print('bias: {}'.format(self.bias))
        print('batch_first: {}'.format(self.batch_first))


    def __str__(self):
        for ln in range(self.num_layers):
            print('(LSTM) {} layer'.format(ln))
            print('Wi: \n{}'.format(self.Wi[ln]))
            print('Ri: \n{}'.format(self.Ri[ln]))
            print('wi: \n{}'.format(self.wi[ln]))
            print('ri: \n{}'.format(self.ri[ln]))

            print('Wf: \n{}'.format(self.Wf[ln]))
            print('Rf: \n{}'.format(self.Rf[ln]))
            print('wf: \n{}'.format(self.wf[ln]))
            print('rf: \n{}'.format(self.rf[ln]))

            print('Wg: \n{}'.format(self.Wg[ln]))
            print('Rg: \n{}'.format(self.Rg[ln]))
            print('wg: \n{}'.format(self.wg[ln]))
            print('rg: \n{}'.format(self.rg[ln]))

            print('Wo: \n{}'.format(self.Wo[ln]))
            print('Ro: \n{}'.format(self.Ro[ln]))
            print('wo: \n{}'.format(self.wo[ln]))
            print('ro: \n{}'.format(self.ro[ln]))
        return '\n'


    def __repr__(self):
        for ln in range(self.num_layers):
            print('(LSTM) {} layer'.format(ln))
            print('Wi: {}'.format(self.Wi[ln].shape))
            print('Ri: {}'.format(self.Ri[ln].shape))
            print('wi: {}'.format(self.wi[ln].shape))
            print('ri: {}'.format(self.ri[ln].shape))

            print('Wf: {}'.format(self.Wf[ln].shape))
            print('Rf: {}'.format(self.Rf[ln].shape))
            print('wf: {}'.format(self.wf[ln].shape))
            print('rf: {}'.format(self.rf[ln].shape))

            print('Wg: {}'.format(self.Wg[ln].shape))
            print('Rg: {}'.format(self.Rg[ln].shape))
            print('wg: {}'.format(self.wg[ln].shape))
            print('rg: {}'.format(self.rg[ln].shape))

            print('Wo: {}'.format(self.Wo[ln].shape))
            print('Ro: {}'.format(self.Ro[ln].shape))
            print('wo: {}'.format(self.wo[ln].shape))
            print('ro: {}'.format(self.ro[ln].shape))
        return '\n'


    def evaluate(self, x, h0=None, c0=None):
        """
            i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
            f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)
            g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)
            o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)
            c[t] = f[t] o c[t-1] + i[t] o g[t]
            h[t] = o[t] o tanh(c[t])
        """

        assert isinstance(x, np.ndarray), 'error: ' + \
        'x should be an numpy array'

        if h0 is not None and c0 is not None:
            assert isinstance(h0, np.ndarray), 'error: ' + \
            'h0 should be an numpy array'
            assert isinstance(c0, np.ndarray), 'error: ' + \
            'c0 should be an numpy array'

        #     h0 = np.float64(h0)
        #     c0 = np.float64(c0)
            
        # x = np.float64(x)
        
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

            bi = self.wi[l] + self.ri[l]
            bf = self.wf[l] + self.rf[l]
            bg = self.wg[l] + self.rg[l]
            bo = self.wo[l] + self.ro[l]

            for b in range(batch_size):
                
                if l == 0:
                    xl = x[b, : , :]
                else:
                    xl = hprev[b, : , :]

                WI = np.matmul(self.Wi[l], xl) + bi[:, np.newaxis]
                WF = np.matmul(self.Wf[l], xl) + bf[:, np.newaxis]
                WG = np.matmul(self.Wg[l], xl) + bg[:, np.newaxis]
                WO = np.matmul(self.Wo[l], xl) + bo[:, np.newaxis]

                for t in range(sequence):

                    if t == 0:
                        if h0 is None or c0 is None:
                            i = LogSig.f(WI[:, t])
                            f = LogSig.f(WF[:, t])
                            g = TanSig.f(WG[:, t])
                            o = LogSig.f(WO[:, t])
                            c = i * g
                            h[b, :, t] = o * TanSig.f(c)

                            continue
                        
                        else:
                            ht_1 = h0[l, b, :]
                            ct_1 = c0[l, b, :]
                            
                    else:
                        ht_1 = h[b, :, t-1]
                        ct_1 = c

                    i = LogSig.f(WI[:, t] + np.matmul(self.Ri[l], ht_1))
                    f = LogSig.f(WF[:, t] + np.matmul(self.Rf[l], ht_1))
                    g = TanSig.f(WG[:, t] + np.matmul(self.Rg[l], ht_1))
                    o = LogSig.f(WO[:, t] + np.matmul(self.Ro[l], ht_1))
                    c = f * ct_1 + i * g
                    h[b, :, t]  = o * TanSig.f(c)

        if x_shape == 3:
            if self.batch_first:
                # from (batch_size, onput_size, sequence)
                # transpose to (batch_size, sequence, onput_size)
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


    def reachApprox_identity(self, I, H0=None, C0=None, lp_solver=None, pool=None, RF=None, DR=None, show=False):
        """
            @output_mode: -one: returns the single output reachable set of the last layer
                         -many: returns many output reachable sets of the last layer

            @H0: the initial hidden state for each sets in the input sequence.
        
            ## currently batch is not supported

            input gate:
              i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
            forgate gate:
              f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)
            cell gate:
              g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)
            output gate:
              o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)
            cell state:
              c[t] = f[t] o c[t-1] + i[t] o g[t], where o is Hadamard product
            hidden state:
              h[t] = o[t] o tanh(c[t])
        """

        assert isinstance(I, list), 'error: ' + \
        'the input, I, should be a list of sparse star sets'

        sequence = len(I)

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

        outMode = self.output_mode == 'one'
        num_layers = self.num_layers

        for l in range(num_layers):
            if show:
                print('(LSTM) layer: {}'.format(l))
            
            WI = []
            WF = []
            WG = []
            WO = []
            if l == 0:
                Xl = I
            else:
                Xl = H1

            for i in range(sequence):
                if l == 0:
                    assert isinstance(Xl[i], SparseStar) or isinstance(Xl[i], Star), 'error: ' + \
                    'Star and SparseStar are only supported for GRULayer reachability'

                WI.append(Xl[i].affineMap(Wi[l], wi[l]+ri[l]))
                WF.append(Xl[i].affineMap(Wf[l], wf[l]+rf[l]))
                WG.append(Xl[i].affineMap(Wg[l], wg[l]+rg[l]))
                WO.append(Xl[i].affineMap(Wo[l], wo[l]+ro[l]))

            H1 = []
            for t in range(sequence):
                
                if show:
                    print('(LSTM) t: {}'.format(t))

                if t == 0:
                    """
                        i[t] = sigmoid(Wi @ x[t] + wi + ri)
                        f[t] = sigmoid(Wf @ x[t] + wf + rf)
                        g[t] = tanh(Wg @ x[t] + wg + rg)
                        o[t] = sigmoid(Wo @ x[t] + wo + ro)
                        c[t] = i[t] o g[t], where o is Hadamard product
                        h[t] = o[t] o tanh(c[t])
                    """

                    if H0 is None or C0 is None:
                        It = LogSig.reach(WI[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                        Ft = LogSig.reach(WF[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                        Gt = TanSig.reach(WG[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                        Ot = LogSig.reach(WO[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                        C1 = IdentityXIdentity.reach(It, Gt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                        TCt = TanSig.reach(C1)
                        if outMode and l == num_layers-1: # one output mode
                            H1 = [IdentityXIdentity.reach(Ot, TCt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)]
                        else: # many output mode
                            H1.append(IdentityXIdentity.reach(Ot, TCt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR))

                        continue

                    else:
                        Ht_1 = H0
                        Ct_1 = C0
                    
                else:
                    if outMode and l == num_layers-1: # one output mode
                        Ht_1 = H1[0]
                    else: # many output mode
                        Ht_1 = H1[t-1]
                    Ct_1 = C1

                """
                    i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
                        => sigmoid(WI + RI_) = It
                    f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)
                        => sigmoid(WF + RF_) = Ft
                    g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)
                        => tanh(WG + RG_) = Gt
                    o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)
                        => sigmoid(WO + RO_) = Ot
                    c[t] = f[t] o c[t-1] + i[t] o g[t], where o is Hadamard product
                        => IdentityXIdentity(Ft, C[t-1]) + IdentityXIdentity(It, Gt)
                        = FCt_1 + IGt = Ct
                    h[t] = o[t] o tanh(c[t])
                        => IdentityXIdentity(Ot, TCt) = H1[t]
                """

                RI_ = Ht_1.affineMap(Ri[l])
                WRi = RI_.minKowskiSum(WI[t])
                It = LogSig.reach(WRi, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                RF_ = Ht_1.affineMap(Rf[l])
                WRf = RF_.minKowskiSum(WF[t])
                Ft = LogSig.reach(WRf, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                RG_ = Ht_1.affineMap(Rg[l])
                WRg = RG_.minKowskiSum(WG[t])
                Gt = TanSig.reach(WRg, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                RO_ = Ht_1.affineMap(Ro[l])
                WRo = RO_.minKowskiSum(WO[t])
                Ot = LogSig.reach(WRo, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                FCt_1 = IdentityXIdentity.reach(Ft, Ct_1, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                IGt = IdentityXIdentity.reach(It, Gt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

                C1 = FCt_1.minKowskiSum(IGt)

                TCt = TanSig.reach(C1)
                if outMode and l == num_layers-1: # one output mode
                    H1 = [IdentityXIdentity.reach(Ot, TCt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)]
                else: # many output mode
                    H1.append(IdentityXIdentity.reach(Ot, TCt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR))

        return H1

    # def reach_withBatch(self, I, H0=None, C0=None,output_mode='one', method='approx', lp_solver=None, pool=None, RF=None, DR=None, show=False):
    #     """
    #         @output_mode: -one: returns the single output reachable set of the last layer
    #                      -many: returns many output reachable sets of the last layer

    #         @H0: the initial hidden state for each sets in the input sequence.
        
    #         ## currently batch is not supported
    #     """

    #     # input gate:
    #     #   i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
        
    #     # forgate gate:
    #     #   f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)

    #     # cell gate:
    #     #   g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)

    #     # output gate:
    #     #   o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)

    #     # cell state:
    #     #   c[t] = f[t] o c[t-1] + i[t] o g[t], where o is Hadamard product

    #     # hidden state:
    #     #   h[t] = o[t] o tanh(c[t])

    #     # check if there is batch in the input,
    #     isBatch = isinstance(I[0], list)
    #     if isBatch:
    #         if self.batch_first:
    #             batch_size = len(I)
    #             sequence = len(I[0])

    #         else:
    #             batch_size = len(I[0])
    #             sequence = len(I)

    #     else:
    #         batch_size = 1
    #         sequence = len(I)

    #     Wi = self.Wi
    #     Ri = self.Ri
    #     wi = self.wi
    #     ri = self.ri

    #     Wf = self.Wf
    #     Rf = self.Rf
    #     wf = self.wf
    #     rf = self.rf

    #     Wg = self.Wg
    #     Rg = self.Rg
    #     wg = self.wg
    #     rg = self.rg

    #     Wo = self.Wo
    #     Ro = self.Ro
    #     wo = self.wo
    #     ro = self.ro

    #     if DR is None:
    #         DR = self.DR
    #     if RF is None:
    #         RF = self.RF
    #     if lp_solver is None:
    #         lp_solver = self.lp_solver
    #     if pool is None:
    #         pool = self.pool

    #     outMode = output_mode == 'one'

    #     for l in range(self.num_layers):
    #         if l > 0:
    #             Hprev = copy.deepcopy(H1)
            
    #         if isBatch:
    #             WI = []
    #             WF = []
    #             WG = []
    #             WO = []
    #             if l == 0:
    #                 for i in range(sequence):
    #                     assert isinstance(I[i], SparseStar) or isinstance(I[i], Star), 'error: ' + \
    #                         'Star and SparseStar are only supported for GRULayer reachability'

    #                     WI.append(I[b][i].affineMap(Wi[l], wi[l]+ri[l]))
    #                     WF.append(I[b][i].affineMap(Wf[l], wf[l]+rf[l]))
    #                     WG.append(I[b][i].affineMap(Wg[l], wg[l]+rg[l]))
    #                     WO.append(I[b][i].affineMap(Wo[l], wo[l]+ro[l]))

    #             else:
    #                 for i in range(sequence):
    #                     WI.append(Hprev[i].affineMap(Wi[l], wi[l]+ri[l]))
    #                     WF.append(Hprev[i].affineMap(Wf[l], wf[l]+rf[l]))
    #                     WG.append(Hprev[i].affineMap(Wg[l], wg[l]+rg[l]))
    #                     WO.append(Hprev[i].affineMap(Wo[l], wo[l]+ro[l]))

    #         else:
    #             WI = []
    #             WF = []
    #             WG = []
    #             WO = []
    #             if l == 0:
    #                 for i in range(sequence):
    #                     assert isinstance(I[i], SparseStar) or isinstance(I[i], Star), 'error: ' + \
    #                         'Star and SparseStar are only supported for GRULayer reachability'

    #                     WI.append(I[i].affineMap(Wi[l], wi[l]+ri[l]))
    #                     WF.append(I[i].affineMap(Wf[l], wf[l]+rf[l]))
    #                     WG.append(I[i].affineMap(Wg[l], wg[l]+rg[l]))
    #                     WO.append(I[i].affineMap(Wo[l], wo[l]+ro[l]))

    #             else:
    #                 for i in range(sequence):
    #                     WI.append(Hprev[i].affineMap(Wi[l], wi[l]+ri[l]))
    #                     WF.append(Hprev[i].affineMap(Wf[l], wf[l]+rf[l]))
    #                     WG.append(Hprev[i].affineMap(Wg[l], wg[l]+rg[l]))
    #                     WO.append(Hprev[i].affineMap(Wo[l], wo[l]+ro[l]))

    #             H1 = []
    #             for t in range(sequence):
    #                 """
    #                     i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
    #                     f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)
    #                     g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)
    #                     o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)
    #                     c[t] = f[t] o c[t-1] + i[t] o g[t]
    #                     h[t] = o[t] o tanh(c[t])
    #                 """

    #                 if show:
    #                     print('\n (LSTM) t: %d\n' % t)
    #                 if t == 0:
                        
    #                     if H0 is not None and C0 is not None:
    #                         RI_ = H0.affineMap(Ri)
    #                         WRi = RI_.minKowskiSum(WI[t])
    #                         It = LogSig.reach(WRi, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

    #                         RF_ = H0.affineMap(Rf)
    #                         WRf = RF_.minKowskiSum(WF[t])
    #                         Ft = LogSig.reach(WRf, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

    #                         RG_ = H0.affineMap(Rg)
    #                         WRg = RG_.minKowskiSum(WG[t])
    #                         Gt = TanSig.reach(WRg, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

    #                         RO_ = H0.affineMap(Ro)
    #                         WRo = RO_.minKowskiSum(WO[t])
    #                         Ot = LogSig.reach(WRo, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

    #                         FCt_1 = IdentityXIdentity.reach(Ft, C0, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
    #                         IGt = IdentityXIdentity.reach(It, Gt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

    #                         C1 = FCt_1.minKowskiSum(IGt)

    #                         TCt = TanSig.reach(C1)
    #                         if outMode: # one output mode
    #                             H1 = IdentityXIdentity.reach(Ot, TCt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
    #                         else: # many output mode
    #                             H1.append(IdentityXIdentity.reach(Ot, TCt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR))
                        
    #                     else:
    #                         It = LogSig.reach(WI[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
    #                         Ft = LogSig.reach(WF[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
    #                         Gt = TanSig.reach(WG[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
    #                         Ot = LogSig.reach(WO[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
    #                         C1 = IdentityXIdentity.reach(It, Gt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

    #                         TCt = TanSig.reach(C1)
    #                         if outMode: # one output mode
    #                             H1 = IdentityXIdentity.reach(Ot, TCt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
    #                         else: # many output mode
    #                             H1.append(IdentityXIdentity.reach(Ot, TCt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR))

    #                 else:
    #                     """
    #                         i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
    #                             => sigmoid(WI + RI_) = It
    #                         f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)
    #                             => sigmoid(WF + RF_) = Ft
    #                         g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)
    #                             => tanh(WG + RG_) = Gt
    #                         o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)
    #                             => sigmoid(WO + RO_) = Ot
    #                         c[t] = f[t] o c[t-1] + i[t] o g[t], where o is Hadamard product
    #                             => IdentityXIdentity(Ft, C[t-1]) + IdentityXIdentity(It, Gt)
    #                             = FCt_1 + IGt = Ct
    #                         h[t] = o[t] o tanh(c[t])
    #                             => IdentityXIdentity(Ot, TCt) = H1[t]
    #                     """
    #                     if outMode: # one output mode
    #                         Ht_1 = H1
    #                     else: # many output mode
    #                         Ht_1 = H1[t-1]

    #                     RI_ = Ht_1.affineMap(Ri)
    #                     WRi = RI_.minKowskiSum(WI[t])
    #                     It = LogSig.reach(WRi, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

    #                     RF_ = Ht_1.affineMap(Rf)
    #                     WRf = RF_.minKowskiSum(WF[t])
    #                     Ft = LogSig.reach(WRf, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

    #                     RG_ = Ht_1.affineMap(Rg)
    #                     WRg = RG_.minKowskiSum(WG[t])
    #                     Gt = TanSig.reach(WRg, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

    #                     RO_ = Ht_1.affineMap(Ro)
    #                     WRo = RO_.minKowskiSum(WO[t])
    #                     Ot = LogSig.reach(WRo, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

    #                     FCt_1 = IdentityXIdentity.reach(Ft, C1, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
    #                     IGt = IdentityXIdentity.reach(It, Gt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)

    #                     C1 = FCt_1.minKowskiSum(IGt)

    #                     TCt = TanSig.reach(C1)
    #                     if outMode: # one output mode
    #                         H1 = IdentityXIdentity.reach(Ot, TCt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
    #                     else: # many output mode
    #                         H1.append(IdentityXIdentity.reach(Ot, TCt, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR))
    #     return H1


    # def get_weights(self, module='default'):

    #     if module == 'default':
    #         return self.Wi, self.Wf, self.Wg, self.Wo, \
    #             self.Ri, self.Rf, self.Rg, self.Ro, \
    #             self.wi, self.wf, self.wg, self.ro, \
    #             self.ri, self.rf, self.rg, self.ro

    #     elif module == 'pytorch':
    #         # In PyTorch,
    #         # i is input, h is recurrent
    #         # i, f, g, o are the input, forget, cell, and output gates, respectively
    #         # weight_ih_l0: (W_ii|W_if|W_ig|W_io)
    #         weight_ih_l0 = torch.nn.Parameter(torch.from_numpy(
    #             np.vstack((self.Wi, self.Wf, self.Wg, self.Wo))))
    #         # weight_hh_l0: (W_hi|W_hf|W_hg|W_ho)
    #         weight_hh_l0 = torch.nn.Parameter(torch.from_numpy(
    #             np.vstack((self.Ri, self.Rf, self.Rg, self.Ro))))
    #         # bias_ih_l0: (b_ii|b_if|b_ig|b_io)
    #         bias_ih_l0 = torch.nn.Parameter(torch.from_numpy(
    #             np.hstack((self.wi, self.wf, self.wg, self.wo))))
    #         # bias_hh_l0: (b_hi|b_hf|b_hg|b_ho)
    #         bias_hh_l0 = torch.nn.Parameter(torch.from_numpy(
    #             np.hstack((self.ri, self.rf, self.rg, self.ro))))
    #         return weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0

    #     elif module == 'onnx':
    #         # In Onnx,
    #         raise Exception('error: ONNX network module is yet unsupported')

    #     else:
    #         raise Exception('error: unsupported network module')

    def reachApprox(self, I, H0=None, C0=None, lp_solver=None, pool=None, RF=None, DR=None, show=False):
        """
            @output_mode: -one: returns the single output reachable set of the last layer
                         -many: returns many output reachable sets of the last layer

            @H0: the initial hidden state for each sets in the input sequence.
        
            ## currently batch is not supported

            input gate:
              i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
            forgate gate:
              f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)
            cell gate:
              g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)
            output gate:
              o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)
            cell state:
              c[t] = f[t] o c[t-1] + i[t] o g[t], where o is Hadamard product
            hidden state:
              h[t] = o[t] o tanh(c[t])
        """

        assert isinstance(I, list), 'error: ' + \
        'the input, I, should be a list of sparse star sets'

        sequence = len(I)

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

        outMode = self.output_mode == 'one'
        num_layers = self.num_layers

        for l in range(num_layers):
            if show:
                print('(LSTM) layer: {}'.format(l))
            
            WI = []
            WF = []
            WG = []
            WO = []
            if l == 0:
                Xl = I
            else:
                Xl = Ht

            for i in range(sequence):
                if l == 0:
                    assert isinstance(Xl[i], SparseStar) or isinstance(Xl[i], Star), 'error: ' + \
                    'Star and SparseStar are only supported for GRULayer reachability'

                WI.append(Xl[i].affineMap(Wi[l], wi[l]+ri[l]))
                WF.append(Xl[i].affineMap(Wf[l], wf[l]+rf[l]))
                WG.append(Xl[i].affineMap(Wg[l], wg[l]+rg[l]))
                WO.append(Xl[i].affineMap(Wo[l], wo[l]+ro[l]))

            Ht = []
            for t in range(sequence):
                
                if show:
                    print('(LSTM) t: {}'.format(t))

                if t == 0:
                    """
                        i[t] = sigmoid(Wi @ x[t] + wi + ri)
                        f[t] = sigmoid(Wf @ x[t] + wf + rf)
                        g[t] = tanh(Wg @ x[t] + wg + rg)
                        o[t] = sigmoid(Wo @ x[t] + wo + ro)
                        c[t] = i[t] o g[t], where o is Hadamard product
                        h[t] = o[t] o tanh(c[t])


                        WI = Wi @ x[t] + wi + ri
                        WF = Wf @ x[t] + wf + rf
                        WG = Wg @ x[t] + wg + rg
                        WO = Wo @ x[t] + wo + ro

                        Ct = sigmoid(WI) o tanh(WG)
                        Ht = sigmoid(WO) o tanh(Ct)
                    """
            
                    if H0 is None or C0 is None:
                        Ct = LogsigXTansig.reach(X=WI[t], H=WG[t], lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                        if outMode and l == num_layers-1: # one output mode
                            T = LogsigXTansig.reach(X=WO[t], H=Ct, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                            Ht = [copy.deepcopy(T)]
                        else: # many output mode
                            T = LogsigXTansig.reach(X=WO[t], H=Ct, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                            Ht.append(copy.deepcopy(T))

                    else:
                        Ht_1 = H0
                        Ct_1 = C0
                    
                else:
                    if outMode and l == num_layers-1: # one output mode
                        Ht_1 = Ht[0]
                    else: # many output mode
                        Ht_1 = Ht[t-1]
                    Ct_1 = Ct

                    """
                        i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
                            => sigmoid(WI + RI_) = It
                        f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)
                            => sigmoid(WF + RF_) = Ft
                        g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)
                            => tanh(WG + RG_) = Gt
                        o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)
                            => sigmoid(WO + RO_) = Ot
                        c[t] = f[t] o c[t-1] + i[t] o g[t], where o is Hadamard product
                            => IdentityXIdentity(Ft, C[t-1]) + IdentityXIdentity(It, Gt)
                            = FCt_1 + IGt = Ct
                        h[t] = o[t] o tanh(c[t])
                            => IdentityXIdentity(Ot, TCt) = H1[t]
                    """


                    """
                        input gate:
                        i[t] = sigmoid(Wi @ x[t] + wi + Ri @ h[t-1] + ri)
                        forgate gate:
                        f[t] = sigmoid(Wf @ x[t] + wf + Rf @ h[t-1] + rf)
                        cell gate:
                        g[t] = tanh(Wg @ x[t] + wg + Rg @ h[t-1] + rg)
                        output gate:
                        o[t] = sigmoid(Wo @ x[t] + wo + Ro @ h[t-1] + ro)
                        cell state:
                        c[t] = f[t] o c[t-1] + i[t] o g[t], where o is Hadamard product
                        hidden state:
                        h[t] = o[t] o tanh(c[t])
                        
                        WI = Wi @ x[t] + wi + ri
                        WF = Wf @ x[t] + wf + rf
                        WG = Wg @ x[t] + wg + rg
                        WO = Wo @ x[t] + wo + ro

                        RI_ = Ri @ Ht
                        RF_ = Rf @ Ht
                        RG_ = Rg @ Ht
                        RO_ = Ro @ Ht

                        WRi = WI + RI
                        WRf = WF + RF
                        WRg = WG + RG
                        WRo = WO + RO
                        
                        Ct = sigmoid(WRf) o Ct_1 + sigmoid(WRi) o tanh(WRg) = FCt + IGt
                        Ht = sigmoid(WRo) o tanh(Ct)
                    
                    """
                    RI_ = Ht_1.affineMap(Ri[l])
                    WRi = RI_.minKowskiSum(WI[t])

                    RF_ = Ht_1.affineMap(Rf[l])
                    WRf = RF_.minKowskiSum(WF[t])

                    RG_ = Ht_1.affineMap(Rg[l])
                    WRg = RG_.minKowskiSum(WG[t])

                    RO_ = Ht_1.affineMap(Ro[l])
                    WRo = RO_.minKowskiSum(WO[t])

                    FCt = LogsigXIdentity.reach(X=WRf, H=Ct_1, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                    IGt = LogsigXTansig.reach(X=WRi, H=WRg, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                    Ct = FCt.minKowskiSum(IGt)

                    if outMode and l == num_layers-1: # one output mode
                        T = LogsigXTansig.reach(X=WRo, H=Ct, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                        Ht = [copy.deepcopy(T)]
                    else: # many output mode
                        T = LogsigXTansig.reach(X=WRo, H=Ct, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR)
                        Ht.append(copy.deepcopy(T))
        return Ht
    

    def reach_identity(self, In, H0=None, C0=None, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
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
            return self.reachApprox_identity(In, H0=H0, C0=C0, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
        raise Exception('error: unknown reachability method')
    

    def reach(self, In, H0=None, C0=None, method='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
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
            return self.reachApprox(In, H0=H0, C0=C0, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
        raise Exception('error: unknown reachability method')


    @staticmethod
    def rand(in_dim, out_dim, num_layers):
        """ Randomly generate a LSTMLayer
            Args:
                @out_dim: number of hidden units
                @in_dim: number of input units
                @num_layers: number of layers


            layer = [W, R, w, r], where
                W = [Wi, Wf, Wg, Wo], a list of input weight matrices
                R = [Ri, Rf, Rg, Ro], a list of reccurent weight matrices
                w = [wi, wf, wg, wo], a list of input bias vectors
                r = [ri, rf, rg, ro], a list of recurrent bias vectors

            Wi = [W_ii[0], ... , W_ii[l]]
            ...
            Wo = [W_io[0], ... , W_io[l]]

            wi = [w_ii[0], ... , w_ii[l]]
            ...
            wo = [w_io[0], ... , w_io[l]]

            similiar logic applies to reccurent side
        """

        assert out_dim > 0, 'error: invalid number of hidden units'
        assert in_dim > 0, 'error: invalid number of input units'
        assert num_layers > 0, 'error: invalid number of layers'
        
        Wi, Wf, Wg, Wo = [], [], [], []
        Ri, Rf, Rg, Ro = [], [], [], []
        wi, wf, wg, wo = [], [], [], []
        ri, rf, rg, ro = [], [], [], []

        for l in range(num_layers):
            Wi.append(np.random.rand(out_dim, in_dim))
            Wf.append(np.random.rand(out_dim, in_dim))
            Wg.append(np.random.rand(out_dim, in_dim))
            Wo.append(np.random.rand(out_dim, in_dim))

            Ri.append(np.random.rand(out_dim, in_dim))
            Rf.append(np.random.rand(out_dim, in_dim))
            Rg.append(np.random.rand(out_dim, in_dim))
            Ro.append(np.random.rand(out_dim, in_dim))

            wi.append(np.random.rand(out_dim))
            wf.append(np.random.rand(out_dim))
            wg.append(np.random.rand(out_dim))
            wo.append(np.random.rand(out_dim))

            ri.append(np.random.rand(out_dim))
            rf.append(np.random.rand(out_dim))
            rg.append(np.random.rand(out_dim))
            ro.append(np.random.rand(out_dim))

        W = [Wi, Wf, Wg, Wo]
        R = [Ri, Rf, Rg, Ro]
        w = [wi, wf, wg, wo]
        r = [ri, rf, rg, ro]
        return LSTMLayer([W, R, w, r], module='default')
    
