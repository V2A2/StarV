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
Main Ceritifier Class: certify robustness of classification neural networks
Sung Woo Choi, 07/17/2023
"""

import os
import copy
import time
import pickle
import multiprocessing
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import jaccard_score
from StarV.net.network import NeuralNetwork
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.set.probsparseimagestar2dcoo import ProbSparseImageStar2DCOO
from StarV.set.probsparseimagestar2dcsr import ProbSparseImageStar2DCSR
from StarV.layer.PixelClassificationLayer import PixelClassificationLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.verifier.crown import (
    reachCROWN,
    certifyRobustness_single_input_CROWN,
    certifyRobustness_CROWN,
    certifyProbabilisticRobustness_single_input_CROWN,
    certifyProbabilisticRobustness_CROWN,
)

class Certifier(object):
    """
        Certifier Class

        Properties: (Certification Settings)

        @lp_solver: lp solver: 'gurobi' (default), 'glpk', 'linprog'
        @method: certification method: 'BFS' ("bread-first-search") or 'DFS' ("depth-first-search")
        @reachMethod: reachability method: 'approx' (default) or 'exact'
        @n_processes: number of processes used for certification
        @RF: relaxation factor in [0.0, 1.0]
        @DR: depth reduction threshold
    """

    def __init__(self, lp_solver='gurobi', method='BFS', reachMethod='approx', n_processes=1, RF=0.0, DR=0):
        self.lp_solver = lp_solver
        self.method = method
        self.reachMethod = reachMethod
        self.n_processes = n_processes
        self.RF = RF
        self.DR = DR

    def _run_with_pool(self, fn, *args, **kwargs):
        if self.n_processes > 1:
            pool = multiprocessing.Pool(self.n_processes)
            try:
                return fn(*args, pool=pool, **kwargs)
            finally:
                pool.close()
                pool.join()
        return fn(*args, pool=None, **kwargs)

    def reach(self, net, inputSet, show=False):
        """Compute reachable set of net on inputSet."""
        return self._run_with_pool(
            reachBFS,
            net=net, inputSet=inputSet, reachMethod=self.reachMethod,
            lp_solver=self.lp_solver, RF=self.RF, DR=self.DR, show=show
        )

    def certify(self, net, inputs, labels=None, return_output=False, show=False):
        """Certify robustness of classification neural network on inputs."""
        return self._run_with_pool(
            certifyRobustness,
            net=net, inputs=inputs, labels=labels, veriMethod=self.method,
            reachMethod=self.reachMethod, lp_solver=self.lp_solver,
            RF=self.RF, DR=self.DR, return_output=return_output, show=show
        )

    def certify_sigmoid(self, net, input, label=None, epsilon=0.01, show=False):
        """Certify robustness of sigmoidal neural network on input."""
        return self._run_with_pool(
            certifyRobustness_sigmoid,
            net=net, input=input, label=label, epsilon=epsilon,
            veriMethod=self.method, reachMethod=self.reachMethod,
            lp_solver=self.lp_solver, RF=self.RF, DR=self.DR, show=show
        )

    def certify_sequence(self, net, inputs, epsilon=0.01, show=False):
        """Certify robustness of sequence/recurrent neural network on inputs."""
        return self._run_with_pool(
            certifyRobustness_sequence,
            net=net, inputs=inputs, epsilon=epsilon,
            veriMethod=self.method, reachMethod=self.reachMethod,
            lp_solver=self.lp_solver, RF=self.RF, DR=self.DR, show=show
        )

    def certify_pixel(self, net, in_sets, in_datas, num_classes, return_output=False, show=False):
        """Certify pixel-level robustness of semantic segmentation neural network."""
        return self._run_with_pool(
            certifyRobustness_pixel,
            net=net, in_sets=in_sets, in_datas=in_datas, num_classes=num_classes,
            veriMethod=self.method, reachMethod=self.reachMethod,
            lp_solver=self.lp_solver, RF=self.RF, DR=self.DR,
            return_output=return_output, show=show
        )

    def certify_probabilistic(
        self, net, inputs, labels=None, relu_approx_mode='analytic_gaussian',
        use_lp=False, return_output=False, show=False
    ):
        """Certify quantitative probabilistic robustness on ProbStar / ProbSparseImageStar inputs."""
        return self._run_with_pool(
            certifyProbabilisticRobustness,
            net=net, inputs=inputs, labels=labels, veriMethod=self.method,
            reachMethod=self.reachMethod, lp_solver=self.lp_solver,
            RF=self.RF, DR=self.DR, relu_approx_mode=relu_approx_mode,
            use_lp=use_lp, return_output=return_output, show=show
        )

    def certify_probabilistic_pixel(
        self, net, in_sets, in_datas, num_classes, relu_approx_mode='analytic_gaussian',
        use_lp=False, return_output=False, show=False
    ):
        """Certify quantitative pixel-level probabilistic robustness of segmentation network."""
        return self._run_with_pool(
            certifyProbabilisticRobustness_pixel,
            net=net, in_sets=in_sets, in_datas=in_datas, num_classes=num_classes,
            veriMethod=self.method, reachMethod=self.reachMethod,
            lp_solver=self.lp_solver, RF=self.RF, DR=self.DR,
            relu_approx_mode=relu_approx_mode, use_lp=use_lp,
            return_output=return_output, show=show
        )


def reachBFS(net, inputSet, reachMethod='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False, relu_approx_mode='analytic_gaussian'):
    """ Verification of neural network with over-apporoximation method.
        Compute Reachable set layer-by-layer
    """

    assert isinstance(net, NeuralNetwork), 'error: first input should be a NeuralNetwork object'
    assert isinstance(
        inputSet,
        (
            list, SparseStar, Star, ProbStar, ImageStar,
            SparseImageStar2DCOO, SparseImageStar2DCSR,
            ProbSparseImageStar2DCOO, ProbSparseImageStar2DCSR,
        ),
    ), (
        'error: second input should be a list of Star/ImageStar/ProbStar/SparseStar/'
        'SparseImageStar2DCOO/SparseImageStar2DCSR sets or a single set'
    )

    if str(reachMethod).lower() in ('crown', 'alpha-crown', 'alpha-beta-crown'):
        start = time.perf_counter()
        outputSet = reachCROWN(
            net, inputSet, mode=str(reachMethod).lower(),
            lp_solver=lp_solver, show=show
        )
        totalReachTime = time.perf_counter() - start
        if isinstance(net.layers[-1], PixelClassificationLayer):
            pixel_classification = net.layers[-1].reach(
                outputSet, method='approx', lp_solver=lp_solver,
                pool=pool, RF=RF, DR=DR, show=show
            )
            return outputSet, totalReachTime, pixel_classification
        return outputSet, totalReachTime

    # compute reachable set
    reachTime = []
    In = inputSet

    # For semantic segmentation neural network
    if isinstance(net.layers[-1], PixelClassificationLayer):
        for i in range(net.n_layers-1):
            if show:
                print(f"\nComputing {net.layers[i].__class__.__name__} layer {i} reachable set...")
            
            start = time.perf_counter()
            if isinstance(net.layers[i], ReLULayer):
                In = net.layers[i].reach(In, method=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show, relu_approx_mode=relu_approx_mode)
            else:
                In = net.layers[i].reach(In, method=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
            vt = time.perf_counter() - start

            # reachSet.append(In)
            reachTime.append(vt)

            if show:
                print('Number of stars/sparsestars: {}'.format(len(In)))
                if reachMethod == 'approx':
                    print(f"Number of predicate variables: {getattr(In, 'num_pred', getattr(In, 'nVars', 0))}")
                    if isinstance(In, ImageStar) or isinstance(In, Star):
                        print(f"Shape of the set: {In.V.shape}")
                    elif isinstance(In, SparseImageStar2DCOO) or isinstance(In, SparseImageStar2DCSR):
                        print(f"Shape of the set: {In.shape + (In.num_pred,)}")
                print(f"Reachability analysis is done in {vt} seconds")

        outputSet = In
        totalReachTime = sum(reachTime)    
        pixel_classification = net.layers[-1].reach(In, method=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
        return outputSet, totalReachTime, pixel_classification
        
    for i in range(net.n_layers):
        if show:
            print(f"\nComputing {net.layers[i].__class__.__name__} layer {i} reachable set...")
        
        start = time.perf_counter()
        if isinstance(net.layers[i], ReLULayer):
            In = net.layers[i].reach(In, method=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show, relu_approx_mode=relu_approx_mode)
        else:
            In = net.layers[i].reach(In, method=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
        vt = time.perf_counter() - start

        # reachSet.append(In)
        reachTime.append(vt)

        if show:
            print('Number of stars/sparsestars: {}'.format(len(In)))
            if reachMethod == 'approx':
                if isinstance(In, list):
                    print(f"Number of predicate variables of In[0]: {In[0].nVars}")
                else:
                    print(f"Number of predicate variables: {In.num_pred}")
                    if isinstance(In, ImageStar) or isinstance(In, Star):
                        print(f"Shape of the set: {In.V.shape}")
                    elif isinstance(In, SparseImageStar2DCOO) or isinstance(In, SparseImageStar2DCSR):
                        print(f"Shape of the set: {In.shape + (In.num_pred,)}")
            print(f"Reachability analysis is done in {vt} seconds")
            
    outputSet = In
    totalReachTime = sum(reachTime)    
    return outputSet, totalReachTime

def certifyRobustness_sigmoid(net, input, label=None, epsilon=0.01, veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
    """
        Certify robustness of neural networks with given inputs

        @input:

        Return:
            @rb: robustness results
                = 1: the network is robust
                = 0: the network is not robust
                = 2: the network is uncertain; unknown

            @vt: verification time
    """
    assert isinstance(DR, int) or isinstance(DR, list), 'depth reduction should be an integer or a list of integers'
    
    RF_ = [RF] if isinstance(RF, int) else RF
    num_rf = len(RF_)
    DR_ = [DR] if isinstance(DR, int) else DR
    num_dr = len(DR_)
    eps = [epsilon] if isinstance(epsilon, float) else epsilon
    num_eps = len(eps)

    # convert input data shape into (num_data, num_image_pixels)
    x = input
    if x.ndim > 1:
        num_data = x.shape[0]

    else:
        num_data = 1
        x = x[np.newaxis, :]
        
    # if label is None:
    #     label = np.zeros(num_data)
    #     for i in range(num_data):
    #         label[i] = net.evaluate(x[i, :]).max().astype(int)
    # else:
    #     if isinstance(label, int):
    #         label = np.array([label])

    #     elif isinstance(label, np.ndarray):
    #         if label.ndim > 1:
    #             raise Exception('error: lable should be an integer or 1D numpy array')

    #         label = label.astype(int)

    #     else:
    #         raise Exception('error: label should be an integer or 1D numpy array')
        
    #     assert num_data == label.shape[0], 'error: inconsistency between number of images and number of labels'
        
    #     for i in range(num_data):
    #         y = net.evaluate(x[i, :]).max().astype(int)
    #         assert y != label[i], 'error: classified image #{} does not match with provided label'.format(i)

    rb = np.zeros([num_rf, num_dr, num_eps, num_data])
    vt = np.zeros([num_rf, num_dr, num_eps, num_data])

    for rf in range(num_rf):
        if show:
                print('Certifying the neural network with relaxation factor {}'.format(RF_[rf]))

        for dr in range(num_dr):
            if show:
                print('Certifying the neural network with depth reduction {}'.format(DR_[dr]))

            for e in range(num_eps):
                if show:
                    print('Certifying the neural network with epsilon {}'.format(eps[e]))

                for i in range(num_data):
                    if show:
                        print('Certifying the neural network with data {}'.format(i))
                    
                    X = SparseStar.inf_attack(data=x[i, :], epsilon=eps[e], data_type='image')
                    
                    start = time.perf_counter()

                    # Compute output reachable sets
                    if veriMethod == 'BFS':
                        Y, _ = reachBFS(net=net, inputSet=X, reachMethod=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF_[rf], DR=DR_[dr], show=show)
                    else:
                        raise Exception('other verification methods is not yet implemented, i.e. DFS')

                    # Certify whether the neural network is robust 
                    y = net.evaluate(x[i, :])
                    max_id = np.argmax(y[np.newaxis], axis=1) # find the classified output

                    max_cands = Y.get_max_point_cadidates()
                    if len(max_cands) == 1:
                        if max_cands == max_id:
                            rb[rf, dr, e, i] = 1
                            
                    else:
                        a = [max_cands == max_id][0]
                        if sum(a) == 0:
                            rb[rf, dr, e, i] = 0
                        
                        else:
                            max_cands = np.delete(max_cands, max_cands == max_id)
                            m = len(max_cands)

                            for j in range(m):
                                if Y.is_p1_larger_than_p2(max_cands[j], max_id):
                                    rb[rf, dr, e, i] = 2
                                    break
                                
                                else:
                                    rb[rf, dr, e, i] = 1

                    vt[rf, dr, e, i] = time.perf_counter() - start  

                    if show:
                        print('Robustness result of data {} (eps = {}, RF = {}, DR = {}): {}'.format(i, eps[e], RF_[rf], DR_[dr], rb[rf, dr, e, i]))
                        print('Verification time of data {} (eps = {}, RF = {}, DR = {}): {}\n'.format(i, eps[e], RF_[rf], DR_[dr], vt[rf, dr, e, i]))

                        cur_time = time.strftime('__%Y-%m-%d_%H-%M-%S__', time.localtime(time.time()))
                        save_dir = 'sigmoidalNet_eval/safety/' + net.type + cur_time + 'eval.pkl'
                        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                        print('Saving the evaluation results to {}'.format(save_dir))

                        with open(save_dir, 'wb') as f:  
                            pickle.dump([rb, vt, epsilon, RF_, DR_, net.type], f)

                if show:
                    print('Robustness result of data {} (eps = {}, RF = {}, DR = {}): {}'.format(i, eps[e], RF_[rf], DR_[dr], (rb[rf, dr, e, :]==1).sum()))
                    print('Verification time of data {} (eps = {}, RF = {}, DR = {}): {}\n'.format(i, eps[e], RF_[rf], DR_[dr], vt[rf, dr, e, :].sum()))
    return rb, vt  


def certifyRobustness_sequence(net, inputs, epsilon=0.01, veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
    """
        Certify robustness of neural network with given an inputs

        @inputs: (input_size, sequence)


        Return:
            @rb: robustness results
                = 1: the network is robust
                = 0: the network is not robust
                = 2: the network is uncertain; unknown

            @vt: verification time
    """
    start = time.perf_counter()
    
    x = inputs
    in_seq = x.shape[1] # sequence

    # Create input reachable sets for rechability analysis
    X = []
    for i in range(in_seq):
        X.append(SparseStar(x[:, i] - epsilon, x[:, i] + epsilon))
        # X.append(SparseStar(np.clip(x[:, i] - epsilon, 0, 1),
        #                     np.clip(x[:, i] + epsilon, 0, 1)))

    # Compute output reachable sets
    if veriMethod == 'BFS':
        Y, _ = reachBFS(net=net, inputSet=X, reachMethod=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
    else:
        raise Exception('other verification methods is not yet implemented, i.e. DFS')
    
    y = net.evaluate(x)
    out_seq = len(Y)

    if out_seq == 1:
        y = y[np.newaxis, :]

    max_id = np.argmax(y, axis=1) # find the classified output
    rb = np.zeros(out_seq)

    for i in range(out_seq):
        max_cands = Y[i].get_max_point_cadidates()
        if len(max_cands) == 1:
            if max_cands == max_id[i]:
                rb[i] = 1
                
        else:
            a = [max_cands == max_id[i]][0]
            if sum(a) == 0:
                rb[i] = 0
            
            else:
                max_cands = np.delete(max_cands, max_cands == max_id[i])
                m = len(max_cands)

                for j in range(m):
                    if Y[i].is_p1_larger_than_p2(max_cands[j], max_id[i]):
                        rb[i] = 2
                        break
                    
                    else:
                        rb[i] = 1

    vt = time.perf_counter() - start       

    return Y, rb, vt 




# def certifyNN(net, inImage, label, veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
#     """
#         Certify robustness of neural network with given an image and a label
        
#         @net:           neural network to verify
#         @inImage:       an input set (Star, SparseStar) 
#         @label:         a classified lable corresponding to the input sets
#         @veriMethod:    verification method: 'BFS' bread-first-search or 'DFS' depth-first-search
#         @reachMethod:   reachability analysis method: 'approx' over-approximate method or 'exact' exact method
#         @lp_solver:     lp solver: 'gurobi' (default), 'glpk', 'linprog', 'estimate'
#         @pool:          parrallel computing
#         @RF:            relaxation factor: between 0.0 (no relaxation) and 1.0 (full relaxation)
#         @DR:            depth reduction: DR > 0, removes predicate variables based on the depth of predicate veriables
#         @show:          display option, comments steps of verification process     

#         Return:
#             @robust = 1: the network is robust
#                     = 0: the network is not robust
#                     = 2: the network is uncertain; unknown
#             @ce:    a counter example
#             @cands: candidate index in the case that the robustness is unkonwn
#             @vt:    verification time
#     """
    
#     assert label < net.out_dim and label >= 0, 'error: invalid classification label'

#     start = time.perf_counter()

#     robust = 2 # unknown first
#     cands = None
#     ce = None

#     if inImage.init_lb is not None:
#         yl = net.evaluate(inImage.init_lb)
#         max_id = np.argmax(yl)
#         if max_id != label:
#             robust = 0
#             ce = inImage.init_lb
        
#         yu = net.evaluate(inImage.init_ub)
#         max_id = np.argmax(yu)
#         if max_id != label:
#             robust = 0
#             ce = inImage.init_ub
        
#     if robust == 2:
#         if veriMethod == 'BFS':
#             R = reachBFS(net=net, inputSet=inImage, reachMethod=reachMethod, lp_solver=lp_solver, pool=None, RF=RF, DR=DR, show=show)
#             [lb, ub] = R.getRanges('estimate')
#             max_val = lb(label)
#             max_cand = np.where(ub > max_val)[0] # max point candidates
#             np.delete(max_cand, max_cand == label) # delete the max labels
            
#             if len(max_cand) == 0:
#                 robust = 1

#             else:
#                 n = len(max_cand)
#                 cnt = 0
#                 for i in range(n):
#                     if R.is_p1_larger_than_p2(max_cand[i], label):
#                         cands = max_cand[i]
#                         break

#                     else:
#                         cnt += 1
                
#                 if cnt == n:
#                     robust = 1

#         else:
#             raise Exception('other verification methods is not yet implemented, i.e. DFS')

#     end = time.perf_counter()
#     vt = end - start
#     return robust, ce, cands, vt

# def certifyRobustness(net, inImages, labels, veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=True):
#     """ 
#         Robustness certification of neural networks (verifies robustness of classification neural networks) 
#         Args:
#             @net:           neural network to verify
#             @inImages:      input sets (Star, SparseStar) 
#             @labels:        a list of correctly classified lables corresponding to the input sets
#             @veriMethod:    verification method: 'BFS' bread-first-search or 'DFS' depth-first-search
#             @reachMethod:   reachability analysis method: 'approx' over-approximate method or 'exact' exact method
#             @lp_solver:     lp solver: 'gurobi' (default), 'glpk', 'linprog', 'estimate'
#             @pool:          parrallel computing
#             @RF:            relaxation factor: between 0.0 (no relaxation) and 1.0 (full relaxation)
#             @DR:            depth reduction: DR > 0, removes predicate variables based on the depth of predicate veriables
#             @show:          display option, comments steps of verification process     

#         Return:
#             @r:     robustness value (in percentage)
#             @rb:    robustness results:
#                 rb = 1: the network is robust
#                    = 0: the network is not robust
#                    = 2: robstness is uncertain / unknown
#             @ce:    counter-examples
#             @cands: candidate indexes
#             @vt:    verification time
    
#     """

#     N = len(inImages)
#     assert len(labels) == N, 'error: inconsistency between the number of correctly classified labels' + \
#     'and the number of input sets'

#     cnt = np.zeros(N)
#     rb = np.zeros(N)
#     ce = [np.empty(0) for _ in range(N)]
#     cands = [np.empty(0) for _ in range(N)]
#     vt = np.zeros(N)

#     #check if

#     for i in range(N):
#         [rb[i], ce[i], cands[i], vt[i]] = certifyNN(net=net, inImage=inImages[i], label=labels[i], veriMethod=veriMethod, \
#                                                     reachMethod=reachMethod, lp_solver='gurobi', pool=None, RF=RF, DR=DR, show=True)
#         cnt[i] = 1 if rb[i] == 1 else 0
    
#     r = sum(cnt) / N
#     return r, rb, ce, cands, vt

def certifyRobustness_pixel(net, in_sets, in_datas, num_classes, veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, return_output=False, show=False):
    assert isinstance(net.layers[-1], PixelClassificationLayer), f"The network's last layer should be PixelClassificationLayer, but network has {net.layers[-1]}"
    assert len(in_sets) == len(in_datas), f"Inconsistent number of elements in in_sets and in_datas"
    start = time.perf_counter()
    N = len(in_sets)
    veri_set = []
    veri_time = np.zeros(N)
    out_sets = []

    num_rbPix  = np.zeros(N) # number of robust pixels
    num_unkPix = np.zeros(N) # number of unknown pixels
    num_misPix = np.zeros(N) # number of missclassified pixels
    num_attPix = np.zeros(N) # number of attacked pixels
    riou = np.zeros(N) # rate of Jaccard similarity coefficient score; iou = Jaccard similarity index (IoU)
	
    num_pixels = np.prod(in_datas[0].shape)

    UNK_PIX = num_classes
    MIS_PIX = num_classes + 1
    
    for i in range(N):
        veri_image, veri_time[i], _, O,  gr_pix_id = certifyPixelRobustness_single_input(net, in_sets[i], in_datas[i], veriMethod, reachMethod, lp_solver, pool, RF, DR, show)
        veri_set.append(veri_image)
        
        num_attPix[i] = in_sets[i].geNumAttackedPixels()
        num_misPix[i] = (veri_image == MIS_PIX).sum()
        num_unkPix[i] = (veri_image == UNK_PIX).sum()
        num_rbPix[i] = num_pixels - (num_misPix[i] + num_unkPix[i])
        iou = jaccard_score(veri_image.ravel(), gr_pix_id.ravel(), average=None) #labels = np.arange(num_classes+2)
        riou[i] = iou.sum()/(num_classes+2)

        if return_output:
            out_sets.append(O)

    avg_numRb = num_rbPix.sum() / N
    avg_numUnk = num_unkPix.sum() / N
    avg_numMis = num_misPix.sum() / N
    avg_numAtt = num_attPix.sum() / N
    avg_riou = riou.sum() / N
    avg_rv = (num_rbPix / num_pixels) / N
    avg_rs = (num_misPix + num_unkPix / num_attPix).sum() / N
    avg_vt = veri_time.sum() / N
    avg_data = [avg_numRb, avg_numUnk, avg_numMis, avg_numAtt, avg_riou, avg_rv, avg_rs, avg_vt]

    vt_total = time.perf_counter() - start 
    return veri_image, veri_time, vt_total, out_sets, avg_data


def certifyRobustness(net, inputs, labels=None, veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, return_output=False, show=False):
    
    start = time.perf_counter()
    N = len(inputs)
    RB = np.zeros(N)
    VT = np.zeros(N)
    Y = []
   
    if not ((isinstance(inputs, list) and len(inputs) > 1) or (isinstance(labels, np.ndarray) and len(labels) > 1)): 
        if isinstance(inputs, list):
            inputs = inputs[0]
            if labels != None:
                labels = labels[0]
        RB, VT, Y = certifyRobustness_single_input(net, inputs, labels, veriMethod, reachMethod, lp_solver, pool, RF, DR, show)
        vt_total = time.perf_counter() - start 
        return RB, VT, vt_total, Y

    if labels is None:
        labels = []
        for input in inputs:
            y = net.evaluate(input)
            labels.append(y.argmax())

    else:
        assert isinstance(labels, list) or isinstance(labels, np.ndarray), \
        f"labels should be a list containing arg_max(F(x))"

    for i, (input, label) in enumerate(zip(inputs, labels)):
        RB[i], VT[i], O = certifyRobustness_single_input(net, input, label, veriMethod, reachMethod, lp_solver, pool, RF, DR, show)
        if return_output:
            Y.append(O)

    vt_total = time.perf_counter() - start 
    return RB, VT, vt_total, Y

def certifyPixelRobustness_single_input(net, in_set, in_data, veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):

    start = time.perf_counter()

    gr_pix_id = net.evaluate(in_data).squeeze(axis=2)

    # Compute output reachable sets
    if veriMethod == 'BFS':
        # outputSet, totalReachTime, pixel_classification
        Y, VT, pixel_labels = reachBFS(net=net, inputSet=in_set, reachMethod=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
    else:
        raise Exception('other verification methods is not yet implemented, i.e. DFS')

    # num_classes = shape[2]
    classes = net.layers[-1].classes
    h, w = pixel_labels.shape

    ver_im = (classes + 1)*np.ones([h, w]) # initially define incorrect classification (unrobust / misslcassified pixels)
    ver_im[pixel_labels == classes] = classes # unknown pixles
    eq = pixel_labels == gr_pix_id 
    ver_im[eq] = pixel_labels[eq] # robust pixels / correctly classified pixels

    vt_total = time.perf_counter() - start  
    return ver_im, VT, vt_total, Y, gr_pix_id


def certifyRobustness_single_input(net, in_set, label=None, veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
    crown_modes = ('crown', 'alpha-crown', 'alpha-beta-crown')
    rm_lower = str(reachMethod).lower() if reachMethod is not None else ''
    vm_lower = str(veriMethod).lower() if veriMethod is not None else ''
    crown_mode = rm_lower if rm_lower in crown_modes else (vm_lower if vm_lower in crown_modes else None)
    if crown_mode is not None:
        return certifyRobustness_single_input_CROWN(
            net, in_set, label=label, mode=crown_mode,
            lp_solver=lp_solver, show=show
        )

    if label is None:
        y = net.evaluate(in_set)
        max_id = np.array([y.argmax()])
    else:
        max_id = np.array([label]).reshape(-1)

    start = time.perf_counter()

    # Compute output reachable sets
    if veriMethod == 'BFS':
        Y, _ = reachBFS(net=net, inputSet=in_set, reachMethod=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
    else:
        raise Exception('other verification methods is not yet implemented, i.e. DFS')
    
    rb = 0
    max_cands = Y.get_max_point_cadidates()
    if len(max_cands) == 1:
        if max_cands == max_id:
            rb = 1
    else:
        a = [max_cands == max_id][0]
        if sum(a) == 0:
            rb = 0
        elif lp_solver == 'estimate':
            max_cands = np.delete(max_cands, max_cands == max_id)
            if hasattr(Y, 'get_c_V'):
                c_vec, V_mat = Y.get_c_V()
            elif isinstance(Y, ImageStar):
                V_flat = Y.V.reshape(Y.num_pixel, Y.num_pred + 1)
                c_vec, V_mat = V_flat[:, 0], V_flat[:, 1:]
            else:
                c_vec, V_mat = Y.V[:, 0], Y.V[:, 1:]
            c_true = int(max_id[0])
            V_csr = V_mat.tocsr() if sp.issparse(V_mat) else None
            rb = 1
            for cand in max_cands:
                cand = int(cand)
                c_diff = float(c_vec[cand] - c_vec[c_true])
                if V_csr is not None:
                    v_diff = (V_csr[[cand]] - V_csr[[c_true]]).toarray().ravel()
                else:
                    v_diff = np.asarray(V_mat[cand] - V_mat[c_true]).ravel()
                pos = np.maximum(v_diff, 0.0)
                neg = np.minimum(v_diff, 0.0)
                ub_diff = c_diff + float(pos @ Y.pred_ub + neg @ Y.pred_lb)
                lb_diff = c_diff + float(pos @ Y.pred_lb + neg @ Y.pred_ub)
                if lb_diff > 0.0:
                    rb = 0
                    break
                if ub_diff >= 0.0:
                    rb = 2
        else:
            max_cands = np.delete(max_cands, max_cands == max_id)
            m = len(max_cands)

            for j in range(m):
                if Y.is_p1_larger_than_p2(max_cands[j], max_id[0], lp_solver=lp_solver):
                    rb = 2
                    break
                else:
                    rb = 1

    vt = time.perf_counter() - start  
    return rb, vt, Y


def certifyProbabilisticRobustness_single_input(
    net, in_set, label, veriMethod='BFS', reachMethod='approx',
    lp_solver='gurobi', pool=None, RF=0.0, DR=0,
    relu_approx_mode='analytic_gaussian', use_lp=False, show=False
):
    """
    Certify quantitative probabilistic robustness for a single classification input set
    (ProbStar, ProbSparseImageStar2DCSR, or ProbSparseImageStar2DCOO).
    Returns:
      res: dict with keys:
        'rb' (1=robust, 0=violated, 2=unknown),
        'status', 'p_min', 'p_max', 'p_hat', 'p_input',
        'cond_p_min', 'cond_p_max', 'cond_p_hat',
        'reach_time', 'verify_time', 'total_time', 'mem_bytes', 'num_pred'
      Y: output reachable set
    """
    t0 = time.perf_counter()
    crown_modes = ('crown', 'alpha-crown', 'alpha-beta-crown')
    rm_lower = str(reachMethod).lower() if reachMethod is not None else ''
    vm_lower = str(veriMethod).lower() if veriMethod is not None else ''
    crown_mode = rm_lower if rm_lower in crown_modes else (vm_lower if vm_lower in crown_modes else None)
    if crown_mode is not None:
        return certifyProbabilisticRobustness_single_input_CROWN(
            net, in_set, label, mode=crown_mode,
            lp_solver=lp_solver, show=show
        )

    Y, reach_time = reachBFS(
        net=net, inputSet=in_set, reachMethod=reachMethod,
        lp_solver=lp_solver, pool=pool, RF=RF, DR=DR,
        show=show, relu_approx_mode=relu_approx_mode
    )
    t_ver0 = time.perf_counter()
    spec_res = Y.certifyLabel(int(label), lp_solver=lp_solver, use_lp=use_lp)
    verify_time = time.perf_counter() - t_ver0
    total_time = time.perf_counter() - t0

    status = spec_res['status']
    if status == 'ROBUST':
        rb = 1
    elif status == 'VIOLATED':
        rb = 0
    else:
        rb = 2

    mem_bytes = Y.nbytes() if hasattr(Y, 'nbytes') else (Y.V.nbytes + Y.mu.nbytes + Y.Sig.nbytes)
    num_pred = getattr(Y, 'num_pred', getattr(Y, 'nVars', 0))

    res = dict(spec_res)
    res.update({
        'rb': rb,
        'reach_time': reach_time,
        'verify_time': verify_time,
        'total_time': total_time,
        'mem_bytes': int(mem_bytes),
        'num_pred': int(num_pred),
    })
    return res, Y


def certifyProbabilisticRobustness(
    net, inputs, labels, veriMethod='BFS', reachMethod='approx',
    lp_solver='gurobi', pool=None, RF=0.0, DR=0,
    relu_approx_mode='analytic_gaussian', use_lp=False,
    return_output=False, show=False
):
    """
    Certify quantitative probabilistic robustness over a single input set or list of input sets.
    """
    if not isinstance(inputs, list):
        lbl = labels[0] if isinstance(labels, (list, np.ndarray)) else labels
        res, Y = certifyProbabilisticRobustness_single_input(
            net, inputs, lbl, veriMethod=veriMethod, reachMethod=reachMethod,
            lp_solver=lp_solver, pool=pool, RF=RF, DR=DR,
            relu_approx_mode=relu_approx_mode, use_lp=use_lp, show=show
        )
        return res, (Y if return_output else None)

    results = []
    outputs = []
    for inp, lbl in zip(inputs, labels):
        res, Y = certifyProbabilisticRobustness_single_input(
            net, inp, int(lbl), veriMethod=veriMethod, reachMethod=reachMethod,
            lp_solver=lp_solver, pool=pool, RF=RF, DR=DR,
            relu_approx_mode=relu_approx_mode, use_lp=use_lp, show=show
        )
        results.append(res)
        if return_output:
            outputs.append(Y)
    return results, outputs


def certifyProbabilisticRobustness_pixel(
    net, in_sets, in_datas, num_classes, veriMethod='BFS', reachMethod='approx',
    lp_solver='estimate', pool=None, RF=0.0, DR=0,
    relu_approx_mode='analytic_gaussian', use_lp=False,
    return_output=False, show=False
):
    """
    Certify pixel-wise probabilistic robustness for a semantic segmentation network.
    For each input set, computes both qualitative pixel classification and
    quantitative per-pixel probability bounds [p_min(h,w), p_max(h,w)] and p_hat(h,w).
    """
    assert isinstance(net.layers[-1], PixelClassificationLayer), (
        f"Expected last layer to be PixelClassificationLayer, got {type(net.layers[-1])}"
    )
    if not isinstance(in_sets, list):
        in_sets = [in_sets]
    if not isinstance(in_datas, list):
        in_datas = [in_datas]

    results = []
    out_sets = []
    for in_set, in_data in zip(in_sets, in_datas):
        t0 = time.perf_counter()
        gr_pix_id = net.evaluate(in_data)
        if gr_pix_id.ndim == 3 and gr_pix_id.shape[2] == 1:
            gr_pix_id = gr_pix_id.squeeze(axis=2)
        Y, reach_time, pixel_labels = reachBFS(
            net=net, inputSet=in_set, reachMethod=reachMethod,
            lp_solver=lp_solver, pool=pool, RF=RF, DR=DR,
            show=show, relu_approx_mode=relu_approx_mode
        )
        t_ver0 = time.perf_counter()
        seg_res = Y.certifySegmentationPixels(gr_pix_id, lp_solver=lp_solver, use_lp=use_lp)
        verify_time = time.perf_counter() - t_ver0
        total_time = time.perf_counter() - t0
        seg_res.update({
            'pixel_labels': pixel_labels,
            'gr_pix_id': gr_pix_id,
            'reach_time': reach_time,
            'verify_time': verify_time,
            'total_time': total_time,
            'mem_bytes': int(Y.nbytes()) if hasattr(Y, 'nbytes') else 0,
            'num_pred': int(getattr(Y, 'num_pred', 0)),
        })
        results.append(seg_res)
        if return_output:
            out_sets.append(Y)
    return results, out_sets