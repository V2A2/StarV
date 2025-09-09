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

import copy
import time
import pickle
import numpy as np
from sklearn.metrics import jaccard_score
from StarV.net.network import NeuralNetwork
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.layer.PixelClassificationLayer import PixelClassificationLayer

class Certifier(object):
    """
        Certifier Class

        Properties: (Certification Settings)

        @lp_solver: lp solver: 'gurobi' (default), 'glpk', 'linprog'
        @method: ceritification method: BFS "bread-first-search" or DFS "depth-first-search"
        @n_processes: number of processes used for verification

    """


def reachBFS(net, inputSet, reachMethod='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
    """ Verification of neural network with over-apporoximation method.
        Compute Reachable set layer-by-layer
    """

    assert isinstance(net, NeuralNetwork), 'error: first input should be a NeuralNetwork object'
    assert isinstance(inputSet, list) or isinstance(inputSet, SparseStar) \
        or isinstance(inputSet, Star) or isinstance(inputSet, ImageStar) \
        or  isinstance(inputSet, SparseImageStar2DCOO) or isinstance(inputSet, SparseImageStar2DCSR), \
        'error: second input should be a list of Star/ImageStar/ProbStar/SparseStar/SparseImageStar2DCOO/SparseImageStar2DCSR sets or a single set'

    # compute reachable set
    reachTime = []
    In = inputSet

    # For semantic segmentation neural network
    if isinstance(net.layers[-1], PixelClassificationLayer):
        for i in range(net.n_layers-1):
            if show:
                print(f"\nComputing {net.layers[i].__class__.__name__} layer {i} reachable set...")
            
            start = time.perf_counter()
            In = net.layers[i].reach(In, method=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show)
            vt = time.perf_counter() - start

            # reachSet.append(In)
            reachTime.append(vt)

            if show:
                print('Number of stars/sparsestars: {}'.format(len(In)))
                if reachMethod == 'approx':
                    print(f"Number of predicate variables: {In.num_pred}")
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
    ver_im[ver_im < classes] = classes # unknown pixles
    eq = pixel_labels == gr_pix_id 
    ver_im[eq] = pixel_labels[eq] # robust pixels / correctly classified pixels

    vt_total = time.perf_counter() - start  
    return ver_im, VT, vt_total, Y, gr_pix_id


def certifyRobustness_single_input(net, in_set, label=None, veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):

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
        else:
            max_cands = np.delete(max_cands, max_cands == max_id)
            m = len(max_cands)

            for j in range(m):
                if Y.is_p1_larger_than_p2(max_cands[j], max_id[0]):
                    rb = 2
                    break
                else:
                    rb = 1

    vt = time.perf_counter() - start  
    return rb, vt, Y