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
Pixel Classification Layer Class
Sung Woo Choi, 08/23/2024
"""

import numpy as np
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR

class PixelClassificationLayer(object):
    """ PixelClassificationLayer Class
        Author: Sung Woo Choi
        Date: 08/23/2024

        Pixel Labels: [0, ..., n, n+1], n is used for unknown case, n+1 is used for unrobust case (missclassification, where a pixel has more than one classification)
    """

    def __init__(
            self,
            num_pix_classes,
            threshold = None
    ):
        self.classes = num_pix_classes
        self.threshold = threshold
        
    def evaluate(self, input):
        if input.shape[2] > 1:
            shape = input.shape
            return np.argmax(input, axis=2)
        
        elif input.shape[2] == 1 and self.threshold is not None:
            input[input >= self.threshold] = 1
            input[input < self.threshold] = 0
        return input

    def reach_single(self, input):
        if isinstance(input, ImageStar):
            shape = input.V.shape[:3]
        else:
            shape = input.shape
        
        h, w = shape[:2]
        lb, ub = input.estimateRanges()
        lb = lb.reshape(shape[:3])
        ub = ub.reshape(shape[:3])
        
        pix_labels = np.empty([h, w], dtype=np.int16)
        reach_classes = 2 if shape[2] == 1 and self.threshold is not None else self.classes
        pix_multiclass_labels = np.zeros([h, w, reach_classes], dtype=bool)

        if shape[2] == 1 and self.threshold is not None:
            for i in range(h):
                for j in range(w):
                    if lb[i, j, 0] >= self.threshold:
                        pix_labels[i, j] = 1
                        if reach_classes > 1:
                            pix_multiclass_labels[i, j, 1] = True
                    elif ub[i, j, 0] < self.threshold:
                        pix_labels[i, j] = 0
                        pix_multiclass_labels[i, j, 0] = True
                    else:
                        pix_labels[i, j] = self.classes + 1 # missclassification/unrobust case
                        pix_multiclass_labels[i, j, 0] = True
                        if reach_classes > 1:
                            pix_multiclass_labels[i, j, 1] = True

        else:
            # multi-class case
            max_lb = np.max(lb, axis=2)
            
            for i in range(h):
                for j in range(w):
                    cand = np.argwhere(ub[i, j, :] >= max_lb[i, j]).ravel()
                    pix_multiclass_labels[i, j, cand] = True
                    
                    if len(cand) != 1:
                        pix_labels[i, j] = self.classes + 1 # missclassification/unrobust case; multiple classification for a single pixel
                    else:
                        pix_labels[i, j] = cand[0]
                        
        results = {
            'pix_labels': pix_labels,
            'pix_multiclass_labels': pix_multiclass_labels,
        }
        return results
        
    def reach_relax_single(self, input, threshold=0.0, RF=0.0, method='area', lp_solver='gurobi', show=False):
        assert method in ['range', 'random', 'area', 'bound'], \
        f"Invalid relaxation method. Options: 'range', 'random', 'area', and 'bound'. Received {method}"
        
        if isinstance(input, ImageStar):
            shape = input.V.shape[:3]
        else:
            shape = input.shape

        h, w = shape[:2]
        lb, ub = input.estimateRanges()
        n1 = round((1 - RF) * len(lb)) # number of LP need to solve

        if  method == 'range':
            if show:
                print('Applying relaxation by range with RF = {}'.format(RF))
                print('(1 - {}) x {} = {} neurons are found by LP solver'.format(RF, len(lb), n1))
            midx = np.argsort(ub - lb) #ascending order
            map = midx[-n1:]
            lb[map] = input.getMins(map, lp_solver=lp_solver)
            ub[map] = input.getMaxs(map, lp_solver=lp_solver)

        elif method == 'random':
            if show:
                print('Applying relaxation by random with RF = {}'.format(RF))
                print('(1 - {}) x {} = {} neurons are found by LP solver'.format(RF, len(lb), n1))
            map = np.random.randint(0, len(ub), n1)
            lb[map] = input.getMins(map, lp_solver=lp_solver)
            ub[map] = input.getMaxs(map, lp_solver=lp_solver)

        elif method == 'area':
            if show:
                print('Applying relaxation by triangular area with RF = {}'.format(RF))
                print('(1 - {}) x {} = {} neurons are found by LP solver'.format(RF, len(lb), n1))
            area = 0.5 * (np.abs(ub) * np.abs(lb))
            midx = np.argsort(area) #ascending order
            map = midx[-n1:]
            lb[map] = input.getMins(map, lp_solver=lp_solver)
            ub[map] = input.getMaxs(map, lp_solver=lp_solver)

        elif method == 'bound':
            if show:
                print('Applying relaxation by bound with RF = {}'.format(RF))

            N = len(ub)
            ul = np.hstack([ub, np.abs(lb)])
            midx = np.argsort(ul)
            midx1 = midx[-2*n1:]
            ub_idx = midx1[midx1 < N]
            lb_idx = midx1[midx1 >= N] - N

            if show:
                print('Applying relaxation by bound')
                print(f"{len(ub_idx)} neurons for upper bound are found by LP solver")
                print(f"{len(lb_idx)} neurons for lower bound are found by LP solver")

            lb[lb_idx] = input.getMins(lb_idx, lp_solver=lp_solver)
            ub[ub_idx] = input.getMaxs(ub_idx, lp_solver=lp_solver)

        else:
            raise Exception('Unknown relaxation methods')

        lb = lb.reshape(shape[:3])
        ub = ub.reshape(shape[:3])
        pix_labels = np.empty([h, w], dtype=np.int16)
        reach_classes = 2 if shape[2] == 1 and self.threshold is not None else self.classes
        pix_multiclass_labels = np.zeros([h, w, reach_classes], dtype=bool)
        
        if shape[2] == 1 and self.threshold is not None:
            for i in range(h):
                for j in range(w):
                    if lb[i, j, 0] >= threshold:
                        pix_labels[i, j] = 1
                        if reach_classes > 1:
                            pix_multiclass_labels[i, j, 1] = True
                    elif ub[i, j, 0] < threshold:
                        pix_labels[i, j] = 0
                        pix_multiclass_labels[i, j, 0] = True
                    else:
                        pix_labels[i, j] = self.classes + 1 # missclassification/unrobust case
                        pix_multiclass_labels[i, j, 0] = True
                        if reach_classes > 1:
                            pix_multiclass_labels[i, j, 1] = True

        else:
            # multi-class case
            max_lb = np.max(lb, axis=2)
            for i in range(h):
                for j in range(w):
                    cand = np.argwhere(ub[i, j, :] >= max_lb[i, j]).ravel()
                    pix_multiclass_labels[i, j, cand] = True
                    if len(cand) != 1:
                        pix_labels[i, j] = self.classes + 1 # missclassification/unrobust case; multiple classification for a single pixel
                    else:
                        pix_labels[i, j] = cand[0]
                    
        results = {
            'pix_labels': pix_labels,
            'pix_multiclass_labels': pix_multiclass_labels,
        }
        return results


    def reach(self, inputSet, method='area', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        assert method in ['approx', 'range', 'random', 'area', 'bound', None], \
        f"Invalid relaxation method. Options: 'approx', 'range', 'random', 'area', and 'bound'. Received {method}"

        if isinstance(inputSet, list):
            outputs = []
            if method in ['approx', None]:
                for i in range(len(inputSet)):
                    outputs.append(self.reach_single(inputSet[i]))
            else:
                for i in range(len(inputSet)):
                    outputs.append(self.reach_relax_single(inputSet[i], method, RF, lp_solver))

            return outputs
        
        return self.reach_single(inputSet)