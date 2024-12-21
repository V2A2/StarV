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

        Pixel Labels: [0, ..., n, n+1], n is used for unknown case, n+1 is used for unrobust case
    """

    def __init__(
            self,
            num_pix_classes,
    ):
        self.classes = num_pix_classes
        
    def evaluate(self, input):
        shape = input.shape
        return np.argmax(input, axis=2)

    def reach_single(self, input):
        if isinstance(input, ImageStar):
            shape = input.V.shape[:3]
        else:
            shape = input.shape
        
        h, w = shape[:2]
        lb, ub = input.estimateRanges()
        lb = lb.reshape(shape[:3])
        ub = ub.reshape(shape[:3])

        max_lb = np.max(lb, axis=2)
        pix_label = np.empty([h, w])
        for i in range(h):
            for j in range(w):
                cand = np.argwhere(ub[i, j, :] >= max_lb[i, j]).ravel()
                if len(cand) > 1:
                    cand = self.classes # unkown case; multiple classification for a single pixel
                pix_label[i, j] = cand
        return pix_label
        
    def reach_relax_single(self, input, RF=0.0, method='area', lp_solver='gurobi', show=False):
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
            ul = np.hstack(ub, np.abs(lb))
            midx = np.argsort(ul)
            midx1 = midx[-2*n1:]
            ub_idx = midx1[midx1 <= N]
            lb_idx = midx1[midx1 > N] - N

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

        max_lb = np.max(lb, axis=2)
        pix_label = np.empty([h, w])
        for i in range(h):
            for j in range(w):
                cand = np.argwhere(ub[i, j, :] >= max_lb[i, j]).ravel()
                if len(cand) > 1:
                    cand = self.classes  # unkown case; multiple classification for a single pixel
                pix_label[i, j] = cand
        return pix_label


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