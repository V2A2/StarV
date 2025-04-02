"""
Reshape Layer Class
Sung Woo Choi, 03/31/2025
"""

import numpy as np
import scipy.sparse as sp
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR

class ReshapeLayer(object):
    """ReshapeLayer class
        Author: Sung Woo Choi
        Date: 03/31/2025
    """

    def __init__(self, out_shape, to_sparse=False, to_dense=False):
        self.out_shape = out_shape
        self.to_sparse = to_sparse
        self.to_dense = to_dense

    def __str__(self):
        print('Reshape Layer')
        print(f'out_shape: {self.out_shape}')
        print(f'to_sparse: {self.to_sparse}')
        print(f'to_dense: {self.to_dense}')
        return '\n'
    
    def info(self):
        print(self)

    def evaluate(self, x):
        """ reshape input x that has batch-last shape"""
        p = np.prod(x.shape[:-1])
        assert p == np.prod(self.out_shape), f'error: cannot reshape ImageStar of size {p} into shape {self.out_shape}'
        b = x.shape[-1]
        return x.reshape(self.out_shape + (b))
    
    def reachSingleInput(self, In):
        if isinstance(In, ImageStar):
            p = In.num_pixel
            assert p == np.prod(self.out_shape), f'error: cannot reshape ImageStar of size {p} into shape {self.out_shape}'
            V = In.V.reshape(self.out_shape + (In.num_pred+1, ))
            return ImageStar(V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        elif isinstance(In, SparseImageStar2DCOO):
            p = np.prod(In.shape)
            assert p == np.prod(self.out_shape), f'error: cannot reshape SparseImageStar2DCOO of size {p} into shape {self.out_shape}'
            
            if In.c is None:
                if self.to_sparse:
                    V = In.V.reshape(-1, In.num_pred+1)
                    c1 = V[:, 0].ravel()
                    V1 = sp.coo_array(V[:, 1:])
                    return SparseImageStar2DCOO(c1, V1, In.C, In.d, In.pred_lb, In.pred_ub, self.out_shape)
                V = In.V.reshape(self.out_shape + (In.num_pred+1, ))
                return SparseImageStar2DCOO(V, In.C, In.d, In.pred_lb, In.pred_ub, self.out_shape)
            
            if self.to_dense:
                c1 = In.c.reshape(self.out_shape + (1,))
                V1 = In.V.toarray().reshape(self.out_shape + (In.num_pred, ))
                V = np.concatenate([c1, V1], axis=3)
                return SparseImageStar2DCOO(V, In.C, In.d, In.pred_lb, In.pred_ub, self.out_shape)
            
            return SparseImageStar2DCOO(In.c, In.V, In.C, In.d, In.pred_lb, In.pred_ub, self.out_shape)

        elif isinstance(In, SparseImageStar2DCSR):
            p = np.prod(In.shape)
            assert p == np.prod(self.out_shape), f'error: cannot reshape SparseImageStar2DCSR of size {p} into shape {self.out_shape}'
            
            if In.c is None:
                if self.to_sparse:
                    V = In.V.reshape(-1, In.num_pred+1)
                    c1 = V[:, 0].ravel()
                    V1 = sp.csr_array(V[:, 1:])
                    return SparseImageStar2DCSR(c1, V1, In.C, In.d, In.pred_lb, In.pred_ub, self.out_shape)
                V = In.V.reshape(self.out_shape + (In.num_pred+1, ))
                return SparseImageStar2DCSR(V, In.C, In.d, In.pred_lb, In.pred_ub, self.out_shape)
            
            if self.to_dense:
                c1 = In.c.reshape(self.out_shape + (1,))
                V1 = In.V.toarray().reshape(self.out_shape + (In.num_pred, ))
                V = np.concatenate([c1, V1], axis=3)
                return SparseImageStar2DCSR(V, In.C, In.d, In.pred_lb, In.pred_ub, self.out_shape)
            
            return SparseImageStar2DCSR(In.c, In.V, In.C, In.d, In.pred_lb, In.pred_ub, self.out_shape)

        raise Exception('unsupported input set, {}'.format(In.__class__.__name__))

    def reach(self, in_sets, method=None, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """ main reshape layer method 
            Args:
                @in_sets: a list of input sets or a single input set

            Return:
                @S: reshaped reachable sets
        """

        if isinstance(in_sets, list):
            F = []
            for i in range(1, len(in_sets[i])):
                F.append(self.reachSingleInput(in_sets[i]))
            return F

        else:
            return self.reachSingleInput(in_sets)