"""
Sparse Image Star Class
Sung Woo Choi, 08/09/2023

"""

# !/usr/bin/python3
import copy
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog
from scipy.linalg import block_diag
import glpk
import polytope as pc

from StarV.set.star import Star
from StarV.set.sparsestar import SparseStar


class SparseImageStar(object):
    """
        
        Sparse Image Star for reachability
        author: Sung Woo Choi
        date: 08/09/2022
        Representation of a SparseImageStar
        ==========================================================================
        Sparse Image Star set defined by


        Channel Last Format
        H W C N
        N:batch_size, H:input_img_height, W:input_img_width, C:no.of.channels 
        https://pytorch.org/blog/accelerating-pytorch-vision-models-with-channels-last-on-cpu/
        ==========================================================================  
    """

    def __init__(self, *args):
        """
            Key Attributes:
            A = []
            C = []
            d = []

            nVars = 0; % number of predicate variables
            nZVars = 0; % number of non-basis (dependent) predicate varaibles
            pred_lb = []; % lower bound of predicate variables
            pred_ub = []; % upper bound of predicate variables
            pred_depth = []; % depth of predicate varaibles
        """

        len_ = len(args)

        if len_ == 6:
            [A, C, d, pred_lb, pred_ub, pred_depth] = copy.deepcopy(args)

            assert isinstance(A, np.ndarray), \
            'error: basis matrix should be a 2D numpy array'
            assert isinstance(pred_lb, np.ndarray), \
            'error: lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray), \
            'error: upper bound vector should be a 1D numpy array'

            if len(d) > 0:
                assert isinstance(C, sp._csc.csc_matrix), \
                'error: non-zero basis matrix should be a scipy sparse csc matrix'
                assert isinstance(d, np.ndarray), \
                'error: non-zero basis matrix should be a 1D numpy array'
                assert d.ndim == 1, \
                'error: constraint vector should be a 1D numpy array'
                assert C.shape[0] == d.shape[0], \
                'error: inconsistency between constraint matrix and constraint vector'
                assert C.shape[1] == pred_lb.shape[0], \
                'error: inconsistent number of predicatve variables between constratint matrix and predicate bound vectors'

            assert len(pred_lb.shape) == 1, \
            'error: lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, \
            'error: upper bound vector should be a 1D numpy array'
            assert pred_ub.shape[0] == pred_lb.shape[0], \
            'error: inconsistent number of predicate variables between predicate lower- and upper-boud vectors'
            assert pred_lb.shape[0] == pred_depth.shape[0], \
            'error: inconsistent number of predicate variables between predicate bounds and predicate depth'

            
            if A.ndim == 4:
                self.height, self.width, self.num_channel = A.shape[:3]

            elif A.ndim > 4:
                raise Exception('error: invalid independent basis matrix')
            
            else:
                raise Exception('error: invalid independent basis matrix')

                # need to clarify if A.ndim is 3 or 2
                if A.ndim == 2:
                    A = A[:, :, np.newaxis]
                self.height, self.width, self.num_channel = A.shape
                num_pixels = self.height * self.width * self.num_channel
                A = np.diag(A.flatten())

                self.nVars = num_pixels
                A = A.reshape(self.height, self.width, self.num_channel, self.nVars)
                A = np.insert(A, 0, 0, axis=3)

            self.A = A
            self.C = C
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.pred_depth = pred_depth

            if len(d) > 0:
                self.nVars = self.C.shape[1]
                self.nZVars = self.C.shape[1] + 1 - A.shape[3]
            else:
                self.nVars = A.shape[3] - 1
                self.nZVars = 0
            self.num_pixels = self.width * self.height * self.num_channel
            
        elif len_ == 2:
            [lb, ub] = copy.deepcopy(args)

            assert isinstance(lb, np.ndarray), \
            'error: lower bound vector should be a numpy array'
            assert isinstance(ub, np.ndarray), \
            'error: upper bound vector should be a numpy array'

            assert lb.shape == ub.shape, \
            'error: inconsistency between lower bound image and upper bound image'

            if np.any(ub < lb):
                raise Exception(
                    'error: the upper bounds must not be less than the lower bounds for all dimensions')

            img_shape = lb.shape
            img_dim = len(img_shape)

            lb = lb.flatten()
            ub = ub.flatten()
            dim = lb.shape[0]
            nv = int(sum(ub > lb))

            c = 0.5 * (lb + ub)
            if dim == nv:
                v = np.diag(0.5 * (ub - lb))
            else:
                v = np.zeros((dim, nv))
                j = 0
                for i in range(self.dim):
                    if ub[i] > lb[i]:
                        v[i, j] = 0.5 * (ub[i] - lb[i])
                        j += 1
            A = np.column_stack([c, v])

            self.nVars = dim

            if img_dim == 3:
                img_shape = img_shape + (self.nVars+1, )
        
            elif img_dim == 2:
                img_shape = img_shape + (1, self.nVars+1)

            self.A = A.reshape(img_shape)
            self.C = sp.csc_matrix((0, dim))
            self.d = np.empty([0])
            self.pred_lb = -np.ones(dim)
            self.pred_ub = np.ones(dim)
            self.pred_depth = np.zeros(dim)
            self.width, self.height, self.num_channel = img_shape[:3]
            self.num_pixels = self.width * self.height * self.num_channel
            self.nVars
            self.nZVars = dim + 1 - self.A.shape[3]

        # elif len_ == 2:
        #     [lb, ub] = copy.deepcopy(args)
        #     lb = lb.astype('float64')
        #     ub = ub.astype('float64')

        #     assert isinstance(lb, np.ndarray), \
        #     'error: lower bound vector should be a numpy array'
        #     assert isinstance(ub, np.ndarray), \
        #     'error: upper bound vector should be a numpy array'

        #     assert lb.shape == ub.shape, \
        #     'error: inconsistency between lower bound image and upper bound image'

        #     if np.any(ub < lb):
        #         raise Exception(
        #             'error: the upper bounds must not be less than the lower bounds for all dimensions')

        #     img_shape = lb.shape
        #     img_dim = lb.ndim

        #     lb = lb.flatten()
        #     ub = ub.flatten()
        #     dim = lb.shape[0]
        #     nv = int(sum(ub > lb))

        #     A = np.zeros((dim, nv+1))
        #     j = 1
        #     for i in range(dim):
        #         if ub[i] > lb[i]:
        #             A[i, j] = 1
        #             j += 1
            
        #     self.nVars = dim

        #     if img_dim == 3:
        #         img_shape = img_shape + (self.nVars+1, )
        
        #     elif img_dim == 2:
        #         img_shape = img_shape + (1, self.nVars+1)
                
        #     self.A = A.reshape(img_shape)
        #     self.C = sp.csc_matrix((0, dim))
        #     self.d = np.empty([0])
        #     self.pred_lb = lb
        #     self.pred_ub = ub
        #     self.pred_depth = np.zeros(dim)
        #     self.width, self.height, self.num_channel = img_shape[0:3]
        #     self.nZVars = dim + 1 - self.A.shape[3]

        elif len_ == 0:
            self.A = np.empty([0, 0])
            self.C = sp.csc_matrix((0, 0))
            self.d = np.empty([0])
            self.pred_lb = np.empty([0])
            self.pred_ub = np.empty([0])
            self.pred_depth = np.empty([0])
            self.height = 0
            self.width = 0
            self.num_channel = 0
            self.num_pixels = 0
            self.nVars = 0
            self.nZVars = 0
            

        else:
            raise Exception(
                'error: invalid number of input arguments (should be 0, 2, 6)')
    
    def __str__(self, toDense=True):
        print('SparseImageStar Set:')
        print('A: \n{}'.format(self.A))
        if toDense:
            print('C_{}: \n{}'.format(self.C.getformat(), self.C.todense()))
        else:
            print('C: {}'.format(self.C))
        print('d: {}'.format(self.d))
        print('pred_lb: {}'.format(self.pred_lb))
        print('pred_ub: {}'.format(self.pred_ub))
        print('pred_depth: {}'.format(self.pred_depth))
        print('height: {}'.format(self.height))
        print('width: {}'.format(self.width))
        print('num_channel: {}'.format(self.num_channel))
        print('nVars: {}'.format(self.nVars))
        print('nZVars: {}'.format(self.nZVars))
        return '\n'

    def __repr__(self):
        print('SparseStar Set:')
        print('A: {}'.format(self.A.shape))
        print('C: {}'.format(self.C.shape))
        print('d: {}'.format(self.d.shape))
        print('pred_lb: {}'.format(self.pred_lb.shape))
        print('pred_ub: {}'.format(self.pred_ub.shape))
        print('pred_depth: {}'.format(self.pred_depth.shape))
        print('height: {}'.format(self.height))
        print('width: {}'.format(self.width))
        print('num_channel: {}'.format(self.num_channel))
        print('nVars: {}'.format(self.nVars))
        print('nZVars: {}'.format(self.nZVars))
        return '\n'
    
    def __len__(self):
        return 1
    
    def num_pixels(self):
        return self.height * self.width * self.num_channel
    
    def c(self, index=None):
        """Get center column vector of SparseStar"""
        if index is None:
            return copy.deepcopy(self.A[:, :, :, 0].reshape(-1, 1))
        else:
            return copy.deepcopy(self.A[index, :, :, 0].reshape(-1, 1))

    def X(self, row=None):
        """Get basis matrix of independent predicate variables"""
        mA = self.A.shape[1]
        if row is None:
            return copy.deepcopy(self.A[:, 1:mA])
        else:
            return copy.deepcopy(self.A[row, 1:mA])

    def V(self, row=None):
        """Get basis matrix"""
        mA = self.A.shape[1]
        if row is None:
            return copy.deepcopy(np.column_stack([np.zeros((self.dim, self.nZVars)), self.X()]))
        else:
            if isinstance(row, int) or isinstance(row, np.integer):
                return copy.deepcopy(np.hstack([np.zeros(self.nZVars), self.X(row)]))
            else:
                return copy.deepcopy(np.column_stack([np.zeros((len(row), self.nZVars)), self.X(row)]))
            
    def translation(self, v=None):
        """Translation of a sparse image star: S = self + v"""
        if v is None:
            return copy.deepcopy(self)

        if isinstance(v, np.ndarray):
            assert isinstance(v, np.ndarray) and v.ndim == 1, \
            'error: the translation vector should be a 1D numpy array or an integer'
            assert v.shape[0] == self.dim, \
            'error: inconsistency between translation vector and SparseStar dimension'

        elif isinstance(v, int) or isinstance(v, float):
            pass
        
        else:
            raise Exception('the translation vector, v, should a 1D numpy array, an integer, or a float')

        A = copy.deepcopy(self.A)
        A[:, :, :, 0] += v
        return SparseImageStar(A, self.C, self.d, self.pred_lb, self.pred_ub, self.pred_depth)

    def toStar(self):

        num_pixel = self.num_pixels()
        return None

    def toSparseStar(self):
        """Convert Sparse Image Star to Sparse Star"""
        num_pixel = self.num_pixels()
        A = self.A.reshape(num_pixel, -1)
        return SparseStar(A, self.C, self.d, self.pred_lb, self.pred_ub, self.pred_depth)
    
    @staticmethod
    def inf_attack(data, epsilon=0.01, data_type='default'):
        """Generate a SparseStar set by infinity norm attack on input dataset"""

        assert isinstance(data, np.ndarray), \
        'error: the data should be a 1D numpy array'

        lb = data - epsilon
        ub = data + epsilon

        if data_type == 'image':
            lb[lb < 0] = 0
            ub[ub > 1] = 1
            return SparseImageStar(lb, ub)

        return SparseImageStar(lb, ub)

    @staticmethod
    def rand_bounds(in_height, in_width, in_channel):
        """Generate a random SparStar by random bounds"""

        lb = -np.random.rand(in_height, in_width, in_channel)
        ub = np.random.rand(in_height, in_width, in_channel)
        return SparseImageStar(lb, ub)

if __name__ == "__main__":
    from IPython.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))

    np.set_printoptions(edgeitems=30, linewidth=100000, 
        formatter=dict(float=lambda x: "%.3g" % x))

    # IM = np.random.rand(2,2,3)
    IM = np.array([
            [0.1683,    0.3099,    0.2993],
            [0.8552,    0.1964,    0.3711],
            [0.2749,    0.3069,    0.1244],
            [0.8106,    0.0844,    0.1902],
    ])
    print(IM)

    SparseImageStar = SparseImageStar.inf_attack(data=IM, epsilon=0.01, data_type='image')
    print(SparseImageStar)

    print(SparseImageStar.A.shape)

    print(SparseImageStar.A[:, :, 0, 0])