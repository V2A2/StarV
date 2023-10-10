"""
Image Star Class
Sung Woo Choi, 08/10/2023

"""
import copy
import numpy as np
import polytope as pc
import glpk
import gurobipy as gp
from gurobipy import GRB
from StarV.set.star import Star

# np.set_default_dtype(np.float64)

class ImageStar(object):
    """
        ImageStar Class for reachability
        date: 08/10/2023
        Representation of a ImageStar
        ===========================================================================================================================
        ImageStar set defined by

        Channel First Format
        H W C N

        ===========================================================================================================================
    """


    def __init__(self, *args):
        """
            Key Attributes:
            V = [] #
                c = V[:, :, :, 0] : anchor image
                X = V[:, :, :, 1:] : generator image
            C = [] # linear constraints matrix of the predicate variables
            d = [] # linear constraints vector of the predicate variables

            num_pred = 0 # number of predicate variables
            pred_lb = [] # lower bound of predicate variables
            pred_ub = [] # upper bound of predicate variables

            height = 0 # height of the image
            width = 0 # width of the image
            num_channel = 0 # number of channels of the image
            num_pred = 0 # number of predicate variables
            num_pixel = 0 # number of pixels in image
        """

        len_ = len(args)

        if len_ == 5:
            [V, C, d, pred_lb, pred_ub] = copy.deepcopy(args)

            assert isinstance(V, np.ndarray), \
            'error: basis matrix should be a numpy array'
            assert isinstance(pred_lb, np.ndarray), \
            'error: lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray), \
            'error: upper bound vector should be a 1D numpy array'

            if len(d) > 0:
                assert isinstance(C, np.ndarray), \
                'error: a linear constraint matrix should be a numpy array'
                assert isinstance(d, np.ndarray), \
                'error: a linear constraint vector should be a 1D numpy array'
                assert d.ndim == 1, \
                'error: a linear constraint vector should be a 1D numpy array'
                assert C.shape[0] == d.shape[0], \
                'error: inconsistency between lienar constraints matrix and linear constraints vector'
                assert C.shape[1] == pred_lb.shape[0], \
                'error: inconsistent number of predicatve variables between linear constratints matrix and predicate bound vectors'

            assert len(pred_lb.shape) == 1, \
            'error: lower bound vector should be a 1D numpy array'
            assert len(pred_ub.shape) == 1, \
            'error: upper bound vector should be a 1D numpy array'
            assert pred_ub.shape[0] == pred_lb.shape[0], \
            'error: inconsistent number of predicate variables between predicate lower- and upper-boud vectors'
            
            if V.ndim == 4:
                self.num_channel, self.height, self.width = V.shape[:3]

            elif V.ndim > 4:
                raise Exception('error: invalid independent basis matrix')
            
            else:
                raise Exception('error: invalid independent basis matrix')

                # need to clarify if A.ndim is 3 or 2
                # if V.ndim == 2:
                #     V = V[:, :, np.newaxis]
                # self.height, self.width, self.num_channel = V.shape
                # num_pixels = self.height * self.width * self.num_channel
                # V = np.diag(A.flatten())

                # self.num_pred = num_pixels
                # V = V.reshape(self.height, self.width, self.num_channel, self.num_pred)
                # V = np.insert(V, 0, 0, axis=3)

            self.V = V
            self.C = C
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub

            if len(d) > 0:
                self.num_pred = self.C.shape[1]
            else:
                self.num_pred = V.shape[3] - 1

            self.num_pixel = self.height * self.width * self.num_channel



        
        # elif len_ == 2:
        #     [lb, ub] = copy.deepcopy(args)

        #     assert isinstance(lb, np.ndarray), \
        #     'error: lower bound vector should be a numpy array'
        #     assert isinstance(ub, np.ndarray), \
        #     'error: upper bound vector should be a numpy array'

        #     assert lb.shape == ub.shape, \
        #     'error: inconsistency between lower bound image and upper bound image'

        #     if (ub < lb).any():
        #         raise Exception(
        #             'error: the upper bounds must not be less than the lower bounds for all dimensions')

        #     lb = lb.type(np.float64)
        #     ub = ub.type(np.float64)

        #     img_shape = lb.shape
        #     img_dim = len(img_shape)

        #     lb = lb.flatten()
        #     ub = ub.flatten()
        #     dim = lb.shape[0]
        #     nv = int(sum(ub > lb))

        #     c = 0.5 * (lb + ub)
        #     if dim == nv:
        #         v = np.diag(0.5 * (ub - lb))
        #     else:
        #         v = np.zeros((dim, nv+1), dtype=np.float64)
        #         j = 1
        #         for i in range(dim):
        #             if ub[i] > lb[i]:
        #                 v[i, j] = 0.5 * (ub[i] - lb[i])
        #                 j += 1
        #     V = np.column_stack([c, v])

        #     self.num_pred = dim

        #     if img_dim == 3:
        #         img_shape = img_shape + (self.num_pred+1, )

        #     elif img_dim == 2:
        #         img_shape = img_shape + (1, self.num_pred+1)

        #     elif img_dim == 1:
        #         img_shape = img_shape + (1, 1, self.num_pred+1)

        #     self.V = V.reshape(img_shape)
        #     self.C = np.empty([0, 0])
        #     self.d = np.empty([0])
        #     self.pred_lb = -np.ones(dim)
        #     self.pred_ub = np.ones(dim)
        #     self.height, self.width, self.num_channel = img_shape[0:3]
        #     self.num_pixel = self.height * self.width * self.num_channel

        elif len_ == 2:
            [lb, ub] = copy.deepcopy(args)

            assert isinstance(lb, np.ndarray), \
            'error: lower bound vector should be a numpy array'
            assert isinstance(ub, np.ndarray), \
            'error: upper bound vector should be a numpy array'
            assert lb.shape == ub.shape, \
            'error: inconsistency between lower bound image and upper bound image'
            assert lb.ndim <= 3, \
            'error: lower and upper bound vectors should be less than 4D tensor'

            if (ub < lb).any():
                raise Exception(
                    'error: the upper bounds must not be less than the lower bounds for all dimensions')
            
            self.dtype = lb.dtype
            img_shape = lb.shape
            img_dim = lb.ndim

            lb = lb.flatten()
            ub = ub.flatten()
            dim = lb.shape[0]
            nv = int(sum(ub > lb))

            V = np.zeros((dim, nv+1), dtype = self.dtype)
            j = 1
            for i in range(dim):
                if ub[i] > lb[i]:
                    V[i, j] = 1
                    j += 1

            self.num_pred = dim

            if img_dim == 3:
                img_shape = img_shape + (self.num_pred+1, )

            elif img_dim == 2:
                img_shape = img_shape + (1, self.num_pred+1)

            elif img_dim == 1:
                img_shape = img_shape + (1, 1, self.num_pred+1)

            self.V = V.reshape(img_shape)
            self.C = np.empty([0, 0])
            self.d = np.empty([0])
            self.pred_lb = lb
            self.pred_ub = ub
            self.height, self.width, self.num_channel = img_shape[0:3]
            self.num_pixel = self.height * self.width * self.num_channel

        elif len_ == 0:
            self.V = np.ndarray([0, 0])
            self.C = np.empty([0, 0])
            self.d = np.empty([0])
            self.pred_lb = np.empty([0])
            self.pred_ub = np.empty([0])
            self.height = 0
            self.width = 0
            self.num_channel = 0
            self.num_pixel = 0
            self.num_pred = 0

        else:
            raise Exception(
                'error: invalid number of input arguments (should be 0, 2, 6)')
    
    def __str__(self, channel_first=False):
        if channel_first:
            print('ImageStar Set (channel first):')
            print('V: \n')
            print(self.V.transpose([3,2,0,1])) # N C H W
        else:
            print('ImageStar Set:')
            print('V: \n{}'.format(self.V))
        print('C: \n{}'.format(self.C))
        print('d: {}'.format(self.d))
        print('pred_lb: {}'.format(self.pred_lb))
        print('pred_ub: {}'.format(self.pred_ub))

        print('height: {}'.format(self.height))
        print('width: {}'.format(self.width))
        print('num_channel: {}'.format(self.num_channel))
        print('num_pred: {}'.format(self.num_pred))
        return '\n'

    def __repr__(self):
        print('ImageStar Set:')
        print('V: {}, {}'.format(self.V.shape, self.V.dtype))
        print('C: {}, {}'.format(self.C.shape, self.C.dtype))
        print('d: {}, {}'.format(self.d.shape, self.d.dtype))
        print('pred_lb: {}, {}'.format(self.pred_lb.shape, self.pred_lb.dtype))
        print('pred_ub: {}, {}'.format(self.pred_ub.shape, self.pred_ub.dtype))

        print('height: {}'.format(self.height))
        print('width: {}'.format(self.width))
        print('num_channel: {}'.format(self.num_channel))
        print('num_pred: {}'.format(self.num_pred))
        return '\n'
    
    def __len__(self):
        return 1
    
    def c(self, index=None):
        """Get anchor image of ImageStar"""
        if index is None:
            return copy.deepcopy(self.V[:, :, :, 0])
        else:
            return copy.deepcopy(self.V[:, :, :, 0].flatten()[index])
    
    def X(self, index=None):
        """Get generator images of ImageStar"""
        if index is None:
            return copy.deepcopy(self.V[:, :, :, 1:])
        else:
            return copy.deepcopy(self.V[:, :, :, index+1])
    
    def getMin(self, index, lp_solver='gurobi'):
        S = self.toStar()
        return S.getMin(index=index, lp_solver=lp_solver)

    def getMax(self, index, lp_solver='gurobi'):
        S = self.toStar()
        return S.getMax(index=index, lp_solver=lp_solver)
    
    def getMins(self, map, lp_solver='gurobi'):
        S = self.toStar()
        n = len(map)
        xmin = np.zeros(n)
        for i in range(n):
            xmin[i] = S.getMin(index=map[i], lp_solver=lp_solver)
        return xmin

    def getMaxs(self, map, lp_solver='gurobi'):
        S = self.toStar()
        n = len(map)
        xmax = np.zeros(n)
        for i in range(n):
            xmax[i] = S.getMax(index=map[i], lp_solver=lp_solver)
        return xmax
    
    def estimateRange(self, index):
        """Estimate the minimum and maximum values of a state x[index]"""
        # supper slow;  Indexing into tensor order of magnitude slower than numpy
        
        p = self.num_pred - self.num_pixel
        
        l = self.pred_lb[p:]
        u = self.pred_ub[p:]

        X = self.X(index).reshape(self.num_pixel)
        pos_f = np.maximum(X, 0.0)
        neg_f = np.minimum(X, 0.0)

        xmin = self.c(index) + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        xmax = self.c(index) + np.matmul(neg_f, l) + np.matmul(pos_f, u)
        return xmin, xmax

    def estimateRanges(self):
        """Estimate the lower and upper bounds of x"""
        
        p = self.num_pred - self.num_pixel
        
        l = self.pred_lb[p:]
        u = self.pred_ub[p:]

        X = self.X().reshape(self.num_pixel, -1)
        pos_f = np.maximum(X, np.ndarray(0))
        neg_f = np.minimum(X, np.ndarray(0))

        xmin = self.c().flatten() + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        xmax = self.c().flatten() + np.matmul(neg_f, l) + np.matmul(pos_f, u)

        shape = self.height, self.width, self.num_channel
        return xmin.reshape(shape), xmax.reshape(shape)

    def getRange(self, index, lp_solver='gurobi'):
        """Get the lower and upper bounds of x[index]"""

        if lp_solver == 'estimate':
            return self.estimate(index)
        else:
            l = self.getMin(index)
            u = self.getMax(index)
            return l, u    

    def getRanges(self, lp_solver='gurobi', RF=0.0, layer=None, delta=0.98):
        """Get the lower and upper bound vectors of the state
            Args:
                lp_solver: linear programming solver. e.g.: 'gurobi', 'estimate', 'linprog'
                RF: relaxation factor \in [0.0, 1.0]
        """
        
        shape = self.height, self.width, self.num_channel

        if RF == 1.0:
            l, u = self.estimateRanges()
        
        elif RF == 0.0:
            if lp_solver == 'estimate':
                l, u = self.estimateRanges()
            else:
                l = self.getMins(np.arange(self.num_pixel))
                u = self.getMaxs(np.arange(self.num_pixel))
            
        else:
            assert RF > 0.0 and RF <= 1.0, \
            'error: relaxation factor should be greater than 0.0 but less than or equal to 1.0'
            l, u = self.estimateRanges()
            
            if layer in ['logsig', 'tansig']:
                n1 = round(1 - RF) * self.num_pixel

                midx = np.argsort((u - l))[::-1]
                midb = np.argwhere((l[midx] >= -delta) & (u[midx] <= delta))
                
                n2 = n1
                check = midb.flatten().shape[0]
                if n2 > check:
                    n2 = check

                mid = midx[midb[0:n2]]
                l1 = self.getMins(mid)
                u1 = self.getMaxs(mid)
                l[mid] = l1
                u[mid] = u1

            elif  layer in ['poslin', 'relu']:
                map = np.argwhere((l < 0) & (u > 0))

                n1 = round(1 - RF) * len(map)

                
                area = 0.5*abs(u[map]*l[map])
                midx = np.argsort(area)[::-1]

                mid = midx[0:n1]
                l1 = self.getMins(mid)
                u1 = self.getMaxs(mid)
                l[mid] = l1
                u[mid] = u1

            else:
                n1 = round(1 - RF) * self.num_pixel

                midx = np.argsort((u - l))[::-1]
                mid = midx[0:n1]
                l1 = self.getMins(mid)
                u1 = self.getMaxs(mid)
                l[mid] = l1
                u[mid] = u1
      
        return l.reshape(shape), u.reshape(shape)

    def isEmptySet(self, lp_solver='gurobi'):
        """Check if a SparseStar is an empty set"""
        res = False
        try:
            self.getMin(0, lp_solver=lp_solver)
        except Exception:
            res = True
        return res
    
    def toStar(self):
        """Convert ImageStarTensor class to Star class"""
        V = self.V.reshape(self.height * self.width * self.num_channel, -1)
        return Star(V, self.C, self.d, self.pred_lb, self.pred_ub)

    def toSparseStar(self):
        """Convert ImageStar class to SparseStar class"""
        pass

    @staticmethod
    def inf_attack(data, epsilon=0.01, data_type='default', dtype = 'float64'):
        """Generate a SparseStar set by infinity norm attack on input dataset"""

        assert isinstance(data, np.ndarray), \
        'error: the data should be a 1D numpy array'

        if dtype =='float64':
            data = data.astype(np.float64)
            data = data.astype(np.float64)
        else:
            data = data.astype(np.float32)
            data = data.astype(np.float32)

        lb = data - epsilon
        ub = data + epsilon

        if data_type == 'image':
            lb[lb < 0] = 0
            ub[ub > 1] = 1

        return ImageStar(lb, ub)
    
    @staticmethod
    def rand_bounds(in_height, in_width, in_channel):
        """Generate a random SparStar by random bounds"""

        lb = -np.random.rand(in_height, in_width, in_channel)
        ub = np.random.rand(in_height, in_width, in_channel)
        return ImageStar(lb, ub)
