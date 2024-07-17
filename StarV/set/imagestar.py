"""
Image Star Class
Sung Woo Choi, 08/10/2023

"""
import copy
import torch
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.linalg import block_diag
import polytope as pc
import glpk
import gurobipy as gp
from gurobipy import GRB
from StarV.set.star import Star

GUROBI_OPT_TOL = 1e-6

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

    def __init__(self, *args, copy_=True):
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
            if copy_ is True:
                [V, C, d, pred_lb, pred_ub] = copy.deepcopy(args)
            else:
                [V, C, d, pred_lb, pred_ub] = args

            assert isinstance(V, np.ndarray), \
            'error: basis matrix should be a numpy array'
            assert isinstance(pred_lb, np.ndarray), \
            'error: lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray), \
            'error: upper bound vector should be a 1D numpy array'

            if d.size > 0:
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
            
            # if V.ndim == 4:
            #     self.height, self.width, self.num_channel = V.shape[:3]
            
            # elif V.ndim == 3:
            #     self.height, self.width = V.shape[:2]
            #     self.num_channel = 1

            # elif V.ndim == 2:
            #     self.height = V.shape[0]
            #     self.width = 1
            #     self.num_channel = 1

            # elif V.ndim > 4:
            #     raise Exception('error: invalid basis matrix')
            
            #     raise Exception('error: invalid dimension of basis matrix')

            if V.ndim == 1:
                V = V[None, None, :, None]

            elif V.ndim == 2:
                V = V[None, None, :, :]
            
            elif V.ndim == 3:
                V = V[:, :, :, None]

            elif V.ndim > 4:
                raise Exception(f"error: invalid dimension of basis matrix, V.shape = {V.shape}")
            
            self.height, self.width, self.num_channel = V.shape[:3]

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

            if d.size > 0:
                self.num_pred = self.C.shape[1]
            else:
                self.num_pred = V.shape[-1] - 1

            self.num_pixel = self.height * self.width * self.num_channel
        
        elif len_ == 2:
            [lb, ub] = copy.deepcopy(args)

            assert isinstance(lb, np.ndarray), \
            'error: lower bound vector should be a numpy array'
            assert isinstance(ub, np.ndarray), \
            'error: upper bound vector should be a numpy array'

            assert lb.shape == ub.shape, \
            'error: inconsistency between lower bound image and upper bound image'

            if (ub < lb).any():
                raise Exception(
                    'error: the upper bounds must not be less than the lower bounds for all dimensions')

            img_shape = lb.shape
            img_dim = len(img_shape)
            dtype = lb.dtype

            lb = lb.reshape(-1)
            ub = ub.reshape(-1)
            dim = lb.shape[0]
            
            gtr = ub > lb
            nv = gtr.sum()

            c = 0.5 * (lb + ub)
            if dim == nv:
                v = np.diag(0.5 * (ub - lb))
            else:
                v = np.zeros((dim, nv), dtype=dtype)
                j = 0
                for i in range(dim):
                    if gtr[i] > 0:
                        v[i, j] = 0.5 * (ub[i] - lb[i])
                        j += 1
            V = np.column_stack([c, v])

            self.num_pred = nv

            if img_dim == 3:
                img_shape = img_shape + (self.num_pred+1, )

            elif img_dim == 2:
                img_shape = img_shape + (1, self.num_pred+1)

            elif img_dim == 1:
                img_shape = img_shape + (1, 1, self.num_pred+1)

            self.V = V.reshape(img_shape)
            self.C = np.empty([0, 0], dtype=dtype)
            self.d = np.empty([0], dtype=dtype)
            # o = np.ones(nv, dtype=dtype)
            # self.C = np.vstack([np.diag(o), np.diag(-o)])
            # self.d = np.hstack([o, o])
            self.pred_lb = -np.ones(nv, dtype=dtype)
            self.pred_ub = np.ones(nv, dtype=dtype)
            self.height, self.width, self.num_channel = img_shape[0:3]
            self.num_pixel = self.height * self.width * self.num_channel

        # elif len_ == 2:
        #     if copy_ is True:
        #         [lb, ub] = copy.deepcopy(args)
        #     else:
        #         [lb, ub] = args

        #     assert isinstance(lb, np.ndarray), \
        #     'error: lower bound vector should be a numpy array'
        #     assert isinstance(ub, np.ndarray), \
        #     'error: upper bound vector should be a numpy array'
        #     assert lb.shape == ub.shape, \
        #     'error: inconsistency between lower bound image and upper bound image'
        #     assert lb.ndim <= 3, \
        #     'error: lower and upper bound vectors should be less than 4D tensor'

        #     if (ub < lb).any():
        #         raise Exception(
        #             'error: the upper bounds must not be less than the lower bounds for all dimensions')
            
        #     img_shape = lb.shape
        #     img_dim = lb.ndim
        #     dtype = lb.dtype

        #     lb = lb.reshape(-1)
        #     ub = ub.reshape(-1)
        #     dim = lb.shape[0]
        #     nv = int(sum(ub > lb))

        #     V = np.zeros((dim, nv+1), dtype=dtype)
        #     j = 1
        #     for i in range(dim):
        #         if ub[i] > lb[i]:
        #             V[i, j] = 1
        #             j += 1

        #     self.num_pred = nv

        #     if img_dim == 3:
        #         img_shape = img_shape + (self.num_pred+1, )

        #     elif img_dim == 2:
        #         img_shape = img_shape + (1, self.num_pred+1)

        #     elif img_dim == 1:
        #         img_shape = img_shape + (1, 1, self.num_pred+1)

        #     self.V = V.reshape(img_shape)
        #     self.C = np.empty([0, 0], dtype=dtype)
        #     self.d = np.empty([0], dtype=dtype)
        #     self.pred_lb = lb
        #     self.pred_ub = ub
        #     self.height, self.width, self.num_channel = img_shape[0:3]
        #     self.num_pixel = self.height * self.width * self.num_channel

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
        return ''

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
        print('')
        return ''
    
    def __len__(self):
        return 1
    
    def nbytes_generator(self):
        return self.V.nbytes
    
    def nbytes_constraints(self):
        return self.C.nbytes + self.d.nbytes
    
    def nbytes(self):
        # V and c (generator image and anchor image)
        nbt = self.nbytes_generator()
        # C and d
        nbt += self.nbytes_constraints()
        # pred_lb and pred_ub
        nbt += self.pred_lb.nbytes + self.pred_ub.nbytes
        return nbt
    
    # def c(self, index=None):
    #     """Get anchor image of ImageStar"""
    #     if index is None:
    #         return copy.deepcopy(self.V[:, :, :, 0])
    #     else:
    #         return copy.deepcopy(self.V[:, :, :, 0].flatten()[index])
    
    # def X(self, index=None):
    #     """Get generator images of ImageStar"""
    #     if index is None:
    #         return copy.deepcopy(self.V[:, :, :, 1:])
    #     else:
    #         return copy.deepcopy(self.V[:, :, :, index+1])

    def c(self, h=None, w=None, c=None):
        """Get anchor image of ImageStar"""
        
        if self.V.ndim == 4:
            if h is None and w is None and c is None:
                return self.V[:, :, :, 0].copy()
            return self.V[h, w, c, 0].copy()
        
        elif self.V.ndim == 2:
            if h is None:
                return self.V[:, 0].copy()
            return self.V[h, 0].copy()
    
    def X(self, h=None, w=None, c=None, n=None):
        """Get generator images of ImageStar"""

        if self.V.ndim == 4:
            
            if h != None and w != None and c != None:
                return self.V[h, w, c, 1:].copy()
            
            if h != None and w != None:
                return self.V[h, w, :, 1:].copy()
            
            return self.V[:, :, :, 1:].copy()
        
        elif self.V.ndim == 2:
            if h is None:
                return self.V[:, 1:]
            return self.V[h, 1:].copy()
        
        else:
            raise Exception('Basis image dimension issue')
                

    def index_to3D(self, index):
        # V is in [height, width, channel] order

        index = copy.deepcopy(index)
        num = self.width * self.num_channel
        h = index // num
        index -= h * num
        w = index // self.num_channel
        c = index % self.num_channel
        return h, w, c
    
    def resetRow(self, index):
        h_indx, w_indx, c_indx = self.index_to3D(index)
        new_V = copy.deepcopy(self.V)
        new_V[h_indx, w_indx, c_indx, :] = 0
        return ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    def resetRows(self, map):
        ndim = self.V.ndim
        if ndim == 4:
            new_V = self.V.reshape(self.num_pixel, self.num_pred + 1).copy()
        elif ndim == 2:
            new_V = self.V.copy()
        else:
            raise Exception('Invalid basis image dimension')
        
        new_V[map, :] = 0
        
        if ndim == 4:
            new_V = new_V.reshape(self.V.shape)

        return ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)


        # h_map, w_map, c_map = self.index_to3D(map)
        # new_V = copy.deepcopy(self.V)
        # for i in range(len(map)):
        #     new_V[h_map[i], w_map[i], c_map[i], :] = 0
        # return ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    def resetRow_hwc(self, h_indx, w_indx, c_indx):
        new_V = copy.deepcopy(self.V)
        new_V[h_indx, w_indx, c_indx, :] = 0
        return ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    def resetRows_hwc(self, h_map, w_map, c_map):
        new_V = copy.deepcopy(self.V)
        for i in range(len(map)):
            new_V[h_map[i], w_map[i], c_map[i], :] = 0
        return ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    def affineMap(self, W=None, b=None):            

        if W is None and b is None:
            return self
        
        elif self.V.shape[0] == 1 and self.V.shape[1] == 1:
            return self.flatten_affineMap(W, b)
        
        if W is not None:
            assert W.ndim == self.V.ndim-1, f"inconsistent number of array dimensions between W and shape of Image; len(shape)={len(self.V.ndim-1)}, W.ndim={W.ndim}"
            V = self.V * W[:, :, :, None]
        else:
            V = self.V.copy()

        if b is not None:
            if b.ndim == self.V.ndim-1:
                V[:, :, :, 0] += b
            elif b.ndim > 1:
                V[:, :, :, 0] += np.expand_dims(b, axis=tuple(np.arange(V.ndim - 1 - b.ndim)+b.ndim))
            else:
                V[:, :, :, 0] += b
        
        return ImageStar(V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    def flatten_affineMap(self, W=None, b=None):
        if W is None and b is None:
            return copy.deepcopy(self)
        
        assert self.V.shape[0] == 1 and self.V.shape[1] == 1, 'error: ImageStar is not flattened to operate affine mapping, V.shape = {}'.format(self.V.shape)

        V = copy.deepcopy(self.V) #.reshape(self.num_pixel, self.num_pred+1)

        if W is not None:
            assert isinstance(W, np.ndarray), 'error: ' + \
            'the mapping matrix should be a 2D numpy array'
            assert W.shape[1] == self.num_pixel, 'error: ' + \
            'inconsistency between mapping matrix and SparseImageStar dimension'

            V = np.matmul(W, V)

        if b is not None:
            assert isinstance(b, np.ndarray), 'error: ' + \
            'the offset vector should be a 1D numpy array'
            assert len(b.shape) == 1, 'error: ' + \
            'offset vector should be a 1D numpy array'

            if W is not None:
                assert W.shape[0] == b.shape[0], 'error: ' + \
                'inconsistency between mapping matrix and offset'
            else:
                assert b.shape[0] == self.num_pixel, 'error: ' + \
                'inconsistency between offset vector and SparseStar dimension'

            V[:, :, :, 0] += b
        
        return ImageStar(V, self.C, self.d, self.pred_lb, self.pred_ub)

    def getMin(self, *args):
        """Get the minimum value of state x[index] or x[h_indx, w_indx, c_indx] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
            h_indx: veritcial index
            w_indx: horizontal index
            c_indx: channel index
            index: flattened index
        """
        len_ = len(args)

        if len_ == 4:
            [h_indx, w_indx, c_indx, lp_solver] = args
            # index = None
            self.getMin_hwc(h_indx, w_indx, c_indx, lp_solver)

        elif len_ == 3:
            [h_indx, w_indx, c_indx] = args
            lp_solver = 'gurobi'
            # index = None
            self.getMin_hwc(h_indx, w_indx, c_indx, lp_solver)

        elif len_ == 2:
            [index, lp_solver] = args
            return self.getMin_index(index, lp_solver)

        elif len_ == 1:
            [index] = args
            lp_solver = 'gurobi'
            return self.getMin_index(index, lp_solver)

        else:
            raise Exception(
                'error: invalid number of input arguments (should be between 1 and 4)')

        # if index is not None:
        #     h_indx, w_indx, c_indx = self.index_to3D(index)

        # return self.getMin_hwc(h_indx, w_indx, c_indx, lp_solver)
    
    def getMax(self, *args):
        """Get the maximum value of state x[index] or x[h_indx, w_indx, c_indx] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
            h_indx: veritcial index
            w_indx: horizontal index
            c_indx: channel index
            index: flattened index
        """
        len_ = len(args)

        if len_ == 4:
            [h_indx, w_indx, c_indx, lp_solver] = args
            # index = None
            return self.getMax_hwc(h_indx, w_indx, c_indx, lp_solver)

        elif len_ == 3:
            [h_indx, w_indx, c_indx] = args
            lp_solver = 'gurobi'
            # index = None
            return self.getMax_hwc(h_indx, w_indx, c_indx, lp_solver)

        elif len_ == 2:
            [index, lp_solver] = args
            return self.getMax_index(index, lp_solver)
        
        elif len_ == 1:
            [index] = args
            lp_solver = 'gurobi'
            return self.getMax_index(index, lp_solver)
        
        else:
            raise Exception(
                'error: invalid number of input arguments (should be between 1 and 4)')

        # if index is not None:
        #     h_indx, w_indx, c_indx = self.index_to3D(index)

        # return self.getMax_hwc(h_indx, w_indx, c_indx, lp_solver)
    
    def getMin_index(self, index, lp_solver='gurobi'):
        """Get the minimum value of state x[index] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
        """

        assert index >= 0 and index < self.num_pixel, \
        'error: invalid index'

        V = self.V.reshape(self.num_pixel, self.num_pred+1)

        f = V[index, 1:]
        if (f == 0).all():
            xmin = V[index, 0]
        else:
            if lp_solver == 'gurobi':  # gurobi is the preferred LP solver

                min_ = gp.Model()
                min_.Params.LogToConsole = 0
                min_.Params.OptimalityTol = GUROBI_OPT_TOL
                if self.pred_lb.size and self.pred_ub.size:
                    x = min_.addMVar(shape=self.num_pred,
                                     lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = min_.addMVar(shape=self.num_pred)
                min_.setObjective(f @ x, GRB.MINIMIZE)
                if len(self.d) > 0:
                    C = self.C
                    d = self.d
                else:
                    C = sp.csr_matrix(np.zeros((1, self.num_pred)))
                    d = 0
                min_.addConstr(C @ x <= d)
                min_.optimize()

                if min_.status == 2:
                    xmin = min_.objVal + V[index, 0]
                else:
                    raise Exception('error: cannot find an optimal solution, ' + \
                        'exitflag = %d' % (min_.status))

            elif lp_solver == 'linprog':

                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
                if len(self.d) == 0:
                    A = np.zeros((1, self.num_pred))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.num_pred, 1))
                ub = ub.reshape((self.num_pred, 1))
                res = linprog(f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))

                if res.status == 0:
                    xmin = res.fun + V[index, 0]
                else:
                    raise Exception('error: cannot find an optimal solution, ' + \
                        'exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                #  https://pyglpk.readthedocs.io/en/latest/examples.html
                #  https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False

                if len(self.d) == 0:
                    A = np.zeros((1, self.num_pred))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.num_pred, 1))
                ub = ub.reshape((self.num_pred, 1))

                lp = glpk.LPX()  # create the empty problem instance
                lp.obj.maximize = False
                lp.rows.add(A.shape[0])  # append rows to this instance
                for r in lp.rows:
                    r.name = chr(ord('p') + r.index)  # name rows if we want
                    lp.rows[r.index].bounds = None, b[r.index]

                lp.cols.add(self.num_pred)
                for c in lp.cols:
                    c.name = 'x%d' % c.index
                    c.bounds = lb[c.index], ub[c.index]

                lp.obj[:] = f.tolist()
                B = A.reshape(A.shape[0]*A.shape[1],)
                lp.matrix = B.tolist()
                # lp.interior()
                lp.simplex()
                # default choice, interior may have a big floating point error

                if lp.status != 'opt':
                    raise Exception('error: cannot find an optimal solution, ' + \
                        'lp.status = {}'.format(lp.status))
                else:
                    xmin = lp.obj.value + V[index, 0]

            else:
                raise Exception(
                    'error: unknown lp solver, should be gurobi or linprog or glpk')
        return xmin
    
    def getMax_index(self, index, lp_solver='gurobi'):
        """Get the maximum value of state x[index] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
        """

        assert index >= 0 and index < self.num_pixel, \
        'error: invalid index'

        V = self.V.reshape(self.num_pixel, self.num_pred+1)
        f = V[index, 1:]
        if (f == 0).all():
            xmax = V[index, 0]
        else:
            if lp_solver == 'gurobi':  # gurobi is the preferred LP solver

                max_ = gp.Model()
                max_.Params.LogToConsole = 0
                max_.Params.OptimalityTol = GUROBI_OPT_TOL
                if self.pred_lb.size and self.pred_ub.size:
                    x = max_.addMVar(shape=self.num_pred,
                                     lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = max_.addMVar(shape=self.num_pred)
                max_.setObjective(f @ x, GRB.MAXIMIZE)
                if len(self.d) > 0:
                    C = self.C
                    d = self.d
                else:
                    C = sp.csr_matrix(np.zeros((1, self.num_pred)))
                    d = 0
                max_.addConstr(C @ x <= d)
                max_.optimize()

                if max_.status == 2:
                    xmax = max_.objVal + V[index, 0]
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = %d' % (max_.status))

            elif lp_solver == 'linprog':
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
                if len(self.d) == 0:
                    A = np.zeros((1, self.num_pred))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.num_pred, 1))
                ub = ub.reshape((self.num_pred, 1))
                res = linprog(-f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))
                if res.status == 0:
                    xmax = -res.fun + V[index, 0]
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                # https://pyglpk.readthedocs.io/en/latest/examples.html
                # https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False  # turn off messages/display

                if len(self.d) == 0:
                    A = np.zeros((1, self.num_pred))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.num_pred, 1))
                ub = ub.reshape((self.num_pred, 1))

                lp = glpk.LPX()  # create the empty problem instance
                lp.obj.maximize = True
                lp.rows.add(A.shape[0])  # append rows to this instance
                for r in lp.rows:
                    r.name = chr(ord('p') + r.index)  # name rows if we want
                    lp.rows[r.index].bounds = None, b[r.index]

                lp.cols.add(self.num_pred)
                for c in lp.cols:
                    c.name = 'x%d' % c.index
                    c.bounds = lb[c.index], ub[c.index]

                lp.obj[:] = f.tolist()
                B = A.reshape(A.shape[0]*A.shape[1],)
                lp.matrix = B.tolist()

                # lp.interior()
                # default choice, interior may have a big floating point error
                lp.simplex()

                if lp.status != 'opt':
                    raise Exception('error: cannot find an optimal solution, \
                    lp.status = {}'.format(lp.status))
                else:
                    xmax = lp.obj.value + V[index, 0]
            else:
                raise Exception('error: \
                unknown lp solver, should be gurobi or linprog or glpk')
        return xmax
    
    def getMin_hwc(self, h_indx, w_indx, c_indx, lp_solver='gurobi'):
        """Get the minimum value of state x[index] or x[h_indx, w_indx, c_indx] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
            h_indx: veritcial index
            w_indx: horizontal index
            c_indx: channel index
        """

        assert h_indx >= 0 and h_indx < self.height, \
        'error: invalid vertical index'
        assert w_indx >= 0 and w_indx < self.width, \
        'error: invalid horizontal index'
        assert c_indx >= 0 and c_indx < self.num_channel, \
        'error: invalid channel index'

        f = self.X(h_indx, w_indx, c_indx)
        if (f == 0).all():
            xmin = self.c(h_indx, w_indx, c_indx)
        else:
            if lp_solver == 'gurobi':  # gurobi is the preferred LP solver

                min_ = gp.Model()
                min_.Params.LogToConsole = 0
                min_.Params.OptimalityTol = GUROBI_OPT_TOL
                if self.pred_lb.size and self.pred_ub.size:
                    x = min_.addMVar(shape=self.num_pred,
                                     lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = min_.addMVar(shape=self.num_pred)
                min_.setObjective(f @ x, GRB.MINIMIZE)
                if len(self.d) > 0:
                    C = self.C
                    d = self.d
                else:
                    C = sp.csr_matrix(np.zeros((1, self.num_pred)))
                    d = 0
                min_.addConstr(C @ x <= d)
                min_.optimize()

                if min_.status == 2:
                    xmin = min_.objVal + self.c(h_indx, w_indx, c_indx)
                else:
                    raise Exception('error: cannot find an optimal solution, ' + \
                        'exitflag = %d' % (min_.status))

            elif lp_solver == 'linprog':

                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
                if len(self.d) == 0:
                    A = np.zeros((1, self.num_pred))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.num_pred, 1))
                ub = ub.reshape((self.num_pred, 1))
                res = linprog(f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))

                if res.status == 0:
                    xmin = res.fun + self.c(h_indx, w_indx, c_indx)
                else:
                    raise Exception('error: cannot find an optimal solution, ' + \
                        'exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                #  https://pyglpk.readthedocs.io/en/latest/examples.html
                #  https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False

                if len(self.d) == 0:
                    A = np.zeros((1, self.num_pred))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.num_pred, 1))
                ub = ub.reshape((self.num_pred, 1))

                lp = glpk.LPX()  # create the empty problem instance
                lp.obj.maximize = False
                lp.rows.add(A.shape[0])  # append rows to this instance
                for r in lp.rows:
                    r.name = chr(ord('p') + r.index)  # name rows if we want
                    lp.rows[r.index].bounds = None, b[r.index]

                lp.cols.add(self.num_pred)
                for c in lp.cols:
                    c.name = 'x%d' % c.index
                    c.bounds = lb[c.index], ub[c.index]

                lp.obj[:] = f.tolist()
                B = A.reshape(A.shape[0]*A.shape[1],)
                lp.matrix = B.tolist()
                # lp.interior()
                lp.simplex()
                # default choice, interior may have a big floating point error

                if lp.status != 'opt':
                    raise Exception('error: cannot find an optimal solution, ' + \
                        'lp.status = {}'.format(lp.status))
                else:
                    xmin = lp.obj.value + self.c(h_indx, w_indx, c_indx)

            else:
                raise Exception(
                    'error: unknown lp solver, should be gurobi or linprog or glpk')
        return xmin


    def getMax_hwc(self, h_indx, w_indx, c_indx, lp_solver='gurobi'):
        """Get the maximum value of state x[h_indx, w_indx, c_indx] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
            h_indx: veritcial index
            w_indx: horizontal index
            c_indx: channel index
        """

        assert h_indx >= 0 and h_indx < self.height, \
        'error: invalid vertical index'
        assert w_indx >= 0 and w_indx < self.width, \
        'error: invalid horizontal index'
        assert c_indx >= 0 and c_indx < self.num_channel, \
        'error: invalid channel index'

        f = self.X(h_indx, w_indx, c_indx)
        if (f == 0).all():
            xmax = self.c(h_indx, w_indx, c_indx)
        else:
            if lp_solver == 'gurobi':  # gurobi is the preferred LP solver

                max_ = gp.Model()
                max_.Params.LogToConsole = 0
                max_.Params.OptimalityTol = GUROBI_OPT_TOL
                if self.pred_lb.size and self.pred_ub.size:
                    x = max_.addMVar(shape=self.num_pred,
                                     lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = max_.addMVar(shape=self.num_pred)
                max_.setObjective(f @ x, GRB.MAXIMIZE)
                if len(self.d) > 0:
                    C = self.C
                    d = self.d
                else:
                    C = sp.csr_matrix(np.zeros((1, self.num_pred)))
                    d = 0
                max_.addConstr(C @ x <= d)
                max_.optimize()

                if max_.status == 2:
                    xmax = max_.objVal + self.c(h_indx, w_indx, c_indx)
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = %d' % (max_.status))

            elif lp_solver == 'linprog':
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
                if len(self.d) == 0:
                    A = np.zeros((1, self.num_pred))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.num_pred, 1))
                ub = ub.reshape((self.num_pred, 1))
                res = linprog(-f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))
                if res.status == 0:
                    xmax = -res.fun + self.c(h_indx, w_indx, c_indx)
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                # https://pyglpk.readthedocs.io/en/latest/examples.html
                # https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False  # turn off messages/display

                if len(self.d) == 0:
                    A = np.zeros((1, self.num_pred))
                    b = np.zeros(1)
                else:
                    A = self.C
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.num_pred, 1))
                ub = ub.reshape((self.num_pred, 1))

                lp = glpk.LPX()  # create the empty problem instance
                lp.obj.maximize = True
                lp.rows.add(A.shape[0])  # append rows to this instance
                for r in lp.rows:
                    r.name = chr(ord('p') + r.index)  # name rows if we want
                    lp.rows[r.index].bounds = None, b[r.index]

                lp.cols.add(self.num_pred)
                for c in lp.cols:
                    c.name = 'x%d' % c.index
                    c.bounds = lb[c.index], ub[c.index]

                lp.obj[:] = f.tolist()
                B = A.reshape(A.shape[0]*A.shape[1],)
                lp.matrix = B.tolist()

                # lp.interior()
                # default choice, interior may have a big floating point error
                lp.simplex()

                if lp.status != 'opt':
                    raise Exception('error: cannot find an optimal solution, \
                    lp.status = {}'.format(lp.status))
                else:
                    xmax = lp.obj.value + self.c(h_indx, w_indx, c_indx)
            else:
                raise Exception('error: \
                unknown lp solver, should be gurobi or linprog or glpk')
        return xmax

    def getMins_all(self, lp_solver='gurobi'):

        xmin = np.zeros([self.height, self.height, self.num_channel], dtype=self.V.dtype)
        for h_ in range(self.height):
            for w_ in range(self.width):
                for c_ in range(self.num_channel):
                    xmin[h_, w_, c_] = self.getMin_hwc(h_, w_, c_, lp_solver)
        return xmin

    def getMaxs_all(self, lp_solver='gurobi'):
        xmax = np.zeros([self.height, self.height, self.num_channel], dtype=self.V.dtype)
        for h_ in range(self.height):
            for w_ in range(self.width):
                for c_ in range(self.num_channel):
                    xmax[h_, w_, c_] = self.getMax_hwc(h_, w_, c_, lp_solver)
        return xmax

    def getMins(self, *args):
        """Get the maximum values of state x corresponding map indexes
        """
        len_ = len(args)

        if len_ == 4:
            [h_map, w_map, c_map, lp_solver] = args
            map = None

        elif len_ == 3:
            [h_map, w_map, c_map] = args
            lp_solver = 'gurobi'
            map = None

        elif len_ == 2:
            [map, lp_solver] = args

        elif len_ == 1:
            [map] = args
            lp_solver = 'gurobi'

        else:
            raise Exception(
                'error: invalid number of input arguments (should be between 1 and 4)')

        n = len(map)
        xmin = np.zeros(n, dtype=self.V.dtype)

        if map is not None:
            # h_map, w_map, c_map = self.index_to3D(map)
            for i in range(n):
                xmin[i] = self.getMin_index(map[i], lp_solver)

        else:
            for i in range(n):
                xmin[i] = self.getMin_hwc(h_map[i], w_map[i], c_map[i], lp_solver)

        return xmin

    def getMaxs(self, *args):
        """Get the maximum values of state x corresponding map indexes
        """
        len_ = len(args)

        if len_ == 4:
            [h_map, w_map, c_map, lp_solver] = args
            map = None

        elif len_ == 3:
            [h_map, w_map, c_map] = args
            lp_solver = 'gurobi'
            map = None

        elif len_ == 2:
            [map, lp_solver] = args

        elif len_ == 1:
            [map] = args
            lp_solver = 'gurobi'

        else:
            raise Exception(
                'error: invalid number of input arguments (should be between 1 and 4)')

        n = len(map)
        xmin = np.zeros(n, dtype=self.V.dtype)

        if map is not None:
            # h_map, w_map, c_map = self.index_to3D(map)
            for i in range(n):
                xmin[i] = self.getMax_index(map[i], lp_solver)

        else:
            for i in range(n):
                xmin[i] = self.getMax_hwc(h_map[i], w_map[i], c_map[i], lp_solver)

        return xmin
    
    def estimateRange(self, h_indx, w_indx, c_indx):
        """Estimate the minimum and maximum values of a state x[index]"""

        assert h_indx >= 0 and h_indx < self.height, \
        'error: invalid vertical index'
        assert w_indx >= 0 and w_indx < self.width, \
        'error: invalid horizontal index'
        assert c_indx >= 0 and c_indx < self.num_channel, \
        'error: invalid channel index'
        
        l = self.pred_lb
        u = self.pred_ub

        X = self.X(h_indx, w_indx, c_indx)
        pos_f = np.maximum(X, 0.0)
        neg_f = np.minimum(X, 0.0)

        xmin = self.c(h_indx, w_indx, c_indx) + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        xmax = self.c(h_indx, w_indx, c_indx) + np.matmul(neg_f, l) + np.matmul(pos_f, u)
        return xmin, xmax

    def estimateRanges(self):
        """Estimate the lower and upper bounds of x"""

        l = self.pred_lb
        u = self.pred_ub

        X = self.X().reshape(self.num_pixel, self.num_pred)
        pos_f = np.maximum(X, 0.0)
        neg_f = np.minimum(X, 0.0)

        xmin = self.c().reshape(-1) + np.matmul(pos_f, l) + np.matmul(neg_f, u)
        xmax = self.c().reshape(-1) + np.matmul(neg_f, l) + np.matmul(pos_f, u)
        return xmin, xmax

    def getRange(self, h_indx, w_indx, c_indx, lp_solver='gurobi'):
        """Get the lower and upper bounds of x[index]"""

        if lp_solver == 'estimate':
            return self.estimateRange(h_indx, w_indx, c_indx)
        else:
            l = self.getMin(h_indx, w_indx, c_indx, lp_solver)
            u = self.getMax(h_indx, w_indx, c_indx, lp_solver)
            return l, u    
        

    def getRanges(self, lp_solver='gurobi', RF=0.0, layer=None, delta=0.98):
        """Get the lower and upper bound vectors of the state
            Args:
                lp_solver: linear programming solver. e.g.: 'gurobi', 'estimate', 'linprog'
        """

        if lp_solver == 'estimate':
            l, u = self.estimateRanges()
        else:
            l = self.getMins_all()
            u = self.getMaxs_all()
        return l, u

    # def getRanges(self, lp_solver='gurobi', RF=0.0, layer=None, delta=0.98):
    #     """Get the lower and upper bound vectors of the state
    #         Args:
    #             lp_solver: linear programming solver. e.g.: 'gurobi', 'estimate', 'linprog'
    #             RF: relaxation factor \in [0.0, 1.0]
    #     """
        
    #     shape = self.height, self.width, self.num_channel

    #     if RF == 1.0:
    #         l, u = self.estimateRanges()
        
    #     elif RF == 0.0:
    #         if lp_solver == 'estimate':
    #             l, u = self.estimateRanges()
    #         else:
    #             l = self.getMins()
    #             u = self.getMaxs()
            
    #     else:
    #         assert RF > 0.0 and RF <= 1.0, \
    #         'error: relaxation factor should be greater than 0.0 but less than or equal to 1.0'
    #         l, u = self.estimateRanges()
            
    #         if layer in ['logsig', 'tansig']:
    #             n1 = round(1 - RF) * self.num_pixel

    #             midx = np.argsort((u - l))[::-1]
    #             midb = np.argwhere((l[midx] >= -delta) & (u[midx] <= delta))
                
    #             n2 = n1
    #             check = midb.flatten().shape[0]
    #             if n2 > check:
    #                 n2 = check

    #             mid = midx[midb[0:n2]]
    #             l1 = self.getMins(mid)
    #             u1 = self.getMaxs(mid)
    #             l[mid] = l1
    #             u[mid] = u1

    #         elif  layer in ['poslin', 'relu']:
    #             map = np.argwhere((l < 0) & (u > 0))

    #             n1 = round(1 - RF) * len(map)

                
    #             area = 0.5*abs(u[map]*l[map])
    #             midx = np.argsort(area)[::-1]

    #             mid = midx[0:n1]
    #             l1 = self.getMins(mid)
    #             u1 = self.getMaxs(mid)
    #             l[mid] = l1
    #             u[mid] = u1

    #         else:
    #             n1 = round(1 - RF) * self.num_pixel

    #             midx = np.argsort((u - l))[::-1]
    #             mid = midx[0:n1]
    #             l1 = self.getMins(mid)
    #             u1 = self.getMaxs(mid)
    #             l[mid] = l1
    #             u[mid] = u1
      
    #     return l.reshape(shape), u.reshape(shape)

    def isEmptySet(self, lp_solver='gurobi'):
        """Check if a SparseStar is an empty set"""
        res = False
        try:
            self.getMin(0, lp_solver)
        except Exception:
            res = True
        return res
    
    def get_max_point_cadidates(self):
        """ Quickly estimate max-point candidates """

        lb, ub = self.getRanges('estimate')
        max_id = np.argmax(lb)
        a = (ub >= lb[max_id])
        if sum(a) == 1:
            return [max_id]
        else:
            return np.where(a)[0]
            
    def is_p1_larger_than_p2(self, p1_indx, p2_indx, lp_solver='gurobi'):
        """
            Check if an index is larger than the other

            Arg:
                @p1_indx: an index of point 1
                @p2_indx: an index of point 2

            return:
                @bool = 1 if there exists the case that p1 >= p2
                        2 if there is no case that p1 >= p2; p1 < p2
        """

        assert p1_indx >= 0 and p1_indx < self.num_pixel, 'error: invalid index for point 1'
        assert p2_indx >= 0 and p2_indx < self.num_pixel, 'error: invalid index for point 2'

        V = self.V.reshape(self.num_pixel, self.num_pred+1)

        d1 = V[p1_indx, 0] - V[p2_indx, 0]
        C1 = V[p2_indx, 1:] - V[p1_indx, 1:]

        if len(self.d) > 0:
            new_d = np.hstack([self.d, d1])
            new_C = np.vstack([self.C, C1])
        else:
            new_d = np.array([d1])
            new_C = C1[None, :]

        # S = Star(V, new_C, new_d, self.pred_lb, self.pred_ub, copy_=False)

        V = V[None, None, :, :]
        S = ImageStar(V, new_C, new_d, self.pred_lb, self.pred_ub, copy_=False)
        
        if S.isEmptySet(lp_solver=lp_solver):
            return False
        else:
            return True
    
    
    def toStar(self, copy_=True):
        """Convert ImageStarTensor class to Star class"""
        V = self.V.reshape(self.num_pixel, self.num_pred+1)
        return Star(V, self.C, self.d, self.pred_lb, self.pred_ub, copy_=True)

    # def toSparseStar(self):
    #     """Convert ImageStar class to SparseStar class"""
    #     pass


    # def toSIM_coo(self):
    #     c = self.V[:, :, :, 0].reshape(-1)
    #     V = sp.coo_array(self.V[:, :, :, 1:])
    #     return SparseImageStar2D(c, V, self.C, self.d, self.pred_lb, self.pred_ub)

    # def toSIM_csr(self):
    #     c = self.V[:, :, :, 0].reshape(-1)
    #     V = sp.csr_array(self.V[:, :, :, 1:])
    #     return SparseImageStar2DCSR(c, V, self.C, self.d, self.pred_lb, self.pred_ub)

    @staticmethod
    def isMax(maxMap, ori_image, center, others, lp_solver='gurobi'):
        """
        Check if a pixel value is the maximum value compared with others
        This is the core step for exactly performing maxpooling operation on an ImageStar set
       
        Args:
            @maxMap: the current maxMap ImageStar
            @ori_image: the original ImageStar to compute the maxMap
            @center: is the center pixel position we want to check
                    center = [index]
            @others: is the other pixel position we want to compare with the cetner one
                    others = [index0, index1]
        """
        assert maxMap.num_pred == ori_image.num_pred, \
        'error: Inconsistency between number of predicates in the currrent maxMap and the original image'
        n = len(others)
        
        # the center may be the max point with some extra constraints on the predicate variables
        new_C = np.zeros([n, maxMap.num_pred], dtype=maxMap.V.dtype)
        new_d = np.zeros(n, dtype=maxMap.V.dtype)

        V = ori_image.V.reshape(ori_image.num_pixel, ori_image.num_pred+1)

        for i in range(n):
            new_C[i, :] = V[others[i], 1:] - V[center, 1:]
            new_d[i] = V[center, 0] - V[others[i], 0]
        
        if len(maxMap.d) > 0:
            C1 = np.vstack([maxMap.C, new_C])
            d1 = np.hstack([maxMap.d, new_d])
        else:
            C1 = new_C
            d1 = new_d

        # # remove redundant constraints
        # E = np.hstack([C1, d1[:, None]])
        # E = np.unique(E, axis=0)

        # C1 = E[:, :-1]
        # d1 = E[:, -1:].reshape(-1)
        # ###

        S = Star(V, C1, d1, ori_image.pred_lb, ori_image.pred_ub, copy_=False)
        
        if S.isEmptySet(lp_solver=lp_solver):
            return None, None
        else:
            return C1, d1


    @staticmethod
    def inf_attack(data, epsilon=0.01, data_type='default', dtype = 'float64'):
        """Generate a ImageStar set by infinity norm attack on input dataset"""

        if isinstance(data, np.ndarray):
            assert data.ndim == 3, \
            'error: data should be a 3D numpy array in [height, width, channel] shape'
        
        elif isinstance(data, torch.Tensor):
            assert data.ndim == 3, \
            'error: data should be a 3D torch tensor in [channel, height, width] shape'

            data = data.permute(1, 2, 0).numpy()

        else:
            raise Exception('the data should be a 3D numpy array or 3D torch tensor')

        if dtype =='float64':
            data = data.astype(np.float64)
        else:
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
