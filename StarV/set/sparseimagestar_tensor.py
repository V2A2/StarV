"""
Sparse Image Star Class
Sung Woo Choi, 08/09/2023

"""

# !/usr/bin/python3
import copy
import torch

class SpImStarT(object):
    """
        
        Sparse Image Star for reachability
        author: Sung Woo Choi
        date: 08/09/2022
        Representation of a SpImStarT
        ==========================================================================
        Sparse Image Star set defined by


        Channel Last Format
        NHWC
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

            assert isinstance(A, torch.Tensor), \
            'error: basis matrix should be a torch tensor'
            assert isinstance(pred_lb, torch.Tensor), \
            'error: lower bound vector should be a 1D torch tensor'
            assert isinstance(pred_ub, torch.Tensor), \
            'error: upper bound vector should be a 1D torch tensor'

            if len(d) > 0:
                assert isinstance(C, torch.Tensor), \
                'error: non-zero basis matrix should be a torch tensor'
                assert isinstance(d, torch.Tensor), \
                'error: non-zero basis matrix should be a 1D torch tensor'
                assert len(d.shape) == 1, \
                'error: constraint vector should be a 1D torch tensor'
                assert C.shape[0] == d.shape[0], \
                'error: inconsistency between constraint matrix and constraint vector'
                assert C.shape[1] == pred_lb.shape[0], \
                'error: inconsistent number of predicatve variables between constratint matrix and predicate bound vectors'

            assert len(pred_lb.shape) == 1, \
            'error: lower bound vector should be a 1D torch tensor'
            assert len(pred_ub.shape) == 1, \
            'error: upper bound vector should be a 1D torch tensor'
            assert pred_ub.shape[0] == pred_lb.shape[0], \
            'error: inconsistent number of predicate variables between predicate lower- and upper-boud vectors'
            assert pred_lb.shape[0] == pred_depth.shape[0], \
            'error: inconsistent number of predicate variables between predicate bounds and predicate depth'

            self.A = A
            self.C = C if C.is_sparse else C.to_sparse()
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub
            self.pred_depth = pred_depth
            if len(d) > 0:
                self.nVars = self.C.shape[1]
                self.nZVars = self.C.shape[1] + 1 - self.A.shape[1]
            else:
                self.nVars = self.A.shape[1] - 1
                self.nZVars = 0

            n = len(A.shape)
            if n == 4:
                assert A.shape[3] == self.nVars + 1, \
                'error: inconsistency between the independent basis matrix and the number of predicate variables'
                self.height, self.width, self.num_channel = A.shape[0:2]

            elif n == 3:
                self.height, self.width, self.num_channel = A.shape

            elif n == 2:
                self.height, self.width = A.shape
                self.num_channel = 1
                
            else:
                raise Exception('error: invalid independent basis matrix')
            
        # elif len_ == 2:
        #     [lb, ub] = copy.deepcopy(args)

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
        #     img_dim = len(img_shape)

        #     lb = lb.flatten()
        #     ub = ub.flatten()
        #     dim = lb.shape[0]
        #     nv = int(sum(ub > lb))

        #     c = 0.5 * (lb + ub)
        #     if dim == nv:
        #         v = np.diag(0.5 * (ub - lb))
        #     else:
        #         v = np.zeros((dim, nv))
        #         j = 0
        #         for i in range(self.dim):
        #             if ub[i] > lb[i]:
        #                 v[i, j] = 0.5 * (ub[i] - lb[i])
        #                 j += 1
        #     A = np.column_stack([c, v])

        #     self.nVars = dim

        #     if img_dim == 3:
        #         img_shape = img_shape + (self.nVars+1, )
        
        #     elif img_dim == 2:
        #         img_shape = img_shape + (1, self.nVars+1)
                
        #     self.A = A.reshape(img_shape)
        #     self.C = sp.csc_matrix((0, dim))
        #     self.d = np.empty([0])
        #     self.pred_lb = -np.ones(dim)
        #     self.pred_ub = np.ones(dim)
        #     self.pred_depth = np.zeros(dim)
        #     self.width, self.height, self.num_channel = img_shape[0:3]
        #     self.nZVars = dim + 1 - self.A.shape[3]

        elif len_ == 2:
            [lb, ub] = copy.deepcopy(args)
            lb = lb.type(torch.float64)
            ub = ub.type(torch.float64)

            assert isinstance(lb, torch.Tensor), \
            'error: lower bound vector should be a torch tensor'
            assert isinstance(ub, torch.Tensor), \
            'error: upper bound vector should be a torch tensor'

            assert lb.shape == ub.shape, \
            'error: inconsistency between lower bound image and upper bound image'

            if torch.any(ub < lb):
                raise Exception(
                    'error: the upper bounds must not be less than the lower bounds for all dimensions')

            img_shape = lb.shape
            img_dim = len(img_shape)

            lb = lb.flatten()
            ub = ub.flatten()
            dim = lb.shape[0]
            nv = int(sum(ub > lb))

            A = torch.zeros((dim, nv+1))
            j = 1
            for i in range(dim):
                if ub[i] > lb[i]:
                    A[i, j] = 1
                    j += 1
            
            self.nVars = dim

            if img_dim == 3:
                img_shape = img_shape + (self.nVars+1, )
        
            elif img_dim == 2:
                img_shape = img_shape + (1, self.nVars+1)
                
            self.A = A.reshape(img_shape)
            self.C = torch.empty((0, dim)).to_sparse()
            self.d = torch.empty([0])
            self.pred_lb = lb
            self.pred_ub = ub
            self.pred_depth = torch.zeros(dim)
            self.width, self.height, self.num_channel = img_shape[0:3]
            self.nZVars = dim + 1 - self.A.shape[3]

        elif len_ == 0:
            self.A = torch.empty([0, 0])
            self.C = torch.empty((0, 0)).to_sparse()
            self.d = torch.empty([0])
            self.pred_lb = torch.empty([0])
            self.pred_ub = torch.empty([0])
            self.pred_depth = torch.empty([0])
            self.height = 0
            self.width = 0
            self.num_channel = 0
            self.nVars = 0
            self.nZVars = 0

        else:
            raise Exception(
                'error: invalid number of input arguments (should be 0, 2, 6)')
    
    def __str__(self, toDense=True):
        print('SpImStarT Set:')
        print('A: \n{}'.format(self.A))
        if toDense:
            print('C: \n{}'.format(self.C.to_dense()))
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
    
    def translation(self, v=None):
        """Translation of a sparse image star: S = self + v"""
        if v is None:
            return copy.deepcopy(self)

        assert isinstance(v, torch.Tensor), 'error: ' + \
        'offset vector should be an 1D numpy array'
        assert len(v.shape) == 1, 'error: ' + \
        'the translation vector should be a 1D numpy array'
        assert v.shape[0] == self.dim, 'error: ' + \
        'inconsistency between translation vector and SparseStar dimension'

        A = copy.deepcopy(self.A)
        A[:, :, 0] += v
        return SpImStarT(A, self.C, self.d, self.pred_lb, self.pred_ub, self.pred_depth)

    # def affineMap(self, W=None, b=None):
    #     """Affine mapping of a sparse image star: S = W*self + b"""

    #     if W is None and b is None:
    #         return copy.deepcopy(self)

    #     if W is not None:
    #         assert isinstance(W, np.ndarray), 'error: ' + \
    #         'the mapping matrix should be a 2D numpy array'
    #         assert W.shape[1] == self.dim, 'error: ' + \
    #         'inconsistency between mapping matrix and SparseStar dimension'

    #         # A = np.matmul(W, self.A)
    #         A = np.multiply()

    #     if b is not None:
    #         assert isinstance(b, np.ndarray), 'error: ' + \
    #         'the offset vector should be a 1D numpy array'
    #         assert len(b.shape) == 1, 'error: ' + \
    #         'offset vector should be a 1D numpy array'

    #         if W is not None:
    #             assert W.shape[0] == b.shape[0], 'error: ' + \
    #             'inconsistency between mapping matrix and offset'
    #         else:
    #             assert b.shape[0] == self.dim, 'error: ' + \
    #             'inconsistency between offset vector and SparseStar dimension'
    #             A = copy.deepcopy(self.A)

    #         A[:, :, 0] += b

    #     return SpImStarT(A, self.C, self.d, self.pred_lb, self.pred_ub, self.pred_depth)

    @staticmethod
    def inf_attack(data, epsilon=0.01, data_type='default'):
        """Generate a SparseStar set by infinity norm attack on input dataset"""

        assert isinstance(data, torch.Tensor), \
        'error: the data should be a 1D numpy array'

        lb = data - epsilon
        ub = data + epsilon

        if data_type == 'image':
            lb[lb < 0] = 0
            ub[ub > 1] = 1
            return SpImStarT(lb, ub)

        return SpImStarT(lb, ub)

if __name__ == "__main__":
    # IM = np.random.rand(2,2,3)
    IM = torch.tensor([
            [0.1683,    0.3099,    0.2993],
            [0.8552,    0.1964,    0.3711],
            [0.2749,    0.3069,    0.1244],
            [0.8106,    0.0844,    0.1902],
    ])
    print(IM)

    SpImStarT = SpImStarT.inf_attack(data=IM, epsilon=0.01, data_type='image')
    print(SpImStarT)

    print(SpImStarT.A.shape)

    print(SpImStarT.A[:, :, 0, 0])