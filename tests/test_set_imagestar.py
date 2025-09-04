import copy
import numpy as np
import scipy
import scipy.linalg

from StarV.set.imagestar import ImageStar
from StarV.fun.matmul import MatMul
from StarV.fun.identityXidentity import IdentityXIdentity

class Test(object):
    """
       Testing Star class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_constructor(self):

        self.n_tests += 1

        shape = (4, 4, 2)
        lb = np.random.rand(*shape) * 2 - 1
        ub = lb + np.random.rand(*shape)
        print('Testing ImageStar Constructor...')
        try:
            ImageStar(lb, ub)
        except Exception:
            print('Fail in constructing ImageStar object with bounds, len(args) = 2')
        else:
            print('Test Successfull!')

    def test_transpose(self):

        self.n_tests += 1

        shape = (3, 2, 1)
        dim = np.prod(shape)
        A = np.arange(dim).reshape(*shape) + 1
        print('A:\n',A)
        eps = 0.5
        lb = A - eps
        ub = A + eps
        print('lb:\n',lb)
        print('ub:\n',ub)
        IM = ImageStar(lb, ub)
        print(IM)
        print(repr)
        LBe, UBe = IM.getRanges('estimate')
        print('LB estimate:\n', LBe.reshape(*shape))
        print('UB estimate:\n', UBe.reshape(*shape))
        LB, UB = IM.getRanges()
        print('shape: ', shape)
        print('LB:\n', LB)
        print('UB:\n', UB)
        print('LB: ', LB.shape)
        print('UB: ', UB.shape)
        I = np.eye(dim)
        C = np.vstack([-I, I])
        d = np.hstack([(np.ones(dim)-0.2), np.ones(dim)-0.2])
        IM1 = ImageStar(IM.V, C, d, IM.pred_lb, IM.pred_ub)
        print(IM1)
        LB1, UB1 = IM1.getRanges()
        print('LB1:\n', LB1)
        print('UB1:\n', UB1)

        IM2 = IM1.transpose()
        LB2, UB2 = IM2.getRanges()
        print('LB2:\n', LB2)
        print('UB2:\n', UB2)

        IM3 = IM1.transpose([1, 0, -1])
        LB3, UB3 = IM3.getRanges()
        print('LB3:\n', LB3)
        print('UB3:\n', UB3)

        print('UB1: ', UB1.shape)
        print('UB2: ', UB2.shape)
        print('UB3: ', UB3.shape)


    def test_sum(self):

        self.n_tests += 1
        
        shape = (3, 2, 1)
        dim = np.prod(shape)
        A = np.arange(dim).reshape(*shape) + 1
        eps = (np.arange(dim).reshape(*shape) + 1)*0.1
        lb, ub = A - eps, A + eps
        IM = ImageStar(lb, ub)

        try:
            IM.sum(axis = 0)
        
        except Exception:
            print('Fail in ImageStar sum function')
            self.n_fails += 1
        
        else:
            print('Test successfull!')
        #TODO: put below to tutorial.
        # shape = (3, 2, 1)
        # dim = np.prod(shape)
        # A = np.arange(dim).reshape(*shape) + 1
        # print('A:\n',A)
        # eps = (np.arange(dim).reshape(*shape) + 1)*0.1
        # lb = A - eps
        # ub = A + eps
        # print('lb:\n',lb)
        # print('ub:\n',ub)
        # IM = ImageStar(lb, ub)
        # print(IM)
        # repr(IM)
        # LBe, UBe = IM.getRanges('estimate')
        # print('LB estimate:\n', LBe.reshape(*shape))
        # print('UB estimate:\n', UBe.reshape(*shape))

        # sum_shape = (1, 2, 1)
        # IM1 = IM.sum(axis=0)
        # print(IM1)
        # repr(IM1)
        # LB1e, UB1e = IM1.getRanges('estimate')
        # print('LB1 estimate:\n', LB1e.reshape(*sum_shape))
        # print('UB1 estimate:\n', UB1e.reshape(*sum_shape))

    def test_sum2(self):

        self.n_tests += 1
        
        shape = (3, 2, 1)
        dim = np.prod(shape)
        A = np.arange(dim).reshape(*shape) + 1
        eps = (np.arange(dim).reshape(*shape) + 1)*0.1
        lb, ub = A - eps, A + eps
        IM = ImageStar(lb, ub)
        I = np.eye(dim)
        C = np.vstack([-I, I])
        d = np.hstack([(np.ones(dim)-0.2), np.ones(dim)-0.2])
        IM1 = ImageStar(IM.V, C, d, IM.pred_lb, IM.pred_ub)
        print('IM')
        print(IM)
        print('IM1')
        print(IM1)
        print('IM')
        repr(IM)
        print('IM1')
        repr(IM1)

        LB, UB = IM.getRanges()
        LB1, UB1 = IM1.getRanges()

        print('LB:\n', LB)
        print('LB1:\n', LB1)
        print()
        print('UB:\n', UB)
        print('UB1:\n', UB1)

        IM2 = IM1.sum(axis=0)
        LB2, UB2 = IM2.getRanges()
        print('LB2:\n', LB2)
        print('UB2:\n', UB2)

        print(IM2)
        repr(IM2)

        print('Testing ImageStar sum')
        try:
            IM.sum(axis = 0)
        
        except Exception:
            print('Fail in ImageStar sum function')
            self.n_fails += 1
        
        else:
            print('Test successfull!')

    def test_element_product(self):
        shape = (3, 2, 1)
        dim = np.prod(shape)
        A = np.arange(dim).reshape(*shape) + 1
        eps = (np.arange(dim).reshape(*shape) + 1)*0.1
        # lb, ub = A - eps, A + eps
        lb, ub = -eps, eps
        IM = ImageStar(lb, ub)
        I = np.eye(dim)
        C = np.vstack([-I, I])
        d = np.hstack([(np.ones(dim)-0.2), np.ones(dim)-0.2])
        IM1 = ImageStar(IM.V, C, d, IM.pred_lb, IM.pred_ub)

        print('IM1')
        print(IM1)
        LB1, UB1 = IM1.getRanges()
        print('LB1\n', LB1)
        print('UB1\n', UB1)

        IM2 = IdentityXIdentity.reach(IM1, IM1)

        print('IM2')
        print(IM2)
        LB2, UB2 = IM2.getRanges()
        print('LB2\n', LB2)
        print('UB2\n', UB2)


        IM3 = IM2.sum(axis=0)
        print(IM3)
        repr(IM3)
        LB3, UB3 = IM3.getRanges()
        print('LB3\n', LB3)
        print('UB3\n', UB3)

    def test_matmul(self):
        shape = (3, 3, 1)
        dim = np.prod(shape)
        A = np.arange(dim).reshape(*shape) + 1
        eps = (np.arange(dim).reshape(*shape) + 1)*0.1
        # lb, ub = A - eps, A + eps
        lb, ub = -eps, eps
        IM = ImageStar(lb, ub)
        I = np.eye(dim)
        C = np.vstack([-I, I])
        d = np.hstack([(np.ones(dim)-0.2), np.ones(dim)-0.2])
        IM1 = ImageStar(IM.V, C, d, IM.pred_lb, IM.pred_ub)

        print('IM1')
        repr(IM1)
        LB1, UB1 = IM1.getRanges()
        print('LB1\n', LB1)
        print('UB1\n', UB1)

        IM2 = IM1.sum(axis=1)

        print('IM2')
        repr(IM2)
        LB2, UB2 = IM2.getRanges()
        print('LB2\n', LB2)
        print('UB2\n', UB2)


        IM3 = MatMul.reach(IM1, IM2)
        print(IM3)
        repr(IM3)
        LB3, UB3 = IM3.getRanges()
        print('LB3\n', LB3)
        print('UB3\n', UB3)

    def test(self):
        shape = (4, 4, 3)
        IM1 = ImageStar.rand_bounds(*shape)
        print('IM1')
        repr(IM1)

        IM2 = ImageStar.rand_bounds(*shape)
        print('IM2')
        repr(IM2)

        C = scipy.linalg.block_diag(IM1.C, IM2.C)
        print(C.shape)



if __name__ == "__main__":

    test_ImageStar = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    # test_ImageStar.test_constructor()
    # test_ImageStar.test_transpose()
    # test_ImageStar.test_sum()
    # test_ImageStar.test_sum2()
    # test_ImageStar.test_element_product()
    test_ImageStar.test_matmul()
    # test_ImageStar.test()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ImageStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_ImageStar.n_fails,
                            test_ImageStar.n_tests - test_ImageStar.n_fails,
                            test_ImageStar.n_tests))