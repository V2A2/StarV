import numpy as np
import scipy.sparse as sp

from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR

class Test(object):
    """
       Testing SparseImageStarCSR class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_hstack_csr(self):

        self.n_tests += 1

        print('=================================================\
        ==============================================')
        print('Testing SparseImageStar2DCSR.hstack_csr...')

        A_data = np.array([1, 2, 3, 4])
        A_rows = np.array([0, 0, 1, 2])
        A_cols = np.array([0, 2, 2, 0])
        A_shape = (3, 3)
        A = sp.csr_array((A_data, (A_rows, A_cols)), shape=A_shape)
        print('A shape: ', A.shape)
        print('A format: ', A.format)
        print('A.data   : ', A.data)
        print('A.indices: ', A.indices)
        print('A.indptr : ', A.indptr)

        B_data = np.array([5, 6, 7])
        B_rows = np.array([0, 1, 2])
        B_cols = np.array([1, 0, 2])
        B_shape = (3, 3)
        B = sp.csr_array((B_data, (B_rows, B_cols)), shape=B_shape)
        print('B shape: ', B.shape)
        print('B format: ', B.format)
        print('B.data   : ', B.data)
        print('B.indices: ', B.indices)
        print('B.indptr : ', B.indptr)

        C = SparseImageStar2DCSR.hstack_csr(A, B, A_shape, B_shape)
        print('C shape: ', C.shape)
        print('C format: ', C.format)
        print('C.data   : ', C.data)
        print('C.indices: ', C.indices)
        print('C.indptr : ', C.indptr)

        expected_data = np.array([1, 2, 5, 3, 6, 4, 7])
        expected_rows = np.array([0, 0, 0, 1, 1, 2, 2])
        expected_cols = np.array([0, 2, 4, 2, 3, 0, 5])
        expected_shape = (3, 6)
        expected_C = sp.csr_array((expected_data, (expected_rows, expected_cols)), shape=expected_shape)
        print('Expected C shape: ', expected_C.shape)
        print('Expected C format: ', expected_C.format)
        print('Expected C.data   : ', expected_C.data)
        print('Expected C.indices: ', expected_C.indices)
        print('Expected C.indptr : ', expected_C.indptr)

        print('A:\n', A.toarray())
        print('B:\n', B.toarray())
        print('C:\n', C.toarray())
        print('Expected C:\n', expected_C.toarray())

        try:
            assert (C != expected_C).nnz == 0
        except AssertionError:
            self.n_fails += 1
            print('Fail in SparseImageStar2DCSR.hstack_csr')
        else:
            print('Test Successful!')

    def test_block_diag_csr(self):
        
        self.n_tests += 1

        print('=================================================\
        ==============================================')
        print('Testing SparseImageStar2DCSR.block_diag_csr...')

        A_data = np.array([1, 2, 3, 4])
        A_rows = np.array([0, 0, 1, 2])
        A_cols = np.array([0, 2, 2, 0])
        A_shape = (3, 3)
        A = sp.csr_array((A_data, (A_rows, A_cols)), shape=A_shape)
        print('A shape: ', A.shape)
        print('A format: ', A.format)
        print('A.data   : ', A.data)
        print('A.indices: ', A.indices)
        print('A.indptr : ', A.indptr)

        B_data = np.array([5, 6, 7])
        B_rows = np.array([0, 1, 2])
        B_cols = np.array([1, 0, 2])
        B_shape = (3, 3)
        B = sp.csr_array((B_data, (B_rows, B_cols)), shape=B_shape)
        print('B shape: ', B.shape)
        print('B format: ', B.format)
        print('B.data   : ', B.data)
        print('B.indices: ', B.indices)
        print('B.indptr : ', B.indptr)

        C = SparseImageStar2DCSR.block_diag_csr(A, B, A_shape, B_shape)
        print('C shape: ', C.shape)
        print('C format: ', C.format)
        print('C.data   : ', C.data)
        print('C.indices: ', C.indices)
        print('C.indptr : ', C.indptr)
        
        expected_data = np.array([1, 2, 3, 4, 5, 6, 7])
        expected_rows = np.array([0, 0, 1, 2, 3, 4, 5])
        expected_cols = np.array([0, 2, 2, 0, 4, 3, 5])
        expected_shape = (6, 6)
        expected_C = sp.csr_array((expected_data, (expected_rows, expected_cols)), shape=expected_shape)
        print('Expected C shape: ', expected_C.shape)
        print('Expected C format: ', expected_C.format)
        print('Expected C.data   : ', expected_C.data)
        print('Expected C.indices: ', expected_C.indices)
        print('Expected C.indptr : ', expected_C.indptr)

        print('A:\n', A.toarray())
        print('B:\n', B.toarray())
        print('C:\n', C.toarray())
        print('Expected C:\n', expected_C.toarray())

        try:
            assert (C != expected_C).nnz == 0
        except AssertionError:
            self.n_fails += 1
            print('Fail in SparseImageStar2DCSR.block_diag_csr')
        else:
            print('Test Successful!')

    def test_minkowskisum(self):
        
        self.n_tests += 1

        print('=================================================\
        ==============================================')
        print('Testing SparseImageStar2DCSR.minKowskiSum...')

        h, w, ch, m = 2, 2, 3, 4
        dim = h * w * ch
        shape = (h, w, ch)
        c = np.arange(dim)
        V = 0.1 * np.arange(dim * m).reshape(dim, m)
        V = sp.csr_array(V)
        C = np.array([[-1, 0, 0, -1],
                      [1, 0, 0, -1],
                      [0, -1, 0, -1],
                      [0, 1, 0, -1]])
        C = sp.csr_array(C)
        d = np.array([-0.5, -0.5, -0.5, -0.5])
        pred_lb = -np.ones(m)
        pred_ub = np.ones(m)
        SIM1 = SparseImageStar2DCSR(c, V, C, d, pred_lb, pred_ub,shape)

        IM = np.arange(dim).reshape(h, w, ch) * 0.2
        LB = IM - 0.3
        UB = IM + 0.3
        SIM2 = SparseImageStar2DCSR(LB, UB)

        print('SIM1 representation:\n', repr(SIM1))
        print('SIM2 representation:\n', repr(SIM2))
        SIM3 = SIM1.minKowskiSum(SIM2)

        V1 = np.hstack((SIM1.c[:, None], SIM1.V.toarray())).reshape(h, w, ch, -1)
        IM1 = ImageStar(V1, SIM1.C.toarray(), SIM1.d, SIM1.pred_lb, SIM1.pred_ub)
        V2 = np.hstack((SIM2.c[:, None], SIM2.V.toarray())).reshape(h, w, ch, -1)
        IM2 = ImageStar(V2, SIM2.C.toarray(), SIM2.d, SIM2.pred_lb, SIM2.pred_ub)

        print('IM1 representation:\n', repr(IM1))
        print('IM2 representation:\n', repr(IM2))
        IM3 = IM1.minKowskiSum(IM2)

        print('Checking the results of minKowskiSum between SparseImageStar2DCSR' +\
        'and ImageStar representations...')
        try:
            slb, sub = SIM3.getRanges()
            ilb, iub = IM3.getRanges()
            assert np.allclose(slb, ilb.ravel()) and np.allclose(sub, iub.ravel())
        except AssertionError:
            self.n_fails += 1
            print('Fail in SparseImageStar2DCSR.minKowskiSum')
        else:
            print('Test Successful!')


    # def test_concatenate(self):
        
    #     self.n_tests += 1

    #     print('=================================================\
    #     ==============================================')
    #     print('Testing SparseImageStar2DCSR.concatenate...')

    #     h, w, ch, m = 2, 2, 3, 4
    #     dim = h * w * ch
    #     shape = (h, w, ch)
    #     c = np.arange(dim)
    #     V = 0.1 * np.arange(dim * m).reshape(dim, m)
    #     V = sp.csr_array(V)
    #     C = np.array([[-1, 0, 0, -1],
    #                   [1, 0, 0, -1],
    #                   [0, -1, 0, -1],
    #                   [0, 1, 0, -1]])
    #     C = sp.csr_array(C)
    #     d = np.array([-0.5, -0.5, -0.5, -0.5])
    #     pred_lb = -np.ones(m)
    #     pred_ub = np.ones(m)
    #     SIM1 = SparseImageStar2DCSR(c, V, C, d, pred_lb, pred_ub,shape)

    #     IM = np.arange(dim).reshape(h, w, ch) * 0.2
    #     LB = IM - 0.3
    #     UB = IM + 0.3
    #     SIM2 = SparseImageStar2DCSR(LB, UB)

    #     print('SIM1 representation:\n', repr(SIM1))
    #     print('SIM2 representation:\n', repr(SIM2))
    #     SIM3 = SIM1.concatenate(SIM2)

    #     V1 = np.hstack((SIM1.c[:, None], SIM1.V.toarray())).reshape(h, w, ch, -1)
    #     IM1 = ImageStar(V1, SIM1.C.toarray(), SIM1.d, SIM1.pred_lb, SIM1.pred_ub)
    #     V2 = np.hstack((SIM2.c[:, None], SIM2.V.toarray())).reshape(h, w, ch, -1)
    #     IM2 = ImageStar(V2, SIM2.C.toarray(), SIM2.d, SIM2.pred_lb, SIM2.pred_ub)

    #     print('IM1 representation:\n', repr(IM1))
    #     print('IM2 representation:\n', repr(IM2))
    #     IM3 = IM1.concatenate(IM2)

    #     print('Checking the results of concatenate between SparseImageStar2DCSR' +\
    #     'and ImageStar representations...')
    #     try:
    #         slb, sub = SIM3.getRanges()
    #         ilb, iub = IM3.getRanges()
    #         assert np.allclose(slb, ilb.ravel()) and np.allclose(sub, iub.ravel())
    #     except AssertionError:
    #         self.n_fails += 1
    #         print('Fail in SparseImageStar2DCSR.concatenate')
    #     else:
    #         print('Test Successful!')


    def test_concatenate_axis(self):
        
        self.n_tests += 1

        print('=================================================\
        ==============================================')
        print('Testing SparseImageStar2DCSR.concatenate...')

        h, w, ch = 4, 4, 3
        dim = h * w * ch
        shape = (h, w, ch)
        axis = 1
        im1 = np.arange(dim).reshape(h, w, ch) * 0.1
        eps = 0.2
        LB1 = im1 - eps
        UB1 = im1 + eps
        SIM1 = SparseImageStar2DCSR(LB1, UB1)
        IM1 = ImageStar(LB1, UB1)

        im2 = np.arange(dim).reshape(h, w, ch)
        eps = 0.3
        LB2 = im2 - eps
        UB2 = im2 + eps
        SIM2 = SparseImageStar2DCSR(LB2, UB2)
        IM2 = ImageStar(LB2, UB2)
        print('SIM1 representation:\n', repr(SIM1))
        print('SIM2 representation:\n', repr(SIM2))
        SIM3 = SIM1.concatenate(SIM2, axis=axis)
        IM3 = IM1.concatenate(IM2, axis=axis)

        print('Checking the results of concatenate between SparseImageStar2DCSR' +\
        'and ImageStar representations...')
        try:
            slb, sub = SIM3.getRanges()
            ilb, iub = IM3.getRanges()
            assert np.allclose(slb, ilb.ravel()) and np.allclose(sub, iub.ravel())
        except AssertionError:
            self.n_fails += 1
            print('Fail in SparseImageStar2DCSR.concatenate')
        else:
            print('Test Successful!')
        
if __name__ == '__main__':
    test = Test()
    # test.test_hstack_csr()
    # test.test_block_diag_csr()
    # test.test_minkowskisum()
    # test.test_concatenate()
    test.test_concatenate_axis()

    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing SparseImageStar2DCSR   Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test.n_fails,
                            test.n_tests - test.n_fails,
                            test.n_tests))