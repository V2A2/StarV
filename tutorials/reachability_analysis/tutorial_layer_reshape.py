import numpy as np
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.layer.ReshapeLayer import ReshapeLayer

def reshape_layer_construct():
    """
    Construct areshape layer
    """
    print('==========================================================================================')
    print('========================== EXAMPLE: Reshape Layer Construction ===========================')
    out_shape = (4, 4, 3)
    to_sparse = True
    to_dense = False
    L_reshape = ReshapeLayer(out_shape, to_sparse=to_sparse)
    print(L_reshape)

    out_shape = (2, 2, 3)
    L_reshape = ReshapeLayer(out_shape)
    L_reshape.to_dense = True
    print(L_reshape)
    print('=========================== DONE: Reshape Layer Construction  ============================')
    print('==========================================================================================\n\n')

def reshape_imagestar():
    """
    Conduct reachability analysis on reshape layer using ImageStar set
    """
    print('==========================================================================================')
    print('============= EXAMPLE: Reachability Analysis on Reshape Layer using ImageStar ============')
    shape = (16, 3)
    data = np.arange(np.prod(shape)).reshape(shape)
    eps = 0.01
    LB = data - eps
    UB = data + eps
    I = ImageStar(LB, UB)
    print('Input ImageStar set:\n')
    repr(I)

    out_shape = (4, 4, 3)
    L_reshape = ReshapeLayer(out_shape)
    R = L_reshape.reach(I)
    print('Output ImageStar set:\n')
    repr(R)
    print('============= DONE: Reachability Analysis on Reshape Layer using ImageStar ===============')
    print('==========================================================================================\n\n')

def reshape_sparseimagestar_csr():
    """
    Conduct reachability analysis on reshape layer using SparseImageStar2DCSR set
    """
    print('==========================================================================================')
    print('======= EXAMPLE: Reachability Analysis on Reshape Layer using SparseImageStar2DCSR =======')
    shape = (16, 3)
    data = np.arange(np.prod(shape)).reshape(shape)
    eps = 0.01
    LB = data - eps
    UB = data + eps
    I = SparseImageStar2DCSR(LB, UB)
    print('Input SparseImageStar2DCSR set:\n')
    repr(I)

    out_shape = (4, 4, 3)
    L_reshape = ReshapeLayer(out_shape)
    R = L_reshape.reach(I)
    print('Output SparseImageStar2DCSR set:\n')
    repr(R)
    print('======== DONE: Reachability Analysis on Reshape Layer using SparseImageStar2DCSR =========')
    print('==========================================================================================\n\n')

def reshape_sparseimagestar_coo():
    """
    Conduct reachability analysis on reshape layer using SparseImageStar2DCOO set
    """
    print('==========================================================================================')
    print('======= EXAMPLE: Reachability Analysis on Reshape Layer using SparseImageStar2DCOO =======')
    shape = (16, 3)
    data = np.arange(np.prod(shape)).reshape(shape)
    eps = 0.01
    LB = data - eps
    UB = data + eps
    I = SparseImageStar2DCOO(LB, UB)
    print('Input SparseImageStar2DCOO set:\n')
    repr(I)

    out_shape = (4, 4, 3)
    L_reshape = ReshapeLayer(out_shape)
    R = L_reshape.reach(I)
    print('Output SparseImageStar2DCOO set:\n')
    repr(R)
    print('======== DONE: Reachability Analysis on Reshape Layer using SparseImageStar2DCOO =========')
    print('==========================================================================================\n\n')

def reshape_sparseimagestar_csr_to_dense():
    """
    Reshape SparseImageStar2DCSR and convert into dense format
    """
    print('==========================================================================================')
    print('======== EXAMPLE: Reshape SparseImageStar2DCSR and convert it to dense format  ===========')
    shape = (4, 4, 3)
    data = np.arange(np.prod(shape)).reshape(shape)
    eps = 0.01
    LB = data - eps
    UB = data + eps
    I = SparseImageStar2DCSR(LB, UB)
    print('Input SparseImageStar2DCSR set:\n')
    repr(I)

    out_shape = (8, 2, 3)
    L_reshape = ReshapeLayer(out_shape, to_dense=True)
    print(L_reshape)

    R = L_reshape.reach(I)
    print('Output SparseImageStar2DCSR set:\n')
    repr(R)
    print('========== DONE: Reshape and convert to dense format of  SparseImageStar2DCSR ============')
    print('==========================================================================================\n\n')

def reshape_sparseimagestar_csr_to_sparse():
    """
    Reshape SparseImageStar2DCSR and convert into sparse format
    """
    print('==========================================================================================')
    print('======== EXAMPLE: Reshape SparseImageStar2DCSR and convert it to sparse format  ==========')
    shape = (4, 4, 3)
    data = np.arange(np.prod(shape)).reshape(shape)
    eps = 0.01
    LB = data - eps
    UB = data + eps
    I = SparseImageStar2DCSR(LB, UB)

    c1 = I.c.reshape(I.shape + (1,))
    V1 = I.V.toarray().reshape(I.shape + (I.num_pred, ))
    V = np.concatenate([c1, V1], axis=3)
    I = SparseImageStar2DCSR(V, I.C, I.d, I.pred_lb, I.pred_ub, I.shape)
    print('Input SparseImageStar2DCSR set:\n')
    repr(I)

    out_shape = (8, 2, 3)
    L_reshape = ReshapeLayer(out_shape, to_sparse=True)
    print(L_reshape)

    R = L_reshape.reach(I)
    print('Output SparseImageStar2DCSR set:\n')
    repr(R)
    print('========== DONE: Reshape and convert to sparse format of  SparseImageStar2DCSR ===========')
    print('==========================================================================================\n\n')


def reshape_sparseimagestar_coo_to_dense():
    """
    Reshape SparseImageStar2DCOO and convert into dense format
    """
    print('==========================================================================================')
    print('======== EXAMPLE: Reshape SparseImageStar2DCOO and convert it to dense format  ===========')
    shape = (4, 4, 3)
    data = np.arange(np.prod(shape)).reshape(shape)
    eps = 0.01
    LB = data - eps
    UB = data + eps
    I = SparseImageStar2DCOO(LB, UB)
    print('Input SparseImageStar2DCOO set:\n')
    repr(I)

    out_shape = (8, 2, 3)
    L_reshape = ReshapeLayer(out_shape, to_dense=True)
    print(L_reshape)

    R = L_reshape.reach(I)
    print('Output SparseImageStar2DCOO set:\n')
    repr(R)
    print('========== DONE: Reshape and convert to dense format of  SparseImageStar2DCOO ============')
    print('==========================================================================================\n\n')

def reshape_sparseimagestar_coo_to_sparse():
    """
    Reshape SparseImageStar2DCOO and convert into sparse format
    """
    print('==========================================================================================')
    print('======== EXAMPLE: Reshape SparseImageStar2DCOO and convert it to sparse format  ==========')
    shape = (4, 4, 3)
    data = np.arange(np.prod(shape)).reshape(shape)
    eps = 0.01
    LB = data - eps
    UB = data + eps
    I = SparseImageStar2DCOO(LB, UB)

    c1 = I.c.reshape(I.shape + (1,))
    V1 = I.V.toarray().reshape(I.shape + (I.num_pred, ))
    V = np.concatenate([c1, V1], axis=3)
    I = SparseImageStar2DCOO(V, I.C, I.d, I.pred_lb, I.pred_ub, I.shape)
    print('Input SparseImageStar2DCOO set:\n')
    repr(I)

    out_shape = (8, 2, 3)
    L_reshape = ReshapeLayer(out_shape, to_sparse=True)
    print(L_reshape)

    R = L_reshape.reach(I)
    print('Output SparseImageStar2DCOO set:\n')
    repr(R)
    print('========== DONE: Reshape and convert to sparse format of  SparseImageStar2DCOO ===========')
    print('==========================================================================================\n\n')

    
if __name__ == "__main__":
    reshape_layer_construct()
    reshape_imagestar()
    reshape_sparseimagestar_csr()
    reshape_sparseimagestar_coo()
    reshape_sparseimagestar_csr_to_dense()
    reshape_sparseimagestar_csr_to_sparse()
    reshape_sparseimagestar_coo_to_dense()
    reshape_sparseimagestar_coo_to_sparse()
    