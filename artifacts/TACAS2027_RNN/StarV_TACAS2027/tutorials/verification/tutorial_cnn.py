import os
import scipy
import numpy as np
import scipy.sparse as sp
import StarV
from StarV.util.load import load_convnet
from StarV.util.attack import brightening_attack
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.verifier.certifier import reachBFS, certifyRobustness
from StarV.layer.Conv2DLayer import Conv2DLayer

path = os.path.dirname(StarV.__file__)

def cnn_rechability_imagestar():
    
    dtype='float32'
    net_type = 'Small'
    folder_dir = f'{path}/util/data/nets/CAV2020_MNIST_ConvNet'
    net_dir = f"{folder_dir}/onnx/{net_type}_ConvNet.onnx"
    print(net_dir)
    starvNet = load_convnet(net_dir, net_type, dtype=dtype)
    print()
    print(starvNet.info())
    
    mat_file = scipy.io.loadmat(f"{folder_dir}/nnv/{net_type}_images.mat")
    data = mat_file['IM_data'].astype(dtype)[:, :, 0]
    label = (mat_file['IM_labels'] - 1)[0]
    
    delta = 0.005
    d = 250
    lb, ub = brightening_attack(data, delta=delta, d=d, dtype=dtype)
    IM = ImageStar(lb, ub)
    print('Input ImageStar:')
    repr(IM)
    
    reachMethod = 'approx'
    lp_solver = 'gurobi'
    pool = None
    RF = 0.0
    DR = 0
    show  = False 
    R, _ = reachBFS(net=starvNet, inputSet=IM, reachMethod=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show) 
    print('Output ImageStar after reachability analysis:')
    repr(R)
    
def cnn_rechability_sparseimagestar_coo():
    
    dtype='float32'
    net_type = 'Small'
    folder_dir = f'{path}/util/data/nets/CAV2020_MNIST_ConvNet'
    net_dir = f"{folder_dir}/onnx/{net_type}_ConvNet.onnx"
    starvNet = load_convnet(net_dir, net_type, dtype=dtype)
    print()
    print(starvNet.info())
    
    mat_file = scipy.io.loadmat(f"{folder_dir}/nnv/{net_type}_images.mat")
    data = mat_file['IM_data'].astype(dtype)[:, :, 0]
    label = (mat_file['IM_labels'] - 1)[0]
    
    delta = 0.005
    d = 250
    lb, ub = brightening_attack(data, delta=delta, d=d, dtype=dtype)
    COO = SparseImageStar2DCOO(lb, ub)
    print('Input SparseImageStar COO:')
    repr(COO)
    
    reachMethod = 'approx'
    lp_solver = 'gurobi'
    pool = None
    RF = 0.0
    DR = 0
    show  = False 
    R, _ = reachBFS(net=starvNet, inputSet=COO, reachMethod=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show) 
    print('Output SparseImageStar COO after reachability analysis:')
    repr(R)


def cnn_rechability_sparseimagestar_csr():
    
    dtype='float32'
    net_type = 'Small'
    folder_dir = f'{path}/util/data/nets/CAV2020_MNIST_ConvNet'
    net_dir = f"{folder_dir}/onnx/{net_type}_ConvNet.onnx"
    starvNet = load_convnet(net_dir, net_type, dtype=dtype)
    print()
    print(starvNet.info())
    
    mat_file = scipy.io.loadmat(f"{folder_dir}/nnv/{net_type}_images.mat")
    data = mat_file['IM_data'].astype(dtype)[:, :, 0]
    label = (mat_file['IM_labels'] - 1)[0]
    
    delta = 0.005
    d = 250
    lb, ub = brightening_attack(data, delta=delta, d=d, dtype=dtype)
    CSR = SparseImageStar2DCSR(lb, ub)
    print('Input SparseImageStar CSR:')
    repr(CSR)
    
    reachMethod = 'approx'
    lp_solver = 'gurobi'
    pool = None
    RF = 0.0
    DR = 0
    show  = False 
    R, _ = reachBFS(net=starvNet, inputSet=CSR, reachMethod=reachMethod, lp_solver=lp_solver, pool=pool, RF=RF, DR=DR, show=show) 
    print('Output SparseImageStar CSR after reachability analysis:')
    repr(R)

    
def cnn_certification_imagestar():
    
    dtype='float32'
    net_type = 'Small'
    folder_dir = f'{path}/util/data/nets/CAV2020_MNIST_ConvNet'
    net_dir = f"{folder_dir}/onnx/{net_type}_ConvNet.onnx"
    starvNet = load_convnet(net_dir, net_type, dtype=dtype)
    print()
    print(starvNet.info())
    
    mat_file = scipy.io.loadmat(f"{folder_dir}/nnv/{net_type}_images.mat")
    data = mat_file['IM_data'].astype(dtype)[:, :, 0]
    label = (mat_file['IM_labels'] - 1)[0]
    
    delta = 0.005
    d = 250
    lb, ub = brightening_attack(data, delta=delta, d=d, dtype=dtype)
    IM = ImageStar(lb, ub)
    print('Input ImageStar:')
    repr(IM)
    
    reachMethod = 'approx'
    lp_solver = 'gurobi'
    pool = None
    RF = 0.0
    DR = 0
    show  = False 
    veriMethod = 'BFS'
    rb, vt, _, _ = certifyRobustness(net=starvNet, inputs=IM, labels=label,
                veriMethod=veriMethod, reachMethod=reachMethod, lp_solver=lp_solver, pool=pool,
                RF=RF, DR=DR, return_output=False, show=show)
    rb = 'Robust' if rb else 'Unknown'
    print('Certification result: ', rb)
    print(f'Verification time: {vt} sec')
    
def cnn_certification_sparseimagestar_coo():
    
    dtype='float32'
    net_type = 'Small'
    folder_dir = f'{path}/util/data/nets/CAV2020_MNIST_ConvNet'
    net_dir = f"{folder_dir}/onnx/{net_type}_ConvNet.onnx"
    starvNet = load_convnet(net_dir, net_type, dtype=dtype)
    print()
    print(starvNet.info())
    
    mat_file = scipy.io.loadmat(f"{folder_dir}/nnv/{net_type}_images.mat")
    data = mat_file['IM_data'].astype(dtype)[:, :, 0]
    label = (mat_file['IM_labels'] - 1)[0]
    
    delta = 0.005
    d = 250
    lb, ub = brightening_attack(data, delta=delta, d=d, dtype=dtype)
    COO = SparseImageStar2DCOO(lb, ub)
    print('Input SparseImageStar COO:')
    repr(COO)
    
    reachMethod = 'approx'
    lp_solver = 'gurobi'
    pool = None
    RF = 0.0
    DR = 0
    show  = False 
    veriMethod = 'BFS'
    rb, vt, _, _ = certifyRobustness(net=starvNet, inputs=COO, labels=label,
                veriMethod=veriMethod, reachMethod=reachMethod, lp_solver=lp_solver, pool=pool,
                RF=RF, DR=DR, return_output=False, show=show)
    rb = 'Robust' if rb else 'Unknown'
    print('Certification result: ', rb)
    print(f'Verification time: {vt} sec')

def cnn_certification_sparseimagestar_csr():
    
    dtype='float32'
    net_type = 'Small'
    folder_dir = f'{path}/util/data/nets/CAV2020_MNIST_ConvNet'
    net_dir = f"{folder_dir}/onnx/{net_type}_ConvNet.onnx"
    starvNet = load_convnet(net_dir, net_type, dtype=dtype)
    print()
    print(starvNet.info())
    
    mat_file = scipy.io.loadmat(f"{folder_dir}/nnv/{net_type}_images.mat")
    data = mat_file['IM_data'].astype(dtype)[:, :, 0]
    label = (mat_file['IM_labels'] - 1)[0]
    
    delta = 0.005
    d = 250
    lb, ub = brightening_attack(data, delta=delta, d=d, dtype=dtype)
    CSR = SparseImageStar2DCSR(lb, ub)
    print('Input SparseImageStar CSR:')
    repr(CSR)
    
    reachMethod = 'approx'
    lp_solver = 'gurobi'
    pool = None
    RF = 0.0
    DR = 0
    show  = False 
    veriMethod = 'BFS'
    rb, vt, _, _ = certifyRobustness(net=starvNet, inputs=CSR, labels=label,
                veriMethod=veriMethod, reachMethod=reachMethod, lp_solver=lp_solver, pool=pool,
                RF=RF, DR=DR, return_output=False, show=show)
    rb = 'Robust' if rb else 'Unknown'
    print('Certification result: ', rb)
    print(f'Verification time: {vt} sec')

if __name__ == '__main__':
    cnn_rechability_imagestar()
    cnn_rechability_sparseimagestar_coo()
    cnn_rechability_sparseimagestar_csr()
    cnn_certification_imagestar()
    cnn_certification_sparseimagestar_coo()
    cnn_certification_sparseimagestar_csr()