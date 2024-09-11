import numpy as np
import scipy
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle

import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.verifier.certifier import certifyRobustness_pixel
from StarV.util.load import *
from sklearn.metrics import jaccard_score

def pixel_attack(image, format='IM', data_type='mnist', N_max=20, br=150, delta=0.001):
    """
        @N_max: maximum allowable number of attacked pixels
        @delta: size of input sets
        @br: threshold to set pixel to 0
    """
    assert N_max <= np.prod(image.shape), f"number of maximum attacked pixels shouldn't be greater than number of image pixels"
    assert format in ['IM', 'SIM_coo', 'SIM_csr'], f"return_class should be \'IM\', \'SIM_coo\', \'SIM_csr\', but received {format}"
    
    if data_type == 'mnist':
        h = image.shape[0] - 1
        w = image.shape[1] - 1
    else:
        h, w = 63, 83
    attack_image = image.copy() 
    cnt = 0
    for i in range(h):
        for j in range(w):
            if image[h, w] > br:
                attack_image[i, j] = 0
                cnt += 1
                if cnt == N_max:
                    break
        else:
             continue
        break
        
    noise = attack_image - image
    V = np.concatenate([image[:, :, None, None], noise[:, :, None, None]], axis=3)
    C = np.array([[1], [-1]])
    d = np.array([1, delta-1])
    IM = ImageStar(V, C, d, np.array([1-delta]), np.array([1]))
    if format == 'IM':
        return IM
    elif format == 'SIM_coo':
        return IM.toSIM('coo')
    else:
        return IM.toSIM('csr')
    
def attack_multiple_images(images, format='IM', data_type='mnist', N_max=[20], br=150, delta=0.001):
    # @N_max: a list containing maximum allowable number of attacked pixels
    # @delta: sizes of input sets
    # @br: threshold to set pixel to 0

    assert isinstance(N_max, list), f"N_max should be a list containing sizes of input sets, but received {type(N_max)}"
    outputs = []
    for n_max in range(len(N_max)):
        out_nmax = []
        for image in images:
            out_nmax.append(pixel_attack(image, format, data_type, n_max, br, delta))
        outputs.append(out_nmax)
    return outputs
            
def load_dataset(folder_dir, data_type, dtype='float64'):
    if data_type == 'mnist':
        mat_file = scipy.io.loadmat(f"{folder_dir}/test_images.mat")
    else:
        mat_file = scipy.io.loadmat(f"{folder_dir}/m2nist_6484_test_images.mat")
    data = mat_file['im_data'].astype(dtype)
    labels = mat_file['im_labels'].ravel() - 1
    return data, labels

def load_network(folder_dir, net_type='relu', data_type='mnist', dtype='float64'):
    assert data_type in ['mnist', 'm2nist'], f"data_type should be either 'mnist' or 'm2nist', but received {data_type}"
    if data_type == 'mnist':
        assert net_type in ['dilated', 'relu_maxpool', 'relu']
        if net_type == 'dilated':
            net_dir = f"{folder_dir}/mnist_{net_type}_net_21_later_83iou.onnx"
        else:
            net_dir = f"{folder_dir}/net_mnist_3_{net_type}.onnx"
    else:
        assert net_type in ['dilatedcnn_avgpool', 'transposedcnn_avgpool', 'dilated']
        if net_type == 'dilated':
            net_dir = f"{folder_dir}/m2nist_{net_type}_72iou_24layer.onnx"
        elif net_type == 'dilatedcnn_avgpool':
            net_dir = f"{folder_dir}/m2nist_62iou_{net_type}.onnx"
        else:
            net_dir = f"{folder_dir}/m2nist_75iou_{net_type}.onnx"
            
    return load_onnx_network(net_dir, num_pixel_classes=10, dtype=dtype)


def verify_CAV2021_MNIST_SSNN(net_type, dtype='float64'):
    print('=========================================================================================')
    print(f"Verification of CAV2021 MNIST Semantic Segmentation Neural Netowrk against Pixel Attacks")
    print('=========================================================================================\n')
    
    data_type = 'mnist'
    folder_dir = f"./SparseImageStar_evaluation/CAV2021_SSNN/MNIST/"
    starvNet = load_network(folder_dir, net_type, data_type, dtype)
    
    print(starvNet.info())
    data, labels = load_dataset(folder_dir, data_type)
    N = 20 # number of tested images
    labels = labels[:N]
    images = [data[:, :, i] for i in range(N)]
    
    N_max = [10, 20, 30, 40, 50]
    M = len(N_max)
    br = 150
    delta = 0.001
    

    print(f"Verifying {net_type} SSNN with ImageStar")
    avg_numRbIM = np.zeros(M)
    avg_numUnkIM = np.zeros(M)
    avg_numMisIM = np.zeros(M)
    avg_numAttIM = np.zeros(M)
    avg_riouIM = np.zeros(M)
    avg_rvIM = np.zeros(M)
    avg_rsIM = np.zeros(M)
    avg_vtIM = np.zeros(M)
    
    IM_sets = attack_multiple_images(images, 'IM', data_type, N_max, br, delta) #returns a list in shape len(N_max) x len(images)
    
    for i in range(M):
        _, _, _, _, avg_data = certifyRobustness_pixel(net=starvNet, in_sets=IM_sets[i], in_datas=images,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=False)
        avg_numRbIM[i], avg_numUnkIM[i], avg_numMisIM[i], avg_numAttIM[i], avg_riouIM[i], avg_rvIM[i], avg_rsIM[i], avg_vtIM[i] = avg_data

    
    print(f"\nVerifying {net_type} SSNN with Sparse Image Star in CSR format")
    avg_numRbCSR = np.zeros(M)
    avg_numUnkCSR = np.zeros(M)
    avg_numMisCSR = np.zeros(M)
    avg_numAttCSR = np.zeros(M)
    avg_riouCSR = np.zeros(M)
    avg_rvCSR = np.zeros(M)
    avg_rsCSR = np.zeros(M)
    avg_vtCSR = np.zeros(M)
    
    CSR_sets = attack_multiple_images(images, 'SIM_csr', data_type, N_max, br, delta) #returns a list in shape len(N_max) x len(images)
    
    for i in range(M):
        _, _, _, _, avg_data = certifyRobustness_pixel(net=starvNet, in_sets=CSR_sets[i], in_datas=images,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=False)
        avg_numRbCSR[i], avg_numUnkCSR[i], avg_numMisCSR[i], avg_numAttCSR[i], avg_riouCSR[i], avg_rvCSR[i], avg_rsCSR[i], avg_vtCSR[i] = avg_data
        
        
    print(f"\nVerifying {net_type} SSNN with Sparse Image Star in COO format")
    avg_numRbCOO = np.zeros(M)
    avg_numUnkCOO = np.zeros(M)
    avg_numMisCOO = np.zeros(M)
    avg_numAttCOO = np.zeros(M)
    avg_riouCOO = np.zeros(M)
    avg_rvCOO = np.zeros(M)
    avg_rsCOO = np.zeros(M)
    avg_vtCOO = np.zeros(M)
    
    COO_sets = attack_multiple_images(images, 'SIM_coo', data_type, N_max, br, delta) #returns a list in shape len(N_max) x len(images)
    
    for i in range(M):
        _, _, _, _, avg_data = certifyRobustness_pixel(net=starvNet, in_sets=COO_sets[i], in_datas=images,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=False)
        avg_numRbCOO[i], avg_numUnkCOO[i], avg_numMisCOO[i], avg_numAttCOO[i], avg_riouCOO[i], avg_rvCOO[i], avg_rsCOO[i], avg_vtCOO[i] = avg_data

    IM_data = [avg_numRbIM, avg_numUnkIM, avg_numMisIM, avg_numAttIM, avg_riouIM, avg_rvIM, avg_rsIM, avg_vtIM]
    CSR_data = [avg_numRbCSR, avg_numUnkCSR, avg_numMisCSR, avg_numAttCSR, avg_riouCSR, avg_rvCSR, avg_rsCSR, avg_vtCSR]
    COO_data = [avg_numRbCOO, avg_numUnkCOO, avg_numMisCOO, avg_numAttCOO, avg_riouCOO, avg_rvCOO, avg_rsCOO, avg_vtCOO]
    
    path = f"./SparseImageStar_evaluation/results"
    save_file = path + f"/{net_type}SSNN_MNIST_CAV2021_results.pkl"
    pickle.dump([IM_data, CSR_data, COO_data], open(save_file, "wb"))
    print('=====================================================')
    print('DONE!')
    print('=====================================================')
    

def verify_CAV2021_M2NIST_SSNN(net_type, dtype='float64'):
    print('==========================================================================================')
    print(f"Verification of CAV2021 M2NIST Semantic Segmentation Neural Netowrk against Pixel Attacks")
    print('==========================================================================================\n')
    
    data_type = 'm2nist'
    folder_dir = f"./SparseImageStar_evaluation/CAV2021_SSNN/M2NIST/"
    starvNet = load_network(folder_dir, net_type, data_type, dtype)
    
    print(starvNet.info())
    data, labels = load_dataset(folder_dir, data_type)
    N = 20 # number of tested images
    labels = labels[:N]
    images = [data[:, :, i] for i in range(N)]
    
    N_max = [10, 20, 30, 40, 50]
    M = len(N_max)
    br = 150
    delta = 0.001

    
    print(f"Verifying {net_type} SSNN with ImageStar")
    avg_numRbIM = np.zeros(M)
    avg_numUnkIM = np.zeros(M)
    avg_numMisIM = np.zeros(M)
    avg_numAttIM = np.zeros(M)
    avg_riouIM = np.zeros(M)
    avg_rvIM = np.zeros(M)
    avg_rsIM = np.zeros(M)
    avg_vtIM = np.zeros(M)
    
    IM_sets = attack_multiple_images(images, 'IM', data_type, N_max, br, delta) #returns a list in shape len(N_max) x len(images)
    
    for i in range(M):
        _, _, _, _, avg_data = certifyRobustness_pixel(net=starvNet, in_sets=IM_sets[i], in_datas=images,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=False)
        avg_numRbIM[i], avg_numUnkIM[i], avg_numMisIM[i], avg_numAttIM[i], avg_riouIM[i], avg_rvIM[i], avg_rsIM[i], avg_vtIM[i] = avg_data

    
    print(f"\nVerifying {net_type} SSNN with Sparse Image Star in CSR format")
    avg_numRbCSR = np.zeros(M)
    avg_numUnkCSR = np.zeros(M)
    avg_numMisCSR = np.zeros(M)
    avg_numAttCSR = np.zeros(M)
    avg_riouCSR = np.zeros(M)
    avg_rvCSR = np.zeros(M)
    avg_rsCSR = np.zeros(M)
    avg_vtCSR = np.zeros(M)
    
    CSR_sets = attack_multiple_images(images, 'SIM_csr', data_type, N_max, br, delta) #returns a list in shape len(N_max) x len(images)
    
    for i in range(M):
        _, _, _, _, avg_data = certifyRobustness_pixel(net=starvNet, in_sets=CSR_sets[i], in_datas=images,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=False)
        avg_numRbCSR[i], avg_numUnkCSR[i], avg_numMisCSR[i], avg_numAttCSR[i], avg_riouCSR[i], avg_rvCSR[i], avg_rsCSR[i], avg_vtCSR[i] = avg_data
        
        
    print(f"\nVerifying {net_type} SSNN with Sparse Image Star in COO format")
    avg_numRbCOO = np.zeros(M)
    avg_numUnkCOO = np.zeros(M)
    avg_numMisCOO = np.zeros(M)
    avg_numAttCOO = np.zeros(M)
    avg_riouCOO = np.zeros(M)
    avg_rvCOO = np.zeros(M)
    avg_rsCOO = np.zeros(M)
    avg_vtCOO = np.zeros(M)
    
    COO_sets = attack_multiple_images(images, 'SIM_coo', data_type, N_max, br, delta) #returns a list in shape len(N_max) x len(images)
    
    for i in range(M):
        _, _, _, _, avg_data = certifyRobustness_pixel(net=starvNet, in_sets=COO_sets[i], in_datas=images,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=False)
        avg_numRbCOO[i], avg_numUnkCOO[i], avg_numMisCOO[i], avg_numAttCOO[i], avg_riouCOO[i], avg_rvCOO[i], avg_rsCOO[i], avg_vtCOO[i] = avg_data

    IM_data = [avg_numRbIM, avg_numUnkIM, avg_numMisIM, avg_numAttIM, avg_riouIM, avg_rvIM, avg_rsIM, avg_vtIM]
    CSR_data = [avg_numRbCSR, avg_numUnkCSR, avg_numMisCSR, avg_numAttCSR, avg_riouCSR, avg_rvCSR, avg_rsCSR, avg_vtCSR]
    COO_data = [avg_numRbCOO, avg_numUnkCOO, avg_numMisCOO, avg_numAttCOO, avg_riouCOO, avg_rvCOO, avg_rsCOO, avg_vtCOO]
    
    path = f"./SparseImageStar_evaluation/results"
    save_file = path + f"/{net_type}SSNN_M2NIST_CAV2021_results.pkl"
    pickle.dump([IM_data, CSR_data, COO_data], open(save_file, "wb"))

    print('=====================================================')
    print('DONE!')
    print('=====================================================')
    

if __name__ == "__main__":
    verify_CAV2021_MNIST_SSNN(net_type='relu', dtype='float64')
    verify_CAV2021_MNIST_SSNN(net_type='relu_maxpool', dtype='float64')
    verify_CAV2021_MNIST_SSNN(net_type='dilated', dtype='float64')
    verify_CAV2021_M2NIST_SSNN(net_type='dilatedcnn_avgpool', dtype='float64')
    verify_CAV2021_M2NIST_SSNN(net_type='transposedcnn_avgpool', dtype='float64')
    verify_CAV2021_M2NIST_SSNN(net_type='dilated', dtype='float64')