
"""
Memory-Efficient Verification of Semantic Segmentation Neural Networks
Artifact Evaluation
"""

import os
import time
import torch
import numpy as np
import scipy.sparse as sp
from skimage.morphology import skeletonize
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no X11 display needed)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import pickle
import scipy

from tabulate import tabulate
from StarV.util.load import *
from StarV.util.data.nets.UNET_Kvasir.train_unet_kvasir_avgpool import KvasirSegDataset as KvasirDataset
from StarV.util.data.nets.UNET_Kvasir.train_unet_kvasir_avgpool import UNetAvgPool as Kvasir_UNetAvgPool
from StarV.util.data.nets.UNET_CamVid.train_unet_camvid_avgpool import CamVidDataset
from StarV.util.data.nets.UNET_CamVid.train_unet_camvid_avgpool import UNetAvgPool as CamV_UNetAvgPool
from StarV.util.vnnlib import *
from StarV.util.attack import *
from StarV.util.semantic_metrics import *
from StarV.verifier.certifier import *
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from scipy.ndimage import label, generate_binary_structure, binary_dilation, binary_fill_holes

import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=1000)

from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import random

artifact = 'ATVA2026_SIM_SSNN'
RESULT_dir = f'artifacts/{artifact}'
STORE_dir = f'artifacts/{artifact}/results'
DATA_dir = f'StarV/util/data/nets/CAV2021_SSNN'

CAMVID_mean=[0.485, 0.456, 0.406]
CAMVID_std=[0.229, 0.224, 0.225]

# ----- Class ID → RGB color (CamVid-ish colors) -----
CAMVID_ID2COLOR = {
    0:  (128, 128, 128),  # Sky
    1:  (128,   0,   0),  # Building
    2:  (192, 192, 128),  # Pole
    3:  (128,  64, 128),  # Road
    4:  (  0,   0, 192),  # Sidewalk
    5:  (128, 128,   0),  # Tree
    6:  (192, 128, 128),  # SignSymbol
    7:  ( 64,  64, 128),  # Fence
    8:  ( 64,   0, 128),  # Car
    9:  ( 64,  64,   0),  # Pedestrian
    10: (  0, 128, 192),  # Bicyclist
    # 255 (Void / ignore) will be black
}

def create_kvasir_dataset(root_dir, img_size=256, batch_size=8,
                       val_ratio=0.1, test_ratio=0.1, seed=42):
    full_dataset = KvasirDataset(root_dir=root_dir, img_size=img_size)
    n = len(full_dataset)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
    return train_ds, val_ds, test_ds


def camvid_unnormalize_image(image):
    """Convert a normalized CamVid tensor back to raw [0, 1] RGB values."""
    mean = torch.tensor(CAMVID_mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(CAMVID_std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return image * std + mean

def generate_unet_indices(
    n_down=4, 
    doubleconv_layers=6, 
    store_mode="before_pool",   # "before_pool" or "after_downblock"
    explicit_concat=True
):
    """
    Generate UNet skip indices for a flattened layer list.

    Assumptions:
    - Each encoder/decoder DoubleConv expands to `doubleconv_layers` layers
    (e.g., Conv, BN, ReLU, Conv, BN, ReLU => 6)
    - Each pool is 1 layer
    - Bottleneck is also a DoubleConv block
    - Each decoder stage has:
        upconv (1 layer)
        concat (1 layer, if explicit_concat=True)
        decoder DoubleConv (`doubleconv_layers`)
    - final conv is after decoder (not included in skip lists)

    Returns:
        unet_down, unet_up
    """
    m = doubleconv_layers
    L = n_down

    # Encoder skip tap points
    # Encoder layout per stage: [DoubleConv(m)] + [Pool(1)]
    pool_indices = [k * (m + 1) + m for k in range(L)]           # e.g., [6,13,20,27] for L=4, m=6
    downblock_out_indices = [p - 1 for p in pool_indices]        # e.g., [5,12,19,26]

    if store_mode == "before_pool":
        # Runtime code stores skip tensors before applying layer i.
        # To capture the encoder output right before pooling, the tap index
        # must therefore be the pool layer index itself.
        unet_down = pool_indices
    elif store_mode == "after_downblock":
        # Alternative mode for call sites that store after executing the
        # encoder block rather than before the pool layer.
        unet_down = downblock_out_indices
    else:
        raise ValueError("store_mode must be 'before_pool' or 'after_downblock'")

    # Start of decoder after encoder+pools+bottleneck
    # Encoder+pools = L*(m+1), bottleneck = m
    decoder_start = L * (m + 1) + m

    if explicit_concat:
        # Per decoder stage: upconv(1), concat(1), DoubleConv(m) => stride = m+2
        first_concat = decoder_start + 1
        stride = m + 2
        unet_up = [first_concat + j * stride for j in range(L)]
    else:
        # If concat is fused into the next decoder block layer, point to decoder block start
        # Per decoder stage without explicit concat layer: upconv(1) + decoder block(m)
        first_decoder_block = decoder_start + 1
        stride = m + 1
        unet_up = [first_decoder_block + j * stride for j in range(L)]

    return unet_down, unet_up

def create_reachable_set(image, set_type, attack_type='brightening', **kwargs):
    """
    Create reachable set for given image and attack type

    Args:
        - image: input image in numpy array format
        - set_type: type of set to create ('ImageStar' or 'SparseImageStar2DCSR' or 'SparseImageStar2DCOO')
        - attack_type: type of attack ('darkening')
        - kwargs: additional parameters for the attack
    Returns:
        reachable set
    """
    dtype = kwargs.get('dtype', np.float32)

    if attack_type == 'ubaa_darkening':
        num_max = kwargs.get('num_max', 20)
        delta = kwargs.get('delta', 0.0)
        d = kwargs.get('d', 240)
        de = kwargs.get('de', 1e-5)
        noise = UBAA_darkening_attack(image, num_max=num_max, delta=delta, d=d, dtype=dtype)

        c = image.astype(dtype)
        V = noise
        C = np.array([[1], [-1]], dtype=dtype)
        d = np.array([1, delta  -1], dtype=dtype)
        pred_lb = np.array([1 - de], dtype=dtype)
        pred_ub = np.array([1], dtype=dtype)

        if set_type == 'ImageStar':
            V = np.concatenate([c[:,:,:,None], V[:,:,:,None]], axis=3)
            return ImageStar(V, C, d, pred_lb, pred_ub)
        
        elif set_type == 'SparseImageStar2DCSR':
            shape = c.shape
            V_csr = sp.csr_array(V.reshape(-1, 1))
            C_csr = sp.csr_array(C)
            return SparseImageStar2DCSR(c.reshape(-1), V_csr, C_csr, d, pred_lb, pred_ub, shape)

        elif set_type == 'SparseImageStar2DCOO':
            shape = c.shape
            V_coo = sp.coo_array(V.reshape(-1, 1))
            C_csr = sp.csr_array(C)
            return SparseImageStar2DCOO(c.reshape(-1), V_coo, C_csr, d, pred_lb, pred_ub, shape)

        else:
            raise ValueError("Unsupported set type. Use 'ImageStar', 'SparseImageStar2DCSR', or 'SparseImageStar2DCOO'.")
        
    elif attack_type == 'brightening':
        lb, ub = brightening_attack(image, delta=kwargs.get('delta', 0.05), d=kwargs.get('d', 240), num_max=kwargs.get('num_max', 20), dtype=dtype)

        if set_type == 'ImageStar':
            return ImageStar(lb, ub)
        
        elif set_type == 'SparseImageStar2DCSR':
            return SparseImageStar2DCSR(lb, ub)
        
        elif set_type == 'SparseImageStar2DCOO':
            return SparseImageStar2DCOO(lb, ub)
        
        else:
            raise ValueError("Unsupported set type. Use 'ImageStar', 'SparseImageStar2DCSR', or 'SparseImageStar2DCOO'.")

    else:
        raise ValueError("Unsupported attack type. Use 'brightening' or 'ubaa_darkening'.")


def apply_inf_norm_attack_to_pixels(image, attack_pixels, eps):
    """Apply an L-infinity attack to all channels of the selected pixel positions."""
    lb = image.copy()
    ub = image.copy()

    if len(attack_pixels) == 0:
        return lb, ub, 0, 0

    atk_pixels = np.asarray(attack_pixels, dtype=np.intp)
    if atk_pixels.ndim != 2 or atk_pixels.shape[1] != 2:
        raise ValueError(f"attack_pixels must have shape (N, 2), got {atk_pixels.shape}")

    rows = atk_pixels[:, 0]
    cols = atk_pixels[:, 1]
    lb[rows, cols, :] = np.clip(lb[rows, cols, :] - eps, 0.0, 1.0)
    ub[rows, cols, :] = np.clip(ub[rows, cols, :] + eps, 0.0, 1.0)

    attacked_channel_values = int(np.count_nonzero(ub > lb))
    attacked_pixel_positions = int(np.count_nonzero(np.any(ub > lb, axis=-1)))
    return lb, ub, attacked_pixel_positions, attacked_channel_values
    
def compute_evaluation_results(pix_ground_truth_labels, pix_multiclass_labels, num_classes, robust_mode="unique"):
    average_wc_r_iou = compute_average_worst_case_robust_iou(
            pix_ground_truth_labels, pix_multiclass_labels, robust_mode="unique",
            num_classes=num_classes, ignore_empty=True)
    
    average_wc_r_dice = compute_average_worst_case_robust_dice_iou(
            pix_ground_truth_labels, pix_multiclass_labels, robust_mode="unique",
            num_classes=num_classes, ignore_empty=True)
    
    average_wc_r_boundary_iou = compute_average_worst_case_robust_boundary_iou(
            pix_ground_truth_labels, pix_multiclass_labels, robust_mode="unique",
            num_classes=num_classes, ignore_empty=True)
    
    average_wc_r_cldice = compute_average_worst_case_robust_cldice(
            pix_ground_truth_labels, pix_multiclass_labels, robust_mode="possible",
            num_classes=num_classes, ignore_empty=True)
    
    return average_wc_r_iou, average_wc_r_dice, average_wc_r_boundary_iou, average_wc_r_cldice

def compute_verification_aware_evaluation_results(pix_ground_truth_labels, verified_image, pix_multiclass_labels, num_classes, robust_mode="unique"):
    metrics = {
        'verification_aware_average_r_iou': compute_average_verification_aware_robust_iou(
            pix_ground_truth_labels, verified_image, pred_is_reachable=False,
            num_semantic_classes=num_classes, ignore_empty=True),
        'verification_aware_average_wc_r_iou': compute_average_verification_aware_worst_case_robust_iou(
            pix_ground_truth_labels, pix_multiclass_labels, robust_mode="unique",
            num_semantic_classes=num_classes, ignore_empty=True),
        'verification_aware_average_r_dice': compute_average_verification_aware_robust_dice_iou(
            pix_ground_truth_labels, verified_image, pred_is_reachable=False,
            num_semantic_classes=num_classes, ignore_empty=True),
        'verification_aware_average_wc_r_dice': compute_average_verification_aware_worst_case_robust_dice_iou(
            pix_ground_truth_labels, pix_multiclass_labels, robust_mode="unique",
            num_semantic_classes=num_classes, ignore_empty=True),
        'verification_aware_average_r_boundary_iou': compute_average_verification_aware_robust_boundary_iou(
            pix_ground_truth_labels, verified_image, pred_is_reachable=False,
            num_semantic_classes=num_classes, ignore_empty=True),
        'verification_aware_average_wc_r_boundary_iou': compute_average_verification_aware_worst_case_robust_boundary_iou(
            pix_ground_truth_labels, pix_multiclass_labels, robust_mode="unique",
            num_semantic_classes=num_classes, ignore_empty=True),
        'verification_aware_average_r_cldice': compute_average_verification_aware_robust_cldice(
            pix_ground_truth_labels, verified_image, pred_is_reachable=False,
            num_semantic_classes=num_classes, ignore_empty=True),
        'verification_aware_average_wc_r_cldice': compute_average_verification_aware_worst_case_robust_cldice(
            pix_ground_truth_labels, pix_multiclass_labels, robust_mode="possible",
            num_semantic_classes=num_classes, ignore_empty=True),
        'verification_aware_average_r_rlc': compute_average_verification_aware_region_level_completeness(
            pix_ground_truth_labels, verified_image, pred_is_reachable=False,
            num_semantic_classes=num_classes),
    }
    return metrics

def colorize_camvid_mask(mask_tensor):
    """
    Convert a (H,W) integer mask with values in {0..10, 255}
    into an RGB color image (H,W,3) uint8.
    """
    mask = mask_tensor.cpu().numpy()
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # default color for void/ignore
    color_mask[mask == 255] = (0, 0, 0)

    for cls_id, color in CAMVID_ID2COLOR.items():
        color_mask[mask == cls_id] = np.array(color, dtype=np.uint8)

    return color_mask

def verify_m2nist_ssnn_ubaa_darkening(dtype=np.float64):
    print('=================================================================================')
    print(f"Verifying M2NIST SSNNs")
    print('=================================================================================\n')
    net_list = ['m2nist_62iou_dilatedcnn_avgpool', 
                'm2nist_75iou_transposedcnn_avgpool', 
                'm2nist_dilated_72iou_24layer']

    # path for saving verification results
    path = STORE_dir
    if not os.path.exists(path):
        os.makedirs(path)
    save_file = f"{path}/{artifact}_m2nist_ssnn_ubaa_darkening_verification_results.pkl"

    # loading test dataset
    data_path = 'StarV/util/data/nets/CAV2021_SSNN/m2nist_6484_test_images.mat'
    data_mat_file = scipy.io.loadmat(data_path)
    image_data = data_mat_file['im_data'].astype(dtype)
    image_data = np.expand_dims(image_data, axis = 2) # shape in [h, w, c, b]

    # number of images to verify
    N = 20
    num_max_pixels_attack = [5, 10, 15, 20, 25]

    starv_nets = [load_cav2021_sssnn(net_name, dtype=dtype) for net_name in net_list]
    in_datas = [image_data[:, :, :, i] for i in range(N)]

    im_results = []
    csr_results = []
    coo_results  = []

    veriMethod='BFS'
    reachMethod='approx'
    lp_solver='gurobi'

    for n_ in range(len(net_list)):
        print('==============================================')
        print(f'Verifying SSNN: {net_list[n_]} with ImageStar...')
        starv_net = starv_nets[n_]
        num_classes = starv_net.layers[-1].classes
       
        avg_data_list = []
        for num_max in num_max_pixels_attack:
            print('==============================================')
            print(f'Verifying SSNNs with ImageStar and UBAA darkening attack with num_max = {num_max}...')
            
            # creating ImageStar input set for all images
            print('Creating ImageStar input set for all images...')
            IM_sets = []
            for i in range(N):
                IM_sets.append(create_reachable_set(image_data[:, :, :, i], 'ImageStar', 'ubaa_darkening', num_max=num_max, de=1e-5, d=150, delta=0.0, dtype=dtype))

            try:
                results = certifyRobustness_pixel(starv_net, IM_sets, in_datas, num_classes, veriMethod, reachMethod, lp_solver, return_max_memory_usage=True)
                avg_data = results['average_data']
                pix_labels = results['pixel_labels']
                pix_multiclass_labels = results['pixel_multiclass_labels']
                pix_ground_truth_labels = results['ground_truth_pixel_labels']

                average_wc_r_iou, average_wc_r_dice, average_wc_r_boundary_iou, average_wc_r_cldice = compute_evaluation_results(pix_ground_truth_labels[0], pix_multiclass_labels[0], num_classes=num_classes, robust_mode="unique")
                verification_aware_metrics = compute_verification_aware_evaluation_results(
                    pix_ground_truth_labels[0], results['verified_images'], pix_multiclass_labels[0], num_classes=num_classes
                )

            except Exception as e:
                print(f"Error during verification with ImageStar input sets: {e}")
                avg_data = ['O/M'] * 9
                average_wc_r_iou = average_wc_r_dice = np.nan
                average_wc_r_boundary_iou = average_wc_r_cldice = np.nan
                verification_aware_metrics = {
                    'verification_aware_average_wc_r_iou': np.nan,
                    'verification_aware_average_wc_r_dice': np.nan,
                    'verification_aware_average_wc_r_boundary_iou': np.nan,
                    'verification_aware_average_wc_r_cldice': np.nan,
                }

            print('==============================================')
            print('Verification results for ImageStar input sets:')
            # print average data
            print('Num of robust pixels: ', avg_data[0])
            print('Num of unknown pixels: ', avg_data[1])
            print('Num of unrobust pixels: ', avg_data[2])
            print('Num of attacked pixels: ', avg_data[3])
            print('RIoU: ', avg_data[4])
            print('RV: ', avg_data[5])
            print('RS: ', avg_data[6])
            print('Average VT: ', avg_data[7])
            print('Max Memory Usage (bytes): ', avg_data[8])
            print('')
            print(f"Average Robust Worst-Case IoU across all classes: {average_wc_r_iou:.4f}")
            print(f"Average Robust Worst-Case Dice across all classes: {average_wc_r_dice:.4f}")
            print(f"Average Robust Worst-Case Boundary IoU across all classes: {average_wc_r_boundary_iou:.4f}")
            print(f"Average Robust Worst-Case Centerline-Dice (R-WC-clDice) across all classes: {average_wc_r_cldice:.4f}")
            print('')
            print(f"Verification-Aware Robust Worst-Case IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_iou']:.4f}")
            print(f"Verification-Aware Robust Worst-Case Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_dice']:.4f}")
            print(f"Verification-Aware Robust Worst-Case Boundary IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']:.4f}")
            print(f"Verification-Aware Robust Worst-Case Centerline-Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_cldice']:.4f}")
            print('==============================================')
            results['average_wc_r_iou'] = average_wc_r_iou
            results['average_wc_r_dice'] = average_wc_r_dice
            results['average_wc_r_boundary_iou'] = average_wc_r_boundary_iou
            results['average_wc_r_cldice'] = average_wc_r_cldice
            results['verification_aware_average_wc_r_iou'] = verification_aware_metrics['verification_aware_average_wc_r_iou']
            results['verification_aware_average_wc_r_dice'] = verification_aware_metrics['verification_aware_average_wc_r_dice']
            results['verification_aware_average_wc_r_boundary_iou'] = verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']
            results['verification_aware_average_wc_r_cldice'] = verification_aware_metrics['verification_aware_average_wc_r_cldice']
            avg_data_list.append(results.copy())

        im_results.append(avg_data_list.copy())
    del IM_sets

    results = [im_results, csr_results, coo_results]
    pickle.dump(results, open(save_file, 'wb'))

    for n_ in range(len(net_list)):
        print('==============================================')
        print(f'Verifying SSNN: {net_list[n_]} with SparseImageStar2DCSR...')
        starv_net = starv_nets[n_]
        num_classes = starv_net.layers[-1].classes
        
        avg_data_list = []
        for num_max in num_max_pixels_attack:
            print('==============================================')
            print(f'Verifying SSNNs with SparseImageStar2DCSR and UBAA darkening attack with num_max = {num_max}...')

            # creating SparseImageStar2DCSR input set for all images
            print('Creating SparseImageStar2DCSR input set for all images...')
            CSR_sets = []
            for i in range(N):
                CSR_sets.append(create_reachable_set(image_data[:, :, :, i], 'SparseImageStar2DCSR', 'ubaa_darkening', num_max=num_max, de=1e-5, d=150, delta=0.0, dtype=dtype))

            try:
                results = certifyRobustness_pixel(starv_net, CSR_sets, in_datas, num_classes, veriMethod, reachMethod, lp_solver, return_max_memory_usage=True)
                avg_data = results['average_data']
                pix_labels = results['pixel_labels']
                pix_multiclass_labels = results['pixel_multiclass_labels']
                pix_ground_truth_labels = results['ground_truth_pixel_labels']

                average_wc_r_iou, average_wc_r_dice, average_wc_r_boundary_iou, average_wc_r_cldice = compute_evaluation_results(pix_ground_truth_labels[0], pix_multiclass_labels[0], num_classes=num_classes, robust_mode="unique")
                verification_aware_metrics = compute_verification_aware_evaluation_results(
                    pix_ground_truth_labels[0], results['verified_images'], pix_multiclass_labels[0], num_classes=num_classes
                )

            except Exception as e:
                print(f"Error during verification with SparseImageStar2DCSR input sets: {e}")
                avg_data = ['O/M'] * 9
                average_wc_r_iou = average_wc_r_dice = np.nan
                average_wc_r_boundary_iou = average_wc_r_cldice = np.nan
                verification_aware_metrics = {
                    'verification_aware_average_wc_r_iou': np.nan,
                    'verification_aware_average_wc_r_dice': np.nan,
                    'verification_aware_average_wc_r_boundary_iou': np.nan,
                    'verification_aware_average_wc_r_cldice': np.nan,
                }

            print('==============================================')
            print('Verification results for SparseImageStar2DCSR"')
            # print average data
            print('Num of robust pixels: ', avg_data[0])
            print('Num of unknown pixels: ', avg_data[1])
            print('Num of unrobust pixels: ', avg_data[2])
            print('Num of attacked pixels: ', avg_data[3])
            print('RIoU: ', avg_data[4])
            print('RV: ', avg_data[5])
            print('RS: ', avg_data[6])
            print('Average VT: ', avg_data[7])
            print('Max Memory Usage (bytes): ', avg_data[8])
            print('')
            print(f"Average Robust Worst-Case IoU across all classes: {average_wc_r_iou:.4f}")
            print(f"Average Robust Worst-Case Dice across all classes: {average_wc_r_dice:.4f}")
            print(f"Average Robust Worst-Case Boundary IoU across all classes: {average_wc_r_boundary_iou:.4f}")
            print(f"Average Robust Worst-Case Centerline-Dice (R-WC-clDice) across all classes: {average_wc_r_cldice:.4f}")
            print('')
            print(f"Verification-Aware Robust Worst-Case IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_iou']:.4f}")
            print(f"Verification-Aware Robust Worst-Case Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_dice']:.4f}")
            print(f"Verification-Aware Robust Worst-Case Boundary IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']:.4f}")
            print(f"Verification-Aware Robust Worst-Case Centerline-Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_cldice']:.4f}")
            print('==============================================')
            results['average_wc_r_iou'] = average_wc_r_iou
            results['average_wc_r_dice'] = average_wc_r_dice
            results['average_wc_r_boundary_iou'] = average_wc_r_boundary_iou
            results['average_wc_r_cldice'] = average_wc_r_cldice
            results['verification_aware_average_wc_r_iou'] = verification_aware_metrics['verification_aware_average_wc_r_iou']
            results['verification_aware_average_wc_r_dice'] = verification_aware_metrics['verification_aware_average_wc_r_dice']
            results['verification_aware_average_wc_r_boundary_iou'] = verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']
            results['verification_aware_average_wc_r_cldice'] = verification_aware_metrics['verification_aware_average_wc_r_cldice']
            avg_data_list.append(results.copy())

        csr_results.append(avg_data_list.copy())
    del CSR_sets

    results = [im_results, csr_results, coo_results]
    pickle.dump(results, open(save_file, 'wb'))

    for n_ in range(len(net_list)):
        print('==============================================')
        print(f'Verifying SSNN: {net_list[n_]} with SparseImageStar2DCOO...')
        starv_net = starv_nets[n_]
        num_classes = starv_net.layers[-1].classes

        avg_data_list = []
        for num_max in num_max_pixels_attack:
            print('==============================================')
            print(f'Verifying SSNNs with SparseImageStar2DCOO and UBAA darkening attack with num_max = {num_max}...')

            # creating SparseImageStar2DCOO input set for all images
            print('Creating SparseImageStar2DCOO input set for all images...')
            COO_sets = []
            for i in range(N):
                COO_sets.append(create_reachable_set(image_data[:, :, :, i], 'SparseImageStar2DCOO', 'ubaa_darkening', num_max=num_max, de=1e-5, d=150, delta=0.0, dtype=dtype))

            try:
                results = certifyRobustness_pixel(starv_net, COO_sets, in_datas, num_classes, veriMethod, reachMethod, lp_solver, return_max_memory_usage=True)
                avg_data = results['average_data']
                pix_labels = results['pixel_labels']
                pix_multiclass_labels = results['pixel_multiclass_labels']
                pix_ground_truth_labels = results['ground_truth_pixel_labels']

                average_wc_r_iou, average_wc_r_dice, average_wc_r_boundary_iou, average_wc_r_cldice = compute_evaluation_results(pix_ground_truth_labels[0], pix_multiclass_labels[0], num_classes=num_classes, robust_mode="unique")
                verification_aware_metrics = compute_verification_aware_evaluation_results(
                    pix_ground_truth_labels[0], results['verified_images'], pix_multiclass_labels[0], num_classes=num_classes
                )

            except Exception as e:
                print(f"Error during verification with SparseImageStar2DCOO input sets: {e}")
                avg_data = ['O/M'] * 9
                average_wc_r_iou = average_wc_r_dice = np.nan
                average_wc_r_boundary_iou = average_wc_r_cldice = np.nan
                verification_aware_metrics = {
                    'verification_aware_average_wc_r_iou': np.nan,
                    'verification_aware_average_wc_r_dice': np.nan,
                    'verification_aware_average_wc_r_boundary_iou': np.nan,
                    'verification_aware_average_wc_r_cldice': np.nan,
                }

            print('==============================================')
            print('Verification results for SparseImageStar2DCSR"')
            # print average data
            print('Num of robust pixels: ', avg_data[0])
            print('Num of unknown pixels: ', avg_data[1])
            print('Num of unrobust pixels: ', avg_data[2])
            print('Num of attacked pixels: ', avg_data[3])
            print('RIoU: ', avg_data[4])
            print('RV: ', avg_data[5])
            print('RS: ', avg_data[6])
            print('Average VT: ', avg_data[7])
            print('Max Memory Usage (bytes): ', avg_data[8])
            print('')
            print(f"Average Robust Worst-Case IoU across all classes: {average_wc_r_iou:.4f}")
            print(f"Average Robust Worst-Case Dice across all classes: {average_wc_r_dice:.4f}")
            print(f"Average Robust Worst-Case Boundary IoU across all classes: {average_wc_r_boundary_iou:.4f}")
            print(f"Average Robust Worst-Case Centerline-Dice (R-WC-clDice) across all classes: {average_wc_r_cldice:.4f}")
            print('')
            print(f"Verification-Aware Robust Worst-Case IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_iou']:.4f}")
            print(f"Verification-Aware Robust Worst-Case Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_dice']:.4f}")
            print(f"Verification-Aware Robust Worst-Case Boundary IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']:.4f}")
            print(f"Verification-Aware Robust Worst-Case Centerline-Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_cldice']:.4f}")
            print('==============================================')
            results['average_wc_r_iou'] = average_wc_r_iou
            results['average_wc_r_dice'] = average_wc_r_dice
            results['average_wc_r_boundary_iou'] = average_wc_r_boundary_iou
            results['average_wc_r_cldice'] = average_wc_r_cldice
            results['verification_aware_average_wc_r_iou'] = verification_aware_metrics['verification_aware_average_wc_r_iou']
            results['verification_aware_average_wc_r_dice'] = verification_aware_metrics['verification_aware_average_wc_r_dice']
            results['verification_aware_average_wc_r_boundary_iou'] = verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']
            results['verification_aware_average_wc_r_cldice'] = verification_aware_metrics['verification_aware_average_wc_r_cldice']
            avg_data_list.append(results.copy())
        
        coo_results.append(avg_data_list.copy())
    del COO_sets

    results = [im_results, csr_results, coo_results]
    pickle.dump(results, open(save_file, 'wb'))
    
    
def plot_mnist_ssnn_ubaa_darkening_results_figure(ax, num_nets, nnv_results, im_results, csr_results, coo_results, result_type, path, y_label):
    num_max_pixels_attack = [5, 10, 15, 20, 25]

    keys_list = ['numRbPixels', 'numUnkPixels', 'numMisPixels', 'numAttPixels', 'RIoU', 'RV', 'RS', 'VT', 'MaxMemoryUsage']
    # Create the index mapping keys to their zero-based index
    key_index = {key: i for i, key in enumerate(keys_list)} 
    def get_average_data(results, net_idx, pixel_idx):
        entry = results[net_idx][pixel_idx]
        if isinstance(entry, dict):
            return entry.get("average_data", [np.nan] * len(keys_list))
        return entry

    
    im_dataset = np.zeros([num_nets, len(num_max_pixels_attack)])
    csr_dataset = np.zeros([num_nets, len(num_max_pixels_attack)])
    coo_dataset = np.zeros([num_nets, len(num_max_pixels_attack)])
    for j in range(num_nets):
        for i in range(len(num_max_pixels_attack)):
            csr_dataset[j, i] = get_average_data(csr_results, j, i)[key_index[result_type]]
            coo_dataset[j, i] = get_average_data(coo_results, j, i)[key_index[result_type]]
            im_dataset[j, i] = get_average_data(im_results, j, i)[key_index[result_type]]

    # for net_idx in range(num_nets):
    if result_type == 'MaxMemoryUsage':
        im_dataset  = im_dataset  / (1024 ** 2)  # Convert to MB
        csr_dataset = csr_dataset / (1024 ** 2)  # Convert to MB
        coo_dataset = coo_dataset / (1024 ** 2)  # Convert to MB
        print('im_dataset: ', im_dataset)

    ax.plot(num_max_pixels_attack, nnv_results[0, :], marker='o', color='blue', linestyle='-', linewidth=3, label='N1 NNV')
    ax.plot(num_max_pixels_attack, nnv_results[1, :], marker='^', color='blue', linestyle='-', linewidth=3, label='N2 NNV')
    ax.plot(num_max_pixels_attack, nnv_results[2, :], marker='+', color='blue', linestyle='-', linewidth=3, label='N3 NNV')

    ax.plot(num_max_pixels_attack, im_dataset[0, :], marker='o', color='red', linestyle='--', linewidth=3, label='N1 IM')
    ax.plot(num_max_pixels_attack, im_dataset[1, :], marker='^', color='red', linestyle='--', linewidth=3, label='N2 IM')
    ax.plot(num_max_pixels_attack, im_dataset[2, :], marker='+', color='red', linestyle='--', linewidth=3, label='N3 IM')
    ax.plot(num_max_pixels_attack, csr_dataset[0, :], marker='o', color='magenta', linestyle='-.', linewidth=3, label='N1 SIM CSR')
    ax.plot(num_max_pixels_attack, csr_dataset[1, :], marker='^', color='magenta', linestyle='-.', linewidth=3, label='N2 SIM CSR')
    ax.plot(num_max_pixels_attack, csr_dataset[2, :], marker='+', color='magenta', linestyle='-.', linewidth=3, label='N3 SIM CSR')
    ax.plot(num_max_pixels_attack, coo_dataset[0, :], marker='o', color='green', linestyle=':', linewidth=3, label='N1 SIM COO')
    ax.plot(num_max_pixels_attack, coo_dataset[1, :], marker='^', color='green', linestyle=':', linewidth=3, label='N2 SIM COO')
    ax.plot(num_max_pixels_attack, coo_dataset[2, :], marker='+', color='green', linestyle=':', linewidth=3, label='N3 SIM COO')

    font_size = 16
    tick_size = 15
    # plt.title(f'Average {result_type} vs. Number of Attacked Pixels for m2nist SSNNs')
    ax.set_xlabel('Number of Atk. Pixels', fontsize=font_size) #'Number of Attacked Pixels'
    if result_type == 'MaxMemoryUsage':
        ax.set_ylabel(f'{y_label}', fontsize=font_size)
    else:
        ax.set_ylabel(f'Avg. {y_label}', fontsize=font_size)
    ax.set_xticks(num_max_pixels_attack)
    ax.ticklabel_format(axis='y', style='plain')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    # ax.legend(fontsize=tick_size)
    ax.grid()




def plot_mnist_ssnn_ubaa_darkening_results(num_images=20):
    # load verification results
    path = STORE_dir
    save_file_full = f"{path}/{artifact}_m2nist_ssnn_ubaa_darkening_verification_results_comparison_figure_full.png"
    save_file_short = f"{path}/{artifact}_m2nist_ssnn_ubaa_darkening_verification_results_comparison_figure_short.png"
    save_file_semantic = f"{path}/{artifact}_m2nist_ssnn_ubaa_darkening_verification_results_comparison_figure_semantic.png"

    # load nnv verification results
    result_path = f'{DATA_dir}/nnv_compare_m2nist_nets_vs_num_attackedpixels_results.mat'
    result_mat_file = scipy.io.loadmat(result_path)
    nnv_avg_riou = result_mat_file['avg_RIoU'] # shape in [num_nets, num_max_pixels_attack]
    nnv_avg_rv = result_mat_file['avg_RV'] # shape in [num_nets, num_max_pixels_attack]
    nnv_avg_rs = result_mat_file['avg_RS'] # shape in [num_nets, num_max_pixels_attack]
    nnv_avg_vt = result_mat_file['VT']/float(num_images) # shape in [num_nets, num_max_pixels_attack]
    nnv_avg_numRbPixels  = result_mat_file['avg_numRbPixels'] # shape in [num_nets, num_max_pixels_attack]
    nnv_avg_numMisPixels = result_mat_file['avg_numMisPixels'] # shape in [num_nets, num_max_pixels_attack]
    nnv_avg_numAttPixels = result_mat_file['avg_numAttPixels'] # shape in [num_nets, num_max_pixels_attack]
    nnv_avg_numUnkPixels = result_mat_file['avg_numUnkPixels'] # shape in [num_nets, num_max_pixels_attack]
    nnv_max_memory_usage = nnv_avg_numAttPixels * np.nan

    load_file = f"{path}/{artifact}_m2nist_ssnn_ubaa_darkening_verification_results.pkl"
    im_results, csr_results, coo_results = pickle.load(open(load_file, 'rb'))
    num_nets = len(im_results)
    
    y_labels = ['Number of Robust Pixels', 'Number of Unknown Pixels', 'Number of Unrobust Pixels',
                'Number of Attacked Pixels', 'Robust IoU', 'Robustness Value', 'Robustness Sensitivity', 'Verification Time (sec)', 'Max Memory Usage (MB)']
    font_size = 16
    tick_size = 15
    legend_size = 12
    fig, axes = plt.subplots(2, 4, figsize=(13, 8), constrained_layout=True)
    plot_mnist_ssnn_ubaa_darkening_results_figure(axes[0, 0], num_nets, nnv_avg_numRbPixels, im_results, csr_results, coo_results, 'numRbPixels', path, y_labels[0])
    plot_mnist_ssnn_ubaa_darkening_results_figure(axes[1, 0], num_nets, nnv_avg_numUnkPixels, im_results, csr_results, coo_results, 'numUnkPixels', path, y_labels[1])
    plot_mnist_ssnn_ubaa_darkening_results_figure(axes[0, 1], num_nets, nnv_avg_numMisPixels, im_results, csr_results, coo_results, 'numMisPixels', path, y_labels[2])
    plot_mnist_ssnn_ubaa_darkening_results_figure(axes[1, 1], num_nets, nnv_avg_riou, im_results, csr_results, coo_results, 'RIoU', path, y_labels[4])
    plot_mnist_ssnn_ubaa_darkening_results_figure(axes[0, 2], num_nets, nnv_avg_rv, im_results, csr_results, coo_results, 'RV', path, y_labels[5])
    plot_mnist_ssnn_ubaa_darkening_results_figure(axes[1, 2], num_nets, nnv_avg_rs, im_results, csr_results, coo_results, 'RS', path, y_labels[6])
    plot_mnist_ssnn_ubaa_darkening_results_figure(axes[0, 3], num_nets, nnv_avg_vt, im_results, csr_results, coo_results, 'VT', path, y_labels[7])
    plot_mnist_ssnn_ubaa_darkening_results_figure(axes[1, 3], num_nets, nnv_max_memory_usage, im_results, csr_results, coo_results, 'MaxMemoryUsage', path, y_labels[8])
    axes[0, 3].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_size)
    plt.savefig(save_file_full, bbox_inches='tight', pad_inches=0.5)
    plt.figure()

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    # plot_mnist_ssnn_ubaa_darkening_results_figure(axes[0], num_nets, nnv_avg_numRbPixels, im_results, csr_results, coo_results, 'numRbPixels', path, y_labels[0])
    plot_mnist_ssnn_ubaa_darkening_results_figure(axes[0], num_nets, nnv_avg_rv, im_results, csr_results, coo_results, 'RV', path, y_labels[5])
    plot_mnist_ssnn_ubaa_darkening_results_figure(axes[1], num_nets, nnv_avg_rs, im_results, csr_results, coo_results, 'RS', path, y_labels[6])
    plot_mnist_ssnn_ubaa_darkening_results_figure(axes[2], num_nets, nnv_avg_vt, im_results, csr_results, coo_results, 'VT', path, y_labels[7])
    plot_mnist_ssnn_ubaa_darkening_results_figure(axes[3], num_nets, nnv_max_memory_usage, im_results, csr_results, coo_results, 'MaxMemoryUsage', path, y_labels[8])
    axes[3].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_size) #(1, 0.5)
    plt.savefig(save_file_short, bbox_inches='tight', pad_inches=0.5)
    plt.figure()

    plot_sematic_metrics(im_results, csr_results, coo_results, 'all', save_file_semantic)


def plot_sematic_metrics(im, csr, coo, metric_name, save_file=None, method_styles=None, legend_mode='outside'):
    metric_specs = {
        'verification_aware_average_wc_r_iou': 'VIoU',
        'verification_aware_average_wc_r_dice': 'VDice',
        'verification_aware_average_wc_r_boundary_iou': 'VBIoU',
        'verification_aware_average_wc_r_cldice': 'VclDice',
    }

    def to_float(value):
        try:
            if isinstance(value, str) and value.strip().lower() in {'o/m', 'nan', 'none', ''}:
                return np.nan
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def pad_series_rows(series_list):
        rows = []
        max_len = 0
        for series in series_list:
            if series is None:
                continue
            arr = np.asarray(series, dtype=np.float64)
            if arr.size == 0:
                continue
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.ndim == 1:
                rows.append(arr)
                max_len = max(max_len, arr.shape[0])
                continue
            if arr.ndim == 2:
                for row in arr:
                    row = np.asarray(row, dtype=np.float64).reshape(-1)
                    rows.append(row)
                    max_len = max(max_len, row.shape[0])
                continue
            arr = arr.reshape(arr.shape[0], -1)
            for row in arr:
                row = np.asarray(row, dtype=np.float64).reshape(-1)
                rows.append(row)
                max_len = max(max_len, row.shape[0])

        if len(rows) == 0:
            return np.array([], dtype=np.float64)

        padded = np.full((len(rows), max_len), np.nan, dtype=np.float64)
        for i, row in enumerate(rows):
            padded[i, :len(row)] = row
        return padded

    def extract_series(results, key):
        if results is None:
            return None
        if isinstance(results, (list, tuple)):
            if len(results) == 0:
                return np.array([], dtype=np.float64)
            first = results[0]
            if isinstance(first, (list, tuple)):
                nested_series = [extract_series(entry, key) for entry in results]
                return pad_series_rows(nested_series)
            if isinstance(first, dict):
                return np.array([to_float(entry.get(key, np.nan)) for entry in results], dtype=np.float64)
        if isinstance(results, dict):
            return np.array([to_float(results.get(key, np.nan))], dtype=np.float64)
        return None

    def extract_x_values(*datasets):
        candidates = []

        def collect(dataset):
            if isinstance(dataset, dict):
                avg_data = dataset.get('average_data', [])
                if len(avg_data) > 3:
                    candidates.append(np.array([to_float(avg_data[3])], dtype=np.float64))
                return
            if not isinstance(dataset, (list, tuple)) or len(dataset) == 0:
                return
            first = dataset[0]
            if isinstance(first, dict):
                x_vals = []
                for entry in dataset:
                    avg_data = entry.get('average_data', [])
                    if len(avg_data) > 3:
                        x_vals.append(to_float(avg_data[3]))
                if x_vals:
                    candidates.append(np.array(x_vals, dtype=np.float64))
                return
            for entry in dataset:
                collect(entry)

        for dataset in datasets:
            collect(dataset)

        if len(candidates) == 0:
            return None
        return max(candidates, key=len)

    if metric_name in metric_specs:
        metrics_to_plot = [metric_name]
    else:
        metrics_to_plot = list(metric_specs.keys())

    semantic_figsize = (14, 4) if len(metrics_to_plot) == 4 else (3.5 * len(metrics_to_plot), 4)
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=semantic_figsize, constrained_layout=True)
    if len(metrics_to_plot) == 1:
        axes = [axes]

    x_values = extract_x_values(im, csr, coo)
    default_method_styles = {
        'IM': ('red', '--'),
        'SIM CSR': ('magenta', '-.'),
        'SIM COO': ('green', ':'),
    }
    if method_styles is not None:
        default_method_styles.update(method_styles)

    method_specs = []
    if im is not None:
        color, linestyle = default_method_styles['IM']
        method_specs.append(('IM', im, color, linestyle))
    if csr is not None:
        color, linestyle = default_method_styles['SIM CSR']
        method_specs.append(('SIM CSR', csr, color, linestyle))
    if coo is not None:
        color, linestyle = default_method_styles['SIM COO']
        method_specs.append(('SIM COO', coo, color, linestyle))
    markers = ['o', '^', '+', 's', 'd']
    any_plotted = False

    font_size = 16
    tick_size = 15
    for ax, key in zip(axes, metrics_to_plot):
        for label, results, color, linestyle in method_specs:
            series = extract_series(results, key)
            if series is None or np.size(series) == 0:
                continue
            series = np.asarray(series, dtype=np.float64)
            if series.ndim == 1:
                series = series.reshape(1, -1)
            current_x = x_values
            if current_x is None or len(current_x) != series.shape[1]:
                current_x = np.arange(1, series.shape[1] + 1, dtype=np.float64)

            for row in range(series.shape[0]):
                if np.all(np.isnan(series[row])):
                    continue
                line_label = label if series.shape[0] == 1 else f'N{row + 1} {label}'
                ax.plot(current_x, series[row], marker=markers[row % len(markers)], color=color, linestyle=linestyle, linewidth=3, label=line_label)
                any_plotted = True

        # ax.set_title(metric_specs[key], fontsize=font_size)
        ax.set_xlabel('Number of Attacked Pixels', fontsize=font_size)
        ax.set_ylabel(metric_specs[key], fontsize=font_size)
        if x_values is not None and len(x_values) > 0:
            ax.set_xticks(x_values)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.tick_params(axis='x', labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        ax.grid()

    if any_plotted:
        if legend_mode == 'inside':
            axes[-1].legend(fontsize=11)
        else:
            axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)

    if not os.path.exists(STORE_dir):
        os.makedirs(STORE_dir)

    if save_file is None:
        save_tag = metric_name if metric_name in metric_specs else 'verification_aware_semantic_metrics'
        save_file = f"{STORE_dir}/{artifact}_{save_tag}.png"
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)


def memory_usage_ssnn(dtype = np.float64):
    print('=================================================================================')
    print(f"Memory Usage of M2NIST SSNNs")
    print('=================================================================================\n')
    net_name = 'm2nist_75iou_transposedcnn_avgpool'

    dtype = np.float64

    # loading test dataset
    data_path = 'StarV/util/data/nets/CAV2021_SSNN/m2nist_6484_test_images.mat'
    data_mat_file = scipy.io.loadmat(data_path)
    image_data = data_mat_file['im_data'].astype(dtype)
    image_data = np.expand_dims(image_data, axis = 2) # shape in [h, w, c, b]

    starv_net = load_cav2021_sssnn(net_name, dtype=dtype)
    image = image_data[:, :, :, 0]

    d = 245
    delta = 0.01
    num_max = 5
    IM  = create_reachable_set(image_data[:, :, :, 0], 'ImageStar', 'brightening', num_max=num_max, d=d, delta=delta, dtype=dtype)
    CSR = create_reachable_set(image_data[:, :, :, 0], 'SparseImageStar2DCSR', 'brightening', num_max=num_max, d=d, delta=delta, dtype=dtype)
    COO = create_reachable_set(image_data[:, :, :, 0], 'SparseImageStar2DCOO', 'brightening', num_max=num_max, d=d, delta=delta, dtype=dtype)

    IM_time = []; COO_time = []; CSR_time = [];
    IM_nb = [IM.nbytes()]; COO_nb = [COO.nbytes()]; CSR_nb = [CSR.nbytes()]
    density = [CSR.density()]

    print('working on SparseImageStar COO')
    for i in range(starv_net.n_layers):
        start = time.perf_counter()
        COO = starv_net.layers[i].reach(COO, method='approx', show=False)
        if starv_net.layers[i].__class__.__name__ == 'PixelClassificationLayer':
            continue
        COO_time.append(time.perf_counter() - start)
        COO_nb.append(COO.nbytes())

    print('working on ImageStar')
    for i in range(starv_net.n_layers):
        start = time.perf_counter()
        # print(f'Layer {i}/{starv_net.n_layers}: {starv_net.layers[i].__class__.__name__}')
        IM = starv_net.layers[i].reach(IM, method='approx', show=False)
        if starv_net.layers[i].__class__.__name__ == 'PixelClassificationLayer': 
            continue
        IM_time.append(time.perf_counter() - start)
        IM_nb.append(IM.nbytes())

    print('working on SparseImageStar CSR')
    for i in range(starv_net.n_layers):
        start = time.perf_counter()
        CSR = starv_net.layers[i].reach(CSR, method='approx', show=False)
        if starv_net.layers[i].__class__.__name__ == 'PixelClassificationLayer':
            continue
        CSR_time.append(time.perf_counter() - start)
        CSR_nb.append(CSR.nbytes())
        density.append(CSR.density())


    # path for saving verification results
    path = STORE_dir
    if not os.path.exists(path):
        os.makedirs(path)
    save_file = f"{path}/{artifact}_m2nist_ssnn_memory_usage2_nmax=5.pkl"
    pickle.dump([IM_time, IM_nb, CSR_time, CSR_nb, COO_time, COO_nb, density], open(save_file, 'wb'))

    x = np.arange(len(IM_time))
    x_ticks_labels = []
    for i in range(starv_net.n_layers):
        if starv_net.layers[i].__class__.__name__ == 'Conv2DLayer':
            l_name = '$L_c$'
        elif starv_net.layers[i].__class__.__name__ == 'ReLULayer':
            l_name = '$L_r$'
        elif starv_net.layers[i].__class__.__name__ == 'FlattenLayer':
            l_name = '$L_{{flat}}$'
        elif starv_net.layers[i].__class__.__name__ == 'FullyConnectedLayer':
            l_name = '$L_f$'
        elif starv_net.layers[i].__class__.__name__ == 'AvgPool2DLayer':
            l_name = '$L_a$'
        elif starv_net.layers[i].__class__.__name__ == 'PixelClassificationLayer':
            # l_name = '$L_{pix}$'
            continue
        elif starv_net.layers[i].__class__.__name__ == 'TransposedConv2DLayer' or starv_net.layers[i].__class__.__name__ == 'ConvTranspose2DLayer':
            l_name = '$L_t$'
        elif starv_net.layers[i].__class__.__name__ == 'BatchNorm2DLayer':
            l_name = '$L_b$'
        elif starv_net.layers[i].__class__.__name__ == 'PadLayer':
            l_name = '$L_{pad}$'
        else:
            l_name = f"${starv_net.layers[i].__class__.__name__}$"
        x_ticks_labels.append(f"{l_name}_{i}")

    font_size = 25
    y_tick_size = 23
    tick_size = 16
    line_width = 5

    plt.rcParams["figure.figsize"] = [22, 7]
    plt.rcParams["figure.autolayout"] = True

    CSR_time = np.array(CSR_time) / 60.0  # convert to minutes
    COO_time = np.array(COO_time) / 60.0  # convert to minutes
    IM_time = np.array(IM_time) / 60.0  # convert to minutes

    fig, ax = plt.subplots(1, 2, layout="constrained")
    ax[1].set_title("Computation Time", fontsize=font_size)
    ax[1].plot(x, IM_time, color="red", linewidth=line_width)
    ax[1].plot(x, COO_time, color='black', linewidth=line_width, linestyle='-.')
    ax[1].plot(x, CSR_time, color="magenta", linewidth=line_width, linestyle=':')
    ax[1].set_xlabel("Layers", fontsize=font_size)
    ax[1].set_ylabel("Computation Time (min)", fontsize=font_size)
    # Set number of ticks for x-axis
    ax[1].set_xticks(x)
    # Set ticks labels for x-axis
    ax[1].set_xticklabels(x_ticks_labels, rotation=80, fontsize=tick_size)
    ax[1].yaxis.set_tick_params(labelsize=y_tick_size)
    # set legend
    ax[1].legend(['ImageStar', 'SIM COO', 'SIM CSR'], fontsize=font_size)
    ax[1].grid()

    IM_nb =  np.array(IM_nb)  / float(1024**3)  # convert to GB
    COO_nb = np.array(COO_nb) / float(1024**3)  # convert to GB
    CSR_nb = np.array(CSR_nb) / float(1024**3)  # convert to GB

    x = np.arange(len(IM_nb))
    x_ticks_labels.insert(0, 'Input')

    ax[0].set_title("Memory Usage", fontsize=font_size)
    ax[0].plot(x, IM_nb, color="red", linewidth=line_width)
    ax[0].plot(x, COO_nb, color='black', linewidth=line_width, linestyle='-.')
    ax[0].plot(x, CSR_nb, color="magenta", linewidth=line_width, linestyle=':')
    ax[0].set_xlabel("Layers", fontsize=font_size)
    ax[0].set_ylabel("Memory Usage (GB)", fontsize=font_size)

    # Set number of ticks for x-axis
    ax[0].set_xticks(x)
    # Set ticks labels for x-axis
    ax[0].set_xticklabels(x_ticks_labels, rotation=80, fontsize=tick_size)
    ax[0].yaxis.set_tick_params(labelsize=y_tick_size)
    # set legend
    ax[0].legend(['ImageStar', 'SIM COO', 'SIM CSR'], loc='center left', fontsize=font_size)
    ax[0].grid()
    ax2 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, density, color="green", linewidth=2)
    ax2.legend(['density'], fontsize=font_size, loc='upper left')
    ax2.set_ylabel("Density", fontsize=font_size)
    ax2.yaxis.set_tick_params(labelsize=y_tick_size)
    plt.tight_layout()
    plt.margins(x=0)
    plt.savefig(f'{path}/{artifact}_m2nist_ssnn_memory_usage_brightening_Nmax5.png', bbox_inches='tight')
    plt.close()

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def verify_m2nist_ssnn_brightening(dtype=np.float64):
    print('=================================================================================')
    print(f"Verifying M2NIST SSNNs with Brightening Attack")
    print('=================================================================================\n')
    net_name = 'm2nist_75iou_transposedcnn_avgpool'

    # path for saving verification results
    path = STORE_dir
    if not os.path.exists(path):
        os.makedirs(path)
    save_file = f"{path}/{artifact}_m2nist_ssnn_bounded_attack_brightening_verification_results.pkl"

    # loading test dataset
    data_path = 'StarV/util/data/nets/CAV2021_SSNN/m2nist_6484_test_images.mat'
    data_mat_file = scipy.io.loadmat(data_path)
    image_data = data_mat_file['im_data'].astype(dtype)
    image_data = np.expand_dims(image_data, axis = 2) # shape in [h, w, c, b]

    # number of images to verify
    N = 1
    i = 0  # index of the image to verify

    num_max_pixels_attack = [5, 10, 15, 20]

    starv_net = load_cav2021_sssnn(net_name, dtype=dtype)
    in_datas = [image_data[:, :, :, i] for i in range(N)]

    csr_results = []
    coo_results  = []

    veriMethod='BFS'
    reachMethod='approx'
    lp_solver='gurobi'

    d = 245
    delta = 0.01

    print('==============================================')
    print(f'Verifying SSNN: {net_name} with SparseImageStar2DCSR...')
    num_classes = starv_net.layers[-1].classes
    
    for num_max in num_max_pixels_attack:
        print('==============================================')
        print(f'Verifying SSNNs with SparseImageStar2DCSR and bounded brightening attack with num_max = {num_max}...')

        # creating SparseImageStar2DCSR input set for all images
        print('Creating SparseImageStar2DCSR input set for all images...')
        CSR_set = create_reachable_set(image_data[:, :, :, i], 'SparseImageStar2DCSR', 'brightening', num_max=num_max, d=d, delta=delta, dtype=dtype)
        results = certifyRobustness_pixel(starv_net, [CSR_set], in_datas, num_classes, veriMethod, reachMethod, lp_solver, return_max_memory_usage=True)
        avg_data = results['average_data']
        pix_labels = results['pixel_labels']
        pix_multiclass_labels = results['pixel_multiclass_labels']
        pix_ground_truth_labels = results['ground_truth_pixel_labels']
        
        average_wc_r_iou, average_wc_r_dice, average_wc_r_boundary_iou, average_wc_r_cldice = compute_evaluation_results(pix_ground_truth_labels[0], pix_multiclass_labels[0], num_classes=num_classes, robust_mode="unique")
        verification_aware_metrics = compute_verification_aware_evaluation_results(
            pix_ground_truth_labels[0], results['verified_images'], pix_multiclass_labels[0], num_classes=num_classes
        )
        
        print('==============================================')
        print('Verification results for SparseImageStar2DCSR"')
        # print average data
        print('Num of robust pixels: ', avg_data[0])
        print('Num of unknown pixels: ', avg_data[1])
        print('Num of unrobust pixels: ', avg_data[2])
        print('Num of attacked pixels: ', avg_data[3])
        print('RIoU: ', avg_data[4])
        print('RV: ', avg_data[5])
        print('RS: ', avg_data[6])
        print('Average VT: ', avg_data[7])
        print('Max Memory Usage (bytes): ', avg_data[8])
        print('')
        print(f"Average Robust Worst-Case IoU across all classes: {average_wc_r_iou:.4f}")
        print(f"Average Robust Worst-Case Dice across all classes: {average_wc_r_dice:.4f}")
        print(f"Average Robust Worst-Case Boundary IoU across all classes: {average_wc_r_boundary_iou:.4f}")
        print(f"Average Robust Worst-Case Centerline-Dice (R-WC-clDice) across all classes: {average_wc_r_cldice:.4f}")
        print('')
        print(f"Verification-Aware Robust Worst-Case IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_iou']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_dice']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Boundary IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Centerline-Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_cldice']:.4f}")
        print('==============================================')
        results['average_wc_r_iou'] = average_wc_r_iou
        results['average_wc_r_dice'] = average_wc_r_dice
        results['average_wc_r_boundary_iou'] = average_wc_r_boundary_iou
        results['average_wc_r_cldice'] = average_wc_r_cldice
        results['verification_aware_average_wc_r_iou'] = verification_aware_metrics['verification_aware_average_wc_r_iou']
        results['verification_aware_average_wc_r_dice'] = verification_aware_metrics['verification_aware_average_wc_r_dice']
        results['verification_aware_average_wc_r_boundary_iou'] = verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']
        results['verification_aware_average_wc_r_cldice'] = verification_aware_metrics['verification_aware_average_wc_r_cldice']

        # avg_data = [avg_numRb, avg_numUnk, avg_numMis, avg_numAtt, avg_riou, avg_rv, avg_rs, avg_vt, max_mem]
        csr_results.append(results.copy())
        results = [csr_results, coo_results]
        pickle.dump(results, open(save_file, 'wb'))
    del CSR_set

    results = [csr_results, coo_results]
    pickle.dump(results, open(save_file, 'wb'))

    print('==============================================')
    print(f'Verifying SSNN: {net_name} with SparseImageStar2DCOO...')
    num_classes = starv_net.layers[-1].classes

    for num_max in num_max_pixels_attack:
        print('==============================================')
        print(f'Verifying SSNNs with SparseImageStar2DCOO and bounded brightening attack with num_max = {num_max}...')

        # creating SparseImageStar2DCOO input set for all images
        print('Creating SparseImageStar2DCOO input set for all images...')
        COO_set = create_reachable_set(image_data[:, :, :, i], 'SparseImageStar2DCOO', 'brightening', num_max=num_max, d=d, delta=delta, dtype=dtype)
        results = certifyRobustness_pixel(starv_net, [COO_set], in_datas, num_classes, veriMethod, reachMethod, lp_solver, return_max_memory_usage=True)
        avg_data = results['average_data']
        pix_labels = results['pixel_labels']
        pix_multiclass_labels = results['pixel_multiclass_labels']
        pix_ground_truth_labels = results['ground_truth_pixel_labels']
        
        average_wc_r_iou, average_wc_r_dice, average_wc_r_boundary_iou, average_wc_r_cldice = compute_evaluation_results(pix_ground_truth_labels[0], pix_multiclass_labels[0], num_classes=num_classes, robust_mode="unique")
        verification_aware_metrics = compute_verification_aware_evaluation_results(
            pix_ground_truth_labels[0], results['verified_images'], pix_multiclass_labels[0], num_classes=num_classes
        )
        
        print('==============================================')
        print('Verification results for SparseImageStar2DCOO')
        # print average data
        print('Num of robust pixels: ', avg_data[0])
        print('Num of unknown pixels: ', avg_data[1])
        print('Num of unrobust pixels: ', avg_data[2])
        print('Num of attacked pixels: ', avg_data[3])
        print('RIoU: ', avg_data[4])
        print('RV: ', avg_data[5])
        print('RS: ', avg_data[6])
        print('Average VT: ', avg_data[7])
        print('Max Memory Usage (bytes): ', avg_data[8])
        print('')
        print(f"Average Robust Worst-Case IoU across all classes: {average_wc_r_iou:.4f}")
        print(f"Average Robust Worst-Case Dice across all classes: {average_wc_r_dice:.4f}")
        print(f"Average Robust Worst-Case Boundary IoU across all classes: {average_wc_r_boundary_iou:.4f}")
        print(f"Average Robust Worst-Case Centerline-Dice (R-WC-clDice) across all classes: {average_wc_r_cldice:.4f}")
        print('')
        print(f"Verification-Aware Robust Worst-Case IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_iou']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_dice']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Boundary IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Centerline-Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_cldice']:.4f}")
        print('==============================================')
        results['average_wc_r_iou'] = average_wc_r_iou
        results['average_wc_r_dice'] = average_wc_r_dice
        results['average_wc_r_boundary_iou'] = average_wc_r_boundary_iou
        results['average_wc_r_cldice'] = average_wc_r_cldice
        results['verification_aware_average_wc_r_iou'] = verification_aware_metrics['verification_aware_average_wc_r_iou']
        results['verification_aware_average_wc_r_dice'] = verification_aware_metrics['verification_aware_average_wc_r_dice']
        results['verification_aware_average_wc_r_boundary_iou'] = verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']
        results['verification_aware_average_wc_r_cldice'] = verification_aware_metrics['verification_aware_average_wc_r_cldice']
        
        coo_results.append(results.copy())
        results = [csr_results, coo_results]
        pickle.dump(results, open(save_file, 'wb'))
    del COO_set



    
def verify_kvasir_unet(dtype=np.float64):
    print('=================================================================================')
    print(f"Verifying Kvasir SSNNs")
    print('=================================================================================\n')
    
    data_dir = f"StarV/util/data/nets/UNET_Kvasir"
    net_name = 'unet_kvasir_avgpool_acc_50_epochs'
    net_dir = f"{data_dir}/{net_name}.pth"
    dataset_dir = f"{data_dir}/data/Kvasir-SEG/"
    
    # Load model
    model = Kvasir_UNetAvgPool(in_ch=3, base_ch=32, out_ch=1)
    ckpt = torch.load(net_dir, map_location="cpu")
    state_dict = ckpt["model_state"]
    model.load_state_dict(state_dict)
    
    # Convert to StarV SSNN
    unet_down, unet_up = generate_unet_indices(n_down=4, doubleconv_layers=6, store_mode="before_pool", explicit_concat=True)
    starv_net = load_neural_network(model, net_type=net_name, dtype=dtype, add_pixel_class_layer=True, pix_threshold=0.0, UNet=[unet_down, unet_up], show=False)
    print(f"starv_net: {starv_net}")
    
    # Load the held-out test split
    _, _, test_ds = create_kvasir_dataset(root_dir=dataset_dir, img_size=256, batch_size=1)
    index = 0
    assert index >= 0 and index < len(test_ds), \
        f'error: test index {index} is out of range for a test split of size {len(test_ds)}'
    requested_index = test_ds.indices[index]
    image, mask = test_ds[index] # image shape in [C, H, W], value in [0, 1]; mask shape in [H, W], value in {0, 1}
    visualize_image_with_overlays(model, image, mask, index=requested_index, dataset_type='kvasir')
    # Convert from PyTorch tensors to numpy arrays for downstream use
    # NOTE: iamge is unnormalized with value in [0, 1] and shape in (C, H, W); mask is in shape (H, W) with integer values representing class labels
    image = image.permute(1, 2, 0).cpu().numpy().astype(dtype)
    mask = mask.cpu().numpy().squeeze()
    
    # attack all channel values for all pixels belonging to the target class (e.g., polyp class in kvasir dataset)
    target_class = 1 # polyp class in kvasir dataset
    mask_attack_pixels, components = build_lidar_attack_order(mask=mask, target_class=target_class, connectivity=8)
    all_selected_indices = list(range(len(mask_attack_pixels)))  # pre-select max(N) attack pixels so subsets are nested
    print(f'Number of pixels belonging to the target class (class {target_class}): {len(mask_attack_pixels)}')    
    
    num_classes = starv_net.layers[-1].classes
    N = np.array([3, 6, 9, 12, 15]) ** 2
    eps = float(0.05) / float(255.0)
    print('Number of attack pixels to verify for each N: ', N )

    save_file = f"{STORE_dir}/{artifact}_kvasir_ssnn_inf_norm_lidar_attack.pkl"
    veriMethod='BFS'
    reachMethod='approx'
    lp_solver='cupdlp' #'gurobi'
    # lp_solver = 'gurobi'
    RF = 0.0
    show = True
    
    csr_results = []
    coo_results = []
    
    # Verification with SparseImageStar2DCSR
    for i, attack_num in enumerate(N):
        selected_indices = all_selected_indices[:attack_num]  # nested: first attack_num from pre-selected pool
        current_attack_pixels = [mask_attack_pixels[idx] for idx in selected_indices]
        print(f"For N={attack_num}, selected {len(current_attack_pixels)} pixel positions for attack.")
        
        print('==============================================')
        print(f'Verifying Kvasir SSNN topological robustness under infinity norm attack on class {target_class} with {len(current_attack_pixels)} pixel positions for all channel values...')
        print('Creating SparseImageStar2DCSR input set for the image...')
        
        lb, ub, attacked_pixel_positions, attacked_channel_values = apply_inf_norm_attack_to_pixels(
            image, current_attack_pixels, eps
        )
        print(f'Number of attacked pixel positions: {attacked_pixel_positions}')
        print(f'Total number of attacked channel values: {attacked_channel_values}')
        if attacked_pixel_positions != len(current_attack_pixels):
            raise RuntimeError(
                f"Selected {len(current_attack_pixels)} attack pixels but only "
                f"{attacked_pixel_positions} pixel positions changed after applying the bounds."
            )
        
        # create the SparseImageStar2DCSR set for the image; no normalization needed
        CSR_set = SparseImageStar2DCSR(lb, ub)
        
        results = certifyRobustness_pixel(starv_net, [CSR_set], [image], num_classes, veriMethod, reachMethod, lp_solver, RF=RF, show=show, return_max_memory_usage=True)    
        avg_data = results['average_data']
        pix_labels = results['pixel_labels']
        pix_multiclass_labels = results['pixel_multiclass_labels']
        pix_ground_truth_labels = results['ground_truth_pixel_labels']
        
        average_wc_r_iou, average_wc_r_dice, average_wc_r_boundary_iou, average_wc_r_cldice = compute_evaluation_results(pix_ground_truth_labels[0], pix_multiclass_labels[0], num_classes=num_classes, robust_mode="unique")
        verification_aware_metrics = compute_verification_aware_evaluation_results(
            pix_ground_truth_labels[0], results['verified_images'], pix_multiclass_labels[0], num_classes=num_classes
        )
        
        print('==============================================')
        print('Verification results for SparseImageStar2DCSR"')
        print('target class: ', target_class)
        # print average data
        print('Num of robust pixels: ', avg_data[0])
        print('Num of unknown pixels: ', avg_data[1])
        print('Num of unrobust pixels: ', avg_data[2])
        print('Num of attacked pixels: ', avg_data[3])
        print('RIoU: ', avg_data[4])
        print('RV: ', avg_data[5])
        print('RS: ', avg_data[6])
        print('Average VT: ', avg_data[7])
        print('Max Memory Usage (bytes): ', avg_data[8])
        print('Attack pixel positions: ', len(current_attack_pixels))
        print('Total number of attacked channel values: ', attacked_channel_values)
        print('')
        print(f"Average Robust Worst-Case IoU across all classes: {average_wc_r_iou:.4f}")
        print(f"Average Robust Worst-Case Dice across all classes: {average_wc_r_dice:.4f}")
        print(f"Average Robust Worst-Case Boundary IoU across all classes: {average_wc_r_boundary_iou:.4f}")
        print(f"Average Robust Worst-Case Centerline-Dice (R-WC-clDice) across all classes: {average_wc_r_cldice:.4f}")
        print('')
        print(f"Verification-Aware Robust Worst-Case IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_iou']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_dice']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Boundary IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Centerline-Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_cldice']:.4f}")
        print('==============================================')
        results['average_wc_r_iou'] = average_wc_r_iou
        results['average_wc_r_dice'] = average_wc_r_dice
        results['average_wc_r_boundary_iou'] = average_wc_r_boundary_iou
        results['average_wc_r_cldice'] = average_wc_r_cldice
        results['verification_aware_average_wc_r_iou'] = verification_aware_metrics['verification_aware_average_wc_r_iou']
        results['verification_aware_average_wc_r_dice'] = verification_aware_metrics['verification_aware_average_wc_r_dice']
        results['verification_aware_average_wc_r_boundary_iou'] = verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']
        results['verification_aware_average_wc_r_cldice'] = verification_aware_metrics['verification_aware_average_wc_r_cldice']
        csr_results.append(results)
        pickle.dump([csr_results, coo_results], open(save_file, 'wb'))
    del CSR_set

    # Verification with SparseImageStar2DCOO
    for i, attack_num in enumerate(N):
        selected_indices = all_selected_indices[:attack_num]  # nested: first attack_num from pre-selected pool
        current_attack_pixels = [mask_attack_pixels[idx] for idx in selected_indices]
        print(f"For N={attack_num}, selected {len(current_attack_pixels)} pixel positions for attack.")
        
        print('==============================================')
        print(f'Verifying Kvasir SSNN topological robustness under infinity norm attack on class {target_class} with {len(current_attack_pixels)} pixel positions for all channel values...')
        print('Creating SparseImageStar2DCOO input set for the image...')
        
        lb, ub, attacked_pixel_positions, attacked_channel_values = apply_inf_norm_attack_to_pixels(
            image, current_attack_pixels, eps
        )
        print(f'Number of attacked pixel positions: {attacked_pixel_positions}')
        print(f'Total number of attacked channel values: {attacked_channel_values}')
        if attacked_pixel_positions != len(current_attack_pixels):
            raise RuntimeError(
                f"Selected {len(current_attack_pixels)} attack pixels but only "
                f"{attacked_pixel_positions} pixel positions changed after applying the bounds."
            )
        
        # create the SparseImageStar2DCOO set for the image; no normalization needed
        COO_set = SparseImageStar2DCOO(lb, ub)
        
        results = certifyRobustness_pixel(starv_net, [COO_set], [image], num_classes, veriMethod, reachMethod, lp_solver, RF=RF, show=show, return_max_memory_usage=True)    
        avg_data = results['average_data']
        pix_labels = results['pixel_labels']
        pix_multiclass_labels = results['pixel_multiclass_labels']
        pix_ground_truth_labels = results['ground_truth_pixel_labels']
        
        average_wc_r_iou, average_wc_r_dice, average_wc_r_boundary_iou, average_wc_r_cldice = compute_evaluation_results(pix_ground_truth_labels[0], pix_multiclass_labels[0], num_classes=num_classes, robust_mode="unique")
        verification_aware_metrics = compute_verification_aware_evaluation_results(
            pix_ground_truth_labels[0], results['verified_images'], pix_multiclass_labels[0], num_classes=num_classes
        )
        
        print('==============================================')
        print('Verification results for SparseImageStar2DCOO"')
        print('target class: ', target_class)
        # print average data
        print('Num of robust pixels: ', avg_data[0])
        print('Num of unknown pixels: ', avg_data[1])
        print('Num of unrobust pixels: ', avg_data[2])
        print('Num of attacked pixels: ', avg_data[3])
        print('RIoU: ', avg_data[4])
        print('RV: ', avg_data[5])
        print('RS: ', avg_data[6])
        print('Average VT: ', avg_data[7])
        print('Max Memory Usage (bytes): ', avg_data[8])
        print('Attack pixel positions: ', len(current_attack_pixels))
        print('Total number of attacked channel values: ', attacked_channel_values)
        print('')
        print(f"Average Robust Worst-Case IoU across all classes: {average_wc_r_iou:.4f}")
        print(f"Average Robust Worst-Case Dice across all classes: {average_wc_r_dice:.4f}")
        print(f"Average Robust Worst-Case Boundary IoU across all classes: {average_wc_r_boundary_iou:.4f}")
        print(f"Average Robust Worst-Case Centerline-Dice (R-WC-clDice) across all classes: {average_wc_r_cldice:.4f}")
        print('')
        print(f"Verification-Aware Robust Worst-Case IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_iou']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_dice']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Boundary IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Centerline-Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_cldice']:.4f}")
        print('==============================================')
        results['average_wc_r_iou'] = average_wc_r_iou
        results['average_wc_r_dice'] = average_wc_r_dice
        results['average_wc_r_boundary_iou'] = average_wc_r_boundary_iou
        results['average_wc_r_cldice'] = average_wc_r_cldice
        results['verification_aware_average_wc_r_iou'] = verification_aware_metrics['verification_aware_average_wc_r_iou']
        results['verification_aware_average_wc_r_dice'] = verification_aware_metrics['verification_aware_average_wc_r_dice']
        results['verification_aware_average_wc_r_boundary_iou'] = verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']
        results['verification_aware_average_wc_r_cldice'] = verification_aware_metrics['verification_aware_average_wc_r_cldice']
        coo_results.append(results)
        pickle.dump([csr_results, coo_results], open(save_file, 'wb'))

def verify_camvid_unet(dtype=np.float64):
    print('=================================================================================')
    print(f"Verifying CamVid SSNNs")
    print('=================================================================================\n')
    
    data_dir = f"StarV/util/data/nets/UNET_CamVid/"
    net_name = 'unet_camvid_avgpool_best'
    net_dir = f"{data_dir}/{net_name}.pth"
    dataset_dir = f"{data_dir}/CamVid/"
    
    # Load model
    model = CamV_UNetAvgPool(in_ch=3, base_ch=64, num_classes=11)
    ckpt = torch.load(net_dir, map_location="cpu")
    state_dict = ckpt["model_state"]
    model.load_state_dict(state_dict)
    
    # Convert to StarV SSNN
    unet_down, unet_up = generate_unet_indices(n_down=3, doubleconv_layers=6, store_mode="before_pool", explicit_concat=True)
    starv_net = load_neural_network(model, net_type=net_name, dtype=dtype, add_pixel_class_layer=True, pix_threshold=0.0, UNet=[unet_down, unet_up], show=False)

    # Load test dataset
    test_imgs = dataset_dir + "test"
    test_masks = dataset_dir + "test_labels"
    test_ds = CamVidDataset(test_imgs, test_masks, img_size=(360, 480), augment=False)
    
    # Get a test image and its mask at index i and visualize them
    index = 170
    image, mask = test_ds[index] # image is normalized by CamVidDataset; mask stores integer class labels
    image = camvid_unnormalize_image(image)
    visualize_image_with_overlays(model, image, mask, index=index)
    # Convert from PyTorch tensors to numpy arrays for downstream use.
    # NOTE: image is now unnormalized with values in [0, 1] and shape (C, H, W).
    image = image.permute(1, 2, 0).cpu().numpy().astype(dtype)
    mask = mask.cpu().numpy().squeeze()
    
    # attack all channel values for all pixels belonging to the target class (e.g., bicyclist class in camvid dataset)
    target_class = 10 # bicyclist class in camvid dataset    
    mask_attack_pixels, components = build_lidar_attack_order(mask=mask, target_class=target_class, connectivity=8)
    all_selected_indices = list(range(len(mask_attack_pixels)))  # pre-select max(N) attack pixels so subsets are nested
    print(f'Number of pixels belonging to the target class (class {target_class}): {len(mask_attack_pixels)}')
    
    num_classes = starv_net.layers[-1].classes
    # eps = float(0.1) / float(255.0)
    eps = float(0.05) / float(255.0)
    N = np.array([3, 6, 9, 12, 15]) ** 2
    # N = np.array([20]) ** 2

    veriMethod='BFS'
    reachMethod='approx'
    lp_solver='cupdlp' #'gurobi'
    RF = 0.0
    show = True
    
    csr_results = []
    coo_results = []
    save_file = f"{STORE_dir}/{artifact}_camvid_ssnn_inf_norm_lidar_attack.pkl"
    
    # Verification with SparseImageStar2DCSR
    for i, attack_num in enumerate(N):
        selected_indices = all_selected_indices[:attack_num]  # nested: first attack_num from pre-selected pool
        current_attack_pixels = [mask_attack_pixels[idx] for idx in selected_indices]
        print(f"For N={attack_num}, selected {len(current_attack_pixels)} pixel positions for attack.")

        print('==============================================')
        print(f'Verifying CamVid SSNN robustness under infinity norm attack on class {target_class} with {len(current_attack_pixels)} pixel positions for all channel values...')
        print('Creating SparseImageStar2DCSR input set for the image...')
        
        lb, ub, attacked_pixel_positions, attacked_channel_values = apply_inf_norm_attack_to_pixels(
            image, current_attack_pixels, eps
        )
        print(f'Number of attacked pixel positions: {attacked_pixel_positions}')
        print(f'Total number of attacked channel values: {attacked_channel_values}')
        if attacked_pixel_positions != len(current_attack_pixels):
            raise RuntimeError(
                f"Selected {len(current_attack_pixels)} attack pixels but only "
                f"{attacked_pixel_positions} pixel positions changed after applying the bounds."
            )
        
        # nomalize the bounds
        lb_norm = (lb - CAMVID_mean) / CAMVID_std
        ub_norm = (ub - CAMVID_mean) / CAMVID_std
        CSR_set = SparseImageStar2DCSR(lb_norm, ub_norm)
        
        # normalize the image since the network is trained on normalized images
        image_norm = (image - CAMVID_mean) / CAMVID_std
        
        results = certifyRobustness_pixel(starv_net, [CSR_set], [image_norm], num_classes, veriMethod, reachMethod, lp_solver, RF=RF, show=show, return_max_memory_usage=True)    
        avg_data = results['average_data']
        pix_labels = results['pixel_labels']
        pix_multiclass_labels = results['pixel_multiclass_labels']
        pix_ground_truth_labels = results['ground_truth_pixel_labels']
        
        average_wc_r_iou, average_wc_r_dice, average_wc_r_boundary_iou, average_wc_r_cldice = compute_evaluation_results(pix_ground_truth_labels[0], pix_multiclass_labels[0], num_classes=num_classes, robust_mode="unique")
        verification_aware_metrics = compute_verification_aware_evaluation_results(
            pix_ground_truth_labels[0], results['verified_images'], pix_multiclass_labels[0], num_classes=num_classes
        )
        
        print('==============================================')
        print('Verification results for SparseImageStar2DCSR"')
        print('target class: ', target_class)
        # print average data
        print('Num of robust pixels: ', avg_data[0])
        print('Num of unknown pixels: ', avg_data[1])
        print('Num of unrobust pixels: ', avg_data[2])
        print('Num of attacked pixels: ', avg_data[3])
        print('RIoU: ', avg_data[4])
        print('RV: ', avg_data[5])
        print('RS: ', avg_data[6])
        print('Average VT: ', avg_data[7])
        print('Max Memory Usage (bytes): ', avg_data[8])
        print('Attack pixel positions: ', len(current_attack_pixels))
        print('Total number of attacked channel values: ', attacked_channel_values)
        print('')
        print(f"Average Robust Worst-Case IoU across all classes: {average_wc_r_iou:.4f}")
        print(f"Average Robust Worst-Case Dice across all classes: {average_wc_r_dice:.4f}")
        print(f"Average Robust Worst-Case Boundary IoU across all classes: {average_wc_r_boundary_iou:.4f}")
        print(f"Average Robust Worst-Case Centerline-Dice (R-WC-clDice) across all classes: {average_wc_r_cldice:.4f}")
        print('')
        print(f"Verification-Aware Robust Worst-Case IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_iou']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_dice']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Boundary IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Centerline-Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_cldice']:.4f}")
        print('==============================================')
        results['average_wc_r_iou'] = average_wc_r_iou
        results['average_wc_r_dice'] = average_wc_r_dice
        results['average_wc_r_boundary_iou'] = average_wc_r_boundary_iou
        results['average_wc_r_cldice'] = average_wc_r_cldice
        results['verification_aware_average_wc_r_iou'] = verification_aware_metrics['verification_aware_average_wc_r_iou']
        results['verification_aware_average_wc_r_dice'] = verification_aware_metrics['verification_aware_average_wc_r_dice']
        results['verification_aware_average_wc_r_boundary_iou'] = verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']
        results['verification_aware_average_wc_r_cldice'] = verification_aware_metrics['verification_aware_average_wc_r_cldice']
        csr_results.append(results)
        pickle.dump([csr_results, coo_results], open(save_file, 'wb'))
    del CSR_set

    # Verification with SparseImageStar2DCOO
    for i, attack_num in enumerate(N):
        selected_indices = all_selected_indices[:attack_num]  # nested: first attack_num from pre-selected pool
        current_attack_pixels = [mask_attack_pixels[idx] for idx in selected_indices]
        print(f"For N={attack_num}, selected {len(current_attack_pixels)} pixel positions for attack.")

        print('==============================================')
        print(f'Verifying CamVid SSNN topological robustness under infinity norm attack on class {target_class} with {len(current_attack_pixels)} pixel positions for all channel values...')
        print('Creating SparseImageStar2DCOO input set for the image...')
        
        lb, ub, attacked_pixel_positions, attacked_channel_values = apply_inf_norm_attack_to_pixels(
            image, current_attack_pixels, eps
        )
        print(f'Number of attacked pixel positions: {attacked_pixel_positions}')
        print(f'Total number of attacked channel values: {attacked_channel_values}')
        if attacked_pixel_positions != len(current_attack_pixels):
            raise RuntimeError(
                f"Selected {len(current_attack_pixels)} attack pixels but only "
                f"{attacked_pixel_positions} pixel positions changed after applying the bounds."
            )
        
        # nomalize the bounds
        lb_norm = (lb - CAMVID_mean) / CAMVID_std
        ub_norm = (ub - CAMVID_mean) / CAMVID_std
        COO_set = SparseImageStar2DCOO(lb_norm, ub_norm)
        
        # normalize the image since the network is trained on normalized images
        image_norm = (image - CAMVID_mean) / CAMVID_std
        
        results = certifyRobustness_pixel(starv_net, [COO_set], [image_norm], num_classes, veriMethod, reachMethod, lp_solver, RF=RF, show=show, return_max_memory_usage=True)    
        avg_data = results['average_data']
        pix_labels = results['pixel_labels']
        pix_multiclass_labels = results['pixel_multiclass_labels']
        pix_ground_truth_labels = results['ground_truth_pixel_labels']
        
        average_wc_r_iou, average_wc_r_dice, average_wc_r_boundary_iou, average_wc_r_cldice = compute_evaluation_results(pix_ground_truth_labels[0], pix_multiclass_labels[0], num_classes=num_classes, robust_mode="unique")
        verification_aware_metrics = compute_verification_aware_evaluation_results(
            pix_ground_truth_labels[0], results['verified_images'], pix_multiclass_labels[0], num_classes=num_classes
        )
        
        print('==============================================')
        print('Verification results for SparseImageStar2DCOO')
        print('target class: ', target_class)
        # print average data
        print('Num of robust pixels: ', avg_data[0])
        print('Num of unknown pixels: ', avg_data[1])
        print('Num of unrobust pixels: ', avg_data[2])
        print('Num of attacked pixels: ', avg_data[3])
        print('RIoU: ', avg_data[4])
        print('RV: ', avg_data[5])
        print('RS: ', avg_data[6])
        print('Average VT: ', avg_data[7])
        print('Max Memory Usage (bytes): ', avg_data[8])
        print('Attack pixel positions: ', len(current_attack_pixels))
        print('Total number of attacked channel values: ', attacked_channel_values)
        print('')
        print(f"Average Robust Worst-Case IoU across all classes: {average_wc_r_iou:.4f}")
        print(f"Average Robust Worst-Case Dice across all classes: {average_wc_r_dice:.4f}")
        print(f"Average Robust Worst-Case Boundary IoU across all classes: {average_wc_r_boundary_iou:.4f}")
        print(f"Average Robust Worst-Case Centerline-Dice (R-WC-clDice) across all classes: {average_wc_r_cldice:.4f}")
        print('')
        print(f"Verification-Aware Robust Worst-Case IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_iou']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_dice']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Boundary IoU across all classes: {verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']:.4f}")
        print(f"Verification-Aware Robust Worst-Case Centerline-Dice across all classes: {verification_aware_metrics['verification_aware_average_wc_r_cldice']:.4f}")
        print('==============================================')
        results['average_wc_r_iou'] = average_wc_r_iou
        results['average_wc_r_dice'] = average_wc_r_dice
        results['average_wc_r_boundary_iou'] = average_wc_r_boundary_iou
        results['average_wc_r_cldice'] = average_wc_r_cldice
        results['verification_aware_average_wc_r_iou'] = verification_aware_metrics['verification_aware_average_wc_r_iou']
        results['verification_aware_average_wc_r_dice'] = verification_aware_metrics['verification_aware_average_wc_r_dice']
        results['verification_aware_average_wc_r_boundary_iou'] = verification_aware_metrics['verification_aware_average_wc_r_boundary_iou']
        results['verification_aware_average_wc_r_cldice'] = verification_aware_metrics['verification_aware_average_wc_r_cldice']
        coo_results.append(results)
        pickle.dump([csr_results, coo_results], open(save_file, 'wb'))


def visualize_image_with_overlays(model, image, mask, index, dataset_type='camvid'):
    """
    Show (image, ground truth overlay, predicted overlay) for a single example from a DataLoader.
    Image is not normalized (pixel values in [0,1]) to visualize the original image and also use it for overlaying the masks. 
    Nnormalize the image before feeding it through the network.
    """
    print(f'Visualizing (image, ground truth overlay, predicted overlay) from a {dataset_type} dataset...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # normalize the image for the model
    if dataset_type == 'camvid':
        mean = torch.tensor(CAMVID_mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
        std = torch.tensor(CAMVID_std, dtype=image.dtype, device=image.device).view(3, 1, 1)
        img = (image - mean) / std
    else:
        img = image
    true_mask = mask.squeeze().cpu()  # (H,W), values in {0..10,255}
    
    with torch.no_grad():
        outputs = model(img.unsqueeze(0).to(device))
        if dataset_type == 'camvid':
            pred_mask = torch.argmax(outputs, dim=1).cpu().squeeze()
        else:
            pred_mask = (torch.sigmoid(outputs) > 0.5).float().cpu().squeeze()
    
    # Convert image back to [0,1] range for visualization
    img_np = image.clone().cpu().permute(1, 2, 0).numpy()
    
    # Colorize masks for visualization
    if dataset_type == 'camvid':
        true_color_mask = colorize_camvid_mask(true_mask)
        pred_color_mask = colorize_camvid_mask(pred_mask)
    else:
        true_color_mask = true_mask
        pred_color_mask = pred_mask
    
    # 4. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # --- Plot 1: Original Image ---
    axes[0].imshow(img_np)
    axes[0].set_title(f"Image (Index {index})")
    axes[0].axis("off")

    # --- Plot 2: Ground Truth Overlay ---
    axes[1].imshow(img_np)
    axes[1].imshow(true_color_mask, alpha=0.4, interpolation="nearest")
    axes[1].set_title("Ground Truth Overlay")
    axes[1].axis("off")

    # --- Plot 3: Prediction Overlay ---
    axes[2].imshow(img_np)
    axes[2].imshow(pred_color_mask, alpha=0.4, interpolation="nearest")
    axes[2].set_title("Prediction Overlay")
    axes[2].axis("off")

    # 5. Save the figure
    if not os.path.exists(STORE_dir):
        os.makedirs(STORE_dir)

    plt.tight_layout()
    save_path = f"{STORE_dir}/{artifact}_visualize_{dataset_type}_image_with_overlaysay.png"
    print('Saving visualization to: ', save_path)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

       
def plot_attack_results(attack_case):
    # load verification results
    path = STORE_dir

    if attack_case == 'kvasir_inf_norm':
        load_file = f"{path}/{artifact}_kvasir_ssnn_inf_norm_lidar_attack.pkl"
        num_attacked_pixels = 3 * np.array([3, 6, 9, 12, 15]) ** 2   # since each pixel has 3 channel values attacked
        save_file_full = f"{path}/{artifact}_kvasir_ssnn_inf_norm_attack_results_full.png"
        save_file_short = f"{path}/{artifact}_kvasir_ssnn_inf_norm_attack_results_short.png"
        save_file_semantic = f"{path}/{artifact}_kvasir_ssnn_inf_norm_attack_results_semantic.png"
    elif attack_case == 'camvid_inf_norm':
        load_file = f"{path}/{artifact}_camvid_ssnn_inf_norm_lidar_attack.pkl"
        num_attacked_pixels = 3 * np.array([3, 6, 9, 12, 15]) ** 2   # since each pixel has 3 channel values attacked
        save_file_full = f"{path}/{artifact}_camvid_ssnn_inf_norm_attack_results_full.png"
        save_file_short = f"{path}/{artifact}_camvid_ssnn_inf_norm_attack_results_short.png"
        save_file_semantic = f"{path}/{artifact}_camvid_ssnn_inf_norm_attack_results_semantic.png"
    # elif attack_case == 'm2nist_ubaa_darkening':
    #     load_file = f"{path}/{artifact}_m2nist_ssnn_ubaa_darkening_verification_results.pkl"
    #     num_attacked_pixels = [5, 10, 15, 20, 25]
    elif attack_case == 'm2nist_brightening':
        load_file = f"{path}/{artifact}_m2nist_ssnn_bounded_attack_brightening_verification_results.pkl"
        num_attacked_pixels = [5, 10, 15, 20]
        save_file_full = f"{path}/{artifact}_m2nist_ssnn_bounded_attack_brightening_verification_results_full.png"
        save_file_short = f"{path}/{artifact}_m2nist_ssnn_bounded_attack_brightening_verification_results_short.png"
        save_file_semantic = f"{path}/{artifact}_m2nist_ssnn_bounded_attack_brightening_verification_results_semantic.png"
    else:
        raise ValueError("Invalid attack_case. Choose from 'kvasir_inf_norm', 'm2nist_ubaa_darkening', 'm2nist_brightening'.")

    loaded_results = pickle.load(open(load_file, 'rb'))
    if isinstance(loaded_results, (list, tuple)) and len(loaded_results) == 2 and len(loaded_results[0]) > 0 and len(loaded_results[1]) > 0:
        csr_results, coo_results = loaded_results
    else:
        csr_results = loaded_results
        coo_results = None

    def to_float(value):
        try:
            if isinstance(value, str) and value.strip().lower() in {'o/m', 'nan', 'none', ''}:
                return np.nan
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def build_average_data_matrix(results):
        if results is None:
            return None
        if isinstance(results, dict):
            avg_data = results.get('average_data', [])
            if not avg_data:
                return np.empty((0, 0), dtype=np.float64)
            return np.array([[to_float(v) for v in avg_data]], dtype=np.float64)
        if isinstance(results, (list, tuple)):
            results = list(results)
            if len(results) == 0:
                return np.empty((0, 0), dtype=np.float64)
            first = results[0]
            if isinstance(first, dict):
                rows = []
                max_width = 0
                for entry in results:
                    avg_data = entry.get('average_data', [])
                    row = [to_float(v) for v in avg_data]
                    rows.append(row)
                    max_width = max(max_width, len(row))
                matrix = np.full((len(rows), max_width), np.nan, dtype=np.float64)
                for row_index, row in enumerate(rows):
                    matrix[row_index, :len(row)] = row
                return matrix
            nested = [build_average_data_matrix(entry) for entry in results]
            nested = [matrix for matrix in nested if matrix is not None and matrix.size > 0]
            if len(nested) == 0:
                return np.empty((0, 0), dtype=np.float64)
            shapes = {matrix.shape for matrix in nested}
            if len(shapes) == 1:
                return np.nanmean(np.stack(nested, axis=0), axis=0)
            if all(matrix.ndim == 2 and matrix.shape[0] == 1 for matrix in nested):
                return np.vstack(nested)
            return nested[0]
        return None

    def get_metric_value(data_matrix, result_index, metric_index):
        if data_matrix is None or data_matrix.ndim != 2:
            return np.nan
        if result_index >= data_matrix.shape[0] or metric_index >= data_matrix.shape[1]:
            return np.nan
        return data_matrix[result_index, metric_index]

    keys_list = ['numRbPixels', 'numUnkPixels', 'numMisPixels', 'numAttPixels', 'RIoU', 'RV', 'RS', 'VT', 'MaxMemoryUsage']
    # Create the index mapping keys to their zero-based index
    key_index = {key: i for i, key in enumerate(keys_list)} 

    csr_average_data = build_average_data_matrix(csr_results)
    coo_average_data = build_average_data_matrix(coo_results)
 
    csr_dataset = np.full([len(keys_list), len(num_attacked_pixels)], np.nan)
    coo_dataset = np.full([len(keys_list), len(num_attacked_pixels)], np.nan) if coo_results is not None else None
    for j in range(len(keys_list)):
        for i in range(len(num_attacked_pixels)):
            if keys_list[j] == 'MaxMemoryUsage':
                csr_dataset[j, i] = get_metric_value(csr_average_data, i, key_index[keys_list[j]]) / (1024**3)  # convert to GB
                if coo_dataset is not None:
                    coo_dataset[j, i] = get_metric_value(coo_average_data, i, key_index[keys_list[j]]) / (1024**3)  # convert to GB
            else:
                csr_dataset[j, i] = get_metric_value(csr_average_data, i, key_index[keys_list[j]])
                if coo_dataset is not None:
                    coo_dataset[j, i] = get_metric_value(coo_average_data, i, key_index[keys_list[j]])

    csr_memory_usage = csr_dataset[key_index['MaxMemoryUsage'], :]
    print('CSR Memory Usage (GB): ', csr_memory_usage)
    csr_vt = csr_dataset[key_index['VT'], :]
    print('CSR VT (sec): ', csr_vt)
    if coo_dataset is not None:
        coo_memory_usage = coo_dataset[key_index['MaxMemoryUsage'], :]
        print('COO Memory Usage (GB): ', coo_memory_usage)
        print('avg coo/csr memory: ', np.sum(coo_memory_usage / csr_memory_usage) / len(csr_memory_usage))
        print('coo/csr memory: ', coo_memory_usage / csr_memory_usage)

        coo_vt = coo_dataset[key_index['VT'], :]
        print('COO VT (sec): ', coo_vt)
        print('coo/csr vt: ', coo_vt / csr_vt)
        print('max coo/csr vt: ', np.max(coo_vt / csr_vt))
    
    # Full set of plots for Appendix
    y_labels = ['Number of Robust Pixels', 'Number of Unknown Pixels', 'Number of Unrobust Pixels',
                'Number of Attacked Pixels', 'Robust IoU', 'Robustness Value', 'Robustness Sensitivity', 'Verification Time (min)', 'Max Memory Usage (GB)']
    font_size = 16
    tick_size = 15
    fig, axes = plt.subplots(2, 4, figsize=(13, 8), constrained_layout=True)
    # fig.suptitle(r'Kvasir SSNN Infinity Norm Attack Verification Results with $\epsilon = 1/255$', fontsize=20)
    i_ = 0
    for i in range(len(keys_list)):
        csr = np.array(csr_dataset[i, :], dtype=np.float64)
        if keys_list[i] == 'VT':
            csr = csr / 60.0
        if keys_list[i] == 'numAttPixels':
            continue
        ax = axes[i_%2, i_//2]
        ax.plot(num_attacked_pixels, csr, marker='o', color='red', linestyle='-', linewidth=3, label='SIM CSR')
        if coo_dataset is not None:
            coo = np.array(coo_dataset[i, :], dtype=np.float64)
            if keys_list[i] == 'VT':
                coo = coo / 60.0
            ax.plot(num_attacked_pixels, coo, marker='+', color='blue', linestyle='-.', linewidth=3, label='SIM COO')
        ax.set_xlabel('Number of Attacked Pixels', fontsize=font_size)
        ax.set_ylabel(f'{y_labels[i]}', fontsize=font_size)
        ax.set_xticks(num_attacked_pixels)
        ax.ticklabel_format(axis='y', style='plain')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        ax.tick_params(axis='x', labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        ax.legend(fontsize=tick_size)
        ax.grid()
        i_ += 1
    plt.tight_layout()
    plt.savefig(save_file_full, bbox_inches='tight', pad_inches=0.5)
    plt.figure()

    # Short set of plots for main paper
    y_labels = ['Number of Robust Pixels', 'Number of Unknown Pixels', 'Number of Unrobust Pixels',
                'Number of Attacked Pixels', 'Robust IoU', 'Robustness Value', 'Robustness Sensitivity', 'Verification Time (min)', 'Max Memory Usage (GB)']
    font_size = 16
    tick_size = 15
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    # fig.suptitle(r'Kvasir SSNN Infinity Norm Attack Verification Results with $\epsilon = 1/255$', fontsize=20)
    i_ = 0
    for i in range(len(keys_list)):
        # if keys_list[i] not in ['numRbPixels', 'RV', 'VT', 'MaxMemoryUsage']:
        if keys_list[i] not in ['RV', 'RS', 'VT', 'MaxMemoryUsage']:
            continue
        csr = np.array(csr_dataset[i, :], dtype=np.float64)
        if keys_list[i] == 'VT':
            csr = csr / 60.0
        if keys_list[i] == 'numAttPixels':
            continue
        ax = axes[i_]
        ax.plot(num_attacked_pixels, csr, marker='o', color='red', linestyle='-', linewidth=3, label='SIM CSR')
        if coo_dataset is not None:
            coo = np.array(coo_dataset[i, :], dtype=np.float64)
            if keys_list[i] == 'VT':
                coo = coo / 60.0
            ax.plot(num_attacked_pixels, coo, marker='+', color='blue', linestyle='-.', linewidth=3, label='SIM COO')
        ax.set_xlabel('Number of Attacked Pixels', fontsize=font_size)
        ax.set_ylabel(f'{y_labels[i]}', fontsize=font_size)
        ax.set_xticks(num_attacked_pixels)
        ax.ticklabel_format(axis='y', style='plain')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        ax.tick_params(axis='x', labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        ax.legend(fontsize=tick_size)
        ax.grid()
        i_ += 1
    plt.tight_layout()
    plt.savefig(save_file_short, bbox_inches='tight', pad_inches=0.5)
    plt.figure()

    # Plot Semantic Metrics (RIoU, RDice, RBIoU, RclDice) for main paper
    plot_sematic_metrics(
        None,
        csr_results,
        None,
        'all',
        save_file_semantic,
        method_styles={'SIM CSR': ('red', '-')},
        legend_mode='inside',
    )

def memory_usage_unet_camvid(dtype = np.float64):
    print('=================================================================================')
    print(f"Memory Usage of CamVid UNet")
    print('=================================================================================\n')

    data_dir = f"StarV/util/data/nets/UNET_CamVid/"
    net_name = 'unet_camvid_avgpool_best'
    net_dir = f"{data_dir}/{net_name}.pth"
    dataset_dir = f"{data_dir}/CamVid/"

    # Load model
    model = CamV_UNetAvgPool(in_ch=3, base_ch=64, num_classes=11)
    ckpt = torch.load(net_dir, map_location="cpu")
    state_dict = ckpt["model_state"]
    model.load_state_dict(state_dict)
    
    # Convert to StarV SSNN
    unet_down, unet_up = generate_unet_indices(n_down=3, doubleconv_layers=6, store_mode="before_pool", explicit_concat=True)
    starv_net = load_neural_network(model, net_type=net_name, dtype=dtype, add_pixel_class_layer=True, pix_threshold=0.0, UNet=[unet_down, unet_up], show=False)

    # # Load test dataset
    # test_imgs = dataset_dir + "test"
    # test_masks = dataset_dir + "test_labels"
    # test_ds = CamVidDataset(test_imgs, test_masks, img_size=(360, 480), augment=False)
    
    # # Get a test image and its mask at index i and visualize them
    # index = 170
    # image, mask = test_ds[index] # image is normalized by CamVidDataset; mask stores integer class labels
    # image = camvid_unnormalize_image(image)
    # # Convert from PyTorch tensors to numpy arrays for downstream use.
    # # NOTE: image is now unnormalized with values in [0, 1] and shape (C, H, W).
    # image = image.permute(1, 2, 0).cpu().numpy().astype(dtype)
    # mask = mask.cpu().numpy().squeeze()
    
    # # attack all channel values for all pixels belonging to the target class (e.g., bicyclist class in camvid dataset)
    # target_class = 10 # bicyclist class in camvid dataset    
    # mask_attack_pixels, components = build_lidar_attack_order(mask=mask, target_class=target_class, connectivity=8)
    # all_selected_indices = list(range(len(mask_attack_pixels)))  # pre-select max(N) attack pixels so subsets are nested
    # print(f'Number of pixels belonging to the target class (class {target_class}): {len(mask_attack_pixels)}')

    # eps = float(0.05) / float(255.0)
    # N = np.array([3, 6, 9, 12, 15]) ** 2

    # CSR_time = []
    # attack_num = N[-1]  # max number of attack pixels to evaluate for memory usage

    # selected_indices = all_selected_indices[:attack_num]  # nested: first attack_num from pre-selected pool
    # current_attack_pixels = [mask_attack_pixels[idx] for idx in selected_indices]
    # print(f"For N={attack_num}, selected {len(current_attack_pixels)} pixel positions for attack.")

    # lb, ub, attacked_pixel_positions, attacked_channel_values = apply_inf_norm_attack_to_pixels(
    #     image, current_attack_pixels, eps
    # )
    # print(f'Number of attacked pixel positions: {attacked_pixel_positions}')
    # print(f'Total number of attacked channel values: {attacked_channel_values}')
    # if attacked_pixel_positions != len(current_attack_pixels):
    #     raise RuntimeError(
    #         f"Selected {len(current_attack_pixels)} attack pixels but only "
    #         f"{attacked_pixel_positions} pixel positions changed after applying the bounds."
    #     )
 
    # print('working on SparseImageStar CSR')
    # CSR = SparseImageStar2DCSR(lb, ub)
    # CSR_nb = [CSR.nbytes()]
    # density = [CSR.density()]
    # stored_X = dict()
    # assert len(unet_down) == len(unet_up), \
    #     f'error: inconsistent UNet skip layout, len(unet_down)={len(unet_down)} != len(unet_up)={len(unet_up)}'
    # down_to_up = {int(d): int(u) for d, u in zip(unet_down[::-1], unet_up)}
    # for i in range(starv_net.n_layers):
    #     print(f'Layer {i}/{starv_net.n_layers}: {starv_net.layers[i].__class__.__name__}')
    #     start = time.perf_counter()
    #     if i in unet_down:
    #         print(f'  Store layer {i} output for skip connection')
    #         stored_X[down_to_up[i]] = CSR
    #     if i in unet_up:
    #         print(f'  Skip layer {i} reachability analysis')
    #         X = stored_X[i]
    #         CSR = starv_net.layers[i].reach([CSR, X], method='approx', show=False)
    #     else:
    #         CSR = starv_net.layers[i].reach(CSR, method='approx', show=False)
    #     if starv_net.layers[i].__class__.__name__ == 'PixelClassificationLayer':
    #         continue
    #     CSR_time.append(time.perf_counter() - start)
    #     CSR_nb.append(CSR.nbytes())
    #     density.append(CSR.density())
    # del CSR, stored_X

    # path for saving verification results
    path = STORE_dir
    if not os.path.exists(path):
        os.makedirs(path)
    save_file = f"{path}/{artifact}_camvid_unet_memory_usage.pkl"
    # pickle.dump([CSR_time, CSR_nb, density], open(save_file, 'wb'))
    CSR_time, CSR_nb, density = pickle.load(open(save_file, 'rb'))

    x = np.arange(len(CSR_time))
    x_ticks_labels = []
    for i in range(starv_net.n_layers):
        if starv_net.layers[i].__class__.__name__ == 'Conv2DLayer':
            l_name = '$L_c$'
        elif starv_net.layers[i].__class__.__name__ == 'ReLULayer':
            l_name = '$L_r$'
        elif starv_net.layers[i].__class__.__name__ == 'FlattenLayer':
            l_name = '$L_{{flat}}$'
        elif starv_net.layers[i].__class__.__name__ == 'FullyConnectedLayer':
            l_name = '$L_f$'
        elif starv_net.layers[i].__class__.__name__ == 'AvgPool2DLayer':
            l_name = '$L_a$'
        elif starv_net.layers[i].__class__.__name__ == 'PixelClassificationLayer':
            # l_name = '$L_{pix}$'
            continue
        elif starv_net.layers[i].__class__.__name__ == 'TransposedConv2DLayer' or starv_net.layers[i].__class__.__name__ == 'ConvTranspose2DLayer':
            l_name = '$L_t$'
        elif starv_net.layers[i].__class__.__name__ == 'BatchNorm2DLayer':
            l_name = '$L_b$'
        elif starv_net.layers[i].__class__.__name__ == 'PadLayer':
            l_name = '$L_{pad}$'
        elif starv_net.layers[i].__class__.__name__ == 'ConcatenateLayer':
            l_name = '$L_{concat}$'
        else:
            l_name = f"${starv_net.layers[i].__class__.__name__}$"
        x_ticks_labels.append(f"{l_name}_{i}")

    font_size = 25
    y_tick_size = 20
    tick_size = 20
    line_width = 5

    CSR_time = np.array(CSR_time) / float(60.0)  # convert to minutes
    CSR_nb = np.array(CSR_nb) / float(1024**3)  # convert to GB

    plt.rcParams["figure.figsize"] = [22, 7]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots(1, 2, layout="constrained")
    ax[1].set_title("Computation Time", fontsize=font_size)
    ax[1].plot(x, CSR_time, color="magenta", linewidth=line_width, linestyle='-')
    ax[1].set_xlabel("Layers", fontsize=font_size)
    ax[1].set_ylabel("Computation Time (min)", fontsize=font_size)
    # Set number of ticks for x-axis
    ax[1].set_xticks(x)
    # Set ticks labels for x-axis
    ax[1].set_xticklabels(x_ticks_labels, rotation=80, fontsize=tick_size)
    ax[1].yaxis.set_tick_params(labelsize=y_tick_size)
    # set legend
    ax[1].legend(['SIM CSR'], fontsize=font_size)
    ax[1].grid()

    x = np.arange(len(CSR_nb))
    x_ticks_labels.insert(0, 'Input')

    ax[0].set_title("Memory Usage", fontsize=font_size)
    ax[0].plot(x, CSR_nb, color="magenta", linewidth=line_width, linestyle='-')
    ax[0].set_xlabel("Layers", fontsize=font_size)
    ax[0].set_ylabel("Memory Usage (GB)", fontsize=font_size)
    # Set number of ticks for x-axis
    ax[0].set_xticks(x)
    # Set ticks labels for x-axis
    ax[0].set_xticklabels(x_ticks_labels, rotation=80, fontsize=tick_size)
    ax[0].yaxis.set_tick_params(labelsize=tick_size)
    # set legend
    ax[0].legend(['SIM CSR'], loc='center left', fontsize=font_size)
    ax2 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, density, color="green", linewidth=2)
    ax2.legend(['density'], fontsize=font_size, loc='upper left')
    ax2.set_ylabel("Density", fontsize=y_tick_size)
    ax2.yaxis.set_tick_params(labelsize=y_tick_size)
    plt.tight_layout()
    plt.margins(x=0)
    plt.savefig(f'{path}/{artifact}_camvid_unet_memory_usage.png')
    plt.close()

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def memory_usage_unet_kvasir(dtype = np.float64):
    print('=================================================================================')
    print(f"Memory Usage of Kavsir UNet")
    print('=================================================================================\n')

    data_dir = f"StarV/util/data/nets/UNET_Kvasir"
    net_name = 'unet_kvasir_avgpool_acc_50_epochs'
    net_dir = f"{data_dir}/{net_name}.pth"
    dataset_dir = f"{data_dir}/data/Kvasir-SEG/"
    
    # Load model
    model = Kvasir_UNetAvgPool(in_ch=3, base_ch=32, out_ch=1)
    ckpt = torch.load(net_dir, map_location="cpu")
    state_dict = ckpt["model_state"]
    model.load_state_dict(state_dict)
    
    # Convert to StarV SSNN
    unet_down, unet_up = generate_unet_indices(n_down=4, doubleconv_layers=6, store_mode="before_pool", explicit_concat=True)
    starv_net = load_neural_network(model, net_type=net_name, dtype=dtype, add_pixel_class_layer=True, pix_threshold=0.0, UNet=[unet_down, unet_up], show=False)

    # # Load the held-out test split
    # _, _, test_ds = create_kvasir_dataset(root_dir=dataset_dir, img_size=256, batch_size=1)
    # index = 0
    # assert index >= 0 and index < len(test_ds), \
    #     f'error: test index {index} is out of range for a test split of size {len(test_ds)}'
    # requested_index = test_ds.indices[index]
    # image, mask = test_ds[index] # image shape in [C, H, W], value in [0, 1]; mask shape in [H, W], value in {0, 1}
    
    # # Convert from PyTorch tensors to numpy arrays for downstream use
    # # NOTE: iamge is unnormalized with value in [0, 1] and shape in (C, H, W); mask is in shape (H, W) with integer values representing class labels
    # image = image.permute(1, 2, 0).cpu().numpy().astype(dtype)
    # mask = mask.cpu().numpy().squeeze()
    
    # # attack all channel values for all pixels belonging to the target class (e.g., polyp class in kvasir dataset)
    # target_class = 1 # polyp class in kvasir dataset
    # mask_attack_pixels, components = build_lidar_attack_order(mask=mask, target_class=target_class, connectivity=8)
    # all_selected_indices = list(range(len(mask_attack_pixels)))  # pre-select max(N) attack pixels so subsets are nested
    # print(f'Number of pixels belonging to the target class (class {target_class}): {len(mask_attack_pixels)}')    

    # eps = float(0.05) / float(255.0)
    # N = np.array([3, 6, 9, 12, 15]) ** 2
    
    # CSR_time = []
    # attack_num = N[-1]  # max number of attack pixels to evaluate for memory usage

    # selected_indices = all_selected_indices[:attack_num]  # nested: first attack_num from pre-selected pool
    # current_attack_pixels = [mask_attack_pixels[idx] for idx in selected_indices]
    # print(f"For N={attack_num}, selected {len(current_attack_pixels)} pixel positions for attack.")

    # lb, ub, attacked_pixel_positions, attacked_channel_values = apply_inf_norm_attack_to_pixels(
    #     image, current_attack_pixels, eps
    # )
    # print(f'Number of attacked pixel positions: {attacked_pixel_positions}')
    # print(f'Total number of attacked channel values: {attacked_channel_values}')
    # if attacked_pixel_positions != len(current_attack_pixels):
    #     raise RuntimeError(
    #         f"Selected {len(current_attack_pixels)} attack pixels but only "
    #         f"{attacked_pixel_positions} pixel positions changed after applying the bounds."
    #     )
 
    # print('working on SparseImageStar CSR')
    # CSR = SparseImageStar2DCSR(lb, ub)
    # CSR_nb = [CSR.nbytes()]
    # density = [CSR.density()]
    # stored_X = dict()
    # assert len(unet_down) == len(unet_up), \
    #     f'error: inconsistent UNet skip layout, len(unet_down)={len(unet_down)} != len(unet_up)={len(unet_up)}'
    # down_to_up = {int(d): int(u) for d, u in zip(unet_down[::-1], unet_up)}
    # for i in range(starv_net.n_layers):
    #     print(f'Layer {i}/{starv_net.n_layers}: {starv_net.layers[i].__class__.__name__}')
    #     start = time.perf_counter()
    #     if i in unet_down:
    #         print(f'  Store layer {i} output for skip connection')
    #         stored_X[down_to_up[i]] = CSR
    #     if i in unet_up:
    #         print(f'  Skip layer {i} reachability analysis')
    #         X = stored_X[i]
    #         CSR = starv_net.layers[i].reach([CSR, X], method='approx', show=False)
    #     else:
    #         CSR = starv_net.layers[i].reach(CSR, method='approx', show=False)
    #     if starv_net.layers[i].__class__.__name__ == 'PixelClassificationLayer':
    #         continue
    #     CSR_time.append(time.perf_counter() - start)
    #     CSR_nb.append(CSR.nbytes())
    #     density.append(CSR.density())
    # del CSR, stored_X

    # path for saving verification results
    path = STORE_dir
    if not os.path.exists(path):
        os.makedirs(path)
    save_file = f"{path}/{artifact}_kvasir_unet_memory_usage.pkl"
    # pickle.dump([CSR_time, CSR_nb, density], open(save_file, 'wb'))
    CSR_time, CSR_nb, density = pickle.load(open(save_file, 'rb'))

    CSR_time = np.array(CSR_time) / float(60.0)
    CSR_nb = np.array(CSR_nb) / float(1024**3)  # convert to GB

    x = np.arange(len(CSR_time))
    x_ticks_labels = []
    for i in range(starv_net.n_layers):
        if starv_net.layers[i].__class__.__name__ == 'Conv2DLayer':
            l_name = '$L_c$'
        elif starv_net.layers[i].__class__.__name__ == 'ReLULayer':
            l_name = '$L_r$'
        elif starv_net.layers[i].__class__.__name__ == 'FlattenLayer':
            l_name = '$L_{{flat}}$'
        elif starv_net.layers[i].__class__.__name__ == 'FullyConnectedLayer':
            l_name = '$L_f$'
        elif starv_net.layers[i].__class__.__name__ == 'AvgPool2DLayer':
            l_name = '$L_a$'
        elif starv_net.layers[i].__class__.__name__ == 'PixelClassificationLayer':
            # l_name = '$L_{pix}$'
            continue
        elif starv_net.layers[i].__class__.__name__ == 'TransposedConv2DLayer' or starv_net.layers[i].__class__.__name__ == 'ConvTranspose2DLayer':
            l_name = '$L_t$'
        elif starv_net.layers[i].__class__.__name__ == 'BatchNorm2DLayer':
            l_name = '$L_b$'
        elif starv_net.layers[i].__class__.__name__ == 'PadLayer':
            l_name = '$L_{pad}$'
        elif starv_net.layers[i].__class__.__name__ == 'ConcatenateLayer':
            l_name = '$L_{concat}$'
        else:
            l_name = f"${starv_net.layers[i].__class__.__name__}$"
        x_ticks_labels.append(f"{l_name}_{i}")

    font_size = 25
    y_tick_size = 20
    tick_size = 20
    line_width = 5


    plt.rcParams["figure.figsize"] = [22, 7]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots(1, 2, layout="constrained")
    ax[1].set_title("Computation Time", fontsize=font_size)
    ax[1].plot(x, CSR_time, color="magenta", linewidth=line_width, linestyle='-')
    ax[1].set_xlabel("Layers", fontsize=font_size)
    ax[1].set_ylabel("Computation Time (min)", fontsize=font_size)
    # Set number of ticks for x-axis
    ax[1].set_xticks(x)
    # Set ticks labels for x-axis
    ax[1].set_xticklabels(x_ticks_labels, rotation=80, fontsize=tick_size)
    ax[1].yaxis.set_tick_params(labelsize=y_tick_size)
    # set legend
    ax[1].legend(['SIM CSR'], fontsize=font_size)
    ax[1].grid()

    x = np.arange(len(CSR_nb))
    x_ticks_labels.insert(0, 'Input')


    ax[0].set_title("Memory Usage", fontsize=font_size)
    ax[0].plot(x, CSR_nb, color="magenta", linewidth=line_width, linestyle='-')
    ax[0].set_xlabel("Layers", fontsize=font_size)
    ax[0].set_ylabel("Memory Usage (GB)", fontsize=font_size)
    # Set number of ticks for x-axis
    ax[0].set_xticks(x)
    # Set ticks labels for x-axis
    ax[0].set_xticklabels(x_ticks_labels, rotation=80, fontsize=tick_size)
    ax[0].yaxis.set_tick_params(labelsize=tick_size)
    # set legend
    ax[0].legend(['SIM CSR'], loc='center left', fontsize=font_size)
    ax2 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, density, color="green", linewidth=2)
    ax2.legend(['density'], fontsize=font_size, loc='upper left')
    ax2.set_ylabel("Density", fontsize=y_tick_size)
    ax2.yaxis.set_tick_params(labelsize=y_tick_size)
    plt.tight_layout()
    plt.margins(x=0)
    plt.savefig(f'{path}/{artifact}_kvasir_unet_memory_usage.png')
    plt.close()

    print('=====================================================')
    print('DONE!')
    print('=====================================================')

if __name__ == '__main__':
    
    print(f'Running {artifact} Artifact Evaluation for Memory-Efficient Verification of Semantic Segmentation Neural Networks...')
    
    # CAV2021 SSNN M2NIST UBAA Darkening Attack (Compare NNV); bounded attack on a single predicate variable
    verify_m2nist_ssnn_ubaa_darkening()
    plot_mnist_ssnn_ubaa_darkening_results(num_images=20)
    
    # SSNN M2NIST Brightening Attack; bounded attack on predicate variables 
    # (number of predicate variables corresponds to the number of attack pixels)
    verify_m2nist_ssnn_brightening()
    plot_attack_results('m2nist_brightening')
    
    # Plots for memory usage 
    memory_usage_ssnn()
    memory_usage_unet_kvasir()
    memory_usage_unet_camvid()
    

    # Kvasir-SEG Infinity Norm Attack; bounded attack on predicate variables
    # (number of predicate variables corresponds to the number of attack pixels)
    verify_kvasir_unet()
    plot_attack_results('kvasir_inf_norm')
    
    # Verify CamVid UNET
    verify_camvid_unet()
    plot_attack_results('camvid_inf_norm')

    verify_kvasir_unet()
    plot_attack_results('kvasir_inf_norm')
    verify_camvid_unet()
    plot_attack_results('camvid_inf_norm')

    print(f'\n\nFinished running {artifact} Artifact Evaluation!')
