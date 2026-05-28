import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize


def _infer_num_classes(gt_labels, pred_labels, num_classes=None):
    if num_classes is not None:
        inferred = int(num_classes)
    elif gt_labels.ndim == 3:
        inferred = gt_labels.shape[2]
    elif pred_labels.ndim == 3:
        inferred = pred_labels.shape[2]
    else:
        inferred = int(max(np.max(gt_labels), np.max(pred_labels)) + 1)

    if inferred == 1 and (np.max(gt_labels) >= 1 or np.max(pred_labels) >= 1):
        inferred = 2
    return inferred

def _get_class_mask(labels, class_id):
    """
    labels: 
        1) (H, W) int class labels, where each pixel has a single class label (0, 1, ..., num_classes-1)
        2) (H, W, C) binary mask for each class, where labels[i, j, c] is True if pixel (i, j) belongs to class c 
            (this can allow for multiple classes per pixel, e.g., in uncertain cases).
    class_id: integer, the class ID for which to extract the mask
    
    Note:
        - The input labels can be either (H, W) int OR (H,W,C) bool, and the output will be a binary mask of shape (H, W) where True indicates pixels belonging to class_id.
        - This function converts the input labels into a binary mask for the specified class_id, which can then be used for computing semantic metrics like boundary IoU or clDice.
        
    Returns:
    - A binary mask of shape (H, W) where True indicates pixels belonging to the specified class_id, and False otherwise. 
    """
    if labels.ndim == 2:
        return (labels == class_id)
   
    if labels.ndim == 3:
        C = labels.shape[2]
        if not (0 <= int(class_id) < C):
            raise ValueError(f"class_id={class_id} out of range for labels with C={C}")
        return labels[..., class_id].astype(bool)
    
    raise ValueError("labels must be 2D (H,W) or 3D (H,W,C)")


def _get_robust_region_from_reachable(reachable_labels, class_id, robust_mode="possible"):
    """
    reachable_labels:
        1) (H, W) int class labels, where each pixel has a single reachable class label (0, 1, ..., num_classes-1)
        2) (H, W, C) binary mask for each class, where reachable_labels[i, j, c] is True if pixel (i, j) is reachable to class c 
            (this can allow for multiple classes per pixel, e.g., in uncertain cases).
    class_id: integer, the class ID for which to extract the mask
    robust_mode:
      - "possible": Omega_max (class_id is possible);   Ωmax​(c) = R(⋅,c)
      - "unique":   Omega_min (only class_id possible); Ωmin​(c) = R(⋅,c) ∧ ∑_k R(⋅,k) = 1
    """
    
    if reachable_labels.ndim == 2:
        return (reachable_labels == class_id)

    if reachable_labels.ndim != 3:
        raise ValueError("reachable_labels must be integer 2D (H,W) or boolean 3D (H,W,C) numpy array")
    
    C = reachable_labels.shape[2]
    if not (0 <= int(class_id) < C):
        raise ValueError(f"class_id={class_id} out of range for reachable_labels with C={C}")
    
    R = reachable_labels.astype(bool)
    R_c = R[..., int(class_id)]
    
    if robust_mode == "possible":
        return R_c
    elif robust_mode == "unique":
        counts = R.sum(axis=-1)
        return R_c & (counts == 1)
    else:
        raise ValueError("robust_mode must be 'possible' or 'unique'")
    
    
####################################################
# Robust IoU Semantic Metric for Pixel Classification
# Ref: 
####################################################

def compute_average_robust_iou(
    gt_labels, pred_labels,
    pred_is_reachable=True, robust_mode="possible",
    num_classes=None, ignore_empty=True):
    """
    Compute the average Robust IoU across all classes.
    """
    assert gt_labels.shape[:2] == pred_labels.shape[:2], "Ground truth and prediction label masks must have the same height and width"
    
    num_classes = _infer_num_classes(gt_labels, pred_labels, num_classes)
    
    total_wc_iou = 0.0
    count = 0
    
    for class_id in range(num_classes):
        gt_mask = _get_class_mask(gt_labels, class_id)
        
        if pred_is_reachable:
            pred_mask = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode=robust_mode) 
        else:
            pred_mask = _get_class_mask(pred_labels, class_id)  # treat pred_labels as hard labels for IoU computation

        if ignore_empty and (not gt_mask.any()) and (not pred_mask.any()):
            continue  # skip this class since it's empty in both GT and prediction

        total_wc_iou += robust_iou_binary(gt_mask, pred_mask)
        count += 1

    if count == 0:
        return 0.0  # Return 0 if no classes were processed

    avg_r_iou = total_wc_iou / count if count > 0 else 0.0
    return float(avg_r_iou)

def robust_iou_per_class(
    gt_labels, pred_labels, class_id,
    pred_is_reachable=True, robust_mode="possible"):
    """
    Compute the Robust IoU for a specific class_id given the ground truth and predicted label masks.
    """
    
    gt_mask = _get_class_mask(gt_labels, class_id)

    if pred_is_reachable:
        pred_mask = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode=robust_mode)
    else:
        pred_mask = _get_class_mask(pred_labels, class_id)
        
    return robust_iou_binary(gt_mask, pred_mask)

def robust_iou_binary(gt_mask, pred_mask):
    """
    Compute the Robust IoU between two binary masks.
    The Robust IoU metric evaluates the worst-case intersection-over-union (IoU) between the ground truth and predicted masks under any possible perturbation of the predicted mask within the robust region. 
    It is defined as the minimum IoU over all subsets of the predicted mask that are consistent with the reachable set, which can be efficiently computed using a closed-form formula based on set operations.
    
    Eq: |G ∩ P| / |G ∪ P|, where G is the GT mask and P is any subset of the reachable predicted mask.
    
    Args:
    - gt_mask: 2D boolean numpy array (h, w) representing the ground truth binary mask for a specific class.
    - pred_mask: 2D boolean numpy array (h, w) representing the predicted binary mask for the same class, where True indicates pixels that are reachable to this class.
    
    Returns:
    - wc_iou: float, the computed Robust IoU score for the specified class_id.
    """
    gt = gt_mask.astype(bool)
    pd = pred_mask.astype(bool)

    intersection = np.logical_and(gt, pd).sum()
    union = np.logical_or(gt, pd).sum()
    
    if union == 0:
        return 1.0
    
    wc_iou = float(intersection) / float(union)
    return float(wc_iou)

####################################################
# Worst Case Robust IoU Semantic Metric for Pixel Classification
# Ref:
####################################################

def compute_average_worst_case_robust_iou(
    gt_labels, pred_labels, robust_mode="possible",
    num_classes=None, ignore_empty=True):
    """
    Compute the average Worst-Case Robust IoU across all classes.
    """
    assert gt_labels.shape[:2] == pred_labels.shape[:2], "Ground truth and prediction label masks must have the same height and width"
    
    num_classes = _infer_num_classes(gt_labels, pred_labels, num_classes)
    
    total_wc_iou = 0.0
    count = 0
    
    for class_id in range(num_classes):
        gt = _get_class_mask(gt_labels, class_id)
        
        om_max = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="possible")  # Ωmax​(c) = R(⋅,c)
        om_min = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="unique")    # Ωmin​(c) = R(⋅,c) ∧ ∑_k R(⋅,k) = 1
        
        if ignore_empty and (not gt.any()) and (not om_min.any()) and (not om_max.any()):
            continue  # skip this class since it's empty in both GT and robust regions

        total_wc_iou += worst_case_robust_iou_binary(gt, om_min, om_max)
        count += 1

    if count == 0:
        return 0.0  # Return 0 if no classes were processed (e.g., all were ignored due to being empty)

    avg_wc_r_iou = total_wc_iou / count if count > 0 else 0.0
    return float(avg_wc_r_iou)

def worst_case_robust_iou_per_class(gt_labels, pred_labels, class_id, robust_mode="possible"):
    """
    Compute the Worst-case Robust IoU for a specific class_id given the ground truth and predicted label masks.
    """
    
    gt_mask = _get_class_mask(gt_labels, class_id)

    omega_max = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="possible")  # Ωmax​(c) = R(⋅,c)
    omega_min = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="unique")    # Ωmin​(c) = R(⋅,c) ∧ ∑_k R(⋅,k) = 1
        
    return worst_case_robust_iou_binary(gt_mask, omega_min, omega_max)

def worst_case_robust_iou_binary(gt_mask, omega_min_mask, omega_max_mask):
    """
    Worst-case (lower-bound) IoU over all possible predicted masks P such that:
        Omega_min ⊆ P ⊆ Omega_max.
        
    The worst-case IoU can be computed using the following closed-form formula:
    Worst-case IoU = |G ∩ Ωmin| / |G ∩ Ωmin| + |Ωmax \ G| + |G \ Ωmin| = tp_min / (tp_min + fp_max + fn_max).
    
    Args:
    - gt_mask: 2D boolean numpy array (h, w) representing the ground truth binary mask for a specific class.
    - omega_min_mask: 2D boolean numpy array (h, w) representing the minimum robust region mask for the same class, where True indicates pixels that are only reachable to this class (unique).
    - omega_max_mask: 2D boolean numpy array (h, w) representing the maximum robust region mask for the same class, where True indicates pixels that are reachable to this class (possible).
    
    Returns:
    - wc_iou: float, the computed Worst-case Robust IoU score for the specified class_id.
    """
    gt = gt_mask.astype(bool)
    om_min = omega_min_mask.astype(bool)
    om_max = omega_max_mask.astype(bool)

    tp_min = np.logical_and(gt, om_min).sum()  # |G ∩ Ωmin|
    fp_max = np.logical_and(~gt, om_max).sum()  # |Ωmax \ G|
    fn_max = np.logical_and(gt, ~om_min).sum()  # |G \ Ωmin| 

    denom = tp_min + fp_max + fn_max

    if denom == 0:
        return 1.0 # Edge case: if both GT and robust region are empty, we consider it a perfect match with IoU=1.0.

    wc_r_iou = float(tp_min) / float(denom)
    return float(wc_r_iou)


####################################################
# Robust Dice IoU Semantic Metric for Pixel Classification
# Ref:
####################################################

def compute_average_robust_dice_iou(
    gt_labels, pred_labels,
    pred_is_reachable=True, robust_mode="possible",
    num_classes=None, ignore_empty=True):
    """
    Compute the average Dice IoU across all classes.
    """
    assert gt_labels.shape[:2] == pred_labels.shape[:2], "Ground truth and prediction label masks must have the same height and width"
    
    num_classes = _infer_num_classes(gt_labels, pred_labels, num_classes)
            
    total_wc_dice_iou = 0.0
    count = 0
    
    for class_id in range(num_classes):

        gt_mask = _get_class_mask(gt_labels, class_id)
        if pred_is_reachable:
            pred_mask = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode=robust_mode) 
        else:
            pred_mask = _get_class_mask(pred_labels, class_id)  # treat pred_labels as hard labels for IoU computation

        if ignore_empty and (not gt_mask.any()) and (not pred_mask.any()):
            continue  # skip this class since it's empty in both GT and robust regions

        total_wc_dice_iou += robust_dice_iou_binary(gt_mask, pred_mask)
        count += 1
    
    avg_wc_dice_iou = total_wc_dice_iou / count if count > 0 else 1.0
    return float(avg_wc_dice_iou)

def robust_dice_iou_per_class(
    gt_labels, pred_labels, class_id,
    pred_is_reachable=True, robust_mode="possible"):
    """
    Compute the Robust Dice IoU for a specific class_id given the ground truth and predicted label masks.
    """
    gt_mask = _get_class_mask(gt_labels, class_id)
    
    if pred_is_reachable:
        pred_mask = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode=robust_mode) 
    else:
        pred_mask = _get_class_mask(pred_labels, class_id)  # treat pred_labels as hard labels for IoU computation

    return robust_dice_iou_binary(gt_mask, pred_mask)

def robust_dice_iou_binary(gt_mask, pred_mask):
    """
    Compute the Robust Dice IoU between two binary masks.
    The Robust Dice IoU metric evaluates the Dice coefficient between the ground truth and predicted masks under any possible perturbation of the predicted mask within the robust region. 
    It is defined as the minimum Dice coefficient over all subsets of the predicted mask that are consistent with the reachable set, which can be efficiently computed using a closed-form formula based on set operations.
    
    EqL 2*|G ∩ P| / (|G| + |P|), where G is the GT mask and P is any subset of the reachable predicted mask.
    
    Args:
    - gt_mask: 2D boolean numpy array (h, w) representing the ground truth binary mask for a specific class.
    - pred_mask: 2D boolean numpy array (h, w) representing the predicted binary mask for the same class, where True indicates pixels that are reachable to this class.
    
    Returns:
    - dice_iou: float, the computed Robust Dice IoU score for the specified class_id.
    """
    gt = gt_mask.astype(bool)
    pd = pred_mask.astype(bool)
    
    # minimum guaranteed Dice coefficient can be computed as: 2*|G ∩ P| / (|G| + |P|), 
    # where G is the GT mask and P is the reachable predicted mask; 
    # this gives a lower bound on the Dice coefficient under any subset of P that could be the actual prediction.

    intersection = np.logical_and(gt, pd).sum()
    size_gt = gt.sum()
    size_pred = pd.sum()
    
    if size_gt + size_pred == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice_iou = (2 * intersection) / (size_gt + size_pred)
    return float(dice_iou)


####################################################
# Robust Dice IoU Semantic Metric for Pixel Classification
# Ref:
####################################################

def compute_average_worst_case_robust_dice_iou(
    gt_labels, pred_labels, robust_mode="possible",
    num_classes=None, ignore_empty=True):
    """
    Compute the average Worst-Case Robust Dice IoU across all classes.
    """
    assert gt_labels.shape[:2] == pred_labels.shape[:2], "Ground truth and prediction label masks must have the same height and width"

    num_classes = _infer_num_classes(gt_labels, pred_labels, num_classes)

    total_wc_dice_iou = 0.0
    count = 0

    for class_id in range(num_classes):
        gt_mask = _get_class_mask(gt_labels, class_id)
        
        omega_max = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="possible")  # Ωmax​(c) = R(⋅,c)
        omega_min = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="unique")    # Ωmin​(c) = R(⋅,c) ∧ ∑_k R(⋅,k) = 1
        
        if ignore_empty and (not gt_mask.any()) and (not omega_min.any()) and (not omega_max.any()):
            continue  # skip this class since it's empty in both GT and robust regions
        
        total_wc_dice_iou += worst_case_robust_dice_iou_binary(gt_mask, omega_min, omega_max)
        count += 1

    avg_wc_dice_iou = total_wc_dice_iou / count if count > 0 else 1.0
    return float(avg_wc_dice_iou)

def worst_case_robust_dice_iou_per_class(gt_labels, pred_labels, class_id, robust_mode="possible"):
    """
    Compute the Worst-case Robust Dice IoU for a specific class_id given the ground truth and predicted label masks.
    """
    
    gt_mask = _get_class_mask(gt_labels, class_id)
    omega_max = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="possible")  # Ωmax​(c) = R(⋅,c)
    omega_min = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="unique")    # Ωmin​(c) = R(⋅,c) ∧ ∑_k R(⋅,k) = 1
        
    return worst_case_robust_dice_iou_binary(gt_mask, omega_min, omega_max)

def worst_case_robust_dice_iou_binary(gt_mask, omega_min_mask, omega_max_mask):
    """
    Worst-case (lower-bound) Dice score over all possible predicted masks P such that:
        Omega_min ⊆ P ⊆ Omega_max.
        
    The worst-case Dice IoU can be computed using the following closed-form formula:
    Worst-case Dice IoU = 2*|G ∩ Ωmin| / (2*|G ∩ Ωmin| + |Ωmax \ G| + |G \ Ωmin|)
    
    Args:
    - gt_mask: 2D boolean numpy array (h, w) representing the ground truth binary mask for a specific class.
    - omega_min_mask: 2D boolean numpy array (h, w) representing the minimum robust region mask for the same class, where True indicates pixels that are only reachable to this class (unique).
    - omega_max_mask: 2D boolean numpy array (h, w) representing the maximum robust region mask for the same class, where True indicates pixels that are reachable to this class (possible).
    
    Returns:
    - wc_dice_iou: float, the computed Worst-case Robust Dice IoU score for the specified class_id.
    """
    gt = gt_mask.astype(bool)
    om_min = omega_min_mask.astype(bool)
    om_max = omega_max_mask.astype(bool)
    
    # Minimum guaranteed True Positives (Intersection of GT and minimum robust region): |G ∩ Ωmin|
    tp_min = np.logical_and(gt, om_min).sum()  # True positives in the worst case (intersection with minimum robust region)
    
    # Maximum possible False Positives (non-GT pixels that are in the maximum robust region): |Ωmax \ G| = |~G ∩ Ωmax|
    fp_max = np.logical_and(~gt, om_max).sum()  # False positives in the worst case (non-GT pixels that are in the maximum robust region)
    
    # Maximum possible False Negatives (GT pixels that are not in the maximum robust region): |G \ Ωmin| = |G ∩ ~Ωmin|
    fn_max = np.logical_and(gt, ~om_min).sum()  # False negatives in the worst case (GT pixels that are not in the maximum robust region)
    
    # The worst-case Dice IoU can be computed as: 2*tp_min / (2*tp_min + fp_max + fn_max)
    denominator = (2 * tp_min + fp_max + fn_max)
    
    if denominator == 0:
        return 1.0 if tp_min == 0 else 0.0  # If both GT and robust region are empty, we consider it a perfect match with Dice=1.0. If GT is not empty but robust region has no true positives, Dice=0.0.

    wc_dice_iou = (2 * tp_min) / denominator
    return float(wc_dice_iou)

####################################################
# Boundary IoU Semantic Metric for Pixel Classification
# Ref: "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation" (https://arxiv.org/pdf/2103.16562)
####################################################

def compute_average_robust_boundary_iou(
    gt_labels, pred_labels, 
    pred_is_reachable=True,
    robust_mode="possible", 
    num_classes=None, d=None, d_ratio=0.02, 
    connectivity=8, ignore_empty=True, use_diagonal=True):
    """
    Compute the average boundary IoU across all classes.
    Args:
    - gt_labels: 2D or 3D numpy array of ground truth class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - pred_labels: 2D or 3D numpy array of predicted class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - num_classes: integer, the number of classes in the label masks.
    - robust_mode: string, either "possible" or "unique", indicating how to interpret the predicted labels.
    - d_ratio: float, the ratio of the minimum image dimension to use as the distance threshold if d is None.
    Returns:
    - average_boundary_iou: float, the average boundary IoU across all classes.
    """
    assert gt_labels.shape[:2] == pred_labels.shape[:2], "Ground truth and prediction label masks must have the same height and width"
    
    num_classes = _infer_num_classes(gt_labels, pred_labels, num_classes)
            
    total_value = 0.0
    count = 0
    
    for class_id in range(num_classes):
        gt_mask = _get_class_mask(gt_labels, class_id)
        
        if pred_is_reachable:
            pred_mask = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode=robust_mode) 
        else:
            pred_mask = _get_class_mask(pred_labels, class_id)  # treat pred_labels as hard labels for boundary IoU computation

        if ignore_empty and (not gt_mask.any()) and (not pred_mask.any()):
            continue  # skip this class since it's empty in both GT and prediction

        total_value += robust_boundary_iou_binary(
            gt_mask, pred_mask, d=d, d_ratio=d_ratio, 
            connectivity=connectivity, use_diagonal=use_diagonal)
        count += 1
        
    return total_value / count if count > 0 else 1.0

def robust_boundary_iou_per_class(
    gt_labels, pred_labels, class_id, 
    pred_is_reachable=True, robust_mode="possible",
    d=None, d_ratio=0.02, connectivity=8, use_diagonal=True):
    """
    computes the boundary IoU for a specific class_id given the ground truth and predicted label masks.
    
    Args:
    - gt_labels: 2D or 3D numpy array of ground truth class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - pred_labels: 2D or 3D numpy array of predicted class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - class_id: integer, the class ID for which to compute the boundary IoU.
    - d: int or None, (pixel distance band width) the distance threshold in pixels for considering a boundary pixel as correctly predicted. If None, it will be set to d_ration*100% of the minimum image dimension.
    - d_ratio: float, the ratio of the minimum image dimension to use as the distance threshold if d is None.
    - connectivity: int, either 4 or 8, to define the connectivity for boundary computation.
    - use_diagonal: bool, whether to use the diagonal length of the image to compute d when d is None (if False, use the minimum dimension instead).
    
    Returns:
    - boundary_iou: float, the computed boundary IoU for the specified class_id.
    """
    gt_mask = _get_class_mask(gt_labels, class_id)

    if pred_is_reachable:
        pred_mask = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode=robust_mode)
    else:
        pred_mask = _get_class_mask(pred_labels, class_id)
        
    return robust_boundary_iou_binary(gt_mask, pred_mask, d=d, d_ratio=d_ratio, 
        connectivity=connectivity, use_diagonal=use_diagonal)

def robust_boundary_iou_binary(gt_mask, pred_mask, d=None, d_ratio=0.02, connectivity=8, use_diagonal=True):
    """
    Computes the Roubst Boundary IoU between the boundaries of two binary masks.
    The boundary is defined as the pixels in the mask that have at least one neighbor outside the mask (8-connectivity).
    
    Boundary IoU between two binary masks:
    | (Gd ∩ G) ∩ (Pd ∩ P) | / | (Gd ∩ G) U (Pd ∩ P) |
    where Gd and Pd are the d-pixel boundary bands of the ground truth and predicted masks, respectively. This allows for a tolerance of d pixels when matching boundary pixels.
    
    Implementation:
      - Compute 1-pixel contour inside mask via erosion: contour = mask & ~erode(mask)
      - Build a band of width d inside the mask via Euclidean distance-to-contour
      
    Args:
        gt_mask: 2D boolean numpy array (h, w) representing the ground truth binary mask for a specific class.
        pred_mask: 2D boolean numpy array (h, w) representing the predicted binary mask for the same class.
        d: int or None, the distance threshold in pixels for considering a boundary pixel as correctly predicted. 
            If None, it will be set to d_ratio*100% of the minimum image dimension.
        d_ratio: float, the ratio of the minimum image dimension to use as the distance threshold if d is None.
        connectivity: int, either 4 or 8, to define the connectivity for boundary computation.
        use_diagonal: bool, whether to use the diagonal length of the image to compute d when d is None (if False, use the minimum dimension instead).
    Returns:
        boundary_iou: Float, the computed boundary IoU score.
    """
    assert gt_mask.shape == pred_mask.shape, "Ground truth and prediction masks must have the same shape"
    
    gt = gt_mask.astype(bool)
    pd = pred_mask.astype(bool)
    
    if connectivity == 8:
        structure = ndi.generate_binary_structure(2, 2)  # 8-connectivity
    elif connectivity == 4:
        structure = ndi.generate_binary_structure(2, 1)  # 4-connectivity
    else:
        raise ValueError("connectivity must be 4 or 8")
    
    h, w = gt.shape
    if d is None:
        if use_diagonal:
            ref_len = np.hypot(h, w)    # sqrt(H^2 + W^2)
        else:
            ref_len = min(h, w)         # min(H, W)
        d = int(max(1, round(d_ratio * ref_len)))
    
    def boundary_band_inside_mask(mask):
        """
        Compute the d-width boundary of a binary mask using binary erosion
        Return (mask_d ∩ mask) to get the boundary band that is inside the original mask, which is more robust to small misalignments and noise in the boundary.
        """
        if not mask.any():
            return np.zeros_like(mask, dtype=bool)  # If the mask is empty, return it as is (no boundary)
        
        eroded = ndi.binary_erosion(mask, structure=structure, border_value=0)
        contour = mask & (~eroded)  # Original boundary (1-pixel wide)
        
        if not contour.any():
            return np.zeros_like(mask, dtype=bool)  # If no boundary pixels, return empty mask
        
        # distance_transform_edt gives distance to nearest zero.
        # Make boundary_pixels be zero; everything else nonzero.
        dist_to_contour = ndi.distance_transform_edt(~contour)
        
        boundary_band = (dist_to_contour <= float(d)) & mask  # Include pixels within distance d of the boundary
        return boundary_band
    
    band_gt = boundary_band_inside_mask(gt) # the boundary band of the ground truth mask (pixels within d of the GT boundary); (Gd ∩ G)
    band_pred = boundary_band_inside_mask(pd) # the boundary band of the predicted mask (pixels within d of the predicted boundary); (Pd ∩ P)
    
    intersection = np.logical_and(band_gt, band_pred).sum() # (G \oplus d) \cap (P \oplus d), where \oplus d is the dilation by d pixels (boundary band); Gd ∩ Pd
    union = np.logical_or(band_gt, band_pred).sum() # (G \oplus d) \cup (P \oplus d); Gd U Pd
    
    if union == 0:
        # If both GT and prediction have no boundary pixels (e.g., both are empty), we consider it a perfect match with IoU=1.0. 
        return 1.0
    
    bound_iou = float(intersection) / float(union)
    return bound_iou


####################################################
# Worst-Case Boundary IoU Semantic Metric for Pixel Classification
# Ref: "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation" (https://arxiv.org/pdf/2103.16562)
####################################################

def compute_average_worst_case_robust_boundary_iou(
    gt_labels, pred_labels, robust_mode="possible",
    num_classes=None, ignore_empty=True):
    """
    Compute the average Worst-Case Robust Boundary IoU across all classes.
    """
    assert gt_labels.shape[:2] == pred_labels.shape[:2], "Ground truth and prediction label masks must have the same height and width"

    num_classes = _infer_num_classes(gt_labels, pred_labels, num_classes)

    total_wc_boundary_iou = 0.0
    count = 0

    for class_id in range(num_classes):
        gt_mask = _get_class_mask(gt_labels, class_id)
        
        omega_max = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="possible")  # Ωmax​(c) = R(⋅,c)
        omega_min = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="unique")    # Ωmin​(c) = R(⋅,c) ∧ ∑_k R(⋅,k) = 1
        
        if ignore_empty and (not gt_mask.any()) and (not omega_min.any()) and (not omega_max.any()):
            continue  # skip this class since it's empty in both GT and robust regions
        
        wc_boundary_iou, _ = worst_case_robust_boundary_iou_binary(gt_mask, omega_min, omega_max)
        total_wc_boundary_iou += wc_boundary_iou
        count += 1

    avg_wc_boundary_iou = total_wc_boundary_iou / count if count > 0 else 1.0
    return float(avg_wc_boundary_iou)

def worst_case_robust_boundary_iou_per_class(gt_labels, pred_labels, class_id, robust_mode="possible"):
    """
    Compute the Worst-case Robust Boundary IoU for a specific class_id given the ground truth and predicted label masks.
    """
    
    gt_mask = _get_class_mask(gt_labels, class_id)
    omega_max = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="possible")  # Ωmax​(c) = R(⋅,c)
    omega_min = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="unique")    # Ωmin​(c) = R(⋅,c) ∧ ∑_k R(⋅,k) = 1
        
    wc_boundary_iou, _ = worst_case_robust_boundary_iou_binary(gt_mask, omega_min, omega_max)
    return wc_boundary_iou

def worst_case_robust_boundary_iou_binary(
    gt_mask, omega_min_mask, omega_max_mask,
    d=None, d_ratio=0.02, connectivity=8,
    use_diagonal=True, strategy="min_of_candidates",
):
    """
    Practical worst-case proxy for Boundary IoU under the constraint:
        Omega_min ⊆ P ⊆ Omega_max.

    Candidates:
      P1 = Omega_min
          - minimal feasible prediction (tends to hurt recall / coverage)

      P2 = Omega_min ∪ (Omega_max \ G)
          - adds all possible false positives while keeping minimal TP
            (tends to hurt precision and distort boundaries)
            
    Args:
    - gt_mask: 2D boolean numpy array (h, w) representing the ground truth binary mask for a specific class.
    - omega_min_mask: 2D boolean numpy array (h, w) representing the minimum robust region mask for the same class, where True indicates pixels that are only reachable to this class (unique).
    - omega_max_mask: 2D boolean numpy array (h, w) representing the maximum robust region mask for the same class, where True indicates pixels that are reachable to this class (possible).
    - d: int or None, the distance threshold in pixels for considering a boundary pixel as correctly predicted.
    - d_ratio: float, the ratio of the minimum image dimension to use as the distance threshold if d is None.
    - connectivity: int, either 4 or 8, to define the connectivity for boundary computation.
    - use_diagonal: bool, whether to use the diagonal length of the image to compute d when d is None (if False, use the minimum dimension instead).
    - strategy: string, either "omega_min", "fp_augmented", or "min_of_candidates", indicating which candidate prediction to use for the worst-case boundary IoU computation.
         "omega_min": use P1 = Omega_min as the worst-case prediction (more conservative, focuses on recall)
         "fp_augmented": use P2 = Omega_min ∪ (Omega_max \ G) as the worst-case prediction (more aggressive, focuses on precision)
         "min_of_candidates": compute boundary IoU for both candidates and take the minimum as the worst-case proxy (more robust, but more computationally expensive)
    
    Returns:
    - wc_boundary_iou: float, the computed Worst-case Robust Boundary IoU score for the specified class_id under the given strategy.
    """
    gt = gt_mask.astype(bool)
    om_min = omega_min_mask.astype(bool)
    om_max = omega_max_mask.astype(bool)

    # Safety: enforce Omega_min ⊆ Omega_max (clamp if violated)
    om_min = np.logical_and(om_min, om_max)

    # Candidate 1: smallest feasible prediction
    P1 = om_min

    # Candidate 2: add all possible false positives (outside GT)
    P2 = np.logical_or(om_min, np.logical_and(om_max, ~gt))

    b1 = robust_boundary_iou_binary(gt, P1, d=d, d_ratio=d_ratio,
                                   connectivity=connectivity, use_diagonal=use_diagonal)
    b2 = robust_boundary_iou_binary(gt, P2, d=d, d_ratio=d_ratio,
                                   connectivity=connectivity, use_diagonal=use_diagonal)

    if strategy == "omega_min":
        return float(b1), "omega_min"
    if strategy == "fp_augmented":
        return float(b2), "fp_augmented"
    if strategy == "min_of_candidates":
        return (float(b1), "omega_min") if b1 <= b2 else (float(b2), "fp_augmented")

    raise ValueError("strategy must be 'omega_min', 'fp_augmented', or 'min_of_candidates'")

####################################################
# clDice Semantic Metric for Pixel Classification
# Ref: "clDice: a novel topology-preserving loss function for tubular structure segmentation" (https://arxiv.org/pdf/2003.07311)
####################################################

def compute_average_robust_cldice(
    gt_labels, pred_labels,
    pred_is_reachable=True, robust_mode="possible",
    num_classes=None, eps=1e-10, ignore_empty=True):
    """    
    Compute the average robust clDice across all classes.
    
    Args:
    - gt_labels: 2D or 3D numpy array of ground truth class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - pred_labels: 2D or 3D numpy array of predicted class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - pred_is_reachable: bool, whether to interpret pred_labels as reachable sets for the class_id (True) or as hard labels (False) when computing clDice.
    - robust_mode: string, either "possible" or "unique", indicating how to interpret the predicted labels if pred_is_reachable is True.
    - num_classes: integer, the number of classes in the label masks. If None, it will be inferred from the shape of gt_labels or pred_labels.
    - ignore_empty: bool, whether to ignore classes that are empty in both GT and prediction when averaging clDice.
    
    Returns:
    - average_cldice: float, the average robust clDice score across all classes.
    """
    assert gt_labels.shape[:2] == pred_labels.shape[:2], "Ground truth and prediction label masks must have the same height and width"
    
    num_classes = _infer_num_classes(gt_labels, pred_labels, num_classes)
            
    total_cldice = 0.0
    count = 0
    
    for class_id in range(num_classes):
        gt_mask = _get_class_mask(gt_labels, class_id)
        
        if pred_is_reachable:
            pred_mask = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode=robust_mode)
        else:
            pred_mask = _get_class_mask(pred_labels, class_id)
        
        if ignore_empty and (not gt_mask.any()) and (not pred_mask.any()):
            continue  # skip this class since it's empty in both GT and prediction
        
        r_cldice, _, _ = robust_cldice_binary(gt_mask, pred_mask, eps=eps)
        total_cldice += r_cldice
        count += 1
            
    return total_cldice / count if count > 0 else 1.0

def robust_cldice_per_class(
    gt_labels, pred_labels, class_id,
    pred_is_reachable=True, robust_mode="possible",
    eps=1e-10):
    """
    Computes the Robust clDice for a specific class_id given the ground truth and predicted label masks.
    
    Args:
    - gt_labels: 2D or 3D numpy array of ground truth class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - pred_labels: 2D or 3D numpy array of predicted class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - class_id: integer, the class ID for which to compute the clDice.
    - pred_is_reachable: bool, whether to interpret pred_labels as reachable sets for the class_id (True) or as hard labels (False) when computing clDice.
    - robust_mode: string, either "possible" or "unique", indicating how to interpret the predicted labels if pred_is_reachable is True.
    
    Returns:
    - r_cldice: float, the computed robust clDice score for the specified class_id.
    - t_prec: float, the precision term used in clDice computation.
    - t_sens: float, the sensitivity term used in clDice computation.
    """
    
    gt_mask = _get_class_mask(gt_labels, class_id)

    if pred_is_reachable:
        pred_mask = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode=robust_mode)
    else:
        pred_mask = _get_class_mask(pred_labels, class_id)
        
    return robust_cldice_binary(gt_mask, pred_mask, eps=eps)
    
    
def robust_cldice_binary(gt_mask, pred_mask, eps=1e-10):
    """
    Robust clDice computation for binary masks of a single class.
    This function computes the clDice score for a single class given the binary GT and predicted masks for that class. 
    It first extracts the skeletons of the GT and predicted masks, then computes 
        the topology sensitivity (t_sens) and 
        topology precision (t_prec) 
    based on the overlap of the skeletons with the masks, and 
    finally computes the robust clDice score as the harmonic mean of t_sens and t_prec.
    
    Args:
    - gt_mask: 2D boolean numpy array (h, w) representing the ground truth binary mask for a specific class.
    - pred_mask: 2D boolean numpy array (h, w) representing the predicted binary mask for the same class.
    - eps: small float to avoid division by zero in clDice computation.
    
    Returns:
    - r_cldice: float, the computed robust clDice score for the specified class_id.
    - t_prec: float, the precision term used in clDice computation.
    - t_sens: float, the sensitivity term used in clDice computation.
    """
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    # extract the skeletons using a morphological skeletonization method
    skel_gt = skeletonize(gt_mask)
    skel_pred = skeletonize(pred_mask)

    # clDice-style terms
    # t_sens = |S(G) ∩ P| / |S(G)|
    intersect_sens = np.logical_and(skel_gt, pred_mask).sum()
    len_skel_gt = skel_gt.sum()
    # topology sensitivity (t_sens) is the fraction of GT skeleton pixels that are covered by the predicted mask; 
    #   this measures how well the prediction covers the GT skeleton (sensitivity of skeleton coverage)
    t_sens = intersect_sens / (len_skel_gt + eps)

    # t_prec = |S(P) ∩ G| / |S(P)|
    intersect_prec = np.logical_and(skel_pred, gt_mask).sum()
    len_skel_pred = skel_pred.sum() 
    # topology precision (t_prec) is the fraction of predicted skeleton pixels that are covered by the GT mask; 
    #   this measures how well the GT mask covers the predicted skeleton (precision of skeleton coverage
    t_prec = intersect_prec / (len_skel_pred + eps)

    # calculate the robust clDice score (harmonic mean of t_prec and t_sens)
    r_cldice = (2 * t_prec * t_sens) / (t_prec + t_sens + eps)
    return float(r_cldice), float(t_prec), float(t_sens)


####################################################
# Worst Case clDice Semantic Metric for Pixel Classification
# Ref: "clDice: a novel topology-preserving loss function for tubular structure segmentation" (https://arxiv.org/pdf/2003.07311)
####################################################

def compute_average_worst_case_robust_cldice(
    gt_labels, pred_labels, robust_mode="possible",
    num_classes=None, eps=1e-10, ignore_empty=True):
    """    
    Compute the average worst-case robust clDice across all classes.
    
    Args:
    - gt_labels: 2D or 3D numpy array of ground truth class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - pred_labels: 2D or 3D numpy array of predicted class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - robust_mode: string, either "possible" or "unique", indicating how to interpret the predicted labels.
    - num_classes: integer, the number of classes in the label masks. If None, it will be inferred from the shape of gt_labels or pred_labels.
    - ignore_empty: bool, whether to ignore classes that are empty in both GT and prediction when averaging clDice.
    
    Returns:
    - average_cldice: float, the average worst-case robust clDice score across all classes.
    """
    assert gt_labels.shape[:2] == pred_labels.shape[:2], "Ground truth and prediction label masks must have the same height and width"
    
    num_classes = _infer_num_classes(gt_labels, pred_labels, num_classes)
            
    total_cldice = 0.0
    count = 0
    
    for class_id in range(num_classes):
        gt_mask = _get_class_mask(gt_labels, class_id)
        
        omega_max = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="possible")
        omega_min = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="unique")

        if ignore_empty and (not gt_mask.any()) and (not omega_min.any()) and (not omega_max.any()):
            continue  # skip this class since it's empty in both GT and prediction
        
        wc_r_cldice, _, _, _ = worst_case_robust_cldice_binary(gt_mask, omega_min, omega_max, eps=eps)
        total_cldice += wc_r_cldice
        count += 1
            
    return total_cldice / count if count > 0 else 1.0

def worst_case_robust_cldice_per_class(
    gt_labels, pred_labels, class_id, robust_mode="possible",
    eps=1e-10):
    """
    Computes the Worst Case Robust clDice for a specific class_id given the ground truth and predicted label masks.
    
    Args:
    - gt_labels: 2D or 3D numpy array of ground truth class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - pred_labels: 2D or 3D numpy array of predicted class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - class_id: integer, the class ID for which to compute the clDice.
    - pred_is_reachable: bool, whether to interpret pred_labels as reachable sets for the class_id (True) or as hard labels (False) when computing clDice.
    - robust_mode: string, either "possible" or "unique", indicating how to interpret the predicted labels if pred_is_reachable is True.
    
    Returns:
    - r_cldice: float, the computed robust clDice score for the specified class_id.
    - t_prec: float, the precision term used in clDice computation.
    - t_sens: float, the sensitivity term used in clDice computation.
    """
    
    gt_mask = _get_class_mask(gt_labels, class_id)

    omega_max = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="possible")
    omega_min = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode="unique")
        
    return worst_case_robust_cldice_binary(gt_mask, omega_min, omega_max, eps=eps)

def worst_case_robust_cldice_binary(gt_mask, omega_min_mask, omega_max_mask,
                                   eps=1e-10, strategy="min_of_candidates"):
    """
    Practical worst-case robust clDice proxy under Omega_min ⊆ P ⊆ Omega_max.

    strategy:
      - "omega_min":          use P = Omega_min
      - "fp_augmented":       use P = Omega_min ∪ (Omega_max \ GT)   (max FP, min TP); 
      - "min_of_candidates":  min( clDice(G, Omega_min), clDice(G, Omega_min ∪ (Omega_max \ GT)) )

    Returns:
      (wc_cldice, t_prec, t_sens, used_strategy)
    """
    gt = gt_mask.astype(bool)
    om_min = omega_min_mask.astype(bool)
    om_max = omega_max_mask.astype(bool)

    # Safety: enforce Omega_min ⊆ Omega_max
    om_min = np.logical_and(om_min, om_max)

    # Candidate 1: smallest feasible prediction
    P1 = om_min
    c1, p1, s1 = robust_cldice_binary(gt, P1) #cldice_binary(gt, P1)

    # Candidate 2: add all possible false positives while keeping minimal TP
    P2 = np.logical_or(om_min, np.logical_and(om_max, ~gt))
    c2, p2, s2 = robust_cldice_binary(gt, P2) #cldice_binary(gt, P2)

    if strategy == "omega_min":
        return c1, p1, s1, "omega_min"
    if strategy == "fp_augmented":
        return c2, p2, s2, "fp_augmented"
    if strategy == "min_of_candidates":
        if c1 <= c2:
            return c1, p1, s1, "omega_min"
        else:
            return c2, p2, s2, "fp_augmented"

    raise ValueError("strategy must be 'omega_min', 'fp_augmented', or 'min_of_candidates'")

####################################################
# Region-Level Completeness (RLC) Metrics for Pixel Classification
# Ref: 
####################################################

def compute_average_robust_region_level_completeness(
    gt_labels, pred_labels,
    pred_is_reachable=True, robust_mode="possible",
    num_classes=None, connectivity=8):
    """
    Compute the average Region-Level Completeness (RLC) across all classes.
    The RLC metric evaluates how well the predicted mask preserves the integrity of the ground truth semantic objects. 
    It identifies connected components in the GT mask and checks how well the predicted mask covers these components with certified robust pixels. 
    The RLC score is averaged across all GT objects to give an overall completeness score for the class.
    
    Args:
    - gt_labels: 2D or 3D numpy array of ground truth class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - pred_labels: 2D or 3D numpy array of predicted class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - pred_is_reachable: bool, whether to interpret pred_labels as reachable sets for the class_id (True) or as hard labels (False) when computing RLC.
    - robust_mode: string, the robust mode to use when computing the robust region from reachable sets (e.g., "possible", "certain").
    - num_classes: integer, the number of classes in the label masks. If None, it will be inferred from the shape of gt_labels or pred_labels.
    - connectivity: int, either 4 or 8, to define the connectivity for identifying connected components in the GT mask.
    
    Returns:
    - average_rlc: float, the average Region-Level Completeness (RLC) score across all classes.
    """
    assert gt_labels.shape[:2] == pred_labels.shape[:2], "Ground truth and prediction label masks must have the same height and width"
    
    num_classes = _infer_num_classes(gt_labels, pred_labels, num_classes)
            
    total_rlc = 0.0
    
    for class_id in range(num_classes):
        rlc_score = robust_region_level_completeness_per_class(
            gt_labels, pred_labels, class_id, 
            pred_is_reachable=pred_is_reachable, robust_mode=robust_mode,
            connectivity=connectivity)
        total_rlc += rlc_score
    
    average_rlc = total_rlc / num_classes if num_classes > 0 else 1.0
    return average_rlc

def robust_region_level_completeness_per_class(
    gt_labels, pred_labels, class_id, 
    pred_is_reachable=True, robust_mode="possible",
    connectivity=8, use_diagonal=True):
    """
    Compute the Region-Level Completeness (RLC) metric for a specific class_id given the ground truth and predicted label masks.
    The RLC metric evaluates how well the predicted mask preserves the integrity of the ground truth semantic objects. 
    It identifies connected components in the GT mask and checks how well the predicted mask covers these components with certified robust pixels. 
    The RLC score is averaged across all GT objects to give an overall completeness score for the class.
     
    Args:
    - gt_labels: 2D or 3D numpy array of ground truth class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - pred_labels: 2D or 3D numpy array of predicted class labels. 
        If 3D, it should be a binary mask of shape (h, w, num_classes).
        If 2D, it should contain integer class labels that will be converted to binary masks for each class.
    - class_id: integer, the class ID for which to compute the RLC.
    - pred_is_reachable: bool, whether to interpret pred_labels as reachable sets for the class_id (True) or as hard labels (False) when computing RLC.
    - robust_mode: string, the robust mode to use when computing the robust region from reachable sets (e.g., "possible", "certain").
    Returns:
    - rlc_score: float, the computed Region-Level Completeness (RLC) score for the specified class_id.
    """
    gt_mask = _get_class_mask(gt_labels, class_id)

    if pred_is_reachable:
        pred_mask = _get_robust_region_from_reachable(pred_labels, class_id, robust_mode=robust_mode)
    else:
        pred_mask = _get_class_mask(pred_labels, class_id)
        
    return robust_region_level_completeness_binary(gt_mask, pred_mask, connectivity=connectivity)
    
def robust_region_level_completeness_binary(gt_mask, pred_mask, connectivity=8):
    """
    Compute the Region-Level Completeness (RLC) metric for a single class given the binary GT and predicted masks for that class.
    The RLC metric evaluates how well the predicted mask preserves the integrity of the ground truth semantic objects. 
    It identifies connected components in the GT mask and checks how well the predicted mask covers these components with certified robust pixels. 
    The RLC score is averaged across all GT objects to give an overall completeness score for the class.
    
    Args:
    - gt_mask: 2D boolean numpy array (h, w) representing the ground truth binary mask for a specific class.
    - pred_mask: 2D boolean numpy array (h, w) representing the predicted binary mask for the same class.
    - connectivity: int, either 4 or 8, to define the connectivity for identifying connected components in the GT mask.
    
    Returns:
    - final_rlc: float, the computed Region-Level Completeness (RLC) score for the specified class_id.
    """
    assert gt_mask.shape == pred_mask.shape, "Ground truth and prediction masks must have the same shape"
    
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    
    if connectivity == 8:
        structure = ndi.generate_binary_structure(2, 2)  # 8-connectivity
    elif connectivity == 4:
        structure = ndi.generate_binary_structure(2, 1)  # 4-connectivity
    else:
        raise ValueError("connectivity must be 4 or 8")
    
    # Identify ground truth semantic objects (O_1, O_2, ..., O_n)
    # compute connected components in the GT mask to identify individual objects; 
    # gt_labeled is a labeled image where each connected component has a unique integer label, and n_objects is the total number of connected components (objects) found in the GT mask.
    gt_labeled, n_objects = ndi.label(gt_mask, structure=structure)
    
    # Edge case: No objects of this class exist in the ground truth image
    if n_objects == 0:
        return 1.0 

    rlc_sum = 0.0
    
    # Iterate over each identified ground truth object O_k
    for k in range(1, n_objects + 1):
        # Mask and area for the specific ground truth object
        ok_mask = (gt_labeled == k)
        area_ok = np.sum(ok_mask)
        
        # Find certified robust pixels that fall STRICTLY WITHIN this object O_k
        pred_in_ok = np.logical_and(pred_mask, ok_mask)
        
        # Identify the certified connected components (cc_i^k) within O_k
        pred_labeled, m_k = ndi.label(pred_in_ok, structure=structure)
        
        if m_k == 0:
            # If the adversary completely destroyed this object (no robust pixels), 
            # its contribution to the RLC sum is 0.
            continue
            
        # Efficiently calculate the area of each certified component (cc_i^k)
        # bincount returns the count of pixels for each label. 
        # We slice [1:] to drop the count for the background (label 0).
        component_areas = np.bincount(pred_labeled.ravel())[1:]
        
        # Calculate the weighted spatial integrity for this object
        total_pred_area = np.sum(component_areas)
        sum_squared_areas = np.sum(component_areas ** 2)
        
        object_rlc = sum_squared_areas / (total_pred_area * area_ok)
        
        rlc_sum += object_rlc
        
    # Average the completeness score across all n ground truth objects
    final_rlc = rlc_sum / n_objects
    return float(final_rlc)


####################################################
# Verification-Aware Semantic Metrics
####################################################

def _infer_semantic_num_classes(gt_labels, pred_labels, num_semantic_classes=None):
    if num_semantic_classes is not None:
        inferred = int(num_semantic_classes)
    elif pred_labels.ndim == 3:
        inferred = pred_labels.shape[2]
    elif gt_labels.ndim == 3:
        inferred = gt_labels.shape[2]
    else:
        inferred = int(np.max(gt_labels) + 1)

    if inferred == 1 and np.max(gt_labels) >= 1:
        inferred = 2
    return inferred


def _collapse_verification_failure_states(pred_labels, num_semantic_classes):
    labels = np.asarray(pred_labels).astype(np.int64, copy=True)
    if labels.ndim != 2:
        raise ValueError("verification-aware hard labels must be a 2D label image")

    failure_label = int(num_semantic_classes)
    labels[labels >= num_semantic_classes] = failure_label
    return labels, failure_label + 1


def _augment_reachable_with_failure_state(pred_labels, num_semantic_classes):
    reachable = np.asarray(pred_labels).astype(bool, copy=False)
    if reachable.ndim != 3:
        raise ValueError("verification-aware reachable labels must be a 3D boolean mask")
    if reachable.shape[2] < num_semantic_classes:
        raise ValueError(
            f"expected at least {num_semantic_classes} semantic channels, got {reachable.shape[2]}"
        )

    semantic = reachable[..., :num_semantic_classes]
    failure = semantic.sum(axis=-1) != 1
    aware = np.concatenate([semantic, failure[..., None]], axis=-1)
    return aware, aware.shape[2]


def compute_average_verification_aware_robust_iou(
    gt_labels, pred_labels, pred_is_reachable=True, robust_mode="possible",
    num_semantic_classes=None, ignore_empty=True):
    num_semantic_classes = _infer_semantic_num_classes(gt_labels, pred_labels, num_semantic_classes)
    if pred_is_reachable:
        aware_pred, aware_classes = _augment_reachable_with_failure_state(pred_labels, num_semantic_classes)
    else:
        aware_pred, aware_classes = _collapse_verification_failure_states(pred_labels, num_semantic_classes)
    return compute_average_robust_iou(
        gt_labels, aware_pred, pred_is_reachable=pred_is_reachable,
        robust_mode=robust_mode, num_classes=aware_classes, ignore_empty=ignore_empty)


def compute_average_verification_aware_worst_case_robust_iou(
    gt_labels, pred_labels, robust_mode="possible",
    num_semantic_classes=None, ignore_empty=True):
    num_semantic_classes = _infer_semantic_num_classes(gt_labels, pred_labels, num_semantic_classes)
    aware_pred, aware_classes = _augment_reachable_with_failure_state(pred_labels, num_semantic_classes)
    return compute_average_worst_case_robust_iou(
        gt_labels, aware_pred, robust_mode=robust_mode,
        num_classes=aware_classes, ignore_empty=ignore_empty)


def compute_average_verification_aware_robust_dice_iou(
    gt_labels, pred_labels, pred_is_reachable=True, robust_mode="possible",
    num_semantic_classes=None, ignore_empty=True):
    num_semantic_classes = _infer_semantic_num_classes(gt_labels, pred_labels, num_semantic_classes)
    if pred_is_reachable:
        aware_pred, aware_classes = _augment_reachable_with_failure_state(pred_labels, num_semantic_classes)
    else:
        aware_pred, aware_classes = _collapse_verification_failure_states(pred_labels, num_semantic_classes)
    return compute_average_robust_dice_iou(
        gt_labels, aware_pred, pred_is_reachable=pred_is_reachable,
        robust_mode=robust_mode, num_classes=aware_classes, ignore_empty=ignore_empty)


def compute_average_verification_aware_worst_case_robust_dice_iou(
    gt_labels, pred_labels, robust_mode="possible",
    num_semantic_classes=None, ignore_empty=True):
    num_semantic_classes = _infer_semantic_num_classes(gt_labels, pred_labels, num_semantic_classes)
    aware_pred, aware_classes = _augment_reachable_with_failure_state(pred_labels, num_semantic_classes)
    return compute_average_worst_case_robust_dice_iou(
        gt_labels, aware_pred, robust_mode=robust_mode,
        num_classes=aware_classes, ignore_empty=ignore_empty)


def compute_average_verification_aware_robust_boundary_iou(
    gt_labels, pred_labels, pred_is_reachable=True, robust_mode="possible",
    num_semantic_classes=None, d=None, d_ratio=0.02,
    connectivity=8, ignore_empty=True, use_diagonal=True):
    num_semantic_classes = _infer_semantic_num_classes(gt_labels, pred_labels, num_semantic_classes)
    if pred_is_reachable:
        aware_pred, aware_classes = _augment_reachable_with_failure_state(pred_labels, num_semantic_classes)
    else:
        aware_pred, aware_classes = _collapse_verification_failure_states(pred_labels, num_semantic_classes)
    return compute_average_robust_boundary_iou(
        gt_labels, aware_pred, pred_is_reachable=pred_is_reachable,
        robust_mode=robust_mode, num_classes=aware_classes, d=d, d_ratio=d_ratio,
        connectivity=connectivity, ignore_empty=ignore_empty, use_diagonal=use_diagonal)


def compute_average_verification_aware_worst_case_robust_boundary_iou(
    gt_labels, pred_labels, robust_mode="possible",
    num_semantic_classes=None, ignore_empty=True):
    num_semantic_classes = _infer_semantic_num_classes(gt_labels, pred_labels, num_semantic_classes)
    aware_pred, aware_classes = _augment_reachable_with_failure_state(pred_labels, num_semantic_classes)
    return compute_average_worst_case_robust_boundary_iou(
        gt_labels, aware_pred, robust_mode=robust_mode,
        num_classes=aware_classes, ignore_empty=ignore_empty)


def compute_average_verification_aware_robust_cldice(
    gt_labels, pred_labels, pred_is_reachable=True, robust_mode="possible",
    num_semantic_classes=None, eps=1e-10, ignore_empty=True):
    num_semantic_classes = _infer_semantic_num_classes(gt_labels, pred_labels, num_semantic_classes)
    if pred_is_reachable:
        aware_pred, aware_classes = _augment_reachable_with_failure_state(pred_labels, num_semantic_classes)
    else:
        aware_pred, aware_classes = _collapse_verification_failure_states(pred_labels, num_semantic_classes)
    return compute_average_robust_cldice(
        gt_labels, aware_pred, pred_is_reachable=pred_is_reachable,
        robust_mode=robust_mode, num_classes=aware_classes, eps=eps, ignore_empty=ignore_empty)


def compute_average_verification_aware_worst_case_robust_cldice(
    gt_labels, pred_labels, robust_mode="possible",
    num_semantic_classes=None, eps=1e-10, ignore_empty=True):
    num_semantic_classes = _infer_semantic_num_classes(gt_labels, pred_labels, num_semantic_classes)
    aware_pred, aware_classes = _augment_reachable_with_failure_state(pred_labels, num_semantic_classes)
    return compute_average_worst_case_robust_cldice(
        gt_labels, aware_pred, robust_mode=robust_mode,
        num_classes=aware_classes, eps=eps, ignore_empty=ignore_empty)


def compute_average_verification_aware_region_level_completeness(
    gt_labels, pred_labels, pred_is_reachable=True, robust_mode="possible",
    num_semantic_classes=None, connectivity=8):
    num_semantic_classes = _infer_semantic_num_classes(gt_labels, pred_labels, num_semantic_classes)
    if pred_is_reachable:
        aware_pred, aware_classes = _augment_reachable_with_failure_state(pred_labels, num_semantic_classes)
    else:
        aware_pred, aware_classes = _collapse_verification_failure_states(pred_labels, num_semantic_classes)
    return compute_average_robust_region_level_completeness(
        gt_labels, aware_pred, pred_is_reachable=pred_is_reachable,
        robust_mode=robust_mode, num_classes=aware_classes, connectivity=connectivity)
