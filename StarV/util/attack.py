#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
##                                                                     ##
##   Primary contacts: Hoang Dung Tran <dungtran@ufl.edu> (UF)         ##
##                     Sung Woo Choi <sungwoo.choi@ufl.edu> (UF)       ##
##                     Yuntao Li <yli17@ufl.edu> (UF)                  ##
##                     Qing Liu <qliu1@ufl.edu> (UF)                   ##
##                                                                     ##
##   See CONTRIBUTORS for full author contacts and affiliations.       ##
##   This program is licensed under the BSD 3‑Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
	Apply adversarial attacks for verifying networks
	
	Sung Woo Choi, 10/20/2024
"""
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def brightening_attack(image, delta=0.05, d=240, num_max=20, dtype=np.float32):
    """
    Apply a brightening attack to an image.
    @image: input image (2D array)
    @delta: factor to increase pixel values (0 < delta <= 1)
    @d: threshold for brightening attack (0 <= d <= 255)
    @num_max: maximum number of pixels to be brightened
    @dtype: data type of the output bounds (e.g., np.float32)
    
    Returns:
        lb, ub: lower and upper bounds for each pixel in the image after the brightening attack.
         - For pixels >= d, lb is set to 0 and ub is multiplied by delta.
         - For pixels < d, lb and ub remain unchanged.
    """
    # d is for threshold for brightnening attack
    shape = image.shape
    n = np.prod(shape)
    
    flatten_img = image.ravel()

    cnt = 0
    lb = flatten_img.copy().astype(dtype)
    ub = flatten_img.copy().astype(dtype)
    for i in range(n):
        if lb[i] >= d:
            lb[i] = 0
            ub[i] *= delta
            cnt += 1
            if cnt == num_max:
                break
            
    lb = lb.reshape(shape)
    ub = ub.reshape(shape)
    return lb, ub

def UBAA_darkening_attack(image, num_max=20, delta=0.0, d=240, dtype='float32', return_type='noise'):
    """
    Unknown, bounded adversairal attack (UBAA) for darkening attack
    (i.e., only darken the pixels that are greater than a threshold d)
    
    @num_max: maximum number of pixels to be darkened
    @delta: the factor to darken the pixels (0 <= delta < 1)
    @d: the threshold for darkening attack (0 <= d <= 255)
    @dtype: data type of the image (e.g., 'float32', 'float64', etc.)
    
    Sung Woo Choi, 9/20/2025    
    """
    
    shape = image.shape
    n = np.prod(shape)
    
    flatten_img = image.astype(dtype).ravel()
    flatten_adv_im = flatten_img.copy()

    cnt = 0
    for i in range(n):
        if flatten_img[i] > d:
            flatten_adv_im[i] *= delta
            cnt += 1
            if cnt == num_max:
                break
        
    if return_type == 'noise':
        noise = flatten_adv_im - flatten_img
        noise = noise.reshape(shape)
        return noise

    elif return_type == 'bounds':
        lb = flatten_adv_im.reshape(shape)
        ub = flatten_img.reshape(shape)
        return lb, ub
    
    
def nearest_component_pixel_to_centroid(rr, cc):
    """
    Find the pixel in the component closest to the centroid (mean of rr, cc).
    Applies the squared Euclidean distance in pixel coordinates.
    Args:
    - rr, cc: arrays of row and column indices of component pixels
    Returns:
    - (row, col) of pixel closest to centroid
    """
    cr = rr.mean()
    ccen = cc.mean()
    d2 = (rr - cr) ** 2 + (cc - ccen) ** 2
    j = int(np.argmin(d2))
    return int(rr[j]), int(cc[j])


def extract_connected_components(mask, target_class=10, connectivity=8, sorted=False):
    """
    Find connected components of target_class and sort by area descending.
    Args:
    - mask: 2D array of class labels
    - target_class: class to extract (e.g., 10 for bicyclist)
    - connectivity: 4 or 8 for pixel connectivity
    - sorted: if True, sort components by area descending
    Returns:
    - labeled: 2D array with component IDs (0=background)
    - components: list of dicts with keys: id, area, center, bbox, rows, cols
    """
    target = (mask == target_class)

    # connectivity=8 -> full 3x3 neighbors, connectivity=4 -> cross neighbors
    if connectivity == 8:
        structure = ndi.generate_binary_structure(rank=2, connectivity=2)
    elif connectivity == 4:
        structure = ndi.generate_binary_structure(rank=2, connectivity=1)
    else:
        raise ValueError("connectivity must be 4 or 8")

    labeled, n_comp = ndi.label(target, structure=structure)

    components = []
    for comp_id in range(1, n_comp + 1):
        rr, cc = np.where(labeled == comp_id)
        if rr.size == 0:
            continue

        area = int(rr.size)
        center = nearest_component_pixel_to_centroid(rr, cc)

        rmin, rmax = int(rr.min()), int(rr.max())
        cmin, cmax = int(cc.min()), int(cc.max())

        components.append({
            "id": comp_id,
            "area": area,
            "center": center,          # global (row,col)
            "bbox": (rmin, rmax, cmin, cmax),
            "rows": rr,
            "cols": cc,
        })

    if sorted:
        components.sort(key=lambda x: x["area"], reverse=True)
    return labeled, components


def ring_coords_clockwise(center_r, center_c, radius):
    """
    Generate coordinates on the square ring at Chebyshev radius `radius`,
    in clockwise order.
    Args:
    - center_r, center_c: center coordinates
    - radius: Chebyshev distance from center (0 = center, 1 = 8-connected neighbors, etc.)
    Yields:
    - (row, col) coordinates of pixels in the ring, in clockwise order.
    Note:
    radius=0 returns only the center.
    """
    if radius == 0:
        yield (center_r, center_c)
        return

    r0, c0 = center_r, center_c
    top = r0 - radius
    bottom = r0 + radius
    left = c0 - radius
    right = c0 + radius

    # Top edge: left -> right
    for c in range(left, right + 1):
        yield (top, c)

    # Right edge: top+1 -> bottom
    for r in range(top + 1, bottom + 1):
        yield (r, right)

    # Bottom edge: right-1 -> left
    for c in range(right - 1, left - 1, -1):
        yield (bottom, c)

    # Left edge: bottom-1 -> top+1
    for r in range(bottom - 1, top, -1):
        yield (r, left)


def component_lidar_order(component_mask, center_local):
    """
    Compute the attack order for a single connected component by expanding 
    from the center pixel outward in clockwise square rings.
    Return                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           an ordered list of component pixels starting from center and expanding
    outward in clockwise square rings ("lidar-like").
    Args:
    - component_mask: 2D bool mask for one connected component where True indicates pixels in the component (local bbox)
    - center_local: (row, col) coordinates of the center pixel in local component_mask
    Returns:
    - ordered: list of (row, col) coordinates of component pixels in attack order
    """
    H, W = component_mask.shape
    cr, cc = center_local

    total = int(component_mask.sum())
    ordered = []
    seen = np.zeros_like(component_mask, dtype=bool)

    # Maximum radius needed to cover the local bbox
    max_radius = max(cr, H - 1 - cr, cc, W - 1 - cc)

    for radius in range(max_radius + 1):
        for r, c in ring_coords_clockwise(cr, cc, radius):
            if 0 <= r < H and 0 <= c < W:
                if component_mask[r, c] and not seen[r, c]:
                    seen[r, c] = True
                    ordered.append((r, c))
                    if len(ordered) == total:
                        return ordered

    return ordered  # fallback (should already return)


def build_lidar_attack_order(mask, target_class=10, connectivity=8):
    """
    Build a global attack order:
    - largest connected component first
    - fill from center outward in clockwise rings
    - then next largest, etc.

    Returns:
        attack_pixels: list of (row,col) in attack order
        components: component metadata (sorted by size)
    """
    labeled, components = extract_connected_components(mask, target_class, connectivity, sorted=True)

    attack_pixels = []

    for rank, comp in enumerate(components, start=1):
        rmin, rmax, cmin, cmax = comp["bbox"]

        # local bbox crop for this component
        local_labels = labeled[rmin:rmax + 1, cmin:cmax + 1]
        local_comp = (local_labels == comp["id"])

        # center in local coordinates
        gr, gc = comp["center"]
        center_local = (gr - rmin, gc - cmin)

        # If center pixel is not inside local_comp (rare), snap to nearest local component pixel
        if not local_comp[center_local]:
            rr, cc = np.where(local_comp)
            if rr.size == 0:
                continue
            # nearest to local centroid
            lr, lc = nearest_component_pixel_to_centroid(rr, cc)
            center_local = (lr, lc)

        local_order = component_lidar_order(local_comp, center_local)

        # convert local -> global coords
        global_order = [(r + rmin, c + cmin) for (r, c) in local_order]
        attack_pixels.extend(global_order)

    return attack_pixels, components


def attack_mask_from_order(mask_shape, attack_pixels, k_pixels):
    """
    Build a boolean attack mask using the first k_pixels in attack order.
    Args:
    - mask_shape: (H, W) shape of the mask
    - attack_pixels: list of (row, col) in attack order
    - k_pixels: number of pixels to include in the attack mask (use min(k_pixels, len(attack_pixels)))
    Returns:
    - A: boolean array of shape (H, W) where True indicates attacked pixels
    """
    H, W = mask_shape
    A = np.zeros((H, W), dtype=bool)
    k = min(k_pixels, len(attack_pixels))
    for r, c in attack_pixels[:k]:
        A[r, c] = True
    return A



def visualize_attack_progress(mask, target_class, components, attack_pixels, ks=(20, 50, 100, 200)):
    """
    Visualize target mask and progressive attack fill.
    Args:
    - mask: 2D array of class labels
    - target_class: class being attacked (e.g., 10 for bicyclist)
    - components: list of component metadata (from extract_connected_components)
    - attack_pixels: list of (row, col) in attack order
    - ks: list of pixel counts to visualize (e.g., [20, 50, 100, 200])
    Returns:
    - Displays matplotlib figures showing the target mask with component centers and the attack progression.
    """
    target = (mask == target_class)
    H, W = mask.shape

    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(4 * (len(ks) + 1), 4))

    # Original target mask + component centers
    ax = axes[0]
    ax.imshow(target, cmap="gray")
    for i, comp in enumerate(components, start=1):
        r, c = comp["center"]
        ax.plot(c, r, "ro", markersize=4)
        ax.text(c + 2, r + 2, f"{i}", color="yellow", fontsize=8)
    ax.set_title("Bicyclist mask + centers")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    # Progressive attack masks
    for j, k in enumerate(ks, start=1):
        ax = axes[j]
        attack_mask = attack_mask_from_order(mask.shape, attack_pixels, k)
        overlay = np.zeros((H, W, 3), dtype=float)
        overlay[target, :] = np.array([0.0, 0.7, 0.0])  # green target
        overlay[attack_mask, :] = np.array([1.0, 0.0, 0.0])  # red attack
        ax.imshow(overlay)
        ax.set_title(f"First {min(k, len(attack_pixels))} attacked pixels")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    plt.tight_layout()
    plt.show()
    
def component_boundary_coords(labeled, cid, structure=None):
    """
    Compute the boundary coordinates of a connected component with ID `cid` in the `labeled` array.
    The boundary is defined as the pixels in the component that have at least one neighbor outside the
    component, according to the specified connectivity structure. If `structure` is None, it defaults to 8-connectivity.
    Args:
    - labeled: 2D array of component IDs (0=background)
    - cid: component ID to extract boundary for
    - structure: binary array defining connectivity (e.g., 3x3 for 8-connectivity, 3x3 with cross for 4-connectivity)
    Returns:
    - (N, 2) array of (row, col) coordinates of boundary pixels for component `cid`
    """
    comp = (labeled == cid)
    eroded = ndi.binary_erosion(comp, structure=structure, border_value=0)
    boundary = comp & (~eroded)
    return np.column_stack(np.nonzero(boundary))  # (N,2) rows=[r,c]

def min_component_distances_kdtree(labeled, connectivity=8):
    """
    Compute the minimum boundary-to-boundary Euclidean distances between all pairs of connected components in the `labeled` array.
    Uses KD-trees for efficient nearest neighbor queries between component boundaries.
    KD-trees: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
        - Build a KD-tree for the boundary coordinates of each component.
        - highly optimized spatial data structure that allows for efficient nearest neighbor queries in 2D space, which is ideal for our boundary pixel coordinates.
        - For each pair of components (i, j), query the boundary points of component j against the KD-tree of component i to find the closest pair of boundary points and their distance.
    Args:
    - labeled: 2D array of component IDs (0=background)
    - connectivity: 4 or 8 for defining component boundaries
    Returns:
    - D: (K,K) array of minimum distances between component boundaries (D[i,j] = distance between component i and j)
    - comp_ids: list of component IDs corresponding to the rows/columns of D
    - closest: dict mapping (cid_i, cid_j) to (distance, (ri, ci), (rj, cj)),
    i.e., dict[(cid_i,cid_j)] = (dist, (ri,ci), (rj,cj)), where 
    (ri, ci) is the closest boundary pixel of component i to component j, and 
    (rj, cj) is the closest boundary pixel of component j to component i.
    """
    comp_ids = [cid for cid in np.unique(labeled) if cid != 0]
    K = len(comp_ids)
    D = np.full((K, K), np.inf, dtype=float)
    np.fill_diagonal(D, 0.0)

    # define boundary connectivity
    if connectivity == 8:
        structure = ndi.generate_binary_structure(2, 2)
    elif connectivity == 4:
        structure = ndi.generate_binary_structure(2, 1)
    else:
        raise ValueError("connectivity must be 4 or 8")

    boundaries = {}
    trees = {}
    for cid in comp_ids:
        coords = component_boundary_coords(labeled, cid, structure=structure) # boundary coords of component cid
        boundaries[cid] = coords
        trees[cid] = cKDTree(coords) if len(coords) > 0 else None

    closest = {}
    for i, cid_i in enumerate(comp_ids):
        for j in range(i + 1, K):
            cid_j = comp_ids[j]
            if trees[cid_i] is None or trees[cid_j] is None:
                continue

            # query boundary points of j against tree of i
            # dists: array of distances from each boundary point of j to nearest boundary point of i
            # nn_idx: array of indices of nearest boundary point in i for each boundary point in
            dists, nn_idx = trees[cid_i].query(boundaries[cid_j], k=1) # nearest neighbor in component i for each boundary point of component j
            k = int(np.argmin(dists)) # index of closest pair of boundary points between component i and j
            d = float(dists[k]) # minimum distance between component i and j
            pj = tuple(map(int, boundaries[cid_j][k]))
            pi = tuple(map(int, boundaries[cid_i][int(nn_idx[k])]))

            D[i, j] = D[j, i] = d
            closest[(cid_i, cid_j)] = (d, pi, pj)

    return D, comp_ids, closest


def build_global_cc_labeled_mask(cc_list, start_global_id=1, dtype=np.int32):
    """
    Build a single global connected-component labeled mask across multiple classes,
    and return a global-id tracker (metadata mapping).

    Args:
    - cc_list:
        List where each entry corresponds to one class:
            (labeled, components)
        - labeled: (H,W) int array with 0=background, 1..K local CC ids for that class
        - components: list of dicts with (at least) keys: 'id', 'area', 'center', 'bbox'
    - start_global_id: first global CC id to use (0 is reserved for background).
    - dtype: dtype for the output labeled mask. Use int32 or larger to avoid overflow.

    Returns:
    - cc_labeled_mask:
        (H,W) array, 0=background, 1..G global CC ids unique across all classes
    - global_id_map:
        dict mapping global_id -> metadata dict with:
            - class_id
            - local_cc_id
            - area
            - center
            - bbox
            - (optional) any other keys present in the component dict
    """
    if not isinstance(cc_list, list) or len(cc_list) == 0:
        raise ValueError("cc_list must be a non-empty list of (labeled, components) tuples")

    # validate shapes
    H, W = cc_list[0][0].shape
    for i, (labeled, _) in enumerate(cc_list):
        if labeled.shape != (H, W):
            raise ValueError(
                f"All labeled masks must have the same shape. "
                f"cc_list[0] has {(H, W)}, but cc_list[{i}] has {labeled.shape}"
            )

    cc_labeled_mask = np.zeros((H, W), dtype=dtype)
    global_id_map = {}

    gid = int(start_global_id)

    # build global CC labeled mask and metadata map
    for class_id, (labeled, components) in enumerate(cc_list):
        if components is None:
            components = []

        for comp in components:
            local_cc_id = int(comp["id"])
            region = (labeled == local_cc_id)
            if not np.any(region):
                continue

            cc_labeled_mask[region] = gid

            meta = dict(comp)  # copy all component fields
            meta.update({
                "class_id": class_id,
                "local_cc_id": local_cc_id,
                "global_id": gid,
            })
            global_id_map[gid] = meta

            gid += 1

    return cc_labeled_mask, global_id_map


def lookup_global_id(global_id_map, global_id):
    """Return metadata for a global CC id (or None if not found)."""
    return global_id_map.get(int(global_id), None)