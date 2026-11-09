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
##   This program is licensed under the BSD 3-Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
Split / Exact ProbStar Abstract Rendering & End-to-End 3D-to-Perception
Probabilistic Certification Pipeline:
  - Physical domain partitioning (`partition_physical_probstar`)
  - Split ProbStar abstract rendering (`render_split_probstar_bounds`)
  - End-to-end 3D-to-Classification certification (`certify_3d_to_vision_probstar`)
  - End-to-end 3D-to-Segmentation pixel certification (`certify_3d_to_ssnn_pixel_probstar`)
  - Monte Carlo 3D rendering + perception baseline & soundness checker (`evaluate_3d_to_vision_mc`)

Author: Hoang Dung Tran & StarV Team
"""

import time
import heapq
import numpy as np

from StarV.renderer.primitives import PhysicalProbStar
from StarV.verifier.certifier import (
    certifyProbabilisticRobustness,
    certifyProbabilisticRobustness_pixel,
)
from StarV.verifier.crown import (
    CROWNNetworkGraph,
    certifyProbabilisticRobustness_CROWN,
)


def partition_physical_probstar(phys, num_branches=4, dim_weights=None):
    """
    Partition a `PhysicalProbStar` domain `[lb0, ub0]` into `num_branches` disjoint
    sub-domains `{phys_1, ..., phys_B}` by priority bisection along the physical
    coordinate with largest weighted width `w_j * (ub[j] - lb[j]) * sqrt(p_in_branch)`.
    """
    if num_branches <= 1:
        return [phys]

    if dim_weights is None:
        # Prioritize 6-DoF SE(3) geometric camera pose axes (pitch, yaw, tx, ty, roll, tz)
        # which govern nonlinear perspective projection and Gaussian conic displacement
        base_w = np.array([3.0, 3.0, 1.5, 2.8, 2.8, 1.5, 1.0, 1.0, 0.8, 0.8], dtype=np.float64)
        if phys.n_phys <= len(base_w):
            dim_weights = base_w[:phys.n_phys]
        else:
            dim_weights = np.ones(phys.n_phys, dtype=np.float64)
            dim_weights[:len(base_w)] = base_w
    else:
        dim_weights = np.asarray(dim_weights, dtype=np.float64).reshape(phys.n_phys)

    # Priority queue ordered by -(p_input * max_weighted_width)
    pq = []
    tie = 0

    def _push(sub_phys):
        nonlocal tie
        widths = (sub_phys.ub0 - sub_phys.lb0) * dim_weights
        score = float(np.max(widths)) * max(sub_phys.p_input, 1e-6)
        heapq.heappush(pq, (-score, tie, sub_phys))
        tie += 1

    _push(phys)
    while len(pq) < num_branches:
        neg_sc, _, cur = heapq.heappop(pq)
        widths = (cur.ub0 - cur.lb0) * dim_weights
        if np.max(widths) <= 1e-14:
            _push(cur)
            break
        split_j = int(np.argmax(widths))
        mid_val = 0.5 * (cur.lb0[split_j] + cur.ub0[split_j])

        lb1, ub1 = cur.lb0.copy(), cur.ub0.copy()
        ub1[split_j] = mid_val
        lb2, ub2 = cur.lb0.copy(), cur.ub0.copy()
        lb2[split_j] = mid_val

        p1 = PhysicalProbStar(
            lb1, ub1, mu0=cur.mu0, Sig0=cur.Sig0,
            C0=cur.C0, d0=cur.d0, param_names=cur.param_names, cov_mode=cur.cov_mode,
        )
        p2 = PhysicalProbStar(
            lb2, ub2, mu0=cur.mu0, Sig0=cur.Sig0,
            C0=cur.C0, d0=cur.d0, param_names=cur.param_names, cov_mode=cur.cov_mode,
        )
        _push(p1)
        _push(p2)

    return [item[2] for item in pq]


def render_split_probstar_bounds(renderer, phys, num_branches=4, mode='probstar'):
    """
    Perform Split ProbStar Abstract Rendering across `num_branches` partitioned
    physical sub-domains. Returns aggregated pixel bounds, law-of-total-variance
    moments, runtime, and the list of rendered `ProbSparseImageStar2DCSR` sets.
    """
    t0 = time.perf_counter()
    sub_domains = partition_physical_probstar(phys, num_branches=num_branches)
    stars = []
    envs = []
    probs = []

    for sub_phys in sub_domains:
        env_b = renderer.render_envelope(sub_phys, mode=mode)
        star_b = env_b.to_probsparse_imagestar_csr(renderer.shape)
        stars.append(star_b)
        envs.append(env_b)
        probs.append(sub_phys.p_input)

    probs = np.asarray(probs, dtype=np.float64)
    tot_prob = max(float(np.sum(probs)), 1e-15)
    weights = probs / tot_prob

    lb_agg = np.min(np.stack([e.lb for e in envs], axis=0), axis=0)
    ub_agg = np.max(np.stack([e.ub for e in envs], axis=0), axis=0)

    mu_stack = np.stack([e.mu for e in envs], axis=0)
    var_stack = np.stack([e.var for e in envs], axis=0)
    mu_agg = np.sum(weights[:, None] * mu_stack, axis=0)
    # Law of total variance: E[Var(I | B)] + Var(E[I | B])
    var_agg = np.sum(
        weights[:, None] * (var_stack + np.square(mu_stack - mu_agg[None, :])),
        axis=0,
    )
    std_agg = np.sqrt(np.maximum(var_agg, 1e-16))
    render_time = time.perf_counter() - t0

    return {
        'lb': lb_agg,
        'ub': ub_agg,
        'mu': mu_agg,
        'var': var_agg,
        'std': std_agg,
        'stars': stars,
        'sub_domains': sub_domains,
        'num_branches': len(stars),
        'render_time': render_time,
        'p_input': float(np.sum(probs)),
    }


def _imagestar_memory_bytes(star_obj):
    """Compute memory footprint in bytes of a Star / ProbSparseImageStar object."""
    if hasattr(star_obj, 'nbytes'):
        return int(star_obj.nbytes())
    total = 0
    for attr in ('c', 'pred_lb', 'pred_ub', 'mu', 'Sig', 'd'):
        v = getattr(star_obj, attr, None)
        if isinstance(v, np.ndarray):
            total += v.nbytes
    for sp_attr in ('V', 'C'):
        M = getattr(star_obj, sp_attr, None)
        if M is not None:
            if hasattr(M, 'data'):
                total += M.data.nbytes
                if hasattr(M, 'indices'):
                    total += M.indices.nbytes + M.indptr.nbytes
                elif hasattr(M, 'row'):
                    total += M.row.nbytes + M.col.nbytes
            elif isinstance(M, np.ndarray):
                total += M.nbytes
    for rec in getattr(star_obj, 'env_dag', []):
        if hasattr(rec, 'nbytes'):
            total += rec.nbytes()
    return int(total)


def certify_3d_to_vision_probstar(
    renderer,
    phys,
    net,
    true_label,
    method='psim_csr',
    num_branches=1,
    alpha_iters=20,
    lp_solver='estimate',
):
    """
    End-to-End 3D-to-Perception Probabilistic Certification:
      3D PhysicalProbStar -> Abstract Renderer -> ProbSparseImageStar -> Vision Model Certification.

    Supported `method` values:
      - `'psim_csr'`: Ours (ProbStar Abstract Rendering + PSIM_CSR forward-backward vision reachability)
      - `'psim_coo'`: Ours (COO sparse format variant)
      - `'ibp_render'`: Interval/Box Abstract Rendering + vision reachability
      - `'crown_end2end'`: ProbStar Abstract Rendering + backward CROWN vision verifier
      - `'alpha_crown_end2end'`: ProbStar Abstract Rendering + alpha-CROWN vision verifier
      - `'split_probstar'`: Branch-and-Bound Split ProbStar over `num_branches` physical sub-domains
    """
    m_name = str(method).lower()

    if m_name == 'split_probstar' or num_branches > 1:
        t_start = time.perf_counter()
        sub_domains = partition_physical_probstar(phys, num_branches=max(num_branches, 2))
        p_min_sum = 0.0
        p_max_sum = 0.0
        p_hat_sum = 0.0
        p_in_sum = 0.0
        t_render_sum = 0.0
        t_vision_sum = 0.0
        max_mem_render = 0
        max_mem_vision = 0
        all_robust = True
        any_violated = False
        lb_list, ub_list = [], []
        total_preds = 0

        for sub_phys in sub_domains:
            t_r0 = time.perf_counter()
            img_star = renderer.render_probstar(sub_phys, mode='probstar', format='csr')
            t_render_sum += time.perf_counter() - t_r0
            max_mem_render = max(max_mem_render, _imagestar_memory_bytes(img_star))
            lb_list.append(img_star.render_lb)
            ub_list.append(img_star.render_ub)
            total_preds += img_star.num_pred

            t_v0 = time.perf_counter()
            res_b, out_b = certifyProbabilisticRobustness(
                net, img_star, labels=int(true_label),
                lp_solver=lp_solver, use_lp=(lp_solver != 'estimate'),
                return_output=True,
            )
            t_vision_sum += time.perf_counter() - t_v0
            max_mem_vision = max(max_mem_vision, int(res_b.get('mem_bytes', 0)))

            p_min_sum += float(res_b['p_min'])
            p_max_sum += float(res_b['p_max'])
            p_hat_sum += float(res_b['p_hat'])
            p_in_sum += float(res_b['p_input'])
            if int(res_b['rb']) != 1:
                all_robust = False
            if int(res_b['rb']) == 0:
                any_violated = False

        p_in_tot = max(p_in_sum, 1e-15)
        cond_p_min = float(np.clip(p_min_sum / p_in_tot, 0.0, 1.0))
        cond_p_max = float(np.clip(p_max_sum / p_in_tot, cond_p_min, 1.0))
        cond_p_hat = float(np.clip(p_hat_sum / p_in_tot, cond_p_min, cond_p_max))

        lb_global = np.min(np.stack(lb_list, axis=0), axis=0)
        ub_global = np.max(np.stack(ub_list, axis=0), axis=0)
        widths = ub_global - lb_global
        rb_code = 1 if (all_robust or cond_p_min >= 1.0 - 1e-9) else (0 if any_violated else 2)
        status_str = 'ROBUST' if rb_code == 1 else ('VIOLATED' if rb_code == 0 else 'UNKNOWN')

        return {
            'method': f'split_probstar_B{len(sub_domains)}',
            'status': status_str,
            'rb': rb_code,
            'p_min': float(p_min_sum),
            'p_max': float(p_max_sum),
            'p_hat': float(p_hat_sum),
            'p_input': float(p_in_sum),
            'cond_p_min': cond_p_min,
            'cond_p_max': cond_p_max,
            'cond_p_hat': cond_p_hat,
            'render_time': t_render_sum,
            'vision_time': t_vision_sum,
            'total_time': time.perf_counter() - t_start,
            'render_mem_bytes': max_mem_render,
            'vision_mem_bytes': max_mem_vision,
            'mean_pixel_width': float(np.mean(widths)),
            'max_pixel_width': float(np.max(widths)),
            'num_render_pred': int(total_preds // len(sub_domains)),
            'num_branches': len(sub_domains),
        }

    # Single-domain pipelines
    t_start = time.perf_counter()
    render_mode = 'ibp' if m_name in ('ibp', 'ibp_render', 'box') else 'probstar'
    render_fmt = 'coo' if m_name == 'psim_coo' else 'csr'

    t_r0 = time.perf_counter()
    img_star = renderer.render_probstar(phys, mode=render_mode, format=render_fmt)
    render_time = time.perf_counter() - t_r0
    render_mem = _imagestar_memory_bytes(img_star)
    widths = img_star.render_ub - img_star.render_lb

    t_v0 = time.perf_counter()
    if m_name in ('crown_end2end', 'crown', 'alpha_crown_end2end', 'alpha-crown'):
        crown_mode = 'alpha-crown' if 'alpha' in m_name else 'crown'
        res_v, out_set = certifyProbabilisticRobustness_CROWN(
            net, img_star, int(true_label),
            mode=crown_mode, alpha_iters=alpha_iters,
            lp_solver=lp_solver, return_output=True,
        )
    else:
        res_v, out_set = certifyProbabilisticRobustness(
            net, img_star, labels=int(true_label),
            lp_solver=lp_solver, use_lp=(lp_solver != 'estimate'),
            return_output=True,
        )
    vision_time = time.perf_counter() - t_v0
    total_time = time.perf_counter() - t_start

    return {
        'method': m_name,
        'status': res_v['status'],
        'rb': int(res_v['rb']),
        'p_min': float(res_v['p_min']),
        'p_max': float(res_v['p_max']),
        'p_hat': float(res_v['p_hat']),
        'p_input': float(res_v['p_input']),
        'cond_p_min': float(res_v['cond_p_min']),
        'cond_p_max': float(res_v['cond_p_max']),
        'cond_p_hat': float(res_v['cond_p_hat']),
        'render_time': render_time,
        'vision_time': vision_time,
        'total_time': total_time,
        'render_mem_bytes': render_mem,
        'vision_mem_bytes': int(res_v.get('mem_bytes', 0)),
        'mean_pixel_width': float(np.mean(widths)),
        'max_pixel_width': float(np.max(widths)),
        'num_render_pred': int(img_star.num_pred),
        'num_vision_pred': int(res_v.get('num_pred', getattr(out_set, 'num_pred', 0))),
        'num_branches': 1,
        'img_star': img_star,
        'out_set': out_set,
    }


def certify_3d_to_ssnn_pixel_probstar(
    renderer,
    phys,
    ssnn_net,
    in_image,
    num_classes=11,
    method='psim_csr',
    lp_solver='estimate',
):
    """
    End-to-End 3D-to-Segmentation Per-Pixel Probabilistic Certification:
      3D PhysicalProbStar -> Abstract Renderer -> ProbSparseImageStar -> SSNN Per-Pixel Probability Maps.
    """
    m_name = str(method).lower()
    render_mode = 'ibp' if m_name in ('ibp', 'ibp_render', 'box') else 'probstar'
    reach_method = 'crown' if 'crown' in m_name else 'approx'

    t_start = time.perf_counter()
    t_r0 = time.perf_counter()
    img_star = renderer.render_probstar(phys, mode=render_mode, format='csr')
    render_time = time.perf_counter() - t_r0
    render_mem = _imagestar_memory_bytes(img_star)
    widths = img_star.render_ub - img_star.render_lb

    t_v0 = time.perf_counter()
    res_list, _ = certifyProbabilisticRobustness_pixel(
        ssnn_net,
        in_sets=[img_star],
        in_datas=[in_image],
        num_classes=num_classes,
        reachMethod=reach_method,
        lp_solver=lp_solver,
        use_lp=(lp_solver != 'estimate'),
        return_output=False,
    )
    res_px = res_list[0] if isinstance(res_list, list) else res_list
    vision_time = time.perf_counter() - t_v0
    total_time = time.perf_counter() - t_start

    p_in = max(float(res_px['p_input']), 1e-15)
    cond_p_min_map = np.clip(res_px['p_min_map'] / p_in, 0.0, 1.0)
    cond_p_max_map = np.clip(res_px['p_max_map'] / p_in, cond_p_min_map, 1.0)
    cond_p_hat_map = np.clip(res_px['p_hat_map'] / p_in, cond_p_min_map, cond_p_max_map)
    px_labels = res_px.get('pixel_labels', res_px.get('pixel_status'))
    rb_ratio = res_px.get(
        'robust_pixel_ratio',
        float(np.mean(np.asarray(px_labels) < int(num_classes))) if px_labels is not None else 0.0,
    )

    return {
        'method': m_name,
        'render_time': render_time,
        'vision_time': vision_time,
        'total_time': total_time,
        'render_mem_bytes': render_mem,
        'vision_mem_bytes': int(res_px.get('mem_bytes', 0)),
        'mean_pixel_width': float(np.mean(widths)),
        'max_pixel_width': float(np.max(widths)),
        'num_render_pred': int(img_star.num_pred),
        'num_vision_pred': int(res_px.get('num_pred', 0)),
        'robust_pixel_pct': 100.0 * float(rb_ratio),
        'mean_cond_p_min': float(np.mean(cond_p_min_map)),
        'mean_cond_p_max': float(np.mean(cond_p_max_map)),
        'mean_cond_p_hat': float(np.mean(cond_p_hat_map)),
        'cond_p_min_map': cond_p_min_map,
        'cond_p_max_map': cond_p_max_map,
        'cond_p_hat_map': cond_p_hat_map,
        'pixel_status': px_labels,
        'gt_labels': res_px.get('gr_pix_id', res_px.get('gt_labels')),
        'img_star': img_star,
    }


def evaluate_3d_to_vision_mc(
    renderer,
    phys,
    net=None,
    true_label=None,
    env_to_check=None,
    n_samples=500,
    seed=42,
):
    """
    Monte Carlo 3D Rendering & Downstream Vision Evaluation + Soundness Verifier:
      1. Draws `n_samples` physical parameter samples alpha_0 in [lb0, ub0].
      2. Renders concrete images I_mc = renderer.evaluate_concrete(alpha_samples).
      3. If `env_to_check` (AffineEnvelopeTensor or ProbSparseImageStar2DCSR) is provided,
         verifies pointwise pixel envelope soundness:
             lb_I(alpha_0) <= I_mc(alpha_0) <= ub_I(alpha_0).
      4. If `net` is provided, evaluates `net` on all `n_samples` rendered images to
         compute empirical classification accuracy `p_mc` (or per-pixel segmentation accuracy).
    """
    t0 = time.perf_counter()
    alpha_samples = phys.sample_truncated(n_samples=n_samples, seed=seed, include_corners=True)
    imgs_mc = renderer.evaluate_concrete(alpha_samples)  # (N_s, H*W*C)

    mu_mc = np.mean(imgs_mc, axis=0)
    std_mc = np.std(imgs_mc, axis=0)
    lb_mc = np.min(imgs_mc, axis=0)
    ub_mc = np.max(imgs_mc, axis=0)

    env_violations = 0
    max_violation_mag = 0.0
    if env_to_check is not None:
        if hasattr(env_to_check, 'evaluate_bounds_at'):
            lb_env, ub_env = env_to_check.evaluate_bounds_at(alpha_samples)
            viol_low = np.maximum(0.0, lb_env - imgs_mc - 1e-7)
            viol_high = np.maximum(0.0, imgs_mc - ub_env - 1e-7)
        else:
            lb_px = getattr(env_to_check, 'render_lb', None)
            ub_px = getattr(env_to_check, 'render_ub', None)
            if lb_px is None or ub_px is None:
                lb_px, ub_px = env_to_check.estimateRanges()
            viol_low = np.maximum(0.0, lb_px[None, :] - imgs_mc - 1e-7)
            viol_high = np.maximum(0.0, imgs_mc - ub_px[None, :] - 1e-7)
        viol_any = np.any((viol_low > 0.0) | (viol_high > 0.0), axis=1)
        env_violations = int(np.sum(viol_any))
        max_violation_mag = float(max(np.max(viol_low), np.max(viol_high)))

    p_mc = None
    logits_mc = None
    if net is not None:
        graph = CROWNNetworkGraph(net, renderer.shape)
        logits_mc = graph.evaluate_flat(imgs_mc.T).T  # (N_s, out_dim)
        if true_label is not None:
            preds = np.argmax(logits_mc, axis=1)
            p_mc = float(np.mean(preds == int(true_label)))

    mc_time = time.perf_counter() - t0
    return {
        'n_samples': n_samples,
        'alpha_samples': alpha_samples,
        'imgs_mc': imgs_mc,
        'mu_mc': mu_mc,
        'std_mc': std_mc,
        'lb_mc': lb_mc,
        'ub_mc': ub_mc,
        'env_violations': env_violations,
        'max_violation_mag': max_violation_mag,
        'cond_p_mc': p_mc,
        'p_mc': (p_mc * phys.p_input) if p_mc is not None else None,
        'logits_mc': logits_mc,
        'mc_time': mc_time,
    }
