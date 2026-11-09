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
ProbStar Abstract Set & Image-Envelope Rendering / Visualization Suite:
  1. `project_probstar_2d`: Extracts 2D polygon vertices, back-substituted
     relational envelope polygon, bounding box, and 2D Gaussian moments (mu_2d, Sig_2d)
     from any `ProbStar`, `ProbSparseImageStar2DCSR`, or `ProbSparseImageStar2DCOO`.
  2. `render_2d_probstar_contour`: Renders a 2D ProbStar projection with clipped
     truncated-Gaussian probability density heatmap, 1-sigma/2-sigma/3-sigma
     iso-density contours, polytope boundary, interval box comparison, and
     optional unsafe half-space specification.
  3. `render_3d_probstar_volume`: Renders a 3D ProbStar projection with translucent
     3D convex-hull polytope facets, 3D Gaussian covariance ellipsoid surfaces,
     and probability-density colored interior samples.
  4. `render_imagestar_envelope_panel`: Multi-panel visualization of a rendered
     `ProbSparseImageStar2DCSR` showing nominal image I_0, lower/upper envelopes
     [I_min, I_max], pixel width reduction vs. IBP, analytical mean E[I], and
     analytical standard deviation Std[I] vs. Monte Carlo.

Author: Hoang Dung Tran & StarV Team
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from StarV.set.probsparse_base import (
    backsubstitute_linear_bounds,
    verify_prob_spec,
    _MIN_VAR,
)


def _extract_c_V_moments(star_set, dir_mat=None, dims=(0, 1)):
    """
    Extract projected center `c_p` of shape (D,), generator matrix `V_p` of shape (D, nVars),
    andprojected Gaussian mean `mu_p` (D,) and covariance `Sig_p` (D, D) from a
    `ProbStar`, `ProbSparseImageStar2DCSR`, or `ProbSparseImageStar2DCOO`.
    """
    if hasattr(star_set, 'get_c_V'):
        c_full, V_full = star_set.get_c_V()
    else:
        c_full = np.asarray(star_set.V[:, 0], dtype=np.float64)
        V_full = np.asarray(star_set.V[:, 1 : star_set.nVars + 1], dtype=np.float64)

    dim = len(c_full)
    if dir_mat is None:
        idx = np.asarray(dims, dtype=np.int64).reshape(-1)
        D = len(idx)
        U = np.zeros((D, dim), dtype=np.float64)
        U[np.arange(D), idx] = 1.0
    else:
        U = np.asarray(dir_mat, dtype=np.float64)
        if U.ndim == 1:
            U = U.reshape(1, -1)
        D = U.shape[0]

    c_p = U @ c_full
    if sp.issparse(V_full):
        V_p = np.asarray(U @ V_full, dtype=np.float64)
    else:
        V_p = np.asarray(U @ V_full, dtype=np.float64)

    mu_pred = np.asarray(star_set.mu, dtype=np.float64).reshape(-1)
    Sig_pred = np.asarray(star_set.Sig, dtype=np.float64)
    mu_p = c_p + V_p @ mu_pred
    if Sig_pred.ndim == 1:
        Sig_p = (V_p * Sig_pred[None, :]) @ V_p.T
    else:
        Sig_p = V_p @ Sig_pred @ V_p.T
    Sig_p = 0.5 * (Sig_p + Sig_p.T) + _MIN_VAR * np.eye(D)

    return U, c_p, V_p, mu_p, Sig_p


def _support_polygon_2d(c_2d, V_2d, lb, ub, n_angles=96):
    """
    Compute exact 2D zonotope/star box-projection polygon vertices by evaluating
    support points over `n_angles` directions on `lb <= alpha <= ub`.
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)  # (K, 2)
    weights = dirs @ V_2d  # (K, nVars)
    alpha_opt = np.where(weights >= 0.0, ub[None, :], lb[None, :])  # (K, nVars)
    pts = c_2d[None, :] + alpha_opt @ V_2d.T  # (K, 2)
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except Exception:
        return pts


def project_probstar_2d(star_set, dir_mat=None, dims=(0, 1), n_angles=96):
    """
    Compute 2D geometric and probabilistic projection of `star_set`:
      - `verts_ext`: 2D polygon vertices of extended star `c_p + V_p @ alpha`
      - `verts_backsub`: 2D polygon vertices of sound back-substituted envelope in `alpha_0`
      - `box_lb`, `box_ub`: 2D interval bounding box
      - `mu_2d`, `Sig_2d`: 2D Gaussian mean and covariance
    """
    U, c_p, V_p, mu_2d, Sig_2d = _extract_c_V_moments(star_set, dir_mat=dir_mat, dims=dims)
    assert len(c_p) == 2, "project_probstar_2d requires a 2D projection"

    lb_pred = np.asarray(star_set.pred_lb, dtype=np.float64).reshape(-1)
    ub_pred = np.asarray(star_set.pred_ub, dtype=np.float64).reshape(-1)
    verts_ext = _support_polygon_2d(c_p, V_p, lb_pred, ub_pred, n_angles=n_angles)

    # Also compute sound back-substituted 2D polygon in root predicates alpha_0
    env_dag = getattr(star_set, 'env_dag', [])
    nVars0 = int(getattr(star_set, 'nVars0', len(lb_pred)))
    lb0 = np.asarray(getattr(star_set, 'pred_lb0', lb_pred), dtype=np.float64).reshape(-1)
    ub0 = np.asarray(getattr(star_set, 'pred_ub0', ub_pred), dtype=np.float64).reshape(-1)

    if len(env_dag) > 0:
        if hasattr(star_set, 'get_c_V'):
            c_full, V_full = star_set.get_c_V()
        else:
            c_full = np.asarray(star_set.V[:, 0], dtype=np.float64)
            V_full = np.asarray(star_set.V[:, 1 : star_set.nVars + 1], dtype=np.float64)
        thetas = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
        dirs_2d = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)  # (K, 2)
        U_dirs = dirs_2d @ U  # (K, dim)
        _, _, d_ub0, W_ub0 = backsubstitute_linear_bounds(
            c_full, V_full, env_dag, nVars0, U=U_dirs
        )
        # Support point in direction theta_k using upper bound of d_k^T y
        alpha0_opt = np.where(W_ub0 >= 0.0, ub0[None, :], lb0[None, :])
        # Map alpha0_opt through midpoint physical generator + slack
        d_lb_xy, W_lb_xy, d_ub_xy, W_ub_xy = backsubstitute_linear_bounds(
            c_full, V_full, env_dag, nVars0, U=U
        )
        c_mid0 = 0.5 * (d_lb_xy + d_ub_xy)
        W_mid0 = 0.5 * (W_lb_xy + W_ub_xy)
        verts_backsub = _support_polygon_2d(c_mid0, W_mid0, lb0, ub0, n_angles=n_angles)
    else:
        verts_backsub = verts_ext.copy()

    box_lb = np.min(verts_ext, axis=0)
    box_ub = np.max(verts_ext, axis=0)
    mu_2d = np.clip(mu_2d, box_lb, box_ub)

    return {
        'U': U,
        'c_2d': c_p,
        'V_2d': V_p,
        'verts_ext': verts_ext,
        'verts_backsub': verts_backsub,
        'box_lb': box_lb,
        'box_ub': box_ub,
        'mu_2d': mu_2d,
        'Sig_2d': Sig_2d,
    }


def render_2d_probstar_contour(
    star_set,
    dir_mat=None,
    dims=(0, 1),
    ax=None,
    unsafe_mat_2d=None,
    unsafe_vec_2d=None,
    mc_points_2d=None,
    show_box=True,
    show_contours=True,
    cmap='viridis',
    poly_edgecolor='#0d3b66',
    title=None,
    axis_labels=('$y_1$', '$y_2$'),
    grid_res=140,
):
    """
    Render a 2D ProbStar projection onto matplotlib Axes `ax`:
      - Clipped 2D truncated-Gaussian probability density heatmap inside the ProbStar polygon
      - 1-sigma, 2-sigma, 3-sigma probability density iso-contours
      - Exact Star polytope boundary vs. outer interval bounding box
      - Optional unsafe region `unsafe_mat_2d @ y <= unsafe_vec_2d` with certified [p_min, p_max], p_hat
      - Optional Monte Carlo scatter points `mc_points_2d` of shape (N_mc, 2)
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5.5, 4.8))

    proj = project_probstar_2d(star_set, dir_mat=dir_mat, dims=dims)
    verts = proj['verts_ext']
    verts_core = proj['verts_backsub']
    lb, ub = proj['box_lb'], proj['box_ub']
    mu = proj['mu_2d']
    Sig = proj['Sig_2d']

    span = np.maximum(ub - lb, 1e-3)
    pad = 0.14 * span
    x_min, x_max = lb[0] - pad[0], ub[0] + pad[0]
    y_min, y_max = lb[1] - pad[1], ub[1] + pad[1]

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    rv = multivariate_normal(mean=mu, cov=Sig, allow_singular=True)
    pdf_vals = rv.pdf(grid_pts).reshape(grid_res, grid_res)

    # Mask PDF strictly inside the ProbStar polygon
    if len(verts) >= 3:
        poly_path = MplPath(verts)
        inside_mask = poly_path.contains_points(grid_pts, radius=1e-6).reshape(grid_res, grid_res)
        pdf_masked = np.ma.masked_where(~inside_mask, pdf_vals)
    else:
        pdf_masked = pdf_vals

    # 1. Interval Box Over-Approximation (dashed gray)
    if show_box:
        box_verts = np.array([
            [lb[0], lb[1]],
            [ub[0], lb[1]],
            [ub[0], ub[1]],
            [lb[0], ub[1]],
        ])
        box_patch = MplPolygon(
            box_verts, closed=True, fill=False,
            edgecolor='#888888', linestyle='--', linewidth=1.3,
            label='Interval Box $[l, u]$', zorder=2,
        )
        ax.add_patch(box_patch)

    # 2. Clipped Probability Density Heatmap inside ProbStar Polytope
    im = ax.pcolormesh(
        xx, yy, pdf_masked, cmap=cmap, shading='auto', alpha=0.82, zorder=3
    )

    # 3. Iso-density 1-sigma, 2-sigma, 3-sigma contours
    if show_contours:
        peak_pdf = float(rv.pdf(mu))
        levels = sorted([
            peak_pdf * np.exp(-0.5 * (k ** 2))
            for k in (2.5, 1.75, 1.0)
        ])
        if levels[-1] > levels[0] + 1e-12:
            ax.contour(
                xx, yy, pdf_masked, levels=levels,
                colors='white', linewidths=0.9, alpha=0.85, zorder=4,
            )

    # 4. ProbStar Polytope Boundary & Core Affine-Physical Polygon
    if len(verts) >= 3:
        poly_patch = MplPolygon(
            verts, closed=True, fill=False,
            edgecolor=poly_edgecolor, linewidth=2.2,
            label='ProbStar Envelope $\\mathcal{P}$', zorder=5,
        )
        ax.add_patch(poly_patch)

    if len(verts_core) >= 3 and np.max(np.abs(verts_core.mean(axis=0) - verts.mean(axis=0))) < 1e3:
        core_patch = MplPolygon(
            verts_core, closed=True, fill=False,
            edgecolor='#00a896', linestyle='-.', linewidth=1.4,
            label='Physical Affine Core $V_{\\mathrm{phys}}\\alpha_0$', zorder=5,
        )
        ax.add_patch(core_patch)

    # 5. Optional Monte Carlo points
    if mc_points_2d is not None:
        pts_mc = np.asarray(mc_points_2d, dtype=np.float64)
        ax.scatter(
            pts_mc[:, 0], pts_mc[:, 1],
            s=8, c='#f4a261', edgecolors='none', alpha=0.55,
            label='MC Samples', zorder=6,
        )

    # 6. Mean marker
    ax.plot(
        mu[0], mu[1], marker='*', color='#e63946', markersize=11,
        markeredgecolor='white', markeredgewidth=0.7,
        label='Analytic Mean $\\mu$', zorder=7,
    )

    # 7. Optional 2D unsafe half-space specification `u_2d @ y <= c_2d`
    if unsafe_mat_2d is not None and unsafe_vec_2d is not None:
        u2 = np.asarray(unsafe_mat_2d, dtype=np.float64).reshape(-1, 2)
        v2 = np.asarray(unsafe_vec_2d, dtype=np.float64).reshape(-1)
        unsafe_grid = np.all((grid_pts @ u2.T) <= v2[None, :], axis=1).reshape(grid_res, grid_res)
        ax.contourf(
            xx, yy, unsafe_grid.astype(np.float64),
            levels=[0.5, 1.5], colors=['#e63946'], alpha=0.22, zorder=4,
        )
        ax.contour(
            xx, yy, (grid_pts @ u2[0] - v2[0]).reshape(grid_res, grid_res),
            levels=[0.0], colors=['#d62828'], linewidths=2.0, linestyles='--', zorder=6,
        )
        # Verify full specification Probability via verify_prob_spec
        U_full = u2 @ proj['U']
        spec_res = verify_prob_spec(star_set, U_full, v2, lp_solver='estimate', use_lp_tightening=False)
        txt = (
            f"$p_{{\\min}}={spec_res['cond_p_min']:.3f},\\; "
            f"\\hat{{p}}={spec_res['cond_p_hat']:.3f},\\; "
            f"p_{{\\max}}={spec_res['cond_p_max']:.3f}$"
        )
        ax.text(
            0.03, 0.96, txt, transform=ax.transAxes,
            fontsize=8.5, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#d62828', alpha=0.9),
            zorder=8,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(axis_labels[0], fontsize=9.5)
    ax.set_ylabel(axis_labels[1], fontsize=9.5)
    if title is not None:
        ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.35)
    return ax, im, proj


def render_3d_probstar_volume(
    star_set,
    dir_mat=None,
    dims=(0, 1, 2),
    ax=None,
    mc_points_3d=None,
    title=None,
    axis_labels=('$y_1$', '$y_2$', '$y_3$'),
):
    """
    Render a 3D ProbStar projection onto a 3D Axes `ax`:
      - Translucent 3D convex hull facets of the ProbStar reachable set
      - 1-sigma and 2-sigma 3D Gaussian covariance ellipsoid wireframes
      - Interior probability-density colored sample points
    """
    if ax is None:
        fig = plt.figure(figsize=(6.0, 5.2))
        ax = fig.add_subplot(111, projection='3d')

    U, c_3d, V_3d, mu_3d, Sig_3d = _extract_c_V_moments(star_set, dir_mat=dir_mat, dims=dims)
    assert len(c_3d) == 3, "render_3d_probstar_volume requires a 3D projection"

    lb_pred = np.asarray(star_set.pred_lb, dtype=np.float64).reshape(-1)
    ub_pred = np.asarray(star_set.pred_ub, dtype=np.float64).reshape(-1)

    # Sample golden-spiral directions on S^2 for 3D support hull vertices
    n_dirs = 120
    idx_s = np.arange(n_dirs, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx_s / n_dirs)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * idx_s
    dirs_3d = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)

    weights = dirs_3d @ V_3d
    alpha_opt = np.where(weights >= 0.0, ub_pred[None, :], lb_pred[None, :])
    pts_3d = c_3d[None, :] + alpha_opt @ V_3d.T

    try:
        hull = ConvexHull(pts_3d)
        facets = [pts_3d[simplex] for simplex in hull.simplices]
        poly3d = Poly3DCollection(
            facets, alpha=0.18, facecolor='#2a9d8f', edgecolor='#264653', linewidths=0.45
        )
        ax.add_collection3d(poly3d)
    except Exception:
        ax.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], s=10, c='#2a9d8f', alpha=0.4)

    # Render 1.5-sigma 3D Gaussian covariance ellipsoid wireframe
    eigvals, eigvecs = np.linalg.eigh(Sig_3d)
    eigvals = np.maximum(eigvals, _MIN_VAR)
    L_ell = eigvecs @ np.diag(np.sqrt(eigvals))

    u_ang = np.linspace(0.0, 2.0 * np.pi, 24)
    v_ang = np.linspace(0.0, np.pi, 14)
    sx = np.outer(np.cos(u_ang), np.sin(v_ang))
    sy = np.outer(np.sin(u_ang), np.sin(v_ang))
    sz = np.outer(np.ones_like(u_ang), np.cos(v_ang))
    sphere = np.stack([sx.ravel(), sy.ravel(), sz.ravel()], axis=0)  # (3, M)

    for scale_sig, col_ell, al_ell in [(1.0, '#e76f51', 0.45), (2.0, '#f4a261', 0.25)]:
        ell_pts = (mu_3d[:, None] + scale_sig * (L_ell @ sphere)).reshape(3, len(u_ang), len(v_ang))
        # Clip ellipsoid coordinates to bounding box of polytope
        b_lb = np.min(pts_3d, axis=0)
        b_ub = np.max(pts_3d, axis=0)
        for d_i in range(3):
            ell_pts[d_i] = np.clip(ell_pts[d_i], b_lb[d_i], b_ub[d_i])
        ax.plot_wireframe(
            ell_pts[0], ell_pts[1], ell_pts[2],
            color=col_ell, linewidth=0.65, alpha=al_ell,
        )

    if mc_points_3d is not None:
        pts_mc = np.asarray(mc_points_3d, dtype=np.float64)
        rv3 = multivariate_normal(mean=mu_3d, cov=Sig_3d, allow_singular=True)
        dens = rv3.pdf(pts_mc)
        ax.scatter(
            pts_mc[:, 0], pts_mc[:, 1], pts_mc[:, 2],
            c=dens, cmap='plasma', s=9, alpha=0.65,
        )

    ax.scatter([mu_3d[0]], [mu_3d[1]], [mu_3d[2]], marker='*', s=90, c='#d62828', label='Mean $\\mu$')
    ax.set_xlabel(axis_labels[0], fontsize=8.5)
    ax.set_ylabel(axis_labels[1], fontsize=8.5)
    ax.set_zlabel(axis_labels[2], fontsize=8.5)
    if title is not None:
        ax.set_title(title, fontsize=9.5, fontweight='bold')
    return ax


def render_imagestar_envelope_panel(
    img_star,
    ibp_star=None,
    mc_res=None,
    nominal_img=None,
    save_path=None,
    title_prefix="ProbStar Abstract Rendering",
):
    """
    Create a 6-panel visualization of a rendered `ProbSparseImageStar2DCSR`:
      1. Nominal Rendered Image I(mu_0)
      2. Sound Lower Envelope Image I_min
      3. Sound Upper Envelope Image I_max
      4. Pixel Bound Width: Ours (I_max - I_min)
      5. Analytical Mean Image E[I]
      6. Analytical Pixel Std[I] (or IBP Width comparison)
    """
    H, W, C = img_star.shape
    lb_px = getattr(img_star, 'render_lb', None)
    ub_px = getattr(img_star, 'render_ub', None)
    if lb_px is None or ub_px is None:
        lb_px, ub_px = img_star.estimateRanges()
    mu_px = getattr(img_star, 'render_mu', img_star.c)
    std_px = getattr(img_star, 'render_std', np.sqrt(np.maximum(ub_px - lb_px, 0.0)))

    def _to_vis(flat_arr):
        img = np.asarray(flat_arr, dtype=np.float64).reshape(H, W, C)
        return img[:, :, 0] if C == 1 else np.clip(img, 0.0, 1.0)

    cmap_img = 'gray' if C == 1 else None
    nom_vis = _to_vis(nominal_img if nominal_img is not None else mu_px)
    lb_vis = _to_vis(lb_px)
    ub_vis = _to_vis(ub_px)
    mu_vis = _to_vis(mu_px)
    width_ours = np.mean((ub_px - lb_px).reshape(H, W, C), axis=2)
    std_vis = np.mean(std_px.reshape(H, W, C), axis=2)

    fig, axes = plt.subplots(1, 6, figsize=(16.5, 2.9))
    panels = [
        (nom_vis, f"{title_prefix}\nNominal $I(\\mu_0)$", cmap_img, (0.0, 1.0)),
        (lb_vis, "Sound Lower\nEnvelope $I_{\\min}$", cmap_img, (0.0, 1.0)),
        (ub_vis, "Sound Upper\nEnvelope $I_{\\max}$", cmap_img, (0.0, 1.0)),
        (width_ours, f"Ours Bound Width\n$\\mathbb{{E}}[\\Delta I]={np.mean(width_ours):.3f}$", 'magma', None),
        (mu_vis, "Analytic Moment\nMean $\\mathbb{E}[I(\\alpha_0)]$", cmap_img, (0.0, 1.0)),
        (std_vis, f"Analytic Moment\nStd $\\sigma_I$ (max={np.max(std_vis):.3f})", 'viridis', None),
    ]

    for ax, (data, ttl, cm_name, vlim) in zip(axes, panels):
        if vlim is not None and cm_name is not None:
            im = ax.imshow(data, cmap=cm_name, vmin=vlim[0], vmax=vlim[1])
        elif cm_name is not None:
            im = ax.imshow(data, cmap=cm_name)
        else:
            im = ax.imshow(data)
        ax.set_title(ttl, fontsize=9, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
    return fig
