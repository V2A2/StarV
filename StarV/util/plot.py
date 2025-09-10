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
Plot module, contains methods for plotting
Dung Tran, 9/11/2022
Update: 12/24/2024 (Sung Woo Choi, merging)
"""

from StarV.set.probstar import ProbStar
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy.stats import multivariate_normal

import pypoman
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import plotly.graph_objects as go
from matplotlib.tri import Triangulation, LinearTriInterpolator


def getVertices(I):
    """Get all vertices of a star"""
    
    assert isinstance(I, ProbStar) or isinstance(I, Star), 'error: input should be a ProbStar or a Star'
    if len(I.C) == 0:
        lb = I.pred_lb
        ub = I.pred_ub
        A = np.eye(I.nVars)
        C = np.vstack((A, -A))
        d = np.concatenate([ub, -lb])
    else:
        if len(I.pred_lb) == 0:
            [lb, ub] = I.estimateRanges()
        else:
            lb = I.pred_lb
            ub = I.pred_ub
        A = np.eye(I.nVars)
        C1 = np.vstack((A, -A))
        d1 = np.concatenate([ub, -lb])
        C = np.vstack((I.C, C1))
        d = np.concatenate([I.d, d1])

    c = I.V[:,0]
    V = I.V[:,1:I.nVars+1]

    proj = (V, c)
    ineq = (C, d)
    verts = pypoman.projection.project_polytope(proj, ineq)

    return verts


def get_bounding_box(A, b):
    'get bounding box of a H-polytope Ax <= b'

    assert isinstance(A, np.ndarray), 'error: A should be a 2d array'
    assert isinstance(b, np.ndarray), 'error: b should be a 1d array'
    assert A.shape[0] == b.shape[0], 'error: inconsistency between unsafe_mat and unsafe_vec'

    n = A.shape[1]
    lb = np.zeros(n)
    ub = np.zeros(n)
    for i in range(0, n):
        c = np.zeros(n)
        c[i] = 1.0
        min_sol = pypoman.lp.solve_lp(c, A, b)
        lb[i] = min_sol[i]
        max_sol = pypoman.lp.solve_lp(-c, A, b)
        ub[i] = max_sol[i]
        
    return lb, ub


def plot_1D_Star(I, show=True, color='g'):
    """Plot a 1D star set
    Yuntao Li"""

    if isinstance(I, ProbStar) or isinstance(I, Star):
        if I.dim != 1:
            raise Exception('error: input set is not 1D star')
        [lb, ub] = I.getRanges()
        if lb == ub:
            plt.plot(lb, 0, 'o', color=color)  # Plot point if lb == ub
        else:
            plt.plot([lb, ub], [0, 0], color=color)  # Plot line otherwise
        if show:
            plt.show()
    elif isinstance(I, list) and len(I) > 1:
        for i in range(0, len(I)):
            if I[i].dim != 1:
                raise Exception('error: input set is not 1D star')
            [lb, ub] = I[i].getRanges()
            if lb == ub:
                plt.plot(lb, 0, 'o', color=color)  # Plot point if lb == ub
            else:
                plt.plot([lb, ub], [0, 0], color=color)  # Plot line otherwise
        if show:
            plt.show()
    elif isinstance(I, list) and len(I) == 1:
        if I[0].dim != 1:
            raise Exception('error: input set is not 1D star')
        [lb, ub] = I[0].getRanges()
        if lb == ub:
            plt.plot(lb, 0, 'o', color=color)  # Plot point if lb == ub
        else:
            plt.plot([lb, ub], [0, 0], color=color)  # Plot line otherwise
        if show:
            plt.show()


def plot_1D_Star_time(I, time_bound,step_size,safety_value=None,show=True, color='g'):
   
    """Plot series 1D star set over time"""

    times = np.arange(0, time_bound + 2*step_size, step_size)
    
    if isinstance(I, list) and len(I) > 1:
        for i in range(0, len(I)):
            if I[i].dim != 1:
                raise Exception('error: input set is not 1D star')
            [lb, ub] = I[i].getRanges()
            plt.plot([times[i], times[i]],[lb, ub], color=color)
            plt.xlabel("Time")
            plt.ylabel("y1")
           
        if show:
            if safety_value is not None:
                plt.axhline(y = safety_value, color = 'r', linestyle = '--',linewidth=1) 
                # plt.text(4, plt.gca().get_ylim()[1] * 0.9, 'unsafe condition', color='r', fontsize=12, verticalalignment='center')

            plt.show()


def plot_2D_Star_with_sampling(I, sample_points=None, show=True, color='g', marker='o'):
    """
    Plot a 2D Star set along with optional sampled points.

    Parameters:
    I (StarSet): The 2D Star set object.
    sample_points (np.ndarray, optional): Array of sampled points to plot, with each column as a point.
    show (bool, optional): Whether to show the plot immediately. Defaults to True.
    color (str, optional): Color for the plot elements. Defaults to 'g' (green).
    marker (str, optional): Marker style for the sampled points. Defaults to 'o' (circle).
    """

    if I.dim != 2:
        raise Exception('Input set is not 2D star')

    # Plot the polygon representing the Star set
    verts = getVertices(I)  # Assuming getVertices is a function to extract vertices
    try:
        pypoman.plot_polygon(verts, color=color)  # Plotting the vertices as a polygon
    except Exception:
        warnings.warn('Potential floating-point error during plotting')

    # If sample points are provided, plot them after transposing if necessary
    if sample_points is not None:
        sample_points = np.array(sample_points)
        if sample_points.shape[0] != 2:
            # Assuming points are stored in columns, transpose them
            sample_points = sample_points.T
        plt.scatter(sample_points.T[:, 0], sample_points.T[:, 1], color=color, marker=marker, label='Sampled Points')

    if show:
        plt.legend()
        plt.show()
        
        
def plot_2D_UnsafeSpec(unsafe_mat, unsafe_vec, show=True, color='r'):
    'plot unsafe spec'

    assert isinstance(unsafe_mat, np.ndarray), 'error: unsafe_mat should be a 2d array'
    assert isinstance(unsafe_vec, np.ndarray), 'error: unsafe_vec should be a 1d array'
    assert unsafe_mat.shape[1] == 2, 'error: unsafe_mat should have 2 row'
    assert unsafe_mat.shape[0] == unsafe_vec.shape[0], 'error: inconsistency between unsafe_mat and unsafe_vec'

    try:
        verts = pypoman.duality.compute_polytope_vertices(unsafe_mat, unsafe_vec)
        pypoman.plot_polygon(verts,color=color)
    except Exception:
        warnings.warn(message='Potential floating-point error or unbounded unsafe spec')
    if show:
        plt.show()
    

def plot_2D_Star(I, show=True, color='g'):

    if I.dim != 2:
        raise Exception('Input set is not 2D star')

    verts = getVertices(I)
    try:
        pypoman.plot_polygon(verts,color=color)
    except Exception:
        warnings.warn(message='Potential floating-point error')
    if show:
        plt.show()
    

def plot_probstar(I, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g', font_size = 15):
    """Plot a star set in a specific direction
       y = dir_mat*x + dir_vec, x in I
    """

    if isinstance(I, ProbStar):
        I1 = I.affineMap(dir_mat, dir_vec)
        
        if I1.dim > 2:
            raise Exception('error: only 2D plot is supported')
        prob = I1.estimateProbability()
        plot_2D_Star(I1, show=False, color=color)
        l, u = I1.getRanges()
        if show_prob:
            ax = plt.gca()
            ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob))
            ax.set_xlim(l[0], u[0])
            ax.set_ylim(l[1], u[1])

    elif isinstance(I, list) and len(I) > 1:
        L = []
        U = []
        for i in range(0,len(I)):
            I2 = I[i].affineMap(dir_mat, dir_vec)
            if I2.dim > 2:
                raise Exception('error: only 2D plot is supported')
            prob = I2.estimateProbability()
            if I2.isEmptySet(): # Can't plot empty set Fixed, Yuntao Li, 2/4/2024
                continue
            else:
                plot_2D_Star(I2, show=False)
            l, u = I2.getRanges()
            if i==0:
                L = l
                U = u
            else:
                if len(L) == 0: # L and U might be empty, Fixed, Yuntao Li, 2/4/2024
                    L = l
                    U = u
                else:
                    L = np.vstack((L, l))
                    U = np.vstack([U, u])
            if show_prob:
                ax = plt.gca()
                ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob))

        Lm = L.min(axis=0)
        Um = U.max(axis=0)
        ax = plt.gca()
        ax.set_xlim(Lm[0], Um[0])
        ax.set_ylim(Lm[1], Um[1])

    elif isinstance(I, list) and len(I) == 1:
        I1 = I[0].affineMap(dir_mat, dir_vec)
        
        if I1.dim > 2:
            raise Exception('error: only 2D plot is supported')
        prob = I1.estimateProbability()
        plot_2D_Star(I1, show=False)
        l, u = I1.getRanges()
        if show_prob:
            ax = plt.gca()
            ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob))
            ax.set_xlim(l[0], u[0])
            ax.set_ylim(l[1], u[1])
        
    else:
        raise Exception('error: first input should be a ProbStar or a list of ProbStar')

    plt.xlabel(label[0], fontsize=font_size)
    plt.ylabel(label[1], fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if show:
        plt.show()


def plot_probstar_signals(traces, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g', font_size = 13):
    """
    plot multiple probstar traces: traces = [T1, T2, T3, ..., Tn], Ti = k-steps trace [R1, R2, ..., Rk]

    *** NOTE *** Used this plot for probstar traces (signals) generated by reachDFS algorithm
   
    Dung Tran: 1/11/2024

    """

    assert isinstance(traces, list), 'error: reachable set should be a list'
    L = []
    U = []
    n = len(traces)
    for i in range(0, n):
        trace = traces[i]
        cpx = []
        cpy = []
        for j in range(0, len(trace)):
            rs = trace[j]
            I = rs.affineMap(dir_mat, dir_vec)
            
            if I.dim > 2:
                
                raise Exception('error: only 2D plot is supported')
            prob = I.estimateProbability()
            plot_2D_Star(I, show=False, color=color)
            l, u = I.getRanges()
            cpx.append(0.5*(l[0] + u[0]))
            cpy.append(0.5*(l[1] + u[1]))
            if i==0 and j==0:
                L = l
                U = u
            else:
                L = np.vstack((L, l))
                U = np.vstack([U, u])
            if show_prob:
                ax = plt.gca()
                ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob))

        ax = plt.gca()
        ax.plot(cpx, cpy, linewidth=1.5)

    Lm = L.min(axis=0)
    Um = U.max(axis=0)
    ax = plt.gca()
    ax.set_xlim(Lm[0], Um[0])
    ax.set_ylim(Lm[1], Um[1])

    plt.xlabel(label[0], fontsize=font_size)
    plt.ylabel(label[1], fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if show:
        plt.show()


def plot_probstar_signal(trace,  dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g', font_size = 13):
    """
    plot a single probstar trace T = [R1, R2, ..., Rk]

    *** NOTE *** Used this plot for a probstar trace generated by reachDFS algorithm

    Dung Tran: 1/11/2024
  
    """

    assert isinstance(trace, list), 'error: reachable set should be a list'
    L = []
    U = []
    n = len(trace)
    cpx = []
    cpy = []
    for i in range(0, n):
        rs = trace[i]
        I = rs.affineMap(dir_mat, dir_vec)         
        if I.dim > 2:
            raise Exception('error: only 2D plot is supported')
        prob = I.estimateProbability()
        plot_2D_Star(I, show=False, color=color)
        l, u = I.getRanges()
        cpx.append(0.5*(l[0] + u[0]))
        cpy.append(0.5*(l[1] + u[1]))
        if i==0:
            L = l
            U = u
        else:
            L = np.vstack((L, l))
            U = np.vstack([U, u])
        if show_prob:
            ax = plt.gca()
            ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob))

    ax = plt.gca()
    ax.plot(cpx, cpy, linewidth=1.5)

    Lm = L.min(axis=0)
    Um = U.max(axis=0)
    ax = plt.gca()
    ax.set_xlim(Lm[0], Um[0])
    ax.set_ylim(Lm[1], Um[1])

    plt.xlabel(label[0], fontsize=font_size)
    plt.ylabel(label[1], fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if show:
        plt.show()


def plot_SAT_trace(sat_trace, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g'):
    'plot a sat trace'

    assert isinstance(sat_trace, list), 'error: sat_trace should be a list of two items, a probstar_trace and a SAT_CDNF'

    trace = sat_trace[0]
    SAT_CDNF = sat_trace[1]
    traces = SAT_CDNF.toProbStarSignals(trace)
    print('traces = {}'.format(traces))

    plot_probstar_signals(traces, dir_mat, dir_vec, show_prob, label, show, color)


def plot_probstar_reachset(rs, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g', font_size = 13):
    """
    plot reachable set rs = [R1, R2, ...,Rk], R1 = [R11, R12, ...]
    
    *** NOTE *** Used this plot for reahable set generated by reachBFS algorithm
    """

    assert isinstance(rs, list), 'error: reachable set should be a list'
    n = len(rs)
    L = []
    U = []
    for i in range(0, n):
        rsi = rs[i]
        m = len(rsi)
        for j in range(0, m):
            rsij = rsi[j]
            I = rsij.affineMap(dir_mat, dir_vec)         
            if I.dim > 2:
                raise Exception('error: only 2D plot is supported')
            prob = I.estimateProbability()
            plot_2D_Star(I, show=False, color=color)
            l, u = I.getRanges()
            if i==0 and j==0:
                L = l
                U = u
            else:
                L = np.vstack((L, l))
                U = np.vstack([U, u])
            if show_prob:
                ax = plt.gca()
                ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob))

    Lm = L.min(axis=0)
    Um = U.max(axis=0)
    ax = plt.gca()
    ax.set_xlim(Lm[0], Um[0])
    ax.set_ylim(Lm[1], Um[1])

    plt.xlabel(label[0], fontsize=font_size)
    plt.ylabel(label[1], fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if show:
        plt.show()


def plot_probstar_reachset_with_unsafeSpec(rs, unsafe_mat, unsafe_vec, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g', font_size = 13):

    """
    plot reachable set rs = [R1, R2, ...,Rk], R1 = [R11, R12, ...] and Unsafe region U
    
    *** NOTE *** Used this plot for reahable set generated by reachBFS algorithm
    """
    assert isinstance(rs, list), 'error: reachable set should be a list'
    n = len(rs)
    L = []
    U = []
    if type(rs[0]) is list:
        for i in range(0, n):
            rsi = rs[i]
            m = len(rsi)
            for j in range(0, m):
                rsij = rsi[j]
                I = rsij.affineMap(dir_mat, dir_vec)         
                if I.dim > 2:
                    raise Exception('error: only 2D plot is supported')
                prob = I.estimateProbability()
                plot_2D_Star(I, show=False, color=color)
                l, u = I.getRanges()
                if i==0 and j==0:
                    L = l
                    U = u
                else:
                    L = np.vstack((L, l))
                    U = np.vstack([U, u])
                if show_prob:
                    ax = plt.gca()
                    ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob))

    else:

        if isinstance(rs, ProbStar):
            I1 = rs.affineMap(dir_mat, dir_vec)

            if I1.dim > 2:
                raise Exception('error: only 2D plot is supported')
            prob = I1.estimateProbability()
            plot_2D_Star(I1, show=False, color=color)
            l, u = I1.getRanges()
            L = l
            U = u
            if show_prob:
                ax = plt.gca()
                ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob))
                ax.set_xlim(l[0], u[0])
                ax.set_ylim(l[1], u[1])

        elif isinstance(rs, list) and len(rs) > 1:
            
            for i in range(0,len(rs)):
                I2 = rs[i].affineMap(dir_mat, dir_vec)
                if I2.dim > 2:
                    raise Exception('error: only 2D plot is supported')
                prob = I2.estimateProbability()
                plot_2D_Star(I2, show=False)
                l, u = I2.getRanges()
                if i==0:
                    L = l
                    U = u
                else:
                    L = np.vstack((L, l))
                    U = np.vstack([U, u])
                if show_prob:
                    ax = plt.gca()
                    ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob))

        elif isinstance(rs, list) and len(rs) == 1:
            I1 = rs[0].affineMap(dir_mat, dir_vec)

            if I1.dim > 2:
                raise Exception('error: only 2D plot is supported')
            prob = I1.estimateProbability()
            plot_2D_Star(I1, show=False)
            l, u = I1.getRanges()
            L = l
            U = u
            if show_prob:
                ax = plt.gca()
                ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob))
                ax.set_xlim(l[0], u[0])
                ax.set_ylim(l[1], u[1])

        else:
            raise Exception('error: first input should be a ProbStar or a list of ProbStar')


    plot_2D_UnsafeSpec(unsafe_mat, unsafe_vec, show=False, color='r')
    lb, ub = get_bounding_box(unsafe_mat, unsafe_vec)
    L = np.vstack((L, lb))
    U = np.vstack((U, ub))
    Lm = L.min(axis=0)
    Um = U.max(axis=0)
    ax = plt.gca()
    ax.set_xlim(Lm[0], Um[0])
    ax.set_ylim(Lm[1], Um[1])

    plt.xlabel(label[0], fontsize=font_size)
    plt.ylabel(label[1], fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if show:
        plt.show()

       
def plot_star(I, dir_mat=None, dir_vec=None, label=('$y_1$', '$y_2$'), show=True, color='g', font_size = 15):
    """Plot a star set in a specific direction
       y = dir_mat*x + dir_vec, x in I
    """

    if isinstance(I, Star) or isinstance(I, SparseStar):
        if isinstance(I, SparseStar):
            I = I.toStar()
        I1 = I.affineMap(dir_mat, dir_vec)
        if I1.dim > 2:
            raise Exception('error: only 2D plot is supported')
        plot_2D_Star(I1, show=False, color=color)
        l, u = I1.getRanges()
        ax = plt.gca()
        ax.set_xlim(l[0], u[0])
        ax.set_ylim(l[1], u[1])

    elif isinstance(I, list) and len(I) == 1:
        I1 = I[0].affineMap(dir_mat, dir_vec)
        if I1.dim > 2:
            raise Exception('error: only 2D plot is supported')
        plot_2D_Star(I1, show=False, color=color)
        l, u = I1.getRanges()
        ax = plt.gca()
        ax.set_xlim(l[0], u[0])
        ax.set_ylim(l[1], u[1])
        
    elif isinstance(I, list) and len(I) > 1:
        L = []
        U = []
        for i in range(0,len(I)):
            I2 = I[i].affineMap(dir_mat, dir_vec)
            if I2.dim > 2:
                raise Exception('error: only 2D plot is supported')
            if I2.isEmptySet(): # Can't plot empty set Fixed, Yuntao Li, 2/4/2024
                continue
            else:
                plot_2D_Star(I2, show=False)
            l, u = I2.getRanges()
            if i==0:
                L = l
                U = u
            else:
                if len(L) == 0: # L and U might be empty, Fixed, Yuntao Li, 2/4/2024
                    L = l
                    U = u
                else:
                    L = np.vstack((L, l))
                    U = np.vstack([U, u])
            
        Lm = L.min(axis=0)
        Um = U.max(axis=0)
        ax = plt.gca()
        ax.set_xlim(Lm[0], Um[0])
        ax.set_ylim(Lm[1], Um[1])
    else:
        raise Exception('error: first input should be a ProbStar or a list of ProbStar')

    plt.xlabel(label[0], fontsize=font_size)
    plt.ylabel(label[1], fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if show:
        plt.show()


def plot_multivariate_normal_distribution(mu, Sig, lb, ub, numMeshPoints=40, xlabel='X', ylabel='Y', zlabel='Probability Density', show=True):
    'plot bivariate normal distribution over variables X and Y'

    # reference: https://gist.github.com/gwgundersen/087da1ac4e2bad5daf8192b4d8f6a3cf
    # Dung Tran: 10/22/2024

    X = np.linspace(lb[0], ub[0], numMeshPoints)
    Y = np.linspace(lb[1], ub[1], numMeshPoints)
    X,Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    F = multivariate_normal(mu, Sig)
    Z = F.pdf(pos)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if show:
        plt.show()


def plot_probstar_distribution_separate_plots(I, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g', numMeshPoints=40, zlabel='Probability Density'):
    'plot a probstar and probstar distribution on a specific direction'

    # references: for distribution transformation
    # https://peterroelants.github.io/posts/multivariate-normal-primer/
    # Dung Tran: 10/22/2024

    
    if isinstance(I, ProbStar):
        I1 = I.affineMap(dir_mat, dir_vec)
        if I1.dim > 2:
            raise Exception('error: only 2D plot is supported')

        lb, ub = I1.getRanges()

        # get transformed distribution: I = c + V*alpha, alpha ~ N(mu, Sig)
        c = I1.V[:,0]
        V = I1.V[:,1:I1.nVars+1]

        new_mu = np.matmul(V, I1.mu) + c
        new_Sig = np.matmul(np.matmul(V, I1.Sig), np.transpose(V))
        xlabel=label[0]
        ylabel=label[1]

        # plot transformed distribution 
        # TODO: neet to get actual predicate bound considering C a <= d
        X = np.linspace(lb[0], ub[0], numMeshPoints)
        Y = np.linspace(lb[1], ub[1], numMeshPoints)
        X,Y = np.meshgrid(X, Y)

        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        F = multivariate_normal(new_mu, new_Sig)
        Z = F.pdf(pos)

        # Plot
        # Set the figure size
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax1= fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X, Y, Z, cmap='viridis', linewidth=0)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_zlabel(zlabel)

        
        prob = I1.estimateProbability()
        ax2 = fig.add_subplot(122)
        plot_2D_Star(I1, show=False, color=color)
        
        if show_prob:
            ax2 = plt.gca()
            ax2.text(0.5*(lb[0] + ub[0]), 0.5*(lb[1] + ub[1]), str(prob))
            ax2.set_xlim(lb[0], ub[0])
            ax2.set_ylim(lb[1], ub[1])
        ax2.set_xlabel(label[0], fontsize=13)
        ax2.set_ylabel(label[1], fontsize=13)


        if show:
            plt.show()  

    else:
        raise RuntimeError('The input I is not a probstar')
        

def plot_3D_Star_ax(I, ax, color='r',alpha=1.0, edgecolor='k', linewidth=1.0, show=True, qhull_option='Qt'):
    """
    From https://github.com/Shaddadi/veritex/blob/master/veritex/utils/plot_poly.py
    eritex/veritex/utils/plot_poly.py 

    Function to plot 3-dimensional polytope
    Parameters:
        set_vs (np.ndarray): Vertices of the set
        ax (AxesSubplot): AxesSubplot
        color (str): Face color
        alpha (float): Color transparency
        edgecolor (str): Edge color
        Linewidth (float): Line width of edges
    """
    set_vs = np.array(getVertices(I))
    hull = ConvexHull(set_vs, qhull_options=qhull_option)
    faces = hull.simplices
    for s in faces:
        sq = [
            [set_vs[s[0], 0], set_vs[s[0], 1], set_vs[s[0], 2]],
            [set_vs[s[1], 0], set_vs[s[1], 1], set_vs[s[1], 2]],
            [set_vs[s[2], 0], set_vs[s[2], 1], set_vs[s[2], 2]]
        ]
        f = a3.art3d.Poly3DCollection([sq])
        f.set_color(color)
        f.set_edgecolor(edgecolor)
        f.set_alpha(alpha)
        f.set_linewidth(linewidth)
        ax.add_collection3d(f)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
    if show:
        plt.show()


def plot_3D_Star_single(I, color='r',alpha=1.0, edgecolor='k', linewidth=1.0, show=True, qhull_option='Qt'):
    """
    From https://github.com/Shaddadi/veritex/blob/master/veritex/utils/plot_poly.py
    eritex/veritex/utils/plot_poly.py 

    Function to plot 3-dimensional polytope
    Parameters:
        set_vs (np.ndarray): Vertices of the set
        color (str): Face color
        alpha (float): Color transparency
        edgecolor (str): Edge color
        Linewidth (float): Line width of edges
        qhull_option (str): Qhull options
    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    set_vs = np.array(getVertices(I))
    hull = ConvexHull(set_vs, qhull_options=qhull_option)
    faces = hull.simplices
    for s in faces:
        sq = [
            [set_vs[s[0], 0], set_vs[s[0], 1], set_vs[s[0], 2]],
            [set_vs[s[1], 0], set_vs[s[1], 1], set_vs[s[1], 2]],
            [set_vs[s[2], 0], set_vs[s[2], 1], set_vs[s[2], 2]]
        ]
        f = a3.art3d.Poly3DCollection([sq])
        f.set_color(color)
        f.set_edgecolor(edgecolor)
        f.set_alpha(alpha)
        f.set_linewidth(linewidth)
        ax.add_collection3d(f)
        ax.set_xlabel('$y_1$')
        ax.set_ylabel('$y_2$')
        ax.set_zlabel('$y_3$')
    if show:
        plt.show()


def plot_3D_Star(I, alpha=0.3, edgecolor='k', linewidth=1.0, show=True, qhull_option='Qt', font_size=15):
    """
    From https://github.com/Shaddadi/veritex/blob/master/veritex/utils/plot_poly.py
    eritex/veritex/utils/plot_poly.py 

    Function to plot 3-dimensional polytope
    Parameters:
        set_vs (np.ndarray): Vertices of the set
        color (str): Face color
        alpha (float): Color transparency
        edgecolor (str): Edge color
        Linewidth (float): Line width of edges
        qhull_option (str): Qhull options
    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    stars = [I] if isinstance(I, Star) else I
    n_stars = len(stars)
    
    # Use Set1 colormap for more distinct colors
    colors = plt.cm.Set1(np.linspace(0, 1, n_stars))

    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')
    
    # Calculate volumes and sort
    star_volumes = []
    for i, star in enumerate(stars):
        if star.isEmptySet():
            continue
        set_vs = np.array(getVertices(star))
        hull = ConvexHull(set_vs, qhull_options=qhull_option)
        star_volumes.append((hull.volume, i))
    
    star_volumes.sort(reverse=True)
    
    # Increase alpha for smaller sets
    for i, (vol, idx) in enumerate(star_volumes):
        star = stars[idx]
        color = colors[idx]
        current_alpha = alpha + (i * 0.2)  # Increase transparency for each smaller set
        
        set_vs = np.array(getVertices(star))
        hull = ConvexHull(set_vs, qhull_options=qhull_option)
        faces = hull.simplices

        x_min = min(x_min, np.min(set_vs[:, 0]))
        x_max = max(x_max, np.max(set_vs[:, 0]))
        y_min = min(y_min, np.min(set_vs[:, 1]))
        y_max = max(y_max, np.max(set_vs[:, 1]))
        z_min = min(z_min, np.min(set_vs[:, 2]))
        z_max = max(z_max, np.max(set_vs[:, 2]))
        
        for s in faces:
            sq = [[set_vs[s[j], k] for k in range(3)] for j in range(3)]
            f = a3.art3d.Poly3DCollection([sq])
            f.set_color(color)
            f.set_edgecolor(edgecolor)
            f.set_alpha(min(current_alpha, 0.9))  # Cap alpha at 0.9
            f.set_linewidth(linewidth)
            ax.add_collection3d(f)
    
    ax.set_xlabel('$y_1$', fontsize=font_size)
    ax.set_ylabel('$y_2$', fontsize=font_size)
    ax.set_zlabel('$y_3$', fontsize=font_size)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.tick_params(labelsize=font_size)

    if show:
        plt.show()


def plot_Mesh3D_Star(I, opacity=1.0,show=True):
    vs = np.array(getVertices(I))
    hull = ConvexHull(vs)
    faces = hull.simplices
    data=go.Mesh3d(
        x = vs[:, 0],
        y = vs[:, 1],
        z = vs[:, 2],
        colorbar_title='z',
        colorscale=[[0, 'gold'],
                    [0.5, 'mediumturquoise'],
                    [1, 'magenta']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity = np.linspace(0, 1, faces.shape[0], endpoint=True),
        intensitymode='cell',
        # i, j and k give the vertices of triangles
        i = faces[:, 0],
        j = faces[:, 1],
        k = faces[:, 2],
        opacity=opacity,
        name='y',
        showscale=True
    )
    if show:
        fig = go.Figure(data=data)
        fig.show()
    else:
        return data


def plot_Mesh3D_Box(lb, ub, opacity=1.0, show=True):
    assert isinstance(lb, np.ndarray), 'error: \
    lower bound vector should be a 1D numpy array'
    assert isinstance(ub, np.ndarray), 'error: \
    upper bound vector should be a 1D numpy array'
    assert len(lb.shape) == 1, 'error: \
    lower bound vector should be a 1D numpy array'
    assert len(ub.shape) == 1, 'error: \
    upper bound vector should be a 1D numpy array'

    lb = lb.reshape(lb.shape[0], 1)
    ub = ub.reshape(ub.shape[0], 1)
    V = []
    for i in range(len(lb)):
        V.append([lb[i], ub[i]])

    vs = np.array(list(itertools.product(*V))).reshape(8,3)
    data = go.Mesh3d(
        # 8 vertices of a cube
        x=vs[:, 0],
        y=vs[:, 1],
        z=vs[:, 2],
        colorbar_title='z',
        colorscale=[[0, 'gold'],
                    [0.5, 'mediumturquoise'],
                    [1, 'magenta']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity = np.linspace(0, 1, 8, endpoint=True),
        # i, j and k give the vertices of triangles
        i=[0,0,0,0,0,0,7,7,7,7,7,7],
        j=[1,1,2,2,4,4,1,1,2,2,4,4],
        k=[3,5,3,6,5,6,3,5,3,6,5,6],
        opacity=opacity,
        name='y',
        showscale=True
    )
    if show:
        fig = go.Figure(data=data)
        fig.show()
    else:
        return data


def plot_probstar_distribution(I, dir_mat=None, dir_vec=None, show_prob=True, 
                             label=('$y_1$', '$y_2$'), show=True, color='g', 
                             numMeshPoints=100, zlabel='Probability Density', 
                             cmap='viridis', font_size=15):
    'plot a probstar distribution on a specific direction'

    # references: for distribution transformation
    # https://peterroelants.github.io/posts/multivariate-normal-primer/
    # Dung Tran: 10/22/2024
    # Updated:
    #   - Sung Woo Choi: 01/16/2025
    #   - Yuntao Li: 01/30/2025

    # TODO: neet to double check more cases

    # Apply affine transformation if provided
    I1 = I.affineMap(dir_mat, dir_vec) if (dir_mat is not None or dir_vec is not None) else I
    
    # Check dimensionality
    if I1.dim > 2:
        raise ValueError(f'Only 1D and 2D plots are supported; received ProbStar has {I1.dim} dimensions')
    
    # Create mesh grid from predicate domain
    lb, ub = I1.pred_lb, I1.pred_ub
    X = np.linspace(lb[0], ub[0], numMeshPoints)
    Y = np.linspace(lb[1], ub[1], numMeshPoints)
    X, Y = np.meshgrid(X, Y)
    
    # Create 3D array of points preserving grid structure
    # pos = np.concatenate([X[:,:,None], Y[:,:,None]], axis=2)
    pos = np.stack([X, Y], axis=2)
    
    # Check predicate constraints
    if len(I1.C) > 0 and len(I1.d) > 0:
        points_2d = pos.reshape(-1, 2)
        constraints = np.matmul(I1.C, points_2d.T)
        valid_mask = np.all(constraints <= I1.d[:, None], axis=0).reshape(X.shape)
        pos = np.where(valid_mask[:,:,None], pos, np.nan)

    # Transform points according to ProbStar definition
    c = I1.V[:, 0]  # center
    V = I1.V[:, 1:] # basis vectors
    transformed_pos = np.einsum('pk,ijk->ijp', V, pos) + c[None,None,:]
    
    # Compute distribution parameters
    new_mu = np.matmul(V, I1.mu) + c
    new_Sig = np.matmul(np.matmul(V, I1.Sig), np.transpose(V))
    
    # Create and evaluate distribution
    F = multivariate_normal(new_mu, new_Sig)
    Z = F.pdf(transformed_pos)
    
    # Create plot
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(transformed_pos[:,:,0], 
                          transformed_pos[:,:,1], 
                          Z, 
                          cmap=cmap, 
                          linewidth=0, 
                          antialiased=True)
    
    # Set labels and title
    ax.set_xlabel(label[0], fontsize=font_size)
    ax.set_ylabel(label[1], fontsize=font_size)
    ax.set_zlabel(zlabel, fontsize=font_size)
    ax.tick_params(labelsize=font_size)

    if show:
        plt.show()


def plot_probstar_contour(I, dir_mat=None, dir_vec=None, show_prob=True, 
                                label=('$y_1$', '$y_2$'), show=True, 
                                numMeshPoints=100, levels=10, cmap='viridis', font_size=15):
    """
    Plot a ProbStar or list of ProbStars using contours in one figure

    - Yuntao Li: 1/16/2025
    """
    # references: for distribution transformation
    # https://peterroelants.github.io/posts/multivariate-normal-primer/
    # Dung Tran: 10/22/2024
    # Updated:
    #   - Sung Woo Choi: 01/16/2025
    #   - Yuntao Li: 01/30/2025

    # TODO: neet to double check more cases
    fig, ax = plt.subplots()
    
    def plot_single_probstar_contour(I1):
        """Helper function to plot a single ProbStar contour"""
        if I1.dim > 2:
            raise ValueError('error: only 2D plot is supported')
        if I1.isEmptySet():
            return None, None
            
        # Create mesh grid from predicate domain
        lb, ub = I1.pred_lb, I1.pred_ub
        X = np.linspace(lb[0], ub[0], numMeshPoints)
        Y = np.linspace(lb[1], ub[1], numMeshPoints)
        X, Y = np.meshgrid(X, Y)
        pos = np.stack([X.ravel(), Y.ravel()], axis=1)
        
        # Check predicate constraints
        if I1.C.size > 0 and I1.d.size > 0:
            constraints = np.matmul(I1.C, pos.T)
            valid_mask = np.all(constraints <= I1.d[:, None], axis=0)
            pos = pos[valid_mask]
            
        # Transform points
        c = I1.V[:, 0]
        V = I1.V[:, 1:]
        transformed_pos = np.matmul(pos, V.T) + c
        
        # Compute distribution
        new_mu = np.matmul(V, I1.mu) + c
        new_Sig = np.matmul(np.matmul(V, I1.Sig), np.transpose(V))
        F = multivariate_normal(new_mu, new_Sig)
        Z = F.pdf(transformed_pos)
        
        # Plot contour
        triang = Triangulation(transformed_pos[:,0], transformed_pos[:,1])
        ax.tricontourf(triang, Z, levels=levels, cmap=cmap)
        ax.tricontour(triang, Z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        return I1.getRanges()
    
    # Handle different input types
    if isinstance(I, ProbStar):
        I1 = I.affineMap(dir_mat, dir_vec)
        l, u = plot_single_probstar_contour(I1)
        if show_prob and l is not None:
            prob = I1.estimateProbability()
            ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob), color='red')
            ax.set_xlim(l[0], u[0])
            ax.set_ylim(l[1], u[1])
            
    elif isinstance(I, list):
        L = []
        U = []
        for i, probstar in enumerate(I):
            I1 = probstar.affineMap(dir_mat, dir_vec)
            l, u = plot_single_probstar_contour(I1)
            if l is not None:  # Not an empty set
                if len(L) == 0:
                    L = l
                    U = u
                else:
                    L = np.vstack((L, l))
                    U = np.vstack([U, u])
                if show_prob:
                    prob = I1.estimateProbability()
                    ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob), color='red')
        
        if len(L) > 0:  # If we have any valid sets
            Lm = L.min(axis=0)
            Um = U.max(axis=0)
            ax.set_xlim(Lm[0], Um[0])
            ax.set_ylim(Lm[1], Um[1])
    
    else:
        raise ValueError('error: first input should be a ProbStar or a list of ProbStar')
        
    ax.set_xlabel(label[0], fontsize=font_size)
    ax.set_ylabel(label[1], fontsize=font_size)
    ax.tick_params(labelsize=font_size)

    ## Optional: Add unsafe condition line and text
    ## For example: Add vertical line at y1 = 4
    # ax.axvline(x=4, color='red', linestyle='--', linewidth=2)

    # # Add 'unsafe condition' text
    # ax.text(4.1, ax.get_ylim()[0] + 0.5*(ax.get_ylim()[1] - ax.get_ylim()[0]), 
    #         'unsafe condition', 
    #         rotation=90, 
    #         color='red',
    #         verticalalignment='center')
    
    if show:
        plt.show()

def plot_probstar_signal_contour(trace, dir_mat=None, dir_vec=None, show_prob=True, 
                                label=('$y_1$', '$y_2$'), show=True, color='g',
                                numMeshPoints=100, levels=10, cmap='viridis', font_size=15):
    """
    Plot a probstar trace as multiple contours in one figure

    - Yuntao Li: 1/16/2025
    """
    assert isinstance(trace, list), 'error: reachable set should be a list'
    
    # Create figure once
    fig, ax = plt.subplots()
    L = []
    U = []
    cpx = []
    cpy = []
    
    # Plot contour for each set in trace
    for i, rs in enumerate(trace):
        I = rs.affineMap(dir_mat, dir_vec) if (dir_mat is not None or dir_vec is not None) else rs
        
        if I.dim > 2:
            raise ValueError('error: only 2D plot is supported')
            
        # Create mesh grid from predicate domain
        lb, ub = I.pred_lb, I.pred_ub
        X = np.linspace(lb[0], ub[0], numMeshPoints)
        Y = np.linspace(lb[1], ub[1], numMeshPoints)
        X, Y = np.meshgrid(X, Y)
        pos = np.stack([X.ravel(), Y.ravel()], axis=1)
        
        # Check predicate constraints
        if I.C.size > 0 and I.d.size > 0:
            constraints = np.matmul(I.C, pos.T)
            valid_mask = np.all(constraints <= I.d[:, None], axis=0)
            pos = pos[valid_mask]
            
        # Transform points
        c = I.V[:, 0]
        V = I.V[:, 1:]
        transformed_pos = np.matmul(pos, V.T) + c
        
        # Compute distribution
        new_mu = np.matmul(V, I.mu) + c
        new_Sig = np.matmul(np.matmul(V, I.Sig), np.transpose(V))
        F = multivariate_normal(new_mu, new_Sig)
        Z = F.pdf(transformed_pos)
        
        # Plot contour
        triang = Triangulation(transformed_pos[:,0], transformed_pos[:,1])
        contour = ax.tricontourf(triang, Z, levels=levels, cmap=cmap)
        ax.tricontour(triang, Z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        # Store ranges for plot limits
        l, u = I.getRanges()
        cpx.append(0.5*(l[0] + u[0]))
        cpy.append(0.5*(l[1] + u[1]))
        if i == 0:
            L = l
            U = u
        else:
            L = np.vstack((L, l))
            U = np.vstack([U, u])
            
        # Add probability text if requested
        if show_prob:
            prob = I.estimateProbability()
            ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob), color='red', fontsize=15)
    
    # Plot center line
    ax.plot(cpx, cpy, linewidth=1.5)
    
    # Set plot limits and labels
    Lm = L.min(axis=0)
    Um = U.max(axis=0)
    ax.set_xlim(Lm[0], Um[0])
    ax.set_ylim(Lm[1], Um[1])
    ax.set_xlabel(label[0], fontsize=font_size)
    ax.set_ylabel(label[1], fontsize=font_size)
    ax.tick_params(labelsize=font_size)
    
    if show:
        plt.show()
