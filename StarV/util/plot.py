"""
Plot module, contains methods for plotting
Dung Tran, 9/11/2022
"""

from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import pypoman
import warnings


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
    

def plot_probstar(I, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g'):
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

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if show:
        plt.show()


def plot_probstar_signals(traces, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g'):
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

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if show:
        plt.show()

   


def plot_probstar_signal(trace,  dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g'):
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

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
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

def plot_probstar_reachset(rs, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g'):
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

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if show:
        plt.show()


def plot_probstar_reachset_with_unsafeSpec(rs, unsafe_mat, unsafe_vec, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g'):

    """
    plot reachable set rs = [R1, R2, ...,Rk], R1 = [R11, R12, ...] and Unsafe region U
    
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

    plot_2D_UnsafeSpec(unsafe_mat, unsafe_vec, show=False, color='r')
    lb, ub = get_bounding_box(unsafe_mat, unsafe_vec)
    L = np.vstack((L, lb))
    U = np.vstack((L, ub))
    Lm = L.min(axis=0)
    Um = U.max(axis=0)
    ax = plt.gca()
    ax.set_xlim(Lm[0], Um[0])
    ax.set_ylim(Lm[1], Um[1])

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if show:
        plt.show()

       
    


def plot_star(I, dir_mat=None, dir_vec=None, label=('$y_1$', '$y_2$'), show=True, color='g'):
    """Plot a star set in a specific direction
       y = dir_mat*x + dir_vec, x in I
    """

    if isinstance(I, Star):
        I1 = I.affineMap(dir_mat, dir_vec)
        if I1.dim > 2:
            raise Exception('error: only 2D plot is supported')
        plot_2D_Star(I, show=False, color=color)
        l, u = I1.getRanges()
        
    elif isinstance(I, list):
        L = []
        U = []
        for i in range(0,len(I)):
            I2 = I[i].affineMap(dir_mat, dir_vec)
            if I2.dim > 2:
                raise Exception('error: only 2D plot is supported')
            plot_2D_Star(I2, show=False)
            l, u = I2.getRanges()
            if i==0:
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

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if show:
        plt.show()

def plot_probstar_2D_distribution(I, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g'):
    """Plot distribution of a probstar set in a specific direction
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

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
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


def plot_probstar_distribution(I, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True, color='g', numMeshPoints=40, zlabel='Probability Density'):
    'plot a probstar distribution on a specific direction'

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
