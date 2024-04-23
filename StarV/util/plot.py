"""
Plot module, contains methods for plotting
Dung Tran, 9/11/2022
"""

from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
import matplotlib.pyplot as plt
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
        A = np.eye(I.dim)
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


def plot_2D_Star(I, show=True):

    if I.dim != 2:
        raise Exception('Input set is not 2D star')

    verts = getVertices(I)
    try:
        pypoman.plot_polygon(verts)
    except Exception:
        warnings.warn(message='Potential floating-point error')
    if show:
        plt.show()

def plot_probstar(I, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True):
    """Plot a star set in a specific direction
       y = dir_mat*x + dir_vec, x in I
    """

    if isinstance(I, ProbStar):
        I1 = I.affineMap(dir_mat, dir_vec)
        if I1.dim > 2:
            raise Exception('error: only 2D plot is supported')
        prob = I1.estimateProbability()
        plot_2D_Star(I, show=False)
        l, u = I1.getRanges()
        if show_prob:
            ax = plt.gca()
            ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob))
            print(l[0], l[1])
            print(u[0], u[1])
            ax.set_xlim(l[0]-2, u[0]+2) # Axis Fixed, Yuntao Li, 2/4/2024
            ax.set_ylim(l[1]-2, u[1]+2) # Axis Fixed, Yuntao Li, 2/4/2024

    elif isinstance(I, list):
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
        ax.set_xlim(l[0], u[0]) # Axis Fixed, Yuntao Li, 2/4/2024
        ax.set_ylim(l[1], u[1]) # Axis Fixed, Yuntao Li, 2/4/2024
    else:
        raise Exception('error: first input should be a ProbStar or a list of ProbStar')

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if show:
        plt.show()

def plot_star(I, dir_mat=None, dir_vec=None, label=('$y_1$', '$y_2$'), show=True):
    """Plot a star set in a specific direction
       y = dir_mat*x + dir_vec, x in I
    """

    if isinstance(I, Star):
        I1 = I.affineMap(dir_mat, dir_vec)
        if I1.dim > 2:
            raise Exception('error: only 2D plot is supported')
        plot_2D_Star(I, show=False)
        l, u = I1.getRanges()
        
    elif isinstance(I, list):
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

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if show:
        plt.show()


def plot_2D_Star_using_Polytope(I, show=True):
    """
    Plot 2D star set using Polytope
    Yuntao Li, 2/4/2024
    """

    assert(I, Star) or isinstance(I, ProbStar), 'error: input should be a Star or a ProbStar'

    if I.dim != 2:
        raise Exception('Input set is not 2D star')
    
    [l, u] = I.estimateRanges()
    P = I.toPolytope()
    try:
        ax = P.plot()
    except Exception:
        warnings.warn(message='Potential floating-point error')
    if show:
        plt.show()
    

def plot_probstar_using_Polytope(I, dir_mat=None, dir_vec=None, show_prob=True, label=('$y_1$', '$y_2$'), show=True):
    """Plot a star set in a specific direction
       y = dir_mat*x + dir_vec, x in I

       Yuntao Li, 2/4/2024
    """

    if isinstance(I, ProbStar):
        I1 = I.affineMap(dir_mat, dir_vec)
        if I1.dim > 2:
            raise Exception('error: only 2D plot is supported')
        prob = I1.estimateProbability()
        plot_2D_Star_using_Polytope(I, show=False)
        l, u = I1.getRanges()
        if show_prob:
            ax = plt.gca()
            ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob))
            print(l[0], l[1])
            print(u[0], u[1])
            ax.set_xlim(l[0]-2, u[0]+2)
            ax.set_ylim(l[1]-2, u[1]+2)

    elif isinstance(I, list):
        L = []
        U = []
        for i in range(0,len(I)):
            I2 = I[i].affineMap(dir_mat, dir_vec)
            if I2.dim > 2:
                raise Exception('error: only 2D plot is supported')
            prob = I2.estimateProbability()
            if I2.isEmptySet():
                continue
            else:
                plot_2D_Star_using_Polytope(I2, show=False)
            l, u = I2.getRanges()
            if i==0:
                L = l
                U = u
            else:
                if len(L) == 0:
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
        ax.set_xlim(l[0]-2, u[0]+2)
        ax.set_ylim(l[1]-2, u[1]+2)
    else:
        raise Exception('error: first input should be a ProbStar or a list of ProbStar')

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if show:
        plt.show()


def plot_star_using_Polytope(I, dir_mat=None, dir_vec=None, label=('$y_1$', '$y_2$'), show=True):
    """Plot a star set in a specific direction
       y = dir_mat*x + dir_vec, x in I

       Yuntao Li, 2/4/2024
    """

    if isinstance(I, Star):
        I1 = I.affineMap(dir_mat, dir_vec)
        if I1.dim > 2:
            raise Exception('error: only 2D plot is supported')
        plot_2D_Star_using_Polytope(I, show=False)
        l, u = I1.getRanges()
        
    elif isinstance(I, list):
        L = []
        U = []
        for i in range(0,len(I)):
            I2 = I[i].affineMap(dir_mat, dir_vec)
            if I2.dim > 2:
                raise Exception('error: only 2D plot is supported')
            if I2.isEmptySet():
                continue
            else:
                plot_2D_Star_using_Polytope(I2, show=False)
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

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if show:
        plt.show()
