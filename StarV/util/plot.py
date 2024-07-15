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

import cdd
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
from scipy.spatial import ConvexHull
import itertools
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def project_polyhedron(proj, ineq, eq=None, canonicalize=True):
    """
    From pypoman library

    ---------------------
    ---> 76 linsys = cdd.Matrix(hstack([b, -A]), number_type='float')

    error: TypeError: only size-1 arrays can be converted to Python scalars

    SOLUTION:

    ---> 76 linsys = cdd.Matrix(hstack([b, -A]).tolist(), number_type='float')
    ---------------------

    Apply the affine projection :math:`y = E x + f` to the polyhedron defined
    by:
    .. math::
        A x & \\leq b \\\\
        C x & = d
    Parameters
    ----------
    proj : pair of arrays
        Pair (`E`, `f`) describing the affine projection.
    ineq : pair of arrays
        Pair (`A`, `b`) describing the inequality constraint.
    eq : pair of arrays, optional
        Pair (`C`, `d`) describing the equality constraint.
    canonicalize : bool, optional
        Apply equality constraints from `eq` to reduce the dimension of the
        input polyhedron. May be a blessing or a curse, see notes below.
    Returns
    -------
    vertices : list of arrays
        List of vertices of the projection.
    rays : list of arrays
        List of rays of the projection.
    Notes
    -----
    When the equality set `eq` of the input polytope is not empty, it is
    usually faster to use these equality constraints to reduce the dimension of
    the input polytope (cdd function: `canonicalize()`) before enumerating
    vertices (cdd function: `get_generators()`). Yet, on some descriptions this
    operation may be problematic: if it fails, or if you get empty outputs when
    the output is supposed to be non-empty, you can try setting
    `canonicalize=False`.
    See also
    --------
    This webpage: https://scaron.info/teaching/projecting-polytopes.html
    """
    # the input [b, -A] to cdd.Matrix represents (b - A * x >= 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    (A, b) = ineq
    b = b.reshape((b.shape[0], 1))
    linsys = cdd.Matrix(np.hstack([b, -A]).tolist(), number_type='float')
    linsys.rep_type = cdd.RepType.INEQUALITY
    

    # the input [d, -C] to cdd.Matrix.extend represents (d - C * x == 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    if eq is not None:
        (C, d) = eq
        d = d.reshape((d.shape[0], 1))
        linsys.extend(np.hstack([d, -C]).tolist(), linear=True)
        if canonicalize:
            linsys.canonicalize()

    # Convert from H- to V-representation
    P = cdd.Polyhedron(linsys)
    generators = P.get_generators()
    if generators.lin_set:
        print("Generators have linear set: {}".format(generators.lin_set))
    V = np.array(generators)

    # Project output wrenches to 2D set
    (E, f) = proj
    vertices, rays = [], []
    free_coordinates = []
    for i in range(V.shape[0]):
        if generators.lin_set and i in generators.lin_set:
            free_coordinates.append(list(V[i, 1:]).index(1.))
        elif V[i, 0] == 1:  # vertex
            vertices.append(np.dot(E, V[i, 1:]) + f)
        else:  # ray
            rays.append(np.dot(E, V[i, 1:]))

    assert not rays, "Projection is not a polytope"
    return vertices


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

    # verts = pypoman.projection.project_polytope(proj, ineq)
    verts = project_polyhedron(proj, ineq)

    return verts


def plot_polytope2d(I, ax, color='r',alpha=1.0, edgecolor='k', linewidth=1.0, zorder=1):
    """
    Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu

    From https://github.com/Shaddadi/veritex/blob/master/veritex/utils/plot_poly.py
    eritex/veritex/utils/plot_poly.py 

    Function to plot 2-dimensional polytope
    Parameters:
        set_vs (np.ndarray): Vertices of the set
        ax (AxesSubplot): AxesSubplot
        color (str): Face color
        alpha (float): Color transparency
        edgecolor (str): Edge color
        Linewidth (float): Line width of edges
        zorder (int): Plotting order
    """
    try: # 2 dimensional hull
        set_vs = np.array(getVertices(I))
        hull = ConvexHull(set_vs)
        fs, ps  = hull.equations, hull.points
        fps = (abs(np.dot(fs[:,0:2], ps.T) + fs[:,[2]])<=1e-6).astype(int)
        bools = np.sum(fps, axis=0)==2
        ps_new = ps[bools,:]
        fps, indx = np.unique(fps[:, bools], return_index=True, axis=1)
        ps_new = ps_new[indx, :]
        ps_adj = np.dot(fps.T, fps) # adjacency between points
        indx_adj = [np.nonzero(arr==1)[0] for arr in ps_adj]

        indx_ps_new = [0] # along the edge
        curr_point = 0
        for n in range(len(indx_adj)-1):
            next_idnx0, next_indx1 = indx_adj[curr_point][0], indx_adj[curr_point][1]
            if next_idnx0 not in indx_ps_new:
                indx_ps_new.append(next_idnx0)
                curr_point = next_idnx0
            elif next_indx1 not in indx_ps_new:
                indx_ps_new.append(next_indx1)
                curr_point = next_indx1
            else:
                raise Exception('error: wrong points')

        ps_final = ps_new[indx_ps_new, :]
    except: # 0 or 1 dimensional hull
        ps_final = set_vs

    poly = Polygon(ps_final, facecolor=color, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_collection(PatchCollection([poly], match_original=True,zorder=zorder))


def plot_3D_Box(lbs, ubs, ax, color='r', alpha=1.0, edgecolor='k', linewidth=1.0, show=True):
    """
    From https://github.com/Shaddadi/veritex/blob/master/veritex/utils/plot_poly.py
    eritex/veritex/utils/plot_poly.py 

    Function to plot 3-dimensional box
    Parameters:
        lbs (list): Lower bounds of the box
        ubs (list): Upper bounds of the box
        ax (AxesSubplot): AxesSubplot
        color (str): Face color
        alpha (float): Color transparency
        edgecolor (str): Edge color
        Linewidth (float): Line width of edges
    """

    V = []
    for i in range(len(lbs)):
        V.append([lbs[i], ubs[i]])

    vs = np.array(list(itertools.product(*V)))
    faces = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    for s in faces:
        sq = [
            [vs[s[0], 0], vs[s[0], 1], vs[s[0], 2]],
            [vs[s[1], 0], vs[s[1], 1], vs[s[1], 2]],
            [vs[s[2], 0], vs[s[2], 1], vs[s[2], 2]],
            [vs[s[3], 0], vs[s[3], 1], vs[s[3], 2]]
        ]
        f = a3.art3d.Poly3DCollection([sq])
        f.set_color(color)
        f.set_edgecolor(edgecolor)
        f.set_alpha(alpha)
        f.set_linewidth(linewidth)
        ax.add_collection3d(f)

    if show:
        ax.set_xlim(lbs[0], ubs[0])
        ax.set_ylim(lbs[1], ubs[1])
        ax.set_zlim(lbs[2], ubs[2])
        plt.show()

def plot_3D_Star(I, ax, color='r',alpha=1.0, edgecolor='k', linewidth=1.0, show=True):
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

    hull = ConvexHull(set_vs)
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



def plot_star2D(I, color='r', transparency=1.0, edgecolor='k', edgewidth=1.0, show=True):
    l, u = I.getRanges('estimate')
    ax = plt.gca()
    ax.set_xlim(l[0], u[0])
    ax.set_ylim(l[1], u[1])
    plot_polytope2d(I, ax, color, transparency, edgecolor, edgewidth)
    if show:
        plt.show()
    return ax
    

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
