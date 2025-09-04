"""
TanSig Class (Hyperbolic tangent sigmoid transfer function or TanH function)
Sung Woo Choi, 04/08/2023

"""

# !/usr/bin/python3
import numpy as np
import scipy.sparse as sp
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star
from StarV.set.imagestar import ImageStar

class TanSig(object):
    """
    TanSig Class for reachability
    Author: Sung Woo Choi
    Date: 04/08/2023

    """
    
    @staticmethod
    def evaluate(x):
        return TanSig.f(x)
    
    @staticmethod
    def f(x):
        return np.tanh(x)
    
    @staticmethod
    def df(x):
        """Derivative of tansig(x)"""
        return 1 - np.tanh(x)**2

    @staticmethod
    def getConstraints_primary(l, u):
        """Gets two tangent line constraints on upper bounds and lower bounds"""

        yl = TanSig.f(l)
        yu = TanSig.f(u)
        dyl  = TanSig.df(l)
        dyu =  TanSig.df(u)

        n = len(l)
        Dl = np.zeros(n)
        Du = np.zeros(n)

        # map0 = np.where(l == u)[0]
        # Dl[map0] = 0
        # Du[map0] = 0

        map1 = np.where((l >= 0) & (l != u))[0]
        # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
        Dl[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])
        # constraint 1: y <= y'(l) * (x - l) + y(l)
        Du[map1] = dyu[map1]     

        map1 = np.where((u <= 0) & (l != u))[0]
        # constraint 1: y >= y'(l) * (x - l) + y(l)
        Dl[map1] = dyl[map1]
        # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
        Du[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])

        map1 = np.where((l < 0) & (u > 0))[0]
        m = np.minimum(dyl[map1], dyu[map1])
        # constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
        Dl[map1] = m
        # constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
        Du[map1] = m

        gl = yl - Dl*l
        gu = yu - Du*u
        Du = np.diag(Du)
        Dl = np.diag(Dl)
        return Dl, Du, gl, gu
    
    @staticmethod
    def getConstraints_primary2(l, u):
        """Gets two tangent line constraints"""

        yl = TanSig.f(l)
        yu = TanSig.f(u)
        dyl  = TanSig.df(l)
        dyu =  TanSig.df(u)

        n = len(l)
        Dl = np.zeros(n)
        Du = np.zeros(n)
        gl = np.zeros(n)
        gu = np.zeros(n)

        map1 = np.where((l >= 0) & (l != u))[0]
        # constraint 2: y <= y'(u) * (x - u) + y(u)
        Du[map1] = dyu[map1]
        gu[map1] = yu[map1] - Du[map1] *u[map1] 
        # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
        Dl[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])
        gl[map1] = yl[map1]  - Dl[map1] *l[map1] 

        map1 = np.where((u <= 0) & (l != u))[0]
        # constraint 2: y >= y'(u) * (x - u) + y(u) 
        Dl[map1] = dyu[map1]
        gl[map1] = yu[map1]  - Dl[map1] *u[map1] 
        # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
        Du[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])
        gu[map1] = yu[map1] - Du[map1] *u[map1] 

        map1 = np.where((l < 0) & (u > 0))[0]
        dmin = np.minimum(dyl[map1], dyu[map1])
        gux = (yu[map1] - dmin * u[map1]) / (1 - dmin)
        guy = gux 
        glx = (yl[map1] - dmin * l[map1]) / (1 - dmin)
        gly = glx

        mu = (yl[map1] - guy) / (l[map1] - gux)
        ml = (yu[map1] - gly) / (u[map1] - glx)
        
        # constraint 3: y[index] >= m_l * (x[index] - u) + y_u
        Dl[map1] = mu
        # constraint 4: y[index] <= m_u * (x[index] - l) + y_l
        Du[map1] = ml
        gl[map1] = yl[map1] - Dl[map1]*l[map1]
        gu[map1] = yu[map1] - Du[map1]*u[map1]

        Du = np.diag(Du)
        Dl = np.diag(Dl)
        return Dl, Du, gl, gu
    
    @staticmethod
    def getConstraints_secondary(l, u):
        """Gets tangent line constraints on upper bounds and lower bounds"""
        
        yl = TanSig.f(l)
        yu = TanSig.f(u)
        dyl  = TanSig.df(l)
        dyu =  TanSig.df(u)

        n = len(l)
        D1 = np.zeros(n)
        D2 = np.zeros(n)
        g1 = np.zeros(n)
        g2 = np.zeros(n)
        pol1 = np.ones(n)
        pol2 = np.ones(n)


        # l >= 0
        map1 = np.where((l >= 0) & (l != u))[0]
        # constraint 2: y <= y'(u) * (x - u) + y(u) 
        D1[map1] = dyu[map1]
        g1[map1] = yu[map1] - D1[map1]*u[map1]
        pol1[map1] = -1
        # xo = (u*u - l*l) / (2*(u-l))
        # constraint 4: y <= y'(xo)*(x - xo) + y(xo)
        # xo = (u[map1]**2 - l[map1]**2) / (2*(u[map1] - l[map1]))
        xo = 0.5*(u[map1] + l[map1])      
        D2[map1] = TanSig.df(xo)
        g2[map1] = TanSig.f(xo) - D2[map1]*xo
        pol2[map1] = -1

        # u <= 0
        map1 = np.where((u <= 0) & (l != u))[0]
        # constraint 2: y >= y'(u) * (x - u) + y(u)
        D1[map1] = dyu[map1]
        g1[map1] = yu[map1] - D1[map1]*u[map1]  
        # xo = (u*u - l*l) / (2*(u-l))
        # constraint 4: y >= y'(xo)*(x - xo) + y(xo) 
        # xo = (u[map1]**2 - l[map1]**2) / (2*(u[map1] - l[map1]))
        xo = 0.5*(u[map1] + l[map1])        
        D2[map1] = TanSig.df(xo)
        g2[map1] = TanSig.f(xo) - D2[map1]*xo 

        # l < 0 & u > 0
        map1 = np.where((l < 0) & (u > 0))[0]
        dmin = np.minimum(dyl[map1], dyu[map1])
        gux = (yu[map1] - dmin * u[map1]) / (1 - dmin)
        guy = gux 
        glx = (yl[map1] - dmin * l[map1]) / (1 - dmin)
        gly = glx

        mu = (yl[map1] - guy) / (l[map1] - gux)
        ml = (yu[map1] - gly) / (u[map1] - glx)
        
        # constraint 3: y[index] >= m_l * (x[index] - u) + y_u
        # constraint 4: y[index] <= m_u * (x[index] - l) + y_l
        D1[map1] = mu
        D2[map1] = ml
        g1[map1] = yl[map1] - D1[map1]*l[map1]
        g2[map1] = yu[map1] - D2[map1]*u[map1]
        pol1[map1] = -1

        D = np.vstack((np.diag(D1), np.diag(D2)))
        g = np.hstack((g1, g2))
        pol = np.hstack((pol1, pol2))
        return D, g, pol
    
    @staticmethod
    def getConstraints_all(l, u):
        """Gets four tangent line constraints"""

        yl = TanSig.f(l)
        yu = TanSig.f(u)
        dyl  = TanSig.df(l)
        dyu =  TanSig.df(u)

        n = len(l)
        PL = np.zeros(n)
        PU = np.zeros(n)
        D1 = np.zeros(n)
        D2 = np.zeros(n)
        g1 = np.zeros(n)
        g2 = np.zeros(n)
        pol1 = np.ones(n)
        pol2 = np.ones(n)

        map1 = np.where((l >= 0) & (l != u))[0]
        # if len(map1):
        # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
        PL[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])
        # constraint 1: y <= y'(l) * (x - l) + y(l)
        PU[map1] = dyu[map1]
        # constraint 2: y <= y'(u) * (x - u) + y(u) 
        D1[map1] = dyu[map1]
        g1[map1] = yu[map1] - D1[map1]*u[map1]
        pol1[map1] = -1
        # xo = (u*u - l*l) / (2*(u-l))
        # constraint 4: y <= y'(xo)*(x - xo) + y(xo)
        # xo = (u[map1]**2 - l[map1]**2) / (2*(u[map1] - l[map1]))
        x0 = 0.5*(u[map1] + l[map1])
        D2[map1] = TanSig.df(xo)
        g2[map1] = TanSig.f(xo) - D2[map1]*xo
        pol2[map1] = -1

        map1 = np.where((u <= 0) & (l != u))[0]
        # if len(map1):
        # constraint 1: y >= y'(l) * (x - l) + y(l)
        PL[map1] = dyl[map1]
        # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
        PU[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])
        # constraint 2: y >= y'(u) * (x - u) + y(u)
        D1[map1] = dyu[map1]
        g1[map1] = yu[map1] - D1[map1]*u[map1] 
        # xo = (u*u - l*l) / (2*(u-l))
        # constraint 4: y >= y'(xo)*(x - xo) + y(xo) 
        xo = (u[map1]**2 - l[map1]**2) / (2*(u[map1] - l[map1]))
        D2[map1] = TanSig.df(xo)
        g2[map1] = TanSig.f(xo) - D2[map1]*xo 

        map1 = np.where((l < 0) & (u > 0))[0]
        # if len(map1):
        dmin = np.minimum(dyl[map1], dyu[map1])
        # constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
        PL[map1] = dmin
        # constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
        PU[map1] = dmin
        gux = (yu[map1] - dmin * u[map1]) / (1 - dmin)
        guy = gux 
        glx = (yl[map1] - dmin * l[map1]) / (1 - dmin)
        gly = glx
        mu = (yl[map1] - guy) / (l[map1] - gux)
        ml = (yu[map1] - gly) / (u[map1] - glx)
        # constraint 3: y[index] >= m_l * (x[index] - u) + y_u
        # constraint 4: y[index] <= m_u * (x[index] - l) + y_l
        D1[map1] = mu 
        D2[map1] = ml
        g1[map1] = yl[map1] - D1[map1]*l[map1]
        g2[map1] = yu[map1] - D2[map1]*u[map1]
        pol1[map1] = -1
            
        Pl = yl - PL*l
        Pu = yu - PU*u
        PU = np.diag(PU)
        PL = np.diag(PL)
        SD = np.vstack((np.diag(D1), np.diag(D2)))
        Sg = np.hstack((g1, g2))
        Spol = np.hstack((pol1, pol2))
        return PL, PU, Pl, Pu, SD, Sg, Spol
    
    @staticmethod
    def optimal_iter_approx_upper(l, u, iter=5):
        xi = u
        for i in range(iter):
            yi = np.sqrt(1 - (  (TanSig.f(xi) - TanSig.f(l)) / (xi - l)  )  )
            xi = 0.5*np.log( (1+yi) / (1-yi) )
        return xi
    
    @staticmethod
    def optimal_iter_approx_lower(l, u, iter=5):
        xi = l
        for i in range(iter):
            yi = np.sqrt(1 - (  (TanSig.f(xi) - TanSig.f(u)) / (xi - u)  )  )
            xi = 0.5*np.log( (1-yi) / (1+yi))
        return xi

    @staticmethod
    def getConstraints_optimal(l, u):
        yl = TanSig.f(l)
        yu = TanSig.f(u)
        dyl  = TanSig.df(l)
        dyu =  TanSig.df(u)

        n = len(l)
        al = np.zeros(n)
        au = np.zeros(n)
        gl = np.zeros(n)
        gu = np.zeros(n)

        map = np.where((l >= 0) & (l != u))[0]
        # xo = (u*u - l*l) / (2*(u-l))
        # constraint 2: y <= y'(xo)*(x - xo) + y(xo)
        xo = (u[map]*u[map] - l[map]*l[map]) / (2*(u[map]-l[map]))
        dyo = TanSig.df(xo)
        au[map] = dyo
        gu[map] = TanSig.f(xo) - dyo*xo
        # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
        al[map] = (yu[map] - yl[map]) / (u[map] - l[map])
        gl[map] = yl[map]  - al[map] *l[map] 

        map = np.where((u <= 0) & (l != u))[0]
        # xo = (u*u - l*l) / (2*(u-l))
        # constraint 2: y >= y'(xo)*(x - xo) + y(xo) 
        # xo = (u[map]*u[map] - l[map]*l[map]) / (2*(u[map]-l[map]))
        xo = 0.5*(u[map] + l[map])
        dyo = TanSig.df(xo)
        al[map] = dyo
        gl[map] = TanSig.f(xo) - dyo*xo
        # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
        au[map] = (yu[map] - yl[map]) / (u[map] - l[map])
        gu[map] = yu[map] - au[map] *u[map] 

        map = np.where((l < 0) & (u > 0))[0]
        ou = TanSig.optimal_iter_approx_upper(l[map], u[map])
        ol = TanSig.optimal_iter_approx_lower(l[map], u[map])


        # constraint 3: y[index] >= y'(xol) * (x[index] - xol) + y(xol)
        al[map] = TanSig.df(ol)
        gl[map] = TanSig.f(ol) - al[map]*ol
        # constraint 4: y[index] <= y'(xou) * (x[index] - xou) + y(xou)
        au[map] = TanSig.df(ou)
        gu[map] = TanSig.f(ou) - au[map]*ou

        Du = np.diag(au)
        Dl = np.diag(al)
        return Dl, Du, gl, gu
    
    
    def reachApprox_sparse(I, opt=True, delta=0.98, lp_solver='gurobi', RF=0.0, DR=0, show=False):
        
        assert isinstance(I, SparseStar), 'error: input set is not a SparseStar set'

        N = I.dim

        l, u = I.getRanges(lp_solver=lp_solver, RF=RF, layer='logsig', delta=delta)
        l = l.reshape(N, 1)
        u = u.reshape(N, 1)

        yl = TanSig.f(l)
        yu = TanSig.f(u)
        dyl = TanSig.df(l)
        dyu = TanSig.df(u)

        ## l != u
        map0 = np.where(l != u)[0]
        m = len(map0)
        A0 = np.zeros((N, m))
        for i in range(m):
            A0[map0[i], i] = 1
        new_A = np.hstack((np.zeros((N, 1)), A0))

        map1 = np.where(l == u)[0]
        if len(map1):
            new_A[map1, 0] = yl[map1].reshape(-1)
            new_A[map1, 1:m+1] = 0

        nv = I.nVars + m

        ## l > 0 & l != u
        map1 = np.where(l[map0] >= 0)[0]
        if len(map1):
            map_ = map0[map1]
            l_ = l[map_]
            u_ = u[map_]
            yl_ = yl[map_]
            yu_ = yu[map_]
            dyl_ = dyl[map_]
            dyu_ = dyu[map_]

            Z = sp.csc_matrix((len(map_), I.nZVars))

            # constraint 1: y <= y'(l) * (x - l) + y(l)
            C11 = sp.hstack((Z, -dyl_*I.X(map_), A0[map_, :]))
            d11 = dyl_*(I.c(map_) - l_) + yl_

            # constraint 2: y <= y'(u) * (x - u) + y(u) 
            C12 = sp.hstack((Z, -dyu_*I.X(map_), A0[map_, :]))
            d12 = dyu_*(I.c(map_) - u_) + yu_

            # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
            g = (yu_ - yl_) / (u_ - l_)
            C13 = sp.hstack((Z, g*I.X(map_), -A0[map_, :]))
            d13 = -g*(I.c(map_) - l_) - yl_

            # xo = (u*u - l*l) / (2*(u-l))
            # constraint 4: y <= y'(xo)*(x - xo) + y(xo)
            xo = 0.5*(u_ + l_)
            # xo = (u_*u_ - l_*l_) / (2*(u_ - l_))
            dyo = TanSig.df(xo)
            C14 = sp.hstack((Z, -dyo*I.X(map_), A0[map_, :]))
            d14 = dyo*(I.c(map_) - xo) + TanSig.f(xo)

            C1 = sp.vstack((C11, C12, C13, C14)).tocsc()
            d1 = np.vstack((d11, d12, d13, d14)).reshape(-1)
        else:
            C1 = sp.csc_matrix((0, nv))
            d1 = np.empty((0))

        ## u <= 0 & l != u
        map1 = np.where(u[map0] <= 0)[0]
        if len(map1):
            map_ = map0[map1]
            l_ = l[map_]
            u_ = u[map_]
            yl_ = yl[map_]
            yu_ = yu[map_]
            dyl_ = dyl[map_]
            dyu_ = dyu[map_]

            Z = sp.csc_matrix((len(map_), I.nZVars))

            # constraint 1: y >= y'(l) * (x - l) + y(l)
            C21 = sp.hstack((Z, dyl_*I.X(map_), -A0[map_, :]))
            d21 = -dyl_*(I.c(map_) - l_) - yl_

            # constraint 2: y >= y'(u) * (x - u) + y(u)
            C22 = sp.hstack((Z, dyu_*I.X(map_), -A0[map_, :]))
            d22 = -dyu_*(I.c(map_) - u_) - yu_

            # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
            g = (yu_ - yl_) / (u_ - l_)
            C23 = sp.hstack((Z, -g*I.X(map_), A0[map_, :]))
            d23 = g*(I.c(map_) - l_) + yl_

            # xo = (u*u - l*l) / (2*(u-l))
            # constraint 4: y >= y'(xo)*(x - xo) + y(xo) 
            # xo = (u_*u_ - l_*l_) / (2*(u_ - l_))
            xo = 0.5*(u_ + l_)
            dyo = TanSig.df(xo)
            C24 = sp.hstack((Z, dyo*I.X(map_), -A0[map_, :]))
            d24 = -dyo*(I.c(map_) - xo) - TanSig.f(xo)

            C2 = sp.vstack((C21, C22, C23, C24)).tocsc()
            d2 = np.vstack((d21, d22, d23, d24)).reshape(-1)
        else:
            C2 = sp.csc_matrix((0, nv))
            d2 = np.empty((0))

        map1 = np.where((l < 0) & (u > 0))[0]
        if len(map1):
            l_ = l[map1]
            u_ = u[map1]
            yl_ = yl[map1]
            yu_ = yu[map1]
            dyl_ = dyl[map1]
            dyu_ = dyu[map1]

            dmin = np.minimum(dyl_, dyu_)
            Z = sp.csc_matrix((len(map1), I.nZVars))

            # constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
            C31 = sp.hstack((Z, dmin*I.X(map1), -A0[map1, :]))
            d31 = -dmin*(I.c(map1) - l_) - yl_

            # constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
            C32 = sp.hstack((Z, -dmin*I.X(map1), A0[map1, :]))
            d32 = dmin*(I.c(map1) - u_) + yu_

            if opt == True:
                xou = TanSig.optimal_iter_approx_upper(l_, u_)
                xol = TanSig.optimal_iter_approx_lower(l_, u_)
                dxou = TanSig.df(xou)
                dxol = TanSig.df(xol)

                # constraint 3: y[index] >= y'(xol)*(x[index] - xol) + y(xol)
                C33 = sp.hstack((Z, dxol*I.X(map1), -A0[map1, :]))
                d33 = -dxol*(I.c(map1) - xol) - TanSig.f(xol)
                
                # constraint 4: y[index] <= y'(xou)*(x[index] - xou) + y(xou)
                C34 = sp.hstack((Z, -dxou*I.X(map1), A0[map1, :]))
                d34 = dxou*(I.c(map1) - xou) + TanSig.f(xou)

            else:
                gux = (yu_ - dmin * u_) / (1 - dmin)
                guy = gux 
                glx = (yl_ - dmin * l_) / (1 - dmin)
                gly = glx

                mu = (yl_ - guy) / (l_ - gux)
                ml = (yu_ - gly) / (u_ - glx)
                
                # constraint 3: y[index] >= m_l * (x[index] - u) + y_u
                C33 = sp.hstack((Z, ml*I.X(map1), -A0[map1, :]))
                d33 = -ml*(I.c(map1) - u_) - yu_

                # constraint 4: y[index] <= m_u * (x[index] - l) + y_l
                C34 = sp.hstack((Z, -mu*I.X(map1), A0[map1, :]))
                d34 = mu*(I.c(map1) - l_) + yl_

            C3 = sp.vstack((C31, C32, C33, C34)).tocsc()
            d3 = np.vstack((d31, d32, d33, d34)).reshape(-1)

        else:
            C3 = sp.csc_matrix((0, nv))
            d3 = np.empty((0))

        n = I.C.shape[0]
        if len(I.d):
            C0 = sp.hstack((I.C, sp.csc_matrix((n, m)))) 
            d0 = I.d
        else:
            C0 = sp.csc_matrix((0, I.nVars+m))
            d0 = np.empty((0))

        new_C = sp.vstack((C0, C1, C2, C3))
        new_d = np.hstack((d0, d1, d2, d3))

        new_pred_lb = np.hstack((I.pred_lb, yl[map0].reshape(-1)))
        new_pred_ub = np.hstack((I.pred_ub, yu[map0].reshape(-1)))
        new_pred_depth = np.hstack((I.pred_depth+1, np.zeros(m)))
        
        S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
        if DR > 0:
            if show:
                if (S.pred_depth >= DR).any():
                    print('Applying depth reduction {}'.format(DR))
            S = S.depthReduction(DR=DR)
        return S
    

    def reachApproxSparse2(I, lp_solver='gurobi', RF=0.0, DR=0):
        
        assert isinstance(I, SparseStar), 'error: input set is not a SparseStar set'

        N = I.dim

        l, u = I.getRanges(lp_solver=lp_solver, RF=RF)

        ## l != u
        map0 = np.where(l != u)[0]
        m = len(map0)
        A0 = np.zeros((N, m))
        for i in range(m):
            A0[map0[i], i] = 1
        new_A = np.hstack((np.zeros((N, 1)), A0))

        map1 = np.where(l == u)[0]
        if len(map1):
            new_A[map1, 0] = yl[map1]
            new_A[map1, 1:m+1] = 0

        nv = I.nVars + m


        # what if l[i] == u[i] for some i?
        # need to figure out to remove zero columns and rows
        PDl, PDu, Pgl, Pgu, SD, Sg, pol = TanSig.getConstraints_all(l, u)

        PDu = np.matmul(PDu, I.A)
        PDl = np.matmul(PDl, I.A)
        PU = -PDu[:, 1:]
        PL = PDl[:, 1:]
        Pu = Pgu + PDu[:, 0]
        Pl = -(Pgl + PDl[:, 0])

        SD = np.matmul(SD, I.A)
        SC = SD[:, 1:]
        Sd = -pol*(Sg + SD[:, 0])

        CpU = np.hstack((PU, A0))
        CpL = np.hstack((PL, -A0))

        E = np.vstack((A0, A0))
        T = np.hstack((SC, -E))
        Cs = np.matmul(np.diag(pol), T)

        new_C = sp.csc_matrix(np.vstack((CpU, CpL, Cs)))
        new_d = np.concatenate((Pu, Pl, Sd))

        yl = TanSig.f(l)
        yu = TanSig.f(u)
        new_pred_lb = np.hstack((I.pred_lb, yl[map0]))
        new_pred_ub = np.hstack((I.pred_ub, yu[map0]))
        
        new_pred_depth = I.pred_depth
        if m == N:
            new_pred_depth += 1
        else:
            new_pred_depth[map0] += 1
        new_pred_depth = np.hstack((new_pred_depth, np.zeros(m)))

        S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
        if DR > 0:
            S = S.depthReduction(DR=DR)
        return S
    

    # used for GRU and LSTM ( logsigXtansig, (1-logsig)Xtansig )
    def getConstraints(c, X, A0, l, u, nZVars,opt=False):

        N = l.shape[0]

        l = l.reshape(N, 1)
        u = u.reshape(N, 1)
        yl = TanSig.f(l)
        yu = TanSig.f(u)
        dyl = TanSig.df(l)
        dyu = TanSig.df(u)

        ## l != u
        map0 = np.where(l != u)[0]

        nv = X.shape[1] + A0.shape[1]

        ## l > 0 & l != u
        map1 = np.where(l[map0] >= 0)[0]
        if len(map1):
            map_ = map0[map1]
            l_ = l[map_]
            u_ = u[map_]
            yl_ = yl[map_]
            yu_ = yu[map_]
            dyl_ = dyl[map_]
            dyu_ = dyu[map_]

            Z = sp.csc_matrix((len(map_), nZVars))

            # constraint 1: y <= y'(l) * (x - l) + y(l)
            C11 = sp.hstack((Z, -dyl_*X[map_, :], A0[map_, :]))
            d11 = dyl_*(c[map_] - l_) + yl_

            # constraint 2: y <= y'(u) * (x - u) + y(u) 
            C12 = sp.hstack((Z, -dyu_*X[map_, :], A0[map_, :]))
            d12 = dyu_*(c[map_] - u_) + yu_

            # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
            g = (yu_ - yl_) / (u_ - l_)
            C13 = sp.hstack((Z, g*X[map_, :], -A0[map_, :]))
            d13 = -g*(c[map_] - l_) - yl_

            # xo = (u*u - l*l) / (2*(u-l))
            # constraint 4: y <= y'(xo)*(x - xo) + y(xo)
            xo = 0.5*(u_ + l_)
            # xo = (u_*u_ - l_*l_) / (2*(u_ - l_))
            dyo = TanSig.df(xo)
            C14 = sp.hstack((Z, -dyo*X[map_], A0[map_, :]))
            d14 = dyo*(c[map_] - xo) + TanSig.f(xo)

            C1 = sp.vstack((C11, C12, C13, C14)).tocsc()
            d1 = np.vstack((d11, d12, d13, d14)).reshape(-1)
        else:
            C1 = sp.csc_matrix((0, nv))
            d1 = np.empty((0))

        ## u <= 0 & l != u
        map1 = np.where(u[map0] <= 0)[0]
        if len(map1):
            map_ = map0[map1]
            l_ = l[map_]
            u_ = u[map_]
            yl_ = yl[map_]
            yu_ = yu[map_]
            dyl_ = dyl[map_]
            dyu_ = dyu[map_]

            Z = sp.csc_matrix((len(map_), nZVars))

            # constraint 1: y >= y'(l) * (x - l) + y(l)
            C21 = sp.hstack((Z, dyl_*X[map_], -A0[map_, :]))
            d21 = -dyl_*(c[map_] - l_) - yl_

            # constraint 2: y >= y'(u) * (x - u) + y(u)
            C22 = sp.hstack((Z, dyu_*X[map_], -A0[map_, :]))
            d22 = -dyu_*(c[map_] - u_) - yu_

            # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
            g = (yu_ - yl_) / (u_ - l_)
            C23 = sp.hstack((Z, -g*X[map_], A0[map_, :]))
            d23 = g*(c[map_] - l_) + yl_

            # xo = (u*u - l*l) / (2*(u-l))
            # constraint 4: y >= y'(xo)*(x - xo) + y(xo) 
            # xo = (u_*u_ - l_*l_) / (2*(u_ - l_))
            xo = 0.5*(u_ + l_)
            dyo = TanSig.df(xo)
            C24 = sp.hstack((Z, dyo*X[map_], -A0[map_, :]))
            d24 = -dyo*(c[map_] - xo) - TanSig.f(xo)

            C2 = sp.vstack((C21, C22, C23, C24)).tocsc()
            d2 = np.vstack((d21, d22, d23, d24)).reshape(-1)
        else:
            C2 = sp.csc_matrix((0, nv))
            d2 = np.empty((0))

        map1 = np.where((l < 0) & (u > 0))[0]
        if len(map1):
            l_ = l[map1]
            u_ = u[map1]
            yl_ = yl[map1]
            yu_ = yu[map1]
            dyl_ = dyl[map1]
            dyu_ = dyu[map1]

            dmin = np.minimum(dyl_, dyu_)
            Z = sp.csc_matrix((len(map1), nZVars))

            # constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
            C31 = sp.hstack((Z, dmin*X[map1, :], -A0[map1, :]))
            d31 = -dmin*(c[map1] - l_) - yl_

            # constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
            C32 = sp.hstack((Z, -dmin*X[map1, :], A0[map1, :]))
            d32 = dmin*(c[map1] - u_) + yu_

            if opt == True:
                xou = TanSig.optimal_iter_approx_upper(l_, u_)
                xol = TanSig.optimal_iter_approx_lower(l_, u_)
                dxou = TanSig.df(xou)
                dxol = TanSig.df(xol)

                # constraint 3: y[index] >= y'(xol)*(x[index] - xol) + y(xol)
                C33 = sp.hstack((Z, dxol*X[map1, :], -A0[map1, :]))
                d33 = -dxol*(c[map1] - xol) - TanSig.f(xol)

                # constraint 4: y[index] <= y'(xou)*(x[index] - xou) + y(xou)
                C34 = sp.hstack((Z, -dxou*X[map1, :], A0[map1, :]))
                d34 = dxou*(c[map1] - xou) + TanSig.f(xou)

            else:
                gux = (yu_ - dmin * u_) / (1 - dmin)
                guy = gux 
                glx = (yl_ - dmin * l_) / (1 - dmin)
                gly = glx

                mu = (yl_ - guy) / (l_ - gux)
                ml = (yu_ - gly) / (u_ - glx)

                # constraint 3: y[index] >= m_l * (x[index] - u) + y_u
                C33 = sp.hstack((Z, ml*X[map1, :], -A0[map1, :]))
                d33 = -ml*(c[map1] - u_) - yu_

                # constraint 4: y[index] <= m_u * (x[index] - l) + y_l
                C34 = sp.hstack((Z, -mu*X[map1, :], A0[map1, :]))
                d34 = mu*(c[map1] - l_) + yl_

            C3 = sp.vstack((C31, C32, C33, C34)).tocsc()
            d3 = np.vstack((d31, d32, d33, d34)).reshape(-1)

        else:
            C3 = sp.csc_matrix((0, nv))
            d3 = np.empty((0))

        new_C = sp.vstack((C1, C2, C3))
        new_d = np.hstack((d1, d2, d3))
        return new_C, new_d, yl[map0], yu[map0]
    

    def reachApprox_star(I, opt=True, lp_solver='gurobi', RF=0.0):
        
        assert isinstance(I, Star), 'error: input set is not a SparseStar set'

        N = I.dim

        l, u = I.getRanges(lp_solver=lp_solver, RF=RF)
        yl, yu = TanSig.f(l), TanSig.f(u)
        dyl, dyu = TanSig.df(l), TanSig.df(u)

        ## l != u
        map0 = np.where(l != u)[0]
        m = len(map0)
        V0 = np.zeros((N, m))
        for i in range(m):
            V0[map0[i], i] = 1
        new_V = np.hstack([np.zeros([N, m+1]), V0])

        map1 = np.where(l == u)[0]
        if len(map1):
            new_V[map1, 0] = yl[map1]
            new_V[map1, 1:m+1] = 0

        nv = I.nVars + m

        ## l >= 0 & l != u
        map1 = np.where(l[map0] >= 0)[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            # constraint 1: y <= y'(l) * (x - l) + y(l)
            C11 = np.hstack([-dyl_*V1, V2])
            d11 = dyl_*(c1 - l) + yl_
            
            # constraint 2: y <= y'(u) * (x - u) + y(u)
            C12 = np.hstack([-dyu_*V1, V2])
            d12 = dyu_*(c1 - u_) + yu_
            
            # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
            g = (yu_ - yl_) / (u_ - l_)
            C13 = np.hstack([g*V1, -V2])
            d13 = -g*(c1 - l_) - yl_

            # xo = (u*u - l*l) / (2*(u-l))
            # constraint 4: y <= y'(xo)*(x - xo) + y(xo)
            xo = 0.5*(u_ + l_)
            # xo = (u_*u_ - l_*l_) / (2*(u_ - l_))
            dyo = TanSig.df(xo)
            C14 = np.hstack([-dyo*V1, V2])
            d14 = dyo*(c1 - xo) + TanSig.f(xo)

            C1 = np.vstack((C11, C12, C13, C14))
            d1 = np.hstack((d11, d12, d13, d14))
        else:
            C1 = np.empty((0, nv))
            d1 = np.empty((0))

        ## u <= 0 & l != u
        map1 = np.where(u[map0] <= 0)[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            # constraint 1: y >= y'(l) * (x - l) + y(l)
            C21 = np.hstack([dyl_*V1, -V2])
            d21 = -dyl_*(c1 - l_) - yl_

            # constraint 2: y >= y'(u) * (x - u) + y(u)
            C22 = np.hstack([dyu_*V1, -V2])
            d22 = -dyu_*(c1 - u_) - yu_

            # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l)
            g = (yu_ - yl_) / (u_ - l_)
            C23 = np.hstack([-g*V1, V2])
            d23 = g*(c1 - l_) + yl_

            # xo = (u*u - l*l) / (2*(u-l))
            # constraint 4: y >= y'(xo)*(x - xo) + y(xo) 
            # xo = (u_*u_ - l_*l_) / (2*(u_ - l_))
            xo = 0.5*(u_ + l_)
            dyo = TanSig.df(xo)
            C24 = sp.hstack([dyo*V1, -V2])
            d24 = -dyo*(c1 - xo) - TanSig.f(xo)

            C2 = np.vstack((C21, C22, C23, C24))
            d2 = np.hstack((d21, d22, d23, d24))
        else:
            C2 = np.empty((0, nv))
            d2 = np.empty((0))

        map1 = np.where((l < 0) & (u > 0))[0]
        if len(map1):
            l_, u_ = l[map1], u[map1]
            yl_, yu_ = yl[map1], yu[map1]
            dyl_, dyu_ = dyl[map1], dyu[map1]
            c1, V1 = I.V[map1, 0], I.V[map1, 1:]
            V2 = V0[map1, :]

            dmin = np.minimum(dyl_, dyu_)

            # constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
            C31 = np.hstack([dmin*V1, -V2])
            d31 = -dmin*(c1 - l_) - yl_

            # constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u)
            C32 = np.hstack([-dmin*V1, V2])
            d32 = dmin*(c1 - u_) + yu_

            if opt == True:
                xou = TanSig.optimal_iter_approx_upper(l_, u_)
                xol = TanSig.optimal_iter_approx_lower(l_, u_)
                dxou = TanSig.df(xou)
                dxol = TanSig.df(xol)

                # constraint 3: y[index] >= y'(xol)*(x[index] - xol) + y(xol)
                C33 = np.hstack([dxol*V1, -V2])
                d33 = -dxol*(c1 - xol) - TanSig.f(xol)
                
                # constraint 4: y[index] <= y'(xou)*(x[index] - xou) + y(xou)
                C34 = np.hstack([-dxou*V1, V2])
                d34 = dxou*(c1 - xou) + TanSig.f(xou)

            else:
                gux = (yu_ - dmin * u_) / (1 - dmin)
                guy = gux 
                glx = (yl_ - dmin * l_) / (1 - dmin)
                gly = glx

                mu = (yl_ - guy) / (l_ - gux)
                ml = (yu_ - gly) / (u_ - glx)
                
                # constraint 3: y[index] >= m_l * (x[index] - u) + y_u
                C33 = np.hstack([ml*V1, -V2])
                d33 = -ml*(c1 - u_) - yu_

                # constraint 4: y[index] <= m_u * (x[index] - l) + y_l
                C34 = np.hstack([-mu*V1, V2])
                d34 = mu*(c1 - l_) + yl_

            C3 = np.vstack((C31, C32, C33, C34))
            d3 = np.hstack((d31, d32, d33, d34))

        else:
            C3 = np.empty((0, nv))
            d3 = np.empty((0))

        n = I.C.shape[0]
        if len(I.d):
            C0 = np.hstack([I.C, np.zeros([n, m])]) 
            d0 = I.d
        else:
            C0 = np.empty([0, I.nVars+m])
            d0 = np.empty([0])

        new_C = np.vstack((C0, C1, C2, C3))
        new_d = np.hstack((d0, d1, d2, d3))

        new_pred_lb = np.hstack((I.pred_lb, yl[map0]))
        new_pred_ub = np.hstack((I.pred_ub, yu[map0]))
        
        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)
    
    
    def reach(I, opt=False, delta=0.98, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        if isinstance(I, SparseStar):
            return TanSig.reachApprox_sparse(I=I, opt=opt, delta=delta, lp_solver=lp_solver, RF=RF, DR=DR, show=show)
        elif isinstance(I, Star):
            return TanSig.reachApprox_star(I, opt=opt, lp_solver=lp_solver, RF=RF)
        elif isinstance(I, ImageStar):
            shape = I.shape()
            S = TanSig.reachApprox_star(I.toStar(), opt=opt, lp_solver=lp_solver, RF=RF)
            return S.toImageStar(image_shape=shape, copy_=False)
        else:
            raise Exception('error: unknown input set')