"""
LogSig Class (Log-sigmoid transfer function or Sigmoid function)
Sung Woo Choi, 04/08/2023

"""

# !/usr/bin/python3
import copy
import numpy as np
import scipy.sparse as sp
from StarV.set.sparsestar import SparseStar
from StarV.set.star import Star

class LogSig(object):
    """
    LogSig Class for reachability   
    Author: Sung Woo Choi
    Date: 04/08/2023

    """

    @staticmethod
    def f(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def df(x):
        """Derivative of logsig(x)"""
        f = LogSig.f(x)
        return f * (1 - f)

    @staticmethod
    def getConstraints_primary(l, u):
        """Gets tangent line constraints on upper bounds and lower bounds"""
        
        yl = LogSig.f(l)
        yu = LogSig.f(u)
        dyl  = LogSig.df(l)
        dyu =  LogSig.df(u)

        n = len(l)
        al = np.zeros(n)
        au = np.zeros(n)

        # map0 = np.where(l == u)[0]
        # al[map0] = 0
        # au[map0] = 0

        map1 = np.where((l >= 0) & (l != u))[0]
        # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
        al[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])
        # constraint 1: y <= y'(l) * (x - l) + y(l)
        au[map1] = dyu[map1]     

        map2 = np.where((u <= 0) & (l != u))[0]
        # constraint 1: y >= y'(l) * (x - l) + y(l)
        al[map2] = dyl[map2]
        # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
        au[map2] = (yu[map2] - yl[map2]) / (u[map2] - l[map2])

        map3 = np.where((l < 0) & (u > 0))[0]
        m = np.minimum(dyl[map3], dyu[map3])
        # constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
        al[map3] = m
        # constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
        au[map3] = m

        gl = yl - al*l
        gu = yu - au*u
        Du = np.diag(au)
        Dl = np.diag(al)
        return Dl, Du, gl, gu
    
    @staticmethod
    def getConstraints_primary2(l, u):
        yl = LogSig.f(l)
        yu = LogSig.f(u)
        dyl  = LogSig.df(l)
        dyu =  LogSig.df(u)

        n = len(l)
        al = np.zeros(n)
        au = np.zeros(n)
        gl = np.zeros(n)
        gu = np.zeros(n)

        map1 = np.where((l >= 0) & (l != u))[0]
        # constraint 2: y <= y'(u) * (x - u) + y(u)
        au[map1] = dyu[map1]
        gu[map1] = yu[map1] - au[map1]*u[map1] 
        # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
        al[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])
        gl[map1] = yl[map1]  - al[map1]*l[map1] 

        map2 = np.where((u <= 0) & (l != u))[0]
        # constraint 2: y >= y'(u) * (x - u) + y(u) 
        al[map2] = dyu[map2]
        gl[map2] = yu[map2]  - al[map2]*u[map2] 
        # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
        au[map2] = (yu[map2] - yl[map2]) / (u[map2] - l[map2])
        gu[map2] = yu[map2] - au[map2]*u[map2] 

        map3 = np.where((l < 0) & (u > 0))[0]
        dmin = np.minimum(dyl[map3], dyu[map3])
        gux = (yu[map3] - dmin * u[map3] - 0.5) / (0.25 - dmin)
        guy = 0.25 * gux + 0.5  
        glx = (yl[map3] - dmin * l[map3] - 0.5) / (0.25 - dmin)
        gly = 0.25 * glx + 0.5

        mu = (yl[map3] - guy) / (l[map3] - gux)
        ml = (yu[map3] - gly) / (u[map3] - glx)
        
        # constraint 3: y[index] >= m_l * (x[index] - u) + y_u
        al[map3] = mu
        # constraint 4: y[index] <= m_u * (x[index] - l) + y_l
        au[map3] = ml
        gl[map3] = yl[map3] - al[map3]*l[map3]
        gu[map3] = yu[map3] - au[map3]*u[map3]

        Du = np.diag(au)
        Dl = np.diag(al)
        return Dl, Du, gl, gu
    
    @staticmethod
    def getConstraints_secondary(l, u):
        """Gets tangent line constraints on upper bounds and lower bounds"""
        
        yl = LogSig.f(l)
        yu = LogSig.f(u)
        dyl  = LogSig.df(l)
        dyu =  LogSig.df(u)

        n = len(l)
        D1 = np.zeros(n)
        D2 = np.zeros(n)
        g1 = np.zeros(n)
        g2 = np.zeros(n)

        # l >= 0
        map1 = np.where((l >= 0) & (l != u))[0]
        # constraint 2: y <= y'(u) * (x - u) + y(u) 
        D1[map1] = dyu[map1]
        g1[map1] = yu[map1] - D1[map1]*u[map1] 
        # xo = (u*u - l*l) / (2*(u-l))
        # constraint 4: y <= y'(xo)*(x - xo) + y(xo)
        #xo = (u[map1]**2 - l[map1]**2) / (2*(u[map1] - l[map1]))
        xo = 0.5*(u[map1] + l[map1])
        D2[map1] = LogSig.df(xo)
        g2[map1] = LogSig.f(xo) - D2[map1]*xo 

        # u <= 0
        map1 = np.where((u <= 0) & (l != u))[0]
        # constraint 2: y >= y'(u) * (x - u) + y(u)
        D1[map1] = dyu[map1]
        g1[map1] = yu[map1] + D1[map1]*u[map1] 
        # xo = (u*u - l*l) / (2*(u-l))
        # constraint 4: y >= y'(xo)*(x - xo) + y(xo) 
        # xo = (u[map1]**2 - l[map1]**2) / (2*(u[map1] - l[map1]))
        xo = 0.5*(u[map1] + l[map1])        
        D2[map1] = LogSig.df(xo)
        g2[map1] = LogSig.f(xo) - D2[map1]*xo 

        # l < 0 & u > 0
        map1 = np.where((l < 0) & (u > 0))[0]
        dmin = np.minimum(dyl[map1], dyu[map1])
        gux = (yu[map1] - dmin * u[map1] - 0.5) / (0.25 - dmin)
        guy = 0.25 * gux + 0.5  
        glx = (yl[map1] - dmin * l[map1] - 0.5) / (0.25 - dmin)
        gly = 0.25 * glx + 0.5

        mu = (yl[map1] - guy) / (l[map1] - gux)
        ml = (yu[map1] - gly) / (u[map1] - glx)
        
        # constraint 3: y[index] >= m_l * (x[index] - u) + y_u
        D1[map1] = mu
        # constraint 4: y[index] <= m_u * (x[index] - l) + y_l
        D2[map1] = ml
        g1[map1] = yl[map1] - D1[map1]*l[map1]
        g2[map1] = yu[map1] - D2[map1]*u[map1]

        D = np.vstack((np.diag(D1), np.diag(D2)))
        g = np.hstack((g1, g2))
        return D, g

    # @staticmethod
    # def getConstraints(l, u):
    #     yl = LogSig.f(l)
    #     yu = LogSig.f(u)
    #     dyl  = LogSig.df(l)
    #     dyu =  LogSig.df(u)

    #     n = len(l)
    #     pl = np.zeros(n)
    #     pu = np.zeros(n)
    #     sl = np.zeros(n)
    #     su = np.zeros(n)
    #     Sl = np.zeros(n)
    #     Su = np.zeros(n)

    #     ##################

    #     map1 = np.where((l >= 0) & (l != u))[0]
    #     # constraint 2: y <= y'(u) * (x - u) + y(u)
    #     su[map1] = dyl[map1]
    #     Pu[map1] = yl[map1] - su[map1] *l[map1] 
    #     # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
    #     sl[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])
    #     Pl[map1] = yl[map1]  - sl[map1] *l[map1] 

    #     map2 = np.where((u <= 0) & (l != u))[0]
    #     # constraint 2: y >= y'(u) * (x - u) + y(u) 
    #     sl[map2] = dyu[map2]
    #     Pl[map2] = yu[map2]  - sl[map2] *u[map2] 
    #     # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
    #     su[map2] = (yu[map2] - yl[map2]) / (u[map2] - l[map2])
    #     Pu[map2] = yu[map2] - su[map2] *u[map2] 

    #     map3 = np.where((l < 0) & (u > 0))[0]
    #     dmin = np.minimum(dyl[map3], dyu[map3])
    #     gux = (yu[map3] - dmin * u[map3] - 0.5) / (0.25 - dmin)
    #     guy = 0.25 * gux + 0.5  
    #     glx = (yl[map3] - dmin * l[map3] - 0.5) / (0.25 - dmin)
    #     gly = 0.25 * glx + 0.5

    #     mu = (yl[map3] - guy) / (l[map3] - gux)
    #     ml = (yu[map3] - gly) / (u[map3] - glx)
        
    #     # constraint 3: y[index] >= m_l * (x[index] - u) + y_u
    #     sl[map3] = mu
    #     # constraint 4: y[index] <= m_u * (x[index] - l) + y_l
    #     su[map3] = ml
    #     Pl[map3] = yl[map3] - sl[map3]*l[map3]
    #     Pu[map3] = yu[map3] - su[map3]*u[map3]

    #     SU = np.diag(su)
    #     SL = np.diag(sl)
    #     return Dl, Du, gl, gu
    #     #############

    #     map0 = np.where(l == u)[0]
    #     pl[map0] = 0
    #     pu[map0] = 0

    #     bl = np.zeros(n - len(map0))
    #     bu = np.zeros(n - len(map0))

    #     map1 = np.where((l >= 0) & (l != u))[0]
    #     # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
    #     pl[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])
    #     # constraint 1: y <= y'(l) * (x - l) + y(l)
    #     pu[map1] = dyu[map1]     

    #     map2 = np.where((u <= 0) & (l != u))[0]
    #     # constraint 1: y >= y'(l) * (x - l) + y(l)
    #     pl[map2] = dyl[map2]
    #     # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
    #     pu[map2] = (yu[map2] - yl[map2]) / (u[map2] - l[map2])

    #     map3 = np.where((l < 0) & (u > 0))[0]
    #     m = np.minimum(dyl[map3], dyu[map3])
    #     # constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
    #     pl[map3] = m
    #     # constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
    #     pu[map3] = m

    #     PU = np.diag(au)
    #     PL = np.diag(al)
    #     Pl = yl - pl*l
    #     Pu = yu - pu*u

    #     return [PL, PU, pl, pu], [SL, SU, sl ,su]


    @staticmethod
    def optimal_iter_approx_upper(l, u, iter=5):
        xi = u
        for i in range(iter):
            yi = 0.5*( 1 + np.sqrt(1 - 4*(  (LogSig.f(xi) - LogSig.f(l)) / (xi - l)  )  )  )
            xi = -np.log( 1 / yi - 1 )
        return xi 
    
    @staticmethod
    def optimal_iter_approx_lower(l, u, iter=5):
        xi = l
        for i in range(iter):
            yi = 0.5*(  1 + np.sqrt(1 - 4*(  (LogSig.f(xi) - LogSig.f(u)) / (xi - u)  )  )  )
            xi = np.log( 1 / yi - 1 )
        return xi

    @staticmethod
    def getConstraints_optimal(l, u):
        yl = LogSig.f(l)
        yu = LogSig.f(u)
        dyl  = LogSig.df(l)
        dyu =  LogSig.df(u)

        n = len(l)
        al = np.zeros(n)
        au = np.zeros(n)
        gl = np.zeros(n)
        gu = np.zeros(n)

        map1 = np.where((l >= 0) & (l != u))[0]
        # xo = (u*u - l*l) / (2*(u-l))
        # constraint 2: y <= y'(xo)*(x - xo) + y(xo)
        xo = (u[map1]*u[map1] - l[map1]*l[map1]) / (2*(u[map1]-l[map1]))
        dyo = LogSig.df(xo)
        au[map1] = dyo
        gu[map1] = LogSig.f(xo) - dyo*xo
        # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
        al[map1] = (yu[map1] - yl[map1]) / (u[map1] - l[map1])
        gl[map1] = yl[map1]  - al[map1] *l[map1] 

        map2 = np.where((u <= 0) & (l != u))[0]
        # xo = (u*u - l*l) / (2*(u-l))
        # constraint 2: y >= y'(xo)*(x - xo) + y(xo) 
        # xo = (u[map2]*u[map2] - l[map2]*l[map2]) / (2*(u[map2]-l[map2]))
        xo = 0.5*(u[map2] + l[map2])
        dyo = LogSig.df(xo)
        al[map2] = dyo
        gl[map2] = LogSig.f(xo) - dyo*xo
        # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
        au[map2] = (yu[map2] - yl[map2]) / (u[map2] - l[map2])
        gu[map2] = yu[map2] - au[map2] *u[map2] 

        map3 = np.where((l < 0) & (u > 0))[0]
        ou = LogSig.optimal_iter_approx_upper(l[map3], u[map3])
        ol = LogSig.optimal_iter_approx_lower(l[map3], u[map3])

        # constraint 3: y[index] >= y'(xol) * (x[index] - xol) + y(xol)
        al[map3] = LogSig.df(ol)
        gl[map3] = LogSig.f(ol) - al[map3]*ol
        # constraint 4: y[index] <= y'(xou) * (x[index] - xou) + y(xou)
        au[map3] = LogSig.df(ou)
        gu[map3] = LogSig.f(ou) - au[map3]*ou

        Du = np.diag(au)
        Dl = np.diag(al)
        return Dl, Du, gl, gu
    

    def reachApprox_sparse(I, delta=0.98, opt=True, lp_solver='gurobi', RF=0.0, DR=0, show=False):
        
        assert isinstance(I, SparseStar), 'error: input set is not a SparseStar set'

        N = I.dim

        l, u = I.getRanges(lp_solver=lp_solver, RF=RF, layer='logsig', delta=delta)
        l = l.reshape(N, 1)
        u = u.reshape(N, 1)

        yl = LogSig.f(l)
        yu = LogSig.f(u)
        dyl = LogSig.df(l)
        dyu = LogSig.df(u)

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
            dyo = LogSig.df(xo)
            C14 = sp.hstack((Z, -dyo*I.X(map_), A0[map_, :]))
            d14 = dyo*(I.c(map_) - xo) + LogSig.f(xo)

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
            dyo = LogSig.df(xo)
            C24 = sp.hstack((Z, dyo*I.X(map_), -A0[map_, :]))
            d24 = -dyo*(I.c(map_) - xo) - LogSig.f(xo)

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
                xou = LogSig.optimal_iter_approx_upper(l_, u_)
                xol = LogSig.optimal_iter_approx_lower(l_, u_)
                dxou = LogSig.df(xou)
                dxol = LogSig.df(xol)
                
                # constraint 3: y[index] >= y'(xol)*(x - xol) + y(xol)
                C33 = sp.hstack((Z, dxol*I.X(map1), -A0[map1, :]))
                d33 = -dxol*(I.c(map1) - xol) - LogSig.f(xol)
                
                # constraint 4: y[index] <= y'(xou)*(x - xou) + y(xou)
                C34 = sp.hstack((Z, -dxou*I.X(map1), A0[map1, :]))
                d34 = dxou*(I.c(map1) - xou) + LogSig.f(xou)

                C3 = sp.vstack((C31, C32, C33, C34)).tocsc()
                d3 = np.vstack((d31, d32, d33, d34)).reshape(-1)

            else:
                gux = (yu_ - dmin * u_ - 0.5) / (0.25 - dmin)
                guy = 0.25 * gux + 0.5 
                glx = (yl_ - dmin * l_ - 0.5) / (0.25 - dmin)
                gly = 0.25 * glx + 0.5

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
        new_pred_depth = np.hstack((I.pred_depth + 1, np.zeros(m)))
        
        S = SparseStar(new_A, new_C, new_d, new_pred_lb, new_pred_ub, new_pred_depth)
        if DR > 0:
            if show:
                if (S.pred_depth >= DR).any():
                    print('Applying depth reduction {}'.format(DR))
            S = S.depthReduction(DR=DR)
        return S
    
    def reach(I, opt=False, delta=0.98, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        if isinstance(I, SparseStar):
            return LogSig.reachApprox_sparse(I=I, opt=opt, delta=delta, lp_solver=lp_solver, RF=RF, DR=DR, show=show)
        elif isinstance(I, Star):
            # return LogSig.reachApproxStar(I, lp_solver, RF)
            raise Exception('error: under development')
        else:
            raise Exception('error: unknown input set')


    # used for GRU and LSTM ( logsigXtansig, (1-logsig)Xtansig, losigXidentity )
    def getConstraints(c, X, A0, l, u, nZVars,opt=False):
    
        N = l.shape[0]
        
        yl = LogSig.f(l)
        yu = LogSig.f(u)
        dyl  = LogSig.df(l)
        dyu =  LogSig.df(u)
        
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
            dyo = LogSig.df(xo)
            C14 = sp.hstack((Z, -dyo*X[map_, :], A0[map_, :]))
            d14 = dyo*(c[map_] - xo) + LogSig.f(xo)

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
            C21 = sp.hstack((Z, dyl_*X[map_, :], -A0[map_, :]))
            d21 = -dyl_*(c[map_] - l_) - yl_

            # constraint 2: y >= y'(u) * (x - u) + y(u)
            C22 = sp.hstack((Z, dyu_*X[map_, :], -A0[map_, :]))
            d22 = -dyu_*(c[map_] - u_) - yu_

            # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
            g = (yu_ - yl_) / (u_ - l_)
            C23 = sp.hstack((Z, -g*X[map_, :], A0[map_, :]))
            d23 = g*(c[map_] - l_) + yl_

            # xo = (u*u - l*l) / (2*(u-l))
            # constraint 4: y >= y'(xo)*(x - xo) + y(xo) 
            # xo = (u_*u_ - l_*l_) / (2*(u_ - l_))
            xo = 0.5*(u_ + l_)
            dyo = LogSig.df(xo)
            C24 = sp.hstack((Z, dyo*X[map_, :], -A0[map_, :]))
            d24 = -dyo*(c[map_] - xo) - LogSig.f(xo)

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
                xou = LogSig.optimal_iter_approx_upper(l_, u_)
                xol = LogSig.optimal_iter_approx_lower(l_, u_)
                dxou = LogSig.df(xou)
                dxol = LogSig.df(xol)

                # constraint 3: y[index] >= y'(xol)*(x - xol) + y(xol)
                C33 = sp.hstack((Z, dxol*X[map1, :], -A0[map1, :]))
                d33 = -dxol*(c[map1] - xol) - LogSig.f(xol)

                # constraint 4: y[index] <= y'(xou)*(x - xou) + y(xou)
                C34 = sp.hstack((Z, -dxou*X[map1, :], A0[map1, :]))
                d34 = dxou*(c[map1] - xou) + LogSig.f(xou)

                C3 = sp.vstack((C31, C32, C33, C34)).tocsc()
                d3 = np.vstack((d31, d32, d33, d34)).reshape(-1)

            else:
                gux = (yu_ - dmin * u_ - 0.5) / (0.25 - dmin)
                guy = 0.25 * gux + 0.5 
                glx = (yl_ - dmin * l_ - 0.5) / (0.25 - dmin)
                gly = 0.25 * glx + 0.5

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