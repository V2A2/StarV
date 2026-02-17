"""
Cosine Class (Cosine function)

Author: Zhuoyang Zhou
Date: 02/09/2026 Update: 02/14/2026
"""

# !/usr/bin/python3
import numpy as np
from StarV.set.star import Star
from StarV.set.imagestar import ImageStar


class Cosine(object):
    """
    Cosine Class for reachability analysis
    Activation function: f(x) = cos(x)
    
    Handles 8 regions based on monotonicity and convexity:
    Region 1: -3π/2 < lb < ub < -π         (decreasing, convex)
    Region 2: -3π/2 < lb < -π < ub < -π/2  (crossing minimum, convex)
    Region 3: -π < lb < ub < -π/2          (increasing, convex)
    Region 4: -π < lb < -π/2 < ub < 0      (crossing zero, changing convexity)
    Region 5: -π/2 < lb < ub < 0           (increasing, concave)
    Region 6: -π/2 < lb < 0, 0 < ub < π/2  (crossing maximum, concave)
    Region 7:  0 < lb < ub < π/2           (decreasing, concave)
    Region 8:  0 < lb < π/2 < ub < π       (crossing zero, changing convexity)
    """

    @staticmethod
    def evaluate(x):
        """Evaluate cosine function"""
        return Cosine.f(x)

    @staticmethod
    def f(x):
        """Cosine activation function: f(x) = cos(x)"""
        return np.cos(x)

    @staticmethod
    def df(x):
        """Derivative of cosine function: f'(x) = -sin(x)"""
        return -np.sin(x)

    def reachApprox_star(I, opt=True, lp_solver='gurobi', RF =0.0):

        assert isinstance(I, Star), 'error: input set is not a Star set'

        N = I.dim

        # Get input ranges for each dimension
        l, u = I.getRanges(lp_solver=lp_solver, RF=RF)
        
        # ------------------------------------------------------------------------
        # Periodic range normalization for cosine:
        # Shift each interval [l_i, u_i] by 2π*k_i so that it falls into
        # the base window [A, B] = [-3π/2, π].
        # ------------------------------------------------------------------------
        A = -3.0 * np.pi / 2.0
        B =  1.0 * np.pi
        T =  2.0 * np.pi

        # Find integer k such that:
        #   l - T*k >= A  and  u - T*k <= B
        # => k <= (l - A)/T  and  k >= (u - B)/T
        k_low  = np.ceil((u - B) / T)      # minimal k to push u down into <= B
        k_high = np.floor((l - A) / T)     # maximal k to keep l above >= A

        # pick k = k_low (should be feasible under width<=B-A)
        k = k_low.astype(np.int64)

        # safety fallback if infeasible (unexpected wide intervals)
        bad = k_low > k_high
        if np.any(bad):
            mid = 0.5 * (l + u)
            k_mid = np.floor((mid - A) / T).astype(np.int64)
            k[bad] = k_mid[bad]

        # apply shift
        shift = T * k
        l = l - shift
        u = u - shift

        # Compute sine function values and derivatives at bounds
        yl, yu = Cosine.f(l), Cosine.f(u)
        dyl, dyu = Cosine.df(l), Cosine.df(u)

        # Identify neurons where l != u (non-constant inputs)
        map0 = np.where(l != u)[0]
        m = len(map0)

        # Create new generator matrix for output variables
        V0 = np.zeros((N, m))
        for i in range(m):
            V0[map0[i], i] = 1

        # Properly preserve the original basis matrix structure
        new_V = np.hstack([np.zeros([N, 1]), np.zeros([N, I.nVars]), V0])

        # Handle constant neurons (l == u)
        map1 = np.where(l == u)[0]
        if len(map1):
            new_V[map1, 0] = yl[map1]
            new_V[map1, 1:] = 0

        nv = I.nVars + m

        # Region 1: -3π/2 < lb < ub < -π
        map1 = np.where((l[map0] >= -3*np.pi/2) & (u[map0] <= -np.pi))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Lower bound: tangents at l and u
            # y >= -sin(l)*(x - l) + cos(l)
            C11 = np.hstack([dyl_diag @ V1, -V2])
            d11 = -dyl_ * (c1 - l_) - yl_

            # y >= -sin(u)*(x - u) + cos(u)
            C12 = np.hstack([dyu_diag @ V1, -V2])
            d12 = -dyu_ * (c1 - u_) - yu_

            # Upper bound: secant line
            # y <= (cos(u) - cos(l)) / (u - l) * (x - l) + cos(l)
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C13 = np.hstack([-g_diag @ V1, V2])
            d13 = g * (c1 - l_) + yl_

            # Additional lower bound: tangent at midpoint
            xo = 0.5 * (u_ + l_)
            dyo = Cosine.df(xo)
            dyo_diag = np.diag(dyo.flatten())
            C14 = np.hstack([dyo_diag @ V1, -V2])
            d14 = -dyo * (c1 - xo) - Cosine.f(xo)

            C1 = np.vstack((C11, C12, C13, C14))
            d1 = np.hstack((d11, d12, d13, d14))
        else:
            C1 = np.empty((0, nv))
            d1 = np.empty((0))

        # Region 2: -3π/2 < lb < -π, -π < ub < -π/2
        map1 = np.where((l[map0] >= -3*np.pi/2) & (l[map0] < -np.pi) & (u[map0] > -np.pi) & (u[map0] <= -np.pi/2))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Lower bound: Must include minimum value -1
            # Use a simple lower bound: y >= -1
            C21 = np.hstack([np.zeros((len(map_), V1.shape[1])), -V2])
            d21 = np.ones(len(map_))

            # Additional lower bound: tangents at endpoints
            C22 = np.hstack([dyl_diag @ V1, -V2])
            d22 = -dyl_ * (c1 - l_) - yl_

            C23 = np.hstack([dyu_diag @ V1, -V2])
            d23 = -dyu_ * (c1 - u_) - yu_

            # Upper bound: secant line
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C24 = np.hstack([-g_diag @ V1, V2])
            d24 = g * (c1 - l_) + yl_

            C2 = np.vstack((C21, C22, C23, C24))
            d2 = np.hstack((d21, d22, d23, d24))
        else:
            C2 = np.empty((0, nv))
            d2 = np.empty((0))

        # Region 3: -π < lb < ub < -π/2
        map1 = np.where((l[map0] >= -np.pi) & (u[map0] <= -np.pi/2))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Lower bounds: tangents
            C31 = np.hstack([dyl_diag @ V1, -V2])
            d31 = -dyl_ * (c1 - l_) - yl_

            C32 = np.hstack([dyu_diag @ V1, -V2])
            d32 = -dyu_ * (c1 - u_) - yu_

            # Upper bound: secant
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C33 = np.hstack([-g_diag @ V1, V2])
            d33 = g * (c1 - l_) + yl_

            # Additional lower bound: midpoint tangent
            xo = 0.5 * (u_ + l_)
            dyo = Cosine.df(xo)
            dyo_diag = np.diag(dyo.flatten())
            C34 = np.hstack([dyo_diag @ V1, -V2])
            d34 = -dyo * (c1 - xo) - Cosine.f(xo)

            C3 = np.vstack((C31, C32, C33, C34))
            d3 = np.hstack((d31, d32, d33, d34))
        else:
            C3 = np.empty((0, nv))
            d3 = np.empty((0))

        # Region 4: -π < lb < -π/2 < ub < 0
        map1 = np.where((l[map0] >= -np.pi) & (l[map0] < -np.pi / 2) & (u[map0] > -np.pi / 2) & (u[map0] <= 0))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]         
            dyl_, dyu_ = dyl[map_], dyu[map_]  
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            # dmin = min(f'(l), f'(u))
            dmin = np.minimum(dyl_, dyu_)
            dmin_diag = np.diag(dmin.flatten())

            # constraint 1: y >= dmin * (x - l) + f(l)
            C41 = np.hstack([dmin_diag @ V1, -V2])
            d41 = -dmin * (c1 - l_) - yl_

            # constraint 2: y <= dmin * (x - u) + f(u)
            C42 = np.hstack([-dmin_diag @ V1, V2])
            d42 = dmin * (c1 - u_) + yu_

            if opt == True:
                xou = Cosine.optimal_iter_approx_upper(l_, u_)
                xol = Cosine.optimal_iter_approx_lower(l_, u_)
                dxou = Cosine.df(xou)   # cos(xou)
                dxol = Cosine.df(xol)   # cos(xol)

                dxol_diag = np.diag(dxol.flatten())
                dxou_diag = np.diag(dxou.flatten())

                # constraint 3: y >= f'(xol) * (x - xol) + f(xol)
                C43 = np.hstack([dxol_diag @ V1, -V2])
                d43 = -dxol * (c1 - xol) - Cosine.f(xol)

                # constraint 4: y <= f'(xou) * (x - xou) + f(xou)
                C44 = np.hstack([-dxou_diag @ V1, V2])
                d44 = dxou * (c1 - xou) + Cosine.f(xou)

            else:
                # Non-optimal branch (direct port)
                # eps = 1e-12
                # denom = np.maximum(1 - dmin, eps)
                b = np.pi / 2
                denom = 1 - dmin

                gux = (yu_ - dmin * u_ - b) / denom
                guy = gux + b
                glx = (yl_ - dmin * l_ - b) / denom
                gly = glx + b

                mu = (yl_ - guy) / (l_ - gux)
                ml = (yu_ - gly) / (u_ - glx)

                mu_diag = np.diag(mu.flatten())
                ml_diag = np.diag(ml.flatten())

                # constraint 3: y >= ml * (x - u) + f(u)
                C43 = np.hstack([ml_diag @ V1, -V2])
                d43 = -ml * (c1 - u_) - yu_

                # constraint 4: y <= mu * (x - l) + f(l)
                C44 = np.hstack([-mu_diag @ V1, V2])
                d44 = mu * (c1 - l_) + yl_

            C4 = np.vstack((C41, C42, C43, C44))
            d4 = np.hstack((d41, d42, d43, d44))

        else:
            C4 = np.empty((0, nv))
            d4 = np.empty((0))

        # Region 5: -π/2 < lb < ub < 0
        map1 = np.where((l[map0] >= -np.pi / 2) & (u[map0] <= 0))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Upper bounds: tangents
            C51 = np.hstack([-dyl_diag @ V1, V2])
            d51 = dyl_ * (c1 - l_) + yl_

            C52 = np.hstack([-dyu_diag @ V1, V2])
            d52 = dyu_ * (c1 - u_) + yu_

            # Lower bound: secant
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C53 = np.hstack([g_diag @ V1, -V2])
            d53 = -g * (c1 - l_) - yl_

            # Additional upper bound: midpoint tangent
            xo = 0.5 * (u_ + l_)
            dyo = Cosine.df(xo)
            dyo_diag = np.diag(dyo.flatten())
            C54 = np.hstack([-dyo_diag @ V1, V2])
            d54 = dyo * (c1 - xo) + Cosine.f(xo)

            C5 = np.vstack((C51, C52, C53, C54))
            d5 = np.hstack((d51, d52, d53, d54))
        else:
            C5 = np.empty((0, nv))
            d5 = np.empty((0))

        # Region 6: -π/2 < lb < 0, 0 < ub < π/2
        map1 = np.where((l[map0] >= -np.pi / 2) & (l[map0] < 0) & (u[map0] > 0) & (u[map0] <= np.pi / 2))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Upper bound: Must not exceed maximum value 1
            # y <= 1
            C61 = np.hstack([np.zeros((len(map_), V1.shape[1])), V2])
            d61 = np.ones(len(map_))

            # Additional upper bounds: tangents at endpoints
            C62 = np.hstack([-dyl_diag @ V1, V2])
            d62 = dyl_ * (c1 - l_) + yl_

            C63 = np.hstack([-dyu_diag @ V1, V2])
            d63 = dyu_ * (c1 - u_) + yu_

            # Lower bound: secant line
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C64 = np.hstack([g_diag @ V1, -V2])
            d64 = -g * (c1 - l_) - yl_

            C6 = np.vstack((C61, C62, C63, C64))
            d6 = np.hstack((d61, d62, d63, d64))
        else:
            C6 = np.empty((0, nv))
            d6 = np.empty((0))

        # Region 7: 0 < lb < ub < π/2
        map1 = np.where((l[map0] >= 0) & (u[map0] <= np.pi / 2))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]
            dyl_, dyu_ = dyl[map_], dyu[map_]
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            dyl_diag = np.diag(dyl_.flatten())
            dyu_diag = np.diag(dyu_.flatten())

            # Upper bounds: tangents
            C71 = np.hstack([-dyl_diag @ V1, V2])
            d71 = dyl_ * (c1 - l_) + yl_

            C72 = np.hstack([-dyu_diag @ V1, V2])
            d72 = dyu_ * (c1 - u_) + yu_

            # Lower bound: secant
            g = (yu_ - yl_) / (u_ - l_)
            g_diag = np.diag(g.flatten())
            C73 = np.hstack([g_diag @ V1, -V2])
            d73 = -g * (c1 - l_) - yl_

            # Additional upper bound: midpoint tangent
            xo = 0.5 * (u_ + l_)
            dyo = Cosine.df(xo)
            dyo_diag = np.diag(dyo.flatten())
            C74 = np.hstack([-dyo_diag @ V1, V2])
            d74 = dyo * (c1 - xo) + Cosine.f(xo)

            C7 = np.vstack((C71, C72, C73, C74))
            d7 = np.hstack((d71, d72, d73, d74))
        else:
            C7 = np.empty((0, nv))
            d7 = np.empty((0))

        # Region 8: 0 < lb < π/2 < ub < π
        map1 = np.where((l[map0] >= 0) & (l[map0] < np.pi / 2) & (u[map0] > np.pi / 2) & (u[map0] <= np.pi))[0]
        if len(map1):
            map_ = map0[map1]
            l_, u_ = l[map_], u[map_]
            yl_, yu_ = yl[map_], yu[map_]          # sin(l_), sin(u_)
            dyl_, dyu_ = dyl[map_], dyu[map_]      # cos(l_), cos(u_) (negative in this region)
            c1, V1 = I.V[map_, 0], I.V[map_, 1:]
            V2 = V0[map_, :]

            # dmin = min(f'(l), f'(u))  (both are negative here)
            dmin = np.minimum(dyl_, dyu_)
            dmin_diag = np.diag(dmin.flatten())

            # constraint 1: y >= dmin * (x - l) + f(l)
            C81 = np.hstack([dmin_diag @ V1, -V2])
            d81 = -dmin * (c1 - l_) - yl_

            # constraint 2: y <= dmin * (x - u) + f(u)
            C82 = np.hstack([-dmin_diag @ V1, V2])
            d82 = dmin * (c1 - u_) + yu_

            if opt == True:
                xou = Cosine.optimal_iter_approx_upper(l_, u_)
                xol = Cosine.optimal_iter_approx_lower(l_, u_)
                dxou = Cosine.df(xou)
                dxol = Cosine.df(xol)

                dxol_diag = np.diag(dxol.flatten())
                dxou_diag = np.diag(dxou.flatten())

                # constraint 3: y >= f'(xol) * (x - xol) + f(xol)
                C83 = np.hstack([dxol_diag @ V1, -V2])
                d83 = -dxol * (c1 - xol) - Cosine.f(xol)

                # constraint 4: y <= f'(xou) * (x - xou) + f(xou)
                C84 = np.hstack([-dxou_diag @ V1, V2])
                d84 = dxou * (c1 - xou) + Cosine.f(xou)

            else:
                b = np.pi / 2
                denom = 1 + dmin

                gux = (b - yu_ + dmin * u_) / denom
                guy = -gux + b

                glx = (b - yl_ + dmin * l_) / denom
                gly = -glx + b

                # Slopes for the other two sides
                mu = (yl_ - guy) / (l_ - gux)
                ml = (yu_ - gly) / (u_ - glx)

                mu_diag = np.diag(mu.flatten())
                ml_diag = np.diag(ml.flatten())

                # constraint 3: y >= ml * (x - u) + y(u)
                C83 = np.hstack([ml_diag @ V1, -V2])
                d83 = -ml * (c1 - u_) - yu_

                # constraint 4: y <= mu * (x - l) + y(l)
                C84 = np.hstack([-mu_diag @ V1, V2])
                d84 = mu * (c1 - l_) + yl_

            C8 = np.vstack((C81, C82, C83, C84))
            d8 = np.hstack((d81, d82, d83, d84))

        else:
            C8 = np.empty((0, nv))
            d8 = np.empty((0))

        # ========================================================================
        # Combine all constraints
        # ========================================================================
        n = I.C.shape[0]
        if len(I.d):
            C0 = np.hstack([I.C, np.zeros([n, m])])
            d0 = I.d
        else:
            C0 = np.empty([0, I.nVars + m])
            d0 = np.empty([0])

        new_C = np.vstack((C0, C1, C2, C3, C4, C5, C6, C7, C8))
        new_d = np.hstack((d0, d1, d2, d3, d4, d5, d6, d7, d8))

        # new_pred_lb = np.hstack((I.pred_lb, yl[map0]))
        # new_pred_ub = np.hstack((I.pred_ub, yu[map0]))

        # ---- Robust β-predicate bounds for all regions (single pass) ----
        idx = map0  # Only set bounds for the newly added m β dimensions
        yL = np.minimum(yl[idx], yu[idx])  # Get min/max by endpoint
        yU = np.maximum(yl[idx], yu[idx])

        # If the interval passes through the global minimum/maximum point, clip it to -1/+1
        cross_min = (l[idx] < -np.pi) & (u[idx] > -np.pi)  # Passing cos minimum -1
        cross_max = (l[idx] < 0) & (u[idx] > 0)  # Passing cos maximum +1
        if np.any(cross_min):
            yL[cross_min] = -1.0
        if np.any(cross_max):
            yU[cross_max] = 1.0

        new_pred_lb = np.hstack((I.pred_lb, yL))
        new_pred_ub = np.hstack((I.pred_ub, yU))

        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)

    @staticmethod
    def reach(I, opt=False, lp_solver='gurobi', RF=0.0):
        """
        Main entry point for reachability analysis with sine activation

        Parameters:
        -----------
        I : Star or ImageStar
            Input set
        opt : bool
            Whether to use optimal approximation (reserved)
        lp_solver : str
            Linear programming solver
        RF : float
            Relaxation factor

        Returns:
        --------
        Star or ImageStar
            Output reachable set
        """
        if isinstance(I, Star):
            return Cosine.reachApprox_star(I, opt=opt, lp_solver=lp_solver, RF=RF)
        elif isinstance(I, ImageStar):
            shape = I.shape()
            S = Cosine.reachApprox_star(I.toStar(), opt=opt, lp_solver=lp_solver, RF=RF)
            return S.toImageStar(image_shape=shape, copy_=False)
        else:
            raise Exception('error: unknown input set type')