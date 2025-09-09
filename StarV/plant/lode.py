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

    Continuous Linear ODE class with Star reachability methods
    
    Dung Tran: 11/29/2022

    Method is from Stanley Bak paper and the tool Hylaa: 

    link: Hylaa: https://github.com/stanleybak/hylaa/blob/master/hylaa/time_elapse_expm.py

    paper: Simlulation-Equivalent Reachability of Large Linear Systems with Inputs, CAV2017
    
"""

from scipy.signal import lti, step, impulse, lsim, cont2discrete
from StarV.plant.dlode import DLODE
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csr_matrix
# import warnings

class LODE(object):
    """
       Continuous Linear ODE class
       Dung Tran: 11/29/2022
      ===========================
       x'[t] = Ax[t] + Bu[t]
       y[t] = Cx[t] + Du[t]
      ===========================

    """

    def __init__(self, A, B=None, C=None, D=None):
        """
         Key Attributes: 
         A, B, C, D: system matrices
         dim: system demension
         nI: number of inputs
         nO: number of outputs
        """
        isinstance(A, np.ndarray), 'error: matrix A should be a numpy array'
        if B is not None:
            isinstance(B, np.ndarray), 'error: matrix B should be a numpy array'
            assert A.shape[0] == B.shape[0], 'error: inconsistent dimension between A and B'

        if C is not None:
            isinstance(C, np.ndarray), 'error: matrix C should be a numpy array'
        if D is not None:
            isinstance (D, np.ndarray), 'error: matrix D should be a numpy array'

        
        if D is not None and C is not None:
            assert C.shape[0] == D.shape[0], 'error: inconsistent dimensions between C and D'
            assert D.shape[1] == B.shape[1], 'error: inconsistent dimensions between B and D'
            assert C.shape[1] == A.shape[0], 'error: inconsistent dimensions between A and C'

        self.A = A
        self.dim = A.shape[0]
        
        self.B = B
        if B is None:
            self.nI = 0
            B1 = np.zeros((self.dim, 1))
        else:
            self.nI = B.shape[1]
            B1 = B
        
        if C is None:
            self.C = np.eye(self.dim)
            self.nO = self.dim
        else:
            self.C = C
            self.nO = C.shape[0]


        self.D = D
        if D is None:
            D1 = np.zeros((self.nO, B1.shape[1]))
            self.plant = lti(A, B1, self.C, D1)
        else:
            self.plant = lti(A, B1, self.C, D)

        self.gA = None # used for one-step reachability gA = e^{A*dt} 
        self.gB = None # used for one-step reachability gB = G(A,h)B in the CAV2017 paper
        self.dt = None # time step used for reachability

    def __str__(self):
        """Print information of the system"""

        print('\n========= LODE ==========\n')
        print('\n Plant Matrices:')
        print('\n A = {}'.format(self.A))
        print('\n B = {}'.format(self.B))
        print('\n C = {}'.format(self.C))
        print('\n D = {}'.format(self.D))

        print('\n Number of inputs: {}'.format(self.nI))
        print('\n Number of outputs: {}'.format(self.nO))
        print('')
        return '\n'


    def info(self):
        print(self)

    def stepResponse(self, x0=None, t=None, n=None):

        """
        Step Response

        Inputs:
          x0: initial state-vector, defaults to zero 
          t : time points, computed if not given
          n : the number of time points to compute (if t is not given)
        
        Outputs:

          t: time values for the output, 1-D array
          y: system response
        
        """

        t, y = step(self.plant, x0, t, n)

        return t, y

    
    def impulseResponse(self, x0=None, t=None, n=None):

        """
        impulse Response

        Inputs:
          x0: initial state-vector, defaults to zero 
          t : time points, computed if not given
          n : the number of time points to compute (if t is not given)
        
        Outputs:

          t: time values for the output, 1-D array
          y: system response

        """

        t, y = impulse(self.plant, x0, t, n)

        return t, y

    def sim(self, u, t=None, x0=None):
        """
        Simulate output of a linear system

        Inputs:
          u: input array describing the input at each time t
             If there are multiple inputs, then each column 
             represents an input

          t: time steps at which the input is defined

          x0: initial conditions on the state vector (zero by default)

        Outputs:

          tout: time values for the output

          yout: system response
        
          xout: time-evolution of the state-vector
       
        """

        tout, yout, xout = lsim(self.plant, u, t, x0)

        return tout, yout, xout


    def toDLODE(self, dt, method='zoh', alpha=None):

        sysd = cont2discrete((self.A, self.B, self.C, self.D), dt, method, alpha)

        sys = DLODE(sysd[0], sysd[1], sysd[2], sysd[3], dt)

        return sys
        

    def stepReach(self, dt, X0=None, U=None, subSetPredicate=False):
        """ step reachability

            Assumption: U is a constant in a single step, U can be a set

            X(t) = e^{At}X0 + int_0^t(e^{A*(t-tau)}Bu(tau))dtau  

        Inputs:
            @dt: time step in which U0 is constant
            @X0: initial condition, should be a set (a Star or ProbStar) or a vector (numpy array)
            @U: control input, should be a set or a vector
            @subSetPredicate: indicate that the predicate of U0 is a subset of X0's predicate (happen in reachability of NNCS)

        Outputs:
            @Xt: state set

        Author: Dung Tran: 12/20/2022

        """

        if self.gA is None:
            self.compute_gA_gB(dt)


        if isinstance(X0, np.ndarray):
            X1 = np.matmul(self.gA, X0)
        elif isinstance(X0, ProbStar) or isinstance(X0, Star):
            X1 = X0.affineMap(A=self.gA)
        elif X0 is None:
            X1 = np.zeros((self.dim,))
        else:
            raise RuntimeError('Unknown datatype for X0')

        if isinstance(U, np.ndarray):
            U1 = np.matmul(self.gB, U)
        elif isinstance(U, ProbStar) or isinstance(U, Star):
            U1 = U.affineMap(A=self.gB)
        elif U is None:
            U1 = np.zeros((self.dim,))
        else:
            raise RuntimeError('Unknown datatype for U')

        Xt = None
        if subSetPredicate is True:  # used in NNCS reachability analysis

            # Minkowski Sum of X1 and U1 does not increase the number of predicate variable

            if isinstance(X1, ProbStar) or isinstance(X1, Star):
                if isinstance(U1, np.ndarray):
                    Xt = X1.affineMap(b=U1)
                else:
                    newV = X1.V + U1.V
                    if isinstance(U1, ProbStar):
                        Xt = ProbStar(newV, U1.C, U1.d, U1.mu, U1.Sig, U1.pred_lb, U1.pred_ub)
                    else:
                        Xt = Star(newV, U1.C, U1.d, U1.pred_lb, U1.pred_ub)
            else:
                if isinstance(U1, np.ndarray):
                    Xt = X1 + U1
                else:
                    Xt = U1.affineMap(b=X1)
                
        else:

            if isinstance(X1, ProbStar) or isinstance(X1, Star):
                if isinstance(U1, np.ndarray):
                    Xt = X1.affineMap(b=U1)
                else:
                    Xt = X1.minKowskiSum(U1)
                    
            else:
                if isinstance(U1, np.ndarray):
                    Xt = X1 + U1
                else:
                    Xt = U1.affineMap(b=X1)
               
        return Xt


    def compute_gA_gB(self, dt):
        """
          Compute one_step_matrix_exp: gA = e^{A*dt}
          Compute one_step_input_effects_matrix: gB = G(A,h)B = ((1-e^{A*dt})/A)*B
        """

        assert dt > 0, 'error: Invalid time step'
        self.dt = dt

        self.gA = expm(self.A * dt)

        if self.B is not None:
            A1 = np.hstack((self.A, self.B))
            A1 = np.vstack((A1, np.zeros((self.nI, self.dim + self.nI))))
            A1 = csr_matrix(A1)
            A1 = A1*self.dt
            Z1 = np.zeros((self.dim, self.nI))
            I1 = np.eye(self.nI)
            C = np.vstack((Z1, I1))  # init vecs
            col = expm_multiply(A1, C) 
            self.gB = col[:self.dim]
        else:
            self.gB = None
        

    def multiStepReach(self, dt, X0=None, U=None, k=1):
        """
        Reachability of LODE in multisteps

        Inputs:

          @dt: timestep, control input is updated (as a constant) every timestep
          @X0: initial condition, can be a state vector or a set (Star or ProbStar)
          @U: input set

          @k: number of steps 

        Outputs:

          @Xt: sequence of state vectors or sets

        ===========================================================================

        Following algorithm 1 in the CAV2017 paper

        
        Author: Dung Tran
        Date: 12/16/2022

        """

        assert k >= 1, 'error: invalid number of steps'
        Xt = []  # reachable set of states

        if X0 is None:
            Xt.append(np.zeros(self.dim, ))
        else:
            Xt.append(X0)
        
        for i in range(1,k+1):
            X1 = self.stepReach(dt, X0=Xt[i-1], U=U)
            Xt.append(X1)
        
        return Xt

    @staticmethod
    def rand(dim, nI, nO=None, dt=0.1):
        """Randomly generate a DLODE"""

        assert dim > 0, 'error: invalid dimension'
        assert nI >= 0, 'error: invalid number of inputs'
        A = np.random.rand(dim, dim)
        if nO is not None:
            assert nO > 0, 'error: invalid number of outputs'

        if nO is not None:
            C = np.random.rand(nO, dim)
        else:
            C = np.eye(dim)

        if nI == 0:
            B = None
            D = None
        else:
            B = np.random.rand(dim, nI)
            if nO is not None:
                D = np.random.rand(nO, nI)
            else:
                D = np.random.rand(dim, nI)

        res = LODE(A, B, C, D)

        return res 

       
        
        
    
