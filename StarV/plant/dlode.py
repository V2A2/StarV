""" Discrete Linear ODE class with Star reachability methods
    Dung Tran: 11/21/2022
"""

from scipy.signal import dlti, dstep, dimpulse, dlsim
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
import warnings

class DLODE(object):
    """
       Discrete Linear ODE class
       Dung Tran: 11/21/2022
      ===========================
       x[k+1] = Ax[k] + Bu[k]
       y[k+1] = Cx[k] + Du[k]
      ===========================

    """

    def __init__(self, A, B=None, C=None, D=None, dt=0.1):
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
            self.plant = dlti(A, B1, self.C, D1)
        else:
            self.plant = dlti(A, B1, self.C, D)

    def info(self):
        """Print information of the system"""

        print('\n========= DLODE ==========\n')
        print('\n Plant Matrices:')
        print('\n A = {}'.format(self.A))
        print('\n B = {}'.format(self.B))
        print('\n C = {}'.format(self.C))
        print('\n D = {}'.format(self.D))

        print('\n Number of inputs: {}'.format(self.nI))
        print('\n Number of outputs: {}'.format(self.nO))

    def stepResponse(self, x0=None, t=None, n=None):

        """
        Step Response

        Inputs:
          x0: initial state-vector, defaults to zero 
          t : time points, computed if not given
          n : the nmber of time points to compute (if t is not given)
        
        Outputs:

          t: time values for the output, 1-D array
          y: system response
        
        """

        t, y = dstep(self.plant, x0, t, n)

        return t, y

    
    def impulseResponse(self, x0=None, t=None, n=None):

        """
        impulse Response

        Inputs:
          x0: initial state-vector, defaults to zero 
          t : time points, computed if not given
          n : the nmber of time points to compute (if t is not given)
        
        Outputs:

          t: time values for the output, 1-D array
          y: system response

        """

        t, y = dimpulse(self.plant, x0, t, n)

        return t, y

    def sim(self, u, t=None, x0=None):
        """
        Simulate output of a discrete-time linear system

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

        tout, yout, xout = dlsim(self.plant, u, t, x0)

        return tout, yout, xout

    def stepReach(self, X0=None, U=None, subSetPredicate=False):
        """ step reachability

            X1 = AX0 + BU
            Y1 = CX0 + DU

        Inputs:
            @X0: initial condition, should be a set (a Star or ProbStar) or a vector (numpy array)
            @U: control input, should be a set or a vector
            @subSetPredicate: indicate that the predicate of U0 is a subset of X0's predicate (happen in reachability of NNCS)

        Outputs:
            @X1: state set
            @Y1: output set

        Author: Dung Tran: 11/24/2022

        """

        
        X1 = None
        Y1 = None
        
        if X0 is None:
            X11 = np.zeros((self.dim,))
            Y11 = np.zeros((self.nO,))
        else:
            
            if isinstance(X0, np.ndarray):
                assert X0.shape[0] == self.dim, 'error: inconsistent dimensions between initial condition state and the plant'
                X11 = np.matmul(self.A, X0)
                Y11 = np.matmul(self.C, X0)
            else:
                assert isinstance(X0, ProbStar) or isinstance(X0, Star), 'error: initial condition should be a star/probstar or a vector'
                X11 = X0.affineMap(self.A)
                Y11 = X0.affineMap(self.C)

        if self.nI == 0 and U is not None:
            warnings.warn("plant has no input, U will not be used", SyntaxWarning)
            X1 = X11
            Y1 = Y11

        else:

            if U is None:
                X1 = X11
                Y1 = Y11
                                
            elif isinstance(U, np.ndarray):
                assert U.shape[0] == self.nI, 'error: inconsistent dimensions between input vector and the plant'
                U11 = np.matmul(self.B, U)
                if self.D is not None:
                    U12 = np.matmul(self.D, U)
                else:
                    U12 = None

                if isinstance(X11, np.ndarray):
                    X1 = X11 + U11
                    if U12 is not None:
                        Y1 = Y11 + U12
                    else:
                        Y1 = Y11
                else:
                    X1 = X11.affineMap(b=U11)
                    if U12 is not None:
                        Y1 = Y11.affineMap(b=U12)
                    else:
                        Y1 = Y11

            else:
                assert isinstance(U, ProbStar) or isinstance(U, Star), 'error: control input should be a vector or a star or a probstar'
                U11 = U.affineMap(self.B)
                if self.D is not None:
                    U12 = U.affineMap(self.D)
                else:
                    U12 = None

                if isinstance(X11, np.ndarray):
                    X1 = U11.affineMap(b=X11)
                    if U12 is not None:
                        Y1 = U12.affineMap(b=Y11)
                    else:
                        Y1 = Y11

                else:
                    if subSetPredicate:
                        VX = X11.V + U11.V
                        if isinstance(U11, Star):
                            X1 = Star(VX, U11.C, U11.d, U11.pred_lb, U11.pred_ub)
                        else:
                            X1 = ProbStar(VX, U11.C, U11.d, U11.mu, U11.Sig, U11.pred_lb, U11.pred_ub)
                        if U12 is not None:
                            VY = Y11.V + U12.V
                            if isinstance(U12, Star):
                                Y1 = Star(VY, U12.C, U12.d, U12.pred_lb, U12.pred_ub)
                            else:
                                Y1 = ProbStar(VY, U12.C, U12.d, U12.mu, U12.Sig, U12.pred_lb, U12.pred_ub)
                        else:
                            Y1 = Y11
                    else:
                        X1 = X11.minKowskiSum(U11)
                        if U12 is not None:
                            Y1 = Y11.minKowskiSum(U12)
                        else:
                            Y1 = Y11

        return X1, Y1

    def multiStepReach(self, X0=None, U=None, k=1):
        """
        Reachability of DLODE in multisteps

        X[k+1] = AX[k] + BU[k]
        Y[k+1] = CX[k] + DU[k]

        Inputs:

          @X0: initial condition, can be a state vector or a set (Star or ProbStar)
          @U: can be a constant control input vector (for multiple step) or 
               can be a sequence of control input vectors for k steps (an array where each row is an input vector) or
               can be a sequence of control sets (Stars or ProbStars) for k steps

          @k: number of steps 

        Outputs:

          @X1: sequence of state vectors or sets
          @Y1: sequence of output vectors or sets

        ============================================================================================================================

        We consider different usecases to model different situations (open loop/closed-loop control, disturbance, no control, etc.,)

        ============================================================================================================================

        *Case 1 (simpliest): Only initial condition, no control inputs/disturbances

        *Case 2: U is a sequence of control input vectors

        *Case 3: U is a sequence of control input sets

        
        Author: Dung Tran
        Date: 11/26/2022

        """

        assert k >= 1, 'error: invalid number of steps'
        X = []  # reachable set of states
        Y = []  # reachable set of outputs   

        if X0 is None:
            X.append(np.zeros((self.dim,)))
        else:
            X.append(X0)

        if U is None:
            for i in range(1, k+1):
                X1,Y1 = self.stepReach(X[i-1])
                X.append(X1)
                Y.append(Y1)
        elif isinstance(U, np.ndarray):
            assert U.shape[0] == k, 'error: U should be an numpy array with k rows for k time steps'
            for i in range(1, k+1):
                X1, Y1 = self.stepReach(X[i-1], U[i-1, :])
                X.append(X1)
                Y.append(Y1)
        elif isinstance(U, list):
            assert len(U) == k, 'error: U0 should be a list of k ProbStars or Stars'
            for i in range(1, k+1):
                print('i = {}'.format(i))
                U[i-1].__str__()
                X[i-1].__str__()
                X1, Y1 = self.stepReach(X[i-1], U[i-1])
                X.append(X1)
                Y.append(Y1)   

        return X, Y


        

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

        res = DLODE(A, B, C, D, dt)

        return res 

       
        
        
    
