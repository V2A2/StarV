"""
ReLU layer class
Dung Tran, 9/10/2022
Update: 12/20/2024 (Sung Woo Choi, merging)
"""

from StarV.fun.poslin import PosLin


class ReLULayer(object):
    """ ReLULayer class for qualitative and quantitative reachability
        Author: Dung Tran
        Date: 9/10/2022
    """
    
    @staticmethod
    def evaluate(x):
        return PosLin.evaluate(x)
    
    @staticmethod
    def reach(In, method='exact', lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """main reachability method
           Args:
               @I: a list of input set (Star or ProbStar)
               @method: method: 'exact', 'approx', or 'relax'
               @lp_solver: lp solver: 'gurobi' (default), 'glpk', or 'linprog'
               @pool: parallel pool: None or multiprocessing.pool.Pool
               @RF: relax-factor from 0 to 1 (0 by default)

            Return: 
               @R: a list of reachable set
        """
        if show:
            print("\nReLULayer reach function\n")

        if method == 'exact':
            print(" type of input set before exactReLu:", type(In))
            S = []
            for i in range(0, len(In)):
                if isinstance(In[i],list):
                    S1 = PosLin.reachExactMultiInputs(In[i], lp_solver, pool)
                    S.append(S1)
                else:
                    S = PosLin.reachExactMultiInputs(In, lp_solver, pool)
            return S
        
        elif method == 'approx':
            if len(In) >1:
                S =[]
                for i in range(len(In)):
                    S1 = PosLin.reachApproxSingleInput(In=In[i], lp_solver=lp_solver, RF=RF, DR=DR, show=show)
                    S.append(S1)
                return S
            else:
                return PosLin.reachApproxSingleInput(In=In, lp_solver=lp_solver, RF=RF, DR=DR, show=show)
        elif method == 'relax':
            return PosLin.reachApproxSingleInput(In=In, lp_solver=lp_solver, RF=RF, DR=DR, show=show)
        elif method == 'basic':
            return PosLin.reachApprox(In=In, lp_solver=lp_solver, show=show)
        else:
            raise Exception(f"error: unknown reachability method: {method}")

    def __str__(self):
        print('Layer type: {}'.format(self.__class__.__name__))
        print('')
        return '\n'

    def info(self):
        print(self)
