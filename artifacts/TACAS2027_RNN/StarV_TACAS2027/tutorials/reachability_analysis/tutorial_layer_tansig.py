import numpy as np
from StarV.set.star import Star
from StarV.set.sparsestar import SparseStar
from StarV.layer.TanSigLayer import TanSigLayer
from StarV.util.plot import plot_star

def tansig_contruct():
    """
    Construct a TanH layer
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: TanH Layer Construction =========================================')
    # Construct a TanH layer
    L_tanh = TanSigLayer()
    print(L_tanh)
    
    print('=============== DONE: TanH Layer Construction ============================================')
    print('==========================================================================================\n\n')


def tansig_reachability_approx_star():
    """
    Perform approximate reachability analysis on a TanH layer using Star sets
    """
    print('==========================================================================================')
    print('============ EXAMPLE: Approx Reachability Analysis on TanH Layer using Star ==============')
    # Construct a TanH layer
    L_tanh = TanSigLayer()

    # Construct input set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    input = Star(lb, ub)
    print('\nInput set:')
    print(input)
    plot_star(input)

    # Reachability analysis
    output = L_tanh.reach(input, method='approx')
    print('\nOutput set:')
    print(output)
    plot_star(output)

    print('============ DONE: Approx Reachability Analysis on TanH Layer using Star =================')
    print('==========================================================================================\n\n')
    

def tansig_reachability_approx_sparsestar():
    """
    Perform approximate reachability analysis on a TanH layer using SparseStar sets
    """
    print('==========================================================================================')
    print('========= EXAMPLE: Approx Reachability Analysis on TanH Layer using SparseStar ===========')
    # Construct a TanH layer
    L_tanh = TanSigLayer()

    # Construct input set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    input = SparseStar(lb, ub)
    print('\nInput set:')
    print(input)
    plot_star(input)

    # Reachability analysis
    output = L_tanh.reach(input, method='approx')
    print('\nOutput set:')
    print(output)
    plot_star(output)

    print('========= DONE: Approx Reachability Analysis on TanH Layer using SparseStar ==============')
    print('==========================================================================================\n\n')
    
    
if __name__ == "__main__":
    tansig_contruct()
    tansig_reachability_approx_star()
    tansig_reachability_approx_sparsestar()