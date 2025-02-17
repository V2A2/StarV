import numpy as np
from StarV.set.star import Star
from StarV.set.sparsestar import SparseStar
from StarV.layer.LogSigLayer import LogSigLayer
from StarV.util.plot import plot_star

def logsig_contruct():
    """
    Construct a Sigmoiod layer
    """
    print('==========================================================================================')
    print('============= EXAMPLE: Sigmoiod Layer Construction =======================================')
    # Construct a Sigmoiod layer
    L_sigmoid = LogSigLayer()
    print(L_sigmoid)
    
    print('=============== DONE: Sigmoiod Layer Construction ============================================')
    print('==========================================================================================\n\n')


def logsig_reachability_approx_star():
    """
    Perform approximate reachability analysis on a Sigmoiod layer using Star sets
    """
    print('==========================================================================================')
    print('========== EXAMPLE: Approx Reachability Analysis on Sigmoiod Layer using Star ============')
    # Construct a Sigmoiod layer
    L_sigmoid = LogSigLayer()

    # Construct input set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    input = Star(lb, ub)
    print('\nInput set:')
    print(input)
    plot_star(input)

    # Reachability analysis
    output = L_sigmoid.reach(input, method='approx')
    print('\nOutput set:')
    print(output)
    plot_star(output)

    print('========== DONE: Approx Reachability Analysis on Sigmoiod Layer using Star ===============')
    print('==========================================================================================\n\n')
    

def logsig_reachability_approx_sparsestar():
    """
    Perform approximate reachability analysis on a Sigmoiod layer using SparseStar sets
    """
    print('==========================================================================================')
    print('======= EXAMPLE: Approx Reachability Analysis on Sigmoiod Layer using SparseStar =========')
    # Construct a TanH layer
    L_sigmoid = LogSigLayer()

    # Construct input set
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    input = SparseStar(lb, ub)
    print('\nInput set:')
    print(input)
    plot_star(input)

    # Reachability analysis
    output = L_sigmoid.reach(input, method='approx')
    print('\nOutput set:')
    print(output)
    plot_star(output)

    print('========= DONE: Approx Reachability Analysis on TanH Layer using SparseStar ==============')
    print('==========================================================================================\n\n')
    
    
if __name__ == "__main__":
    logsig_contruct()
    logsig_reachability_approx_star()
    logsig_reachability_approx_sparsestar()