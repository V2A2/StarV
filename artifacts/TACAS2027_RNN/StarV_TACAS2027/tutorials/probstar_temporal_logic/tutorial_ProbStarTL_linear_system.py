
import math
import time
import numpy as np
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
from StarV.util.plot import plot_probstar_signal,plot_probstar
from StarV.verifier.krylov_func.simKrylov_with_projection import simReachKrylov
from StarV.verifier.krylov_func.simKrylov_with_projection import random_two_dims_mapping
from StarV.verifier.krylov_func.LCS_verifier import quantiVerifier_LCS
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_,_OR_


def harmonic_initial_states():
    """
    Constructing initial states for timed harmonic oscilaator example
    """
    print('==========================================================================================')
    print('=============== EXAMPLE: Timed Harmonic Initial States ===================================')
    
    # system dynamics
    A = np.array([[0,1,1,0],[-1,0,1,0],[0,0,0,0],[0,0,0,0]])

    # input sets 
    init_state_bounds_list = []
    dims = A.shape[0]
    for dim in range(dims):
        if dim == 0: 
            lb = -6
            ub = -5
        elif dim == 1: 
            lb = 0
            ub = 1
        elif dim == 2:
            lb = 0.5
            ub = 0.5
        elif dim == 3: 
            lb = 0.5
            ub = 0.5
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))

        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]
    print(init_state_lb)
    print(init_state_ub)

    # construct satr set using state lower and upper bound 
    X0 = Star(init_state_lb,init_state_ub)

    # construct probsatr set 
    mu_U = 0.5*(X0.pred_lb + X0.pred_ub) 
    a  = 3
    sig_U = (X0.pred_ub-mu_U )/a
    epsilon = 1e-10
    sig_U = np.maximum(sig_U, epsilon)
    Sig_U = np.diag(np.square(sig_U))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)
    print(X0_probstar)

    print('==========================================================================================')
    print('============================ Done: Timed Harmonic Initial States =========================\n\n')



def probstarTL_harmonic_temporal_specs():
    """
    Creating temporal specifications
    """

    print('==========================================================================================')
    print('=============== EXAMPLE: Timed Harmonic Example Temporal Specifications ==================')

    # load temporal and logic operator
    h = math.pi/4
    time_bound = math.pi
    T = int (time_bound/ h)
    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()
    EVOT =_EVENTUALLY_(0,T)
    EVOT1 =_EVENTUALLY_(2,3)
    AWOT = _ALWAYS_(0,T)
    AWOT1 = _ALWAYS_(1,2)

    # create atomic predicates
    A1 = np.array([-1., 0.])
    b1 = np.array([-4])
    P1 = AtomicPredicate(A1,b1)
    A2 = np.array([1,0])
    b2 = np.array([4])
    P2 = AtomicPredicate(A2,b2)

    # convert atomic predicate to temporal specification
    # phi1 : eventually_[0, T](x >= 4) : A1x <= b1
    phi1 = Formula([EVOT,P1])
    print(phi1)
    # phi2 : always_[1,2](x <=4 OR eventually_[2, 3](x >=4 ))
    phi2 = Formula([AWOT1,lb,P2,OR,lb,EVOT1,P1,rb,rb])
    print(phi2)

    specs =[phi1,phi2]

    print('==========================================================================================')
    print('============================ Done: Timed Harmonic Example Temporal Specifications ========\n\n')


def probstarTL_harmonic_verifying():
    """
    Verifying the Harmonic example using ProbSatrTL
    """

    print('==========================================================================================')
    print('=============== EXAMPLE: Timed Harmonic Example Verification =============================')

    # system dynamics
    A = np.array([[0,1,1,0],[-1,0,1,0],[0,0,0,0],[0,0,0,0]])

    # input sets 
    init_state_bounds_list = []
    dims = A.shape[0]
    for dim in range(dims):
        if dim == 0: 
            lb = -6
            ub = -5
        elif dim == 1: 
            lb = 0
            ub = 1
        elif dim == 2:
            lb = 0.5
            ub = 0.5
        elif dim == 3: 
            lb = 0.5
            ub = 0.5

        init_state_bounds_list.append((lb, ub))
    init_state_bounds_array = np.array(init_state_bounds_list)
    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]
    print(init_state_lb)
    print(init_state_ub)

    # construct satr set using state lower and upper bound 
    X0 = Star(init_state_lb,init_state_ub)

    # construct probsatr set 
    mu_U = 0.5*(X0.pred_lb + X0.pred_ub) 
    a  = 3
    sig_U = (X0.pred_ub-mu_U )/a
    epsilon = 1e-10
    sig_U = np.maximum(sig_U, epsilon)
    Sig_U = np.diag(np.square(sig_U))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)
    print(X0_probstar)

    # load temporal and logic operator
    h = math.pi/4
    time_bound = math.pi
    T = int (time_bound/ h) 
    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()
    EVOT =_EVENTUALLY_(0,T)
    EVOT1 =_EVENTUALLY_(2,3)
    AWOT = _ALWAYS_(0,T)
    AWOT1 = _ALWAYS_(1,2)

    # create atomic predicates
    A1 = np.array([-1., 0.])
    b1 = np.array([-4])
    P1 = AtomicPredicate(A1,b1)
    A2 = np.array([1,0])
    b2 = np.array([4])
    P2 = AtomicPredicate(A2,b2)

    # convert atomic predicate to temporal specification
    # phi1 : eventually_[0, T](x >= 4) : A1x <= b1
    phi1 = Formula([EVOT,P1])
    print(phi1)
    # phi2 : always_[1,2](x <=4 OR eventually_[2, 3](x >=4 ))
    phi2 = Formula([AWOT1,lb,P2,OR,lb,EVOT1,P1,rb,rb])
    print(phi2)

    specs =[phi1,phi2]

    # verify the system

    # set paprameter for reachability
    m = 2
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9
    use_arnoldi = True
    use_init_space = False
    output_space = np.array([[1,0,0,0],[0,1,0,0]])
    initial_space = X0_probstar.V 

    reach_start_time = time.time()
    R,_= simReachKrylov(A,X0_probstar,h, T, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
    reach_time = time.time() - reach_start_time

    reachTime = []
    checkingTime = []
    verifyTime = []
    p_SAT_MAX = []
    p_SAT_MIN= []
   
    for i in range(0,len(specs)):
        check_start = time.time()
        spec = specs[i]
        DNF_spec = spec.getDynamicFormula()
        _,p_max, p_min,_= DNF_spec.evaluate(R)
        end = time.time()
        reachTime.append(reach_time)
        check_time = end -check_start
        checkingTime.append(check_time)
        verifyTime.append(check_time + reach_time)
        p_SAT_MAX.append(p_max)
        p_SAT_MIN.append(p_min)
    

    print('p_SAT_MAX = {}'.format(p_SAT_MAX))
    print('p_SAT_MIN = {}'.format(p_SAT_MIN))
    print('reachTime = {}'.format(reachTime))
    print('checkingTime = {}'.format(checkingTime))
    print('verifyTime = {}'.format(verifyTime))
    
    plot_probstar_signal(R)

    print('==========================================================================================')
    print('============================ Done: Timed Harmonic Example Verification ===================\n\n')



if __name__ == '__main__':
    """
    Main function to run the ProbStarTL of Timed Harmonic Example tutorials
    """
    harmonic_initial_states()
    probstarTL_harmonic_temporal_specs()
    probstarTL_harmonic_verifying()