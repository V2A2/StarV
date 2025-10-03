import time
import os
import math
import pickle
import pandas as pd
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.util.plot import plot_probstar_signal_contour
from StarV.verifier.verifier import checkSafetyProbStar
from StarV.verifier.krylov_func.simKrylov_with_projection import combine_mats
from StarV.verifier.krylov_func.simKrylov_with_projection import simReachKrylov as sim3
from StarV.verifier.krylov_func.simKrylov_with_projection import random_two_dims_mapping
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_,_OR_
from StarV.util.load import load_building_model,load_beam_model,load_pde_model,load_MNA5_model,load_MNA1_model,load_iss_model,load_mcs_model,load_fom_model

def run_harmonic( use_arnoldi =True,use_init_space=False):

    A = np.array([[0,1,1,0],[-1,0,1,0],[0,0,0,0],[0,0,0,0]])
    h = math.pi/4
    N = int((math.pi)/h)
    m = 2
    target_error = 1e-9
    tolerance = 1e-9
    samples = 51
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

    X0 = Star(init_state_lb,init_state_ub)

    # create initial input Probstar set
    mu_U = 0.5*(X0.pred_lb + X0.pred_ub) 
    a  = 3
    sig_U = (X0.pred_ub-mu_U )/a
    epsilon = 1e-10
    sig_U = np.maximum(sig_U, epsilon)
    Sig_U = np.diag(np.square(sig_U))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)

    # parameters
    h = math.pi/4
    time_bound = math.pi
    N = int (time_bound/ h)
    m = 2
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9

    # define output and initial sapces
    output_space= np.array([[1,0,0,0],[0,1,0,0]])
    initial_space = X0_probstar.V 

    # compute Probsatr signal(a sequences of rechable sets)
    reach_start_time = time.time()
    R,krylov_time = sim3(A,X0_probstar,h, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space) 
    reach_time_duration = time.time() - reach_start_time

    # create temporal specifications
    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()

    A1 = np.array([-1., 0.])
    b1 = np.array([-4])
    P1 = AtomicPredicate(A1,b1)

    EVOT =_EVENTUALLY_(0,4)
    EVOT1 =_EVENTUALLY_(2,3)
    AWOT1 = _ALWAYS_(1,2)
    
    A3 = np.array([-1.,0])
    b3 = np.array([-4])
    P3 = AtomicPredicate(A3,b3)

    A4 = np.array([1,0])
    b4 = np.array([4])
    P4 = AtomicPredicate(A4,b4)

    specs =[]
    spec = Formula([EVOT,P1])
    spec1 = Formula([AWOT1,lb,P4,OR,lb,EVOT1,P1,rb,rb])
    specs =[spec,spec1]

    checking_time = []
    data=[]

    # verification using ProbSatrTL
    for i in range(0,len(specs)):
        check_start = time.time()
        spec = specs[i]
        print('\n==================Specification{}====================: '.format(i))
        spec.print()
        DNF_spec = spec.getDynamicFormula()
        Nadnf = DNF_spec.length
        print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
        _,p_max, p_min,Ncdnf = DNF_spec.evaluate(R)
        end = time.time()
        checking_time = end -check_start 
        print("p_min:",p_min)
        print("p_max:",p_max) 
        verify_time=checking_time + reach_time_duration    
        data.append(["Harmonic",i,Nadnf,Ncdnf,p_min,p_max, reach_time_duration,checking_time,verify_time])

    print(tabulate(data,headers=[ "Model","spec","prob-Min","prob-Max", "tr","tc","tv"],tablefmt='latex'))

    # save verification results
    cur_path = os.path.dirname(os.path.abspath(__file__))
    path = cur_path + '/results'  
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/Table_1_Harmonic_results.tex", "w") as f:
        print(tabulate(data, headers=["Model","spec","Nadnf","Ncdnf","prob-Min","prob-Max", "tr","tc","tv"], tablefmt='latex'), file=f)
    
    print("Table_1 has been generated and saved to 'results/Table_1_Harmonic_results.tex'")

    # plot ProbStar signal and save to file --- gnerate figure_1a
    plot_probstar_signal_contour(R,show=None)
    plt.savefig(path+"/Figure_1a.png")
    # plt.show()
    print("\nFigure_1a has been generated and saved to 'results/Figure_1a.tex'")


    # plot unsafe ProbStar signal and save to file --- generate figure_1b
    unsafe_mat = np.array([[-1,0]])
    unsafe_vec =np.array([-4])
    P = []  # unsafe output set

    # handle one output constraints
    for i in range(1,len(R)):
        P1,prob1 = checkSafetyProbStar(unsafe_mat, unsafe_vec,R[i])
        if prob1 > 0 and not P1.isEmptySet():
            P.append(P1)

    plot_probstar_signal_contour(P,show= None)
    plt.savefig(path+"/Figure_1b.png")
    # plt.show()
    print("Figure_1b has been generated and saved to 'results/Figure_1b.tex'")

    return R
     
def run_mcs_model(use_arnoldi = None,use_init_space=None):
    
    print("\n\n=====================================================")
    print('Quantitative Verification of MCS Model ')
    print('=====================================================')

    plant = load_mcs_model()
    combined_mat = combine_mats(plant.A,plant.B)
    dims = combined_mat.shape[0]

    # returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    initial_dim = plant.A.shape[0]
    for dim in range(dims):
        if dim == 1:
            lb =0.002
            ub = 0.0025
        elif dim == 2:
            lb =0.001
            ub = 0.0015    
        elif dim < initial_dim:
            lb = ub = 0 
        elif dim == initial_dim :
            # first input
            lb = 0.16
            ub = 0.3
        elif dim > initial_dim:
            # second input
            lb = 0.2
            ub = 0.4
        else:         
            raise RuntimeError('Unknown dimension: {}'.format(dim))        
        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    # create Star for initial state 
    X0 = Star(init_state_lb,init_state_ub)

    # create input ProbStar Star set
    mu_X0 = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_X0 = (mu_X0 - X0.pred_lb)/a
    Sig_X0 = np.diag(np.square(sig_X0))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_X0, Sig_X0,X0.pred_lb,X0.pred_ub)

    h = [0.1]
    time_bound = 20
    m = 4
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9

    output_space= plant.C
    expand_mat = np.zeros((2,2))
    output_space = np.hstack((output_space,expand_mat))
    initial_space = X0_probstar.V 

    # create temporal specifications
    A1 = np.array([-1,0])
    b1 = np.array([-0.3])
    P1 = AtomicPredicate(A1, b1)
    P11= AtomicPredicate(-A1, -b1)
    
    A2 = np.array([1,0])
    b2 = np.array([0.4])
    P2 = AtomicPredicate(A2, b2)
    P22 = AtomicPredicate(-A2, -b2)

    A3 = np.array([0,-1])
    b3 = np.array([-0.4])
    P3 = AtomicPredicate(A3, b3)
    P33 = AtomicPredicate(-A3, -b3)
    
    A4 = np.array([0,1])
    b4 = np.array([0.6])
    P4 = AtomicPredicate(A4, b4)
    P44 = AtomicPredicate(-A4, -b4)

    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()

    datas=[]

    for hi in h:
        N = int (time_bound/ hi)
        reach_start_time = time.time()
        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time

        EV0T = _EVENTUALLY_(0,N)
        EV0T1 = _EVENTUALLY_(10,15)
        AW0T = _ALWAYS_(0,N)
        AW0T1 = _ALWAYS_(10,15)

        # create temporal specifications
        specs =[]
        spec = Formula([EV0T,P1,AND,P2,AND,P3,AND,P4])
        spec1 = Formula([EV0T1,lb,P1,AND,P2,AND,P3,AND,P4,OR,lb,AW0T1,P11,AND,P22,AND,P33,AND,P44,rb,rb])
        spec2 = Formula([AW0T1,P1,OR,P2,OR,P3,OR,P4])
        specs=[spec,spec1,spec2]

        for spec_id in range(0,len(specs)):
            check_start = time.time()
            spec = specs[spec_id]
            print("\n===================Specification{}: =====================".format(spec_id))
            spec.print()
            # S  = spec.render(R)
            DNF_spec = spec.getDynamicFormula()
            print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
            Nadnf = DNF_spec.length
            _,p_max, p_min,Ncdnf = DNF_spec.evaluate(R)
            end = time.time()
            checking_time = end -check_start 
            print("p_min:",p_min)
            print("p_max:",p_max)

            verify_time = checking_time + reach_time_duration    

            # data.append(["MCS",hi,spec_id,Nadnf,Ncdnf,p_min,p_max, reach_time_duration,checking_time,verify_time])
            data = {
                    'Model': 'Motor',
                    'Spec': spec_id,
                    'Nadnf': Nadnf,
                    'Ncdnf': Ncdnf,
                    'p_min':p_min,
                    'p_max':p_max,
                    't_r':reach_time_duration,
                    't_c':checking_time,
                    't_v': verify_time
                }
            datas.append(data)

    return datas

def run_building_model(use_arnoldi =None,use_init_space=None):
    
    print('\n=====================================================')
    print('Quantitative Verification of Building Model ')
    print('=====================================================')

    plant = load_building_model()
    combined_mat = combine_mats(plant.A,plant.B)
    dims = combined_mat.shape[0]
    
    #  returns list of initial states for each dimension
    init_state_bounds_list = []
    initial_dim = plant.A.shape[0]

    for dim in range(dims):
        if dim < 10:
            lb = 0.0002
            ub = 0.00025
        elif dim == 24:
            lb = -0.0001
            ub = 0.0001
        elif dim < initial_dim: 
            lb = ub = 0 
        elif dim >= initial_dim:
            lb = 0.8
            ub = 1
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))
        
        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)
    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    X0 = Star(init_state_lb,init_state_ub)

    # create input ProbStar set
    mu_U = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_U = (mu_U - X0.pred_lb)/a
    Sig_U = np.diag(np.square(sig_U))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)

    # deine output and initial spaces
    output_space= plant.C
    expand_mat = np.zeros((1,1))
    output_space = np.hstack((output_space,expand_mat))
    initial_space = X0_probstar.V 

    inputProb = X0_probstar.estimateProbability()

    h = [0.1]
    time_bound = 20
    m = 4
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9    

    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()

    A1 = np.array([-1])
    b1 = np.array([-0.004])
    P = AtomicPredicate(A1,b1)
    P1 = AtomicPredicate(-A1,-b1)

    datas = []

    for hi in h:

        N = int (time_bound/ hi)
        reach_start_time = time.time()
        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time

        # create temporal specifications
        EVOT =_EVENTUALLY_(0,N)
        EVOT1 =_EVENTUALLY_(0,10)
        AWOT=_ALWAYS_(0,N)
        AWOT1=_ALWAYS_(0,10)

        spec0 = Formula([EVOT,P]) # be unsafe at some step 0 to N
        spec1 = Formula([AWOT,P1]) # alwyas safe between 0 to N
        spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P,rb,rb])

        check_start = time.time()
        spec = spec2
        print("\n===================Specification{}: =====================".format(2))
        spec.print()
        # S  = spec.render(R)
        DNF_spec = spec.getDynamicFormula()
        Nadnf = DNF_spec.length
        print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
        _,p_max, p_min, Ncdnf = DNF_spec.evaluate(R)
        end = time.time()
        checking_time = end - check_start 
        print("p_min:",p_min)
        print("p_max:",p_max)
        verify_time = checking_time + reach_time_duration    
        
        data_spec2 = {
            'Model': 'Building',
            'Spec': 2,
            'Nadnf': Nadnf,
            'Ncdnf': Ncdnf,
            'p_min':p_min,
            'p_max':p_max,
            't_r':reach_time_duration,
            't_c':checking_time,
            't_v': verify_time
        }
        
        specs=[spec0, spec1]
        # Using for loop
        for spec_id in range(0,len(specs)):
            check_start = time.time()
            spec = specs[spec_id]
            print("\n===================Specification{}: =====================".format(spec_id))
            spec.print()
            # S  = spec.render(R)
            DNF_spec = spec.getDynamicFormula()
            Nadnf = DNF_spec.length
            print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
            print(DNF_spec)
            _, p_max, p_min, Ncdnf = DNF_spec.evaluate(R)
            end = time.time()
            checking_time = end - check_start 
            print("p_min:",p_min)
            print("p_max:",p_max)
            verify_time = checking_time + reach_time_duration    
         
            data = {
                'Model': 'Building',
                'Spec': spec_id,
                'Nadnf': Nadnf,
                'Ncdnf': Ncdnf,
                'p_min':p_min,
                'p_max':p_max,
                't_r':reach_time_duration,
                't_c':checking_time,
                't_v': verify_time
            }
            
            datas.append(data)
        datas.append(data_spec2)

    return datas

def run_pde_model(use_arnoldi = None,use_init_space=None):
        
    print('\n\n=====================================================')
    print('Quantitative Verification of PDE Model ')
    print('=====================================================')
    plant = load_pde_model()
    combined_mat = combine_mats(plant.A,plant.B)
    dims = combined_mat.shape[0]

    #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    initial_dim = plant.A.shape[0]

    for dim in range(dims):
        if dim < 64:
            lb = ub = 0
        elif dim < 80:
            lb = 0.001
            ub = 0.0015
        elif dim < initial_dim:
            lb = -0.002
            ub = -0.0015  
        elif dim >= initial_dim:
            lb = 0.5
            ub = 1      
        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    # create Star for initial state 
    X0 = Star(init_state_lb,init_state_ub)
    
    # create input ProbStar set
    mu_X0 = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_X0 = (mu_X0 - X0.pred_lb)/a
    Sig_X0 = np.diag(np.square(sig_X0))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_X0, Sig_X0,X0.pred_lb,X0.pred_ub)


    h = [0.1]
    time_bound = 20
    m = 8
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9

    output_space = plant.C
    expand_mat = np.zeros((1,1))
    output_space = np.hstack((output_space,expand_mat))
    initial_space = X0_probstar.V 

    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()

    A1 = np.array([-1])
    b1 = np.array([-10.75])
    P = AtomicPredicate(A1,b1)
    P1 = AtomicPredicate(-A1,-b1)
    
    datas = []

    for hi in h:
        N = int (time_bound/ hi)

        EVOT =_EVENTUALLY_(0,N)
        EVOT1 =_EVENTUALLY_(0,20)
        AWOT=_ALWAYS_(0,N)
        AWOT1=_ALWAYS_(10,20)

        spec = Formula([EVOT,P]) # be unsafe at some step 0 to N
        spec1 = Formula([AWOT,P1]) # alwyas safe between 0 to N
        spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P,rb,rb])
        specs=[spec,spec1,spec2]

        reach_start_time = time.time()
        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time

        for spec_id in range(0,len(specs)):
            check_start = time.time()
            spec = specs[spec_id]
            print("\n===================Specification{}: =====================".format(spec_id))
            spec.print()
            DNF_spec = spec.getDynamicFormula()
            Nadnf = DNF_spec.length
            print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
            _,p_max, p_min,Ncdnf = DNF_spec.evaluate(R)
            end = time.time()
            checking_time = end -check_start 
            print("p_min:",p_min)
            print("p_max:",p_max)
            verify_time = checking_time + reach_time_duration    

            data = {
                    'Model': 'PDE',
                    'Spec': spec_id,
                    'Nadnf': Nadnf,
                    'Ncdnf': Ncdnf,
                    'p_min':p_min,
                    'p_max':p_max,
                    't_r':reach_time_duration,
                    't_c':checking_time,
                    't_v': verify_time
                }
            datas.append(data)

    return datas
     
def run_iss_model(use_arnoldi=None,use_init_space=None):
       
    print('\n\n=====================================================')
    print('Quantitative Verification of ISS Model ')
    print('=====================================================')

    plant = load_iss_model()
    combined_mat = combine_mats(plant.A,plant.B)
    dims = combined_mat.shape[0]

    #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    initial_dim = plant.A.shape[0]

    for dim in range(dims):
        if dim < 100:
            lb = -0.0001 
            ub = 0.0001 
        elif dim < initial_dim:
            lb = 0
            ub = 0
        elif dim == initial_dim: # input 1
            lb = 0
            ub = 0.1
        elif dim ==initial_dim+1: # input 2
            lb = 0.8
            ub = 1.0
        elif dim == initial_dim+2: # input 3
            lb = 0.9
            ub = 1.0
        else:
            raise RuntimeError('incorrect dimension: {}'.format(dim))
            
        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    # create Star for initial state 
    X0 = Star(init_state_lb,init_state_ub)


    # create ProbStar for initial state 
    mu_X0 = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_U = (mu_X0 - X0.pred_lb)/a

    Sig_U = np.diag(np.square(sig_U))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_X0, Sig_U,X0.pred_lb,X0.pred_ub)


    h = [0.1]
    time_bound = 20
    m = 8
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9


    output_space= plant.C[2,:].reshape(1,initial_dim) #y3
    expand_mat = np.zeros((1,3))
    output_space = np.hstack((output_space,expand_mat))
    initial_space = X0_probstar.V 


    datas = []

    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()

    A1 = np.array([1])
    b1 = np.array([-0.0001])
    P = AtomicPredicate(A1,b1)
    P1 = AtomicPredicate(-A1,-b1)
    
    A2 = np.array([-1])
    b2 = np.array([-0.0001])

    P2 = AtomicPredicate(A2,b2)
    P3= AtomicPredicate(-A2,-b2)
    
    for hi in h:
        N = int (time_bound/ hi)
        reach_start_time = time.time()

        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi= use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time
       
        EVOT =_EVENTUALLY_(0,N)
        EVOT1 =_EVENTUALLY_(50,100)
        AWOT=_ALWAYS_(0,N)
        AWOT1=_ALWAYS_(50,100)

        spec = Formula([EVOT,P,OR,P2]) # be unsafe at some step 0 to N
        spec1 = Formula([AWOT,P1,AND,P3]) # alwyas safe between 0 to N
        spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P,rb,rb])
        specs=[spec,spec1,spec2]

        for spec_id in range(0,len(specs)):
            check_start = time.time()
            spec = specs[spec_id]
            print("\n==================Specification{}: =====================".format(spec_id))
            spec.print()
            DNF_spec = spec.getDynamicFormula()
            Nadnf = DNF_spec.length
            # DNF_spec.print()
            print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
            _,p_max, p_min,Ncdnf=DNF_spec.evaluate(R)
            end = time.time()
            checking_time = end -check_start 
            print("p_min:",p_min)
            print("p_max:",p_max)
            verify_time = checking_time + reach_time_duration    

            # data.append(["ISS",hi,spec_id,Nadnf,Ncdnf,p_min,p_max, reach_time_duration,checking_time,verify_time])
            data = {
                    'Model': 'ISS',
                    'Spec': spec_id,
                    'Nadnf': Nadnf,
                    'Ncdnf': Ncdnf,
                    'p_min':p_min,
                    'p_max':p_max,
                    't_r':reach_time_duration,
                    't_c':checking_time,
                    't_v': verify_time
                }
            datas.append(data)


    return datas

def run_beam_model(use_arnoldi = None,use_init_space=None):
       
    print('\n\n=====================================================')
    print('Quantitative Verification of Beam Model ')
    print('=====================================================')
    plant = load_beam_model()
    combined_mat = combine_mats(plant.A,plant.B)
    dims = combined_mat.shape[0]


    #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    initial_dim = plant.A.shape[0]
    # print("init_dim:",initial_dim)

    for dim in range(dims):
        if dim < 300 :
            lb = ub = 0
        elif dim < initial_dim:
            lb = 0.0015
            ub = 0.002
        elif dim >= initial_dim:
            lb = 0.2
            ub = 0.8
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))
        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    # create Star for initial state 
    X0 = Star(init_state_lb,init_state_ub)

    
    # create ProbStar for initial state 
    mu_X0 = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_X0 = (mu_X0 - X0.pred_lb)/a
    Sig_X0 = np.diag(np.square(sig_X0))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_X0, Sig_X0,X0.pred_lb,X0.pred_ub)

    h = [0.1]
    time_bound = 20
    m = 8
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9


    output_space= plant.C
    expand_mat = np.zeros((1,1))
    output_space = np.hstack((output_space,expand_mat))
    initial_space = X0_probstar.V 

    datas = []
    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()

    A1 = np.array([-1])
    b1 = np.array([-500])
    P = AtomicPredicate(A1,b1)

    P1 = AtomicPredicate(-A1,-b1)
    
    for hi in h:
        N = int (time_bound/ hi)
        reach_start_time = time.time()

        R,krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time

        EVOT =_EVENTUALLY_(0,N)
        EVOT1 =_EVENTUALLY_(100,200)
        AWOT=_ALWAYS_(0,N)
        AWOT1=_ALWAYS_(100,200)

        spec = Formula([EVOT,P]) # be unsafe at some step 0 to N
        spec1 = Formula([AWOT,P1]) # alwyas safe between 0 to N
        spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P,rb,rb])
        specs=[spec,spec1,spec2]

        for spec_id in range(0,len(specs)):
            check_start = time.time()
            spec = specs[spec_id]
            print("\n===================Specification{}: =====================".format(spec_id))
            spec.print()
            # S  = spec.render(R)
            DNF_spec = spec.getDynamicFormula()
            Nadnf = DNF_spec.length

            print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
            _,p_max, p_min,Ncdnf = DNF_spec.evaluate(R)
            end = time.time()
            checking_time = end -check_start 
            print("p_min:",p_min)
            print("p_max:",p_max)

            verify_time = checking_time + reach_time_duration    
    
            data = {
                    'Model': 'Beam',
                    'Spec': spec_id,
                    'Nadnf': Nadnf,
                    'Ncdnf': Ncdnf,
                    'p_min':p_min,
                    'p_max':p_max,
                    't_r':reach_time_duration,
                    't_c':checking_time,
                    't_v': verify_time
                }
            datas.append(data)


    return datas

def run_MNA1_model(use_arnoldi = None,use_init_space=None):

    print('\n\n=====================================================')
    print('Quantitative Verification of MNA1 Model ')
    print('=====================================================')
    plant = load_MNA1_model()
    combined_mat = combine_mats(plant.A,plant.B)
    dims = combined_mat.shape[0]

    #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    initial_dim = plant.A.shape[0]
    for dim in range(dims):
        if dim < 2:
            lb = 0.001
            ub = 0.0015
        elif dim < initial_dim:
            lb = ub = 0
        elif dim >= initial_dim and dim < initial_dim + 5:
            # first 5 inputs
            lb = ub = 0.1
        elif dim >= initial_dim + 5 and dim < initial_dim + 9:
            # second 4 inputs
            lb = ub = 0.2
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))
            
        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)
    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    # create Star for initial state 
    X0 = Star(init_state_lb,init_state_ub)

    # create ProbStar for initial state 
    mu_U = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig = (mu_U - X0.pred_lb)/a
    Sig_U = np.diag(np.square(sig))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)

    h = [0.1]
    time_bound = 20
    m = 8
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9

    output_space= np.zeros((1,dims))
    output_space[0,0]=1
    initial_space = X0_probstar.V 

    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()
    A1 = np.array([-1])
    b1 = np.array([-0.2])
    P = AtomicPredicate(A1,b1)
    P1 = AtomicPredicate(-A1,-b1)

    datas = []

    for hi in h:
        N = int (time_bound/ hi)
        reach_start_time = time.time()
        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time

        EVOT =_EVENTUALLY_(0,N)
        EVOT1 =_EVENTUALLY_(150,195)
        AWOT=_ALWAYS_(0,N)
        AWOT1=_ALWAYS_(150,195)

        sepcs=[]
        spec = Formula([EVOT,P]) # be unsafe at some step 0 to N
        spec1 = Formula([AWOT,P1]) # alwyas safe between 0 to N
        spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P,rb,rb])
        specs=[spec,spec1,spec2]

        for spec_id in range(0,len(specs)):
            check_start = time.time()
            spec = specs[spec_id]
            print("\n===================Specification{}: =====================".format(spec_id))
            spec.print()
            DNF_spec = spec.getDynamicFormula()
            Nadnf = DNF_spec.length
            # DNF_spec.print()
            print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
            _,p_max, p_min, Ncdnf= DNF_spec.evaluate(R)
            end = time.time()
            checking_time = end -check_start 
            print("p_min:",p_min)
            print("p_max:",p_max)
            verify_time = checking_time + reach_time_duration    

            data = {
                    'Model': 'MNA1',
                    'Spec': spec_id,
                    'Nadnf': Nadnf,
                    'Ncdnf': Ncdnf,
                    'p_min':p_min,
                    'p_max':p_max,
                    't_r':reach_time_duration,
                    't_c':checking_time,
                    't_v': verify_time
                }
            datas.append(data)

    return datas

def run_fom_model(use_arnoldi = None,use_init_space=None):
       
    print('\n\n=====================================================')
    print('Quantitative Verification of FOM Model ')
    print('=====================================================')
    plant = load_fom_model()
    combined_mat = combine_mats(plant.A,plant.B)
    dims = combined_mat.shape[0]

    #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    initial_dim = plant.A.shape[0]

    for dim in range(dims):
        if dim < 50:
            lb = -0.0002  
            ub = 0.00025 
        # elif dim < 500:
        #     lb = 0.0002  
        #     ub = 0.00025 
        elif dim < initial_dim: 
            lb = ub = 0 
        elif dim >= initial_dim:
            lb =-1
            ub = 1
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))
    
        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    # create Star for initial state 
    X0 = Star(init_state_lb,init_state_ub)

    # create ProbStar for initial state 
    mu_X0 = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_X0 = (mu_X0 - X0.pred_lb)/a
    Sig_X0 = np.diag(np.square(sig_X0))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_X0, Sig_X0,X0.pred_lb,X0.pred_ub)
    
    h = [0.1]
    time_bound = 20
    m = 8
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9

    output_space= plant.C
    expand_mat = np.zeros((1,1))
    output_space = np.hstack((output_space,expand_mat))
    initial_space = X0_probstar.V 

    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()
    A1 = np.array([-1])
    b1 = np.array([-7])
    P = AtomicPredicate(A1,b1)
    P1 = AtomicPredicate(-A1,-b1)

    datas = []

    for hi in h:
        N = int (time_bound/ hi)
        reach_start_time = time.time()
        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time

        EVOT =_EVENTUALLY_(0,N)
        EVOT1 =_EVENTUALLY_(50,100)
        AWOT=_ALWAYS_(0,N)
        AWOT1=_ALWAYS_(50,100)

        spec = Formula([EVOT,P]) # be unsafe at some step 0 to N
        spec1 = Formula([AWOT,P1]) # alwyas safe between 0 to N
        spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P,rb,rb])
        specs=[spec,spec1,spec2]

        for spec_id in range(0,len(specs)):
            check_start = time.time()
            spec = specs[spec_id]
            print("\n===================Specification{}: =====================".format(spec_id))
            spec.print()
            DNF_spec = spec.getDynamicFormula()
            Nadnf = DNF_spec.length
            print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
            _,p_max, p_min,Ncdnf = DNF_spec.evaluate(R)
            end = time.time()
            checking_time = end -check_start 
            print("p_min:",p_min)
            print("p_max:",p_max)
            verify_time = checking_time + reach_time_duration    
    
            data = {
                    'Model': 'FOM',
                    'Spec': spec_id,
                    'Nadnf': Nadnf,
                    'Ncdnf': Ncdnf,
                    'p_min':p_min,
                    'p_max':p_max,
                    't_r':reach_time_duration,
                    't_c':checking_time,
                    't_v': verify_time
                }
            datas.append(data)

    return datas

def run_MNA5_model(use_arnoldi=None,use_init_space=None):
    print('\n\n=====================================================')
    print('Quantitative Verification of MNA5 Model ')
    print('=====================================================')

    plant = load_MNA5_model()
    combined_mat = combine_mats(plant.A,plant.B)
    dims = combined_mat.shape[0]

    #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    initial_dim = plant.A.shape[0]
    for dim in range(dims):
        if dim < 10:
            lb = 0.0002
            ub = 0.00025
        elif dim < initial_dim:
            lb = ub = 0
        elif dim >= initial_dim and dim < initial_dim + 5:
            # first 5 inputs
            lb = ub = 0.1
        elif dim >= initial_dim + 5 and dim < initial_dim + 9:
            # second 4 inputs
            lb = ub = 0.2
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))
            
        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)
    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    # create Star for initial state 
    X0 = Star(init_state_lb,init_state_ub)

    # create ProbStar for initial state 
    mu_U = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_U = (mu_U - X0.pred_lb)/a
    epsilon = 1e-10
    sig_U = np.maximum(sig_U, epsilon)
    Sig_U = np.diag(np.square(sig_U))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)

    h = [0.1]
    time_bound = 20
    m = 10
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9

    output_space= random_two_dims_mapping(X0_probstar,1,2)
    initial_space = X0_probstar.V 

    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()
    A1 = np.array([-1,0])
    b1 = np.array([-0.1])
    P = AtomicPredicate(A1,b1)
    P1 = AtomicPredicate(-A1,-b1)
    A2 = np.array([0,-1])
    b2 = np.array([-0.15])
    P2 = AtomicPredicate(A2,b2)

    datas= []
    
    for hi in h:
        N = int (time_bound/ hi)
        reach_start_time = time.time()
        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time

        EVOT =_EVENTUALLY_(0,N)
        EVOT1 =_EVENTUALLY_(100,200)
        AWOT=_ALWAYS_(0,N)
        AWOT1=_ALWAYS_(100,200)

        spec = Formula([EVOT,P,OR,P2]) # be unsafe at some step 0 to N
        spec1 = Formula([AWOT,P1]) # alwyas safe between 0 to N
        spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P,rb,rb])
        specs=[spec,spec1,spec2]

        for spec_id in range(0,len(specs)):
            check_start = time.time()
            spec = specs[spec_id]
            print("\n===================Specification{}: =====================".format(spec_id))
            spec.print()
            DNF_spec = spec.getDynamicFormula()
            Nadnf = DNF_spec.length
            print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
            _,p_max, p_min,Ncdnf = DNF_spec.evaluate(R)
            end = time.time()
            checking_time = end -check_start 
            print("p_min:",p_min)
            print("p_max:",p_max)
            verify_time = checking_time + reach_time_duration    
    
            data = {
                    'Model': 'MNA5',
                    'Spec': spec_id,
                    'Nadnf': Nadnf,
                    'Ncdnf': Ncdnf,
                    'p_min':p_min,
                    'p_max':p_max,
                    't_r':reach_time_duration,
                    't_c':checking_time,
                    't_v': verify_time
                }
            datas.append(data)

    return datas

def heat3D_create_dia(samples, diffusity_const, heat_exchange_const):

    samples_sq = samples**2
    dims = samples**3
    step = 1.0 / (samples + 1)

    a = diffusity_const * 1.0 / step**2
    d = -6.0 * a  

    # Initialize dense matrix
    matrix = np.zeros((dims, dims))

    for i in range(dims):
        z = i // samples_sq
        y = (i % samples_sq) // samples
        x = i % samples

        if z > 0:
            matrix[i, i - samples_sq] = a  
        if y > 0:
            matrix[i, i - samples] = a     
        if x > 0:
            matrix[i, i - 1] = a           

        matrix[i, i] = d

        if z == 0 or z == samples - 1:
            matrix[i, i] += a  
        if y == 0 or y == samples - 1:
            matrix[i, i] += a  
        if x == 0:
            matrix[i, i] += a  
        if x == samples - 1:
            matrix[i, i] += a / (1 + heat_exchange_const * step)  

        if x < samples - 1:
            matrix[i, i + 1] = a           
        if y < samples - 1:
            matrix[i, i + samples] = a     
        if z < samples - 1:
            matrix[i, i + samples_sq] = a 
            
    return matrix

def heat3D_star_vectors(a,samples):


    dims = a.shape[0]

    data = []
    inds = []

    assert samples >= 10 and samples % 10 == 0, "init region isn't evenly divided by discretization"

    for z in range(int(samples / 10 + 1)):
        zoffset = z * samples * samples

        for y in range(int(2 * samples / 10 + 1)):
            yoffset = y * samples

            for x in range(int(4 * samples / 10 + 1)):
                dim = x + yoffset + zoffset

                data.append(1)
                inds.append(dim)

    init_space = np.zeros((dims, 1))
    for i in inds:
        init_space[i, 0] = 1

    init_mat = np.array([[1], [-1.]], dtype=float)
    init_mat_rhs = np.array([1.1, -0.9], dtype=float)

    return init_space,init_mat,init_mat_rhs

def run_heat3D_model(use_arnoldi=None,use_init_space=None):
    
    print('\n\n=====================================================')
    print('Quantitative Verification of Heat3D Model ')
    print('=====================================================')

    diffusity_const = 0.01
    heat_exchange_const = 0.5
    samples_per_side = 20
    dims =samples_per_side**3

    print ("Making {}x{}x{} ({} dims) 3d Heat Plate ODEs...".format(samples_per_side, samples_per_side, \
                                                                samples_per_side, samples_per_side**3))

    a_matrix = heat3D_create_dia(samples_per_side, diffusity_const, heat_exchange_const)
    init_space,init_mat,init_mat_rhs = heat3D_star_vectors(a_matrix,samples_per_side)
    
    #  returns list of initial states for each dimension
    init_state_bounds_list = []

    for dim in range(dims):
       if init_space[dim] == 1:
            lb = 0.9
            ub = 1.1
            init_state_bounds_list.append((lb, ub))
       else:
            lb = 0
            ub = 0
            init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    X0 = Star(init_state_lb,init_state_ub)
    
    mu_U = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_U = (mu_U - X0.pred_lb)/a
    Sig_U = np.diag(np.square(sig_U))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)

    h = [0.1]
    time_bound = 20
    m = 8
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9

    center_x = int(math.floor(samples_per_side/2.0))
    center_y = int(math.floor(samples_per_side/2.0))
    center_z = int(math.floor(samples_per_side/2.0))
    center_dim = center_z * samples_per_side * samples_per_side + center_y * samples_per_side + center_x
    output_space= np.zeros((1,dims))
    output_space[0,center_dim] =1
    initial_space = X0_probstar.V 

    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()
    A1 = np.array([-1])
    b1 = np.array([-0.012])
    P = AtomicPredicate(A1,b1)
    P1 = AtomicPredicate(-A1,-b1)
    
    datas = []
    
    for hi in h:
        N = int (time_bound/ hi)
        reach_start_time = time.time()
        R, krylov_time = sim3(a_matrix,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time

        EVOT =_EVENTUALLY_(0,N)
        EVOT1 =_EVENTUALLY_(50,180)
        AWOT=_ALWAYS_(0,N)
        AWOT1=_ALWAYS_(50,180)

        spec = Formula([EVOT,P]) # be unsafe at some step 0 to N
        spec1 = Formula([AWOT,P1]) # alwyas safe between 0 to N
        spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P,rb,rb])
        specs=[spec,spec1,spec2]

        for spec_id in range(0,len(specs)):
            check_start = time.time()
            spec = specs[spec_id]
            print("\n===================Specification{}: =====================".format(spec_id))
            spec.print()
            DNF_spec = spec.getDynamicFormula()
            Nadnf=DNF_spec.length
            print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
            _,p_max, p_min,Ncdnf = DNF_spec.evaluate(R)
            end = time.time()
            checking_time = end -check_start 
            print("p_min:",p_min)
            print("p_max:",p_max)

            verify_time = checking_time + reach_time_duration    

            data = {
                    'Model': 'Heat3D',
                    'Spec': spec_id,
                    'Nadnf': Nadnf,
                    'Ncdnf': Ncdnf,
                    'p_min':p_min,
                    'p_max':p_max,
                    't_r':reach_time_duration,
                    't_c':checking_time,
                    't_v': verify_time
                }
            datas.append(data)

    return datas
     
def full_evaluation_results():

    results = []

    building_result = run_building_model(use_arnoldi=True, use_init_space=False)
    mcs_result = run_mcs_model(use_arnoldi=True, use_init_space=False)
    pde_result = run_pde_model(use_arnoldi=True, use_init_space=False)
    iss_result = run_iss_model(use_arnoldi=True, use_init_space=False)
    beam_result = run_beam_model(use_arnoldi=True, use_init_space=False)
    fom_result = run_fom_model(use_arnoldi=True, use_init_space=False)
    mna1_result = run_MNA1_model(use_arnoldi=True, use_init_space=False)
    mna5_result = run_MNA5_model(use_arnoldi=True, use_init_space=False)

    # Heat#d model using Lanczos iteration due to symmetric dynamics, set use_arnoldi to False
    heat3D_result = run_heat3D_model(use_arnoldi=False, use_init_space=False)

    results.extend(mcs_result + building_result + pde_result + iss_result + beam_result + mna1_result + fom_result + mna5_result + heat3D_result)

    print(tabulate([[r['Model'], r['Spec'], r['Nadnf'],r['Ncdnf'],r['p_min'], r['p_max'], 
                    r['t_r'], r['t_c'], r['t_v']] for r in results],
                  headers=["Model", "Spec","Nadnf","Ncdnf","p_min", "p_max",
                          "t_r", "t_c", "t_v"]))

    # Save results to pickle file
    cur_path = os.path.dirname(__file__)
    path = cur_path + '/results' 
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/full_results.pkl', 'wb') as f:
        pickle.dump(results, f)

def verification_Hylaa_tool():
    """
    Verify all benchamrks with using Hylaa verification tool
    """
    data = [
    {"Model": 'Motor', "Spec": 0, "SAT": 'YES', "t_v": 0.005317},
    {"Model": 'Building', "Spec": 0, "SAT": 'NO', "t_v": 0.160578}, 
    {"Model": 'PDE', "Spec": 0, "SAT": 'YES', "t_v": 0.371748},
    {"Model": 'ISS', "Spec": 0, "SAT": 'YES', "t_v": 3.292322},
    {"Model": 'Beam', "Spec": 0, "SAT": 'YES', "t_v": 4.475225},
    {"Model": 'MNA1', "Spec": 0, "SAT": 'YES', "t_v": 3.634282},
    {"Model": 'FOM', "Spec": 0, "SAT": 'YES', "t_v": 22.683661},
    {"Model": 'MNA5', "Spec": 0, "SAT": 'YES', "t_v": 4.217258},
    {"Model": 'Heat3D', "Spec": 0, "SAT": 'YES', "t_v": 0.066165}
    ]

    results = []
    for entry in data:
        result = {
            'Model': entry['Model'],
            'Spec': 0,
            'SAT': entry['SAT'],
            't_v': entry['t_v'],
        }
        results.append(result)
    
    # Save to pickle file
    cur_path = os.path.dirname(__file__)
    path = cur_path + '/results' 
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/Hylaa_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def generate_table_3_and_table_4():

    print('\n\n==================================================================')
    print('======================= Run All Benchmarks =======================')
    print('==================================================================\n')

    """
    Generate LaTeX table combining verification results for massive linear system:
    1. Quantitative Verification (ProbStar)
    2. Hylaa tool verification results
    """
    def format_number(value):
        """Preserve scientific notation for very small numbers and original format"""
        if pd.isna(value) or value is None:
            return ''
        elif isinstance(value, int):
            return f'{value}'
        elif isinstance(value, float):
            if abs(value) ==0: 
                return f'{0}'
            if abs(value) < 1e-5:  # Use scientific notation for very small numbers
                return f'{value:.6e}'
            elif abs(value) >= 1:  # Use fixed notation with reasonable precision
                return f'{value:.6f}'.rstrip('0').rstrip('.')
            else:  # For numbers between 0 and 1
                return f'{value:.6f}'.rstrip('0').rstrip('.')
        return str(value)

    def load_pickle_file(filename):
        """Load data from pickle file with error handling"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            return []
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return []
        
    full_evaluation_results()
    verification_Hylaa_tool()

    cur_path = os.path.dirname(__file__)
    path = cur_path + '/results'     
    # Load all data sources
    probstarTL_data = load_pickle_file(path + '/full_results.pkl')

    Hylaa_data = load_pickle_file(path + '/Hylaa_results.pkl')

    print("\n=============== Begin generating Table 3 =======================")
    print("Verification results of all models compare with Hylaa\n")

    # Create lookup dictionaries for MC and other tools data
    Hylaa_dict = {(d['Model'], d['Spec']): d for d in Hylaa_data}

    # Generate LaTeX table
    table_lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllllll|ll}",
        r"\hline",
        r"    \multicolumn{7}{c}{\textbf{ProbStarTL}}  & " + 
        r"\multicolumn{2}{c}{\textbf{Hylaa}} \\",
        r"\hline",
        r"\text{model} & \textbf{$\varphi$} & \textbf{$p_{min}$} & " +
        r"\textbf{$p_{max}$} & \textbf{$t_r$} & " +
        r"\textbf{$t_c$} & \textbf{$t_v$} & \text{SAT} & \textbf{$t_v$} \\",
        r"\hline"
    ]

    # Add data rows
    for entry in probstarTL_data:
        model = entry['Model']
        spec = entry['Spec']
        
        # Base row with Quantitative Verification data
        row = [
            str(model),
            spec,
            format_number(entry['p_min']),
            format_number(entry['p_max']),
            format_number(entry['t_r']),
            format_number(entry['t_c']),
            format_number(entry['t_v']),
        ]

        # Add Hylaa tool data 
        if spec==0:
            Hylaa_entry = Hylaa_dict.get((model, spec), {})
            
            row.extend([
                format_number(Hylaa_entry.get('SAT', '')),
                format_number(Hylaa_entry.get('t_v', ''))

            ])
        else:
            # Add empty cells for Hlyaa Tool columns
            row.extend([''] * 2)

        table_lines.append(' & '.join(map(str,row)) + r' \\')

    # Add table footer
    table_lines.extend([
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ])

    # Join all lines with newlines and save to file
    table_content = '\n'.join(table_lines)
    
    with open(path + '/Table_3_vs_Hylaa.tex', 'w') as f:
        f.write(table_content)

    print("Table3(full results of nine benchmarks)has been generated and saved to 'results/Table_3_vs_Hylaa.tex' \n")

    print("=============== Begin generating Table 4 ======================= ")
    print("Number of ADNF and CDNF of each specification\n")

    data_dict = {
        (d['Model'], d['Spec']): d
        for d in probstarTL_data
    }

    models = ['Motor','Building','PDE','ISS','Beam','MNA1','FOM','MNA5','Heat3D']
    specs  = [0,1,2]

    table_lines = [
        r"\begin{table*}[h]",
        r"\vspace{-4mm}",
        r"    \centering",
        r"    % \scriptsize",
        r"    \resizebox{0.9\textwidth}{!}{%",
        # build the column spec: one 'c' + '|' + nine groups of 'ccc|', but last group without trailing |
        r"    \begin{tabular}{c|" + "|".join(["ccc"]*(len(models)-1)) + "|ccc}",
        r"    \hline",
        # 1st header row: blank cell + one multicol for each model
        "    & " + " & ".join(
            rf"\multicolumn{{3}}{{c|}}{{{m}}}" for m in models[:-1]
        ) + " & " + rf"\multicolumn{{3}}{{c}}{{{models[-1]}}} \\",
        r"    \hline",
        # 2nd header row: φ labels repeated
        "    & " + " & ".join(
            r"$\varphi_1$ & $\varphi_2$ & $\varphi_3$"
            for _ in models
        ) + r" \\",
        r"    \hline",
    ]

    # N_adnf
    row_adnf = ["    $N_{adnf}$"]
    for m in models:
        for s in specs:
            ent = data_dict.get((m, s), {})
            row_adnf.append(str(ent.get("Nadnf", "")))
    table_lines.append("    " + " & ".join(row_adnf) + r" \\")

    # N_cdnf
    row_cdnf = ["    $N_{cdnf}$"]
    for m in models:
        for s in specs:
            ent = data_dict.get((m, s), {})
            row_cdnf.append(str(ent.get("Ncdnf", "")))
    table_lines.append("    " + " & ".join(row_cdnf) + r" \\")

    # footer
    table_lines += [
        r"    \hline",
        r"    \end{tabular}",
        r"    }",
        r"    \vspace{3pt}",
        r"    \caption{Length of ADNF and CDNF of each specification.}",
        r"    \label{tab: Number of CDNF of all benchmarks}",
        r"    \vspace{-10mm}",
        r"\end{table*}"
    ]

    # table_lines.append(" & ".join(row_cdnf) + r" \\")

    # write it out
    with open(path + '/Table_4.tex','w') as f:
        f.write("\n".join(table_lines))
    print("Table_4(the length of ADNF and CDNF) has been generated and saved to 'results/Table_4.tex' \n")

    return table_content


if __name__ == '__main__':

    # generate Table 3: Verification results of all models compare with Hylaa and Table 4: Length of ADNF and CDNF of each specification.
    # generate_table_3_and_table_4()
   
    # generate Table 1: Verification results of Harmonic example and Figure 1: Computation of the ProbStar signal.
    run_harmonic()


