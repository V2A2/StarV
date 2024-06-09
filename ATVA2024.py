"""
Verify ACC and AEBS system
Author: Anomynous
Date: 4/19/2024
"""


from StarV.verifier.verifier import quantiVerifyBFS
from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load import load_acc_model, load_AEBS_model
from StarV.util.plot import plot_probstar_reachset, plot_probstar_reachset_with_unsafeSpec
from StarV.net.network import reachExactBFS
import time
from StarV.util.plot import plot_probstar
from matplotlib import pyplot as plt
from matplotlib.pyplot import step, show
from StarV.set.star import Star
from tabulate import tabulate
from StarV.nncs.nncs import VerifyPRM_NNCS, verifyBFS_DLNNCS, reachBFS_AEBS, ReachPRM_NNCS, AEBS_NNCS
import os
import copy
import multiprocessing
from matplotlib.patches import Rectangle



def generate_exact_reachset_figs(net_id='5x20'):
    'generate 4 pictures and save in NEURIPS2024/pics/'

    if net_id == '5x20':
        net = 'controller_5_20'
    elif net_id == '3x20':
        net = 'controller_3_20'
    elif net_id == '7x30':
        net = 'controller_7_20'
    else:
        raise RuntimeError('Invalid net_id')
    
    plant='linear'
    initSet_id=5
    pf=0.0
    numSteps=30
    numCores=4

    # load NNCS ACC system
    print('Loading the ACC system...')
    ncs, initSets, refInputs, unsafe_mat, unsafe_vec = load_acc_model(net,plant)
    
    

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
    
    print('Verifying the ACC system for {} timesteps using exact reachability...'.format(numSteps))
    res = verifyBFS_DLNNCS(ncs, verifyPRM)
    RX = res.RX
    Ce = res.CeIn
    Co = res.CeOut
    
    n = len(RX)
    CE = []
    CO = []
    for i in range(0,n):
        if len(Ce[i]) > 0:
            CE.append(Ce[i])  # to plot counterexample set
            CO.append(Co[i])

    # plot reachable set  (d_actual - d_safe) vs. (v_ego)
    dir_mat1 = np.array([[0., 0., 0., 0., 1., 0., 0.],
                        [1., 0., 0., -1., -1.4, 0., 0.]])
    dir_vec1 = np.array([0., -10.])

    # plot reachable set  d_rel vs. d_safe
    dir_mat2 = np.array([[1., 0., 0., -1., 0., 0., 0.],
                        [0., 0., 0., 0., 1.4, 0., 0.]])
    dir_vec2 = np.array([0., 10.])

    # plot counter input set
    dir_mat3 = np.array([[0., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 1., 0., 0.]])
    dir_vec3 = np.array([0., 0.])


    path = "artifacts/NEURIPS2024_Algebra/ACC/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    print('Plot reachable set...')
    plot_probstar_reachset(RX, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=False, \
                           label=('$v_{ego}$','$d_r - d_{safe}$'), show=False)
    plt.savefig(path+"/dr_dsafe_vs_vego_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure
    plt.show()

    plot_probstar_reachset(RX, dir_mat=dir_mat2, dir_vec=dir_vec2, show_prob=False, \
                           label=('$d_{r}$','$d_{safe}$'), show=False)
    plt.savefig(path+"/dr_vs_dsafe_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure
    plt.show()
    

    if net_id=='5x20':
        print('Plot counter output set...')
        plot_probstar_reachset(CO, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=False, \
                               label=('$v_{ego}$','$d_r - d_{safe}$'), show=False)
        plt.savefig(path+"/counterOutputSet_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure
        plt.show()

        print('Plot counter init set ...')
        plot_probstar_reachset(CE, dir_mat=dir_mat3, dir_vec=dir_vec3, show_prob=False, \
                               label=('$v_{lead}[0]$','$v_{ego}[0]$'), show=False)
        plt.savefig(path+"/counterInitSet_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure
        plt.show()

    print('Done!')



def generate_exact_Q2_verification_results(net_id='5x20'):
    'generate Q2 verification results'

    if net_id == '5x20':
        net = 'controller_5_20'
    elif net_id == '3x20':
        net = 'controller_3_20'
    elif net_id == '7x30':
        net = 'controller_7_20'
    else:
        raise RuntimeError('Invalid net_id')
    
    plant='linear'
    initSet_id=5
    pf=0.0
    numSteps=30
    numCores=4

    # load NNCS ACC system
    print('Loading the ACC system...')
    ncs, initSets, refInputs, unsafe_mat, unsafe_vec = load_acc_model(net,plant)
    
    

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
    
    print('Verifying the ACC system controlled by Net_3x20 for {} timesteps using exact reachability...'.format(numSteps))
    res = verifyBFS_DLNNCS(ncs, verifyPRM)
    Ql = res.Ql
    Qt = res.Qt
    Qt_max = res.Qt_max

    t = range(0, 1, numSteps)
    
    path = "artifacts/NEURIPS2024_Algebra/ACC/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    print('Plot Q2 verification results...')

    print('Plot exact qualitative results...')
    fig1 = plt.figure()
    xaxis = np.arange(0, len(Ql))
    yaxis = np.array(Ql)
    step(xaxis, yaxis)
    label=('t', 'SAT')
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(path+"/Ql_exact_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure
    #show()

    fig2 = plt.figure()
    
    print('Plot exact quantitative results Qt...')
    yaxis1 = np.array(Qt)
    yaxis2 = np.array(Qt_max)

    plt.plot(xaxis, yaxis1, color='blue', marker='o', label='$Qt$')
    plt.plot(xaxis, yaxis2, color='red', marker='x', label='$Qt_{max}$')
    label=('t', '$Qt, Qt_{max}$')
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend()
    plt.savefig(path+"/Qt_Qt_max_exact_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure
    show()

    print('Done!')

    
def generate_approx_Q2_verification_results(net_id='5x20', pf=0.001):
    'generate approx Q2 verification results'

    if net_id == '5x20':
        net = 'controller_5_20'
    elif net_id == '3x20':
        net = 'controller_3_20'
    elif net_id == '7x30':
        net = 'controller_7_20'
    else:
        raise RuntimeError('Invalid net_id')
    
    plant='linear'
    initSet_id=5
    numSteps=30
    numCores=4

    # load NNCS ACC system
    print('Loading the ACC system...')
    ncs, initSets, refInputs, unsafe_mat, unsafe_vec = load_acc_model(net,plant)
    
    

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
    
    print('Verifying the ACC system for {} timesteps using approx reachability. with pf = {}..'.format(numSteps, pf))
    res = verifyBFS_DLNNCS(ncs, verifyPRM)
    Ql = res.Ql
    Qt = res.Qt
    Qt_ub = res.Qt_ub
    Qt_max = res.Qt_max
    
    t = range(0, 1, numSteps)
    
    path = "artifacts/NEURIPS2024_Algebra/ACC/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    print('Plot Q2 verification results...')

    print('Plot approx qualitative results...')
    fig1 = plt.figure()
    xaxis = np.arange(0, len(Ql))
    yaxis = np.array(Ql)
    step(xaxis, yaxis)
    label=('t', 'SAT')
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(path+"/Ql_approx_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure
    #show()

    fig2 = plt.figure()
    
    print('Plot approx quantitative results...')
    yaxis1 = np.array(Qt)
    yaxis2 = np.array(Qt_ub)
    yaxis3 = np.array(Qt_max)

    plt.plot(xaxis, yaxis1, color='blue', marker='o', label='$Qt$')
    plt.plot(xaxis, yaxis2, color='orange', marker='*', label='$Qt_{ub}$')
    plt.plot(xaxis, yaxis3, color='red', marker='x', label='$Qt_{max}$')
    label=('t', '$Qt, Qt_{lb}, Qt_{ub}$')
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend()
    plt.savefig(path+"/Qt_Qtub_Qt_max_approx_Net_{}.png".format(net_id), bbox_inches='tight')  # save figure
    show()
    
    print('Done!')



def generate_VT_Conv_vs_pf_net():
    'generate verification time vesus pf and networks'

    nets=['controller_3_20', 'controller_5_20']
    plant='linear'
    initSet_id=5
    #pf = [0.0, 0.1]
    pf = [0.0, 0.005, 0.01, 0.015]
    numSteps=30
    numCores=4

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()


    
    m = len(pf)
    n = len(nets)
    VT = np.zeros((m, n))
    NO = np.zeros((m,n))
    VT_improv = np.zeros((m, n))  # improvement in verification time
    Conv = np.zeros((m, n))       # conservativeness of prediction
    Qt_ub_sum = np.zeros((m, n))
    Qt_sum = np.zeros((m, n))
    Qt_exact_sum = np.zeros((n,))
    data = []
    for i in range(0, m): 
        for j in range(0, n):
            # load NNCS ACC system
            print('Loading the ACC system with network {}...'.format(nets[j]))
            ncs, initSets, refInputs, unsafe_mat, unsafe_vec = load_acc_model(nets[j],plant)
            verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
            verifyPRM.refInputs = copy.deepcopy(refInputs)
            verifyPRM.numSteps = numSteps
            verifyPRM.pf = pf[i]
            verifyPRM.numCores = numCores
            verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
            
            print('Verifying the ACC system with {} for {} \
            timesteps using approx reachability with pf = {}...'.format(nets[j], numSteps, pf[i]))
            start_time = time.time()
            res = verifyBFS_DLNNCS(ncs, verifyPRM)
            end_time = time.time()


            RX = res.RX
            Qt = res.Qt
            Qt_ub = res.Qt_ub
            Qt_ub_sum[i,j] = sum(Qt_ub)
            Qt_sum[i,j] = sum(Qt)
            if i==0:
                Qt_exact_sum[j] = sum(Qt)          
            M = len(RX)
            NO[i,j] = len(RX[M-1])
            VT[i, j] = end_time - start_time
            VT_improv[i,j] = (VT[0,j] - VT[i, j])*100/VT[0,j]
            Conv[i,j] = 100*(Qt_ub_sum[i,j] - Qt_exact_sum[j])/Qt_exact_sum[j]    
            strg1 = '{}'.format(pf[i])
            strg2 = '{}'.format(nets[j])
            data.append([strg1, strg2, VT[i,j], NO[i,j], VT_improv[i,j], Conv[i,j]])
            
    print(tabulate(data, headers=["p_filter", "network", "verification time", "number of output sets", "VT_improve (in %)", "Convativeness (in %)"]))   
    
    path = "artifacts/NEURIPS2024_Algebra/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/VT_Conv_vs_pf_net.tex", "w") as f:
         print(tabulate(data, headers=["p_filter", "network", "verification time", "number of output sets", "VT_improve", "Conservativeness"], tablefmt='latex'), file=f)
       
    
    print('Done!')


            

def generate_exact_reachset_figs_AEBS():

    # load initial conditions of NNCS AEBS system
    print('Loading initial conditions of AEBS system...')
    
    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
    AEBS = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)
    

    # stepReach computation for AEBS

    # step 0: initial state of plant
    # step 1: normalizing state
    # step 2: compute brake output of RL controller
    # step 3: compose brake output and normalized speed
    # step 4: compute transformer output
    # step 5: get control output
    # step 6: scale control output
    # step 7: compute reachable set of the plannt with the new control input and initial state

    # step 8: go back to step 1, .... (another stepReach)

    # initial bound on states
    d_lb = [97., 90., 48., 5.0]
    d_ub = [97.5, 90.5, 48.5, 5.2]
    v_lb = [25.2, 27., 30.2, 1.0]
    v_ub = [25.5, 27.2, 30.4, 1.2]

   

    reachPRM = ReachPRM_NNCS()
    reachPRM.numSteps = 50
    reachPRM.filterProb = 0.0
    reachPRM.numCores = 4
    

    for i in range(0, len(initSets)):
        reachPRM.initSet = initSets[i]     
        X, _ = reachBFS_AEBS(AEBS, reachPRM)

        # plot reachable set  (d) vs. (v_ego)
        dir_mat1 = np.array([[1., 0., 0.],
                            [0., 1., 0]])
        dir_vec1 = np.array([0., 0.])



        path = "artifacts/NEURIPS2024_Algebra/AEBS/pics"
        if not os.path.exists(path):
            os.makedirs(path)

        # 0.5 <= d_k <= 2.5 AND 0.2 <= v_k <= v_ub
        unsafe_mat = np.array([[1.0, 0.0], [-1., 0.], [0., 1.], [0., -1.]])
        unsafe_vec = np.array([2.5, -0.5, v_ub[i], -0.2])
        
        print('Plot reachable set...')
        plot_probstar_reachset_with_unsafeSpec(X, unsafe_mat, unsafe_vec, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=False, \
                               label=('$d$','$v_{ego}$'), show=False, color='g')
        plt.savefig(path+"/d_vs_vego_init_{}.png".format(i), bbox_inches='tight')  # save figure
        plt.show()



def generate_AEBS_exact_Q2_verification_results(initSet_id=0):
    'generate Q2 verification results'

    # load NNCS AEBS system
    print('Loading the AEBS system...')
    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
    AEBS = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)


    numSteps = 50
    numCores = 4
    pf = 0.0
    # 0.5 <= d_k <= 2.5 and v_k >= 0.2
    unsafe_mat = np.array([[1., 0., 0.], [-1., 0., 0], [0., -1., 0.]])
    unsafe_vec = np.array([2.5, -0.5, -0.2])


    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
    
    
    
    print('Verifying the AEBS for {} timesteps using exact reachability...'.format(numSteps))
    res = verifyBFS_DLNNCS(AEBS, verifyPRM)
    Ql = res.Ql
    Qt = res.Qt
    Qt_max = res.Qt_max

    t = range(0, 1, numSteps)
    
    path = "artifacts/NEURIPS2024_Algebra/AEBS/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    print('Plot Q2 verification results...')

    print('Plot exact qualitative results...')
    fig1 = plt.figure()
    xaxis = np.arange(0, len(Ql))
    yaxis = np.array(Ql)
    step(xaxis, yaxis)
    label=('t', 'SAT')
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(path+"/Ql_exact_initSet_{}.png".format(initSet_id), bbox_inches='tight')  # save figure
    #show()

    fig2 = plt.figure()
    
    print('Plot exact quantitative results Qt...')
    yaxis1 = np.array(Qt)
    yaxis2 = np.array(Qt_max)

    plt.plot(xaxis, yaxis1, color='blue', marker='o', label='$Qt$')
    plt.plot(xaxis, yaxis2, color='red', marker='x', label='$Qt_{max}$')
    label=('t', '$Qt, Qt_{max}$')
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend()
    plt.savefig(path+"/Qt_Qt_max_exact_initSet_{}.png".format(initSet_id), bbox_inches='tight')  # save figure
    show()

    print('Done!')

def generate_AEBS_approx_Q2_verification_results(initSet_id=0, pf=0.01):
    'generate Q2 verification results'

    # load NNCS AEBS system
    print('Loading the AEBS system...')
    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
    AEBS = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)


    numSteps = 50
    numCores = 4
    # 0.5 <= d_k <= 2.5 and v_k >= 0.2
    unsafe_mat = np.array([[1., 0., 0.], [-1., 0., 0], [0., -1., 0.]])
    unsafe_vec = np.array([2.5, -0.5, -0.2])


    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
    
    
    
    print('Verifying the AEBS for {} timesteps using exact reachability...'.format(numSteps))
    res = verifyBFS_DLNNCS(AEBS, verifyPRM)

    Qt = res.Qt
    Qt_ub = res.Qt_ub
    Qt_max = res.Qt_max

    t = range(0, 1, numSteps)
    
    path = "artifacts/NEURIPS2024_Algebra/AEBS/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    print('Plot Q2 approximate verification results...')


    fig2 = plt.figure()
    xaxis = np.arange(0, len(Qt))
    yaxis1 = np.array(Qt)
    yaxis2 = np.array(Qt_ub)
    yaxis3 = np.array(Qt_max)

    plt.plot(xaxis, yaxis1, color='blue', marker='o', label='$Qt$')
    plt.plot(xaxis, yaxis2, color='orange', marker='*', label='$Qt_{ub}$')
    plt.plot(xaxis, yaxis3, color='red', marker='x', label='$Qt_{max}$')
    label=('t', '$Qt, Qt_{ub}, Qt_{max}$')
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend()
    plt.savefig(path+"/Qt_Qt_ub_Qt_max_approx_initSet_{}.png".format(initSet_id), bbox_inches='tight')  # save figure
    show()

    print('Done!')



def generate_AEBS_Q2_verification_results(initSet_id=0, pf=0.005):
    'generate Q2 verification results'

    # load NNCS AEBS system
    print('Loading the AEBS system...')
    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
    AEBS = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)


    numSteps = 50
    numCores = 4
    # 0.5 <= d_k <= 2.5 and v_k >= 0.2
    unsafe_mat = np.array([[1., 0., 0.], [-1., 0., 0], [0., -1., 0.]])
    unsafe_vec = np.array([2.5, -0.5, -0.2])


    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.numSteps = numSteps
    
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
    
    
    verifyPRM.pf = 0.0
    res = verifyBFS_DLNNCS(AEBS, verifyPRM)
    Qt_exact = res.Qt

    verifyPRM.pf = pf
    res = verifyBFS_DLNNCS(AEBS, verifyPRM)
    Qt_approx = res.Qt
    Qt_ub = res.Qt_ub
    Qt_max = res.Qt_max

    t = range(0, 1, numSteps)
    
    path = "artifacts/NEURIPS2024_Algebra/AEBS/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    print('Plot Q2 verification results...')


    fig2 = plt.figure()
    xaxis = np.arange(0, len(Qt_exact))
    yaxis1 = np.array(Qt_exact)
    yaxis2 = np.array(Qt_approx)
    yaxis3 = np.array(Qt_ub)
    yaxis4 = np.array(Qt_max)

    plt.plot(xaxis, yaxis1, color='blue', marker='o', label='$Qt_{exact}$')
    plt.plot(xaxis, yaxis2, color='green', marker='>', label='$Qt_{approx}$')
    plt.plot(xaxis, yaxis3, color='orange', marker='*', label='$Qt_{ub}$')
    plt.plot(xaxis, yaxis4, color='red', marker='x', label='$Qt_{max}$')
    label=('t', '$Qt_{exact}, Qt_{approx}, Qt_{ub}, Qt_{max}$')
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend()
    plt.savefig(path+"/Qt_exact_Qt_approx_Qt_ub_Qt_max_approx_initSet_{}.png".format(initSet_id), bbox_inches='tight')  # save figure
    show()

    print('Done!')



def generate_AEBS_VT_Conv_vs_pf_initSets():
    'generate verification time vesus pf and networks'


    # load NNCS AEBS system
    print('Loading the AEBS system...')
    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
    AEBS = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)


    numSteps = 50
    numCores = 4
    #pf = [0.0, 0.2]
    pf = [0.0, 0.0025, 0.005]

    # 0.5 <= d_k <= 2.5 and v_k >= 0.2
    unsafe_mat = np.array([[1., 0., 0.], [-1., 0., 0], [0., -1., 0.]])
    unsafe_vec = np.array([2.5, -0.5, -0.2])

    d_lb = [97., 90., 48., 5.0]
    d_ub = [97.5, 90.5, 48.5, 5.2]
    v_lb = [25.2, 27., 30.2, 1.0]
    v_ub = [25.5, 27.2, 30.4, 1.2]

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    
    m = len(pf)
    n = len(initSets)
    VT = np.zeros((m, n))
    NO = np.zeros((m,n))
    VT_improv = np.zeros((m, n))  # improvement in verification time
    Conv = np.zeros((m, n))       # conservativeness of prediction
    Qt_ub_sum = np.zeros((m, n))
    Qt_sum = np.zeros((m, n))
    Qt_exact_sum = np.zeros((n,))
    data = []
    for i in range(0, m): 
        for j in range(0, n):
            
            verifyPRM.initSet = copy.deepcopy(initSets[j])
            verifyPRM.numSteps = numSteps
            verifyPRM.pf = pf[i]
            verifyPRM.numCores = numCores
            verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
            
            print('Verifying the AEBS system with initSet {} for {} \
            timesteps using approx reachability with pf = {}...'.format(j, numSteps, pf[i]))
            start_time = time.time()
            res = verifyBFS_DLNNCS(AEBS, verifyPRM)
            end_time = time.time()

            RX = res.RX
            Qt = res.Qt
            Qt_ub = res.Qt_ub
            Qt_ub_sum[i,j] = sum(Qt_ub)
            Qt_sum[i,j] = sum(Qt)
            if i==0:
                Qt_exact_sum[j] = sum(Qt)          
            M = len(RX)
            NO[i,j] = len(RX[M-1])
            VT[i, j] = end_time - start_time
            VT_improv[i,j] = (VT[0,j] - VT[i, j])*100/VT[0,j]
            Conv[i,j] = 100*(Qt_ub_sum[i,j] - Qt_exact_sum[j])/Qt_exact_sum[j]    
            strg1 = '{}'.format(pf[i])
            strg2 = '[{},{}][{},{}]'.format(d_lb[j], d_ub[j], v_lb[j], v_ub[j])
            data.append([strg1, strg2, VT[i,j], NO[i,j], VT_improv[i,j], Conv[i,j]])
            
    print(tabulate(data, headers=["p_filter", "initSet", "verification time", "number of output sets", "VT_improve (in %)", "Convativeness (in %)"]))   
    
    path = "artifacts/NEURIPS2024_Algebra/AEBS/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/VT_Conv_vs_pf_initSet.tex", "w") as f:
         print(tabulate(data, headers=["p_filter", "initSet", "verification time", "number of output sets", "VT_improve", "Conservativeness"], tablefmt='latex'), file=f)
       
    
    print('Done!')



    
if __name__ == "__main__":

    # verify ACC model
    
    #generate_exact_reachset_figs(net_id='3x20')
    #generate_exact_reachset_figs(net_id='5x20')
    #generate_exact_Q2_verification_results(net_id='3x20')
    #generate_exact_Q2_verification_results(net_id='5x20')
    #generate_approx_Q2_verification_results(net_id='3x20', pf=0.01)
    #generate_approx_Q2_verification_results(net_id='5x20', pf=0.01)
    generate_VT_Conv_vs_pf_net()

    # verify AEBS model
    #generate_exact_reachset_figs_AEBS()
    #generate_AEBS_exact_Q2_verification_results(initSet_id=0)
    #generate_AEBS_exact_Q2_verification_results(initSet_id=1)
    #generate_AEBS_exact_Q2_verification_results(initSet_id=2)
    #generate_AEBS_exact_Q2_verification_results(initSet_id=3)
    #generate_AEBS_approx_Q2_verification_results(initSet_id=0, pf=0.005)
    #generate_AEBS_approx_Q2_verification_results(initSet_id=1, pf=0.005)
    #generate_AEBS_approx_Q2_verification_results(initSet_id=2, pf=0.005)
    #generate_AEBS_approx_Q2_verification_results(initSet_id=3, pf=0.005)
    #generate_AEBS_Q2_verification_results(initSet_id=0, pf=0.005)
    #generate_AEBS_Q2_verification_results(initSet_id=1, pf=0.005)
    #generate_AEBS_Q2_verification_results(initSet_id=2, pf=0.005)
    #generate_AEBS_Q2_verification_results(initSet_id=3, pf=0.005)
    #generate_AEBS_VT_Conv_vs_pf_initSets()
