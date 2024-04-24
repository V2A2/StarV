"""
Verify ACC and AEBS system
Author: Anomynous
Date: 4/19/2024
"""


from StarV.verifier.verifier import quantiVerifyBFS
from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load import load_acc_model, load_AEBS_model
from StarV.util.plot import plot_probstar_reachset
from StarV.net.network import reachExactBFS
import time
from StarV.util.plot import plot_probstar
from matplotlib import pyplot as plt
from matplotlib.pyplot import step, show
from StarV.set.star import Star
from tabulate import tabulate
from StarV.nncs.nncs import VerifyPRM_NNCS, verifyBFS_DLNNCS
import os
import copy
import multiprocessing


def generate_exact_reachset_figs():
    'generate 4 pictures and save in ATVA2024/pics/'

    net='controller_5_20'
    plant='linear'
    initSet_id=5
    pf=0.0
    numSteps=19
    numCores=4

    # load NNCS ACC system
    print('Loading the ACC system...')
    ncs, initSets, refInputs, unsafe_mat, unsafe_vec = load_acc_model(net,plant)
    
    

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.verifyMethod = 'Q2'
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
    
    print('Verifying the ACC system for {} timesteps using exact reachability...'.format(numSteps))
    RX, _, _, Ce, Co, _, _, _, _, _ = verifyBFS_DLNNCS(ncs, verifyPRM)
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


    path = "artifacts/DAC2024/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    RX1 = RX[18:19]
    print('Plot reachable set...')
    plot_probstar_reachset(RX1, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=True, \
                           label=('$v_{ego}$','$d_r - d_{safe}$'), show=False)
    plt.savefig(path+"/dr_dsafe_vs_vego.png", bbox_inches='tight')  # save figure
    #plt.show()

    plot_probstar_reachset(RX, dir_mat=dir_mat2, dir_vec=dir_vec2, show_prob=True, \
                           label=('$d_{r}$','$d_{safe}$'), show=False)
    plt.savefig(path+"/dr_vs_dsafe.png", bbox_inches='tight')  # save figure
    #plt.show()
    

    print('Plot counter output set...')
    plot_probstar_reachset(CO, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=True, \
                           label=('$v_{ego}$','$d_r - d_{safe}$'), show=False)
    plt.savefig(path+"/counterOutputSet.png", bbox_inches='tight')  # save figure
    #plt.show()

    print('Plot counter init set ...')
    plot_probstar_reachset(CE, dir_mat=dir_mat3, dir_vec=dir_vec3, show_prob=True, \
                           label=('$v_{lead}[0]$','$v_{ego}[0]$'), show=False)
    plt.savefig(path+"/counterInitSet.png", bbox_inches='tight')  # save figure
    #plt.show()

    print('Done!')
    
def generate_approx_reachset_figs():
    'generate 4 pictures and save in DAC2024/pics/'

    net='controller_5_20'
    plant='linear'
    initSet_id=5
    pf=0.1
    numSteps=18
    numCores=4

    # load NNCS ACC system
    print('Loading the ACC system...')
    ncs, initSets, refInputs, unsafe_mat, unsafe_vec = load_acc_model(net,plant)
    
    

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.verifyMethod = 'Q2'
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
    
    print('Verifying the ACC system for {} timesteps using exact reachability...'.format(numSteps))
    RX, _, _, Ce, Co, _, _, _, _, _ = verifyBFS_DLNNCS(ncs, verifyPRM)
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


    path = "artifacts/DAC2024/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    RX1 = RX[18:19]
    print('Plot reachable set...')
    plot_probstar_reachset(RX1, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=True, \
                           label=('$v_{ego}$','$d_r - d_{safe}$'), show=False)
    plt.savefig(path+"/dr_dsafe_vs_vego_approx.png", bbox_inches='tight')  # save figure
    #plt.show()

    plot_probstar_reachset(RX1, dir_mat=dir_mat2, dir_vec=dir_vec2, show_prob=True, \
                           label=('$d_{r}$','$d_{safe}$'), show=False)
    plt.savefig(path+"/dr_vs_dsafe_approx.png", bbox_inches='tight')  # save figure
    #plt.show()
    

    print('Plot counter output set...')
    plot_probstar_reachset(CO, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=True, \
                           label=('$v_{ego}$','$d_r - d_{safe}$'), show=False)
    plt.savefig(path+"/counterOutputSet_approx.png", bbox_inches='tight')  # save figure
    #plt.show()

    print('Plot counter init set ...')
    plot_probstar_reachset(CE, dir_mat=dir_mat3, dir_vec=dir_vec3, show_prob=True, \
                           label=('$v_{lead}[0]$','$v_{ego}[0]$'), show=False)
    plt.savefig(path+"/counterInitSet_approx.png", bbox_inches='tight')  # save figure
    #plt.show()

    print('Done!')

    
def generate_numberReachSets_vs_pf():
    'generate number of reachable sets and verification time vesus pf'

    net='controller_5_20'
    plant='linear'
    initSet_id=5
    pf=[0.0, 0.1]
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
    verifyPRM.verifyMethod = 'Q2'
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]

    R = []
    VT = []
    for p in pf:
        verifyPRM.pf = copy.deepcopy(p)
        print('Verifying the ACC system for {} timesteps using approx reachability with pf = {}...'.format(numSteps, p))
        start_time = time.time()
        RX, _, _, _, _, _, _, _, _, _ = verifyBFS_DLNNCS(ncs, verifyPRM)
        end_time = time.time()
        VT.append(end_time - start_time)
        R.append(RX)

    t = range(0, 1, numSteps)
    
    path = "artifacts/DAC2024/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    

    print('Plot verification time vs. pf...')

    fig1 = plt.figure()
    xaxis = pf
    yaxis = VT
    plt.plot(xaxis, yaxis, color='blue', marker='o', label='VT')
    label=('$p_f$', '$VT(sec)$')
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(path+"/VT_vs_pf.png", bbox_inches='tight')  # save figure
    show()

    fig2 = plt.figure()
    
    print('Plot number of reachable sets via time steps...')

    nR = []
    for Ri in R:
        nRi = []
        for Rj in Ri:
            nRi.append(len(Rj))

        nR.append(nRi)

    markers = ['*', '+', 'x', 'o']
    colors = ['green', 'blue', 'red', 'black']
    for i in range (0, len(nR)):
        xaxis = np.arange(0,len(nR[i]))
        plt.plot(xaxis, nR[i], color=colors[i], marker=markers[i], label='$p_f = {}$'.format(pf[i]))

    label=('t', '$N_{rs}$')
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend()
    plt.savefig(path+"/numReachSet.png", bbox_inches='tight')  # save figure
    show()

    print('Done!')

def generate_exact_Q2_verification_results():
    'generate Q2 verification results'

    net='controller_5_20'
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
    verifyPRM.verifyMethod = 'Q2'
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
    
    print('Verifying the ACC system for {} timesteps using exact reachability...'.format(numSteps))
    _, _, _, _, _, Ql, Qt, _, _, _ = verifyBFS_DLNNCS(ncs, verifyPRM)

    t = range(0, 1, numSteps)
    
    path = "artifacts/DAC2024/pics"
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
    plt.savefig(path+"/Ql_exact.png", bbox_inches='tight')  # save figure
    #show()

    fig2 = plt.figure()
    
    print('Plot exact quantitative results...')
    yaxis1 = np.array(Qt)

    plt.plot(xaxis, yaxis1, color='blue', marker='o', label='Qt')
    label=('t', 'Qt')
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(path+"/Qt_exact.png", bbox_inches='tight')  # save figure
    show()
    
    print('Done!')

def generate_approx_Q2_verification_results():
    'generate approx Q2 verification results'

    net='controller_5_20'
    plant='linear'
    initSet_id=5
    pf=0.1
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
    verifyPRM.verifyMethod = 'Q2'
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
    
    print('Verifying the ACC system for {} timesteps using approx reachability. with pf = {}..'.format(numSteps, pf))
    _, _, _, _, _, Ql, Qt, Qt_lb, Qt_ub, _ = verifyBFS_DLNNCS(ncs, verifyPRM)

    t = range(0, 1, numSteps)
    
    path = "artifacts/DAC2024/pics"
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
    plt.savefig(path+"/Ql_approx.png", bbox_inches='tight')  # save figure
    #show()

    fig2 = plt.figure()
    
    print('Plot approx quantitative results...')
    yaxis1 = np.array(Qt)
    yaxis2 = np.array(Qt_lb)
    yaxis3 = np.array(Qt_ub)

    plt.plot(xaxis, yaxis1, color='blue', marker='o', label='Qt')
    plt.plot(xaxis, yaxis2, color='black', marker='x', label='Qt_lb')
    plt.plot(xaxis, yaxis3, color='red', marker='*', label='Qt_ub')
    label=('t', 'Qt, Qt_lb, Qt_ub')
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend()
    plt.savefig(path+"/Qt_Qtlb_Qtub_approx.png", bbox_inches='tight')  # save figure
    show()
    
    print('Done!')

def generate_VT_vs_nets():
    'generate verification time vesus pf and networks'

    nets=['controller_3_20', 'controller_5_20', 'controller_7_20']
    plant='linear'
    initSet_id=5
    pf = [0.0, 0.02, 0.04]
    numSteps=30
    numCores=4

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()


    n = len(nets)
    m = len(pf)

    VT = np.zeros((m, n))
    NO = np.zeros((m,n))
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
            verifyPRM.verifyMethod = 'Q2'
            verifyPRM.numCores = numCores
            verifyPRM.unsafeSpec = [unsafe_mat, unsafe_vec]
            
            print('Verifying the ACC system with {} for {} \
            timesteps using approx reachability with pf = {}...'.format(nets[j], numSteps, pf[i]))
            start_time = time.time()
            RX, _, _, _, _, _, _, _, _, p_ignored = verifyBFS_DLNNCS(ncs, verifyPRM)
            end_time = time.time()
            M = len(RX)
            NO[i,j] = len(RX[M-1])
            VT[i, j] = end_time - start_time
            strg1 = '{}'.format(pf[i])
            strg2 = '{}'.format(nets[j])
            data.append([strg1, strg2, VT[i,j], NO[i,j], p_ignored])
            
    print(tabulate(data, headers=["p_filter", "network", "verification time", "number of output sets", "p_ignored"]))   
    
    path = "artifacts/DAC2024/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/VT.tex", "w") as f:
         print(tabulate(data, headers=["p_filter", "network", "verification time", "number of output sets", "p_ignored"], tablefmt='latex'), file=f)
       
    
    print('Done!')



def generate_exact_reachset_figs_AEBS():

    # load NNCS AEBS system
    print('Loading the AEBS system...')
    
    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()

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



    X0 = initSets[0]    
    T = 1
    numCores = 1

    if numCores > 1:
        pool = multiprocessing.Pool(numCores)
    else:
        pool = None
    
    
    X = multiStepsReach_AEBS(controller, transformer, norm_mat, scale_mat, plant, X0, T, pool)
    print('X ={}'.format(X))
    


def multiStepsReach_AEBS(controller, transformer, norm_mat, scale_mat, plant, X0, T, pool):
    'compute the reachable set of AEBS for multiple steps T'

    if T < 1:
        raise RuntimeError('Invalid number of steps')

    X = []
    X.append([X0])
    for i in range(0, T):
        if i==0:
            X1 = stepReach_AEBS_multipleInitSets(controller, transformer, norm_mat, scale_mat, plant, X[0], pool)
        else:
            X1 = stepReach_AEBS_multipleInitSets(controller, transformer, norm_mat, scale_mat, plant, X[i-1], pool)
        X.append(X1)

    return X

def stepReach_AEBS_multipleInitSets(controller, transformer, norm_mat, scale_mat, plant, X0, pool):
    'step reach of AEBS with multiple initial sets'

    n = len(X0)
    X1 = []
    for i in range(0, n):
        X1i = stepReach_AEBS(controller, transformer, norm_mat, scale_mat, plant, X0[i], pool)
        X1.append(X1i)

    return X1


def stepReach_AEBS(controller, transformer, norm_mat, scale_mat, plant, X0, pool):
    'step reachability of AEBS system'
    

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

    norm_X = X0.affineMap(norm_mat)
    print('Computing reachable set of RL controller ...\n')
    brake = reachExactBFS(controller, [norm_X], pool=pool)
    m = len(brake)

    print('Geting exact input sets to transformer ...\n')
    # get exact inputs to transformer
    speed_brake = [] # exact inputs to transformer
    for i in range(0, m):
        V = np.vstack((norm_X.V[2, :], brake[i].V))
        speed_brake_i = ProbStar(V, brake[i].C, brake[i].d, brake[i].mu, brake[i].Sig, brake[i].pred_lb, brake[i].pred_ub)
        speed_brake.append(speed_brake_i)


    print('Computing exact transformer output ... \n')
    # get exact transformer output
    tf_outs = []
    for i in range(0, m):
        tf_out = reachExactBFS(transformer, [speed_brake[i]], pool=pool)
        tf_outs.extend(tf_out)

    print('Getting control input set to the plant and scale it ...\n')
    # get control input to the plant and scale it using scale matrix
    n = len(tf_outs)
    controls = []
    for i in range(0, n):
        V = np.vstack((norm_X.V[2, :], tf_outs[i].V))
        control = ProbStar(V, tf_outs[i].C, tf_outs[i].d, tf_outs[i].mu, tf_outs[i].Sig, tf_outs[i].pred_lb, tf_outs[i].pred_ub)
        #control.__str__()
        control = control.affineMap(scale_mat)
        controls.append(control)

    print('Compute the next step reachable set for the plant ...\n')
    # compute the next step reachable set for the plant
    X1 = []
    for i in range(0, n):
        X1i = plant.stepReach(X0, controls[i], subSetPredicate=True)
        X1.append(X1i)

    return X1
    
if __name__ == "__main__":

    # verify ACC model
    
    #generate_exact_reachset_figs()
    #generate_approx_reachset_figs()
    #generate_exact_Q2_verification_results()
    #generate_approx_Q2_verification_results()
    #generate_numberReachSets_vs_pf()
    #generate_VT_vs_nets()

    # verify AEBS model
    generate_exact_reachset_figs_AEBS()
    
