"""
Verify ACC system
Author: Anomynous
Date: 05/2024
"""


from StarV.verifier.verifier import quantiVerifyBFS
from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load import load_acc_model
from StarV.util.plot import plot_probstar_reachset, plot_probstar_signal, plot_probstar_signals, plot_SAT_trace
import time
from StarV.util.plot import plot_probstar
from matplotlib import pyplot as plt
from matplotlib.pyplot import step, show
from matplotlib.ticker import MaxNLocator
from StarV.set.star import Star
from tabulate import tabulate
from StarV.nncs.nncs import VerifyPRM_NNCS, verifyBFS_DLNNCS, ReachPRM_NNCS, reachBFS_DLNNCS, reachDFS_DLNNCS, verify_temporal_specs_DLNNCS, verify_temporal_specs_DLNNCS_for_full_analysis
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_
from StarV.spec.dProbStarTL import DynamicFormula
import os
import copy



def test_verification_ACC():
    
    net='controller_5_20'
    plant='linear'
    spec_ids=[6]
    initSet_id=5  
    numSteps=29
    T = numSteps
    numCores=1
    pf = 0.0
    t=5

    # load NNCS ACC system
    print('Loading the ACC system...')
    ncs, specs, initSet, refInputs = load_acc_model(net, plant, spec_ids, initSet_id, T, t)

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSet)
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.temporalSpecs = copy.deepcopy(specs)

    traces, p_SAT_MAX, p_SAT_MIN, reachTime, checkingTime, verifyTime = verify_temporal_specs_DLNNCS(ncs, verifyPRM)

    print('Number of traces = {}'.format(len(traces)))
    print('p_SAT_MAX = {}'.format(p_SAT_MAX))
    print('p_SAT_MIN = {}'.format(p_SAT_MIN))
    print('reachTime = {}'.format(reachTime))
    print('checkingTime = {}'.format(checkingTime))
    print('verifyTime = {}'.format(verifyTime))

def test_verification_ACC_full_analysis():
    
    net='controller_5_20'
    plant='linear'
    spec_ids=[6]
    initSet_id=5  
    numSteps=30
    T = numSteps
    numCores=1
    pf = 0.0
    t=5

    # load NNCS ACC system
    print('Loading the ACC system...')
    ncs, specs, initSet, refInputs = load_acc_model(net, plant, spec_ids, initSet_id, T, t)

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSet)
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = numSteps
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.temporalSpecs = copy.deepcopy(specs)

    res = verify_temporal_specs_DLNNCS_for_full_analysis(ncs, verifyPRM)

    traces = res[0] # reachable set trace
    p_SAT_MAX = res[1]  # upper bound of satisfaction probability
    p_SAT_MIN = res[2]  # lower bound of satisfaction probability
    reachTime = res[3]  # reachability time
    checkingTime = res[4] # checking tim
    verifyTime = res[5]   # total verification time
    p_IG = res[6]         # probability of ignored CDNF in verification
    p_ig0 = res[7]        # probability of ignored traces in analysis
    CDNF_SAT = res[8]     # 
    CDNF_IG = res[9]
    conserv = res[10]
    constit = res[11]
    
    print('Number of traces = {}'.format(len(traces)))
    print('p_SAT_MAX = {}'.format(p_SAT_MAX))
    print('p_SAT_MIN = {}'.format(p_SAT_MIN))
    print('reachTime = {}'.format(reachTime))
    print('checkingTime = {}'.format(checkingTime))
    print('verifyTime = {}'.format(verifyTime))
    print('p_IG = {}'.format(p_IG))
    print('p_ig0 = {}'.format(p_ig0))
    print('CDNF satisfying the properties = {}'.format(CDNF_SAT))
    print('CDNF ignored in verification = {}'.format(CDNF_IG))
    print('Conservativeness (in percentage) = {}'.format(conserv))
    print('Consitution of ignored traces in estimating pmax = {}'.format(constit))

    
    
def verify_temporal_specs_ACC():
    'verify temporal properties of Le-ACC, tables are generated and stored in artifacts/2024POPL/table'


    net='controller_5_20'
    plant='linear'
    #spec_ids = [6] # for testing
    spec_ids=[0,1,2,3,4,6]
    initSet_id=5 
    T = [10, 20, 30] # numSteps
    #T = [30] # for testing
    numCores=4
    t=5
    
    
    # verification with pf = 0.0
    verification_data_pf_0_0 = []
    for T1 in T:
        
        # load NNCS ACC system
        print('Loading the ACC system...')
        ncs, specs, initSet, refInputs = load_acc_model(net, plant, spec_ids, initSet_id, T1, t)

         # verify parameters
        verifyPRM = VerifyPRM_NNCS()
        verifyPRM.initSet = copy.deepcopy(initSet)
        verifyPRM.refInputs = copy.deepcopy(refInputs)
        verifyPRM.numSteps = T1
        verifyPRM.pf = 0.0
        verifyPRM.numCores = numCores
        verifyPRM.temporalSpecs = copy.deepcopy(specs)

        traces, p_max, p_min, reachTime, checkingTime, verifyTime = verify_temporal_specs_DLNNCS(ncs, verifyPRM)
        #print('pmax = {}'.format(p_max))
        #print('pmin = {}'.format(p_min))

        for spec_id in spec_ids:
            if spec_id == 6:
                spec_id1 = 5
            else:
                spec_id1 = spec_id
            verification_data_pf_0_0.append(['p_{}'.format(spec_id1), T1 , \
                                             p_max[spec_id1], p_min[spec_id1], \
                                             reachTime, checkingTime[spec_id1], \
                                             verifyTime[spec_id1]])


     # vefification with pf = 0.1
    verification_data_pf_0_1 = []
    for T1 in T:
        
        # load NNCS ACC system
        print('Loading the ACC system...')
        ncs, specs, initSet, refInputs = load_acc_model(net, plant, spec_ids, initSet_id, T1, t)

         # verify parameters
        verifyPRM = VerifyPRM_NNCS()
        verifyPRM.initSet = copy.deepcopy(initSet)
        verifyPRM.refInputs = copy.deepcopy(refInputs)
        verifyPRM.numSteps = T1
        verifyPRM.pf = 0.1
        verifyPRM.numCores = numCores
        verifyPRM.temporalSpecs = copy.deepcopy(specs)
        verifyPRM.computeProbMethod = 'approx'

        traces, p_max, p_min, reachTime, checkingTime, verifyTime = verify_temporal_specs_DLNNCS(ncs, verifyPRM)

        for spec_id in spec_ids:
            
            if spec_id == 6:
                spec_id1 = 5
            else:
                spec_id1 = spec_id
            
            verification_data_pf_0_1.append(['p_{}'.format(spec_id1), T1 , \
                                             p_max[spec_id1], p_min[spec_id1], \
                                             reachTime, checkingTime[spec_id1], \
                                             verifyTime[spec_id1]])


    print('=======================VERIFICATION RESULTS WITHOUT FILTERING pf = 0 ==========================')
    print(tabulate(verification_data_pf_0_0, headers=["Spec.", "T", "p_max", "p_min", "reachTime", "checkTime", "verifyTime"]))   
    
    path = "artifacts/2024POPL/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/verification_tab_pf_0_0.tex", "w") as f:
         print(tabulate(verification_data_pf_0_0, headers=["Spec.", "T", "p_max", "p_min", "reachTime", "checkTime", "verifyTime"], tablefmt='latex'), file=f)


    print('=======================VERIFICATION RESULTS WITH FILTERING pf = 0.1 ==========================')
    print(tabulate(verification_data_pf_0_1, headers=["Spec.", "T", "p_max", "p_min", "reachTime", "checkTime", "verifyTime"]))   
    
    path = "artifacts/CAV2024/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/verification_tab_pf_0_1.tex", "w") as f:
         print(tabulate(verification_data_pf_0_1, headers=["Spec.", "T", "p_max", "p_min",  "reachTime", "checkTime", "verifyTime"], tablefmt='latex'), file=f)


def analyze_timing_performance():
    """
    Generate figures to analyze timing performance of the verification, 
    figures are stored in artifacts/2024POPL/pics

    """


    # we use phi_4' to analyze the timing performance

    # # figure 1: reachTime, checkingTime, verifyTime vs. number of time step

    net='controller_5_20'
    plant='linear'
    spec_ids=[6]
    initSet_id=5  
    T = [10, 15, 20, 25, 30] # numSteps
    #T = [4, 6, 8]
    numCores=4
    t=5

    reachTime = []
    checkingTime = []
    verifyTime = []
    for T1 in T:
        
        # load NNCS ACC system
        print('Loading the ACC system. T = {}..'.format(T1))
        ncs, specs, initSet, refInputs = load_acc_model(net, plant, spec_ids, initSet_id, T1, t)

        # verify parameters
        verifyPRM = VerifyPRM_NNCS()
        verifyPRM.initSet = copy.deepcopy(initSet)
        verifyPRM.refInputs = copy.deepcopy(refInputs)
        verifyPRM.numSteps = T1
        verifyPRM.pf = 0.0
        verifyPRM.numCores = numCores
        verifyPRM.temporalSpecs = copy.deepcopy(specs)

        _, _, _, reachTime1, checkingTime1, verifyTime1 = verify_temporal_specs_DLNNCS(ncs, verifyPRM)
        reachTime.append(reachTime1)
        checkingTime.append(checkingTime1)
        verifyTime.append(verifyTime1)

    path = "artifacts/2024POPL/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure()
    ax = plt.figure().gca()
    plt.plot(T, reachTime, 'bo-', label='$t_r$')
    plt.plot(T, checkingTime, 'rx-', label='$t_c$')
    plt.plot(T, verifyTime, '>-', label='$t_v$')
    plt.xlabel('$T$ (number of steps)', fontsize=13)
    plt.ylabel('Analysis Time (s)', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(path+"/rt_ct_vt_vs_T.png", bbox_inches='tight')  # save figure
    #plt.show()


    # # # figure 2: verification time vs. pf

    net='controller_5_20'
    plant='linear'
    spec_ids=[6]
    initSet_id=5  # corresponding to X0_0 and X0_1 in the paper
    pf = [0.0, 0.05, 0.1, 0.15, 0.2]
    #pf = [0.0, 0.05, 0.1]
    T = 30
    numCores=4
    t=5

    reachTime = []
    checkingTime = []
    verifyTime = []
    for pf1 in pf:
        
        # load NNCS ACC system
        print('Loading the ACC system. pf = {}..'.format(pf1))
        ncs, specs, initSet, refInputs = load_acc_model(net, plant, spec_ids, initSet_id, T, t)

        # verify parameters
        verifyPRM = VerifyPRM_NNCS()
        verifyPRM.initSet = copy.deepcopy(initSet)
        verifyPRM.refInputs = copy.deepcopy(refInputs)
        verifyPRM.numSteps = T
        verifyPRM.pf = pf1
        verifyPRM.numCores = numCores
        verifyPRM.temporalSpecs = copy.deepcopy(specs)

        _, _, _, reachTime1, checkingTime1, verifyTime1 = verify_temporal_specs_DLNNCS(ncs, verifyPRM)
        reachTime.append(reachTime1)
        checkingTime.append(checkingTime1)
        verifyTime.append(verifyTime1)

    path = "artifacts/2024POPL/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure()
    plt.plot(pf, reachTime, 'bo-', label='$t_r$')
    plt.plot(pf, checkingTime, 'rx-', label='$t_c$')
    plt.plot(pf, verifyTime, '>-', label='$t_v$')
    plt.xlabel('$p_f$', fontsize=13)
    plt.ylabel('Analysis Time (s)', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    
    plt.savefig(path+"/rt_ct_vt_vs_pf.png", bbox_inches='tight')  # save figure
    #plt.show()


    # # figure 3: analysis time vs. number of cores
    net='controller_5_20'
    plant='linear'
    spec_ids=[6]
    initSet_id=5  # corresponding to X0_0 
    pf = 0.0
    T = 30
    numCores=[1, 2, 4, 6, 8]
    #numCores=[1,4]
    t=5

    reachTime = []
    checkingTime = []
    verifyTime = []
    for numCore in numCores:
        
        # load NNCS ACC system
        print('Loading the ACC system...')
        ncs, specs, initSet, refInputs = load_acc_model(net, plant, spec_ids, initSet_id, T, t)

        # verify parameters
        verifyPRM = VerifyPRM_NNCS()
        verifyPRM.initSet = copy.deepcopy(initSet)
        verifyPRM.refInputs = copy.deepcopy(refInputs)
        verifyPRM.numSteps = T
        verifyPRM.pf = pf
        verifyPRM.numCores = numCore
        verifyPRM.temporalSpecs = copy.deepcopy(specs)

        _, _, _, reachTime1, checkingTime1, verifyTime1 = verify_temporal_specs_DLNNCS(ncs, verifyPRM)
        reachTime.append(reachTime1)
        checkingTime.append(checkingTime1)
        verifyTime.append(verifyTime1)

    path = "artifacts/2024POPL/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure()
    plt.plot(numCores, reachTime, 'bo-', label='$t_r$')
    plt.plot(numCores, checkingTime, 'rx-', label='$t_c$')
    plt.plot(numCores, verifyTime, '>-', label='$t_v$')
    plt.xlabel('$N_{cores}$', fontsize=13)
    plt.ylabel('Analysis Time (s)', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    
    plt.savefig(path+"/rt_ct_vt_vs_numCores.png", bbox_inches='tight')  # save figure
    #plt.show()


def analyze_conservativeness():
    'analyze conservativeness of verification results, figures are generated and stored in artifacts/2024POPL/pics'

    
    net='controller_5_20'
    plant='linear'
    spec_ids=[1]
    initSet_id=5  
    T = [10, 15, 20, 25, 30]
    #T = [6, 8]
    numCores=4
    pf = [0.0, 0.05, 0.1]
    t=5

    m = len(pf)
    n = len(T)

    conserv = np.zeros((m, n), dtype=float)
    constit = np.zeros((m, n), dtype=float)

    for i in range(0,m):
        pf1 = pf[i]
        for j in range(0,n):
            T1 = T[j]
            
            # load NNCS ACC system
            print('Loading the ACC system...')
            ncs, specs, initSet, refInputs = load_acc_model(net, plant, spec_ids, initSet_id, T1, t)

            # verify parameters
            verifyPRM = VerifyPRM_NNCS()
            verifyPRM.initSet = copy.deepcopy(initSet)
            verifyPRM.refInputs = copy.deepcopy(refInputs)
            verifyPRM.numSteps = T1
            verifyPRM.pf = pf1
            verifyPRM.numCores = numCores
            verifyPRM.temporalSpecs = copy.deepcopy(specs)

            res = verify_temporal_specs_DLNNCS_for_full_analysis(ncs, verifyPRM)
            conserv1 = res[10]
            constit1 = res[11]

            conserv[i,j] = conserv1[0]
            constit[i,j] = constit1[0]

    # figures: conservativeness and constitution (of phi1') vs. time steps      
    path = "artifacts/2024POPL/pics"
    if not os.path.exists(path):
        os.makedirs(path)

   
    plt.figure()
    ax = plt.figure().gca()
    plt.plot(T, conserv[0,:], 'bo-', label='$p_f = 0.0$')
    plt.plot(T, conserv[1,:], 'rx-', label='$p_f = 0.05$')
    plt.plot(T, conserv[2,:], '>-', label='$p_f = 0.1$')
    plt.xlabel('$T$ (number of steps)', fontsize=13)
    plt.ylabel('Conservativeness ($100*(p_{max} - p_{min})/p_{max}$)', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(path+"/conservertiveness_vs_numSteps.png", bbox_inches='tight')  # save figure
    #plt.show()

    plt.figure()
    ax = plt.figure().gca()
    plt.plot(T, constit[0,:], 'bo-', label='$p_f = 0.0$')
    plt.plot(T, constit[1,:], 'rx-', label='$p_f = 0.05$')
    plt.plot(T, constit[2,:], '>-', label='$p_f = 0.1$')
    plt.xlabel('$T$ (number of steps)', fontsize=13)
    plt.ylabel('Constitution ($100*p_{ig}/p_{max}$)', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(path+"/constitution_vs_numSteps.png", bbox_inches='tight')  # save figure
    #plt.show()c


def analyze_verification_complexity():
    'analyze verification complexity, figures are generated and stored in artifacts/2024POPL/pics'

    # figure 1: number of traces vs networks vs time steps
    # figure 2: number of computable CDNF and ignored CDNF

    net=['controller_3_20','controller_5_20','controller_7_20']

    plant='linear'
    spec_ids=[0]
    initSet_id=5  
    T = [10, 15, 20, 25, 30]
    #T = [6, 8]
    numCores=4
    pf = 0.0
    t=5

    m = len(net)
    n = len(T)

    n_traces = np.zeros((m, n), dtype=int)
    n_cdnfs = np.zeros((m, n), dtype=int)
    n_ig_cdnfs = np.zeros((m,n), dtype=int)
    n_sat_cdnfs = np.zeros((m,n), dtype=int)

    for i in range(0,m):
        net1 = net[i]
        for j in range(0,n):
            T1 = T[j]
            
            # load NNCS ACC system
            print('Loading the ACC system...')
            ncs, specs, initSet, refInputs = load_acc_model(net1, plant, spec_ids, initSet_id, T1, t)

            # verify parameters
            verifyPRM = VerifyPRM_NNCS()
            verifyPRM.initSet = copy.deepcopy(initSet)
            verifyPRM.refInputs = copy.deepcopy(refInputs)
            verifyPRM.numSteps = T1
            verifyPRM.pf = pf
            verifyPRM.numCores = numCores
            verifyPRM.temporalSpecs = copy.deepcopy(specs)

            res = verify_temporal_specs_DLNNCS_for_full_analysis(ncs, verifyPRM)
            

            traces = res[0] # reachable set trace
            n_traces[i,j] = len(traces)
            CDNF_SAT = res[8]
            CDNF_SAT = CDNF_SAT[0]
            CDNF_SAT_short = [ele for ele in CDNF_SAT if ele != []]
            n_sat_cdnfs[i,j] = len(CDNF_SAT_short)
            CDNF_IG = res[9]
            CDNF_IG = CDNF_IG[0]
            CDNF_IG_short = [ele for ele in CDNF_IG if ele !=[]]
            n_ig_cdnfs[i,j] = len(CDNF_IG_short)
            n_cdnfs[i,j] = n_sat_cdnfs[i,j] + n_ig_cdnfs[i,j]


    print('n_traces = {}'.format(n_traces))
    print('n_cdnfs = {}'.format(n_cdnfs))
    print('n_sat_cdnfs = {}'.format(n_sat_cdnfs))
    print('n_ig_cdnfs = {}'.format(n_ig_cdnfs))
            
    # figures: conservativeness and constitution (of phi1') vs. time steps      
    path = "artifacts/2024POPL/pics"
    if not os.path.exists(path):
        os.makedirs(path)

   
    plt.figure()
    ax = plt.figure().gca()
    plt.plot(T, n_traces[0,:], 'bo-', label='$N_{3x20}$')
    plt.plot(T, n_traces[1,:], 'rx-', label='$N_{5x20}$')
    plt.plot(T, n_traces[2,:], '>-', label='$N_{7x20}$')
    plt.xlabel('$T$ (number of steps)', fontsize=13)
    plt.ylabel('Number of traces', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(path+"/ntraces_vs_numSteps.png", bbox_inches='tight')  # save figure
    #plt.show()


def analyze_verification_complexity_2():
    'produce a figure of n_CDNF_SAT and n_CDNF_UNSAT and store in artifacts/2024POPL/data'


    net='controller_5_20'

    plant='linear'
    spec_ids=[6]
    initSet_id=5  
    T = [10, 20, 30]
    numCores=4
    pf = 0.0
    t=5

    n_cdnfs = []
    n_ig_cdnfs = []
    n_sat_cdnfs = []
    n_traces = []

    for T1 in T:
            
        # load NNCS ACC system
        print('Loading the ACC system...')
        ncs, specs, initSet, refInputs = load_acc_model(net, plant, spec_ids, initSet_id, T1, t)

        # verify parameters
        verifyPRM = VerifyPRM_NNCS()
        verifyPRM.initSet = copy.deepcopy(initSet)
        verifyPRM.refInputs = copy.deepcopy(refInputs)
        verifyPRM.numSteps = T1
        verifyPRM.pf = pf
        verifyPRM.numCores = numCores
        verifyPRM.temporalSpecs = copy.deepcopy(specs)

        res = verify_temporal_specs_DLNNCS_for_full_analysis(ncs, verifyPRM)


        n_traces.append(len(res[0]))
        CDNF_SAT1 = res[8]
        CDNF_SAT = CDNF_SAT1[0]
        CDNF_SAT_short = [ele for ele in CDNF_SAT if ele != []]
        n_sat_cdnfs.append(len(CDNF_SAT_short))
        CDNF_IG1 = res[9]
        CDNF_IG = CDNF_IG1[0]
        CDNF_IG_short = [ele for ele in CDNF_IG if ele !=[]]
        n_ig_cdnfs.append(len(CDNF_IG_short))
        n_cdnfs.append(len(CDNF_SAT_short) + len(CDNF_IG_short))
        
                                                  

    print('n_traces = {}'.format(n_traces))
    print('n_cdnfs = {}'.format(n_cdnfs))
    print('n_sat_cdnfs = {}'.format(n_sat_cdnfs))
    print('n_ig_cdnfs = {}'.format(n_ig_cdnfs))
            
    # figures:     
    path = "artifacts/2024POPL/pics"
    if not os.path.exists(path):
        os.makedirs(path)


    T = ('$T = 10$', '$T = 20$', '$T = 30$')
    cdnf_counts = {'SAT': n_sat_cdnfs, 'IGNORED': n_ig_cdnfs}
    width=0.6

    fig, ax = plt.subplots()
    bottom = np.zeros(3)
    for cdnf, cdnf_count in cdnf_counts.items():
        p = ax.bar(T, cdnf_count, width, label=cdnf, bottom=bottom)
        bottom += cdnf_count
        ax.bar_label(p, label_type='center')
    ax.legend()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(path+"/ncdnfs_vs_T.png", bbox_inches='tight')  # save figure
    plt.show()

    
def analyze_verification_complexity_3():
    'pictures are generated and stored in artifacts/2024POPL/pics'

    net='controller_5_20'

    plant='linear'
    spec_ids=[6]
    initSet_id=5  
    T = 30
    numCores=4
    pf = 0.0
    t=5

    n_cdnfs = []
    n_ig_cdnfs = []
    n_sat_cdnfs = []
    n_traces = []

    
    # load NNCS ACC system
    print('Loading the ACC system...')
    ncs, specs, initSet, refInputs = load_acc_model(net, plant, spec_ids, initSet_id, T, t)

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSet)
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = T
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.temporalSpecs = copy.deepcopy(specs)

    res = verify_temporal_specs_DLNNCS_for_full_analysis(ncs, verifyPRM)

    CDNF_SAT1 = res[8]
    CDNF_SAT = CDNF_SAT1[0]
    CDNF_SAT_short = [ele for ele in CDNF_SAT if ele != []]
    CDNF_IG1 = res[9]
    CDNF_IG = CDNF_IG1[0]
    CDNF_IG_short = [ele for ele in CDNF_IG if ele !=[]]
    CDNFs = CDNF_SAT_short + CDNF_IG_short
    
    # figures: length of CDNFs      
    path = "artifacts/CAV2024/pics"
    if not os.path.exists(path):
        os.makedirs(path)

    L_CDNF = []
    indexs = []
    for i in range(0, len(CDNFs)):
        L_CDNF.append(CDNFs[i].length)
        indexs.append(i)

    fig, ax = plt.subplots()

    rect = ax.bar(indexs, L_CDNF)
    ax.bar_label(rect)
    ax.set_ylabel('Length of CDNF')
    ax.set_xlabel('CDNF index')
    ax.set_xticks(indexs)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(path+"/length_of_cdnf.png", bbox_inches='tight')  # save figure
    #plt.show()

   

def visualize_satisfied_traces():
    'visualize satisfied traces in verification, pictures are generated and stored in artifacts/CAV2024/pics'

    net='controller_5_20'

    plant='linear'
    spec_ids=[6]
    initSet_id=5  
    T = 30
    numCores=4
    pf = 0.0
    t=5

    n_cdnfs = []
    n_ig_cdnfs = []
    n_sat_cdnfs = []
    n_traces = []

    
    # load NNCS ACC system
    print('Loading the ACC system...')
    ncs, specs, initSet, refInputs = load_acc_model(net, plant, spec_ids, initSet_id, T, t)

    # verify parameters
    verifyPRM = VerifyPRM_NNCS()
    verifyPRM.initSet = copy.deepcopy(initSet)
    verifyPRM.refInputs = copy.deepcopy(refInputs)
    verifyPRM.numSteps = T
    verifyPRM.pf = pf
    verifyPRM.numCores = numCores
    verifyPRM.temporalSpecs = copy.deepcopy(specs)

    res = verify_temporal_specs_DLNNCS_for_full_analysis(ncs, verifyPRM)

    SAT_traces = res[12]
    SAT_traces = SAT_traces[0]

    # plot SAT_traces
    if len(SAT_traces) != 0:
        sat_trace = SAT_traces[0]
        #print('sat_trace = {}'.format(sat_trace))
        
        # plot reachable set  (d_actual - d_safe) vs. (v_ego)
        dir_mat1 = np.array([[0., 0., 0., 0., 1., 0., 0.],
                            [1., 0., 0., -1., -1.4, 0., 0.]])
        dir_vec1 = np.array([0., -10.])

        probstar_sigs = sat_trace[1].toProbStarSignals(sat_trace[0])

        path = "artifacts/2024POPL/pics"
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(0, len(probstar_sigs)):
            sig = probstar_sigs[i]
            plt.figure()
            print('Plot SAT trace ...')
            plot_probstar_signal(sig[15:25], dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=True, \
                                   label=('$v_{ego}$','$D_r - D_{safe}$'), show=False)
            plt.savefig(path+"/sat_trace_{}.png".format(i), bbox_inches='tight')  # save figure
            plt.show()

        
    
def verify_temporal_specs_ACC_full():
    'verify temporal properties of Le-ACC for all 3  networks, table is generated and stored in CAV2024/data'

    net=['controller_3_20', 'controller_5_20', 'controller_7_20']
    plant='linear'
    #spec_ids = [0] # for testing
    spec_ids=[0,1,2,3,4,6]
    initSet_id=5  # corresponding to X0_0 and X0_1 in the paper
    T = [10, 20, 30] # numSteps
    #T = [10]
    pf = [0.0, 0.1]
    #pf = [0.05]
    numCores=4
    t=5

    verification_data_full = []
    for net1 in net:
        for T1 in T:
            for pf1 in pf:
                # load NNCS ACC system
                print('Loading the ACC system...')
                ncs, specs, initSet, refInputs = load_acc_model(net1, plant, spec_ids, initSet_id, T1, t)

                 # verify parameters
                verifyPRM = VerifyPRM_NNCS()
                verifyPRM.initSet = copy.deepcopy(initSet)
                verifyPRM.refInputs = copy.deepcopy(refInputs)
                verifyPRM.numSteps = T1
                verifyPRM.pf = pf1
                verifyPRM.numCores = numCores
                verifyPRM.temporalSpecs = copy.deepcopy(specs)

                _, p_max, p_min, reachTime, checkingTime, verifyTime = verify_temporal_specs_DLNNCS(ncs, verifyPRM)

                for spec_id in spec_ids:
                    if spec_id == 6:
                       
                        spec_id1 = 5
                    else:
                        spec_id1 = spec_id

                    verification_data_full.append([net1, 'p_{}'.format(spec_id1), T1 , pf1, p_max[spec_id1], p_min[spec_id1],\
                                                   reachTime, checkingTime[spec_id1], verifyTime[spec_id1]])

        
    print('=======================FULL VERIFICATION RESULTS ==========================')
    print(tabulate(verification_data_full, headers=["Net.", "Spec.", "T", "pf" , "p_max", "p_min", "reachTime", "checkTime", "verifyTime"]))   
    
    path = "artifacts/2024POPL/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/verification_tab_full.tex", "w") as f:
         print(tabulate(verification_data_full, headers=["Net.", "Spec.", "T", "pf", "p_max", "p_min", "reachTime", "checkTime", "verifyTime"], tablefmt='latex'), file=f)




    
if __name__ == "__main__":

    
    #verify_temporal_specs_ACC()
    #analyze_timing_performance()
    #analyze_conservativeness()
    #analyze_verification_complexity()
    #analyze_verification_complexity_2()
    visualize_satisfied_traces()    
    #verify_temporal_specs_ACC_full()
    #test_verification_ACC()
    #test_verification_ACC_full_analysis()

