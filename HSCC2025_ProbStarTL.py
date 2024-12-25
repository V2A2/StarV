"""
HSCC2025: ProbStar Temporal Logic for Verifying Complex Behaviors of
          Learning-enabled Systems
Author: Anomynous
Date: 12/25/2024
"""

import os
import copy
import pickle
import scipy
import time
import numpy as np

from tabulate import tabulate
from matplotlib import pyplot as plt
from matplotlib.pyplot import step, show
from matplotlib.ticker import MaxNLocator

from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.verifier.verifier import quantiVerifyBFS
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_
from StarV.spec.dProbStarTL import DynamicFormula
from StarV.nncs.nncs import VerifyPRM_NNCS, verifyBFS_DLNNCS, ReachPRM_NNCS, reachBFS_DLNNCS, reachDFS_DLNNCS, verify_temporal_specs_DLNNCS, verify_temporal_specs_DLNNCS_for_full_analysis, reachBFS_AEBS, AEBS_NNCS
from StarV.util.load import load_acc_model, load_acc_trapezius_model, load_AEBS_model, load_AEBS_temporal_specs
from StarV.util.plot import plot_probstar_reachset, plot_probstar_signal, plot_probstar_signals, plot_SAT_trace
from StarV.util.plot import plot_probstar, plot_probstar_reachset_with_unsafeSpec


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
    'verify temporal properties of Le-ACC, tables are generated and stored in artifacts/HSCC2025_ProbStarTL/table'


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
    
    path = "artifacts/HSCC2025_ProbStarTL/ACC/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/verification_tab_pf_0_0.tex", "w") as f:
         print(tabulate(verification_data_pf_0_0, headers=["Spec.", "T", "p_max", "p_min", "reachTime", "checkTime", "verifyTime"], tablefmt='latex'), file=f)


    print('=======================VERIFICATION RESULTS WITH FILTERING pf = 0.1 ==========================')
    print(tabulate(verification_data_pf_0_1, headers=["Spec.", "T", "p_max", "p_min", "reachTime", "checkTime", "verifyTime"]))   
    
    path = "artifacts/HSCC2025_ProbStarTL/ACC/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/verification_tab_pf_0_1.tex", "w") as f:
         print(tabulate(verification_data_pf_0_1, headers=["Spec.", "T", "p_max", "p_min",  "reachTime", "checkTime", "verifyTime"], tablefmt='latex'), file=f)


def analyze_timing_performance():
    """
    Generate figures to analyze timing performance of the verification, 
    figures are stored in artifacts/HSCC2025_ProbStarTL/pics

    """


    # we use phi_4' to analyze the timing performance

    # # # figure 3a: reachTime, checkingTime, verifyTime vs. number of time step

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

    path = "artifacts/HSCC2025_ProbStarTL/ACC/pics"
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
    plt.savefig(path+"/Figure_3a__rt_ct_vt_vs_T.png", bbox_inches='tight')  # save figure
    #plt.show()


    # # # figure 3b: verification time vs. pf

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

    path = "artifacts/HSCC2025_ProbStarTL/ACC/pics"
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
    
    plt.savefig(path+"/Figure_3b__rt_ct_vt_vs_pf.png", bbox_inches='tight')  # save figure
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

    path = "artifacts/HSCC2025_ProbStarTL/ACC/pics"
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
    'analyze conservativeness of verification results, figures are generated and stored in artifacts/HSCC2025_ProbStarTL/pics'

    
    net='controller_5_20'
    plant='linear'
    spec_ids=[0]
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
    path = "artifacts/HSCC2025_ProbStarTL/ACC/pics"
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
    plt.savefig(path+"/Figure_2a__conservertiveness_vs_numSteps.png", bbox_inches='tight')  # save figure
    plt.show()

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
    plt.savefig(path+"/Figure_2b__constitution_vs_numSteps.png", bbox_inches='tight')  # save figure
    plt.show()

def analyze_verification_complexity():
    'analyze verification complexity, figures are generated and stored in artifacts/HSCC2025_ProbStarTL/pics'

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
    
    # figures: conservativeness and constitution (of phi1') vs. time steps      
    path = "artifacts/HSCC2025_ProbStarTL/ACC/pics"
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
    plt.savefig(path+"/Figure_4a__ntraces_vs_numSteps.png", bbox_inches='tight')  # save figure
    #plt.show()


def analyze_verification_complexity_2():
    'produce a figure of n_CDNF_SAT and n_CDNF_UNSAT and store in artifacts/HSCC2025_ProbStarTL/data'


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
    path = "artifacts/HSCC2025_ProbStarTL/ACC/pics"
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
    plt.savefig(path+"/Figure_4b__ncdnfs_vs_T.png", bbox_inches='tight')  # save figure
    plt.show()

    
def analyze_verification_complexity_3():
    'pictures are generated and stored in artifacts/HSCC2025_ProbStarTL/pics'

    net='controller_5_20'

    plant='linear'
    spec_ids=[6]
    initSet_id=5  
    T = 20
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
    path = "artifacts/HSCC2025_ProbStarTL/ACC/pics"
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
    plt.savefig(path+"/Figure_5a__length_of_cdnf.png", bbox_inches='tight')  # save figure
    #plt.show()

def visualize_satisfied_traces():
    'visualize satisfied traces in verification, pictures are generated and stored in artifacts/HSCC2025_ProbStarTL/pics'

    net='controller_5_20'

    plant='linear'
    spec_ids=[6]
    initSet_id=5  
    T = 20
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

        path = "artifacts/HSCC2025_ProbStarTL/ACC/pics"
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(0, len(probstar_sigs)):
            sig = probstar_sigs[i]
            plt.figure()
            print('Plot SAT trace ...')
            plot_probstar_signal(sig, dir_mat=dir_mat1, dir_vec=dir_vec1, show_prob=True, \
                                   label=('$v_{ego}$','$D_r - D_{safe}$'), show=False)
            plt.savefig(path+"/Figure_5b__sat_trace_{}.png".format(i), bbox_inches='tight')  # save figure
            plt.show()

        
    
def verify_temporal_specs_ACC_full():
    'verify temporal properties of Le-ACC for all 3  networks, table is generated and stored in HSCC2025_ProbStarTL/data'

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
    
    path = "artifacts/HSCC2025_ProbStarTL/ACC/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/verification_tab_full.tex", "w") as f:
         print(tabulate(verification_data_full, headers=["Net.", "Spec.", "T", "pf", "p_max", "p_min", "reachTime", "checkTime", "verifyTime"], tablefmt='latex'), file=f)



def generate_exact_reachset_figs_AEBS():

    # load initial conditions of NNCS AEBS system
    print('Loading initial conditions of AEBS system...')
    
    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
    AEBS = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)
    
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



        path = "artifacts/HSCC2025_ProbStarTL/AEBS/pics"
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


def verify_AEBS():

    # load initial conditions of NNCS AEBS system
    print('Loading initial conditions of AEBS system...')
    
    controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
    AEBS = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)

    print('Loading temporal specifications of AEBS system...')
    specs = load_AEBS_temporal_specs()

    print('specs = {}'.format(specs))


    #initSet_ids = [0]   # for testing
    initSet_ids=[0, 1, 2, 3]  
    #T = [10]   # for testing
    T = [10, 20, 40, 50] # numSteps
    #pf = [0.0] # for testing
    pf = [0.0, 0.01]
    numCores=4
    
    verification_data_full = []
   
    for initSet_id in initSet_ids:
        for T1 in T:
            for pf1 in pf:
            
                # verify parameters
                verifyPRM = VerifyPRM_NNCS()
                verifyPRM.initSet = copy.deepcopy(initSets[initSet_id])
                verifyPRM.numSteps = T1
                verifyPRM.pf = pf1
                verifyPRM.numCores = numCores
                verifyPRM.temporalSpecs = copy.deepcopy(specs)

                
                traces, p_max, p_min, reachTime, checkingTime, verifyTime = verify_temporal_specs_DLNNCS(AEBS, verifyPRM)

                for spec_id in range(0, len(specs)):
                    
                    verification_data_full.append(['X0_{}'.format(initSet_id), 'p_{}'.format(spec_id), T1 , pf1, \
                                                   p_max[spec_id], p_min[spec_id], reachTime, checkingTime[spec_id], \
                                                   verifyTime[spec_id], len(traces)])

        
    print('=======================FULL VERIFICATION RESULTS ==========================')
    print(tabulate(verification_data_full, headers=["X0", "Spec.", "T", "pf" , "p_max", "p_min", "reachTime", "checkTime", "verifyTime", "N_traces"]))   
    
    path = "artifacts/HSCC2025_ProbStarTL/AEBS/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"/AEBS_verification_tab_full.tex", "w") as f:
         print(tabulate(verification_data_full, headers=["X0", "Spec.", "T", "pf", "p_max", "p_min", "reachTime", "checkTime", "verifyTime", "N_traces"], tablefmt='latex'), file=f)


def verify_temporal_specs_ACC_trapezius(net='controller_3_20'):
    'verify temporal properties of Le-ACC, tables are generated and stored in artifacts/HSCC2025_ProbStarTL/table'

    plant='linear'
    spec_ids = [8]
    T = [10, 20, 30, 50] # numSteps
    numCores = 1
    t = 3
    
    verification_acc_trapezius_data = []
    for T1 in T:
        
        # load NNCS ACC system
        print('Loading the ACC system...')
        ncs, specs, initSet, refInputs = load_acc_trapezius_model(net, plant, spec_ids, T1, t)
        p = initSet.estimateProbability()

         # verify parameters
        verifyPRM = VerifyPRM_NNCS()
        verifyPRM.initSet = copy.deepcopy(initSet)
        verifyPRM.refInputs = copy.deepcopy(refInputs)
        verifyPRM.numSteps = T1
        verifyPRM.pf = 0.0
        verifyPRM.numCores = numCores
        verifyPRM.temporalSpecs = copy.deepcopy(specs)

        traces, p_max, p_min, reachTime, checkingTime, verifyTime = verify_temporal_specs_DLNNCS(ncs, verifyPRM)
        
        for i, spec_id in enumerate(spec_ids):
            if spec_id == 6:
                spec_id1 = 5
            else:
                spec_id1 = spec_id

            pc_max = p - np.array(p_max)
            pc_min = p - np.array(p_min)
            verification_acc_trapezius_data.append(['p_{}'.format(spec_id), T1 , \
                                             pc_max[i], pc_min[i], \
                                             reachTime, checkingTime[i], \
                                             verifyTime[i]])
            
    print(tabulate(verification_acc_trapezius_data, headers=["Spec.", "T", "pc_max", "pc_min", "reachTime", "checkTime", "verifyTime"]))   
    
    path = "artifacts/HSCC2025_ProbStarTL/ACC/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+f"/verification_acc_trapezius_{net}_tab.tex", "w") as f:
         print(tabulate(verification_acc_trapezius_data, headers=["Spec.", "T", "p_max", "p_min", "reachTime", "checkTime", "verifyTime"], tablefmt='latex'), file=f)

    path = "artifacts/HSCC2025_ProbStarTL/ACC/results"
    if not os.path.exists(path):
        os.makedirs(path)
    save_file = path + f"/verification_acc_trapezius_{net}_results.pkl"
    pickle.dump(verification_acc_trapezius_data, open(save_file, "wb"))

def generate_temporal_specs_ACC_trapezius_table():
    T = [10, 20, 30, 50]
    Nets = ['controller_3_20', 'controller_5_20']

    path = "artifacts/HSCC2025_ProbStarTL/ACC/results"
    NS_path = "StarV/util/data/nets/ACC/NeuroSymbolic_results/"

    header = ['CtrlNet', 'Method']
    for t in T:
        header.extend([f'[T = {t}] range', f'[T = {t}] vt (sec)'])
    header

    result_table = []
    for net in Nets:
        NeuroSymb_interval = []
        NeuroSymb_vt = []
        
        if net == 'controller_3_20':
            netname = 'N_3_20'
        else:
            netname = 'N_5_20'
            
        load_file = path + f"/verification_acc_trapezius_{net}_results.pkl"
        with open(load_file, 'rb') as f:
            data = pickle.load(f)
            
        for t in T:
            mat_file = NS_path + f"NeuroSymbolic_{net}_phi3_linear_exact_t{t}_linprog_results.mat"
            file = scipy.io.loadmat(mat_file)
            NeuroSymb_interval.append(file['interval'])
            NeuroSymb_vt.append(file['Computation_time'])
        
        table_row = []
        table_row.extend([netname, 'ProbStar'])
        for j in range(len(T)):
            table_row.extend([f'[{data[j][2]:2.4f}, {data[j][3]:2.4f}]', f'{data[j][6]:0.4f}'])
        result_table.append(table_row)

        table_row = []
        table_row.extend([netname, 'NeuroSymbolic'])
        for j in range(len(T)):
            table_row.extend([f'[{NeuroSymb_interval[j][0, 0]:0.4f}, {NeuroSymb_interval[j][0, 1]:0.4f}]', f'{NeuroSymb_vt[j].item():0.4f}'])
            
        result_table.append(table_row)

    print(tabulate(result_table, headers=header))

    path = "artifacts/HSCC2025_ProbStarTL/ACC/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+f"/verification_acc_trapezius_full_tab.tex", "w") as f:
        print(tabulate(result_table, headers=header, tablefmt='latex'), file=f)


if __name__ == "__main__":
    
    #test_verification_ACC()
    #test_verification_ACC_full_analysis()

    # Table 2: Verification results for Le-ACC with the network controller 𝑁5×20 (i.e., 5 layers, 20 neurons per layer)
    verify_temporal_specs_ACC()                     # Table 2
    #verify_temporal_specs_ACC_full() # Le-ACC with the network controllers: 𝑁3×20, 𝑁5×20, 𝑁7×20.
    
    # Figure 2: Conservativeness analysis of 𝜑1.
    analyze_conservativeness()                      # Figure 2

    # Figure 3: Verification timing performance of 𝜑′4. 
    analyze_timing_performance()                    # Figure 3

    # Figure 4: Verification complexity depends on 
    #   1) the number of traces (which varies for different networks and different initial conditions), 
    #   2) the number of CDNFs, and 
    #   3) the lengths of CDNFs.
    analyze_verification_complexity()               # Figure 4a
    analyze_verification_complexity_2()             # Figure 4b
    

    # Figure 5: Length of CDNFs for 𝜑′ 4 verification with 𝑇 = 20 and 
    # the visualization of a trace satisfying the specification.
    analyze_verification_complexity_3()             # Figure 5a
    visualize_satisfied_traces()                    # Figure 5b

    # Table 3:
    verify_temporal_specs_ACC_trapezius(net='controller_3_20')
    verify_temporal_specs_ACC_trapezius(net='controller_5_20')
    generate_temporal_specs_ACC_trapezius_table()   # Table 3

    # Table 4: Quantitative verification results of AEBS system
    # against property 𝜑 = ⋄[0,𝑇 ] (𝑑𝑘 ≤ 𝐿 ∧ 𝑣𝑘 ≥ 0.2) 
    verify_AEBS()                                   # Table 4

    # Figure 6: 50-step reachable sets (𝑑𝑘 vs. 𝑣𝑘 ) of AEBS system (in green) and 
    # the unsafe region (in red) for different initial conditions (scenarios) 𝑑0 × 𝑣0. 𝑑𝑘
    generate_exact_reachset_figs_AEBS()             # Figure 6