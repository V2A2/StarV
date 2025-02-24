"""
CAV2025: StarV: A Qualitative and Quantitative
                Verification Tool for Learning-enabled Systems

Author: Anomynous
Date: 02/20/2025
"""

import time
import scipy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from matplotlib.ticker import StrMethodFormatter
from StarV.util.load import *
from StarV.util.vnnlib import *
from StarV.util.print_util import print_util
from StarV.util.attack import brightening_attack
from StarV.util.load_piecewise import load_ACASXU_ReLU
from StarV.verifier.certifier import certifyRobustness
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.util.load import load_GRU_network, load_LSTM_network
from StarV.verifier.certifier import certifyRobustness_sequence
from StarV.verifier.verifier import quantiVerifyBFS, quantiVerifyMC
from StarV.verifier.krylov_func.simKrylov_with_projection import combine_mats
from StarV.util.plot import plot_probstar_signal,plot_probstar
from StarV.verifier.krylov_func.simKrylov_with_projection import simReachKrylov as sim3
from StarV.verifier.krylov_func.simKrylov_with_projection import random_two_dims_mapping
from StarV.verifier.krylov_func.LCS_verifier import quantiVerifier_LCS
from StarV.nncs.nncs import VerifyPRM_NNCS, verifyBFS_DLNNCS, ReachPRM_NNCS, reachBFS_DLNNCS, reachDFS_DLNNCS, verify_temporal_specs_DLNNCS, verify_temporal_specs_DLNNCS_for_full_analysis, reachBFS_AEBS, AEBS_NNCS
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_,_OR_

artifact = 'CAV2025_StarV_Tool'

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text)]

def verify_MNIST_LSTM_GRU(type='lstm', hidden=15):
    data_dir = f'StarV/util/data/nets/{artifact}'
    with open(f'{data_dir}/mnist_test.csv', 'r') as x:
        test_data = list(csv.reader(x, delimiter=","))

    test_data = np.array(test_data)
    # MNIST test dataset
    # data
    XTest = copy.deepcopy(test_data[:, 1:]).astype('float32').T / 255
    XTest = XTest.reshape([392, 2, 10000], order='F')
    # labels
    YTest = test_data[:, 0].astype(int)

    # import onnx network to StarV net
    net_name = f'MNIST_{type.upper()}{hidden}net'
    net_dir = f'{data_dir}/{net_name}.onnx'

    if type == 'gru':
        net = load_GRU_network(net_dir, net_name)
    else:
        net = load_LSTM_network(net_dir, net_name)

    print(f'\nVerifying {net_name}')

    NSample = 100
    # get first 100 correctly classified images
    x = []
    y = []
    for i in range(10000):
        if net.evaluate(XTest[:, :, i]).argmax() == YTest[i]:
            x.append(XTest[:, :, i])
            y.append(YTest[i])
            if len(x) == NSample:
                break

    if hidden == 15:
        eps = [0.005, 0.01, 0.015, 0.02, 0.025]
    else:
        eps = [0.0025, 0.005, 0.0075, 0.01, 0.0125]

    ###
    RB_est = [np.array([]) for _ in range(len(eps))]
    VT_est = [np.array([]) for _ in range(len(eps))]
    lp_solver = 'estimate'
    DR = 0

    print('==============================================')
    print('Estimate')
    for e in range(len(eps)):
        print('working epsilon: ', eps[e])
        for i in range(len(x)):
            print('working image #:', i)
            _, rb, vt = certifyRobustness_sequence(net, x[i], epsilon=eps[e], lp_solver=lp_solver, DR=DR, show=False)
            RB_est[e] = np.hstack([RB_est[e], rb[0]])
            VT_est[e] = np.hstack([VT_est[e], vt])
        print()

    RB_lp = [np.array([]) for _ in range(len(eps))]
    VT_lp = [np.array([]) for _ in range(len(eps))]
    lp_solver = 'gurobi'
    DR = 0

    print('LP')
    for e in range(len(eps)):
        print('working epsilon: ', eps[e])
        for i in range(len(x)):
            print('working image #:', i)
            _, rb, vt = certifyRobustness_sequence(net, x[i], epsilon=eps[e], lp_solver=lp_solver, DR=DR, show=False)
            RB_lp[e] = np.hstack([RB_lp[e], rb[0]])
            VT_lp[e] = np.hstack([VT_lp[e], vt])
        print()


    ####
    RB_dr1 = [np.array([]) for _ in range(len(eps))]
    VT_dr1 = [np.array([]) for _ in range(len(eps))]
    lp_solver = 'gurobi'
    DR = 1

    print('\n==============================================')
    print('DR1')
    for e in range(len(eps)):
        print('working epsilon: ', eps[e])
        for i in range(len(x)):
            print('working image #:', i)
            _, rb, vt = certifyRobustness_sequence(net, x[i], epsilon=eps[e], lp_solver=lp_solver, DR=DR, show=False)
            RB_dr1[e] = np.hstack([RB_dr1[e], rb[0]])
            VT_dr1[e] = np.hstack([VT_dr1[e], vt])
        print()

    ###
    RB_dr2 = [np.array([]) for _ in range(len(eps))]
    VT_dr2 = [np.array([]) for _ in range(len(eps))]
    lp_solver = 'gurobi'
    DR = 2

    print('\n==============================================')
    print('DR2')
    for e in range(len(eps)):
        print('working epsilon: ', eps[e])
        for i in range(len(x)):
            print('working image #:', i)
            _, rb, vt = certifyRobustness_sequence(net, x[i], epsilon=eps[e], lp_solver=lp_solver, DR=DR, show=False)
            RB_dr2[e] = np.hstack([RB_dr2[e], rb[0]])
            VT_dr2[e] = np.hstack([VT_dr2[e], vt])
        print()

    rb_est = np.array([]) 
    vt_est = np.array([])
    rb_lp = np.array([]) 
    vt_lp = np.array([])
    rb_dr1 = np.array([]) 
    vt_dr1 = np.array([])
    rb_dr2 = np.array([]) 
    vt_dr2 = np.array([])
    for e in range(len(eps)):
        rb_est = np.hstack([rb_est, (RB_est[e] == 1).sum()])
        vt_est = np.hstack([vt_est, VT_est[e].sum()/NSample])
        rb_lp = np.hstack([rb_lp, (RB_lp[e] == 1).sum()])
        vt_lp = np.hstack([vt_lp, VT_lp[e].sum()/NSample])
        rb_dr1 = np.hstack([rb_dr1, (RB_dr1[e] == 1).sum()])
        vt_dr1 = np.hstack([vt_dr1, VT_dr1[e].sum()/NSample])
        rb_dr2 = np.hstack([rb_dr2, (RB_dr2[e] == 1).sum()])
        vt_dr2 = np.hstack([vt_dr2, VT_dr2[e].sum()/NSample])

    # save verification results
    path = f"artifacts/{artifact}/results"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'{path}/mnist_{type}_{hidden}_results.npy', 'wb') as f:
        np.save(f, rb_est)
        np.save(f, rb_lp)
        np.save(f, rb_dr1)
        np.save(f, rb_dr2)
        
        np.save(f, vt_est)
        np.save(f, vt_lp)
        np.save(f, vt_dr1)
        np.save(f, vt_dr2)


def plot_rnns_results(type='lstm', hidden=15):

    ft_dir = f"artifacts/{artifact}/"
    f_dir = ft_dir + "results/"
    with open(f_dir + f'mnist_{type}_{hidden}_results.npy', 'rb') as f:
        rb_est = np.load(f)
        rb_lp = np.load(f)
        rb_dr1 = np.load(f)
        rb_dr2 = np.load(f)
        
        vt_est = np.load(f)
        vt_lp = np.load(f)
        vt_dr1 = np.load(f)
        vt_dr2 = np.load(f)

    with open(ft_dir + f'auto_lirpa_results/auto_lirpa_{type}_{hidden}_results.pkl', 'rb') as f:
        rb_autolirpa, vt_autolirpa, _ = pickle.load(f)

    if type == 'lstm':
        with open(ft_dir + f'popqorn_results/popqorn_{type}_{hidden}_results.pkl', 'rb') as f:
            rb_popqorn, vt_popqorn, _ = pickle.load(f)

    if hidden == 15:
        eps = [0.005, 0.01, 0.015, 0.02, 0.025]
        rb_loc = 'upper right'
        vt_loc = 'center right'
    else:
        eps = [0.0025, 0.005, 0.0075, 0.01, 0.0125]
        rb_loc = 'lower left'
        vt_loc = 'center right'

    fig_num = 2

    ft = 13 #font_size
    cm = 1/2.54  # centimeters in inches
    lw = 2.5

    legend = ['EST', 'LP', 'DR=1', 'DR=2', 'AutoLirpa']
    if type == 'lstm':
        legend.append('$POPQORN$')

    fig, ax = plt.subplots(figsize=(12*cm, 15*cm), tight_layout=True)
    plt.title(f"MNIST {type.upper()}{hidden} RB")
    
    plt.plot(eps, rb_est, "--gD", linewidth=lw)
    plt.plot(eps, rb_lp, "-rs", linewidth=lw)
    plt.plot(eps, rb_dr1, "--co", linewidth=lw)
    plt.plot(eps, rb_dr2, "--bv", linewidth=lw)
    plt.plot(eps, rb_autolirpa, "--y*", linewidth=lw)
    
    if type == 'lstm':
        plt.plot(eps, rb_popqorn, "--mx", linewidth=lw)
    plt.legend(legend, fontsize=ft, loc=rb_loc)

    plt.xticks(fontsize=ft)
    plt.yticks(fontsize=ft)
    plt.xlabel("$\epsilon$", fontsize=ft)
    plt.ylabel("Robustness", fontsize=ft)
    ax.set_xticks(eps)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
    plt.savefig(f_dir + f'Figure_{fig_num}__MNIST_{type.upper()}_{hidden}_net_RB.png')
    plt.show()


def verify_convnet_network(net_type='Small', dtype='float32'):

    print('=================================================================================')
    print(f"Verification of CAV2020 {net_type} ConvNet Network against Brightnening Attack")
    print('=================================================================================\n')

    folder_dir = f"StarV/util/data/nets/CAV2020_MNIST_ConvNet"
    net_dir = f"{folder_dir}/onnx/{net_type}_ConvNet.onnx"
    starvNet = load_convnet(net_dir, net_type, dtype=dtype)
    print()
    print(starvNet.info())

    mat_file = scipy.io.loadmat(f"{folder_dir}/nnv/{net_type}_images.mat")
    data = mat_file['IM_data'].astype(dtype)
    labels = mat_file['IM_labels'] - 1

    delta = [0.005, 0.01, 0.015]
    d = [250, 245, 240]
    N = 100 # number of test images used for robustness verification
    M = len(delta)
    P = len(d)

    IML = []
    CSRL = []
    COOL = []
    labelL = []

    for i, d_ in enumerate(d):
        IM_j = []
        CSR_j = []
        COO_j = []
        label_j = []

        for j, delta_ in enumerate(delta):

            count = 0
            IM_k = []
            CSR_k = []
            COO_k = []
            label_k = []
            for k in range(2000):
                lb, ub = brightening_attack(data[:, :, k], delta=delta_, d=d_, dtype=dtype)
                IM_k.append(ImageStar(lb, ub))
                CSR_k.append(SparseImageStar2DCSR(lb, ub))
                COO_k.append(SparseImageStar2DCOO(lb, ub))
                label_k.append(labels[k])
                count += 1

                if count == N:
                    break

            IM_j.append(IM_k)
            CSR_j.append(CSR_k)
            COO_j.append(COO_k)
            label_j.append(label_k)

        IML.append(IM_j)
        CSRL.append(CSR_j)
        COOL.append(COO_j)
        labelL.append(label_j)

    rbIM = np.zeros([P, M, N])
    vtIM = np.zeros([P, M, N])
    rbCSR = np.zeros([P, M, N])
    vtCSR = np.zeros([P, M, N])
    rbCOO = np.zeros([P, M, N])
    vtCOO = np.zeros([P, M, N])

    print(f"Verifying {net_type} ConvNet with ImageStar")
    rbIM_table = []
    vtIM_table = []
    for i in range(P):
        rb_delta_table = ['ImageStar']
        vt_delta_table = ['ImageStar']
        for j in range(M):
            IMs = IML[i][j]
            IDs = labelL[i][j]
            print(f"Verifying netowrk with d = {d[i]}, delta = {delta[j]}")

            rbIM[i, j, :], vtIM[i, j, :], _, _ = certifyRobustness(net=starvNet, inputs=IMs, labels=IDs,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None,
                RF=0.0, DR=0, return_output=False, show=False)
            rb_delta_table.append((rbIM[i, j, :]==1).sum())
            vt_delta_table.append((vtIM[i, j, :].sum() / N))
        rbIM_table.append(rb_delta_table)
        vtIM_table.append(vt_delta_table)

    print(f"\nVerifying {net_type} ConvNet with Sparse Image Star in CSR format")
    rbCSR_table = []
    vtCSR_table = []
    for i in range(P):
        rb_delta_table = ['SIM_CSR']
        vt_delta_table = ['SIM_CSR']
        for j in range(M):
            CSRs = CSRL[i][j]
            IDs = labelL[i][j]
            print(f"Verifying netowrk with d = {d[i]}, delta = {delta[j]}")

            rbCSR[i, j, :], vtCSR[i, j, :], _, _ = certifyRobustness(net=starvNet, inputs=CSRs, labels=IDs,
                    veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None,
                    RF=0.0, DR=0, return_output=True, show=False)
            rb_delta_table.append((rbCSR[i, j, :]==1).sum())
            vt_delta_table.append((vtCSR[i, j, :].sum() / N))
        rbCSR_table.append(rb_delta_table)
        vtCSR_table.append(vt_delta_table)

    print(f"\nVerifying {net_type} ConvNet with Sparse Image Star in COO format")
    rbCOO_table = []
    vtCOO_table = []
    for i in range(P):
        rb_delta_table = ['SIM_COO']
        vt_delta_table = ['SIM_COO']
        for j in range(M):
            COOs = COOL[i][j]
            IDs = labelL[i][j]
            print(f"Verifying netowrk with d = {d[i]}, delta = {delta[j]}")

            rbCOO[i, j, :], vtCOO[i, j, :], _, _ = certifyRobustness(net=starvNet, inputs=COOs, labels=IDs,
                    veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None,
                    RF=0.0, DR=0, return_output=True, show=False)
            rb_delta_table.append((rbCOO[i, j, :]==1).sum())
            vt_delta_table.append((vtCOO[i, j, :].sum() / N))
        rbCOO_table.append(rb_delta_table)
        vtCOO_table.append(vt_delta_table)

    # save verification results
    path = f"artifacts/{artifact}/results"
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(P):
        headers = [f"d={d[i]}", f"delta={delta[0]}", f"delta={delta[1]}", f"delta={delta[2]}"]
        # Robustness Resluts
        data = [rbIM_table[i], rbCSR_table[i], rbCOO_table[i]]
        print('-----------------------------------------------------')
        print('Robustness')
        print('-----------------------------------------------------')
        print(tabulate(data, headers=headers))
        print()

        Tlatex = tabulate(data, headers=headers, tablefmt='latex')
        with open(path+f"/{net_type}ConvNet_brightAttack_Table_d{d[i]}_rb.tex", "w") as f:
            print(Tlatex, file=f)

        # Verification Time Results
        data = [vtIM_table[i], vtCSR_table[i], vtCOO_table[i]]
        print('-----------------------------------------------------')
        print('Verification Time')
        print('-----------------------------------------------------')
        print(tabulate(data, headers=headers))
        print()

        Tlatex = tabulate(data, headers=headers, tablefmt='latex')
        with open(path+f"/{net_type}ConvNet_brightAttack_Table_d{d[i]}_vt.tex", "w") as f:
            print(Tlatex, file=f)

    save_file = path + f"/{net_type}ConvNet_brightAttack_results.pkl"
    pickle.dump([rbIM, vtIM, rbIM_table, vtIM_table, rbCSR, vtCSR, rbCSR_table, vtCSR_table, rbCOO, vtCOO, rbCOO_table, vtCOO_table], open(save_file, "wb"))

    print('=====================================================')
    print('DONE!')
    print('=====================================================')
    
    
def plot_table_covnet_network_all():
    net_types = ['Small', 'Medium', 'Large']
    dir = f"artifacts/{artifact}/results/"
    save_dir = dir + f'Table_2__Verification_results_of_the_MNIST_CNN.tex'
    
    rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO = [], [], [], [], [], []
    for net_type in net_types:
        result_dir = dir + f'{net_type}ConvNet_brightAttack_results.pkl'
        
        with open(result_dir, 'rb') as f:
            [_, _, rbim, vtim, _, _, rbcsr, vtcsr, _, _, rbcoo, vtcoo] = pickle.load(f)
        rbIM.append(rbim)
        vtIM.append(vtim)
        rbCSR.append(rbcsr)
        vtCSR.append(vtcsr)
        rbCOO.append(rbcoo)
        vtCOO.append(vtcoo)
    
    rbNNV, vtNNV = [], []
    mat_path = f"StarV/util/data/nets/CAV2020_MNIST_ConvNet/nnv/"
    for net_type in net_types:
        mat_file = scipy.io.loadmat(mat_path + f"NNV_{net_type}_ConvNet_Results_brightAttack.mat")
        rbNNV.append(mat_file['r_star'])
        vtNNV.append(mat_file['VT_star'])        

    delta = [0.005, 0.01, 0.015]
    d = [250, 245, 240]
    N = 100
    
    file = open(save_dir, "w")
    L = [
        r"\begin{table}[]" + '\n',
        r"\centering" + '\n',
        r"\resizebox{\columnwidth}{!}{" + '\n',
        r"\footnotesize" + '\n',
        r"\begin{tabular}{r||c||c:c:c||c:c:c||c:c:c ||c:c:c||c:c:c||c:c:c}" + '\n',
        r"      & & \multicolumn{9}{c||}{Robustness results ($\%$)}  & \multicolumn{9}{c}{Verification time (sec)} \\" + '\n',
        r"\hline" + '\n',
        r"      & & \multicolumn{3}{c||}{Small} & \multicolumn{3}{c||}{Medium} & \multicolumn{3}{c||}{Large} & " +
        r"\multicolumn{3}{c||}{Small} & \multicolumn{3}{c||}{Medium} & \multicolumn{3}{c}{Large} \\"  + '\n',
        r"      & & $\delta = 0.005$ & $\delta = 0.01$ & $\delta = 0.015$ & $\delta = 0.005$ & $\delta = 0.01$ & $\delta = 0.015$ & " +
        r"$\delta = 0.005$ & $\delta = 0.01$ & $\delta = 0.015$ & $\delta = 0.005$ & $\delta = 0.01$ & $\delta = 0.015$ & " +
        r"$\delta = 0.005$ & $\delta = 0.01$ & $\delta = 0.015$ & $\delta = 0.005$ & $\delta = 0.01$ & $\delta = 0.015$ \\" + '\n',
        r"\hline" + '\n',
    ]
    file.writelines(L)
    
    for i in range(len(d)):
        file.write(r"\hline" + '\n')
        line = r"\multirow{4}{*}{\rotatebox{90}{$d=" + f"{d[i]}" + r"$}}" + '\n'
        file.write(line)
        
        line = f'& IM'
        for j in range(len(net_types)):
            line += f' & {rbIM[j][i][1]} & {rbIM[j][i][2]} &  {rbIM[j][i][3]}'
        for j in range(len(net_types)):
            line += f' & {vtIM[j][i][1] :.3f} & {vtIM[j][i][2] :.3f} &  {vtIM[j][i][3] :.3f}'
        file.write(line + ' \\\\\n')
        
        line = f'& SIM\\_csr'
        for j in range(len(net_types)):
            line += f' & {rbCSR[j][i][1]} & {rbCSR[j][i][2]} &  {rbCSR[j][i][3]}'
        for j in range(len(net_types)):
            line += f' & {vtCSR[j][i][1] :.3f} & {vtCSR[j][i][2] :.3f} &  {vtCSR[j][i][3] :.3f}'
        file.write(line + ' \\\\\n')
        
        line = f'& SIM\\_coo'
        for j in range(len(net_types)):
            line += f' & {rbCOO[j][i][1]} & {rbCOO[j][i][2]} &  {rbCOO[j][i][3]}'
        for j in range(len(net_types)):
            line += f' & {vtCOO[j][i][1] :.3f} & {vtCOO[j][i][2] :.3f} &  {vtCOO[j][i][3] :.3f}'
        file.write(line + ' \\\\\n')
        
        line = f'& NNV'
        for j in range(len(net_types)):
            line += f' & {int(rbNNV[j][i, 0]*100)} & {int(rbNNV[j][i, 1]*100)} &  {int(rbNNV[j][i, 2]*100)}'
        for j in range(len(net_types)):
            line += f' & {vtNNV[j][i, 0]/100 :.3f} & {vtNNV[j][i, 1]/100 :.3f} &  {vtNNV[j][i, 2]/100 :.3f}'
        file.write(line + ' \\\\\n')
        
        file.write(r"\hline" + '\n')
            
    L = [
        r"\end{tabular}" + '\n',
        r"}" + '\n',
        r"\caption{Verification results of the MNIST CNN \cite{tran2020cav}.}" + '\n',
        r"\label{tab:CAV2020_mnist_convnet} + '\n'"
        r"\end{table} + '\n'"
    ]
    file.writelines(L)
    file.close()

    print('=====================================================')
    print('DONE!')
    print('=====================================================')

def verify_vgg16_network_spec_cn(dtype='float64'):

    print('=================================================================================')
    print(f"Verification of VGG16 Network against Infinity Norm Attack Spec_cn")
    print('=================================================================================\n')

    folder_dir = f"StarV/util/data/nets/vggnet16"
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())


    shape = (3, 224, 224)

    vnnlib_dir = f"{folder_dir}/vnnlib/spec_cn"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)

    # save verification results
    path = f"artifacts/{artifact}/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_spec_cn_results.pkl"

    N = len(vnnlib_files)
    rbCSR = np.zeros(N)
    vtCSR = np.zeros(N)
    rbCOO = np.zeros(N)
    vtCOO = np.zeros(N)
    numPred = np.zeros(N)

    show = True

    print(f"\n\nVerifying vggnet16 with SparseImageStar in CSR format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        print(f"\n Loading a VNNLIB file")
        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        CSR = SparseImageStar2DCSR(lb, ub)
        del lb, ub, bounds

        rbCSR[i], vtCSR[i], _, Y = certifyRobustness(net=starvNet, inputs=CSR, labels=label,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None,
            RF=0.0, DR=0, return_output=False, show=show)
        numPred[i] = Y.num_pred

        if rbCSR[i] == 1:
            print(f"ROBUSTNESS RESULT: ROBUST")
        elif rbCSR[i] == 2:
            print(f"ROBUSTNESS RESULT: UNKNOWN")
        elif rbCSR[i] == 0:
            print(f"ROBUSTNESS RESULT: UNROBUST")

        print(f"VERIFICATION TIME: {vtCSR[i]}")
        print(f"NUM_PRED: {numPred[i]}")
        pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))
    del CSR, Y

    print(f"\n\nVerifying vggnet16 with SparseImageStar in COO format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        print(f"\n Loading a VNNLIB file")
        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        COO = SparseImageStar2DCOO(lb, ub)
        del lb, ub, bounds

        rbCOO[i], vtCOO[i], _, _ = certifyRobustness(net=starvNet, inputs=COO, labels=label,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None,
            RF=0.0, DR=0, return_output=False, show=show)

        if rbCOO[i] == 1:
            print(f"ROBUSTNESS RESULT: ROBUST")
        elif rbCOO[i] == 2:
            print(f"ROBUSTNESS RESULT: UNKNOWN")
        elif rbCOO[i] == 0:
            print(f"ROBUSTNESS RESULT: UNROBUST")

        print(f"VERIFICATION TIME: {vtCOO[i]}")
        pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))

    pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))

    headers = [f"SIM_csr, SIM_coo"]

    # Robustness Resluts
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), rbCSR, rbCOO], headers=headers))
    print()

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), vtCSR, vtCOO], headers=headers))
    print()

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def plot_table_vgg16_network():
    folder_dir = f"artifacts/{artifact}/results/"
    file_dir = folder_dir + 'vggnet16_vnncomp23_spec_cn_results.pkl'
    with open(file_dir, 'rb') as f:
        numPred_cn, rbCSR_cn, vtCSR_cn, rbCOO_cn, vtCOO_cn = pickle.load(f)

    f_dir = f"StarV/util/data/nets/vggnet16"
    net_dir = f"{f_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)
    vnnlib_dir = f"{f_dir}/vnnlib"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)
    shape = (3, 224, 224)

    num_attack_pixel = []
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype='float32')
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0])
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0])
        num_attack_pixel.append(int((lb != ub).sum()))

    headers = ['Specs', 'e', 'Result', 'm', 'IM', 'SIM_csr', 'SIM_coo', 'NNV', 'DeepPoly',  'Marabou', 'IM', 'NNV', 'NNENUM', 'ab-CROWN', 'b-CROWN']
    result = 'UNSAT'

    num_attack_pixel_cn = [200, 300, 400, 500, 1000, 2000, 3000]
    N_cn = len(numPred_cn)
    vt_NNENUM_cn = [744.02, 1060.96, 1354.75, 1781.26, 'T/O', 'T/O', 'O/M']
    vt_bcrown_cn = [26782.327130317688, 37052.68477010727, 'T/O', 'T/O', 'T/O', 'T/O', 'T/O']
    vt_abcrown_cn = 'T/O'
    vt_DP = 'O/M'
    vt_marabou = 'T/O'
    
    data = []
    for i in range(N_cn):
        vt_im = 'O/M'
        vt_imc = 'O/M'
        vt_nnv = 'O/M'
        vt_nnvc = 'O/M'
        vt_bcrown_cd = vt_bcrown_cn[i] if vt_bcrown_cn[i] == 'T/O' else f"{np.array(vt_bcrown_cn[i], dtype='float64'):0.1f}"

        nPred = 'NA' if np.isnan(vtCSR_cn[i]) else f"{numPred_cn[i]}"
        data.append([f"c_{i}", num_attack_pixel_cn[i], result, nPred,  vt_im, f"{vtCSR_cn[i]:0.1f}", f"{vtCOO_cn[i]:0.1f}", vt_nnv, vt_DP, vt_marabou, vt_imc, vt_nnvc, vt_NNENUM_cn[i], vt_abcrown_cn, vt_bcrown_cd])

    print(tabulate(data, headers=headers))

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(folder_dir+f"Table_3__vggnet16_vnncomp23_results.tex", "w") as f:
        print(Tlatex, file=f)

    print('=====================================================')
    print('DONE!')
    print('=====================================================')
    
    
def memory_usage_vgg16(spec):

    print('=================================================================================')
    print(f"Memory Usage of VGG16")
    print('=================================================================================\n')

    dtype = 'float64'

    folder_dir = 'StarV/util/data/nets/vggnet16'
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())

    mat_file = scipy.io.loadmat(f"{folder_dir}/nnv/nnv_vgg16_memory_usage.mat")
    nnv_nb = mat_file['memory_usage'].ravel()
    nnv_time = mat_file['reach_time'].ravel()

    shape = (3, 224, 224)

    vnnlib_dir = f"{folder_dir}/vnnlib"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)

    vnnlib_file = vnnlib_files[spec]
    vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

    with open(vnnlib_file_dir) as f:
        first_line = f.readline().strip('\n')
    label = int(re.findall(r'\b\d+\b', first_line)[0])

    vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

    box, spec_list = vnnlib_rv[0]
    bounds = np.array(box, dtype=inp_dtype)
    # transpose from [C, H, W] to [H, W, C]
    lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
    ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

    num_attack_pixel = (lb != ub).sum()
    print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

    IM = ImageStar(lb, ub)
    COO = SparseImageStar2DCOO(lb, ub)
    CSR = SparseImageStar2DCSR(lb, ub)

    IM_time = []; COO_time = []; CSR_time = [];
    IM_nb = [IM.nbytes()]; COO_nb = [COO.nbytes()]; CSR_nb = [CSR.nbytes()]
    IM_shape = [IM.V.shape]; COO_shape = [COO.shape + (COO.num_pred, )]; CSR_shape = [CSR.shape + (CSR.num_pred, )]
    nPred = [CSR.num_pred]
    density = [CSR.density()]

    for i in range(starvNet.n_layers):
        start = time.perf_counter()
        IM = starvNet.layers[i].reach(IM, method='approx', show=False)
        IM_time.append(time.perf_counter() - start)
        IM_nb.append(IM.nbytes())
        IM_shape.append(IM.V.shape)
    del IM

    for i in range(starvNet.n_layers):
        start = time.perf_counter()
        CSR = starvNet.layers[i].reach(CSR, method='approx', show=False)
        CSR_time.append(time.perf_counter() - start)
        CSR_nb.append(CSR.nbytes())
        nPred.append(CSR.num_pred)
        CSR_shape.append(CSR.shape)
        density.append(CSR.density())
    del CSR

    for i in range(starvNet.n_layers):
        start = time.perf_counter()
        COO = starvNet.layers[i].reach(COO, method='approx', show=False)
        COO_time.append(time.perf_counter() - start)
        COO_nb.append(COO.nbytes())
        COO_shape.append(COO.shape)
    del COO


    # save verification results
    path = f"artifacts/{artifact}/results"
    if not os.path.exists(path):
        os.makedirs(path)

    x = np.arange(len(CSR_time))
    x_ticks_labels = []
    for i in range(starvNet.n_layers):
        if starvNet.layers[i].__class__.__name__ == 'Conv2DLayer':
            l_name = '$L_c$'
        elif starvNet.layers[i].__class__.__name__ == 'ReLULayer':
            l_name = '$L_r$'
        elif starvNet.layers[i].__class__.__name__ == 'FlattenLayer':
            l_name = '$L_{{flat}}$'
        elif starvNet.layers[i].__class__.__name__ == 'FullyConnectedLayer':
            l_name = '$L_f$'
        elif starvNet.layers[i].__class__.__name__ == 'MaxPool2DLayer':
            l_name = '$L_m$'
        elif starvNet.layers[i].__class__.__name__ == 'BatchNorm2DLayer':
            l_name = '$L_b$'
        else:
            raise Exception('Unknown layer')
        x_ticks_labels.append(f"{l_name}_{i}")

    plt.rcParams["figure.figsize"] = [8.50, 5.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1,1)
    plt.title("Computation Time", fontsize=14)
    plt.plot(x, IM_time, color='red', linewidth=2)
    plt.plot(x, COO_time, color='black', linewidth=2)
    plt.plot(x, CSR_time, color="magenta", linewidth=2)
    plt.plot(x, nnv_time[1:], color='blue', linewidth=2)
    plt.xlabel("Layers", fontsize=12)
    plt.ylabel("Computation Time (sec)", fontsize=12)

    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks_labels, rotation=80, fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    # set legend
    ax.legend(['ImageStar', 'SIM COO', 'SIM CSR', 'NNV'], fontsize=12)

    plt.savefig(f'{path}/computation_time_vgg16_spec_{spec}.png')
    # plt.show()
    plt.close()


    x = np.arange(len(IM_nb))
    x_ticks_labels.insert(0, 'Input')

    plt.rcParams["figure.figsize"] = [8.50, 5.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1,1)
    plt.title("Memory Usage", fontsize=14)
    plt.plot(x, IM_nb, color="red", linewidth=2)
    plt.plot(x, COO_nb, color='black', linewidth=2)
    plt.plot(x, CSR_nb, color="magenta", linewidth=2)
    plt.plot(x, nnv_nb, color='blue', linewidth=2)
    plt.xlabel("Layers", fontsize=12)
    plt.ylabel("Bytes", fontsize=12)

    # Set number of ticks for x-axis
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(x_ticks_labels, rotation=80, fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    # set legend
    ax.legend(['ImageStar', 'SIM COO', 'SIM CSR', 'NNV'], loc='center right', fontsize=12)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.plot(x, density, color="green", linewidth=2)
    ax2.legend(['density'], loc='upper center', fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)

    plt.savefig(f'{path}/Figure_3__memory_usage_vgg16_spec_{spec}.png')
    # plt.show()
    plt.close()

    save_file = path + f"/memory_usage_vgg16_results_spec_{spec}.pkl"
    pickle.dump([IM_time, COO_time, CSR_time, IM_nb, COO_nb, CSR_nb, \
                 IM_shape, COO_shape, CSR_nb, nPred, density], open(save_file, "wb"))

    print('=====================================================')
    print('DONE!')
    print('=====================================================')
    

def format_net_name(x, y):
    """Format network name as x-y"""
    return f"{x}-{y}"

def quantiverify_ACASXU_ReLU_table_3(x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
    """Verify all ACASXU ReLU networks with spec_id"""
    print_util('h2')
    results = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')
    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for p_filter in p_filters:
            print_util('h3')
            print('quanti verify of ACASXU N_{}_{} ReLU network under specification {}...'.format(x[i], y[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_ACASXU_ReLU(x[i], y[i], spec_ids[i])
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            a = 3.0  # coefficient to adjust the distribution
            sig = (mu - S.pred_lb)/a
            print('Mean of predicate variables: mu = {}'.format(mu))
            print('Standard deviation of predicate variables: sig = {}'.format(sig))
            Sig = np.diag(np.square(sig))
            print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            inputSetProb = inputSet[0].estimateProbability()
            
            start = time.time()
            OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(
                net=net, inputSet=inputSet, unsafe_mat=unsmat,
                unsafe_vec=unsvec, numCores=numCores, p_filter=p_filter
            )
            end = time.time()
            verifyTime = end-start

            # Store results in dictionary format
            result = {
                'Prop': spec_ids[i],
                'Net': format_net_name(x[i], y[i]),
                'p_f': p_filter,
                'O': len(OutputSet),
                'US-O': len(unsafeOutputSet),
                'C': len(counterInputSet),
                'US-Prob-LB': prob_lb,
                'US-Prob-UB': prob_ub,
                'US-Prob-Min': prob_min,
                'US-Prob-Max': prob_max,
                'I-Prob': inputSetProb,
                'VT': verifyTime
            }
            results.append(result)
            print_util('h3')

    # Print verification results
    print(tabulate([[r['Prop'], r['Net'], r['p_f'], r['O'], r['US-O'], r['C'], 
                    r['US-Prob-LB'], r['US-Prob-UB'], r['US-Prob-Min'], r['US-Prob-Max'],
                    r['I-Prob'], r['VT']] for r in results],
                  headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",
                          "UnsafeProb-LB", "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max",
                          "inputSet Probability", "VerificationTime"]))

    # Save results to pickle file
    path = f"artifacts/{artifact}/ACASXU/ReLU"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/AcasXu_ReLU_ProbStar.pkl', 'wb') as f:
        pickle.dump(results, f)

    print_util('h2')
    return results


def quantiverify_ACASXU_ReLU_MC_table_3(x, y, spec_ids, unsafe_mat, unsafe_vec, numSamples, nTimes, numCore):
    """Verify all ACASXU ReLU networks with spec_id using Monte Carlo"""
    print_util('h2')
    results = []

    if len(x) != len(y):
        raise Exception('length(x) should equal length(y)')
    if len(x) != len(spec_ids):
        raise Exception('length(x) should equal length(spec_ids)')

    for i in range(len(x)):
        for numSample in numSamples:
            print_util('h3')
            print('quanti verify using Monte Carlo of ACASXU N_{}_{} ReLU network under specification {}...'.format(x[i], y[i], spec_ids[i]))
            net, lb, ub, unsmat, unsvec = load_ACASXU_ReLU(x[i], y[i], spec_ids[i])
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            a = 3.0  # coefficient to adjust the distribution
            sig = (mu - S.pred_lb)/a
            print('Mean of predicate variables: mu = {}'.format(mu))
            print('Standard deviation of predicate variables: sig = {}'.format(sig))
            Sig = np.diag(np.square(sig))
            print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)

            start = time.time()
            unsafe_prob = quantiVerifyMC(net=net, inputSet=In, unsafe_mat=unsmat, 
                                         unsafe_vec=unsvec, numSamples=numSample, nTimes=nTimes, numCores=numCore)
            end = time.time()
            verifyTime = end-start

            # Store results in dictionary format
            result = {
                'Prop': spec_ids[i],
                'Net': format_net_name(x[i], y[i]),
                'p_f': 0,  # Monte Carlo results only correspond to p_f = 0
                'MC_US-Prob': unsafe_prob,
                'MC_VT': verifyTime
            }
            results.append(result)
            print_util('h3')

    # Print verification results
    print(tabulate([[r['Prop'], r['Net'], r['MC_US-Prob'], r['MC_VT']] for r in results],
                  headers=["Prop.", "Net", "UnsafeProb", "VerificationTime"]))

    # Save results to pickle file
    path = f"artifacts/{artifact}/ACASXU/ReLU"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/AcasXu_ReLU_MC.pkl', 'wb') as f:
        pickle.dump(results, f)

    print_util('h2')
    return results


def qualiverify_ACASXU_ReLU_other_tools_table_3():
    """
    Verify all ACASXU ReLU networks with spec_id using other verification tools
    (NNV, Marabou, NNenum)
    """
    data = [
    {"x": 1, "y": 6, "s": 2, "Result": "violated", "NNV_exact": 13739.970, "Marabou": 166.58, "NNenum": 1.5938},
    {"x": 2, "y": 2, "s": 2, "Result": "violated", "NNV_exact": 21908.227, "Marabou": 9.27, "NNenum": 0.89502},
    {"x": 2, "y": 9, "s": 2, "Result": "violated", "NNV_exact": 74328.776, "Marabou": 31.19, "NNenum": 1.0538},
    {"x": 3, "y": 1, "s": 2, "Result": "violated", "NNV_exact": 5601.779, "Marabou": 3.50, "NNenum": 0.86799},
    {"x": 3, "y": 6, "s": 2, "Result": "violated", "NNV_exact": 74664.104, "Marabou": 36.28, "NNenum": 0.91804},
    {"x": 3, "y": 7, "s": 2, "Result": "violated", "NNV_exact": 23282.763, "Marabou": 105.32, "NNenum": 61.198},
    {"x": 4, "y": 1, "s": 2, "Result": "violated", "NNV_exact": 17789.960, "Marabou": 9.58, "NNenum": 0.87820},
    {"x": 4, "y": 7, "s": 2, "Result": "violated", "NNV_exact": 40696.630, "Marabou": 8.67, "NNenum": 0.90657},
    {"x": 5, "y": 3, "s": 2, "Result": "violated", "NNV_exact": 2740.739, "Marabou": 113.12, "NNenum": 1.9434},
    {"x": 1, "y": 7, "s": 3, "Result": "violated", "NNV_exact": 0.943, "Marabou": 0.25, "NNenum": 0.86683},
    {"x": 1, "y": 9, "s": 4, "Result": "violated", "NNV_exact": 1.176, "Marabou": 0.31, "NNenum": 0.86635}
    ]

    results = []
    for entry in data:
        result = {
            'Prop': entry['s'],
            'Net': format_net_name(entry['x'], entry['y']),
            'p_f': 0,  # Other tools results only correspond to p_f = 0
            'NNV': entry['NNV_exact'],
            'Marabou': entry['Marabou'],
            'NNenum': entry['NNenum']
        }
        results.append(result)
    
    # Save to pickle file
    path = f"artifacts/{artifact}/ACASXU/ReLU"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/AcasXu_ReLU_other_tools.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results


def generate_table_3_AcasXu_ReLU_quanti_verify_vs_other_tools():
    """
    Generate LaTeX table combining verification results for ACASXU ReLU networks:
    1. Quantitative Verification (ProbStar)
    2. Monte Carlo results
    3. Other verification tools (NNV, Marabou, NNenum)
    """
    def format_number(value):
        """Preserve scientific notation for very small numbers and original format"""
        if pd.isna(value) or value is None:
            return ''
        elif isinstance(value, int):
            return f'{value}'
        elif isinstance(value, float):
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
        
    path = f"artifacts/{artifact}/ACASXU/ReLU"
    # Load all data sources
    probstar_data = load_pickle_file(path + '/AcasXu_ReLU_ProbStar.pkl')
    mc_data = load_pickle_file(path + '/AcasXu_ReLU_MC.pkl')
    other_tools_data = load_pickle_file(path + '/AcasXu_ReLU_other_tools.pkl')

    # Create lookup dictionaries for MC and other tools data
    mc_dict = {(d['Prop'], d['Net']): d for d in mc_data}
    other_dict = {(d['Prop'], d['Net']): d for d in other_tools_data}

    # Sort probstar data by Prop, Net, and p_f
    sorted_data = sorted(probstar_data, key=lambda x: (x['Prop'], x['Net'], x['p_f']))

    # Generate LaTeX table
    table_lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllllllllll||ll||l|l|l}",
        r"\hline",
        r"    \multicolumn{12}{c||}{\textbf{Quantitative Verification}}  & " + 
        r"\multicolumn{2}{c||}{\textbf{Monte Carlo (NS: $10^7$)}} & " +
        r"\multicolumn{3}{c}{\textbf{Qualitative Verification}} \\",
        r"\hline",
        r"\textbf{Prop} & \textbf{Net} & \textbf{$p_f$} & " +
        r"\textbf{$\mathcal{O}$} & \textbf{$\mathcal{US-O}$} & " +
        r"\textbf{$\mathcal{C}$} & \textbf{US-Prob-LB} & \textbf{US-Prob-UB} & " +
        r"\textbf{US-Prob-Min} & \textbf{US-Prob-Max} & \textbf{I-Prob} & " +
        r"\textbf{VT} & \textbf{US-Prob} & \textbf{VT} & \textbf{NNV} & " +
        r"\textbf{Marabou} & \textbf{NNenum}\\",
        r"\hline"
    ]

    # Add data rows
    for entry in sorted_data:
        prop = entry['Prop']
        net = entry['Net']
        p_f = entry['p_f']
        
        # Base row with Quantitative Verification data
        row = [
            str(prop),
            net,
            str(p_f),
            format_number(entry['O']),
            format_number(entry['US-O']),
            format_number(entry['C']),
            format_number(entry['US-Prob-LB']),
            format_number(entry['US-Prob-UB']),
            format_number(entry['US-Prob-Min']),
            format_number(entry['US-Prob-Max']),
            format_number(entry['I-Prob']),
            format_number(entry['VT'])
        ]

        # Add Monte Carlo and Other Tools data only if p_f = 0
        if p_f == 0:
            mc_entry = mc_dict.get((prop, net), {})
            other_entry = other_dict.get((prop, net), {})
            
            row.extend([
                format_number(mc_entry.get('MC_US-Prob', '')),
                format_number(mc_entry.get('MC_VT', '')),
                format_number(other_entry.get('NNV', '')),
                format_number(other_entry.get('Marabou', '')),
                format_number(other_entry.get('NNenum', ''))
            ])
        else:
            # Add empty cells for Monte Carlo and Other Tools columns
            row.extend([''] * 5)

        table_lines.append(' & '.join(row) + r' \\')

    # Add table footer
    table_lines.extend([
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ])

    # Join all lines with newlines and save to file
    table_content = '\n'.join(table_lines)
    with open(path + '/Table_4__AcasXu_ReLU_quanti_verify_vs_other_tools.tex', 'w') as f:
        f.write(table_content)

    print("Table has been generated and saved to 'Table_4__AcasXu_ReLU_quanti_verify_vs_other_tools.tex'")
    return table_content


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
    
    path = f"artifacts/{artifact}/ACC/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+f"/Table_5__verification_acc_trapezius_{net}_tab.tex", "w") as f:
         print(tabulate(verification_acc_trapezius_data, headers=["Spec.", "T", "p_max", "p_min", "reachTime", "checkTime", "verifyTime"], tablefmt='latex'), file=f)

    path = f"artifacts/{artifact}/ACC/results"
    if not os.path.exists(path):
        os.makedirs(path)
    save_file = path + f"/verification_acc_trapezius_{net}_results.pkl"
    pickle.dump(verification_acc_trapezius_data, open(save_file, "wb"))

def generate_temporal_specs_ACC_trapezius_table():
    T = [10, 20, 30, 50]
    Nets = ['controller_3_20', 'controller_5_20']

    path = f"artifacts/{artifact}/ACC/results"
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

    path = f"artifacts/{artifact}/ACC/table"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+f"/Table_5__verification_acc_trapezius_full_tab.tex", "w") as f:
        print(tabulate(result_table, headers=header, tablefmt='latex'), file=f)

def verify_temporal_specs_ACC_trapeziu_full():
    verify_temporal_specs_ACC_trapezius(net='controller_3_20')
    verify_temporal_specs_ACC_trapezius(net='controller_5_20')
    generate_temporal_specs_ACC_trapezius_table()
    
     
def generate_table_3_vs_Hylaa_tool():
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
        
    path = "results"
    # Load all data sources
    probstarTL_data = load_pickle_file(path + '/full_results.pkl')

    Hylaa_data = load_pickle_file(path + '/Hylaa_results.pkl')


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

        # Add Monte Carlo and Other Tools data only if p_f = 0
        if spec==0:
            Hylaa_entry = Hylaa_dict.get((model, spec), {})
            
            row.extend([
                format_number(Hylaa_entry.get('SAT', '')),
                format_number(Hylaa_entry.get('t_v', ''))

            ])
        else:
            # Add empty cells for Monte Carlo and Other Tools columns
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
    
    with open(path + '/Table_6__vs_Hylaa.tex', 'w') as f:
        f.write(table_content)

    print("Table has been generated and saved to 'Table_6__vs_Hylaa.tex'")
    return table_content


def harmonic(use_arnoldi =None,use_init_space=None):
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
    # print("init_state_bounds_list_shape:",len(init_state_bounds_list))
    # print("init_sate_bounds_list:",init_state_bounds_list)
    # print("init_state_lb:",init_state_lb)

    X0 = Star(init_state_lb,init_state_ub)


    mu_U = 0.5*(X0.pred_lb + X0.pred_ub) 
    a  = 3
    sig_U = (X0.pred_ub-mu_U )/a
    epsilon = 1e-10
    sig_U = np.maximum(sig_U, epsilon)
    Sig_U = np.diag(np.square(sig_U))



    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)

    h = math.pi/4
    time_bound = math.pi
    N = int (time_bound/ h)
    m = 2
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9

    output_space= random_two_dims_mapping(X0_probstar,1,2)
    initial_space = X0_probstar.V 

    unsafe_mat = np.array([[-1,0]])
    unsafe_vec = np.array([-4])


    unsafe_mat_list =[unsafe_mat]
    unsafe_vec_list = [unsafe_vec]


    reach_start_time = time.time()
    R,krylov_time = sim3(A,X0_probstar,h, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
    reach_time_duration = time.time() - reach_start_time

    p_min,smallest_prob_time_step, p_max, largest_prob_time_step,unsafeOutputSet, counterInputSet= quantiVerifier_LCS(R = R, inputSet=X0_probstar, unsafe_mat=unsafe_mat_list, \
                                                                                    unsafe_vec=unsafe_vec_list,time_step=h)

    # plot_probstar(unsafeOutputSet) 

            
# #============================ Temproal logic test =====================
    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()

    A1 = np.array([-1., 0.])
    b1 = np.array([-4])
    P1 = AtomicPredicate(A1,b1)

    A2 = np.array([0., -1])
    b2 = np.array([-4])
    P2 = AtomicPredicate(A2,b2)

    EVOT =_EVENTUALLY_(0,4)
    EVOT1 =_EVENTUALLY_(2,3)
    
    A3 = np.array([-1.,0])
    b3 = np.array([-4])
    P3 = AtomicPredicate(A3,b3)

    A4 = np.array([1,0])
    b4 = np.array([4])
    P4 = AtomicPredicate(A4,b4)

    AWOT = _ALWAYS_(0,4)
    AWOT1 = _ALWAYS_(1,2)

    spec = Formula([EVOT,P1])
    spec1 = Formula([AWOT1,lb,P4,OR,lb,EVOT1,P1,rb,rb])
    specs =[spec,spec1]
    # plot_probstar_signal(R)
    checking_time = []
    data=[]

    for i in range(0,len(specs)):
        check_start = time.time()
        spec = specs[i]
        print('\n==================Specification{}====================: '.format(i))
        spec.print()
        DNF_spec = spec.getDynamicFormula()
        DNF_spec.print()
        print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))

        _,p_max, p_min,Ncdnf, = DNF_spec.evaluate(R)

        end = time.time()
        checking_time = end -check_start 
        print("p_min:",p_min)
        print("p_max:",p_max)

        verify_time=checking_time + reach_time_duration    

        data.append(["Harmonic",i,p_min,p_max, reach_time_duration,checking_time,verify_time])

    return R,unsafeOutputSet,data
     
def run_mcs_model(use_arnoldi = None,use_init_space=None):
    
    print('=====================================================')
    print('Quantitative Verification of MCS Model Using Krylov Subspace')
    print('=====================================================')

    plant = load_mcs_model()
    combined_mat = combine_mats(plant.A,plant.B)
    dims = combined_mat.shape[0]


    #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
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

    specs=[]
    datas=[]

    for hi in h:
        N = int (time_bound/ hi)

        EV0T = _EVENTUALLY_(0,N)
        EV0T1 = _EVENTUALLY_(10,15)
        AW0T = _ALWAYS_(0,N)
        AW0T1 = _ALWAYS_(10,15)

        spec = Formula([EV0T,P1,AND,P2,AND,P3,AND,P4])
        spec1 = Formula([EV0T1,lb,P1,AND,P2,AND,P3,AND,P4,OR,lb,AW0T1,P11,AND,P22,AND,P33,AND,P44,rb,rb])
        spec2 = Formula([AW0T1,P1,OR,P2,OR,P3,OR,P4])

        specs=[spec,spec1,spec2]

        reach_start_time = time.time()
        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time

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
                    'Model': 'MCS',
                    'Spec': spec_id,
                    # 'Nadnf': Nadnf,
                    # 'Ncdnf': Ncdnf,
                    'p_min':p_min,
                    'p_max':p_max,
                    't_r':reach_time_duration,
                    't_c':checking_time,
                    't_v': verify_time
                }
            datas.append(data)


    return datas


def run_building_model(use_arnoldi =None,use_init_space=None):
    
    print('\n\n=====================================================')
    print('Quantitative Verification of Building Model Using Krylov Subspace')
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

    # init_sate_bounds_array=[np.array(list).reshape(48, 1) for list in init_sate_bounds_list]
    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]

    X0 = Star(init_state_lb,init_state_ub)

    
    # create ProbStar for initial state 
    mu_U = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_U = (mu_U - X0.pred_lb)/a
    Sig_U = np.diag(np.square(sig_U))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)

    output_space= plant.C
    expand_mat = np.zeros((1,1))
    output_space = np.hstack((output_space,expand_mat))
    initial_space = X0_probstar.V 


    inputProb = X0_probstar.estimateProbability()

    datas = []
    h = [0.1]
    time_bound = 20
    m = 4
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9
    
    for hi in h:

        AND = _AND_()
        OR = _OR_()                                        
        lb = _LeftBracket_()
        rb = _RightBracket_()

        A1 = np.array([-1])
        b1 = np.array([-0.004])
        P = AtomicPredicate(A1,b1)

        P1 = AtomicPredicate(-A1,-b1)

        N = int (time_bound/ hi)
        reach_start_time = time.time()

        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time

        EVOT =_EVENTUALLY_(0,N)
        EVOT1 =_EVENTUALLY_(0,10)
        AWOT=_ALWAYS_(0,N)
        AWOT1=_ALWAYS_(0,10)

        spec = Formula([EVOT,P]) # be unsafe at some step 0 to N
        spec1 = Formula([AWOT,P1]) # alwyas safe between 0 to N
        spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P,rb,rb])
        specs=[spec,spec1]
    

    inputProb = X0_probstar.estimateProbability()

    datas = []
    for hi in h:

        AND = _AND_()
        OR = _OR_()                                        
        lb = _LeftBracket_()
        rb = _RightBracket_()

        A1 = np.array([-1])
        b1 = np.array([-0.004])
        P = AtomicPredicate(A1,b1)

        P1 = AtomicPredicate(-A1,-b1)

        N = int (time_bound/ hi)
        reach_start_time = time.time()

        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
        reach_time_duration = time.time() - reach_start_time

        EVOT =_EVENTUALLY_(0,N)
        EVOT1 =_EVENTUALLY_(0,10)
        AWOT=_ALWAYS_(0,N)
        AWOT1=_ALWAYS_(0,10)

        spec = Formula([EVOT,P]) # be unsafe at some step 0 to N
        spec1 = Formula([AWOT,P1]) # alwyas safe between 0 to N
        spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P,rb,rb])
        specs=[spec2,spec,spec1]

        for spec_id in range(0,len(specs)):
            check_start = time.time()
            spec = specs[spec_id]
            print("\n===================Specification{}: =====================".format(spec_id))
            spec.print()
            # S  = spec.render(R)
            DNF_spec = spec.getDynamicFormula()
            Nadnf = DNF_spec.length
            print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
            _,p_max, p_min, Ncdnf = DNF_spec.evaluate(R)
            end = time.time()
            checking_time = end -check_start 
            print("p_min:",p_min)
            print("p_max:",p_max)
            verify_time = checking_time + reach_time_duration    

            if spec_id == 0:
                spec_id = 2
                data2 = {
                    'Model': 'Building',
                    'Spec': spec_id,
                    # 'Nadnf': Nadnf,
                    # 'Ncdnf': Ncdnf,
                    'p_min':p_min,
                    'p_max':p_max,
                    't_r':reach_time_duration,
                    't_c':checking_time,
                    't_v': verify_time
                }
            else:
                data = {
                    'Model': 'Building',
                    'Spec': spec_id,
                    # 'Nadnf': Nadnf,
                    # 'Ncdnf': Ncdnf,
                    'p_min':p_min,
                    'p_max':p_max,
                    't_r':reach_time_duration,
                    't_c':checking_time,
                    't_v': verify_time
                }
                datas.append(data)
        datas.append(data2)


    return datas


def run_pde_model(use_arnoldi = None,use_init_space=None):

        
    print('\n\n=====================================================')
    print('Quantitative Verification of PDE Model Using Krylov Subspace')
    print('=====================================================')
    plant = load_pde_model()
    combined_mat = combine_mats(plant.A,plant.B)
    dims = combined_mat.shape[0]

    #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []
    initial_dim = plant.A.shape[0]
    print("init_dim:",initial_dim)

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
        else:
           raise RuntimeError('Unknown dimension: {}'.format(dim))        
        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]
    print("init_state_lb:",init_state_lb.shape)

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

    output_space = plant.C
    expand_mat = np.zeros((1,1))
    output_space = np.hstack((output_space,expand_mat))
    initial_space = X0_probstar.V 


    datas = []

    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()

    A1 = np.array([-1])
    b1 = np.array([-10.75])
    P = AtomicPredicate(A1,b1)

    P1 = AtomicPredicate(-A1,-b1)
    
    for hi in h:
        N = int (time_bound/ hi)
        reach_start_time = time.time()

        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)

        reach_time_duration = time.time() - reach_start_time

        EVOT =_EVENTUALLY_(0,N)
        EVOT1 =_EVENTUALLY_(0,20)
        AWOT=_ALWAYS_(0,N)
        AWOT1=_ALWAYS_(10,20)

        spec = Formula([EVOT,P]) # be unsafe at some step 0 to N
        spec1 = Formula([AWOT,P1]) # alwyas safe between 0 to N
        spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P,rb,rb])
        specs=[spec,spec1,spec2]

        for spec_id in range(0,len(specs)):
            check_start = time.time()
            spec = specs[spec_id]
            print("\\===================Specification{}: =====================".format(spec_id))
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
                    # 'Nadnf': Nadnf,
                    # 'Ncdnf': Ncdnf,
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
    print('Quantitative Verification of ISS Model Using Krylov Subspace')
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

            print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
            _,p_max, p_min,Ncdnf=DNF_spec.evaluate(R)
            end = time.time()
            checking_time = end -check_start 
            print("p_min:",p_min)
            print("p_max:",p_max)
            verify_time = checking_time + reach_time_duration    

            # data.append(["ISS",hi,spec_id,Nadnf,Ncdnf,p_min,p_max, reach_time_duration,checking_time,verify_time])
            data = {
                    'Model': 'MCS',
                    'Spec': spec_id,
                    # 'Nadnf': Nadnf,
                    # 'Ncdnf': Ncdnf,
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
    print('Quantitative Verification of Beam Model Using Krylov Subspace')
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
                    # 'Nadnf': Nadnf,
                    # 'Ncdnf': Ncdnf,
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
    print('Quantitative Verification of MNA1 Model Using Krylov Subspace')
    print('=====================================================')
        #--------------------krylov method -------------------------
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
    
    datas = []
    sepcs=[]

    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()

    A1 = np.array([-1])
    b1 = np.array([-0.2])
    P = AtomicPredicate(A1,b1)

    P1 = AtomicPredicate(-A1,-b1)

    
    for hi in h:
        N = int (time_bound/ hi)
        reach_start_time = time.time()

        R, krylov_time = sim3(combined_mat,X0_probstar,hi, N, m,samples,tolerance,target_error=target_error,initial_space=initial_space,output_space=output_space,use_arnoldi = use_arnoldi,use_init_space=use_init_space)
       
        print("R_length:",len(R))
        reach_time_duration = time.time() - reach_start_time

        EVOT =_EVENTUALLY_(0,N)
        EVOT1 =_EVENTUALLY_(150,195)
        AWOT=_ALWAYS_(0,N)
        AWOT1=_ALWAYS_(150,195)

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
            _,p_max, p_min, Ncdnf= DNF_spec.evaluate(R)
            end = time.time()
            checking_time = end -check_start 
            print("p_min:",p_min)
            print("p_max:",p_max)
            verify_time = checking_time + reach_time_duration    

            data = {
                    'Model': 'MNA1',
                    'Spec': spec_id,
                    # 'Nadnf': Nadnf,
                    # 'Ncdnf': Ncdnf,
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
    print('Quantitative Verification of FOM Model Using Krylov Subspace')
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

    datas = []

    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()

    A1 = np.array([-1])
    b1 = np.array([-7])
    P = AtomicPredicate(A1,b1)

    P1 = AtomicPredicate(-A1,-b1)
    
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
                    # 'Nadnf': Nadnf,
                    # 'Ncdnf': Ncdnf,
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
    print('Quantitative Verification of MNA5 Model Using Krylov Subspace')
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
    print("init_state_bounds_list_shape:",len(init_state_bounds_list))

    # create Star for initial state 
    X0 = Star(init_state_lb,init_state_ub)


    # create ProbStar for initial state 
    mu_U = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3
    sig_U = (mu_U - X0.pred_lb)/a
    epsilon = 1e-10
    sig_U = np.maximum(sig_U, epsilon)
    Sig_U = np.diag(np.square(sig_U))
    # U_probstar = ProbStar(U.V, U.C, U.d,mu_U, Sig_U,U.pred_lb,U.pred_ub)
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_U, Sig_U,X0.pred_lb,X0.pred_ub)



    h = [0.1]
    time_bound = 20
    m = 10
    target_error = 1e-6
    samples = 51
    tolerance = 1e-9


    output_space= random_two_dims_mapping(X0_probstar,1,2)
    initial_space = X0_probstar.V 

    datas = []
    sepcs=[]

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
        # spec2 = Formula([AWOT1,lb, P,AND,lb,EVOT1,P1,rb,rb])
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

            # data.append(["MNA5",hi,spec_id,Nadnf,Ncdnf,p_min,p_max, reach_time_duration,checking_time,verify_time])
    
            data = {
                    'Model': 'MNA5',
                    'Spec': spec_id,
                    # 'Nadnf': Nadnf,
                    # 'Ncdnf': Ncdnf,
                    'p_min':p_min,
                    'p_max':p_max,
                    't_r':reach_time_duration,
                    't_c':checking_time,
                    't_v': verify_time
                }
            datas.append(data)

    return datas

def heat3D_create_dia(samples, diffusity_const, heat_exchange_const):
    '''fast dense matrix construction for heat3d dynamics'''

    samples_sq = samples**2
    dims = samples**3
    step = 1.0 / (samples + 1)

    a = diffusity_const * 1.0 / step**2
    print("a:",a)
    d = -6.0 * a  # Since we have six neighbors in 3D
    print("d:",d)

    # Initialize dense matrix
    matrix = np.zeros((dims, dims))

    for i in range(dims):
        z = i // samples_sq
        # print("z:",z)
        y = (i % samples_sq) // samples
        # print("y:",y)
        x = i % samples
        # print("x:",x)

        if z > 0:
            matrix[i, i - samples_sq] = a  # Interaction with the point below
        if y > 0:
            matrix[i, i - samples] = a     # Interaction with the point in front
        if x > 0:
            matrix[i, i - 1] = a           # Interaction with the point to the left

        matrix[i, i] = d

        if z == 0 or z == samples - 1:
            matrix[i, i] += a  # Boundary adjustment for z-axis
        if y == 0 or y == samples - 1:
            matrix[i, i] += a  # Boundary adjustment for y-axis
        if x == 0:
            matrix[i, i] += a  # Boundary adjustment for x=0
        if x == samples - 1:
            matrix[i, i] += a / (1 + heat_exchange_const * step)  # Boundary adjustment for x=samples-1

        if x < samples - 1:
            matrix[i, i + 1] = a           # Interaction with the point to the right
        if y < samples - 1:
            matrix[i, i + samples] = a     # Interaction with the point behind
        if z < samples - 1:
            matrix[i, i + samples_sq] = a  # Interaction with the point above

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
    

    datas = []

    AND = _AND_()
    OR = _OR_()                                        
    lb = _LeftBracket_()
    rb = _RightBracket_()

    A1 = np.array([-1])
    b1 = np.array([-0.012])
    P = AtomicPredicate(A1,b1)

    P1 = AtomicPredicate(-A1,-b1)
    
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
                    # 'Nadnf': Nadnf,
                    # 'Ncdnf': Ncdnf,
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


    results.extend(run_mcs_model(use_arnoldi=True, use_init_space=False))
    results.extend(run_building_model(use_arnoldi=True, use_init_space=False))
    results.extend(run_pde_model(use_arnoldi=True, use_init_space=False))
    results.extend(run_iss_model(use_arnoldi=True, use_init_space=False))
    results.extend(run_beam_model(use_arnoldi=True, use_init_space=False))
    results.extend(run_MNA1_model(use_arnoldi=True, use_init_space=False))
    results.extend(run_fom_model(use_arnoldi=True, use_init_space=False))
    results.extend(run_MNA5_model(use_arnoldi=True, use_init_space=False))

    # Heat#d model using Lanczos iteration due to symmetric dynamics
    results.extend(run_heat3D_model(use_arnoldi=False, use_init_space=False))


    print(tabulate([[r['Model'], r['Spec'], r['p_min'], r['p_max'], 
                    r['t_r'], r['t_c'], r['t_v']] for r in results],
                  headers=["Model", "Spec", "Nadnf", "Ncdnf", "p_min", "p_max",
                          "t_r", "t_c", "t_v"]))

    # Save results to pickle file
    path = "results"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/full_results.pkl', 'wb') as f:
        pickle.dump(results, f)

def verification_Hylaa_tool():
    """
    Verify all benchamrks with using Hylaa verification tool
    (NNV, Marabou, NNenum)
    """
    data = [
    {"Model": 'MCS', "Spec": 0, "SAT": 'YES', "t_v": 0.005317},
    {"Model": 'Building', "Spec": 0, "SAT": 'NO', "t_v": 0.160578}, 
    {"Model": 'PDE', "Spec": 0, "SAT": 'YES', "t_v": 0.371748},
    {"Model": 'ISS', "Spec": 0, "SAT": 'YES', "t_v": 3.292322},
    {"Model": 'BEAM', "Spec": 0, "SAT": 'YES', "t_v": 4.475225},
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
    path = "results"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/Hylaa_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def generate_table_3_vs_Hylaa_tool():
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
        
    path = "results"
    # Load all data sources
    probstarTL_data = load_pickle_file(path + '/full_results.pkl')

    Hylaa_data = load_pickle_file(path + '/Hylaa_results.pkl')


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

        # Add Monte Carlo and Other Tools data only if p_f = 0
        if spec==0:
            Hylaa_entry = Hylaa_dict.get((model, spec), {})
            
            row.extend([
                format_number(Hylaa_entry.get('SAT', '')),
                format_number(Hylaa_entry.get('t_v', ''))

            ])
        else:
            # Add empty cells for Monte Carlo and Other Tools columns
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

    print("Table has been generated and saved to 'Table_3_vs_Hylaa.tex'")
    return table_content

if __name__ == "__main__":
    # Figure 2: L15 and G15 verification results 
    verify_MNIST_LSTM_GRU(type='gru', hidden=15)
    plot_rnns_results(type='gru', hidden=15)
    verify_MNIST_LSTM_GRU(type='lstm', hidden=15)
    plot_rnns_results(type='lstm', hidden=15)
    
    
    # Table 2: Verification results of the MNIST CNN
    verify_convnet_network(net_type='Small', dtype='float64')
    verify_convnet_network(net_type='Medium', dtype='float64')
    verify_convnet_network(net_type='Large', dtype='float64')
    plot_table_covnet_network_all()
    
    
    # Table 3: Verification results of VGG16
    verify_vgg16_network_spec_cn()
    plot_table_vgg16_network()
    
    
    # Figure 3: Memory usage and computation time comparison between ImageStar and
    # SparseImageStar (SIM) in verifying the vggnet16 network (vnncomp2023) with spec 11 image
    memory_usage_vgg16(spec=11)  
    
    
    # Table 4: Combined AcasXu ReLU networks, ProbStar vs MC vs other tools
    x = [1, 1, 1]
    y = [6, 7, 9] 
    s = [2, 3, 4] # property id
    quantiverify_ACASXU_ReLU_table_3(x=x, y=y, spec_ids=s, numCores=16, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5]) # AcasXu ReLU networks, ProbStar
    numSamplesList = [10000000]
    quantiverify_ACASXU_ReLU_MC_table_3(x=x, y=y, spec_ids=s, unsafe_mat=None, unsafe_vec=None, numSamples=numSamplesList, nTimes=10, numCore=16) # AcasXu ReLU networks, Monte Carlo
    qualiverify_ACASXU_ReLU_other_tools_table_3() # AcasXu ReLU networks, other tools
    generate_table_3_AcasXu_ReLU_quanti_verify_vs_other_tools() 
    
    
    # Table 5: Verification results (robustness intervals) of NeuroSymbolic  
    verify_temporal_specs_ACC_trapeziu_full()


    # Table 6: Verification results of all models for Quantitative verification of Massive linear system
    full_evaluation_results()
    verification_Hylaa_tool()
    generate_table_3_vs_Hylaa_tool()
