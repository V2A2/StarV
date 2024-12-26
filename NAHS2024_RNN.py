"""
NAHS2024: Reachability Analysis of Recurrent Neural Networks (Vaniall, LSTM, GRU RNNs)
Evaluation

Author: Sung Woo Choi
Date: 12/25/2024
"""

import os
import csv
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from StarV.util.load import load_neural_network_file, load_GRU_network, load_LSTM_network
from StarV.verifier.certifier import certifyRobustness_sequence

def verify_MNIST_LSTM_GRU(type='lstm', hidden=15):
    data_dir = 'StarV/util/data/nets/NAHS2024_RNN'
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
    path = f"artifacts/NAHS2024_RNN/results"
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

    f_dir = f"artifacts/NAHS2024_RNN/results/"
    with open(f_dir + f'mnist_{type}_{hidden}_results.npy', 'rb') as f:
        rb_est = np.load(f)
        rb_lp = np.load(f)
        rb_dr1 = np.load(f)
        rb_dr2 = np.load(f)
        
        vt_est = np.load(f)
        vt_lp = np.load(f)
        vt_dr1 = np.load(f)
        vt_dr2 = np.load(f)

    with open(f_dir + f'auto_lirpa_results/auto_lirpa_{type}_{hidden}_results.pkl', 'rb') as f:
        rb_autolirpa, vt_autolirpa, _ = pickle.load(f)

    if type == 'lstm':
        with open(f_dir + f'popqorn_results/popqorn_{type}_{hidden}_results.pkl', 'rb') as f:
            rb_popqorn, vt_popqorn, _ = pickle.load(f)

    if hidden == 15:
        eps = [0.005, 0.01, 0.015, 0.02, 0.025]

        rb_loc = 'upper right'
        vt_loc = 'center right'

        if type == 'lstm':
            fig_num = 9
        else:
            fig_num = 8
    else:
        eps = [0.0025, 0.005, 0.0075, 0.01, 0.0125]
        rb_loc = 'lower left'
        vt_loc = 'center right'

        if type == 'lstm':
            fig_num = 11
        else:
            fig_num = 10


    ft = 13 #font_size
    cm = 1/2.54  # centimeters in inches
    lw = 2.5

    legend = ['EST', 'LP', 'DR=1', 'DR=2', 'AutoLirpa']
    if type == 'lstm':
        legend.append('$POPQORN^0$')

    fig, ax = plt.subplots(figsize=(11*cm, 15*cm), tight_layout=True)
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
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}')) # 2 decimal places
    plt.savefig(f_dir + f'Figure_{fig_num}__MNIST_{type.upper()}_{hidden}_net_RB.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(9*cm, 15*cm), tight_layout=True) # 10 15
    plt.title(f"MNIST {type.upper()}{hidden} VT")

    plt.plot(eps, vt_est, "--gD", linewidth=lw)
    plt.plot(eps, vt_lp, "-rs", linewidth=lw)
    plt.plot(eps, vt_dr1, "--co", linewidth=lw)
    plt.plot(eps, vt_dr2, "--bv", linewidth=lw)
    plt.plot(eps, vt_autolirpa, "--y*", linewidth=lw)
    if type == 'lstm':
        plt.plot(eps, vt_popqorn, "--mx", linewidth=lw)
    plt.legend(legend, fontsize=ft, loc=vt_loc)

    plt.xticks(fontsize=ft)
    plt.yticks(fontsize=ft)
    plt.xlabel("$\epsilon$", fontsize=ft)
    plt.ylabel("Avg. Verification Time (s)", fontsize=ft)
    ax.set_xticks(eps)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}')) # 2 decimal places
    plt.savefig(f_dir + f'Figure_{fig_num}__MNIST_{type.upper()}_{hidden}_net_VT.png')
    plt.show()

if __name__ == "__main__":

    # Figure 8: G15 certification results with different attack bounds \epsilon.
    verify_MNIST_LSTM_GRU(type='gru', hidden=15)
    plot_rnns_results(type='gru', hidden=15)

    # Figure 9: L15 certification results with different attack bounds \epsilon.
    verify_MNIST_LSTM_GRU(type='lstm', hidden=15)
    plot_rnns_results(type='lstm', hidden=15)

    # Figure 10: G30 certification results with different attack bounds \epsilon.
    verify_MNIST_LSTM_GRU(type='gru', hidden=30)
    plot_rnns_results(type='gru', hidden=30)

    # Figure 11: L30 certification results with different attack bounds \epsilon.
    verify_MNIST_LSTM_GRU(type='lstm', hidden=30)
    plot_rnns_results(type='lstm', hidden=30)