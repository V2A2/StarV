"""
Create Table for MNIST ConvNet (CAV2020) against brightening attack results
Author: Sung Woo Choi
Date: 06/24/2024
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle

import os
import sys
sys.path.append(os.path.abspath(os.curdir))

def plot_table_covnet_network(net_type):
    assert net_type in ['Small', 'Medium', 'Large'], \
    f"There are 3 types of ConvNet networks: /'Small/', /'Medium/', and /'Large/'"

    folder_dir = 'SparseImageStar_evaluation/results/'
    mat_file = scipy.io.loadmat(folder_dir + f"NNV_{net_type}_ConvNet_Results_brightAttack.mat")
    rbNNV = mat_file['r_star']
    vtNNV = mat_file['VT_star']
    
    delta = [0.005, 0.01, 0.015]
    d = [250, 245, 240]
    N = 100
    
    dir = folder_dir + f'{net_type}ConvNet_brightAttack'
    result_dir = dir + '_results.pkl'
    save_dir = dir + '_full_table.tex'

    print('result_dir: ', result_dir)
    with open(result_dir, 'rb') as f:
        [rbIM, vtIM, rbIM_table, vtIM_table, rbCSR, vtCSR, rbCSR_table, vtCSR_table, rbCOO, vtCOO, rbCOO_table, vtCOO_table] = pickle.load(f)

    file = open(save_dir, "w")
    L = [
    f"\\begin{{table}}[!]\n",
    f"\\scriptsize\n",
    f"\\centering\n",
    f"\\begin{{tabular}}{{c||c:c:c||c:c:c}}\n",
    f"      & \\multicolumn{{3}}{{c||}}{{Robustness results ($\%$)}} & \multicolumn{{3}}{{c}}{{Verification time (sec)}} \\\\\n",
    f"\\hline\n \n"
    ]
    file.writelines(L)
    
    for i in range(len(d)):
        file.write(f"\\hline\n")
        line = f"$d = {d[i]}$"
        for j in delta:
            line += f" & $\\delta = {j}$"
        for j in delta:
            line += f" & $\\delta = {j}$"
        line += f" \\\\\n \\hline\n"
        file.write(line)
        file.write(f'IM & {rbIM_table[i][1]} & {rbIM_table[i][2]} &  {rbIM_table[i][3]} & {vtIM_table[i][1] :.3f} & {vtIM_table[i][2] :.3f} &  {vtIM_table[i][3] :.3f} \\\\ \\hline\n')
        file.write(f'SIM\\_csr & {rbCSR_table[i][1]} & {rbCSR_table[i][2]} &  {rbCSR_table[i][3]} & {vtCSR_table[i][1] :.3f} & {vtCSR_table[i][2] :.3f} &  {vtCSR_table[i][3] :.3f} \\\\ \\hline\n')
        file.write(f'SIM\\_coo & {rbCOO_table[i][1]} & {rbCOO_table[i][2]} &  {rbCOO_table[i][3]} & {vtCOO_table[i][1] :.3f} & {vtCOO_table[i][2] :.3f}  &  {vtCOO_table[i][3] :.3f} \\\\ \\hline\n')
        file.write(f'NNV & {int(rbNNV[i, 0]*100)} & {int(rbNNV[i, 1]*100)} & {int(rbNNV[i, 2]*100)} & {vtNNV[i, 0]/100 :.3f} & {vtNNV[i, 1]/100 :.3f} & {vtNNV[i, 2]/100 :.3f} \\\\ \\hline\n')

        file.write(f"\\hline\n\n")

    L = [
    f"\\end{{tabular}}\n",
    f"\\caption{{Verification results of the {net_type} MNIST CNN (CAV2020).}}\n",
    f"\\label{{tab:CAV2020_convNet_{net_type}}}\n",
    f"\\end{{table}}",
    ]
    file.writelines(L)
    file.close()

if __name__ == "__main__":
    plot_table_covnet_network(net_type = 'Small')
    plot_table_covnet_network(net_type = 'Medium')
    plot_table_covnet_network(net_type = 'Large')
