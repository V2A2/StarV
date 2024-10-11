"""
Create Table for VGG16 vnncomp2023 results
Author: Sung Woo Choi
Date: 10/09/2024
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle

import os
import sys
sys.path.append(os.path.abspath(os.curdir))

def plot_table_vgg16_network():
    folder_dir = 'SparseImageStar_evaluation/results/'
    file_dir = folder_dir + 'vggnet16_vnncomp23_results.pkl'
    with open(file_dir, 'rb') as f:
        rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, rb_table, vt_table = pickle.load(f)

    N = 15
    vt_NNENUM = [3.5, 3.4, 9.3, 4.8, 18.1, 35.7, 6.5, 18.3, 133.85, 10.6, 40.9, 57.6, 'T/O', 236.52, 746.60]

    headers = ['Specs', 'Result', 'IM', 'SIM_csr', 'SIM_coo', 'NNENUM']
    result = 'UNSAT'
    
    data = []
    for i in range(N):
        vt_im = 'O/M' if np.isnan(vtIM[i]) else f"{vtIM[i]:0.1f}"
        data.append([i, result, vt_im, f"{vtCSR[i]:0.1f}", f"{vtCOO[i]:0.1f}", vt_NNENUM[i]])
    print(tabulate(data, headers=headers))

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(folder_dir+f"vggnet16_vnncomp23_results_full_table.tex", "w") as f:
        print(Tlatex, file=f)

if __name__ == "__main__":
    plot_table_vgg16_network()