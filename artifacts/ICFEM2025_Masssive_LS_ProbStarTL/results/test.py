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

##################################
cur_path = os.path.dirname(__file__)
print("cur_path:",cur_path)
probstarTL_data = load_pickle_file(cur_path + '/full_results.pkl')

# print(probstarTL_data[0])

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
cur_path = os.path.dirname(__file__)
with open(cur_path + '/Table_4.tex','w') as f:
    f.write("\n".join(table_lines))
print("Generated Table_4_The_length_of_ADNF-and _CDNF.tex")


