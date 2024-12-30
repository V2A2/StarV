import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from StarV.set.star import Star
from StarV.util import load_fairness
from StarV.verifier import verifier_fairness
import os
import numpy as np

import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager for timeout"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

@dataclass
class ModelResults:
    model_name: str
    num_outputs_set1: int
    num_outputs_set2: int
    num_counter_examples: int
    reach_time_set1: float
    reach_time_set2: float
    verification_time: float
    total_time: float
    timeout: bool = False  # New field for timeout status


def run_all_models() -> Dict[str, List[ModelResults]]:
    # Model IDs for each category
    adult_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    bm_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    GC_ids = [1, 2, 3, 4, 5]


    # 1-35
    # adult_ids = [8, 9, 10]
    # bm_ids = [6]
    # GC_ids = [3, 4]


    # 36-65
    # adult_ids = [1, 6, 11, 12]
    # bm_ids = [2, 4]
    # GC_ids = [1]

    # # 66-150
    # adult_ids = [2, 3, 5, 7]
    # bm_ids = [1, 3, 7, 8]
    # GC_ids = [2, 5]


    # # 151-320
    # adult_ids = [4]
    # bm_ids = [5]
    # GC_ids = []
    
    
    results = {
        'AC': [],  # Adult Credit
        'BM': [],  # Bank Marketing
        'GC': []   # German Credit Marketing
    }

    model_configs = [
        ('AC', adult_ids, load_fairness.load_fairness_adult),
        ('BM', bm_ids, load_fairness.load_fairness_bank),
        ('GC', GC_ids, load_fairness.load_fairness_german)
    ]
    
    for category, ids, load_func in model_configs:
        for model_id in ids:
            print(f"\nRunning {category} model {model_id}...")
            try:
                with time_limit(3600):  # 1 hour timeout
                    net = load_func(model_id)
                    lb1, ub1 = verifier_fairness.parse_inputs_strict(category)
                    inputSet1 = Star(lb1, ub1)
                    inputSet2 = verifier_fairness.parse_protected_attr(category, 1, inputSet1)
                    
                    model_result = run_single_model(f"{category}-{model_id}", net, inputSet1, inputSet2)
                    results[category].append(model_result)
            except TimeoutException:
                print(f"Timeout occurred for {category} model {model_id}")
                results[category].append(ModelResults(
                    model_name=f"{category}-{model_id}",
                    num_outputs_set1=-1,
                    num_outputs_set2=-1,
                    num_counter_examples=-1,
                    reach_time_set1=-1,
                    reach_time_set2=-1,
                    verification_time=-1,
                    total_time=-1,
                    timeout=True
                ))
            except Exception as e:
                print(f"Error occurred for {category} model {model_id}: {str(e)}")
                results[category].append(ModelResults(
                    model_name=f"{category}-{model_id}",
                    num_outputs_set1=-1,
                    num_outputs_set2=-1,
                    num_counter_examples=-1,
                    reach_time_set1=-1,
                    reach_time_set2=-1,
                    verification_time=-1,
                    total_time=-1,
                    timeout=True
                ))
    
    return results

def run_single_model(model_name: str, net, inputSet1, inputSet2) -> ModelResults:
    try:
        results = verifier_fairness.fairness_verification_exact_strict_both(net, inputSet1, inputSet2, numCores=4)
        timing = results['timing']
        
        return ModelResults(
            model_name=model_name,
            num_outputs_set1=len(results['pos_first']['S1']),
            num_outputs_set2=len(results['pos_first']['S2']),
            num_counter_examples=len(results['pos_first']['C1']) + len(results['neg_first']['C1']),
            reach_time_set1=timing['reach_S1_time'],
            reach_time_set2=timing['reach_S2_time'],
            verification_time=(timing['pos_first']['total'] + timing['neg_first']['total']),
            total_time=timing['total_time'],
            timeout=False
        )
    except Exception as e:
        print(f"Error in run_single_model for {model_name}: {str(e)}")
        raise

def generate_latex_table(results: Dict[str, List[ModelResults]]) -> str:
    latex = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\begin{tabular}{|l|r|r|r|r|r|r|r|}",
        "\\hline",
        "Model & \\#Out$_1$ & \\#Out$_2$ & \\#Counter & Reach$_1$ (s) & Reach$_2$ (s) & Verify (s) & Total (s) \\\\",
        "\\hline",
    ]
    
    for category in ['AC', 'BM', 'GC']:
        latex.append(f"\\multicolumn{{8}}{{|c|}}{{\\textbf{{{category} Models}}}} \\\\")
        latex.append("\\hline")
        
        for result in results[category]:
            if result.timeout:
                latex.append(
                    f"{result.model_name} & - & - & - & - & - & - & TO \\\\"
                )
            else:
                latex.append(
                    f"{result.model_name} & "
                    f"{result.num_outputs_set1} & "
                    f"{result.num_outputs_set2} & "
                    f"{result.num_counter_examples} & "
                    f"{result.reach_time_set1:.2f} & "
                    f"{result.reach_time_set2:.2f} & "
                    f"{result.verification_time:.2f} & "
                    f"{result.total_time:.2f} \\\\"
                )
        latex.append("\\hline")
    
    latex.extend([
        "\\end{tabular}",
        "\\caption{Strict fairness results using exact approach. TO indicates timeout (>1h).}",
        "\\label{tab:strict_fair_exact}",
        "\\end{table*}"
    ])
    
    return "\n".join(latex)

def save_results(results: Dict[str, List[ModelResults]], filename: str = "fairness_results"):
    path = "artifacts/HSCC2024/exact/strict/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    latex_file = os.path.join(path, f"{filename}.tex")
    csv_file = os.path.join(path, f"{filename}.csv")
    
    with open(latex_file, "w") as f:
        f.write(generate_latex_table(results))
    
    with open(csv_file, "w") as f:
        f.write("Category,Model,#Outputs1,#Outputs2,#Counter,ReachTime1,ReachTime2,VerifyTime,TotalTime,Timeout\n")
        for category, category_results in results.items():
            for result in category_results:
                if result.timeout:
                    f.write(f"{category},{result.model_name},-,-,-,-,-,-,TO,Yes\n")
                else:
                    f.write(f"{category},{result.model_name},{result.num_outputs_set1},"
                           f"{result.num_outputs_set2},{result.num_counter_examples},"
                           f"{result.reach_time_set1:.2f},{result.reach_time_set2:.2f},"
                           f"{result.verification_time:.2f},{result.total_time:.2f},No\n")
    
    print(f"Results saved to:\n  LaTeX: {latex_file}\n  CSV: {csv_file}")

if __name__ == "__main__":
    results = run_all_models()
    save_results(results, "exact_strict_results")
    
    for category, category_results in results.items():
        print(f"\n{category} Models Summary:")
        for result in category_results:
            print(f"\n{result.model_name}:")
            if result.timeout:
                print("  TIMEOUT (>1h)")
            else:
                print(f"  Output sets: {result.num_outputs_set1} (Set1), {result.num_outputs_set2} (Set2)")
                print(f"  Counter examples: {result.num_counter_examples}")
                print(f"  Total time: {result.total_time:.2f} seconds")