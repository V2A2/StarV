"""
Test different options of multithreading computation
1. standard python multithreading
2. IPython
3. Ray

References: Anyscale-Parallelizing Python Code

Dung Tran, 8/30/2022
"""

import math
import numpy as np
from multiprocessing import Pool
# import ipyparallel as ipp
# import subprocess
import time
from timebudget import timebudget
timebudget.set_quiet()  # don't show measurements as they happen
# timebudget.report_at_exit()  # Generate report when the program exits
# refs: https://pypi.org/project/timebudget/

iterations_count = round(1e7)


def complex_operation(input_index):
    print("Complex operation. Input index: {:2d}".format(input_index))

    [math.exp(i) * math.sinh(i) for i in [1] * iterations_count]


def complex_operation_numpy(input_index):
    print("Complex operation (numpy). \
    Input index: {:2d}".format(input_index))

    data = np.ones(iterations_count)
    np.exp(data) * np.sinh(data)


@timebudget
def run_complex_operations(operation, input, pool):
    pool.map(operation, input)


@timebudget
def run_complex_operations_numpy(operation, input, pool):
    pool.map(operation, input)


@timebudget
def run_complex_operations_ipp(operation, input, pool):
    pool.map(operation, input)


@timebudget
def run_complex_operations_numpy_ipp(operation, input, pool):
    pool.map(operation, input)


def unwrap_self_f(arg, **kwarg):
    return C.f(*arg, **kwarg)


class C:
    def f(self, name):
        print('hello %s,' % name)
        time.sleep(5)
        print('nice to meet you')

    def run(self):
        pool = Pool(processes=2)
        names = ('frank', 'justin', 'osi', 'thomas')
        pool.map(unwrap_self_f, zip([self]*len(names), names))

# processes_count = 10
# input = range(10)
# subprocess.Popen(["ipcluster", "start", "-n={:d}".format(processes_count)])
# client_ids = ipp.Client()
# pool1 = client_ids[:]


if __name__ == '__main__':
    # processes_pool = Pool(processes_count)
    # print('\nWithout Numpy - Python multiprocessing')
    # run_complex_operations(complex_operation, input, processes_pool)

    # print('\nNumpy - Python multiprocessing')
    # run_complex_operations_numpy(complex_operation_numpy,
    #                             input, processes_pool)

    # print('\nWithout Numpy- IPython multiprocessing')
    # run_complex_operations_ipp(complex_operation, input, pool1)

    # print('\nNumpy - IPython multiprocessing')
    # run_complex_operations_numpy_ipp(complex_operation_numpy, input, pool1)

    # timebudget.report(reset=True)
    c = C()
    c.run()
