"""
Verify the RNN bechmark
Author: Bryan Duong
Date: 11/20/2024
"""

import os
import time
import numpy as np
from StarV.util.load_rnn import load_rnn, load_data_points
from tabulate import tabulate
from matplotlib import pyplot as plt
from StarV.set.probstar import ProbStar
from StarV.util.plot import plot_probstar
from StarV.net.network import NeuralNetwork
from StarV.layer.RecurrentLayer import RecurrentLayer
from StarV.verifier.verifier import quantiVerifyBFS


def quantiverify_RecurrentNN(numCores):
    """Verify all Recurent Neural Network ReLU"""

    print("=====================================================")
    print("Quantitative Verification of Recurrent Neural Network")
    print("=====================================================")

    # net = load_rnn(2, 0)
    net = load_rnn(2, 2)
    # net = load_rnn(4, 0)
    # net = load_rnn(4, 2)
    # net = load_rnn(4, 4)
    # net = load_rnn(8, 0)

    point_idx = 0
    t_step = 2
    inputSet = load_data_points(index=point_idx, M=t_step)

    unsafe_mat = np.array([[1.0, 0.0]])
    unsafe_vec = np.array([3.0])
    # unsafe_mat = np.array([[1.0, 1.0]])
    # unsafe_vec = np.array([2])
    inputProb = inputSet[0].estimateProbability()

    data = []
    p_filter = 0.0

    # start verification
    start = time.time()
    OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(
        net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, unsafe_vec=unsafe_vec, numCores=numCores, p_filter=p_filter
    )
    end = time.time()
    verifyTime = end - start

    data.append([p_filter, len(OutputSet[t_step - 1]), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputProb, verifyTime])

    print(
        tabulate(
            data,
            headers=[
                "p_filter",
                "OutputSet",
                "UnsafeOutputSet",
                "CounterInputSet",
                "UnsafeProb-LB",
                "UnsafeProb-UB",
                "UnsafeProb-Min",
                "UnsafeProb-Max",
                "inputSet Probability",
                "VerificationTime",
            ],
        )
    )

    # save verification results
    path = "artifacts/JOURNAL2025/RecurrentNet"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/RecurrentNetTable.tex", "w") as f:
        print(
            tabulate(
                data,
                headers=[
                    "p_filter",
                    "OutputSet",
                    "UnsafeOutputSet",
                    "CounterInputSet",
                    "UnsafeProb-LB",
                    "UnsafeProb-UB",
                    "UnsafeProb-Min",
                    "UnsafeProb-Max",
                    "inputSet Probability",
                    "VerificationTime",
                ],
                tablefmt="latex",
            ),
            file=f,
        )

    # plot reachable sets and unsafe reachable sets

    dir_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    plot_probstar(OutputSet[t_step - 1], dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
    plt.savefig(path + "/OutputSet.png", bbox_inches="tight")  # save figure
    plt.show()
    plot_probstar(unsafeOutputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
    plt.savefig(path + "/UnsafeOutputSet.png", bbox_inches="tight")  # save figure
    plt.show()
    print("=====================================================")
    print("DONE!")
    print("=====================================================")


if __name__ == "__main__":

    quantiverify_RecurrentNN(numCores=1)
