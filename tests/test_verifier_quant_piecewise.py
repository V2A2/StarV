"""
Test verifier module for piecewise linear networks
Author: Yuntao Li
Date: 2/8/2024
"""

from StarV.net.network import NeuralNetwork
from StarV.verifier.verifier import reachExactBFS, checkSafetyStar, checkSafetyProbStar, quantiVerifyExactBFS, quantiVerifyBFS, quantiVerifyMC, quantiVerifyProbStarTL
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.LeakyReLULayer import LeakyReLULayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.SatLinsLayer import SatLinsLayer
from StarV.set.probstar import ProbStar
import numpy as np
import multiprocessing
from StarV.util.load_piecewise import load_2017_IEEE_TNNLS_ReLU, load_ACASXU_ReLU, load_tiny_network_ReLU
from StarV.util.load_piecewise import load_2017_IEEE_TNNLS_LeakyReLU, load_ACASXU_LeakyReLU, load_tiny_network_LeakyReLU
from StarV.util.load_piecewise import load_2017_IEEE_TNNLS_SatLin, load_ACASXU_SatLin, load_tiny_network_SatLin
from StarV.util.load_piecewise import load_2017_IEEE_TNNLS_SatLins, load_ACASXU_SatLins, load_tiny_network_SatLins
import time
from StarV.util.plot import plot_probstar_using_Polytope, plot_probstar
from matplotlib import pyplot as plt
from StarV.set.star import Star
from tabulate import tabulate
import os
from StarV.util.print_util import print_util

class Test(object):
    """
    Testing module net class and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0


    def test_quantiverify_tiny_network_ReLU(self, numCores):
        """verify a tiny ReLU network"""

        print_util('h2')
        print_util('h3')
        print('Quantitative Verification of Tiny ReLU Network')
        print_util('h3')

        self.n_tests = self.n_tests + 1
         
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_tiny_network_ReLU()
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            a = 2.5 # coefficience to adjust the distribution
            sig = (mu - S.pred_lb)/a
            print('Mean of predicate variables: mu = {}'.format(mu))
            print('Standard deviation of predicate variables: sig = {}'.format(sig))
            Sig = np.diag(np.square(sig))
            print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
            # Sig = 1e-2*np.eye(S.nVars)
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            inputProb = inputSet[0].estimateProbability()

            p_filters = [0.0, 0.1]
            data = []
            for p_filter in p_filters:
                start = time.time()
                OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                         unsafe_vec=unsafe_vec, numCores=numCores, p_filter=p_filter)
                end = time.time()
                verifyTime = end-start
                data.append([p_filter, len(OutputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputProb, verifyTime])

            # print verification results
            print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                      "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

            # save verification results
            path = "artifacts/Test/tinyNet/ReLU/"
            if not os.path.exists(path):
                os.makedirs(path)

            with open(path+"/tinyNetTable.tex", "w") as f:
                print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                              "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
        

            # plot reachable sets and unsafe reachable sets
            # OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
            #                                                                           unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
            
            # dir_mat = np.array([[1., 0.], [0., 1.]])
            # plot_probstar_using_Polytope(OutputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
            # plt.savefig(path+"/OutputSet_ReLU.png", bbox_inches='tight')  # save figure
            # # plt.show()
            # plot_probstar_using_Polytope(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True, show=False)
            # plt.savefig(path+"/UnsafeOutputSet_ReLU.png", bbox_inches='tight')  # save figure
            # # plt.show()
            # plot_probstar_using_Polytope(counterInputSet,dir_mat=dir_mat,dir_vec=None, label= ("$x_1$", "$x_2$"), show_prob=True, show=False)
            # plt.savefig(path+"/CounterInputSet_ReLU.png", bbox_inches='tight')  # save figure
            # # plt.show()

            print_util('h3')
            print('DONE!')
            print_util('h3')

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')


    def test_quantiverify_tiny_network_LeakyReLU(self, numCores):
        """verify a tiny LeakyReLU network"""

        print_util('h2')
        print_util('h3')
        print('Quantitative Verification of Tiny LeakyReLU Network')
        print_util('h3')

        self.n_tests = self.n_tests + 1
         
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_tiny_network_LeakyReLU()
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            a = 2.5 # coefficience to adjust the distribution
            sig = (mu - S.pred_lb)/a
            print('Mean of predicate variables: mu = {}'.format(mu))
            print('Standard deviation of predicate variables: sig = {}'.format(sig))
            Sig = np.diag(np.square(sig))
            print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
            # Sig = 1e-2*np.eye(S.nVars)
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            inputProb = inputSet[0].estimateProbability()

            p_filters = [0.0, 0.1]
            data = []
            for p_filter in p_filters:
                start = time.time()
                OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                         unsafe_vec=unsafe_vec, numCores=numCores, p_filter=p_filter)
                end = time.time()
                verifyTime = end-start
                data.append([p_filter, len(OutputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputProb, verifyTime])

            # print verification results
            print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                      "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

            # save verification results
            path = "artifacts/Test/tinyNet/LeakyReLU/"
            if not os.path.exists(path):
                os.makedirs(path)

            with open(path+"/tinyNetTable.tex", "w") as f:
                print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                              "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
        

            # plot reachable sets and unsafe reachable sets
            # OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
            #                                                                           unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
            # dir_mat = np.array([[1., 0.], [0., 1.]])
            # plot_probstar_using_Polytope(OutputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
            # plt.savefig(path+"/OutputSet_LeakyReLU.png", bbox_inches='tight')  # save figure
            # # plt.show()
            # plot_probstar_using_Polytope(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True, show=False)
            # plt.savefig(path+"/UnsafeOutputSet_LeakyReLU.png", bbox_inches='tight')  # save figure
            # # plt.show()
            # plot_probstar_using_Polytope(counterInputSet,dir_mat=dir_mat,dir_vec=None, label= ("$x_1$", "$x_2$"), show_prob=True, show=False)
            # plt.savefig(path+"/CounterInputSet_LeakyReLU.png", bbox_inches='tight')  # save figure
            # # plt.show()

            print_util('h3')
            print('DONE!')
            print_util('h3')

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')

    
    def test_quantiverify_tiny_network_SatLin(self, numCores):
        """verify a tiny SatLin network"""

        print_util('h2')
        print_util('h3')
        print('Quantitative Verification of Tiny SatLin Network')
        print_util('h3')

        self.n_tests = self.n_tests + 1
         
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_tiny_network_SatLin()
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            a = 2.5 # coefficience to adjust the distribution
            sig = (mu - S.pred_lb)/a
            print('Mean of predicate variables: mu = {}'.format(mu))
            print('Standard deviation of predicate variables: sig = {}'.format(sig))
            Sig = np.diag(np.square(sig))
            print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
            # Sig = 1e-2*np.eye(S.nVars)
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            inputProb = inputSet[0].estimateProbability()

            p_filters = [0.0, 0.1]
            data = []
            for p_filter in p_filters:
                start = time.time()
                OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                         unsafe_vec=unsafe_vec, numCores=numCores, p_filter=p_filter)
                end = time.time()
                verifyTime = end-start
                data.append([p_filter, len(OutputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputProb, verifyTime])

            # print verification results
            print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                      "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

            # save verification results
            path = "artifacts/Test/tinyNet/SatLin/"
            if not os.path.exists(path):
                os.makedirs(path)

            with open(path+"/tinyNetTable.tex", "w") as f:
                print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                              "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
        

            # plot reachable sets and unsafe reachable sets
            OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                                      unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
            dir_mat = np.array([[1., 0.], [0., 1.]])
            plot_probstar(OutputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
            plt.savefig(path+"/OutputSet_SatLin.png", bbox_inches='tight')  # save figure
            # plt.show()
            plot_probstar(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True, show=False)
            plt.savefig(path+"/UnsafeOutputSet_SatLin.png", bbox_inches='tight')  # save figure
            # plt.show()
            plot_probstar(counterInputSet,dir_mat=dir_mat,dir_vec=None, label= ("$x_1$", "$x_2$"), show_prob=True, show=False)
            plt.savefig(path+"/CounterInputSet_SatLin.png", bbox_inches='tight')  # save figure
            # plt.show()

            print_util('h3')
            print('DONE!')
            print_util('h3')

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')


    def test_quantiverify_tiny_network_SatLins(self, numCores):
        """verify a tiny SatLins network"""

        print_util('h2')
        print_util('h3')
        print('Quantitative Verification of Tiny SatLins Network')
        print_util('h3')

        self.n_tests = self.n_tests + 1
         
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_tiny_network_SatLins()
            net.info()
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            a = 2.5 # coefficience to adjust the distribution
            sig = (mu - S.pred_lb)/a
            print('Mean of predicate variables: mu = {}'.format(mu))
            print('Standard deviation of predicate variables: sig = {}'.format(sig))
            Sig = np.diag(np.square(sig))
            print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
            # Sig = 1e-2*np.eye(S.nVars)
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            inputProb = inputSet[0].estimateProbability()

            p_filters = [0.0, 0.1]
            data = []
            for p_filter in p_filters:
                start = time.time()
                OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                         unsafe_vec=unsafe_vec, numCores=numCores, p_filter=p_filter)
                end = time.time()
                verifyTime = end-start
                data.append([p_filter, len(OutputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputProb, verifyTime])

            # print verification results
            print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                      "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

            # save verification results
            path = "artifacts/Test/tinyNet/SatLins/"
            if not os.path.exists(path):
                os.makedirs(path)

            with open(path+"/tinyNetTable.tex", "w") as f:
                print(tabulate(data, headers=["p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                              "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
        

            # plot reachable sets and unsafe reachable sets
            OutputSet, unsafeOutputSet, counterInputSet, _, _, _, _ = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, \
                                                                                      unsafe_vec=unsafe_vec, numCores=numCores, p_filter=0.0)
            dir_mat = np.array([[1., 0.], [0., 1.]])
            plot_probstar_using_Polytope(OutputSet, dir_mat=dir_mat, dir_vec=None, show_prob=True, show=False)
            plt.savefig(path+"/OutputSet_SatLins.png", bbox_inches='tight')  # save figure
            plt.show()
            plot_probstar_using_Polytope(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True, show=False)
            plt.savefig(path+"/UnsafeOutputSet_SatLins.png", bbox_inches='tight')  # save figure
            plt.show()
            plot_probstar_using_Polytope(counterInputSet,dir_mat=dir_mat,dir_vec=None, label= ("$x_1$", "$x_2$"), show_prob=True, show=False)
            plt.savefig(path+"/CounterInputSet_SatLins.png", bbox_inches='tight')  # save figure
            plt.show()

            print_util('h3')
            print('DONE!')
            print_util('h3')

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')


    def test_quantiverify_ACASXU_all_ReLU(self, x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
        """Verify all ACASXU ReLU networks with spec_id"""

        print_util('h2')

        self.n_tests = self.n_tests + 1
        res = []
        data = []

        try:
            if len(x) != len(y):
                raise Exception('length(x) should equal length(y)')

            if len(x) != len(spec_ids):
                raise Exception('length(x) should equal length(spec_ids)')

            for i in range(len(x)):
                for p_filter in p_filters:
                    print_util('h3')
                    print('Test quanti verify of ACASXU N_{}_{} ReLU network under specification {}...'.format(x[i], y[i], spec_ids[i]))
                    net, lb, ub, unsmat, unsvec = load_ACASXU_ReLU(x[i], y[i], spec_ids[i])
                    net.info()
                    S = Star(lb, ub)
                    mu = 0.5*(S.pred_lb + S.pred_ub)
                    a  = 3.0 # coefficience to adjust the distribution
                    sig = (mu - S.pred_lb)/a
                    print('Mean of predicate variables: mu = {}'.format(mu))
                    print('Standard deviation of predicate variables: sig = {}'.format(sig))
                    Sig = np.diag(np.square(sig))
                    print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
                    In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
                    inputSet = []
                    inputSet.append(In)
                    inputSetProb = inputSet[0].estimateProbability()
                    netName = '{}-{}'.format(x[i], y[i])
                    start = time.time()
                    OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsmat, \
                                                                            unsafe_vec=unsvec, numCores=numCores, p_filter=p_filter)
                    end = time.time()
                    verifyTime = end-start
                    data.append([spec_ids[i], netName, p_filter, len(OutputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputSetProb, verifyTime])
                    print_util('h3')
            # print verification results
            print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                    "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

            # save verification results
            path = "artifacts/Test/ACASXU_ReLU"
            if not os.path.exists(path):
                os.makedirs(path)

            with open(path+"/ACASXUTable.tex", "w") as f:
                print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                    "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
            
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')

        return res, data


    def test_quantiverify_ACASXU_all_LeakyReLU(self, x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
        """Verify all ACASXU LeakyReLU networks with spec_id"""

        self.n_tests = self.n_tests + 1
        res = []
        data = []
        
        try: 
            if len(x) != len(y):
                raise Exception('length(x) should equal length(y)')

            if len(x) != len(spec_ids):
                raise Exception('length(x) should equal length(spec_ids)')

            for i in range(len(x)):
                for p_filter in p_filters:
                    print_util('h3')
                    print('Test quanti verify of ACASXU N_{}_{} LeakyReLU network under specification {}...'.format(x[i], y[i], spec_ids[i]))
                    net, lb, ub, unsmat, unsvec = load_ACASXU_LeakyReLU(x[i], y[i], spec_ids[i])
                    net.info()
                    S = Star(lb, ub)
                    mu = 0.5*(S.pred_lb + S.pred_ub)
                    a  = 3.0 # coefficience to adjust the distribution
                    sig = (mu - S.pred_lb)/a
                    print('Mean of predicate variables: mu = {}'.format(mu))
                    print('Standard deviation of predicate variables: sig = {}'.format(sig))
                    Sig = np.diag(np.square(sig))
                    print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
                    In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
                    inputSet = []
                    inputSet.append(In)
                    inputSetProb = inputSet[0].estimateProbability()
                    netName = '{}-{}'.format(x[i], y[i])
                    start = time.time()
                    OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsmat, \
                                                                            unsafe_vec=unsvec, numCores=numCores, p_filter=p_filter)
                    end = time.time()
                    verifyTime = end-start
                    data.append([spec_ids[i], netName, p_filter, len(OutputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputSetProb, verifyTime])
                    print_util('h3')
            # print verification results
            print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                    "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

            # save verification results
            path = "artifacts/Test/ACASXU_LeakyReLU"
            if not os.path.exists(path):
                os.makedirs(path)

            with open(path+"/ACASXUTable.tex", "w") as f:
                print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                    "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
            
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')

        return res, data
    
    
    def test_quantiverify_ACASXU_all_SatLin(self, x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
        """Verify all ACASXU SatLin networks with spec_id"""
        print_util('h2')

        self.n_tests = self.n_tests + 1
        res = []
        data = []

        try: 
            if len(x) != len(y):
                raise Exception('length(x) should equal length(y)')

            if len(x) != len(spec_ids):
                raise Exception('length(x) should equal length(spec_ids)')

            for i in range(len(x)):
                for p_filter in p_filters:
                    print_util('h3')
                    print('Test quanti verify of ACASXU N_{}_{} SatLin network under specification {}...'.format(x[i], y[i], spec_ids[i]))
                    net, lb, ub, unsmat, unsvec = load_ACASXU_SatLin(x[i], y[i], spec_ids[i])
                    net.info()
                    S = Star(lb, ub)
                    mu = 0.5*(S.pred_lb + S.pred_ub)
                    a  = 4.0 # coefficience to adjust the distribution
                    sig = (mu - S.pred_lb)/a
                    print('Mean of predicate variables: mu = {}'.format(mu))
                    print('Standard deviation of predicate variables: sig = {}'.format(sig))
                    Sig = np.diag(np.square(sig))
                    print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
                    In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
                    inputSet = []
                    inputSet.append(In)
                    inputSetProb = inputSet[0].estimateProbability()
                    netName = '{}-{}'.format(x[i], y[i])
                    start = time.time()
                    OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsmat, \
                                                                            unsafe_vec=unsvec, numCores=numCores, p_filter=p_filter)
                    end = time.time()
                    verifyTime = end-start
                    data.append([spec_ids[i], netName, p_filter, len(OutputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputSetProb, verifyTime])
                    print_util('h3')
            # print verification results
            print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                    "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

            # save verification results
            path = "artifacts/Test/ACASXU_SatLin"
            if not os.path.exists(path):
                os.makedirs(path)

            with open(path+"/ACASXUTable.tex", "w") as f:
                print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                    "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
            
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')

        return res, data
    
    
    def test_quantiverify_ACASXU_all_SatLins(self, x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
        """Verify all ACASXU SatLins networks with spec_id"""

        print_util('h2')

        self.n_tests = self.n_tests + 1
        res = []
        data = []
        
        try:    
            if len(x) != len(y):
                raise Exception('length(x) should equal length(y)')

            if len(x) != len(spec_ids):
                raise Exception('length(x) should equal length(spec_ids)')

            for i in range(len(x)):
                for p_filter in p_filters:
                    print_util('h3')
                    print('Test quanti verify of ACASXU N_{}_{} SatLins network under specification {}...'.format(x[i], y[i], spec_ids[i]))
                    net, lb, ub, unsmat, unsvec = load_ACASXU_SatLins(x[i], y[i], spec_ids[i])
                    net.info()
                    S = Star(lb, ub)
                    mu = 0.5*(S.pred_lb + S.pred_ub)
                    a  = 3.0 # coefficience to adjust the distribution
                    sig = (mu - S.pred_lb)/a
                    print('Mean of predicate variables: mu = {}'.format(mu))
                    print('Standard deviation of predicate variables: sig = {}'.format(sig))
                    Sig = np.diag(np.square(sig))
                    print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
                    In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
                    inputSet = []
                    inputSet.append(In)
                    inputSetProb = inputSet[0].estimateProbability()
                    netName = '{}-{}'.format(x[i], y[i])
                    start = time.time()
                    OutputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, inputSet=inputSet, unsafe_mat=unsmat, \
                                                                            unsafe_vec=unsvec, numCores=numCores, p_filter=p_filter)
                    end = time.time()
                    verifyTime = end-start
                    data.append([spec_ids[i], netName, p_filter, len(OutputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputSetProb, verifyTime])
                    print_util('h3')
            # print verification results
            print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                    "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

            # save verification results
            path = "artifacts/Test/ACASXU_SatLins"
            if not os.path.exists(path):
                os.makedirs(path)

            with open(path+"/ACASXUTable.tex", "w") as f:
                print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                    "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
                
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
        print_util('h2')

        return res, data

if __name__ == "__main__":

    test_verifier = Test()
    print_util('h1')
    
    # test_verifier.test_quantiverify_tiny_network_ReLU(8)
    # test_verifier.test_quantiverify_tiny_network_LeakyReLU(8)
    test_verifier.test_quantiverify_tiny_network_SatLin(8)
    # test_verifier.test_quantiverify_tiny_network_SatLins(8)

    x = [1]
    y = [2]
    s = [2]
    # x = [1, 1, 1]
    # y = [2, 3, 4] 
    # s = [2, 2, 2] # property id
 
    # test_verifier.test_quantiverify_ACASXU_all_ReLU(x=x, y=y, spec_ids=s, \
    #                                             numCores=8, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5])
    # test_verifier.test_quantiverify_ACASXU_all_ReLU(x=x, y=y, spec_ids=s, \
    #                                             numCores=8, unsafe_mat=None, unsafe_vec=None, p_filters=[1e-5])
    # test_verifier.test_quantiverify_ACASXU_all_LeakyReLU(x=x, y=y, spec_ids=s, \
    #                                             numCores=8, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5])
    # test_verifier.test_quantiverify_ACASXU_all_LeakyReLU(x=x, y=y, spec_ids=s, \
    #                                             numCores=8, unsafe_mat=None, unsafe_vec=None, p_filters=[1e-5])
    # test_verifier.test_quantiverify_ACASXU_all_SatLin(x=x, y=y, spec_ids=s, \
    #                                             numCores=8, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5])
    # test_verifier.test_quantiverify_ACASXU_all_SatLin(x=x, y=y, spec_ids=s, \
    #                                             numCores=8, unsafe_mat=None, unsafe_vec=None, p_filters=[1e-5])
    # test_verifier.test_quantiverify_ACASXU_all_SatLins(x=x, y=y, spec_ids=s, \
    #                                             numCores=8, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5])

    print_util('h1')
    print('Testing verifier: fails: {}, successfull: {}, \
    total tests: {}'.format(test_verifier.n_fails,
                            test_verifier.n_tests - test_verifier.n_fails,
                            test_verifier.n_tests))
