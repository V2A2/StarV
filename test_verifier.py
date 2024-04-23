"""
Test verifier module
Author: Dung Tran
Date: 8/10/2022
"""

from StarV.net.network import NeuralNetwork
from StarV.verifier.verifier import reachExactBFS, checkSafetyStar, quantiVerifyExactBFS, quantiVerifyBFS, quantiVerifyMC, quantiVerifyProbStarTL
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.set.probstar import ProbStar
import numpy as np
import multiprocessing
from StarV.util.load import load_2017_IEEE_TNNLS, load_ACASXU, load_DRL, load_tiny_network, load_harmonic_oscillator_model
import time
from StarV.util.plot import plot_probstar
from matplotlib import pyplot as plt
from StarV.set.star import Star
from tabulate import tabulate
from StarV.spec.dProbStarTL import Formula, AtomicPredicate, _ALWAYS_, _AND_
import os

class Test(object):
    """
    Testing module net class and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_reachExactBFS(self):

        self.n_tests = self.n_tests + 2
        W = np.eye(2)
        b = np.zeros(2,)
        L1 = fullyConnectedLayer(W, b)
        L2 = ReLULayer()
        layers = []
        layers.append(L1)
        layers.append(L2)
        net = NeuralNetwork(layers, 'ffnn')

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        mu = np.zeros(2,)
        Sig = np.eye(2)

        In = ProbStar(mu, Sig, lb, ub)

        print('Test reachExactBFS method ...')

        inputSet = []
        inputSet.append(In)

        
        try:
            print('Test without parallel computing...')
            S = reachExactBFS(net=net, inputSet=inputSet)
            print('Number of input sets: {}'.format(len(inputSet)))
            print('Number of output sets: {}'.format(len(S)))
            # for i in range(0, len(S)):
            #     S[i].__str__()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

        try:
            print('Test with parallel computing...')
            pool = multiprocessing.Pool(2)
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            print('Number of input sets: {}'.format(len(inputSet)))
            print('Number of output sets: {}'.format(len(S)))
            pool.close()
            # for i in range(0, len(S)):
            #     S[i].__str__()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_reach_2017_IEEE_TNNLS(self):
        """reachability analysis for 2017 IEEE TNNLS network"""

        self.n_tests = self.n_tests + 1

        print('Test exact reachability of 2017 IEEE TNNLS network...')
        
        try:
            lb = np.array([-1.0, -1.0, -1.0])
            ub = np.array([1.0, 1.0, 1.0])
            mu = np.zeros(3,)
            Sig = np.eye(3)
            In = ProbStar(mu, Sig, lb, ub)
            inputSet = []
            inputSet.append(In)
            net = load_2017_IEEE_TNNLS()
            net.info()
            pool = multiprocessing.Pool(8)
            start = time.time()
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            pool.close()
            print('Number of output sets: {}'.format(len(S)))
            end = time.time()
            print('Reachability time = {}'.format(end - start))
            plot_probstar(S)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_reach_ACASXU(self, x, y, spec_id):
        """Reachability analysis of ACASXU network"""

        self.n_tests = self.n_tests + 1

        print('Test probabilistic reachability of ACASXU N_{}_{} network under specification {}...'.format(x, y, spec_id))
        
        try:
            net, lb, ub, _, _ = load_ACASXU(x, y, spec_id)
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            Sig = 0.1*np.eye(S.nVars)
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            pool = multiprocessing.Pool(8)
            start = time.time()
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            pool.close()
            print('Number of output sets: {}'.format(len(S)))
            end = time.time()
            print('Reachability time = {}'.format(end - start))
        
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def  test_checkSafetyStar(self):

        self.n_tests = self.n_tests + 1
        print('Test intersectWithUnsafeRegion method...')
        
        
        try:
            lb = -np.random.rand(3,)
            ub = np.random.rand(3,)
            V = np.random.rand(3, 4)
            C = np.random.rand(3, 3)
            d = np.random.rand(3,)
            S = Star(V, C, d, lb, ub)
            unsafe_mat = np.random.rand(2, 3,)
            unsafe_vec = np.random.rand(2,)
            P = checkSafetyStar(unsafe_mat, unsafe_vec, S)
            S.__str__()
            if isinstance(P, Star):
                print('\nUnsafe Set')
                P.__str__()
            else:
                print('\nSafe')
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_quantiverify_ACASXU(self, x, y, spec_id, numCores):
        """Quantitative Verififcation of ACASXU network"""

        self.n_tests = self.n_tests + 1

        print('Test quantitative verification of ACASXU network: \
        N_{}_{} under specification {}...'.format(x, y, spec_id))

        # net, lb, ub, _, _ = load_ACASXU(x, y, spec_id)
        # S = Star(lb, ub)
        # mu = 0.5*(S.pred_lb + S.pred_ub)
        # Sig = 0.1*np.eye(S.nVars)
        # In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
        # inputSet = []
        # inputSet.append(In)
        # # unsafe constraints: Weak-Lelf (x[1]) <= 0.02175 
        # unsafe_mat = np.array([[0., 1., 0., 0., 0.]])
        # unsafe_vec = np.array([0.02175])
        # start = time.time()
        # OutputSet, unsafeOutputSet, prob = quantiVerifyExactBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, unsafe_vec=unsafe_vec, numCores=1)
        # print('Number of output sets: {}'.format(len(OutputSet)))
        # print('Number of unsafe output sets: {}'.format(len(unsafeOutputSet)))
        # print('Unsafe Probability: {}'.format(prob))
        # print('Input Set Probability: {}'.format(In.estimateProbability()))
        # In.__str__()
        # end = time.time()
        # print('Verification time = {}'.format(end - start))
        # dir_mat = np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]])
        # plot_probstar(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True)


        
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_ACASXU(x, y, spec_id)
            S = Star(lb, ub)
            mu = 0.5*(S.pred_lb + S.pred_ub)
            Sig = 1e-6*np.eye(S.nVars)
            In = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
            inputSet = []
            inputSet.append(In)
            # unsafe constraints: Weak-Lelf (x[1]) <= 0.02175 
            # unsafe_mat = np.array([[0., 1., 0., 0., 0.]])
            # unsafe_vec = np.array([0.02175])
            start = time.time()
            OutputSet, unsafeOutputSet, prob = quantiVerifyExactBFS(net=net, inputSet=inputSet, unsafe_mat=unsafe_mat, unsafe_vec=unsafe_vec, numCores=numCores)
            print('Number of output sets: {}'.format(len(OutputSet)))
            print('Number of unsafe output sets: {}'.format(len(unsafeOutputSet)))
            print('Unsafe Probability: {}'.format(prob))
            print('Input Set Probability: {}'.format(In.estimateProbability()))
            In.__str__()
            end = time.time()
            print('Verification time = {}'.format(end - start))
            dir_mat = np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]])
            plot_probstar(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True)

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
                

    def test_quantiverify_ACASXU_all(self, x, y, spec_ids, numCores, unsafe_mat, unsafe_vec, p_filters):
        """Verify all ACASXU networks with spec_id"""

        # example:
        # x = [3]
        # y = [7, 8]

        # unsafe constraints: Weak-Lelf (x[1]) <= 0.02175 
        # unsafe_mat = np.array([[0., 1., 0., 0., 0.]])
        # unsafe_vec = np.array([0.02175])
        res = []
        data = []
        if len(x) != len(y):
            raise Exception('length(x) should equal length(y)')

        if len(x) != len(spec_ids):
            raise Exception('length(x) should equal length(spec_ids)')

        for i in range(len(x)):
            for p_filter in p_filters:
                net, lb, ub, unsmat, unsvec = load_ACASXU(x[i], y[i], spec_ids[i])
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
        # print verification results
        print(tabulate(data, headers=["Prop.", "Net", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                  "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"]))

        # save verification results
        path = "artifacts/HSCC2022/ACASXU"
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path+"/ACASXUTable.tex", "w") as f:
            print(tabulate(data, headers=["Net", "Prob", "p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet",  "UnsafeProb-LB", \
                                  "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)
            
        
        return res, data
      
    def test_quantiverify_RocketNet(self, numCores, net_ids, spec_ids, p_filters):
        """Verify all DRL networks"""

        data = []
       
        for net_id in net_ids:           
            for spec_id in spec_ids: # running property 2 need more memory
                prob_lbs = []
                prob_ubs = []
                prob_mins = []
                prob_maxs = []
                counterInputSets = []
                verifyTimes = []
                net, lb, ub, unsmat, unsvec = load_DRL(net_id, spec_id)
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
                name = 'rocketNet_{}'.format(net_id)
                spec = 'P_{}'.format(spec_id)

                for p_filter in p_filters:
                    start = time.time()
                    outputSet, unsafeOutputSet, counterInputSet, prob_lb, prob_ub, prob_min, prob_max = quantiVerifyBFS(net=net, \
                                                                                                                        inputSet=inputSet, unsafe_mat=unsmat, \
                                                                                                                        unsafe_vec=unsvec, numCores=numCores, p_filter=p_filter)
                    end = time.time()
                    verifyTime = end-start

                    # store data for plotting
                    prob_lbs.append(prob_lb)
                    prob_ubs.append(prob_ub)
                    prob_maxs.append(prob_max)
                    prob_mins.append(prob_min)
                    counterInputSets.append(len(counterInputSet))
                    verifyTimes.append(verifyTime)

                    data.append([name, spec, p_filter, len(outputSet), len(unsafeOutputSet), len(counterInputSet), prob_lb, prob_ub, prob_min, prob_max, inputSetProb, verifyTime])

                fig, (ax1, ax2, ax3) = plt.subplots(3,1)
                ax1.plot(p_filters, prob_lbs, marker='o', label='prob_lb')
                ax1.plot(p_filters, prob_ubs, marker='*', label='prob_ub')
                ax1.plot(p_filters, prob_mins, marker='x', label='prob-min')
                ax1.plot(p_filters, prob_maxs, marker= 'p',label='prob-max')
                # ax1.set_title('Probability')
                ax1.set_ylabel('$p$', fontsize=13)
                ax1.set_xlabel('$p_f$', fontsize=13)
                ax1.legend(loc='upper left')

                ax2.plot(p_filters, counterInputSets)
                # ax2.set_title('CounterSet', fontsize=13)
                ax2.set_xlabel('$p_f$', fontsize=13)
                ax2.set_ylabel('$N_C$', fontsize=13)

                ax3.plot(p_filters, verifyTimes)
                # ax3.set_title('VT')
                ax3.set_xlabel('$p_f$', fontsize=13)
                ax3.set_ylabel('$VT (s)$', fontsize=13)

                # save verification results
                path = "artifacts/HSCC2022/rocketNet"
                if not os.path.exists(path):
                    os.makedirs(path)

                plt.savefig(path+"/rocketNet_{}_spec_{}.png".format(net_id, spec_id), bbox_inches='tight')  # save figure


        print(tabulate(data, headers=["Network_ID", "Property", "p_filter", "OutputSet", "UnsafeOutputSet", "counterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", "UnsafeProb-Min", "UnsafeProb-Max", "InputSetProbability", "VerificationTime"]))

        

        with open(path+"/rocketNetTable.tex", "w") as f:
            print(tabulate(data, headers=["Network_ID", "Property","p_filter", "OutputSet", "UnsafeOutputSet", "CounterInputSet", "UnsafeProb-LB", "UnsafeProb-UB", \
                                              "UnsafeProb-Min", "UnsafeProb-Max", "inputSet Probability", "VerificationTime"], tablefmt='latex'), file=f)

        
        return data

    def test_quantiverify_tiny_network(self, numCores):
        """verify a tiny network"""

        self.n_tests = self.n_tests + 1

         
        try:
            print('=====================================================')
            print('Quantitative Verification of Tiny Network')
            print('=====================================================')
            net, lb, ub, unsafe_mat, unsafe_vec = load_tiny_network()
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
            path = "artifacts/HSCC2022/tinyNet"
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
            plt.savefig(path+"/OutputSet.png", bbox_inches='tight')  # save figure
            plt.show()
            plot_probstar(unsafeOutputSet,dir_mat=dir_mat,dir_vec=None,show_prob=True, show=False)
            plt.savefig(path+"/UnsafeOutputSet.png", bbox_inches='tight')  # save figure
            plt.show()
            plot_probstar(counterInputSet,dir_mat=dir_mat,dir_vec=None, label= ("$x_1$", "$x_2$"), show_prob=True, show=False)
            plt.savefig(path+"/CounterInputSet.png", bbox_inches='tight')  # save figure
            plt.show()

            print('=====================================================')
            print('DONE!')
            print('=====================================================')

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_quantiverifyMC_ACASXU_all(self, x, y, spec_ids, N_samples, nTimes, numCores=1):
        """Verify all ACASXU networks with spec_id using Monte Carlo sampling-based method"""

        res = []
        data = []
        if len(x) != len(y):
            raise Exception('length(x) should equal length(y)')

        if len(x) != len(spec_ids):
            raise Exception('length(x) should equal length(spec_ids)')

        for i in range(len(x)):
            for N in N_samples:
                for nC in numCores:
                    
                    net, lb, ub, unsmat, unsvec = load_ACASXU(x[i], y[i], spec_ids[i])
                    S = Star(lb, ub)
                    mu = 0.5*(S.pred_lb + S.pred_ub)
                    a  = 3.0 # coefficience to adjust the distribution
                    sig = (mu - S.pred_lb)/a
                    # print('Mean of predicate variables: mu = {}'.format(mu))
                    # print('Standard deviation of predicate variables: sig = {}'.format(sig))
                    Sig = np.diag(np.square(sig))
                    # print('Variance matrix of predicate variables: Sig = {}'.format(Sig))
                    inputSet = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)
                    inputSetProb = inputSet.estimateProbability()
                    netName = '{}-{}'.format(x[i], y[i])
                    start = time.time()
                    unsafeprob = quantiVerifyMC(net=net, inputSet=inputSet, unsafe_mat=unsmat, \
                                                                             unsafe_vec=unsvec, numSamples=N, nTimes=nTimes, numCores=nC)
                    end = time.time()
                    verifyTime = end-start
                    data.append([spec_ids[i], netName, N, unsafeprob, inputSetProb, nC, verifyTime])
        # print verification results
        print(tabulate(data, headers=["Prop.", "Net", "N-Samples", "UnsafeProb", "inputSet Probability", "numCores", "VerificationTime"]))

        # save verification results
        path = "artifacts/HSCC2022/ACASXU"
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path+"/ACASXUTable_MC.tex", "w") as f:
            print(tabulate(data, headers=["Net", "N-Samples", "UnsafeProb", "inputSet Probability", "numCores", "VerificationTime"], tablefmt='latex'), file=f)
            
        
        return res, data


    def test_quantiverifyProbStarTL_harmonic_oscillator(self):
        'Verify harmonic oscillator againts a ProbStar temporal logic specification'


        print('Quantitative Reachability Analysis of Harmonic Osscillator Model againts ProbStarTL property')
        print('Loading model, initial conditions and inputs...')
        
        plant, lb, ub, input_lb, input_ub = load_harmonic_oscillator_model()
        
        dt = np.pi/4
        numsteps = 4

        print('Constructing initial set, input set....')
        S = Star(lb, ub)
        mu = 0.5*(S.pred_lb + S.pred_ub)
        a = 3.0 # coefficience to adjust the distribution
        sig = (mu - S.pred_lb)/a
        Sig = np.diag(np.square(sig))
        X0 = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)

        S = Star(input_lb, input_ub)
        mu = 0.5*(S.pred_lb + S.pred_ub)
        a = 3.0 # coefficience to adjust the distribution
        sig = (mu - S.pred_lb)/a
        Sig = np.diag(np.square(sig))
        U = ProbStar(S.V, S.C, S.d, mu, Sig, S.pred_lb, S.pred_ub)

        # specification:
        # always_[0, 2](x[0] >= 0)
        A = np.array([-1., 0.])
        b = np.array([0.])
        P = AtomicPredicate(A,b)
        op = _ALWAYS_(0,2)
        spec1 = Formula([op, P])

        # spec2: always_[0, 2](x[1] >= 0)
         
        A = np.array([0., -1.])
        b = np.array([0.])
        P = AtomicPredicate(A,b)
        spec2 = Formula([op, P])

        # spec3: always_[0, 2](x[1] >= 0.5 <-> 0*x[0] - x[1] <= -0.5)
         
        A = np.array([0., -1.])
        b = np.array([-0.5])
        P = AtomicPredicate(A,b)
        op = _ALWAYS_(0,4)
        spec3 = Formula([op, P])

        # spec4: always_[0, 4] (-x[0] + x[1] <= 0.0)
        A = np.array([0, -1.0])
        b = np.array([-0.5])
        P1 = AtomicPredicate(A, b)

        A = np.array([-1.0, -1.0])
        b = np.array([0.0])
        P2 = AtomicPredicate(A,b)

        op1 = _AND_()

        op = _ALWAYS_(2,4)

        spec4 = Formula([op, P2])

        spec = spec4
        
        print('Specification: ')
        spec.print()

        probSAT, Xt = quantiVerifyProbStarTL(plant, spec, X0=X0, U=U, timeStep=dt, numSteps = numsteps)

        print('Probability of satisfaction: {}'.format(probSAT))
        print('Plot reachable set...')
        plot_probstar(I=Xt, label=('$x$', '$y$'))

        
        

if __name__ == "__main__":

    test_verifier = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_verifier.test_reachExactBFS()
    # test_verifier.test_reach_2017_IEEE_TNNLS()
    # test_verifier.test_reach_ACASXU(x=3,y=7,spec_id=3)
    test_verifier.test_checkSafetyStar()
    # test_verifier.test_quantiverify_ACASXU(x=3,y=7,spec_id=3,numCores=8)
    # test_verifier.test_quantiverify_tiny_network(numCores=4)

    # x[0] = COC (Clear-of-Conflict)
    # x[1] = Weak-Left
    # x[2] = Weak-Right
    # x[3] = Strong-Left
    # x[4] = Strong-Right
    # Look at load_ACASXU

    x = [1]
    y = [2]
    s = [2]
    # x = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1]
    # y = [2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 4, 5, 6, 7, 8, 9, 1, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9] 
    # s = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4] # property id
    # test_verifier.test_quantiverify_ACASXU_all(x=x, y=y, spec_ids=s, \
    #                                             numCores=8, unsafe_mat=None, unsafe_vec=None, p_filters=[0.0, 1e-5])

    # test_verifier.test_quantiverifyMC_ACASXU_all(x=x, y=y, spec_ids=s, N_samples=[1000, 10000, 100000, 1000000, 10000000], nTimes=1, numCores=[1, 4])

    # test_verifier.test_quantiverify_ACASXU_all(x=[1], y=[7, 8, 9], spec_id=4, \
    #                                             numCores=8, unsafe_mat=None, unsafe_vec=None, p_filter=0.0)

    # test_verifier.test_quantiverify_RocketNet(numCores=8, net_ids=[0, 1], spec_ids=[1,2], p_filters=[0.0, 1e-8, 1e-5, 1e-3])

    # test_verifier.test_quantiverifyProbStarTL_harmonic_oscillator()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing verifier: fails: {}, successfull: {}, \
    total tests: {}'.format(test_verifier.n_fails,
                            test_verifier.n_tests - test_verifier.n_fails,
                            test_verifier.n_tests))
