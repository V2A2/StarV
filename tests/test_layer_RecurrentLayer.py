"""
Test RecurrentLayer Class
Author: Qing Liu
Date: 9/28/2025
"""

from curses.ascii import RS
import numpy as np
from StarV.RNN_ProbStar_rechability import reachability_with_RNN_exact_branches, verify_tl_over_branches
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.RecurrentLayer import RecurrentLayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.net.network import NeuralNetwork
from StarV.util.load_rnn import load_simple_rnn, get_Star_set,get_ProbStar_set
from StarV.util.plot import plot_2D_Star,plot_probstar_signal,plot_probstar,plot_SAT_trace,plot_probstar_reachset_with_unsafeSpec
import matplotlib.pyplot as plt
from StarV.fun.poslin import PosLin
from StarV.verifier.verifier import checkSafetyProbStar
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_,_OR_
from StarV.layer.RecurrentLayer import RecurrentLayer


class Test(object):
    """
    Testing RecurrenltLayer class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_constructor(self):

        self.n_tests = self.n_tests + 1
        Whx = np.random.rand(2, 3)
        Whh = np.random.rand(3, 3)
        bh = np.random.rand(3)
        Woh = np.random.rand(3, 3)
        bo = np.random.rand(3)
        print('\nTest RecurrentLayer Constructor\n')

        try:
            L = RecurrentLayer(Whx, Whh, bh, Woh, bo)
            # L.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_rand(self):
        self.n_tests = self.n_tests + 1
        print('\nTest RecurrentLayer random method\n')

        try:
            RecurrentLayer.rand(3, 2)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_reachExact(self):
        self.n_tests = self.n_tests + 1
        print('\nTest RecurrentLayer reachExact method\n')

        try:
            L = RecurrentLayer.rand(2, 3)
            # L.info()
            In =[]
            I=Star.rand(2)
            print('Input Set:')
            I.__str__()
            I.C = np.zeros([1,I.nVars])  
            I.d = np.zeros([1])
            In.__str__()
            for i in range(5):
                In.append(I)
            S = L.reachExact(In)
            print('Number of utput Set:'.format(len(S)))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_reachApprox(self):
        self.n_tests = self.n_tests + 1
        print('\nTest RecurrentLayer reachExact method\n')

        try:
            L = RecurrentLayer.rand(2, 3)
            In = []
            I = Star.rand(2)
            print('Input Set:')
            I.__str__()
            I.C = np.zeros([1,I.nVars])  
            I.d = np.zeros([1])
            for i in range(5):
                In.append(I)
            S = L.reachApprox(In)
            print('Number of utput Set:'.format(len(S)))
            # S[i].__str__()

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_reach(self):
        self.n_tests = self.n_tests + 1
        print('\nTest RecurrentLayer reach method\n')
        
        try:
            L = RecurrentLayer.rand(2, 3)
            In = []
            I = Star.rand(2)
            I.C = np.zeros([1,I.nVars])  
            I.d = np.zeros([1])
            for i in range(5):
                In.append(I)
            print('Number of input sets: {}'.format(len(In)))
            S = L.reach(In)
            print('Number of output Set: {}'.format(len(S)))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_simple_rnn(self):

        self.n_tests = self.n_tests + 1
        print('\n Test Simple Recurrent Neural Network\n')

        Whx,Whh,bh,Woh,bo,data_points, W_ff,b_ff = load_simple_rnn()

        # create NN
        L1 = RecurrentLayer(Whx,Whh,bh,Woh,bo)
        mat= []
        for i in range(len(W_ff[0])):
            print(f"w_ff[0]{i} shape:",W_ff[0][i].shape)
            W_b = [W_ff[0][i],b_ff[0][i].reshape(-1)]
            mat.append(W_b)
            # print("W_b:",W_b[0].shape())

        L2 = FullyConnectedLayer(mat[0])
        L3 = ReLULayer()
        L4 = FullyConnectedLayer(mat[1])
        L5 = ReLULayer()
        L4 = FullyConnectedLayer(mat[2])
        L6 = ReLULayer()
        # L5 = FullyConnectedLayer(mat[3],fo='relu')
        # L6 = FullyConnectedLayer(mat[4],fo='relu')
        # L7 = FullyConnectedLayer(mat[5],fo='relu')


        layers = [L1,L2,L3,L4,L5,L6]
        net = NeuralNetwork(layers=layers)

        num_input_seq= 1
        x = data_points[:num_input_seq,:]
        x = x.T
    
        eps = 0.01
        time_steps = 5 
        results = []
        # try :
        for k in range(0,num_input_seq):
            xk = np.array(x[:, k]).reshape(-1,1)
            col_point = xk
            S = get_ProbStar_set(col_point,eps,time_steps)
            Numlayers = len(layers)
            Layer_RS = []
            RS = S
            for j in range(0,Numlayers):
                layers[j].info()
                RS1 = net.layers[j].reach(RS, method = "exact", lp_solver='gurobi', pool=None, RF=0.0, DR=0)
                print("\nnumer of Output sets after layer {} : {}".format(j+1,len(RS1)))
                print("output set types after layer {} : {}".format(j+1,type(RS1)))
                print("output set[0] types after layer {} : {}{}".format(j+1,type(RS1[0]),RS1[0]))
                print("num of output set[0]  after layer {} : {}".format(j+1,len(RS1[0])))
                RS = RS1
                Layer_RS.append(RS1)
            result = RS1
            results.append(result)
        # except Exception:
        #     print('Test Fail!')
        #     self.n_fails = self.n_fails + 1
        # else:
        #     print('Test Successfull!')

        
        return results

    def test_multiRandomLayers(self):
        self.n_tests = self.n_tests + 1
        print('\nTest multiRecurrentLayer\n')
        np.random.seed(42)

        # try:
        L1 = RecurrentLayer.rand(8, 2)
        L2 = RecurrentLayer.rand(2, 2)
        # L3 = FullyConnectedLayer.rand(8,12)
        layers = [L1,L2]
        net = NeuralNetwork(layers=layers)
        Numlayers = len(layers)
        In = []
        I = Star.rand(8)
        print("I dim:",I.dim)
        I.C = np.zeros([1,I.nVars])  
        I.d = np.zeros([1])
        for i in range(5):
            In.append(I)
        # print('Number of input sets: {}'.format(len(In)))
        results = []
        RS = In
        for j in range(0,Numlayers):
            print("\n====== Process the {}th layer of NN ===========".format(j+1))
            # layers[j].info()
            RS1 = net.layers[j].reach(RS, method = "exact", lp_solver='gurobi', pool=None, RF=0.0, DR=0)
            for i in range(len(RS1)):
                print('Number of Output sets in O{}:{}'.format(i,len(RS1[i])))
            #     for k in range(len(RS1[i])):
            #         print(" Layer {} at time {} output set {} :".format(j+1,i,RS1[i][k].V))
            RS = RS1
        result = RS1
        results.append(result)
        # except Exception:
        #     print('Test Fail!')
        #     self.n_fails = self.n_fails + 1
        # else:
        #     print('Test Successfull!')

    def test_multiMinsum(self):
            
            np.random.seed(42)
            self.n_tests = self.n_tests + 1
            print('\nTest multiMinkowskiSum\n')
            L = RecurrentLayer.rand(2,2)
            Whx = L.Whx
            bh = L.bhx 
            Whh = L.Whh
            print('Hidden to Hidden weight matrix:',Whh)
            print('Input to Hidden weight matrix:',Whx)
            print('Hidden layer bias vector:',bh)
            def star_rand(dim):
                # lb = np.round(np.random.uniform(0.01, 1.0, size=dim), 2)
                # gap = np.round(np.random.uniform(0.01, 1.0, size=dim), 2)
                # ub = lb + gap
                lb = -np.round(np.random.rand(dim,),2)
                ub = np.round(np.random.rand(dim,),2)
                print('lb:',lb)
                return Star(lb, ub)
            # # try:
            O1 = star_rand(2)
            O1.C = np.zeros([1,O1.nVars])  
            O1.d = np.zeros([1])
            print('Input Set O1:',O1)
            # plot_2D_Star(O1)
            X2 = star_rand(2)
            print('Input Set X2:',X2)
            X2.C = np.zeros([1,O1.nVars])  
            X2.d = np.zeros([1])

            # X =[O1,X2]

            # S1 = star_rand(2)
            # S2 =star_rand(2)
            # S3 =star_rand(2)
            # S = [S1,S2,S3]


            # Relu(X2)
            relu_X2 = ReLULayer.reach([X2], method="exact")
            for j in range(len(relu_X2)):
                print('Relu X2 Set {}:{}\n'.format(j,relu_X2[j].V))
            print("number of sets after relu_X2:{}\n".format(len(relu_X2)))

            # Relu(O1)
            relu_O1= ReLULayer.reach([O1], method="exact")
            for i in range(len(relu_O1)):
                print('Relu O1 Set {}:{}\n'.format(i,relu_O1[i].V))
            print("number of sets after relu_O1:{}\n".format(len(relu_O1)))

            # all_out = []
            # for i in range(len(relu_O1)):
            #     print('Relu O1 Set {}:{}\n'.format(i,relu_O1[i].V))
            #     # print("dim of relu O1 set {}: {}\n".format(i,relu_O1[i].dim))
            #     out = relu_O1[i].minKowskiSum(X2)
            #     all_out.append(out)
            #     print('Relu O1 Set {} minsum X2:{}\n'.format(i,out.V))
            # print("number of sets after relu_O1.minKowskiSum(X2):{}\n".format(len(all_out)))


            # relu_out = []
            # for j in range(len(all_out)):
            #     out1 = ReLULayer.reach([all_out[j]], method="exact")
            #     relu_out.extend(out1)
            # for i in range(len(relu_out)):
            #     print('relu(relu(O1)_minsum_X2) Set {}:{}\n'.format(i,relu_out[i].V))
            # print("number of set after relu(relu(O1)_minsum_x2):{}\n".format(len(relu_out)))

            # print("++++++++++++++++++++++++++++++++++++++++++++++++++")
            # relu_relu_O1 = []
            # for i in range(len(relu_O1)):
            #     print('Relu O1 Set {}:{}\n'.format(i,relu_O1[i].V))
            #     out2= ReLULayer.reach([relu_O1[i]],method="exact")
            #     print("relu(relu_O1[i]):{}".format(out2[0].V))
                # print("number of set after relu(relu(O1_i))):{}\n".format(len(out2)))
                # print("relu(relu_O1[{}]):{}\n".format(i,out2[0].V))
                # print("relu(relu_O1[{}]):{}\n".format(i,out2[0].dim))
                # relu_relu_O1.extend(out2)
            # for i in range(len(relu_relu_O1)):
            #     print('relu(relu(O1)) Set {}:{}\n'.format(i,relu_relu_O1[i].V))
            # print("number of set after relu(relu(O1))):{}\n".format(len(relu_relu_O1)))

            # relu_out_2 =[]
            # for i in range(len(relu_relu_O1)):
            #     R =[]
            #     for j in range(len(relu_relu_O1)):
            #         out3 = relu_relu_O1[i].minKowskiSum(relu_X2[j])
            #         R.append(out3)
            #     relu_out_2.extend(R)
            # print("number of set after relu(relu(O1))_minsum_Relu(X2):{}\n".format(len(relu_out_2)))
            # for i in range(len(relu_out_2)):
            #     print('relu(relu(O1))_minsum_Relu(X2) Set {}:{}\n'.format(i,relu_out_2[i].V))

             # Plot lRelu(O1)_minsum_Relu(X2)_approx_2, N sets
            # for S in relu_out:
            #     plot_2D_Star(S, show=False)
            #     print("plot the {}th set in relu_out".format(relu_out.index(S)))
            #     lb,ub = S.getRanges()
            #     print("lower bound {} and upper bound range of the {}th set in relu_out".format(lb,ub,relu_out.index(S)))
            #     plt.show()
            # plt.savefig("Relu(O1)_minsum_Relu(X2)_approx_1.png") 
            # plt.show()

            # for S in relu_out_2:
            #     plot_2D_Star(S, show=False)
            #     print("plot the {}th set in relu_out_2".format(relu_out_2.index(S)))
            #     lb,ub = S.getRanges()
            #     print("lower bound {} and upper bound range of the {}th set in relu_out_2".format(lb,ub,relu_out_2.index(S)))
            #     plt.show()
            # plt.savefig("Relu(O1)_minsum_Relu(X2)_approx_1.png") 
            # plt.show()

            # ========================Relu(O1_minsum_X2)======================
            sum1 = O1.minKowskiSum(X2)
            print('O1 minsum X2 Set:',sum1)
            relu_1 = ReLULayer.reach([sum1], method="exact")
            for k in range(len(relu_1)):
                print('Relu O1_X1 Set {}:{}\n'.format(k,relu_1[k].V))
                print("dim of relu O1_X2 set {}: {}\n".format(k,relu_1[k].dim))
            print("number of sets after relu_O1_X2:{}\n".format(len(relu_1)))




            # =================Relu(O1)_minsum_Relu(X2)================= N x Msets exact method
            # [O1,O2,O3] minsum [X1,X2,X3] = [(O1 minsum X1),(O1 minsum X2),(O1 minsum X3),(O2 minsum X1),(O2 minsum X2), (O2 minsum X3),(O3 minsum X1), (O3 minsum X2), (O3 minsum X3)] = [R1,R2,R3,R4,R5,R6,R7,R8,R9]
            all_exact_set = []
            for y in range(len(relu_O1)):
                summed = []
                for z in range(len(relu_X2)):
                    sum = relu_O1[y].minKowskiSum(relu_X2[z])
                    print('N x M sets{}{}:{}\n'.format(y,z,sum.V))
                    summed.append(sum)
                all_exact_set.extend(summed)
            print("number of sets after Relu(O1)_minsum_Relu(X2) using exact method:{}\n".format(len(all_exact_set)))

            # =================Relu(O1)_minsum_Relu(X2)================= N sets approx method 1, reduce output set amount,but still need many minsum operations
            # [O1,O2,O3] minsum [X1,X2,X3] = [(((O1 minsum X1) minsum X2) minsum X3),(((O2 minsum X1) minsum X2) minsum X3),(((O3 minsum X1) minsum X2) minsum X3)] = [ R1,R2,R3]
            all_approx_set = []
            for y in range(len(relu_O1)):
                for z in range(len(relu_X2)):
                    set_from_O1 = relu_O1[y]
                    sum = set_from_O1.minKowskiSum(relu_X2[z])
                    set_from_O1 = sum
                all_approx_set.append(set_from_O1)


            print("number of sets after Relu(O1)_minsum_Relu(X2) using approx method 1:{}\n".format(len(all_approx_set)))

            # =================Relu(O1)_minsum_Relu(X2)================= N sets approx method 2, reduce output set amount and reduce minsum operations
            # **** The output of this method is same as computing Relu(O1_minsum_X2)***
            all_approx_set_1 = []
            for y in range(len(relu_O1)):
                for z in range(len(relu_X2)):
                    if z == y:
                        print(" ======z=y=====")
                        sum = relu_O1[y].minKowskiSum(relu_X2[z])
                        all_approx_set_1.append(sum)

            print("number of sets after Relu(O1)_minsum_Relu(X2) using approx method 2:{}\n".format(len(all_approx_set_1)))

            # =================Approximate to single set: Minsum(Relu(O1))_minsum_Minsum(Relu(X2))================= 1 set
            O1_1 = relu_O1[0]
            for y in range(1,len(relu_O1)):
                summed = O1_1.minKowskiSum(relu_O1[y])
                summed_O1= summed
            print("number of sets after minsum(relu_O1):{}\n".format(len(summed_O1)))

            X2_1 = relu_X2[0]
            for y in range(1,len(relu_X2)):
                summed = X2_1.minKowskiSum(relu_O1[y])
                summed_X2= summed
            print("number of sets after minsum(relu_X2):{}\n".format(len(summed_X2)))

            All_Sum = summed_O1.minKowskiSum(summed_X2)  
            print("All_Sum_Minsum(Relu(O1))_minsum_Minsum(Relu(X2)):{}\n".format(All_Sum.V))

            # =================Approximate to single set: Minsum(Relu(O1))_minsum_Relu(X2)================= len(relu(x2)) sets

            min_all =[]
            for i in range(len(relu_X2)):
                summed = summed_O1.minKowskiSum(relu_X2[i])
                min_all.append(summed)
            print("number of sets after minsum(relu_O1)_(relu_X2):{}\n".format(len(min_all)))


            # min_all =[]
            # for i in range(len(relu_O1)):
            #     summed = relu_O1[i].minKowskiSum(summed_X2)
            #     min_all.append(summed)
            # print("number of sets after minsum(relu_O1)_(relu_X2):{}\n".format(len(min_all)))


            # # =========================Plot results=========================

            color1 = 'g'
            color2 = 'r' 
            color3 = 'b' 
            color4 = 'y'
            color5 = 'm'

            # # Plot Relu(O1_minsum_X2)
            # print("\n-----------------------------------------------------------------------------\n")
            # for S in relu_1:
            #     plot_2D_Star(S, color=color1, show=False)
            #     print("plot the {}th set in relu_1".format(relu_1.index(S)))
            #     lb,ub = S.getRanges()
            #     print("lower bound {} and upper bound range of the {}th set in relu_1".format(lb,ub,relu_1.index(S)))
            # plt.savefig("Relu(O1_minsum_X2).png") 
            # plt.show()

            # Plot Relu(O1)_minsum_Relu(X2)_exact, N x M sets
            print("-----------------------------------------------------------------------------\n")
            for S in all_exact_set:
                plot_2D_Star(S, color=color2, show=False)
                print("plot the {}th set in all_exact_set".format(all_exact_set.index(S)))
                lb,ub = S.getRanges()
                print("lower bound {} and upper bound range of the {}th set in all_exact_set".format(lb,ub,all_exact_set.index(S)))
                # plt.show()
            # plt.savefig("Relu(O1)_minsum_Relu(X2)_exact.png") 
            plt.show()


            # print("-----------------------------------------------------------------------------\n")
            # # Plot Relu(O1)_minsum_Relu(X2)_approx_1, N sets
            # for S in all_approx_set:
            #     plot_2D_Star(S, color=color3, show=False)
            #     print("plot the {}th set in all_approx_set".format(all_approx_set.index(S)))
            #     lb,ub = S.getRanges()
            #     print("lower bound {} and upper bound range of the {}th set in all_approx_set".format(lb,ub,all_approx_set.index(S)))
            #     # plt.show()
            # plt.savefig("Relu(O1)_minsum_Relu(X2)_approx.png") 
            # plt.show()


            print("-----------------------------------------------------------------------------\n")
            # Plot lRelu(O1)_minsum_Relu(X2)_approx_2, N sets
            for S in all_approx_set_1:
                plot_2D_Star(S, color=color4, show=False)
                print("plot the {}th set in all_approx_set_1".format(all_approx_set_1.index(S)))
                lb,ub = S.getRanges()
                print("lower bound {} and upper bound range of the {}th set in all_approx_set".format(lb,ub,all_approx_set_1.index(S)))
                # plt.show()
            plt.savefig("Relu(O1)_minsum_Relu(X2)_approx_1.png") 
            plt.show()


            print("-----------------------------------------------------------------------------\n")
            # Plot Minsum(Relu(O1))_minsum_Minsum(Relu(X2)), 1 set
            
            plot_2D_Star(All_Sum, color=color5, show=False)
            lb,ub = All_Sum.getRanges()
            print("lower bound {} and upper bound {} range of the set in All_Sum".format(lb,ub))
            plt.savefig("Minsum(Relu(O1))_minsum_Minsum(Relu(X2)).png") 
            plt.show()


            print("-----------------------------------------------------------------------------\n")
            # Plot Minsum(Relu(O1))_minsum_(Relu(X2)), len(relu(x2)) set
            for s in min_all:
                plot_2D_Star(s, color=color5, show=False)
                lb,ub = s.getRanges()
                print("lower bound {} and upper bound {} range of the {}th set in All_Sum".format(lb,ub,min_all.index(s)))
            # plt.savefig("Minsum(Relu(O1))_minsum_Minsum(Relu(X2)).png") 
            plt.show()


            # # Affine mapping testing
            # af_O1 = O1.affineMap(Whh)
            # print('Affine Map O1:',af_O1)
            # af_X2 = X2.affineMap(Whx)
            # print('Affine Map R X2:',af_X2)
            # af_msum = af_O1.minKowskiSum(af_X2)
            # print('MinSum Affine Map Set:',af_msum)

            # O_1_X2 = O1.minKowskiSum(X2)
            # print('Minsum O1 X2 Set:',O_1_X2)
            # af_O_1_X2 = O_1_X2.affineMap(Whx)
            # print('Affine Map O1 minsum O1_X2 Set:',af_O_1_X2)



            # sum3 = I.minKowskiSum(I3)
            # print('Sum3 Set:',sum3)
            # for i in range(len(S)):
            #     O = O.minKowskiSum(S[i])
            #     print(f"O{i}:{O}")
            # print('Result Set:',O)
            
            # except Exception:
            #     print('Test Fail!')
            #     self.n_fails = self.n_fails + 1
            # else:
            #     print('Test Successfull!')

    def test_probstar_construct(self):
        np.random.seed(42)
        X =Star.rand(8)
        print("First Star info:",X)
        mu = 0.5*(X.pred_ub + X.pred_lb) 
        a  = 3
        sig= (X.pred_ub-mu )/a
        epsilon = 1e-10
        sig = np.maximum(sig, epsilon)
        Sig = np.diag(np.square(sig))
        P = ProbStar(X.V,X.C,X.d, mu, Sig,X.pred_lb,X.pred_ub)
        print("ProbStar info:",P,"probility:",P.estimateProbability())

        lb_X = X.getRanges()[0]
        ub_X = X.getRanges()[1]
        print("Star lower bound:",lb_X," upper bound:",ub_X)

        mu_1 = 0.5*(ub_X + lb_X)
        sig_1= (ub_X - mu_1)/a
        print("mu_1:",mu_1,"sig:",sig)
        sig_1 = np.maximum(sig_1, epsilon)
        Sig_1 = np.diag(np.square(sig_1))
        print("Sig_1:",Sig_1)
        P_1 = ProbStar(mu_1, Sig_1, lb_X, ub_X)
        print("ProbStar_1 info:",P_1,"probility_1:",P_1.estimateProbability())


    def test_probstar_combine(self):
        # np.random.seed(42)
        lb = np.array([0,1])
        ub = np.array([1.5,2.5])
        # X = Star.rand(2)
        X = Star(lb,ub)
        print("First Star info:",X)
        mu = 0.5*(X.pred_ub + X.pred_lb) 
        a  = 3
        sig= (X.pred_ub-mu )/a
        epsilon = 1e-10
        sig = np.maximum(sig, epsilon)
        Sig = np.diag(np.square(sig))
        P = ProbStar(X.V,X.C,X.d, mu, Sig,X.pred_lb,X.pred_ub)
        print("ProbStar info:",P,"probility:",P.estimateProbability())
        plot_probstar(P)
        
        # X1 =Star.rand(2)
        lb1 = np.array([-1,0])
        ub2 = np.array([0.5,1.5])
        # X = Star.rand(2)
        X1 = Star(lb1,ub2)
        print("Second Star info:",X1)
        mu = 0.5*(X1.pred_ub + X1.pred_lb) 
        a  = 3
        sig= (X1.pred_ub-mu )/a
        epsilon = 1e-10
        sig = np.maximum(sig, epsilon)
        Sig = np.diag(np.square(sig))
        P1 = ProbStar(X1.V,X1.C,X1.d, mu, Sig,X1.pred_lb,X1.pred_ub)
        print("ProbStar info:",P1,"probility:",P1.estimateProbability())
        plot_probstar(P1)

        Combine_P = P.Combine(P1)

        plot_probstar_signal([P,P1])
        # plot_2D_Star(S)
        plot_probstar(Combine_P)

    def test_relu(self):
        np.random.seed(42)
        self.n_tests = self.n_tests + 1
        # try:
        # I = ProbStar.rand(2)
        I = Star.rand(2)
        # map_mat = np.array([[0,0,1,0],[0,0,0,1]])
        # I_aff = I.affineMap(map_mat)
        # a = 
        # plot_2D_Star(I_aff)
        # plot_probstar(I_aff)
        # S =[I1,I2,I3]
        print(f"I:{I}, type:{type(I)}")
        R = PosLin.reachExactSingleInput(I,'gurobi')
        print(f"\n==============Number of sets after relu: {len(R)}\n")
        for i in range(len(R)):
            if i ==0:
                R1 = R[i]
            if i == 1:
                R2 = R[i]
            if i == 2:
                R3 = R[i]
            if i == 3:
                R4 = R[i]
        result =[]
        result1 = R1.intersectStar(R2)
        result.append(result1)
        result2 = R2.intersectStar(R3)
        result.append(result2)
        result3 = R1.intersectStar(R3)
        result.append(result3)
        print(f"result of intersecting the two sets after relu: {result}")

        unsafe_mat = np.array([[0,-1]])
        unsafe_vec = np.array([-0.6])

        # S =[]
        # SAT=[]
        # SAT_prob=[]
        # for i in range(len(R)):
        #     if R[i].isEmptySet():
        #         print(f"R{i}isEmpty")
        #     else:
        #         print(f"R{i}:{R[i]}, :prob after relu = {R[i].estimateProbability()}")
        #         # aff_set = R[i].affineMap(map_mat)
        #         # S.append(aff_set)
        #         sat,prob = checkSafetyProbStar(unsafe_mat,unsafe_vec,R[i])
        #         SAT_prob.append(prob)
        #         SAT.append(sat)
        # plot_probstar_signal(S)
        # sat,prob = checkSafetyProbStar(unsafe_mat,unsafe_vec,R[i])
        # print(f"prob for sat :{SAT_prob}")
        # plot_SAT_trace(SAT,dir_mat=unsafe_mat,dir_vec=unsafe_vec)
        # plot_probstar_signal(SAT,dir_mat=map_mat)
        
        # except Exception:
        #     print('Test Fail!')
        #     self.n_fails = self.n_fails + 1
        # else:
        #     print('Test Successfull!')

 
    def test_ProbSatrTL(self):
        np.random.seed(42)
        self.n_tests = self.n_tests + 1
        # try:
        L1 = RecurrentLayer.rand(4, 4)
        In = []
        for i in range(5):
            X0 = Star.rand(4)
            print("X0:",X0)
            mu = 0.5*(X0.pred_lb + X0.pred_ub) 
            a  = 3
            sig= (mu - X0.pred_lb)/a
            epsilon = 1e-6
            sig = np.maximum(sig, epsilon)
            Sig = np.diag(np.square(sig))
            X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu, Sig,X0.pred_lb,X0.pred_ub)
            print(f"Input ProbStar {i}:",X0_probstar)
            In.append(X0_probstar)

        RS = L1.reach(In, method="exact")
        print("Number of output sets after RecurrentLayer:",len(RS))
        print("Output set types after RecurrentLayer:",type(RS))
        if len(RS) > 0 and isinstance(RS[0], list):
            RS1=[]
            for j in range(len(RS)):
                print('\n=======================')
                print(f"Output set {j} is a list of sets, number of sets in output set RS{j}: {len(RS[j])}")
                if len(RS[j]) == 1:
                    RS1.append(RS[j][0])
                else:
                    for i, set in enumerate(RS[j]):
                        print(f"\nSet {i} in output set RS{j}: {set}, probability: {set.estimateProbability()}")
                        print(f"output set:{RS[j][i]}")
                        # mat_mat= np.array([[0,0,1,0],[0,0,0,1]])
                        # All_map_sets=[]
                    #     for i in range(len(RS[j])):
                    #         Aff_RS= RS[j][i].affineMap(mat_mat)
                    #         All_map_sets.append(Aff_RS)
                        
                    # plot_probstar_signal(All_map_sets)
            if len(RS1) > 0:
                for i in range(len(RS1)):
                    print(f"\nSet {i} in RS1: {RS1[i]}, probability: {RS1[i].estimateProbability()}")
                print("lens of RS1:",len(RS1))
                RS =RS1
        else:
            print("Output set[0] is a single ProbStar, probability:",RS[0].estimateProbability())


        # create temporal specifications
        AND = _AND_()
        OR = _OR_()                                        
        lb = _LeftBracket_()
        rb = _RightBracket_()

        A1 = np.array([-1., 0.,0,0])
        b1 = np.array([-6])
        P1 = AtomicPredicate(A1,b1)

        A2 = np.array([0,-1,0,0])
        b2 = np.array([-13])
        P2 = AtomicPredicate(A2,b2)

        EVOT =_EVENTUALLY_(0,5)
        AWOT = _ALWAYS_(2,3)
        EVOT1 =_EVENTUALLY_(5,15)
        AWOT1 = _ALWAYS_(0,5)
        

        specs =[]
        spec = Formula([EVOT,P1])
        spec1 = Formula([AWOT,lb,P2,rb])
        spec2 = Formula([EVOT1,lb,P1,OR,lb,AWOT1,P2,rb,rb])
        specs =[spec]

        #mapping
        mat_mat= np.array([[1,0,0,0],[0,1,0,0]])
        All_map_sets=[]
        for i in range(len(RS)):
            Aff_RS= RS[i].affineMap(mat_mat)
            All_map_sets.append(Aff_RS)
        
        # plot_probstar(All_map_sets[1])


        print(f"\n========Start Verification Ca<=b===================")        
        # verify output reachable sets 
        unsafe_mat = np.array([[-1,0]])
        unsafe_vec = np.array([-6])

        all_check_sets=[]
        all_check_prob=[]
        for i, S1 in enumerate(All_map_sets):
            P = []
            prob = []
            if len(S1) > 1:
                for j,S2 in enumerate(S1):
                    P1, prob1 = checkSafetyProbStar(unsafe_mat, unsafe_vec, S2)
                    if isinstance(P1, ProbStar):
                        print(f"prob1 of S{i}{j} = {prob1}")
                        P.append(P1)
                        prob.append(prob1)
                    else:
                        print(f"S{i}{j} is an empty set, prob = 0.0")
            else:
                P1, prob1 = checkSafetyProbStar(unsafe_mat, unsafe_vec, S1)
                if isinstance(P1, ProbStar):
                    print(f"prob1 of S{i} = {prob1}")
                    P.append(P1)
                    prob.append(prob1)
                else:
                    print(f"S{i} is an empty set, prob = 0.0")

        
            if len(P) != 0:
                all_check_sets.append(P)
                all_check_prob.append(prob)


        # verification using ProbSatrTL
        print(f"\n========Start Verification TL===================")
        for i in range(0,len(specs)):
            spec = specs[i]
            print('\n==================Specification{}====================: '.format(i))
            spec.print()
            DNF_spec = spec.getDynamicFormula()
            print(f"====== Dynamic Formula=======\n {DNF_spec.print()}")
            Nadnf = DNF_spec.length
            print('Length of abstract DNF_spec = {}'.format(DNF_spec.length))
            _,p_max, p_min,Ncdnf = DNF_spec.evaluate_for_RNN(RS)
            print("p_min:",p_min)
            print("p_max:",p_max) 
            # verify_time=checking_time + reach_time_duration    
              
    def test_ProbStar_TL_verification(self):
        np.random.seed(42)
        self.n_tests = self.n_tests + 1
        # try:
        L1 = RecurrentLayer.rand(4, 4)
        L2 = FullyConnectedLayer.rand(4, 4)
        L3 = ReLULayer()
        L4 = FullyConnectedLayer.rand(4, 4)
        

        In = []
        for i in range(5):
            X0 = Star.rand(4)
            print("X0:",X0)
            mu = 0.5*(X0.pred_lb + X0.pred_ub) 
            a  = 3
            sig= (mu - X0.pred_lb)/a
            epsilon = 1e-6
            sig = np.maximum(sig, epsilon)
            Sig = np.diag(np.square(sig))
            X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu, Sig,X0.pred_lb,X0.pred_ub)
            print(f"Input ProbStar {i}:",X0_probstar)
            In.append(X0_probstar)

        branches = L1.reachExactBranches(
            In,
            post_layers=[L2,L3,L4],
            lp_solver="gurobi",
            pool=None,
            p_filter=None,
            show=True,
        )
        print("\n\nNumber of branches after RecurrentLayer:",len(branches))
        print("Branch types after RecurrentLayer:",type(branches))
        print("Branch 0 type:",type(branches[0]))
        print(f"Branch 0 number of sets: {len(branches[0])}")
        for i in range(len(branches[0])):
            print(f"Branch 0 set {i} type: {type(branches[0][i])}")
            print(f"Branch 0 set {i} probability: {branches[0][i].estimateProbability()}")
            print(f"Branch 0 set {i} info: {branches[0][i]}")
            
        # Example specs (uncomment and edit as needed)
          # create temporal specifications
        AND = _AND_()
        OR = _OR_()                                        
        lb = _LeftBracket_()
        rb = _RightBracket_()

        A1 = np.array([-1., 0.,0,0])
        b1 = np.array([-600])
        P1 = AtomicPredicate(A1,b1)

        A2 = np.array([0,-1,0,0])
        b2 = np.array([-50])
        P2 = AtomicPredicate(A2,b2)

        EVOT =_EVENTUALLY_(0,5)
        AWOT = _ALWAYS_(2,3)
        EVOT1 =_EVENTUALLY_(5,15)
        AWOT1 = _ALWAYS_(0,5)
        

        specs =[]
        spec = Formula([EVOT,P1])
        spec1 = Formula([AWOT,lb,P2,rb])
        spec2 = Formula([EVOT,lb,P1,OR,lb,AWOT,P2,rb,rb])
        specs =[spec2]


        #mapping
        # mat_mat= np.array([[1,0,0,0],[0,1,0,0]])
        # All_map_sets=[]
        # for i in range(len(RS)):
        #     Aff_RS= RS[i].affineMap(mat_mat)
        #     All_map_sets.append(Aff_RS)
        
        # plot_probstar(All_map_sets[1])

        map_mat =None
        

        for k, spec in enumerate(specs):
            print(f"\n==================Branch TL Spec {k}====================")
            spec.print()
            p_total, p_per_branch = verify_tl_over_branches(branches, spec, map_mat=map_mat, map_vec=None)
            print(f"p_total: {p_total}")
            print(f"p_per_branch (len={len(p_per_branch)}): {p_per_branch}")

if __name__ == "__main__":

    test_RecurrentLayer = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    # test_RecurrentLayer.test_constructor()
    # test_RecurrentLayer.test_rand()
    # test_RecurrentLayer.test_reachExact()
    # test_RecurrentLayer.test_reachApprox()
    # test_RecurrentLayer.test_reach()
    # test_RecurrentLayer.test_simple_rnn()
    # test_RecurrentLayer.test_multiRandomLayers()
    # test_RecurrentLayer.test_multiMinsum()
    # test_RecurrentLayer.test_probstar_construct()
    # test_RecurrentLayer.test_probstar_combine()
    # test_RecurrentLayer.test_relu()
    # test_RecurrentLayer.test_ProbSatrTL()
    test_RecurrentLayer.test_ProbStar_TL_verification()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing Recurrent Layer Class: fails: {}, successfull: {}, \
    total tests: {}'.format(    test_RecurrentLayer.n_fails,
                                test_RecurrentLayer.n_tests - test_RecurrentLayer.n_fails,
                                test_RecurrentLayer.n_tests))

    