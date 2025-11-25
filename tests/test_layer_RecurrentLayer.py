"""
Test RecurrentLayer Class
Author: Qing Liu
Date: 9/28/2025
"""

import numpy as np
from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.layer.RecurrentLayer import RecurrentLayer
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.net.network import NeuralNetwork
from StarV.util.load_rnn import load_simple_rnn, get_Star_set,get_ProbStar_set
from StarV.util.plot import plot_2D_Star
import matplotlib.pyplot as plt


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
            W_b = [W_ff[0][i],b_ff[0][i].reshape(-1)]
            mat.append(W_b)

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

        num_input_seq= 5
        x = data_points[:num_input_seq,:]
        x = x.T
    
        eps = 0.01
        time_steps = 5 
        results = []
        try :
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
                    # print("\nnumer of Output sets after layer {} : {}".format(j+1,len(RS1)))
                    # print("output set types after layer {} : {}".format(j+1,type(RS1)))
                    # print("output set[0] types after layer {} : {}".format(j+1,type(RS1[0])))
                    # print("num of output set[0] types after layer {} : {}".format(j+1,len(RS1[0])))
                    RS = RS1
                    Layer_RS.append(RS1)
                result = RS1
                results.append(result)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

        
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



            # =========================Plot results=========================

            color1 = 'g'
            color2 = 'r' 
            color3 = 'b' 
            color4 = 'y'

            # Plot Relu(O1_minsum_X2)
            print("\n-----------------------------------------------------------------------------\n")
            for S in relu_1:
                plot_2D_Star(S, color=color1, show=False)
                print("plot the {}th set in relu_1".format(relu_1.index(S)))
                lb,ub = S.getRanges()
                print("lower bound {} and upper bound range of the {}th set in relu_1".format(lb,ub,relu_1.index(S)))
            plt.savefig("Relu(O1_minsum_X2).png") 
            plt.show()

            # Plot Relu(O1)_minsum_Relu(X2)_exact, N x M sets
            print("-----------------------------------------------------------------------------\n")
            for S in all_exact_set:
                plot_2D_Star(S, color=color2, show=False)
                print("plot the {}th set in all_exact_set".format(all_exact_set.index(S)))
                lb,ub = S.getRanges()
                print("lower bound {} and upper bound range of the {}th set in all_exact_set".format(lb,ub,all_exact_set.index(S)))
                # plt.show()
            plt.savefig("Relu(O1)_minsum_Relu(X2)_exact.png") 
            plt.show()


            print("-----------------------------------------------------------------------------\n")
            # Plot Relu(O1)_minsum_Relu(X2)_approx_1, N sets
            for S in all_approx_set:
                plot_2D_Star(S, color=color3, show=False)
                print("plot the {}th set in all_approx_set".format(all_approx_set.index(S)))
                lb,ub = S.getRanges()
                print("lower bound {} and upper bound range of the {}th set in all_approx_set".format(lb,ub,all_approx_set.index(S)))
                # plt.show()
            plt.savefig("Relu(O1)_minsum_Relu(X2)_approx.png") 
            plt.show()


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
    test_RecurrentLayer.test_multiMinsum()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing Recurrent Layer Class: fails: {}, successfull: {}, \
    total tests: {}'.format(    test_RecurrentLayer.n_fails,
                                test_RecurrentLayer.n_tests - test_RecurrentLayer.n_fails,
                                test_RecurrentLayer.n_tests))

    

