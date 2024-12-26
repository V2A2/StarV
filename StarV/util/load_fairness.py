"""
load fairness models (ReLU networks) for testing/evaluation, 
including Adult, Bank, German networks.
Yuntao Li, 3/22/2024
"""
import os
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.net.network import NeuralNetwork
import tensorflow as tf
import numpy as np
from scipy.io import savemat
# import torch
# import math
# import keras2onnx

# def export_fairness_keras2onnx(id, type):
#     if type =='adult':
#         t = 'AC'
#     elif type == 'bank':
#         t = 'BM'
#     else:
#         t = 'GC'
    
#     model_name = '{t}-{id}.h5'
#     cur_path = os.path.dirname(__file__)
#     data_path = '/data/nets/Fairness_Models/adult/'
#     model_path = cur_path + data_path + model_name
#     onnx_path = cur_path + data_path + f'{t}-{id}.onnx'
#     # Check if the model file exists
#     if os.path.exists(model_path):

#         # Load the model
#         model = tf.keras.models.load_model(model_path)
#         # Print the model summary to see its structure
#         model.summary()
#         onnx_model = keras2onnx.convert_keras(model, model.name)
#         keras2onnx.save_model(onnx_model, onnx_path)

def load_fairness(id, type, show=False):
    """Load network dataset
    """
    if type =='adult':
        t = 'AC'
    elif type == 'bank':
        t = 'BM'
    else:
        t = 'GC'

    model_name = f'{t}-{id}'
    cur_path = os.path.dirname(__file__)
    model_path = f'{cur_path}/data/nets/Fairness_Models/{type}/{model_name}.h5'

    # Check if the model file exists
    if os.path.exists(model_path):

        # Load the model
        model = tf.keras.models.load_model(model_path)
        # Print the model summary to see its structure
        if show:
            model.summary()

        layers = []
        # Access the weights and biases of each layer
        n_layers = len(model.layers)
        for i, layer in enumerate(model.layers):
            weights = layer.get_weights()[0]
            biases = layer.get_weights()[1]

            L1 = fullyConnectedLayer(weights.T, biases[:, None])
            layers.append(L1)
            if i < n_layers-1:
                L2 = ReLULayer()
                layers.append(L2)

        return NeuralNetwork(layers, net_type=f'ffnn_{model_name}')

    print(f"Model file not found at {model_path}")



def load_fairness_adult(id):
    """Load network from adult dataset
    """
    model_name = 'AC-{}.h5'.format(id)
    cur_path = os.path.dirname(__file__)
    model_path = cur_path + '/data/nets/Fairness_Models/adult/' + model_name

    # Check if the model file exists
    if os.path.exists(model_path):

        # Load the model
        model = tf.keras.models.load_model(model_path)
        # Print the model summary to see its structure
        model.summary()
        W = []
        b = []

        # Access the weights and biases of each layer
        for layer in model.layers:
            weights = layer.get_weights()[0]
            biases = layer.get_weights()[1]
            # print(f"Weights shape of {layer.name}: {weights.shape}")
            # print(f"Weights of {layer.name}: \n{weights}\n")
            # print(f"Biases shape of {layer.name}: {biases.shape}")
            # print(f"Biases of {layer.name}: \n{biases}\n")

            W.append(np.transpose(weights))
            b.append(biases)

        # mdic = {"W": W, "b": b}
        # savemat("{}.mat".format(model_name), mdic)

        # print("b: ", W)
        # print("b: ", b)

        layers = []
        for i in range(0, len(W)-1):
            Wi = W[i]
            # print(f"Wi shape: {Wi.shape}")
            bi = b[i]
            bi = bi.reshape(bi.shape[0],)
            L1 = fullyConnectedLayer(Wi, bi)
            # print(f"L1: {L1}")
            layers.append(L1)
            L2 = ReLULayer()
            layers.append(L2)

        Wi = W[len(W)-1]
        bi = b[len(W)-1]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        net = NeuralNetwork(layers, net_type='ffnn_AC-{}'.format(id))
        
    else:
        print(f"Model file not found at {model_path}")

    return net

def load_fairness_bank(id):
    """Load network from bank dataset
    """
    model_name = 'BM-{}.h5'.format(id)
    cur_path = os.path.dirname(__file__)
    model_path = cur_path + '/data/nets/Fairness_Models/bank/' + model_name

    # Check if the model file exists
    if os.path.exists(model_path):

        # Load the model
        model = tf.keras.models.load_model(model_path)
        # Print the model summary to see its structure
        model.summary()
        W = []
        b = []

        # Access the weights and biases of each layer
        for layer in model.layers:
            weights = layer.get_weights()[0]
            biases = layer.get_weights()[1]
            # print(f"Weights shape of {layer.name}: {weights.shape}")
            # print(f"Weights of {layer.name}: \n{weights}\n")
            # print(f"Biases shape of {layer.name}: {biases.shape}")
            # print(f"Biases of {layer.name}: \n{biases}\n")

            W.append(np.transpose(weights))
            b.append(biases)

        # mdic = {"W": W, "b": b}
        # savemat("{}.mat".format(model_name), mdic)
        # print("b: ", W)
        # print("b: ", b)

        layers = []
        for i in range(0, len(W)-1):
            Wi = W[i]
            # print(f"Wi shape: {Wi.shape}")
            bi = b[i]
            bi = bi.reshape(bi.shape[0],)
            L1 = fullyConnectedLayer(Wi, bi)
            # print(f"L1: {L1}")
            layers.append(L1)
            L2 = ReLULayer()
            layers.append(L2)

        Wi = W[len(W)-1]
        bi = b[len(W)-1]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        net = NeuralNetwork(layers, net_type='ffnn_BM-{}'.format(id))
        
    else:
        print(f"Model file not found at {model_path}")

    return net

def load_fairness_german(id):
    """Load network from bank dataset
    """
    model_name = 'GC-{}.h5'.format(id)
    cur_path = os.path.dirname(__file__)
    model_path = cur_path + '/data/nets/Fairness_Models/german/' + model_name

    # Check if the model file exists
    if os.path.exists(model_path):

        # Load the model
        model = tf.keras.models.load_model(model_path)
        # Print the model summary to see its structure
        model.summary()
        W = []
        b = []

        # Access the weights and biases of each layer
        for layer in model.layers:
            weights = layer.get_weights()[0]
            biases = layer.get_weights()[1]
            # print(f"Weights shape of {layer.name}: {weights.shape}")
            # print(f"Weights of {layer.name}: \n{weights}\n")
            # print(f"Biases shape of {layer.name}: {biases.shape}")
            # print(f"Biases of {layer.name}: \n{biases}\n")

            W.append(np.transpose(weights))
            b.append(biases)

        # mdic = {"W": W, "b": b}
        # savemat("{}.mat".format(model_name), mdic)
        # print("b: ", W)
        # print("b: ", b)

        layers = []
        for i in range(0, len(W)-1):
            Wi = W[i]
            # print(f"Wi shape: {Wi.shape}")
            bi = b[i]
            bi = bi.reshape(bi.shape[0],)
            L1 = fullyConnectedLayer(Wi, bi)
            # print(f"L1: {L1}")
            layers.append(L1)
            L2 = ReLULayer()
            layers.append(L2)

        Wi = W[len(W)-1]
        bi = b[len(W)-1]
        bi = bi.reshape(bi.shape[0],)
        L1 = fullyConnectedLayer(Wi, bi)
        layers.append(L1)
        net = NeuralNetwork(layers, net_type='ffnn_GM-{}'.format(id))
        
    else:
        print(f"Model file not found at {model_path}")

    return net
