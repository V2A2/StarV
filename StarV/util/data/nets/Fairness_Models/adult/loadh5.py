import tensorflow as tf
import os

# Specify the path to the .h5 model file
model_path = '/home/yuntao/Documents/Verification/StarV_Project/StarV/StarV/util/data/nets/Fairness_Models/adult/AC-1.h5'


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
        print(f"Weights shape of {layer.name}: {weights.shape}")
        print(f"Weights of {layer.name}: \n{weights}\n")
        print(f"Biases shape of {layer.name}: {biases.shape}")
        print(f"Biases of {layer.name}: \n{biases}\n")

        W.append(tf.transpose(weights))
        b.append(biases)
        print(f"Transposed weights shape: {tf.transpose(weights).shape}")
        print(f"Transposed weights: \n{tf.transpose(weights)}\n")

else:
    print(f"Model file not found at {model_path}")