import nnfs
import numpy as np
from nnfs.datasets import spiral_data

nnfs.init()

# Training data set
X, y = spiral_data(100, 3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)  # Creating random weight per neuron
        self.biases = np.zeros((1, n_neurons))  # Creating random bias per neuron

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases  # Dot product of the layer


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)  # taking the list of inputs and appending them to the output array if inputs > 0 else append 0


layer1 = Layer_Dense(2, 5)  # Shape of each layer, index0 = input of the layer, index1 = the amount of neurons on that layer
activation1 = Activation_ReLU()

layer1.forward(X)  # passing the data through the layer the parameter being the input layer
activation1.forward(X)

print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)