import nnfs
import numpy as np
from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)  # Creating random weight per neuron
        self.biases = np.zeros((1, n_neurons))  # Creating random bias per neuron

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases  # Dot product of the layer


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)  # taking the list of inputs and appending them to the output array if inputs > 0 else append 0


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])