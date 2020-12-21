import numpy as np

np.random.seed(0)

# Training data set
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)  # Creating random weight per neuron
        self.biases = np.zeros((1, n_neurons))  # Creating random bias per neuron

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases  # Dot product of the layer


layer1 = Layer_Dense(4, 5)  # Shape of each layer, index0 = input of the layer, index1 = the amount of neurons on that layer
layer2 = Layer_Dense(5, 2)

layer1.forward(X)  # passing the data through the layer the parameter being the input layer
# print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
