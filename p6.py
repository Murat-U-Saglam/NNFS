import numpy as np
import nnfs

# Input -> Exponentiation -> Normalise -> Output | Exponentiation -> Normalise = Softmax

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, 1.81, 0.2],
                 [1.41, 1.051, 0.026]]


exp_values = np.exp(layer_outputs)  # goes through the layer_outputs array and does e**layer_outputs[EachElement]
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # Makes an array that is the normalized value of the exponents

print(norm_values)

