import math

softmax_output = [0.7, 0.1, 0.2]  # Output from softmax activation function from the final nueron
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2]
         )

print(loss)

loss = -math.log(softmax_output[0])

print(loss)     # The target_output array 0 values cancel out therefore only the value of the target class index remains