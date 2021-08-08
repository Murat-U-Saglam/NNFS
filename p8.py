import numpy as np
from numpy.lib.function_base import average


softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0,1,1]

print(softmax_outputs[range(len(softmax_outputs)), class_targets]) 
#Gets the index of class targets for each sample softmax output, (Each row of the sotmax_outputs array)

neg_log  = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])    #Calculating loss check p7.py

average_loss = np.mean(neg_log)
print(average_loss)