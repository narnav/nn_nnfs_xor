import numpy as np
import matplotlib.pyplot as plt
import math 

# V4 - P6 
layer_outputs= [[4.8,1.21,-2.85],[1.7,-2.21,2.35],[-3.1,4.1,2.38]]

# exp big numbers will couse overflow
exp_values =np.exp(layer_outputs)

norm_values=exp_values/ np.sum(exp_values,axis=1,keepdims=True)

# print(np.sum(exp_values,axis=1,keepdims=True))
print(norm_values)