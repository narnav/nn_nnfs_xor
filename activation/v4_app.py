import numpy as np
import matplotlib.pyplot as plt
import math 

# V4 - P6 
layer_outputs= [4.8,1.21,2.385]

E =math.e
print(E)

# long version
exp_values=[]
for output in layer_outputs:
    exp_values.append(E ** output)

# short version
exp_values=np.exp(layer_outputs)

# print(layer_outputs)
# print(exp_values)

# long version
norm_values=[]
norm_base = sum(exp_values)
for val in exp_values:
    norm_values.append(val /norm_base)

# short values
norm_values=exp_values/np.sum(exp_values)

print(norm_values)
print(sum(norm_values)) # convert to 1
