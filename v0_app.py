import sys
import numpy as np
import matplotlib

# V0 , p1-p3
inputs =[1.2,5.1,2.1]
weights =[3.1,2.1,8.7]
bias =3

# very basic NN
output = inputs[0]*weights[0]+inputs[1]*weights[1]+inputs[2]*weights[2]+bias
print(output)

