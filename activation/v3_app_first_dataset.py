import sys
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
# V3 - P4 -5
np.random.seed(0)

X, y = spiral_data(100,3)
# print(X,y)

plt.scatter(X[:,0],X[:,1])
# plt.show()

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights =.1 * np.random.randn(n_inputs,n_neurons) #this order redant the transpose
        self.biases =np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights) +self.biases

class Activation_ReLU:
    def forward(self,inputs ):
        self.output= np.maximum(0,inputs)

layer1=Layer_Dense(2,5)
activation1 =Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

# If many values are becoming zero (dying) when using the ReLU activation function,
#  you can adjust the bias to prevent this issue. 