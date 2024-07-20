import sys
import numpy as np
import matplotlib

# V1 - P4
# same numbers
np.random.seed(0)
# shape (3,4)
X =[[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-.8]]

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights =.1 * np.random.randn(n_inputs,n_neurons) #this order redant the transpose
        self.biases =np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights) +self.biases

class Activation_ReLU:
    def forward(self,inputs ):
        self.output= np.maximum(0,inputs)

layer1=Layer_Dense(4,5)
layer2=Layer_Dense(5,2)
layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)