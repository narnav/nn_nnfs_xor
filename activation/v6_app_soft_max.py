import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
# V6 - P6 
np.random.seed(0)

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights =.1 * np.random.randn(n_inputs,n_neurons) #this order redant the transpose
        self.biases =np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights) +self.biases

class Activation_ReLU:
    def forward(self,inputs ):
        self.output= np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self,inputs ):
        exp_values= np.exp(inputs -np.max(inputs,axis=1,keepdims=True))
        probabilities =exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output =probabilities

X, y = spiral_data(100,3)

dense1= Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2= Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])