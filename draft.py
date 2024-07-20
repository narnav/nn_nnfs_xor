import numpy as np
import matplotlib.pyplot as plt

DISPLAY_CHART = False

# XOR inputs
x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
# XOR outputs
y = np.array([[0, 1, 1, 0]])

# Number of inputs
n_x = 2
# Number of neurons in output layer
n_y = 1
# Number of neurons in hidden layer
n_h = 2
# Total training examples
m = x.shape[1]

# Learning rate
lr = 0.1

# Optionally set a seed for reproducibility
# np.random.seed(2)  # Comment out this line for different results each run

# Define weight matrices for neural network
w1 = np.random.rand(n_h, n_x)   # Weight matrix for hidden layer
w2 = np.random.rand(n_y, n_h)   # Weight matrix for output layer

print("Weight matrix for hidden layer (w1):")
print(w1)
print("\nWeight matrix for output layer (w2):")
print(w2)
