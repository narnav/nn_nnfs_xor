import numpy as np
import nnfs

nnfs.init()

def relu(x):
    return np.maximum(0, x)

# Example usage
x = np.array([-2, -1, 0, 1, 2])
print(relu(x))  # Output: [0 0 0 1 2]
