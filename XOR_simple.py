import numpy as np

# XOR input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 1],
              [1, 0]])

y = np.array([[0],
              [1],
              [0],
              [1]])

# np.random.seed(42)
input_size = 2
hidden_size = 2
output_size = 1

learning_rate = 0.1
epochs = 10000

# Weights and biases
W1 = np.random.uniform(size=(input_size, hidden_size))
b1 = np.random.uniform(size=(1, hidden_size))
W2 = np.random.uniform(size=(hidden_size, output_size))
b2 = np.random.uniform(size=(1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train():
    global  W1,b1 ,W2 ,b2
    for epoch in range(epochs):
        # Forward pass
        # print(X)
        hidden_input = np.dot(X, W1) + b1
        
        hidden_output = sigmoid(hidden_input)
        
        final_input = np.dot(hidden_output, W2) + b2
        final_output = sigmoid(final_input)
        
        # Compute the error
        error = y - final_output
        
        # Backward pass
        d_final_output = error * sigmoid_derivative(final_output)
        error_hidden_layer = d_final_output.dot(W2.T)
        d_hidden_output = error_hidden_layer * sigmoid_derivative(hidden_output)
        
        # Update weights and biases
        W2 += hidden_output.T.dot(d_final_output) * learning_rate
        b2 += np.sum(d_final_output, axis=0, keepdims=True) * learning_rate
        W1 += X.T.dot(d_hidden_output) * learning_rate
        b1 += np.sum(d_hidden_output, axis=0, keepdims=True) * learning_rate
        
        if epoch % 1000 == 0:
            loss = np.mean(np.abs(error))
            # print(f'Epoch {epoch}, Loss:{round(loss, 3)}')

train()
def test(test_input):
    hidden_input = np.dot(test_input, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    print(test_input,"Predicted outputs:", final_output ,final_output[0][0]>.5 )

test(np.array([[1, 0]]))
test(np.array([[1, 1]]))