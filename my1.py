import numpy as np

# Sigmoid function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Generate the dataset
def generate_data(n_samples):
    x1 = np.random.randint(0, 100, n_samples)
    x2 = np.random.randint(0, 100, n_samples)
    y = (x1 == x2).astype(int)
    x = np.column_stack((x1, x2))
    return x, y

# Generate training and test data
n_train = 10000
n_test = 2000
x_train, y_train = generate_data(n_train)
# print(x_train)
# for ind,xxx in x_train:
#     print(xxx,y_train[0][ind])
# exit()
x_test, y_test = generate_data(n_test)

# Normalize input data
x_train = x_train / 100.0
x_test = x_test / 100.0

# Initialize parameters
n_x = 2     # Number of input features
n_h = 10    # Number of neurons in hidden layer
n_y = 1     # Number of output neurons
lr = 0.1    # Learning rate
epochs = 1000  # Number of epochs

# Weights and biases initialization
np.random.seed(1)
w1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))
w2 = np.random.randn(n_y, n_h)
b2 = np.zeros((n_y, 1))

# Training the model
for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(w1, x_train.T) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    # Compute the loss (mean squared error)
    m = y_train.shape[0]
    loss = (1 / m) * np.sum((a2 - y_train) ** 2)

    # Backward propagation
    dz2 = a2 - y_train
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    
    dz1 = np.dot(w2.T, dz2) * sigmoid_derivative(z1)
    dw1 = (1 / m) * np.dot(dz1, x_train)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    # Update parameters
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2

    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Evaluate the model
z1 = np.dot(w1, x_test.T) + b1
a1 = sigmoid(z1)
z2 = np.dot(w2, a1) + b2
a2 = sigmoid(z2)
predictions = (a2 > 0.5).astype(int)

accuracy = np.mean(predictions == y_test) * 100
print(f'Test Accuracy: {accuracy:.2f}%')

# Example predictions
example_pairs = np.array([[30, 30]])
example_pairs = example_pairs / 100.0
z1 = np.dot(w1, example_pairs.T) + b1
a1 = sigmoid(z1)
z2 = np.dot(w2, a1) + b2
a2 = sigmoid(z2)
print(a2)
example_predictions = (a2 > 0.5).astype(int)
print(f'Predictions for {example_pairs * 100}: {example_predictions.ravel()}')
