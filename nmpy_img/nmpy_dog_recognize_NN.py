import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder, image_size):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img = img.resize((image_size, image_size))
        img = np.array(img)
        if img is not None:
            images.append(img)
    return np.array(images)

# Load the dataset
image_size = 64  # Example size, you can change this
train_images = load_images_from_folder('./dogs/', image_size)

# Normalize the images
train_images = train_images / 255.0

# Flatten the images
train_images = train_images.reshape((train_images.shape[0], -1))


# Initialize model parameters
input_size = image_size * image_size * 3  # For RGB images
hidden_size = 64
output_size = 1  # Assuming binary classification (dog or not dog)

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Forward and Backward Propagation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def compute_loss(Y, A2):
    m = Y.shape[0]
    loss = -np.mean(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return loss

def backward_propagation(X, Y, Z1, A1, Z2, A2):
    m = X.shape[0]
    
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2

# Train the Model
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def train(X, Y, learning_rate=0.1, epochs=100):
    global W1, b1, W2, b2
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X)
        loss = compute_loss(Y, A2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example labels (0 for not a dog, 1 for a dog)
train_labels = np.ones((train_images.shape[0], 1))  # Assuming all images are of dogs

train(train_images, train_labels, learning_rate=0.1, epochs=100)

# Evaluate the Model
def predict(X):
    _, _, _, A2 = forward_propagation(X)
    return (A2 > 0.5).astype(int)

train_predictions = predict(train_images)
train_accuracy = np.mean(train_predictions == train_labels)
print(f"Training accuracy: {train_accuracy:.4f}")
