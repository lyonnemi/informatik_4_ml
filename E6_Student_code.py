import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(np.int64)

X = X / 255.0

num_classes = 10
Y = np.eye(num_classes, dtype=np.float32)[y]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.14285, random_state=42
)

input_size = 784
h1_size = 64
h2_size = 64
output_size = 10

np.random.seed(42)

Theta1 = np.random.randn(h1_size, input_size + 1).astype(np.float32) * np.sqrt(2.0 / input_size)
Theta2 = np.random.randn(h2_size, h1_size + 1).astype(np.float32) * np.sqrt(2.0 / h1_size)
Theta3 = np.random.randn(output_size, h2_size + 1).astype(np.float32) * np.sqrt(2.0 / h2_size)

Theta1[:, 0] = 0.0
Theta2[:, 0] = 0.0
Theta3[:, 0] = 0.0


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):# to return a step function for positive inputs
    return Z > 0


def softmax(Z):
    shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def forward(X, Theta1, Theta2, Theta3):
    # TODO 1: forward propagation
    # Return output (y_hat).
    A1 = np.hstack([np.ones((X.shape[0], 1)), X])
    Z2 = A1 @ Theta1.T
    A2 = relu(Z2)

    A2 = np.hstack([np.ones((A2.shape[0], 1)), A2])
    Z3 = A2 @ Theta2.T
    A3 = relu(Z3)

    A3 = np.hstack([np.ones((A3.shape[0], 1)), A3])
    Z4 = A3 @ Theta3.T
    y_hat = softmax(Z4)

    return y_hat



def cross_entropy_cost(y_hat, Y_true):
    # TODO 2: cross-entropy cost
    # Compute the average loss over all training examples.
    m = Y_true.shape[0]
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    J = -1 / m * np.sum(Y_true * np.log(y_hat))
    return J

def backward(X, Y_true, Theta1, Theta2, Theta3):
    # TODO 3: backpropagation
    # Compute dTheta1, dTheta2, and dTheta3.
    m = X.shape[0]

    A1 = np.hstack([np.ones((m, 1)), X])
    Z2 = A1 @ Theta1.T
    A2 = relu(Z2)

    A2_bias = np.hstack([np.ones((A2.shape[0], 1)), A2])
    Z3 = A2_bias @ Theta2.T
    A3 = relu(Z3)

    A3_bias = np.hstack([np.ones((A3.shape[0], 1)), A3])
    Z4 = A3_bias @ Theta3.T
    y_hat = softmax(Z4)

    delta3 = y_hat - Y_true
    dTheta3 = (1 / m) * delta3.T @ A3_bias

    delta2 = (delta3 @ Theta3[:, 1:]) * relu_derivative(Z3)
    dTheta2 = (1 / m) * delta2.T @ A2_bias

    delta1 = (delta2 @ Theta2[:, 1:]) * relu_derivative(Z2)
    dTheta1 = (1 / m) * delta1.T @ A1

    return dTheta1, dTheta2, dTheta3


learning_rate = 0.1
epochs = 100
costs = []

for epoch in range(epochs):
    y_hat = forward(X_train, Theta1, Theta2, Theta3)
    cost = cross_entropy_cost(y_hat, Y_train)
    costs.append(cost)

    dTheta1, dTheta2, dTheta3 = backward(X_train, Y_train, Theta1, Theta2, Theta3)

    # TODO: gradient descent update of Theta1, Theta2, Theta3
    # Theta1 = ...
    # Theta2 = ...
    # Theta3 = ...
    Theta1 -= learning_rate * dTheta1
    Theta2 -= learning_rate * dTheta2
    Theta3 -= learning_rate * dTheta3

    print("Epoch", epoch + 1, "/", epochs, "cost =", cost)


plt.figure()
plt.plot(costs)
plt.xlabel("Epoch")
plt.ylabel("cost")
plt.title("Training cost")
plt.show()


def predict(X, Theta1, Theta2, Theta3):
    y_hat = forward(X, Theta1, Theta2, Theta3)
    return np.argmax(y_hat, axis=1)


y_test_true = np.argmax(Y_test, axis=1)
y_test_pred = predict(X_test, Theta1, Theta2, Theta3)

acc = np.mean(y_test_pred == y_test_true)
print("Test Accuracy =", round(acc * 100, 2), "%")


def show_predictions(X, y_true, y_pred, num=10):
    plt.figure(figsize=(12, 3))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap="gray")
        plt.title("T:" + str(y_true[i]) + "\nP:" + str(y_pred[i]))
        plt.axis("off")
    plt.show()


show_predictions(X_test, y_test_true, y_test_pred, num=10)
