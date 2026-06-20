import numpy as np
import matplotlib.pyplot as plt

# load data
features_path = "dataset_1/features.txt"
labels_path = "dataset_1/labels.txt"

X = np.loadtxt(features_path)
y = np.loadtxt(labels_path)

# add bias term (column of ones)
X = np.hstack([np.ones((X.shape[0], 1)), X])  # shape = (m, n+1)

# sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# cost functoin
def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5  # to avoid log(0)
    cost = -(1/m) * np.sum(y * np.log(h + epsilon) +
                           (1 - y) * np.log(1 - h + epsilon))
    return cost

# gradient function
def gradient(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (1/m) * (X.T @ (h - y))
    return grad

# gradient descent
def logistic_regression(X, y, alpha=0.1, iterations=5000):
    theta = np.zeros(X.shape[1])
    for i in range(iterations):
        grad = gradient(theta, X, y)
        theta -= alpha * grad
        if i % 500 == 0:
            print(f"iteration {i}: cost = {cost_function(theta, X, y):.4f}")
    return theta

# train
theta = logistic_regression(X, y)

print("\nfinal parameters (theta):")
print(theta)

# plot
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 1], X[y == 0, 2], c='pink', label='class 0')
plt.scatter(X[y == 1, 1], X[y == 1, 2], c='yellow', label='class 1')

x_min, x_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5
y_min, y_max = X[:, 2].min()-0.5, X[:, 2].max()+0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

grid = np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()]
probs = sigmoid(grid @ theta).reshape(xx.shape)

plt.contour(xx, yy, probs, levels=[0.5], cmap="Greys", vmin=0, vmax=1)
plt.title("simple logistic regression (part a)")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.legend()
plt.show()
