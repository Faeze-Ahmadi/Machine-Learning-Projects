import numpy as np
import matplotlib.pyplot as plt
import random

# Data Generation
np.random.seed(42)

x_train = np.sort(np.array(random.sample(range(1, 100), 5)))
x_train = x_train[:, np.newaxis] / 100
y_train = -0.3 + 0.5 * np.sin(2 * np.pi * x_train)

x_test = np.linspace(0.01, 0.99, 99)[:, np.newaxis]
y_test = -0.3 + 0.5 * np.sin(2 * np.pi * x_test)

# Feature mapping , phi(x) = [1, x, x^2]
phi_train = np.concatenate(
    (np.ones((len(x_train),1)), x_train, x_train**2), axis=1
)
phi_test = np.concatenate(
    (np.ones((len(x_test),1)), x_test, x_test**2), axis=1
)

N, d = phi_train.shape

# Gradeint Descent (GD)
lr_gd = 0.1
epochs_gd = 500

w_gd = np.zeros((d,1))
loss_gd = []

for epoch in range(epochs_gd):
    y_pred = phi_train @ w_gd
    error = y_pred - y_train
    grad = (2/N) * phi_train.T @ error
    w_gd -= lr_gd * grad
    loss_gd.append(np.mean(error**2))

# Stochastic Gradeint Descent (SGD)
lr_sgd = 0.05
epochs_sgd = 200

w_sgd = np.zeros((d,1))
loss_sgd = []

for epoch in range(epochs_sgd):
    indices = np.random.permutation(N)
    for i in indices:
        xi = phi_train[i].reshape(1, -1)
        yi = y_train[i]
        error = xi @ w_sgd - yi
        grad = 2 * xi.T @ error
        w_sgd -= lr_sgd * grad

    total_error = phi_train @ w_sgd - y_train
    loss_sgd.append(np.mean(total_error**2))

# Plot Loss Curves
plt.figure(figsize=(8,5))
plt.plot(loss_gd, label="Gradient Descent")
plt.plot(loss_sgd, label="Stochastic Gradient Descent")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("GD vs SGD Convergence")
plt.grid()
plt.show()
