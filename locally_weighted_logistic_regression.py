import numpy as np
import matplotlib.pyplot as plt

features_path = "dataset_1/features.txt"
labels_path = "dataset_1/labels.txt"

X = np.loadtxt(features_path)
y = np.loadtxt(labels_path)  

# add intercept term (x_0 = 1) to X
m = X.shape[0]
X_train = np.hstack((np.ones((m, 1)), X))
y_train = y.reshape(-1, 1)

def sigmoid(z):
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))

def compute_weights(X_train, x_query, tau):
    # x_train[:, 1:] because we don't include the intercept term in distance calculation
    diff = X_train[:, 1:] - x_query[1:]
    distances_squared = np.sum(diff**2, axis=1)
    weights = np.exp(-distances_squared / (2 * tau**2))
    return np.diag(weights)

# locally weighted logistic regression
def fit_lwlr(X_train, y_train, x_query, tau, iterations=10):
    m, n = X_train.shape
    theta = np.zeros((n, 1))
    W = compute_weights(X_train, x_query, tau)
    
    for _ in range(iterations):
        z = np.dot(X_train, theta)
        h = sigmoid(z)
        gradient = np.dot(X_train.T, np.dot(W, (h - y_train)))
        S_diag = np.diag(W) * (h * (1 - h)).flatten()
        S = np.diag(S_diag)
        Hessian = np.dot(X_train.T, np.dot(S, X_train))
        Hessian += 1e-5 * np.eye(n)
        theta = theta - np.dot(np.linalg.inv(Hessian), gradient)
    return theta

def predict_lwlr(X_train, y_train, x_query, tau):
    theta = fit_lwlr(X_train, y_train, x_query, tau)
    prob = sigmoid(np.dot(x_query.reshape(1, -1), theta))
    return prob[0, 0]


tau = 0.8
print(f"training locally weighted logistic regression (tau={tau})")

x1_min, x1_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
x2_min, x2_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 50),
                       np.linspace(x2_min, x2_max, 50))
grid_probs = np.zeros(xx1.shape)

for i in range(xx1.shape[0]):
    for j in range(xx1.shape[1]):
        x_query = np.array([1.0, xx1[i, j], xx2[i, j]])
        grid_probs[i, j] = predict_lwlr(X_train, y_train, x_query, tau)

plt.figure(figsize=(8, 6))
plt.contourf(xx1, xx2, grid_probs, levels=[0, 0.5, 1], colors=['#ffb6c1', '#add8e6'], alpha=0.6)
plt.contour(xx1, xx2, grid_probs, levels=[0.5], colors='k', linewidths=2)
plt.scatter(X[y==0, 0], X[y==0, 1], c='pink', edgecolors='k', label='class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], c='yellow', edgecolors='k', label='class 1')
plt.title(f"locally weighted logistic regression (tau = {tau})")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.legend()
plt.show()
