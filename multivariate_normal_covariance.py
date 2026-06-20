import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set random seed for reproducibility
np.random.seed(42)

# Paramters
n_samples = 1000
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Case 1: Spherical (equal variances, no correlation)
Sigma1 = np.array([[1, 0], [0, 1]])
mu = np.array([0, 0])
Z1 = np.exp(-0.5 * ((X - mu[0])**2 + (Y - mu[1])**2))

# Case 2: Anisotropic (different variances, no correlation)
Sigma2 = np.array([[4, 0], [0, 1]])  # variances 4 and 1
Z2 = np.exp(-0.5 * (X**2/4 + Y**2))

# Case 3: Correlated (positive correlation)
Sigma3 = np.array([[1, 0.8], [0.8, 1]])
inv_Sigma3 = np.linalg.inv(Sigma3)
Z3 = np.exp(-0.5 * (inv_Sigma3[0, 0]*X**2 + 2 *
            inv_Sigma3[0, 1]*X*Y + inv_Sigma3[1, 1]*Y**2))

# Sample data
samples1 = np.random.multivariate_normal(mu, Sigma1, n_samples)
samples2 = np.random.multivariate_normal(mu, Sigma2, n_samples)
samples3 = np.random.multivariate_normal(mu, Sigma3, n_samples)

# Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Contour plots
axes[0, 0].contour(X, Y, Z1, levels=10, cmap='Blues')
axes[0, 0].set_title("Case 1: Spherical")
axes[0, 0].set_xlim(-4, 4)
axes[0, 0].set_ylim(-4, 4)

axes[0, 1].contour(X, Y, Z2, levels=10, cmap='Blues')
axes[0, 1].set_title("Case 2: Anisotropic")
axes[0, 1].set_xlim(-4, 4)
axes[0, 1].set_ylim(-4, 4)

axes[0, 2].contour(X, Y, Z3, levels=10, cmap='Blues')
axes[0, 2].set_title("Case 3: Correlated")
axes[0, 2].set_xlim(-4, 4)
axes[0, 2].set_ylim(-4, 4)

# Scatter plots
axes[1, 0].scatter(samples1[:, 0], samples1[:, 1], alpha=0.7, s=10)
axes[1, 0].set_title("Samples: Case 1")
axes[1, 0].set_xlim(-4, 4)
axes[1, 0].set_ylim(-4, 4)

axes[1, 1].scatter(samples2[:, 0], samples2[:, 1], alpha=0.7, s=10)
axes[1, 1].set_title("Samples: Case 2")
axes[1, 1].set_xlim(-4, 4)
axes[1, 1].set_ylim(-4, 4)

axes[1, 2].scatter(samples3[:, 0], samples3[:, 1], alpha=0.7, s=10)
axes[1, 2].set_title("Samples: Case 3")
axes[1, 2].set_xlim(-4, 4)
axes[1, 2].set_ylim(-4, 4)

plt.tight_layout()
plt.show()
