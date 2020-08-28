import time
import numpy as np
import matplotlib.pyplot as plt

from pycheeger import compute_cheeger, Disk, SimpleSet
from tvsfw import GaussianPolynomial, WeightedIndicatorFunction, SimpleFunction


std = 0.1

E1 = Disk(np.array([-0.4, 0]), 0.3)
E2 = Disk(np.array([0.4, -0.3]), 0.2)
u = SimpleFunction([WeightedIndicatorFunction(1, E1), WeightedIndicatorFunction(1, E2)])

x_coarse, y_coarse = np.linspace(-1, 1, 30), np.linspace(-1, 1, 30)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
grid = np.stack([X_coarse, Y_coarse], axis=2)


def aux(x, i, j):
    return np.exp(-np.linalg.norm(x - grid[i, j, :, np.newaxis], axis=0) ** 2 / (2 * std ** 2))


def compute_obs(f):
    y = np.zeros((30, 30))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for atom in f.atoms:
                y[i, j] += atom.weight * atom.support.compute_weighted_area(lambda x: f(x, i, j))

    return y


y = compute_obs(u)
noise = np.random.normal(0, 0.002, y.shape)
noisy_y = y + noise

plt.imshow(noisy_y, cmap='gray', origin='lower')
plt.axis('off')
plt.show()

eta = GaussianPolynomial(X_coarse, Y_coarse, y, std)

E, _, _ = compute_cheeger(eta, max_tri_area=0.001, max_primal_dual_iter=20000, max_iter=500, convergence_tol=1e-3)

u_hat = SimpleFunction([WeightedIndicatorFunction(0, E)])
u_hat.fit_weights()
