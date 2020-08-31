import numpy as np
import matplotlib.pyplot as plt

from pycheeger import compute_cheeger, Disk, SimpleSet
from tvsfw import GaussianPolynomial, WeightedIndicatorFunction, SimpleFunction


std = 0.1
alpha = 1e-3

E1 = Disk(np.array([-0.4, 0]), 0.3)
E2 = Disk(np.array([0.4, -0.3]), 0.2)
u = SimpleFunction([WeightedIndicatorFunction(1, E1), WeightedIndicatorFunction(1, E2)])

x_coarse, y_coarse = np.linspace(-1, 1, 30), np.linspace(-1, 1, 30)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
grid = np.stack([X_coarse, Y_coarse], axis=2)


def aux(x, i, j):
    return np.exp(-np.linalg.norm(x - grid[i, j, :, np.newaxis], axis=0) ** 2 / (2 * std ** 2))


y = u.compute_obs(grid, aux)
noise = np.random.normal(0, 0.002, y.shape)
noisy_y = y + noise

plt.imshow(noisy_y, cmap='gray', origin='lower')
plt.axis('off')
plt.show()

eta = GaussianPolynomial(X_coarse, Y_coarse, y, std)

E, _, _ = compute_cheeger(eta, max_tri_area=0.001, max_primal_dual_iter=20000, max_iter=500, convergence_tol=1e-3)

u_hat = SimpleFunction([WeightedIndicatorFunction(0, E)])
u_hat.fit_weights(y, aux, alpha / y.size)

lala = u_hat.compute_obs(grid, aux)
new_eta = GaussianPolynomial(X_coarse, Y_coarse, lala - y, std)

E, _, _ = compute_cheeger(new_eta, max_tri_area=0.001, max_primal_dual_iter=20000, max_iter=500, convergence_tol=1e-3)

u_hat.extend_support(E)
u_hat.fit_weights(y, aux, alpha / y.size)

for atom in u.atoms:
    simple_set = atom.support
    x_curve = np.append(simple_set.boundary_vertices[:, 0], simple_set.boundary_vertices[0, 0])
    y_curve = np.append(simple_set.boundary_vertices[:, 1], simple_set.boundary_vertices[0, 1])
    plt.plot(x_curve, y_curve, color='black')

for atom in u_hat.atoms:
    simple_set = atom.support
    x_curve = np.append(simple_set.boundary_vertices[:, 0], simple_set.boundary_vertices[0, 0])
    y_curve = np.append(simple_set.boundary_vertices[:, 1], simple_set.boundary_vertices[0, 1])
    plt.plot(x_curve, y_curve, color='red')

plt.show()
