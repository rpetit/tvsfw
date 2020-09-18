import time
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, log
from pycheeger import compute_cheeger, Disk, plot_simple_functions
from tvsfw import GaussianPolynomial, SampledGaussianKernel, WeightedIndicatorFunction, SimpleFunction


std = 0.1

E1 = Disk(np.array([-0.4, 0.3]), 0.3, max_tri_area=0.01, num_vertices=40)
E2 = Disk(np.array([0.2, -0.1]), 0.2, max_tri_area=0.01, num_vertices=30)
u = SimpleFunction([WeightedIndicatorFunction(1, E1), WeightedIndicatorFunction(1, E2)])

x_coarse, y_coarse = np.linspace(-1, 1, 30), np.linspace(-1, 1, 30)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
grid = np.stack([X_coarse, Y_coarse], axis=2)

phi = SampledGaussianKernel(grid, std)

y = u.compute_obs(phi, version=0)

vmax = np.max(y)

std_noise = 1e-3
noise = np.random.normal(0, std_noise, y.shape)
noisy_y = y + noise

alpha = sqrt(2 * log(y.size)) * std_noise

fig, ax = plt.subplots()

im = ax.imshow(noisy_y, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
ax.axis('off')

fig.colorbar(im, ax=ax)

plt.show()

start = time.time()

eta = GaussianPolynomial(grid, y, std)

E, _, _ = compute_cheeger(eta, max_tri_area=0.001, max_primal_dual_iter=20000, max_iter=500, convergence_tol=1e-2)

u_hat = SimpleFunction([WeightedIndicatorFunction(0, E)])
u_hat.fit_weights(noisy_y, phi, alpha / y.size)
obj_tab, grad_norm_tab = u_hat.perform_sliding(noisy_y, phi, alpha, 1.0, 500, 1e-7)

print(u_hat.atoms[0].weight)

plt.plot(obj_tab)
plt.show()

plt.plot(grad_norm_tab)
plt.show()

plot_simple_functions(u, u_hat)

new_y = u_hat.compute_obs(phi, version=0)

fig, ax = plt.subplots()

im = ax.imshow(new_y - noisy_y, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
ax.axis('off')

fig.colorbar(im, ax=ax)

plt.show()

new_eta = GaussianPolynomial(grid, new_y - noisy_y, std)

E, _, _ = compute_cheeger(new_eta, max_tri_area=0.001, max_primal_dual_iter=20000, max_iter=500, convergence_tol=1e-2)

u_hat.extend_support(E)

u_hat.fit_weights(noisy_y, phi, alpha / y.size)
obj_tab, grad_norm_tab = u_hat.perform_sliding(noisy_y, phi, alpha, 0.5, 500, 1e-7)

print([atom.weight for atom in u_hat.atoms])

final_y = u_hat.compute_obs(phi, version=0)

fig, ax = plt.subplots()

im = ax.imshow(final_y - y, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
ax.axis('off')

fig.colorbar(im, ax=ax)

plt.show()

end = time.time()

print(end - start)

plt.plot(obj_tab)
plt.show()

plt.plot(grad_norm_tab)
plt.show()

plot_simple_functions(u, u_hat)

u_hat.fit_weights(noisy_y, phi, alpha / y.size)
print([atom.weight for atom in u_hat.atoms])
