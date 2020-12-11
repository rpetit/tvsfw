import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, log
from pycheeger import disk, plot_simple_functions
from tvsfw import SampledGaussianFilter, WeightedIndicatorFunction, SimpleFunction, sfw


std = 0.1

E1 = disk(np.array([-0.4, 0.3]), 0.3, max_tri_area=0.01, num_vertices=40)
E2 = disk(np.array([0.2, -0.3]), 0.3, max_tri_area=0.01, num_vertices=30)
u = SimpleFunction([WeightedIndicatorFunction(1, E1), WeightedIndicatorFunction(0.7, E2)])

x_coarse, y_coarse = np.linspace(-1, 1, 30), np.linspace(-1, 1, 30)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
grid = np.stack([X_coarse, Y_coarse], axis=2)

phi = SampledGaussianFilter(grid, std)

y = u.compute_obs(phi, version=0)

vmax = np.max(y)

std_noise = 1e-3
noise = np.random.normal(0, std_noise, y.shape)
noisy_y = y + noise

alpha = sqrt(2 * log(y.size)) * std_noise

u_hat = sfw(phi, noisy_y, alpha, 4, u=u)

x_fine, y_fine = np.linspace(-1, 1, 300), np.linspace(-1, 1, 300)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
Z = np.array([[u(np.array([X_fine[i, j], Y_fine[i, j]])) for j in range(X_fine.shape[1])] for i in range(X_fine.shape[0])])
Z_hat = np.array([[u_hat(np.array([X_fine[i, j], Y_fine[i, j]])) for j in range(X_fine.shape[1])] for i in range(X_fine.shape[0])])

fig, axs = plt.subplots(3, figsize=(7, 20))

vmax = np.max(np.abs(Z))

im = axs[0].imshow(Z, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
axs[0].axis('off')

fig.colorbar(im, ax=axs[0])

im = axs[1].imshow(Z_hat, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
axs[1].axis('off')

fig.colorbar(im, ax=axs[1])

im = axs[2].imshow(Z - Z_hat, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
axs[2].axis('off')

fig.colorbar(im, ax=axs[2])

plt.show()
