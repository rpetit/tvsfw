import time
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, log
from pycheeger import compute_cheeger, disk, plot_simple_functions
from tvsfw import SampledGaussianFilter, WeightedIndicatorFunction, SimpleFunction


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


def compute_objective(u):
    obs = u.compute_obs(phi, version=0)
    tv = np.sum([np.abs(atom.weight) * atom.support.compute_perimeter() for atom in u.atoms])
    return 0.5 * np.linalg.norm(obs - y) ** 2 + alpha * tv


fig, ax = plt.subplots()

im = ax.imshow(noisy_y, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
ax.axis('off')

fig.colorbar(im, ax=ax)

plt.show()

start = time.time()

eta = phi.apply_adjoint(y)

print("cheeger")
E, obj_tab, grad_norm_tab = compute_cheeger(eta,
                                            max_tri_area_fm=1e-3, max_iter_fm=10000, plot_results_fm=True,
                                            num_boundary_vertices_ld=75, max_tri_area_ld=1e-2,
                                            step_size_ld=1e-2, max_iter_ld=300, convergence_tol_ld=1e-6,
                                            num_iter_resampling_ld=50, plot_results_ld=True)

u_hat = SimpleFunction([WeightedIndicatorFunction(0, E)])
u_hat.fit_weights(noisy_y, phi, alpha)
print(u_hat.atoms[0].weight)

print("sliding")
obj_tab, grad_norm_tab = u_hat.perform_sliding(noisy_y, phi, alpha, 0.5, 500, 0.5, 0.1, 1e-7, 50, 25, 0.004)

# print(u_hat.atoms[0].weight)
#
# plt.plot(obj_tab)
# plt.show()
#
# plt.plot(grad_norm_tab)
# plt.show()
#
# plot_simple_functions(u, u_hat)
#
# new_y = u_hat.compute_obs(phi, version=0)
#
# fig, ax = plt.subplots()
#
# im = ax.imshow(new_y - noisy_y, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
# ax.axis('off')
#
# fig.colorbar(im, ax=ax)
#
# plt.show()
#
# new_eta = phi.apply_adjoint(new_y - noisy_y)
#
# print("cheeger")
# E, obj_tab, grad_norm_tab = compute_cheeger(new_eta,
#                                             max_tri_area_fm=1e-3, max_iter_fm=10000, plot_results_fm=True,
#                                             num_boundary_vertices_ld=75, max_tri_area_ld=1e-2,
#                                             step_size_ld=1e-2, max_iter_ld=300, convergence_tol_ld=1e-6,
#                                             num_iter_resampling_ld=50, plot_results_ld=True)
#
# u_hat.extend_support(E)
#
# u_hat.fit_weights(noisy_y, phi, alpha)
# print([atom.weight for atom in u_hat.atoms])
#
# print("sliding")
# obj_tab, grad_norm_tab = u_hat.perform_sliding(noisy_y, phi, alpha, 0.3, 500, 0.1, 0.05, 1e-7, 50, 25, 0.004)
#
# final_y = u_hat.compute_obs(phi, version=0)
#
# fig, ax = plt.subplots()
#
# im = ax.imshow(final_y - y, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
# ax.axis('off')
#
# fig.colorbar(im, ax=ax)
#
# plt.show()
#
# end = time.time()
#
# print(end - start)
#
# plt.plot(obj_tab)
# plt.show()
#
# plt.plot(grad_norm_tab)
# plt.show()
#
# plot_simple_functions(u, u_hat)
#
# x_fine, y_fine = np.linspace(-1, 1, 200), np.linspace(-1, 1, 200)
# X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
# Z = np.array([[u(np.array([X_fine[i, j], Y_fine[i, j]])) for j in range(X_fine.shape[1])] for i in range(X_fine.shape[0])])
# Z_hat = np.array([[u_hat(np.array([X_fine[i, j], Y_fine[i, j]])) for j in range(X_fine.shape[1])] for i in range(X_fine.shape[0])])
#
# fig, axs = plt.subplots(3, figsize=(7, 20))
#
# vmax = np.max(np.abs(Z))
#
# im = axs[0].imshow(Z, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
# axs[0].axis('off')
#
# fig.colorbar(im, ax=axs[0])
#
# im = axs[1].imshow(Z_hat, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
# axs[1].axis('off')
#
# fig.colorbar(im, ax=axs[1])
#
# im = axs[2].imshow(Z - Z_hat, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
# axs[2].axis('off')
#
# fig.colorbar(im, ax=axs[2])
#
# plt.show()
