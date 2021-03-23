import time
import os

import matplotlib as mpl
import matplotlib.cm as cm

from math import sqrt, log
from pycheeger import SimpleSet, compute_cheeger, plot_simple_functions
from tvsfw import SampledGaussianFilter
from tvsfw.utils import *


np.random.seed(0)

# path = '/home/petit/Documents/these/code/python/tvsfw/experiments/exp1_ter/'
path = 'lala'
saves_path = os.path.join(path, 'saves')

if not os.path.exists(saves_path):
    os.mkdir(saves_path)

points1 = 1.1 * np.array([[-0.6, 0.8], [-0.75, 0.5], [-0.65, 0.31], [-0.75, 0.0], [-0.5, -0.5],
                          [-0.15, 0.15], [0.1, 0.5], [-0.25, 0.78]]) + np.array([0.0, -0.25])

points2 = 1.2 * np.array([[-0.2, 0.75], [-0.3, 0.5], [-0.2, 0.3], [0.25, 0.45], [0.3, 0.6],
                          [0.25, 0.8]]) + np.array([-0.05, -0.25])

points3 = np.array([[0.5, 0.25], [0.35, 0.1], [0.3, -0.1], [0.15, -0.3], [0.5, -0.5],
                    [0.8, 0.0]]) + np.array([0.1, -0.25])

weights = np.array([1.0, -2, 2.0])

norm = mpl.colors.Normalize(vmin=-np.max(np.abs(weights) + 0.1), vmax=np.max(np.abs(weights) + 0.1))
cmap = cm.bwr
m = cm.ScalarMappable(norm=norm, cmap=cmap)

std = 0.05
std_noise = 1e-3
grid_size = 75
alpha = 0.85 * sqrt(2 * log(grid_size)) * std_noise

curve1 = interpolate_points(points1)
curve2 = interpolate_points(points2)
curve3 = interpolate_points(points3)

E1 = SimpleSet(curve1, max_tri_area=0.001)
E2 = SimpleSet(curve2, max_tri_area=0.001)
E3 = SimpleSet(curve3, max_tri_area=0.001)

u = SimpleFunction([WeightedIndicatorFunction(weights[0], E1),
                    WeightedIndicatorFunction(weights[1], E2),
                    WeightedIndicatorFunction(weights[2], E3)])

np.save(os.path.join(saves_path, 'true_weights'), weights)

for i in range(u.num_atoms):
    np.save(os.path.join(saves_path, 'true_vertices_{}'.format(i + 1)), u.atoms[i].support.boundary_vertices)

x_coarse, y_coarse = np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
grid = np.stack([X_coarse, Y_coarse], axis=2)

phi = SampledGaussianFilter(grid, std)

y = u.compute_obs(phi, version=0)

noise = np.random.normal(0, std_noise, y.shape)
noisy_y = y + noise

plot_simple_function(u, m)
plot_obs(noisy_y, cmap, v_abs_max=np.max(1.1 * np.abs(noisy_y)))

np.save(os.path.join(saves_path, 'obs'), noisy_y)

u_hat = SimpleFunction([])

start = time.time()

for iter in range(3):
    new_y = u_hat.compute_obs(phi, version=0)
    residual = noisy_y - new_y

    plot_obs(residual, cmap, v_abs_max=np.max(1.1 * np.abs(noisy_y)))

    np.save(os.path.join(saves_path, 'residual_{}'.format(iter + 1)), residual)

    eta = phi.apply_adjoint(noisy_y - new_y)

    E, obj_tab, grad_norm_tab = compute_cheeger(eta,
                                                max_tri_area_fm=1e-3, max_iter_fm=10000, plot_results_fm=True,
                                                num_boundary_vertices_ld=50, max_tri_area_ld=1e-2, step_size_ld=1e-2,
                                                max_iter_ld=200, convergence_tol_ld=1e-6, num_iter_resampling_ld=60,
                                                plot_results_ld=True)

    np.save(os.path.join(saves_path, 'cheeger_obj_tab_{}'.format(iter + 1)), obj_tab)
    np.save(os.path.join(saves_path, 'cheeger_grad_norm_tab_{}'.format(iter + 1)), grad_norm_tab)
    np.save(os.path.join(saves_path, 'cheeger_vertices_{}'.format(iter + 1)), E.boundary_vertices)

    u_hat.extend_support(E)
    u_hat.fit_weights(noisy_y, phi, alpha)

    np.save(os.path.join(saves_path, 'fitted_weights_{}'.format(iter + 1)),
            np.array([atom.weight for atom in u_hat.atoms]))

    obj_tab, grad_norm_tab = u_hat.perform_sliding(noisy_y, phi, alpha, 0.15, 250, None, None, 1e-8, 60, 50, 0.004)

    plot_simple_function(u_hat, m)
    plot_simple_function(simple_function_diff(u, u_hat), m)
    plot_simple_functions(u, u_hat)

    print([u_hat.atoms[i].weight for i in range(u_hat.num_atoms)])

    np.save(os.path.join(saves_path, 'sliding_obj_tab_{}'.format(iter + 1)), obj_tab)
    np.save(os.path.join(saves_path, 'sliding_grad_norm_tab_{}'.format(iter + 1)), grad_norm_tab)

    np.save(os.path.join(saves_path, 'sliding_weights_{}'.format(iter + 1)),
            np.array([atom.weight for atom in u_hat.atoms]))

    for i in range(u_hat.num_atoms):
        np.save(os.path.join(saves_path, 'sliding_vertices_{}_{}'.format(iter + 1, i+1)),
                u_hat.atoms[i].support.boundary_vertices)

final_y = u_hat.compute_obs(phi, version=0)
np.save(os.path.join(saves_path, 'final_residual'), noisy_y - final_y)

end = time.time()

print("total time: {}".format(end - start))
