import os
import numpy as np
import time

import matplotlib as mpl
import matplotlib.cm as cm

from skimage import measure, io
from skimage.color import rgb2gray

from pycheeger import SimpleSet, compute_cheeger
from tvsfw import SampledGaussianFilter
from tvsfw.utils import *

from math import sqrt, log

np.random.seed(0)

path = '/home/petit/Documents/these/code/python/tvsfw/experiments/exp4/'
saves_path = os.path.join(path, 'pacman/saves_1')

if not os.path.exists(saves_path):
    os.mkdir(saves_path)

im = rgb2gray(io.imread(os.path.join(path, 'imgs/pacman.png'))[:, :, :-1])

tmp_curve_1 = measure.find_contours(im, 0.1)[0] / im.shape[0]
tmp_curve_1 = 2.0 * (tmp_curve_1 - np.array([0.5, 0.5]))
curve1 = np.zeros_like(tmp_curve_1)
curve1[:, 0] = tmp_curve_1[:, 1]
curve1[:, 1] = -tmp_curve_1[:, 0]

i1 = np.argmin(np.linalg.norm(curve1 - np.array([0.0, 0.0]), axis=1))
i2 = np.argmin(np.linalg.norm(curve1 - np.array([0.683, 0.414]), axis=1))
i3 = np.argmin(np.linalg.norm(curve1 - np.array([0.683, -0.423]), axis=1))

curve1 = curve1[np.array([i for i in range(len(curve1)) if i % 40 == 0 or i == i1 or i == i2 or i == i3]), :]

tmp_curve_2 = measure.find_contours(im, 0.1)[1] / im.shape[0]
tmp_curve_2 = 2.0 * (tmp_curve_2 - np.array([0.5, 0.5]))
curve2 = np.zeros_like(tmp_curve_2)
curve2[:, 0] = tmp_curve_2[:, 1]
curve2[:, 1] = -tmp_curve_2[:, 0]
curve2 = curve2[::30, :]

norm = mpl.colors.Normalize(vmin=-1.1, vmax=1.1)
cmap = cm.bwr
m = cm.ScalarMappable(norm=norm, cmap=cmap)

E1 = SimpleSet(curve1, max_tri_area=0.004)
E2 = SimpleSet(curve2, max_tri_area=0.004)

u = SimpleFunction([WeightedIndicatorFunction(1, E1),
                    WeightedIndicatorFunction(-1, E2)])

np.save(os.path.join(saves_path, 'true_weights'), np.array([1, -1]))
np.save(os.path.join(saves_path, 'true_vertices_1'), curve1)
np.save(os.path.join(saves_path, 'true_vertices_2'), curve2)

plot_simple_function(u, m)

std = 0.03
std_noise = 4e-4
grid_size = 75
alpha = 0.05 * sqrt(2 * log(grid_size)) * std_noise

x_coarse, y_coarse = np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
grid = np.stack([X_coarse, Y_coarse], axis=2)

phi = SampledGaussianFilter(grid, std)

y = u.compute_obs(phi, version=0)

noise = np.random.normal(0, std_noise, y.shape)
noisy_y = y + noise

plot_obs(noisy_y, cmap, v_abs_max=np.max(1.1 * np.abs(noisy_y)))

np.save(os.path.join(saves_path, 'obs'), noisy_y)

u_hat = SimpleFunction([])

start = time.time()

step_size_tab = [2.0, 2.0]

# atoms = []
# weights = np.load(os.path.join(saves_path, 'sliding_weights_1.npy'))
#
# for i in range(1):
#     vertices = np.load(os.path.join(saves_path, 'sliding_vertices_1_{}.npy'.format(i+1)))
#     simple_set = SimpleSet(vertices, max_tri_area=0.004)
#     atoms.append(WeightedIndicatorFunction(weights[i], simple_set))
#
# u_hat = SimpleFunction(atoms)
#
# plot_simple_function(u_hat, m)

for iter in range(2):
    new_y = u_hat.compute_obs(phi, version=0)
    residual = noisy_y - new_y

    plot_obs(residual, cmap, v_abs_max=np.max(1.1 * np.abs(noisy_y)))

    np.save(os.path.join(saves_path, 'residual_{}'.format(iter + 1)), residual)

    eta = phi.apply_adjoint(10 * (noisy_y - new_y))

    E, obj_tab, grad_norm_tab = compute_cheeger(eta,
                                                max_tri_area_fm=1e-3, max_iter_fm=10000, plot_results_fm=True,
                                                num_boundary_vertices_ld=50, max_tri_area_ld=1e-3, step_size_ld=1e-2,
                                                max_iter_ld=100, convergence_tol_ld=1e-4, num_iter_resampling_ld=50,
                                                plot_results_ld=True)

    plt.plot(obj_tab)
    plt.show()
    plt.plot(grad_norm_tab)
    plt.show()

    np.save(os.path.join(saves_path, 'cheeger_obj_tab_{}'.format(iter + 1)), obj_tab)
    np.save(os.path.join(saves_path, 'cheeger_grad_norm_tab_{}'.format(iter + 1)), grad_norm_tab)
    np.save(os.path.join(saves_path, 'cheeger_vertices_{}'.format(iter + 1)), E.boundary_vertices)

    u_hat.extend_support(E)
    u_hat.fit_weights(noisy_y, phi, alpha)

    np.save(os.path.join(saves_path, 'fitted_weights_{}'.format(iter + 1)),
            np.array([atom.weight for atom in u_hat.atoms]))

    obj_tab, grad_norm_tab = u_hat.perform_sliding(noisy_y, phi, alpha, step_size_tab[iter],
                                                   200, None, None, 1e-9, 150, 25, 0.001)

    plt.plot(obj_tab)
    plt.show()
    plt.plot(grad_norm_tab)
    plt.show()

    plot_simple_function(u_hat, m)

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
