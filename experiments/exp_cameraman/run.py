import time
import os

import matplotlib as mpl
import matplotlib.cm as cm

from math import sqrt, log
from pycheeger import SimpleSet, compute_cheeger, plot_simple_functions
from tvsfw import SampledGaussianFilter
from tvsfw.utils import *

from skimage import io

from scipy.ndimage import gaussian_filter

np.random.seed(0)

path = '/home/petit/Documents/these/code/python/tvsfw/experiments/exp_cameraman/'
saves_path = os.path.join(path, 'saves_6')

if not os.path.exists(saves_path):
    os.mkdir(saves_path)

im = io.imread(os.path.join(path, 'cameraman.png')) / 255
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.show()

blurred_im = gaussian_filter(im, sigma=2.0)[::4, ::4]
plt.imshow(blurred_im, cmap='gray')
plt.axis('off')
plt.show()

std = 2*2.0/256
std_noise = 1e-6
grid_size = blurred_im.shape[0]
alpha = 1.0

x_coarse, y_coarse = np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
grid = np.stack([X_coarse, Y_coarse], axis=2)

phi = SampledGaussianFilter(grid, std, normalization=True)

y = blurred_im[::-1, :]

noise = np.random.normal(0, std_noise, y.shape)
noisy_y = y + noise

v_abs_max = np.max(np.abs(noisy_y))
norm = mpl.colors.Normalize(vmin=0, vmax=v_abs_max)
cmap = cm.gray
m = cm.ScalarMappable(norm=norm, cmap=cmap)

u_hat = SimpleFunction([])

plot_obs(noisy_y, cmap, v_abs_max=np.max(1.1 * np.abs(noisy_y)))

np.save(os.path.join(saves_path, 'obs'), noisy_y)

atoms = []
# weights = np.load(os.path.join(saves_path, 'sliding_weights_{}.npy'.format(2 + 1)))
#
# for i in range(len(weights)):
#         vertices = np.load(os.path.join(saves_path, 'sliding_vertices_{}_{}.npy'.format(2 + 1, i + 1)))
#         simple_set = SimpleSet(vertices, max_tri_area=0.004)
#         atoms.append(WeightedIndicatorFunction(weights[i], simple_set))

u_hat = SimpleFunction(atoms)
plot_simple_function(u_hat, m)

for iter in range(10):
    new_y = u_hat.compute_obs(phi, version=0)
    residual = noisy_y - new_y

    plot_obs(residual, cmap, v_abs_max=np.max(1.1 * np.abs(noisy_y)))

    np.save(os.path.join(saves_path, 'residual_{}'.format(iter + 1)), residual)

    eta = phi.apply_adjoint(0.1 * (noisy_y - new_y))

    E, obj_tab, grad_norm_tab = compute_cheeger(eta,
                                                max_tri_area_fm=1e-3, max_iter_fm=10000, plot_results_fm=True,
                                                num_boundary_vertices_ld=50, max_tri_area_ld=1e-2, step_size_ld=1e-2,
                                                max_iter_ld=50, convergence_tol_ld=1e-4, num_iter_resampling_ld=50,
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

    obj_tab, grad_norm_tab = u_hat.perform_sliding(noisy_y, phi, alpha, 2e-4, 300, 0.1, 0.25, 1e-8, 200, 25, 0.004)

    plt.plot(obj_tab)
    plt.show()
    plt.plot(grad_norm_tab)
    plt.show()

    try:
        plot_simple_function(u_hat, m)
    except:
        pass

    print([u_hat.atoms[i].weight for i in range(u_hat.num_atoms)])

    np.save(os.path.join(saves_path, 'sliding_obj_tab_{}'.format(iter + 1)), obj_tab)
    np.save(os.path.join(saves_path, 'sliding_grad_norm_tab_{}'.format(iter + 1)), grad_norm_tab)

    np.save(os.path.join(saves_path, 'sliding_weights_{}'.format(iter + 1)),
            np.array([atom.weight for atom in u_hat.atoms]))

    for i in range(u_hat.num_atoms):
        np.save(os.path.join(saves_path, 'sliding_vertices_{}_{}'.format(iter + 1, i+1)),
                u_hat.atoms[i].support.boundary_vertices)
