import time

from math import sqrt, log
from pycheeger import SimpleSet, compute_cheeger, plot_simple_functions
from tvsfw import SampledGaussianFilter
from tvsfw.utils import *

import matplotlib as mpl
import matplotlib.cm as cm


path = '/home/petit/Documents/these/code/python/tvsfw/experiments/exp1/figures/'

points1 = 1.1 * np.array([[-0.6, 0.8], [-0.75, 0.5], [-0.65, 0.31], [-0.75, 0.0], [-0.5, -0.5],
                    [-0.15, 0.15], [0.1, 0.5], [-0.25, 0.78]]) + np.array([0.0, -0.25])

points2 = 1.2 * np.array([[-0.2, 0.75], [-0.3, 0.5], [-0.2, 0.3], [0.25, 0.45], [0.3, 0.6], [0.25, 0.8]]) + np.array([-0.05, -0.25])

points3 = np.array([[0.5, 0.25], [0.35, 0.1], [0.3, -0.1], [0.15, -0.3], [0.5, -0.5], [0.8, 0.0]]) + np.array([0.1, -0.25])

vals = np.array([1.0, -2, 2.0])

norm = mpl.colors.Normalize(vmin=-np.max(np.abs(vals)+0.1), vmax=np.max(np.abs(vals)+0.1))
cmap = cm.bwr
m = cm.ScalarMappable(norm=norm, cmap=cmap)

std = 0.05

curve1 = interpolate_points(points1)
curve2 = interpolate_points(points2)
curve3 = interpolate_points(points3)

E1 = SimpleSet(curve1, max_tri_area=0.001)
E2 = SimpleSet(curve2, max_tri_area=0.001)
E3 = SimpleSet(curve3, max_tri_area=0.001)

u = SimpleFunction([WeightedIndicatorFunction(vals[0], E1),
                    WeightedIndicatorFunction(vals[1], E2),
                    WeightedIndicatorFunction(vals[2], E3)])

plot_simple_function(u, m, save_path=path+'signal.png')

x_coarse, y_coarse = np.linspace(-1, 1, 75), np.linspace(-1, 1, 75)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
grid = np.stack([X_coarse, Y_coarse], axis=2)

phi = SampledGaussianFilter(grid, std)

y = u.compute_obs(phi, version=0)

std_noise = 1e-3
noise = np.random.normal(0, std_noise, y.shape)
noisy_y = y + noise

alpha = sqrt(2 * log(y.size)) * std_noise

plot_obs(noisy_y, cmap, v_abs_max=np.max(1.1 * np.abs(noisy_y)), save_path=path+'obs.png')

u_hat = SimpleFunction([])

start = time.time()

for iter in range(3):
    new_y = u_hat.compute_obs(phi, version=0)
    plot_obs(noisy_y - new_y, cmap,
             v_abs_max=np.max(1.1 * np.abs(noisy_y)), save_path=path+'residual_obs_{}.png'.format(iter))

    eta = phi.apply_adjoint(noisy_y - new_y)

    E, obj_tab, grad_norm_tab = compute_cheeger(eta,
                                                max_tri_area_fm=1e-3, max_iter_fm=10000, plot_results_fm=True,
                                                num_boundary_vertices_ld=50, max_tri_area_ld=1e-2,
                                                step_size_ld=1e-2, max_iter_ld=200, convergence_tol_ld=1e-6,
                                                num_iter_resampling_ld=60, plot_results_ld=True)

    u_hat.extend_support(E)
    u_hat.fit_weights(noisy_y, phi, alpha)

    obj_tab, grad_norm_tab = u_hat.perform_sliding(noisy_y, phi, alpha, 0.1, 200, 0.05, 0.5, 1e-8, 60, 25, 0.004)

    plt.plot(obj_tab)
    plt.show()

    plt.plot(grad_norm_tab)
    plt.show()

    plot_simple_function_bis(u_hat, m, save_path=path+'estim_{}.png'.format(iter))
    plot_simple_function_bis(simple_function_diff(u, u_hat), m, save_path=path+'diff_{}.png'.format(iter))
    plot_simple_functions(u, u_hat, save_path=path+'diff_support_{}.png'.format(iter))

    print([u_hat.atoms[i].weight for i in range(u_hat.num_atoms)])

final_y = u_hat.compute_obs(phi, version=0)
plot_obs(noisy_y - final_y, cmap,
         v_abs_max=np.max(1.1 * np.abs(noisy_y)), save_path=path+'residual_obs_final.png')

end = time.time()

print("total time: {}".format(end - start))
