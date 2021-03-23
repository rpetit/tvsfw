import os
import matplotlib as mpl
import matplotlib.cm as cm

from pycheeger import SimpleSet, plot_simple_functions
from tvsfw.utils import *

path = '/home/petit/Documents/these/code/python/tvsfw/experiments/exp1/'
saves_path = os.path.join(path, 'saves')
plots_path = os.path.join(path, 'plots')

if not os.path.exists(plots_path):
    os.mkdir(plots_path)

weights = np.load(os.path.join(saves_path, 'true_weights.npy'))

norm = mpl.colors.Normalize(vmin=-np.max(np.abs(weights) + 0.1), vmax=np.max(np.abs(weights) + 0.1))
cmap = cm.bwr
m = cm.ScalarMappable(norm=norm, cmap=cmap)

boundary_vertices = []

for i in range(len(weights)):
    boundary_vertices.append(np.load(os.path.join(saves_path, 'true_vertices_{}.npy'.format(i+1))))

u = SimpleFunction([WeightedIndicatorFunction(weights[i], SimpleSet(boundary_vertices[i])) for i in range(len(weights))])

noisy_y = np.load(os.path.join(saves_path, 'obs.npy'))

plot_simple_function(u, m, save_path=os.path.join(plots_path, 'signal.png'))
plot_obs(noisy_y, cmap, v_abs_max=np.max(1.1 * np.abs(noisy_y)), save_path=os.path.join(plots_path, 'obs.png'))

u_hat = SimpleFunction([])

# for iter in range(3):
#     # plot residual
#     residual = np.load(os.path.join(saves_path, 'residual_{}.npy'.format(iter+1)))
#     plot_obs(residual, cmap, v_abs_max=np.max(1.1 * np.abs(noisy_y)),
#              save_path=os.path.join(plots_path, 'residual_obs_{}.png'.format(iter+1)))
#
#     # plot cheeger
#     obj_tab = np.load(os.path.join(saves_path, 'cheeger_obj_tab_{}.npy'.format(iter + 1)))
#     grad_norm_tab = np.load(os.path.join(saves_path, 'cheeger_grad_norm_tab_{}.npy'.format(iter + 1)))
#
#     # plt.plot(obj_tab)
#     # plt.show()
#     #
#     # plt.plot(grad_norm_tab)
#     # plt.show()
#
#     cheeger_vertices = np.load(os.path.join(saves_path, 'cheeger_vertices_{}.npy'.format(iter + 1)))
#     cheeger_set = SimpleSet(cheeger_vertices)
#
#     fitted_weights = np.load(os.path.join(saves_path, 'fitted_weights_{}.npy'.format(iter + 1)))
#
#     former_atoms = u_hat.atoms
#     new_atoms = [WeightedIndicatorFunction(fitted_weights[i], former_atoms[i].support) for i in range(len(former_atoms))]
#     new_atoms.append(WeightedIndicatorFunction(fitted_weights[-1], cheeger_set))
#     u_hat = SimpleFunction(new_atoms)
#
#     plot_simple_function(u_hat, m, save_path=os.path.join(plots_path, 'cheeger_{}.png'.format(iter + 1)))
#
#     # plot sliding
#     obj_tab = np.load(os.path.join(saves_path, 'sliding_obj_tab_{}.npy'.format(iter+1)))
#     grad_norm_tab = np.load(os.path.join(saves_path, 'sliding_grad_norm_tab_{}.npy'.format(iter + 1)))
#
#     # plt.plot(obj_tab)
#     # plt.show()
#     #
#     # plt.plot(grad_norm_tab)
#     # plt.show()
#
#     new_weights = np.load(os.path.join(saves_path, 'sliding_weights_{}.npy'.format(iter + 1)))
#     new_atoms = []
#     for i in range(u_hat.num_atoms):
#         vertices = np.load(os.path.join(saves_path, 'sliding_vertices_{}_{}.npy'.format(iter + 1, i + 1)))
#         simple_set = SimpleSet(vertices)
#         new_atoms.append(WeightedIndicatorFunction(new_weights[i], simple_set))
#
#     u_hat = SimpleFunction(new_atoms)
#
#     plot_simple_function(u_hat, m, save_path=os.path.join(plots_path, 'sliding_{}.png'.format(iter + 1)))
#     plot_simple_function(simple_function_diff(u, u_hat), m, save_path=os.path.join(plots_path, 'diff_{}.png'.format(iter + 1)))
#     plot_simple_functions(u, u_hat, save_path=os.path.join(plots_path, 'diff_support_{}.png'.format(iter + 1)))
#
#     print([u_hat.atoms[i].weight for i in range(u_hat.num_atoms)])

# plot_obs(noisy_y - final_y, cmap,
#          v_abs_max=np.max(1.1 * np.abs(noisy_y)), save_path=path+'residual_obs_final.png')

# bandeaux itérations
u_hat_tab = []

for iter in range(3):
    weights = np.load(os.path.join(saves_path, 'sliding_weights_{}.npy'.format(iter + 1)))
    atoms = []
    i = 0
    while os.path.exists(os.path.join(saves_path, 'sliding_vertices_{}_{}.npy'.format(iter + 1, i + 1))):
        vertices = np.load(os.path.join(saves_path, 'sliding_vertices_{}_{}.npy'.format(iter + 1, i + 1)))
        simple_set = SimpleSet(vertices)
        atoms.append(WeightedIndicatorFunction(weights[i], simple_set))
        i += 1

    u_hat = SimpleFunction(atoms)
    u_hat_tab.append(u_hat)

plot_iter(u_hat_tab, u, m, [0.83, 0.29, 0.01, 0.425], save_path=os.path.join(plots_path, 'iter.png'))
