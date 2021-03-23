import os
import matplotlib as mpl
import matplotlib.cm as cm

from pycheeger import SimpleSet, plot_simple_functions
from tvsfw.utils import *

path = '/home/petit/Documents/these/code/python/tvsfw/experiments/exp4/pacman'
saves_path = os.path.join(path, 'saves_1')
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

weights = np.load(os.path.join(saves_path, 'sliding_weights_2.npy'))

atoms = []
for i in range(2):
    vertices = np.load(os.path.join(saves_path, 'sliding_vertices_{}_{}.npy'.format(2, i + 1)))
    simple_set = SimpleSet(vertices)
    atoms.append(WeightedIndicatorFunction(weights[i], simple_set))

    u_hat = SimpleFunction(atoms)

plot_simple_function(u_hat, m, save_path=os.path.join(plots_path, 'u_hat_final.png'))
plot_simple_function(simple_function_diff(u, u_hat), m, save_path=os.path.join(plots_path, 'diff_final.png'))
plot_simple_functions(u, u_hat, save_path=os.path.join(plots_path, 'diff_support_final.png'))

print([u_hat.atoms[i].weight for i in range(u_hat.num_atoms)])
