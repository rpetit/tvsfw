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
saves_path = os.path.join(path, 'saves_5')
plots_path = os.path.join(path, 'plots_5')

if not os.path.exists(saves_path):
    os.mkdir(saves_path)

if not os.path.exists(plots_path):
    os.mkdir(plots_path)

img = io.imread(os.path.join(path, 'cameraman.png')) / 255
blurred_img = gaussian_filter(img, sigma=2.0)[::4, ::4]

std = 2*2.0/256
std_noise = 1e-6
grid_size = blurred_img.shape[0]
alpha = 1e-2

x_coarse, y_coarse = np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
grid = np.stack([X_coarse, Y_coarse], axis=2)

phi = SampledGaussianFilter(grid, std, normalization=True)

y = blurred_img[::-1, :]

noise = np.random.normal(0, std_noise, y.shape)
noisy_y = y + noise

v_abs_max = np.max(np.abs(noisy_y))
norm = mpl.colors.Normalize(vmin=0, vmax=v_abs_max)
cmap = cm.gray
m = cm.ScalarMappable(norm=norm, cmap=cmap)

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect('equal')

im = ax.imshow(img, cmap=cmap, vmin=0, vmax=v_abs_max)

ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(plots_path, 'signal.png'), dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect('equal')

im = ax.imshow(y, origin='lower', cmap=cmap, vmin=0, vmax=v_abs_max)

ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(plots_path, 'obs.png'), dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

for iter in range(11):
    atoms = []
    weights = np.load(os.path.join(saves_path, 'sliding_weights_{}.npy'.format(iter + 1)))

    for i in range(len(weights)):
        vertices = np.load(os.path.join(saves_path, 'sliding_vertices_{}_{}.npy'.format(iter + 1, i + 1)))
        simple_set = SimpleSet(vertices, max_tri_area=0.004)
        atoms.append(WeightedIndicatorFunction(weights[i], simple_set))

    u_hat = SimpleFunction(atoms)
    plot_simple_function(u_hat, m, save_path=os.path.join(plots_path, 'u_hat_{}.png'.format(iter + 1)))
