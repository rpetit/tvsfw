from pymesh import triangle

import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, log
from pycheeger import SimpleSet
from tvsfw import SampledGaussianKernel, WeightedIndicatorFunction, SimpleFunction

from scipy import interpolate

import matplotlib as mpl
import matplotlib.cm as cm

norm = mpl.colors.Normalize(vmin=-2, vmax=2)
cmap = cm.bwr
m = cm.ScalarMappable(norm=norm, cmap=cmap)

fig, ax = plt.subplots()

x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
X, Y = np.meshgrid(x, y)

r = 0.7
eps = 0.1

Z = np.where(np.logical_and(r**2 - eps <= X ** 2 + Y ** 2, X ** 2 + Y ** 2 <= r**2 + eps), 1, -1)

circle = plt.Circle((0.05, 0.05), r, color='black', alpha=0.5)
ax.add_artist(circle)

ax.pcolor(X, Y, Z, cmap='bwr', alpha=0.3)

ax.axis('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axis('off')

# plt.savefig('grid.png', dpi=200, bbox_inches='tight', pad_inches=0)
plt.show()

fig, ax = plt.subplots()

r = 0.7

t = np.linspace(0, 2 * np.pi, 21)
x = r * np.cos(t)
y = r * np.sin(t)

circle = plt.Circle((0.0, 0.0), r, color='black', alpha=0.5)
ax.add_artist(circle)

ax.pcolor(X, Y, -np.ones_like(X), cmap='bwr', alpha=0.3)

ax.plot(x, y, 'r', linewidth=2, alpha=0.5)
ax.scatter(x, y, color='red', s=40.0, alpha=0.5)

ax.axis('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axis('off')

plt.savefig('polygon.png', dpi=200, bbox_inches='tight', pad_inches=0)
plt.show()


# points1 = np.array([[-0.6, 0.8], [-0.75, 0.5], [-0.65, 0.31], [-0.75, 0.0], [-0.5, -0.5],
#                     [-0.15, 0.15], [0.1, 0.5], [-0.25, 0.78], [-0.6, 0.8]]) + np.array([0.0, -0.1])
#
# points2 = np.array([[-0.2, 0.75], [-0.3, 0.5], [-0.2, 0.3], [0.25, 0.45], [0.3, 0.6], [0.25, 0.8], [-0.2, 0.75]]) + np.array([0.0, -0.1])
#
# points3 = np.array([[0.5, 0.25], [0.35, 0.1], [0.3, -0.1], [0.15, -0.3], [0.5, -0.5], [0.8, 0.0], [0.5, 0.25]]) + np.array([0.0, -0.1])
#
# points_tab = [points1, points2, points3]
# curves_tab = []
#
# vals = [1, -2, 1.5]
#
# plt.figure(figsize=(20, 20))
#
# for i in range(3):
#     points = points_tab[i]
#
#     x = points[:, 0]
#     y = points[:, 1]
#
#     tck, u = interpolate.splprep([x, y], s=0)
#
#     unew = np.arange(0, 1.01, 0.01)
#
#     out = interpolate.splev(unew, tck)
#
#     curves_tab.append(out)
#     plt.plot(out[0], out[1], color=m.to_rgba(vals[i]), linewidth=5.0)
#
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.axis('off')
# plt.axis('equal')
# plt.savefig('grad.png', bbox_inches='tight', pad_inches=0)
# plt.show()
#
# std = 0.05
#
# points1 = points1[:-1]
# points2 = points2[:-1]
# points3 = points3[:-1]
#
# E1 = SimpleSet(points1, max_area=0.001)
# E2 = SimpleSet(points2, max_area=0.001)
# E3 = SimpleSet(points3, max_area=0.001)
#
# u = SimpleFunction([WeightedIndicatorFunction(1, E1),
#                     WeightedIndicatorFunction(-2, E2),
#                     WeightedIndicatorFunction(1.5, E3)])
#
# from shapely.geometry import Polygon
#
# p1 = Polygon([(curves_tab[0][0][i], curves_tab[0][1][i]) for i in range(len(curves_tab[0][0]))])
# p2 = Polygon([(curves_tab[1][0][i], curves_tab[1][1][i]) for i in range(len(curves_tab[1][0]))])
# intersect_coords = list(p1.intersection(p2).boundary.coords)
#
# plt.figure(figsize=(20, 20))
#
# plt.fill([-1, 1, 1, -1], [-1, -1, 1, 1], color=m.to_rgba(0))
# plt.fill(curves_tab[0][0], curves_tab[0][1], color=m.to_rgba(1))
# plt.fill(curves_tab[1][0], curves_tab[1][1], color=m.to_rgba(-2))
# plt.fill(curves_tab[2][0], curves_tab[2][1], color=m.to_rgba(1.5))
#
# p3_x = np.array([intersect_coords[i][0] for i in range(len(intersect_coords))])
# p3_y = np.array([intersect_coords[i][1] for i in range(len(intersect_coords))])
#
# plt.fill(p3_x, p3_y, color=m.to_rgba(-1))
#
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.axis('off')
# plt.axis('equal')
# plt.savefig('signal.png', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()
#
# x_coarse, y_coarse = np.linspace(-1, 1, 75), np.linspace(-1, 1, 75)
# X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)
# grid = np.stack([X_coarse, Y_coarse], axis=2)
#
# phi = SampledGaussianKernel(grid, std)
#
# y = u.compute_obs(phi, version=0)
#
# vmax = np.max(y)
#
# std_noise = 3e-3
# noise = np.random.normal(0, std_noise, y.shape)
# noisy_y = y + noise
#
# alpha = sqrt(2 * log(y.size)) * std_noise
# alpha = 5e-4
#
# fig, ax = plt.subplots(figsize=(20, 20))
#
# im = ax.imshow(noisy_y, origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax)
# plt.axis('off')
# plt.savefig('noisy_signal.png', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()