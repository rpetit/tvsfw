import matplotlib.pyplot as plt
import numpy as np
import itertools

from scipy import interpolate
from shapely.geometry import Polygon
from .simple_function import SimpleFunction, WeightedIndicatorFunction

import matplotlib.ticker as tick
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def simple_function_diff(u, v):
    atoms = []

    for atom in u.atoms:
        atoms.append(atom)
    for atom in v.atoms:
        atoms.append(WeightedIndicatorFunction(-atom.weight, atom.support))

    return SimpleFunction(atoms)


def plot_simple_function(f, m, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')
    ax.fill([-1, 1, 1, -1], [-1, -1, 1, 1], color=m.to_rgba(0))

    n = f.num_atoms
    idx_set = set(list(range(n)))

    for k in range(1, n + 1):
        for indices in itertools.combinations(idx_set, k):
            p = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
            weight = 0
            for i in indices:
                points_i = f.atoms[i].support.boundary_vertices
                p_i = Polygon([tuple(points_i[k]) for k in range(len(points_i))])
                p = p.intersection(p_i)
                weight += f.atoms[i].weight

            if not p.is_empty:
                coords = list(p.boundary.coords)
                x = np.array([coords[i][0] for i in range(len(coords))])
                y = np.array([coords[i][1] for i in range(len(coords))])

                ax.fill(x, y, color=m.to_rgba(weight))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')

    cbar = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04, format=tick.FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=30)
    cbar.set_ticks([-1, 0, 1])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()


def plot_supports(u1, u2, save_path=None):
    fig, ax = plt.subplots(figsize=(1690/300, 1690/300), dpi=300)
    ax.set_aspect('equal')

    for i in range(2):
        if i == 0:
            u = u1
            boundary_color = 'black'
        else:
            u = u2
            boundary_color = 'red'

        for atom in u.atoms:
            simple_set = atom.support

            x_curve = np.append(simple_set.boundary_vertices[:, 0], simple_set.boundary_vertices[0, 0])
            y_curve = np.append(simple_set.boundary_vertices[:, 1], simple_set.boundary_vertices[0, 1])

            ax.plot(x_curve, y_curve, color=boundary_color)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()


def periodize(x):
    return np.append(x, [x[0]], axis=0)


def interpolate_points(input_points):
    x = periodize(input_points[:, 0])
    y = periodize(input_points[:, 1])

    tck, u = interpolate.splprep([x, y], s=0, per=0)

    unew = np.linspace(0, 1.0, 100)

    out = interpolate.splev(unew, tck)

    return np.stack([out[0][:-1], out[1][:-1]], axis=1)


def plot_obs(y, cmap, v_abs_max=None, save_path=None):
    if v_abs_max is None:
        v_abs_max = np.max(y)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')

    im = ax.imshow(y, origin='lower', cmap=cmap, vmin=-v_abs_max, vmax=v_abs_max)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=tick.FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=30)

    ax.axis('off')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()
