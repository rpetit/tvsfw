import numpy as np
import quadpy

from numpy import exp
from numba import jit, prange

from pycheeger import GaussianPolynomial


def generate_square_aux(grid, std, normalization):
    scheme = quadpy.c2.get_good_scheme(3)
    scheme_weights = scheme.weights
    scheme_points = (1 + scheme.points.T) / 2
    scale = - 1 / (2 * std ** 2)

    @jit(nopython=True, parallel=True)
    def aux(grid_size, res):
        h = 2 / grid_size
        for i in prange(grid_size):
            for j in prange(grid_size):
                for n in range(scheme_weights.size):
                    x = -1 + h * (scheme_points[n, 0] + i)
                    y = -1 + h * (scheme_points[n, 1] + j)
                    for k in range(grid.shape[0]):
                        squared_norm = (x - grid[k, 0]) ** 2 + (y - grid[k, 1]) ** 2
                        res[i, j, k] += scheme_weights[n] * exp(scale * squared_norm)

                res[i, j] *= h ** 2

        if normalization:
            res *= (-scale)

    return aux


def generate_triangle_aux(grid, std, normalization):
    scheme = quadpy.t2.get_good_scheme(5)
    scheme_weights = scheme.weights
    scheme_points = scheme.points.T
    scale = - 1 / (2 * std ** 2)

    @jit(nopython=True, parallel=True)
    def aux(meshes, res):
        for i in range(len(meshes)):
            for j in prange(len(meshes[i])):
                a = np.sqrt((meshes[i, j, 1, 0] - meshes[i, j, 0, 0]) ** 2 + (meshes[i, j, 1, 1] - meshes[i, j, 0, 1]) ** 2)
                b = np.sqrt((meshes[i, j, 2, 0] - meshes[i, j, 1, 0]) ** 2 + (meshes[i, j, 2, 1] - meshes[i, j, 1, 1]) ** 2)
                c = np.sqrt((meshes[i, j, 2, 0] - meshes[i, j, 0, 0]) ** 2 + (meshes[i, j, 2, 1] - meshes[i, j, 0, 1]) ** 2)
                p = (a + b + c) / 2
                area = np.sqrt(p * (p - a) * (p - b) * (p - c))

                for m in range(grid.shape[0]):
                    for n in range(scheme_weights.size):
                        x = scheme_points[n, 0] * meshes[i, j, 0, 0] + \
                            scheme_points[n, 1] * meshes[i, j, 1, 0] + \
                            scheme_points[n, 2] * meshes[i, j, 2, 0]

                        y = scheme_points[n, 0] * meshes[i, j, 0, 1] + \
                            scheme_points[n, 1] * meshes[i, j, 1, 1] + \
                            scheme_points[n, 2] * meshes[i, j, 2, 1]

                        squared_norm = (x - grid[m, 0]) ** 2 + (y - grid[m, 1]) ** 2
                        res[i, j, m] += scheme_weights[n] * exp(scale * squared_norm)

                    res[i, j, m] *= area

        if normalization:
            res *= (-scale)

    return aux


def generate_line_aux(grid, std, normalization):
    scheme = quadpy.c1.gauss_patterson(3)
    scheme_weights = scheme.weights
    scheme_points = (scheme.points + 1) / 2
    scale = - 1 / (2 * std ** 2)

    @jit(nopython=True, parallel=True)
    def aux(curves, mask, res):
        for i in range(len(curves)):
            num_vertices_i = np.sum(mask[i])
            for j in prange(num_vertices_i):
                edge_length = np.sqrt((curves[i, (j + 1) % num_vertices_i, 0] - curves[i, j, 0]) ** 2 +
                                      (curves[i, (j + 1) % num_vertices_i, 1] - curves[i, j, 1]) ** 2)

                for n in range(scheme_weights.size):
                    x = scheme_points[n] * curves[i, j] + (1 - scheme_points[n]) * curves[i, (j + 1) % num_vertices_i]

                    for m in range(grid.shape[0]):
                        squared_norm = (x[0] - grid[m, 0]) ** 2 + (x[1] - grid[m, 1]) ** 2
                        res[i, j, m, 0] += scheme_weights[n] * scheme_points[n] * exp(scale * squared_norm)

                res[i, j, :, 0] *= edge_length / 2

                edge_length = np.sqrt((curves[i, j, 0] - curves[i, j - 1, 0]) ** 2 +
                                      (curves[i, j, 1] - curves[i, j - 1, 1]) ** 2)

                for n in range(scheme_weights.size):
                    x = scheme_points[n] * curves[i, j] + (1 - scheme_points[n]) * curves[i, j - 1]

                    for m in range(grid.shape[0]):
                        squared_norm = (x[0] - grid[m, 0]) ** 2 + (x[1] - grid[m, 1]) ** 2
                        res[i, j, m, 1] += scheme_weights[n] * scheme_points[n] * exp(scale * squared_norm)

                res[i, j, :, 1] *= edge_length / 2

        if normalization:
            res *= (-scale)

    return aux


class SampledGaussianFilter:
    def __init__(self, grid, std, normalization=False):
        self.grid = grid
        self.std = std
        self.normalization = normalization

        self._square_aux = generate_square_aux(self.grid, self.std, self.normalization)
        self._triangle_aux = generate_triangle_aux(self.grid, self.std, self.normalization)
        self._line_aux = generate_line_aux(self.grid, self.std, self.normalization)

    @property
    def grid_size(self):
        return len(self.grid)

    def integrate_on_pixel_grid(self, grid_size):
        res = np.zeros((grid_size, grid_size, self.grid_size))
        self._square_aux(grid_size, res)
        return res

    def integrate_on_meshes(self, meshes):
        res = np.zeros((len(meshes), len(meshes[0]), self.grid_size))
        self._triangle_aux(meshes, res)
        return res

    def integrate_on_curves(self, curves):
        max_num_vertices = max([len(vertices) for vertices in curves])
        res = np.zeros((len(curves), max_num_vertices, self.grid_size, 2))
        self._line_aux(curves, res)
        return [res[i, :len(curves[i])] for i in range(len(curves))]

    def apply_adjoint(self, weights):
        return GaussianPolynomial(self.grid, weights, self.std, normalization=self.normalization)
