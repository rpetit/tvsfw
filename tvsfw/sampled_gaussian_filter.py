import numpy as np

from math import exp
from numba import jit, prange


def generate_func1(grid, std):
    @jit(nopython=True, parallel=True)
    def aux(x, res):
        for i in prange(x.shape[0]):
            for j in prange(grid.shape[0]):
                for k in prange(grid.shape[1]):
                    squared_norm = (x[i, 0] - grid[j, k, 0]) ** 2 + (x[i, 1] - grid[j, k, 1]) ** 2
                    res[i, j, k] = exp(-squared_norm / (2 * std ** 2))

    return aux


class SampledGaussianFilter:
    def __init__(self, grid, std):
        self.grid = grid
        self.std = std

        self._aux = generate_func1(self.grid, self.std)

    def __call__(self, x):
        if x.ndim == 1:
            tmp = np.zeros((1, self.grid.shape[0], self.grid.shape[1]))
            self._aux(np.reshape(x, (1, 2)), tmp)
            res = tmp[0]
        else:
            res = np.zeros((x.shape[0], self.grid.shape[0], self.grid.shape[1]))
            self._aux(x, res)
        return res

    def apply_adjoint(self, weights):
        return GaussianPolynomial(self.grid, weights, self.std)


def generate_func2(grid, weights, std):
    @jit(nopython=True, parallel=True)
    def aux(x, res):
        for i in prange(x.shape[0]):
            for j in prange(weights.shape[0]):
                for k in prange(weights.shape[1]):
                    squared_norm = (x[i, 0] - grid[j, k, 0]) ** 2 + (x[i, 1] - grid[j, k, 1]) ** 2
                    res[i] += weights[j, k] * exp(-squared_norm / (2 * std ** 2))

    return aux


class GaussianPolynomial:
    def __init__(self, grid, weights, std):
        self.grid = grid
        self.weights = weights
        self.std = std

        self._aux = generate_func2(self.grid, self.weights, self.std)

    def __call__(self, x):
        if x.ndim == 1:
            tmp = np.zeros(1)
            self._aux(np.reshape(x, (1, 2)), tmp)
            res = tmp[0]
        else:
            res = np.zeros(x.shape[0])
            self._aux(x, res)
        return res
