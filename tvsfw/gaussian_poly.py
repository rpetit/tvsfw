import numpy as np

from math import exp
from numba import jit, prange


def generate_func1(x_grid, y_grid, weights, std):
    @jit(nopython=True, parallel=True)
    def aux(x):
        squared_norm = (x[0] - x_grid[0, 0]) ** 2 + (x[1] - y_grid[0, 0]) ** 2
        res = weights[0, 0] * exp(-squared_norm / (2 * std ** 2))

        for i in prange(weights.shape[0]):
            for j in prange(weights.shape[1]):
                if i != 0 or j != 0:
                    squared_norm = (x[0] - x_grid[i, j]) ** 2 + (x[1] - y_grid[i, j]) ** 2
                    res += weights[i, j] * exp(-squared_norm / (2 * std ** 2))

        return res

    return aux


def generate_func2(x_grid, y_grid, weights, std):
    @jit(nopython=True, parallel=True)
    def aux(x, res):
        for k in prange(x.shape[0]):
            for i in prange(weights.shape[0]):
                for j in prange(weights.shape[1]):
                    squared_norm = (x[k, 0] - x_grid[i, j]) ** 2 + (x[k, 1] - y_grid[i, j]) ** 2
                    res[k] += weights[i, j] * exp(-squared_norm / (2 * std ** 2))

    return aux


def generate_func3(x_grid, y_grid, weights, std):
    @jit(nopython=True, parallel=True)
    def aux(x, res):
        for k in prange(x.shape[0]):
            for l in prange(x.shape[1]):
                for i in prange(weights.shape[0]):
                    for j in prange(weights.shape[1]):
                        squared_norm = (x[k, l, 0] - x_grid[i, j]) ** 2 + (x[k, l, 1] - y_grid[i, j]) ** 2
                        res[k, l] += weights[i, j] * exp(-squared_norm / (2 * std ** 2))

    return aux


class GaussianPolynomial:
    def __init__(self, x_grid, y_grid, weights, std):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.weights = weights
        self.std = std

        self._aux1 = generate_func1(self.x_grid, self.y_grid, self.weights, self.std)
        self._aux2 = generate_func2(self.x_grid, self.y_grid, self.weights, self.std)
        self._aux3 = generate_func3(self.x_grid, self.y_grid, self.weights, self.std)

    def __call__(self, x):
        if x.ndim == 1:
            res = self._aux1(x)
        elif x.ndim == 2:
            res = np.zeros(x.shape[0])
            self._aux2(x, res)
        else:
            res = np.zeros((x.shape[0], x.shape[1]))
            self._aux3(x, res)
        return res
