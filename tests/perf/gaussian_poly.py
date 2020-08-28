import time
import numpy as np

from math import exp
from scipy.signal import fftconvolve
from numba import jit, prange

from tvsfw import GaussianPolynomial


std = 0.1

x_fine, y_fine = np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

u = np.where(np.logical_or((X_fine + 0.4)**2 + Y_fine**2 <= 0.1,
                           (X_fine - 0.4)**2 + (Y_fine + 0.3)**2 <= 0.05), 1, 0)

kernel = np.exp(-(X_fine**2 + Y_fine**2) / (2 * std**2))
v = fftconvolve(u, kernel, mode='same')
v = v / np.max(v)
w = v[::37, ::37]

noise = np.random.normal(0, 0.04, w.shape)

x_coarse, y_coarse = np.linspace(-1, 1, w.shape[0]), np.linspace(-1, 1, w.shape[1])
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)

y = w + noise

eta = GaussianPolynomial(X_coarse, Y_coarse, y, std)

x = np.random.random((2, 100))
lala = eta(x)

start = time.time()
for i in range(1000):
    lala = eta(x)
stop = time.time()

print(stop - start)


@jit(nopython=True, parallel=True)
def eta_aux1(x):
    squared_norm = (x[0] - X_coarse[0, 0]) ** 2 + (x[1] - Y_coarse[0, 0]) ** 2
    res = y[0, 0] * exp(-squared_norm / (2 * std ** 2))

    for i in prange(w.shape[0]):
        for j in prange(w.shape[1]):
            if i != 0 or j != 0:
                squared_norm = (x[0] - X_coarse[i, j]) ** 2 + (x[1] - Y_coarse[i, j]) ** 2
                res += y[i, j] * exp(-squared_norm / (2 * std ** 2))

    return res


@jit(nopython=True, parallel=True)
def eta_aux2(x, res):
    for k in prange(x.shape[1]):
        for i in prange(w.shape[0]):
            for j in prange(w.shape[1]):
                squared_norm = (x[0, k] - X_coarse[i, j]) ** 2 + (x[1, k] - Y_coarse[i, j]) ** 2
                res[k] += y[i, j] * exp(-squared_norm / (2 * std ** 2))


def eta(x):
    if x.ndim == 1:
        return eta_aux1(x) / 10
    else:
        res = np.zeros(x.shape[1])
        eta_aux2(x, res)
        return res / 10


x = np.random.random((2, 100))
lala = eta(x)

start = time.time()
for i in range(1000):
    lala = eta(x)
stop = time.time()

print(stop - start)
