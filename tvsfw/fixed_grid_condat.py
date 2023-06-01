import numpy as np
import scipy.ndimage
import scipy.sparse

from numba import jit, prange

import matplotlib.pyplot as plt


@jit(nopython=True, parallel=True)
def one_two_norm(x):
    res = 0
    for i in prange(x.shape[1]):
        res += np.sqrt(x[0, i] ** 2 + x[1, i] ** 2)
    return res


@jit(nopython=True, parallel=True)
def prox_one_two_norm_aux(x, alpha, res):
    for i in prange(x.shape[1]):
        norm_i = np.sqrt(x[0, i] ** 2 + x[1, i] ** 2)
        if norm_i <= alpha:
            res[:, i] = 0
        else:
            res[:, i] = x[:, i] * (1 - alpha / norm_i)


def prox_one_two_norm(x, alpha):
    res = np.zeros_like(x)
    prox_one_two_norm_aux(x, alpha, res)
    return res


def make_grad_op(n):
    data = []
    row = []
    col = []

    for i in range(n-1):
        for j in range(n):
            row.append(n*i+j)
            col.append(n*i+j)
            data.append(-1)

            row.append(n*i+j)
            col.append(n*(i+1)+j)
            data.append(1)

    for j in range(n):
        row.append(n*(n-1)+j)
        col.append(n*(n-1)+j)
        data.append(0)

    grad_op_1 = scipy.sparse.coo_matrix((data, (row, col)), shape=(n*n, n*n))

    data = []
    row = []
    col = []

    for i in range(n):
        for j in range(n-1):
            row.append(n*i+j)
            col.append(n*i+j)
            data.append(-1)

            row.append(n*i+j)
            col.append(n*i+j+1)
            data.append(1)

    for i in range(n):
        row.append(n*i+(n-1))
        col.append(n*i+(n-1))
        data.append(0)

    grad_op_2 = scipy.sparse.coo_matrix((data, (row, col)), shape=(n*n, n*n))

    grad_op = scipy.sparse.vstack([grad_op_1, grad_op_2])

    return grad_op.tocsc()


def make_l_op(n):
    data = []
    row = []
    col = []

    for i in range(n):
        for j in range(n):
            data.append(1)
            row.append(n*i+j)
            col.append(n*i+j)

    l_op_1_1 = scipy.sparse.coo_matrix((data, (row, col)), shape=(n*n, 2*n*n))

    data = []
    row = []
    col = []

    for i in range(n):
        for j in range(n-1):
            data.append(1/4)
            row.append(n*i+j)
            col.append(n*i+j)

            data.append(1/4)
            row.append(n*i+j)
            col.append(n*i+(j+1))

            if i != 0:
                data.append(1/4)
                row.append(n*i+j)
                col.append(n*(i-1)+j)

                data.append(1/4)
                row.append(n*i+j)
                col.append(n*(i-1)+(j+1))

    l_op_2_1 = scipy.sparse.coo_matrix((data, (row, col)), shape=(n*n, 2*n*n))

    data = []
    row = []
    col = []

    for i in range(n):
        for j in range(n):
            data.append(1/2)
            row.append(n*i+j)
            col.append(n*i+j)

            if i != 0:
                data.append(1/2)
                row.append(n*i+j)
                col.append(n*(i-1)+j)

    l_op_3_1 = scipy.sparse.coo_matrix((data, (row, col)), shape=(n*n, 2*n*n))

    data = []
    row = []
    col = []

    for i in range(n-1):
        for j in range(n):
            data.append(1/4)
            row.append(n*i+j)
            col.append(n*n+n*i+j)

            data.append(1/4)
            row.append(n*i+j)
            col.append(n*n+n*(i+1)+j)

            if j != 0:
                data.append(1/4)
                row.append(n*i+j)
                col.append(n*n+n*i+(j-1))

                data.append(1/4)
                row.append(n*i+j)
                col.append(n*n+n*(i+1)+(j-1))

    l_op_1_2 = scipy.sparse.coo_matrix((data, (row, col)), shape=(n*n, 2*n*n))

    data = []
    row = []
    col = []

    for i in range(n):
        for j in range(n):
            data.append(1)
            row.append(n*i+j)
            col.append(n*n+n*i+j)

    l_op_2_2 = scipy.sparse.coo_matrix((data, (row, col)), shape=(n*n, 2*n*n))

    data = []
    row = []
    col = []

    for i in range(n):
        for j in range(n):
            data.append(1/2)
            row.append(n*i+j)
            col.append(n*n+n*i+j)

            if j != 0:
                data.append(1/2)
                row.append(n*i+j)
                col.append(n*n+n*i+(j-1))

    l_op_3_2 = scipy.sparse.coo_matrix((data, (row, col)), shape=(n*n, 2*n*n))

    l_op = scipy.sparse.vstack([l_op_1_1, l_op_2_1, l_op_3_1, l_op_1_2, l_op_2_2, l_op_3_2])

    return l_op.tocsc()


def run_primal_dual(y, dsampling_factor, kernel, reg_param, max_iter=10000):
    x = scipy.ndimage.zoom(y, dsampling_factor, order=2)

    n = x.shape[0]
    x = x.ravel()

    kernel_f = np.fft.fft2(kernel)
    kernel_f_conj = np.conj(kernel_f)

    y_f = np.fft.fft2(y)

    grad_op = make_grad_op(n)
    l_op = make_l_op(n)

    u = np.zeros(x.size*2)
    v = np.zeros(x.size*3*2)

    tmp = grad_op @ x
    v[:n*n] = tmp[:n*n]
    v[4*n*n:5*n*n] = tmp[n*n:]

    d_norm = np.sqrt(8)
    mu = 1.0
    tau = 0.99 / d_norm**2
    gamma = 0.99 / 3


    convergence = False
    n_iter = 0

    obj_tab = []

    while not convergence:
        x_f = np.fft.fft2(x.reshape(n, n))
        residual = np.real(np.fft.ifft2(x_f * kernel_f))[::dsampling_factor, ::dsampling_factor] - y
        upsampled_residual = np.zeros((residual.shape[0] * dsampling_factor, residual.shape[1] * dsampling_factor))
        upsampled_residual[::dsampling_factor, ::dsampling_factor] = residual
        phi_adj_residual = np.real(np.fft.ifft2(np.fft.fft2(upsampled_residual) * kernel_f_conj))
        x -= tau * (grad_op.T @ (grad_op @ x - l_op.T @ v + mu * u) + mu * phi_adj_residual.ravel())

        v += gamma * l_op @ (grad_op @ x - l_op.T @ v + mu * u)
        v = v.reshape((2, 3*x.size))
        v = prox_one_two_norm(v, gamma*mu*reg_param)
        v = v.ravel()

        u += (grad_op @ x - l_op.T @ v) / mu

        x_f = np.fft.fft2(x.reshape(n, n))

        tv = one_two_norm(v.reshape((2, 3*x.size)))
        obj = 0.5 * np.sum((np.real(np.fft.ifft2(x_f * kernel_f)[::dsampling_factor, ::dsampling_factor]) - y)**2) \
              + reg_param * tv
        obj_tab.append(obj)

        if n_iter % 1000 == 0:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_aspect('equal')

            im = ax.imshow(x.reshape(n, n), vmin=-3.1, vmax=3.1,
                           origin='lower', cmap='bwr')

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=30)

            ax.axis('off')
            plt.tight_layout()
            plt.show()

            # plt.plot(obj_tab)
            # plt.show()

        n_iter += 1

        convergence = n_iter > max_iter

    return x
