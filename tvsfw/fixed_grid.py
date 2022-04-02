import numpy as np
import scipy.ndimage
import scipy.sparse

from numba import jit, prange

import matplotlib.pyplot as plt


@jit(nopython=True, parallel=True)
def proj_two_inf_ball_aux(x, alpha, res):
    for i in prange(x.shape[1]):
        norm_i = np.sqrt(x[0, i] ** 2 + x[1, i] ** 2)
        if norm_i > alpha:
            res[:, i] = alpha * x[:, i] / norm_i
        else:
            res[:, i] = x[:, i]


def proj_two_inf_ball(x, alpha):
    res = np.zeros_like(x)
    proj_two_inf_ball_aux(x, alpha, res)
    return res


def prox_conv_dsample_fft(u_f, y_f, kernel_f, tau):
    kernel_f_conj = np.conj(kernel_f)
    return (u_f + tau * kernel_f_conj * y_f) / (1 + tau * kernel_f_conj * kernel_f)


def make_grad_op(n):
    data = []
    row = []
    col = []

    for i in range(n+1):
        for j in range(1, n+1):
            if i != n:
                row.append((n+1)*i+j)
                col.append(n*i+j-1)
                data.append(1)

            if i != 0:
                row.append((n+1)*i+j)
                col.append(n*(i-1)+j-1)
                data.append(-1)

    grad_op_1 = scipy.sparse.coo_matrix((data, (row, col)), shape=((n+1)*(n+1), n*n))

    data = []
    row = []
    col = []

    for i in range(1, n+1):
        for j in range(n+1):
            if j != n:
                row.append((n+1)*i+j)
                col.append(n*(i-1)+j)
                data.append(1)

            if j != 0:
                row.append((n+1)*i+j)
                col.append(n*(i-1)+j-1)
                data.append(-1)

    grad_op_2 = scipy.sparse.coo_matrix((data, (row, col)), shape=((n+1)*(n+1), n*n))

    grad_op = (2/n) * scipy.sparse.vstack([grad_op_1, grad_op_2])

    return grad_op.tocsc()


def run_primal_dual(y, dsampling_factor, kernel, reg_param, max_iter=10000, convergence_tol=None, verbose=False):
    n = y.shape[0] * dsampling_factor
    u = scipy.ndimage.zoom(y, dsampling_factor, order=2)
    y_f = np.fft.fft2(y)
    kernel_f = np.fft.fft2(kernel)

    u = u.ravel()

    p = np.zeros((n+1) * (n+1) * 2)

    grad_op = make_grad_op(n)

    L = 1.0
    tau = 1 / L
    sigma = 1 / tau / L**2
    theta = 1.0

    convergence = False
    n_iter = 0

    obj_tab = []

    while not convergence:
        u_aux = u.copy()
        u -= tau * (grad_op.T @ p)

        u_f = np.fft.fft2(u.reshape(n, n))
        u_f = prox_conv_dsample_fft(u_f, y_f, kernel_f, tau)
        u = np.real(np.fft.ifft2(u_f)).reshape(n*n)

        u_aux = u + theta * (u - u_aux)

        p += sigma * (grad_op @ u_aux)

        p = p.reshape(2, (n+1)*(n+1))
        p = proj_two_inf_ball(p, reg_param)
        p = p.ravel()

        tv = np.sum(np.sqrt(np.sum(((grad_op @ u).reshape(2, (n+1)*(n+1)))**2, axis=0)))
        obj = 0.5 * np.sum((np.real(np.fft.ifft2(u_f * kernel_f)) - y)**2) + reg_param * tv
        obj_tab.append(obj)

        n_iter += 1

        if n_iter % 10000 == 0:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_aspect('equal')

            im = ax.imshow(u.reshape(n, n), vmin=-2.1, vmax=2.1,
                           origin='lower', cmap='bwr')

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=30)

            ax.axis('off')
            plt.tight_layout()
            plt.show()

            plt.plot(obj_tab)
            plt.show()

        if convergence_tol is None:
            convergence = n_iter > max_iter
        else:
            convergence = (np.linalg.norm(u - u_aux) / np.linalg.norm(u) < convergence_tol) or (n_iter > max_iter)

    if verbose:
        print(np.linalg.norm(u - u_aux) / np.linalg.norm(u))

    return u

