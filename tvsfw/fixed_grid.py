import numpy as np
import scipy.ndimage
import scipy.sparse

from numba import jit, prange
from pycheeger.tools import power_method, update_grad, update_adj_grad

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

    grad_op = (2/(n-1)) * scipy.sparse.vstack([grad_op_1, grad_op_2])

    return grad_op.tocsc()


# update_grad(2 * u - former_u, grad_buffer)
# update_obs(2 * u - former_u, phi_mat, obs_buffer)
#
# p = proj_two_inf_ball(p + sigma * grad_buffer, reg_param)
#
# q = (q + sigma * (obs_buffer - y)) / (1 + sigma)
#
# update_adj_grad(p, adj_grad_buffer)
# update_adj_obs(q, phi_mat, adj_obs_buffer)
#
# former_u = np.copy(u)
# u -= tau * adj_grad_buffer
# u[1:grid_size+1, 1:grid_size+1] -= tau * adj_obs_buffer


def run_primal_dual(y, dsampling_factor, kernel, reg_param, max_iter=10000, convergence_tol=None, verbose=False):
    m = y.shape[0]
    n = y.shape[0] * dsampling_factor
    u = scipy.ndimage.zoom(y, dsampling_factor, order=2)
    kernel_f = np.fft.fft2(kernel)
    kernel_f_conj = np.conj(kernel_f)

    u = u.ravel()

    p = np.zeros((n+1) * (n+1) * 2)
    q = np.zeros((m, m))

    grad_op = make_grad_op(n)

    L = 0.0005
    tau = 1 / L
    sigma = 1 / tau / L**2
    theta = 1.0

    convergence = False
    n_iter = 0

    obj_tab = []

    while not convergence:
        u_aux = u.copy()

        q_f = np.fft.fft2(q)
        phi_adj_q = np.real(np.fft.ifft2(kernel_f_conj * np.tile(q_f, (dsampling_factor, dsampling_factor))))

        u -= tau * (grad_op.T @ p + phi_adj_q.ravel())
        u_f = np.fft.fft2(u.reshape(n, n))

        u_aux = u + theta * (u - u_aux)

        p += sigma * (grad_op @ u_aux)
        p = p.reshape(2, (n+1)*(n+1))
        p = proj_two_inf_ball(p, reg_param)
        p = p.ravel()

        u_aux_f = np.fft.fft2(u_aux.reshape(n, n))

        q += sigma * np.real(np.fft.ifft2(u_aux_f * kernel_f)[::dsampling_factor, ::dsampling_factor])
        q = (q - sigma * y) / (1 + sigma)

        tv = np.sum(np.sqrt(np.sum(((grad_op @ u).reshape(2, (n+1)*(n+1)))**2, axis=0)))
        obj = 0.5 * np.sum((np.real(np.fft.ifft2(u_f * kernel_f)[::dsampling_factor, ::dsampling_factor]) - y)**2) \
              + reg_param * tv
        obj_tab.append(obj)

        n_iter += 1

        if n_iter % 5000 == 0:
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
        elif n_iter > 1:
            convergence = (np.linalg.norm(u - u_aux) / np.linalg.norm(u) < convergence_tol) or (n_iter > max_iter)

    if verbose:
        print(np.linalg.norm(u - u_aux) / np.linalg.norm(u))

    return u

