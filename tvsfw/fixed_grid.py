import numpy as np

from numba import jit, prange
from pycheeger.tools import power_method, update_grad, update_adj_grad

import matplotlib.pyplot as plt


@jit(nopython=True, parallel=True)
def proj_two_inf_ball_aux(x, alpha, res):
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            norm_ij = np.sqrt(x[i, j, 0] ** 2 + x[i, j, 1] ** 2)
            if norm_ij > alpha:
                res[i, j] = alpha * x[i, j] / norm_ij
            else:
                res[i, j] = x[i, j]


def proj_two_inf_ball(x, alpha):
    res = np.zeros_like(x)
    proj_two_inf_ball_aux(x, alpha, res)
    return res


@jit(nopython=True, parallel=True)
def update_obs(u, phi_mat, res):
    grid_size = u.shape[0] - 2
    for k in prange(res.shape[0]):
        res[k] = 0
        for i in range(grid_size):
            for j in range(grid_size):
                res[k] += u[i, j] * phi_mat[i, j, k]


@jit(nopython=True, parallel=True)
def update_adj_obs(q, phi_mat, res):
    grid_size = res.shape[0]
    for i in prange(grid_size):
        for j in prange(grid_size):
            res[i, j] = 0
            for k in range(q.size):
                res[i, j] += q[k] * phi_mat[i, j, k]


def run_primal_dual(grid_size, y, phi, reg_param, max_iter=10000, convergence_tol=None, verbose=False, plot=False):
    phi_mat = phi.integrate_on_pixel_grid(grid_size)
    num_obs = phi_mat.shape[-1]

    grad_op_norm = power_method(grid_size)
    # meas_op_norm = np.linalg.norm(np.reshape(phi_mat, (grid_size * grid_size, num_obs)).T, ord=2)
    meas_op_norm = 0
    op_norm = np.sqrt(grad_op_norm ** 2 + meas_op_norm ** 2)

    grad_buffer = np.zeros((grid_size + 1, grid_size + 1, 2))
    adj_grad_buffer = np.zeros((grid_size + 2, grid_size + 2))

    obs_buffer = np.zeros(num_obs)
    adj_obs_buffer = np.zeros((grid_size, grid_size))

    L = 20.0
    tau = 1/L
    sigma = 1/tau/L**2

    p = np.zeros((grid_size + 1, grid_size + 1, 2))  # first dual variable
    q = np.zeros(num_obs)  # second dual variable
    u = np.zeros((grid_size + 2, grid_size + 2))  # primal variable
    former_u = u

    convergence = False
    iter = 0

    obj_tab = [np.linalg.norm(y) ** 2 / 2]

    while not convergence:
        update_grad(2 * u - former_u, grad_buffer)
        update_obs(2 * u - former_u, phi_mat, obs_buffer)

        p = proj_two_inf_ball(p + sigma * grad_buffer, reg_param)

        q = (q + sigma * (obs_buffer - y)) / (1 + sigma)

        update_adj_grad(p, adj_grad_buffer)
        update_adj_obs(q, phi_mat, adj_obs_buffer)

        former_u = np.copy(u)
        u -= tau * adj_grad_buffer
        u[1:grid_size+1, 1:grid_size+1] -= tau * adj_obs_buffer

        if iter % 500 == 0:
            v_abs_max = np.max(u)

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_aspect('equal')

            n = np.int(np.sqrt(y.size))

            im = ax.imshow(y.reshape((n, n)), origin='lower', cmap='bwr', vmin=-2.1, vmax=2.1)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=30)

            ax.axis('off')
            plt.tight_layout()
            plt.show()

            plt.plot(obj_tab)
            plt.show()

        obj = np.linalg.norm(obs_buffer - y) ** 2 / 2 + reg_param * np.sum(np.linalg.norm(grad_buffer, axis=-1))
        obj_tab.append(obj)

        iter += 1

        if convergence_tol is None:
            convergence = iter > max_iter
        else:
            convergence = np.linalg.norm(u - former_u) / np.linalg.norm(u) < convergence_tol

    if verbose:
        print(np.linalg.norm(u - former_u) / np.linalg.norm(u))

    return u