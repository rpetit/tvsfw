from pycheeger import compute_cheeger, plot_simple_functions
from tvsfw import SimpleFunction, SlidingOptimizer

import matplotlib.pyplot as plt


def perform_sliding(initial_function, y, f, reg_param, step_size, max_iter, alpha, beta, eps_stop,
                    num_points, num_iter_resampling, max_tri_area):
    optimizer = SlidingOptimizer(step_size, max_iter, eps_stop, num_points, max_tri_area, num_iter_resampling,
                                 alpha, beta)

    new_function, obj_tab, grad_norm_tab = optimizer.run(initial_function, f, reg_param, y)

    return new_function, obj_tab, grad_norm_tab


def sfw(phi, y, reg_param, n_iter, u=None):
    u_hat = SimpleFunction([])

    for i in range(n_iter):
        residual = y - u_hat.compute_obs(phi, version=0)
        eta = phi.apply_adjoint(residual)

        E, obj_tab, grad_norm_tab = compute_cheeger(eta,
                                                    max_tri_area_fm=1e-3, max_iter_fm=10000, plot_results_fm=True,
                                                    num_boundary_vertices_ld=75, max_tri_area_ld=1e-2,
                                                    step_size_ld=1e-2, max_iter_ld=300, convergence_tol_ld=1e-6,
                                                    num_iter_resampling_ld=50, plot_results_ld=True)

        u_hat.extend_support(E)
        u_hat.fit_weights(y, phi, reg_param)
        u_hat, obj_tab, grad_norm_tab = perform_sliding(u_hat, y, phi, reg_param, 0.2, 250, 0.1, 0.5, 1e-7, 50, 25, 0.004)

        plt.plot(obj_tab)
        plt.show()

        plt.plot(grad_norm_tab)
        plt.show()

        plot_simple_functions(u, u_hat)

    return u_hat
