import pytest
import numpy as np

from pycheeger import Disk
from tvsfw import SimpleFunction, WeightedIndicatorFunction, SampledGaussianKernel


def test_weights_update():
    E1 = Disk(np.array([-0.4, 0]), 0.3)
    E2 = Disk(np.array([0.4, -0.3]), 0.2)
    u = SimpleFunction([WeightedIndicatorFunction(1, E1), WeightedIndicatorFunction(1, E2)])

    x_grid, y_grid = np.linspace(-1, 1, 40), np.linspace(-1, 1, 40)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid = np.stack([X_grid, Y_grid], axis=2)
    std = 0.1

    phi = SampledGaussianKernel(grid, std)
    y = u.compute_obs(phi)

    E3 = Disk(np.array([1, 1]), 1)
    u_hat = SimpleFunction([WeightedIndicatorFunction(0, E1),
                            WeightedIndicatorFunction(0, E2),
                            WeightedIndicatorFunction(0, E3)])

    alpha = 1e-4
    u_hat.fit_weights(y, phi, alpha / y.size)

    np.testing.assert_allclose(np.array([atom.weight for atom in u_hat.atoms][:-1]),
                               np.array([atom.weight for atom in u.atoms]), rtol=1e-2)

    assert u_hat.atoms[-1].weight == pytest.approx(0)


def test_sliding():
    E1 = Disk(np.array([-0.4, 0]), 0.4)
    E2 = Disk(np.array([0.4, -0.3]), 0.3)
    u = SimpleFunction([WeightedIndicatorFunction(1, E1), WeightedIndicatorFunction(1, E2)])

    x_grid, y_grid = np.linspace(-1, 1, 40), np.linspace(-1, 1, 40)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid = np.stack([X_grid, Y_grid], axis=2)
    std = 0.1

    phi = SampledGaussianKernel(grid, std)
    y = u.compute_obs(phi)

    E1_noisy = E1
    E2_noisy = E2
    u_noisy = SimpleFunction([WeightedIndicatorFunction(1.1, E1_noisy), WeightedIndicatorFunction(0.9, E2_noisy)])

    u_noisy.perform_sliding(y, phi, 1e-3, 0.1, 500, 1e-6)

    np.testing.assert_allclose(np.array([atom.weight for atom in u_noisy.atoms]),
                               np.array([atom.weight for atom in u.atoms]), rtol=0.05)

    # E1_noisy.boundary_vertices += 0 * (1 - np.random.random((E1_noisy.num_boundary_vertices, 2)))
    # E1_noisy.mesh_vertices[E1_noisy.boundary_vertices_indices] = E1_noisy.boundary_vertices
    #
    # E2_noisy.boundary_vertices += 0 * (1 - np.random.random((E2_noisy.num_boundary_vertices, 2)))
    # E2_noisy.mesh_vertices[E2_noisy.boundary_vertices_indices] = E2_noisy.boundary_vertices
