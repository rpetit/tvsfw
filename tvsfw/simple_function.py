import numpy as np
import matplotlib.pyplot as plt

from math import copysign, fabs
from copy import deepcopy
from celer import Lasso
from pycheeger import integrate_on_triangles


sign = lambda x: copysign(x, 1)


class WeightedIndicatorFunction:
    def __init__(self, weight, simple_set):
        self.weight = weight
        self.support = simple_set

    def __call__(self, x):
        if self.support.contains(x):
            return self.weight
        else:
            return 0


class SimpleFunction:
    def __init__(self, atoms):
        self.atoms = atoms
        self.num_atoms = len(atoms)

    def __call__(self, x):
        res = 0
        for f in self.atoms:
            res += f(x)
        return res

    def compute_obs(self, f, version=0):
        obs = [atom.support.compute_weighted_areas(f) for atom in self.atoms]

        if version == 0:
            if len(obs) == 0:
                res = 0
            else:
                res = np.zeros(obs[0].shape[1:])
            for i in range(self.num_atoms):
                res += self.atoms[i].weight * np.sum(obs[i], axis=0)
        else:
            res = obs

        return res

    def extend_support(self, simple_set):
        new_atom = WeightedIndicatorFunction(0, simple_set)
        self.atoms.append(new_atom)
        self.num_atoms += 1

    def fit_weights(self, y, phi, reg_param, tol_factor=1e-4):
        obs = self.compute_obs(phi, version=1)
        mat = np.array([np.sum(obs[i], axis=0) for i in range(self.num_atoms)])
        mat = mat.reshape((self.num_atoms, -1)).T

        tol = tol_factor * np.linalg.norm(y)**2 / y.size
        perimeters = np.array([self.atoms[i].support.compute_perimeter() for i in range(self.num_atoms)])

        lasso = Lasso(alpha=reg_param/y.size, fit_intercept=False, tol=tol, weights=perimeters)
        lasso.fit(mat, y.reshape(-1))

        new_weights = lasso.coef_
        self.atoms = [WeightedIndicatorFunction(new_weights[i], self.atoms[i].support) for i in range(self.num_atoms) if np.abs(new_weights[i]) > 0.1]
        self.num_atoms = len(self.atoms)

    def perform_sliding(self, y, f, reg_param, step_size, max_iter, alpha, beta, eps_stop,
                        num_points, iter_remesh, max_tri_area):
        obj_tab = []
        grad_norm_tab = []

        convergence = False
        n_iter = 0

        obs = self.compute_obs(f, version=1)

        best_obj = None
        best_weights = None
        best_supports = None

        weights = np.array([atom.weight for atom in self.atoms])
        supports = [atom.support for atom in self.atoms]
        perimeters = np.array([support.compute_perimeter() for support in supports])

        y_hat = np.sum(weights[:, np.newaxis, np.newaxis] * np.array([np.sum(obs[i], axis=0) for i in range(self.num_atoms)]), axis=0)
        diff_obs = y_hat - y
        obj = 0.5 * np.sum(diff_obs ** 2) + reg_param * np.sum(np.abs(weights) * perimeters)
        obj_tab.append(obj)

        while not convergence and n_iter < max_iter:
            grad_weights = []
            grad_boundary_vertices = []
            grad_norm_squared = 0

            for i in range(self.num_atoms):
                grad_weight = np.sum(np.sum(obs[i], axis=0) * diff_obs) + reg_param * sign(weights[i]) * perimeters[i]
                grad_weights.append(grad_weight)

                grad_norm_squared += grad_weight ** 2

                grad_perimeter = supports[i].compute_perimeter_gradient()
                grad_area = supports[i].compute_weighted_area_gradient(f)
                grad_boundary_vertices.append(np.sum(weights[i] * np.expand_dims(diff_obs, axis=(2, 3)) * grad_area, axis=(0, 1))
                                              + reg_param * fabs(weights[i]) * grad_perimeter)

                grad_norm_squared += np.linalg.norm(grad_boundary_vertices[i]) ** 2

            t = step_size

            ag_condition = False

            former_obj = obj
            former_weights = weights.copy()
            former_supports = deepcopy(supports)

            while not ag_condition:
                ag_condition = True
                weights = np.array(former_weights) - t * np.array(grad_weights)

                for i in range(self.num_atoms):
                    supports[i].boundary_vertices = former_supports[i].boundary_vertices - t * grad_boundary_vertices[i]
                    supports[i].mesh_vertices[np.arange(supports[i].num_boundary_vertices)] = supports[i].boundary_vertices

                    obs[i][supports[i].boundary_faces_indices, :, :] = integrate_on_triangles(f,
                                                                                              supports[i].mesh_vertices[supports[i].mesh_faces[supports[i].boundary_faces_indices]])

                perimeters = np.array([support.compute_perimeter() for support in supports])

                y_hat = np.sum(weights[:, np.newaxis, np.newaxis] * np.array([np.sum(obs[i], axis=0) for i in range(self.num_atoms)]), axis=0)
                diff_obs = y_hat - y
                obj = 0.5 * np.sum(diff_obs ** 2) + reg_param * np.sum(np.abs(weights) * perimeters)

                # ag_condition = (obj <= former_obj - alpha * t * grad_norm_squared)
                # t = beta * t

            if n_iter > 0:
                weights = weights + n_iter / (n_iter + 3) * (weights - former_weights)

                for i in range(self.num_atoms):
                    supports[i].boundary_vertices = supports[i].boundary_vertices + \
                                                    n_iter / (n_iter + 3) * (supports[i].boundary_vertices - former_supports[i].boundary_vertices)

                    supports[i].mesh_vertices[np.arange(supports[i].num_boundary_vertices)] = supports[i].boundary_vertices

                    obs[i][supports[i].boundary_faces_indices, :, :] = integrate_on_triangles(f,
                                                                                              supports[i].mesh_vertices[supports[i].mesh_faces[supports[i].boundary_faces_indices]])

                    perimeters = np.array([support.compute_perimeter() for support in supports])

                    y_hat = np.sum(weights[:, np.newaxis, np.newaxis] * np.array(
                        [np.sum(obs[i], axis=0) for i in range(self.num_atoms)]), axis=0)
                    diff_obs = y_hat - y
                    obj = 0.5 * np.sum(diff_obs ** 2) + reg_param * np.sum(np.abs(weights) * perimeters)

            self.atoms = [WeightedIndicatorFunction(weights[i], supports[i]) for i in range(self.num_atoms)]

            n_iter += 1
            grad_norm_tab.append(grad_norm_squared)

            convergence = grad_norm_squared <= eps_stop

            if n_iter % iter_remesh == 0:
                print(n_iter)
                plt.plot(obj_tab)
                plt.show()
                plt.plot(grad_norm_tab)
                plt.show()
                for i in range(self.num_atoms):
                    self.atoms[i].support.resample_boundary(num_points, max_tri_area)

                obs = self.compute_obs(f, version=1)

                weights = np.array([atom.weight for atom in self.atoms])
                supports = [atom.support for atom in self.atoms]
                perimeters = np.array([support.compute_perimeter() for support in supports])

                y_hat = np.sum(weights[:, np.newaxis, np.newaxis] * np.array([np.sum(obs[i], axis=0) for i in range(self.num_atoms)]), axis=0)
                diff_obs = y_hat - y
                obj = 0.5 * np.sum(diff_obs ** 2) + reg_param * np.sum(np.abs(weights) * perimeters)
                obj_tab.append(obj)
            else:
                obj_tab.append(obj)

            if obj_tab[-1] > obj_tab[0]:
                convergence = True

            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_weights = weights.copy()
                best_supports = deepcopy(supports)

        for i in range(self.num_atoms):
            self.atoms = [WeightedIndicatorFunction(best_weights[i], best_supports[i]) for i in range(self.num_atoms)]

        obs = self.compute_obs(f, version=1)

        weights = np.array([atom.weight for atom in self.atoms])
        supports = [atom.support for atom in self.atoms]
        perimeters = np.array([support.compute_perimeter() for support in supports])

        y_hat = np.sum(
            weights[:, np.newaxis, np.newaxis] * np.array([np.sum(obs[i], axis=0) for i in range(self.num_atoms)]),
            axis=0)
        diff_obs = y_hat - y
        obj = 0.5 * np.sum(diff_obs ** 2) + reg_param * np.sum(np.abs(weights) * perimeters)

        assert np.isclose(obj, best_obj, rtol=1e-2) and np.isclose(np.min(np.array(obj_tab)), obj)

        for i in range(self.num_atoms):
            self.atoms[i].support.resample_boundary(num_points, max_tri_area)

        return obj_tab, grad_norm_tab

