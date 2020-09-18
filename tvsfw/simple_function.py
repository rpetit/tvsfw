import numpy as np

from math import copysign, fabs
from copy import deepcopy
from sklearn.linear_model import Lasso
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

    def fit_weights(self, y, phi, reg_param):
        obs = self.compute_obs(phi, version=1)
        mat = np.array([np.sum(obs[i], axis=0) for i in range(self.num_atoms)])
        mat = mat.reshape((self.num_atoms, -1)).T

        lasso = Lasso(alpha=reg_param)
        lasso.fit(mat, y.reshape(-1))

        new_weights = lasso.coef_

        self.atoms = [WeightedIndicatorFunction(new_weights[i], self.atoms[i].support) for i in range(self.num_atoms)]

    def perform_sliding(self, y, aux, reg_param, step_size, max_iter, eps_stop):
        obj_tab = []
        grad_norm_tab = []

        convergence = False
        n_iter = 0

        obs = self.compute_obs(aux, version=1)

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
                grad_area = supports[i].compute_weighted_area_gradient(aux)
                grad_boundary_vertices.append(np.sum(weights[i] * np.expand_dims(diff_obs, axis=(2, 3)) * grad_area, axis=(0, 1))
                                              + reg_param * fabs(weights[i]) * grad_perimeter)

                grad_norm_squared += np.linalg.norm(grad_boundary_vertices[i]) ** 2

            alpha = 0.1
            beta = 0.5
            t = step_size

            ag_condition = False

            former_obj = obj
            former_weights = weights.copy()
            former_supports = deepcopy(supports)

            while not ag_condition:
                weights = np.array(former_weights) - t * np.array(grad_weights)

                for i in range(self.num_atoms):
                    supports[i].boundary_vertices = former_supports[i].boundary_vertices - t * grad_boundary_vertices[i]
                    supports[i].mesh_vertices[supports[i].boundary_vertices_indices] = supports[i].boundary_vertices

                    obs[i][supports[i].boundary_faces_indices, :, :] = integrate_on_triangles(aux,
                                                                                              supports[i].mesh_vertices[supports[i].mesh_faces[supports[i].boundary_faces_indices]])

                perimeters = np.array([support.compute_perimeter() for support in supports])

                y_hat = np.sum(weights[:, np.newaxis, np.newaxis] * np.array([np.sum(obs[i], axis=0) for i in range(self.num_atoms)]), axis=0)
                diff_obs = y_hat - y
                obj = 0.5 * np.sum(diff_obs ** 2) + reg_param * np.sum(np.abs(weights) * perimeters)

                ag_condition = (obj <= former_obj - alpha * t * grad_norm_squared)
                t = beta * t

            if n_iter > 0:
                weights = weights + n_iter / (n_iter + 3) * (weights - former_weights)

                for i in range(self.num_atoms):
                    supports[i].boundary_vertices = supports[i].boundary_vertices + \
                                                    n_iter / (n_iter + 3) * (supports[i].boundary_vertices - former_supports[i].boundary_vertices)

                    supports[i].mesh_vertices[supports[i].boundary_vertices_indices] = supports[i].boundary_vertices

                    obs[i][supports[i].boundary_faces_indices, :, :] = integrate_on_triangles(aux,
                                                                                              supports[i].mesh_vertices[supports[i].mesh_faces[supports[i].boundary_faces_indices]])

                    perimeters = np.array([support.compute_perimeter() for support in supports])

                    y_hat = np.sum(weights[:, np.newaxis, np.newaxis] * np.array(
                        [np.sum(obs[i], axis=0) for i in range(self.num_atoms)]), axis=0)
                    diff_obs = y_hat - y
                    obj = 0.5 * np.sum(diff_obs ** 2) + reg_param * np.sum(np.abs(weights) * perimeters)

            self.atoms = [WeightedIndicatorFunction(weights[i], supports[i]) for i in range(self.num_atoms)]

            n_iter += 1
            obj_tab.append(obj)
            grad_norm_tab.append(grad_norm_squared)

            convergence = grad_norm_squared <= eps_stop

            if n_iter % 50 == 0:
                for i in range(self.num_atoms):
                    self.atoms[i].support.mesh(0.005)

                obs = self.compute_obs(aux, version=1)

                weights = np.array([atom.weight for atom in self.atoms])
                supports = [atom.support for atom in self.atoms]
                perimeters = np.array([support.compute_perimeter() for support in supports])

                y_hat = np.sum(weights[:, np.newaxis, np.newaxis] * np.array([np.sum(obs[i], axis=0) for i in range(self.num_atoms)]), axis=0)
                diff_obs = y_hat - y
                obj = 0.5 * np.sum(diff_obs ** 2) + reg_param * np.sum(np.abs(weights) * perimeters)
                obj_tab.append(obj)

        for i in range(self.num_atoms):
            self.atoms[i].support.mesh(0.005)

        return obj_tab, grad_norm_tab

