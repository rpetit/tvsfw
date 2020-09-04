import numpy as np

from copy import deepcopy
from sklearn.linear_model import Lasso
from pymesh import form_mesh
from pycheeger import SimpleSet, integrate_on_triangle


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

    def compute_obs(self, f, grid_shape, version=0):
        if version == 0:
            res = np.zeros((grid_shape[0], grid_shape[1]))
        if version == 1:
            num_max_faces = np.max([len(atom.support.mesh_faces) for atom in self.atoms])
            res = np.zeros((grid_shape[0], grid_shape[1], self.num_atoms, num_max_faces))

        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(self.num_atoms):
                    if version == 0:
                        res[i, j] += self.atoms[k].weight * self.atoms[k].support.compute_weighted_area(lambda x: f(x, i, j))
                    if version == 1:
                        areas = self.atoms[k].support.compute_weighted_areas(lambda x: f(x, i, j))
                        res[i, j, k] = np.pad(areas, (0, num_max_faces - len(areas)), 'constant')

        return res

    def extend_support(self, simple_set):
        new_atom = WeightedIndicatorFunction(0, simple_set)
        self.atoms.append(new_atom)
        self.num_atoms += 1

    def fit_weights(self, y, aux, reg_param):
        mat = np.sum(self.compute_obs(aux, y.shape, version=1), axis=-1)
        mat = mat.reshape((-1, self.num_atoms))

        lasso = Lasso(alpha=reg_param)
        lasso.fit(mat, y.reshape(-1))

        new_weights = lasso.coef_

        self.atoms = [WeightedIndicatorFunction(new_weights[i], self.atoms[i].support) for i in range(self.num_atoms)]

    def perform_sliding(self, y, aux, reg_param, step_size, max_iter, eps_stop):
        obj_tab = []

        convergence = False
        n_iter = 0

        obs = self.compute_obs(aux, y.shape, version=1)

        weights = np.array([atom.weight for atom in self.atoms])
        supports = [atom.support for atom in self.atoms]
        perimeters = np.array([support.compute_perimeter() for support in supports])

        diff_obs = np.sum(weights * np.sum(obs, axis=-1), axis=2) - y
        obj = 0.5 * np.sum(diff_obs ** 2) + reg_param * np.sum(np.abs(weights) * perimeters)
        obj_tab.append(obj)

        while not convergence and n_iter < max_iter:
            grad_weights = np.sum(np.sum(obs, axis=-1) * diff_obs[:, :, np.newaxis], axis=(0, 1)) + reg_param * np.sign(weights) * perimeters

            grad_supports = []

            for k in range(self.num_atoms):
                grad_perimeter = supports[k].compute_perimeter_gradient()
                grad_area = np.array([[supports[k].compute_weighted_area_gradient(lambda x: aux(x, i, j)) for j in range(obs.shape[1])] for i in range(obs.shape[0])])
                grad_supports.append(np.sum(weights[k] * np.expand_dims(diff_obs, axis=(2, 3)) * grad_area, axis=(0, 1)) + reg_param * np.abs(weights[k]) * grad_perimeter)

            grad_norm_squared = np.linalg.norm(grad_weights) ** 2

            for k in range(self.num_atoms):
                grad_norm_squared += np.linalg.norm(grad_supports[k]) ** 2

            alpha = 0.1
            beta = 0.5
            t = step_size

            ag_condition = False

            former_obj = obj
            former_weights = weights.copy()
            former_supports = deepcopy(supports)

            while not ag_condition:
                weights = former_weights - t * grad_weights

                for k in range(self.num_atoms):
                    supports[k].boundary_vertices = former_supports[k].boundary_vertices - t * grad_supports[k]
                    supports[k].mesh_vertices[supports[k].boundary_vertices_indices] = supports[k].boundary_vertices

                    for l in supports[k].boundary_faces_indices:
                        for i in range(obs.shape[0]):
                            for j in range(obs.shape[1]):
                                obs[i, j, k, l] = integrate_on_triangle(lambda x: aux(x, i, j), supports[k].mesh_vertices[supports[k].mesh_faces[l]])

                perimeters = np.array([support.compute_perimeter() for support in supports])

                diff_obs = np.sum(weights * np.sum(obs, axis=-1), axis=2) - y
                obj = 0.5 * np.sum(diff_obs ** 2) + reg_param * np.sum(np.abs(weights) * perimeters)

                ag_condition = (obj <= former_obj - alpha * t * grad_norm_squared)
                t = beta * t
            n_iter += 1
            obj_tab.append(obj)

        return obj_tab

