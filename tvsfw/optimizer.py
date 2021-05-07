import numpy as np

from math import copysign, fabs
from .simple_function import SimpleFunction, WeightedIndicatorFunction
from pycheeger import resample, SimpleSet


sign = lambda x: copysign(x, 1)


class SlidingOptimizerState:
    def __init__(self, initial_function, f, reg_param, y):
        self.function = None
        self.obs_tab = None
        self.diff_obs = None
        self.perimeter_tab = None
        self.obj = None

        self.update_function(initial_function, f, reg_param, y)

    def update_obj(self, y, reg_param):
        weights = self.function.weights

        self.perimeter_tab = []
        for atom in self.function.atoms:
            self.perimeter_tab.append(atom.support.compute_perimeter())

        sum_obs = np.array([np.sum(self.obs_tab[i], axis=0) for i in range(self.function.num_atoms)])
        y_hat = np.sum(weights[:, np.newaxis, np.newaxis] * sum_obs, axis=0)
        self.diff_obs = y_hat - y

        self.obj = 0.5 * np.sum(self.diff_obs ** 2) + reg_param * np.sum(np.abs(weights) * self.perimeter_tab)

    def update_weights(self, new_weights):
        for i in range(self.function.num_atoms):
            self.function.atoms[i].weight = new_weights[i]

    def update_boundary_vertices(self, new_boundary_vertices, f, y, reg_param):
        for i in range(self.function.num_atoms):
            self.function.atoms[i].support.boundary_vertices = new_boundary_vertices[i]
            support = self.function.atoms[i].support
            self.obs_tab[i][support.mesh_boundary_faces_indices, :, :] = support.compute_weighted_area_tab(f, boundary_faces_only=True)

        self.update_obj(y, reg_param)

    def update_function(self, new_function, f, reg_param, y):
        self.function = new_function
        self.obs_tab = self.function.compute_obs(f, version=1)
        self.update_obj(y, reg_param)

    def compute_gradient(self, f, reg_param):
        grad_weights = []
        grad_boundary_vertices = []
        grad_norm_squared = 0

        weights = self.function.weights
        supports = self.function.supports

        for i in range(self.function.num_atoms):
            grad_weight = np.sum(np.sum(self.obs_tab[i], axis=0) * self.diff_obs) \
                          + reg_param * sign(weights[i]) * self.perimeter_tab[i]
            grad_weights.append(grad_weight)

            grad_norm_squared += grad_weight ** 2

            grad_perimeter = supports[i].compute_perimeter_gradient()
            grad_area = supports[i].compute_weighted_area_gradient(f)
            grad_shape = np.sum(weights[i] * np.expand_dims(self.diff_obs, axis=(2, 3)) * grad_area, axis=(0, 1)) \
                         + reg_param * fabs(weights[i]) * grad_perimeter

            grad_boundary_vertices.append(grad_shape)

            grad_norm_squared += np.linalg.norm(grad_boundary_vertices[i]) ** 2

        return grad_weights, grad_boundary_vertices, grad_norm_squared


class SlidingOptimizer:
    def __init__(self, step_size, max_iter, eps_stop, num_points, max_tri_area, num_iter_resampling, alpha, beta):
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps_stop = eps_stop
        self.num_points = num_points
        self.max_tri_area = max_tri_area
        self.num_iter_resampling = num_iter_resampling
        self.alpha = alpha
        self.beta = beta

        self.state = None

    def perform_linesearch(self, f, y, reg_param, grad_weights, grad_boundary_vertices, grad_norm_squared, no_linesearch=True):
        t = self.step_size

        ag_condition = False

        former_obj = self.state.obj
        former_weights = self.state.function.weights
        former_boundary_vertices = self.state.function.support_boundary_vertices

        while not ag_condition:
            new_weights = np.array(former_weights) - t * np.array(grad_weights)
            new_boundary_vertices = []

            for i in range(self.state.function.num_atoms):
                new_boundary_vertices.append(former_boundary_vertices[i] - t * grad_boundary_vertices[i])

            self.state.update_weights(new_weights)
            self.state.update_boundary_vertices(new_boundary_vertices, f, y, reg_param)
            new_obj = self.state.obj

            if no_linesearch:
                ag_condition = True
            else:
                ag_condition = (new_obj <= former_obj - self.alpha * t * grad_norm_squared)
                t = self.beta * t

    def run(self, initial_function, f, reg_param, y):
        convergence = False
        obj_tab = []
        grad_norm_tab = []

        iteration = 0
        num_atoms = initial_function.num_atoms

        self.state = SlidingOptimizerState(initial_function, f, reg_param, y)

        while not convergence and iteration < self.max_iter:
            grad_weights, grad_boundary_vertices, grad_norm_squared = self.state.compute_gradient(f, reg_param)
            grad_norm_tab.append(grad_norm_squared)
            obj_tab.append(self.state.obj)

            self.perform_linesearch(f, y, reg_param, grad_weights, grad_boundary_vertices, grad_norm_squared)

            iteration += 1
            convergence = False

        if self.num_iter_resampling is not None and iteration % self.num_iter_resampling == 0:
            new_weights = []
            new_sets = []

            for atom in self.state.function.atoms:
                new_weights.append(atom.weight)
                new_boundary_vertices = resample(atom.support.boundary_vertices, num_points=self.num_points)
                new_sets.append(SimpleSet(new_boundary_vertices, max_tri_area=self.max_tri_area))

            new_function = SimpleFunction([WeightedIndicatorFunction(new_weights[i], new_sets[i])
                                           for i in range(num_atoms)])

            self.state.update_function(new_function, f, reg_param, y)

        return self.state.function, obj_tab, grad_norm_tab
