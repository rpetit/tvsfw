import numpy as np

from math import copysign
from .simple_function import SimpleFunction, WeightedIndicatorFunction
from pycheeger import resample, SimpleSet, plot_simple_set


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
        y_hat = np.sum(weights[:, None] * sum_obs, axis=0)
        self.diff_obs = y_hat - y

        self.obj = 0.5 * np.sum(self.diff_obs ** 2) + reg_param * np.sum(np.abs(weights) * self.perimeter_tab)

    def update_weights(self, new_weights):
        for i in range(self.function.num_atoms):
            self.function.atoms[i].weight = new_weights[i]

    def update_boundary_vertices(self, new_boundary_vertices, f, y, reg_param):
        max_num_boundary_faces = max(len(atom.support.mesh_boundary_faces) for atom in self.function.atoms)
        obs = np.zeros((self.function.num_atoms, max_num_boundary_faces, f.grid_size))
        meshes = np.zeros((self.function.num_atoms, max_num_boundary_faces, 3, 2))

        for i in range(self.function.num_atoms):
            support_i = self.function.atoms[i].support
            support_i.boundary_vertices = new_boundary_vertices[i]
            meshes[i, :len(support_i.mesh_boundary_faces)] = support_i.mesh_vertices[support_i.mesh_boundary_faces]

        f._triangle_aux(meshes, obs)

        for i in range(self.function.num_atoms):
            support_i = self.function.atoms[i].support
            self.obs_tab[i][support_i.mesh_boundary_faces_indices, :] = obs[i, :len(support_i.mesh_boundary_faces)]

        self.update_obj(y, reg_param)

    def update_function(self, new_function, f, reg_param, y):
        self.function = new_function
        self.obs_tab = self.function.compute_obs(f, version=1)
        self.update_obj(y, reg_param)

    def compute_gradient(self, f, reg_param):
        grad_weights = []
        grad_boundary_vertices = []
        grad_norm_squared = 0

        max_num_boundary_vertices = max(atom.support.num_boundary_vertices for atom in self.function.atoms)
        grad_area_weights = np.zeros((self.function.num_atoms, max_num_boundary_vertices, f.grid_size, 2))
        curves = np.zeros((self.function.num_atoms, max_num_boundary_vertices, 2))

        for i in range(self.function.num_atoms):
            support_i = self.function.atoms[i].support
            curves[i, :support_i.num_boundary_vertices, :] = support_i.boundary_vertices

        f._line_aux(curves, grad_area_weights)

        for i in range(self.function.num_atoms):
            weight_i = self.function.atoms[i].weight
            support_i = self.function.atoms[i].support

            grad_weight = np.sum(np.sum(self.obs_tab[i], axis=0) * self.diff_obs) + \
                          reg_param * sign(weight_i) * self.perimeter_tab[i]
            grad_weights.append(grad_weight)

            grad_norm_squared += grad_weight ** 2

            grad_perimeter = support_i.compute_perimeter_gradient()

            num_boundary_vertices = support_i.num_boundary_vertices
            grad_area = support_i.compute_weighted_area_gradient(f, weights=grad_area_weights[i, :num_boundary_vertices])

            grad_shape = weight_i * np.sum(self.diff_obs[None, :, None] * grad_area, axis=1) + \
                         reg_param * abs(weight_i) * grad_perimeter

            grad_boundary_vertices.append(grad_shape)

            grad_norm_squared += np.sum(grad_boundary_vertices[i] ** 2)

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

    def perform_linesearch(self, f, y, reg_param, grad_weights, grad_boundary_vertices, grad_norm_squared, no_linesearch=False):
        t = self.step_size

        ag_condition = False

        former_obj = self.state.obj
        former_weights = self.state.function.weights
        former_boundary_vertices = self.state.function.support_boundary_vertices

        iteration = 0

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

            iteration += 1

        max_displacement = 0
        for i in range(self.state.function.num_atoms):
            max_displacement = max(max_displacement, np.max(np.linalg.norm(new_boundary_vertices[i] - former_boundary_vertices[i], axis=-1)))

        return iteration, max_displacement

    def run(self, initial_function, f, reg_param, y, verbose=True):
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

            n_iter_linesearch, max_displacement = self.perform_linesearch(f, y, reg_param, grad_weights, grad_boundary_vertices, grad_norm_squared, no_linesearch=False)

            iteration += 1
            convergence = (max_displacement < self.eps_stop)

            if verbose:
                print("iteration {}: {} linesearch steps".format(iteration, n_iter_linesearch))

            if self.num_iter_resampling is not None and iteration % self.num_iter_resampling == 0:

                new_weights = []
                new_sets = []

                for atom in self.state.function.atoms:
                    new_weights.append(atom.weight)
                    new_boundary_vertices = resample(atom.support.boundary_vertices, num_points=self.num_points)
                    new_set = SimpleSet(new_boundary_vertices, max_tri_area=self.max_tri_area)
                    new_sets.append(new_set)

                new_function = SimpleFunction([WeightedIndicatorFunction(new_weights[i], new_sets[i])
                                               for i in range(num_atoms)])

                self.state.update_function(new_function, f, reg_param, y)

        return self.state.function, obj_tab, grad_norm_tab
