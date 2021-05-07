import numpy as np

from celer import Lasso


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

    def __call__(self, x):
        res = 0
        for f in self.atoms:
            res += f(x)
        return res

    @property
    def num_atoms(self):
        return len(self.atoms)

    @property
    def weights(self):
        return np.array([atom.weight for atom in self.atoms])

    @property
    def supports(self):
        return [atom.support for atom in self.atoms]

    @property
    def support_boundary_vertices(self):
        return [atom.support.boundary_vertices for atom in self.atoms]

    def compute_obs(self, f, version=0):
        obs = [atom.support.compute_weighted_area_tab(f) for atom in self.atoms]

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

    def fit_weights(self, y, phi, reg_param, tol_factor=1e-4):
        obs = self.compute_obs(phi, version=1)
        mat = np.array([np.sum(obs[i], axis=0) for i in range(self.num_atoms)])
        mat = mat.reshape((self.num_atoms, -1)).T

        tol = tol_factor * np.linalg.norm(y)**2 / y.size
        perimeters = np.array([self.atoms[i].support.compute_perimeter() for i in range(self.num_atoms)])

        lasso = Lasso(alpha=reg_param/y.size, fit_intercept=False, tol=tol, weights=perimeters)
        lasso.fit(mat, y.reshape(-1))

        new_weights = lasso.coef_
        self.atoms = [WeightedIndicatorFunction(new_weights[i], self.atoms[i].support)
                      for i in range(self.num_atoms) if np.abs(new_weights[i]) > 0.1]
