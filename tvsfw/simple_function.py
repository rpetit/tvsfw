import numpy as np

from sklearn.linear_model import Lasso


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

    def compute_obs(self, grid, f):
        y = np.zeros((grid.shape[0], grid.shape[1]))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for atom in self.atoms:
                    y[i, j] += atom.weight * atom.support.compute_weighted_area(lambda x: f(x, i, j))

        return y

    def extend_support(self, simple_set):
        new_atom = WeightedIndicatorFunction(0, simple_set)
        self.atoms.append(new_atom)
        self.num_atoms += 1

    def fit_weights(self, y, aux, reg_param):
        mat = np.zeros(y.shape + (self.num_atoms,))

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                for k in range(self.num_atoms):
                    atom = self.atoms[k]
                    support = atom.support
                    mat[i, j, k] = support.compute_weighted_area(lambda x: aux(x, i, j))

        mat = mat.reshape((-1, self.num_atoms))

        lasso = Lasso(alpha=reg_param)
        lasso.fit(mat, y.reshape(-1))

        new_weights = lasso.coef_

        self.atoms = [WeightedIndicatorFunction(new_weights[i], self.atoms[i].support) for i in range(self.num_atoms)]

    def perform_sliding(self, y, aux, reg_param, n_iter):
        for _ in range(n_iter):
            print("coucou")

