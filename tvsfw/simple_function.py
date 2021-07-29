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
        if self.num_atoms == 0:
            return np.zeros(f.grid_size)

        max_num_triangles = max(len(atom.support.mesh_faces) for atom in self.atoms)
        meshes = np.zeros((self.num_atoms, max_num_triangles, 3, 2))
        obs = np.zeros((self.num_atoms, max_num_triangles, f.grid_size))

        for i in range(self.num_atoms):
            support_i = self.atoms[i].support
            meshes[i, :len(support_i.mesh_faces)] = support_i.mesh_vertices[support_i.mesh_faces]

        f._triangle_aux(meshes, obs)

        if version == 1:
            res = [obs[i, :len(self.atoms[i].support.mesh_faces), :] for i in range(self.num_atoms)]
        else:
            res = np.zeros(f.grid_size)
            for i in range(self.num_atoms):
                res += self.atoms[i].weight * np.sum(obs[i], axis=0)

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
                      for i in range(self.num_atoms) if np.abs(new_weights[i]) > 1e-2]
        # TODO: clean zero weight condition
