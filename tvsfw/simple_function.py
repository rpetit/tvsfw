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

    def extend_support(self, simple_set):
        new_atom = WeightedIndicatorFunction(0, simple_set)
        self.atoms.append(new_atom)

    def fit_weights(self, phi, y, reg_param):
        pass

    def perform_sliding(self, phi, y, reg_param):
        pass
