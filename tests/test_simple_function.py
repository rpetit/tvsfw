import numpy as np

from pytest import approx

from pycheeger import Disk
from tvsfw import SimpleFunction, WeightedIndicatorFunction


# def test_weights_update():
#     E1 = Disk(np.array([-0.4, 0]), 0.3)
#     E2 = Disk(np.array([0.4, -0.3]), 0.2)
#     u = SimpleFunction([WeightedIndicatorFunction(1, E1), WeightedIndicatorFunction(1, E2)])
#     u_hat = SimpleFunction([WeightedIndicatorFunction(0, E1), WeightedIndicatorFunction(0, E2)])


