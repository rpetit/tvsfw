import pytest
import quadpy

from numpy import pi, exp, sqrt, cosh, sinh
from scipy.special import erf

from tvsfw.sampled_gaussian_filter import *
from pycheeger.simple_set import *

from pycheeger.plot_utils import plot_simple_set


std = 0.1
radius = 0.5

phi = SampledGaussianFilter(np.array([[0.0, 0.0]]), std)
simple_set = disk(np.array([0.0, 0.0]), radius, num_vertices=30, max_tri_area=1e-2)


def test_integral():
    res = phi.integrate_on_meshes(np.array([simple_set.mesh_vertices[simple_set.mesh_faces]]))
    my_val = np.sum(res[0])

    # scheme = quadpy.s2.get_good_scheme(15)
    # quadpy_val = scheme.integrate(lambda x: eta(x.T), [0.0, 0.0], radius)
    # assert quadpy_val == pytest.approx(weight * 2 * pi * std ** 2 * (1 - exp(-0.5 * radius ** 2 / std ** 2)))

    scheme = quadpy.t2.get_good_scheme(10)
    quadpy_val = 0
    triangles = simple_set.mesh_vertices[simple_set.mesh_faces]

    for i in range(len(triangles)):
        quadpy_val += scheme.integrate(lambda x: exp(-(x[0] ** 2 + x[1] ** 2) / (2 * std ** 2)), triangles[i])

    analytic_val = 2 * pi * std ** 2 * (1 - exp(-0.5 * radius ** 2 / std ** 2))

    assert quadpy_val == pytest.approx(my_val, rel=1e-4)

    assert quadpy_val == pytest.approx(analytic_val, rel=1e-4)

    assert my_val == pytest.approx(analytic_val, rel=1e-4)


def test_line_integral():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0]])

    scheme = quadpy.c1.gauss_patterson(8)

    def func_1(t):
        x = np.multiply.outer(t, vertices[0]) + np.multiply.outer(1 - t, vertices[1])
        return t * exp(-(x[:, 0] ** 2 + x[:, 1] ** 2) / (2 * std ** 2))

    quadpy_val_1 = scheme.integrate(func_1, [0.0, 1.0])

    def func_2(t):
        x = np.multiply.outer(1 - t, vertices[0]) + np.multiply.outer(t, vertices[1])
        return t * exp(-(x[:, 0] ** 2 + x[:, 1] ** 2) / (2 * std ** 2))

    quadpy_val_2 = scheme.integrate(func_2, [0.0, 1.0])

    my_val = phi.integrate_on_curves(np.array([vertices]))[0][:, 0, :]

    assert quadpy_val_1 == pytest.approx(my_val[0, 0], rel=1e-4)
    assert quadpy_val_2 == pytest.approx(my_val[1, 0], rel=1e-4)
    assert my_val[0, 0] == my_val[0, 1]
    assert my_val[1, 0] == my_val[1, 1]

    eta = GaussianPolynomial(np.array([[0.0, 0.0]]), np.array([1.0]), std)
    my_val_bis = eta.integrate_on_polygonal_curve(vertices)

    assert my_val_bis[0, 0] == pytest.approx(my_val[0, 0], rel=1e-4)
    assert my_val_bis[1, 0] == pytest.approx(my_val[1, 0], rel=1e-4)
    assert my_val_bis[0, 0] == my_val[0, 1]
    assert my_val_bis[1, 0] == my_val[1, 1]


# # TODO: revoir...
# def test_perimeter_grad():
#     perimeter = simple_set.compute_perimeter()
#     grad = simple_set.compute_perimeter_gradient()
#     assert grad.shape == (simple_set.num_boundary_vertices, 2)
#
#     t = 1e-5
#     new_boundary_vertices = simple_set.boundary_vertices + t * grad
#     new_simple_set = SimpleSet(new_boundary_vertices)
#     new_perimeter = new_simple_set.compute_perimeter()
#
#     finite_diff = (new_perimeter - perimeter - t * np.sum(grad * grad)) / t
#
#     assert abs(finite_diff) < 1e-4
#
#
# # TODO: revoir...
# def test_weighted_area_grad():
#     weighted_area = simple_set.compute_weighted_area(eta)
#     grad = simple_set.compute_weighted_area_gradient(eta)
#     assert grad.shape == (simple_set.num_boundary_vertices, 2)
#
#     t = 1e-5
#     new_boundary_vertices = simple_set.boundary_vertices + t * grad
#     simple_set.boundary_vertices = new_boundary_vertices
#     new_weighted_area = simple_set.compute_weighted_area(eta)
#
#     finite_diff = (new_weighted_area - weighted_area - t * np.sum(grad * grad)) / t
#
#     assert abs(finite_diff) < 1e-4def test_line_integral():
#     vertices = np.array([[0.0, 0.0], [1.0, 0.0]])
#
#     scheme = quadpy.c1.gauss_patterson(8)
#
#     def func_1(t):
#         return t * eta(np.multiply.outer(t, vertices[0]) + np.multiply.outer(1 - t, vertices[1]))
#
#     # def func_1(t):
#     #     return weight * t * exp(-np.linalg.norm(np.multiply.outer(t, vertices[0]) + np.multiply.outer(1-t, vertices[1]), axis=-1) ** 2 / (2 * std ** 2))
#
#     quadpy_val_1 = scheme.integrate(func_1, [0.0, 1.0])
#
#     def func_2(t):
#         return t * eta(np.multiply.outer(1 - t, vertices[0]) + np.multiply.outer(t, vertices[1]))
#
#     # def func_2(t):
#     #     return weight * t * exp(-np.linalg.norm(np.multiply.outer(1-t, vertices[0]) + np.multiply.outer(t, vertices[1]), axis=-1) ** 2 / (2 * std ** 2))
#
#     quadpy_val_2 = scheme.integrate(func_2, [0.0, 1.0])
#
#     my_val = eta.integrate_on_polygonal_curve(vertices)
#
#     assert quadpy_val_1 == pytest.approx(my_val[0, 0], rel=1e-4)
#     assert quadpy_val_2 == pytest.approx(my_val[0, 1], rel=1e-4)
#     assert my_val[0, 0] == my_val[1, 1]
#     assert my_val[0, 1] == my_val[1, 0]
#
#
# # TODO: revoir...
# def test_perimeter_grad():
#     perimeter = simple_set.compute_perimeter()
#     grad = simple_set.compute_perimeter_gradient()
#     assert grad.shape == (simple_set.num_boundary_vertices, 2)
#
#     t = 1e-5
#     new_boundary_vertices = simple_set.boundary_vertices + t * grad
#     new_simple_set = SimpleSet(new_boundary_vertices)
#     new_perimeter = new_simple_set.compute_perimeter()
#
#     finite_diff = (new_perimeter - perimeter - t * np.sum(grad * grad)) / t
#
#     assert abs(finite_diff) < 1e-4
#
#
# # TODO: revoir...
# def test_weighted_area_grad():
#     weighted_area = simple_set.compute_weighted_area(eta)
#     grad = simple_set.compute_weighted_area_gradient(eta)
#     assert grad.shape == (simple_set.num_boundary_vertices, 2)
#
#     t = 1e-5
#     new_boundary_vertices = simple_set.boundary_vertices + t * grad
#     simple_set.boundary_vertices = new_boundary_vertices
#     new_weighted_area = simple_set.compute_weighted_area(eta)
#
#     finite_diff = (new_weighted_area - weighted_area - t * np.sum(grad * grad)) / t
#
#     assert abs(finite_diff) < 1e-4
