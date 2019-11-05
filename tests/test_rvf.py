from examples.rvf.rvf import RvfSolver
from anneal import helpers
import numpy as np
import pytest
import random


def rvf_1_basic(x):
    return x**4 - 1


def rvf_1_basic_inverted(x):
    return -1*rvf_1_basic(x)


def rvf_2_basic(x, y):
    return x**4 + y**4 - 1


def rvf_3_basic(x, y, z):
    return x**3 + y**3 + z**3 - 1


@pytest.mark.parametrize("bad_value", [[[1, -1]], [[1, 1]]])
def test_initialized_with_bad_bounds(bad_value):
    with pytest.raises(ValueError):
        RvfSolver(rvf_1_basic, [1], bad_value)


@pytest.mark.parametrize("bad_value", [None, "foo", 4])
def test_initialized_with_bad_function(bad_value):
    with pytest.raises(TypeError):
        solver = RvfSolver(bad_value, [0, 0], [[-1, 1], [-1, 1]])


def test_initialized_with_bad_objective():
    with pytest.raises(ValueError):
        solver = RvfSolver(rvf_1_basic, [1], [[-1, 1]], objective="blah")


@pytest.mark.parametrize("function, point, bounds", [
        (rvf_2_basic, [0], [[-1, 1], [-1, 1]]),
        (rvf_3_basic, [2, 0, 1], [[-1, 1], [-1, 1]])
        ])
def test_initialized_with_wrong_shapes(function, point, bounds):
    with pytest.raises(ValueError):
        RvfSolver(function, point, bounds)


@pytest.mark.parametrize("function, point, bounds", [
        (rvf_2_basic, [2, 0], [[-1, 1], [-1, 1]]),
        (rvf_3_basic, [2, 0, 1], [[-1, 1], [-1, 1], [-1, 1]])
        ])
def test_initialized_with_initial_point_out_of_bounds(function, point, bounds):
    with pytest.raises(ValueError):
        RvfSolver(function, point, bounds)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("function, initial_point, bounds, obj, actual", [
        (rvf_1_basic, [1], [[-2, 2]], 'min', [0]),
        (rvf_2_basic, [1, 2], [[-2, 2], [-2, 2]], 'min', [0, 0]),
        (rvf_1_basic_inverted, [1], [[-2, 2]], 'max', [0])
    ])
def test_optimum_inside_bounds(function, initial_point, bounds, obj, actual,
                               seed):
    random.seed(seed)
    np.random.seed(seed)

    solver = RvfSolver(function, initial_point, bounds, objective=obj)
    point, value = solver.anneal(max_steps=4000)

    # tolerances
    tol_point = 0.1
    tol_value = 0.1

    actual = np.array(actual)

    assert abs(value - function(*actual)) < tol_value
    assert helpers.distance(actual, point) < tol_point


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("function, initial_point, bounds, actual", [
        (rvf_3_basic, [1, 1, -1], [[-2, 2], [-2, 2], [-2, 2]], [-2, -2, -2])
    ])
def test_optimum_on_bounds(function, initial_point, bounds, actual, seed):
    random.seed(seed)
    np.random.seed(seed)

    solver = RvfSolver(function, initial_point, bounds)
    point, value = solver.anneal(max_steps=4000)

    # tolerances
    tol_point = 0.1
    tol_value = 0.1

    actual = np.array(actual)

    assert abs(value - function(*actual)) < tol_value
    assert helpers.distance(actual, point) < tol_point
