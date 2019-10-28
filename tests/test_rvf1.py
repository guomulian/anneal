from examples.rvf1 import rvf1
from anneal import helpers
import pytest
import random


def test_initialized_with_bad_bounds():
    with pytest.raises(ValueError):
        rvf1.Rvf1(None, 0, 1000, [1, -1])

    with pytest.raises(ValueError):
        rvf1.Rvf1(None, 0, 1000, [1, 1])


def test_initialized_with_bad_function():
    with pytest.raises(TypeError):
        rvf1.Rvf1(None, 0, 1000, [-1, 1])

    with pytest.raises(TypeError):
        rvf1.Rvf1('Not a function', 0, 1000, [-1, 1])


def test_initialized_with_bad_objective():
    with pytest.raises(ValueError):
        rvf1.Rvf1(None, 0, 1000, [-1, 1], 'blah')


def test_initialized_with_initial_point_out_of_bounds():
    with pytest.raises(ValueError):
        rvf1.Rvf1(None, 2, 1000, [-1, 1])


def test_optimum_inside_bounds():
    random.seed(0)

    def fun(x):
        return x**4 - 1

    initial_point = -1
    max_steps = 1000
    bounds = [-2, 2]
    optimizer = rvf1.Rvf1(fun, initial_point, max_steps, bounds)
    min_point, min_value = optimizer.anneal()

    min_point_actual = 0
    min_value_actual = -1

    tol_point = 0.1
    tol_value = 0.01

    assert abs(min_value - min_value_actual) < tol_value
    assert helpers.distance(min_point, min_point_actual) < tol_point


def test_optimum_on_bounds():
    random.seed(0)

    def fun(x):
        return x**3 - 6*x

    initial_point = 1
    max_steps = 1000
    bounds = [-1, 1]
    objective = 'max'
    optimizer = rvf1.Rvf1(fun, initial_point, max_steps, bounds, objective)
    max_point, max_value = optimizer.anneal()

    assert max_point == -1
    assert max_value == 5
