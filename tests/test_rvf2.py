from examples.rvf2 import rvf2
import pytest
import random
import math


def test_initialized_with_bad_bounds():
    with pytest.raises(ValueError):
        rvf2.Rvf2(None, (0, 0), 1000, [[1, -1], [-1, 1]])
    
    with pytest.raises(ValueError):
        rvf2.Rvf2(None, (0, 0), 1000, [[-1, 1], [1, 1]])

def test_initialized_with_bad_function():
    with pytest.raises(TypeError):
        rvf2.Rvf2(None, (0, 0), 1000, [[-1, 1], [-1, 1]])

    with pytest.raises(TypeError):
        rvf2.Rvf2('Not a function', (0, 0), 1000, [[-1, 1], [-1, 1]])

def test_initialized_with_bad_objective():
    with pytest.raises(ValueError):
        rvf2.Rvf2(None, (0, 0), 1000, [[-1, 1], [-1, 1]], 'blah')

def test_initialized_with_initial_point_out_of_bounds():
    with pytest.raises(ValueError):
        rvf2.Rvf2(None, (2, 0), 1000, [[-1, 1], [-1, 1]])

def test_clip():
    # lower == upper
    assert rvf2.Rvf2.clip(2, 1, 1) == 1
   
    # item within bounds
    assert rvf2.Rvf2.clip(0, -1, 1) == 0

def test_optimum_inside_bounds():
    random.seed(0)

    def fun(x, y):
        return x**4 - 3*x**2 + y**4 - 3*y**2 + 1

    initial_point = (0, 0)
    max_steps = 1000
    bounds = [[-2, 2], [-2, 2]]
    optimizer = rvf2.Rvf2(fun, initial_point, max_steps, bounds)
    min_point, min_value = optimizer.anneal()

    # there are multiple solutions for this function
    min_point_actual = [(math.sqrt(3/2), math.sqrt(3/2)), (-math.sqrt(3/2), math.sqrt(3/2))]
    min_value_actual = -7/2

    tol_point = 0.1
    tol_value = 0.01

    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    min_point_passed = list(map(lambda p: distance(min_point, p) < tol_point, min_point_actual))

    assert abs(min_value - min_value_actual) < tol_value
    assert any(min_point_passed)

def test_optimum_on_bounds():
    random.seed(0)

    def fun(x, y):
        return x**3 + y**3
    
    initial_point = (0, 0)
    max_steps = 1000
    bounds = [[-1, 1], [-1, 1]]
    objective = 'max'
    optimizer = rvf2.Rvf2(fun, initial_point, max_steps, bounds, objective)
    max_point, max_value = optimizer.anneal()

    assert max_point == (1,1)
    assert max_value == 2
