from examples.tsp.tsp import TravelingSalesPerson
import pytest
import random


def test_neighbor_reverse_across_end():
    random.seed(0)

    n_points = 10
    max_steps = 500
    cities = [(i, i) for i in range(n_points)]

    solver = TravelingSalesPerson(cities, max_steps)

    # this is specific to seed 0
    before = solver.initial_state
    before_subroute = before[5:] + before[:2]
    after = solver._neighbor(solver.initial_state)
    after_subroute = after[5:] + after[:2]

    assert before_subroute == after_subroute[::-1]


def test_energy():
    n_points = 5
    cities = [(i, 0) for i in range(n_points)]
    solver = TravelingSalesPerson(cities, max_steps=100)
    assert solver._energy(cities) == 8


def test_invalid_cities():
    with pytest.raises(ValueError):
        solver = TravelingSalesPerson([], max_steps=100)
