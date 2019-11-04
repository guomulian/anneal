from examples.tsp.tsp import TravelingSalesPerson
import pytest
import random
import numpy as np


def is_rotation(a, b):
    if len(a) != len(b):
        raise ValueError("States must have the same length.")

    n = len(a)

    for shift in range(n):
        rolled = np.roll(a, shift, axis=0)
        if (rolled == b).all():
            return True

    return False


@pytest.fixture
def five_cities():
    n_cities = 5
    max_steps = 1000
    cities = np.random.randn(n_cities, 2)
    solver = TravelingSalesPerson(cities, max_steps)
    return solver


def test_neighbor_reverse_across_end():
    random.seed(0)
    np.random.seed(0)

    n_points = 10
    max_steps = 500
    cities = [(i, i) for i in range(n_points)]

    solver = TravelingSalesPerson(cities, max_steps)

    # this is specific to the given seed
    before = np.copy(solver.initial_state)
    after = np.copy(solver.neighbor(before))

    before_subroute = np.concatenate((before[4:], before[:3]))
    after_subroute = np.concatenate((after[4:], after[:3]))

    assert (before_subroute == after_subroute[::-1]).all()


def test_energy():
    n_points = 5
    cities = [(i, 0) for i in range(n_points)]
    assert TravelingSalesPerson.energy(None, cities) == 8


def test_invalid_cities():
    with pytest.raises(ValueError):
        solver = TravelingSalesPerson([], max_steps=100)


def test_small_cities(five_cities):
    n_runs = 100
    _, energies = five_cities.run(n_runs=n_runs,
                                  max_steps=10000,
                                  energy_exit_rounds=10,
                                  energy_exit_tol=1e-7)
    _, bf_energy = five_cities.brute_force()

    n_good = sum(abs(e - bf_energy)/bf_energy < 1e-2 for e in energies)

    assert n_good/n_runs > 0.8
