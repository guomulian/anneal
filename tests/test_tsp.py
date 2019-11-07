from examples.tsp.tsp import TravelingSalesPerson
import pytest
import random
import numpy as np


@pytest.fixture
def five_cities():
    """TravelingSalesPerson solver with five cities."""
    n_cities = 5
    cities = np.random.randn(n_cities, 2)
    solver = TravelingSalesPerson(cities)
    return solver


@pytest.fixture
def five_cities_on_line():
    """TravelingSalesPerson solver with five cities on the horizontal axis."""
    n_points = 5
    cities = [(i, 0) for i in range(n_points)]
    solver = TravelingSalesPerson(cities)
    return solver


def test_neighbor_reverse_across_end():
    random.seed(0)
    np.random.seed(0)

    n_points = 10
    cities = [(i, i) for i in range(n_points)]

    solver = TravelingSalesPerson(cities)

    # this is specific to this particular seed
    before = np.copy(solver.initial_state)
    after = np.copy(solver.neighbor(before))

    before_subroute = np.concatenate((before[4:], before[:3]))
    after_subroute = np.concatenate((after[4:], after[:3]))

    assert (before_subroute == after_subroute[::-1]).all()


def test_energy_method(five_cities_on_line):
    assert five_cities_on_line.energy == 8


def test_invalid_cities():
    with pytest.raises(ValueError):
        solver = TravelingSalesPerson([])


def test_small_cities_compared_to_brute_force(five_cities):
    random.seed(0)
    np.random.seed(0)

    n_runs = 20
    _, energies = five_cities.run(n_runs=n_runs,
                                  max_steps=1000,
                                  energy_break_rounds=10,
                                  energy_break_tol=1e-5)
    _, bf_energy = five_cities.brute_force()

    n_good = sum(abs(e - bf_energy)/bf_energy < 1e-2 for e in energies)

    assert n_good/n_runs > 0.8
