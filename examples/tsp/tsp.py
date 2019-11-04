from anneal import anneal, helpers
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations


class TravelingSalesPerson(anneal.BaseAnnealer):
    def __init__(self, cities, max_steps):
        n_cities = len(cities)

        if not n_cities > 0:
            raise ValueError("cities must be a non-empty list.")

        initial_state = np.array(cities, copy=True)
        np.random.shuffle(initial_state)
        super().__init__(initial_state, max_steps)

    def energy(self, state):
        """Returns the total distance of the (closed) route given by state."""
        total_distance = 0
        n = len(state)

        for i in range(n):
            total_distance += helpers.distance(state[(i + 1) % n],
                                               state[i % n])

        return total_distance

    def neighbor(self, state):
        """Reverses a random subroute."""
        n = len(state)
        neighbor = np.copy(state)

        # start of the subroute
        subroute_start = np.random.randint(n)

        # length of the subroute
        subroute_length = np.random.randint(2, n-1)

        for i in range(subroute_start, subroute_start + subroute_length // 2):
            a = i % n
            b = (subroute_length + subroute_start - i + subroute_start) % n
            neighbor[[a, b]] = neighbor[[b, a]]

        return neighbor

    def plot_state(self, state=None):  # pragma: no cover
        if state is None:
            state = np.copy(self._state)
        else:
            state = np.copy(state)

        state = np.array(state).T
        x, y = np.column_stack((state, state[:, 0]))

        plt.plot(x, y)
        plt.show()

    def brute_force(self):
        cities = np.copy(self.initial_state)
        n_cities = len(cities)

        if n_cities > 10:  # pragma: no cover
            raise RuntimeError("This is only intended for testing small-sized "
                               "problems. Right now, the limit for the length "
                               "of cities is 10. (The current length of "
                               "cities is {}.)".format(n_cities))

        best_state = np.copy(cities)
        best_energy = self.energy(cities)

        possible_solutions = permutations(cities)

        for solution in possible_solutions:
            energy = self.energy(solution)
            if energy < best_energy:
                best_state = np.copy(solution)
                best_energy = energy

        return best_state, best_energy
