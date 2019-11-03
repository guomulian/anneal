from examples.tsp import tsp
import random
import logging
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(message)s')

random.seed(0)

n_points = 30
max_steps = 10000
cities = [(random.uniform(-10, 10), random.uniform(-10, 10))
          for _ in range(n_points)]

solver = tsp.TravelingSalesPerson(cities, max_steps)
state, energy = solver.anneal(verbose=2,  energy_exit_rounds=50,
                              energy_exit_tol=1e-6)


def get_xy(state):
    """Append the first city to the end and transpose so that state is in a
    plt.plot-friendly shape.
    """
    state = np.array(state).T
    state = np.column_stack((state, state[:, 0]))
    return state


x, y = get_xy(state)

plt.plot(x, y)
plt.show()
