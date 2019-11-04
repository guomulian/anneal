from examples.tsp import tsp
import random
import logging
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(message)s')


np.random.seed(1)
random.seed(0)

n_points = 10
max_steps = 5000
cities = np.random.rand(n_points, 2)

solver = tsp.TravelingSalesPerson(cities, max_steps)
_, energy = solver.anneal(verbose=2)

print("Shortest distance found: {}".format(energy))
solver.plot_state()
