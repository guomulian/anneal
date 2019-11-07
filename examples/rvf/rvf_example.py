import rvf
import random
import numpy as np


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    # ignore divide by zero warning
    np.seterr(divide='ignore')

    # Example from:
    # https://mathoverflow.net/questions/253450/can-you-give-me-good-examples-of-non-convex-functions-that-are-problematic-for-o
    def f_1(x):
        return x**2 + np.exp(-1/(100*(x - 1))**2) - 1

    def f_2(x, y):
        return np.sin(x) + y**2

    def f_3(x, y, z):
        return x + y**2 + z

    point_1 = [1]
    point_2 = [-1, 1]
    point_3 = [-1, 1, 1]

    bounds_1 = [[-2, 2]]
    bounds_2 = [[-2, 2], [-2, 2]]
    bounds_3 = [[-2, 2], [-2, 2], [-2, 2]]

    solver_1 = rvf.RvfSolver(f_1, point_1, bounds_1, max_steps=5000)
    solver_2 = rvf.RvfSolver(f_2, point_2, bounds_2, max_steps=5000)
    solver_3 = rvf.RvfSolver(f_3, point_3, bounds_3, max_steps=5000)

    # pick one
    solver = solver_1
    n_runs = 10

    print("Optimizing {} with max_steps = {} for {} runs:".format(
                                                        solver.function,
                                                        solver.max_steps,
                                                        n_runs))

    for i in range(n_runs):
        s, e = solver.anneal(n_runs)
        print("Run {}: {} {}".format(i, s, e))
