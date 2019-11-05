import rvf
import random
import numpy as np


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    def f_1(x):
        return x**4 - 1

    def f_2(x, y):
        return np.sin(x) + y**2

    def f_3(x, y, z):
        return x + y**2 + z

    def f_4(x, y, z, w):
        return x + y**2 + z*w

    bounds_1 = [[-2, 2]]
    bounds_2 = [[-2, 2], [-2, 2]]
    bounds_3 = [[-2, 2], [-2, 2], [-2, 2]]
    bounds_4 = [[-2, 2], [-2, 2], [-2, 2], [-2, 2]]

    solver_1 = rvf.RvfSolver(f_1, [1], bounds_1, max_steps=4000,
                             objective='max')
    solver_2 = rvf.RvfSolver(f_2, [-1, 1], bounds_2, max_steps=5000)
    solver_3 = rvf.RvfSolver(f_3, [-1, 1, 1], bounds_3, max_steps=5000)
    solver_4 = rvf.RvfSolver(f_4, [-1, 1, 1, -1], bounds_4, max_steps=5000)

    # pick one
    solver = solver_2

    print("Optimizing {} with max_steps = {}".format(solver.function,
                                                     solver.max_steps))
    s, e = solver.run(10)

    for i in range(10):
        print("Run #{}:".format(i+1))
        print("\tState: {}".format(s[i]))
        print("\tEnergy: {}".format(e[i]))
