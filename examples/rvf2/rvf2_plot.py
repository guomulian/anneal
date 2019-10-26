from anneal import anneal
from examples.rvf2 import rvf2
import random

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    DIR_NAME = os.path.dirname(os.path.abspath(__file__))
    PICKLE_NAME = 'rvf_2.pickle'
    FILENAME = os.path.join(DIR_NAME, PICKLE_NAME)

    def fun_1(x, y):
        return x**4 - 3*x**2 + y**4 - 3*y**2 + 1

    def fun_2(x, y):
        return x**3 + y**3

    def fun_3(x, y):
        return np.sin(x) + y**2

    def fun_4(x, y):
        return x**2 + y**2

    def fun_5(x, y):
        return np.sin(x*y)

    # random.seed(0)
    fun = fun_3
    initial_point = (1, 0)
    bounds = [[-5, 5], [-5, 5]]
    max_steps = 100
    objective = 'min'
    solver = rvf2.Rvf2(fun, initial_point, max_steps, bounds, objective)

    best_state, best_energy = solver.anneal(
                                verbose=2,
                                debug=True,
                                filename=FILENAME
                                )

    intermediate_states = solver.unpickle_states(FILENAME)

    xp = np.array([s[0] for s in intermediate_states])
    yp = np.array([s[1] for s in intermediate_states])
    zp = fun(xp, yp)
    tp = np.arange(len(intermediate_states))

    steps = 100
    x = np.linspace(*bounds[0], num=steps)
    y = np.linspace(*bounds[1], num=steps)

    X, Y = np.meshgrid(x, y)
    Z = fun(X, Y)

    # plt.style.use('ggplot')
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle("Anneal steps for {} with initial_point={} and max_steps={}\n\
                  Best State: {} Best Energy: {}"
                 .format(fun, initial_point, max_steps, best_state,
                         best_energy))

    # contour plot
    ax = fig.add_subplot(1, 2, 1)
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])

    contourf = ax.contourf(X, Y, Z, zorder=0, alpha=0.5)
    contour = ax.contour(X, Y, Z, zorder=1)
    ax.clabel(contour, inline=1, fontsize=10)

    plt.plot(xp, yp, c='gray', alpha=0.8, zorder=2)
    sc = plt.scatter(xp, yp, c=tp, cmap=cm.coolwarm, edgecolors='k',
                     clip_on=False, zorder=3)

    cb = plt.colorbar(sc)
    cb.ax.set_ylabel("Step")

    # surface plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot3D(xp, yp, zp, c='gray', )
    ax.scatter3D(xp, yp, zp, c=tp, cmap=cm.coolwarm,
                 s=10, edgecolors='k')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False,
                    alpha=0.15)

    plt.show()

    print("Solution: {}\nMin Value: {}\n".format(best_state, best_energy))
