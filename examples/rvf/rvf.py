from anneal import anneal
import numpy as np
import random
from inspect import signature


class RvfSolver(anneal.BaseAnnealer):
    """For estimating a global optimum of a function f: R^n -> R."""

    def __init__(self, function, initial_state, bounds, *args, **kwargs):
        bounds = np.array(bounds)
        initial_state = np.array(initial_state)

        if (bounds[:, 0] < bounds[:, 1]).all():
            self.bounds = bounds
        else:
            raise ValueError('Invalid bounds.')

        if not ((bounds[:, 0] <= initial_state).all() and
                (bounds[:, 1] >= initial_state).all()):
            raise ValueError('Initial state is out of bounds.')

        n_parameters = len(signature(function).parameters)

        if not n_parameters == len(bounds) == len(initial_state):
            raise ValueError("Number of arguments of function, length of"
                             "bounds, and length of initial state all have to"
                             "be the same.")

        objective = kwargs.get("objective", "min")

        if objective in ["min", "max"]:
            self.objective = objective
        else:
            raise ValueError("Objective must be either 'min' or 'max'.")

        self.function = function
        self.n_parameters = n_parameters

        super().__init__(initial_state, *args, **kwargs)

    def copy_method(self, state):
        return np.copy(state)

    def neighbor(self, state, scale=1):
        # "scale" of each dimension
        sizes = abs(self.bounds[:, 1] - self.bounds[:, 0])

        # pick direction: each component in [-1, 1)
        dx = 2*np.random.random(self.n_parameters) - 1
        dx = np.multiply(dx, scale*sizes)

        moved = state + dx

        # Make sure the coordinate stays in the bounding region.
        # There are likely better ways to do this; this is just a simple option

        moved = np.clip(moved, self.bounds[:, 0], self.bounds[:, 1])

        return moved

    def energy_method(self, state):
        if self.objective == 'min':
            return self.function(*state)
        elif self.objective == 'max':
            return -self.function(*state)
        else:
            raise ValueError('Objective should be either "min" or "max".')

    def format_output(self, output):
        if self.objective == 'max':
            return output[0], -output[1]
        elif self.objective == 'min':
            return output
        else:
            raise ValueError('Objective should be either "min" or "max".')
