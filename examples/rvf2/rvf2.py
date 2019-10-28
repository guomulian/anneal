from anneal import anneal, helpers
import pickle
import random


class Rvf2(anneal.BaseAnnealer):
    """For estimating a global optimum of a function f: R^2 -> R."""

    def __init__(self, fun, initial_state, max_steps, bounds, objective='min'):
        self.fun = fun
        self.objective = objective

        # bounds must be of the form (b_min, b_max), where b_min < b_max
        if bounds[0][0] < bounds[0][1] and bounds[1][0] < bounds[1][1]:
            self.bounds = bounds
        else:
            raise ValueError('Invalid bounds.')

        # initial state must be within the bounds
        if not bounds[0][0] <= initial_state[0] <= bounds[0][1]\
                or not bounds[1][0] <= initial_state[1] <= bounds[1][1]:
            raise ValueError('Initial state is out of bounds.')

        super().__init__(initial_state, max_steps)

    def _neighbor(self, state):
        x_bounds, y_bounds = self.bounds

        dx = 0.1*abs(x_bounds[1]-x_bounds[0])
        dy = 0.1*abs(y_bounds[1]-y_bounds[0])

        dx *= random.uniform(-1, 1)
        dy *= random.uniform(-1, 1)

        # Make sure the coordinate stays in the bounding region.
        # There are likely better ways to do this; this is just a simple option

        x = helpers.clip(state[0] + dx, *x_bounds)
        y = helpers.clip(state[1] + dy, *y_bounds)

        return (x, y)

    def _energy(self, state):
        if self.objective == 'min':
            return self.fun(*state)
        elif self.objective == 'max':
            return -self.fun(*state)
        else:
            raise ValueError('Objective should be either "min" or "max".')

    def formatter(self, output):
        if self.objective == 'max':
            return output[0], -output[1]
        else:
            return output
