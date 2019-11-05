from anneal import anneal, helpers
import pickle
import random


class Rvf1(anneal.BaseAnnealer):
    """For estimating a global optimum of a function f: R -> R."""

    def __init__(self, fun, initial_state, max_steps, bounds, objective='min'):
        self.fun = fun
        self.objective = objective

        # bounds must be of the form (b_min, b_max), where b_min < b_max
        if bounds[0] < bounds[1]:
            self.bounds = bounds
        else:
            raise ValueError('Invalid bounds.')

        # initial state must be within the bounds
        if not bounds[0] <= initial_state <= bounds[1]:
            raise ValueError('Initial state is out of bounds.')

        super().__init__(initial_state, max_steps)

    def neighbor(self, state):
        bounds = self.bounds

        dx = 0.1*abs(bounds[1]-bounds[0])
        dx *= random.uniform(-1, 1)

        moved = state + dx

        # Make sure the coordinate stays in the bounding region.
        # There are likely better ways to do this; this is just a simple option

        moved = helpers.clip(moved, *bounds)

        return moved

    def energy_method(self, state):
        if self.objective == 'min':
            return self.fun(state)
        elif self.objective == 'max':
            return -self.fun(state)
        else:
            raise ValueError('Objective should be either "min" or "max".')

    def format_output(self, output):
        if self.objective == 'max':
            return output[0], -output[1]
        else:
            return output
