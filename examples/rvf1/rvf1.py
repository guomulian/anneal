from anneal import anneal
import pickle
import random


class Rvf1(anneal.SimulatedAnnealer):
    """For estimating a global optimum of a function f: R^2 -> R."""

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

    @staticmethod
    def clip(item, lower, upper):
        """Clip item to be in [lower, upper]."""
        return max(lower, min(item, upper))

    def _neighbor(self):
        bounds = self.bounds

        dx = 0.1*abs(bounds[1]-bounds[0])
        dx *= random.uniform(-1, 1)

        moved = self.state + dx

        # Make sure the coordinate stays in the bounding region.
        # There are likely better ways to do this; this is just a simple option

        moved = Rvf1.clip(moved, *bounds)

        return moved

    def _energy(self, state):
        if self.objective == 'min':
            return self.fun(state)
        elif self.objective == 'max':
            return -self.fun(state)
        else:
            raise ValueError('Objective should be either "min" or "max".')

    def formatter(self, output):
        if self.objective == 'max':
            return output[0], -output[1]
        else:
            return output

    def debug_method(self, *args, **kwargs):
        filename = kwargs['filename']

        if self.step == 0:
            mode = 'wb'
        else:
            mode = 'ab'

        with open(filename, mode) as file:
            pickle.dump(self.state, file)

    @staticmethod
    def unpickle_states(filename):
        states = []

        with open(filename, 'rb') as file:
            while True:
                try:
                    states.append(pickle.load(file))
                except EOFError:
                    break
        return states


if __name__ == '__main__':
    random.seed(0)

    def f_1(x):
        return x**4 - 1

    def f_2(x):
        return x**3 - 6*x

    bounds_1 = [-2, 2]

    example_11 = Rvf1(f_1, 1, 1000, bounds_1)
    example_12 = Rvf1(f_1, 2, 1000, bounds_1)
    example_13 = Rvf1(f_1, -1, 1000, bounds_1)

    print("Minimizing: {}...".format(f_1))
    print("Solution: {}\nMin Value: {}\n".format(*example_11.anneal()))
    print("Solution: {}\nMin Value: {}\n".format(*example_12.anneal()))
    print("Solution: {}\nMin Value: {}\n".format(*example_13.anneal()))

    bounds_2 = [-2, 2]

    example_21 = Rvf1(f_2, 0, 1000, bounds_2, 'max')
    example_22 = Rvf1(f_2, 0.5, 1000, bounds_2, 'max')
    example_23 = Rvf1(f_2, -1, 1000, bounds_2, 'max')

    print("Maximizing: {}".format(f_2))
    print("Solution: {}\nMax Value: {}\n".format(*example_21.anneal()))
    print("Solution: {}\nMax Value: {}\n".format(*example_22.anneal()))
    print("Solution: {}\nMax Value: {}\n".format(*example_23.anneal()))