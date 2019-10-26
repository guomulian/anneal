from anneal import anneal
import random


class Rvf2(anneal.SimulatedAnnealer):
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

    @staticmethod
    def clip(item, lower, upper):
        """Clip item to be in [lower, upper]."""
        return max(lower, min(item, upper))

    def _neighbor(self):
        x_bounds, y_bounds = self.bounds

        dx = 0.1*abs(x_bounds[1]-x_bounds[0])
        dy = 0.1*abs(y_bounds[1]-y_bounds[0])

        dx *= random.uniform(-1, 1)
        dy *= random.uniform(-1, 1)

        moved = list(map(sum, zip(self.state, (dx, dy))))

        # Make sure the coordinate stays in the bounding region.
        # There are likely better ways to do this; this is just a simple option

        moved[0] = Rvf2.clip(moved[0], x_bounds[0], x_bounds[1])
        moved[1] = Rvf2.clip(moved[1], y_bounds[0], y_bounds[1])

        return tuple(moved)

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


if __name__ == '__main__':
    random.seed(0)

    def f_1(x, y):
        return x**4-3*x**2+y**4-3*y**2+1

    def f_2(x, y):
        return x**3 + y**3

    bounds_1 = [[-2, 2], [-2, 2]]

    example_11 = Rvf2(f_1, (1, 0), 1000, bounds_1)
    example_12 = Rvf2(f_1, (2, 0), 1000, bounds_1)
    example_13 = Rvf2(f_1, (0, 0), 1000, bounds_1)

    print("Minimizing: {}...".format(f_1))
    print("Solution: {}\nMin Value: {}\n".format(*example_11.anneal()))
    print("Solution: {}\nMin Value: {}\n".format(*example_12.anneal()))
    print("Solution: {}\nMin Value: {}\n".format(*example_13.anneal()))

    bounds_2 = [[-1, 1], [-1, 1]]

    example_21 = Rvf2(f_2, (1, 0), 1000, bounds_2, 'max')
    example_22 = Rvf2(f_2, (0.5, 0), 1000, bounds_2, 'max')
    example_23 = Rvf2(f_2, (-1, 0), 1000, bounds_2, 'max')

    print("Maximizing: {}".format(f_2))
    print("Solution: {}\nMax Value: {}\n".format(*example_21.anneal()))
    print("Solution: {}\nMax Value: {}\n".format(*example_22.anneal()))
    print("Solution: {}\nMax Value: {}\n".format(*example_23.anneal()))
