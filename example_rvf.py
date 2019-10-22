import anneal
import random


class Rvf2Optimize(anneal.SimulatedAnnealer):
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
        if not bounds[0][0] <= initial_state[0] <= bounds[0][1] and \
                not bounds[1][0] <= initial_state[1] <= bounds[1][1]:
            raise ValueError('Initial state is out of bounds.')

        super().__init__(initial_state, max_steps)

    def clip(self, item, lower, upper):
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

        moved[0] = self.clip(moved[0], x_bounds[0], x_bounds[1])
        moved[1] = self.clip(moved[1], y_bounds[0], y_bounds[1])

        return tuple(moved)

    def _energy(self, state):
        if self.objective == 'min':
            return self.fun(*state)
        elif self.objective == 'max':
            return -self.fun(*state)
        else:
            raise ValueError('Objective should be either "min" or "max".')


if __name__ == '__main__':
    random.seed(0)

    def example_1(x, y):
        return x**4-3*x**2+y**4-3*y**2+1

    def example_2(x, y):
        return x**3 + y**3

    max_steps_1 = 1000
    bounds_1 = [[-2, 2], [-2, 2]]

    example_1_1 = Rvf2Optimize(example_1, (1, 0), max_steps_1, bounds_1)
    example_1_2 = Rvf2Optimize(example_1, (2, 0), max_steps_1, bounds_1)
    example_1_3 = Rvf2Optimize(example_1, (0, 0), max_steps_1, bounds_1)

    print("Solution: {}\nMin Value: {}\n".format(*example_1_1.anneal()))
    print("Solution: {}\nMin Value: {}\n".format(*example_1_2.anneal()))
    print("Solution: {}\nMin Value: {}\n".format(*example_1_3.anneal()))

    example_2_1 = Rvf2Optimize(example_2, (1, 0), max_steps_1, bounds_1)
    example_2_2 = Rvf2Optimize(example_2, (2, 0), max_steps_1, bounds_1)
    example_2_3 = Rvf2Optimize(example_2, (0, 0), max_steps_1, bounds_1)

    print("Solution: {}\nMin Value: {}\n".format(*example_2_1.anneal()))
    print("Solution: {}\nMin Value: {}\n".format(*example_2_2.anneal()))
    print("Solution: {}\nMin Value: {}\n".format(*example_2_3.anneal()))
