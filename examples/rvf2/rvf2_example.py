import rvf2
import random


if __name__ == '__main__':
    random.seed(0)

    def f_1(x, y):
        return x**4-3*x**2+y**4-3*y**2+1

    def f_2(x, y):
        return x**3 + y**3

    bounds_1 = [[-2, 2], [-2, 2]]

    example_11 = rvf2.Rvf2(f_1, (1, 0), 1000, bounds_1)
    example_12 = rvf2.Rvf2(f_1, (2, 0), 1000, bounds_1)
    example_13 = rvf2.Rvf2(f_1, (0, 0), 1000, bounds_1)

    print("Minimizing: {}...".format(f_1))
    print("Solution: {}\nMin Value: {}\n".format(*example_11.anneal()))
    print("Solution: {}\nMin Value: {}\n".format(*example_12.anneal()))
    print("Solution: {}\nMin Value: {}\n".format(*example_13.anneal()))

    bounds_2 = [[-1, 1], [-1, 1]]

    example_21 = rvf2.Rvf2(f_2, (1, 0), 1000, bounds_2, 'max')
    example_22 = rvf2.Rvf2(f_2, (0.5, 0), 1000, bounds_2, 'max')
    example_23 = rvf2.Rvf2(f_2, (-1, 0), 1000, bounds_2, 'max')

    print("Maximizing: {}".format(f_2))
    print("Solution: {}\nMax Value: {}\n".format(*example_21.anneal()))
    print("Solution: {}\nMax Value: {}\n".format(*example_22.anneal()))
    print("Solution: {}\nMax Value: {}\n".format(*example_23.anneal()))
