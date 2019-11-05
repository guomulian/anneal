import rvf1
import random


if __name__ == '__main__':
    random.seed(0)

    def f_1(x):
        return x**4 - 1

    def f_2(x):
        return x**3 - 6*x

    bounds_1 = [-2, 2]

    example_11 = rvf1.Rvf1(f_1, 1, 1000, bounds_1)
    example_12 = rvf1.Rvf1(f_1, 2, 1000, bounds_1)
    example_13 = rvf1.Rvf1(f_1, -1, 1000, bounds_1)

    print("Minimizing: {}...".format(f_1))
    print("Solution: {}\nMin Value: {}\n".format(*example_11.anneal()))
    print("Solution: {}\nMin Value: {}\n".format(*example_12.anneal()))
    print("Solution: {}\nMin Value: {}\n".format(*example_13.anneal()))

    bounds_2 = [-2, 2]

    example_21 = rvf1.Rvf1(f_2, 0, 1000, bounds_2, 'max')
    example_22 = rvf1.Rvf1(f_2, 0.5, 1000, bounds_2, 'max')
    example_23 = rvf1.Rvf1(f_2, -1, 1000, bounds_2, 'max')

    print("Maximizing: {}".format(f_2))
    print("Solution: {}\nMax Value: {}\n".format(*example_21.anneal()))
    print("Solution: {}\nMax Value: {}\n".format(*example_22.anneal()))
    print("Solution: {}\nMax Value: {}\n".format(*example_23.anneal()))
