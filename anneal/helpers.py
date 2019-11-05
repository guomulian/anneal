import copy
import math
import numpy as np
import time
import timeit


def clip(item, lower, upper):
    """Clip item to be in [lower, upper]."""
    return max(lower, min(item, upper))


def distance(p1, p2):
    """Return the Euclidean distance between two points."""
    r1 = np.array(p1, copy=True)
    r2 = np.array(p2, copy=True)

    return np.linalg.norm(r1 - r2)


def generate_filename(obj, extension):
    """Generates a timestamp-based filename prefixed with the class name of the
    given object.
    """
    timestamp = time.strftime("-%Y%m%d-%H%M%S")
    filename = obj.__class__.__name__ + timestamp + extension
    return filename


def timed(function):
    def timed_function(*args, **kwargs):
        start = timeit.default_timer()
        result = function(*args, **kwargs)
        end = timeit.default_timer()
        print("Execution of {} took {} seconds"
              .format(function.__name__, end - start))
        return result
    return timed_function
