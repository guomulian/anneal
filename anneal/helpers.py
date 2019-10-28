import math
import numpy as np
import time


def clip(item, lower, upper):
    """Clip item to be in [lower, upper]."""
    return max(lower, min(item, upper))


def distance(p1, p2):
    """Return the Euclidean distance between two points."""
    p1 = np.array(p1)
    p2 = np.array(p2)

    return np.linalg.norm(p1 - p2)


def generate_filename(obj, extension):
    """Generates a timestamp-based filename prefixed with the class name of the
    given object.
    """
    timestamp = time.strftime("-%Y%m%d-%H%M%S")
    filename = obj.__class__.__name__ + timestamp + extension
    return filename
