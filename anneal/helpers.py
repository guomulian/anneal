import math
import numpy as np


def clip(item, lower, upper):
    """Clip item to be in [lower, upper]."""
    return max(lower, min(item, upper))


def distance(p1, p2):
    """Return the Euclidean distance between two points."""
    p1 = np.array(p1)
    p2 = np.array(p2)

    return np.linalg.norm(p1 - p2)
