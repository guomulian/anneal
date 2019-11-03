from anneal import anneal
import random
import math
import copy


class TravelingSalesPerson(anneal.BaseAnnealer):
    def __init__(self, cities, max_steps):
        if not len(cities) > 0:
            raise ValueError("cities must be a non-empty list.")

        initial_state = copy.deepcopy(cities)
        random.shuffle(initial_state)
        super().__init__(initial_state, max_steps)

    @staticmethod
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _energy(self, state):
        """Returns the total distance of the (closed) route given by state."""
        total_distance = 0
        n = len(state)

        for i in range(n):
            total_distance += TravelingSalesPerson.distance(state[(i + 1) % n],
                                                            state[i % n])

        return total_distance

    def _neighbor(self, state):
        """Reverses a random subroute."""
        n = len(state)
        neighbor = copy.deepcopy(self.state)

        # start of the subroute
        subroute_start = random.randrange(n)

        # length of the subroute
        subroute_length = random.randrange(2, n-1)

        for i in range(subroute_start, subroute_start + subroute_length // 2):
            a = i % n
            b = (subroute_length - i) % n
            neighbor[a], neighbor[b] = neighbor[b], neighbor[a]

        return neighbor
