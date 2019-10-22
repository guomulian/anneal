import anneal
import collections
import copy
import random


class AutoLabelPlacer(anneal.SimulatedAnnealer)
    """For placing labels automatically on a map or chart.

    Randomly places the labels on the map, then generates "neighboring"
    placements by selecting a random label and moving it slightly.

    The energy/cost function should penalize undesired qualities, such as
    overlapping labels and labels far from their anchor points.
    """

    def __init__(self, initial_state, max_steps):
        super().__init__(initial_state, max_steps)

    def _neighbor(self):
        """Returns a randomly selected "neighboring" map.

        This is done by picking a random label from the list then
        translating it slightly.
        """
        pass

    def _energy(self, state):
        """Scores a map based on the "quality" of the labeling.
        
        This will be defined by the following helper "cost" functions;
        new ones may be added as desired.
        """
         pass


if __name__ == '__main__':
    initial_state = None
    max_steps = 500
    placer = AutoLabelPlacer(inital_state, max_steps)
