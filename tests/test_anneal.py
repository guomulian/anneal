from anneal import anneal
import pytest
import random


def test_initialized_without_abstract_methods():
    with pytest.raises(TypeError):
        anneal.SimulatedAnnealer(None, 100)


class AnnealerWithBadMaxSteps(anneal.SimulatedAnnealer):
    def __init__(self):
        super().__init__(None, 0)

    def _energy(self, state):
        return 0

    def _neighbor(self, state):
        return state


def test_initialized_with_bad_max_steps():
    with pytest.raises(ValueError):
        AnnealerWithBadMaxSteps()
