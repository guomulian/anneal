from anneal import anneal
import pytest


class TrivialAnnealer(anneal.BaseAnnealer):
    def energy_method(self, state):
        return 0

    def neighbor(self, state):
        return state


class BadMaxStepsAnnealer(TrivialAnnealer):
    def __init__(self):
        super().__init__(initial_state=0, max_steps=-1)


class CustomDebugAnnealer(TrivialAnnealer):
    def debug_method(self):
        print("This is a custom debug method.")


class PlusOneAnnealer(anneal.BaseAnnealer):
    def __init__(self):
        super().__init__(initial_state=0)

    def energy_method(self, state):
        return state

    def neighbor(self, state):
        return state + 1

    def temperature(self, step):
        return 1e128  # "always" accept new states; p = exp(-1/temp)


class SmallTempAnnealer(PlusOneAnnealer):
    def temperature(self, step):
        return 1e-128  # "never" accept new states; p = exp(-1/temp)


@pytest.fixture
def trivial_annealer():
    """Annealer with constant (zero) energy and constant (zero) state."""
    return TrivialAnnealer(0)


@pytest.fixture
def custom_debug_annealer():
    """Trivial annealer with a custom debug method."""
    return CustomDebugAnnealer(0)


@pytest.fixture
def plus_one_annealer():
    """Annealer that increases its energy by one every step. (Neighboring
    states are always accepted.)
    """
    return PlusOneAnnealer()


@pytest.fixture
def small_temp_annealer():
    """Annealer with a constant small temperature."""
    return SmallTempAnnealer()
