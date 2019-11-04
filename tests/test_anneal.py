from anneal import anneal
from collections import deque
import logging
import pytest
import random
import sys


class AnnealerWithBadMaxSteps(anneal.BaseAnnealer):
    def __init__(self):
        super().__init__(initial_state=None, max_steps=-1)

    def energy(self, state):
        return 0

    def neighbor(self, state):
        return state


class AnnealerWithConstantEnergy(anneal.BaseAnnealer):
    def __init__(self):
        super().__init__(initial_state=None, max_steps=100)

    def energy(self, state):
        return 0

    def neighbor(self, state):
        return state


class AnnealerWithCustomDebugMethod(anneal.BaseAnnealer):
    def __init__(self):
        super().__init__(initial_state=None, max_steps=1000)

    def energy(self, state):
        return 0

    def neighbor(self, state):
        return state

    def debug_method(self, *args, **kwargs):
        print("This is a custom debug method.")


class AnnealerWithConstantSmallTemperature(anneal.BaseAnnealer):
    def __init__(self):
        super().__init__(initial_state=0, max_steps=100)

    def energy(self, state):
        return state

    def neighbor(self, state):
        return state - 1

    def temperature(self, step):
        return 1e-128

    def debug_method(self, *args, **kwargs):
        print("State accepted.")


class AnnealerMinusOne(anneal.BaseAnnealer):
    def __init__(self):
        super().__init__(initial_state=0, max_steps=100)

    def energy(self, state):
        return state

    def neighbor(self, state):
        return state - 1

    def temperature(self, step):
        return 1e-128


def test_initialized_without_abstract_methods():
    with pytest.raises(TypeError):
        anneal.BaseAnnealer(None, 100)


def test_anneal_with_bad_max_steps():
    with pytest.raises(ValueError):
        annealer = AnnealerWithBadMaxSteps()
        annealer.anneal()


def test_anneal_with_max_step_change():
    annealer = AnnealerWithBadMaxSteps()
    annealer.anneal(max_steps=50)


def test_anneal_unpickle_states_without_file():
    annealer = AnnealerWithConstantEnergy()
    annealer.anneal()

    with pytest.raises(FileNotFoundError):
        annealer.unpickle_states()


def test_accept_state_overflow(capsys):
    annealer = AnnealerWithConstantSmallTemperature()
    annealer.anneal(verbose=2, debug=True, temp_tol=-1)
    captured = capsys.readouterr()
    expected = "State accepted.\n"

    # We expect the state to be accepted every step that we get an
    # OverflowError from _acceptance_probability. This should happen every step
    # when the temperature is very small, which is always for this annealer.

    # Since debug_method is called every time a state is accepted, we expect
    # max_steps prints of the output of debug_method.

    # (We disable the "temperature exit" by setting temp_tol to -1.)

    assert captured.out == expected*annealer.max_steps


def test_run():
    annealer = AnnealerWithConstantEnergy()
    _, energies = annealer.run(50)

    assert all(e == 0 for e in energies)


def test_run_parameter_change():
    annealer = AnnealerWithConstantEnergy()
    energies = annealer.run(50, max_steps=10)
    assert annealer.max_steps == 10


def test_bad_verbose_value():
    annealer = AnnealerWithConstantEnergy()

    with pytest.raises(ValueError):
        annealer.anneal(verbose=5)


def test_bad_energy_exit_rounds_value():
    annealer = AnnealerWithConstantEnergy()

    with pytest.raises(TypeError):
        annealer.anneal(energy_exit_rounds=0.2)


def test_energy_queue_clears():
    annealer = AnnealerWithConstantEnergy()
    annealer._energy_queue = deque([0], maxlen=2)
    annealer._energy_queue_handler(energy=1, tol=0.1)
    assert annealer._energy_queue == deque([1], maxlen=2)


def test_debug_handler(capsys):
    annealer = AnnealerWithCustomDebugMethod()
    custom_string = "This is a custom debug method.\n"

    # should print custom_string 10 times
    annealer.anneal(debug=True)
    captured = capsys.readouterr()
    assert captured.out == custom_string*10

    # should print custom_string 100 times
    annealer.anneal(verbose=1, debug=True)
    captured = capsys.readouterr()
    assert captured.out == custom_string*100

    # should print custom_string 1000 times
    annealer.anneal(verbose=2, debug=True)
    captured = capsys.readouterr()
    assert captured.out == custom_string*1000


def test_verbose_max_steps(caplog):
    annealer = AnnealerWithConstantEnergy()

    with caplog.at_level(logging.INFO):
        annealer.anneal(verbose=1)
        expected_end = "Finished - Reached max steps (max_steps = {}).\n"\
                       .format(annealer.max_steps)
        assert caplog.text.endswith(expected_end)

        annealer.anneal(verbose=2)
        expected_end = "Finished - Reached max steps (max_steps = {}).\n"\
                       .format(annealer.max_steps)
        assert caplog.text.endswith(expected_end)


def test_verbose_temp_break(caplog):
    annealer = AnnealerWithConstantSmallTemperature()
    temp_tol = 0.1

    with caplog.at_level(logging.INFO):
        annealer.anneal(verbose=1, temp_tol=temp_tol)
        expected_end = "Finished - Reached temperature tolerance (tol = {})."\
                       "\n".format(temp_tol)
        assert caplog.text.endswith(expected_end)

        annealer.anneal(verbose=2, temp_tol=temp_tol)
        expected_end = "Finished - Reached temperature tolerance (tol = {})."\
                       "\n".format(temp_tol)
        assert caplog.text.endswith(expected_end)


def test_verbose_energy_break(caplog):
    annealer = AnnealerWithConstantEnergy()
    temp_tol = -1
    energy_exit_rounds = 3
    energy_exit_tol = 0.1

    with caplog.at_level(logging.INFO):
        annealer.anneal(verbose=1, energy_exit_rounds=energy_exit_rounds,
                        energy_exit_tol=energy_exit_tol, temp_tol=temp_tol)
        expected_end = "Finished - Energy within tolerance for {} rounds "\
                       "(tol = {}).\n".format(energy_exit_rounds,
                                              energy_exit_tol)

        assert caplog.text.endswith(expected_end)


def test_anneal_with_best_state_specified():
    annealer = AnnealerMinusOne()
    best_state, best_energy = annealer.anneal(best_state=-200, debug=True)

    assert best_state == -200
    assert best_energy == -200
