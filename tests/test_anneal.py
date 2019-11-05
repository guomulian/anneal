from anneal import anneal
from collections import deque
import logging
import pytest
import random
import sys


def test_initialized_without_abstract_methods():
    with pytest.raises(TypeError):
        anneal.BaseAnnealer(None, 100)


def test_initialized_with_bad_max_steps(trivial_annealer):
    with pytest.raises(ValueError):
        trivial_annealer.__class__(initial_state=0, max_steps=-1)


@pytest.mark.parametrize("bad_value", [3.2, -1, "foo"])
def test_max_step_validation(trivial_annealer, bad_value):
    with pytest.raises(ValueError):
        trivial_annealer.max_steps = bad_value


def test_anneal_with_max_step_change(trivial_annealer):
    trivial_annealer.anneal(max_steps=50)
    assert trivial_annealer.step == 50


def test_anneal_unpickle_states_without_file(trivial_annealer):
    trivial_annealer.anneal()

    with pytest.raises(FileNotFoundError):
        trivial_annealer.unpickle_states()


def test_run_method(trivial_annealer):
    _, energies = trivial_annealer.run(50)

    assert all(e == 0 for e in energies)
    assert len(energies) == 50


def test_run_with_parameter_change(trivial_annealer):
    energies = trivial_annealer.run(50, max_steps=10)
    assert trivial_annealer.max_steps == 10


@pytest.mark.parametrize("bad_value", [3.2, -1, 5, "foo"])
def test_bad_verbose_value(trivial_annealer, bad_value):
    with pytest.raises(Exception):
        trivial_annealer.anneal(verbose=bad_value)


def test_bad_energy_break_rounds_value(trivial_annealer):
    tol = 0.1

    with pytest.raises(TypeError):
        trivial_annealer.anneal(energy_break_rounds=5.2, energy_break_tol=tol)

    trivial_annealer.anneal(energy_break_rounds=0.2, energy_break_tol=tol)
    assert trivial_annealer.energy_queue is None


def test_energy_queue_clears(plus_one_annealer):
    plus_one_annealer.anneal(energy_break_rounds=2, energy_break_tol=0.2)
    assert len(plus_one_annealer.energy_queue) == 1


def test_handle_debug(capsys, custom_debug_annealer):
    custom_string = "This is a custom debug method.\n"

    # should print custom_string 10 times
    custom_debug_annealer.anneal(debug=True)
    captured = capsys.readouterr()
    assert captured.out == custom_string*10

    # should print custom_string 100 times
    custom_debug_annealer.anneal(verbose=1, debug=True)
    captured = capsys.readouterr()
    assert captured.out == custom_string*100

    # should print custom_string 1000 times
    custom_debug_annealer.anneal(verbose=2, debug=True)
    captured = capsys.readouterr()
    assert captured.out == custom_string*1000


def test_verbose_max_steps(caplog, trivial_annealer):
    with caplog.at_level(logging.INFO):
        trivial_annealer.anneal(verbose=1)
        expected_end = "Finished - Reached max steps (max_steps = {}).\n"\
                       .format(trivial_annealer.max_steps)
        assert caplog.text.endswith(expected_end)

        trivial_annealer.anneal(verbose=2)
        expected_end = "Finished - Reached max steps (max_steps = {}).\n"\
                       .format(trivial_annealer.max_steps)
        assert caplog.text.endswith(expected_end)


def test_verbose_temp_break(caplog, small_temp_annealer):
    temp_tol = 0.1

    with caplog.at_level(logging.INFO):
        small_temp_annealer.anneal(verbose=1, temp_tol=temp_tol)
        expected_end = "Finished - Reached temperature tolerance (tol = {})."\
                       "\n".format(temp_tol)
        assert caplog.text.endswith(expected_end)

        small_temp_annealer.anneal(verbose=2, temp_tol=temp_tol)
        expected_end = "Finished - Reached temperature tolerance (tol = {})."\
                       "\n".format(temp_tol)
        assert caplog.text.endswith(expected_end)


def test_verbose_energy_break(caplog, trivial_annealer):
    temp_tol = -1
    energy_break_rounds = 3
    energy_break_tol = 0.1

    with caplog.at_level(logging.INFO):
        trivial_annealer.anneal(verbose=1,
                                energy_break_rounds=energy_break_rounds,
                                energy_break_tol=energy_break_tol,
                                temp_tol=temp_tol)
        expected_end = "Finished - Energy within tolerance for {} rounds "\
                       "(tol = {}).\n".format(energy_break_rounds,
                                              energy_break_tol)

        assert caplog.text.endswith(expected_end)


def test_anneal_with_best_state_specified(plus_one_annealer):
    best_state, best_energy = plus_one_annealer.anneal(best_state=-200)

    assert best_state == -200
    assert best_energy == -200
