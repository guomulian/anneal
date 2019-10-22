import abc
import copy
import math
import random


class SimulatedAnnealer(metaclass=abc.ABCMeta):
    """Template method pattern for perfoming simulated annealing."""

    def __init__(self, initial_state, max_steps):
        self.step = 0

        self.energy = self._energy(initial_state)
        self.state = copy.deepcopy(initial_state)
        self.initial_state = copy.deepcopy(initial_state)

        self.best_state = copy.deepcopy(initial_state)
        self.best_energy = self._energy(self.best_state)

        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        else:
            raise ValueError("Max steps must be a positive integer")

    def __str__(self):
        pattern = """
{}(
    step={}/{},
    temp={},
    best_state={},
    best_energy={}
)     """
        return pattern.format(type(self).__name__, self.step, self.max_steps, self._temp(self.step), self.best_state, self.best_energy)

    def _reset(self, best_state=None):
        """Resets the state of the annealer, with the possibility of
           pre-specifying a best_state.

        """
        self.step = 0
        self.state = copy.deepcopy(self.initial_state)
        self.energy = self._energy(self.state)

        if best_state:
            self.best_state = best_state
        else:
            self.best_state = copy.deepcopy(self.state)

        self.best_energy = self._energy(self.best_state)

    @abc.abstractmethod
    def _neighbor(self):
        """Returns a random neighbor of the current state."""
        pass

    @abc.abstractmethod
    def _energy(self, state):
        """Returns the energy of a given state.

        The goal is to bring the system to a state that minimizes this value.
        """
        pass

    def _temp(self, step):
        """The annealing schedule.

        This method may be overwritten in a subclass if desired.
        """
        return 1.00001 - step/self.max_steps

    def _acceptance_probability(self, new_state, temp):
        """Probability of moving from the current state to the new state.

        As temp goes to zero, this should go to zero if E_new > E_old.
        """
        return math.exp(-(self._energy(new_state) - self._energy(self.state))
                        / temp)

    def _accept_state(self, new_state):
        """Returns True if the new_state is accepted."""
        try:
            p = self._acceptance_probability(new_state, self._temp(self.step))

            if p >= 1 or p >= random.random():
                return True
            else:
                return False

        except OverflowError:
            return True

    def anneal(self, verbose=False, debug=False):
        """The annealing procedure."""

        self._reset()

        for _ in range(self.max_steps):
            self.step += 1

            if debug:
                print(self)

            neighbor = self._neighbor()

            if self._accept_state(neighbor):
                self.state = neighbor

            if self._energy(self.state) < self.energy:
                self.energy = self._energy(self.state)
                self.best_energy = self.energy

            if self._temp(self.step) < 0.0001:
                if verbose:
                    print("Finished - Reached temperature 0")

                return self.state, self.energy

        if verbose:
            print("Finished - Reached max steps")

        return self.state, self.energy
