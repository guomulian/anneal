import abc
import copy
import math
import random


class SimulatedAnnealer(metaclass=abc.ABCMeta):
    """Template method pattern for perfoming simulated annealing.

    ...

    Attributes
    ----------
    step : int
        The current step the annealer is on.
    max_steps : int
        The maximum number of steps the annealer is permitted to take.
    energy : float
        The energy of the current state, as defined by the _energy() method.
    initial_state : <>
        The initial state passed in. This is kept simply for the _reset()
        method.
    state : <>
        The current state.
    best_energy : float
        The current best energy.
    best_state : <>
        The current best state. The final value of this will be the solution.
    """

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
                    state={},
                    energy={},
                    best_state={},
                    best_energy={}
                )     """
        return pattern.format(type(self).__name__, self.step, self.max_steps,
                              self.temp(self.step), self.state, self.energy,
                              self.best_state, self.best_energy)

    def _reset(self, best_state=None):
        """Resets the state of the annealer, with the possibility of
           pre-specifying a "best state".

        """
        self.step = 0
        self.state = copy.deepcopy(self.initial_state)
        self.energy = self._energy(self.state)

        # this right now is not really helpful for anything
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

        The annealing procedure will try to bring the system to a state that
        minimizes this value.
        """
        pass

    def temp(self, step):
        """Defines the temperature/annealing schedule for the problem.

        This method may be overwritten in a subclass if desired.

        Parameters
        ----------
        step : int
            The number of steps elapsed.
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
            p = self._acceptance_probability(new_state, self.temp(self.step))

            if p >= 1 or p >= random.random():
                return True
            else:
                return False

        except OverflowError:
            return True

    def formatter(self, output):
        """Function for processing the output of anneal. May be overwritten if
        desired.

        This may be used, for example, if one wants the anneal method to output
        only the best state (or only the best energy).

        Another possible use is if one wants to use SimulatedAnnealer to search
        for a global maximum rather than a minimum. (By default, energy will be
        minimized; the formatter can be used to take the negative of the final
        energy.)

        Paramters
        ---------
        output : (<>, float)
            First argument is the best state found by the algorithm, the second
            is the best energy.
        """

        return output

    def anneal(self, temp_tol=0.0001, best_state=None, verbose=0, debug=False):
        """Tries to find the state which minimizes the energy given by the
        _energy method via simulated annealing.

        Parameters
        ----------
        temp_tol : float, optional
            The minimum allowed temperature before the program aborts. Default
            is 0.0001.

        best_state : <>, optional
            For if you want the algorithm to start with a different state than
            originally specified.

        verbose : int, optional
            Must be one of 0, 1, 2.
                0 (default) will print no output (except when debug=True,
                            however, it will minimize the debug output)
                1           will print less output
                2           will print all output

        debug : bool, optional
            At the moment, setting this to True will print the current step,
            temperature, best state, and best energy
            as the process goes on.

        Returns
        -------
        (<>, float)
            This is (best_state, best_energy).
        """

        if verbose not in [0, 1, 2]:
            raise ValueError("verbose must be one of 0 (none), 1 (less), or 2\
                (all).")

        self._reset(best_state=best_state)

        for _ in range(self.max_steps):
            self.step += 1

            if debug:
                if verbose == 0:
                    interval = self.max_steps // 10
                elif verbose == 1:
                    interval = self.max_steps // 100
                else:
                    interval = 1

                if self.step % interval == 0:
                    print(self)

            neighbor = self._neighbor()

            if self._accept_state(neighbor):
                self.state = neighbor

            if self._energy(self.state) < self.energy:
                self.energy = self._energy(self.state)
                self.best_energy = self.energy

            if self.temp(self.step) < temp_tol:
                if verbose != 0:
                    print("Finished - Reached temperature 0")
                break
        else:
            if verbose != 0:
                print("Finished - Reached max steps")

        return self.formatter((self.state, self.energy))
