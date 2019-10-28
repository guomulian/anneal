import abc
import copy
import logging
import math
import pickle
import random
from collections import deque
from anneal import helpers


class BaseAnnealer(metaclass=abc.ABCMeta):
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
        pattern = "{}(\n"\
                  "\tstep={},\n"\
                  "\tmax_steps={},\n"\
                  "\ttemp={},\n"\
                  "\tstate={},\n"\
                  "\tenergy={},\n"\
                  "\tbest_state={},\n"\
                  "\tbest_energy={}\n)"
        return pattern.format(type(self).__name__,
                              self.step,
                              self.max_steps,
                              self.temperature(self.step),
                              self.state,
                              self.energy,
                              self.best_state,
                              self.best_energy)

    def _reset(self, best_state=None):
        """Resets the state of the annealer, with the possibility of
           pre-specifying a "best state".

        """
        self.step = 0
        self.state = copy.deepcopy(self.initial_state)
        self.energy = self._energy(self.state)

        if best_state:
            self.best_state = copy.deepcopy(best_state)
        else:
            self.best_state = copy.deepcopy(self.state)

        self.best_energy = self._energy(self.best_state)

    @abc.abstractmethod
    def _neighbor(self, state):  # pragma: no cover
        """Returns a random neighbor of a given state."""
        pass

    @abc.abstractmethod
    def _energy(self, state):  # pragma: no cover
        """Returns the energy of a given state.

        The annealing procedure will try to bring the system to a state that
        minimizes this value.
        """
        pass

    def temperature(self, step):
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
            temp = self.temperature(self.step)
            p = self._acceptance_probability(new_state, temp)

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

        Another possible use is if one wants to use BaseAnnealer to search
        for a global maximum rather than a minimum. (By default, energy will be
        minimized; the formatter can be used to take the negative of the final
        energy.)

        Parameters
        ---------
        output : (<>, float)
            First argument is the best state found by the algorithm, the second
            is the best energy.
        """

        return output

    def debug_method(self, *args, **kwargs):  # pragma: no cover
        """Defines behavior when anneal is run with debug=True. Default is to
        print __str__(self). Note that debug_method will not be run for every
        step unless verbose is set to 2.

        May be overwritten to display a visualization of the current state,
        for example.
        """
        print(self)

    def _debug_handler(self, verbose, *args, **kwargs):
        """Takes care of debugging with different verbosity options."""

        # maps verbose to number of times to execute debug_method
        n_intervals = {0: 10, 1: 100, 2: self.max_steps}
        debug_interval = self.max_steps // n_intervals[verbose]

        if self.step % debug_interval == 0:
            self.debug_method(*args, **kwargs)

    def _pickle_state(self, *args, **kwargs):
        """Pickles the current state to a file.

        Parameters
        ----------
        Keyword Args:
            pickle_file : str, optional
                File to write to. If this is not specified, a .pickle file will
                be created with the filename given by the class name and a
                timestamp.
        """
        try:
            filename = kwargs["pickle_file"]
        except KeyError:
            filename = helpers.generate_filename(self, ".pickle")

        if self.step == 0:
            mode = 'wb'
        else:
            mode = 'ab'

        self.__last_pickle = filename

        with open(filename, mode) as file:
            pickle.dump(self.state, file)

    def unpickle_states(self, filename=None):
        """Returns a list of states found in a given pickle file. If no
        filename is provided, unpickle_states() will try to unpickle the
        latest file pickled by anneal.
        """
        if not filename:
            try:
                filename = self.__last_pickle
            except AttributeError as error:
                raise FileNotFoundError(
                    "Could not find anything to unpickle. Try first running "
                    "anneal() with pickle=True, or run unpickle_states() with "
                    "a filename specified.") from None

        states = []

        with open(filename, 'rb') as file:
            while True:
                try:
                    states.append(pickle.load(file))
                except EOFError:
                    break

        return states

    def _energy_exit_handler(self, energy, tol):
        try:
            if abs(self.__energy_queue[-1] - energy) < tol:
                self.__energy_queue.append(energy)
            else:
                self.__energy_queue.clear()
                self.__energy_queue.append(energy)

        except AttributeError:
            pass

    def _energy_break(self, energy_exit_rounds):
        try:
            self.__energy_queue
        except AttributeError:
            return False

        if len(self.__energy_queue) == energy_exit_rounds:
            return True
        else:
            return False

    def _temp_break(self, temp_tol):
        return self.temperature(self.step) < temp_tol

    def anneal(self,
               best_state=None,
               verbose=0,
               debug=False,
               pickle=False,
               energy_exit_rounds=-1,
               energy_exit_tol=-1,
               temp_tol=0.0001,
               *args, **kwargs):
        """Tries to find the state which minimizes the energy given by the
        _energy method via simulated annealing.

        Parameters
        ----------
        best_state : <>, optional
            For if you want the algorithm to start with a specific "best
            state".

        verbose : int, optional
            Must be one of 0, 1, 2.
                0 (default) will print no output (except when debug=True,
                            where it will print minimal output)
                1           will print less output
                2           will print all output

        debug : bool, optional
            Default is False. Execute debug_method at certain intervals. By
            default, setting this to True will display the step number,
            temperature, best state, and best energy at each step.

        pickle : bool, optional
            Default is False. Pickle the intermediate steps and write to a
            file, optionally given by the pickle_filen keyword argument.

        pickle_file : str, optional
            File to pickle to. If not specified and pickle is set to True,
            a timestamp will be used as the filename with the class name
            as a prefix.

        energy_exit_rounds : int, optional
            Number of rounds of "slowly changing energy" to allow before
            exiting. Must be at least 2. Default value is -1 (algorithm will
            not terminate in this manner).

            If the change in energy remains within some tolerance (specified
            with energy_exit_tol) for energy_exit_rounds rounds, the algorithm
            will abort early and return the best state and energy found up to
            that point.

        energy_exit_tol : float, optional
            Tolerance for energy_exit_rounds. Default value is -1.

        temp_tol : float, optional
            The minimum allowed temperature before the program aborts. Default
            is 0.0001.

        Returns
        -------
        (<>, float)
            This is (best_state, best_energy).
        """

        if verbose not in [0, 1, 2]:
            raise ValueError("verbose must be one of 0 (none), 1 (less), or 2\
                (all).")

        if not isinstance(energy_exit_rounds, int):
            raise TypeError("energy_exit_rounds must be an int.")

        if energy_exit_rounds > 1 and energy_exit_tol > 0:
            self.__energy_queue = deque([self.energy],
                                        maxlen=energy_exit_rounds)

        self._reset(best_state=best_state)

        for _ in range(self.max_steps):
            if debug:
                self._debug_handler(verbose, *args, **kwargs)

            if pickle:
                self._pickle_state(*args, **kwargs)

            self.step += 1

            neighbor = self._neighbor(copy.deepcopy(self.state))

            if self._accept_state(neighbor):
                new_energy = self._energy(neighbor)

                if new_energy < self.energy:
                    self.energy = new_energy
                    self.best_state = copy.deepcopy(neighbor)
                    self.best_energy = new_energy

                self.state = copy.deepcopy(neighbor)

                self._energy_exit_handler(new_energy, energy_exit_tol)

                # Test for energy break
                if self._energy_break(energy_exit_rounds):
                    if verbose != 0:
                        logging.info("Finished - Energy within tolerance for "
                                     "{} rounds (tol = {})."
                                     .format(energy_exit_rounds,
                                             energy_exit_tol))
                    break

            # Test for temperature break
            if self._temp_break(temp_tol):
                if verbose != 0:
                    logging.info("Finished - Reached temperature tolerance"
                                 "(tol = {}).".format(temp_tol))
                break
        else:
            if verbose != 0:
                logging.info("Finished - Reached max steps"
                             "(max_steps = {}).".format(self.max_steps))

        # if there was a break, pickle last state
        if pickle:
            self._pickle_state(*args, **kwargs)

        return self.formatter((self.best_state, self.best_energy))
