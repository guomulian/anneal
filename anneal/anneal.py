import abc
import copy
import logging
import math
import pickle
import random
from collections import deque
from anneal import helpers


class BaseAnnealer(metaclass=abc.ABCMeta):
    """Template method pattern for perfoming simulated annealing."""

    def __init__(self, initial_state, max_steps=None, *args, **kwargs):
        """
        Parameters
        ----------
        initial_state : <>
            The state the algorithm will begin with.

        max_steps : int, optional
            Default is 1000.

            The maximum steps the algorithm is allowed to take. May be changed
            later.
        """
        self._initial_state = self.copy_method(initial_state)

        if max_steps is not None:
            self.max_steps = max_steps
        else:
            self.max_steps = self.defaults["max_steps"]

        self._reset(*args, **kwargs)

    def __str__(self):
        return "{}(step={}/{}: energy={})".format(type(self).__name__,
                                                  self.step,
                                                  self.max_steps,
                                                  self.energy)

    @property
    def defaults(self):
        """Default values for various parameters."""
        return dict(max_steps=1000,
                    energy_break_rounds=-1,
                    energy_break_tol=-1,
                    temp_tol=-1,
                    verbose=0,
                    debug=False,
                    pickle=False,
                    pickle_file=None)

    @property
    def step(self):
        """The current step the anneal() method is on."""
        return self._step

    @property
    def max_steps(self):
        """The maximum number of steps anneal() is allowed to take."""
        return self._max_steps

    @property
    def energy(self):
        """The energy of the current state."""
        return self.energy_method(self.state)

    @property
    def state(self):
        """The current state."""
        return self._state

    @property
    def initial_state(self):
        """The state the annealer object was initialized with."""
        return self._initial_state

    @property
    def best_state(self):
        """The state with the lowest known energy."""
        return self._best_state

    @property
    def best_energy(self):
        """The energy of best_state."""
        return self.energy_method(self.best_state)

    @property
    def energy_queue(self):
        """Queue to keep track of energies for an energy break.

        (Keeping this readable for debugging purposes.)
        """
        return self.__energy_queue

    @max_steps.setter
    def max_steps(self, value):
        if isinstance(value, int) and value > 0:
            self._max_steps = value
        else:
            raise ValueError("Max steps must be a positive integer.")

    @property
    def last_exit(self):
        """Prints the type of exit of the last run of anneal()."""
        try:
            return self.__last_exit
        except AttributeError:
            return None

    def copy_method(self, state):
        """Method for copying states; this may be overwritten.

        Default is copy.deepcopy(). This may not be the most efficient option
        for certain problems.
        """
        return copy.deepcopy(state)

    def _reset(self, *args, **kwargs):
        """Resets the state of the annealer with the given options."""

        self._step = 0
        self._state = self.copy_method(self.initial_state)

        self.max_steps = kwargs.get("max_steps", self.max_steps)

        self.__verbose = kwargs.get("verbose", 0)
        assert self.__verbose in [0, 1, 2], "verbose must be in [0, 1, 2]."

        self.__debug = kwargs.get(
                "debug", self.defaults["debug"])
        self.__pickle = kwargs.get(
                "pickle", self.defaults["pickle"])
        self.__pickle_file = kwargs.get(
                "pickle_file", self.defaults["pickle_file"])
        self.__energy_break_rounds = kwargs.get(
                "energy_break_rounds", self.defaults["energy_break_rounds"])
        self.__energy_break_tol = kwargs.get(
                "energy_break_tol", self.defaults["energy_break_tol"])
        self.__temp_tol = kwargs.get(
                "temp_tol", self.defaults["temp_tol"])
        self.__last_pickle = None

        best_state = kwargs.get("best_state", None)

        if best_state:
            self._best_state = self.copy_method(best_state)
        else:
            self._best_state = self.copy_method(self._state)

        if self.__energy_break_rounds > 1 and self.__energy_break_tol > 0:
            self.__energy_queue = deque([self.energy],
                                        maxlen=self.__energy_break_rounds)
        else:
            self.__energy_queue = None

    @abc.abstractmethod
    def neighbor(self, state):  # pragma: no cover
        """Returns a random neighbor of a given state.

        Parameters
        ----------
        state : <>
            The (current) state to find a neighbor of.
        """
        pass

    @abc.abstractmethod
    def energy_method(self, state):  # pragma: no cover
        """Returns the energy of a given state.

        The annealing procedure will try to bring the system to a state that
        minimizes this value.

        Parameters
        ----------
        state : <>
            The state to find the energy of.
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
        return 1 - step/self.max_steps

    def _acceptance_probability(self, state, temp):
        """Probability of moving from the current state to the new state.

        As temp goes to zero, this should go to zero for E_new > E_old.
        """
        return math.exp(-(self.energy_method(state) -
                          self.energy_method(self.state))
                        / temp)

    def _accept_state(self, state):
        """Returns True if the given state is accepted."""
        try:
            temp = self.temperature(self.step)
            p = self._acceptance_probability(state, temp)

            if p >= 1 or p >= random.random():
                return True
            else:
                return False

        except OverflowError:
            return True

        except ZeroDivisionError:
            return self.energy_method(state) < self.energy_method(self.state)

    def format_output(self, output):
        """Function for processing the output of anneal. May be overwritten if
        desired.

        This may be used, for example, if one wants the anneal method to output
        only the best state (or only the best energy).

        Another possible use is if one wants to use BaseAnnealer to search
        for a global maximum rather than a minimum. (By default, energy will be
        minimized; format_output can be used to take the negative of the final
        energy.)

        Parameters
        ---------
        output : (<>, float)
            At the moment, output is of the form (best_state, best_energy).
        """
        return output

    def debug_method(self):  # pragma: no cover
        """Defines behavior when anneal is run with debug=True. Runs at the
        start of every step.

        Note that debug_method will not be run for every step unless verbose
        is set to the max (2).
        """
        print(self)

    def _handle_debug(self):
        """Takes care of debugging with different verbosity options."""
        if self.__debug:
            # {verbose: # of times to call debug_method()}
            n_intervals = {0: 10,
                           1: 100,
                           2: self.max_steps}
            debug_interval = self.max_steps // n_intervals[self.__verbose]

            if self.step % debug_interval == 0:
                self.debug_method()

    def pickle_state(self, pickle_file=None, append=False):
        """Pickles the current state to a file.

        Parameters
        ----------
        pickle_file : str, optional
            Default is None.

            File to write to. If this is not specified, a .pickle file will
            be created with the filename given by the class name and a
            timestamp.

        append : bool, optional
            Default is False.

            If True, opens pickle_file in append mode.
        """
        if pickle_file is None:
            pickle_file = helpers.generate_filename(self, ".pickle")

        if append:
            mode = 'ab'
        else:
            mode = 'wb'

        self.__last_pickle = pickle_file

        with open(pickle_file, mode) as file:
            pickle.dump(self.state, file)

    def unpickle_states(self, filename=None):
        """Returns a list of states found in a given pickle file. If no
        filename is provided, unpickle_states() will try to unpickle the
        latest file pickled by anneal.
        """
        if not filename:
            if self.__last_pickle is not None:
                filename = self.__last_pickle
            else:
                raise FileNotFoundError(
                    "Could not find anything to unpickle. Try first running "
                    "anneal() with pickle=True, or run unpickle_states() with "
                    "a filename specified.")

        states = []

        with open(filename, 'rb') as file:
            while True:
                try:
                    states.append(pickle.load(file))
                except EOFError:
                    break

        return states

    def _energy_break(self):
        """Tests whether conditions for an energy break are met."""
        if self.__energy_queue is None:
            return False
        elif len(self.__energy_queue) == self.__energy_break_rounds:
            return True
        else:
            return False

    def _temp_break(self):
        """Tests whether conditions for a temperature break are met."""
        return abs(self.temperature(self.step - 1) -
                   self.temperature(self.step)) < self.__temp_tol

    def _handle_pickle(self, append=False):
        if self.__pickle:
            self.pickle_state(self.__pickle_file, append)

    def _handle_energy_queue(self, energy):
        """Tests if given energy should be added to the queue (in other words,
        is within the given tolerance. If it's not, resets the queue.
        ."""
        if self.__energy_queue is not None:
            if abs(self.__energy_queue[-1] - energy) < self.__energy_break_tol:
                self.__energy_queue.append(energy)
            else:
                self.__energy_queue.clear()
                self.__energy_queue.append(energy)

    def _handle_exit(self, exit):
        messages = {
                "energy": "Energy within tolerance for {} rounds (tol = {})."
                          .format(self.__energy_break_rounds,
                                  self.__energy_break_tol),
                "temp": "Reached temperature tolerance (tol = {})."
                        .format(self.__temp_tol),
                "max_steps": "Reached max steps (max_steps = {})."
                             .format(self.max_steps)
                }

        self.__last_exit = messages[exit]

        if self.__verbose != 0:
            logging.info("Finished - " + messages[exit])

    def anneal(self, *args, **kwargs):
        """Tries to find the state which minimizes the energy given by
        energy_method via simulated annealing.

        Parameters
        ----------
        max_steps : int, optional
            For if you want to run anneal with a different max_steps than
            originally specified.

        best_state : <>, optional
            Default is None.

            For if you want the algorithm to start with a specific "best
            state".

        verbose : int, optional
            Default is 0.

            Must be one of 0, 1, 2.
                0 (default) will print no output (unless debug=True, where
                            minimal output will be printed)
                1           will print less output
                2           will print all output

        debug : bool, optional
            Default is False.

            Execute debug_method at certain intervals. By default, setting this
            to True will display the step number, temperature, best state, and
            best energy at each step.

        pickle : bool, optional
            Default is False.

            Pickle the intermediate steps and write to a file, optionally given
            by the pickle_file keyword argument.

        pickle_file : str, optional
            Default is None.

            File to pickle to. If not specified and pickle is set to True,
            a timestamp will be used as the filename with the class name
            as a prefix.

        energy_break_rounds : int, optional
            Default is -1.

            Number of rounds of "slowly changing energy" to allow before
            exiting. Must be at least 2. If left at the default value, the
            algorithm won't terminate in this manner.

            In other words, if the change in energy remains within some
            tolerance (specified by energy_break_tol) for energy_break_rounds
            rounds, the algorithm will abort early and return the best state
            and energy found up to that point.

        energy_break_tol : float, optional
            Default is -1.

            Tolerance for energy_break_rounds.

        temp_tol : float, optional
            Default is -1.

            If the change in temperature becomes smaller than this, the program
            will abort.

        Returns
        -------
        (<>, float)
            This is (best_state, best_energy).
        """
        self._reset(*args, **kwargs)

        # pickle first state
        self._handle_pickle(append=False)

        for _ in range(self.max_steps):
            self._handle_debug()

            neighbor = self.neighbor(self.copy_method(self.state))

            if self._accept_state(neighbor):
                new_energy = self.energy_method(neighbor)

                if new_energy < self.best_energy:
                    self._best_state = self.copy_method(neighbor)

                self._state = self.copy_method(neighbor)

                self._handle_pickle(append=True)
                self._handle_energy_queue(new_energy)

                if self._energy_break():
                    self._handle_exit("energy")
                    break

            if self._temp_break():
                self._handle_exit("temp")
                break

            self._step += 1

        else:
            self._handle_exit("max_steps")

        return self.format_output((self.best_state, self.best_energy))

    def run(self, n_runs, *args, **kwargs):
        """Run anneal method multiple times with a given set of parameters.
        (*args and **kwargs will be passed to anneal.)

        Parameters
        ----------
        n_runs : int
            Number of times to run anneal.
        """
        states = []
        energies = []

        for _ in range(n_runs):
            s, e = self.anneal(*args, **kwargs)
            states.append(s)
            energies.append(e)

        return states, energies
