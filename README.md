# anneal
A template method pattern for implementing simulated annealing. Written in Python.

- [Overview](#overview)
    - [Algorithm](#algorithm)
    - [Examples](#examples)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview

### Algorithm
The [Wikipedia article](https://en.wikipedia.org/wiki/Simulated_annealing) provides a good description of the procedure.
> Simulated annealing is a probabalistic technique for approximating the global optimum of a given function.

### Examples
- Traveling Salesman
- Solving Sudoku
- Map Label Placement

## Getting Started

## Usage

To use, simply import the `annealer` module and subclass `SimulatedAnnealer`, making sure to define `_energy` and `_neighbor` methods.

```python
import annealer


class MySolver(annealer.SimulatedAnnealer):
    """Class docstring for your solver."""

    def __init__(self, *args, **kwargs):
        # Initialization code for your problem.
        # All you need to make sure is to pass an initial_state and
        # the max_steps parameter to the statement below.

        super().__init__(initial_state, max_steps)

    def _energy(self, state):
        """Returns the energy of a given state."""
        pass

    def _neighbor(self):
        """Returns a random neighbor of the current state."""
        pass
```

## Contributing

```bash
$ git clone https://github.com/guomulian/anneal
```