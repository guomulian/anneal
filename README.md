# anneal
A template method pattern for implementing simulated annealing in Python, along with implementations of several example problems.

- [Overview](#overview)
    - [Algorithm](#algorithm)
    - [Examples](#examples)
- [Installing](#installing)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview
> Simulated annealing is a probabilistic technique for approximating the global optimum of a given function.

### Algorithm

One problem of [hill climbing optimization techniques](https://en.wikipedia.org/wiki/Hill_climbing) is that they run the risk of getting "stuck" in a local optimum.

For example, if we apply [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) to the following function, we might find ourself in the shorter "hill" (shown on the left in the image).

[![A surface with two local maxima.][non-convex-example-image]](https://commons.wikimedia.org/wiki/File:Local_maximum.png)

**[Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing)** tries to overcome this problem by initially allowing transitions to "worse" solutions than the current one, then slowly reducing the acceptability of such "bad" transitions. This gives the algorithm the opportunity to "escape" any local optima it might find itself in initially.

The **acceptance probability** of these "bad" solutions is controlled by a parameter (often referred to as the **temperature**, due to analogy with the annealing of metallurgy), which is eventually reduced to zero according to some **schedule**.

For example, the schedule shown in the following image has the temperature decreasing at a rate of `0.1` per step.

[![A simulated annealing search for a maximum on a noisy example function][wikipedia-image]](https://commons.wikimedia.org/wiki/File:Hill_Climbing_with_Simulated_Annealing.gif)

The [Wikipedia article for simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) provides a more detailed overview of the algorithm.


### Examples
Included are a number of well-known example use cases.

- Real-valued function of two real variables
- Sudoku
- Traveling salesperson
- Automatic map label placement

## Installing


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

#### Optional Methods
##### `temp(self, step)`
The `temp` method, which provides the annealing schedule, may also be overwritten if desired. The default is for the temperature to decrease at a constant rate from `1` towards `0`, where the rate is `-1/max_steps`. 

###### Example
```python
def temp(self, step):
    # exponential scheme
    return 0.8**step
```
##### `formatter(self, output)`
This method is given as an option for post-processing the results of the `anneal()` method.

`output` is of the form `(best_state, best_energy)`.

###### Example
```python

def formatter(self, output):
    # only return the final state
    return output[0]
```

## Contributing
To clone this repository:
```bash
$ git clone https://github.com/guomulian/anneal
```

[wikipedia-image]: https://upload.wikimedia.org/wikipedia/commons/d/d5/Hill_Climbing_with_Simulated_Annealing.gif
[non-convex-example-image]: https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Local_maximum.png/260px-Local_maximum.png