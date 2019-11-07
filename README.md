# anneal
A template class for implementing the [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) algorithm in Python, along with [implementations of several well-known problems](#examples).

- [Overview](#overview)
    - [Algorithm](#algorithm)
    - [Examples](#examples)
- [Installing](#installing)
- [Usage](#usage)

## Overview

### Algorithm
> Simulated annealing is a probabilistic technique for approximating the global optimum of a given function.

One problem of [hill climbing optimization techniques](https://en.wikipedia.org/wiki/Hill_climbing) is that they run the risk of getting "stuck" in a local optimum.

For example, if we apply [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) to maximize the following function, we might find ourself in the shorter "hill" (shown on the left in the image).

[![A surface with two local maxima.][non-convex-example-image]](https://commons.wikimedia.org/wiki/File:Local_maximum.png)

**[Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing)** tries to overcome this problem by initially allowing transitions to "worse" solutions than the current one, then slowly reducing the acceptability of such "bad" transitions. This gives the algorithm the opportunity to "escape" any local optima it might find itself in initially.

The **acceptance probability** of these "bad" solutions is controlled by a parameter (often referred to as the **temperature**, due to analogy with the annealing of metallurgy), which is eventually reduced to zero according to some **schedule**.

For example, the schedule shown in the following image (from Wikipedia) has the temperature decreasing at a rate of `0.1` per step.

[![A simulated annealing search for a maximum on a noisy example function][wikipedia-image]](https://commons.wikimedia.org/wiki/File:Hill_Climbing_with_Simulated_Annealing.gif)

The [Wikipedia article for simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) provides a more detailed overview of the algorithm.


### Examples
Included are implementations of a few example use cases. (Unchecked items are not yet complete.)

- [x] [Real-valued function of several real variables](./examples/rvf/)
- [x] [Sudoku](./examples/sudoku/)
- [x] [Traveling salesperson](./examples/tsp/)
- [ ] Automatic map label placement

## Installing
To run the examples:
```bash
$ git clone https://github.com/guomulian/anneal
$ cd anneal

# create/activate virtual environment
$ pip install -r requirements.txt
$ pip install [-e] .
$ python examples/rvf2/rvf2_example.py
```

## Usage

To use, simply import the `anneal` module and subclass `BaseAnnealer`, making sure to define `energy` and `neighbor` methods.

```python
import anneal


class MySolver(anneal.BaseAnnealer):
    """Class docstring for your solver."""

    def __init__(self, initial_state, *args, **kwargs):
        # Initialization code for your problem.
        # All you need to make sure is to pass an initial_state and,
        # optionally, the max_steps parameter to the statement below.

        super().__init__(initial_state, max_steps=1000)

    def energy_method(self, state):
        """Returns the energy of a given state."""
        pass

    def neighbor(self, state):
        """Returns a random neighbor of the given state."""
        pass
```

Calling the `anneal()` method on your `MySolver` instance will return the optimal state and its associated energy.

```python
solver = MySolver(initial_state)

best_state, best_energy = solver.anneal(max_steps=2000)
```

That's all there is to it!

#### Optional Methods
##### `temperature(self, step)`
The `temperature` method, which provides the annealing schedule, may also be overwritten if desired. The default is for the temperature to decrease at a constant rate from `1` towards `0`, where the rate is `-1/max_steps`.

###### Example
```python
def temperature(self, step):
    # exponential scheme
    return 0.8**step
```
`step` runs from `0` to `max_steps - 1`.

##### `format_output(self, output)`
This method is given as an option for post-processing the results of the `anneal()` method.

`output` is of the form `(best_state, best_energy)`.

###### Example
```python

def format_output(self, output):
    # only return the final state
    return output[0]
```
##### `debug_method(self)`
This is run at the start of every step when the `anneal` method is set with `debug=True`.

###### Example
```python

def debug_method(self):
    # only print the current step
    print("Current Step: {}".format(self.step))
```

##### `copy_method(self, state)`
Default is `copy.deepcopy(state)`. If you know your states won't be the sort of objects that require deep copying, this can (and should) be overwritten to something with better performance.

###### Example
```python

def copy_method(self, state):
    return state
```

[wikipedia-image]: https://upload.wikimedia.org/wikipedia/commons/d/d5/Hill_Climbing_with_Simulated_Annealing.gif
[non-convex-example-image]: https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Local_maximum.png/260px-Local_maximum.png
