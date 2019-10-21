from anneal import SimulatedAnnealer
import random

class Rvf2Optimize(SimulatedAnnealer):
    def __init__(self, fun, initial_state, max_steps, bounds=[[-1,1],[-1,1]], objective='min'):
        self.fun = fun
        self.objective = objective
        
        if bounds[0][0] < bounds[0][1] and bounds[1][0] < bounds[1][1]:
            self.bounds = bounds
        else:
            raise ValueError('Invalid bounds.')

        # test if state in bounds

        super().__init__(initial_state, max_steps)

    def _neighbor(self):
        x_bounds, y_bounds = self.bounds
        
        dx = 0.1*abs(x_bounds[1]-x_bounds[0])
        dy = 0.1*abs(y_bounds[1]-y_bounds[0])

        dx *= random.uniform(-1,1)
        dy *= random.uniform(-1,1)

        return tuple(map(sum, zip(self.state, (dx,dy))))
        

    def _energy(self, state):
        if self.objective == 'min':
            return self.fun(*state)
        elif self.objective == 'max':
            return -self.fun(*state)
        else:
            raise ValueError('Objective should be either "min" or "max".')


def rvf(x,y):
    return x**4-3*x**2+y**4-3*y**2+1

rvf2 = Rvf2Optimize(rvf, (1,0), 50, bounds=[[-2,2],[-2,2]])
print(rvf2.anneal())
