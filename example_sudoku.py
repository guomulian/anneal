from anneal import SimulatedAnnealer
import random


class SudokuSolver(SimulatedAnnealer):
    def __init__(self, initial_state, max_steps):
        super().__init__(initial_state, max_steps)

    def _neighbor(self):
        pass

    def _energy(self, state):
        pass


if __name__ == '__main__':

    puzzle_1 = [[0, 0, 0, 2, 6, 0, 7, 0, 1],
                [6, 8, 0, 0, 7, 0, 0, 9, 0],
                [1, 9, 0, 0, 0, 4, 5, 0, 0],
                [8, 2, 0, 1, 0, 0, 0, 4, 0],
                [0, 0, 4, 6, 0, 2, 9, 0, 0],
                [0, 5, 0, 0, 0, 3, 0, 2, 8],
                [0, 0, 9, 3, 0, 0, 0, 7, 4],
                [0, 4, 0, 0, 5, 0, 0, 3, 6],
                [7, 0, 3, 0, 1, 8, 0, 0, 0]]

    solution_1 = [[4, 3, 5, 2, 6, 9, 7, 8, 1],
                  [6, 8, 2, 5, 7, 1, 4, 9, 3],
                  [1, 9, 7, 8, 3, 4, 5, 6, 2],
                  [8, 2, 6, 1, 9, 5, 3, 4, 7],
                  [3, 7, 4, 6, 8, 2, 9, 1, 5],
                  [9, 5, 1, 7, 4, 3, 6, 2, 8],
                  [5, 1, 9, 3, 2, 6, 8, 7, 4],
                  [2, 4, 8, 9, 5, 7, 1, 3, 6],
                  [7, 6, 3, 4, 1, 8, 2, 5, 9]]
