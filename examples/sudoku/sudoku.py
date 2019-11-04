from anneal import anneal
import collections
import copy
import random


class SudokuSolver(anneal.BaseAnnealer):
    """For solving Sudoku puzzles.

    Randomly fills in the board from the set of missing values, then
    switches the inserted numbers until the number of errors is minimal.

    The number of errors is quantified by the energy/cost function, which
    is computed by counting the number of unique elements in each row, column,
    and 3x3 block.
    """

    def __init__(self, puzzle, max_steps):
        if not all(all(e in range(10) for e in row) for row in puzzle):
            raise ValueError('Cells in the board must be in [0,9] (where 0 \
                  represents an empty cell).')

        # in case the original puzzle is needed in the future
        self.puzzle = copy.deepcopy(puzzle)

        initial_state, unknown = SudokuSolver.fill_puzzle(puzzle)

        # we keep a list of unknown cells; we only want to make swaps with
        # these, not the pre-filled cells
        self.unknown = unknown

        super().__init__(initial_state, max_steps)

    @staticmethod
    def pretty_print(board):  # pragma: no cover
        """Prints the board in a readable format."""
        result = '\n'.join(' '.join(map(str, row)) for row in board)
        print(result)

    @staticmethod
    def fill_puzzle(board):
        """Fills the unknown cells in the board and returns the filled board
        and the list of the indices of the unknown cells.

        Filling is done so that each block contains the correct set of numbers,
        i.e., 1-9.
        """

        unknown = []

        for block_index in range(9):
            block_indices = SudokuSolver.block_indices(block_index)
            missing = [num for num in range(1, 10) if num not in
                       [board[i][j] for i, j in block_indices]]

            random.shuffle(missing)

            for i, j in block_indices:
                if board[i][j] == 0:
                    board[i][j] = missing.pop()
                    unknown.append((i, j))

        return board, unknown

    @staticmethod
    def block_indices(block_index):
        """Gets the indices of a given block."""

        # indicies of the upper leftmost cell of the block
        start_row = 3*(block_index // 3)
        start_col = 3*(block_index % 3)

        indices = []

        for row in range(start_row, start_row + 3):
            for col in range(start_col, start_col + 3):
                indices.append((row, col))

        return indices

    @staticmethod
    def block(block_index, board):
        """Returns a (flattened) block of a given board."""
        return [board[i][j] for i, j in
                SudokuSolver.block_indices(block_index)]

    @staticmethod
    def blockify(board):
        """Returns a list of the 3x3 blocks."""
        return [SudokuSolver.block(b, board) for b in range(9)]

    def neighbor(self, state):
        """Returns a randomly selected "neighboring" board.

        First, we pick a random 3x3 block. Then, we randomly pick two "unknown"
        cells in the block. (If there are less than two, we pick a different
        block.) Then, we swap the contents of these two cells and return the
        resulting board.
        """

        potential_blocks = list(range(9))

        while len(potential_blocks) > 0:
            block = random.choice(potential_blocks)
            candidates = [c for c in self.unknown if c in
                          SudokuSolver.block_indices(block)]

            if len(candidates) < 2:
                potential_blocks.remove(block)
                continue

            else:
                break

        else:
            # len(candidates) < 2 in every block;
            # puzzle should be already solved
            return state

        (i1, j1), (i2, j2) = random.sample(candidates, 2)

        neighbor = copy.deepcopy(state)

        neighbor[i1][j1], neighbor[i2][j2] = neighbor[i2][j2], neighbor[i1][j1]

        return neighbor

    def energy(self, state):
        """Adds -1 to the energy/score for every unique value in each
        row/column.

        "Best" score is 9*(-9) + 9*(-9) = -162
        """

        row_score = sum(-len(set(row)) for row in state)
        col_score = sum(-len(set(col)) for col in map(list, zip(*state)))

        return row_score + col_score
