from examples.sudoku.sudoku import SudokuSolver
from anneal import helpers
import pytest
import random
import os


@pytest.fixture
def solver():
    def _solver(puzzle, max_steps=1000):
        return SudokuSolver(puzzle, max_steps)
    return _solver


@pytest.fixture
def puzzle_all_zeros():
    return [[0 for _ in range(9)] for _ in range(9)]


@pytest.fixture
def puzzle_cols():
    return [[n for n in range(9)] for _ in range(9)]


@pytest.fixture
def puzzle_valid():
    return [[0, 0, 0, 2, 6, 0, 7, 0, 1],
            [6, 8, 0, 0, 7, 0, 0, 9, 0],
            [1, 9, 0, 0, 0, 4, 5, 0, 0],
            [8, 2, 0, 1, 0, 0, 0, 4, 0],
            [0, 0, 4, 6, 0, 2, 9, 0, 0],
            [0, 5, 0, 0, 0, 3, 0, 2, 8],
            [0, 0, 9, 3, 0, 0, 0, 7, 4],
            [0, 4, 0, 0, 5, 0, 0, 3, 6],
            [7, 0, 3, 0, 1, 8, 0, 0, 0]]


@pytest.fixture
def puzzle_valid_solution():
    return [[4, 3, 5, 2, 6, 9, 7, 8, 1],
            [6, 8, 2, 5, 7, 1, 4, 9, 3],
            [1, 9, 7, 8, 3, 4, 5, 6, 2],
            [8, 2, 6, 1, 9, 5, 3, 4, 7],
            [3, 7, 4, 6, 8, 2, 9, 1, 5],
            [9, 5, 1, 7, 4, 3, 6, 2, 8],
            [5, 1, 9, 3, 2, 6, 8, 7, 4],
            [2, 4, 8, 9, 5, 7, 1, 3, 6],
            [7, 6, 3, 4, 1, 8, 2, 5, 9]]


@pytest.fixture
def grid():
    return [[(row, col) for col in range(9)] for row in range(9)]


def test_initialized_with_bad_puzzle():
    # bad shape
    with pytest.raises(IndexError):
        SudokuSolver([], 1000)

    # bad values
    with pytest.raises(ValueError):
        board = [[num for num in range(-1, 8)] for _ in range(9)]
        SudokuSolver(board, 1000)


def test_block_indices():
    # test first block
    test_indices = SudokuSolver.block_indices(0)
    actual_indices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0),
                      (2, 1), (2, 2)]
    assert sorted(test_indices) == sorted(actual_indices)


def test_fill_puzzle(puzzle_cols):
    # [[0,...,8],...,[0,...,8]] should fill in the puzzle
    # with 1-9 in the first column, and return the first
    # column's indices in unknown

    filled, unknown = SudokuSolver.fill_puzzle(puzzle_cols)

    assert all(filled[row][0] in list(range(1, 10)) for row in range(9))
    assert all(u[1] == 0 for u in unknown)


def test_block_correct(puzzle_cols, grid):
    assert SudokuSolver.block(0, puzzle_cols) == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert SudokuSolver.block(8, puzzle_cols) == [6, 7, 8, 6, 7, 8, 6, 7, 8]
    assert SudokuSolver.block(4, grid) == [(3, 3), (3, 4), (3, 5),
                                           (4, 3), (4, 4), (4, 5),
                                           (5, 3), (5, 4), (5, 5)]


def test_blockify_correct(puzzle_cols):
    assert SudokuSolver.blockify(puzzle_cols) == [[0, 1, 2, 0, 1, 2, 0, 1, 2],
                                                  [3, 4, 5, 3, 4, 5, 3, 4, 5],
                                                  [6, 7, 8, 6, 7, 8, 6, 7, 8],
                                                  [0, 1, 2, 0, 1, 2, 0, 1, 2],
                                                  [3, 4, 5, 3, 4, 5, 3, 4, 5],
                                                  [6, 7, 8, 6, 7, 8, 6, 7, 8],
                                                  [0, 1, 2, 0, 1, 2, 0, 1, 2],
                                                  [3, 4, 5, 3, 4, 5, 3, 4, 5],
                                                  [6, 7, 8, 6, 7, 8, 6, 7, 8]]


def test_energy_correct(puzzle_all_zeros, puzzle_valid_solution):
    assert SudokuSolver.energy_method(None, puzzle_all_zeros) == -18
    assert SudokuSolver.energy_method(None, puzzle_valid_solution) == -162


def test_neighbor_switches_two_in_same_block(solver, puzzle_valid, grid):
    random.seed(0)
    s = solver(puzzle_valid)
    state = s.initial_state
    neighbor = s.neighbor(state)

    diffs = []

    for row in range(9):
        for col in range(9):
            if state[row][col] != neighbor[row][col]:
                diffs.append((row, col))

    assert len(diffs) == 2

    (r1, c1), (r2, c2) = diffs[0], diffs[1]

    assert state[r1][c1] == neighbor[r2][c2]
    assert state[r2][c2] == neighbor[r1][c1]

    # check that they are both in same block (and only one block)
    blocks = SudokuSolver.blockify(grid)
    assert sum(all(p in block for p in diffs) for block in blocks) == 1


def test_neighbor_on_already_solved(solver, puzzle_valid_solution):
    random.seed(0)
    s = solver(puzzle_valid_solution)
    state = s.initial_state
    neighbor = s.neighbor(state)

    assert state == neighbor


def test_energy_break_on_solved_puzzle(tmpdir, solver, puzzle_valid_solution):
    random.seed(0)
    file = tmpdir.join(helpers.generate_filename(SudokuSolver, ".pickle"))
    rounds = 3

    s = solver(puzzle_valid_solution)
    s.anneal(pickle=True, pickle_file=file, energy_break_rounds=rounds,
             energy_break_tol=0.05)

    states = s.unpickle_states()

    assert len(states) == rounds
