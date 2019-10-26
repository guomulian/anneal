from examples.sudoku import sudoku
import pytest


def test_initialized_with_bad_puzzle():
    # bad shape
    with pytest.raises(IndexError):
        sudoku.SudokuSolver([], 1000)

    # bad values
    with pytest.raises(ValueError):
        board = [[num for num in range(-1, 8)] for _ in range(9)]
        sudoku.SudokuSolver(board, 1000)


def test_block_indices():
    # test first block
    test_indices = sudoku.SudokuSolver.block_indices(0)
    actual_indices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0),
                      (2, 1), (2, 2)]
    assert sorted(test_indices) == sorted(actual_indices)


def test_fill_puzzle():
    # [[0,...,8],...,[0,...,8]] should fill in the puzzle
    # with 1-9 in the first column, and return the first
    # column's indices in unknown

    board = [[n for n in range(9)] for _ in range(9)]
    filled_board, unknown = sudoku.SudokuSolver.fill_puzzle(board)

    print(filled_board)

    assert all(filled_board[row][0] in list(range(1, 10)) for row in range(9))
    assert all(u[1] == 0 for u in unknown)
