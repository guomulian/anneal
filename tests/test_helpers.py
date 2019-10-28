from anneal import helpers


def test_clip():
    # lower == upper
    assert helpers.clip(2, 1, 1) == 1

    # item within bounds
    assert helpers.clip(0, -1, 1) == 0
