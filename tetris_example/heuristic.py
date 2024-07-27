"""
Heuristics from Boumaza 2009.
"""
import itertools

import numpy as np


def bool_field2feature_heuristic(rows: np.ndarray):
    """Heuristic measures from Boumaza 2009, Table 1

    | Id |       Name         |     Description                                                                                             |
    |----|--------------------|-------------------------------------------------------------------------------------------------------------|
    | f1 | Landing height     | Height where the last piece was added                                                                       |
    | f2 | Eroded pieces      | (# of lines cleared in the last move) × (# of cells of the last piece that were eliminated in the last move)|
    | f3 | Row transitions    | # of horizontal cell transitions (from filled to empty)                                                     |
    | f4 | Column transitions | # of vertical cell transitions                                                                              |
    | f5 | Holes              | # of empty cells covered by at least one filled cell                                                        |
    | f6 | Cumulative wells   | ∑ w ∈ wells(1 + 2 + . . . + depth(w))                                                                       |
    | f7 | Hole depth         | # of filled cells on top of each hole                                                                       |
    | f8 | Rows holes         | # of rows with at least one hole                                                                            |
    """
    columns = rows.T[:, ::-1]
    column_heights = [column2height(col) for col in columns]

    # f1: Landing height  # TODO: do properly
    # NOTE: using max height instead for simplicity
    max_height = max(column_heights)

    # f2: Eroded pieces
    # A * B where
    #     A = # of lines cleared in the last move
    #     B = # of cells of the last piece that were eliminated in the last move)
    # TODO: implement
    eroded_pieces = 0

    r_trans = np.diff(rows, axis=1).astype(int).sum()  # f3: Row transitions
    c_trans = np.diff(rows, axis=0).astype(int).sum()  # f4: Column transitions

    n_hole = 0  # f5: Number of holes
    d_hole = 0  # f7: Hole depth
    r_hole = 0  # f8: Rows holes
    for col, h in zip(columns, column_heights):
        if h == 0:
            continue
        _n_hole = sum(col[:h] == 0)

        # f5: Number of holes
        n_hole += _n_hole

        # f7: Hole depth
        # Count sequence of holes as one hole
        d_hole = 0
        # last_holes = np.nonzero(np.diff(col[:h]) == 1)[0]
        # d_hole += (h - 1 - last_holes).sum()
        # diff = np.diff(col[:h]) == 1
        # if diff.sum() > 0:
        #     last_hole = np.argmax(np.diff(col[:h]) == 1)
        #     d_hole += h - 1 - last_hole

        # f8: Rows holes
        r_hole += _n_hole > 0

    # f6: Cumulative wells
    # ∑ w ∈ wells(1 + 2 + . . . + depth(w))
    sum_wells = sum(abs(h1 - h2) for h1, h2 in zip(column_heights, column_heights[1:]))

    return max_height, eroded_pieces, r_trans, c_trans, n_hole, sum_wells, d_hole, r_hole


def column2height(c):
    is_any_nonzero, h = max(zip(c != 0, itertools.count()))
    return (h + 1) if is_any_nonzero else 0


# from Boumaza 2009, Appendix A
f2weight = np.array([-0.3213,  0.0744, -0.2851, -0.5907,  -0.2188, -0.2650, -0.0822, -0.5499])
def board2score(b):
    return (f2weight * bool_field2feature_heuristic(b > 0)).sum()


# from TetrisAI
# f2weight = np.array([-0.798752914564018, 0, 0, 0, -0.24921408023878, -0.164626498034284, 0, 0])
# def board2score(b):
#     return (f2weight * bool_field2feature_heuristic(b > 0)).sum()
