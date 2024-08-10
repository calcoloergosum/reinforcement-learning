"""
Heuristics from Boumaza 2009.
"""
import itertools

import numpy as np
from typing import List, Tuple
import rust_tetris
from .action_cluster import MINO2ACTIONS
import random

np.set_printoptions(precision=3, threshold=1e5, suppress=True)


def piece_dependency(game: rust_tetris.Game):
    rows = np.array(game.playfield).reshape(22, 10)
    rows = game2bool_field(game)
    heights = column2heights(rows.T)
    # IJLOSTZ

    same1 = heights[:-1] == heights[1:]
    same2 = same1[:-1] == same1[1:]
    same3 = same2[:-1] == same2[1:]
    up1  = heights[:-1] == heights[1:] - 1
    down1= heights[:-1] == heights[1:] + 1
    up2  = heights[:-1] == heights[1:] - 2
    down2= heights[:-1] == heights[1:] + 2

    n_up1_same = (up1[:-1] & same1[1:]).sum()
    n_same1_down1 = (same1[:-1] & down1[1:]).sum()
    n_same1_up1  = (same1[:-1] & up1[1:]).sum()
    n_same1 = same1.sum()
    n_down1 = down1.sum()
    n_down1_same1 = (down1[:-1] & same1[1:]).sum()
    n_down1_up1 = (down1[1:] & up1[:1]).sum()
    n_down2 = down2.sum()
    n_up1 = up1.sum()
    n_up2 = up2.sum()
    n_same2 = same2.sum()
    n_same3 = same3.sum()
    ns = [
        # I
        7 + n_same3,
        # J
        n_same2 + n_same1 + n_up2 + n_same1_down1,
        # L
        n_same2 + n_same1 + n_down2 + n_up1_same,
        # O
        n_same1,
        # S
        n_down1 + n_same1_up1,
        # T
        n_same2 + n_up1 + n_down1 + n_down1_up1,
        # Z
        n_up1 + n_down1_same1,
    ]
    return ns


def bool_field2feature_heuristic(rows: np.ndarray):
    """Return f3 ~ f8 of Heuristic measures from Boumaza 2009, Table 1

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
    rows = rows.astype(np.int8)
    columns = rows.T[:, ::-1]
    column_heights = column2heights(columns)

    # f3: Row transitions
    row_border = np.ones((rows.shape[0], 1), dtype=rows.dtype)
    _rows = np.hstack((row_border, rows, row_border))
    _rows_diff = np.diff(_rows, axis=1)
    r_trans = np.abs(_rows_diff).astype(int).sum()

    # f4: Column transitions
    col_border = np.ones((columns.shape[0], 1), dtype=rows.dtype)
    _columns = np.hstack((col_border, columns, col_border))
    c_trans = np.abs(np.diff(_columns, axis=1)).astype(int).sum()

    # f6: Cumulative wells
    # ∑ w ∈ wells(1 + 2 + . . . + depth(w))
    sum_wells = ((_rows_diff == -1)[:, :-1] * (_rows_diff == 1)[:, 1:]).sum()

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
        # d_hole = 0
        # last_holes = np.nonzero(np.diff(col[:h]) == 1)[0]
        # d_hole += (h - 1 - last_holes).sum()
        # diff = np.diff(col[:h]) == 1
        # if diff.sum() > 0:
        #     last_hole = np.argmax(np.diff(col[:h]) == 1)
        #     d_hole += h - 1 - last_hole

        # f8: Rows holes
        r_hole += _n_hole > 0

    return r_trans, c_trans, n_hole, sum_wells, d_hole, r_hole


def column2heights(cs):
    return np.argmax((cs != 0) + np.arange(22)[None, :] * 0.01, axis=1)


# from Boumaza 2009, Appendix A
f2weight = np.array([-0.3213,  0.0744, -0.2851, -0.5907,  -0.2188, -0.2650, -0.0822, -0.5499])
def board2score(b):
    return (f2weight * bool_field2feature_heuristic(b > 0)).sum()


def game2bool_field(game: rust_tetris.Game) -> np.ndarray:
    arr = np.array(game.playfield, dtype=np.int8).reshape(22, 10)
    arr[arr == 8] = 0  # Remove ghost
    arr[arr == 10] = 0  # Remove current piece
    return arr > 0


f2weight_dellacherie = np.array([
    -4.500158825082766,
    3.4181268101392694,
    -3.2178882868487753,
    -3.2178882868487753,
    -7.899265427351652,
    -3.3855972247263626,
    0, 0
])
def game2score_dellacherie(game: rust_tetris.Game) -> float:
    if game.is_game_over:
        return -1e5
    x, n = game.delta_last or (0, 0)
    feature = (22 - x, 2 * n * n, *bool_field2feature_heuristic(game2bool_field(game)))
    return f2weight_dellacherie @ feature


# From github.com/takado8
def bool_field2feature_takado8(rows):
    columns = rows.T[:, ::-1]
    column_heights = column2heights(columns)

    sum_height = sum(column_heights)

    n_hole = 0  # f5: Number of holes
    for col, h in zip(columns, column_heights):
        if h == 0:
            continue
        n_hole += sum(col[:h] == 0)

    bumpiness = np.abs(np.diff(column_heights)).sum()

    return sum_height, n_hole, bumpiness
    

# Original
# f2weight_takado8 = np.array([0.522287506868767, -0.798752914564018, -0.24921408023878, -0.164626498034284])

# Modified
# Don't worry about line clear too much. Focus on flattening it out.
f2weight_takado8 = np.array([0.1, -0.798752914564018, -0.24921408023878, -0.164626498034284])
def game2score_takado8(game: rust_tetris.Game) -> float:
    if game.is_game_over:
        return -1e5
    return f2weight_takado8 @ (game.line_clears, *bool_field2feature_takado8(game2bool_field(game)))


def game2score_dellacherie_vanilla_mix(game: rust_tetris.Game) -> float:
    if game.is_game_over:
        return -1e5
    return 0.5 * game.line_clears + game2score_dellacherie(game)


def game2score_vanilla(game: rust_tetris.Game) -> float:
    if game.is_game_over:
        return -1e5
    return game.score


def game2actions(game: rust_tetris.Game) -> List[Tuple[int, int]]:
    i_kind, *_ = game.piece
    str_kind = rust_tetris.piece_kind_int2str(i_kind)
    return MINO2ACTIONS[str_kind]


def play(game, action: int) -> rust_tetris.Game:
    game = game.copy()
    r, t = game2actions(game)[action]
    for _ in range(r):
        game.rotate_c()
    for _ in range(0, t):
        game.right()
    for _ in range(0, -t):
        game.left()
    game.hard_drop()
    return game


def action2commands(r: int, t: int) -> List[str]:
    ret = []
    if r != 0:
        ret.extend(('rotate',) * r)
    if t > 0:
        ret.extend(('right',) * t)
    else:
        ret.extend(('left',) * (-t))
    ret.append('hard_drop')
    return ret


def get_policy(game2score, epsilon):
    def select_action_cluster(game):
        actions = game2actions(game)
        if epsilon > random.random():
            # Random action
            i_action, action = random.choice(list(enumerate(actions)))
        else:
            scores = [(game2score(play(game, i)), i, a) for i, a in enumerate(actions)]
            _, i_action, action = max(scores)
        return i_action, action2commands(*action)
    return select_action_cluster


def game2score_piece_dependency(g: rust_tetris.Game):
    v_max = -1000
    for i_a1, _ in enumerate(game2actions(g)):
        g1 = play(g, i_a1)
        for i_a2, _ in enumerate(game2actions(g1)):
            g2 = play(g1, i_a2)
            v = game2score_dellacherie(g2)
            if v_max < v:
                v_max = v
    return v_max
