from array import array
import tetris

import numpy as np
import cv2

from typing import Literal


CV2_WINDOW_NAME = 'tetris'


# IJLOSTZ
COLORMAP = [
    (0,     0,   0),  # BG; Black
    (255, 255,   0),  # I; Cyan
    (255,   0,   0),  # J; Blue
    (  0, 128, 255),  # L; Orange
    (  0, 255, 255),  # O; Yellow
    (  0, 255,   0),  # S; Green
    (255,  50, 200),  # T; Purple
    (  0,   0, 255),  # Z; Red
    (30, 30, 30),  # Ghost
]


N_QUEUE = 3

BOARD_HEIGHT = 5
BOARD_WIDTH = 10


def get_render_func(game: tetris.BaseGame):
    BG_COLOR = 80
    SCALE_FIELD = 80
    SCALE_QUEUE = 40
    BUFFER = 50
    MARGIN = 50

    # Prepare canvas
    h, w = game.playfield.shape
    h += 3
    h_queue_cell = 2
    w_queue_cell = 4
    board = np.zeros((h, w, 3), dtype=np.uint8)
    board_queue = np.zeros((N_QUEUE * (h_queue_cell + 1), w_queue_cell, 3), dtype=np.uint8)
    canvas = BG_COLOR * np.ones((h * SCALE_FIELD + MARGIN, w * SCALE_FIELD + w_queue_cell * SCALE_QUEUE + BUFFER + 2 * MARGIN, 3), dtype=np.uint8)
    # Prepare canvas done

    def render():
        nonlocal canvas

        # playfield
        board.fill(0)
        arr = np.array(game.get_playfield(buffer_lines=3).data).reshape(h, w)
        for i, v in enumerate(COLORMAP):
            board[arr == i] = v
        # playfield done

        # queue
        board_queue.fill(BG_COLOR)
        for i, v in enumerate(game.queue[:N_QUEUE]):
            v = game.queue[i]
            board_queue[i * (h_queue_cell + 1): (i + 1) * (h_queue_cell + 1) - 1] = 0
            for x, y in game.rs.spawn(v).minos:
                board_queue[i * (h_queue_cell + 1) + x, y] = COLORMAP[v.value]
        # queue done

        # Draw to canvas
        playfield = cv2.resize(board, (w * SCALE_FIELD, h * SCALE_FIELD), interpolation=cv2.INTER_NEAREST)
        queue     = cv2.resize(board_queue, (w_queue_cell * SCALE_QUEUE, N_QUEUE * (h_queue_cell + 1) * SCALE_QUEUE), interpolation=cv2.INTER_NEAREST)
        canvas[:playfield.shape[0], MARGIN: MARGIN + playfield.shape[1]] = playfield
        canvas[MARGIN: MARGIN + queue.shape[0], MARGIN + playfield.shape[1] + BUFFER: MARGIN + playfield.shape[1] + BUFFER + queue.shape[1],] = queue
        cv2.imshow(CV2_WINDOW_NAME, canvas)
        # Draw to canvas done

        return cv2.waitKey
    return render


Action = Literal["left", "right", "wait", "rotate"]


def number_of_blocks_dropped(g: tetris.BaseGame) -> float:
    return (sum(np.array(g.board.data) > 0) + BOARD_WIDTH * g.scorer.line_clears) / 4


def immediate_reward(g_bef: tetris.BaseGame, g_aft: tetris.BaseGame) -> float:
    reward = 0.

    # *_, n_hole_aft, diff_total_aft, max_height_aft = game2feature_heuristic(g_aft)
    # *_, n_hole_bef, diff_total_bef, max_height_bef = game2feature_heuristic(g_bef)
    reward += 10 * (g_aft.scorer.line_clears - g_bef.scorer.line_clears) ** 2
    if number_of_blocks_dropped(g_aft) != number_of_blocks_dropped(g_bef):
        reward += 1
    # reward += .1 * (n_hole_bef - n_hole_aft)

    return reward


def get_next_game(g: tetris.BaseGame, a: Action, engine) -> tetris.BaseGame | None:
    _g = copy_game(g, engine)
    if a == 'wait':
        if engine is not None:
            _g.gravity.tick()
        _g.tick()
    elif a == 'unknown':
        pass
    else:
        getattr(_g, a)()
        if engine is not None:
            _g.gravity.tick()
        _g.tick()
    return immediate_reward(g, _g) , _g, _g.lost


def copy_game(g: tetris.BaseGame, engine):
    is_engine_auto = engine is None
    _g = tetris.BaseGame(tetris.impl.presets.Modern if is_engine_auto else engine,
                         board=g.board.copy(), board_size=(BOARD_HEIGHT, BOARD_WIDTH))
    _g.queue._pieces = g.queue._pieces.copy()
    _g.scorer.goal = g.scorer.goal
    _g.scorer.level = g.scorer.level
    _g.scorer.score = g.scorer.score
    _g.scorer.line_clears = g.scorer.line_clears
    if not is_engine_auto:
        _g.gravity.last_drop = g.gravity.last_drop
        _g.gravity.now = g.gravity.now
    _g.piece = g.piece.__class__(
        type=g.piece.type,
        x=g.piece.x,
        y=g.piece.y,
        r=g.piece.r,
        minos=g.piece.minos,
    )
    # _g.board._data = g.board._data
    # print(_g.playfield)
    return _g


def select_action(game: tetris.BaseGame) -> Action | Literal['unknown']:
    # Constant: Interface
    KEY_A = ord('a')
    KEY_D = ord('d')
    KEY_S = ord('s')
    KEY_W = ord('w')
    KEY_SPACE = 32
    # cv2.namedWindow(CV2_WINDOW_NAME, cv2.WINDOW_NORMAL)

    key = get_render_func(game)()(30)
    if key == -1:
        return 'wait'
    if key == KEY_SPACE:
        return 'hard_drop'
    if key == KEY_A:
        return 'left'
    if key == KEY_D:
        return 'right'
    if key == KEY_S:
        return 'soft_drop'
    if key == KEY_W:
        return 'rotate'
    print(f"Unknown key {key}")
    return 'unknown'


def get_random_state(buff: int, engine, seed: int | None = None) -> tetris.BaseGame:
    q = [1,2,3,4,5,6,7] * 100
    game = tetris.BaseGame(
        tetris.impl.presets.Modern if engine is None else engine,
        board_size=(BOARD_HEIGHT, BOARD_WIDTH), seed=seed, queue=q)

    b = np.array(game.board.data).reshape(BOARD_HEIGHT * 2, BOARD_WIDTH)
    _b = np.round(np.random.random((b.shape[0] // 2 - buff, b.shape[1])))
    _b[(_b > 0).all(axis=1)] = 0
    b[BOARD_HEIGHT + buff:] = np.round(_b)
    game.board._data = array('B', b.flatten())
    return game
