import cv2
import numpy as np
import tetris

# Constant: Interface
KEY_A = ord('a')
KEY_D = ord('d')
KEY_S = ord('s')
KEY_W = ord('w')
KEY_SPACE = 32
CV2_WINDOW_NAME = 'tetris'
# cv2.namedWindow(CV2_WINDOW_NAME, cv2.WINDOW_NORMAL)

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
    (128, 128, 128),  # Ghost
]


N_QUEUE = 1

BOARD_HEIGHT = 10
BOARD_WIDTH = 5


def get_render_func(game: tetris.BaseGame):
    BG_COLOR = 30
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


def main():
    game = tetris.BaseGame(board_size=(BOARD_HEIGHT, BOARD_WIDTH), seed=128)
    render = get_render_func(game)

    while game.status == tetris.PlayingStatus.PLAYING:
        game.tick()

        wait_func = render()
        key = wait_func(30)
        if key == -1:
            continue
        if key == KEY_SPACE:
            game.hard_drop()
        if key == KEY_A:
            game.left()
        if key == KEY_D:
            game.right()
        if key == KEY_S:
            game.soft_drop()
        if key == KEY_W:
            game.rotate()


if __name__ == '__main__':
    main()
