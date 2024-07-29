from typing import List

import numpy as np
import rust_tetris


def game2serialize(game: rust_tetris.Game) -> List[int]:
    arr = np.array(game.playfield, dtype=np.int8)
    arr[arr == 8] = 0  # Remove ghost
    arr[arr == 10] = -1  # Flip current piece
    piece = game.piece
    queue = game.queue
    return (*arr.tolist(), *queue, piece[0], game.score)
