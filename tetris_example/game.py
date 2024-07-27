import numpy as np
import rust_tetris
from typing import List
import pathlib
from .board import select_action as manual_policy
from .heuristic import board2score
from .uct import get_policy as get_uct_policy
import time


def game2bool_field(game: rust_tetris.Game) -> np.ndarray:
    arr = np.array(game.playfield, dtype=np.int8).reshape(22, 10)
    arr[arr == 8] = 0  # Remove ghost
    arr[arr == 10] = 0  # Remove current piece
    return arr


def game2score_vanilla(game: rust_tetris.Game) -> float:
    if game.is_game_over:
        return -1e5
    return game.score


def game2score_heuristic(game: rust_tetris.Game) -> float:
    if game.is_game_over:
        return -1e5
    return board2score(game2bool_field(game))


def game2score_mixed(game: rust_tetris.Game) -> float:
    if game.is_game_over:
        return -1e5
    return 0.5 * game.line_clears + board2score(game2bool_field(game))


def game2serialize(game: rust_tetris.Game) -> List[int]:
    arr = np.array(game.playfield, dtype=np.int8)
    arr[arr == 8] = 0  # Remove ghost
    arr[arr == 10] = -1  # Flip current piece
    piece = game.piece
    queue_action = game.queue
    return (*arr, *piece, *queue_action, game.score)


strategies = {
    "mixed": {
        "n_search": 3000,
        "ucb_constant": 3,
        "game2score": game2score_mixed,
    },
    "vanilla": {
        "n_search": 100000,
        "ucb_constant": 40,
        "game2score": game2score_vanilla,
    },
    "heuristic": {
        "n_search": 1000,
        "ucb_constant": 10,
        "game2score": game2score_heuristic,
    },
    # "heuristic": {
    #     "n_search": 1000,
    #     "ucb_constant": 10,
    #     "game2score": game2score_heuristic,
    # },
}


UCB_CONSTANT_HEURISTIC = 10
def game2score_heuristic(game: rust_tetris.Game) -> float:
    return board2score(game2bool_field(game))


def main():
    """Play interactively"""
    method = 'mcts'
    # method = 'manual'
    strategy_name = 'mixed'
    # strategy_name = 'heuristic'
    # strategy_name = 'vanilla'
    epsilon = 0.0

    if method == 'mcts':
        policy = get_uct_policy(**strategies[strategy_name], epsilon=epsilon)
    elif method == 'manual':
        policy = manual_policy
    else:
        raise KeyError(f"Unknown policy {method}")

    game = rust_tetris.Game()

    # Sparse hole pattern
    board = np.zeros((22, 10))
    board[10:] = 1  # first 2 + 8 lines empty
    n_hole_per_line = 4
    board[
        np.arange(0, 22)[:, None].repeat(n_hole_per_line, axis=1),
        np.random.randint(0, 10, (22, n_hole_per_line)),
    ] = 0
    game.board = board.astype(int).tolist()

    history = [game.copy()]
    while not game.is_game_over and len(history) < 300:
        action = policy(game.copy())
        if action == 'unknown':
            game.soft_drop()
            pass
        elif action == 'hard_drop':
            game.hard_drop()
        elif action == 'left':
            game.left()
        elif action == 'right':
            game.right()
        elif action == 'soft_drop':
            game.soft_drop()
        elif action == 'rotate':
            game.rotate_c()
        elif action == 'go_back':
            print("#history:", len(history))
            try:
                game = history.pop()
                game = history.pop()
            except:
                print("undo list empty")
                pass
        elif action == 'swap':
            game.swap()
        else:
            print(f"Unknown action {action}")
        history.append(game.copy())

    logdir = pathlib.Path(f"logs/{method}_{strategy_name}_{epsilon}")
    logdir.mkdir(exist_ok=True, parents=True)
    with (logdir / f"log_{int(time.time())}.txt").open("w", encoding='utf8') as fp:
        for g in history:
            print(' '.join(map(str, game2serialize(g))), file=fp)

if __name__ == '__main__':
    main()
