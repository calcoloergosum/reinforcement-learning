import numpy as np
import rust_tetris
from typing import List
import pathlib
from .board import select_action as manual_policy, get_render_func
from .heuristic import board2score
from .uct import get_policy as get_uct_policy
import time
import json


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
    queue = game.queue
    piece_onehot = [0 for _ in range(7)]
    piece_onehot[piece[0] - 1] = 1
    return (*arr.tolist(), *queue, piece[0], game.score)


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
    strategy_name = 'mixed'
    # strategy_name = 'heuristic'
    # strategy_name = 'vanilla'
    # Epsilon greedy; Random movement ratio.
    # With 5%, it seems to breeze through the randomness.
    # With 10%, it seems to suffer too much.
    epsilon = 0.07
    policy = get_uct_policy(**strategies[strategy_name], epsilon=epsilon)

    game = rust_tetris.Game()

    # Sparse hole pattern
    # board = np.zeros((22, 10))
    # board[10:] = 1  # first 2 + 8 lines empty
    # n_hole_per_line = 4
    # board[
    #     np.arange(0, 22)[:, None].repeat(n_hole_per_line, axis=1),
    #     np.random.randint(0, 10, (22, n_hole_per_line)),
    # ] = 0
    # game.board = board.astype(int).tolist()

    states = [game.copy()]
    actions = []
    while not game.is_game_over and len(states) < 200:
        print(f"Move #{len(states):0>4}")
        i_action, action_cluster = policy(game.copy())
        for action in action_cluster:
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
            elif action == 'swap':
                game.swap()
            else:
                raise KeyError(action)
            _ = get_render_func(game)()(1)
        states.append(game.copy())
        actions.append((game.piece[0], int(i_action)))
    
    logdir = pathlib.Path(f"logs/{strategy_name}_{epsilon}")
    logdir.mkdir(exist_ok=True, parents=True)
    (logdir / f"log_{int(time.time())}.json").write_text(
        json.dumps({
            "actions": actions,
            "states": [game2serialize(state) for state in states],
            "over": states[-1].is_game_over,
        }),
        encoding='utf8',
    )


def main_manual():
    """Play interactively"""
    policy = manual_policy

    game = rust_tetris.Game()

    history = [game.copy()]
    while not game.is_game_over:
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


if __name__ == '__main__':
    main()
