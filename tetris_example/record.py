import argparse
import copy
import json
import pathlib
import random
import time

import humanfriendly
import numpy as np
import rust_tetris

from . import game, heuristic, uct
from .board import get_render_func
from .board import select_action as manual_policy

settings = {
    "log_dir": pathlib.Path("logs"),
    "strategy_name": "dellacherie",
    # "strategy_name": 'takado8_small',
    # "strategy_name": 'vanilla',
    # "strategy_name": 'path_dependency',
    "max_history": 500,
    "render_every": 1000,
    "strategy": {
        # Epsilon greedy; Random movement ratio.
        "epsilon": 0.5,
    },
    "board": {
        "type": "sparse-hole",  # ['sparse-hole', 'empty']
        "n_hole_per_line": 4,  # ['sparse-hole', 'empty']
    }
}
strategies = {
    "takado8_small": {
        # With 5%, it seems to breeze through the randomness.
        # With 10%, it seems to suffer just enough to make great episodes.
        "game2score": heuristic.game2score_takado8,
        "score_type": "immediate",
    },
    "dellacherie": {
        "game2score": heuristic.game2score_dellacherie,
        "score_type": "immediate",
    },
    "piece_dependency": {
        "game2score": heuristic.game2score_piece_dependency,
        "score_type": "immediate",
    },
    "vanilla": {
        "game2score": heuristic.game2score_vanilla,
        "score_type": "cumulative",
        "n_search": 3000,
        "ucb_constant": 40,
    },
}


def get_policy(config):
    strategy_name = config["strategy_name"]
    strategy = strategies[strategy_name].copy()
    score_type = strategy.pop("score_type")
    if score_type == 'cumulative':
        policy = uct.get_policy(**strategy, **config["strategy"], n_thread=0)
    elif score_type == 'immediate':
        policy = heuristic.get_policy(**strategy, **config["strategy"])
    else:
        raise NotImplementedError(score_type)
    return policy


def main():
    """Play interactively"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("-e", type=float, default=0.01)
    args = parser.parse_args()
    settings["strategy"]["epsilon"] = args.e

    for i_episode in range(1, args.n + 1):
        config = copy.deepcopy(settings)
        g = rust_tetris.Game()

        # Sparse hole pattern

        config_board = config["board"]
        match config_board["type"]:
            case 'empty':
                pass
            case "sparse-hole":
                board = np.zeros((22, 10))
                board[10:] = 1  # first 2 + 8 lines empty
                n_hole_per_line = config_board["n_hole_per_line"]
                board[
                    np.arange(0, 22)[:, None].repeat(n_hole_per_line, axis=1),
                    np.random.randint(0, 10, (22, n_hole_per_line)),
                ] = 0
                g.board = board.astype(int).tolist()
            case _:
                raise KeyError(_)

        states = [g.copy()]
        actions = []
        render_every = config["render_every"]
        max_history = config["max_history"]
        policy = get_policy(config)
        while not g.is_game_over:
            # print(f"Move #{len(states):0>4} (score: {g.score: >10})")
            if len(states) >= max_history:
                config["strategy"]["epsilon"] = 0.3
                policy = get_policy(config)

            i_action, action_cluster = policy(g)
            actions.append((g.piece[0], int(i_action)))
            for action in action_cluster:
                if action == 'unknown':
                    g.soft_drop()
                    pass
                elif action == 'hard_drop':
                    g.hard_drop()
                elif action == 'left':
                    g.left()
                elif action == 'right':
                    g.right()
                elif action == 'soft_drop':
                    g.soft_drop()
                elif action == 'rotate':
                    g.rotate_c()
                elif action == 'swap':
                    g.swap()
                else:
                    raise KeyError(action)
                if render_every and len(states) % render_every == 0:
                    _ = get_render_func(g)()(1)
            states.append(g.copy())
        print(f"[{i_episode:0>4}] Game over after {len(states):0>4} moves " +
              f"(lines: {humanfriendly.format_number(g.line_clears): >7} " +
              f"score: {humanfriendly.format_number(g.score): >10}).")
        logdir = (
            config["log_dir"] /
            config["strategy_name"] /
            f"eps{settings['strategy']['epsilon']}_max{config['max_history']}"
        )
        logdir.mkdir(exist_ok=True, parents=True)
        salt = hex(int(random.random() * 2 ** 32))[2:]
        (logdir / f"log_{int(time.time())}_{salt}.json").write_text(
            json.dumps({
                "actions": actions,
                "states": [game.game2serialize(state) for state in states],
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
