import itertools
import random
import resource
from multiprocessing.pool import ThreadPool
from typing import Callable, List, Literal, Tuple

import humanfriendly
import numpy as np
import rust_tetris

from . import board, heuristic
from .uct_base import UCTNode

State = Tuple[bool, rust_tetris.Game]


# MODE = 'random'
MODE = 'greedy'



def play(state: State, action: int) -> State:
    ready_to_drop, game = state
    game = game.copy()
    if ready_to_drop:
        if MODE == 'random':
            game = game.copy()
            rest = game.queue
            i = board.N_QUEUE + action
            rest[board.N_QUEUE], rest[i] = rest[i], rest[board.N_QUEUE]
            game.queue = rest
            game.hard_drop()
            return (False, game)
        elif MODE == 'greedy':
            game.hard_drop()
            return (False, game)
        else:
            raise KeyError(f"Unknown mode {MODE}")

    r, t = heuristic.game2actions(game)[action]
    for _ in range(r):
        game.rotate_c()
    for _ in range(0, t):
        game.right()
    for _ in range(0, -t):
        game.left()
    return (True, game)


def turn(state: State) -> Literal[-1] | 0 | 1:
    ready_to_drop, _ = state
    return 0 if ready_to_drop else 1


def report_statistics(root: UCTNode,
                      game2score: Callable[[rust_tetris.Game], float] | None = None,
) -> None:
    game2score = game2score or heuristic.game2score_vanilla

    x = root
    for i in itertools.count():
        if x is None:
            break

        n_visit = x.child_number_visits.sum()
        n_visit_all = x.parent.number_visits - 1
        n_visit_ratio = n_visit / int(n_visit_all) if n_visit_all else 0
        # visit_ratio = n_visit / n_visit_all if n_visit_all else 0.0
        print(f"depth: {i: >3} " +
              f"line_clears: {x.inner[1].line_clears: >3} " +
              f"score: {game2score(x.inner[1]): >8.3f} " +
              f"visited: {n_visit: >5} / {int(n_visit_all): >5} ({n_visit_ratio:5.1%})")
        x = x.children.get(x.best_child())


def main() -> None:
    n_search = 3000
    ucb_constant = 140
    game = rust_tetris.Game()
    policy = get_policy(n_search, ucb_constant, 0.)
    eval_func = policy.__closure__[2].cell_contents
    root, i = UCTNode.uct_search((False, game), n_search, eval_func, play, turn)
    report_statistics(root)


def get_policy(n_search: int,
               ucb_constant: float,
               epsilon: float,
               n_thread: int,
               game2score: Callable[[rust_tetris.Game], float] | None = None,
):
    """ε-greedy UCT search.

    UCB Constant represents how pessimistic we will be.
    The lower the deeper. The higher the wider.

    Roughly speaking, averaged reward deviation should be good.
    
    Technically speaking, a value σ that the reward becomes sub-σ gaussian is optimal.
    Practically, the algorithm is not too sensitive to this value - just increase `n_search`!
    """
    game2score = game2score or heuristic.game2score_vanilla

    def evaluate_uniform(state):
        ready_to_drop, game = state
        actions = heuristic.game2actions(game)
        score = game2score(game) / ucb_constant

        if ready_to_drop:  # next player is random player
            # if N_QUEUE == 3
            # oooooooxxxxxxx
            # *ooo^
            #     ooo
            #   3 choices
            #
            # oooooooxxxxxxx
            #   *ooo^
            #       o
            #   1 choice
            #
            # oooooooxxxxxxx
            #      *oxx^
            #          xxxxx
            #   5 choice
            #
            # oooooooxxxxxxx
            #  *
            #
            # if N_QUEUE == 5
            # oooooooxxxxxxx
            #   ^
            if MODE == 'greedy':
                n_case = 1
            elif MODE == 'random':
                n_case = (len(game.queue) - board.N_QUEUE) % 7
                if n_case == 0:
                    n_case = 7
            else:
                raise NotImplementedError("Should be unreachable")
            child_prior = np.ones(n_case)
        else:  # player
            child_prior = 1e5 * np.ones(len(actions)) + 100 * np.random.random(len(actions))
        return child_prior, score

    evaluate = evaluate_uniform
    root = None
    thread_pool = ThreadPool(16) if n_thread else None

    def select_action_cluster(game: rust_tetris.Game) -> Tuple[int, List[str]]:
        nonlocal root

        if root and root.turn == 0:
            old_queue = root.inner[1].queue[1:]
            new_queue = game.queue
            i_action = old_queue[board.N_QUEUE - 1:].index(new_queue[board.N_QUEUE - 1])
            if MODE == 'greedy' and i_action != 0:
                root = None  # reset!
            else:
                root_new = root.safe_get_child(i_action, evaluate, play, turn)
                evaluate(root.inner)
                # Replace old root with new root of chosen action,
                # and update dummy
                dummy = root.parent
                dummy.child_total_value  [None] = root.child_total_value  [i_action]
                dummy.child_number_visits[None] = root.child_number_visits[i_action]
                root = root_new 
                root.parent = dummy
                root.move = None

        # sanity check
        if root:
            _, _game = root.inner
            if game.queue[:board.N_QUEUE] != _game.queue[:board.N_QUEUE]:
                if MODE == 'greedy':
                    pass
                else:
                    import code
                    code.interact(local={**globals(), **locals()})
            if root.inner[1].playfield != game.playfield:
                if MODE == 'greedy':
                    root = None
                else:
                    import code
                    code.interact(local={**globals(), **locals()})
                    _ = board.get_render_func(game)()()
                    _ = board.get_render_func(root.inner[1])()()
                    raise RuntimeError("Random action is not working correctly")

        actions = heuristic.game2actions(game)

        if epsilon > random.random():
            # Random action
            i_action, _ = random.choice(list(enumerate(actions)))
            root, _ = UCTNode.uct_search((False, game), 1, evaluate, play, turn, thread_pool)
        else:
            if root is None:
                root, i_action = UCTNode.uct_search((False, game), n_search, evaluate, play, turn, thread_pool)
            else:
                root, i_action = UCTNode.uct_search_continue_parallel(root, n_search - root.number_visits, evaluate, play, turn, thread_pool)
        report_statistics(root, game2score)

        # Cache tree for later use
        assert root.turn == 1
        root_new = root.safe_get_child(i_action, evaluate, play, turn)
        # Replace old root with new root of chosen action,
        # and update dummy
        dummy = root.parent
        dummy.child_total_value  [None] = root.child_total_value  [i_action]
        dummy.child_number_visits[None] = root.child_number_visits[i_action]
        root = root_new
        root.parent = dummy
        root.move = None

        ret = heuristic.action2commands(*actions[i_action])

        # Report
        n_bytes = humanfriendly.format_size(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(f"Consumed {n_bytes} memory")
        # Report done
        return i_action, ret

    return select_action_cluster


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    # sample_statistics()
    main()
