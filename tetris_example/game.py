import tetris

from . import board
from .gravity import ManualGravity


def main():
    """Play interactively"""
    my_engine = tetris.EngineFactory(
        gravity=ManualGravity,
        queue=tetris.impl.queue.SevenBag,
        rotation_system=tetris.impl.rotation.SRS,
        scorer=tetris.impl.scorer.GuidelineScorer,
    )

    game = board.get_random_state(board.BOARD_HEIGHT // 2, my_engine)

    while game.status == tetris.PlayingStatus.PLAYING:
        action = board.select_action(game)
        _, game, _ = board.get_next_game(game, action, my_engine)


if __name__ == '__main__':
    main()
