import tetris
from . import board



def main():
    game = board.get_random_state(3)

    while game.status == tetris.PlayingStatus.PLAYING:
        game.tick()

        action = board.select_action(game)
        game = board.get_next_game(game, action, None)        


if __name__ == '__main__':
    main()
