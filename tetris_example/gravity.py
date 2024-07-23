"""Collection of gravity implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Optional

from tetris.engine import Gravity
from tetris.types import Move, MoveDelta, MoveKind

if TYPE_CHECKING:
    from tetris import BaseGame

SECOND: Final[int] = 1_000_000_000  # in nanoseconds


class ManualGravity(Gravity):
    """NES gravity without lock delay, typically played without hard drops.

    Notes
    -----
    See <https://tetris.wiki/Tetris_(NES,_Nintendo)>.
    """
    rule_overrides = {"can_hard_drop": True}
    FRAMES_PER_SEC = 60.0988

    def __init__(self, game: BaseGame, time_step = 0):
        super().__init__(game)

        self.last_drop = time_step
        self.now = time_step

    def tick(self):
        """pass time"""
        self.now += 1 / self.FRAMES_PER_SEC

    def calculate(self, delta: Optional[MoveDelta] = None) -> None:  # noqa: D102
        piece = self.game.piece

        # for i in NES_GRAV_FRAMES:  # set gravity based on level
        #     if level >= i:
        #         drop_delay = NES_GRAV_FRAMES[i] / self.FRAMES_PER_SEC
        #         break
        drop_delay = 3 / self.FRAMES_PER_SEC

        now = self.now

        since_drop = now - self.last_drop
        if since_drop >= drop_delay:
            if self.game.rs.overlaps(minos=piece.minos, px=piece.x + 1, py=piece.y):
                # hard drop if there is a piece below
                self.game.push(Move(kind=MoveKind.HARD_DROP, auto=True))
            else:
                self.game.push(
                    Move(
                        kind=MoveKind.SOFT_DROP,
                        x=int(since_drop / drop_delay),
                        auto=True,
                    )
                )
            self.last_drop = now
