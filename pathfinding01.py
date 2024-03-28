"""
Pathfinding by reinforcement learning.
Demonstration of TD(λ)

Problem setting
    - You are in the middle of `2n+1` by `2n+1` grid.
    - Goal is to reach an unknown fixed position {x0, y0} as soon as possible.
    - You can move up, down, left, right.
    - You cannot move out of the grid.
    - You have to get there before `10n` movement

Problem setting as MDP
    - State:
        - Starting coordinate is (0, 0).
        - Grid coordinates is {-n, ..., n} x {-n, ..., n}
    - Action:
        - Direction (Up, Down, Left, Right)
    - Next State:
        - Not on edge:
            - 85% intended direction
            - 15% wrong direction (5% each)
        - On edge:
            - 90% intended direction
            - 10% wrong direction (5% each)
        - On corner:
            - 95% intended direction
            - 5% wrong direction
"""
from collections import Counter
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn

State = tuple[int, int]
Action = int

ACTION_LEFT  = 0
ACTION_RIGHT = 1
ACTION_UP    = 2
ACTION_DOWN  = 3

Action = TypeAlias[int]
RNG = np.random.Generator
Reward = TypeAlias[float]


MRPSample = Tuple[State, List[Tuple[State, Reward]]]


def vec2action(vec: tuple[int, int]) -> Action:
    if vec == (-1, 0):
        return ACTION_LEFT
    if vec == (1, 0):
        return ACTION_RIGHT
    if vec == (0, 1):
        return ACTION_UP
    if vec == (0, -1):
        return ACTION_DOWN
    raise KeyError(vec)


class RandomWalkGridMDP:
    def __init__(self, radius: int, target: State, is_episodic: bool) -> None:
        assert - radius <= target[0] <= radius
        assert - radius <= target[1] <= radius
        self.n = radius
        self.target = target
        self.is_episodic = is_episodic

    @property
    def grid_size(self) -> int:
        return 2 * self.n + 1

    def get_actions(self, loc: State) -> List[Action]:
        x, y = loc
        opts = []
        if x != - self.n:
            opts.append(ACTION_LEFT)
        if x !=   self.n:
            opts.append(ACTION_RIGHT)
        if y != - self.n:
            opts.append(ACTION_DOWN)
        if y !=   self.n:
            opts.append(ACTION_UP)
        return opts

    def move(self, loc: State, act: Action, rng: RNG) -> Tuple[State, Reward, bool]:
        acts = self.get_actions(loc)
        assert act in acts, "Invalid option"

        reward = 0.
        should_continue = True
        val = rng.random()

        acts.remove(act)
        if val >= 0.05 * len(acts):
            next_loc = self._move(loc, act)
        else:
            _act = rng.choice(acts)
            next_loc = self._move(loc, _act)
        if next_loc == self.target:
            next_loc = self.target
            reward = 1.
            should_continue = not self.is_episodic
        return (next_loc, reward, should_continue)

    def _move(self, loc: State, act: Action) -> State:
        x, y = loc
        if act == ACTION_LEFT:
            return (x-1, y)
        if act == ACTION_RIGHT:
            return (x+1, y)
        if act == ACTION_UP:
            return (x, y + 1)
        if act == ACTION_DOWN:
            return (x, y - 1)
        raise NotImplementedError

    def sample(self,
               initial_state: State,
               policy: Callable[[State, RNG], Action],
               rng: RNG,
               trajectory_length: int,
    ) -> MRPSample:
        state = initial_state

        srs: List[Tuple[State, Reward]] = []
        for _ in range(trajectory_length):
            action = policy(state, rng)
            _state, reward, should_continue = self.move(state, action, rng)
            action2diff2n[action][(_state[0] - state[0], _state[1] - state[1])] += 1
            srs.append((_state, reward))
            if not should_continue:
                break
            state = _state
        return (initial_state, srs)

action2diff2n = [{(0, 1): 0, (1, 0): 0, (-1, 0): 0, (0, -1): 0} for _ in range(4)]


def test_mdp():
    mdp = RandomWalkGridMDP(10, (10, 10), False)
    assert len(mdp.get_actions((0, 0))) == 4
    assert len(mdp.get_actions((-10, -10))) == 2
    assert len(mdp.get_actions((-10, 0))) == 3
    assert len(mdp.get_actions((10, 0))) == 3

    rng = np.random.default_rng(1)
    actions = []
    for _ in range(100):
        (x, y), _, _ = mdp.move((10, 0), ACTION_LEFT, rng)
        act = vec2action((x - 10, y))
        actions.append(act)
    assert dict(Counter(actions)) == {ACTION_LEFT: 92, ACTION_UP: 7, ACTION_DOWN:1}


def td0(samples: List[MRPSample], state2value, discount: float) -> None:
    """Model free estimation"""
    assert 0 < discount < 1
    lr = 0.9  # learning rate
    for init_state, asr_list in samples:
        prev = init_state
        for _next, reward in asr_list:
            td                 = reward + discount * state2value[_next] - state2value[prev]
            state2value[prev] += lr * td
            prev = _next
    return state2value


def main():
    """Simple demonstration of TD(0)"""
    # Setup
    np.seterr(all='raise')
    np.set_printoptions(precision=2, suppress=True)
    # setup done

    is_episodic = False

    mdp = RandomWalkGridMDP(3, (3, 3), is_episodic=is_episodic)
    rng = np.random.default_rng(1)

    def _policy(state: State, rng: RNG) -> Action:
        """Complete random policy"""
        opts = mdp.get_actions(state)
        return rng.choice(opts)

    samples = []
    for _ in range(10000):
        sample = mdp.sample((0, 0), _policy, rng, 1000)
        samples += [sample]

    state2value = np.zeros((mdp.grid_size, mdp.grid_size), dtype=float)
    state2value = np.roll(np.roll(td0(samples, state2value, .9), 3, 0), 3, 1)

    print(state2value)
    if is_episodic:
        plt.title(    "Episodic random walk on grid")
    else:
        plt.title("Non-episodic random walk on grid")
    seaborn.heatmap(np.log(.01 + state2value))
    plt.show()


if __name__ == '__main__':
    main()
