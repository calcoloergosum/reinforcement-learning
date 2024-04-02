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
import scipy.special

State = tuple[int, int]
Action = int

ACTION_LEFT  = 0
ACTION_RIGHT = 1
ACTION_UP    = 2
ACTION_DOWN  = 3

Action = int
RNG = np.random.Generator
Reward = float


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
    
    def true_state2value(self, discount: float):
        assert 0 < discount < 1
        neighborhood_mat = np.zeros((self.grid_size,) * 4, dtype=bool)
        for x in range(-self.n, self.n + 1):
            for y in range(-self.n, self.n):
                neighborhood_mat[y, x, y+1, x] = 1.
            for y in range(-self.n + 1, self.n + 1):
                neighborhood_mat[y, x, y-1, x] = 1.
        for y in range(-self.n, self.n + 1):
            for x in range(-self.n, self.n):
                neighborhood_mat[y, x, y, x + 1] = 1.
            for x in range(-self.n + 1, self.n + 1):
                neighborhood_mat[y, x, y, x - 1] = 1.
        neighborhood_mat[self.target]              = 0.
        neighborhood_mat[self.target][self.target] = 1.

        prob_kernel = neighborhood_mat / neighborhood_mat.sum(axis=(2, 3))
        prob_kernel = prob_kernel.reshape((self.grid_size ** 2,) * 2)
        immediate_reward = np.zeros((self.grid_size,) * 2, dtype=float)
        immediate_reward[self.target[1] - 1, self.target[0]] = 1/3
        immediate_reward[self.target[1], self.target[0] - 1] = 1/3
        return (
            np.linalg.inv(
                np.identity(self.grid_size ** 2) - discount * prob_kernel
            ) @ immediate_reward.reshape(-1)
        ).reshape(self.grid_size, self.grid_size)

action2diff2n = [{(0, 1): 0, (1, 0): 0, (-1, 0): 0, (0, -1): 0} for _ in range(4)]


def test_mdp():
    mdp = RandomWalkGridMDP(10, (10, 10), False)
    assert len(mdp.get_actions((0, 0))) == 4
    assert len(mdp.get_actions((-10, -10))) == 2
    assert len(mdp.get_actions((-10, 0))) == 3
    assert len(mdp.get_actions((10, 0))) == 3

    rng = np.random.default_rng(1)
    actions = []
    for _ in range(10000):
        (x, y), _, _ = mdp.move((10, 0), ACTION_LEFT, rng)
        act = vec2action((x - 10, y))
        actions.append(act)
    assert dict(Counter(actions)) == {ACTION_LEFT: 92, ACTION_UP: 7, ACTION_DOWN:1}


def td0(samples: List[MRPSample], state2value, discount: float) -> None:
    """Temporal Difference; exploit markov property to find solution of bellman equation"""
    assert 0 < discount < 1
    for i_sample, (init_state, asr_list) in enumerate(samples, start=1):
        lr = 1 / i_sample
        prev = init_state
        for _next, reward in asr_list:
            term = reward + discount * state2value[_next]
            state2value[prev] += lr * (term - state2value[prev])
            prev = _next
    return state2value


def td_multi_step(samples: List[MRPSample], n_step: int, state2value, discount: float) -> None:
    """Multi-step temporal difference"""
    assert 0 < discount < 1
    for i_sample, (init_state, asr_list) in enumerate(samples, start=1):
        lr = 1 / i_sample
        prev = init_state
        for t in range(1, len(asr_list)):
            term = 0
            for i_step, (s, r) in enumerate(asr_list[t:t+n_step]):
                term += r * (discount ** i_step)
            term += discount ** (i_step + 1) * state2value[s]
            state2value[prev] += lr * (term  - state2value[prev])
            prev, _ = asr_list[t]
    return state2value


def first_visit_monte_carlo(episodes: List[MRPSample], state2value, discount) -> None:
    """First-visit Monte Carlo"""
    s2t_first = -1 * np.ones(state2value.shape, dtype=np.uint32)
    state2n_visit = np.zeros(state2value.shape, dtype=np.uint32)
    for i_update, (init_state, asr_list) in enumerate(episodes, start=1):
        # record first appearance
        s2t_first.fill(-1)
        s2t_first[init_state] = 0
        for t, (s, _) in zip(range(len(asr_list), 0, -1), asr_list[::-1]):
            s2t_first[s] = t
        # record first appearance done

        reward_sum = 0.
        states, rewards = list(zip(*([(init_state, 0)] + asr_list)))
        for t in range(len(asr_list) - 1, 0, -1):
            s = states[t]
            reward_sum = rewards[t + 1] + discount * reward_sum
            if s2t_first[s] == t:
                state2n_visit[s] += 1
                state2value[s]   += reward_sum

    state2n_visit[state2n_visit == 0] = 1
    return state2value / state2n_visit


def every_visit_monte_carlo(episodes: List[MRPSample], state2value, state2n_visit, discount) -> None:
    """Every-visit Monte Carlo"""
    for init_state, asr_list in episodes:
        reward_sum = 0.
        states, rewards = list(zip(*([(init_state, 0)] + asr_list)))
        for t in range(len(asr_list) - 1, 0, -1):
            s = states[t]
            reward_sum = rewards[t + 1] + discount * reward_sum
            state2n_visit[s] += 1
            state2value[s]   += reward_sum

    state2n_visit[state2n_visit == 0] = 1  # to suppress warning
    return state2value / state2n_visit


def td_lambda(samples: List[MRPSample], state2value, discount, _lambda) -> None:
    assert 0 < discount < 1
    assert 0 <= _lambda <= 1
    eligability_trace = np.zeros_like(state2value)

    for i_sample, (init_state, asr_list) in enumerate(samples, start=1):
        lr = 1 / i_sample
        prev = init_state
        for _next, reward in asr_list:
            term = reward + discount * state2value[_next] - state2value[prev]
            eligability_trace *= _lambda * discount
            eligability_trace[prev] = 1
            state2value += lr * term * eligability_trace
            prev = _next
    return state2value
    

def main():
    """Simple demonstration of TD(0)"""
    # Setup
    # Raise on anything other than underflow
    np.seterr(all='raise')
    np.seterr(under='ignore')
    np.set_printoptions(precision=2, suppress=True)
    # setup done

    is_episodic = True
    N_TRIAL = 100
    N_EPISODE = 100
    MAX_EPISODE_LENGTH = 1000
    DISCOUNT = 0.9

    RADIUS = 3
    mdp = RandomWalkGridMDP(RADIUS, (RADIUS, RADIUS), is_episodic=is_episodic)
    rng = np.random.default_rng()

    def _policy(state: State, rng: RNG) -> Action:
        """Complete random policy"""
        opts = mdp.get_actions(state)
        return rng.choice(opts)
    
    def _show(state2value, title):
        state2value = np.roll(np.roll(state2value, RADIUS, 0), RADIUS, 1)
        print(state2value)
        plt.title(title)
        seaborn.heatmap(np.log(.01 + state2value))
        plt.show()

    def loss(_state2value):
        return scipy.special.kl_div(_state2value.flatten(), state2value_gt.flatten()).sum()

    # GT
    state2value_gt = mdp.true_state2value(DISCOUNT)
    # _show(state2value_gt,
    #     "Value function of " +"episodic random walk on grid\n(Ground truth)"
    #     if is_episodic else
    #     "Value function of non-episodic random walk on grid\n(Ground truth)"
    # )

    # Various estimation methods
    # Sample
    method2errors = {
        'td0': [],
        # **{f'td_multi_step_{n_step:0>2}': [] for n_step in range(2, 10)},
        **{f'td_lambda_{_lambda:3.1f}': [] for _lambda in np.linspace(0.01, 0.99, 11)},
        'first_visit_monte_carlo': [],
    }
    for i_trial in range(N_TRIAL):
        samples = [mdp.sample((0, 0), _policy, rng, MAX_EPISODE_LENGTH) for _ in range(N_EPISODE)]
        # print(f"[{i_trial:0>5}] Average episode length:",
        #       sum(len(s[1]) + 1 for s in samples) / len(samples))

        state2value   = np.empty((mdp.grid_size, mdp.grid_size), dtype=float)

        # TD 0
        state2value.fill(0)
        _state2value = td0(samples, state2value, DISCOUNT)
        method2errors['td0'].append(loss(_state2value))
        # _show(
        #     td0(samples, state2value, DISCOUNT),
        #     "Value function of " +"episodic random walk on grid\n(Temporal Difference)")
        #     if is_episodic else
        #     "Value function of non-episodic random walk on grid\n(Temporal Difference)")
        # )

        # TD n-step
        # for n_step in range(2, 10):
        #     state2value.fill(0)
        #     _state2value = td_multi_step(samples, n_step, state2value, DISCOUNT)
        #     method2errors[f'td_multi_step_{n_step:0>2}'].append(loss(_state2value))
            # _show(
            #     td_multi_step(samples, 100, state2value, DISCOUNT),
            #     "Value function of " +"episodic random walk on grid\n(Temporal Difference; 3-step)"
            #     if is_episodic else
            #     "Value function of non-episodic random walk on grid\n(Temporal Difference; 3-step)"
            # )
            
        # TD(λ)
        for _lambda in np.linspace(0.01, 0.99, 11):
            state2value.fill(0)
            _state2value = td_lambda(samples, state2value, DISCOUNT, _lambda)
            method2errors[f'td_lambda_{_lambda:3.1f}'].append(loss(_state2value))

        # Monte Carlo
        state2value.fill(0)
        _state2value = first_visit_monte_carlo(samples, state2value, DISCOUNT)
        method2errors['first_visit_monte_carlo'].append(loss(_state2value))
        # _show(
        #     first_visit_monte_carlo(samples, state2value, state2n_visit, DISCOUNT),
        #     "Value function of " +"episodic random walk on grid\n(Monte Carlo)"
        #     if is_episodic else
        #     "Value function of non-episodic random walk on grid\n(Monte Carlo)"
        # )

    for method, errors in method2errors.items():
        error = np.mean(errors)
        std   = np.std(errors)
        print(f"{method: >30}: {error:.4f} ± {2 * std:.4f}")


if __name__ == '__main__':
    main()
