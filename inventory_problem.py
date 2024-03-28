"""
**WORK IN PROGRESS**

Example 1.1 Inventory problem
Algorithms for Reinforcement Learning (Csaba Szepesvari)

Find best policy for inventory problem
Mr.Nishimatsu has a business.
- He has a storage of maximum size `M`. Never below 0.
- He has to pay `K` for storage per item per night.
- He earns `C` for each item sold.
  Number of items sold in a day is random i.i.d. (known)
- He order to restock the storage every night,
  when business is closed. Cost per item is 'X'.

At a night when he has `N` items in the storage,
How much should he order?
"""
import random
from typing import *
import numpy as np
import itertools


random.seed(0)

State = int  # = Inventory at night; >= 0
Action = int  # = Restock Order; >= 0
Reward = float  # = Profit
StateTransition = Callable[
    [Tuple[State, Action]],  # night, order
     Tuple[State, Reward]    # next_night, profit
]

STORAGE_COST_PER_ITEM: float = 100
COST_PER_ITEM: float = 100
COST_PER_ORDER: float = 3_000
REVENUE_PER_ITEM: float = 1_000
MAX_INVENTORY = 10

get_transition: StateTransition
def get_transition(avg: int, method: str) -> StateTransition:
    """Random order of uniform probability over [0,C]
    """
    if method == 'uniform':
        def get_random_order():
            return random.randint(0, 2 * avg)
    elif method == 'poisson':
        def get_random_order():
            return int(random.expovariate(1/avg))
    else:
        raise KeyError(method)

    def transition(s: State, a: Action) -> Tuple[State, Reward]:
        # NOTE: 0 <= inventory <= M
        s_morning = min(s + a, MAX_INVENTORY)  # <= M
        order_received = get_random_order()
        item_sold = min(order_received, s_morning)
        s_night = s_morning - item_sold  # next day; >= 0
        # = s_night = max(0, s_morning - item_sold)
        profit = (
            REVENUE_PER_ITEM * item_sold
            - COST_PER_ITEM * a
            - STORAGE_COST_PER_ITEM * s_night
            - (COST_PER_ORDER if a > 0 else 0)
        )
        return s_night, profit
    return transition


def test_get_transition():
    t = get_transition(5, 'poisson')
    random.seed(0)
    s, r = t(10, 0)
    assert r > 0
    random.seed(0)
    s, r = t(0, 10)
    assert r > 0


Policy = Callable[[State], Action]  # Greedy

ValueEvaluator = Callable[
    [State, Policy, StateTransition],
    Reward,
]

ImmediateRewardFunction = Callable[
    [State, Action],
    Reward
]
TransitionKernel = Dict[Tuple[State, Action, State], float]

expectation: ValueEvaluator
def get_q(
    poli: Policy,
    t: StateTransition,
    n_sample: int = 1000,
    n_step_per_sample: int = 100,
) -> Tuple[ImmediateRewardFunction, TransitionKernel]:
    state_action2reward_sum = np.zeros((MAX_INVENTORY + 1, MAX_INVENTORY + 1), dtype=np.float32)
    state_action2n          = np.zeros((MAX_INVENTORY + 1, MAX_INVENTORY + 1), dtype=np.uint32)
    state_action_state      = np.zeros((MAX_INVENTORY + 1, MAX_INVENTORY + 1, MAX_INVENTORY + 1), dtype=np.uint32)

    for i in itertools.count():
        # Sample (A0, R0), (X1, R1), ..., (X100, R100)
        xrs = []
        actions = []
        s_init = random.randint(0, MAX_INVENTORY)
        s = s_init
        for _ in range(n_step_per_sample):
            act = random.randint(0, MAX_INVENTORY)
            # act = poli(s) if random.random() > 0.1 else random.randint(0, MAX_INVENTORY)
            actions.append(act)
            s_next, reward = t(s, act)
            xrs.append((s_next, reward))
            s = s_next

        xs, rs = list(zip(*xrs))

        for x_bef, x_aft, a, r in zip([s_init] + list(xs), xs, actions, rs):
            state_action2reward_sum[x_bef, a] += r
            state_action2n         [x_bef, a] += 1
            state_action_state     [x_bef, a, x_aft] += 1

        # if i > n_sample and (state_action2n > 0).all():
        # all state has to visited at least once!
        if i > n_sample:
            break

    return (
        state_action2reward_sum / state_action2n,
        state_action_state / state_action2n[..., None],
    )


def main():
    transition = get_transition(5, method='poisson')

    # N = 13
    # def policy(s: State) -> Action:
    #     return max(0, N - s)

    # Policy
    # Initial policy as uniform distribution
    state2action_p = np.ones((MAX_INVENTORY + 1, MAX_INVENTORY + 1))
    state2action_p[np.arange(MAX_INVENTORY + 1), MAX_INVENTORY - np.arange(MAX_INVENTORY + 1)] = 10
    state2action_p /= state2action_p.sum(axis=1)[:, None]

    def policy(s: State) -> Action:
        return int(np.random.choice(range(MAX_INVENTORY + 1), size=1, p=state2action_p[s]).squeeze(0))
    # s_init = MAX_INVENTORY // 2
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    discount = 0.9
    while True:
        state_action2reward_immediate, state_action_state2p = get_q(
            policy, transition, n_sample = 10000, n_step_per_sample=10,
        )

        # Find Q* by repeated Bellman operator
        state_action2reward = state_action2reward_immediate.copy()
        for _ in range(10):
            state_action2reward = (
                state_action2reward_immediate +
                discount * (
                    state_action_state2p *
                    state_action2reward.max(axis=1)[None, None, :]
                ).sum(axis=-1)
            )

        # update policy
        best_actions = state_action2reward.argmax(axis=-1)
        _state2action_p = np.zeros_like(state2action_p)
        _state2action_p[np.arange(MAX_INVENTORY + 1), best_actions] = 1
        # state2action_p = .9 * state2action_p + .1 * _state2action_p
        state2action_p = .5 * state2action_p + .5 * _state2action_p

        # log
        print(f"Current Policy:\n{state2action_p}")
        values = [f"{v:7.2f}" for v in (state2action_p * state_action2reward).sum(axis=-1)]
        print(f"Current Values:\n{' '.join(values)}")


if __name__ == '__main__':
    main()
