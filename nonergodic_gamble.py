"""Try to average out trajectory"""
import math
import random

import matplotlib.pyplot as plt


def fair_coin_toss() -> bool:
    """A fair coin"""
    return random.random() > 0.5


def trajectory(init: float, length: int, coin_toss):
    """Trajectory by given coin"""
    s = init
    ss = [s]
    for _ in range(length):
        if coin_toss():
            s *= 1.5
        else:
            s *= 0.6
        ss.append(s)
    return ss


def main():
    """:)"""
    n_trial = 10_000
    trajectory_length = 1_000
    init_state = 100

    print()
    print("Example of non-ergodic markov model reward")
    print("=" * 42)
    print("Coin toss where")
    print("     win  = 1.5")
    print("     loss = 0.6")
    print(f"Initial state:                {init_state:.2e}")
    print(f"Trajectory length:            {trajectory_length}")
    print(f"Expectation after trajectory: {init_state*1.05**trajectory_length:.2e}")
    print("=" * 42)
    print(f"Averages over the number of {n_trial} trials:")
    trajectories = []
    for i in range(10):
        _trajectories = [trajectory(init_state, 1000, fair_coin_toss)
                         for _ in range(trajectory_length)]
        last_states = [t[-1] for t in _trajectories]
        print(f"{i+1: >2}: {sum(last_states) / len(last_states):10.5f}")
        trajectories.extend(_trajectories)

    print("Much smaller than the expectation! @_@")
    plt.title("Coin toss")
    plt.plot([0, 1000], [math.log(init_state), math.log(init_state) + 1000 * math.log(1.05)],
             linestyle='dotted', label="expectation")
    for t in trajectories:
        plt.plot(range(1001), [math.log(s) for s in t], alpha=0.5, linewidth=0.5)
    plt.legend()
    plt.show()

main()
