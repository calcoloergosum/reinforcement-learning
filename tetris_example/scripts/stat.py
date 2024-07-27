import pathlib

method2strat2epsilon2n_actions = {}
for p in pathlib.Path("logs").rglob("*.txt"):
    method, strat, epsilon = p.parent.name.split("_")
    n_actions = p.read_text().count("\n")
    (
        method2strat2epsilon2n_actions
        .setdefault(method, {})
        .setdefault(strat, {})
        .setdefault(float(epsilon), [])
    ).append(n_actions)

import matplotlib.pyplot as plt

for method, strat2epsilon2n_actions in method2strat2epsilon2n_actions.items():
    for strat, epsilon2n_actions in strat2epsilon2n_actions.items():
        plt.title(f"Trend in epsilon-greedy ({method}, {strat})")
        plt.xlabel("Epsilon")
        plt.ylabel("Number of cleared lines")
        print(method, strat, {e: len(x) for e, x in epsilon2n_actions.items()})
        epsilons, n_actions = list(zip(*sorted(epsilon2n_actions.items())))
        plt.boxplot(n_actions, tick_labels=epsilons)
        plt.show()
