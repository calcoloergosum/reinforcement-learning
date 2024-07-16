from .gravity import NESGravity
import tetris.impl.queue
import tetris.impl.rotation
import tetris.impl.scorer
from tetris.engine import EngineFactory
import pathlib
from typing import Literal, List, Tuple, Iterator
import itertools
import json
import numpy as np
import tetris
import torch
import torchvision
import random
from .game import BOARD_WIDTH, BOARD_HEIGHT, N_QUEUE, get_render_func
from torch.utils.tensorboard import SummaryWriter

# Modeling tetris agent as Markov model M = (State, Action, Probability)
# State := (Field, Queue):
#     Field is boolean of W x H = 10 x 20
#     Queue is Pieces of size `N_q`.
#     Piece is one hot vector of size 7; one of IJLOSTZ.
#
#     field is current field = 200 bit
#     queue is 7 x `N_q` bit
#
# Action := left | right | hard_drop | soft_drop | rotate
#
# Probability := S x A -> R+ such that:
#     With 1% chance, the action is ignored
#     With 1% chance, the action is doubled
#     With 98% chance, the action is conducted as intended

State = tetris.BaseGame
ACTIONS = ["left", "right", "hard_drop", "soft_drop", "rotate"]
Action = Literal["left", "right", "hard_drop", "soft_drop", "rotate"]
RANDOM_ACTION_EPSILON_INIT = 0.1
RANDOM_ACTION_EPSILON_DECAY = 1.
# RANDOM_ACTION_EPSILON_DECAY = 0.999
RENDER = True
LOGGER = SummaryWriter(f'runs/{N_QUEUE}')
SAVE_EVERY = 100
RENDER_EVERY = 1000
LR = 0.005

class TetrisNet(torch.nn.Module):
    CONV_CH = 8
    ROI_HEIGHT = 4
    QUEUE_CH = 32

    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(1, self.CONV_CH, (self.ROI_HEIGHT, BOARD_WIDTH))
        self.queuemap = torchvision.ops.MLP(N_QUEUE * 7, [self.QUEUE_CH])
        self.head = torchvision.ops.MLP(self.QUEUE_CH + self.CONV_CH * BOARD_HEIGHT, [32, 32, 1])

    def __call__(self, x):
        return self.head(
            torch.concat([
                self.conv(
                    x[..., :- N_QUEUE * 7]
                    .reshape(-1, 1, BOARD_HEIGHT + 3, BOARD_WIDTH))
                    .reshape(-1, self.CONV_CH * BOARD_HEIGHT),
                    self.queuemap(x[..., -N_QUEUE * 7:]),
                ],
                dim=1,
            )
        )


def get_discount(i: int = 0) -> Iterator[float]:
    VALUE_START = 0.4
    VALUE_END   = 0.95
    ITER_MAX = 100000

    discount = VALUE_START + (VALUE_END - VALUE_START) * i / ITER_MAX
    return min(0.99, max(0, discount))


def get_state2value_full():
    # mlp = torchvision.ops.MLP(20 * 10 + N_QUEUE * 7, [256, 32, 1],)
    mlp = TetrisNet()
    if (mlp_path := pathlib.Path(f"runs/{N_QUEUE}/mlp_full.pt")).exists():
        print("Loaded checkpoint! (model)")
        mlp.load_state_dict(torch.load(mlp_path))
    else:
        print("Could not find checkpoint! Making new ...")
    return mlp, lambda x: mlp(x[None])[0, 0]


def get_feature_heuristic2value():
    mlp = torchvision.ops.MLP(7, [14, 16, 8, 4, 1], norm_layer=None)
    if (mlp_path := pathlib.Path(f"runs/{N_QUEUE}/mlp_heuristic.pt")).exists():
        print("Loaded checkpoint! (optimizer)")
        mlp.load_state_dict(torch.load(mlp_path))
    else:
        print("Could not find checkpoint! Making new ...")
    return mlp


def get_optimizer_heuristic(params):
    optimizer = torch.optim.SGD(params, lr=LR)
    # optimizer = torch.optim.SGD(params, lr=LR, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    # optimizer = torch.optim.Adam(params)
    if (optimizer_path := pathlib.Path(f"runs/{N_QUEUE}/optim_heuristic.pt")).exists():
        optimizer.load_state_dict(torch.load(optimizer_path))
    else:
        print("Could not find checkpoint! Making new ...")
    for g in optimizer.param_groups:
        g['lr'] = LR
    return optimizer


def get_optimizer_full(params):
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    # optimizer = torch.optim.Adam(params)
    if (optimizer_path := pathlib.Path(f"runs/{N_QUEUE}/optim_full.pt")).exists():
        optimizer.load_state_dict(torch.load(optimizer_path))
    else:
        print("Could not find checkpoint! Making new ...")
    for g in optimizer.param_groups:
        g['lr'] = LR
    return optimizer


def piece2tensor(p) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(p.value) - 1, num_classes=7)


def game2tensor(g: tetris.BaseGame) -> torch.Tensor:
    board = torch.tensor(g.board[17:].data)
    field = torch.tensor(g.get_playfield(buffer_lines=3).data)
    field[field == 8] = 0  # Remove ghost
    field_mask = field - board > 0
    board = (field > 0).type(torch.float32)
    board[field_mask] *= -1
    queue = [piece2tensor(p) for p in g.queue[:N_QUEUE]]
    vec = torch.concat((board, *queue)).type(torch.float32)
    return vec


def copy_game(g: tetris.BaseGame):
    _g = tetris.BaseGame(my_engine, board=g.board.copy(), board_size=(20, 10))
    _g.queue._pieces = g.queue._pieces.copy()
    _g.scorer.goal = g.scorer.goal
    _g.scorer.level = g.scorer.level
    _g.scorer.score = g.scorer.score
    _g.scorer.line_clears = g.scorer.line_clears
    _g.gravity.last_drop = g.gravity.last_drop
    _g.gravity.now = g.gravity.now
    _g.piece = g.piece.__class__(
        type=g.piece.type,
        x=g.piece.x,
        y=g.piece.y,
        r=g.piece.r,
        minos=g.piece.minos,
    )
    # _g.board._data = g.board._data
    # print(_g.playfield)
    return _g


def get_next_games(g: tetris.BaseGame, actions=ACTIONS) -> Tuple[List[Action], List[tetris.BaseGame]]:
    ret_g = []
    ret_a = []
    for a in actions:
        if (_g := get_next_game(g, a)) is not None:
            ret_g += [_g]
            ret_a += [a]
    return ret_a, ret_g


def get_next_game(g: tetris.BaseGame, a: Action) -> tetris.BaseGame | None:
    _g = copy_game(g)
    try:
        getattr(_g, a)()
        _g.gravity.tick()
        _g.tick()
    except AssertionError:  # cannot rotate
        return None
    return _g
    


def column2height(c):
    is_any_nonzero, h = max(zip(c != 0, itertools.count()))
    return (h + 1) if is_any_nonzero else 0


def game2feature_heuristic(g: tetris.BaseGame):
    """Heuristic measures useful at the beginning of the learning"""
    board = np.array(g.board[BOARD_HEIGHT:])
    columns = board.T[:, ::-1]
    column_heights = [column2height(col) for col in columns]

    # Number of holes
    n_hole = 0
    for col, h in zip(columns, column_heights):
        if h == 0:
            continue
        n_hole += sum(col[:h] == 0)
    # Number of holes done

    # Bumpyness
    diff_total = sum(abs(h1 - h2)
                     for h1, h2 in zip(column_heights, column_heights[1:]))
    # diff_max = max(abs(h1 - h2) for h1, h2 in zip(column_heights, column_heights[1:]))
    # Bumpyness done

    # Height
    max_height = max(column_heights)
    # min_height = min(column_heights)
    # Height done

    return (
        *scorer2feature_heuristic(g.scorer),
        n_hole, diff_total, max_height,
    )


def scorer2feature_heuristic(s):
    return (s.level, s.line_clears, s.back_to_back, s.combo)


my_engine = EngineFactory(
    gravity=NESGravity,
    queue=tetris.impl.queue.SevenBag,
    rotation_system=tetris.impl.rotation.SRS,
    scorer=tetris.impl.scorer.GuidelineScorer,
)


def immediate_reward(a: Action, g_bef: tetris.BaseGame, g_aft: tetris.BaseGame) -> float:
    i_episode = 0
    reward = 0.

    # Reward some actions
    match a:
        case 'rotate':
            reward -= .1
        case 'hard_drop':
            pass
        case 'left':
            pass
        case 'right':
            pass
        case 'soft_drop':
            reward -= 0.1
        case _:
            raise RuntimeError("Not Implemented!")
    # Reward some actions done

    # Reward surviving
    reward += 0.1
    # Reward surviving done

    if i_episode < 1000000:
        *_, n_hole_aft, diff_total_aft, max_height_aft = game2feature_heuristic(g_aft)
        *_, n_hole_bef, diff_total_bef, max_height_bef = game2feature_heuristic(g_bef)
        reward += 10 * (g_aft.scorer.line_clears - g_bef.scorer.line_clears)
        reward += .5 * (n_hole_bef - n_hole_aft)
        # reward += diff_total_bef - diff_total_aft
        reward += max_height_bef - 2 * max_height_aft
        if n_hole_aft - n_hole_bef == 0:
            pass
        elif diff_total_aft - diff_total_bef == 0:
            pass
        else:
            pass
            # print(n_hole_bef - n_hole_aft)
            # print(diff_total_bef - diff_total_aft)
            # print(max_height_bef - max_height_aft)
    else:
        # reward += g_aft.score - g_bef.score
        pass
    # reward += g_aft.score - g_bef.score
    return reward


def main():
    state2value_module, state2value = get_state2value_full()
    optim_full = get_optimizer_full(state2value_module.parameters())

    def game2value(g: tetris.BaseGame):
        return state2value(game2tensor(g))
    # feature_heuristic2value = get_feature_heuristic2value()
    # optim_heuristic = get_optimizer_heuristic(feature_heuristic2value.parameters())
    # def game2value(g):
    #     return feature_heuristic2value(
    #         torch.tensor(
    #             game2feature_heuristic(g),
    #             dtype=torch.float32,
    #         )[None]
    #     )[0]

    criterion = torch.nn.L1Loss()

    random.seed(0)
    i_episode = 0
    i_update = 0
    loss_per_update = 0

    # Load state
    if (txtpath := pathlib.Path(f"runs/{N_QUEUE}/progress.json")).exists():
        d = json.loads(txtpath.read_text())
        i_episode = d.get("i_episode", 0)
        i_update = d.get("i_update", 0)
        loss_per_update = d.get("loss_per_update", 0)
        a, b, c = d["random_state"]
        random.setstate((a, tuple(b), c))
    # Load state done

    q = [1, 2, 3, 4, 5, 6, 7] * 200
    for i_episode in range(i_episode + 1, 100001):
        random_action_epsilon = 0.1
        discount = get_discount(i_episode)
        print(f"Reset! seed={i_episode} epsilon={random_action_epsilon:.2f}" + " " * 50)
        reward_total = 0.
        loss_total = 0.
        n_tick = 0

        # g = tetris.BaseGame(my_engine, board_size=(20, 10), seed=i_episode)
        random.seed(0)
        g = tetris.BaseGame(my_engine, board_size=(20, 10), seed=0, queue=q.copy())

        if i_episode % SAVE_EVERY == 0:
            print("Saving checkpoints ...")
            torch.save(state2value_module.state_dict(), f"runs/{N_QUEUE}/mlp_full.pt")
            torch.save(optim_full.state_dict(),
                       f"runs/{N_QUEUE}/optim_full.pt")
            pathlib.Path(f"runs/{N_QUEUE}/progress.json").write_text(json.dumps({
                "i_episode": i_episode,
                "random_state":     random.getstate(),
                "i_update": i_update,
                "loss_per_update": loss_per_update,
            }))

        if RENDER and i_episode % RENDER_EVERY == 0:
            should_render = True
            render = get_render_func(g)
        else:
            should_render = False

        # suppress repeated action (falls into infinite loop in tetris)
        should_game_end = False
        # suppress repeated action done

        a = random.choice(ACTIONS)
        g_aft = get_next_game(g, a)
        assert g_aft is not None
        reward = immediate_reward(a, g, g_aft)
        reward_total += reward
        n_tick += 1
        g = g_aft

        while True:
            if i_update % 100 == 0:
                LOGGER.add_scalar('Update/Loss', loss_per_update / 100, i_update)
                LOGGER.add_scalar('Update/LearningRate', LR, i_update)
                loss_per_update = 0

            if g.lost:
                assert should_game_end
                # immediate reward is zero
                n_tick += 1

                optim_full.zero_grad()
                loss = criterion(game2value(g), torch.tensor(0))
                loss_total      += float(loss)
                loss_per_update += float(loss)
                loss.backward()
                torch.nn.utils.clip_grad_value_(state2value_module.parameters(), 100)
                optim_full.step()
                i_update += 1

                *_, n_hole, diff_total, _ = game2feature_heuristic(g)
                LOGGER.add_scalar('Episode/NumberOfHoles',        n_hole,                i_episode)
                LOGGER.add_scalar('Episode/Bumpyness',            diff_total,            i_episode)
                LOGGER.add_scalar('Episode/RewardAveragePerTick', reward_total / n_tick, i_episode)
                LOGGER.add_scalar('Episode/RewardTotal',          reward_total,          i_episode)
                LOGGER.add_scalar('Episode/NumberOfTicks',        n_tick,                i_episode)
                LOGGER.add_scalar('Episode/NumberOfLineClear',    g.scorer.line_clears,  i_episode)
                LOGGER.add_scalar('Episode/LossPerTick',          loss_total / n_tick,   i_episode)
                LOGGER.add_scalar('Episode/RandomActionRatio',    random_action_epsilon, i_episode)
                LOGGER.add_scalar('Episode/Discount',             discount,              i_episode)
                break

            actions = ACTIONS.copy()
            actions, gs_next = get_next_games(g, actions)
            with torch.no_grad():
                v_aft, i_aft, g_aft = max(zip([game2value(g_next) for g_next in gs_next], itertools.count(), gs_next))

            # if random.random() <= random_action_epsilon:
            #     i_best, g_best = random.choice(list(zip(itertools.count(), gs_next)))
            # else:

            if g_aft.lost:
                should_game_end = True

            # debug
            if should_render:
                wait_func = render()
                wait_func(10)
            # debug done

            # forward again for learning
            # optim_heuristic.zero_grad()
            optim_full.zero_grad()
            v = game2value(g)
            loss = criterion(reward + v_aft - v, torch.tensor(0))
            # print(f"{float(rvs[i_best]):.2f} {float(v):.2f}")
            loss_total      += loss
            loss_per_update += float(loss)
            loss.backward()
            # torch.nn.utils.clip_grad_value_(feature_heuristic2value.parameters(), 100)
            torch.nn.utils.clip_grad_value_(state2value_module.parameters(), 100)
            # optim_heuristic.step()
            optim_full.step()
            i_update += 1

            # apply action
            assert id(g) == id(g.gravity.game)
            assert id(g_aft) == id(g_aft.gravity.game)

            a = actions[i_aft]
            if random.random() <= random_action_epsilon:
                g_aft = random.choice(gs_next)
                if g_aft.lost:
                    should_game_end = True
                else:
                    should_game_end = False

            assert should_game_end == g_aft.lost, f"{should_game_end} != {g_aft.lost}"
            g = g_aft
            reward = immediate_reward(a, g, g_aft)
            reward_total += reward
            n_tick += 1
            # apply action done

            # g = g_best
            if should_render:
                render = get_render_func(g)

    # state = game2tensor(game)
    # feat = feature_extractor(torch.stack((state, state)))


if __name__ == '__main__':
    main()