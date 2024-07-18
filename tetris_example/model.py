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
# Action = Literal["left", "right", "soft_drop", "rotate"]
ACTIONS = ["left", "right", "wait", "rotate"]
Action = Literal["left", "right", "wait", "rotate"]
RENDER = True
LOGGER = SummaryWriter(f'runs/{N_QUEUE}')
SAVE_EVERY = 100
RENDER_EVERY = 100
LR = 0.01

class TetrisNet(torch.nn.Module):
    ROI_HEIGHT = 4
    CONV_CH = 64
    QUEUE_CH = 32
    INTERMED_CH = 32

    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.CONV_CH, (self.ROI_HEIGHT, BOARD_WIDTH)),
            torch.nn.ReLU(True),
        )
        self.queuemap = torchvision.ops.MLP(N_QUEUE * 7 + len(ACTIONS), [self.QUEUE_CH])
        self.head = torchvision.ops.MLP(
            self.QUEUE_CH + self.CONV_CH * BOARD_HEIGHT,
            [self.INTERMED_CH, self.INTERMED_CH, 1],
        )

    def forward(self, x):
        return self.head(
            torch.concat(
                [
                    self.conv(
                        x[..., :- N_QUEUE * 7 - len(ACTIONS)]
                        .reshape(-1, 1, BOARD_HEIGHT + 3, BOARD_WIDTH
                    )).reshape(-1, self.CONV_CH * BOARD_HEIGHT),
                    self.queuemap(x[..., - (N_QUEUE * 7 + len(ACTIONS)):]),
                ],
                dim=1,
            )
        )


def get_discount(i: int = 0) -> Iterator[float]:
    VALUE_START = 0.99
    VALUE_END   = 0.999
    ITER_MAX = 10000000

    discount = VALUE_START + (VALUE_END - VALUE_START) * i / ITER_MAX
    return min(0.999, max(0, discount))


def get_state2value_full():
    mlp = torchvision.ops.MLP(
        BOARD_WIDTH * (BOARD_HEIGHT + TetrisNet.ROI_HEIGHT - 1) + N_QUEUE * 7 + len(ACTIONS),
        # [1024, 1024, 64, 1],
        [256, 256, 32, 1],
    )
    # mlp = TetrisNet()
    # mlp = mlp.cuda()
    if (mlp_path := pathlib.Path(f"runs/{N_QUEUE}/mlp_full.pt")).exists():
        print("Loaded checkpoint! (model)")
        mlp.load_state_dict(torch.load(mlp_path))
    else:
        print("Could not find checkpoint! Making new ...")
    return mlp, lambda xs: mlp(xs)[:, 0].cpu()


def get_feature_heuristic2value():
    mlp = torchvision.ops.MLP(7, [14, 16, 8, 4, 1], norm_layer=None)
    if (mlp_path := pathlib.Path(f"runs/{N_QUEUE}/mlp_heuristic.pt")).exists():
        print("Loaded checkpoint! (optimizer)")
        mlp.load_state_dict(torch.load(mlp_path))
    else:
        print("Could not find checkpoint! Making new ...")
    return mlp


def get_optimizer_heuristic(params):
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.,)
    # optimizer = torch.optim.SGD(params, lr=LR, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    # optimizer = torch.optim.Adam(params, lr=LR)
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


def action2tensor(a: Action) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(ACTIONS.index(a)), num_classes=len(ACTIONS))


def game2tensor(g: tetris.BaseGame, a: Action) -> torch.Tensor:
    board = torch.tensor(g.board[BOARD_HEIGHT - TetrisNet.ROI_HEIGHT + 1:].data)
    field = torch.tensor(g.get_playfield(buffer_lines=TetrisNet.ROI_HEIGHT - 1).data)
    field[field == 8] = 0  # Remove ghost
    field_mask = field - board > 0
    board = (field > 0).type(torch.float32)
    board[field_mask] *= -1
    queue_action = [piece2tensor(p) for p in g.queue[:N_QUEUE]] + [action2tensor(a)]
    vec = torch.concat((board, *queue_action)).type(torch.float32)
    return vec


def copy_game(g: tetris.BaseGame):
    _g = tetris.BaseGame(my_engine, board=g.board.copy(), board_size=(BOARD_HEIGHT, BOARD_WIDTH))
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
    if a == 'wait':
        _g.gravity.tick()
        _g.tick()
        return _g
    getattr(_g, a)()
    _g.gravity.tick()
    _g.tick()
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


def number_of_blocks_dropped(g: tetris.BaseGame) -> float:
    return (sum(np.array(g.board.data) > 0) + BOARD_WIDTH * g.scorer.line_clears) / 4


def immediate_reward(a: Action, g_bef: tetris.BaseGame, g_aft: tetris.BaseGame) -> float:
    reward = 0.

    # Reward some actions
    match a:
        case 'rotate':
            pass
        case 'left':
            pass
        case 'right':
            pass
        case 'soft_drop':
            pass
        case 'wait':
            pass
        case _:
            raise RuntimeError("Not Implemented!")
    # Reward some actions done

    # *_, n_hole_aft, diff_total_aft, max_height_aft = game2feature_heuristic(g_aft)
    # *_, n_hole_bef, diff_total_bef, max_height_bef = game2feature_heuristic(g_bef)
    reward += (g_aft.scorer.line_clears - g_bef.scorer.line_clears)
    if number_of_blocks_dropped(g_aft) != number_of_blocks_dropped(g_bef):
        reward += 1
    # reward += .1 * (n_hole_bef - n_hole_aft)

    if g_aft.lost:
        reward -= 100
    return reward


def get_random_action_ratio(i):
    if i < 10000:
        return 0.01
    if i < 100000:
        return 0.01
    return 0.01


def main():
    state2value_module, state2value = get_state2value_full()
    optim_full = get_optimizer_full(state2value_module.parameters())

    def game2value(g: tetris.BaseGame, a: Action) -> torch.Tensor:
        return state2value(game2tensor(g, a)[None])

    criterion = torch.nn.L1Loss()

    random.seed(1)
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

    # q = [4] * 200
    q = [1] * 200
    for i_episode in range(i_episode + 1, 10000001):
        random_action_ratio = get_random_action_ratio(i_episode)
        discount = get_discount(i_episode)
        print(f"Reset! seed={i_episode} epsilon={random_action_ratio:.2f}" + " " * 50)
        reward_total = 0.
        loss_total = 0.
        loss_max = -10000000.
        loss_min = +10000000.
        n_tick = 0

        # random.seed(0)
        g_bef = tetris.BaseGame(my_engine, board_size=(BOARD_HEIGHT, BOARD_WIDTH), seed=i_episode, queue=q.copy())
        # g_bef = tetris.BaseGame(my_engine, board_size=(BOARD_HEIGHT, BOARD_WIDTH))

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

        a_bef = random.choice(ACTIONS)
        g_aft = get_next_game(g_bef, a_bef)
        assert g_aft is not None
        reward = immediate_reward(a_bef, g_bef, g_aft)
        reward_total += reward
        n_tick += 1

        if RENDER and i_episode % RENDER_EVERY == 0:
            should_render = True
            render = get_render_func(g_aft)
        else:
            should_render = False

        v_aft, _, a_aft = max((game2value(g_aft, a), i, a) for i, a in enumerate(ACTIONS))
        episode = []
        episode.append((g_bef, a_bef, reward, g_aft, a_aft, v_aft))
        while True:
            if len(episode) >= 1000:
                break
            if i_update % 1000 == 0:
                LOGGER.add_scalar('Update/Loss', loss_per_update / 1000, i_update)
                LOGGER.add_scalar('Update/LearningRate', LR, i_update)
                LOGGER.add_scalar('Update/RandomActionRatio', random_action_ratio, i_update)
                loss_per_update = 0

            if g_aft.lost:
                *_, n_hole, diff_total, _ = game2feature_heuristic(g_aft)
                LOGGER.add_scalar('Episode/NumberOfHoles',         n_hole,                i_episode)
                LOGGER.add_scalar('Episode/Bumpyness',             diff_total,            i_episode)
                LOGGER.add_scalar('Episode/RewardAveragePerTick',  reward_total / n_tick, i_episode)
                LOGGER.add_scalar('Episode/RewardTotal',           reward_total,          i_episode)
                LOGGER.add_scalar('Episode/NumberOfTicks',         n_tick,                i_episode)
                LOGGER.add_scalar('Episode/NumberOfLineClear',     g_aft.scorer.line_clears,  i_episode)
                LOGGER.add_scalar('Episode/LossAve',               loss_total / n_tick,   i_episode)
                LOGGER.add_scalar('Episode/LossMax',               loss_max,              i_episode)
                LOGGER.add_scalar('Episode/LossMin',               loss_min,              i_episode)
                LOGGER.add_scalar('Episode/RandomActionRatio',     random_action_ratio,   i_episode)
                LOGGER.add_scalar('Episode/Discount',              discount,              i_episode)
                LOGGER.add_scalar('Episode/NumberOfBlocksDropped', number_of_blocks_dropped(g_aft), i_episode)
                if should_render:
                    wait_func = render()
                    wait_func(10)
                break

            actions = ACTIONS.copy()
            with torch.no_grad():
                v_aft, _, a_aft = max((game2value(g_aft, a), i, a) for i, a in enumerate(actions))
            # debug
            if should_render:
                wait_func = render()
                wait_func(10)
            # debug done

            # forward again for learning
            # optim_heuristic.zero_grad()
            # optim_full.zero_grad()
            # v = game2value(g_bef, a_bef)
            # loss = criterion(v, reward + discount * v_aft)
            # loss_float = float(loss)
            # loss_total      += loss_float
            # loss_max        = max(loss_max, loss_float)
            # loss_min        = min(loss_min, loss_float)
            # loss_per_update += loss_float
            # loss.backward()
            # torch.nn.utils.clip_grad_value_(state2value_module.parameters(), 10)
            # optim_full.step()
            i_update += 1

            # apply action
            assert id(g_bef) == id(g_bef.gravity.game)
            assert id(g_aft) == id(g_aft.gravity.game)

            episode.append((g_bef, a_bef, reward, g_aft, a_aft, v_aft))
            if random.random() <= random_action_ratio:
                a_aft = random.choice(ACTIONS)
            g_aft_aft = get_next_game(g_aft, a_aft)

            g_aft, g_bef = g_aft_aft, g_aft
            reward = immediate_reward(a_aft, g_bef, g_aft)
            a_bef = a_aft
            reward_total += reward
            n_tick += 1

            if should_render:
                render = get_render_func(g_aft_aft)

        gs_bef, as_bef, rs, gs_aft, as_aft, vs_aft = list(zip(*episode))

        # Monte Carlo target
        arr_rwrd = np.array(rs + (0,) * (len(rs) - 1))
        window_size = len(rs)
        xs = np.lib.stride_tricks.as_strided(
            arr_rwrd,
            (arr_rwrd.size - window_size + 1, window_size), [arr_rwrd.strides[0], arr_rwrd.strides[0]]
        )
        # sum(rs[i] * discount ** i for i in np.nonzero(rs)[0])
        targs = (xs * (discount ** np.arange(len(rs)))).sum(axis=1)
        targs_mc = torch.tensor(targs)
        # Monte Carlo target done

        # Q learning target
        with torch.no_grad():
            targs_td = discount * torch.stack(vs_aft).flatten() + torch.tensor(rs)
        # Q learning target done

        optim_full.zero_grad()
        vs = state2value(torch.stack([game2tensor(g, a) for g, a in zip(gs_bef, as_bef)]))
        loss = criterion(vs, .5 * (targs_mc + targs_td))
        loss.backward()
        optim_full.step()
    # state = game2tensor(game)
    # feat = feature_extractor(torch.stack((state, state)))


if __name__ == '__main__':
    main()
