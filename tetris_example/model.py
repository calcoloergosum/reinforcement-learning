from .gravity import NESGravity
import tetris.impl.queue
import tetris.impl.rotation
import tetris.impl.scorer
from tetris.engine import EngineFactory
import pathlib
from typing import Iterator
import itertools
import json
import numpy as np
import tetris
import torch
import torchvision
import random
from torch.utils.tensorboard import SummaryWriter
from .board import get_next_game, Action, BOARD_WIDTH, BOARD_HEIGHT, N_QUEUE, get_random_state, number_of_blocks_dropped
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
ACTIONS = ["left", "right", "wait", "rotate"]
RENDER = True
LOGGER = SummaryWriter(f'runs/{N_QUEUE}')
SAVE_EVERY = 100
RENDER_EVERY = 100
LR = 0.001


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


def get_state2value():
    # mlp = torchvision.ops.MLP(
    #     BOARD_WIDTH * (BOARD_HEIGHT + TetrisNet.ROI_HEIGHT - 1) + N_QUEUE * 7 + len(ACTIONS),
    #     # [1024, 1024, 64, 1],
    #     [256, 256, 32, 1],
    # )
    mlp = TetrisNet()
    # mlp = mlp.cuda()
    if (mlp_path := pathlib.Path(f"runs/{N_QUEUE}/mlp_full.pt")).exists():
        mlp.load_state_dict(torch.load(mlp_path))
        print("Loaded checkpoint! (model)")
    else:
        print("Could not find checkpoint! Making new ...")
    return mlp


def get_optimizer(params):
    # optimizer = torch.optim.SGD(params, lr=LR, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    optimizer = torch.optim.Adam(params, lr=LR)
    if (optimizer_path := pathlib.Path(f"runs/{N_QUEUE}/optim_full.pt")).exists():
        optimizer.load_state_dict(torch.load(optimizer_path))
        print("Loaded checkpoint! (optimizer)")
    else:
        print("Could not find checkpoint! Making new ...")
    for g in optimizer.param_groups:
        g['lr'] = LR
    return optimizer


def piece2tensor(p) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(p.value) - 1, num_classes=7)


def action2tensor(a: Action) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(ACTIONS.index(a)), num_classes=len(ACTIONS))


def state_action2tensor(g: tetris.BaseGame, a: Action) -> torch.Tensor:
    board = torch.tensor(g.board[BOARD_HEIGHT - TetrisNet.ROI_HEIGHT + 1:].data)
    field = torch.tensor(g.get_playfield(buffer_lines=TetrisNet.ROI_HEIGHT - 1).data)
    field[field == 8] = 0  # Remove ghost
    field_mask = field - board > 0
    board = (field > 0).type(torch.float32)
    board[field_mask] *= -1
    queue_action = [piece2tensor(p) for p in g.queue[:N_QUEUE]] + [action2tensor(a)]
    vec = torch.concat((board, *queue_action)).type(torch.float32)
    return vec


def state2tensor(g: tetris.BaseGame) -> torch.Tensor:
    board = torch.tensor(g.board[BOARD_HEIGHT - TetrisNet.ROI_HEIGHT + 1:].data)
    field = torch.tensor(g.get_playfield(buffer_lines=TetrisNet.ROI_HEIGHT - 1).data)
    field[field == 8] = 0  # Remove ghost
    field_mask = field - board > 0
    board = (field > 0).type(torch.float32)
    board[field_mask] *= -1
    queue_action = [piece2tensor(p) for p in g.queue[:N_QUEUE]]
    vec = torch.concat((board, *queue_action)).type(torch.float32)
    return vec


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


from collections import deque

class DequeSARSA:
    def __init__(self, maxlen: int) -> None:
        self.sar = None
        self.queue = deque(maxlen=maxlen)

    def push(self, sarsd):
        t_, r_, _, done = state_action2tensor(*sarsd[:2]), *sarsd[2:]
        if self.sar is not None:
            self.queue.append((*self.sar, t_))

        if done:
            # assert self.sar is not None
            self.queue.append((*self.sar, None))
            self.sar = None
            return
        else:
            self.sar = (t_, r_)
        return


class DequeSAR:
    def __init__(self, maxlen: int, acceptance_ratio: float) -> None:
        self.queue = deque(maxlen=maxlen)
        self.acceptance_ratio = acceptance_ratio

    def push(self, sarsd):
        accept = random.random() < self.acceptance_ratio
        accept |= sarsd[2] > 1  # Positive reward; line clear
        if accept:
            t_, r_, s, done = state_action2tensor(*sarsd[:2]), sarsd[2], state2tensor(sarsd[3]), sarsd[4]
            self.queue.append((t_, r_, s, done))
        return


class Accumulator():
    def __init__(self):
        self.sum = 0
        self.n = 0
        self._max = -1e7
        self._min = 1e7
        
    def clear(self):
        self.sum = 0
        self.n = 0
        self._max = -1e7
        self._min = 1e7
    
    def add(self, v):
        self.sum += v
        self.n += 1
        self._min = min(self._min, v)
        self._max = max(self._max, v)

    @property
    def average(self):
        return 0 if self.n == 0 else self.sum / self.n

    @property
    def min(self):
        return 0 if self.n == 0 else self._min

    @property
    def max(self):
        return 0 if self.n == 0 else self._max


def get_random_action_ratio(v_initial: float, v_final: float, n_current: int, n_final: int):
    return (
        v_final + (max(n_final - n_current, 0) * (v_initial - v_final) / n_final)
    )


def main():
    # constants
    method = 'Q-learning'
    REPLAY_QUEUE_SIZE = 500_000
    N_STEP = 1_000_000
    BATCH_SIZE = 512
    discount = 0.99
    RANDOM_ACTION_INITIAL_VALUE = 1.
    RANDOM_ACTION_FINAL_VALUE   = 1e-2
    RANDOM_ACTION_FINAL_STEP    = 5_000
    # constants done
    q_func = get_state2value()
    q_func.train()
    optim_full = get_optimizer(q_func.parameters())

    state = get_random_state(3, my_engine, None)
    match method:
        case 'SARSA':
            replay_queue = DequeSARSA(REPLAY_QUEUE_SIZE)  # SARSA
        case 'Q-learning':
            replay_queue = DequeSAR(REPLAY_QUEUE_SIZE, acceptance_ratio=0.5)  # Q-learning
        case _:
            raise NotImplementedError(method)
    loss_acc = Accumulator()
    reward_acc = Accumulator()

    i_episode = 0
    i_step = 1
    if (p := pathlib.Path(f"runs/{N_QUEUE}/progress.txt")).exists():
        progress = json.loads(p.open('r').read())
        i_step, i_episode = progress['i_step'], progress['i_episode']

    for i_step in range(i_step, N_STEP):
        # policy
        # arsd_list = get_next_n_best_game(state, action, my_engine)
        random_action_ratio = get_random_action_ratio(RANDOM_ACTION_INITIAL_VALUE, RANDOM_ACTION_FINAL_VALUE, i_episode, RANDOM_ACTION_FINAL_STEP)
        if random.random() <= random_action_ratio:
            action = random.choice(ACTIONS)
        else:
            q_func.eval()
            with torch.no_grad():
                i_a = torch.argmax(q_func(torch.stack([state_action2tensor(state, a) for a in ACTIONS])).reshape(-1))
                action = ACTIONS[i_a]
            q_func.train()
        reward, state_, over = get_next_game(state, action, my_engine)
        replay_queue.push((state, action, reward, state_, over))
        reward_acc.add(reward)
        # policy done

        state = state_

        if over or loss_acc.n > 10_000:  # force terminate if tick more than 10k
            i_episode += 1
            report(state_, reward_acc, loss_acc, random_action_ratio, i_episode)
            loss_acc.clear()
            reward_acc.clear()
            state = get_random_state(3, my_engine, None)

        if len(replay_queue.queue) < REPLAY_QUEUE_SIZE / 10:
            continue

        samples = random.sample(replay_queue.queue, min(len(replay_queue.queue), BATCH_SIZE))

        match method:
            case 'SARSA':
                sa_bef, r, sa_aft = zip(*samples)
                q_func.eval()
                with torch.no_grad():
                    targets = torch.tensor(r)
                    active_idxs = [i for i, v in enumerate(sa_aft) if v is not None]
                    targets[active_idxs] += discount * q_func(torch.stack([sa_aft[i] for i in active_idxs]))[:, 0]
                q_func.train()
            case 'Q-learning':
                sa_bef, r, s_aft, done = zip(*samples)
                q_func.eval()
                with torch.no_grad():
                    targets = torch.tensor(r)

                    done = torch.tensor(done)
                    size_sa = len(sa_bef[0])
                    n_not_done = sum(~done)
                    sa_aft = torch.empty((n_not_done, len(ACTIONS), size_sa))
                    sa_aft[:, :, :-len(ACTIONS)] = torch.stack(s_aft)[~done, None, :]
                    sa_aft[:, :, -len(ACTIONS):] = torch.stack([action2tensor(a) for a in ACTIONS])[None, :, :]
                    vs = torch.max(
                        q_func(sa_aft.reshape(-1, size_sa)).reshape(n_not_done, -1),
                        dim=1,
                    )
                    targets[~done] += discount * vs.values
                q_func.train()

        optim_full.zero_grad()
        loss = torch.nn.functional.mse_loss(
            q_func(torch.stack(sa_bef))[:, 0],
            targets,
        )
        loss.backward()
        optim_full.step()
        loss_acc.add(float(loss))

        if i_step % 1000 == 0:
            print("Saving checkpoints ...")
            torch.save(q_func.state_dict(), f"runs/{N_QUEUE}/mlp_full.pt")
            torch.save(optim_full.state_dict(),         f"runs/{N_QUEUE}/optim_full.pt")
            with pathlib.Path(f"runs/{N_QUEUE}/progress.txt").open('w') as fp:
                fp.write(json.dumps({
                    "i_step": i_step,
                    "i_episode": i_episode,
                }))


def report(g: tetris.BaseGame, r: Accumulator, l: Accumulator, rar: float, i: int):
    print            ('Episode/NumberOfLineClear',     g.scorer.line_clears, i)
    LOGGER.add_scalar('Episode/NumberOfLineClear',     g.scorer.line_clears, i)
    LOGGER.add_scalar('Episode/LossAverage',           l.average,            i)
    LOGGER.add_scalar('Episode/LossMax',               l.max,                i)
    LOGGER.add_scalar('Episode/LossMin',               l.min,                i)
    LOGGER.add_scalar('Episode/NumberOfTicks',         r.n,                  i)
    LOGGER.add_scalar('Episode/RewardAverage',         r.average,            i)
    LOGGER.add_scalar('Episode/RewardTotal',           r.sum,                i)
    LOGGER.add_scalar('Episode/RandomActionRatio',     rar,                  i)
    LOGGER.add_scalar('Episode/NumberOfTicks',  number_of_blocks_dropped(g), i)


if __name__ == '__main__':
    main()
