import itertools
import json
import pathlib
import random
from typing import Iterator

import numpy as np
import tetris
import tetris.impl.queue
import tetris.impl.rotation
import tetris.impl.scorer
import torch
import torchvision
from tetris.engine import EngineFactory
from torch.utils.tensorboard import SummaryWriter

from . import board, utils
from .board import (Action, get_next_game, get_random_state,
                    number_of_blocks_dropped)
from .gravity import ManualGravity

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
LOGGER = SummaryWriter(f'runs/{board.N_QUEUE}')
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
            torch.nn.Conv2d(1, self.CONV_CH, (self.ROI_HEIGHT, board.BOARD_WIDTH)),
            torch.nn.ReLU(True),
        )
        self.queuemap = torchvision.ops.MLP(board.N_QUEUE * 7 + len(ACTIONS), [self.QUEUE_CH])
        self.head = torchvision.ops.MLP(
            self.QUEUE_CH + self.CONV_CH * board.BOARD_HEIGHT,
            [self.INTERMED_CH, self.INTERMED_CH, 1],
        )

    def forward(self, x):
        return self.head(
            torch.concat(
                [
                    self.conv(
                        x[..., :- board.N_QUEUE * 7 - len(ACTIONS)]
                        .reshape(-1, 1, board.BOARD_HEIGHT + 3, board.BOARD_WIDTH
                    )).reshape(-1, self.CONV_CH * board.BOARD_HEIGHT),
                    self.queuemap(x[..., - (board.N_QUEUE * 7 + len(ACTIONS)):]),
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
    #     BOARD_WIDTH * (BOARD_HEIGHT + TetrisNet.ROI_HEIGHT - 1) + board.N_QUEUE * 7 + len(ACTIONS),
    #     # [1024, 1024, 64, 1],
    #     [256, 256, 32, 1],
    # )
    mlp = TetrisNet()
    # mlp = mlp.cuda()
    if (mlp_path := pathlib.Path(f"runs/{board.N_QUEUE}/mlp_full.pt")).exists():
        mlp.load_state_dict(torch.load(mlp_path))
        print("Loaded checkpoint! (model)")
    else:
        print("Could not find checkpoint! Making new ...")
    return mlp


def get_optimizer(params):
    # optimizer = torch.optim.SGD(params, lr=LR, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    optimizer = torch.optim.Adam(params, lr=LR)
    if (optimizer_path := pathlib.Path(f"runs/{board.N_QUEUE}/optim_full.pt")).exists():
        optimizer.load_state_dict(torch.load(optimizer_path))
        print("Loaded checkpoint! (optimizer)")
    else:
        print("Could not find checkpoint! Making new ...")
    for g in optimizer.param_groups:
        g['lr'] = LR
    return optimizer


def action2tensor(a: Action) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(ACTIONS.index(a)), num_classes=len(ACTIONS))


def state_action2tensor(g: tetris.BaseGame, a: Action) -> torch.Tensor:
    b = torch.tensor(g.board[board.BOARD_HEIGHT - TetrisNet.ROI_HEIGHT + 1:].data)
    field = torch.tensor(g.get_playfield(buffer_lines=TetrisNet.ROI_HEIGHT - 1).data)
    field[field == 8] = 0  # Remove ghost
    field_mask = field - b > 0
    b = (field > 0).type(torch.float32)
    b[field_mask] *= -1
    queue_action = [board.piece2tensor(p) for p in g.queue[:b.N_QUEUE]] + [action2tensor(a)]
    vec = torch.concat((b, *queue_action)).type(torch.float32)
    return vec


def state2tensor(g: tetris.BaseGame) -> torch.Tensor:
    board = torch.tensor(g.board[board.BOARD_HEIGHT - TetrisNet.ROI_HEIGHT + 1:].data)
    field = torch.tensor(g.get_playfield(buffer_lines=TetrisNet.ROI_HEIGHT - 1).data)
    field[field == 8] = 0  # Remove ghost
    field_mask = field - board > 0
    board = (field > 0).type(torch.float32)
    board[field_mask] *= -1
    queue_action = [board.piece2tensor(p) for p in g.queue[:board.N_QUEUE]]
    vec = torch.concat((board, *queue_action)).type(torch.float32)
    return vec


my_engine = EngineFactory(
    gravity=ManualGravity,
    queue=tetris.impl.queue.SevenBag,
    rotation_system=tetris.impl.rotation.SRS,
    scorer=tetris.impl.scorer.GuidelineScorer,
)


def get_random_action_ratio(v_initial: float, v_final: float, n_current: int, n_final: int):
    return (
        v_final + (max(n_final - n_current, 0) * (v_initial - v_final) / n_final)
    )
    

# constants
METHOD = 'Q-learning'
REPLAY_QUEUE_SIZE = 500_000
N_STEP = 1_000_000
BATCH_SIZE = 512
DISCOUNT = 0.99
RANDOM_ACTION_INITIAL_VALUE = 1.
RANDOM_ACTION_FINAL_VALUE   = 1e-2
RANDOM_ACTION_FINAL_STEP    = 5_000
# constants done


def main():
    q_func = get_state2value()
    q_func.train()
    optim_full = get_optimizer(q_func.parameters())

    state = get_random_state(2, my_engine, None)
    match METHOD:
        case 'SARSA':
            replay_queue = utils.DequeSARSA(REPLAY_QUEUE_SIZE)  # SARSA
        case 'Q-learning':
            replay_queue = utils.DequeSAR(REPLAY_QUEUE_SIZE, acceptance_ratio=0.5)  # Q-learning
        case _:
            raise NotImplementedError(METHOD)
    loss_acc = utils.Accumulator()
    reward_acc = utils.Accumulator()

    i_episode = 0
    i_step = 1
    if (p := pathlib.Path(f"runs/{board.N_QUEUE}/progress.txt")).exists():
        progress = json.loads(p.read_text(encoding='utf8'))
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

        match METHOD:
            case 'SARSA':
                sa_bef, r, sa_aft = zip(*samples)
                q_func.eval()
                with torch.no_grad():
                    targets = torch.tensor(r)
                    active_idxs = [i for i, v in enumerate(sa_aft) if v is not None]
                    targets[active_idxs] += DISCOUNT * q_func(torch.stack([sa_aft[i] for i in active_idxs]))[:, 0]
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
                    targets[~done] += DISCOUNT * vs.values
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
            torch.save(q_func.state_dict(), f"runs/{board.N_QUEUE}/mlp_full.pt")
            torch.save(optim_full.state_dict(),         f"runs/{board.N_QUEUE}/optim_full.pt")
            pathlib.Path(f"runs/{board.N_QUEUE}/progress.txt").write_text(
                json.dumps({
                    "i_step": i_step,
                    "i_episode": i_episode,
                }),
                encoding='utf8',
            )


def report(g: tetris.BaseGame, r: utils.Accumulator, l: utils.Accumulator, rar: float, i: int):
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
