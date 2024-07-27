import json
import pathlib
import random
from typing import List

import numpy as np
import rust_tetris
import torch
import torch.utils
import torch.utils.tensorboard
import torchvision

from . import board, utils
from .gravity import ManualGravity

# Grouped action definition
MINO2ACTIONS = {
    # rotation, translation
    "I": [
        (0, -3),
        (0, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, -5),
        (1, -4),
        (1, -3),
        (1, -2),
        (1, -1),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ],
    "J": [
        (0, -3),
        (0, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, -4),
        (1, -3),
        (1, -2),
        (1, -1),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, -3),
        (2, -2),
        (2, -1),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, -3),
        (3, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 5),
    ],
    "L": [
        (0, -3),
        (0, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, -4),
        (1, -3),
        (1, -2),
        (1, -1),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, -3),
        (2, -2),
        (2, -1),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, -3),
        (3, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 5),
    ],
    "O": [
        (0, -4),
        (0, -3),
        (0, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
    ],
    "S": [
        (0, -3),
        (0, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, -4),
        (1, -3),
        (1, -2),
        (1, -1),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ],
    "T": [
        (0, -3),
        (0, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, -4),
        (1, -3),
        (1, -2),
        (1, -1),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, -3),
        (2, -2),
        (2, -1),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, -3),
        (3, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 5),
    ],
    "Z": [
        (0, -3),
        (0, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, -4),
        (1, -3),
        (1, -2),
        (1, -1),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ],
}
MINO2ACTION2INDEX = {
    mino: {a: i for i, a in enumerate(acts)}
    for mino, acts in MINO2ACTIONS.items()
}
ACTION_SIZE = max(len(acts) for acts in MINO2ACTIONS.values())
# Grouped action definition done


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
        self.queuemap = torchvision.ops.MLP(board.N_QUEUE * 7 + ACTION_SIZE, [self.QUEUE_CH])
        self.head = torchvision.ops.MLP(
            self.QUEUE_CH + self.CONV_CH * board.BOARD_HEIGHT,
            [self.INTERMED_CH, self.INTERMED_CH, 1],
        )

    def forward(self, x):
        return self.head(
            torch.concat(
                [
                    self.conv(
                        x[..., :- board.N_QUEUE * 7 - ACTION_SIZE]
                        .reshape(-1, 1, board.BOARD_HEIGHT + 3, board.BOARD_WIDTH
                    )).reshape(-1, self.CONV_CH * board.BOARD_HEIGHT),
                    self.queuemap(x[..., - (board.N_QUEUE * 7 + ACTION_SIZE):]),
                ],
                dim=1,
            )
        )


def get_q_function(root: pathlib.Path) -> torch.nn.Module:
    # net = TetrisNet()
    net = torchvision.ops.MLP(
        board.BOARD_WIDTH * (board.BOARD_HEIGHT + TetrisNet.ROI_HEIGHT - 1) +
        board.N_QUEUE * 7 + ACTION_SIZE,
        # [1024, 1024, 64, 1],
        [1024, 1024, 256, 256, 32, 1],
    )
    if (net_path := root / "mlp_full.pt").exists():
        net.load_state_dict(torch.load(net_path))
        print("Loaded checkpoint! (model)")
    else:
        print("Could not find checkpoint! Making new ...")
    return net


def get_optimizer(root: pathlib.Path, params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    optim = torch.optim.SGD(params, lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    # optim = torch.optim.Adam(params, lr=lr)
    if (optim_path := root / "optim_full.pt").exists():
        optim.load_state_dict(torch.load(optim_path))
        print("Loaded checkpoint! (optimizer)")
    else:
        print("Could not find checkpoint! Making new ...")
    for g in optim.param_groups:
        g['lr'] = lr
    return optim


def action2tensor(mino, a):
    return torch.nn.functional.one_hot(
        torch.tensor(MINO2ACTION2INDEX[mino][a]),
        num_classes=ACTION_SIZE,
    )


def state_action2tensor(g: rust_tetris.BaseGame, a):
    b = torch.tensor(g.board[board.BOARD_HEIGHT - TetrisNet.ROI_HEIGHT + 1:].data)
    field = torch.tensor(g.get_playfield(buffer_lines=TetrisNet.ROI_HEIGHT - 1).data)
    field[field == 8] = 0  # Remove ghost
    field_mask = field - b > 0
    b = (field > 0).type(torch.float32)
    b[field_mask] *= -1
    queue_action = [board.piece2tensor(p) for p in g.queue[:board.N_QUEUE]] + \
                   [action2tensor(g.piece.type.name, a)]
    vec = torch.concat((b, *queue_action)).type(torch.float32)
    return vec


def state2tensor(g: rust_tetris.BaseGame):
    b = torch.tensor(g.board[board.BOARD_HEIGHT - TetrisNet.ROI_HEIGHT + 1:].data)
    field = torch.tensor(g.get_playfield(buffer_lines=TetrisNet.ROI_HEIGHT - 1).data)
    field[field == 8] = 0  # Remove ghost
    field_mask = field - b > 0
    b = (field > 0).type(torch.float32)
    b[field_mask] *= -1
    queue_action = [board.piece2tensor(p) for p in g.queue[:board.N_QUEUE]]
    vec = torch.concat((b, *queue_action)).type(torch.float32)
    return vec


def get_next_game(g: rust_tetris.BaseGame, a, engine):
    _g = board.copy_game(g, engine)
    r, t = a
    if r != 0:
        _g.rotate(turns=r)
    if t < 0:
        _g.left(tiles=-t)
    if t > 0:
        _g.right(tiles=t)

    _g.hard_drop()
    if engine is not None:
        _g.gravity.tick()
    _g.tick()
    return board.immediate_reward(g, _g) , _g, _g.lost


def get_random_action_ratio(v_initial, v_final, n_current, n_final):
    return (
        v_final + (max(n_final - n_current, 0) * (v_initial - v_final) / n_final)
    )


# constants
#     progress tracking
ROOT = pathlib.Path(f"runs/{board.N_QUEUE}_comp_act")
ROOT.mkdir(exist_ok=True)
#     schedule
REPLAY_QUEUE_SIZE = 300_000
LR = 0.01
N_STEP = 1_000_000
BATCH_SIZE = 512
#     environment, agent
EMPTY_HEIGHT = board.BOARD_HEIGHT // 2
METHOD = 'Q-learning-PER'
DISCOUNT = 0.95
RANDOM_ACTION_INITIAL_VALUE = 5e-2
RANDOM_ACTION_FINAL_VALUE   = 5e-2
RANDOM_ACTION_FINAL_STEP    = 2_000
#     logging
LOGGER = torch.utils.tensorboard.SummaryWriter(ROOT)
np.set_printoptions(precision=3, threshold=1e5, suppress=True)
# constants done


def main():
    # Set random seed
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # Set random seed done

    q_func = get_q_function(ROOT)
    q_func.train()
    optim = get_optimizer(ROOT, q_func.parameters(), LR)

    my_engine = rust_tetris.EngineFactory(
        gravity=ManualGravity,
        queue=rust_tetris.impl.queue.SevenBag,
        rotation_system=rust_tetris.impl.rotation.SRS,
        scorer=rust_tetris.impl.scorer.GuidelineScorer,
    )

    state = board.get_random_state(EMPTY_HEIGHT, my_engine, None)
    match METHOD:
        case 'SARSA':
            replay_queue = utils.DequeSARSA(REPLAY_QUEUE_SIZE, state_action2tensor)  # SARSA
        case 'Q-learning':
            replay_queue = utils.DequeSAR(maxlen=REPLAY_QUEUE_SIZE,
                                          acceptance_ratio=1.,
                                          sa2t=state_action2tensor, s2t=state2tensor)  # Q-learning
        case 'Q-learning-PER':
            # Q-learning with Prioritized Experience Replay (PER)
            replay_queue = utils.PrioritizedSAR(
                maxlen=REPLAY_QUEUE_SIZE,
                batch_size=BATCH_SIZE,
                sa2t=state_action2tensor,
                s2t =state2tensor
            )
        case _:
            raise NotImplementedError(METHOD)
    loss_acc = utils.Accumulator()
    reward_acc = utils.Accumulator()

    i_episode = 0
    i_step = 1
    if (p := ROOT / "progress.txt").exists():
        progress = json.loads(p.open('r').read())
        i_step, i_episode = progress['i_step'], progress['i_episode']

    for i_step in range(i_step, N_STEP):
        # policy
        actions = MINO2ACTIONS[state.piece.type.name]
        random_action_ratio = get_random_action_ratio(RANDOM_ACTION_INITIAL_VALUE,
                                                      RANDOM_ACTION_FINAL_VALUE,
                                                      i_episode,
                                                      RANDOM_ACTION_FINAL_STEP)
        if replay_queue.size() < REPLAY_QUEUE_SIZE / 10 or random.random() <= random_action_ratio:
            action = random.choice(actions)
        else:
            q_func.eval()
            with torch.no_grad():
                i_a = torch.argmax(
                    q_func(torch.stack([
                        state_action2tensor(state, a)
                        for a in actions
                    ])).reshape(-1)
                )
                action = actions[i_a]
            q_func.train()
        reward, state_, over = get_next_game(state, action, my_engine)
        replay_queue.push((state, action, reward, state_, over))
        reward_acc.add(reward)
        state = state_
        del action, reward, state_, actions
        # policy done

        # schedule
        if over or loss_acc.n > 10_000:  # force terminate if tick more than 10k
            i_episode += 1
            report(LOGGER, state, reward_acc, loss_acc, random_action_ratio, replay_queue, i_episode)
            loss_acc.clear()
            reward_acc.clear()
            state = board.get_random_state(EMPTY_HEIGHT, my_engine, None)

        if replay_queue.size() < REPLAY_QUEUE_SIZE / 10:
            continue
        # schedule done

        # learning
        samples, update_queue = replay_queue.sample()
        loss_np, loss_mean = learn(METHOD, samples, q_func, optim, DISCOUNT)
        update_queue(loss_np)
        loss_acc.add(float(loss_mean))
        # learning done

        # logging
        if i_step % 1000 == 0:
            print("Score Samples", np.array([x[0] for x in replay_queue.ipq.heap[:1000]]))
            print("(UCB) Arm visit Samples", np.array(replay_queue.ucb.i2n_raw[:1000]))
            print("Saving checkpoints ...")
            torch.save(q_func.state_dict(), ROOT / "mlp_full.pt")
            torch.save(optim.state_dict(), ROOT / "optim_full.pt")
            with (ROOT / "progress.txt").open('w') as fp:
                fp.write(json.dumps({
                    "i_step": i_step,
                    "i_episode": i_episode,
                }))
        # logging done


def learn(method, samples, q_func, optim, discount):
    match method:
        case 'SARSA':
            sa_bef, r, sa_aft = zip(*samples)
            q_func.eval()
            with torch.no_grad():
                targets = torch.tensor(r)
                active_idxs = [i for i, v in enumerate(sa_aft) if v is not None]
                targets[active_idxs] += discount * q_func(torch.stack([sa_aft[i] for i in active_idxs]))[:, 0]
            q_func.train()
        case 'Q-learning' | 'Q-learning-PER':
            pieces, sa_bef, r, s_aft, done = zip(*samples)
            q_func.eval()
            with torch.no_grad():
                targets = torch.tensor(r)

                done = torch.tensor(done)
                size_sa = len(sa_bef[0])

                s_aft = torch.stack(s_aft)
                vs = torch.inf * torch.ones(sum(~done))

                pieces_np = np.array(pieces)[~done]
                for mino, actions in MINO2ACTIONS.items():
                    _idxs = torch.from_numpy(pieces_np == mino)
                    n_mino = _idxs.sum()
                    if n_mino == 0:
                        continue
                    action_size = len(actions)
                    sa_aft = torch.empty((n_mino, action_size, size_sa))
                    sa_aft[:, :, :-ACTION_SIZE] = s_aft[~done, None, :][_idxs]
                    sa_aft[:, :, -ACTION_SIZE:] = torch.nn.functional.one_hot(
                        torch.arange(action_size), ACTION_SIZE)[None, :, :]
                    vs[_idxs] = torch.max(
                        q_func(
                            sa_aft
                            .reshape(-1, size_sa)
                        )
                        .reshape(n_mino, action_size),
                        dim=1,
                    ).values
                assert (vs != torch.inf).all()
                targets[~done] += discount * vs
            assert (targets != torch.inf).all()
            q_func.train()

    optim.zero_grad()
    loss = torch.nn.functional.smooth_l1_loss(
        q_func(torch.stack(sa_bef)).flatten(),
        targets, reduction='none'
    )
    loss_mean = loss.mean()
    loss_mean.backward()
    # torch.nn.utils.clip_grad_norm_(q_func.parameters(), 10)
    optim.step()
    return loss.detach().numpy(), float(loss_mean)


def report(logger: torch.utils.tensorboard.SummaryWriter,
           g: rust_tetris.BaseGame,
           r: utils.Accumulator,  # reward
           l: utils.Accumulator,  # loss
           rar: float,            # random action ratio
           q,
           i: int,                # episode index
) -> None:
    logger.add_scalar('Episode/NumberOfLineClear',     g.scorer.line_clears, i)
    logger.add_scalar('Episode/LossAverage',           l.average,            i)
    logger.add_scalar('Episode/LossMax',               l.max,                i)
    logger.add_scalar('Episode/LossMin',               l.min,                i)
    logger.add_scalar('Episode/NumberOfTicks',         r.n,                  i)
    logger.add_scalar('Episode/RewardAverage',         r.average,            i)
    logger.add_scalar('Episode/RewardTotal',           r.sum,                i)
    logger.add_scalar('Episode/RandomActionRatio',     rar,                  i)
    logger.add_scalar('Episode/NumberOfTicks',  board.number_of_blocks_dropped(g), i)

    # histogram report
    priorities = np.array([x for x, *_ in q.ipq.heap if x < 1e4])
    if len(priorities > 0):
        logger.add_histogram('ReplayQueue/Scores', priorities,  i)


if __name__ == '__main__':
    main()
