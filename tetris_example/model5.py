import json
import pathlib
import random
from typing import List

import numpy as np
import rust_tetris
import torch
import torch.utils
import torch.utils.tensorboard
import tqdm

from . import board, game, utils
from .action_cluster import MINO2ACTIONS
from .net import LinearDDQN as DDQN


def get_q_function(root: pathlib.Path) -> torch.nn.Module:
    in_channels = board.BOARD_WIDTH * (board.BOARD_HEIGHT + board.BOARD_HEIGHT_ROI) +\
        (1 + board.N_QUEUE) * 7
    net = DDQN(in_channels)
    net.to(device)
    # net = torchvision.ops.MLP(in_channels, [1024, 1024, 32, ACTION_SIZE])
    if (net_path := root / model_filename).exists():
        net.load_state_dict(torch.load(net_path))
        print("Loaded checkpoint! (model)")
    else:
        print("Could not find checkpoint! Making new ...")
    return net


def get_optimizer(root: pathlib.Path, params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    # optim = torch.optim.SGD(params, lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    optim = torch.optim.Adam(params, lr=lr)
    if (optim_path := root / optim_filename).exists():
        optim.load_state_dict(torch.load(optim_path))
        print("Loaded checkpoint! (optimizer)")
    else:
        print("Could not find checkpoint! Making new ...")
    for g in optim.param_groups:
        g['lr'] = lr
    return optim


def state2tensor(g: rust_tetris.Game):
    b = game.game2bool_field(g)
    queue = [board.piece2tensor(p) for p in g.queue[:board.N_QUEUE]]
    vec = torch.concat((b, *queue)).type(torch.float32)
    return vec


# constants
#     progress tracking
ROOT = pathlib.Path(f"runs/ddqn_{board.N_QUEUE}")
ROOT.mkdir(exist_ok=True)
#     schedule & Sampling
REPLAY_QUEUE_SIZE = 300_000
REPLAY_QUEUE_SIZE_MIN = 30_000
PER_ALPHA = 0.5
# LR = 0.00001   # for mlp
LR = 0.0001   # for ddqn
N_STEP = 1_000_000
BATCH_SIZE = 64
DOUBLE_Q_REPLACE_EVERY = 10_000
#     environment, agent
DISCOUNT = 0

#     logging
LOGGER = torch.utils.tensorboard.SummaryWriter(ROOT)
np.set_printoptions(precision=3, threshold=1e5, suppress=True)
model_filename = "model.pt"
optim_filename = "optim.pt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# constants done


def get_data_loader():
        paths = list(pathlib.Path("logs").rglob("log_*.json"))
        random.shuffle(paths)
        for path in paths:
            try:
                e = json.loads(path.read_text())
            except FileNotFoundError:
                print(f"file deleted ({path.as_posix()})")
                continue
            is_over = [False for _ in e['actions']]
            is_over[-1] = e['over']
            for i, (s, a, s_, over) in enumerate(zip(e['states'], e['actions'], e['states'][1:], is_over)):
                s,  p,  score  = s [:-1], *s[-2:]
                s_, p_, score_ = s_[:-1], *s_[-2:]
                assert 220 + 8 + 1 <= len(s)  <= 220 + 14 + 1, f"{len(s) } {p}"
                assert 220 + 8 + 1 <= len(s_) <= 220 + 14 + 1, f"{len(s_)} {p_}"

                q = [0 for _ in range(board.N_QUEUE * 7)]
                for i, _p in enumerate(s[220:220+board.N_QUEUE]):
                    q[7*i + _p - 1] = 1

                q_ = [0 for _ in range(board.N_QUEUE * 7)]
                for i, _p in enumerate(s[220:220+board.N_QUEUE]):
                    q_[7*i + _p - 1] = 1

                assert score_ - score >= 0, f"{path}, {i}"
                yield (s[:220] + q, p), a, score_ - score, (s_[:220] + q_, p_), over
            assert over


def main():
    # Set random seed
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # Set random seed done

    # Load model
    q_func_old = get_q_function(ROOT)
    q_func_new = get_q_function(ROOT)
    q_func_old.eval()
    q_func_new.train()
    optim = get_optimizer(ROOT, q_func_new.parameters(), LR)
    # Load model done

    iter_sars = get_data_loader()

    # replay_queue = utils.PrioritizedSAR(
    #     maxlen=REPLAY_QUEUE_SIZE,
    #     batch_size=BATCH_SIZE,
    #     per_power=PER_ALPHA,
    # )
    replay_queue = utils.SimpleSAR(
        maxlen=REPLAY_QUEUE_SIZE,
        batch_size=BATCH_SIZE,
    )
    loss_acc = utils.Accumulator()
    reward_acc = utils.Accumulator()

    i_step = 1
    if (p := ROOT / "progress.txt").exists():
        progress = json.loads(p.open('r').read())
        i_step = progress['i_step']

    progress_bar = tqdm.tqdm(total=1000)
    for i_step in range(i_step, N_STEP):
        progress_bar.update()

        try:
            (state, action, reward, state_, over) = next(iter_sars)
        except StopIteration:
            iter_sars = get_data_loader()
            (state, action, reward, state_, over) = next(iter_sars)

        replay_queue.push((state, action, reward, state_, over))
        reward_acc.add(reward)
        if over:
            report(LOGGER, state, reward_acc, loss_acc, 0., replay_queue, i_step)
            loss_acc.clear()
            reward_acc.clear()
        state = state_
        del action, reward, state_
        # policy done

        # schedule
        if replay_queue.size() < REPLAY_QUEUE_SIZE_MIN:
            continue
        # schedule done

        # learning
        samples, update_queue = replay_queue.sample()
        loss_np, loss_mean = learn(samples, q_func_old, q_func_new, optim, DISCOUNT)
        update_queue(loss_np)
        loss_acc.add(float(loss_mean))
        # print(f"{loss_mean:.3f}\r")
        # learning done

        # replace every N steps (Double Q-learning)
        if i_step % DOUBLE_Q_REPLACE_EVERY == 0:
            print("Replace!")
            q_func_old.load_state_dict(q_func_new.state_dict())
            q_func_old.eval()
            q_func_new.train()
        # replace every N steps done

        # logging
        if i_step % 1000 == 0:
            # print("Score Samples", np.array([x[0] for x in replay_queue.ipq.heap[:1000]]))
            # print("(UCB) Arm visit Samples", np.array(replay_queue.ucb.i2n_raw[:1000]))
            print("Saving checkpoints ...")
            torch.save(q_func_new.state_dict(), ROOT / model_filename)
            torch.save(optim.state_dict(), ROOT / optim_filename)
            with (ROOT / "progress.txt").open('w') as fp:
                fp.write(json.dumps({
                    "i_step": i_step,
                }))
            progress_bar.close()
            progress_bar = tqdm.tqdm(total=1000)
        # logging done


def learn(samples, q_func_old, q_func_new, optim, discount):
    q_func_new.eval()
    sp_bef, pa, r, sp_aft, done = zip(*samples)
    s_bef, p_bef = zip(*sp_bef)
    pieces, a_raw = zip(*pa)
    a_raw = torch.tensor(a_raw).to(device)
    s_aft, p_aft = zip(*sp_aft)
    s_bef = torch.hstack((
        torch.tensor(s_bef, dtype=torch.float32),
        torch.nn.functional.one_hot(torch.tensor(p_bef) - 1, num_classes=7),
    )).to(device)
    s_aft = torch.hstack((
        torch.tensor(s_aft, dtype=torch.float32),
        torch.nn.functional.one_hot(torch.tensor(p_aft) - 1, num_classes=7)
    )).to(device)
    targets = torch.tensor(r, dtype=torch.float32).to(device)  # value of future will be added here

    with torch.no_grad():
        done = torch.tensor(done).to(device)

        vs = torch.inf * torch.ones(sum(~done)).to(device)

        pieces_np = np.array(pieces)[~done]
        for mino, actions in MINO2ACTIONS.items():
            _idxs = torch.from_numpy(pieces_np == rust_tetris.piece_kind_str2int(mino))
            n_mino = _idxs.sum()
            if n_mino == 0:
                continue

            # Pure q-learning
            qmax = q_func_old(s_aft[~done][_idxs])[:, :len(actions)].max(dim=1)

            # If Double Q-learning
            future_vals = q_func_new(s_aft[~done][_idxs])
            vs[_idxs] = future_vals.take_along_dim(qmax.indices.reshape(-1, 1), dim=1).reshape(-1)

            # Otherwise
            # vs[_idxs] = qmax.values

            # Testing
            # assert vs[_idxs][0] == future_vals[0, qmax.indices[0]], f"{vs[_idxs][0]} != {future_vals[0, qmax.indices[0]]}"
            # assert vs[_idxs][1] == future_vals[1, qmax.indices[1]], f"{vs[_idxs][1]} != {future_vals[1, qmax.indices[1]]}"
            # assert vs[_idxs][2] == future_vals[2, qmax.indices[2]], f"{vs[_idxs][2]} != {future_vals[2, qmax.indices[2]]}"
            # Testing done

        assert (vs != torch.inf).all(), "Model seems corrupt"
        targets[~done] += discount * vs
    assert (targets != torch.inf).all(), "Missing assignment?"

    # x = q_func_old.feature2advantage[0]._parameters['weight'][0][0]
    q_func_new.train()
    optim.zero_grad()

    cur_vals = q_func_new(s_bef)
    _targets = cur_vals.take_along_dim(a_raw.reshape(-1, 1), dim=1).reshape(-1)
    loss = torch.nn.functional.mse_loss(
        _targets,
        targets, reduction='none'
    )
    loss[done] *= 10  # make `done == True` more important
    # Testing
    # assert _targets[0] == cur_vals[0, a_raw[0]]
    # assert _targets[1] == cur_vals[1, a_raw[1]]
    # assert _targets[2] == cur_vals[2, a_raw[2]]
    # Testing done

    loss_mean = loss.mean()
    loss_mean.backward()
    torch.nn.utils.clip_grad_norm_(q_func_new.parameters(), 10 / LR)
    optim.step()

    # assert q_func_old.feature2advantage[0]._parameters['weight'][0][0] == x, "old q function is updated??"
    assert torch.isfinite(loss_mean)
    return np.sqrt(loss.detach().cpu().numpy()), np.sqrt(float(loss_mean.cpu()))


def report(logger: torch.utils.tensorboard.SummaryWriter,
           g: rust_tetris.Game,
           r: utils.Accumulator,  # reward
           l: utils.Accumulator,  # loss
           rar: float,            # random action ratio
           q,
           i: int,                # episode index
) -> None:
    # logger.add_scalar('Episode/NumberOfLineClear',     g.line_clears, i)
    # logger.add_scalar('Episode/NumberOfTicks',  board.number_of_blocks_dropped(g), i)
    logger.add_scalar('Episode/LossAverage',           l.average,     i)
    logger.add_scalar('Episode/LossMax',               l.max,         i)
    logger.add_scalar('Episode/LossMin',               l.min,         i)
    logger.add_scalar('Episode/NumberOfTicks',         r.n,           i)
    logger.add_scalar('Episode/RewardAverage',         r.average,     i)
    logger.add_scalar('Episode/RewardTotal',           r.sum,         i)
    logger.add_scalar('Episode/RandomActionRatio',     rar,           i)

    # histogram report
    # priorities = np.array([x for x, *_ in q.ipq.heap if x < 1e4])
    # if len(priorities > 0):
    #     logger.add_histogram('ReplayQueue/Scores', priorities,  i)


if __name__ == '__main__':
    main()
