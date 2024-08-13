import torch
import torchvision

from . import board
from .action_cluster import ACTION_SIZE


class LinearDDQN(torch.nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.input2feature = torchvision.ops.MLP(in_channels, [1024,] * 4,)
        self.feature2value = torchvision.ops.MLP(1024, [64, 16, 1])
        self.feature2advantage = torchvision.ops.MLP(1024, [64, ACTION_SIZE])

    def forward(self, x):
        feat = self.input2feature(x)
        v = self.feature2value(feat)
        a = self.feature2advantage(feat)
        q = v + a - a.mean(axis=1)[:, None]
        return q


class ConvDDQN(torch.nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.board2feature = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (3, 3), padding=(1, 1), padding_mode='zeros'),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, (3, 3)),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, (3, 3)),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, (3, 3)),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, (3, 3)),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, (14, 2)),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
        )
        self.queue2feature = torchvision.ops.MLP((1 + board.N_QUEUE) * 7, [32] * 2,)
        self.feature2value = torchvision.ops.MLP(32 + 32, [64, 64, 32, 16, 1])
        self.feature2advantage = torchvision.ops.MLP(32 + 32, [64, 64, ACTION_SIZE])

    def forward(self, x):
        _board = x[:, :220].reshape(-1, 1, 22, 10) + 1
        flat = self.board2feature(_board).reshape(-1, 32)
        queue = self.queue2feature(x[:, 220:])
        feat = torch.hstack((flat, queue))
        v = self.feature2value(feat)
        a = self.feature2advantage(feat)
        q = v + a - a.mean(axis=1)[:, None]
        return q
