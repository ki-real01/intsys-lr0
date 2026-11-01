"""Полносвязная сеть: 64 входа -> n_layers x (Linear + ReLU + Dropout) -> 10 выходов."""
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_layers, hidden, dropout, n_in=64, n_out=10):
        super().__init__()
        layers = []
        width = n_in
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
