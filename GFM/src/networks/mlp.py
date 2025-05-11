import torch
import torch.nn as nn
from typing import List, Optional


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "lrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
    "swish": swish,
}


class SimpleDenseNet(nn.Module):
    def __init__(self, input_size: int, target_size: int, activation: str, batch_norm: bool = False,
                 hidden_dims: List[int] = None):
        super().__init__()
        dims = [input_size, *hidden_dims, target_size]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(ACTIVATION_MAP[activation]())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)
        self.target_size = target_size

    def forward(self, x):
        return self.model(x)


class DeepSetModel(SimpleDenseNet):
    def __init__(
        self,
        input_size: int,
        target_size: int,
        activation: str,
        batch_norm: bool = False,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__(input_size, target_size, activation, batch_norm, hidden_dims)
        self.final_activation = ACTIVATION_MAP[activation]()

    def forward(self, x):
        x = super().forward(x)
        x = self.final_activation(x)
        x = x.mean(dim=0, keepdim=True)
        return x