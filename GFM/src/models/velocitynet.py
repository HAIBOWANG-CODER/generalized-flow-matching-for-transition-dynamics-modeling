import torch
import torch.nn as nn
from typing import List, Optional
from src.networks.mlp import SimpleDenseNet


class VelocityNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 activation: str,
                 batch_norm: bool = True,
                 hidden_dims: Optional[List[int]] = None,):

        super().__init__()

        self.model = SimpleDenseNet(
            input_size=input_dim + 1,
            target_size=input_dim,
            activation=activation,
            batch_norm=batch_norm,
            hidden_dims=hidden_dims,
        )

    def forward(self, t, x, **kwargs):

        if t.dim() < 1 or t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])[:, None]
        if t.dim() < 2:
            t = t[:, None]

        x = torch.cat([t, x], dim=-1)
        device = x.device
        self.model.to(device)
        return self.model(x)