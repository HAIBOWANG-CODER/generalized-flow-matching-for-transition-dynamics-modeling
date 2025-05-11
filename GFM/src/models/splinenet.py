import torch
import torch.nn as nn
from typing import List, Optional
from src.networks.mlp import DeepSetModel
from src.networks.mlp import SimpleDenseNet


class SplineNet(nn.Module):
    def __init__(self, input_dim: int, activation: str, batch_norm: bool = True,
                 hidden_dims: Optional[List[int]] = None, hidden_dim_deepset: Optional[list[int]] = None,
                 time_spline: bool = False, flatten_input_reshape_output=None,):
        super().__init__()
        self.input_dim = input_dim
        self.time_spline = time_spline
        self.deepset = None

        if hidden_dim_deepset is not None:
            self.deepset = DeepSetModel(
                input_size=2 * input_dim,
                target_size=input_dim,
                activation=activation,
                batch_norm=batch_norm,
                hidden_dims=hidden_dim_deepset,
            )
        self.mainnet = SimpleDenseNet(
            input_size=(2 + (1 if hidden_dim_deepset else 0)) * input_dim + 1,
            target_size=input_dim,
            activation=activation,
            batch_norm=batch_norm,
            hidden_dims=hidden_dims,
        )

        self.flatten_input_reshape_output = None
        if flatten_input_reshape_output is not None:
            self.flatten_input_reshape_output = flatten_input_reshape_output

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.flatten_input_reshape_output is not None:
            x0 = x0.view(x0.shape[0], -1)
            x1 = x1.view(x1.shape[0], -1)
            t = t.view(t.shape[0], -1)

        out = self._forward_cat(x0, x1, t)
        if self.flatten_input_reshape_output is not None:
            out = out.view(out.shape[0], *self.flatten_input_reshape_output)
        return out

    def _forward_cat(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=-1)
        if self.deepset is not None:
            deepset_output = self.deepset(x).repeat(x0.size(0), 1)
            x = torch.cat([x, deepset_output], dim=-1)

        if self.time_spline:
            x = torch.cat([x, t], dim=-1)
        return self.mainnet(x)