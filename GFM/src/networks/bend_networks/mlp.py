import torch
import torch.nn as nn
from typing import List, Optional
import math

from src.networks.base import SimpleDenseNet, DeepSetModel, ACTIVATION_MAP


class LineBendMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        activation: str,
        batch_norm: bool = True,
        hidden_dims: Optional[List[int]] = None,
        hidden_dim_deepset: Optional[int] = None,
        time_bend: bool = False,
        time_embedding_type: str = "cat",
        flatten_input_reshape_output=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_bend = time_bend
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
            input_size=(2 + (1 if hidden_dim_deepset else 0)) * input_dim
            + (1 if time_bend and time_embedding_type == "cat" else 0),
            target_size=input_dim,
            activation=activation,
            batch_norm=batch_norm,
            hidden_dims=hidden_dims,
        )
        if time_embedding_type == "cat":
            self._forward_func = self._forward_cat
        elif time_embedding_type == "mlp":
            self._forward_func = self._forward_mlp
            self.time_mlp = nn.Sequential(
                nn.Linear(1, input_dim),
                ACTIVATION_MAP[activation](),
                nn.Linear(input_dim, hidden_dims[0]),
            )
        elif time_embedding_type == "sin":
            self._forward_func = self._forward_sin
            self.time_mlp = nn.Sequential(
                nn.Linear(2 * input_dim, 2 * input_dim),
                ACTIVATION_MAP[activation](),
                nn.Linear(2 * input_dim, 2 * input_dim),
            )
        else:
            raise NotImplementedError(
                f"Time embedding type {time_embedding_type} not implemented"
            )
        self.flatten_input_reshape_output = None
        if flatten_input_reshape_output is not None:
            self.flatten_input_reshape_output = flatten_input_reshape_output

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if self.flatten_input_reshape_output is not None:
            x0 = x0.view(x0.shape[0], -1)
            x1 = x1.view(x1.shape[0], -1)
            t = t.view(t.shape[0], -1)
        out = self._forward_func(x0, x1, t)
        if self.flatten_input_reshape_output is not None:
            out = out.view(out.shape[0], *self.flatten_input_reshape_output)
        return out

    def _forward_cat(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
        if self.deepset is not None:
            deepset_output = self.deepset(x).repeat(x0.size(0), 1)
            x = torch.cat([x, deepset_output], dim=1)
        if self.time_bend:
            x = torch.cat([x, t], dim=1)
        return self.mainnet(x)

    def _forward_mlp(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
        if self.deepset is not None:
            deepset_output = self.deepset(x).repeat(x0.size(0), 1)
            x = torch.cat([x, deepset_output], dim=1)
        if self.time_bend:
            t_embedded = self.time_mlp(t)
            for i, layer in enumerate(self.mainnet.model):
                if isinstance(layer, nn.Linear):
                    x = layer(x)
                    if i != len(self.mainnet.model) - 1:
                        x += t_embedded
                else:
                    x = layer(x)
        else:
            x = self.mainnet(x)
        return x

    def sinusoidal_embedding(self, t, num_features):
        position = t  # Make it (batch_size, 1)
        div_term = torch.exp(
            torch.arange(0, num_features, 2).float()
            * -(math.log(10000.0) / num_features)
        )
        pe = torch.zeros(t.size(0), num_features)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _forward_sin(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
        if self.deepset is not None:
            deepset_output = self.deepset(x).repeat(x0.size(0), 1)
            x = torch.cat([x, deepset_output], dim=1)
        if self.time_bend:
            t_embedded = self.sinusoidal_embedding(t, x.size(1))
            t_embedded = self.time_mlp(t_embedded)
            x = x + t_embedded
        return self.mainnet(x)
