import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from src.networks.mlp import SimpleDenseNet


class BaseVAE(SimpleDenseNet):
    def __init__(self, input_size: int, target_size: int, activation: str, batch_norm: bool = False, hidden_dims: List[int] = None):
        super().__init__(input_size=input_size, activation=activation, target_size=hidden_dims[-1], hidden_dims=hidden_dims[:-1])
        self.fc_mu = nn.Linear(hidden_dims[-1], target_size)
        self.fc_var = nn.Linear(hidden_dims[-1], target_size)

    def forward(self, x):
        latent = self.model(x)
        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)
        return mu, log_var

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encode = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

    return MSE, KLD