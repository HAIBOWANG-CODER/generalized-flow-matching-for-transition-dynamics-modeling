import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from torch.optim import Adam
from torchdyn.core import NeuralODE
from torchcfm.optimal_transport import OTPlanSampler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from tqdm import tqdm
import wandb
import yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchmetrics.functional import mean_squared_error
from pdb_data.internal import cartesian_to_internal
from pdb_data.internal import internal_to_cartesian
from src.train.parsers import parse_args
from src.resample.weight import resample_trajectory
from src.resample.angles import plot_phi_psi_angles
from src.resample.md_unbiased import plot_paths_energy

wandb.init(project="latent_space_separate")
# =============================================== Load Data ===================================================== #


class sampleLoader:
    def __init__(self, path, data_type, max_atoms=66):
        self.path = path
        self.max_atoms = max_atoms
        self.data_type = data_type

    def load_all_data(self):
        all_coords = torch.load(self.path)

        if self.data_type in ('ADC', 'ADI') and all_coords.size(2) != self.max_atoms:
            raise ValueError("The number of atoms read in is not equal to alanine dipeptide")

        all_coords = np.array(all_coords, dtype=np.float32)

        if self.data_type == 'ADC':
            all_coords = all_coords.squeeze() * 100  # Convert nm to 0.1angstrom
        elif self.data_type == 'ADI':
            # Convert angstrom to nm and transform Cartesian to Internal
            all_coords = cartesian_to_internal(all_coords.squeeze())
            all_coords = all_coords.numpy()
        return all_coords


class SampleDataset(Dataset):
    def __init__(self, directory, data_type, max_atoms=66, vae_key=False, directory1=None):
        self.loader = sampleLoader(directory, data_type, max_atoms=max_atoms)
        self.data = torch.tensor(self.loader.load_all_data(), dtype=torch.float32)
        if vae_key:
            if directory1 is None:
                raise ValueError("vae_key=True and directory1 is not specified")
            self.loader = sampleLoader(directory1, data_type, max_atoms=max_atoms)
            self.data1 = torch.tensor(self.loader.load_all_data(), dtype=torch.float32)
            self.data = torch.cat((self.data, self.data1), dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# =============================================== Train ===================================================== #

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


def dataset_name2datapath(dataset_name, data_on_cluster):

    if data_on_cluster is not None:
        if dataset_name == "alanine":
            path = f"{data_on_cluster}/pdb_data/alanine_data_cartesian"
            return path

        else:
            raise ValueError("Dataset not recognized")
    else:
        raise ValueError("Dataset not recognized")


class SimpleDenseNet(nn.Module):
    def __init__(self, input_size: int, target_size: int, activation: str, batch_norm: bool = False, hidden_dims: List[int] = None):
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

class VelocityNet(SimpleDenseNet):
    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(input_size=dim + 1, target_size=dim, *args, **kwargs)

    def forward(self, t, x, **kwargs):
        if t.dim() < 1 or t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])[:, None]
        if t.dim() < 2:
            t = t[:, None]
        x = torch.cat([t, x], dim=-1)
        device = x.device
        self.model.to(device)
        return self.model(x)

class LineSplineMLP(nn.Module):
    def __init__(self, input_dim: int, activation: str, batch_norm: bool = True, hidden_dims: Optional[List[int]] = None, time_spline: bool = False, time_embedding_type: str = "cat", flatten_input_reshape_output=None):
        super().__init__()
        self.input_dim = input_dim
        self.time_spline = time_spline
        self.mainnet = SimpleDenseNet(
            input_size=2 * input_dim + (1 if time_spline and time_embedding_type == "cat" else 0),
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
            raise NotImplementedError(f"Time embedding type {time_embedding_type} not implemented")
        self.flatten_input_reshape_output = None
        if flatten_input_reshape_output is not None:
            self.flatten_input_reshape_output = flatten_input_reshape_output

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.flatten_input_reshape_output is not None:
            x0 = x0.view(x0.shape[0], -1)
            x1 = x1.view(x1.shape[0], -1)
            t = t.view(t.shape[0], -1)
        out = self._forward_func(x0, x1, t)
        if self.flatten_input_reshape_output is not None:
            out = out.view(out.shape[0], *self.flatten_input_reshape_output)
        return out

    def _forward_cat(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
        if self.time_spline:
            x = torch.cat([x, t], dim=1)
        return self.mainnet(x)

    def _forward_mlp(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
        if self.time_spline:
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

    def _forward_sin(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
        if self.time_spline:
            t_embedded = self.sinusoidal_embedding(t, x.size(1))
            t_embedded = self.time_mlp(t_embedded)
            x = x + t_embedded
        return self.mainnet(x)


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


class spline(nn.Module):
    def __init__(self, model, forward_net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.forward_net = forward_net

        for param in self.model.parameters():
            param.requires_grad = False

    def reparameterize(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x0, x1, t):

        mu0, logvar0 = self.model.encode(x0)
        mu1, logvar1 = self.model.encode(x1)

        z0 = self.reparameterize(mu0, logvar0)
        z1 = self.reparameterize(mu1, logvar1)

        z0 = z0.clone().detach().to(x0.device)
        z1 = z1.clone().detach().to(x0.device)

        zt = slerp(z0, z1, t)
        xt_recon = self.model.decoder(zt)

        xt, d_xt = interpolate(x0, x1, t, self.forward_net)
        return xt_recon, xt, d_xt


def slerp(z0, z1, t):
    """
    Assumes all inputs can be broadcasted to the same shape (..., D).
    Performs the slerp on the last dimension.
    """
    if t.shape != 1:
        t = t.squeeze()
    low_norm = z0 / torch.norm(z0, dim=-1, keepdim=True)
    high_norm = z1 / torch.norm(z1, dim=-1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(-1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - t) * omega) / so).unsqueeze(-1) * z0 + (
        torch.sin(t * omega) / so
    ).unsqueeze(-1) * z1
    return res


def interpolate(x0, x1, t, net):
    x0.requires_grad_(True)
    x1.requires_grad_(True)
    t.requires_grad_(True)
    net_out = net(x0, x1, t)
    net_out.requires_grad_(True)

    d_net = torch.cat([torch.autograd.grad(
        net_out[:, i],
        t,
        grad_outputs=torch.ones_like(net_out[:, i]),
        create_graph=False,
        retain_graph=True,
    )[0] for i in range(net_out.shape[1])], dim=1)

    xt = (1-t)*x0 + t*x1 + t*(1-t)*net_out
    d_xt = x1 - x0 + t*(1-t)*d_net + (1-2*t)*net_out

    return xt, d_xt


def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def merge_config(args, config_updates):
    for key, value in config_updates.items():
        setattr(args, key, value)
    return args

# ======================================== Train vae ================================================== #

def train_vae(vae, train_loader, optimizer, epoch):
    vae.train()
    train_loss = 0
    mse_loss = 0
    kl_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.view(-1, 66)
        data = Variable(data)
        optimizer.zero_grad()
        recon_x, mu, logvar = vae(data)
        MSE, KL = loss_function(recon_x, data, mu, logvar)

        kl = 0.01 * KL
        loss = MSE + kl

        loss.backward()
        train_loss += loss.item()

        mse_loss += MSE.item()
        kl_loss += kl.item()

        optimizer.step()

    num = batch_idx + 1
    train_loss_epoch = train_loss / num
    MSE_epoch = mse_loss / num
    KLD_epoch = kl_loss / num

    wandb.log({
        "VAE/MSE": MSE_epoch,
        "VAE/KL": KLD_epoch,
        "VAE/total_loss": train_loss_epoch,
    })


def train_spline(spline, x0_loader, x1_loader, optimizer, ot_sampler, num_epochs, batch_size):
    spline.train()

    for epoch in tqdm(range(num_epochs), desc="Spline Training Epochs"):
        total_loss_epoch = 0
        l_recon_epoch = 0
        ke_epoch = 0

        for x0_batch, x1_batch in zip(x0_loader, x1_loader):
            optimizer.zero_grad()

            x0_batch = x0_batch.to(next(spline.parameters()).device)
            x1_batch = x1_batch.to(next(spline.parameters()).device)

            t = torch.rand(batch_size).reshape(-1,1).to(x0_batch.device)
            x0, x1 = ot_sampler.sample_plan(x0_batch, x1_batch, replace=True)
            xt_recon, xt, d_xt = spline(x0, x1, t)

            l_recon = mean_squared_error(xt, xt_recon)
            ke = 1/2 * torch.mean((d_xt ** 2).sum(dim=-1))
            total_loss = l_recon + ke

            total_loss.backward()
            optimizer.step()
            total_loss_epoch += total_loss.item()
            l_recon_epoch += l_recon.item()
            ke_epoch += ke.item()

        avg_loss = total_loss_epoch / len(x0_loader)

        avg_recon = l_recon_epoch / len(x0_loader)
        avg_ke = ke_epoch / len(x0_loader)
        wandb.log({
            "SplineNet/l_recon_epoch": avg_recon,
            "SplineNet/ke_epoch": avg_ke,
            "SplineNet/total_loss_epoch": avg_loss,
        })


def train_velocity(x0_loader, x1_loader, spline, velocity_net, ot_sampler, velocity_optimizer, num_epochs, batch_size):

    for epoch in tqdm(range(num_epochs), desc="Velocity Training Epochs"):
        total_loss_epoch = 0

        for x0, x1 in zip(x0_loader, x1_loader):

            x0 = x0.to(next(spline.parameters()).device)
            x1 = x1.to(next(spline.parameters()).device)
            t = torch.rand(batch_size).reshape(-1, 1).to(x0.device)

            x0, x1 = ot_sampler.sample_plan(x0, x1, replace=True)

            _, xt, ut = spline(x0, x1, t)

            vt = velocity_net(t, xt)
            loss = mean_squared_error(vt, ut)

            velocity_optimizer.zero_grad()
            loss.backward()
            velocity_optimizer.step()

            total_loss_epoch += loss.item()

        avg_loss = total_loss_epoch / len(x0_loader)

        # Log the average loss using Weights & Biases (wandb)
        wandb.log({
            "VelocityNet/velocity_total_loss_epoch": avg_loss,
        })


args = parse_args()
if args.config_path:
    config = load_config(args.config_path)

    for experiment_config in config["experiments"]:
        updated_args = merge_config(args, experiment_config)

args.data_path = dataset_name2datapath(args.data_name, args.data_on_cluster)
batch_size = args.batch_size
vae_epochs = args.vae_epochs
spline_epochs = args.epochs
velocity_epochs = args.epochs
data_type = args.data_type

file0 = f"{args.data_path}/x0s.pt"
file1 = f"{args.data_path}/x1s.pt"

vae_dataset = SampleDataset(file0, data_type, max_atoms=args.dim, vae_key=True, directory1=file1)
train_loader = DataLoader(vae_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

encoder = BaseVAE(
    input_size=args.dim,
    target_size=args.latent_dim,
    activation=args.activation_encoder,
    batch_norm=False,
    hidden_dims=args.hidden_dims_encoder,
)

decoder = SimpleDenseNet(
    input_size=args.latent_dim,
    target_size=args.dim,
    activation=args.activation_encoder,
    batch_norm=False,
    hidden_dims=args.hidden_dims_encoder,
)

vae = VAE(encoder=encoder, decoder=decoder)

vae_optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)
for epoch in tqdm(range(1, vae_epochs + 1),desc="VAE Training Epochs"):
    train_vae(vae, train_loader, vae_optimizer, epoch)

# ======================================== Train spline ================================================== #

ot_sampler = OTPlanSampler(method="exact")
spline_net = LineSplineMLP(
    input_dim=args.dim,
    hidden_dims=args.hidden_dims_spline,
    time_spline=args.time_spline,
    activation=args.activation_spline,
    batch_norm=False,
)
spline = spline(
    model=vae,
    forward_net=spline_net,
).to('cuda' if torch.cuda.is_available() else 'cpu')


spline_optimizer = Adam(spline.parameters(), lr=args.spline_lr)

dataset0 = SampleDataset(file0, data_type, max_atoms=args.dim)
x0_loader = DataLoader(dataset0, batch_size=args.batch_size, shuffle=True, drop_last=True)
dataset1 = SampleDataset(file1, data_type, max_atoms=args.dim)
x1_loader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, drop_last=True)

train_spline(spline, x0_loader, x1_loader, spline_optimizer, ot_sampler, num_epochs=spline_epochs, batch_size=batch_size)

# ======================================== Train velocity ================================================== #

velocity_net = VelocityNet(
    dim=args.dim,
    hidden_dims=args.hidden_dims_velocity,
    activation=args.activation_velocity,
    batch_norm=False,
)

velocity_optimizer = Adam(velocity_net.parameters(), lr=args.velocity_lr)
train_velocity(x0_loader, x1_loader, spline, velocity_net, ot_sampler, velocity_optimizer, num_epochs=velocity_epochs, batch_size=batch_size)

# ======================================== neuralODE ================================================== #

class model_torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(t, x)

node = NeuralODE(
            model_torch_wrapper(velocity_net),
            solver="euler",
            sensitivity="adjoint",
            atol=1e-10,
            rtol=1e-10,
        )

loader0 = sampleLoader(file0, data_type, max_atoms=args.dim)
x0 = torch.tensor(loader0.load_all_data(), dtype=torch.float32)

traj = node.trajectory(
                x0,
                t_span=torch.linspace(
                    0, 1, 501
                ),
            )

if data_type == "ADC":
    xts = traj.detach() / 100        # Convert units to nanometers
elif data_type == "ADI":
    # Transform internal coords to Cartesian coords
    xts = internal_to_cartesian(traj.detach())
else:
    raise ValueError(f'Unknown data_type {data_type}')

torch.save(xts, f'{args.save_address}/velocitynet_tensor.pt')

xt, uts, weights = resample_trajectory(xts)  # Computing weights (Cartesian Coords)
torch.save(xt, f'{args.save_address}/xt.pt')

img = mpimg.imread(r"background\background.png")
plt.figure(figsize=(10, 10))
plt.imshow(img, extent=[-np.pi, np.pi, -np.pi, np.pi], alpha=0.7)
plot_phi_psi_angles(xts, 'r', 'Trajectories', point_size=5, skip_index=600)
plot_paths_energy(xt, threshold=2000, last_time_threshold=10000, num_indices=100)

wandb.finish()
