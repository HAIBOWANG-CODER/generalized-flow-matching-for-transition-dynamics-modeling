import torch
import torch.nn as nn
import math
import numpy as np
import yaml
from torch.optim import Adam
from torchdyn.core import NeuralODE
from torchcfm.optimal_transport import OTPlanSampler
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchmetrics.functional import mean_squared_error
from src.resample.weight import resample_trajectory
from src.resample.angles import plot_phi_psi_angles
from src.resample.md_unbiased import plot_paths_energy
from src.flow_matchers.models.bend_cfm import BendConditionalFlowMatcher
from src.train.parsers import parse_args
from src.geo_metrics.velocity_functions import VelocityOnManifold
from src.flow_matchers.bend_net_train import BendNetTrain
from src.dataloaders.temporal_batch import TemporalDataModule
from pdb_data.internal import internal_to_cartesian
from pdb_data.internal import cartesian_to_internal
from toy_data.main_2D import calculate_full_tensor_energy_and_plot


wandb.init(project="Generalized_flow_matching_simultaneous_metric")


class BidirectionalDataset(Dataset):
    def __init__(self, xt_path):

        self.xt = torch.load(xt_path)

    def __len__(self):
        return self.xt.shape[0]

    def __getitem__(self, idx):
        return self.xt[idx]


def get_bidirectional_dataloader(xt_forward_path, xt_backward_path, batch_size=32, shuffle=False):
    dataset_forward = BidirectionalDataset(xt_forward_path)
    dataset_backward = BidirectionalDataset(xt_backward_path)

    dataloader_forward = DataLoader(dataset_forward, batch_size=batch_size, shuffle=shuffle)
    dataloader_backward = DataLoader(dataset_backward, batch_size=batch_size, shuffle=shuffle)
    return dataloader_forward, dataloader_backward


class sampleLoader:
    def __init__(self, path, data_type, max_atoms=66):
        self.path = path
        self.max_atoms = max_atoms
        self.data_type = data_type

    def load_all_data(self):
        all_coords = torch.load(self.path)

        if self.data_type in ('ADC', 'ADI') and all_coords.size(2) != self.max_atoms:
            raise ValueError("The number of atoms read in is not equal to alanine dipeptide")
        if self.data_type == 'muller' and all_coords.size(1) != self.max_atoms:
            raise ValueError("The dimension of muller brown data read in is not equal to 2")

        all_coords = np.array(all_coords, dtype=np.float32)

        if self.data_type == 'ADC':
            all_coords = all_coords.squeeze() * 100  # Convert nm to 0.1angstrom
        elif self.data_type == 'ADI':
            # Convert angstrom to nm and transform Cartesian to Internal
            all_coords = cartesian_to_internal(all_coords.squeeze())
            all_coords = all_coords.numpy()

        elif self.data_type == 'muller':
            all_coords = all_coords
        else:
            raise ValueError("Data type not recognized")
        return all_coords


class SampleDataset(Dataset):
    def __init__(self, directory, data_type, max_atoms=66):
        self.loader = sampleLoader(directory, data_type, max_atoms=max_atoms)
        self.data = torch.tensor(self.loader.load_all_data(), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
    def __init__(self, input_dim: int, activation: str, batch_norm: bool = True, hidden_dims: Optional[List[int]] = None, time_bend: bool = False, time_embedding_type: str = "cat", flatten_input_reshape_output=None):
        super().__init__()
        self.input_dim = input_dim
        self.time_bend = time_bend
        self.mainnet = SimpleDenseNet(
            input_size=2 * input_dim + (1 if time_bend and time_embedding_type == "cat" else 0),
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
        if self.time_bend:
            x = torch.cat([x, t], dim=1)
        return self.mainnet(x)

    def _forward_mlp(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
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

    def _forward_sin(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
        if self.time_bend:
            t_embedded = self.sinusoidal_embedding(t, x.size(1))
            t_embedded = self.time_mlp(t_embedded)
            x = x + t_embedded
        return self.mainnet(x)


def dataset_name2datapath(dataset_name, data_on_cluster):

    if data_on_cluster is not None:
        if data_on_cluster is not None:
            if dataset_name == "alanine":
                path = f"{data_on_cluster}/pdb_data/alanine_data_cartesian"
                return path

            elif dataset_name == "muller":
                path = f"{data_on_cluster}/toy_data"
                return path
        else:
            raise ValueError("Dataset not recognized")
    else:
        raise ValueError("Dataset not recognized")

def process_trajectory(flow_net, flow_matcher_base, x0, x1, t_steps, traj_path_prefix, num, data_type):
    """
    Process forward and backward trajectories, visualize, and save key results.

    Parameters:
    - flow_net: Neural network for flow computation.
    - flow_matcher_base: Base object for flow matching.
    - x0, x1: Initial and final states.
    - t_steps: Number of time steps for trajectory generation.
    - traj_path_prefix: Directory prefix for saving trajectory data.
    """
    # Generate time points
    t = torch.linspace(0, 1, steps=t_steps)

    # Generate trajectory using flow matching
    xt = flow_matcher_base.sample_location_and_conditional_flow(x0, x1, 0, 1, t=t, bend_test=True)

    if data_type == 'ADC':
        xt = xt / 100  # Convert units from 0.1 angstrom back to nm
    elif data_type == 'ADI':
        xt = internal_to_cartesian(xt)  # Transform internal coords to Cartesian coords
    else:
        raise ValueError(f'Unknown data_type {data_type}')
    torch.save(xt.detach(), f'{traj_path_prefix}/splinenet_tensor_{num}.pt')

    # Visualize phi-psi angles
    img = mpimg.imread(r"background/background.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(img, extent=[-np.pi, np.pi, -np.pi, np.pi], alpha=0.7)
    plot_phi_psi_angles(xt.detach(), 'r', 'Trajectories', point_size=5, skip_index=600)

    # Compute the best energy of spline output
    xts_b, _, _ = resample_trajectory(xt.detach())
    torch.save(xts_b, f'{traj_path_prefix}/xts_b_{num}.pt')
    plot_paths_energy(xts_b, threshold=200000, last_time_threshold=100000, num_indices=100)

    # Use NeuralODE to compute forward and backward trajectories
    node = NeuralODE(flow_net, solver="euler", sensitivity="adjoint", atol=1e-5, rtol=1e-5)
    traj_forward = node.trajectory(x0, t_span=torch.linspace(0, 1, t_steps))
    traj_backward = node.trajectory(x1, t_span=torch.linspace(1, 0, t_steps))

    # Save forward and backward trajectories
    save_forward_and_backward_trajectories(traj_forward, traj_backward, x0, x1, traj_path_prefix)

    # Process and save flow-matching trajectory
    if data_type == 'ADC':
        traj_forward_nm = traj_forward / 100
    elif data_type == 'ADI':
        traj_forward_nm = internal_to_cartesian(traj_forward)
    else:
        raise ValueError(f'Unknown data_type {data_type}')
    torch.save(traj_forward_nm.detach(), f'{traj_path_prefix}/flownet_tensor_{num}.pt')

    # Visualize phi-psi angles and energy paths
    plt.figure(figsize=(10, 10))
    plt.imshow(img, extent=[-np.pi, np.pi, -np.pi, np.pi], alpha=0.7)
    plot_phi_psi_angles(traj_forward_nm.detach(), 'r', 'Trajectories', point_size=5, skip_index=600)

    xts, uts, weights = resample_trajectory(traj_forward_nm.detach())
    # Saved for resample iterative training
    torch.save(traj_forward, f'{traj_path_prefix}/flownet_tensor.pt')
    torch.save(weights, f'{traj_path_prefix}/weights.pt')

    torch.save(xts, f'{traj_path_prefix}/xts_{num}.pt')
    plot_paths_energy(xts, threshold=200000, last_time_threshold=1000000, num_indices=100)


def save_forward_and_backward_trajectories(traj_forward, traj_backward, x0, x1, traj_path_prefix):
    """
    Save forward and backward trajectories for bidirectional experiments.

    Parameters:
    - traj_forward, traj_backward: Generated trajectories.
    - x0, x1: Initial and final states.
    - traj_path_prefix: Directory prefix for saving trajectory data.
    """
    # Prepare and save forward trajectory
    x0 = x0.unsqueeze(1)
    x1_forward = traj_forward[-1].unsqueeze(1)
    x_forward = torch.cat((x0, x1_forward), dim=1)
    torch.save(x_forward, f'{traj_path_prefix}/ode_x_forward.pt')

    # Prepare and save backward trajectory
    x1 = x1.unsqueeze(1)
    x0_backward = traj_backward[-1].unsqueeze(1)
    x_backward = torch.cat((x0_backward, x1), dim=1)
    torch.save(x_backward, f'{traj_path_prefix}/ode_x_backward.pt')


def run_processing_for_pdb(files0, files1, flow_net, flow_matcher_base, traj_path_prefix, data_type, t_steps=501, num=0):
    """
    Main function to load data, process trajectories, and save results.

    Parameters:
    - files0, files1: Paths to input alanine dipeptide data files for states x0 and x1.
    - flow_net: Neural network for flow computation.
    - flow_matcher_base: Base object for flow matching.
    - traj_path_prefix: Directory prefix for saving trajectory data.
    - t_steps: Number of time steps for trajectory generation.
    """
    with torch.no_grad():
        # Load data from alanine dipeptide data files
        loader0 = sampleLoader(files0, data_type)
        x0 = torch.tensor(loader0.load_all_data(), dtype=torch.float32)
        loader1 = sampleLoader(files1, data_type)
        x1 = torch.tensor(loader1.load_all_data(), dtype=torch.float32)

        # Process trajectories
        process_trajectory(flow_net, flow_matcher_base, x0, x1, t_steps, traj_path_prefix, num, data_type)


def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def merge_config(args, config_updates):
    for key, value in config_updates.items():
        setattr(args, key, value)
    return args


def train_gfm(
            spline_model,
              flow_net,
              optimizer,
              x0_loader,
              x1_loader,
              num_epochs=0,
              desc=None,
              converge=False,
              reflow=False,
              resample=False,
              batch_size=None,
              direc="unbidirectional"
              ):

    spline_model.train()
    flow_net.train()

    if resample:
        paths = torch.load(f"{args.save_address}/flownet_tensor.pt")
        weights = torch.load(f"{args.save_address}/weights.pt")
        T = torch.linspace(0, 1, paths.size(0))

    n = 1
    for epoch in tqdm(range(num_epochs), desc=desc):
        total_spline_loss = 0
        total_velocity_loss = 0
        total_loss = 0

        for x0_batch, x1_batch in zip(x0_loader, x1_loader):
            # Train spline_model
            optimizer.zero_grad()

            if reflow:
                x0 = x0_batch[:, 0, :]
                x1T = x0_batch[:, 1, :]

                x0T = x1_batch[:, 0, :]
                x1 = x1_batch[:, 1, :]
            else:
                x0_batch = x0_batch.to(next(spline_model.parameters()).device)
                x1_batch = x1_batch.to(next(spline_model.parameters()).device)
                x0_batch, x1_batch = ot_sampler.sample_plan(x0_batch, x1_batch, replace=True)

                x0 = x0_batch
                x1 = x1_batch

                if n % 2 == 1 or converge:
                    if direc == "bidirectional":
                        x0_batch1, x1_batch1 = next(zip(x0_loader, x1_loader))
                        x0_batch1, x1_batch1 = ot_sampler.sample_plan(x0_batch1, x1_batch1, replace=True)
                        x1T = x1_batch1
                        x0T = x0_batch1

                        # x1T = x1_batch
                        # x0T = x0_batch

                    elif direc == "unbidirectional":
                        x1T = x1_batch
                    else:
                        raise ValueError(f"Don't know the direction")

                else:
                    with torch.no_grad():
                        node = NeuralODE(
                            flow_net,
                            solver="euler",
                            sensitivity="adjoint",
                            atol=1e-5,
                            rtol=1e-5,
                        )
                        traj_forward = node.trajectory(
                            x0,
                            t_span=torch.linspace(
                                0, 1, 501
                            ),
                        )
                        x1T = traj_forward[-1].detach()
                        del traj_forward

                        if direc == "bidirectional":
                            traj_backard = node.trajectory(
                                x1,
                                t_span=torch.linspace(
                                    1, 0, 501
                                ),
                            )
                            x0T = traj_backard[-1].detach()
                            del traj_backard
                n += 1

            indice = torch.randint(0, x0_batch.size(0), (x0_batch.size(0),))

            t = torch.rand(batch_size).reshape(-1, 1).to(x0_batch.device)
            t.requires_grad_(True)

            if direc == "bidirectional":
                # Spline
                loss0 = spline_model._compute_loss([[x0, indice], [x1T, indice]], [x0, x1], exp_G=True)
                loss1 = spline_model._compute_loss([[x0T, indice], [x1, indice]], [x0, x1], exp_G=True)
                spline_loss = loss0 + loss1
                # Velocity
                ts0, xts0, uts0 = spline_model._process_flow([x0], [x1T], t, exp_G=True)
                ts1, xts1, uts1 = spline_model._process_flow([x0T], [x1], t, exp_G=True)
                vt0 = flow_net(ts0[0].detach(), xts0[0].detach())
                vt1 = flow_net(ts1[0].detach(), xts1[0].detach())
                velocity_loss = mean_squared_error(vt0,
                                                   uts0[0].detach()) + mean_squared_error(vt1, uts1[0].detach())

            elif direc == "unbidirectional":
                # Spline
                if resample:
                    loss0 = spline_model._compute_loss([[x0, indice], [x1T, indice]], [x0, x1],
                                                     paths=paths, weights=weights, T=T,
                                                     batch=batch_size, exp_G=True, resample=resample)
                else:
                    loss0 = spline_model._compute_loss([[x0, indice], [x1T, indice]], [x0, x1], exp_G=True)
                spline_loss = loss0
                # Velocity
                ts0, xts0, uts0 = spline_model._process_flow([x0], [x1T], t, exp_G=True)
                vt0 = flow_net(ts0[0].detach(), xts0[0].detach())
                velocity_loss = mean_squared_error(vt0, uts0[0].detach())
            else:
                raise ValueError(f"Don't know the direction, so can't compute loss")

            loss = spline_loss + velocity_loss

            loss.backward()
            optimizer.step()

            wandb.log({
                "spline_step": spline_loss,
                "velocity_step": velocity_loss,
                "loss_step": loss,
            })

            total_spline_loss += spline_loss
            total_velocity_loss += velocity_loss
            total_loss += loss

        spline_epoch = total_spline_loss / len(x0_loader)
        vel_epoch = total_velocity_loss / len(x0_loader)
        loss_epoch = total_loss / len(x0_loader)

        wandb.log({
            "spline_epoch": spline_epoch,
            "velocity_epoch": vel_epoch,
            "loss_epoch": loss_epoch,
            "epochs": epoch,
        })


args = parse_args()

if args.config_path:
    config = load_config(args.config_path)
    for experiment_config in config["experiments"]:
        updated_args = merge_config(args, experiment_config)

args.data_path = dataset_name2datapath(args.data_name, args.data_on_cluster)
args.gamma_current = args.gammas[0]
t_exclude = 1
epochs = args.epochs
converge = args.converge
reflow = args.reflow
resample = args.resample
direc = args.direc
data_type = args.data_type

ot_sampler = OTPlanSampler(method="exact")
data_module_spline = TemporalDataModule(
    args=args,
    skipped_datapoint=t_exclude,
)
spline_net = LineSplineMLP(
    input_dim=args.dim,
    hidden_dims=args.hidden_dims_bend,
    time_bend=args.time_bend,
    activation=args.activation_bend,
    batch_norm=False,
)
flow_matcher_base = BendConditionalFlowMatcher(
    spline_net=spline_net,
    sigma=args.sigma,
    alpha=args.alpha,
)
velocity_on_manifold = VelocityOnManifold(
    args=args,
    skipped_time_points=[t_exclude],
    datamodule=data_module_spline,
)
spline_model = BendNetTrain(
    flow_matcher=flow_matcher_base,
    skipped_time_points=[t_exclude],
    ot_sampler=ot_sampler,
    velocity_on_manifold=velocity_on_manifold,
    args=args,
)
flow_net = VelocityNet(
    dim=args.dim,
    hidden_dims=args.hidden_dims_flow,
    activation=args.activation_flow,
    batch_norm=False,
)

optimizer = Adam([
    {'params': spline_net.parameters(), 'lr': args.spline_lr},
    {'params': flow_net.parameters(), 'lr': args.flow_lr}
])

if args.data_name == "alanine":
    desc = "Train alanine dipeptide data. Epochs:"
    files0 = f"{args.data_path}/x0s.pt"
    dataset0 = SampleDataset(files0, data_type, max_atoms=args.dim)
    x0_loader = DataLoader(dataset0, batch_size=args.batch_size, shuffle=True, drop_last=True)

    files1 = f"{args.data_path}/x1s.pt"
    dataset1 = SampleDataset(files1, data_type, max_atoms=args.dim)
    x1_loader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    train_gfm(spline_model,
              flow_net,
              optimizer,
              x0_loader,
              x1_loader,
              desc=desc,
              num_epochs=epochs,
              converge=converge,
              batch_size=args.batch_size,
              direc=direc)
    traj_path_prefix = args.save_address
    # Test the output alanine dipeptide of velocity.
    run_processing_for_pdb(files0, files1, flow_net, flow_matcher_base,
                   traj_path_prefix, data_type, t_steps=501, num=0)
    
    # Reflow for staged experiments
    if converge and reflow:
        reflow_num = args.reflow_num
        for i in range(reflow_num):
            desc = f"Train reflow {i+1}. Epochs:"
            xt_forward_path = f"{args.save_address}/ode_x_forward.pt"
            xt_backward_path = f"{args.save_address}/ode_x_backward.pt"
            dataloader_forward, dataloader_backward = get_bidirectional_dataloader(xt_forward_path,
                                                                                   xt_backward_path,
                                                                                   batch_size=args.batch_size)
            train_gfm(spline_model,
                        flow_net,
                        optimizer,
                        dataloader_forward,
                        dataloader_backward,
                        desc=desc,
                        num_epochs=epochs,
                        converge=converge,
                        reflow=reflow,
                        batch_size=args.batch_size,
                        direc=direc)
            run_processing_for_pdb(files0, files1, flow_net, flow_matcher_base,
                           traj_path_prefix, data_type, t_steps=501, num=i+1)

    # Resample train
    if resample:
        resample_num = args.resample_num
        for j in range(resample_num):
            desc = f"Train resample {j+1}. Epochs:"
            x0_loader = DataLoader(dataset0, batch_size=args.batch_size_i, shuffle=True, drop_last=True)
            x1_loader = DataLoader(dataset1, batch_size=args.batch_size_i, shuffle=True, drop_last=True)

            train_gfm(spline_model,
                      flow_net,
                      optimizer,
                      x0_loader,
                      x1_loader,
                      desc=desc,
                      num_epochs=args.iter_epochs,
                      converge=converge,
                      resample=resample,
                      batch_size=args.batch_size_i,
                      direc=direc)

            traj_path_prefix = f"{args.save_address}"
            run_processing_for_pdb(files0, files1, flow_net, flow_matcher_base,
                           traj_path_prefix, data_type, t_steps=501, num=j+1)


elif args.data_name == 'muller':
    desc = "Train toy data. Epochs:"
    toy_fles0 = f"{args.data_path}/x0s.pt"
    toy_fles1 = f"{args.data_path}/x1s.pt"
    dataset0 = SampleDataset(toy_fles0, data_type, max_atoms=args.dim)
    x0_loader = DataLoader(dataset0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset1 = SampleDataset(toy_fles1, data_type, max_atoms=args.dim)
    x1_loader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, drop_last=True)

    train_gfm(spline_model,
              flow_net,
              optimizer,
              x0_loader,
              x1_loader,
              num_epochs=epochs,
              desc=desc,
              converge=converge,
              batch_size=args.batch_size,
              direc=direc)

    loader0 = sampleLoader(toy_fles0, data_type, max_atoms=args.dim)
    x0 = torch.tensor(loader0.load_all_data(), dtype=torch.float32)
    node = NeuralODE(flow_net, solver="euler", sensitivity="adjoint", atol=1e-5, rtol=1e-5)
    traj = node.trajectory(x0, t_span=torch.linspace(0, 1, 501))
    torch.save(traj.detach(), f"{args.save_address}/ode_x.pt")
    # Reading in 2D trajectories data
    calculate_full_tensor_energy_and_plot([
        f"{args.save_address}",
    ], step=1)

wandb.finish()
