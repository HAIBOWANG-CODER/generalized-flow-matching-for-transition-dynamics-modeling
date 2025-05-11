import torch
import torch.optim as optim
from torch.optim import Adam
from torchcfm.optimal_transport import OTPlanSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import yaml
from src.networks.mlp import SimpleDenseNet
from src.models.vae import BaseVAE
from src.models.vae import VAE
from src.flow_matchers.latent_interpolate import Spline
from src.models.splinenet import SplineNet
from src.models.velocitynet import VelocityNet
from src.dataloaders.dataload import sampleLoader
from src.dataloaders.dataload import SampleDataset
from src.train.train_latent import train_vae
from src.train.train_latent import train_gfm
from src.evaluate.eval_latent import run_spline_and_ode_pipeline
from src.parsers import parse_args

wandb.init(project="latent_space_simultaneous")

# =============================================== Train ===================================================== #

def dataset_name2datapath(dataset_name, data_on_cluster):

    if data_on_cluster is not None:
        if dataset_name == "alanine":
            path = f"{data_on_cluster}/pdb_data/alanine_data_cartesian"
            return path

        else:
            raise ValueError("Dataset not recognized")
    else:
        raise ValueError("Dataset not recognized")


def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def merge_config(args, config_updates):
    for key, value in config_updates.items():
        setattr(args, key, value)
    return args


def main(args):

    if args.config_path:
        config = load_config(args.config_path)

        for experiment_config in config["experiments"]:
            updated_args = merge_config(args, experiment_config)


    epochs = args.epochs
    args.data_path = dataset_name2datapath(args.data_name, args.data_on_cluster)
    batch_size = args.batch_size
    vae_epochs = args.vae_epochs
    data_type = args.data_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # ======================================== Train vae ================================================== #
    vae = VAE(encoder=encoder, decoder=decoder)
    optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)

    for epoch in tqdm(range(1, vae_epochs + 1), desc="VAE Training Epochs"):
        train_vae(vae, train_loader, optimizer, epoch)

    # ======================================== Train GFM ================================================== #

    ot_sampler = OTPlanSampler(method="exact")
    spline_net = SplineNet(
        input_dim=args.dim,
        hidden_dims=args.hidden_dims_spline,
        time_spline=args.time_spline,
        activation=args.activation_spline,
        batch_norm=False,
    )
    velocity_net = VelocityNet(
        input_dim=args.dim,
        hidden_dims=args.hidden_dims_velocity,
        activation=args.activation_velocity,
        batch_norm=False,
    )
    spline = Spline(
        model=vae,
        forward_net=spline_net,
    ).to(device)

    optimizer = Adam([
        {'params': spline_net.parameters(), 'lr': args.spline_lr},
        {'params': velocity_net.parameters(), 'lr': args.velocity_lr}
    ])

    dataset0 = SampleDataset(file0, data_type, max_atoms=args.dim)
    x0_loader = DataLoader(dataset0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataset1 = SampleDataset(file1, data_type, max_atoms=args.dim)
    x1_loader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, drop_last=True)

    train_gfm(spline,
              velocity_net,
              x0_loader,
              x1_loader,
              batch_size,
              ot_sampler,
              optimizer,
              epochs)

    # ======================================== neuralODE ================================================== #
    spline_net.eval()
    velocity_net.eval()
    run_spline_and_ode_pipeline(args,
                                sampleLoader,
                                data_type,
                                spline_net,
                                velocity_net,)

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)
