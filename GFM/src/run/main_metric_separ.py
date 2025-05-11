import torch
import yaml
import wandb
from torch.optim import Adam
from torchdyn.core import NeuralODE
from torchcfm.optimal_transport import OTPlanSampler
from torch.utils.data import DataLoader
from src.flow_matchers.spline_cfm import SplineConditionalFlowMatcher
from src.parsers import parse_args
from src.geo_metrics.velocity_functions import VelocityOnManifold
from src.flow_matchers.compute_spline_loss import SplineNetTrain
from src.dataloaders.temporal_batch import TemporalDataModule
from src.models.splinenet import SplineNet
from src.models.velocitynet import VelocityNet
from src.evaluate.eval_metric import run_processing_for_pdb
from src.dataloaders.dataload import get_bidirectional_dataloader
from src.dataloaders.dataload import sampleLoader
from src.dataloaders.dataload import SampleDataset
from train_data.toy_data.main_2D import estimation_for_muller
from src.train.train_metric import train_spline
from src.train.train_metric import train_velocity


wandb.init(project="bidirectional")


def dataset_name2datapath(dataset_name, data_on_cluster):
    if data_on_cluster is not None:
        if dataset_name == "alanine":
            path = f"{data_on_cluster}/train_data/pdb_data/alanine_data_cartesian"
            return path

        elif dataset_name == "muller":
            path = f"{data_on_cluster}/train_data/toy_data"
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

    args.data_path = dataset_name2datapath(args.data_name, args.data_on_cluster)
    args.gamma_current = args.gammas[0]
    t_exclude = 1
    epochs = args.epochs
    reflow = args.reflow
    resample = args.resample
    direc = args.direc
    data_type = args.data_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ot_sampler = OTPlanSampler(method="exact")
    data_module_metric = TemporalDataModule(
        args=args,
        skipped_datapoint=t_exclude,
    )
    spline_net = SplineNet(
        input_dim=args.dim,
        hidden_dims=args.hidden_dims_spline,
        hidden_dim_deepset=args.hidden_dim_deepset if args.deepset else None,
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

    train = True

    if train==False:
        # Load trained networks
        spline_net.load_state_dict(torch.load(f"{args.save_address}/spline_net.pth", map_location=device))
        velocity_net.load_state_dict(torch.load(f"{args.save_address}/velocity_net.pth", map_location=device))
        spline_net.eval()
        velocity_net.eval()

    flow_matcher_base = SplineConditionalFlowMatcher(
        spline_net=spline_net,
        sigma=args.sigma,
        alpha=args.alpha,
    )
    velocity_on_manifold = VelocityOnManifold(
        args=args,
        skipped_time_points=[t_exclude],
        datamodule=data_module_metric,
    )
    spline_model = SplineNetTrain(
        flow_matcher=flow_matcher_base,
        skipped_time_points=[t_exclude],
        velocity_on_manifold=velocity_on_manifold,
        args=args,
    )

    spline_net.to(device)
    velocity_net.to(device)

    optimizer_spline = Adam([
        {'params': spline_net.parameters(), 'lr': args.spline_lr},
    ])

    optimizer_velocity = Adam([
        {'params': velocity_net.parameters(), 'lr': args.velocity_lr}
    ])

    if args.data_type in ('ADC', 'ADI'):

        desc = "Train the spline net. Now epoch/Epochs:"

        files0 = f"{args.data_path}/x0s.pt"
        dataset0 = SampleDataset(files0, data_type, max_atoms=args.dim)
        x0_loader = DataLoader(dataset0, batch_size=args.batch_size, shuffle=True, drop_last=True)

        files1 = f"{args.data_path}/x1s.pt"
        dataset1 = SampleDataset(files1, data_type, max_atoms=args.dim)
        x1_loader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, drop_last=True)

        if train:
            train_spline(args,
                        device,
                        ot_sampler,
                        spline_model,
                        optimizer_spline,
                        x0_loader,
                        x1_loader,
                        desc=desc,
                        num_epochs=epochs,
                        batch_size=args.batch_size,
                        direc=direc)

            flow_matcher_base.spline_net = spline_model.spline_net

            train_velocity(ot_sampler,
                           x0_loader,
                           x1_loader,
                           flow_matcher_base,
                           velocity_net,
                           epochs,
                           optimizer_velocity,
                           direc=direc)

            torch.save(spline_net.state_dict(), f"{args.save_address}/spline_net.pth")
            torch.save(spline_model.spline_net.state_dict(), f"{args.save_address}/spline_model_spline_net.pth")
            torch.save(velocity_net.state_dict(), f"{args.save_address}/velocity_net.pth")
        else:
            flow_matcher_base.spline_net = spline_model.spline_net

        traj_path_prefix = f'{args.save_address}'
        # Test the output alanine dipeptide of velocity.
        run_processing_for_pdb(files0, files1, velocity_net, flow_matcher_base,
                               traj_path_prefix, data_type, ot_sampler, t_steps=501, num=0)

        # Reflow for staged experiments
        if reflow:
            reflow_num = args.reflow_num
            for i in range(reflow_num):
                desc = f"Train reflow {i + 1}. Epochs:"
                xt_forward_path = f'{args.save_address}/ode_x_forward.pt'
                xt_backward_path = f'{args.save_address}/ode_x_backward.pt'
                dataloader_forward, dataloader_backward = get_bidirectional_dataloader(xt_forward_path,
                                                                                       xt_backward_path,
                                                                                       batch_size=args.batch_size)
                train_spline(args,
                          device,
                          ot_sampler,
                          spline_model,
                          optimizer_spline,
                          dataloader_forward,
                          dataloader_backward,
                          desc=desc,
                          num_epochs=epochs,
                          reflow=reflow,
                          batch_size=args.batch_size,
                          direc=direc)

                flow_matcher_base.spline_net = spline_model.spline_net

                train_velocity(ot_sampler,
                               dataloader_forward,
                               dataloader_backward,
                               flow_matcher_base,
                               velocity_net,
                               epochs,
                               optimizer_velocity,
                               reflow=True,
                               direc=direc)

                run_processing_for_pdb(files0, files1, velocity_net, flow_matcher_base,
                                       traj_path_prefix, data_type, ot_sampler, t_steps=501, num=i + 1)

        # Resample train
        if resample:
            resample_num = args.resample_num
            for j in range(resample_num):
                desc = f"Train resample {j + 1}. Epochs:"
                x0_loader = DataLoader(dataset0, batch_size=args.batch_size_i, shuffle=True, drop_last=True)
                x1_loader = DataLoader(dataset1, batch_size=args.batch_size_i, shuffle=True, drop_last=True)

                train_spline(args,
                          device,
                          ot_sampler,
                          spline_model,
                          optimizer_spline,
                          x0_loader,
                          x1_loader,
                          desc=desc,
                          num_epochs=args.iter_epochs,
                          resample=resample,
                          batch_size=args.batch_size_i,
                          direc=direc)

                flow_matcher_base.spline_net = spline_model.spline_net

                train_velocity(ot_sampler,
                               x0_loader,
                               x1_loader,
                               flow_matcher_base,
                               velocity_net,
                               epochs,
                               optimizer_velocity)

                run_processing_for_pdb(files0, files1, velocity_net, flow_matcher_base,
                                       traj_path_prefix, data_type, ot_sampler, t_steps=501, num=j + 1)


    elif args.data_type == 'muller':
        desc = "Train toy data. Epochs:"
        toy_fles0 = f"{args.data_path}/x0s.pt"
        toy_fles1 = f"{args.data_path}/x1s.pt"

        dataset0 = SampleDataset(toy_fles0, data_type, max_atoms=args.dim)
        x0_loader = DataLoader(dataset0, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dataset1 = SampleDataset(toy_fles1, data_type, max_atoms=args.dim)
        x1_loader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, drop_last=True)

        train_spline(args,
                    device,
                    ot_sampler,
                    spline_model,
                    optimizer_spline,
                    x0_loader,
                    x1_loader,
                    epochs,
                    desc=desc,
                    batch_size=args.batch_size,
                    direc=direc)

        flow_matcher_base.spline_net = spline_model.spline_net

        loader0 = sampleLoader(toy_fles0, data_type, max_atoms=args.dim)
        x0 = torch.tensor(loader0.load_all_data(), dtype=torch.float32)

        train_velocity(ot_sampler,
                        x0_loader,
                        x1_loader,
                        flow_matcher_base,
                        velocity_net,
                        epochs,
                        optimizer_velocity)

        node = NeuralODE(velocity_net, solver="euler", sensitivity="adjoint", atol=1e-5, rtol=1e-5)
        traj = node.trajectory(x0, t_span=torch.linspace(0, 1, 501))
        torch.save(traj.detach(), f"{args.save_address}/velocity_x.pt")
        # Reading in 2D trajectories data
        estimation_for_muller([
            f"{args.save_address}",
        ], step=1)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)