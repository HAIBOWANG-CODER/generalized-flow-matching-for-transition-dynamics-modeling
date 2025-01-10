from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchcfm.optimal_transport import OTPlanSampler
import wandb
import secrets
import yaml
import string
import torch
from src.flow_matchers.models.bend_cfm import BendConditionalFlowMatcher
from src.geo_metrics.velocity_functions import VelocityOnManifold
from src.flow_matchers.flow_net_train import FlowNetTrain
from src.flow_matchers.bend_net_train import BendNetTrain
from src.dataloaders.temporal_batch import TemporalDataModule
from src.networks.flow_networks.mlp import VelocityNet
from src.networks.bend_networks.mlp import LineBendMLP
from src.utils import set_seed
from src.train.parsers import parse_args


def main(args):
    read_model = False
    device = torch.device("cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu")

    group_name = generate_group_string()
    args.data_path = dataset_name2datapath(args.data_name, args.data_on_cluster)
    if not args.reflow:
        args.reflow_num = 0
    if not args.resample:
        args.resample_num = 0

    for seed in args.seeds:

        for i, t_exclude in enumerate(args.t_exclude):
            set_seed(seed)
            args.t_exclude_current = t_exclude
            args.seed_current = seed
            args.gamma_current = args.gammas[i]

            for j in range(args.resample_num + 1):             # The resampling iteration loop
                for n in range(args.reflow_num + 1):            # The reflow iteration loop

                    flow_net = VelocityNet(
                        dim=args.dim,
                        hidden_dims=args.hidden_dims_flow,
                        activation=args.activation_flow,
                        batch_norm=False,
                    )

                    ot_sampler = OTPlanSampler(method=args.optimal_transport_method)

                    bend_net = LineBendMLP(
                        input_dim=args.dim,
                        hidden_dims=args.hidden_dims_bend,
                        hidden_dim_deepset=args.hidden_dim_deepset if args.deepset else None,
                        time_bend=args.time_bend,
                        activation=args.activation_bend,
                        batch_norm=False,
                        time_embedding_type=args.time_embedding_type,
                    )

                    wandb.init(
                        project=f"bend-cfm-{args.data_type}-{args.data_name}",
                        group=group_name,
                        config=vars(args),
                        dir=args.logs_dir,
                    )

                    flow_matcher_base = BendConditionalFlowMatcher(
                        bend_net=bend_net,
                        sigma=args.sigma,
                        alpha=args.alpha,
                    )

                    if read_model == False:
                        if args.alpha != 0:

                            data_module_bend = TemporalDataModule(
                                args=args,
                                now_resample=j,    # The current resample iteration number
                                now_reflow=n,       # The current reflow iteration number
                                skipped_datapoint=t_exclude,
                                direc=args.direc,
                            )
                            velocity_on_manifold = VelocityOnManifold(
                                args=args,
                                skipped_time_points=[t_exclude],
                                datamodule=data_module_bend,
                                now_reflow=n,       # The current reflow iteration number
                            )
                            early_stop_callback = EarlyStopping(
                                monitor="BendNet/val_loss_bend" if j == 0 else "iteration_BendNet/iter_val_loss_bend",
                                patience=args.patience_bend,
                                verbose=True,
                                mode="min",
                            )
                            checkpoint_callback = ModelCheckpoint(
                                dirpath=args.logs_dir,
                                monitor="BendNet/val_loss_bend" if j == 0 else "iteration_BendNet/iter_val_loss_bend",
                                mode="min",
                                save_top_k=1,
                                verbose=True,
                            )
                            bend_model = BendNetTrain(
                                flow_matcher=flow_matcher_base,
                                skipped_time_points=[t_exclude],
                                ot_sampler=ot_sampler,
                                velocity_on_manifold=velocity_on_manifold,
                                args=args,
                                now_resample=j,  # The current resample iteration number
                                now_reflow=n,     # The current reflow iteration number
                                direc=args.direc,
                            )

                            wandb_logger = WandbLogger()

                            trainer = Trainer(
                                max_epochs=args.epochs if j == 0 else args.iter_epochs,  # First train epoch and iteration epochs
                                check_val_every_n_epoch=args.check_val_every_n_epoch,
                                callbacks=[early_stop_callback, checkpoint_callback],
                                accelerator=args.accelerator,
                                logger=wandb_logger,
                                num_sanity_val_steps=0,
                                default_root_dir=args.logs_dir,
                            )
                            trainer.fit(bend_model, datamodule=data_module_bend)

                            test_spline = False
                            if test_spline:
                                trainer.test(bend_model, datamodule=data_module_bend)

                            best_model_path = checkpoint_callback.best_model_path
                            bend_model = BendNetTrain.load_from_checkpoint(best_model_path)

                            flow_matcher_base.bend_net = bend_model.bend_net
                    else:

                        best_model_path = r"scratch\data\~"
                        print(f'Reading {best_model_path}')
                        bend_model = BendNetTrain.load_from_checkpoint(best_model_path)
                        bend_model = bend_model.to(device)

                        flow_matcher_base.bend_net = bend_model.bend_net

                    data_module = TemporalDataModule(
                        args=args,
                        skipped_datapoint=t_exclude,
                        now_reflow=n,       # The current reflow iteration number
                        direc=args.direc,
                    )

                    early_stop_callback = EarlyStopping(
                        monitor="FlowNet/val_loss_cfm",
                        patience=args.patience,
                        verbose=False,
                        mode="min",
                    )

                    checkpoint_callback = ModelCheckpoint(
                        dirpath=args.logs_dir,
                        mode="min",
                        save_top_k=1,
                        verbose=True,
                    )

                    flow_matcher = FlowNetTrain(
                        flow_matcher=flow_matcher_base,
                        flow_net=flow_net,
                        ot_sampler=ot_sampler,
                        skipped_time_points=[t_exclude],
                        args=args,
                        now_resample=j,    # The current resample iteration number
                        now_reflow=n,       # The current reflow iteration number
                        direc=args.direc,
                    )

                    wandb_logger = WandbLogger()

                    trainer = Trainer(
                        max_epochs=args.epochs if j == 0 else args.iter_epochs,
                        callbacks=[early_stop_callback, checkpoint_callback],
                        check_val_every_n_epoch=args.check_val_every_n_epoch,
                        accelerator=args.accelerator,
                        logger=wandb_logger,
                        default_root_dir=args.logs_dir,
                    )

                    trainer.fit(flow_matcher, datamodule=data_module)
                    trainer.test(flow_matcher, datamodule=data_module)

            wandb.finish()


def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def merge_config(args, config_updates):
    for key, value in config_updates.items():
        setattr(args, key, value)
    return args


def generate_group_string(length=16):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def dataset_name2datapath(dataset_name, data_on_cluster):

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


if __name__ == "__main__":
    args = parse_args()
    if args.config_path:
        config = load_config(args.config_path)

        for experiment_config in config["experiments"]:
            updated_args = merge_config(args, experiment_config)
            main(updated_args)
    else:
        main(args)
