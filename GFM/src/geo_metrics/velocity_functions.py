import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader
from src.geo_metrics.rbf_learned import RBFNetwork


class VelocityOnManifold:
    def __init__(
        self,
        args,
        skipped_time_points=None,
        datamodule=None,
        now_reflow=0,
    ):
        self.skipped_time_points = skipped_time_points
        self.datamodule = datamodule

        self.gamma = args.gamma_current
        self.rho = args.rho
        self.n_centers = args.n_centers
        self.kappa = args.kappa
        self.clustering_method = args.clustering_method
        self.variance_aggregation = args.variance_aggregation
        self.metric_epochs = args.metric_epochs
        self.metric_patience = args.metric_patience
        self.lr = args.metric_lr
        self.alpha_metric = args.alpha_metric
        self.image_hx = args.image_hx
        self.now_reflow = now_reflow

        self.called_first_time = True if self.now_reflow == 0 else False

        if args.dataset_num == 2 or args.dataset_num == 3:
            self.skipped_time_points = []

    def calculate_metric(self, x_t, current_timestep):
        M_dd_x_t = None
        if self.called_first_time:
            self.rbf_networks = []
            for timestep in range(self.datamodule.num_timesteps - 1):
                if timestep in self.skipped_time_points:
                    continue
                print("Learning RBF networksss, timestep: ", timestep)
                if self.image_hx:
                    rbf_network = RBFNetwork(
                        input_dim=x_t.shape[1],
                        current_timestep=timestep,
                        next_timestep=timestep
                        + 1
                        + (1 if timestep + 1 in self.skipped_time_points else 0),
                        n_centers=self.n_centers,
                        kappa=self.kappa,
                        clustering_method=self.clustering_method,
                        variance_aggregation=self.variance_aggregation,
                        lr=self.lr,
                        datamodule=self.datamodule,
                    )
                    trainer = pl.Trainer(
                        max_epochs=self.metric_epochs,
                        accelerator=(
                            "gpu"
                            if self.datamodule.device == "cuda"
                            else self.datamodule.device
                        ),
                        logger=WandbLogger(),
                        num_sanity_val_steps=0,
                    )

                    all_data = self.datamodule.all_data
                    self.dataloader = DataLoader(
                        TensorDataset(all_data, self.datamodule.manifold_weights),
                        batch_size=128,
                        shuffle=True,
                    )
                    self.dataloader.x0 = self.datamodule.train_x0
                    self.dataloader.x1 = self.datamodule.train_x1
                    trainer.fit(rbf_network, self.dataloader)

                    self.rbf_networks.append(rbf_network)
                else:
                    rbf_network = RBFNetwork(
                        input_dim=x_t.shape[1],
                        current_timestep=timestep,
                        next_timestep=timestep
                        + 1
                        + (1 if timestep + 1 in self.skipped_time_points else 0),
                        n_centers=self.n_centers,
                        kappa=self.kappa,
                        clustering_method=self.clustering_method,
                        variance_aggregation=self.variance_aggregation,
                        lr=self.lr,
                    )
                    early_stop_callback = pl.callbacks.EarlyStopping(
                        monitor="MetricModel/val_loss_learn_metric",
                        patience=self.metric_patience,
                        verbose=False,
                        mode="min",
                    )
                    device = "gpu" if torch.cuda.is_available() else "cpu"
                    trainer = pl.Trainer(
                        max_epochs=self.metric_epochs,
                        accelerator=device,
                        logger=WandbLogger(),
                        num_sanity_val_steps=0,
                        callbacks=[early_stop_callback],
                    )
                    trainer.fit(rbf_network, self.datamodule)
                    self.rbf_networks.append(rbf_network)
                    torch.save(rbf_network.state_dict(), f"RBF.pth")
            self.called_first_time = False
            print("Learning RBF networksss... Done")
        if self.now_reflow != 0:
            rbf_network = RBFNetwork(
                input_dim=x_t.shape[1],
                current_timestep=0,
                next_timestep=1,
                n_centers=self.n_centers,
                kappa=self.kappa,
                clustering_method=self.clustering_method,
                variance_aggregation=self.variance_aggregation,
                lr=self.lr,
                now_reflow=self.now_reflow,
            )
            self.rbf_networks = []
            rbf_network.load_state_dict(torch.load(f"RBF.pth"))
            self.rbf_networks.append(rbf_network)

        M_dd_x_t = self.rbf_networks[current_timestep].compute_metric(
            x_t,
            epsilon=self.rho,
            alpha=self.alpha_metric,
            image_hx=self.image_hx,
        )
        return M_dd_x_t

    def calculate_velocity(self, x_t, u_t, timestep):

        if len(u_t.shape) > 2:
            u_t = u_t.reshape(u_t.shape[0], -1)
            x_t = x_t.reshape(x_t.shape[0], -1)
        M_dd_x_t = self.calculate_metric(x_t, timestep)

        velocity = torch.sqrt(((u_t**2) * M_dd_x_t).sum(dim=-1))
        return velocity
