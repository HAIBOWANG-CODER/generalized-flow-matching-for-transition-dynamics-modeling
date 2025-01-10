import torch
import pytorch_lightning as pl
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torchmetrics.functional import mean_squared_error
from torchdyn.core import NeuralODE
from src.networks.utils import flow_model_torch_wrapper
from src.resample.weight import resample_trajectory
from pdb_data.internal import internal_to_cartesian
from src.resample.angles import plot_phi_psi_angles
from src.resample.md_unbiased import plot_paths_energy
from toy_data.main_2D import calculate_full_tensor_energy_and_plot

class FlowNetTrain(pl.LightningModule):
    def __init__(
        self,
        flow_matcher,
        flow_net,
        skipped_time_points=None,
        ot_sampler=None,
        args=None,
        now_resample=0,
        now_reflow=0,
        direc=None,
    ):
        super().__init__()
        self.flow_matcher = flow_matcher
        self.flow_net = flow_net
        self.ot_sampler = ot_sampler
        self.skipped_time_points = skipped_time_points
        self.micro_batch_size = args.micro_batch_size
        self.ambient_space_metric_only = args.ambient_space_metric_only
        self.OT_in_ambient_space = args.OT_in_ambient_space
        self.save_address = args.save_address

        self.optimizer_name = args.flow_optimizer
        self.lr = args.flow_lr
        self.weight_decay = args.flow_weight_decay
        self.data_type = args.data_type
        self.data_name = args.data_name

        self.now_resample = now_resample     # Current resample number
        self.now_reflow = now_reflow
        self.reflow_num = args.reflow_num
        self.direc = direc

        if args.dataset_num == 2 or args.dataset_num == 3:
            self.skipped_time_points = []

    def forward(self, t, xt):
        return self.flow_net(t, xt)

    def _compute_loss(self, main_batch, metric_samples_batch):
        main_batch_filtered = [
            x[0] for i, x in enumerate(main_batch) if i not in self.skipped_time_points
        ]
        indices = [
            x[1] for i, x in enumerate(main_batch) if i not in self.skipped_time_points
        ]

        x0s, x1s = main_batch_filtered[:-1], main_batch_filtered[1:]
        ts, xts, uts = self._process_flow(x0s, x1s, indices=indices)

        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)
        vt = self(t[:, None], xt.detach())
        loss = mean_squared_error(vt, ut.detach())

        return loss

    def _process_flow(self, x0s, x1s, indices=None, exp_G=False):
        ts, xts, uts = [], [], []
        t_start = self.timesteps[0]
        i_start = 0

        for i, (x0, x1) in enumerate(zip(x0s, x1s)):
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)
            if not exp_G:

                if self.ot_sampler is not None:
                    if self.ambient_space_metric_only and self.OT_in_ambient_space:
                        indices0s, indices1s = indices[0], indices[1]
                        pi = self.ot_sampler.get_map(indices0s, indices1s)
                        i_ot, j_ot = self.ot_sampler.sample_map(
                            pi, x0.shape[0], replace=True
                        )
                        x0 = x0[i_ot]
                        x1 = x1[j_ot]

                    else:
                        x0, x1 = self.ot_sampler.sample_plan(
                            x0,
                            x1,
                            replace=True,
                        )

            if self.skipped_time_points and i + 1 >= self.skipped_time_points[0]:
                t_start_next = self.timesteps[i + 2]
            else:
                t_start_next = self.timesteps[i + 1]

            if x0.size(0) > self.micro_batch_size:
                indices = torch.randperm(x0.size(0))[: self.micro_batch_size]
                x0 = x0[indices]
                x1 = x1[indices]

            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(  # ================================== #
                x0, x1, t_start, t_start_next
            )

            ts.append(t)

            xts.append(xt)
            uts.append(ut)
            t_start = t_start_next

        return ts, xts, uts

    def training_step(self, batch, batch_idx):
        if self.direc == 'unbidirectional':
            main_batch, metric_batch = [], []
            if self.now_reflow == 0:
                main_batch = batch["train_samples"][0]
                metric_batch = batch["metric_samples"][0]
                self.timesteps = torch.linspace(0.0, 1.0, len(main_batch)).tolist()

            else:
                for x in batch["train_samples"][0]:
                    tensor = x[0]
                    indice = x[1]
                    tensor_a = tensor[:, 0, :]
                    tensor_b = tensor[:, 1, :]
                    a = [tensor_a, indice]
                    b = [tensor_b, indice]
                    main_batch = [a, b]

                for m in batch["metric_samples"][0]:
                    tensor_a = m[:, 0, :]
                    tensor_b = m[:, 1, :]
                    metric_batch = [tensor_a, tensor_b]
                self.timesteps = torch.linspace(0.0, 1.0, 2).tolist()

            loss = self._compute_loss(main_batch, metric_batch)

        elif self.direc == 'bidirectional':
            main_batch_f, metric_batch_f = [], []
            main_batch_b, metric_batch_b = [], []
            if self.now_reflow == 0:
                main_batch_f = batch["train_samples"][0]
                metric_batch_f = batch["metric_samples"][0]
                self.timesteps = torch.linspace(0.0, 1.0, len(main_batch_f)).tolist()

                dataloader = self.trainer.datamodule.train_dataloader()
                next_batch = next(iter(dataloader))
                main_batch_b = next_batch[0]["train_samples"][0]
                metric_batch_b = next_batch[0]["metric_samples"][0]

            else:
                for x in batch["train_samples"][0]:
                    tensor = x[0]
                    indice = x[1]
                    tensor_a_f = tensor[:, 0, 0, :]
                    tensor_b_f = tensor[:, 1, 0, :]
                    a_f = [tensor_a_f, indice]
                    b_f = [tensor_b_f, indice]
                    main_batch_f = [a_f, b_f]

                    tensor_a_b = tensor[:, 0, 1, :]
                    tensor_b_b = tensor[:, 1, 1, :]
                    a_b = [tensor_a_b, indice]
                    b_b = [tensor_b_b, indice]
                    main_batch_b = [a_b, b_b]

                for m in batch["metric_samples"][0]:
                    tensor_a_f = m[:, 0, 0, :]
                    tensor_b_f = m[:, 1, 0, :]
                    metric_batch_f = [tensor_a_f, tensor_b_f]

                    tensor_a_b = m[:, 0, 1, :]
                    tensor_b_b = m[:, 1, 1, :]
                    metric_batch_b = [tensor_a_b, tensor_b_b]
                self.timesteps = torch.linspace(0.0, 1.0, 2).tolist()

            loss_f = self._compute_loss(main_batch_f, metric_batch_f)
            loss_b = self._compute_loss(main_batch_b, metric_batch_b)
            loss = loss_f + loss_b
        else:
            raise ValueError(f"The 'direc' setting is not found")

        if self.flow_matcher.alpha != 0:
            self.log(
                "FlowNet/mean_bend_cfm",
                (self.flow_matcher.bend_net_output.abs().mean()),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.log(
            "FlowNet/train_loss_cfm",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = 0
        if self.direc == 'unbidirectional':
            main_batch, metric_batch = [], []
            if self.now_reflow == 0:
                main_batch = batch["val_samples"][0]
                metric_batch = batch["metric_samples"][0]
                self.timesteps = torch.linspace(0.0, 1.0, len(main_batch)).tolist()

            else:
                for x in batch["val_samples"][0]:
                    tensor = x[0]
                    indice = x[1]
                    tensor_a = tensor[:, 0, :]
                    tensor_b = tensor[:, 1, :]
                    a = [tensor_a, indice]
                    b = [tensor_b, indice]
                    main_batch = [a, b]

                for m in batch["metric_samples"][0]:
                    tensor_a = m[:, 0, :]
                    tensor_b = m[:, 1, :]
                    metric_batch = [tensor_a, tensor_b]
                self.timesteps = torch.linspace(0.0, 1.0, 2).tolist()

            val_loss = self._compute_loss(main_batch, metric_batch)

        elif self.direc == 'bidirectional':
            main_batch_f, metric_batch_f = [], []
            main_batch_b, metric_batch_b = [], []
            if self.now_reflow == 0:
                main_batch_f = batch["val_samples"][0]
                metric_batch_f = batch["metric_samples"][0]
                self.timesteps = torch.linspace(0.0, 1.0, len(main_batch_f)).tolist()
                dataloader = self.trainer.datamodule.val_dataloader()
                next_batch = next(iter(dataloader))
                main_batch_b = next_batch[0]["val_samples"][0]
                metric_batch_b = next_batch[0]["metric_samples"][0]

            else:
                for x in batch["val_samples"][0]:
                    tensor = x[0]
                    indice = x[1]
                    tensor_a_f = tensor[:, 0, 0, :]
                    tensor_b_f = tensor[:, 1, 0, :]
                    a_f = [tensor_a_f, indice]
                    b_f = [tensor_b_f, indice]
                    main_batch_f = [a_f, b_f]

                    tensor_a_b = tensor[:, 0, 1, :]
                    tensor_b_b = tensor[:, 1, 1, :]
                    a_b = [tensor_a_b, indice]
                    b_b = [tensor_b_b, indice]
                    main_batch_b = [a_b, b_b]

                for m in batch["metric_samples"][0]:
                    tensor_a_f = m[:, 0, 0, :]
                    tensor_b_f = m[:, 1, 0, :]
                    metric_batch_f = [tensor_a_f, tensor_b_f]

                    tensor_a_b = m[:, 0, 1, :]
                    tensor_b_b = m[:, 1, 1, :]
                    metric_batch_b = [tensor_a_b, tensor_b_b]
                self.timesteps = torch.linspace(0.0, 1.0, 2).tolist()

            loss_f = self._compute_loss(main_batch_f, metric_batch_f)
            loss_b = self._compute_loss(main_batch_b, metric_batch_b)
            val_loss = loss_f + loss_b

        self.log(
            "FlowNet/val_loss_cfm",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return val_loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)


    def test_step(self, batch, batch_idx):
        x0, x1 = [], []
        node = NeuralODE(
            flow_model_torch_wrapper(self.flow_net),
            solver="euler",
            sensitivity="adjoint",
            atol=1e-5,
            rtol=1e-5,
        )

        # Generate the trajectory
        t_exclude = self.skipped_time_points[0] if self.skipped_time_points else 1

        if self.now_reflow == 0:
            x0 = batch[t_exclude - 1]
            x1 = batch[-1]
        else:
            if self.direc == 'unbidirectional':
                x0 = batch[0][:, 0, :]
                x1 = batch[0][:, -1, :]
            elif self.direc == 'bidirectional':
                x0 = batch[0][:, 0, 0, :]
                x1 = batch[0][:, -1, -1, :]

        if t_exclude is not None:
            traj_forward = node.trajectory(
                x0,
                t_span=torch.linspace(
                    self.timesteps[t_exclude - 1], self.timesteps[-1], 501
                ),
            )

            if self.data_name == 'alanine':
                traj_backward = node.trajectory(
                    x1,
                    t_span=torch.linspace(
                        self.timesteps[-1], self.timesteps[t_exclude - 1], 501
                    ),
                )
                # Saved for resample iterative training
                torch.save(traj_forward, f"{self.save_address}//flownet_tensor.pt")

                x0 = x0.unsqueeze(1)
                x1_forward = traj_forward[-1].unsqueeze(1)
                x_forward = torch.cat((x0, x1_forward), dim=1)
                # For reflow iteration and bidirectional experiments
                torch.save(x_forward, f"{self.save_address}//ode_x_forward.pt")

                x1 = x1.unsqueeze(1)
                x0_backward = traj_backward[-1].unsqueeze(1)
                x_backward = torch.cat((x0_backward, x1), dim=1)
                # For bidirectional experiment
                torch.save(x_backward, f"{self.save_address}//ode_x_backward.pt")

                traj = []
                if self.data_type == 'ADC':
                    traj = traj_forward/100     # Convert units from 0.1 angstrom back to nm
                elif self.data_type == 'ADI':
                    traj = internal_to_cartesian(traj_forward)     # Transform internal coords to Cartesian coords
                else:
                    raise ValueError(f'Unknown data_type {self.data_type}')

                # torch.save(traj,
                #            f"{self.save_address}//flownet_tensor_{self.now_resample}_{self.now_reflow}_{self.direc}.pt")
                xts, _, weights = resample_trajectory(traj)  # Computing weights (Cartesian Coords)
                torch.save(weights, f"{self.save_address}//weights.pt")
                # torch.save(xts, f"{self.save_address}//xts_{self.now_resample}_{self.now_reflow}_{self.direc}.pt")

                img = mpimg.imread(r"background\background.png")
                plt.figure(figsize=(10, 10))
                plt.imshow(img, extent=[-np.pi, np.pi, -np.pi, np.pi], alpha=0.7)
                plot_phi_psi_angles(traj, 'r', 'Trajectories', point_size=5, skip_index=600)
                plot_paths_energy(xts, threshold=2000, last_time_threshold=10000, num_indices=100)

            elif self.data_name == 'muller':
                torch.save(traj_forward, f"{self.save_address}//flownet_tensor.pt")
                calculate_full_tensor_energy_and_plot([
                    f"{self.save_address}",
                ], step=1)

            else:
                raise ValueError(f'Unknown data_name {self.data_name}')


    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
            )

        return optimizer
