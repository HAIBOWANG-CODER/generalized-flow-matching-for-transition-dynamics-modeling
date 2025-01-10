import torch
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torchmetrics.functional import mean_squared_error
from pdb_data.internal import internal_to_cartesian
from src.resample.angles import plot_phi_psi_angles
from src.resample.weight import resample_trajectory
from src.resample.md_unbiased import plot_paths_energy

class BendNetTrain(pl.LightningModule):
    def __init__(
        self,
        flow_matcher,
        args,
        skipped_time_points: list = None,
        ot_sampler=None,
        velocity_on_manifold=None,
        now_resample=0,
        now_reflow=0,
        direc=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.flow_matcher = flow_matcher
        self.bend_net = flow_matcher.bend_net
        self.ot_sampler = ot_sampler
        self.skipped_time_points = skipped_time_points if skipped_time_points else []
        self.optimizer_name = args.bend_optimizer
        self.lr = args.spline_lr
        self.weight_decay = args.bend_weight_decay
        self.args = args
        self.velocity_on_manifold = velocity_on_manifold
        self.multiply_validation = args.multiply_validation
        self.micro_batch_size = args.micro_batch_size
        self.ambient_space_metric_only = args.ambient_space_metric_only
        self.OT_in_ambient_space = args.OT_in_ambient_space
        self.data_type = args.data_type
        self.save_address = args.save_address

        self.now_resample = now_resample   # Current resample iteration number
        self.now_reflow = now_reflow
        self.direc = direc

        self.first_loss = None
        self.timesteps = None
        self.computing_reference_loss = False

        if args.dataset_num == 2 or args.dataset_num == 3:
            self.skipped_time_points = []

    def forward(self, x0, x1, t):
        return self.bend_net(x0, x1, t)

    def on_train_start(self):
        self.first_loss = self.compute_initial_loss()

    def compute_initial_loss(self):
        print("Computing reference loss....")
        self.bend_net.train(mode=False)
        total_loss = 0
        total_count = 0
        with torch.enable_grad():
            self.t_val = []
            for i in range(
                self.trainer.datamodule.num_timesteps - len(self.skipped_time_points)
            ):
                self.t_val.append(
                    torch.rand(
                        self.trainer.datamodule.batch_size * self.multiply_validation,
                        requires_grad=True,
                    )
                )
            if self.now_resample != 0:    # loading resample iteration paths and weights
                self.paths = torch.load(f"{self.save_address}//flownet_tensor.pt")
                self.weights = torch.load(f"{self.save_address}//weights.pt")
                self.T = torch.linspace(0, 1, self.paths.size(0))

        self.computing_reference_loss = True
        with torch.no_grad():
            old_alpha = self.flow_matcher.alpha
            self.flow_matcher.alpha = 0
            for batch in self.trainer.datamodule.train_dataloader():

                if self.direc == 'unbidirectional':
                    if self.now_reflow == 0:
                        self.timesteps = torch.linspace(
                            0.0, 1.0, len(batch[0]["train_samples"][0])
                        )
                        # train_samples = batch[0]["train_samples"][0],
                        # metric_samples = batch[0]["metric_samples"][0],

                        loss = self._compute_loss(
                            batch[0]["train_samples"][0],
                            batch[0]["metric_samples"][0],
                        )

                    else:
                        self.timesteps = torch.linspace(
                            0.0, 1.0, 2
                        )
                        for x in batch[0]["train_samples"][0]:
                            tensor = x[0]
                            indice = x[1]

                            tensor_a = tensor[:, 0, :]
                            tensor_b = tensor[:, 1, :]
                            a = [tensor_a, indice]
                            b = [tensor_b, indice]
                            train_samples = [a, b]

                        metric_samples = []
                        for m in batch[0]["metric_samples"][0]:
                            tensor_a = m[:, 0, :]
                            tensor_b = m[:, 1, :]
                            metric_samples = [tensor_a, tensor_b]

                        loss = self._compute_loss(train_samples, metric_samples)

                elif self.direc == 'bidirectional':
                    if self.now_reflow == 0:
                        self.timesteps = torch.linspace(
                            0.0, 1.0, len(batch[0]["train_samples"][0])
                        )
                        # train_samples_f = batch[0]["train_samples"][0],
                        # metric_samples_f = batch[0]["metric_samples"][0],
                        loss_f = self._compute_loss(
                            batch[0]["train_samples"][0],
                            batch[0]["metric_samples"][0],
                        )

                        dataloader = self.trainer.datamodule.train_dataloader()
                        next_batch = next(iter(dataloader))
                        # train_samples_b = next_batch[0]["train_samples"][0],
                        # metric_samples_b = next_batch[0]["metric_samples"][0],
                        loss_b = self._compute_loss(
                            next_batch[0]["train_samples"][0],
                            next_batch[0]["metric_samples"][0],
                        )

                    else:
                        for x in batch[0]["train_samples"][0]:
                            tensor = x[0]
                            indice = x[1]
                            tensor_a_f = tensor[:, 0, 0, :]
                            tensor_b_f = tensor[:, 1, 0, :]
                            a_f = [tensor_a_f, indice]
                            b_f = [tensor_b_f, indice]
                            train_samples_f = [a_f, b_f]

                            tensor_a_b = tensor[:, 0, 1, :]
                            tensor_b_b = tensor[:, 1, 1, :]
                            a_b = [tensor_a_b, indice]
                            b_b = [tensor_b_b, indice]
                            train_samples_b = [a_b, b_b]

                        for m in batch[0]["metric_samples"][0]:
                            tensor_a_f = m[:, 0, 0, :]
                            tensor_b_f = m[:, 1, 0, :]
                            metric_samples_f = [tensor_a_f, tensor_b_f]

                            tensor_a_b = m[:, 0, 1, :]
                            tensor_b_b = m[:, 1, 1, :]
                            metric_samples_b = [tensor_a_b, tensor_b_b]
                        self.timesteps = torch.linspace(0.0, 1.0, 2).tolist()

                        loss_f = self._compute_loss(train_samples_f, metric_samples_f)
                        loss_b = self._compute_loss(train_samples_b, metric_samples_b)

                    loss = loss_f + loss_b

                total_loss += loss.item()
                total_count += 1
            self.flow_matcher.alpha = old_alpha
        self.computing_reference_loss = False
        self.bend_net.train(mode=True)
        print("Reference loss computed: ", total_loss / total_count)
        return total_loss / total_count if total_count > 0 else 1.0

    def _compute_loss(
            self,
            main_batch,
            metric_samples_batch,
            paths=None,
            weights=None,
            T=None,
            batch=None,
            exp_G=False,
            resample=False,
    ):
        main_batch_filtered = [
            x[0].to(self.device)
            for i, x in enumerate(main_batch)
            if i not in self.skipped_time_points
        ]
        indices = [
            x[1].to(self.device)
            for i, x in enumerate(main_batch)
            if i not in self.skipped_time_points
        ]
        metric_samples_batch_filtered = [
            x.to(self.device)
            for i, x in enumerate(metric_samples_batch)
            if i not in self.skipped_time_points
        ]

        x0s, x1s = main_batch_filtered[:-1], main_batch_filtered[1:]
        samples0, samples1 = (
            metric_samples_batch_filtered[:-1],
            metric_samples_batch_filtered[1:],
        )

        if exp_G and resample:
            self.now_resample = 1

        if self.now_resample == 0:   # Tran spline for the first time
            t = None
            ts, xts, uts = self._process_flow(x0s, x1s, t, indices, exp_G)

            velocities = []
            matrics = []
            for i in range(len(ts)):
                samples = torch.cat([samples0[i], samples1[i]], dim=0)
                vel, matric = self.velocity_on_manifold.calculate_velocity(
                    xts[i], uts[i], samples, i
                )
                matric_norm = torch.norm(matric).unsqueeze(0)
                velocities.append(vel)
                matrics.append(matric_norm)

            mean_matric = torch.mean(torch.cat(matrics))

            loss = torch.mean(torch.cat(velocities) ** 2)
            self.log(
                "BendNet/mean_velocity_bend",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            self.log(
                "BendNet/matrics_norm_bend",
                mean_matric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        # =================================== Resample Iterative Train ========================================== #
        else:
            if exp_G:

                self.paths = paths
                self.weights = weights
                self.T = T
                path_index = torch.multinomial(self.weights, batch, replacement=True)
                tensor = torch.ones(self.paths.shape[0])
                time_step = torch.multinomial(tensor.float(), batch, replacement=True)

            else:
                path_index = torch.multinomial(self.weights, self.trainer.datamodule.batch_size, replacement=True)
                tensor = torch.ones(self.paths.shape[0])
                time_step = torch.multinomial(tensor.float(), self.trainer.datamodule.batch_size, replacement=True)

            path = self.paths[:, path_index, :]
            t = self.T[time_step]
            t.requires_grad_(True)

            Xt = path[time_step, torch.arange(time_step.size(0)), :]

            ts, xts, uts = self._process_flow(x0s, x1s, t, indices, exp_G)
            xt = torch.cat(xts)
            ut = torch.cat(uts)
            Xt = Xt.to(xt.device)

            loss = mean_squared_error(xt, Xt) + torch.mean((ut**2).sum(dim=-1))

            self.log(
                "iteration_BendNet/iter_mean_bend_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def _process_flow(self, x0s, x1s, t=None, indices=None, exp_G=False):
        ts, xts, uts = [], [], []
        self.timesteps = torch.linspace(0.0, 1.0, 2)
        t_start = self.timesteps[0]

        for i, (x0, x1) in enumerate(zip(x0s, x1s)):
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)

            if not exp_G:
                if self.now_resample == 0:                  # Tran spline for the first time in resample experiments
                    if self.trainer.validating or self.computing_reference_loss:
                        repeat_tuple = (self.multiply_validation, 1) + (1,) * (
                            len(x0.shape) - 2
                        )
                        x0 = x0.repeat(repeat_tuple)
                        x1 = x1.repeat(repeat_tuple)
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

            if self.now_resample == 0:
                t = None
                if not exp_G:
                    if (self.trainer.validating or self.computing_reference_loss):
                        t = self.t_val[i]

                if x0.size(0) > self.micro_batch_size:
                    indices = torch.randperm(x0.size(0))[: self.micro_batch_size]
                    x0 = x0[indices]
                    x1 = x1[indices]
                    if t is not None:
                        with torch.enable_grad():
                            t = t[indices]

            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                x0, x1, t_start, t_start_next, training_bend_net=True, t=t
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

            tangential_velocity_loss = self._compute_loss(main_batch, metric_batch)

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
            tangential_velocity_loss = loss_f + loss_b

        else:
            raise ValueError(f"The 'direc' setting is not found")

        if self.first_loss:
            tangential_velocity_loss = tangential_velocity_loss / self.first_loss

        if self.now_resample == 0:
            describe_0 = "BendNet/mean_bend_bend"
            describe_1 = "BendNet/train_loss_bend"
        else:
            describe_0 = "iteration_BendNet/iter_mean_bend_bend"
            describe_1 = "iteration_BendNet/iter_train_loss_bend"
        self.log(
            describe_0,
            (self.flow_matcher.bend_net_output.abs().mean()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            describe_1,
            tangential_velocity_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return tangential_velocity_loss

    def validation_step(self, batch, batch_idx):
        if self.direc == 'unbidirectional':
            main_batch, metric_batch = [], []
            if self.now_reflow == 0:
                main_batch = batch["val_samples"][0]
                metric_batch = batch["metric_samples"][0]

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
            tangential_velocity_loss = self._compute_loss(main_batch, metric_batch)

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
            tangential_velocity_loss = loss_f + loss_b

        else:
            raise ValueError(f"The 'direc' setting is not found")

        if self.first_loss:
            tangential_velocity_loss = tangential_velocity_loss / self.first_loss

        if self.now_resample == 0:
            describe = "BendNet/val_loss_bend"
        else:
            describe = "iteration_BendNet/iter_val_loss_bend"

        self.log(
            describe,
            tangential_velocity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return tangential_velocity_loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def test_step(self, batch, batch_idx):

        # Generate the trajectory
        t_exclude = self.skipped_time_points[0] if self.skipped_time_points else 1
        if t_exclude is not None:
            t = torch.linspace(0, 1, steps=501)

            if self.now_reflow == 0:
                x0 = batch[t_exclude - 1]
                x1 = batch[t_exclude]
            else:
                x0 = batch[0][:, 0, :]
                x1 = batch[0][:, 1, :]

            traj = self.flow_matcher.sample_location_and_conditional_flow(
                x0, x1, 0, 1, t=t, bend_test=True
            )
            
            if self.data_type == 'ADI':
                traj = internal_to_cartesian(traj)
            elif self.data_type == 'ADC':
                traj = traj / 100     # Convert units from 0.1 angstrom back to nm

            xts, _, weights = resample_trajectory(traj)  # Computing weights (Cartesian Coords)
            img = mpimg.imread(r"background/background.png")
            plt.figure(figsize=(10, 10))
            plt.imshow(img, extent=[-np.pi, np.pi, -np.pi, np.pi], alpha=0.7)
            plot_phi_psi_angles(traj, 'r', 'Trajectories', point_size=5, skip_index=600)
            plot_paths_energy(xts, threshold=2000, last_time_threshold=10000, num_indices=100)

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.bend_net.parameters(),
                lr=self.lr,
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.bend_net.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        return {
            'optimizer': optimizer,
            # 'gradient_clip_val': 0.5,
        }
