import torch
import pytorch_lightning as pl
from torchmetrics.functional import mean_squared_error


class SplineNetTrain(pl.LightningModule):
    def __init__(
            self,
            flow_matcher,
            args,
            skipped_time_points: list = None,
            velocity_on_manifold=None,
            now_resample=0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.flow_matcher = flow_matcher
        self.spline_net = flow_matcher.spline_net
        self.skipped_time_points = skipped_time_points if skipped_time_points else []
        self.lr = args.spline_lr
        self.args = args
        self.velocity_on_manifold = velocity_on_manifold
        self.micro_batch_size = args.micro_batch_size
        self.now_resample = now_resample  # Current resample iteration number

        self.timesteps = None

        if args.dataset_num == 2 or args.dataset_num == 3:
            self.skipped_time_points = []

    def forward(self, x0, x1, t):
        return self.bend_net(x0, x1, t)

    def _compute_loss(
            self,
            main_batch,
            paths=None,
            weights=None,
            T=None,
            batch=None,
            resample=False,
    ):

        x0s, x1s = main_batch[:-1], main_batch[1:]
        if resample:
            self.now_resample = 1

        if self.now_resample == 0:  # Train spline for the first time
            t = None
            ts, xts, uts = self._process_flow(x0s, x1s, t)

            velocities = []
            for i in range(len(ts)):
                vel = self.velocity_on_manifold.calculate_velocity(
                    xts[i], uts[i], i
                )
                velocities.append(vel)
            loss = torch.mean(torch.cat(velocities) ** 2)
            self.log(
                "SplineNet/spline_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        # =================================== Resample Iterative Train ========================================== #
        else:
            self.paths = paths
            self.weights = weights
            self.T = T
            path_index = torch.multinomial(self.weights, batch, replacement=True)
            tensor = torch.ones(self.paths.shape[0])
            time_step = torch.multinomial(tensor.float(), batch, replacement=True)

            path = self.paths[:, path_index, :]
            t = self.T[time_step]
            t.requires_grad_(True)

            Xt = path[time_step, torch.arange(time_step.size(0)), :]

            ts, xts, uts = self._process_flow(x0s, x1s, t)
            xt = torch.cat(xts)
            ut = torch.cat(uts)
            Xt = Xt.to(xt.device)

            loss = mean_squared_error(xt, Xt) + torch.mean((ut ** 2).sum(dim=-1))

            self.log(
                "SplineNet/resample_spline_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def _process_flow(self, x0s, x1s, t=None):
        ts, xts, uts = [], [], []
        self.timesteps = torch.linspace(0.0, 1.0, 2)
        t_start = self.timesteps[0]
        i_start = 0

        for i, (x0, x1) in enumerate(zip(x0s, x1s)):
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)

            if self.skipped_time_points and i + 1 >= self.skipped_time_points[0]:
                t_start_next = self.timesteps[i + 2]
            else:
                t_start_next = self.timesteps[i + 1]

            if self.now_resample == 0:
                t = None

                if x0.size(0) > self.micro_batch_size:
                    indices = torch.randperm(x0.size(0))[: self.micro_batch_size]
                    x0 = x0[indices]
                    x1 = x1[indices]

            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                x0, x1, t_start, t_start_next, training_spline_net=True, t=t
            )

            ts.append(t)
            xts.append(xt)
            uts.append(ut)
            t_start = t_start_next

        return ts, xts, uts
