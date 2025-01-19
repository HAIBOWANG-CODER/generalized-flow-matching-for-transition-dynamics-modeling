import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, pad_t_like_x


class BendConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(
        self, bend_net: torch.nn.Module = None, alpha: float = 1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

        self.bend_net = bend_net
        if self.alpha != 0:
            assert bend_net is not None, "Bend model must be provided if alpha != 0"

    def gamma(self, t, t_min, t_max):
        return (
            1.0
            - ((t - t_min) / (t_max - t_min)) ** 2
            - ((t_max - t) / (t_max - t_min)) ** 2
        )

    def d_gamma(self, t, t_min, t_max):
        return 2 * (-2 * t + t_max + t_min) / (t_max - t_min) ** 2

    def compute_mu_t(self, x0, x1, t, t_min, t_max):

        with torch.enable_grad():
            t = pad_t_like_x(t, x0)
            if self.alpha == 0:
                return (t_max - t) / (t_max - t_min) * x0 + (t - t_min) / (
                    t_max - t_min
                ) * x1

            self.bend_net_output = self.bend_net(x0, x1, t)

            if self.bend_net.time_bend:
                if self.bend_test == False:
                    self.doutput_dt = torch.autograd.grad(
                        self.bend_net_output,
                        t,
                        grad_outputs=torch.ones_like(self.bend_net_output),
                        create_graph=False,
                        retain_graph=True,
                    )[0]
        return (
            (t_max - t) / (t_max - t_min) * x0
            + (t - t_min) / (t_max - t_min) * x1
            + self.gamma(t, t_min, t_max) * self.bend_net_output
        )

    def sample_xt(self, x0, x1, t, epsilon, t_min, t_max):
        mu_t = self.compute_mu_t(x0, x1, t, t_min, t_max)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def sample_location_and_conditional_flow(
        self,
        x0,
        x1,
        t_min,
        t_max,
        training_bend_net=False,
        midpoint_only=False,
        t=None,
        bend_test=False,
    ):
        # ===================================== #
        self.bend_test = bend_test
        if self.bend_test:
            with torch.no_grad():
                xts = []
                for i in range(t.shape[0]):

                    min_size = min(x0.size(0), x1.size(0))
                    x0 = x0[:min_size]
                    x1 = x1[:min_size]

                    T = t[i] * torch.ones(x0.shape[0], dtype=x0.dtype, device=x0.device)
                    T = T.type_as(x0)
                    xt = self.compute_mu_t(x0, x1, T, t_min, t_max)
                    # eps = self.sample_noise_like(x0)
                    # xt = self.sample_xt(x0, x1, t, eps, t_min, t_max)
                    xts.append(xt)
                xt_bend_traj = torch.stack(xts)
                return xt_bend_traj
        # ===================================== #
        else:
            self.training_bend_net = training_bend_net
            with torch.enable_grad():
                if t is None:
                    t = torch.rand(x0.shape[0], requires_grad=True)
                t = t.type_as(x0)
                t = t * (t_max - t_min) + t_min
                if midpoint_only:
                    t = (t_max + t_min) / 2 * torch.ones_like(t).type_as(x0)
            assert len(t) == x0.shape[0], "t has to have batch size dimension"

            xt = self.compute_mu_t(x0, x1, t, t_min, t_max)
            # eps = self.sample_noise_like(x0)
            # xt = self.sample_xt(x0, x1, t, eps, t_min, t_max)
            ut = self.compute_conditional_flow(x0, x1, t, xt, t_min, t_max)

            return t, xt, ut

    def compute_conditional_flow(self, x0, x1, t, xt, t_min, t_max):
        del xt
        t = pad_t_like_x(t, x0)
        if self.alpha == 0:
            return (x1 - x0) / (t_max - t_min)
        return (
            (x1 - x0) / (t_max - t_min)
            + self.d_gamma(t, t_min, t_max) * self.bend_net_output
            + (
                self.gamma(t, t_min, t_max) * self.doutput_dt
                if self.bend_net.time_bend
                else 0
            )
        )
