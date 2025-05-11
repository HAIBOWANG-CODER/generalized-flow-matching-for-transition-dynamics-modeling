import torch
import torch.nn as nn


class Spline(nn.Module):
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
