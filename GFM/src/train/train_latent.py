import torch
import wandb
from tqdm import tqdm
from torch.autograd import Variable
from src.models.vae import loss_function
from torchmetrics.functional import mean_squared_error


def train_vae(vae, train_loader, optimizer, epoch):
    vae.train()
    train_loss = 0
    mse_loss = 0
    kl_loss = 0
    batch_idx = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.view(-1, 498)
        data = Variable(data)
        optimizer.zero_grad()
        recon_x, mu, logvar = vae(data)
        MSE, KL = loss_function(recon_x, data, mu, logvar)

        kl = 0.01 * KL
        loss = MSE + kl

        loss.backward()
        train_loss += loss.item()

        mse_loss += MSE.item()
        kl_loss += kl.item()

        optimizer.step()

    num = batch_idx + 1
    train_loss_epoch = train_loss / num
    MSE_epoch = mse_loss / num
    KLD_epoch = kl_loss / num

    wandb.log({
        "VAE/MSE": MSE_epoch,
        "VAE/KL": KLD_epoch,
        "VAE/total_loss": train_loss_epoch,
    })


def train_spline(spline, x0_loader, x1_loader, optimizer, ot_sampler, num_epochs, batch_size):
    spline.train()
    epoch = 0
    avg_loss = 0
    avg_recon = 0
    avg_ke = 0

    for epoch in tqdm(range(num_epochs), desc="Spline Training Epochs"):
        total_loss_epoch = 0
        l_recon_epoch = 0
        ke_epoch = 0

        for x0_batch, x1_batch in zip(x0_loader, x1_loader):
            optimizer.zero_grad()

            x0_batch = x0_batch.to(next(spline.parameters()).device)
            x1_batch = x1_batch.to(next(spline.parameters()).device)

            t = torch.rand(batch_size).reshape(-1,1).to(x0_batch.device)
            x0, x1 = ot_sampler.sample_plan(x0_batch, x1_batch, replace=True)
            xt_recon, xt, d_xt = spline(x0, x1, t)

            l_recon = mean_squared_error(xt, xt_recon)
            ke = 1/2 * torch.mean((d_xt ** 2).sum(dim=-1))
            total_loss = l_recon + ke

            total_loss.backward()
            optimizer.step()
            total_loss_epoch += total_loss.item()
            l_recon_epoch += l_recon.item()
            ke_epoch += ke.item()

        avg_loss = total_loss_epoch / len(x0_loader)

        avg_recon = l_recon_epoch / len(x0_loader)
        avg_ke = ke_epoch / len(x0_loader)
        wandb.log({
            "SplineNet/train_l_recon_epoch": avg_recon,
            "SplineNet/train_ke_epoch": avg_ke,
            "SplineNet/train_total_loss_epoch": avg_loss,
        })

    # 打印训练和验证损失
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KE: {avg_ke:.4f}")


def train_velocity(x0_loader, x1_loader, spline, velocity_net, ot_sampler, velocity_optimizer, num_epochs, batch_size):

    for epoch in tqdm(range(num_epochs), desc="Velocity Training Epochs"):
        total_loss_epoch = 0

        for x0, x1 in zip(x0_loader, x1_loader):

            x0 = x0.to(next(spline.parameters()).device)
            x1 = x1.to(next(spline.parameters()).device)
            t = torch.rand(batch_size).reshape(-1, 1).to(x0.device)

            x0, x1 = ot_sampler.sample_plan(x0, x1, replace=True)

            _, xt, ut = spline(x0, x1, t)

            vt = velocity_net(t, xt)
            loss = mean_squared_error(vt, ut)

            velocity_optimizer.zero_grad()
            loss.backward()
            velocity_optimizer.step()

            total_loss_epoch += loss.item()

        avg_loss = total_loss_epoch / len(x0_loader)

        # 记录到 wandb
        wandb.log({
            "VelocityNet/train_velocity_total_loss_epoch": avg_loss,
        })

        # 打印训练和验证损失
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_loss:.4f}")


def train_gfm(spline, velocity_net, x0_loader, x1_loader, batch_size, ot_sampler, optimizer, num_epochs):
    spline.train()

    for epoch in tqdm(range(num_epochs), desc="Train latent gfm. Epochs:"):
        total_loss_epoch = 0
        l_recon_epoch = 0
        ke_epoch = 0

        for x0_batch, x1_batch in zip(x0_loader, x1_loader):
            optimizer.zero_grad()

            x0_batch = x0_batch.to(next(spline.parameters()).device)
            x1_batch = x1_batch.to(next(spline.parameters()).device)

            t = torch.rand(batch_size).reshape(-1,1).to(x0_batch.device)
            x0, x1 = ot_sampler.sample_plan(x0_batch, x1_batch, replace=True)
            xt_recon, xt, ut = spline(x0, x1, t)
            # spline
            l_recon = mean_squared_error(xt, xt_recon)
            ke = 1/2 * torch.mean((ut ** 2).sum(dim=-1))
            spline_loss = 100*l_recon + ke
            # velocity
            vt = velocity_net(t, xt.detach())
            velocity_loss = mean_squared_error(vt, ut.detach())

            loss = spline_loss + velocity_loss

            loss.backward()
            optimizer.step()
            total_loss_epoch += spline_loss.item()
            l_recon_epoch += l_recon.item()
            ke_epoch += ke.item()

        avg_loss = total_loss_epoch / len(x0_loader)

        avg_recon = l_recon_epoch / len(x0_loader)
        avg_ke = ke_epoch / len(x0_loader)
        wandb.log({
            "SplineNet/l_recon_epoch": avg_recon,
            "SplineNet/ke_epoch": avg_ke,
            "SplineNet/total_loss_epoch": avg_loss,
        })