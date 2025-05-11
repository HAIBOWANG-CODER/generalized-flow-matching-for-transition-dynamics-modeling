import torch
import wandb
from tqdm import tqdm
from torchdyn.core import NeuralODE
from torchmetrics.functional import mean_squared_error


def train_spline(
        args,
        device,
        ot_sampler,
        spline_model,
        optimizer,
        x0_loader,
        x1_loader,
        num_epochs=0,
        desc=None,
        reflow=False,
        resample=False,
        batch_size=None,
        direc="unbidirectional"
):

    spline_model.train()
    spline_model.to(device)

    paths = 0
    weights = 0
    T = 0
    best_val_loss = float("inf")  # 用于保存最佳验证损失
    best_model_path = f"{args.save_address}/best_spline_model.pth"
    if resample:
        paths = torch.load(f'{args.save_address}/velocitynet_tensor.pt')
        weights = torch.load(f'{args.save_address}/weights.pt')
        T = torch.linspace(0, 1, paths.size(0))

    for epoch in tqdm(range(num_epochs), desc=desc):
        total_spline_loss = 0

        for x0_batch, x1_batch in zip(x0_loader, x1_loader):
            # Train spline_model
            optimizer.zero_grad()

            if reflow:
                x0 = x0_batch[:, 0, :]
                x1T = x0_batch[:, 1, :]

                x0T = x1_batch[:, 0, :]
                x1 = x1_batch[:, 1, :]
            else:
                x0_batch = x0_batch.to(next(spline_model.parameters()).device)
                x1_batch = x1_batch.to(next(spline_model.parameters()).device)
                x0_batch, x1_batch = ot_sampler.sample_plan(x0_batch, x1_batch, replace=True)

                x0 = x0_batch
                x1 = x1_batch

                if direc == "bidirectional":
                    x0_batch1, x1_batch1 = next(zip(x0_loader, x1_loader))
                    x0_batch1, x1_batch1 = ot_sampler.sample_plan(x0_batch1, x1_batch1, replace=True)
                    x1T = x1_batch1
                    x0T = x0_batch1

                elif direc == "unbidirectional":
                    x1T = x1_batch
                else:
                    raise ValueError(f"Don't know the direction")

            if direc == "bidirectional":
                loss0 = spline_model._compute_loss([x0, x1T])
                loss1 = spline_model._compute_loss([x0T, x1])
                spline_loss = loss0 + loss1

            elif direc == "unbidirectional":
                if resample:
                    loss0 = spline_model._compute_loss([x0, x1T], paths=paths, weights=weights, T=T,
                                                       batch=batch_size, resample=resample)
                else:
                    loss0 = spline_model._compute_loss([x0, x1T])
                spline_loss = loss0
            else:
                raise ValueError(f"Don't know the direction, so can't compute loss")

            loss = spline_loss

            loss.backward()
            optimizer.step()

            wandb.log({
                "spline_step": spline_loss,
            })

            total_spline_loss += spline_loss

        spline_epoch = total_spline_loss / len(x0_loader)

        wandb.log({
            "spline_epoch": spline_epoch,
        })


def train_velocity(ot_sampler,
                   x0s,
                   x1s,
                   flow_matcher_base,
                   velocity_net,
                   num_epochs,
                   optimizer,
                   reflow=False,
                   direc="unbidirectional"):

    velocity_net.to('cpu')

    for epoch in tqdm(range(num_epochs), desc=f"Train the velocity net. Now epoch/Epochs:"):
        total_velocity_loss = 0
        for x0_batch, x1_batch in zip(x0s, x1s):

            if reflow:
                x0 = x0_batch[:, 0, :]
                x1T = x0_batch[:, 1, :]

            else:
                x0 = x0_batch.to('cuda')  # .to(next(velocity_net.parameters()).device)
                x1T = x1_batch.to('cuda')  # .to(next(velocity_net.parameters()).device)
                x0, x1T = ot_sampler.sample_plan(x0, x1T, replace=True)

            t, xt, ut = flow_matcher_base.sample_location_and_conditional_flow(
                x0, x1T, 0, 1
            )

            t = t.to(next(velocity_net.parameters()).device)
            xt = xt.to(next(velocity_net.parameters()).device)
            ut = ut.to(next(velocity_net.parameters()).device)

            vt = velocity_net(t, xt.detach())
            loss = mean_squared_error(vt, ut.detach())

            if direc == "bidirectional":
                if reflow:
                    x0T = x1_batch[:, 0, :]
                    x1 = x1_batch[:, 1, :]
                else:
                    x0_batch1, x1_batch1 = next(zip(x0s, x1s))
                    x0T, x1 = ot_sampler.sample_plan(x0_batch1, x1_batch1, replace=True)

                t1, xt1, ut1 = flow_matcher_base.sample_location_and_conditional_flow(
                    x0T, x1, 0, 1
                )

                t1 = t1.to(next(velocity_net.parameters()).device)
                xt1 = xt1.to(next(velocity_net.parameters()).device)
                ut1 = ut1.to(next(velocity_net.parameters()).device)

                vt1 = velocity_net(t1, xt1.detach())
                loss1 = mean_squared_error(vt1, ut1.detach())
                loss = loss + loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_velocity_loss += loss

            wandb.log({
                "velocity_loss_step": loss,
            })

        velocity_epoch = total_velocity_loss / len(x0s)
        wandb.log({
            "velocity_loss_step": velocity_epoch,
        })


def train_gfm(args,
              ot_sampler,
              spline_model,
              velocity_net,
              optimizer,
              x0_loader,
              x1_loader,
              num_epochs=0,
              desc=None,
              converge=False,
              reflow=False,
              resample=False,
              batch_size=None,
              direc="unbidirectional"
              ):

    spline_model.train()
    velocity_net.train()
    paths = 0
    weights = 0
    T = 0

    if resample:
        paths = torch.load(f'{args.save_address}/velocitynet_tensor.pt')
        weights = torch.load(f'{args.save_address}/weights.pt')
        T = torch.linspace(0, 1, paths.size(0))

    n = 1
    for epoch in tqdm(range(num_epochs), desc=desc):
        total_spline_loss = 0
        total_velocity_loss = 0
        total_loss = 0

        for x0_batch, x1_batch in zip(x0_loader, x1_loader):
            # Train spline_model
            optimizer.zero_grad()

            if reflow:
                x0 = x0_batch[:, 0, :]
                x1T = x0_batch[:, 1, :]

                x0T = x1_batch[:, 0, :]
                x1 = x1_batch[:, 1, :]
            else:
                x0_batch = x0_batch.to(next(spline_model.parameters()).device)
                x1_batch = x1_batch.to(next(spline_model.parameters()).device)
                x0_batch, x1_batch = ot_sampler.sample_plan(x0_batch, x1_batch, replace=True)

                x0 = x0_batch
                x1 = x1_batch

                if n % 1 == 0 or converge:
                    if direc == "bidirectional":
                        x0_batch1, x1_batch1 = next(zip(x0_loader, x1_loader))
                        x0_batch1, x1_batch1 = ot_sampler.sample_plan(x0_batch1, x1_batch1, replace=True)
                        x1T = x1_batch1
                        x0T = x0_batch1

                        # x1T = x1_batch
                        # x0T = x0_batch

                    elif direc == "unbidirectional":
                        x1T = x1_batch
                    else:
                        raise ValueError(f"Don't know the direction")

                else:
                    with torch.no_grad():
                        node = NeuralODE(
                            velocity_net,
                            solver="euler",
                            sensitivity="adjoint",
                            atol=1e-5,
                            rtol=1e-5,
                        )
                        traj_forward = node.trajectory(
                            x0,
                            t_span=torch.linspace(
                                0, 1, 501
                            ),
                        )
                        x1T = traj_forward[-1].detach()
                        del traj_forward

                        if direc == "bidirectional":
                            traj_backard = node.trajectory(
                                x1,
                                t_span=torch.linspace(
                                    1, 0, 501
                                ),
                            )
                            x0T = traj_backard[-1].detach()
                            del traj_backard
                n += 1

            t = torch.rand(batch_size).reshape(-1, 1).to(x0_batch.device)
            t.requires_grad_(True)

            if direc == "bidirectional":
                # Two different ways. Spline
                loss0 = spline_model._compute_loss([x0, x1T])
                loss1 = spline_model._compute_loss([x0T, x1])
                spline_loss = loss0 + loss1
                # Two different ways. Velocity
                ts0, xts0, uts0 = spline_model._process_flow([x0], [x1T], t)
                ts1, xts1, uts1 = spline_model._process_flow([x0T], [x1], t)
                vt0 = velocity_net(ts0[0].detach(), xts0[0].detach())
                vt1 = velocity_net(ts1[0].detach(), xts1[0].detach())
                velocity_loss = mean_squared_error(vt0,
                                                   uts0[0].detach()) + mean_squared_error(vt1, uts1[0].detach())

            elif direc == "unbidirectional":
                # One way. Spline
                if resample:
                    loss0 = spline_model._compute_loss([x0, x1T], paths=paths, weights=weights, T=T,
                                                       batch=batch_size, resample=resample)
                else:
                    loss0 = spline_model._compute_loss([x0, x1T])
                spline_loss = loss0
                # One way. Velocity
                ts0, xts0, uts0 = spline_model._process_flow([x0], [x1T], t)
                vt0 = velocity_net(ts0[0].detach(), xts0[0].detach())
                velocity_loss = mean_squared_error(vt0, uts0[0].detach())
            else:
                raise ValueError(f"Don't know the direction, so can't compute loss")

            loss = spline_loss + velocity_loss

            loss.backward()
            optimizer.step()

            wandb.log({
                "spline_step": spline_loss,
                "velocity_step": velocity_loss,
                "loss_step": loss,
            })

            total_spline_loss += spline_loss
            total_velocity_loss += velocity_loss
            total_loss += loss

        spline_epoch = total_spline_loss / len(x0_loader)
        vel_epoch = total_velocity_loss / len(x0_loader)
        loss_epoch = total_loss / len(x0_loader)

        wandb.log({
            "spline_epoch": spline_epoch,
            "velocity_epoch": vel_epoch,
            "loss_epoch": loss_epoch,
            "epochs": epoch,
        })

