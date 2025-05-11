import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchdyn.core import NeuralODE
from src.resample.weight import resample_trajectory
from src.resample.angles import plot_phi_psi_angles
from src.resample.md_unbiased import plot_paths_energy
from train_data.pdb_data.internal import internal_to_cartesian


def spline_test(x0, x1, T, net):
    with torch.no_grad():
        xts = []
        for i in range(T.shape[0]):
            min_size = min(x0.size(0), x1.size(0))
            x0 = x0[:min_size]
            x1 = x1[:min_size]

            t = T[i] * torch.ones(x0.shape[0], dtype=x0.dtype, device=x0.device)
            t = t.type_as(x0)
            t = t.unsqueeze(1)
            net_out = net(x0, x1, t)
            xt = (1-t)*x0 + t*x1 + t*(1-t)*net_out
            xts.append(xt)
        xt_spline_traj = torch.stack(xts)
    return xt_spline_traj


def run_spline_and_ode_pipeline(
    args,
    sampleLoader,
    data_type: str,
    spline_net: torch.nn.Module,
    velocity_net: torch.nn.Module,
):

    file_test0 = f"{args.data_path}/x_test.pt"
    loader0 = sampleLoader(file_test0, data_type, max_atoms=args.dim)
    x0 = torch.tensor(loader0.load_all_data(), dtype=torch.float32)

    file_test1 = f"{args.data_path}/x_test1.pt"
    loader1 = sampleLoader(file_test1, data_type, max_atoms=args.dim)
    x1 = torch.tensor(loader1.load_all_data(), dtype=torch.float32)

    t = torch.linspace(0, 1, steps=500)
    xt = spline_test(x0, x1, t, spline_net)

    torch.save(xt, f'{args.save_address}/spline_output.pt')

    class model_torch_wrapper(torch.nn.Module):

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, t, x, *args, **kwargs):
            return self.model(t, x)

    node = NeuralODE(
                model_torch_wrapper(velocity_net),
                solver="euler",
                sensitivity="adjoint",
                atol=1e-10,
                rtol=1e-10,
            )


    traj = node.trajectory(
                    x0,
                    t_span=torch.linspace(
                        0, 1, 501
                    ),
                )

    if data_type == "ADC":
        xts = traj.detach() / 10        # Convert units to nanometers
    elif data_type == "ADI":
        # Transform internal coords to Cartesian coords
        xts = internal_to_cartesian(traj.detach())
    else:
        raise ValueError(f'Unknown data_type {data_type}')


    torch.save(xts, f'{args.save_address}/velocitynet_tensor.pt')


    xt, uts, weights = resample_trajectory(xts)  # Computing weights (Cartesian Coords)
    torch.save(xt, f'{args.save_address}/xt.pt')

    img = mpimg.imread(r"background/background.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(img, extent=[-np.pi, np.pi, -np.pi, np.pi], alpha=0.7)
    plot_phi_psi_angles(xts, 'r', 'Trajectories', point_size=5, skip_index=600)
    plot_paths_energy(xt, threshold=2000, last_time_threshold=10000, num_indices=100)