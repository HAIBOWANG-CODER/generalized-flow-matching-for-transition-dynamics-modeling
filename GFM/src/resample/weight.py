import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob
import matplotlib.image as mpimg
from src.resample.angles import plot_phi_psi_angles_from_tensor
from src.resample.md_unbiased import V_t_func

def impt_weight_fn(s, xs, us):
    """
    s: (S,)
    xs: (S, N, D)
    us: (S, N, D)
    sigma: float
    ===
    weights: (N,)
    """
    S, N, D = xs.shape
    assert s.shape == (S,)

    dt = torch.cat([s[[1]] - s[[0]], s[1:] - s[:-1]]).reshape(1, -1)

    assert dt.shape == (1, S) and (dt > 0).all()

    pdb_file_path = r"pdb_data\AD_A.pdb"
    xs = xs.to(dt.device)
    xt_path = xs.numpy()
    us = us.to(dt.device)

    # Kinetic energy
    control_cost = 0.5 * (us ** 2).sum(dim=-1) * dt.reshape(-1, 1)

    # potential energy
    V = V_t_func(xt_path, pdb_file_path)
    state_cost = V * dt.reshape(-1, 1)

    total_cost = (state_cost + control_cost).mean(dim=0)
    total_cost = total_cost - total_cost.min(dim=0, keepdim=True)[0]

    weights = torch.exp(-total_cost)
    weights = weights / weights.sum(dim=0, keepdim=True)
    assert weights.shape == (N,)
    return weights

def resample_trajectory(xs):
    """
       Resamples trajectories based on importance weights to focus on high-energy paths.

       Args:
           xs: (S, N, D)  # Input trajectories, where S is the number of time steps,
                          # N is the number of trajectories, and D is the state dimension.
    """
    S, N, D = xs.shape
    s = torch.linspace(0.0, 1.0, steps=S)
    sigma = 0.05

    dt = 1.0 / S
    us = torch.zeros_like(xs)
    for i in range(S - 1):
        us[i+1] = (xs[i + 1] - xs[i]) / dt

    weights = impt_weight_fn(s, xs, us)

    weight_sorted, sorted_indices = torch.sort(weights, descending=True)
    # Select the 500 paths with the largest energy weights
    top_count_indices = min(500, int(weights.numel() * 0.1))
    top_indices = sorted_indices[:top_count_indices]

    if xs.dim() == 3:
        xts = xs[:, top_indices, :]
        uts = us[:, top_indices, :]
    else:
        raise ValueError("The shape of tensor is not (S, N, D)")

    return xts, uts, weights

if __name__ == "__main__":
    traj_tensor_folder = r"traj"
    resample_tensor_folder = r"traj\resample_traj"

    pdb_file = r"pdb_data\AD_A.pdb"
    img = mpimg.imread(r"background\background.png")
    tensor_files = glob.glob(os.path.join(traj_tensor_folder, '*.pt'))
    for tensor_file in tqdm(tensor_files, desc=f"Processing "):
        xs = torch.load(tensor_file)
        xs = xs.to('cpu')

        xts, uts, weight = resample_trajectory(xs)
        torch.save(xts, r"traj/resample_traj/resample_data.pt")
        torch.save(weight, r"traj/weights.pt")

        plt.figure(figsize=(10, 10))
        plt.imshow(img, extent=[-np.pi, np.pi, -np.pi, np.pi], alpha=1.0)
        plot_phi_psi_angles_from_tensor(resample_tensor_folder, 'r', 'Trajectories', point_size=5, skip_index=600)
        plt.tight_layout()
        plt.show()

