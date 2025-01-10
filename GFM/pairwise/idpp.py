import os
import glob
import torch
import numpy as np
import torch.nn as nn
import openmm.app as app
import openmm.unit as unit
import torch.optim as optim
from torchmetrics.functional import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchcfm.optimal_transport import OTPlanSampler
from src.resample.angles import plot_phi_psi_angles
from src.resample.md_unbiased import plot_paths_energy
import matplotlib.image as mpimg

def read_all_pdb_files_in_directory(path):
    all_data = []
    pdb_files = glob.glob(os.path.join(path, '*.pdb'))

    for pdb_file in pdb_files:
        pdb = app.PDBFile(pdb_file)
        data_x = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        adata = data_x.flatten()
        if len(adata) != 66:
            adata = np.pad(adata, (0, 66 - len(adata)), 'constant')
        adata_tensor = torch.tensor(adata, dtype=torch.float32)
        all_data.append(adata_tensor)
    combined_data = torch.stack(all_data)
    return combined_data


def calculate_pairwise_distance(state):
    """Compute pairwise distance matrix"""
    state = state.reshape(-1, 22, 3)
    diff = state.unsqueeze(2) - state.unsqueeze(1)  # Shape: (N, 22, 22, 3)
    dist_squared = torch.sum(diff ** 2, dim=-1)  # Sum across the last dimension (3D coordinates), shape: (N, 22, 22)
    # pairwise_distances = torch.sqrt(dist_squared)  # Shape: (N, 22, 22)
    pairwise_distances = torch.sqrt(dist_squared + 1e-16)  # Shape: (N, 22, 22)
    return pairwise_distances


def calculate_pairwise(x0, x1, x_t, t):
    pair_A = calculate_pairwise_distance(x0)
    pair_B = calculate_pairwise_distance(x1)

    # Compute pairwise distances for x_t (already interpolated xt values)
    pair_x_t = calculate_pairwise_distance(x_t)

    # Initialize a list to store results for each t
    pairwise_t_list = []

    # Iterate over time steps in t
    for i in range(t.size(0)):
        t_i = t[i].reshape(-1, 1, 1)  # Reshape t[i] for broadcasting

        # Interpolate pairwise distances between pair_A and pair_B
        pairwise = (1 - t_i) * pair_A + t_i * pair_B

        # Store the result for this time step
        pairwise_t_list.append(pairwise)

    # Stack all pairwise_t results into a single tensor
    pairwise_t = torch.stack(pairwise_t_list, dim=0)
    pairwise_t = pairwise_t.reshape(-1, pairwise_t.size(2), pairwise_t.size(3))

    return pair_x_t, pairwise_t


xt = nn.Parameter(torch.rand(501, 2000, 66))
t = torch.linspace(0, 1, 501)

x0 = read_all_pdb_files_in_directory(r"pdb_data\x0s")
x1 = read_all_pdb_files_in_directory(r"pdb_data\x1s")

optimizer = optim.Adam([xt], lr=0.01)
ot_sampler = OTPlanSampler(method='exact')
x0, x1 = ot_sampler.sample_plan(x0, x1, replace=True)

grad_norms = []
loss_plt = []

epochs = 500

for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()

    pair_x_t, pairwise_t = calculate_pairwise(x0, x1, xt, t)
    loss = mean_squared_error(pair_x_t, pairwise_t)
    loss.backward()

    if torch.isnan(xt.grad).any():
        print("Gradient contains NaN values.")
    if torch.isinf(xt.grad).any():
        print("Gradient contains Inf values.")

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    optimizer.step()
    loss_plt.append(loss.detach())

plt.plot(loss_plt)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

xt = xt.detach()
torch.save(xt, r"traj\neb.pt")

img = mpimg.imread(r"background\background.png")
plt.figure(figsize=(10, 10))
plt.imshow(img, extent=[-np.pi, np.pi, -np.pi, np.pi], alpha=0.7)
plot_phi_psi_angles(xt, 'r', 'Trajectories', point_size=5, skip_index=40)
plot_paths_energy(xt, threshold=2000, last_time_threshold=10000, num_indices=100)
