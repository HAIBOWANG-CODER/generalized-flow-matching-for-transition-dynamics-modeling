import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchcfm.optimal_transport import OTPlanSampler
from src.resample.angles import plot_phi_psi_angles
from src.resample.md_unbiased import plot_paths_energy
import matplotlib.image as mpimg


def load_all_data(path):
    all_coords = torch.load(path)
    return all_coords


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

T, N, D = 501, 2000, 66

xt = nn.Parameter(torch.rand(T, N, D))
t = torch.linspace(0, 1, T)

x0 = load_all_data(r"pdb_data\alanine_data_cartesian\x0s.pt")
x1 = load_all_data(r"pdb_data\alanine_data_cartesian\x1s.pt")

x0 = torch.squeeze(x0)
x1 = torch.squeeze(x1)

indices = torch.randint(low=0, high=x1.size(0), size=(N,))

x0 = x0[indices]
x1 = x1[indices]

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
plot_paths_energy(xt, threshold=2000, last_time_threshold=10000, num_indices=100, )
