import torch
import numpy as np
import mdtraj as md
from torchcfm.optimal_transport import OTPlanSampler
from train_data.pdb_data.internal import cartesian_to_internal
from train_data.pdb_data.internal import internal_to_cartesian


ot_sampler = OTPlanSampler(method="exact")


def load_all_data(path):
    all_coords = torch.load(path)

    return all_coords

"""
x0_path = r"C:/Users/Administrator/Desktop/GFM/pdb_data/alanine_data_cartesian/x0s.pt"
x1_path = r"C:/Users/Administrator/Desktop/GFM/pdb_data/alanine_data_cartesian/x1s.pt"

x0 = load_all_data(x0_path)
x1 = load_all_data(x1_path)

x0 = torch.squeeze(x0)
x1 = torch.squeeze(x1)
"""

# folder_path = paths
dcd_path = r"E:\chignolin_results\align_chig\aligned_trajectory.dcd"
topology_file = r"C:\Users\Administrator\Desktop\GFM\pdb_data\test.pdb"

x0_indices = np.load(r"C:\Users\Administrator\Desktop\GFM\x0.npy")
x1_indices = np.load(r"C:\Users\Administrator\Desktop\GFM\x1.npy")
# pt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]
# pt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcd')]
# pt_files.sort()
# traj = md.load(pt_files, top=topology_file)

traj = md.load(dcd_path, top=topology_file)
x = traj.xyz

x0_indices = np.random.choice(x0_indices, size=20000, replace=False)
x1_indices = np.random.choice(x1_indices, size=20000, replace=False)

x0 = torch.tensor(x[x0_indices, :, :]).reshape(-1, 1, 498)
x1 = torch.tensor(x[x1_indices, :, :]).reshape(-1, 1, 498)

# x0 = cartesian_to_internal(x0)
# x1 = cartesian_to_internal(x1)

# indices = torch.randperm(x0.shape[0])[:1000]
# x0 = x0[indices]
# x1 = x1[indices]


x0, x1 = ot_sampler.sample_plan(
                        x0,
                        x1,
                        replace=True,
                    )


t_values = torch.linspace(0, 1, 101)

X = []
for i in range(101):
    x = (1 - t_values[i]) * x0 + t_values[i] * x1
    X.append(x)

xt = torch.stack(X, dim=0)

# xts = internal_to_cartesian(xt)

torch.save(xt, r"C:\Users\Administrator\Desktop\GFM\traj\linear_internal.pt")
