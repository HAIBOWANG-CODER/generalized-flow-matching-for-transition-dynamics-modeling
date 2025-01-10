import torch
import numpy as np
import os
import glob
import openmm.unit as unit
import openmm.app as app
from torchcfm.optimal_transport import OTPlanSampler
from pdb_data.internal import cartesian_to_internal
from pdb_data.internal import internal_to_cartesian


ot_sampler = OTPlanSampler(method="exact")


def read_2D(x0_path):
    pt_files = [f for f in os.listdir(x0_path) if f.endswith('.pt')]

    # pt_files = pt_files[:100]

    tensor_list = []

    for pt_file in pt_files:
        file_path = os.path.join(x0_path, pt_file)

        tensor = torch.load(file_path)
        tensor_list.append(tensor)

    x = torch.stack(tensor_list, dim=0)
    return x


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
    # combined_data = cartesian_to_internal(combined_data)
    return combined_data


x0_path = r"C:\Users\Administrator\Desktop\GFM\pdb_data\x0s"
x1_path = r"C:\Users\Administrator\Desktop\GFM\pdb_data\x1s"
# x0 = read_2D(x0_path)
# x1 = read_2D(x1_path)

# x0 = read_all_pdb_files_in_directory(x0_path)
# x1 = read_all_pdb_files_in_directory(x1_path)

x0 = torch.load(r"C:\Users\Administrator\Desktop\GFM\pdb_data\internal_data\x0s.pt")
x1 = torch.load(r"C:\Users\Administrator\Desktop\GFM\pdb_data\internal_data\x1s.pt")

# indices = torch.randperm(x0.shape[0])[:1000]
# x0 = x0[indices]
# x1 = x1[indices]
"""
x0, x1 = ot_sampler.sample_plan(
                        x0,
                        x1,
                        replace=True,
                    )
"""
t_values = torch.linspace(0, 1, 501)

X = []
for i in range(501):
    x = (1 - t_values[i]) * x0 + t_values[i] * x1
    X.append(x)

xt = torch.stack(X, dim=0)
# xts = internal_to_cartesian(xt)

torch.save(xt, r"C:\Users\Administrator\Desktop\GFM\traj\linear_internal.pt")
