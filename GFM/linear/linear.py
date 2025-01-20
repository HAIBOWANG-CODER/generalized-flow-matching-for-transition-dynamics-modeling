import torch
from torchcfm.optimal_transport import OTPlanSampler
from pdb_data.internal import cartesian_to_internal
from pdb_data.internal import internal_to_cartesian


ot_sampler = OTPlanSampler(method="exact")


def load_all_data(path):
    all_coords = torch.load(path)

    return all_coords


x0_path = r"./pdb_data/alanine_data_cartesian/x0s.pt"
x1_path = r"./pdb_data/alanine_data_cartesian/x1s.pt"

x0 = load_all_data(x0_path)
x1 = load_all_data(x1_path)

x0 = torch.squeeze(x0)
x1 = torch.squeeze(x1)

# x0 = cartesian_to_internal(x0)
# x1 = cartesian_to_internal(x1)

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

torch.save(xt, r"./traj/linear_internal.pt")
