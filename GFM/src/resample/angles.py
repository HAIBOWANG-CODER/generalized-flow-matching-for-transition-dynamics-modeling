import os
import glob
import jax.numpy as jnp
import jax
import numpy as np
from mdtraj.geometry import indices_phi, indices_psi
import matplotlib.pyplot as plt
import mdtraj as md
from tqdm import tqdm
import torch
import matplotlib.image as mpimg


def to_md_traj(mdtraj_topology, trajectory):
    return md.Trajectory(trajectory.reshape(-1, mdtraj_topology.n_atoms, 3), mdtraj_topology)

@jax.jit
def dihedral(p):

    b = p[:-1] - p[1:]
    b = b.at[0].set(-b[0])
    v = jnp.array(
        [v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    v /= jnp.sqrt(jnp.einsum('...i,...i', v, v)).reshape(-1, 1)
    b1 = b[1] / jnp.linalg.norm(b[1])
    x = jnp.dot(v[0], v[1])
    m = jnp.cross(v[0], b1)
    y = jnp.dot(m, v[1])
    return jnp.arctan2(y, x)


def phi_psi_from_mdtraj(mdtraj_topology):
    angles_phi = indices_phi(mdtraj_topology)[0]
    angles_psi = indices_psi(mdtraj_topology)[0]

    @jax.jit
    @jax.vmap
    def phi_psi(p):
        p = p.reshape(mdtraj_topology.n_atoms, 3)
        phi = dihedral(p[angles_phi, :])
        psi = dihedral(p[angles_psi, :])
        return jnp.array([phi, psi])

    return phi_psi


def read_topology_from_pdb(pdb_file):
    pdb = md.load(pdb_file)
    return pdb.topology


def process_tensor_file(tensor_file):
    data = torch.load(tensor_file)
    if isinstance(data, list):
        data = torch.cat(data)
    return jnp.array(data.detach().numpy())


def plot_phi_psi_angles_from_tensor(tensor_folder,
                                    color,
                                    label,
                                    point_size=5,
                                    skip_index=100,
):
    pdb_file = r"pdb_data\AD_A.pdb"
    topology = read_topology_from_pdb(pdb_file)
    phi_psi = phi_psi_from_mdtraj(topology)
    tensor_files = glob.glob(os.path.join(tensor_folder, '*.pt'))
    all_phi = []
    all_psi = []

    for tensor_file in tqdm(tensor_files, desc=f"Processing {label}"):
        positions = process_tensor_file(tensor_file)
        if positions.ndim < 3:
            positions = jnp.expand_dims(positions, axis=1)

        for time_index in tqdm(range(positions.shape[0]), desc=f"Processing time steps in {tensor_file}", leave=False):
            if time_index % 1 == 0:
                for model_index in range(positions.shape[1]):
                    if model_index % skip_index == 0:
                        positions_data = positions[time_index, model_index]

                        # Reshape positions_data from (66,) to (22, 3)
                        positions_data = positions_data.reshape(-1, 3)

                        angles = phi_psi(positions_data.reshape(1, topology.n_atoms, 3))
                        phi = angles.squeeze()[0]
                        psi = angles.squeeze()[1]

                        if phi.size > 0 and psi.size > 0:
                            all_phi.append(np.expand_dims(phi, axis=0))
                            all_psi.append(np.expand_dims(psi, axis=0))

    if all_phi and all_psi:
        all_phi = np.concatenate(all_phi)
        all_psi = np.concatenate(all_psi)

        k = 1.0
        plt.scatter(all_phi, all_psi, c=color, label=label, s=point_size, alpha=1.0)
        plt.xlim(-k * np.pi, k * np.pi)
        plt.ylim(-k * np.pi, k * np.pi)
        plt.xlabel('Phi (radians)', fontsize=36)
        plt.ylabel('Psi (radians)', fontsize=36)
        plt.tick_params(axis='both', which='major', labelsize=32)
        plt.tight_layout()

    else:
        print(f"No valid phi/psi angles found in {label}")


def plot_phi_psi_angles(tensor,
                        color,
                        label,
                        point_size=5,
                        skip_index=1,
):

    pdb_file = r"pdb_data\AD_A.pdb"
    topology = read_topology_from_pdb(pdb_file)
    phi_psi = phi_psi_from_mdtraj(topology)
    all_phi = []
    all_psi = []

    positions = tensor  # Use the input tensor directly

    if positions.ndim < 3:
        positions = jnp.expand_dims(positions, axis=1)

    for time_index in tqdm(range(positions.shape[0]), desc=f"Processing time steps in {label}", leave=False):
        if time_index % 1 == 0:
            for model_index in range(positions.shape[1]):
                if model_index % skip_index == 0:
                    positions_data = positions[time_index, model_index]

                    # Reshape positions_data from (66,) to (22, 3)
                    positions_data = positions_data.reshape(-1, 3)

                    if isinstance(positions_data, torch.Tensor):
                        positions_data = positions_data.cpu().numpy()

                    angles = phi_psi(positions_data.reshape(1, topology.n_atoms, 3))
                    phi = angles.squeeze()[0]
                    psi = angles.squeeze()[1]

                    if phi.size > 0 and psi.size > 0:
                        all_phi.append(np.expand_dims(phi, axis=0))
                        all_psi.append(np.expand_dims(psi, axis=0))

    if all_phi and all_psi:
        all_phi = np.concatenate(all_phi)
        all_psi = np.concatenate(all_psi)

        k = 1.0
        plt.scatter(all_phi, all_psi, c=color, label=label, s=point_size, alpha=1.0)
        plt.xlim(-k * np.pi, k * np.pi)
        plt.ylim(-k * np.pi, k * np.pi)
        plt.xlabel('Phi (radians)', fontsize=36)
        plt.ylabel('Psi (radians)', fontsize=36)
        plt.tick_params(axis='both', which='major', labelsize=32)
        plt.tight_layout()

    else:
        print(f"No valid phi/psi angles found in {label}")


if __name__ == '__main__':

    tensor_folder = r"traj\test"
    img = mpimg.imread(r"background\background.png")

    # Read in a tensor of dimensions (T, N, D) containing all coordinates, and then plot the dihedral angles
    plt.figure(figsize=(10, 10))
    plt.imshow(img, extent=[-np.pi, np.pi, -np.pi, np.pi], alpha=0.7)
    plot_phi_psi_angles_from_tensor(tensor_folder, 'r', 'Trajectories', point_size=5, skip_index=100)
    plt.tight_layout()
    plt.show()
