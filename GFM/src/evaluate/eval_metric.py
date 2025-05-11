import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchdyn.core import NeuralODE
from train_data.pdb_data.internal import internal_to_cartesian
from src.dataloaders.dataload import sampleLoader
from src.resample.weight import resample_trajectory
from src.resample.angles import plot_phi_psi_angles
from src.resample.md_unbiased import plot_paths_energy

def compute_spline_traj(x0, x1, t, flow_matcher_base):
    B = x0.size(0)
    N = 100
    chunk_size = (B + N - 1) // N

    x0_chunks = x0.split(chunk_size, dim=0)
    x1_chunks = x1.split(chunk_size, dim=0)
    t = t.to('cuda')

    xt_chunks = []
    for x0_chunk, x1_chunk in zip(x0_chunks, x1_chunks):
        x0c = x0_chunk.to('cuda')
        x1c = x1_chunk.to('cuda')

        xtc = flow_matcher_base.sample_location_and_conditional_flow(
            x0c, x1c,
            0, 1,
            t=t,
            spline_test=True
        )
        xt_chunks.append(xtc.to('cpu'))
        del x0c, x1c, xtc
        torch.cuda.empty_cache()

    xt = torch.cat(xt_chunks, dim=1)

    return xt

def process_trajectory(velocity_net, flow_matcher_base, x0, x1, t_steps, traj_path_prefix, num, data_type):
    """
    Process forward and backward trajectories, visualize, and save key results.

    Parameters:
    - velocity_net: Neural network for flow computation.
    - flow_matcher_base: Base object for flow matching.
    - x0, x1: Initial and final states.
    - t_steps: Number of time steps for trajectory generation.
    - traj_path_prefix: Directory prefix for saving trajectory data.
    """
    # Generate time points
    t = torch.linspace(0, 1, steps=t_steps)

    # Generate trajectory using flow matching (from Spline Network)
    xt = compute_spline_traj(x0, x1, t, flow_matcher_base)

    if data_type == 'ADC':
        xt = xt / 100  # Convert units from 0.1 angstrom back to nm
    elif data_type == 'ADI':
        xt = internal_to_cartesian(xt)  # Transform internal coords to Cartesian coords
    else:
        raise ValueError(f'Unknown data_type {data_type}')

    torch.save(xt.detach(), f"D:/chignolin_result/splinenet_tensor_{num}.pt")
    del xt

    # Visualize phi-psi angles
    # img = mpimg.imread(r"background/background.png")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img, extent=[-np.pi, np.pi, -np.pi, np.pi], alpha=0.7)
    # plot_phi_psi_angles(xt.detach(), 'r', 'Trajectories', point_size=5, skip_index=600)

    # Compute the best energy of spline output
    # xts_b, _, _ = resample_trajectory(xt.detach())
    # torch.save(xts_b, f'{traj_path_prefix}/xts_b_{num}.pt')
    # plot_paths_energy(xts_b, threshold=200000, last_time_threshold=100000, num_indices=100)

    # Use NeuralODE to compute forward and backward trajectories (from Velocity Network)
    node = NeuralODE(velocity_net, solver="euler", sensitivity="adjoint", atol=1e-5, rtol=1e-5)
    traj_forward = node.trajectory(x0, t_span=torch.linspace(0, 1, t_steps))
    # traj_backward = node.trajectory(x1, t_span=torch.linspace(1, 0, t_steps))

    # Save forward and backward trajectories
    # save_forward_and_backward_trajectories(traj_forward, traj_backward, x0, x1, traj_path_prefix)

    # Process and save flow-matching trajectory
    if data_type == 'ADC':
        traj_forward_nm = traj_forward / 100
    elif data_type == 'ADI':
        traj_forward_nm = internal_to_cartesian(traj_forward)
    else:
        raise ValueError(f'Unknown data_type {data_type}')

    # torch.save(traj_forward_nm.detach(), f'{traj_path_prefix}/velocitynet_tensor_{num}.pt')
    torch.save(traj_forward_nm.detach(), f"D:/chignolin_result/velocitynet_tensor_{num}.pt")

    # Visualize phi-psi angles and energy paths
    img = mpimg.imread(r"background/background.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(img, extent=[-np.pi, np.pi, -np.pi, np.pi], alpha=0.7)
    plot_phi_psi_angles(traj_forward_nm.detach(), 'r', 'Trajectories', point_size=5, skip_index=600)

    xts, uts, weights = resample_trajectory(traj_forward_nm.detach())
    # Saved for resample iterative training
    torch.save(traj_forward, f'{traj_path_prefix}/velocitynet_tensor.pt')
    torch.save(weights, f'{traj_path_prefix}/weights.pt')

    torch.save(xts, f'{traj_path_prefix}/xts_{num}.pt')
    plot_paths_energy(xts, threshold=200000, last_time_threshold=1000000, num_indices=100)


def save_forward_and_backward_trajectories(traj_forward, x0, x1, traj_path_prefix):
    """
    Save forward and backward trajectories for bidirectional experiments.

    Parameters:
    - traj_forward, traj_backward: Generated trajectories.
    - x0, x1: Initial and final states.
    - traj_path_prefix: Directory prefix for saving trajectory data.
    """
    # Prepare and save forward trajectory
    x0 = x0.unsqueeze(1)
    x1_forward = traj_forward[-1].unsqueeze(1)
    x_forward = torch.cat((x0, x1_forward), dim=1)
    torch.save(x_forward, f'{traj_path_prefix}/ode_x_forward.pt')

    # Prepare and save backward trajectory
    x1 = x1.unsqueeze(1)
    # x0_backward = traj_backward[-1].unsqueeze(1)
    # x_backward = torch.cat((x0_backward, x1), dim=1)
    # torch.save(x_backward, f'{traj_path_prefix}/ode_x_backward.pt')


def run_processing_for_pdb(files0, files1, velocity_net, flow_matcher_base, traj_path_prefix, data_type, ot_sampler,
                           t_steps=501, num=0):
    """
    Main function to load data, process trajectories, and save results.

    Parameters:
    - files0, files1: Paths to input PDB files for states x0 and x1.
    - velocity_net: Neural network for flow computation.
    - flow_matcher_base: Base object for flow matching.
    - traj_path_prefix: Directory prefix for saving trajectory data.
    - t_steps: Number of time steps for trajectory generation.
    """
    with torch.no_grad():
        loader0 = sampleLoader(files0, data_type, 'x0')
        x0 = torch.tensor(loader0.load_all_data(), dtype=torch.float32).to('cpu')
        loader1 = sampleLoader(files1, data_type, 'x1')
        x1 = torch.tensor(loader1.load_all_data(), dtype=torch.float32).to('cpu')
        x0 = x0[:20000, :]
        x1 = x1[:20000, :]
        x0, x1 = ot_sampler.sample_plan(x0, x1, replace=True)

        # flow_matcher_base.spline_net = flow_matcher_base.spline_net.to('cpu')
        velocity_net = velocity_net.to('cpu')

        # Process trajectories
        process_trajectory(velocity_net, flow_matcher_base, x0, x1, t_steps, traj_path_prefix, num, data_type)