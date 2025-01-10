import numpy as np
import torch
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import umap
import phate
import scprep
import scanpy as sc
from sklearn.decomposition import PCA

import ot as pot
from tqdm import tqdm
from functools import partial
from typing import Optional


def set_seed(seed):
    """
    Sets the seed for reproducibility in PyTorch, Numpy, and Python's Random.

    Parameters:
    seed (int): The seed for the random number generators.
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # CPU and GPU (deterministic)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # CUDA
        torch.cuda.manual_seed_all(seed)  # all GPU devices
        torch.backends.cudnn.deterministic = True  # CuDNN behavior
        torch.backends.cudnn.benchmark = False


def wasserstein_distance(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 1,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret


def plot_trajectories(
    true_datasets, traj, time_steps, dimensions=(0, 1), n_samples=200, fname="traj.png"
):
    traj = traj.cpu().numpy() if torch.is_tensor(traj) else traj
    fig = plt.figure(figsize=(10, 6))
    first = True

    # Get the colormap
    cmap = plt.cm.get_cmap("Spectral")
    source_color = cmap(0.0)  # Start of the colormap
    target_color = cmap(1.0)  # End of the colormap

    for i, dataset in enumerate(true_datasets):
        dataset = dataset.cpu().numpy() if torch.is_tensor(dataset) else dataset
        if i in time_steps:
            if len(dataset) > n_samples:
                indices = np.random.choice(len(dataset), size=n_samples, replace=False)
            else:
                indices = range(len(dataset))
            if i == time_steps[0]:
                label = r"$p_0$"
                color = source_color  # Darker orange
                text_x_pos = -1
            else:
                label = r"$p_1$"
                color = target_color  # Darker violet
                text_x_pos = 1
            plt.scatter(
                dataset[indices, dimensions[0]],
                dataset[indices, dimensions[1]],
                s=15,  # Slightly larger size
                alpha=1,
                color=color,
            )
            # Adding text label
            plt.text(
                text_x_pos,
                0.9,
                label,
                fontsize=40,  # Increased font size
                color=color,
                ha="center",
                va="center",
            )
            if first:
                first = False
                traj_label_added = False

                # Select half of the indices for plotting lines
                line_indices = np.random.choice(
                    indices, size=len(indices) // 2, replace=False
                )

                for idx in line_indices:
                    if not traj_label_added:
                        plt.plot(
                            traj[:, idx, dimensions[0]],
                            traj[:, idx, dimensions[1]],
                            linewidth=0.5,
                            alpha=0.2,
                            color="grey",
                        )
                        traj_label_added = True
                    else:
                        plt.plot(
                            traj[:, idx, dimensions[0]],
                            traj[:, idx, dimensions[1]],
                            linewidth=0.5,
                            alpha=0.2,
                            color="grey",
                        )

    plt.axis("off")
    plt.savefig(fname, dpi=200)
    plt.show()
    return fig


def plot_lidar(ax, dataset, xs=None, S=25):
    # Combine the dataset and trajectory points for sorting
    combined_points = []
    combined_colors = []
    combined_sizes = []

    # Normalize the z-coordinates for alpha scaling
    z_coords = (
        dataset[:, 2].numpy() if torch.is_tensor(dataset[:, 2]) else dataset[:, 2]
    )
    z_min, z_max = z_coords.min(), z_coords.max()
    z_norm = (z_coords - z_min) / (z_max - z_min)

    # Add surface points with a lower z-order
    for i, point in enumerate(dataset):
        # alpha = 2 + z_norm[i]
        grey_value = 0.95 - 0.7 * z_norm[i]
        combined_points.append(point.numpy())
        combined_colors.append(
            (
                grey_value,
                grey_value,
                grey_value,
            )
        )  # Grey color with transparency
        combined_sizes.append(0.1)

    # Add trajectory points with a higher z-order
    if xs is not None:
        cmap = plt.get_cmap("Spectral")
        B, T, D = xs.shape
        steps_to_log = np.linspace(0, T - 1, S).astype(int)
        xs = xs.cpu().detach().clone()
        for idx, step in enumerate(steps_to_log):
            for point in xs[:512, step]:
                combined_points.append(
                    point.numpy() if torch.is_tensor(point) else point
                )
                combined_colors.append(cmap(idx / (len(steps_to_log) - 1)))
                combined_sizes.append(0.8)

    # Convert to numpy array for easier manipulation
    combined_points = np.array(combined_points)
    combined_colors = np.array(combined_colors)
    combined_sizes = np.array(combined_sizes)

    # Sort by z-coordinate (depth)
    sorted_indices = np.argsort(combined_points[:, 2])
    combined_points = combined_points[sorted_indices]
    combined_colors = combined_colors[sorted_indices]
    combined_sizes = combined_sizes[sorted_indices]

    # Plot the sorted points
    ax.scatter(
        combined_points[:, 0],
        combined_points[:, 1],
        combined_points[:, 2],
        s=combined_sizes,
        c=combined_colors,
        depthshade=True,
        # alpha=1.0,
    )

    ax.set_xlim3d(left=-4.8, right=4.8)
    ax.set_ylim3d(bottom=-4.8, top=4.8)
    ax.set_zlim3d(bottom=0.0, top=2.0)
    ax.set_zticks([0, 1.0, 2.0])
    ax.grid(False)
    plt.axis("off")

    return ax


def plot_images_trajectory(trajectories, vae, processor, use_ambient_space, num_steps):

    # Compute trajectories for each image
    t_span = torch.linspace(0, trajectories.shape[1] - 1, num_steps)
    t_span = [int(t) for t in t_span]
    num_images = trajectories.shape[0]

    # Decode images at each step in each trajectory
    if not use_ambient_space:
        decoded_images = [
            [
                processor.postprocess(
                    vae.decode(
                        trajectories[i_image, traj_step].unsqueeze(0)
                    ).sample.detach()
                )[0]
                for traj_step in t_span
            ]
            for i_image in range(num_images)
        ]
    else:
        decoded_images = [
            [
                (trajectories[i_image, traj_step].detach().cpu().numpy())
                for traj_step in t_span
            ]
            for i_image in range(num_images)
        ]

    # Plotting
    fig, axes = plt.subplots(
        num_images, num_steps, figsize=(num_steps * 2, num_images * 2)
    )
    if num_images == 1:
        axes = [axes]  # Ensure axes is iterable
    for img_idx, img_traj in enumerate(decoded_images):
        for step_idx, img in enumerate(img_traj):
            ax = axes[img_idx][step_idx] if num_images > 1 else axes[step_idx]
            if (
                isinstance(img, np.ndarray) and img.shape[0] == 3
            ):  # Assuming 3 channels (RGB)
                img = img.transpose(1, 2, 0)
            ax.imshow(img)
            ax.axis("off")
            if img_idx == 0:
                ax.set_title(f"t={t_span[step_idx]/t_span[-1]:.2f}")
    plt.tight_layout()
    return fig
