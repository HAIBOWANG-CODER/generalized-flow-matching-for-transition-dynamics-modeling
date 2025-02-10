import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def U1(xs, scale1=0.15, scale2=15, beta=0.1):
    x, y = xs[:, 0], xs[:, 1]
    e1 = -200 * torch.exp(-(x - 1)**2 - 10 * y**2)
    e2 = -100 * torch.exp(-x**2 - 10 * (y - 0.5)**2)
    e3 = -170 * torch.exp(-6.5 * (0.5 + x)**2 + 11 * (x + 0.5) * (y - 1.5) - 6.5 * (y - 1.5)**2)
    e4 = 15.0 * torch.exp(0.7 * (1 + x)**2 + 0.6 * (x + 1) * (y - 1) + 0.7 * (y - 1)**2) + 146.7
    return beta * (scale1 * (e1 + e2 + e3 + e4))

def U(xs):
    x, y = xs[:, 0], xs[:, 1]
    e1 = -200 * torch.exp(-(x - 1)**2 - 10 * y**2)
    e2 = -100 * torch.exp(-x**2 - 10 * (y - 0.5)**2)
    e3 = -170 * torch.exp(-6.5 * (0.5 + x)**2 + 11 * (x + 0.5) * (y - 1.5) - 6.5 * (y - 1.5)**2)
    e4 = 15.0 * torch.exp(0.7 * (1 + x)**2 + 0.6 * (x + 1) * (y - 1) + 0.7 * (y - 1)**2)
    return e1 + e2 + e3 + e4

def dUdx(xs, scale1=0.15, scale2=15, beta=0.1):
    xs.requires_grad_(True)
    # energy = U(xs, scale1, scale2, beta).sum()
    energy = U(xs).sum()
    energy.backward()
    grad = xs.grad
    xs.requires_grad_(False)
    return grad


def generate_data_and_save(initial_point, file, num_points=2000, dt=0.0001, save_interval=1, directory="toy_data"):
    os.makedirs(directory, exist_ok=True)
    old_x = initial_point.clone()
    all_xs = []
    for t in tqdm(range(1, num_points + 1), desc="Generating Data"):  # Starting from 1 to num_points
        eps = torch.randn((1, 2))
        # eps = torch.randn((1, 2))/2.5
        grad = dUdx(old_x)
        if grad is None:
            print(f"Gradient is None at iteration {t}")
            break
        # new_x = old_x - grad * dt + torch.sqrt(torch.tensor(2.0)) * torch.sqrt(torch.tensor(dt)) * eps
        new_x = old_x - grad * dt + torch.tensor(5.0) * torch.sqrt(torch.tensor(dt)) * eps

        old_x = new_x
        if t % save_interval == 0:
            all_xs.append(new_x)

    all_xs = torch.cat(all_xs, dim=0)
    torch.save(all_xs, os.path.join(directory, f"{file}.pt"))


def plot_muller_train_data(directory, step=1):
    def plot_energy(energy, data, step=100, extent=[(-1.5, 0.9), (-0.5, 1.7)], resolution=150, dim=2):

        # plt.figure(figsize=(5,5))
        x, y = np.linspace(extent[0][0], extent[0][1], resolution), np.linspace(extent[1][0], extent[1][1], resolution)
        x, y = np.meshgrid(x, y)
        xy = torch.tensor(np.stack([x, y], -1).reshape(-1, 2), dtype=torch.float32)
        us = energy(xy).reshape([resolution, resolution]).detach().numpy()
        plt.contourf(x, y, us, 50)
        colors = ['red', 'blue', 'green', 'yellow']
        labels = ['Data from A', 'Data from B', 'Data from C', 'Data from Dir4']
        for i, d in enumerate(data):
            plt.scatter(d[::step, 0], d[::step, 1], marker='.', color=colors[i], alpha=0.5, label=labels[i])

        plt.scatter(-0.55828035, 1.44169, marker='.', color=colors[2], alpha=1, s=400)
        plt.scatter(0.62361133, 0.02804632, marker='.', color=colors[2], alpha=1, s=400)
        plt.xlim(extent[0])
        plt.ylim(extent[1])
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    data_list = []
    for filename in tqdm(os.listdir(directory), desc=f"Reading Data from {directory}"):
        if filename.endswith(".pt"):
            file_path = os.path.join(directory, filename)
            data = torch.load(file_path)
            data_list.append(data)
    plot_energy(U, data_list, step=step)


def estimation_for_muller(directories, step=1):
    all_data = []

    for directory in directories:
        for filename in tqdm(os.listdir(directory), desc=f"Reading Data from {directory}"):
            if filename.endswith(".pt"):
                file_path = os.path.join(directory, filename)
                data = torch.load(file_path)
                data = data.to('cpu')
                #data = data * 100

                if data.dim() != 3:
                    data = data.unsqueeze(0)
                T, N, D = data.shape
                sampling_step = 10
                n = N // sampling_step  # sampling_step
                sampled_data = data[:, ::sampling_step, :]

                """
                # Mark the endpoints at both ends of the path
                first_last = torch.stack([sampled_data[0], sampled_data[-1]], dim=0)
                sampled_data = sampled_data[1:-1]
                data_first_last = first_last.reshape(2 * n, D)
                data = sampled_data.reshape((T-2) * n, D)       # =========
                all_data.append(data_first_last.reshape(-1, 2))
                """

                data1 = sampled_data.reshape(T * n, D)

                all_data.append(data1.reshape(-1, 2))

    energies = [U(data) for data in all_data]
    target_point1 = torch.tensor([-0.767, 0.635])
    target_point2 = torch.tensor([0.22, 0.3])
    energy1 = U(target_point1.unsqueeze(0))
    energy2 = U(target_point2.unsqueeze(0))
    print("energy of saddle point (-0.767, 0.635): ", energy1)
    print("energy of saddle point (0.22, 0.3): ", energy2)

    energies = torch.cat(energies, dim=0)
    energies = energies.reshape(T, -1)

    maxenergy, _ = torch.max(energies, dim=0)
    min_maxenergy = torch.min(maxenergy)
    mean_energy = torch.mean(maxenergy)
    std_energy = torch.std(maxenergy)

    print("energies: ", energies)
    print("Max energy: ", maxenergy)
    print("MIN max energy: ", min_maxenergy)
    print("mean: ", mean_energy)
    print("std: ", std_energy)

    alldata = torch.cat(all_data, dim=0)
    alldata = alldata.reshape(T, -1, 2)
    # target_point = torch.tensor([0.22, 0.3])
    target_point = torch.tensor([-0.766, 0.632])
    diff = alldata - target_point
    squared_diff = diff ** 2
    distances = torch.sqrt(torch.sum(squared_diff, dim=-1))

    min_dist,_ = torch.min(distances, dim=0)
    mean_distance = torch.mean(min_dist)
    std_distance = torch.std(min_dist)

    print(f"{target_point} distances: ", distances)
    print("mean distance: ", mean_distance)
    print("std distance: ", std_distance)

    def plot_energy(energy, data, step=100, extent=[(-1.5, 0.9), (-0.5, 1.7)], resolution=150, dim=2):
        x, y = np.linspace(extent[0][0], extent[0][1], resolution), np.linspace(extent[1][0], extent[1][1],
                                                                                resolution)
        x, y = np.meshgrid(x, y)
        xy = torch.tensor(np.stack([x, y], -1).reshape(-1, 2), dtype=torch.float32)
        us = energy(xy).reshape([resolution, resolution]).detach().numpy()
        plt.contourf(x, y, us, 50)
        colors = ['red', 'blue', 'green', 'yellow']
        labels = ['trajectories', 'saddle point 1', 'saddle point 2', 'Data from Dir4']
        for i, d in enumerate(data):
            if i == 1:
                plt.scatter(d[::step, 0], d[::step, 1], marker='.', color=colors[3], alpha=0.5, s=5)
            else:
                plt.scatter(d[::step, 0], d[::step, 1], marker='.', color=colors[0], alpha=0.5, s=0.5)
                plt.scatter([], [], marker='.', color=colors[0],  alpha=0.5, s=50)  # label=labels[i],

        # saddle points
        plt.scatter(-0.767, 0.635, marker='*', color=colors[3], alpha=1, s=150)
        plt.scatter(0.22, 0.3, marker='*', color=colors[3], alpha=1, s=150)
        plt.scatter(-0.55828035, 1.44169, marker='.', color=colors[2], alpha=1, s=400)
        plt.scatter(0.62361133, 0.02804632, marker='.', color=colors[2], alpha=1, s=400)
        # plt.legend()
        plt.xlim(extent[0])
        plt.ylim(extent[1])
        plt.xticks(fontsize=15)

        current_yticks = plt.gca().get_yticks()
        reduced_yticks = current_yticks[::2]
        plt.yticks(reduced_yticks, fontsize=15)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    plot_energy(U, all_data, step=step)


if __name__ == "__main__":
    A = torch.tensor([[-0.55828035, 1.44169]], dtype=torch.float32)
    B = torch.tensor([[0.62361133, 0.02804632]], dtype=torch.float32)
    C = torch.tensor([[-0.05004308, 0.46666032]], dtype=torch.float32)

    # generate_data_and_save(A, file='x0s', num_points=2000, dt=0.0001, save_interval=1, directory=r"C:\Users\Administrator\Desktop\GFM\toy_data")
    # generate_data_and_save(B, file='x1s', num_points=2000, dt=0.0001, save_interval=1, directory=r"C:\Users\Administrator\Desktop\GFM\toy_data")
    # generate_data_and_save(C, file='xms', num_points=4000, dt=0.0001, save_interval=1, directory=r"C:\Users\Administrator\Desktop\GFM\toy_data")

    # plot_muller_train_data(
    #      "C:/Users/Administrator/Desktop/GFM/toy_data", step=1
    # )

    # Reading in 2D trajectories data
    estimation_for_muller([
         r"C:/Users/Administrator/Desktop/GFM/traj",
    ], step=1)
