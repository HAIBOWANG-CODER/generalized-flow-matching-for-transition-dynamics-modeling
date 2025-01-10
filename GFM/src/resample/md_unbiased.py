from tqdm import tqdm
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import torch
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from src.resample.angles import plot_phi_psi_angles


def V_t_func(xt_path, pdb_file_path):
    T, N, D = xt_path.shape
    energies = torch.zeros((T, N))
    total_energy = 0.0

    pdb = app.PDBFile(pdb_file_path)

    forcefield = app.ForceField('amber99sbildn.xml')

    system = forcefield.createSystem(
        pdb.topology,
        # nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds
    )

    integrator = mm.LangevinIntegrator(
        300 * unit.kelvin,
        1 / unit.picoseconds,
        2 * unit.femtoseconds
    )

    simulation = app.Simulation(pdb.topology, system, integrator)

    energies = torch.rand(T, N)
    for t in tqdm(range(T), desc='computing the energy:'):
        for n in range(N):
            xt = xt_path[t, n].reshape(-1, 3)
            positions = [mm.Vec3(x[0], x[1], x[2]) for x in xt]

            simulation.context.setPositions(positions)
            state = simulation.context.getState(getEnergy=True)
            potential_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            total_energy = potential_energy

            energies[t, n] = total_energy

    return energies


# Generate a training data set and store it in a tensor, and then store it in a pt file
def run_simulation(temp,
                   saveFreq,
                   unbiasDir,
                   total_steps,
                   threshold_step,
                   device_index,
                   precision,
                   pdb_path,
                   output_filename,
):
    # Create unbiased directory if it doesn't exist
    if not os.path.isdir(unbiasDir):
        os.mkdir(unbiasDir)

    # Load PDB file
    pdb = app.PDBFile(pdb_path)

    # Create force field and system
    forcefield = app.ForceField('amber99sbildn.xml')
    system = forcefield.createSystem(
        pdb.topology,
        # nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds
    )

    # Create integrator and simulation
    integrator = mm.LangevinIntegrator(
        temp * unit.kelvin,
        1 / unit.picoseconds,
        2 * unit.femtoseconds
    )

    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': device_index, 'Precision': precision}

    # Initialize simulation
    modeller = app.Modeller(pdb.topology, pdb.positions)
    simulation = app.Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temp)

    # Reporters
    simulation.reporters.append(app.StateDataReporter(
        "unbiased.log",
        saveFreq,
        step=True,
        potentialEnergy=True,
        temperature=True,
        volume=True,
        speed=True,
        remainingTime=True,
        elapsedTime=True,
        totalSteps=total_steps
    ))

    # Collect positions over time
    positions_list = []

    for i in tqdm(range(total_steps // saveFreq), desc='Generating train data:'):
        simulation.step(saveFreq)
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if i*saveFreq >= threshold_step:
            positions_list.append(positions)

    # Convert positions to a tensor of shape (T, N, D)
    positions_tensor = torch.tensor(np.array(positions_list))
    T, N, D = positions_tensor.shape
    x = positions_tensor.reshape([T, 1, N*D])
    print('The shape of the first data set is', x.shape)

    plot_phi_psi_angles(x, 'r', 'Trajectories', point_size=5)

    # Save tensor to .pt file
    tensor_file_path = os.path.join(unbiasDir, output_filename)

    torch.save(x, tensor_file_path)
    print(f"Tensor 0 saved to {tensor_file_path}")


def plot_energies(energies, indices):
    T, N = energies.shape
    time_steps = range(T)

    plt.figure(figsize=(12, 6))
    for n in indices:
        plt.plot(time_steps, energies[:, n].cpu().numpy())

    plt.xlabel('T')
    plt.ylabel('U(x) (kJ/mol)')
    plt.show()


def plot_energies_picture(energies, indices):
    T, N = energies.shape
    time_steps = range(T)

    plt.figure(figsize=(28, 4))
    for n in indices:
        plt.plot(time_steps, energies[:, n].cpu().numpy(), linewidth=2)

        positions = [0, 99, 199, 299, 399, 499]
        # =================================
        for pos in positions:
            plt.annotate('', xy=(pos, energies[pos, n]), xytext=(pos, energies[pos, n] + 1000),
                         arrowprops=dict(arrowstyle="->", color='orange', linestyle='--', lw=3,
                                         mutation_scale=30))

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


def plot_paths_energy(xt_path,
                       threshold=2000,
                       last_time_threshold=10000,
                       num_indices=100):
    """
    Processes the trajectory file, filters paths based on energy thresholds,
    and plots or saves the resulting paths with low energy.

    Parameters:
    xt_path (tensor): The dims of trajectories is (T, N, D).
    threshold (float): Maximum energy threshold for filtering.
    last_time_threshold (float): Energy threshold for the last point in the path.
    num_random_indices (int): Number of random indices to select for processing.
    """

    pdb_file_path_A = r"pdb_data\AD_A.pdb"

    # Randomly select indices
    xt_path = xt_path[:, :num_indices, :]

    # Calculate energies
    energies = V_t_func(xt_path, pdb_file_path_A)
    # Energy statistics
    maxenergy, _ = torch.max(energies, dim=0)
    min_maxenergy = min(maxenergy)
    mean_energy = torch.mean(maxenergy)
    std_energy = torch.std(maxenergy)

    print("energies: ", energies)
    print("Max energy: ", maxenergy)
    print("MIN max energy: ", min_maxenergy)
    print("mean: ", mean_energy)
    print("std: ", std_energy)

    # Filter based on last point energy threshold
    last_elements = energies[-1, :]
    indices = np.where(last_elements < last_time_threshold)[0]
    energies_extra = energies[:, indices]
    xt_path_extra = xt_path[:, indices, :]

    # Further filter based on overall energy threshold
    low_energy_indices = [n for n in range(energies_extra.shape[1]) if energies_extra[:, n].max() < threshold]

    # Save or plot the results (uncomment the following lines if needed)
    # torch.save(xt_path_extra[:, low_energy_indices, :], r"traj\best.pt")  # Save the paths with low energy
    plot_energies(energies_extra, low_energy_indices)                 # For test picture
    # vplot_energies_picture(energies_extra, low_energy_indices)       # For paper picture


# Generate alanine dipeptide data (.pt file)
def generate_alanine_data(temp=300, saveFreq=20, unbiasDir="pdb_data/alanine_data_cartesian",
                          total_steps=int(6e5), threshold_step=0, device_index='0', precision='mixed',
                          init_pdb_path_A="pdb_data\AD_A.pdb", init_pdb_path_B="pdb_data\AD_B.pdb",
                          out_tensor_A=None, out_tensor_B=None, img_path=r"background\background.png",
                          create_file=None):

    if out_tensor_A is None:
        out_tensor_A = "x0s.pt"
    if out_tensor_B is None:
        out_tensor_B = "x1s.pt"

    plt.figure(figsize=(10, 10))
    img = mpimg.imread(img_path)
    plt.imshow(img, extent=[-np.pi, np.pi, -np.pi, np.pi], alpha=0.7)

    if create_file == 'x0s':
        run_simulation(temp, saveFreq, unbiasDir, total_steps, threshold_step,
                       device_index, precision, init_pdb_path_A, out_tensor_A)
    elif create_file == 'x1s':
        run_simulation(temp, saveFreq, unbiasDir, total_steps, threshold_step,
                       device_index, precision, init_pdb_path_B, out_tensor_B)
    plt.show()


if __name__ == '__main__':

    generate_alanine_data(create_file='x0s')
    generate_alanine_data(create_file='x1s')

    # Search the path energy with different energy thresholds
    plot = True

    if plot:
        xt_path = torch.load(r"~\resample_data.pt")
        plot_paths_energy(xt_path, threshold=510, last_time_threshold=1000, num_indices=100)
