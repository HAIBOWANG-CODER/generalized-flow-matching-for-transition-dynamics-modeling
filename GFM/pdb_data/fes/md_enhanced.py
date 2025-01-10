import openmm as mm
import openmm.app as app
import openmm.unit as unit
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

dim = 25
total_steps = int(4e8)


def rmsd(V, W):

    # Calculate Root-mean-square deviation from two sets of vectors V and W.
    # Parameters
    # ----------
    # V : array
    #     (N,D) matrix, where N is points and D is dimension.
    # W : array
    #     (N,D) matrix, where N is points and D is dimension.
    # Returns
    # -------
    # rmsd : float
    #     Root-mean-square deviation between the two vectors

    diff = np.array(V) - np.array(W)
    N = len(V)
    return np.sqrt((diff * diff).sum() / N)


temp = 300
init_pdb = app.PDBFile("./AD_A.pdb")
# init_pdb = app.PDBFile("./AD_B.pdb")
# target_pdb

forcefield = app.ForceField('amber99sbildn.xml')

system = forcefield.createSystem(
    init_pdb.topology,
    # nonbondedMethod=app.PME,
    # nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds,
    # nonbondedCutoff=1.0 * unit.nanometer,
    # ewaldErrorTolerance=0.0005
)

pi = float(np.pi)
metadFreq = 1000
biasDir = "./bias"
if not os.path.isdir(biasDir):
    os.mkdir(biasDir)

torsionCVForce_phi = mm.CustomTorsionForce("theta")
torsionCVForce_phi.addTorsion(1, 6, 8, 14)

biasVar_phi = app.BiasVariable(
    torsionCVForce_phi, 
    minValue=-pi,
    maxValue=pi, 
    biasWidth=0.05,
    gridWidth=dim,
    periodic=True,
    # pld=True
)

torsionCVForce_psi = mm.CustomTorsionForce("theta")
torsionCVForce_psi.addTorsion(6, 8, 14, 16)
biasVar_psi = app.BiasVariable(
    torsionCVForce_psi, 
    minValue=-pi,
    maxValue=pi, 
    biasWidth=0.05,
    gridWidth=dim,
    periodic=True,
    # pld=True,
)

metad = app.Metadynamics(
    system, 
    [biasVar_phi, biasVar_psi], 
    temperature=temp*unit.kelvin,
    biasFactor=8.0,
    height=0.2*unit.kilojoules_per_mole,
    frequency=metadFreq,        # metadFreq
    saveFrequency=metadFreq,
    biasDir="./bias"
)

modeller = app.Modeller(init_pdb.topology, init_pdb.positions)

integrator = mm.LangevinIntegrator(
    temp*unit.kelvin, 
    1/unit.picoseconds,
    2*unit.femtoseconds)

# ============================================================
platform = mm.Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
# ============================================================

# simulation = app.Simulation(modeller.topology, system, integrator)
simulation = app.Simulation(modeller.topology, system, integrator, platform, properties)

simulation.context.setPositions(modeller.positions)
simulation.context.setVelocitiesToTemperature(temp)
# simulation.context.setVelocities(nptVelocities)
# simulation.context.setPeriodicBoxVectors(*nptBoxVec)

# simulation.reporters.append(app.PDBReporter("pdb", metadFreq))
simulation.reporters.append(app.DCDReporter("output.dcd", metadFreq*10000))


simulation.reporters.append(
    app.StateDataReporter(
        "metaD.log", 
        metadFreq*10000,
        step=True, 
        potentialEnergy=True, 
        temperature=True, 
        volume=True, 
        speed=True,
        remainingTime=True,
        elapsedTime=True,
        totalSteps=total_steps
    )
)

past_CV, new_CV = None, None
for i in tqdm(range(total_steps // metadFreq)):
    metad.step(simulation, metadFreq)
    if i % 100 == 0:
        cv = metad.getCollectiveVariables(simulation)
        free_energy = metad.getFreeEnergy()
        if i > 0:
            past_CV = new_CV
            new_CV = free_energy
            free_energy_diff = rmsd(past_CV - past_CV.min(), new_CV - new_CV.min())
            # print (free_energy, free_energy.shape, len(free_energy), free_energy_diff)
            with open(f'./{biasDir}/fes_diff.txt', 'a') as f:
                f.write(f"{free_energy_diff}\n")
        new_CV = free_energy
        with open(f'./{biasDir}/cv.txt', 'a') as f:
            f.write(f"{cv[0]}, {cv[1]}\n")
        with open(f'./{biasDir}/fes.txt', 'w') as f:
            # f.write(f"{free_energy}\n")
            for row in free_energy:
                values = [float(value._value) for value in row]
                f.write(' '.join(f'{val:.6f}' for val in values) + '\n')

metaDpositions = simulation.context.getState(getPositions=True).getPositions()
# finalizeReporters(simulation.reporters)

# write final frame into PDB file anyhow
# if "pdb" not in prodReportFormats:
#     with open(f"./{biasDir}/metaD.pdb", 'w') as f:
#         app.PDBFile.writeFile(modeller.topology, metaDpositions, f)

# ================================= Plot fes================================================ #


def read_fes_file(filepath):
    last_block = None
    current_block = []

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip().replace('[', '').replace(']', '').replace('kJ/mol', '').replace(',', '')
            if line:
                try:
                    values = [float(x) for x in line.split()]
                    current_block.extend(values)
                    if len(current_block) >= dim*dim:
                        last_block = np.array(current_block[:dim*dim]).reshape((dim, dim))
                        current_block = []
                except ValueError as e:
                    print(f"Skipping line due to error: {e} - {line}")

    if len(current_block) == dim*dim:
        last_block = np.array(current_block)  # .reshape((dim,dim))

    if last_block is not None:
        return last_block
    else:
        raise ValueError("The file does not contain enough data for a 20x20 block.")

def read_cv_file(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip().replace('[', '').replace(']', '')
            if line:
                try:
                    data.append([float(x) for x in line.split(',')])
                except ValueError as e:
                    print(f"Skipping line due to error: {e} - {line}")
    return np.array(data)


fes_file = "bias/fes.txt"
cv_file = "bias/cv.txt"

# read data
fes_data = read_fes_file(fes_file)
cv_data = read_cv_file(cv_file)

fes_data = fes_data.reshape(-1, dim, dim)

fes_data = fes_data[-1, :, :].squeeze()
fes_data = fes_data - fes_data.min()
# fes_data = np.flipud(fes_data)

# ==================================
# fes_data = np.log(np.flipud(fes_data)+.000001)
# fes_data = -(0.001987*300)*fes_data
# ==================================

print(f"FES data shape: {fes_data.shape}")
print(f"CV data shape: {cv_data.shape}")

phi = cv_data[:, 0]
psi = cv_data[:, 1]

# phi_unique = np.unique(phi)
# psi_unique = np.unique(psi)

# ==============
phi_unique = np.linspace(-3.14, 3.14, dim)
psi_unique = np.linspace(-3.14, 3.14, dim)
psi_grid_high_res, phi_grid_high_res = np.meshgrid(psi_unique, phi_unique)
cmap = plt.get_cmap('jet')
levels = np.linspace(fes_data.min(), fes_data.max(), 15)
fig, ax = plt.subplots(figsize=(8, 6))
contour_filled = ax.contourf(psi_grid_high_res, phi_grid_high_res, fes_data, levels=levels, cmap=cmap)

contour_lines = ax.contour(psi_grid_high_res, phi_grid_high_res, fes_data, levels=levels, colors='black', linewidths=0.5, linestyles='solid')

# add the colorbar
cbar = plt.colorbar(contour_filled, ax=ax, ticks=levels, format='%.1f', aspect=15)
cbar.set_label('kJ/mol', fontsize=20, labelpad=15)

plt.xlim(-3.14, 3.14)
plt.ylim(-3.14, 3.14)
plt.ylabel("psi", size=25, labelpad=15)
plt.xlabel("phi", size=25, labelpad=15)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.show()
# ==================

# Load the data from the file
file_path = "bias/fes_diff.txt"
data = np.loadtxt(file_path)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data, marker='o', markersize=2, linestyle='-', color='blue')
plt.title('RMSD Values Over Time')
plt.xlabel('Iteration')
plt.ylabel('RMSD (kJ/mol)')
plt.grid(True)
plt.tight_layout()
plt.show()
