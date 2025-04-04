import os
import pyemma
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt


dcd_file = r"E:\chignolin_results\DEShaw_research_chignolin\DESRES-Trajectory_CLN025-0-protein\DESRES-Trajectory_CLN025-0-protein\CLN025-0-protein"  # DCD 文件路径
topology_file = r"E:\chignolin_results\DEShaw_research_chignolin\DESRES-Trajectory_CLN025-0-protein\DESRES-Trajectory_CLN025-0-protein\CLN025-0-protein\chignolin.pdb"  # PDB 文件路径，包含分子的拓扑信息

dcd_files = [os.path.join(dcd_file, f) for f in os.listdir(dcd_file) if f.endswith('.dcd')]
dcd_files.sort()

# ******************************** RMSD **************************************** #

traj = md.load(dcd_files, top=topology_file)
x = traj.xyz

reference = md.load_dcd(dcd_file + '/CLN025-0-protein-000.dcd', top=topology_file)
# reference = md.load_dcd('ref.dcd', top=topology_file)

CA_atoms = reference.topology.select('name CA and resid 2 to 36')

rmsd = []

for traj_name in dcd_files:
    traj = md.load_dcd(traj_name, top=topology_file)
    for element in md.rmsd(traj, reference, 1500, atom_indices=CA_atoms):
        rmsd.append(element)

# fig = plt.figure(figsize=(17, 2))
# plt.plot(rmsd[::500])
# plt.axis([0, 100, 0.0, 0.5])
# plt.ylabel('RMSD(nm)')
# plt.xlabel('Snapshot Num./500')
# plt.show()


#histogram
fig = plt.figure(figsize=(5, 3))

ax1 = fig.add_subplot(111)
ax1.hist(rmsd[::100], density=True, bins=30, color='g', alpha=0.5, edgecolor='r')
ax1.set_xlabel('RMSD$(\AA)$', fontsize=12)
ax1.set_ylabel('Probability Dens.', fontsize=12)
plt.show()

# to Angstrom
rmsd = np.array(rmsd) * 10.0

threshold = 2

above_threshold = rmsd > threshold
below_threshold = rmsd <= threshold

above_indices = np.where(above_threshold)[0]
below_indices = np.where(below_threshold)[0]

above_values = rmsd[above_threshold]
below_values = rmsd[below_threshold]

print(f"大于 {threshold} 的索引: {above_indices}")
print(f"大于 {threshold} 的值: {above_values}")
print(f"小于等于 {threshold} 的索引: {below_indices}")
print(f"小于等于 {threshold} 的值: {below_values}")
print(f"\n统计信息:")
print(f"总数据点: {len(rmsd)}")
print(f"大于 {threshold} 的数量: {len(above_values)} ({len(above_values)/len(rmsd):.1%})")
print(f"小于等于 {threshold} 的数量: {len(below_values)} ({len(below_values)/len(rmsd):.1%})")

# ------------------------ #
x0 = x[above_indices, :, :]
x1 = x[below_indices, :, :]

# ******************************** TICA **************************************** #

features = pyemma.coordinates.featurizer(topology_file)
features.add_residue_mindist()

fig, axes = plt.subplots(1, 1, figsize=(8, 8))
source_dcd = pyemma.coordinates.source([dcd_files], features=features, chunksize=x.shape[0])
tica = pyemma.coordinates.tica(data=source_dcd, lag=10, dim=2)
tica_result_d = tica.get_output()[0]

tica_result = tica_result_d
x0_indices = np.random.choice(above_indices, size=300, replace=False)
x1_indices = np.random.choice(below_indices, size=300, replace=False)

pyemma.plots.plot_free_energy(*tica_result.T, ax=axes)

axes.scatter(*tica_result[x0_indices, :].T, marker='.', c='r', s=30, label='x0')
axes.scatter(*tica_result[x1_indices, :].T, marker='.', c='b', s=30, label='x1')

axes.legend()
axes.set_xlabel('TIC 1 / a.u.')
axes.set_ylabel('TIC 2 / a.u.')

fig.tight_layout()

plt.show()

# MSM estimation
# msm = [pyemma.msm.estimate_markov_model(cluster.dtrajs, lag=lag, dt_traj='0.0002 us') for lag in lags]
