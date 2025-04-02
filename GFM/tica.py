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

ref_pdb = md.load(topology_file)
ref = ref_pdb.xyz
ref_traj = md.Trajectory(ref, ref_pdb.topology)
ref_traj.save('ref.dcd')

rmsd = md.rmsd(traj, ref_pdb, 0)

plt.figure(figsize=(10, 6))
plt.hist(rmsd, bins=50, color='skyblue', edgecolor='black')
plt.title('RMSD Distribution Histogram', fontsize=14)
plt.xlabel('RMSD (nm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
# plt.legend()
plt.grid(True, alpha=0.3)
# plt.show()

threshold = 0.7

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

x0_traj = md.Trajectory(x0, ref_pdb.topology)
x1_traj = md.Trajectory(x1, ref_pdb.topology)
x0_traj.save('x0.dcd')
x1_traj.save('x1.dcd')

# ******************************** TICA **************************************** #

features = pyemma.coordinates.featurizer(topology_file)
features.add_residue_mindist()

x0_file = 'x0.dcd'
x1_file = 'x1.dcd'
ref_file = 'ref.dcd'

files = [dcd_files, x0_file, x1_file, ref_file]
chunk_sizes = [x.shape[0], x0.shape[0], x1.shape[0], ref.shape[0]]
colors = ['grey', 'red', 'blue', 'green']
labels = ['centers', 'centers x0', 'centers x1', 'centers ref']
markers = ['x', '.', '.', '*']
ks = [64, 256, 256, 1]
sizes = [30, 30, 30, 500]

fig, axes = plt.subplots(1, 1, figsize=(8, 8))

source_dcd = pyemma.coordinates.source([dcd_files], features=features, chunk_size=chunk_sizes[0])
tica = pyemma.coordinates.tica(data=source_dcd, lag=10, dim=2)
tica_result_dcd = tica.get_output()[0]

for file, chunk_size, color, label, marker, k, size in zip(files, chunk_sizes, colors, labels, markers, ks, sizes):

    if file == dcd_files:
        tica_result = tica_result_dcd
        pyemma.plots.plot_free_energy(*tica_result.T, ax=axes)

    else:
        source = pyemma.coordinates.source([file], features=features, chunk_size=chunk_size)
        data = source.get_output()[0]
        tica_result = tica.transform(data)
        cluster_result = pyemma.coordinates.cluster_kmeans(tica_result, k=k, max_iter=100)
        axes.scatter(*cluster_result.clustercenters.T, marker=marker, c=color, s=size, label=label)

axes.legend()
axes.set_xlabel('TIC 1 / a.u.')
axes.set_ylabel('TIC 2 / a.u.')

fig.tight_layout()

plt.show()

# MSM estimation
# msm = [pyemma.msm.estimate_markov_model(cluster.dtrajs, lag=lag, dt_traj='0.0002 us') for lag in lags]
print(1)