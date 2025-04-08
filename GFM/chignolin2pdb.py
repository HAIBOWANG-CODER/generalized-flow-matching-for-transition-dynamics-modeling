import torch
import mdtraj as md
import os
from tqdm import tqdm

topology_file = r"E:\chignolin_results\DEShaw_research_chignolin\DESRES-Trajectory_CLN025-0-protein\DESRES-Trajectory_CLN025-0-protein\CLN025-0-protein\chignolin.pdb"
traj = torch.load("D:\chignolin_result\splinenet_tensor_0.pt")
# traj = traj/10
traj = traj.cpu().numpy()

ref_pdb = md.load(topology_file)

output_dir = r"E:\chignolin_results\paths"
os.makedirs(output_dir, exist_ok=True)

for i in tqdm(range(traj.shape[1]), desc=f'Save paths:'):
    path_coords = traj[:, i, :].reshape(-1, 166, 3)
    traj_instance = md.Trajectory(path_coords, ref_pdb.topology)
    pdb_filename = os.path.join(output_dir, f'path_{i+1}.pdb')
    traj_instance.save(pdb_filename)

print(f"所有PDB文件已保存到: {output_dir}")