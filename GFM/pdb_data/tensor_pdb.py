import mdtraj as md
import torch
import os
import tempfile
from tqdm import tqdm


def read_pdb(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    pdb = md.load_pdb(file_path)
    return pdb, lines


def write_pdb(file_path, lines):
    with open(file_path, 'w') as file:
        file.writelines(lines)


def replace_coordinates(pdb, tensor):
    new_xyz = tensor.reshape(-1, 3)
    pdb.xyz[0] = new_xyz / 10
    return pdb


def extract_conect_records(lines):
    conect_records = []
    for line in lines:
        if line.startswith("CONECT"):
            conect_records.append(line)
    return conect_records


def remove_conect_records(lines):
    return [line for line in lines if not line.startswith("CONECT")]


def remove_end_records(lines):
    return [line for line in lines if not line.startswith("END")]


def remove_models_records(lines):
    return [line for line in lines if not line.startswith("MODEL")]


pdb_file_path = r"~\pdb_data\AD_A.pdb"

tensor_path = r"~\traj\best.pt"     # Conservative trajectories file path

extraction = False

assert os.path.isfile(pdb_file_path), f"PDB file not found: {pdb_file_path}"


def save(pdb_file_path,tensor_path):
    pdb, pdb_lines = read_pdb(pdb_file_path)
    pdb_ext = extract_conect_records(pdb_lines)
    pdb_lines = remove_conect_records(pdb_lines)

    tensor_data = torch.load(tensor_path)
    # tensor_data = torch.stack(tensor_data, dim=1)
    if tensor_data.dim() != 3:
        tensor_data = tensor_data.unsqueeze(1)

    tensor_data = tensor_data.detach().numpy()
    tensor_data_base = tensor_data

    if extraction == False:

        for j in range(tensor_data_base.shape[1]):
            if j % 1 == 0:
                tensor_data = tensor_data_base[:, j:j + 1, :] * 10    # Tensor of reading (t, b, n) generates the corresponding PDB file
                # tensor_data = tensor_data_base[j:j+1, :] * 10       # Tensor of reading (b, n) generates the corresponding PDB file

                tensor_data = tensor_data.squeeze(1)
                merged_pdb_lines = []
                for i in tqdm(range(tensor_data.shape[0]), desc="Processing models"):
                    sub_tensor = tensor_data[i]
                    sub_tensor = sub_tensor.reshape(22, 3)
                    modified_pdb = replace_coordinates(pdb, sub_tensor)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_file:
                        modified_pdb.save_pdb(temp_file.name)
                        temp_file_path = temp_file.name

                    with open(temp_file_path, 'r') as temp_file:
                        lines = temp_file.readlines()

                    lines = remove_conect_records(lines)
                    lines = remove_end_records(lines)
                    lines = remove_models_records(lines)

                    merged_pdb_lines.append(f"MODEL          {i + 1}\n")
                    merged_pdb_lines.extend(lines)
                    merged_pdb_lines.append('ENDMDL\n')

                merged_pdb_lines.extend(pdb_ext)
                merged_pdb_lines.append('END\n')
                temp_file_path = f"~/traj/traj_{j}.pdb"
                write_pdb(temp_file_path, merged_pdb_lines)

    else:
        merged_pdb_lines = []

        for i in range(tensor_data_base.shape[1]):
            if i % 1000 == 0:
                tensor_data = tensor_data_base[:, i:i + 1, :] * 10
                tensor_data = tensor_data.squeeze()
                for j in range(tensor_data.shape[0]):
                    sub_tensor = tensor_data[j]
                    sub_tensor = sub_tensor.reshape(22, 3)

                    modified_pdb = replace_coordinates(pdb, sub_tensor)

                    temp_file_path = f"tmp/temp_model_{i}.pdb"
                    modified_pdb.save_pdb(temp_file_path)

                    with open(temp_file_path, 'r') as temp_file:
                        lines = temp_file.readlines()

                    lines = remove_conect_records(lines)
                    lines = remove_end_records(lines)
                    lines = remove_models_records(lines)
                    merged_pdb_lines.append(f"MODEL          0\n")
                    merged_pdb_lines.extend(lines)
                    merged_pdb_lines.append('ENDMDL\n')

                    merged_pdb_lines.extend(pdb_ext)

                    merged_pdb_lines.append('END\n')
                    temp_file_path = f"xts/xt_{i}_{j}.pdb"
                    write_pdb(temp_file_path, merged_pdb_lines)
                    merged_pdb_lines = []

                    # modified_pdb.save_pdb(temp_file_path)


save(pdb_file_path, tensor_path)


