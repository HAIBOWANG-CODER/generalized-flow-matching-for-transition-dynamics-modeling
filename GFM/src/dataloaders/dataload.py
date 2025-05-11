import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from train_data.pdb_data.internal import cartesian_to_internal


class BidirectionalDataset(Dataset):
    def __init__(self, xt_path):
        self.xt = torch.load(xt_path)

    def __len__(self):
        return self.xt.shape[0]

    def __getitem__(self, idx):
        return self.xt[idx]


def get_bidirectional_dataloader(xt_forward_path, xt_backward_path, batch_size=32, shuffle=False):
    dataset_forward = BidirectionalDataset(xt_forward_path)
    dataset_backward = BidirectionalDataset(xt_backward_path)

    dataloader_forward = DataLoader(dataset_forward, batch_size=batch_size, shuffle=shuffle)
    dataloader_backward = DataLoader(dataset_backward, batch_size=batch_size, shuffle=shuffle)
    return dataloader_forward, dataloader_backward


class sampleLoader:
    def __init__(self, path, data_type, max_atoms=66):
        self.path = path
        self.max_atoms = max_atoms
        self.data_type = data_type

    def load_all_data(self):
        all_coords = torch.load(self.path)

        if self.data_type in ('ADC', 'ADI') and all_coords.size(2) != self.max_atoms:
            raise ValueError("The number of atoms read in is not equal to alanine dipeptide")
        if self.data_type == 'muller' and all_coords.size(1) != self.max_atoms:
            raise ValueError("The dimension of muller brown data read in is not equal to 2")

        all_coords = np.array(all_coords, dtype=np.float32)

        if self.data_type == 'ADC':
            all_coords = all_coords.squeeze() * 10  # Convert nm to 0.1angstrom
            all_coords = all_coords.reshape(-1, self.max_atoms)

        elif self.data_type == 'ADI':
            # Convert angstrom to nm and transform Cartesian to Internal
            all_coords = cartesian_to_internal(all_coords.squeeze())
            all_coords = all_coords.numpy()

        elif self.data_type == 'muller':
            all_coords = all_coords
        else:
            raise ValueError("Data type not recognized")
        return all_coords


class SampleDataset(Dataset):
    def __init__(self, directory, data_type, max_atoms, vae_key=False, directory1=None):
        self.loader = sampleLoader(directory, data_type, max_atoms=max_atoms)
        self.data = torch.tensor(self.loader.load_all_data(), dtype=torch.float32)

        if vae_key:
            if directory1 is None:
                raise ValueError("vae_key=True and directory1 is not specified")
            self.loader = sampleLoader(directory1, data_type, max_atoms=max_atoms)
            self.data1 = torch.tensor(self.loader.load_all_data(), dtype=torch.float32)
            self.data = torch.cat((self.data, self.data1), dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
