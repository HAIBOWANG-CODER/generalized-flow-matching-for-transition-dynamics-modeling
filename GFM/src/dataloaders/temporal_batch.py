import torch
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.combined_loader import CombinedLoader
import numpy as np
from torch.utils.data import Dataset
import os
from pdb_data.internal import cartesian_to_internal
from src.resample.md_unbiased import generate_alanine_data
from toy_data.main_2D import generate_data_and_save
from toy_data.main_2D import plot_muller_train_data


class IndexDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        return data_sample, idx


def check_data_files(directory):
    required_files = ['x0s.pt', 'x1s.pt']
    existing_files = os.listdir(directory)
    existing_files_without_ext = [os.path.splitext(f)[0] for f in existing_files]
    missing_files = [os.path.splitext(f)[0] for f in required_files if
                     os.path.splitext(f)[0] not in existing_files_without_ext]

    return missing_files


def check_and_create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"This directory already exists: {directory}")


def read_data(paths, data_type):

    check_and_create_directory(paths)
    no_data_dirs = check_data_files(paths)
    if no_data_dirs:
        for directory in no_data_dirs:
            if data_type in ('ADC', 'ADI'):
                generate_alanine_data(create_file=directory)

            elif data_type == 'muller':
                if directory == 'x0s':
                    initial_pos = torch.tensor([[-0.55828035, 1.44169]], dtype=torch.float32)
                elif directory == 'x1s':
                    initial_pos = torch.tensor([[0.62361133, 0.02804632]], dtype=torch.float32)
                else:
                    raise ValueError
                
                generate_data_and_save(initial_pos, file=directory, num_points=2000, dt=0.0001, save_interval=1)
        if data_type == 'muller':
            plot_muller_train_data(paths, step=1)
    all_data = []
    all_labels = []

    folder_path = paths
    pt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]

    for i, file_path in enumerate(pt_files):
        adata = torch.load(file_path)
        if data_type in ('ADC', 'ADI') and adata.size(2) != 66:
            raise ValueError("The number of atoms read in is not equal to alanine dipeptide")
        if data_type == 'muller' and adata.size(1) != 2:
            raise ValueError("The dimension of muller brown data read in is not equal to 2")

        all_data.append(adata)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if file_name == 'x0s':
            all_labels.extend([0] * adata.size(0))
        elif file_name == 'x1s':
            all_labels.extend([10] * adata.size(0))
        else:
            all_labels.extend([i] * adata.size(0))

    all_data = torch.cat(all_data, dim=0)
    ulabels = np.unique(all_labels)
    combined_labels = np.array(all_labels)
    combined_data = np.vstack(all_data)

    if data_type == 'ADC':
        combined_data1 = combined_data * 100   # Convert units nm to 0.1 angstrom
    elif data_type == 'ADI':
        combined_data0 = cartesian_to_internal(combined_data)
        combined_data1 = combined_data0.numpy()
    elif data_type == 'muller':
        combined_data1 = combined_data
    else:
        raise ValueError("Data type not recognized")

    return (combined_data1, combined_labels, ulabels)


def read_reflow_data(save_address):
    xt = torch.load(f"{save_address}//ode_x_forward.pt")
    if xt.is_cuda:
        xt = xt.cpu()
    xt = xt.numpy()
    label = np.ones(xt.shape[0])
    unique_label = np.unique(label)
    return xt, label, unique_label


def read_bidirectional_data(save_address):
    xt_forward = torch.load(f"{save_address}//ode_x_forward.pt")
    xt_backward = torch.load(f"{save_address}//ode_x_backward.pt")

    min_length = min(xt_forward.shape[0], xt_backward.shape[0])
    xt_forward = xt_forward[:min_length]
    xt_backward = xt_backward[:min_length]

    xt = torch.stack([xt_forward, xt_backward], dim=2)
    if xt.is_cuda:
        xt = xt.cpu()
    xt = xt.numpy()
    label = np.ones(xt.shape[0])
    unique_label = np.unique(label)
    return xt, label, unique_label


class TemporalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
        now_resample=0,
        now_reflow=0,
        skipped_datapoint=-1,
        direc=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_type = args.data_type
        self.data_path = args.data_path
        self.batch_size = args.batch_size if now_resample == 0 else args.batch_size_i
        self.split_ratios = args.split_ratios
        self.max_dim = args.dim
        self.hvg = args.hvg
        self.whiten = args.whiten
        self.skipped_datapoint = skipped_datapoint
        self.whiten_test = args.whiten_test
        self.dataset_num = args.dataset_num
        self.now_reflow = now_reflow
        self.direc = direc
        self.save_address = args.save_address
        self._prepare_data()

    def _prepare_data(self):
        self.train_dataloaders = []
        self.val_dataloaders = []
        self.test_dataloaders = []
        self.metric_samples_dataloaders = []
        labels = []
        unique_labels = []
        ds = []

        if self.now_reflow == 0:
            ds, labels, unique_labels = read_data(self.data_path, self.data_type)

            if self.dataset_num != len(unique_labels):
                raise ValueError(f"The dataset_num setting is not equal to {len(unique_labels)}")
            if self.max_dim != ds.shape[-1]:
                raise ValueError(f"The dim setting is not equal to {ds.shape[-1]}")

            if self.whiten:
                self.scaler = StandardScaler()
                ds = self.scaler.fit_transform(ds)
            elif self.whiten_test:
                self.scaler = StandardScaler()
                self.scaler.fit(ds)
            self.num_timesteps = len(unique_labels)
        else:
            if self.direc == 'unbidirectional':
                ds, labels, unique_labels = read_reflow_data(self.save_address)
            elif self.direc == 'bidirectional':
                ds, labels, unique_labels = read_bidirectional_data(self.save_address)
            self.num_timesteps = 2

        ds_tensor = torch.tensor(ds, dtype=torch.float32)
        label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
        print("Label to numeric: ", label_to_numeric)
        frame_indices = {
            label_to_numeric[label]: (labels == label).nonzero()[0]
            for label in unique_labels
        }

        min_frame_size = min([len(indices) for indices in frame_indices.values()])
        for label, indices in frame_indices.items():
            print("Processing Label: ", label)
            frame_data = ds_tensor[indices]
            split_index = int(len(frame_data) * self.split_ratios[0])

            # Adjust split_index to ensure minimum validation samples
            if len(frame_data) - split_index < self.batch_size:
                split_index = (
                    len(frame_data) - self.batch_size
                )  # Adjust to leave at least one batch for validation
            # Shuffle data
            shuffled_indices = torch.randperm(len(frame_data))
            frame_data = frame_data[shuffled_indices]
            train_data = frame_data[:split_index]
            val_data = frame_data[split_index:]
            self.train_dataloaders.append(
                DataLoader(
                    IndexDataset(train_data),
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                )
            )
            self.val_dataloaders.append(
                DataLoader(
                    IndexDataset(val_data),
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=True,
                )
            )
            self.test_dataloaders.append(
                DataLoader(
                    frame_data,
                    batch_size=frame_data.shape[0],
                    shuffle=False,
                    drop_last=False,
                )
            )
            self.metric_samples_dataloaders.append(
                DataLoader(
                    frame_data,
                    batch_size=min_frame_size,  # balanced batches
                    shuffle=True,
                    drop_last=False,
                )
            )

    def train_dataloader(self):
        combined_loaders = {
            "train_samples": CombinedLoader(self.train_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }
        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def val_dataloader(self):
        combined_loaders = {
            "val_samples": CombinedLoader(self.val_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }

        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def test_dataloader(self):
        return CombinedLoader(self.test_dataloaders, "max_size")
