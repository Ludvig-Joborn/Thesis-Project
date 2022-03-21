import torch
from torch.utils.data import DataLoader
from pathlib import Path


class DatasetManager:
    """
    Class for loading datasets into memory and storing them in a dictionary.
    """

    def __init__(self):
        self.datasets = {}

    def _get_ds_entry(self, key: str):
        if key not in self.datasets:
            raise ValueError(f"Dataset with name: {key} is not initialized")
        return self.datasets[key]

    def get_dataset(self, name: str):
        return self._get_ds_entry(name)[0]

    def get_dataloader(self, name: str):
        return self._get_ds_entry(name)[1]

    def load_dataset(
        self,
        DS,  # Dataset Class
        name: str,
        path_annotations: Path,
        path_audio: Path,
        clip_len: int,
        NUM_WORKERS: int,
        PIN_MEMORY: bool,
        batch_size: int,
        shuffle: bool,
    ):
        """
        Loads the dataset into memory and creates a dataloader for it.
        Note: Dictionary 'datasets' contains datasets on the format::

                {name: (dataset, dataloader)}
        """
        DS = DS(name, path_annotations, path_audio, clip_len)
        DS_loader = DataLoader(
            DS,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        self.datasets[name] = (DS, DS_loader)
        return DS_loader

    ### User help-functions ###
    def display_loaded_datasets(self) -> str:
        """
        Display all datasets that are loaded into memory.
        """
        return self.datasets.keys()
