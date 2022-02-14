import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List

# User defined imports
from config import *

# Dataset classes
from datasets.desed import DESED_Strong


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

    def _dataset_helper(
        self,
        DS_class,  # Dataset Class
        name: str,
        path_to_annotations: Path,
        path_to_audio: Path,
        clip_len: int,
        NUM_WORKERS: int,
        PIN_MEMORY: bool,
    ):
        """
        Loads the dataset into memory and creates a dataloader for it.
        Note: Dictionary 'datasets' contains datasets on the format::

                {name: (dataset, dataloader)}
        """
        DS = DS_class(name, path_to_annotations, path_to_audio, clip_len)
        DS_loader = DataLoader(
            DS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        self.datasets[name] = (DS, DS_loader)
        return DS_loader

    ### Available Datasets ###
    def load_DESED_strong_synth_train(self, name: str):
        DS_loader = self._dataset_helper(
            DESED_Strong,
            name,
            PATH_TO_SYNTH_TRAIN_DESED_TSV,
            PATH_TO_SYNTH_TRAIN_DESED_WAVS,
            DESED_CLIP_LEN_SECONDS,
            NUM_WORKERS_TRAIN_DESED,
            PIN_MEMORY,
        )
        return DS_loader

    def load_DESED_strong_synth_val(self, name: str):
        DS_loader = self._dataset_helper(
            DESED_Strong,
            name,
            PATH_TO_SYNTH_VAL_DESED_TSV,
            PATH_TO_SYNTH_VAL_DESED_WAVS,
            DESED_CLIP_LEN_SECONDS,
            NUM_WORKERS_VAL_DESED,
            PIN_MEMORY,
        )
        return DS_loader

    def load_DESED_strong_public_test(self, name: str):
        DS_loader = self._dataset_helper(
            DESED_Strong,
            name,
            PATH_TO_PUBLIC_TEST_DESED_TSV,
            PATH_TO_PUBLIC_TEST_DESED_WAVS,
            DESED_CLIP_LEN_SECONDS,
            NUM_WORKERS_TEST_DESED,
            PIN_MEMORY,
        )
        return DS_loader

    ### User help-functions ###

    def display_loaded_datasets(self) -> str:
        """
        Display all datasets that are loaded into memory.
        """
        return self.datasets.keys()

    def get_all_function_names(self) -> List[str]:
        """
        User help-function to show available function names in DatasetManager class.
        """
        method_list = [
            attribute
            for attribute in dir(DatasetManager)
            if callable(getattr(DatasetManager, attribute))
            and attribute.startswith("_") is False
        ]
        return method_list

    def get_loadable_datasets(self) -> List[str]:
        """
        Useful to see what datasets are available to load.
        """
        method_list = self.get_all_function_names()
        return [f for f in method_list if f.startswith("load_")]


class DatasetWrapper:
    """
    Class for handling training, validation and test-datasets/dataloaders.
    Acts as a wrapper for DatasetManager.
    """

    def __init__(self):
        """
        Uses the DatasetManager class to load a training, validation and test-dataset.
        """
        self.DM = DatasetManager()

        # Display function names to see what datasets are available to load
        # print(self.DM.get_all_function_names())

        # Specify names for train, val and test-datasets
        self.name_train = TRAIN_DESED_NAME
        self.name_val = VAL_DESED_NAME
        self.name_test = TEST_DESED_NAME

        # Specify datasets to load here:
        self.DM.load_DESED_strong_synth_train(self.name_train)
        self.DM.load_DESED_strong_synth_val(self.name_val)
        self.DM.load_DESED_strong_public_test(self.name_test)

        # Display loaded datasets
        # print(self.DM.display_loaded_datasets())

        # Assign datasets and dataloaders for train, val and test
        self.train_ds = self.DM.get_dataset(self.name_train)
        self.val_ds = self.DM.get_dataset(self.name_val)
        self.test_ds = self.DM.get_dataset(self.name_test)

        self.train_dl = self.DM.get_dataloader(self.name_train)
        self.val_dl = self.DM.get_dataloader(self.name_val)
        self.test_dl = self.DM.get_dataloader(self.name_test)

    # Getters for datasets and dataloaders
    def get_train_loader(self):
        return self.train_dl

    def get_val_loader(self):
        return self.val_dl

    def get_test_loader(self):
        return self.test_dl

    def get_train_ds(self):
        return self.train_ds

    def get_val_ds(self):
        return self.val_ds

    def get_test_ds(self):
        return self.test_ds

    # Setters for train, val and test
    def set_train(self, name: str):
        """
        Set training dataset and dataloader.
        Note: The new dataset must be loaded into memory for this to work.
        """
        self.name_train = name
        self.train_ds = self.DM.get_dataset(self.name_train)
        self.train_dl = self.DM.get_dataloader(self.name_train)

    def set_val(self, name: str):
        """
        Set validation dataset and dataloader.
        Note: The new dataset must be loaded into memory for this to work.
        """
        self.name_val = name
        self.val_ds = self.DM.get_dataset(self.name_val)
        self.val_dl = self.DM.get_dataloader(self.name_val)

    def set_test(self, name: str):
        """
        Set test dataset and dataloader.
        Note: The new dataset must be loaded into memory for this to work.
        """
        self.name_test = name
        self.test_ds = self.DM.get_dataset(self.name_test)
        self.test_dl = self.DM.get_dataloader(self.name_test)

    # User help-functions
    def display_loaded_datasets(self) -> str:
        """
        Display all datasets that are loaded into memory.
        """
        return self.DM.display_loaded_datasets()

    def get_loadable_datasets(self) -> List[str]:
        """
        Useful to see what datasets are available to load.
        """
        return self.DM.get_loadable_datasets()
