import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from typing import Tuple
from collections import OrderedDict
import warnings
import julius

# User defined imports
from datasets.datasets_utils import get_wav_filenames, to_mono, split_waveform
import config


class CustomUnlabeled(Dataset):
    """
    Custom dataset to load audio clips from a directory with no annotations.
    (Files are usually larger than 'CLIP_LEN_SECONDS'.)

    NOTE: Data loaders using this dataset needs to use 'shuffle=False',
    or output predictions will be incorrect.
    """

    def __init__(self, name: str, _: Path, wav_dir: Path, clip_len_seconds: int):
        self.name = name
        self.wav_dir = wav_dir
        self.filenames = get_wav_filenames(self.wav_dir)
        self.clip_len_seconds = clip_len_seconds
        self.sample_rates = set()  # Set of unique sample rates
        self.resample_table = {}

        ### Build dictionary with key=filename, value=n_segments ###
        self.file_seg_table = OrderedDict()
        for file in self.filenames:
            wav_path = self.wav_dir / file
            if not wav_path.exists():
                raise ValueError(f"File {wav_path} does not exist")
            waveform, sample_rate = torchaudio.load(wav_path)
            self.sample_rates.add(sample_rate)
            n_segments = -(waveform.shape[1] // (-sample_rate * self.clip_len_seconds))
            self.file_seg_table[file] = n_segments

        ### Initialize ResampleFrac for each sample rate in dataset ###
        for sample_rate in self.sample_rates:
            self.resample_table[sample_rate] = julius.resample.ResampleFrac(
                sample_rate, config.SAMPLE_RATE
            )

    def __len__(self) -> int:
        return sum(self.file_seg_table.values())

    def get_file_seg_ids(self, idx: int):
        """
        Calculates the file and segment id given a global index of all segments
        in sequence.
        """
        seg_id = 0
        for file_id, (filename, n_segments) in enumerate(self.file_seg_table.items()):
            if seg_id + n_segments > idx:
                return filename, file_id, idx - seg_id
            seg_id += n_segments
        raise IndexError(
            f"Index {idx} is out of bounds, max index is {self.__len__() - 1}."
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        filename, file_id, seg_id = self.get_file_seg_ids(idx)

        wav_path = self.wav_dir / filename
        if not wav_path.exists():
            raise ValueError(f"File {wav_path} does not exist")
        full_waveform, sample_rate = torchaudio.load(wav_path)

        # Waveform to mono
        full_waveform = to_mono(full_waveform)

        # Get segment of waveform with seg_id
        waveform = split_waveform(full_waveform, sample_rate, self.clip_len_seconds)
        waveform = waveform[seg_id]
        waveform = torch.squeeze(waveform)

        # Resample waveform to SAMPLE_RATE (found in config.py)
        waveform = self.resample_table[sample_rate](waveform)

        return waveform, config.SAMPLE_RATE, file_id, seg_id, idx

    def __str__(self) -> str:
        return self.name

    def filename(self, idx: int) -> str:
        filename, file_id, _ = self.get_file_seg_ids(idx)
        return filename

    def get_annotations(self) -> None:
        warnings.warn(f"No annotations exist for dataset {self.name}.")
        return None

    def get_sample_rate(self) -> int:
        return config.SAMPLE_RATE
