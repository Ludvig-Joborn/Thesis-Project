import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import pandas as pd
from typing import Tuple

# User defined imports
from datasets.datasets_utils import *


class DESED_Strong(Dataset):
    def __init__(
        self, name: str, annotations_tsv: Path, wav_dir: Path, clip_len_seconds: int
    ):
        self.name = name
        self.df_annotations = pd.read_table(annotations_tsv)
        self.wav_dir = wav_dir
        self.filenames = get_wav_filenames(self.wav_dir)
        self.clip_len_seconds = clip_len_seconds

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        filename = self.filenames[idx]
        wav_path = self.wav_dir / filename
        if not wav_path.exists():
            raise ValueError(f"File {wav_path} does not exist")
        waveform, sample_rate = torchaudio.load(wav_path)

        # Waveform to mono
        waveform = to_mono(waveform)

        # pad and trim waveform
        waveform = post_pad_audio(waveform, sample_rate, self.clip_len_seconds)
        waveform = trim_audio(waveform, sample_rate, self.clip_len_seconds)

        # Get annotations as a pandas dataframe
        labels = get_rows_from_annotations(self.df_annotations, filename)

        # Convert labels to one hot encoding of speech activity
        labeled_frames = label_frames(labels)

        return waveform, labeled_frames, idx

    def __str__(self) -> str:
        return f"{self.name} dataset"

    def filename(self, idx: int) -> str:
        return self.filenames[idx]

    def get_annotations(self) -> pd.DataFrame:
        return self.df_annotations

    def get_sample_rate(self) -> int:
        """
        Assumes that all audio files in 'wav_dir' have the same sample rate.
        """
        len_ = self.__len__()
        i = 0
        wav_path = self.wav_dir / self.filenames[i]
        while not wav_path.exists():
            i += 1
            wav_path = self.wav_dir / self.filenames[i]
            if i > len_:  # Check if i > number of files/entries in dataset
                raise ValueError(
                    f"Could not locate audio files in path: {self.wav_dir}"
                )

        _, sample_rate = torchaudio.load(wav_path)
        return sample_rate
