import torch
import pandas as pd
from pathlib import Path
import math
from typing import List

# User defined imports
import config


def get_wav_filenames(path: Path) -> List[str]:
    """
    Read all wav files in path and return a list of filenames.
    """
    files = []
    for file in path.iterdir():
        if file.suffix == ".wav" and file.is_file():
            files.append(file.name)
    return files


def get_rows_from_annotations(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Returns all rows containing filename in annotations.
    Also adds a new column with Speech or non-Speech (as 1 or 0).

    The input-format for the dataframe annotations is:
    filename    onset   offset  event_label
    """
    matched = df.loc[df["filename"].isin([filename])]
    if matched.empty:
        raise ValueError(f"No matching filename found in annotations: {filename}")

    matched = matched.assign(Speech=0)
    matched.loc[matched["event_label"] == "Speech", "Speech"] = 1
    return matched


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    # If multiple channels are present, average them
    return (
        torch.unsqueeze(torch.mean(waveform, 0), 0)
        if waveform.shape[0] > 1
        else waveform
    )


def split_waveform(
    waveform: torch.Tensor, sr: int, len_seconds: int
) -> List[torch.Tensor]:
    """
    Split the waveform into chunks of len_seconds
    and apply padding to the last element.
    """
    list_ = []
    waveform_len = waveform.shape[1]
    target_len = sr * len_seconds
    i = 0
    while waveform_len > target_len:
        wf = waveform[:, i : (i + target_len)]
        list_.append(wf)
        i += target_len
        waveform_len = waveform_len - target_len

    if waveform_len <= target_len and waveform_len > 0:
        # Apply padding to last split.
        wf = post_pad_audio(waveform[:, i:], sr, len_seconds)
        list_.append(wf)
    return list_


def post_pad_audio(waveform: torch.Tensor, sr: int, len_seconds: int) -> torch.Tensor:
    """
    Pad the waveform to be of length 'len_seconds' with the last part being zeros.
    """
    target_len = sr * len_seconds
    waveform_len = waveform.shape[1]

    if waveform_len < target_len:
        num_missing_samples = target_len - waveform_len
        last_dim_padding = (0, num_missing_samples)
        waveform = torch.nn.functional.pad(waveform[:, 0:target_len], last_dim_padding)
    return waveform


def trim_audio(waveform: torch.Tensor, sr: int, len_seconds: int) -> torch.Tensor:
    """
    Trims the input waveform to be of length 'len_seconds'. Trims half of the excess
    datapoints from the front, and half of excess from the back.
    """
    target_len = sr * len_seconds
    waveform_len = waveform.shape[1]

    if waveform_len > target_len:
        half_onset = (waveform_len - target_len) // 2
        waveform = waveform[:, half_onset : target_len + half_onset]
    return waveform


def label_frames(labels: pd.DataFrame) -> torch.Tensor:
    """
    Converts labels in seconds to frames and returns an array
    where '1' indicates the presence of speech.
    """
    if config.NR_CLASSES == 1:
        # Create a tensor of zeros
        labeled_frames = torch.zeros(config.NR_CLASSES, config.N_MELSPEC_FRAMES)

        # Get rows from annotations with speech
        speech = labels.loc[labels["Speech"] == 1]

        if speech.empty:
            return labeled_frames

        # Map seconds to frame number. Round onset down and offset up.
        for _, row in speech.iterrows():
            onset = math.floor(
                config.N_MELSPEC_FRAMES * row["onset"] / config.CLIP_LEN_SECONDS
            )
            offset = math.ceil(
                config.N_MELSPEC_FRAMES * row["offset"] / config.CLIP_LEN_SECONDS
            )
            labeled_frames[:, onset : (offset + 1)] = 1

        return labeled_frames
    else:
        raise NotImplementedError(
            "Labeling of frames is not implemented for more than one class."
        )
