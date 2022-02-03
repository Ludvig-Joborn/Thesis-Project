import torch
from torch.utils.data import DataLoader

# User defined imports
from config import *
from melspec import *
from datasets.datasets_utils import *
from datasets.desed import DESED_Strong

# Use cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"


# Test code to plot mel spectrogram
if True:
    # Load dataset DESED Train (Synthetic)
    DESED_train = DESED_Strong(
        "DESED Synthetic Training",
        PATH_TO_SYNTH_TRAIN_DESED_TSV,
        PATH_TO_SYNTH_TRAIN_DESED_WAVS,
        DESED_CLIP_LEN_SECONDS,
    )

    # Load dataset DESED Validation (Synthetic)
    DESED_val = DESED_Strong(
        "DESED Synthetic Validation",
        PATH_TO_SYNTH_VALIDATION_DESED_TSV,
        PATH_TO_SYNTH_VALIDATION_DESED_WAVS,
        DESED_CLIP_LEN_SECONDS,
    )

    # Load dataset DESED Test (Public Evaluation)
    DESED_test = DESED_Strong(
        "DESED Public Evaluation",
        PATH_TO_PUBLIC_EVAL_DESED_TSV,
        PATH_TO_PUBLIC_EVAL_DESED_WAVS,
        DESED_CLIP_LEN_SECONDS,
    )

    print("Name:", DESED_train)
    print("length:", len(DESED_train))
    print("Sample 0 in DESED:", DESED_train[0], "\n")

    # One sample
    waveform, sample_rate, labels = DESED_train.__getitem__(0)

    # Parameters for mel spectrogram
    n_fft = 2048  # 1024
    win_length = None
    hop_length = 1024  # 512
    n_mels = 128

    # Plot mel spectrogram
    if True:
        melspec_fcn = get_mel_spectrogram(
            sample_rate, n_fft, win_length, hop_length, n_mels
        )
        melspec = melspec_fcn(waveform)
        print("Size of melspectrogram:", melspec.size())
        plot_spectrogram(melspec[0], title="MelSpectrogram", ylabel="mel freq")

    ######################################################
    DESED_dataloader_train = DataLoader(
        DESED_train, batch_size=64, shuffle=True, pin_memory=True
    )
