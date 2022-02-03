import torch
from torch.utils.data import DataLoader
import julius

# User defined imports
from config import *
from utils import *
from datasets.datasets_utils import *
from datasets.desed import DESED_Strong
from src.logger import CustomLogger as Logger

log = Logger("train-Logger")

# Use cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    log.error("Please use a GPU to train this model.")
    exit()

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

    # Plot mel spectrogram
    if False:
        waveform, sample_rate, labels = DESED_train.__getitem__(4)
        # resample = julius.ResampleFrac(old_sr=44100, new_sr=16000)
        # waveform = resample(waveform)
        melspec_fcn = get_mel_spectrogram(
            sample_rate, N_FFT, WIN_LENGTH, HOP_LENGTH, N_MELS
        )
        melspec = melspec_fcn(waveform)
        print("Size of melspectrogram:", melspec.size())
        print("waveform shape: ", waveform.shape)
        plot_spectrogram(melspec[0], title="MelSpectrogram", ylabel="mel freq")
        exit()

    if True:
        DESED_dataloader_train = DataLoader(
            DESED_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
        )

        # TODO: Initialize model and send it GPU.

        # for epoch in range(EPOCHS)

        for i, sample in enumerate(DESED_dataloader_train):
            waveform, sample_rate, labels = sample
            print("Batch:", i)
            print("Batch waveform:", waveform)
            print("Labels shape:", labels.shape)
            exit()
