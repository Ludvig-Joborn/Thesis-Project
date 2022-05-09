"""
This is the configuration file for this application.
"""

from pathlib import Path
import numpy as np
import math
from enum import Enum

# Dataset classes
import datasets.desed as desed
import datasets.custom_unlabeled as custom_ds

### Train model with deterministic behaviour ###
# A deterministic run might take longer
DETERMINISTIC_RUN = True
SEED = 12345


##############
### MODELS ###
##############
EPOCHS = 20
BATCH_SIZE = 32

ACT_THRESHOLD = 0.5

################
### DATASETS ###
################

### GLOBAL ###
CLIP_LEN_SECONDS = 10
NR_CLASSES = 1
PIN_MEMORY = True

### DESED ###
# Train Dataset - Synthetic
PATH_TO_SYNTH_TRAIN_DESED_TSV = Path(
    "E:/Datasets/desed_zenodo/V3/dcase21_synth/metadata/train/synhtetic21_train/soundscapes.tsv"
)
PATH_TO_SYNTH_TRAIN_DESED_WAVS = Path(
    "E:/Datasets/desed_zenodo/V3/dcase21_synth/audio/train/synthetic21_train/soundscapes"
)
DESED_SYNTH_TRAIN_ARGS = {
    "name": "DESED_Synthetic_Training",
    "DS": desed.DESED_Strong,
    "path_annotations": PATH_TO_SYNTH_TRAIN_DESED_TSV,
    "path_audio": PATH_TO_SYNTH_TRAIN_DESED_WAVS,
    "clip_len": CLIP_LEN_SECONDS,
    "NUM_WORKERS": 0,
    "PIN_MEMORY": PIN_MEMORY,
    "batch_size": BATCH_SIZE,
    "shuffle": True,
}

# Validation Dataset - Synthetic
PATH_TO_SYNTH_VAL_DESED_TSV = Path(
    "E:/Datasets/desed_zenodo/V3/dcase21_synth/metadata/validation/synhtetic21_validation/soundscapes.tsv"
)
PATH_TO_SYNTH_VAL_DESED_WAVS = Path(
    "E:/Datasets/desed_zenodo/V3/dcase21_synth/audio/validation/synthetic21_validation/soundscapes"
)
DESED_SYNTH_VAL_ARGS = {
    "name": "DESED_Synthetic_Validation",
    "DS": desed.DESED_Strong,
    "path_annotations": PATH_TO_SYNTH_VAL_DESED_TSV,
    "path_audio": PATH_TO_SYNTH_VAL_DESED_WAVS,
    "clip_len": CLIP_LEN_SECONDS,
    "NUM_WORKERS": 0,
    "PIN_MEMORY": PIN_MEMORY,
    "batch_size": BATCH_SIZE,
    "shuffle": True,
}

# Test Dataset - Public Eval
PATH_TO_PUBLIC_EVAL_DESED_TSV = Path(
    "E:/Datasets/desed_zenodo/DESEDpublic_eval/dataset/metadata/eval/public.tsv"
)
PATH_TO_PUBLIC_EVAL_DESED_WAVS = Path(
    "E:/Datasets/desed_zenodo/DESEDpublic_eval/dataset/audio/eval/public"
)
DESED_PUBLIC_EVAL_ARGS = {
    "name": "DESED_Public_Eval",
    "DS": desed.DESED_Strong,
    "path_annotations": PATH_TO_PUBLIC_EVAL_DESED_TSV,
    "path_audio": PATH_TO_PUBLIC_EVAL_DESED_WAVS,
    "clip_len": CLIP_LEN_SECONDS,
    "NUM_WORKERS": 0 if DETERMINISTIC_RUN else 2,
    "PIN_MEMORY": PIN_MEMORY,
    "batch_size": BATCH_SIZE,
    "shuffle": True,
}

# Dataset - Desed Real
PATH_TO_DESED_REAL_TSV = Path(
    "E:/Datasets/DESED_REAL_DOWNLOAD/DESED/real/metadata/validation/validation.tsv"
)
PATH_TO_DESED_REAL_WAVS = Path(
    "E:/Datasets/DESED_REAL_DOWNLOAD/DESED/real/audio/validation"
)
DESED_REAL_ARGS = {
    "name": "DESED_Real",
    "DS": desed.DESED_Strong,
    "path_annotations": PATH_TO_DESED_REAL_TSV,
    "path_audio": PATH_TO_DESED_REAL_WAVS,
    "clip_len": CLIP_LEN_SECONDS,
    "NUM_WORKERS": 0,
    "PIN_MEMORY": PIN_MEMORY,
    "batch_size": BATCH_SIZE,
    "shuffle": True,
}

# Dataset - independent wav-files.
PATH_TO_CUSTOM_WAVS = Path("E:/Datasets/custom")
CUSTOM_ARGS = {
    "name": "Custom_Dataset",
    "DS": custom_ds.CustomUnlabeled,
    "path_annotations": None,
    "path_audio": PATH_TO_CUSTOM_WAVS,
    "clip_len": CLIP_LEN_SECONDS,
    "NUM_WORKERS": 0,
    "PIN_MEMORY": PIN_MEMORY,
    "batch_size": 1,
    "shuffle": False,
}

######################
### MELSPECTROGRAM ###
######################

SAMPLE_RATE = 16000
N_FFT = 2048
WIN_LENGTH = None
N_MELS = 128
HOP_LENGTH = 1024
FMIN = 0
FMAX = 8000

N_MELSPEC_FRAMES = math.ceil((CLIP_LEN_SECONDS * SAMPLE_RATE) / HOP_LENGTH)

PARAMS_TO_MELSPEC = {
    "sr": SAMPLE_RATE,
    "n_fft": N_FFT,
    "win_length": WIN_LENGTH,
    "n_mels": N_MELS,
    "hop_length": HOP_LENGTH,
    "window": "hann",
    "center": True,
    "pad_mode": "reflect",
    "power": 2.0,
    "htk": True,  # Sets to log-scale
    "fmin": FMIN,
    "fmax": FMAX,
    "norm": 1,
    "trainable_mel": False,
    "trainable_STFT": False,
    "verbose": False,
}

################
### TRAINING ###
################

# NOTE
# Models to train/evaluate are imported and specified in models_dict.py

### Optimizer SGD ###
LR_SGD = 0.02
MOMENTUM = 0.8

### Scheduler ###
GAMMA = 0.9

##########################
### PLOTS / EVALUATION ###
##########################
class PLOT_MODES(Enum):
    """
    Options (with metadata) when plotting.
    First argument is what to plot (which is saved in dictionaries on disk),
    second argument is the y-axis label and the last is the plot title.
    """

    TR_ACC = ("tr_epoch_accs", "Accuracy", "Training Accuracy")
    TR_LOSS = ("tr_epoch_losses", "Loss", "Training Loss")
    VAL_ACC = ("val_acc_table", "Accuracy", "Validation Accuracy")
    VAL_LOSS = ("val_epoch_losses", "Loss", "Validation Loss")
    PSDS = ("psds", "PSD-Score", "Validation PSD-Score")
    FSCORE = ("Fscores", "F1-Score", "Validation F1-Score")


# Choose what to inlude in plot.
WHAT_TO_PLOT = [
    PLOT_MODES.TR_ACC,
    PLOT_MODES.TR_LOSS,
    PLOT_MODES.VAL_ACC,
    PLOT_MODES.VAL_LOSS,
    PLOT_MODES.PSDS,
    PLOT_MODES.FSCORE,
]
SAVE_PLOT = True


### PSDS ###
PLOT_PSD_ROC = False  # Will pause execution of code if multiple models are tested
SAVE_PSD_ROC = True
N_THRESHOLD = 50
OPERATING_POINTS = np.linspace(0.01, 0.99, N_THRESHOLD)

PSDS_PARAMS_01 = {
    "duration_unit": "hour",
    "dtc_threshold": 0.1,
    "gtc_threshold": 0.1,
    "cttc_threshold": 0.1,
    "alpha_ct": 0.0,
    "alpha_st": 0.0,
    "max_efpr": 100,  # Max eFPR per minute. Set to follow DCASE numbers.
}
PSDS_PARAMS_07 = {
    "duration_unit": "hour",
    "dtc_threshold": 0.7,
    "gtc_threshold": 0.7,
    "cttc_threshold": 0.1,
    "alpha_ct": 0.0,
    "alpha_st": 0.0,
    "max_efpr": 100,  # Max eFPR per minute. Set to follow DCASE numbers.
}

##################
### PREDICTION ###
##################
# Prediction intervals smaller than this are joined together.
# (applies *only* to predict.py)
MIN_DETECTION_INTERVAL_SEC = 0.8


############
### MISC ###
############
SAVED_MODELS_DIR = (
    "E:/saved_models/"
    + (f"seed_{SEED}/" if DETERMINISTIC_RUN else "nondeterministic/")
    + f"SR{SAMPLE_RATE}_"
    f"lr0{str(LR_SGD).split('.')[1]}_"
    f"M0{str(MOMENTUM).split('.')[1]}_"
    f"G0{str(GAMMA).split('.')[1]}/"
)
LOG_DIR = "logs/"

LOGGER_TRAIN = "train-logger"
LOGGER_TEST = "test-logger"

# TQDM colors
TQDM_MODELS = "magenta"
TQDM_EPOCHS = "cyan"
TQDM_BATCHES = "green"  # "white"
