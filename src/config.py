"""
This is the configuration file for this application.
"""

from pathlib import Path
import numpy as np
import math

# Dataset classes
import datasets.desed as desed
import datasets.custom_unlabeled as custom_ds

### Train model with deterministic behaviour ###
# A deterministic run might take longer
SEED = 12345
DETERMINISTIC_RUN = True


##############
### MODELS ###
##############
EPOCHS = 30
BATCH_SIZE = 32

ACT_THRESHOLD = 0.5

### Save & Load Model ###
CONTINUE_TRAINING = False
LOAD_MODEL_PATH = Path("E:/saved_models/improved_baseline.pt")


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
    "name": "DESED Synthetic Training",
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
    "name": "DESED Synthetic Validation",
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
    "name": "DESED Public Eval",
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
USE_CONCAT_VAL = True  # Concatinates synth val and real as new validation set

PATH_TO_DESED_REAL_TSV = Path(
    "E:/Datasets/DESED_REAL_DOWNLOAD/DESED/real/metadata/validation/validation.tsv"
)
PATH_TO_DESED_REAL_WAVS = Path(
    "E:/Datasets/DESED_REAL_DOWNLOAD/DESED/real/audio/validation"
)
DESED_REAL_ARGS = {
    "name": "DESED Real",
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
PATH_TO_CUSTOM_WAVS = Path("E:/Datasets/test_audio")
CUSTOM_ARGS = {
    "name": "Custom Dataset",
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

### Optimizer Adam ###
LR_adam = 0.01
WD = 0.0001

### Optimizer SGD ###
LR_sgd = 0.03
MOMENTUM = 0.8

### Scheduler 1 ###
GAMMA_1 = 0.9


##################
### EVALUATION ###
##################

PLOT_TR_VAL_ACC = True

### PSDS ###
# If PSDS and F1-Score should be calculated
CALC_PSDS = True
PLOT_PSD_ROC = True
N_THRESHOLD = 50
OPERATING_POINTS = np.linspace(0.01, 0.99, N_THRESHOLD)

PSDS_PARAMS = {
    "duration_unit": "minute",
    "dtc_threshold": 0.1,
    "gtc_threshold": 0.1,
    "cttc_threshold": 0.1,
    "alpha_ct": 0.0,
    "alpha_st": 0.0,
    "max_efpr": None,  # Will be set to max value
}


##################
### PREDICTION ###
##################

LOG_PREDS = True
SAVE_PREDS = True  # Creates tsv file with predictions

# Prediction intervals smaller than this are joined together.
# (applies *only* to predict.py)
MIN_DETECTION_INTERVAL_SEC = 0.8

############
### MISC ###
############
SAVED_MODELS_DIR = "E:/saved_models/"
LOG_DIR = "logs/"
PREDS_DIR = "E:/predictions/"

LOGGER_TRAIN = "train-logger"
LOGGER_TEST = "test-logger"
