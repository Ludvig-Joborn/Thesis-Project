"""
This is the configuration file for this application.
"""

from pathlib import Path

################
### DATASETS ###
################

### GLOBAL ###
CLIP_LEN_SECONDS = 10
NR_CLASSES = 1
PIN_MEMORY = True

### DESED ###
# Clip length of audio files
DESED_CLIP_LEN_SECONDS = CLIP_LEN_SECONDS

# Train Dataset - Synthetic
TRAIN_DESED_NAME = "DESED Synthetic Training"
NUM_WORKERS_TRAIN_DESED = 0
PATH_TO_SYNTH_TRAIN_DESED_TSV = Path(
    "E:/Datasets/desed_zenodo/V3/dcase21_synth/metadata/train/synhtetic21_train/soundscapes.tsv"
)
PATH_TO_SYNTH_TRAIN_DESED_WAVS = Path(
    "E:/Datasets/desed_zenodo/V3/dcase21_synth/audio/train/synthetic21_train/soundscapes"
)

# Validation Dataset - Synthetic
VAL_DESED_NAME = "DESED Synthetic Validation"
NUM_WORKERS_VAL_DESED = 0
PATH_TO_SYNTH_VAL_DESED_TSV = Path(
    "E:/Datasets/desed_zenodo/V3/dcase21_synth/metadata/validation/synhtetic21_validation/soundscapes.tsv"
)
PATH_TO_SYNTH_VAL_DESED_WAVS = Path(
    "E:/Datasets/desed_zenodo/V3/dcase21_synth/audio/validation/synthetic21_validation/soundscapes"
)

# Test Dataset - Public Eval
TEST_DESED_NAME = "DESED Public Eval"
NUM_WORKERS_TEST_DESED = 2
PATH_TO_PUBLIC_TEST_DESED_TSV = Path(
    "E:/Datasets/desed_zenodo/DESEDpublic_eval/dataset/metadata/eval/public.tsv"
)
PATH_TO_PUBLIC_TEST_DESED_WAVS = Path(
    "E:/Datasets/desed_zenodo/DESEDpublic_eval/dataset/audio/eval/public"
)


######################
### MELSPECTROGRAM ###
######################

SAMPLE_RATE = 44100
N_FFT = 2048
WIN_LENGTH = None
N_MELS = 128
HOP_LENGTH = 1024
FMIN = 0
FMAX = 8000

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
    "verbose": True,
}

##############
### MODELS ###
##############
EPOCHS = 20
BATCH_SIZE = 32

ACT_THRESHOLD = 0.5

### Save & Load Model ###
CONTINUE_TRAINING = False
LOAD_MODEL_PATH = Path("E:/saved_models/2022-02-23_16-24.pt")


################
### TRAINING ###
################

### Optimizer Adam ###
LR_adam = 0.01
WD = 0.0001

### Optimizer SGD ###
LR_sgd = 0.03
MOMENTUM = 0.9

### Scheduler 1 ###
GAMMA_1 = 0.9

### Scheduler 2 ###
GAMMA_2 = 0.1
MILESTONES = [15, 40, 75]


##################
### EVALUATION ###
##################


############
### MISC ###
############
SAVED_MODELS_DIR = "E:/saved_models/"
LOG_DIR = "logs/"
LOGGER_TRAIN = "train-logger"
LOGGER_TEST = "test-logger"
