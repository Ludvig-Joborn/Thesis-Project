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

### DESED ###
# Clip length of audio files
DESED_CLIP_LEN_SECONDS = CLIP_LEN_SECONDS

# Train Dataset
PATH_TO_SYNTH_TRAIN_DESED_TSV = Path(
    "C:/Users/ludvi/Desktop/Thesis-Project/data/dcase21_synth/metadata/train/synhtetic21_train/soundscapes.tsv"
)
PATH_TO_SYNTH_TRAIN_DESED_WAVS = Path(
    "C:/Users/ludvi/Desktop/Thesis-Project/data/dcase21_synth/audio/train/synthetic21_train/soundscapes"
)

# Validation Dataset
PATH_TO_SYNTH_VALIDATION_DESED_TSV = Path(
    "C:/Users/ludvi/Desktop/Thesis-Project/data/dcase21_synth/metadata/validation/synhtetic21_validation/soundscapes.tsv"
)
PATH_TO_SYNTH_VALIDATION_DESED_WAVS = Path(
    "C:/Users/ludvi/Desktop/Thesis-Project/data/dcase21_synth/audio/validation/synthetic21_validation/soundscapes"
)

# Test Dataset
PATH_TO_PUBLIC_EVAL_DESED_TSV = Path(
    # "E:/Datasets/desed_zenodo/DESEDpublic_eval/dataset/metadata/eval/public.tsv"
    "C:/Users/ludvi/Desktop/Thesis-Project/data/desed_real_eval/metadata/eval/public.tsv"
)
PATH_TO_PUBLIC_EVAL_DESED_WAVS = Path(
    # "E:/Datasets/desed_zenodo/DESEDpublic_eval/dataset/audio/eval/public"
    "C:/Users/ludvi/Desktop/Thesis-Project/data/desed_real_eval/audio/eval/public"
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
    "htk": False,
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
EPOCHS = 2
BATCH_SIZE = 8


################
### TRAINING ###
################


##################
### EVALUATION ###
##################


############
### MISC ###
############
