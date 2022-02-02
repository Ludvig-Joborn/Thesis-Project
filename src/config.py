"""
This is the configuration file for this application.
"""

from pathlib import Path

################
### DATASETS ###
################

### GLOBAL ###
CLIP_LEN_SECONDS = 10

### DESED ###
# Clip length of audio files
PUBLIC_EVAL_DESED_CLIP_LEN_SECONDS = CLIP_LEN_SECONDS

PATH_TO_PUBLIC_EVAL_DESED_TSV = Path(
    "E:/Datasets/desed_zenodo/DESEDpublic_eval/dataset/metadata/eval/public.tsv"
)
PATH_TO_PUBLIC_EVAL_DESED_WAVS = Path(
    "E:/Datasets/desed_zenodo/DESEDpublic_eval/dataset/audio/eval/public"
)


##############
### MODELS ###
##############


################
### TRAINING ###
################


##################
### EVALUATION ###
##################


############
### MISC ###
############
