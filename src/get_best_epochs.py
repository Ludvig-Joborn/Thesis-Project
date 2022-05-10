#########################################
# File for getting best epochs for test #
#########################################

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time

# User-defined imports
import config
from models_dict import *
from datasets.dataset_handler import DatasetManager
from models.model_utils import load_model
from utils import psd_score


# Ignore warnings
warnings.simplefilter("ignore", UserWarning)

#
DS_train_name = config.DESED_SYNTH_TRAIN_ARGS["name"]
DS_train_basestring = str(config.SAVED_MODELS_DIR) + "/" + str(DS_train_name)

DM = DatasetManager()
params_DS_val = config.DESED_SYNTH_VAL_ARGS
DS_val_loader = DM.load_dataset(**params_DS_val)
DS_val_name = params_DS_val["name"]
DS_val = DM.get_dataset(DS_val_name)


PSDS_PARAMS_01 = {
    "duration_unit": "hour",
    "dtc_threshold": 0.1,
    "gtc_threshold": 0.1,
    "cttc_threshold": 0.1,
    "alpha_ct": 0.0,
    "alpha_st": 0.0,
    "max_efpr": 100,
}

PSDS_PARAMS_07 = {
    "duration_unit": "hour",
    "dtc_threshold": 0.7,
    "gtc_threshold": 0.7,
    "cttc_threshold": 0.1,
    "alpha_ct": 0.0,
    "alpha_st": 0.0,
    "max_efpr": 100,
}


def _get_PSDS(state, DS_val, psds_params):
    operating_points = config.OPERATING_POINTS
    op_table = state["op_table"]
    psds, _ = psd_score(
        op_table,
        DS_val.get_annotations(),
        psds_params,
        operating_points,
    )
    return psds.value


def get_PSDS_best_epoch(val_model_basepath, PSDS_PARAMS=PSDS_PARAMS_01):
    d_ = f"{PSDS_PARAMS['dtc_threshold']}_model epoch"
    best_epoch = 0
    best_psds = 0
    for epoch in tqdm(
        range(MIN_EPOCHS + 1, EPOCHS + 1), desc=d_, position=2, leave=False
    ):
        state = load_model(val_model_basepath / f"Ve{epoch}.pt")
        psds = _get_PSDS(state, DS_val, PSDS_PARAMS)
        if best_psds < psds:
            best_epoch = epoch
            best_psds = psds
    return best_epoch


MIN_EPOCHS = 2  # Save best epoch from 3 epochs and forward
EPOCHS = 20

DO_SNR = False

if DO_SNR:  # SNRs
    # model name: validation model basepaths
    SNRS = [20, 15, 10, 5, 0, -5, -10]
    SNRS = [30]
    SNRS = [-15, -20, -30]
    MODLES = ["baseline", "improved_baseline"]  # NOTE: must have these names!
    val_end_string = "validation/DESED_Synthetic_Validation"
    TO_EVAL = {
        f"{model}_SNR{snr}": Path(
            f"{DS_train_basestring}_SNR{snr}/{model}/{val_end_string}"
        )
        for snr in SNRS
        for model in MODLES
    }
    if False:  # Add baseline and imp.bs. without SNR
        for m in MODLES:
            TO_EVAL[m] = Path(f"{DS_train_basestring}/{m}/{val_end_string}")

if not DO_SNR:  # SRs
    TO_EVAL = {}
    SRs = [8000, 16000, 22050, 44100]
    for SR in SRs:
        p_bs = Path(
            f"E:/saved_models/seed_12345/SR{SR}_lr002_M08_G09/{DS_train_name}/baseline"
        )
        p_imp_bs = Path(
            f"E:/saved_models/seed_12345/SR{SR}_lr002_M08_G09/{DS_train_name}/improved_baseline"
        )
        val_model_basepath_bs = p_bs / "validation" / str(DS_val)
        val_model_basepath_imp_bs = p_imp_bs / "validation" / str(DS_val)

        TO_EVAL[f"baseline_{SR}"] = val_model_basepath_bs
        TO_EVAL[f"improved_baseline_{SR}"] = val_model_basepath_imp_bs


for psds_pars in tqdm([0.1, 0.7], desc="PSDS pars", position=0, leave=False):
    td_ = "".join(str(psds_pars).split("."))
    tqdm.write(f"DICT_{td_} = " + "{")
    psds_params = PSDS_PARAMS_01 if psds_pars == 0.1 else PSDS_PARAMS_07
    for model_name, path_ in tqdm(
        TO_EVAL.items(), desc="model", position=1, leave=False
    ):
        epoch = get_PSDS_best_epoch(path_, psds_params)
        if DO_SNR:
            tqdm.write(
                f'\tModel("{model_name}", {model_name.split("_SNR")[0]}): {epoch},'
            )
        else:
            tqdm.write(f'\tModel("{model_name}", {model_name.split("_")[0]}): {epoch},')
    tqdm.write("}")
