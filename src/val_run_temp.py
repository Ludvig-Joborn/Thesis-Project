#################################################################
# File for getting scores from validation (for multiple models) #
#################################################################

import torch
import matplotlib.pyplot as plt
from typing import Callable, List, Dict
import numpy as np
from pathlib import Path
from enum import Enum
from tqdm import tqdm
import pandas as pd
import warnings
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import collections

# User-defined imports
import config
from models_dict import *
from datasets.dataset_handler import DatasetManager
from models.model_utils import *
from eval import *


def get_val_model_value_epoch(
    picked_model: Model,
    basepath: Path,
    epoch: int,
    act_threshold: float = config.ACT_THRESHOLD,
    DS_val: Dataset = None,
    psds_params: Dict = config.PSDS_PARAMS,
    operating_points: np.ndarray = config.OPERATING_POINTS,
) -> Dict[str, Dict[str, List]]:
    """
    Creates and returns a dictionary with metrics for a given model.
    """

    # Get the threshold value in 'operating_points' that is closest to 'act_threshold'
    val_model_basepath = basepath / str(picked_model) / "validation" / str(DS_val)
    used_threshold = min(
        operating_points, key=lambda input_list: abs(input_list - act_threshold)
    )

    # Calculate PSDS and Fscores for each epoch of the input model
    values = {}
    psd_scores = []
    fscores_epochs = []

    state = load_model(val_model_basepath / f"Ve{epoch}.pt")
    op_table = state["op_table"]
    psds, fscores = psd_score(
        op_table,
        DS_val.get_annotations(),
        psds_params,
        operating_points,
    )
    try:
        fscores_epochs.append(fscores.Fscores.loc[used_threshold])
    except:
        fscores_epochs.append(0)

    psd_scores.append(psds.value)

    # Add values
    values["tr_epoch_accs"] = state["tr_epoch_accs"]
    values["tr_epoch_losses"] = state["tr_epoch_losses"]
    values["val_acc_table"] = state["val_acc_table"][used_threshold]
    values["val_epoch_losses"] = state["val_epoch_losses"]
    values["Fscores"] = fscores_epochs[0]
    values["psds"] = psd_scores[0]

    return values


# MODEL: BEST_EPOCH
DICT_01 = {
    Model("baseline", baseline): 12,
    Model("lp_pool", lp_pool): 10,
    Model("baseline_ks33_l44", baseline_ks33_l44): 8,
    Model("gru_2", gru_2): 13,
    Model("gru_2_drop01", gru_2_drop01): 16,
    Model("lstm_1", lstm_1): 8,
    Model("b_ks33_l22_gru_2", b_ks33_l22_gru_2): 10,
    Model("swish", swish): 4,
}

DICT_07 = {
    Model("baseline", baseline): 11,
    Model("lp_pool", lp_pool): 9,
    Model("baseline_ks33_l44", baseline_ks33_l44): 7,
    Model("gru_2", gru_2): 13,
    Model("gru_2_drop01", gru_2_drop01): 8,
    Model("lstm_1", lstm_1): 8,
    Model("b_ks33_l22_gru_2", b_ks33_l22_gru_2): 16,
    Model("swish", swish): 18,
}

DS_train_name = config.DESED_SYNTH_TRAIN_ARGS["name"]
DS_train_basepath = Path(config.SAVED_MODELS_DIR) / DS_train_name

DM = DatasetManager()
params_DS_val = config.DESED_SYNTH_VAL_ARGS
DS_val_loader = DM.load_dataset(**params_DS_val)
DS_val = DM.get_dataset(params_DS_val["name"])


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

OP_THRESHOLD = 0.5

# NOTE: CHOOSE! #######
# PSDS_CONFIG = 0.7  ####
#######################

# Ignore warnings
warnings.simplefilter("ignore", UserWarning)

if False:
    for PSDS_CONFIG in [0.1, 0.7]:
        if PSDS_CONFIG == 0.1:
            PSDS_PARAMS = PSDS_PARAMS_01
            DICT_ = DICT_01
        elif PSDS_CONFIG == 0.7:
            PSDS_PARAMS = PSDS_PARAMS_07
            DICT_ = DICT_07

        WHAT_TO_PRINT = [
            config.PLOT_MODES.PSDS,
            config.PLOT_MODES.FSCORE,
            # config.PLOT_MODES.TR_ACC,
            # config.PLOT_MODES.TR_LOSS,
            # config.PLOT_MODES.VAL_ACC,
            # config.PLOT_MODES.VAL_LOSS,
        ]

        # cols = ["Model", "PSD-Score-1", "F1-Score-1", "PSD-Score-2", "F1-Score-2", "Total"]
        # pandas_res = pd.DataFrame(columns=cols)

        T1 = 18
        T2 = 10

        print(f"RESULTS -- PSDS_CONFIG: {PSDS_CONFIG}")
        for wtp in WHAT_TO_PRINT:
            vals_to_print = {}
            for picked_model, epoch in DICT_.items():
                RET = get_val_model_value_epoch(
                    picked_model,
                    DS_train_basepath,
                    epoch,
                    OP_THRESHOLD,
                    DS_val,
                    PSDS_PARAMS,
                    config.OPERATING_POINTS,
                )

                val = RET[wtp.value[0]]
                name_of_metric = wtp.value[1]
                s_ = " " * (T1 - len(str(picked_model)))
                s2_ = " " * (T2 - len(name_of_metric))
                vals_to_print[
                    val
                ] = f"model: {str(picked_model)}{s_}| {name_of_metric}:{s2_}{val}"

            # sort 'vals_to_print' on val
            od = collections.OrderedDict(sorted(vals_to_print.items(), reverse=True))
            [print(v) for _, v in od.items()]
            print()
        print("#######")
else:
    _list = []

    # NOTE: len 01 == 07 (and order the same)
    for index, (model, epoch_01) in enumerate(DICT_01.items()):
        epoch_07 = list(DICT_07.values())[index]

        def _(picked_model, epoch, PSDS_PARAMS):
            # for picked_model, epoch in DICT_.items():
            RET = get_val_model_value_epoch(
                picked_model,
                DS_train_basepath,
                epoch,
                OP_THRESHOLD,
                DS_val,
                PSDS_PARAMS,
                config.OPERATING_POINTS,
            )
            val_psds = RET[config.PLOT_MODES.PSDS.value[0]]
            val_fscore = RET[config.PLOT_MODES.FSCORE.value[0]]
            return val_psds, val_fscore

        val_psds_01, val_fscore_01 = _(model, epoch_01, PSDS_PARAMS_01)
        val_psds_07, val_fscore_07 = _(model, epoch_07, PSDS_PARAMS_07)

        _row = [
            str(model),
            val_psds_01,
            val_fscore_01,
            val_psds_07,
            val_fscore_07,
            val_psds_01 + val_psds_07,
        ]
        _list.append(_row)

    cols = [
        "Model",
        "PSD-Score-1",
        "F1-Score-1",
        "PSD-Score-2",
        "F1-Score-2",
        "Total",
    ]
    pandas_res = pd.DataFrame(_list, columns=cols).sort_values(
        axis=0, by=["Total"], ascending=False
    )
    print(str(pandas_res))
    print()

    # To latex
    res = pandas_res.to_latex(buf=None, index=False)
    print(res)
