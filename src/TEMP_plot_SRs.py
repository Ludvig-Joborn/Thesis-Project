import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
from torch.utils.data import Dataset
from typing import Callable, List, Dict
import numpy as np
from pathlib import Path
from tqdm import tqdm

# User-defined imports
import config
from models.model_utils import load_model
from utils import get_datetime, psd_score
from models_dict import Model
from logger import CustomLogger as Logger

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
import math

# User-defined imports
import config
from models_dict import *
from datasets.dataset_handler import DatasetManager
from models.model_utils import *
from eval import *
from utils import timer
import time
from utils import create_basepath

# Ignore warnings
warnings.simplefilter("ignore", UserWarning)


def get_val_model_value_epoch(
    state,
    act_threshold: float = config.ACT_THRESHOLD,
    DS_val: Dataset = None,
    psds_params: Dict = config.PSDS_PARAMS,
):
    operating_points = config.OPERATING_POINTS
    # Get the threshold value in 'operating_points' that is closest to 'act_threshold'
    used_threshold = min(
        operating_points, key=lambda input_list: abs(input_list - act_threshold)
    )

    op_table = state["op_table"]
    psds, fscores = psd_score(
        op_table,
        DS_val.get_annotations(),
        psds_params,
        operating_points,
    )

    values = {}
    try:
        values["Fscores"] = fscores.Fscores.loc[used_threshold]
    except:
        values["Fscores"] = 0

    try:
        values["psds"] = psds.value
    except:
        values["psds"] = 0

    # Add values
    values["tr_epoch_accs"] = state["tr_epoch_accs"][-1]
    values["tr_epoch_losses"] = state["tr_epoch_losses"][-1]
    values["val_acc_table"] = state["val_acc_table"][used_threshold][-1]
    values["val_epoch_losses"] = state["val_epoch_losses"][-1]

    return values


DS_train_name = config.DESED_SYNTH_TRAIN_ARGS["name"]
DS_train_basepath = Path(config.SAVED_MODELS_DIR) / DS_train_name

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

OP_THRESHOLD = 0.5
EPOCHS = 20
SAVE_PLOTS_TO_PATH = Path("C:/Users/Forensic/Downloads/Results/SRs")


def _G_(val_model_basepath, PSDS_PARAMS=PSDS_PARAMS_01):
    d_ = f"{PSDS_PARAMS['dtc_threshold']}_model epoch"
    values_all_epochs = []
    for epoch in tqdm(range(1, EPOCHS + 1), desc=d_, position=1, leave=False):
        state = load_model(val_model_basepath / f"Ve{epoch}.pt")
        vals = get_val_model_value_epoch(state, OP_THRESHOLD, DS_val, PSDS_PARAMS)
        values_all_epochs.append(vals)
    return values_all_epochs


WHAT_TO_PLOT = [
    config.PLOT_MODES.TR_ACC,
    config.PLOT_MODES.TR_LOSS,
    config.PLOT_MODES.VAL_ACC,
    config.PLOT_MODES.VAL_LOSS,
    config.PLOT_MODES.PSDS,
    config.PLOT_MODES.FSCORE,
]

s_time = time.time()
plot_vals_01_bs = {}
plot_vals_07_bs = {}
plot_vals_01_imp_bs = {}
plot_vals_07_imp_bs = {}

SRs = [8000, 16000, 22050, 44100]
for SR in tqdm(SRs, desc="SR", position=0):
    p_bs = Path(
        f"E:/saved_models/seed_12345/SR{SR}_lr002_M08_G09/{DS_train_name}/baseline"
    )
    p_imp_bs = Path(
        f"E:/saved_models/seed_12345/SR{SR}_lr002_M08_G09/{DS_train_name}/improved_baseline"
    )
    p_imp_bs_ks33_l22_gru_2 = Path(
        f"E:/saved_models/seed_12345/SR{SR}_lr002_M08_G09/{DS_train_name}/b_ks33_l22_gru_2"
    )

    # bs
    val_model_basepath_bs = p_bs / "validation" / str(DS_val)
    values_all_epochs_01_bs = _G_(val_model_basepath_bs, PSDS_PARAMS_01)
    values_all_epochs_07_bs = _G_(val_model_basepath_bs, PSDS_PARAMS_07)

    plot_vals_01_bs[f"baseline_{SR}"] = values_all_epochs_01_bs
    plot_vals_07_bs[f"baseline_{SR}"] = values_all_epochs_07_bs

    if SR == 16000:
        # imp. bs (b_ks33_l22_gru_2)
        val_model_basepath_imp_bs = p_imp_bs_ks33_l22_gru_2 / "validation" / str(DS_val)
        values_all_epochs_01_imp_bs = _G_(val_model_basepath_imp_bs, PSDS_PARAMS_01)
        values_all_epochs_07_imp_bs = _G_(val_model_basepath_imp_bs, PSDS_PARAMS_07)
    else:
        # imp. bs
        val_model_basepath_imp_bs = p_imp_bs / "validation" / str(DS_val)
        values_all_epochs_01_imp_bs = _G_(val_model_basepath_imp_bs, PSDS_PARAMS_01)
        values_all_epochs_07_imp_bs = _G_(val_model_basepath_imp_bs, PSDS_PARAMS_07)

    plot_vals_01_imp_bs[f"improved_baseline_{SR}"] = values_all_epochs_01_imp_bs
    plot_vals_07_imp_bs[f"improved_baseline_{SR}"] = values_all_epochs_07_imp_bs

print(f"Took: {timer(s_time, time.time())}")


def di_c_T(list_of_dicts):
    dict_ = list_of_dicts[0]
    master_dict = {key: [] for key in dict_.keys()}
    for dict_ in list_of_dicts:
        for key, val in dict_.items():
            master_dict[key].append(val)
    return master_dict


def plot_models(what_to_plot: List[config.PLOT_MODES], plot_vals, title):
    # Parameters for plotting
    plot_params = {
        "linestyle": "-",
        "marker": "o",
        "markersize": 4,
    }
    plt.style.use("ggplot")
    # plt.figure(title, figsize=(18, 10))
    plt.figure(title, figsize=(9, 12))

    # Determine plot-grid
    if len(what_to_plot) == 1:
        plot_grid = (1, 1)
    else:
        plot_grid = (2, -(-len(what_to_plot) // 2))

    ### Plots ###
    x = range(1, EPOCHS + 1)
    for i, wtp in enumerate(what_to_plot):
        for model_name, list_of_dicts in plot_vals.items():
            values = di_c_T(list_of_dicts)

            _vals = values[wtp.value[0]]
            if wtp.name == config.PLOT_MODES.TR_LOSS.name:
                _vals = [10000 * 157 * x for x in _vals]

            plt.subplot(plot_grid[0], plot_grid[1], i + 1)
            plt.plot(x, _vals, **plot_params, label=model_name)

            # Force x-axis to use integers
            plt.xticks(range(math.floor(min(x)), math.ceil(max(x)) + 1))

            plt.title(wtp.value[2])
            plt.ylabel(wtp.value[1])
            plt.xlabel("Epoch")
            plt.legend()

    # plt.show()
    path_to_plot = create_basepath(SAVE_PLOTS_TO_PATH) / f"{title}_OP{OP_THRESHOLD}.pdf"
    plt.savefig(path_to_plot)
    print(f"Plot {title} saved to: \n{path_to_plot}")


###############
### Results ###
###############

### Baseline ###
# Training
plot_models(
    [config.PLOT_MODES.TR_ACC, config.PLOT_MODES.TR_LOSS],
    plot_vals_01_bs,
    "baseline_tr_acc_loss",
)
# Validation
plot_models(
    [config.PLOT_MODES.VAL_ACC, config.PLOT_MODES.VAL_LOSS],
    plot_vals_01_bs,
    "baseline_val_acc_loss",
)
# PSDS + F1-Score
plot_models(
    [config.PLOT_MODES.PSDS, config.PLOT_MODES.FSCORE],
    plot_vals_01_bs,
    "baseline_PSDS01",
)
plot_models(
    [config.PLOT_MODES.PSDS, config.PLOT_MODES.FSCORE],
    plot_vals_07_bs,
    "baseline_PSDS07",
)


### Improved Baseline ###
# Training
plot_models(
    [config.PLOT_MODES.TR_ACC, config.PLOT_MODES.TR_LOSS],
    plot_vals_01_imp_bs,
    "improved_baseline_tr_acc_loss",
)
# Validation
plot_models(
    [config.PLOT_MODES.VAL_ACC, config.PLOT_MODES.VAL_LOSS],
    plot_vals_01_imp_bs,
    "improved_baseline_val_acc_loss",
)
# PSDS + F1-Score
plot_models(
    [config.PLOT_MODES.PSDS, config.PLOT_MODES.FSCORE],
    plot_vals_01_imp_bs,
    "improved_baseline_PSDS01",
)
plot_models(
    [config.PLOT_MODES.PSDS, config.PLOT_MODES.FSCORE],
    plot_vals_07_imp_bs,
    "improved_baseline_PSDS07",
)
