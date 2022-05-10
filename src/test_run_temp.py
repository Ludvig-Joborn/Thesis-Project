##################################################################
# File for getting scores from testdataset (for multiple models) #
##################################################################

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import warnings
import collections
import logging
from torch.nn.modules import loss
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import time
import gc
import multiprocessing
from functools import partial
import psutil

# User-defined imports
import config
from models_dict import *
from datasets.dataset_handler import DatasetManager
from models.model_utils import load_model
from eval import te_val_batches, plot_psd_roc
from utils import (
    outrow_to_detections,
    create_basepath,
    get_datetime,
    psd_score,
    outrow_to_detections,
    frames_to_intervals,
)
from models.model_utils import activation


params_DS_train = config.DESED_SYNTH_TRAIN_ARGS

# NOTE:
params_DS_test = config.DESED_PUBLIC_EVAL_ARGS
params_DS_test = config.DESED_REAL_ARGS

DM = DatasetManager()
DS_train_name = config.DESED_SYNTH_TRAIN_ARGS["name"]
DS_train_basepath = Path(config.SAVED_MODELS_DIR) / DS_train_name

DS_test_loader = DM.load_dataset(**params_DS_test)
DS_test = DM.get_dataset(params_DS_test["name"])
sample_rates = set()
sample_rates.add(DS_test.get_sample_rate())

device = "cuda"

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
# NOTE:
DICT_01 = {
    Model("baseline_8000", baseline): 5,
    Model("improved_baseline_8000", improved_baseline): 14,
    Model("baseline_16000", baseline): 12,
    Model("improved_baseline_16000", improved_baseline): 10,
    Model("baseline_22050", baseline): 10,
    Model("improved_baseline_22050", improved_baseline): 8,
    Model("baseline_44100", baseline): 10,
    Model("improved_baseline_44100", improved_baseline): 11,
}
DICT_07 = {
    Model("baseline_8000", baseline): 16,
    Model("improved_baseline_8000", improved_baseline): 7,
    Model("baseline_16000", baseline): 11,
    Model("improved_baseline_16000", improved_baseline): 16,
    Model("baseline_22050", baseline): 9,
    Model("improved_baseline_22050", improved_baseline): 8,
    Model("baseline_44100", baseline): 6,
    Model("improved_baseline_44100", improved_baseline): 11,
}


DICT_01 = {
    Model("baseline", baseline): 12,
    Model("improved_baseline", improved_baseline): 10,
}
DICT_07 = {
    Model("baseline", baseline): 11,
    Model("improved_baseline", improved_baseline): 16,
}

# Loss function
criterion = nn.BCELoss()


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


def get_detection_table_OLD(
    output_table: torch.Tensor,
    file_ids: torch.Tensor,
    dataset,
) -> Dict[float, pd.DataFrame]:
    """
    Used when calculating PSD-Score.
    Calculates detection (speech) intervals on the CPU and returns a dictionary
    with detections for each operating point.
    """
    # Dictionary containing detections (pandas DataFrames)
    op_tables = {}

    # Send to CPU
    output_table = output_table.to(device="cpu", non_blocking=True)
    file_ids = file_ids.tolist()  # Creates a list on cpu

    # Iterate over operating points to add predictions to each operating point table
    for op in tqdm(
        config.OPERATING_POINTS,
        desc="Generating detection intervals",
        leave=False,
        position=1,
        colour=config.TQDM_BATCHES,
        total=len(config.OPERATING_POINTS),
    ):
        detections = []
        for i, out_row in enumerate(output_table):
            out_act = activation(out_row, op)
            filename = dataset.filename(
                int(file_ids[i])
            )  # Get filename from file index
            detection_intervals = frames_to_intervals(out_act, filename)
            for speech_row in detection_intervals:
                detections.append(speech_row)

        # Add detections (as pandas DataFrame) to op_tables
        cols = ["event_label", "onset", "offset", "filename"]
        op_tables[op] = pd.DataFrame(detections, columns=cols)

    return op_tables


def get_detection_table_TEMP(
    output_table: torch.Tensor,
    file_ids: torch.Tensor,
    operating_points: np.ndarray,
    dataset: Dataset,
    testing: bool = False,
) -> Dict[float, pd.DataFrame]:
    """
    Used when calculating PSD-Score.
    Calculates detection (speech) intervals on the CPU and returns a dictionary
    with detections for each operating point.
    """
    # Dictionary containing detections (pandas DataFrames)
    op_tables = {}

    # Send to CPU
    output_table = output_table.to(device="cpu")
    file_ids = file_ids.tolist()  # Creates a list on cpu

    # Partially define function
    func = partial(outrow_to_detections, output_table, file_ids, dataset)
    # Get number of available CPUs (NOTE: lower this number if crash!)
    # nb_workers = psutil.cpu_count(logical=False)
    # Instantiate a pool of workers and make an iterable imap
    nb_workers = 1
    pool = multiprocessing.Pool(nb_workers)
    it = pool.imap_unordered(func, operating_points)

    # Generate table of detections with operating_points as keys, utilizing multiprocess.
    for op_index, op_detections in enumerate(it):
        cols = ["event_label", "onset", "offset", "filename"]
        op_tables[operating_points[op_index]] = pd.DataFrame(
            op_detections, columns=cols
        )

    return op_tables


def calc_test_psds_TEMP(
    act_threshold: float,
    device: str,
    model_basepath: Path,
    DS_test_loader: DataLoader,
    DS_test: Dataset,
    len_te: int,
    model,
    criterion: loss._Loss,
    psds_params: Dict = config.PSDS_PARAMS,
    operating_points: np.ndarray = config.OPERATING_POINTS,
    plot_psds_roc: bool = False,
    save_psds_roc: bool = True,
) -> Tuple[object, pd.DataFrame]:
    """
    Calculates the test PSD-Score with the parameters specificed in 'psds_params'.
    """
    output_table, file_ids, file_ids, _ = te_val_batches(
        device, DS_test_loader, len_te, model, criterion, testing=True
    )
    ### PSDS and F1-Score ###
    operating_points_table = get_detection_table_OLD(
        # output_table, file_ids, operating_points, DS_test, testing=True
        output_table,
        file_ids,
        DS_test,
    )
    psds, fscores = psd_score(
        operating_points_table,
        DS_test.get_annotations(),
        psds_params,
        operating_points,
    )
    # Used Fscore
    used_threshold = min(
        operating_points, key=lambda input_list: abs(input_list - act_threshold)
    )
    op_fscore = fscores.loc[used_threshold]["Fscores"]

    # SAVE
    if save_psds_roc:
        plt.style.use("fast")
        plot_filepath = model_basepath / f"{get_datetime()}_PSDS_{psds.value:2f}.pdf"
        plot_psd_roc(psds, show=False, filename=plot_filepath)
        # tqdm.write(f"PSD-ROC saved to: {plot_filepath}")

    if plot_psds_roc:
        # Plot the PSD-ROC
        plt.style.use("fast")
        plot_psd_roc(psds, show=True)

    return psds.value, op_fscore


def func_(dict_=DICT_01, psds_params_=PSDS_PARAMS_01):
    for picked_model, epoch in tqdm(
        iterable=dict_.items(),
        desc="Models",
        leave=False,
        position=0,
        colour=config.TQDM_MODELS,
    ):
        tqdm.write(f"Model: {str(picked_model)}")

        model = picked_model.get_NN()(sample_rates, config.SAMPLE_RATE).to(
            device, non_blocking=True
        )
        # load Model state
        model_name = str(picked_model)

        # NOTE:
        SRs = False

        if SRs:
            model_name = str(picked_model).rsplit("_", 1)[0]
            SR = str(picked_model).rsplit("_", 1)[1]

        model_path = DS_train_basepath / model_name / "train" / f"e{epoch}.pt"
        state = load_model(model_path)
        # Load state
        model.load_state_dict(state["state_dict"])

        # TEST
        model_basepath = create_basepath(
            DS_train_basepath / model_name / "test" / str(DS_test)
        )
        if SRs:
            model_basepath = create_basepath(model_basepath / SR)

        psds_val, fscore_val = calc_test_psds_TEMP(
            OP_THRESHOLD,
            device,
            model_basepath,
            DS_test_loader,
            DS_test,
            len(DS_test),
            model,
            criterion,
            psds_params_,
            config.OPERATING_POINTS,
            config.PLOT_PSD_ROC,
            config.SAVE_PSD_ROC,
        )
        yield str(picked_model), psds_val, fscore_val


# print("### PSDS_PARAMS 0.1 ###")
it_01 = func_(dict_=DICT_01, psds_params_=PSDS_PARAMS_01)
res_01 = {
    model_name: (psds_val, fscore_val) for model_name, psds_val, fscore_val in it_01
}

# print("### PSDS_PARAMS 0.7 ###")
it_07 = func_(dict_=DICT_07, psds_params_=PSDS_PARAMS_07)
res_07 = {
    model_name: (psds_val, fscore_val) for model_name, psds_val, fscore_val in it_07
}

_list = []
for model_name, (psds_val_01, fscore_val_01) in res_01.items():
    psds_val_07 = res_07[model_name][0]
    fscore_val_07 = res_07[model_name][1]
    _list.append(
        [
            model_name,
            psds_val_01,
            fscore_val_01,
            psds_val_07,
            fscore_val_07,
            psds_val_01 + psds_val_07,
        ]
    )

_cols = [
    "Model",
    "PSD-Score-1",
    "F1-Score-1",
    "PSD-Score-2",
    "F1-Score-2",
    "Total",
]
pandas_res = pd.DataFrame(_list, columns=_cols).sort_values(
    axis=0, by=["Total"], ascending=False
)
print(str(pandas_res))
print()

# To latex
res = pandas_res.to_latex(buf=None, index=False)
print(res)
print("Dataset: ", str(DS_test))
