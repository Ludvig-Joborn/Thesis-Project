import torch
from torch.nn import Module
from torch.nn.modules import loss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from pathlib import Path
import pandas as pd

# User defined imports
import config
from utils import get_detection_table, psd_score, get_datetime
from psds_eval import plot_psd_roc
from models.model_utils import calc_nr_correct_predictions, activation
from logger import CustomLogger as Logger
from validate import te_val_batches


def calc_test_acc_loss(
    device: str,
    DS_test_loader: DataLoader,
    len_te: int,
    model: Module,
    criterion: loss._Loss,
    act_threshold: float,
    log: Logger,
) -> Tuple[float, float]:
    """
    Calculates the test accuracy and loss with the given activation threshold.
    """
    test_loss, test_acc = te_batches_acc_loss(
        device,
        DS_test_loader,
        len_te,
        model,
        criterion,
        act_threshold,
    )
    # Log test loss and accuracy to file-log only
    log.info(f"Test Loss: {test_loss}", display_console=False)
    log.info(f"Test Accuracy: {test_acc}", display_console=False)
    # Write using tqdm (keeps terminal tidy)
    tqdm.write(f"Test Accuracy: {test_acc}")

    return test_acc, test_loss


def calc_test_psds(
    device: str,
    model_basepath: Path,
    DS_test_loader: DataLoader,
    DS_test: Dataset,
    len_te: int,
    model: Module,
    criterion: loss._Loss,
    log: Logger,
    psds_params: Dict = config.PSDS_PARAMS,
    operating_points: np.ndarray = config.OPERATING_POINTS,
    plot_psds_roc: bool = config.PLOT_PSD_ROC,
    save_psds_roc: bool = config.SAVE_PSD_ROC,
) -> Tuple[object, pd.DataFrame]:
    """
    Calculates the test PSD-Score with the parameters specificed in 'psds_params'.
    """
    output_table, file_ids, file_ids, _ = te_val_batches(
        device, DS_test_loader, len_te, model, criterion, testing=True
    )
    ### PSDS and F1-Score ###
    operating_points_table = get_detection_table(
        output_table, file_ids, operating_points, DS_test, testing=True
    )
    psds, fscores = psd_score(
        operating_points_table,
        DS_test.get_annotations(),
        psds_params,
        operating_points,
    )

    # Strings to log and write to terminal
    str_pars = (
        f"Parameters: dtc={config.PSDS_PARAMS['dtc_threshold']} "
        f"| gtc={config.PSDS_PARAMS['gtc_threshold']}"
    )
    str_psds = f"PSD-Score: {psds.value:5f}"
    str_fscore = f"F1-Scores:\n{str(fscores)}"
    # Log all output to file-log only
    log.info(str_pars, display_console=False)
    log.info(str_psds, display_console=False)
    log.info(str_fscore, display_console=False)
    # Write using tqdm (keeps terminal tidy)
    tqdm.write(str_pars)
    tqdm.write(str_psds)
    tqdm.write(str_fscore)

    if save_psds_roc:
        plt.style.use("fast")
        plot_filepath = model_basepath / f"{get_datetime()}_PSDS_{psds.value:2f}.png"
        plot_psd_roc(psds, show=False, filename=plot_filepath)
        tqdm.write(f"PSD-ROC saved to: {plot_filepath}")

    if plot_psds_roc:
        # Plot the PSD-ROC
        plt.style.use("fast")
        plot_psd_roc(psds, show=True)

    return psds, fscores


### PSDS ###
# NOTE: see te_val_batches() in validate.py
############


def te_batches_acc_loss(
    device: str,
    data_loader: DataLoader,
    len_data: int,
    model: Module,
    criterion: loss._Loss,
    act_threshold: float,
) -> Tuple[float, float]:
    """
    Runs batches for the dataloader sent in without updating any model weights.
    Used for calculating test accuracy and loss.
    """
    # Track loss
    total_loss = 0
    total = 0
    # Track correct and total predictions to calculate epoch accuracy
    correct = torch.zeros(-(len_data // -config.BATCH_SIZE)).to(
        device, non_blocking=True
    )

    # Set model state to evaluation
    model.eval()

    ##### Evaluate on validation dataset #####
    with torch.no_grad():
        desc = "Test Batch progress"
        for i, sample in enumerate(
            tqdm(
                iterable=data_loader,
                desc=desc,
                leave=False,
                position=1,
                colour=config.TQDM_BATCHES,
            )
        ):
            waveform, sample_rate, labels, file_id = sample

            # Send parameters to device
            waveform = waveform.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # forward + loss
            outputs = model.forward(waveform, sample_rate)
            loss = criterion(outputs, labels)

            # Track loss statistics
            total_loss += loss
            correct[i], total_i = calc_nr_correct_predictions(
                activation(outputs, act_threshold), labels
            )
            total += total_i

    # Calculate accuracy and loss
    avg_loss = float(total_loss.item() / total)
    avg_acc = float(correct.sum().item() / total)
    return avg_acc, avg_loss
