import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import pandas as pd
from typing import Tuple
from pathlib import Path

# User defined imports
import config
from utils import get_detections, join_predictions, preds_to_tsv
from models.model_utils import load_model, nr_parameters
from logger import CustomLogger as Logger
from datasets.dataset_handler import DatasetManager

# Model to load
from models.improved_baseline import NeuralNetwork as NN

# Use cuda if available, exit otherwise
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    exit("Please use a GPU to train this model.")


def pred_batches(
    data_loader: DataLoader, len_data: int, model
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runs batches for the dataloader sent in without updating any model weights.
    Used for calculating validation and test accuracy and loss.
    """

    output_table = torch.empty(
        (len_data, config.N_MELSPEC_FRAMES),
        device=device,
    )
    # Add indices to retrieve filename later
    indices = torch.empty((len_data), device=device)
    # variable for number of added elements
    n_added_elems = torch.tensor(0, device=device)

    # Set model state to evaluation
    model.eval()

    ##### Evaluate on validation dataset #####
    with torch.no_grad():
        for i, sample in enumerate(
            tqdm(iterable=data_loader, desc="Prediction progress")
        ):
            waveform, sample_rate, file_id, seg_id, idx = sample

            # Send parameters to device
            waveform = waveform.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)

            # forward + loss
            outputs = model.forward(waveform, sample_rate)

            # Add index to tensor (used for getting filenames later)
            len_ = idx.shape[0]
            indices[n_added_elems : (n_added_elems + len_)] = idx

            # Reduce dimensions
            outputs = torch.squeeze(outputs, dim=1)

            # Add outputs to output_table
            len_ = outputs.shape[0]
            output_table[n_added_elems : (n_added_elems + len_), :] = outputs

            # Track number of added elements so far
            n_added_elems += len_

    return output_table, indices


def predict(
    data_loader: DataLoader,
    dataset,
    len_te: int,
    model,
    log: Logger,
    path_to_audio: Path,
):
    """
    Test loop. Calculates test loss and accuracy with no weight updates.
    """

    ##### Run test batches #####
    output_table, file_ids = pred_batches(data_loader, len_te, model)

    detections = get_detections(output_table, file_ids, config.ACT_THRESHOLD)
    detections = join_predictions(detections, dataset)

    # Swap first and last columns and put in dataframe
    cols_from = ["event_label", "onset", "offset", "filename"]
    cols_to = ["filename", "onset", "offset", "event_label"]
    det = pd.DataFrame(detections, columns=cols_from)[cols_to]

    # Append line that has info about min detection interval
    line = pd.DataFrame(
        {
            "filename": ["min_det_interval"],
            "onset": [""],
            "offset": [""],
            "event_label": [config.MIN_DETECTION_INTERVAL_SEC],
        },
        index=[0],
    )
    det = pd.concat([det, line], ignore_index=True)

    if config.SAVE_PREDS:
        preds_to_tsv(det, config.PREDS_DIR, path_to_audio)

    if config.LOG_PREDS:
        log.info("Predictions", add_header=True)
        log.info("\n" + str(det))


if __name__ == "__main__":
    ### MISC ###
    # Create new log (only a console logger - does not create file)
    log = Logger(config.LOGGER_TEST, Path(""), logging.DEBUG, logging.NOTSET)

    ### Load Datasets ###

    # Load datasets via DatasetWrapper
    DM = DatasetManager()
    custom_loader = DM.load_dataset(**config.CUSTOM_ARGS)
    custom_ds = DM.get_dataset(config.CUSTOM_ARGS["name"])
    len_custom = len(custom_ds)

    # Prerequisite: All sample rates within a dataset must be equal (or resampled
    # at dataset level) but may differ between datasets.
    sample_rates = set()
    sample_rates.add(custom_ds.get_sample_rate())

    ### Declare Model ###

    # Network
    model = NN(sample_rates, config.SAMPLE_RATE).to(device, non_blocking=True)

    # Load model from disk
    state = load_model(config.LOAD_MODEL_PATH)
    model.load_state_dict(state["state_dict"])
    log.info(f"Loaded model from {config.LOAD_MODEL_PATH}")

    # Number of trained parameters in network
    log.info(f"Number of trained parameters: {nr_parameters(model)}")

    ### Predictions ###
    predict(
        custom_loader,
        custom_ds,
        len_custom,
        model,
        log,
        config.CUSTOM_ARGS["path_audio"],
    )
