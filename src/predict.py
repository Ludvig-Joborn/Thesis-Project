import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from typing import Tuple
from pathlib import Path

# User defined imports
import config
from utils import get_detections, join_predictions, preds_to_tsv


def pred_batches(
    device: str, data_loader: DataLoader, len_data: int, model: Module
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
            tqdm(
                iterable=data_loader,
                desc="Prediction progress",
                leave=False,
                colour=config.TQDM_BATCHES,
            )
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
    device: str,
    data_loader: DataLoader,
    dataset: Dataset,
    len_dataset: int,
    model: Module,
    save_to_tsv: bool,
    path_preds: Path,
) -> pd.DataFrame:
    """
    Calculates prediction intervals and returns a
    pandas DataFrame with the following columns::

    ```
    ["filename", "onset", "offset", "event_label"].
    ```
    """

    ##### Run test batches #####
    output_table, file_ids = pred_batches(device, data_loader, len_dataset, model)

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

    if save_to_tsv:
        preds_to_tsv(det, path_preds)

    return det
