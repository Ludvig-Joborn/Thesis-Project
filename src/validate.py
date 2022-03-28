import torch
from torch.nn import Module
from torch.nn.modules import loss
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from tqdm import tqdm

# User defined imports
import config
from utils import get_detection_table, get_accuracy_table, create_basepath
from models.model_utils import load_model, save_model


### Evaluation Metrics ###


def get_tr_val_acc_loss(
    val_model_save: Dict, act_threshold: float
) -> Tuple[List[float], List[float], List[float], List[float], float]:
    """
    From the loaded 'val_model_save' dictionary, get the training and validtion
    accuracies and losses over all epochs (number of epochs is included in the dictionary).
    """
    op_table = val_model_save["op_table"]
    tr_epoch_losses = val_model_save["tr_epoch_losses"]
    tr_epoch_accs = val_model_save["tr_epoch_accs"]
    val_epoch_losses = val_model_save["val_epoch_losses"]
    val_acc_table = val_model_save["val_acc_table"]

    # Finds closest act. threshold key from operating points
    used_threshold = min(
        list(op_table.keys()), key=lambda input_list: abs(input_list - act_threshold)
    )
    val_epoch_accs = val_acc_table[used_threshold]
    return (
        tr_epoch_accs,
        tr_epoch_losses,
        val_epoch_accs,
        val_epoch_losses,
        used_threshold,
    )


### Dataset Evaluation ###


def te_val_batches(
    device: str,
    data_loader: DataLoader,
    len_data: int,
    model: Module,
    criterion: loss._Loss,
    testing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Runs batches for the dataloader sent in without updating any model weights.
    Used for calculating validation or test PSD-Score.
    """
    output_table = torch.empty(
        (len_data, config.N_MELSPEC_FRAMES),
        device=device,
    )  # Size: (692, 157) for desed public eval and 16kHz sampling
    label_table = torch.empty(
        (len_data, config.N_MELSPEC_FRAMES),
        device=device,
    )
    file_ids = torch.empty((len_data), device=device)
    # variable for number of added elements
    n_added_elems = torch.tensor(0, device=device)

    # Track loss
    total_loss = 0

    # Set model state to evaluation
    model.eval()

    ##### Evaluate on validation dataset #####
    with torch.no_grad():
        desc = "Validation Batch progress" if not testing else "Test Batch progress"
        for sample in tqdm(
            iterable=data_loader,
            desc=desc,
            position=1 if testing else 2,
            leave=False,
            colour=config.TQDM_BATCHES,
        ):
            waveform, sample_rate, labels, file_id = sample

            # Send parameters to device
            waveform = waveform.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # forward + loss
            outputs = model.forward(waveform, sample_rate)
            loss = criterion(outputs, labels)

            # Add file id to tensor (used for getting filenames later)
            len_ = file_id.shape[0]
            file_ids[n_added_elems : (n_added_elems + len_)] = file_id

            # Reduce dimensions
            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels)

            # Add outputs to output_table
            len_ = outputs.shape[0]
            output_table[n_added_elems : (n_added_elems + len_), :] = outputs

            # Add labels to label_table
            label_table[n_added_elems : (n_added_elems + len_), :] = labels

            # Track number of added elements so far
            n_added_elems += len_

            # Track loss statistics
            total_loss += loss

    return output_table, label_table, file_ids, total_loss.item()


def validate_model(
    device: str,
    start_epoch: int,
    end_epoch: int,  # Inclusive
    model_basepath: Path,
    model: Module,
    criterion: loss._Loss,
    DS_val_loader: DataLoader,
    DS_val: Dataset,
    len_val: int,
    operating_points: np.ndarray = config.OPERATING_POINTS,
):
    """
    Calculates and saves evaluation metrics for a given model and dataset.

    Uses validation data to save values in 'op_table' to later
    calculate acc and/or PSDS.

    'op_table' has keys representing different activation thresholds (betw. 0 and 1).
    These keys are called operating points and is a key concept in the
    evaluation metric PSDS. The 'op_table' entry is saved in 'model_save'.
    """
    # Parameters to save from validation
    val_losses = []
    val_acc_table = {op: [] for op in operating_points}

    for epoch in tqdm(
        iterable=range(start_epoch, end_epoch + 1),
        desc="Epoch",
        position=1,
        leave=False,
        colour=config.TQDM_EPOCHS,
    ):
        # Load Model
        model_path = model_basepath / f"e{epoch}.pt"
        state = load_model(model_path)

        # Load states
        model.load_state_dict(state["state_dict"])

        ##### Model validation #####
        output_table, label_table, file_ids, val_loss = te_val_batches(
            device, DS_val_loader, len_val, model, criterion, testing=False
        )

        val_losses.append(val_loss)

        operating_points_table = get_detection_table(
            output_table, file_ids, operating_points, DS_val, testing=False
        )

        # Append accuracies this epoch to 'val_acc_table'
        epoch_val_acc_table = get_accuracy_table(
            output_table, label_table, operating_points
        )
        for op, acc in epoch_val_acc_table.items():
            val_acc_table[op].append(acc)

        # Save model
        val_basepath = create_basepath(
            model_basepath.parent / "validation" / str(DS_val)
        )
        save_validation = {
            "val_basepath": val_basepath,
            "epoch": epoch,
            "op_table": operating_points_table,
            "tr_epoch_losses": state["tr_epoch_losses"][0:epoch],
            "tr_epoch_accs": state["tr_epoch_accs"][0:epoch],
            "val_epoch_losses": val_losses,
            "val_acc_table": val_acc_table,
        }
        # Save parameters from validation
        save_model(
            save_validation,
            val_basepath / f"Ve{epoch}.pt",
        )
