import torch
from torchinfo import summary
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import logging
import math

# User defined imports
from config import *
from utils import *
from psds_utils import *
from models.model_utils import *
from logger import CustomLogger as Logger
from datasets.dataset_handler import DatasetWrapper
from models.baseline import NeuralNetwork as NN

# Use cuda if available, exit otherwise
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    exit("Please use a GPU to train this model.")


def te_val_batches(
    data_loader: DataLoader,
    len_data: int,
    model,
    criterion,
    testing: bool = False,
) -> Tuple[float, torch.Tensor, int, int, torch.Tensor, torch.Tensor]:
    """
    Runs batches for the dataloader sent in without updating any model weights.
    Used for calculating validation and test accuracy and loss.
    """
    if CALC_PSDS:
        ### PSDS ###
        # Mel-Spectrogram frames
        frames = math.ceil((CLIP_LEN_SECONDS * SAMPLE_RATE) / HOP_LENGTH)
        output_table = torch.empty(
            (len_data, frames),
            device=device,
        )  # Size: (692, 157) for desed public eval and 16kHz sampling
        file_ids = torch.empty((len_data), device=device)
        # variable for number of added elements
        n_added_elems = torch.tensor(0, device=device)
        ############

    # Track loss
    total_loss = 0

    # Track correct and total predictions to calculate epoch accuracy
    correct = torch.zeros(-(len_data // -BATCH_SIZE)).to(device, non_blocking=True)
    total = 0

    # Set model state to evaluation
    model.eval()

    ##### Evaluate on validation dataset #####
    with torch.no_grad():
        desc = "Test Batch progress:" if testing else "Valdiation Batch progress:"
        for i, sample in enumerate(tqdm(iterable=data_loader, desc=desc)):
            waveform, labels, file_id = sample

            # Send parameters to device
            waveform = waveform.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # forward + loss
            outputs = model.forward(waveform)
            loss = criterion(outputs, labels)

            # Track loss statistics
            total_loss += loss
            correct[i], total = update_acc(
                activation(outputs, ACT_THRESHOLD), labels, total
            )

            if CALC_PSDS:
                ### PSDS ###
                # Add file id to tensor (used for getting filenames later)
                len_ = file_id.shape[0]
                file_ids[n_added_elems : (n_added_elems + len_)] = file_id

                # Reduce dimensions
                outputs = torch.squeeze(outputs)

                # Add outputs to output_table
                len_ = outputs.shape[0]
                output_table[n_added_elems : (n_added_elems + len_), :] = outputs

                # Track number of added elements so far
                n_added_elems += len_
                ############
    if CALC_PSDS:
        return total_loss, correct, total, i, output_table, file_ids

    return total_loss, correct, total, i, torch.tensor([]), torch.tensor([])


def test(
    te_loader: DataLoader,
    len_te: int,
    model,
    criterion,
    log: Logger,
    testing: bool = False,
) -> Tuple[float, float]:
    """
    Test loop. Calculates test loss and accuracy with no weight updates.
    """

    ##### Run test batches #####
    (
        test_loss,
        correct,
        total,
        total_te_batches,
        output_table,
        file_ids,
    ) = te_val_batches(te_loader, len_te, model, criterion, testing)

    # Log test loss and accuracy
    test_loss = test_loss.item()
    test_acc = correct.sum().item() / total
    log.info(
        f"Average testing loss: {test_loss/(total_te_batches+1)}",
        display_console=False,
    )
    log.info(f"Average test accuracy: {test_acc}", display_console=True)

    return test_loss, test_acc, output_table, file_ids


if __name__ == "__main__":

    ### MISC ###
    # Create new log (only a console logger - does not create file)
    log = Logger(LOGGER_TEST, Path(""), logging.DEBUG, logging.NOTSET)

    ### Load Datasets ###

    # Load datasets via DatasetWrapper
    DW = DatasetWrapper()
    DS_test_loader = DW.get_test_loader()
    DS_test = DW.get_test_ds()
    len_te = len(DS_test)

    # Prerequisite: All datasets have the same sample rate.
    sample_rate = DW.get_test_ds().get_sample_rate()

    ### Declare Model ###

    # Network
    model = NN(sample_rate, SAMPLE_RATE).to(device, non_blocking=True)
    # summary(model, input_size=(BATCH_SIZE, 1, 10 * sample_rate), device=device)

    # Loss function
    criterion = nn.BCELoss()

    # Load model from disk
    state = load_model(LOAD_MODEL_PATH)
    model.load_state_dict(state["state_dict"])
    log.info(f"Loaded model from {LOAD_MODEL_PATH}")

    # Number of trained parameters in network
    log.info(f"Number of trained parameters: {nr_parameters(model)}")

    ### Test Model ###
    log.info("Testing", add_header=True)

    te_loss, te_acc, output_table, file_ids = test(
        DS_test_loader, len_te, model, criterion, log, testing=True
    )

    # Dictionary with paths, losses and accuracies
    model_save = state["model_save"]
    tr_epoch_losses = model_save["tr_epoch_losses"]
    tr_epoch_accs = model_save["tr_epoch_accs"]
    val_epoch_losses = model_save["val_epoch_losses"]
    val_epoch_accs = model_save["val_epoch_accs"]

    # Plot training and validation accuracy
    if PLOT_TR_VAL_ACC:
        plot_tr_val_acc_loss(
            tr_epoch_losses, tr_epoch_accs, val_epoch_losses, val_epoch_accs
        )

    if CALC_PSDS:
        log.info("PSD-Score", add_header=True)

        ### PSDS and F1-Score ###
        operating_points_table = get_detection_table(output_table, file_ids, DS_test)

        # Get PSD-Score, F1-Score and optionally plot the PSD-ROC curve
        psds, f1_score_obj = psd_score(
            operating_points_table, DS_test.get_annotations(), plot=PLOT_PSD_ROC
        )

        log.info(
            "Parameters: "
            f"dtc={PSDS_PARAMS['dtc_threshold']}"
            f" | gtc={PSDS_PARAMS['gtc_threshold']}"
            f" | cttc={PSDS_PARAMS['cttc_threshold']}"
            f" | alpha_ct={PSDS_PARAMS['alpha_ct']}"
            f" | alpha_st={PSDS_PARAMS['alpha_st']}"
        )
        log.info(f"PSD-Score:    {psds:5f}")
        log.info(f"TPR:          {f1_score_obj.TPR[0]:5f}")
        log.info(f"FPR:          {f1_score_obj.FPR[0]:5f}")
        log.info(f"OP-threshold: {f1_score_obj.threshold[0]}")
        log.info(f"F1-Score:     {f1_score_obj.Fscore[0]:5f}")
