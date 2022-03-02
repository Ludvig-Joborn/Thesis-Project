import torch
from torchinfo import summary
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import logging

# User defined imports
from config import *
from utils import *
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
) -> Tuple[float, torch.Tensor, int, int]:
    """
    Runs batches for the dataloader sent in without updating any model weights.
    Used for calculating validation and test accuracy and loss.
    """
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
            waveform, labels = sample

            # Send parameters to device
            waveform = waveform.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # forward + loss
            outputs = model.forward(waveform)
            loss = criterion(outputs, labels)

            # Track loss statistics
            total_loss += loss
            correct[i], total = update_acc(activation(outputs), labels, total)

        return total_loss, correct, total, i


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
    test_loss, correct, total, total_te_batches = te_val_batches(
        te_loader, len_te, model, criterion, testing
    )

    # Log test loss and accuracy
    test_loss = test_loss.item()
    test_acc = correct.sum().item() / total
    log.info(
        f"Average testing loss: {test_loss/(total_te_batches+1)}",
        display_console=False,
    )
    log.info(f"Average test accuracy: {test_acc}", display_console=True)

    return test_loss, test_acc


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

    te_loss, te_acc = test(DS_test_loader, len_te, model, criterion, log, testing=True)

    # Dictionary with paths, losses and accuracies
    model_save = state["model_save"]
    tr_epoch_losses = model_save["tr_epoch_losses"]
    tr_epoch_accs = model_save["tr_epoch_accs"]
    val_epoch_losses = model_save["val_epoch_losses"]
    val_epoch_accs = model_save["val_epoch_accs"]

    plot_tr_val_acc_loss(
        tr_epoch_losses, tr_epoch_accs, val_epoch_losses, val_epoch_accs
    )
