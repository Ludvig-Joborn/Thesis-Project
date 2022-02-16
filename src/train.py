import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
import numpy as np
from typing import Tuple

# User defined imports
from config import *
from utils import *
from datasets.datasets_utils import *
from logger import CustomLogger as Logger
from models.basic_nn import NeuralNetwork as NN
from datasets.dataset_handler import DatasetWrapper
from models.model_utils import nr_parameters


def train(
    tr_loader: DataLoader,
    val_loader: DataLoader,
    model,
    criterion,
    optimizer,
    scheduler1,
    scheduler2,
    log: Logger,
) -> Tuple[List[float], float]:
    """
    Training loop. Uses validation data for early stopping.

    """
    # TODO: accuracy
    epoch_losses, epoch_accs = [], []

    # Used for early stopping
    min_validation_loss = np.inf

    for epoch in range(1, EPOCHS + 1):
        log.info(f"Epoch {epoch}")

        running_losses, running_loss = [], 0
        epoch_loss = 0

        ##### Model training #####
        for i, sample in enumerate(
            tqdm(iterable=tr_loader, desc="Training Batch progress:")
        ):
            waveform, labels = sample

            # Send parameters to device
            waveform = waveform.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # reset the parameter gradients
            optimizer.zero_grad()

            # forward + loss + backward + optimize
            outputs = model.forward(waveform)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track loss statistics
            loss = loss.detach()
            running_loss += loss
            epoch_loss += loss

            # log every 100 batches
            if i % 100 == 99:
                log.info(
                    f"[{epoch}, {i + 1:5d}] loss: {running_loss / 100:.3f}",
                    display_console=False,
                )
                # running_losses.append(running_loss.item())
                running_loss = 0.0

        # Update learning_rate
        scheduler1.step()
        scheduler2.step()

        epoch_losses.append(epoch_loss.item() / (i + 1))

        ##### Model validation #####
        validation_loss = 0
        with torch.no_grad():
            for i, sample in enumerate(
                tqdm(iterable=val_loader, desc="Validation Batch progress:")
            ):
                waveform, labels = sample

                # Send parameters to device
                waveform = waveform.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # forward + loss
                outputs = model.forward(waveform)
                loss = criterion(outputs, labels)

                # Track loss statistics
                validation_loss += loss

            log.info(
                f"Average validation loss: {validation_loss.item()/(i+1)}",
                display_console=False,
            )

            # Early stopping:
            # Has validation loss decreased? If yes, keep training. If no, stop training (overfit).
            if min_validation_loss > validation_loss.item():
                min_validation_loss = validation_loss.item()
            else:
                log.info(
                    f"Model overfitting occurred at epoch {epoch}. Halting training.",
                    display_console=False,
                )
                return epoch_losses, min_validation_loss / (i + 1)

    return epoch_losses, min_validation_loss / (i + 1)


def test(te_loader: DataLoader, model, criterion, log: Logger) -> float:
    # TODO: Move to eval.py
    # TODO: accuracy
    test_loss, acc = 0, []
    with torch.no_grad():
        for i, sample in enumerate(
            tqdm(iterable=te_loader, desc="Test Batch progress:")
        ):
            waveform, labels = sample

            # Send parameters to device
            waveform = waveform.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # forward + loss
            outputs = model.forward(waveform)
            loss = criterion(outputs, labels)

            # Track loss statistics
            test_loss += loss

    test_loss = test_loss.item()
    log.info(f"Average testing loss: {test_loss/(i+1)}", display_console=False)

    return test_loss


if __name__ == "__main__":

    ### MISC ###
    log = Logger(LOGGER_TRAIN)

    # Use cuda if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        log.error("Please use a GPU to train this model.")
        exit()

    ### Load Datasets ###

    # Load datasets via DatasetWrapper
    DW = DatasetWrapper()
    DS_train_loader = DW.get_train_loader()
    DS_val_loader = DW.get_val_loader()
    DS_test_loader = DW.get_test_loader()

    # Prerequisite: All datasets have the same sample rate.
    sample_rate = DW.get_train_ds().get_sample_rate()

    ### Declare Model ###

    # Network
    model = NN(sample_rate, SAMPLE_RATE).to(device, non_blocking=True)

    # Loss function
    criterion = nn.MSELoss()

    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Schedulers for updating learning rate
    scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    ### Train Model ###
    log.info("Training", add_header=True)

    # Log number of parameters to train in network
    log.info(f"Number of parameters to train: {nr_parameters(model)}")

    tr_epoch_loss, val_epoch_loss = train(
        DS_train_loader,
        DS_val_loader,
        model,
        criterion,
        optimizer,
        scheduler1,
        scheduler2,
        log,
    )

    ### Test Model (temporary) ###
    log.info("Testing", add_header=True)

    te_epoch_loss = test(DS_test_loader, model, criterion, log)
