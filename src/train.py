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
from models.model_utils import nr_parameters, activation, update_acc


def train(
    tr_loader: DataLoader,
    len_tr: int,
    val_loader: DataLoader,
    len_val: int,
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
    # Tracks training losses and accuracies across epochs
    tr_epoch_losses, tr_epoch_accs = [], []

    # Tracks validation loss and accuracy across epochs
    val_epoch_losses, val_epoch_accs = [], []

    # Used for early stopping
    min_validation_loss = np.inf

    for epoch in range(1, EPOCHS + 1):
        log.info(f"Epoch {epoch}")

        # Track loss while epoch is running
        running_loss = 0
        # Track loss for a whole epoch
        training_loss = 0

        # Track correct and total predictions to calculate epoch accuracy
        correct = torch.zeros(-(len_tr // -BATCH_SIZE)).to(device, non_blocking=True)
        total = 0

        # Set model state to training
        model.train()

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

            # Track statistics
            loss = loss.detach()
            running_loss += loss
            training_loss += loss
            correct[i], total = update_acc(activation(outputs), labels, total)

            # log every 100 batches
            if i % 100 == 99:
                log.info(
                    f"[{epoch}, {i + 1:5d}] loss: {running_loss / 100:.3f}",
                    display_console=False,
                )
                running_loss = 0.0

        # Update learning_rate
        scheduler1.step()
        scheduler2.step()

        # Log training loss and accuracy
        log.info(
            f"Epoch {epoch}: Average training loss: {training_loss.item()/(i+1)}",
            display_console=False,
        )
        log.info(
            f"Epoch {epoch}: Average training accuracy: {correct.sum().item() / total}",
            display_console=False,
        )

        # Save this epoch's training loss and accuracy
        tr_epoch_losses.append(training_loss.item() / (i + 1))
        tr_epoch_accs.append(correct.sum().item() / total)

        # Track validation loss
        validation_loss = 0

        # Track correct and total predictions to calculate epoch accuracy
        correct = torch.zeros(-(len_val // -BATCH_SIZE)).to(device, non_blocking=True)
        total = 0

        # Set model state to evaluation
        model.eval()

        ##### Model validation #####
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
                correct[i], total = update_acc(activation(outputs), labels, total)

            # Log validation loss and accuracy
            log.info(
                f"Epoch {epoch}: Average validation loss: {validation_loss.item()/(i+1)}",
                display_console=False,
            )
            log.info(
                f"Epoch {epoch}: Average validation acc: {correct.sum().item() / total}",
                display_console=False,
            )

            # Save this epoch's training loss and accuracy
            val_epoch_losses.append(validation_loss.item() / (i + 1))
            val_epoch_accs.append(correct.sum().item() / total)

            # Early stopping:
            # Has validation loss decreased? If yes, keep training. If no, stop training (overfit).
            if min_validation_loss > validation_loss.item():
                min_validation_loss = validation_loss.item()
            else:
                log.info(
                    f"Model overfitting occurred at epoch {epoch}. Halting training.",
                    display_console=False,
                )
                return (
                    tr_epoch_losses,
                    tr_epoch_accs,
                    val_epoch_losses,
                    val_epoch_accs,
                )

    return (tr_epoch_losses, tr_epoch_accs, val_epoch_losses, val_epoch_accs)


def test(te_loader: DataLoader, len_te: int, model, criterion, log: Logger) -> float:
    """
    Test loop. Calculates test loss and accuracy.
    """
    # TODO: Move to eval.py
    # Track test loss
    test_loss = 0

    # Track correct and total predictions to calculate test accuracy
    correct = torch.zeros(-(len_te // -BATCH_SIZE)).to(device, non_blocking=True)
    total = 0

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
            correct[i], total = update_acc(activation(outputs), labels, total)

    # Log test loss and accuracy
    test_loss = test_loss.item()
    test_acc = correct.sum().item() / total
    log.info(f"Average testing loss: {test_loss/(i+1)}", display_console=False)
    log.info(f"Average test accuracy: {test_acc}", display_console=True)

    return test_loss, test_acc


if __name__ == "__main__":

    # Used for optimizing CNN training (when data is of same-sized inputs)
    torch.backends.cudnn.benchmark = True

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

    DS_train = DW.get_train_ds()
    DS_val = DW.get_val_ds()
    DS_test = DW.get_test_ds()

    len_tr = len(DS_train)
    len_val = len(DS_val)
    len_te = len(DS_test)

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

    tr_epoch_losses, tr_epoch_accs, val_epoch_loss, val_epoch_accs = train(
        DS_train_loader,
        len_tr,
        DS_val_loader,
        len_val,
        model,
        criterion,
        optimizer,
        scheduler1,
        scheduler2,
        log,
    )

    ### Test Model (temporary) ###
    log.info("Testing", add_header=True)

    te_loss, te_acc = test(DS_test_loader, len_te, model, criterion, log)
