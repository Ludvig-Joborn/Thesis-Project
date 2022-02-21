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
from models.model_utils import *


def train(
    start_epoch: int,
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
    model_save: Dict,
) -> Tuple[List[float], float]:
    """
    Training loop. Uses validation data for early stopping.
    """
    # Load model paramters (if empty params, start training from scratch)
    model_path = model_save["model_path"]
    best_model_path = model_save["best_model_path"]
    # Tracks losses and accuracies across epochs
    tr_epoch_losses = model_save["tr_epoch_losses"]
    tr_epoch_accs = model_save["tr_epoch_accs"]
    val_epoch_losses = model_save["val_epoch_losses"]
    val_epoch_accs = model_save["val_epoch_accs"]

    # Used for early stopping
    min_validation_loss = np.inf

    for epoch in range(start_epoch, EPOCHS + 1):
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

            # Save model after each epoch
            model_save = {
                "model_path": model_path,
                "best_model_path": best_model_path,
                "tr_epoch_losses": tr_epoch_losses,
                "tr_epoch_accs": tr_epoch_accs,
                "val_epoch_losses": val_epoch_losses,
                "val_epoch_accs": val_epoch_accs,
                "log_path": log.path(),
            }
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler1": scheduler1.state_dict(),
                "scheduler2": scheduler2.state_dict(),
                "model_save": model_save,
            }
            save_model(state, False, model_path)

            # Early stopping:
            # Has validation loss decreased? If yes, keep training. If no, stop training (overfit).
            if min_validation_loss > validation_loss.item():
                min_validation_loss = validation_loss.item()

                # Save best model so far
                save_model(state, True, model_path, best_model_path)
            else:
                log.info(
                    f"Model overfitting occurred at epoch {epoch}. Halting training."
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

    # Set model state to evaluation
    model.eval()

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
    if LOAD_MODEL:
        state = load_model(LOAD_MODEL_PATH)
        log_path = state["model_save"]["log_path"]
    else:
        # Create new logfile
        filename = get_datetime() + "_" + LOGGER_TRAIN.split("-")[0]
        log_path = create_path(Path(LOG_DIR), filename, ".log")

    log = Logger(LOGGER_TRAIN, log_path)

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
    start_epoch = 1
    model = NN(sample_rate, SAMPLE_RATE).to(device, non_blocking=True)

    # Loss function
    criterion = nn.MSELoss()

    # optimizer = optim.Adam(model.parameters(), lr=LR_adam, weight_decay=WD)
    optimizer = optim.SGD(model.parameters(), lr=LR_sgd, momentum=MOMENTUM)

    # Schedulers for updating learning rate
    scheduler1 = ExponentialLR(optimizer, gamma=GAMMA_1)
    scheduler2 = MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA_2)

    # Load model from disk to continue training
    if LOAD_MODEL:
        start_epoch = state["epoch"] + 1  # Start from next epoch
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler1.load_state_dict(state["scheduler1"])
        scheduler2.load_state_dict(state["scheduler2"])

        # Dictionary with path, losses and accuracies
        model_save = state["model_save"]

        log.info("")
        log.info("")
        log.info("")
        log.info(f"Loaded model from {LOAD_MODEL_PATH}")
    else:
        # Model Paths for saving model during training
        model_path = create_path(Path(SAVED_MODELS_DIR))
        best_model_path = create_path(Path(SAVED_MODELS_DIR), ".pt", best=True)
        model_save = {
            "model_path": model_path,
            "best_model_path": best_model_path,
            "tr_epoch_losses": [],
            "tr_epoch_accs": [],
            "val_epoch_losses": [],
            "val_epoch_accs": [],
            "log_path": log.path(),
        }

    ### Train Model ###
    log.info("Training", add_header=True)

    # Log number of parameters to train in network
    log.info(f"Number of parameters to train: {nr_parameters(model)}")

    tr_epoch_losses, tr_epoch_accs, val_epoch_losses, val_epoch_accs = train(
        start_epoch,
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
        model_save,
    )

    ### Test Model (temporary) ###
    log.info("Testing", add_header=True)

    te_loss, te_acc = test(DS_test_loader, len_te, model, criterion, log)

    plot_tr_val_acc_loss(
        tr_epoch_losses, tr_epoch_accs, val_epoch_losses, val_epoch_accs
    )
