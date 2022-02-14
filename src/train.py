import torch
from torch.utils.data import DataLoader
from torch import float32, nn
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
import numpy as np
from typing import Tuple

# User defined imports
from config import *
from utils import *
from datasets.datasets_utils import *
from datasets.desed import DESED_Strong
from logger import CustomLogger as Logger
from models.basic_nn import NeuralNetwork as NN

# from models.model_utils import nr_parameters

log = Logger("train-Logger")

# Use cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    log.error("Please use a GPU to train this model.")
    exit()

# Test code to plot mel spectrogram
def load_datasets():
    # TODO: Use dataset handler
    # Load dataset DESED Train (Synthetic)
    DESED_train = DESED_Strong(
        "DESED Synthetic Training",
        PATH_TO_SYNTH_TRAIN_DESED_TSV,
        PATH_TO_SYNTH_TRAIN_DESED_WAVS,
        DESED_CLIP_LEN_SECONDS,
    )

    # Load dataset DESED Validation (Synthetic)
    DESED_val = DESED_Strong(
        "DESED Synthetic Validation",
        PATH_TO_SYNTH_VAL_DESED_TSV,
        PATH_TO_SYNTH_VAL_DESED_WAVS,
        DESED_CLIP_LEN_SECONDS,
    )

    # Load dataset DESED Test (Public Evaluation)
    DESED_test = DESED_Strong(
        "DESED Public Evaluation",
        PATH_TO_PUBLIC_TEST_DESED_TSV,
        PATH_TO_PUBLIC_TEST_DESED_WAVS,
        DESED_CLIP_LEN_SECONDS,
    )

    _, sample_rate, __ = DESED_train.__getitem__(0)  # Get sample rate
    tr_loader = DataLoader(
        DESED_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        DESED_val, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0
    )
    te_loader = DataLoader(
        DESED_test, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2
    )

    return sample_rate, tr_loader, val_loader, te_loader


def train(
    tr_loader: DataLoader,
    val_loader: DataLoader,
    model,
    criterion,
    optimizer,
    scheduler1,
    scheduler2,
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
            waveform, _, labels = sample

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
                waveform, _, labels = sample

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


def test(te_loader: DataLoader, model, criterion) -> float:
    # TODO: Move to eval.py
    # TODO: accuracy
    test_loss, acc = 0, []
    with torch.no_grad():
        for i, sample in enumerate(
            tqdm(iterable=te_loader, desc="Test Batch progress:")
        ):
            waveform, _, labels = sample

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
    sample_rate, tr_loader, val_loader, te_loader = load_datasets()
    # Prerequisite: Sample rate is the same for all clips in the dataset
    model = NN(sample_rate, SAMPLE_RATE).to(device, non_blocking=True)
    criterion = nn.MSELoss()

    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Use schedulers ?
    scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    log.info("Training", add_header=True)
    tr_epoch_loss, val_epoch_loss = train(
        tr_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        scheduler1,
        scheduler2,
    )
    log.info("Testing", add_header=True)
    te_epoch_loss = test(te_loader, model, criterion)
