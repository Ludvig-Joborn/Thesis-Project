import torch
from torch.nn import Module
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim import lr_scheduler
from tqdm import tqdm
from typing import Tuple
from pathlib import Path

# User defined imports
import config
from models.model_utils import save_model, calc_nr_correct_predictions, activation
from logger import CustomLogger as Logger


def train_batches(
    device: str,
    tr_loader: DataLoader,
    len_tr: int,
    model: Module,
    criterion: loss._Loss,
    optimizer: Optimizer,
    scheduler: lr_scheduler,
    SNR_DB: torch.Tensor,
) -> Tuple[float, float]:
    """
    Runs training batches and updates model weights.
    Also tracks the training loss,
    as well as the number of correct and total predictions.
    """
    # Track loss for a whole epoch
    training_loss = 0
    # Track correct and total predictions to calculate epoch accuracy
    correct_tr = torch.zeros(-(len_tr // -config.BATCH_SIZE)).to(
        device, non_blocking=True
    )
    total_tr = 0

    # Set model state to training
    model.train()

    ##### Model training #####
    for i, sample in enumerate(
        tqdm(
            iterable=tr_loader,
            desc="Training Batch progress",
            position=2,
            leave=False,
            colour=config.TQDM_BATCHES,
        )
    ):
        waveform, sample_rate, labels, _ = sample

        # Send parameters to device
        waveform = waveform.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # reset the parameter gradients
        optimizer.zero_grad()

        # forward + loss + backward + optimize
        outputs = model.forward(waveform, sample_rate, SNR_DB)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Track statistics
        loss = loss.detach()
        training_loss += loss
        correct_tr[i], total_i = calc_nr_correct_predictions(
            activation(outputs, config.ACT_THRESHOLD), labels
        )
        total_tr += total_i

    tr_loss = float(training_loss.item() / total_tr)
    tr_acc = float(correct_tr.sum().item() / total_tr)

    # Update learning rate
    scheduler.step()

    return tr_loss, tr_acc


def train(
    device: str,
    start_epoch: int,
    end_epoch: int,  # Inclusive
    DS_train_loader: DataLoader,
    len_tr: int,
    model: Module,
    criterion: loss._Loss,
    optimizer: Optimizer,
    scheduler: lr_scheduler,
    log: Logger,
    model_basepath: Path,
    SNR_DB: torch.Tensor,
):
    """
    Trains a model on the given dataset and saves the weights (with the training loss and accuracy).
    """
    tr_epoch_losses = []
    tr_epoch_accs = []
    for epoch in tqdm(
        iterable=range(start_epoch, end_epoch + 1),
        desc="Epoch",
        position=1,
        leave=False,
        colour=config.TQDM_EPOCHS,
    ):
        log.info(f"Epoch {epoch}", display_console=False)

        epoch_filepath = model_basepath / f"e{epoch}.pt"

        ##### Model batch training #####
        tr_loss, tr_acc = train_batches(
            device,
            DS_train_loader,
            len_tr,
            model,
            criterion,
            optimizer,
            scheduler,
            SNR_DB,
        )

        ### Log training loss and accuracy ###
        log.info(
            f"Epoch {epoch}: Average training loss: {tr_loss}",
            display_console=False,
        )
        log.info(
            f"Epoch {epoch}: Average training accuracy: {tr_acc}",
            display_console=False,
        )

        ### Save Model ###
        # Update for this epoch's training loss and accuracy
        tr_epoch_losses.append(tr_loss)
        tr_epoch_accs.append(tr_acc)
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "tr_epoch_losses": tr_epoch_losses,
            "tr_epoch_accs": tr_epoch_accs,
        }
        save_model(state, epoch_filepath)
