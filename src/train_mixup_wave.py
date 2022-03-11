import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from typing import Tuple
from torch.autograd import Variable
import numpy as np

# User defined imports
from config import *
from utils import *
from eval import te_val_batches, test
from models.model_utils import *
from datasets.datasets_utils import *
from logger import CustomLogger as Logger
from models.baseline import NeuralNetwork as NN
from datasets.dataset_handler import DatasetWrapper

"""
This file is for training models by applying mixup on waveforms.
"""


# Mixup alpha parameter (for beta-distribution) to generate lambda.
# Set to 0 for no mixup anywhere.
ALPHA = 0.2

# Use cuda if available, exit otherwise.
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    exit("Please use a GPU to train this model.")


def mixup_data(x, y, alpha=1.0):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    Imported from: https://github.com/facebookresearch/mixup-cifar10/blob/eaff31ab397a90fbc0a4aac71fb5311144b3608b/train.py#L76
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # Generate a random permutation inputs and labels
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    # Create a mixed input
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # Create permuted labels
    y, y_permuted = y, y[index]
    return (mixed_x, y, y_permuted, lam)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_batches(
    tr_loader: DataLoader,
    len_tr: int,
    model,
    criterion,
    optimizer,
) -> Tuple[int, torch.Tensor, int, int]:
    """
    Runs training batches and updates model weights.
    Also calculates training accuracy and loss.
    """
    # Track loss for a whole epoch
    training_loss = 0

    # Track correct and total predictions to calculate epoch accuracy
    correct_tr = torch.zeros(-(len_tr // -BATCH_SIZE)).to(device, non_blocking=True)
    total_tr = 0

    # Set model state to training
    model.train()

    ##### Model training #####
    for i, sample in enumerate(
        tqdm(iterable=tr_loader, desc="Training Batch progress:")
    ):
        waveform, labels, _ = sample

        # Send parameters to device
        waveform = waveform.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply mixup to input and store th pair of labels used
        inputs, labels, labels_permuted, lam = mixup_data(waveform, labels, ALPHA)
        labels, labels_permuted = map(Variable, (labels, labels_permuted))

        # reset the parameter gradients
        optimizer.zero_grad()

        # forward + loss + backward + optimize
        outputs = model.forward(inputs)
        loss = mixup_criterion(criterion, outputs, labels, labels_permuted, lam)
        loss.backward()
        optimizer.step()

        # Track statistics
        loss = loss.detach()
        training_loss += loss
        total_tr += torch.numel(labels)
        predictions = activation(outputs, ACT_THRESHOLD)
        # Calculate mixup-based correct predictions.
        correct_tr[i] = (
            lam * predictions.eq(labels).sum().float()
            + (1 - lam) * predictions.eq(labels_permuted).sum().float()
        )

    return training_loss, correct_tr, total_tr, i


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
    Training loop. Uses validation data to save the best model.
    """
    # Load model paramters (if empty params, start training from scratch)
    model_path = model_save["model_path"]
    best_model_path = model_save["best_model_path"]
    # Tracks losses and accuracies across epochs
    tr_epoch_losses = model_save["tr_epoch_losses"]
    tr_epoch_accs = model_save["tr_epoch_accs"]
    val_epoch_losses = model_save["val_epoch_losses"]
    val_epoch_accs = model_save["val_epoch_accs"]
    # Used for saving the best model
    best_val_acc = model_save["best_val_acc"]

    for epoch in range(start_epoch, EPOCHS + 1):
        log.info(f"Epoch {epoch}")

        ##### Model batch training #####
        training_loss, correct_tr, total_tr, total_tr_epochs = train_batches(
            tr_loader, len_tr, model, criterion, optimizer
        )

        # Update learning_rate
        scheduler1.step()
        scheduler2.step()

        # Log training loss and accuracy
        log.info(
            f"Epoch {epoch}: Average training loss: {training_loss.item()/(total_tr_epochs+1)}",
            display_console=False,
        )
        log.info(
            f"Epoch {epoch}: Average training accuracy: {correct_tr.sum().item() / total_tr}",
            display_console=False,
        )

        # Save this epoch's training loss and accuracy
        tr_epoch_losses.append(training_loss.item() / (total_tr_epochs + 1))
        tr_epoch_accs.append(correct_tr.sum().item() / total_tr)

        ##### Model validation #####
        validation_loss, correct_val, total_val, total_val_epochs = te_val_batches(
            val_loader, len_val, model, criterion
        )

        # Log validation loss and accuracy
        log.info(
            f"Epoch {epoch}: Average validation loss: {validation_loss.item()/(total_val_epochs+1)}",
            display_console=False,
        )
        log.info(
            f"Epoch {epoch}: Average validation acc: {correct_val.sum().item() / total_val}",
            display_console=False,
        )

        # Save this epoch's training loss and accuracy
        val_acc = correct_val.sum().item() / total_val
        val_epoch_losses.append(validation_loss.item() / (total_val_epochs + 1))
        val_epoch_accs.append(val_acc)

        # Save model after each epoch
        model_save = {
            "model_path": model_path,
            "best_model_path": best_model_path,
            "tr_epoch_losses": tr_epoch_losses,
            "tr_epoch_accs": tr_epoch_accs,
            "val_epoch_losses": val_epoch_losses,
            "val_epoch_accs": val_epoch_accs,
            "best_val_acc": best_val_acc,
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

        # Has validation accuracy increased?
        # If yes, save the current model as 'best' model.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            log.info(f"New best model at epoch {epoch}. Saving model.")
            save_model(state, True, model_path, best_model_path)

    return (tr_epoch_losses, tr_epoch_accs, val_epoch_losses, val_epoch_accs)


if __name__ == "__main__":

    # Used for optimizing CNN training (when data is of same-sized inputs)
    torch.backends.cudnn.benchmark = True

    ### MISC ###
    if CONTINUE_TRAINING:
        state = load_model(LOAD_MODEL_PATH)
        log_path = state["model_save"]["log_path"]
    else:
        # Create new logfile
        filename = get_datetime() + "_" + LOGGER_TRAIN.split("-")[0]
        log_path = create_path(Path(LOG_DIR), filename, ".log")

    log = Logger(LOGGER_TRAIN, log_path)

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
    criterion = nn.BCELoss()

    # optimizer = optim.Adam(model.parameters(), lr=LR_adam, weight_decay=WD)
    optimizer = optim.SGD(model.parameters(), lr=LR_sgd, momentum=MOMENTUM)

    # Schedulers for updating learning rate
    scheduler1 = ExponentialLR(optimizer, gamma=GAMMA_1)
    scheduler2 = MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA_2)

    # Load model from disk to continue training
    if CONTINUE_TRAINING:
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
        start_epoch = 1
        # Model Paths for saving model during training
        model_path = create_path(Path(SAVED_MODELS_DIR))
        best_model_path = create_path(Path(SAVED_MODELS_DIR), best=True)
        model_save = {
            "model_path": model_path,
            "best_model_path": best_model_path,
            "tr_epoch_losses": [],
            "tr_epoch_accs": [],
            "val_epoch_losses": [],
            "val_epoch_accs": [],
            "best_val_acc": 0,
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

    te_loss, te_acc = test(DS_test_loader, len_te, model, criterion, log, testing=True)

    plot_tr_val_acc_loss(
        tr_epoch_losses, tr_epoch_accs, val_epoch_losses, val_epoch_accs
    )
