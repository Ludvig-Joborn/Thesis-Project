import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
import numpy as np

# User defined imports
import logging
from config import *
from utils import *
from temp_train import train
from models.model_utils import *
from datasets.datasets_utils import *
from logger import CustomLogger as Logger
from datasets.dataset_handler import DatasetWrapper

# from models.basic_nn import NeuralNetwork as NN
from models.model_extensions_1.b1 import NeuralNetwork as NN_b1
from models.basic_nn import NeuralNetwork as NN_basic


def model_selection(filename: str, network):
    torch.backends.cudnn.benchmark = True

    # Create new logfile
    log_path = create_path(Path(LOG_DIR), filename, ".log")
    log = Logger(LOGGER_TRAIN + "-" + filename, log_path, logging.DEBUG, logging.DEBUG)

    # Use cuda if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.error("Please use a GPU to train this model.")
        exit()

    ##### Load Datasets #####

    # Load datasets via DatasetWrapper
    DW = DatasetWrapper()
    DS_train_loader = DW.get_train_loader()
    DS_val_loader = DW.get_val_loader()

    DS_train = DW.get_train_ds()
    DS_val = DW.get_val_ds()

    len_tr = len(DS_train)
    len_val = len(DS_val)

    # Prerequisite: All datasets have the same sample rate.
    sample_rate = DW.get_train_ds().get_sample_rate()

    ##### Declare Model #####
    # Network
    model = network(sample_rate, SAMPLE_RATE).to(device, non_blocking=True)
    # summary(model, input_size=(8, 1, 441000), device=device)

    # Loss function
    criterion = nn.BCELoss()

    # optimizer = optim.Adam(model.parameters(), lr=LR_adam, weight_decay=WD)
    optimizer = optim.SGD(model.parameters(), lr=LR_sgd, momentum=MOMENTUM)

    # Schedulers for updating learning rate
    scheduler1 = ExponentialLR(optimizer, gamma=GAMMA_1)
    scheduler2 = MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA_2)

    # Load model from disk to continue training

    start_epoch = 1
    # Model Paths for saving model during training
    model_path = create_path(Path(SAVED_MODELS_DIR), filename)
    best_model_path = create_path(Path(SAVED_MODELS_DIR), filename, best=True)
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

    # Calling train will modify model_save, which is used to track losses and accuracies.
    train(
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

    return model_save


if __name__ == "__main__":
    model_saves = {}
    epochs = np.inf

    # Load pre trained models and add to dict
    trained_modules = {"b1": load_model(Path("E:/saved_models/b1.pt"))["model_save"]}
    for key, model_save in trained_modules.items():
        model_saves[key] = model_save
        if len(model_save["tr_epoch_losses"]) < EPOCHS:
            exit(
                f"ERROR: Pre-trained model {key} has not been sufficiently trained. Please train it to {EPOCHS} epochs."
            )

    # Initialize what models to run.
    # Change 'modules' to include more models in the selection process.
    modules_to_train = {"baseline": NN_basic}

    # Train each model and store the 'model_save'
    for key, model in modules_to_train.items():
        model_save = model_selection(key, model)
        model_saves[key] = model_save

    plot_model_selection(model_saves)
