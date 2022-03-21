import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

# User defined imports
import logging
from utils import *
from train import train
from eval import test
from models.model_utils import *
from datasets.datasets_utils import *
from logger import CustomLogger as Logger
from datasets.dataset_handler import DatasetManager

# Baseline, basic
from models.model_extensions_1.basic_nn import NeuralNetwork as basic_nn
from models.baseline import NeuralNetwork as baseline

# b2
from models.model_extensions_1.b2 import NeuralNetwork as b2
from models.model_extensions_1.b2_cbam import NeuralNetwork as b2_cbam
from models.model_extensions_1.b2_cbam_drop01 import NeuralNetwork as b2_cbam_drop01
from models.model_extensions_1.b2_cbam_drop01_lindrop import (
    NeuralNetwork as b2_cbam_drop01_lindrop,
)
from models.model_extensions_1.b2_cbam_drop02 import NeuralNetwork as b2_cbam_drop02

# b1
from models.model_extensions_1.b1 import NeuralNetwork as b1
from models.model_extensions_1.b1_cbam import NeuralNetwork as b1_cbam
from models.model_extensions_1.b1_cbam_drop01 import NeuralNetwork as b1_cbam_drop01
from models.model_extensions_1.b1_cbam_drop01_lindrop import (
    NeuralNetwork as b1_cbam_drop01_lindrop,
)
from models.model_extensions_1.b1_cbam_drop02 import NeuralNetwork as b1_cbam_drop02

# Pooling
from models.pooling_extensions.lp_pool import NeuralNetwork as lp_pool
from models.pooling_extensions.max_pool import NeuralNetwork as max_pool

# Model extensions 2
from models.model_extensions_2.baseline_ks33_l22 import (
    NeuralNetwork as baseline_ks33_l22,
)
from models.model_extensions_2.baseline_ks33_l44 import (
    NeuralNetwork as baseline_ks33_l44,
)
from models.model_extensions_2.baseline_ks53_l24 import (
    NeuralNetwork as baseline_ks53_l24,
)
from models.model_extensions_2.baseline_ks73_l12 import (
    NeuralNetwork as baseline_ks73_l12,
)

# Recurrent momory units like LSTM and GRU
from models.recurrent_memory_ext.gru_2_drop01 import NeuralNetwork as gru_2_drop01
from models.recurrent_memory_ext.gru_2 import NeuralNetwork as gru_2
from models.recurrent_memory_ext.gru_4 import NeuralNetwork as gru_4
from models.recurrent_memory_ext.lstm_1 import NeuralNetwork as lstm_1
from models.recurrent_memory_ext.lstm_2_drop01 import NeuralNetwork as lstm_2_drop01
from models.recurrent_memory_ext.lstm_3_drop01 import NeuralNetwork as lstm_3_drop01

# Activation functions
from models.act_funcs.ls_relu import NeuralNetwork as ls_relu
from models.act_funcs.ls_relu_tr import NeuralNetwork as ls_relu_tr
from models.act_funcs.swish import NeuralNetwork as swish
from models.act_funcs.swish_tr import NeuralNetwork as swish_tr


def model_selection(filename: str, network: nn.Module):
    """
    Trains a model on a given network-structure.
    """
    if config.DETERMINISTIC_RUN:
        import numpy as np
        import random

        random.seed(config.SEED)
        np.random.seed(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
    else:
        # Used for optimizing CNN training (when data is of same-sized inputs)
        torch.backends.cudnn.benchmark = True

    # Create new logfile
    log_path = create_path(Path(config.LOG_DIR), filename, ".log")
    log = Logger(
        config.LOGGER_TRAIN + "-" + filename, log_path, logging.DEBUG, logging.DEBUG
    )

    # Use cuda if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.error("Please use a GPU to train this model.")
        exit()

    ##### Load Datasets #####

    # Load datasets via DatasetManager
    DM = DatasetManager()
    # Load Train
    DS_train_loader = DM.load_dataset(**config.DESED_SYNTH_TRAIN_ARGS)
    DS_train = DM.get_dataset(config.DESED_SYNTH_TRAIN_ARGS["name"])
    # Load Validation
    DS_val_loader = DM.load_dataset(**config.DESED_SYNTH_VAL_ARGS)
    DS_val = DM.get_dataset(config.DESED_SYNTH_VAL_ARGS["name"])
    # Load Test
    DS_test_loader = DM.load_dataset(**config.DESED_PUBLIC_EVAL_ARGS)
    DS_test = DM.get_dataset(config.DESED_PUBLIC_EVAL_ARGS["name"])

    len_tr = len(DS_train)
    len_val = len(DS_val)
    len_te = len(DS_test)

    # Prerequisite: All datasets have the same sample rate.
    sample_rate = DS_train.get_sample_rate()

    ##### Declare Model #####
    # Network
    model = network(sample_rate, config.SAMPLE_RATE).to(device, non_blocking=True)
    # summary(model, input_size=(config.BATCH_SIZE, 1, CLIP_LEN_SECONDS * SAMPLE_RATE), device=device)

    # Loss function
    criterion = nn.BCELoss()

    # optimizer = optim.Adam(model.parameters(), lr=LR_adam, weight_decay=WD)
    optimizer = optim.SGD(
        model.parameters(), lr=config.LR_sgd, momentum=config.MOMENTUM
    )

    # Schedulers for updating learning rate
    scheduler1 = ExponentialLR(optimizer, gamma=config.GAMMA_1)

    start_epoch = 1
    # Model Paths for saving model during training
    model_path = create_path(Path(config.SAVED_MODELS_DIR), filename)
    best_model_path = create_path(Path(config.SAVED_MODELS_DIR), filename, best=True)
    # Define model_save dict that is mutable and updated in train()
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
        log,
        model_save,
    )

    ### Test Model (temporary) ###
    log.info("Testing", add_header=True)

    # Run on test data and log the accuracy and loss
    test(DS_test_loader, len_te, model, criterion, log, testing=True)

    return model_save


if __name__ == "__main__":
    model_saves = {}

    # Load pre trained models and add to dict
    trained_modules = {
        "baseline": load_model(Path("E:/saved_models/baseline.pt"))["model_save"],
        "b1_cbam": load_model(Path("E:/saved_models/b1_cbam.pt"))["model_save"],
        "b1_cbam_drop01": load_model(Path("E:/saved_models/b1_cbam_drop01.pt"))[
            "model_save"
        ],
    }
    for key, model_save in trained_modules.items():
        model_saves[key] = model_save
        if len(model_save["tr_epoch_losses"]) < EPOCHS:
            exit(
                f"ERROR: Pre-trained model {key} has not been sufficiently trained. Please train it to {EPOCHS} epochs."
            )

    # Initialize what models to run.
    # Change 'modules' to include more models in the selection process.
    modules_to_train = {
        # Pooling
        # "lp_pool_all": lp_pool,
        # "max_pool_all": max_pool,
        #
        # Different conv-layer-structures
        # "baseline_ks33_l22": baseline_ks33_l22,
        # "baseline_ks33_l44": baseline_ks33_l44,
        # "baseline_ks53_l24": baseline_ks53_l24,
        # "baseline_ks73_l12": baseline_ks73_l12,
        #
        # GRU
        # "gru_2_drop01": gru_2_drop01,
        # "gru_2": gru_2,
        # "gru_4": gru_4,
        #
        # LSTM
        # "lstm_1": lstm_1,
        # "lstm_2_drop01": lstm_2_drop01,
        # "lstm_3_drop01": lstm_3_drop01,
        #
        # Activation function LS-ReLU - VERY Slow -> Don't Run!
        # "ls_relu": ls_relu,
        # "ls_relu_tr": ls_relu_tr,
        #
        # Activation function Swish
        # "swish": swish,
        # "swish_tr": swish_tr,
    }

    # Train each model and store the 'model_save'
    for key, model in modules_to_train.items():
        model_save = model_selection(key, model)
        model_saves[key] = model_save

    if model_saves:
        plot_model_selection(model_saves)
    else:
        exit("Nothing to plot, please specify models evaluate.")
