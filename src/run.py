import torch
from torch import nn
from torch.nn.modules import loss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import argparse
import re
import numpy as np
import random
import time

# User defined imports
import config
from models_dict import MODELS, Model
from utils import get_datetime, create_basepath, timer
from plot_utils import plot_models
from models.model_utils import load_model, nr_parameters
from logger import CustomLogger as Logger
from datasets.dataset_handler import DatasetManager

from train import train
from validate import validate_model
from eval import calc_test_acc_loss, calc_test_psds
from predict import predict

# Use cuda if available, exit otherwise.
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    exit("Please use a GPU to train this model.")


def pick_multiple_models(models: List[Model]):
    """
    Function taken (and modified) from
    https://github.com/MattiasBeming/LiU-AI-Project-Active-Learning-for-Music.

    User-interface to pick a subset of models given a list of models.
    """
    # Initial prompt
    print("Pick models to run.")

    indexed_models = {i: m for i, m in enumerate(models)}
    picked_inds = []
    while True:
        # Print unpicked Ps
        print("Models to pick from:")
        if len(picked_inds) == len(indexed_models):
            print("\t-")
        else:
            for i, model in indexed_models.items():
                if i not in picked_inds:
                    print(f"\t{i}: {str(model)}")

        # Print picked models
        print("Picked Models:")
        if not picked_inds:
            print("\t-")
        else:
            for i in sorted(picked_inds):
                print(f"\t{i}: {str(indexed_models[i])}")

        # Input prompt
        print("Enter indices on format 'i' or 'i-j'.")
        print("Drop staged Models with 'drop i'.")
        print("Write 'done' when you are done.")

        # Handle input
        try:
            idx = input("> ")
            if idx == "done":  # Check if done
                break
            elif bool(re.match("^[0-9]+-[0-9]+$", idx)):  # Check if range
                span_str = idx.split("-")
                picked_inds += [
                    i
                    for i in range(int(span_str[0]), int(span_str[1]) + 1)
                    if i not in picked_inds
                ]
            elif bool(re.match("^drop [0-9]+$", idx)):
                picked_inds.remove(int(idx.split()[1]))
            elif (
                int(idx) in indexed_models.keys() and int(idx) not in picked_inds
            ):  # Check if singular
                picked_inds.append(int(idx))
        except ValueError:
            continue

    return [indexed_models[i] for i in picked_inds]


def deterministic_run(
    deterministic: bool = config.DETERMINISTIC_RUN, seed: int = config.SEED
):
    """
    Seeds the execution of training if 'deterministic' is True.
    """
    if deterministic:
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        # Used for optimizing CNN training (when data is of same-sized inputs)
        torch.backends.cudnn.benchmark = True


def add_run_args(parser: argparse.PARSER):
    """
    Add 'run.py'-specific arguments to the given argument parser.
    """
    # Train
    parser.add_argument(
        "-tr",
        "--train",
        action="store_true",
        help=("Training of models."),
    )
    parser.add_argument(
        "-ftr",
        "--force-retrain",
        action="store_true",
        help=("Force retraining of models."),
    )
    # Validate
    parser.add_argument(
        "-val",
        "--validate",
        action="store_true",
        help=("Validation of models."),
    )
    # Test
    parser.add_argument(
        "-te_acc_loss",
        "--test_acc_loss",
        action="store_true",
        help=("Testing of models - Metrics provided: Accuracy and Loss."),
    )
    parser.add_argument(
        "-te_psds",
        "--test_psds",
        action="store_true",
        help=("Testing of models - Metrics provided: PSDS"),
    )
    parser.add_argument(
        "-te_epoch",
        "--test_epoch",
        type=int,
        default=config.EPOCHS,
        help=("Specify which model-epoch to load and evaluate."),
    )
    # Predict
    parser.add_argument(
        "-pred",
        "--predict",
        action="store_true",
        help=("Get Predictions from custom dataset (look in config.py)"),
    )
    parser.add_argument(
        "-save_preds",
        "--save_predictions_to_tsv",
        action="store_true",
        help=("Save predictions in a tsv file."),
    )
    # Others
    parser.add_argument(
        "-op",
        "--op_threshold",
        type=float,
        default=config.ACT_THRESHOLD,
        help=("Operating Point (Activation) Threshold for accuracy/PSDS/F1-Score."),
    )
    # Plot arguments
    parser.add_argument(
        "-plot",
        "--plot",
        action="store_true",
        help=("The instructions for what to plot will be given at a later time."),
    )
    return parser


def train_main(
    device: str,
    models_to_train: List[Model],
    log: Logger,
    DS_train_loader: DataLoader,
    DS_train: Dataset,
    len_tr: int,
    criterion: loss._Loss,
):
    if not models_to_train:
        log.info("No models to train.")
        return

    log.info("Training", add_header=True)
    log.info(
        f"Training the following models: "
        f"{''.join([f'{m}, ' for m in models_to_train]).strip(', ')}"
    )

    ### Train models in 'models_to_train' ###
    for model_tr in tqdm(
        iterable=models_to_train,
        desc="Models",
        leave=False,
        position=0,
        colour=config.TQDM_MODELS,
    ):
        deterministic_run(config.DETERMINISTIC_RUN, config.SEED)
        log.info(f"Training model {str(model_tr)}...", display_console=False)
        tqdm.write(f"Training model {str(model_tr)}...")
        s_time = time.time()

        start_epoch = 1
        end_epoch = config.EPOCHS
        # Model Paths for saving model during training
        model_basepath = create_basepath(
            Path(config.SAVED_MODELS_DIR) / str(DS_train) / str(model_tr) / "train"
        )

        # Prerequisite: All sample rates within a dataset must be equal (or resampled
        # at dataset level) but may differ between datasets.
        sample_rates = set()
        sample_rates.add(DS_train.get_sample_rate())
        model = model_tr.get_NN()(sample_rates, config.SAMPLE_RATE).to(
            device, non_blocking=True
        )

        # Optimizer for updating weights
        optimizer = optim.SGD(
            model.parameters(), lr=config.LR_SGD, momentum=config.MOMENTUM
        )
        # Schedulers for updating learning rate
        scheduler = ExponentialLR(optimizer, gamma=config.GAMMA)

        # Log number of parameters to train in network
        log.info(
            f"Number of parameters to train: {nr_parameters(model)}",
            display_console=False,
        )

        train(
            device,
            start_epoch,
            end_epoch,  # Inclusive
            DS_train_loader,
            len_tr,
            model,
            criterion,
            optimizer,
            scheduler,
            log,
            model_basepath,
        )
        tqdm.write(f"> {str(model_tr)} done! Took: {timer(s_time, time.time())}")
        log.info(
            f"> {str(model_tr)} done! Took: {timer(s_time, time.time())}",
            display_console=False,
        )


def validation_main(
    device: str,
    picked_models: List[Model],
    DS_train_basepath: Path,
    log: Logger,
    criterion: loss._Loss,
    DS_val_loader: DataLoader,
    DS_val: Dataset,
    len_val: int,
    operating_points: np.ndarray = config.OPERATING_POINTS,
):
    log.info("Validation", add_header=True)
    log.info(f"Validate using dataset: {str(DS_val)}")

    # Prerequisite: All sample rates within a dataset must be equal (or resampled
    # at dataset level) but may differ between datasets.
    sample_rates = set()
    sample_rates.add(DS_val.get_sample_rate())

    for picked_model in tqdm(
        iterable=picked_models,
        desc="Models",
        leave=False,
        position=0,
        colour=config.TQDM_MODELS,
    ):
        s_time = time.time()
        train_model_basepath = DS_train_basepath / str(picked_model) / "train"
        tqdm.write(f"Validating model {str(picked_model)}...")
        log.info(f"Validating model {str(picked_model)}...", display_console=False)

        start_epoch = 1
        end_epoch = config.EPOCHS

        model = picked_model.get_NN()(sample_rates, config.SAMPLE_RATE).to(
            device, non_blocking=True
        )

        validate_model(
            device,
            start_epoch,
            end_epoch,  # Inclusive
            train_model_basepath,
            model,
            criterion,
            DS_val_loader,
            DS_val,
            len_val,
            operating_points,
        )
        tqdm.write(f"> {str(picked_model)} done! Took: {timer(s_time, time.time())}")
        log.info(
            f"> {str(picked_model)} done! Took: {timer(s_time, time.time())}",
            display_console=False,
        )


def test_main(
    args_run: argparse.PARSER,
    picked_models: List[Model],
    DS_train_basepath: Path,
    log: Logger,
    DS_test_loader: DataLoader,
    DS_test: Dataset,
    len_te: int,
    criterion: loss._Loss,
):
    log.info("Testing", add_header=True)
    log.info(f"Evaluate using dataset: {str(DS_test)}")

    # Prerequisite: All sample rates within a dataset must be equal (or resampled
    # at dataset level) but may differ between datasets.
    sample_rates = set()
    sample_rates.add(DS_test.get_sample_rate())

    for picked_model in tqdm(
        iterable=picked_models,
        desc="Models",
        leave=False,
        position=0,
        colour=config.TQDM_MODELS,
    ):
        log.info(f"Model: {str(picked_model)}", display_console=False)
        tqdm.write(f"Model: {str(picked_model)}")

        model = picked_model.get_NN()(sample_rates, config.SAMPLE_RATE).to(
            device, non_blocking=True
        )
        # load Model state
        model_path = (
            DS_train_basepath
            / str(picked_model)
            / "train"
            / f"e{args_run.test_epoch}.pt"
        )
        state = load_model(model_path)
        # Load state
        model.load_state_dict(state["state_dict"])

        if args_run.test_acc_loss:
            calc_test_acc_loss(
                device,
                DS_test_loader,
                len_te,
                model,
                criterion,
                args_run.op_threshold,
                log,
            )

        if args_run.test_psds:
            model_basepath = create_basepath(
                DS_train_basepath / str(picked_model) / "test" / str(DS_test)
            )
            calc_test_psds(
                device,
                model_basepath,
                DS_test_loader,
                DS_test,
                len_te,
                model,
                criterion,
                log,
                config.PSDS_PARAMS,
                config.OPERATING_POINTS,
                config.PLOT_PSD_ROC,
                config.SAVE_PSD_ROC,
            )


def predict_main(
    args_run: argparse.PARSER,
    picked_models: List[Model],
    log: Logger,
    DS_custom_loader: DataLoader,
    DS_custom: Dataset,
    len_custom: int,
    DS_train_basepath: Path,
    save_to_tsv: bool = False,
):
    log.info("Predictions", add_header=True)
    log.info(f"Evaluate using dataset: {str(DS_custom)}")

    # Prerequisite: All sample rates within a dataset must be equal (or resampled
    # at dataset level) but may differ between datasets.
    sample_rates = set()
    sample_rates.add(DS_custom.get_sample_rate())

    for picked_model in tqdm(
        iterable=picked_models,
        desc="Models",
        leave=False,
        position=0,
        colour=config.TQDM_MODELS,
    ):
        log.info(f"Model: {str(picked_model)}", display_console=False)
        tqdm.write(f"Model: {str(picked_model)}")

        model = picked_model.get_NN()(sample_rates, config.SAMPLE_RATE).to(
            device, non_blocking=True
        )
        # load Model state
        model_path = (
            DS_train_basepath
            / str(picked_model)
            / "train"
            / f"e{args_run.test_epoch}.pt"
        )
        state = load_model(model_path)
        # Load state
        model.load_state_dict(state["state_dict"])

        preds_model_basepath = create_basepath(
            DS_train_basepath / str(picked_model) / "predictions" / str(DS_custom)
        )
        det = predict(
            device,
            DS_custom_loader,
            DS_custom,
            len_custom,
            model,
            save_to_tsv,
            preds_model_basepath,
        )
        log.info(f"{str(det)}", display_console=False)
        tqdm.write(f"{str(det)}")


def plot_main(
    picked_models: List[Model],
    DS_train_basepath: Path,
    DS_val: Dataset,
    log: Logger,
):
    if not picked_models:
        exit("Nothing to plot.")

    # if config.EPOCHS == 1:
    #     exit("Cannot plot model with only 1 epoch")

    log.info("Plots", add_header=True)

    plots_basepath = create_basepath(DS_train_basepath / "plots" / str(DS_val))
    plot_models(
        config.WHAT_TO_PLOT,
        picked_models,
        DS_train_basepath,
        log,
        config.EPOCHS,
        config.ACT_THRESHOLD,
        DS_val,
        config.PSDS_PARAMS,
        config.OPERATING_POINTS,
        plots_basepath if config.SAVE_PLOT else None,
    )


def run(
    params_DS_train: Dict = config.DESED_SYNTH_TRAIN_ARGS,
    params_DS_val: Dict = config.DESED_SYNTH_VAL_ARGS,
    params_DS_test: Dict = config.DESED_PUBLIC_EVAL_ARGS,
    params_DS_preds: Dict = config.CUSTOM_ARGS,
):
    ### Instantiate parser ###
    parser_run = argparse.ArgumentParser(
        description="Which models to train/evaluate are specified in the next step. "
        "All other parameters are specified in 'config.py'"
    )
    parser_run = add_run_args(parser_run)
    args_run = parser_run.parse_args()

    if not (0.0 <= args_run.op_threshold and args_run.op_threshold <= 1.0):
        parser_run.error("'op_threshold' must be in range 0-1.")

    if (
        not args_run.train
        and not args_run.validate
        and not args_run.test_acc_loss
        and not args_run.test_psds
        and not args_run.force_retrain
        and not args_run.predict
        and not args_run.plot
    ):
        parser_run.error("Nothing to do (no parameters given). Exiting...")
        exit()

    # Loss function
    criterion = nn.BCELoss()

    # Create DatasetManager to load datasets
    DM = DatasetManager()

    # Terminal interface to choose which models to proceed with
    picked_models = pick_multiple_models(MODELS)

    # Fetching training dataset name
    DS_train_name = params_DS_train["name"]

    # Loading training dataset is only neccessary if the conditions are met.
    if (
        args_run.train
        or args_run.validate
        or args_run.test_acc_loss
        or args_run.test_psds
        or args_run.force_retrain
    ):
        print(f"Loading train dataset: {DS_train_name}")
        DS_train_loader = DM.load_dataset(**params_DS_train)
        DS_train = DM.get_dataset(DS_train_name)
        print("> Done!")

    # Train logger will be used for training, validation and testing
    filename = f"{get_datetime()}_Logger_{DS_train_name}.log"
    log_path = create_basepath(Path(config.LOG_DIR)) / filename
    log = Logger("RUN-Logger", log_path)

    DS_train_basepath = Path(config.SAVED_MODELS_DIR) / DS_train_name

    # Force retraining of all models from 'picked_models'
    if args_run.force_retrain:
        args_run.train = True
        models_to_train = picked_models
    else:
        # Determine which models to train (those that can be loaded are skipped)
        models_to_train = []
        for picked_model in picked_models:
            try:
                path = (
                    DS_train_basepath
                    / str(picked_model)
                    / "train"
                    / f"e{config.EPOCHS}.pt"
                )
                load_model(path)
            except:
                models_to_train.append(picked_model)

    if models_to_train and not args_run.force_retrain:
        if not args_run.train:
            log.warning(
                f"The following models are NOT trained for {config.EPOCHS} epochs. "
                f'models: {"".join([f"{m}, " for m in models_to_train]).strip(", ")}. '
                "\nStart training (note: from epoch 1) (y/n):"
            )
            while True:
                train_models = input("> ").lower()
                if train_models == "n":
                    exit("Exiting...")
                elif train_models == "y":
                    args_run.train = True
                    break

    ### Training ###
    if args_run.train:
        train_main(
            device,
            models_to_train,
            log,
            DS_train_loader,
            DS_train,
            len(DS_train),
            criterion,
        )

    ### Validation ###
    if args_run.validate:
        # Load DESED Synthetic Validation dataset
        DS_val_loader = DM.load_dataset(**params_DS_val)
        DS_val = DM.get_dataset(params_DS_val["name"])
        validation_main(
            device,
            picked_models,
            DS_train_basepath,
            log,
            criterion,
            DS_val_loader,
            DS_val,
            len(DS_val),
            config.OPERATING_POINTS,
        )

    ### Testing ###
    if args_run.test_acc_loss or args_run.test_psds:
        # Load Test dataset
        DS_test_loader = DM.load_dataset(**params_DS_test)
        DS_test = DM.get_dataset(params_DS_test["name"])
        test_main(
            args_run,
            picked_models,
            DS_train_basepath,
            log,
            DS_test_loader,
            DS_test,
            len(DS_test),
            criterion,
        )

    ### Predictions ###
    if args_run.predict:
        # Load Custom dataset
        DS_custom_loader = DM.load_dataset(**params_DS_preds)
        DS_custom = DM.get_dataset(params_DS_preds["name"])
        predict_main(
            args_run,
            picked_models,
            log,
            DS_custom_loader,
            DS_custom,
            len(DS_custom),
            DS_train_basepath,
            args_run.save_predictions_to_tsv,
        )

    ### Plots ###
    if args_run.plot:
        DS_val_loader = DM.load_dataset(**params_DS_val)
        DS_val = DM.get_dataset(params_DS_val["name"])
        plot_main(
            picked_models,
            DS_train_basepath,
            DS_val,
            log,
        )


if __name__ == "__main__":
    run(
        params_DS_train=config.DESED_SYNTH_TRAIN_ARGS,
        params_DS_val=config.DESED_SYNTH_VAL_ARGS,
        # params_DS_val = config.DESED_REAL_ARGS,
        params_DS_test=config.DESED_PUBLIC_EVAL_ARGS,
        params_DS_preds=config.CUSTOM_ARGS,
    )
