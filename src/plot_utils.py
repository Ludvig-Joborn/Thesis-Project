import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import Callable, List, Dict
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math

# User-defined imports
import config
from models.model_utils import load_model
from utils import get_datetime, psd_score, create_basepath
from models_dict import Model
from logger import CustomLogger as Logger



def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=True)


def get_mel_spectrogram(
    sample_rate: int, n_fft: int, win_length: int, hop_length: int, n_mels: int
) -> Callable:
    """
    Returns a function that takes a waveform and returns a mel spectrogram.
    """
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )
    return mel_spectrogram


def get_val_model_values(
    picked_models: List[Model],
    basepath: Path,
    epochs: int,
    log: Logger,
    act_threshold: float = config.ACT_THRESHOLD,
    DS_val: Dataset = None,
    psds_params: Dict = config.PSDS_PARAMS,
    operating_points: np.ndarray = config.OPERATING_POINTS,
) -> Dict[str, Dict[str, List]]:
    """
    Creates and returns a dictionary with model names as keys and dictionaries with their metrics as values.
    """
    outputs = {}
    for picked_model in tqdm(
        iterable=picked_models,
        desc="Models",
        leave=False,
        position=0,
        colour=config.TQDM_MODELS,
    ):
        log.info(f"Model: {str(picked_model)}", display_console=False)
        tqdm.write(f"Model: {str(picked_model)}")

        # Get the threshold value in 'operating_points' that is closest to 'act_threshold'
        val_model_basepath = basepath / str(picked_model) / "validation" / str(DS_val)
        used_threshold = min(
            operating_points, key=lambda input_list: abs(input_list - act_threshold)
        )

        # Calculate PSDS and Fscores for each epoch of the input model
        values = {}
        psd_scores = []
        fscores_epochs = []
        for e in tqdm(
            iterable=range(1, epochs + 1),
            desc="Epoch",
            leave=False,
            position=1,
            colour=config.TQDM_EPOCHS,
        ):
            state = load_model(val_model_basepath / f"Ve{e}.pt")
            op_table = state["op_table"]
            psds, fscores = psd_score(
                op_table,
                DS_val.get_annotations(),
                psds_params,
                operating_points,
            )
            try:
                fscores_epochs.append(fscores.Fscores.loc[used_threshold])
            except:
                fscores_epochs.append(0)

            psd_scores.append(psds.value)

        # Add values
        values["tr_epoch_accs"] = state["tr_epoch_accs"][0:epochs]
        values["tr_epoch_losses"] = state["tr_epoch_losses"][0:epochs]
        values["val_acc_table"] = state["val_acc_table"][used_threshold][0:epochs]
        values["val_epoch_losses"] = state["val_epoch_losses"][0:epochs]
        values["Fscores"] = fscores_epochs
        values["psds"] = psd_scores

        outputs[str(picked_model)] = values
    return outputs


def plot_models(
    what_to_plot: List[config.PLOT_MODES],
    picked_models: List[Model],
    basepath: Path,
    log: Logger,
    epochs: int = config.EPOCHS,
    act_threshold: float = config.ACT_THRESHOLD,
    DS_val: Dataset = None,
    psds_params: Dict = config.PSDS_PARAMS,
    operating_points: np.ndarray = config.OPERATING_POINTS,
    plots_basepath: Path = None,
):
    """
    Plots metrics specified in 'what_to_plot' in config.py.
    """
    if not what_to_plot:
        exit("What to plot is specified in config.py")

    plot_vals = get_val_model_values(
        picked_models,
        basepath,
        epochs,
        log,
        act_threshold,
        DS_val,
        psds_params,
        operating_points,
    )

    # Parameters for plotting
    plot_params = {
        "linestyle": "-",
        "marker": "o",
        "markersize": 4,
    }
    plt.style.use("ggplot")
    plt.figure(figsize=(18, 10))

    # Determine plot-grid
    if len(what_to_plot) == 1:
        plot_grid = (1, 1)
    else:
        plot_grid = (2, -(-len(what_to_plot) // 2))

    ### Plots ###
    x = range(1, epochs + 1)
    for i, wtp in enumerate(what_to_plot):
        for model_name, values in plot_vals.items():
            plt.subplot(plot_grid[0], plot_grid[1], i + 1)
            plt.plot(x, values[wtp.value[0]], **plot_params, label=model_name)

            # Force x-axis to use integers
            plt.xticks(range(math.floor(min(x)), math.ceil(max(x)) + 1))

            plt.title(wtp.value[2])
            plt.ylabel(wtp.value[1])
            plt.xlabel("Epoch")
            plt.legend()

    if plots_basepath is not None:
        path_to_plot = create_basepath(
            plots_basepath
            / f"DTC{psds_params['dtc_threshold']}_GTC{psds_params['gtc_threshold']}"
        ) / (f"{get_datetime()}_OP{act_threshold}.png")
        plt.savefig(path_to_plot)
        tqdm.write(f"Plot saved to: {path_to_plot}")

    plt.show()
