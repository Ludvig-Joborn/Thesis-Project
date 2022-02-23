import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
from datetime import datetime
from typing import Callable, List
from pathlib import Path

from config import EPOCHS


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=True)


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


def get_datetime() -> str:
    """
    Returns the current date and time as a string.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def create_path(
    dir: Path, filename: str = get_datetime(), ending: str = ".pt", best: bool = False
) -> Path:
    # Create directory
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)

    b = "_best" if best else ""
    path = dir / f"{filename}{b}{ending}"
    return path


def plot_tr_val_acc_loss(
    tr_losses: List[float],
    tr_accs: List[float],
    val_losses: List[float],
    val_accs: List[float],
):
    """
    Plots the training and validation accuracy and loss.
    """
    plt.style.use("ggplot")

    x = range(1, len(tr_losses) + 1)

    # Parameters for plotting
    c_tr = "m"
    c_val = "c"
    plot_params = {
        "linestyle": "-",
        "marker": "o",
        "markersize": 4,
    }

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, tr_accs, color=c_tr, **plot_params, label="Training Acc")
    plt.plot(x, val_accs, color=c_val, **plot_params, label="Validation Acc")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, tr_losses, color=c_tr, **plot_params, label="Training Loss")
    plt.plot(x, val_losses, color=c_val, **plot_params, label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


def plot_model_selection(model_saves):
    """
    Plots the training and validation accuracy and loss.
    """

    plt.style.use("ggplot")
    x = range(1, EPOCHS + 1)

    # Parameters for plotting
    plot_params = {
        "linestyle": "-",
        "marker": "o",
        "markersize": 4,
    }

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    for key, model_save in model_saves.items():
        plt.plot(x, model_save["tr_epoch_accs"][0:EPOCHS], **plot_params, label=key)
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()

    plt.subplot(2, 2, 2)
    for key, model_save in model_saves.items():
        plt.plot(x, model_save["tr_epoch_losses"][0:EPOCHS], **plot_params, label=key)
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(2, 2, 3)
    for key, model_save in model_saves.items():
        plt.plot(x, model_save["val_epoch_accs"][0:EPOCHS], **plot_params, label=key)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.subplot(2, 2, 4)
    for key, model_save in model_saves.items():
        plt.plot(x, model_save["val_epoch_losses"][0:EPOCHS], **plot_params, label=key)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Validation Loss")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    dict = {
        "b1": {
            "model_path": None,
            "best_model_path": None,
            "tr_epoch_losses": [0.2],
            "tr_epoch_accs": [0.8],
            "val_epoch_losses": [0.3],
            "val_epoch_accs": [0.7],
            "best_val_acc": 0,
            "log_path": None,
        },
        "b2": {
            "model_path": None,
            "best_model_path": None,
            "tr_epoch_losses": [0.25],
            "tr_epoch_accs": [0.65],
            "val_epoch_losses": [0.4],
            "val_epoch_accs": [0.5],
            "best_val_acc": 0,
            "log_path": None,
        },
    }
    epochs = len(dict["b1"]["tr_epoch_losses"])
    plot_model_selection(dict, epochs)
