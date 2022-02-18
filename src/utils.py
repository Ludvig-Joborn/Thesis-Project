import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
from typing import Callable, List


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
