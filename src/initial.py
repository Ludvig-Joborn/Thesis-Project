import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
from torch import nn
from models.basic_nn import NeuralNetwork as NN
from datasets.datasets_utils import *
from datasets.desed import DESED_Strong
from config import *

# Use cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

RECORD_LEN_SECONDS = 10

train_data = torchaudio.datasets.TEDLIUM(root="data", subset="train", download=False)
test_data = torchaudio.datasets.TEDLIUM(root="data", subset="test", download=False)

DES = DESED_Strong(
    "DESED NUTS",
    PATH_TO_PUBLIC_EVAL_DESED_TSV,
    PATH_TO_PUBLIC_EVAL_DESED_WAVS,
    PUBLIC_EVAL_DESED_CLIP_LEN_SECONDS,
)

print(DES)
print(len(DES))
print(DES[691])

# One sample
(
    waveform,
    sample_rate,
    transcript,
    talk_id,
    speaker_id,
    identifier,
) = train_data.__getitem__(10)

### Sequence the waveform into chunks of RECORD_LEN_SECONDS
print("\nBefore waveform.shape:", waveform.shape)
print("target len:", sample_rate * RECORD_LEN_SECONDS)
print()

waveform = to_mono(waveform)

# list_ = split_waveform(waveform, sample_rate, RECORD_LEN_SECONDS)
list_ = [trim_audio(waveform, sample_rate, RECORD_LEN_SECONDS)]

print("After:")
print("list of waveforms -> len:", len(list_))
[print("shape:", (l.shape[0], l.shape[1]), "| elements:", l) for l in list_]


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


if False:
    (
        waveform,
        sample_rate,
        transcript,
        talk_id,
        speaker_id,
        identifier,
    ) = train_data.__getitem__(1)

    waveform_len = waveform.shape[1]
    if waveform_len > sample_rate * RECORD_LEN_SECONDS:
        waveform = waveform[:, : sample_rate * RECORD_LEN_SECONDS]
    elif waveform_len < sample_rate * RECORD_LEN_SECONDS:
        num_missing_samples = sample_rate * RECORD_LEN_SECONDS - waveform_len
        last_dim_padding = (0, num_missing_samples)
        waveform = torch.nn.functional.pad(waveform, last_dim_padding)


n_fft = 2048
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128

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

# plot spec
for waveform in list_:
    melspec = mel_spectrogram(waveform)
    print(melspec.size())
    plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")


######################################################

exit()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NN().to(device)
print(model)

logits = model(melspec.to(device))
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
